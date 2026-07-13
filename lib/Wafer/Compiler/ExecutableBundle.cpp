//===- ExecutableBundle.cpp - Static-rank executable bundle -------------===//

#include "ExecutableBundleInternal.h"

#include "AcceptedCallClosure.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Pipelines/Pipelines.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

struct ExecutableBundleBuilder {
  static RankExecutable
  makeRank(int64_t logicalRank, mlir::OwningOpRef<mlir::ModuleOp> module,
           llvm::StringRef entrySymbol,
           std::vector<RankProgramBinding> programBindings) {
    return RankExecutable(logicalRank, std::move(module), entrySymbol,
                          std::move(programBindings));
  }

  static ExecutableBundle makeBundle(ExecutionConfig executionConfig,
                                     std::shared_ptr<mlir::MLIRContext> context,
                                     std::vector<RankExecutable> ranks) {
    return ExecutableBundle(executionConfig, std::move(context),
                            std::move(ranks));
  }
};

namespace {

mlir::LogicalResult verifyAcceptedRankModule(mlir::ModuleOp module,
                                             const ExecutionConfig &config,
                                             int64_t logicalRank) {
  if (logicalRank < 0 || logicalRank >= config.getRankCount())
    return module.emitOpError("logical rank is outside ExecutionConfig");
  if (mlir::failed(detail::verifyExactExecutionConfig(module, config)) ||
      mlir::failed(mlir::verify(module)))
    return mlir::failure();

  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    llvm::StringRef name = operation->getName().getStringRef();
    bool allowed = dialect == "builtin" || dialect == "func" ||
                   dialect == "arith" || dialect == "math" ||
                   dialect == "memref" || dialect == "scf" || dialect == "cf";
    if (dialect == "wafer")
      allowed = mlir::isa<wafer::TargetTopologyOp, wafer::ExecutionMeshOp,
                          wafer::TileRegionOp, wafer::TileYieldOp>(operation) ||
                name.starts_with("wafer.instr.");
    if (!allowed || mlir::isa<wafer::GroupOp, wafer::GroupYieldOp>(operation)) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    if (mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp,
                  wafer::InstrDTEWaitOp>(operation)) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }

    auto verifyType = [](mlir::Type type) {
      return !mlir::isa<mlir::BaseMemRefType>(type) ||
             wafer::isWaferMemRefType(type);
    };
    if (!llvm::all_of(operation->getOperandTypes(), verifyType) ||
        !llvm::all_of(operation->getResultTypes(), verifyType)) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        if (!llvm::all_of(block.getArgumentTypes(), verifyType)) {
          illegal = operation;
          return mlir::WalkResult::interrupt();
        }

    if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(operation)) {
      auto memory = wafer::getWaferMemoryAttr(alloc.getType());
      if (!memory ||
          (memory.getSpace() == wafer::MemorySpace::SPM &&
           !alloc->getAttrOfType<wafer::SPMOffsetAttr>(
               wafer::kWaferSPMOffsetAttrName)) ||
          (memory.getSpace() == wafer::MemorySpace::DDR &&
           !alloc->getAttrOfType<wafer::DDROffsetAttr>(
               wafer::kWaferDDROffsetAttrName))) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  if (!illegal)
    return mlir::success();
  if (mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp,
                wafer::InstrDTEWaitOp>(illegal))
    return illegal->emitOpError(
        "unsupported_transport: physical DTE acceptance is not implemented");
  return illegal->emitOpError(
      "is not legal in an accepted static-rank executable");
}

std::optional<frontend::ProgramRankSlice>
findRankSlice(llvm::ArrayRef<frontend::ProgramRankSlice> slices,
              int64_t logicalRank) {
  const frontend::ProgramRankSlice *match = nullptr;
  for (const frontend::ProgramRankSlice &slice : slices) {
    if (slice.logicalRank != logicalRank)
      continue;
    if (match)
      return std::nullopt;
    match = &slice;
  }
  if (!match)
    return std::nullopt;
  return *match;
}

mlir::FailureOr<std::vector<RankProgramBinding>> buildRankProgramBindings(
    const frontend::FrontendProgramVerificationResult &program,
    int64_t logicalRank, mlir::ModuleOp diagnosticAnchor) {
  std::vector<RankProgramBinding> bindings;
  auto appendBoundary = [&](const frontend::ProgramBoundaryBinding &binding,
                            ProgramResourceRole role) -> mlir::LogicalResult {
    std::optional<frontend::ProgramRankSlice> slice =
        findRankSlice(binding.rankSlices, logicalRank);
    if (!slice)
      return diagnosticAnchor.emitOpError(
          "typed program boundary does not contain exactly one rank slice");
    bindings.push_back({role,
                        binding.index,
                        {},
                        binding.dtype,
                        binding.distribution,
                        binding.globalShape,
                        binding.localShape,
                        std::move(*slice)});
    return mlir::success();
  };
  for (const frontend::ProgramBoundaryBinding &binding :
       program.distributedInputs)
    if (mlir::failed(appendBoundary(binding, ProgramResourceRole::UserInput)))
      return mlir::failure();
  for (const frontend::ProgramParameterBinding &parameter :
       program.parameters) {
    std::optional<frontend::ProgramRankSlice> slice =
        findRankSlice(parameter.rankSlices, logicalRank);
    if (!slice) {
      diagnosticAnchor.emitOpError(
          "typed parameter metadata does not contain exactly one rank slice");
      return mlir::failure();
    }
    bindings.push_back({ProgramResourceRole::Parameter, parameter.argumentIndex,
                        parameter.name, parameter.dtype, parameter.distribution,
                        parameter.globalShape, parameter.localShape,
                        std::move(*slice)});
  }
  for (const frontend::ProgramConstantBinding &constant : program.constants) {
    frontend::ProgramRankSlice slice;
    slice.logicalRank = logicalRank;
    slice.replicaId = logicalRank;
    slice.offsets.assign(constant.shape.size(), 0);
    slice.sizes = constant.shape;
    slice.strides.assign(constant.shape.size(), 1);
    slice.payloadPath = constant.payloadPath;
    bindings.push_back({ProgramResourceRole::Constant,
                        constant.argumentIndex,
                        {},
                        constant.dtype,
                        frontend::ProgramDistributionKind::Replicated,
                        constant.shape,
                        constant.shape,
                        std::move(slice)});
  }
  for (const frontend::ProgramBoundaryBinding &binding :
       program.distributedOutputs)
    if (mlir::failed(appendBoundary(binding, ProgramResourceRole::Output)))
      return mlir::failure();
  return bindings;
}

} // namespace

llvm::Expected<ExecutableBundle> detail::buildExecutableBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp groupedModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank) {
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    diagnostics << "wafer-compile: " << message << "\n";
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };

  bool requiresPhysicalTransport = false;
  groupedModule.walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferLinalgExtCollectiveOpInterface>(operation)) {
      requiresPhysicalTransport = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (requiresPhysicalTransport)
    return fail("unsupported_transport: physical DTE acceptance is not "
                "implemented for executable bundles");

  std::vector<RankExecutable> ranks;
  ranks.reserve(executionConfig.getRankCount());
  for (int64_t logicalRank = 0; logicalRank < executionConfig.getRankCount();
       ++logicalRank) {
    mlir::OwningOpRef<mlir::ModuleOp> rankModule = groupedModule.clone();
    mlir::PassManager manager(context.get());
    wafer::buildLowerGroupsToSelectedInstrPipeline(manager, logicalRank);
    if (mlir::failed(manager.run(*rankModule)))
      return fail("rank lowering failed for logical rank " +
                  std::to_string(logicalRank));
    if (mlir::failed(verifyAcceptedRankModule(*rankModule, executionConfig,
                                              logicalRank)))
      return fail("accepted-rank verification failed for logical rank " +
                  std::to_string(logicalRank));

    llvm::Expected<detail::AcceptedCallClosure> closure =
        detail::analyzeAcceptedCallClosure(*rankModule);
    if (!closure)
      return fail("accepted-rank call closure failed for logical rank " +
                  std::to_string(logicalRank) + ": " +
                  llvm::toString(closure.takeError()));
    mlir::FailureOr<std::vector<RankProgramBinding>> bindings =
        buildRankProgramBindings(program, logicalRank, *rankModule);
    if (mlir::failed(bindings))
      return fail("typed rank resource projection failed for logical rank " +
                  std::to_string(logicalRank));
    if (failAfterLogicalRank && logicalRank == *failAfterLogicalRank)
      return fail("test-only injected failure after logical rank " +
                  std::to_string(logicalRank));

    std::string entrySymbol = closure->entry.getSymName().str();
    ranks.push_back(ExecutableBundleBuilder::makeRank(
        logicalRank, std::move(rankModule), entrySymbol, std::move(*bindings)));
  }
  if (ranks.size() != static_cast<size_t>(executionConfig.getRankCount()))
    return fail("executable bundle rank domain is incomplete");
  for (auto [expectedRank, rank] : llvm::enumerate(ranks))
    if (rank.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return fail("executable bundle rank domain is not canonical");

  return ExecutableBundleBuilder::makeBundle(
      executionConfig, std::move(context), std::move(ranks));
}

} // namespace wafer::compiler
