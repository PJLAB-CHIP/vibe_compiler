//===- CoordinatedExecutableAdmission.cpp - Exact all-rank gate --------===//

#include "CoordinatedExecutableAdmission.h"

#include "AcceptedCallClosure.h"
#include "BoundedRankExecutor.h"
#include "DirectDTETransport.h"
#include "ExecutableBundleInternal.h"
#include "TargetArtifactInternal.h"
#include "WholeVariantResourceAcceptance.h"

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/PhysicalDataflow.h"
#include "Wafer/Transforms/SoftwarePipelining.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

static mlir::LogicalResult
lowerAcceptedRankIndexExpressions(mlir::ModuleOp module) {
  mlir::PassManager manager(module.getContext());
  wafer::support::attachCompileTiming(manager, "accepted-rank-index-lowering");
  manager.addPass(mlir::createLowerAffinePass());
  return manager.run(module);
}

static llvm::Expected<RuntimeLaunchContract> formAcceptedRuntimeLaunchContract(
    const ExecutionConfig &executionConfig,
    llvm::ArrayRef<mlir::ModuleOp> acceptedRankModules,
    uint64_t programSlotCount) {
  constexpr std::array main{RuntimeLaunchPhaseRole::Main};
  constexpr std::array prepareMain{RuntimeLaunchPhaseRole::Prepare,
                                   RuntimeLaunchPhaseRole::Main};
  bool requiresRuntimePrepare = false;
  for (mlir::ModuleOp module : acceptedRankModules)
    module.walk([&](mlir::Operation *operation) {
      if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(operation))
        requiresRuntimePrepare = true;
    });

  if (executionConfig.getRuntimeLaunchKind() == RuntimeLaunchKind::Model) {
    if (executionConfig.getRankCount() != 16 || requiresRuntimePrepare)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "model runtime launch requires 16 ranks without a prepare phase");
    return RuntimeLaunchContract::createModel(ModelEntryABI::Tx81ModelBootParam,
                                              main);
  }
  if (executionConfig.getRankCount() == 1) {
    if (requiresRuntimePrepare)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "rank-one kernel runtime launch does not support a prepare phase");
    return RuntimeLaunchContract::createKernel(
        KernelLaunchForm::PerRank, KernelEntryABI::RankLocalPointerBlock, main);
  }
  const KernelLaunchForm form = requiresRuntimePrepare
                                    ? KernelLaunchForm::Cluster
                                    : KernelLaunchForm::Grid;
  const uint64_t packetBytes = requiresRuntimePrepare
                                   ? kTx81ClusterKernelArgumentBytesMax
                                   : kTx81KernelArgumentBytesMax;
  const uint64_t directSlotsPerRank =
      packetBytes / sizeof(uint64_t) /
      static_cast<uint64_t>(executionConfig.getRankCount());
  // Keep ordinary and profile compilation on the same entry ABI. Target ABI
  // preparation can append one default-DDR workspace, one profile record and,
  // for cluster launches, one transport-status slot after program-boundary
  // bindings have been fixed.
  const uint64_t possibleCompilerSlots =
      2 + static_cast<uint64_t>(requiresRuntimePrepare);
  const KernelEntryABI entryABI =
      programSlotCount > directSlotsPerRank ||
              possibleCompilerSlots > directSlotsPerRank - programSlotCount
          ? KernelEntryABI::RankRowPointerTable
          : KernelEntryABI::RankMajorPointerTable;
  if (requiresRuntimePrepare)
    return RuntimeLaunchContract::createKernel(form, entryABI, prepareMain);
  return RuntimeLaunchContract::createKernel(form, entryABI, main);
}

static std::optional<unsigned>
resolveTransparentLoopIterArgIndex(mlir::Value value, mlir::scf::ForOp loop) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() != loop.getBody() || argument.getArgNumber() == 0)
        return std::nullopt;
      return argument.getArgNumber() - 1;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *definition = result ? result.getOwner() : nullptr;
    if (auto cast = mlir::dyn_cast_or_null<mlir::memref::CastOp>(definition)) {
      auto sourceType =
          mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
      auto resultType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
      if (!sourceType || !resultType ||
          sourceType.getShape() != resultType.getShape() ||
          sourceType.getElementType() != resultType.getElementType() ||
          sourceType.getMemorySpace() != resultType.getMemorySpace())
        return std::nullopt;
      value = cast.getSource();
      continue;
    }
    if (auto view =
            mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(definition)) {
      mlir::Value source = view.getViewSource();
      auto sourceType = source
                            ? mlir::dyn_cast<mlir::MemRefType>(source.getType())
                            : mlir::MemRefType();
      auto resultType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
      if (!sourceType || !resultType ||
          sourceType.getElementType() != resultType.getElementType() ||
          sourceType.getMemorySpace() != resultType.getMemorySpace())
        return std::nullopt;
      value = source;
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

static mlir::Value
resolveDDRMovementRootImpl(mlir::Value value,
                           llvm::DenseSet<mlir::Value> seen) {
  if (!value || !seen.insert(value).second)
    return {};
  auto valueType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!valueType || !isWaferDDRMemRefType(valueType))
    return {};

  auto resolveLoopCarried = [&](mlir::scf::ForOp loop, unsigned index) {
    if (!loop || index >= loop.getInitArgs().size())
      return mlir::Value{};
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != loop.getInitArgs().size())
      return mlir::Value{};

    // Prove the complete recurrence component reached by this operand. A
    // fixed-slot loop may rotate multiple subviews of the same returned DDR
    // allocation across iter-arg indices. That remains boundary-only when
    // every reached init and every external yield resolve to one exact root.
    // Alternating distinct roots and unknown yield producers fail closed.
    llvm::SmallVector<unsigned, 4> pending{index};
    llvm::DenseSet<unsigned> visited;
    mlir::Value componentRoot;
    while (!pending.empty()) {
      const unsigned current = pending.pop_back_val();
      if (!visited.insert(current).second)
        continue;
      if (current >= loop.getInitArgs().size())
        return mlir::Value{};
      mlir::Value initRoot =
          resolveDDRMovementRootImpl(loop.getInitArgs()[current], seen);
      if (!initRoot || (componentRoot && componentRoot != initRoot))
        return mlir::Value{};
      componentRoot = initRoot;

      mlir::Value next = yield.getOperand(current);
      if (std::optional<unsigned> yieldedIterArg =
              resolveTransparentLoopIterArgIndex(next, loop)) {
        if (*yieldedIterArg >= loop.getInitArgs().size())
          return mlir::Value{};
        pending.push_back(*yieldedIterArg);
        continue;
      }
      mlir::Value yieldedRoot = resolveDDRMovementRootImpl(next, seen);
      if (!yieldedRoot || yieldedRoot != componentRoot)
        return mlir::Value{};
    }
    return componentRoot;
  };

  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = argument.getOwner();
    mlir::Operation *parent = owner ? owner->getParentOp() : nullptr;
    if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(parent)) {
      if (tileRegion.getBody().empty() ||
          owner != &tileRegion.getBody().front() ||
          argument.getArgNumber() >= tileRegion.getInputs().size())
        return {};
      return resolveDDRMovementRootImpl(
          tileRegion.getInputs()[argument.getArgNumber()], std::move(seen));
    }
    if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
      if (owner != loop.getBody() || argument.getArgNumber() == 0)
        return {};
      return resolveLoopCarried(loop, argument.getArgNumber() - 1);
    }
    auto function = mlir::dyn_cast_or_null<mlir::func::FuncOp>(parent);
    return function && !function.empty() && owner == &function.front()
               ? value
               : mlir::Value{};
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  mlir::Operation *definition = result ? result.getOwner() : nullptr;
  if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(definition)) {
    if (tileRegion.getBody().empty() ||
        result.getResultNumber() >=
            tileRegion.getBody().front().getTerminator()->getNumOperands())
      return {};
    return resolveDDRMovementRootImpl(
        tileRegion.getBody().front().getTerminator()->getOperand(
            result.getResultNumber()),
        std::move(seen));
  }
  if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(definition))
    return resolveLoopCarried(loop, result.getResultNumber());
  if (auto cast = mlir::dyn_cast_or_null<mlir::memref::CastOp>(definition)) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
    if (!sourceType || sourceType.getShape() != valueType.getShape() ||
        sourceType.getElementType() != valueType.getElementType() ||
        sourceType.getMemorySpace() != valueType.getMemorySpace())
      return {};
    return resolveDDRMovementRootImpl(cast.getSource(), std::move(seen));
  }
  if (auto view =
          mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(definition)) {
    mlir::Value source = view.getViewSource();
    auto sourceType = source
                          ? mlir::dyn_cast<mlir::MemRefType>(source.getType())
                          : mlir::MemRefType();
    if (!sourceType ||
        sourceType.getElementType() != valueType.getElementType() ||
        sourceType.getMemorySpace() != valueType.getMemorySpace())
      return {};
    return resolveDDRMovementRootImpl(source, std::move(seen));
  }
  return definition && mlir::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp,
                                 mlir::memref::GetGlobalOp>(definition)
             ? value
             : mlir::Value{};
}

static mlir::Value resolveDDRMovementRoot(mlir::Value value) {
  return resolveDDRMovementRootImpl(value, {});
}

/// A NoC-resident collective may still perform boundary loads and required
/// output publication. It must not use a private DDR allocation as an
/// intermediate spill. Artifact-kind metadata records how a candidate was
/// generated and is deliberately not used as this semantic proof.
} // namespace

bool hasBoundaryOnlyDDRMovementEvidence(const RankExecutable &rank) {
  mlir::ModuleOp module = rank.getModule();
  mlir::func::FuncOp entry =
      module.lookupSymbol<mlir::func::FuncOp>(rank.getEntrySymbol());
  if (!entry || entry.empty())
    return false;

  llvm::DenseSet<mlir::Value> readableRoots;
  for (mlir::BlockArgument argument : entry.getArguments()) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
    if (type && isWaferDDRMemRefType(type))
      readableRoots.insert(argument);
  }

  llvm::SmallVector<mlir::func::ReturnOp, 2> returns;
  entry.walk([&](mlir::func::ReturnOp operation) {
    if (operation->getParentOfType<mlir::func::FuncOp>() == entry)
      returns.push_back(operation);
  });
  if (returns.size() != 1)
    return false;

  llvm::DenseSet<mlir::Value> writableRoots;
  for (mlir::Value output : returns.front().getOperands()) {
    if (!mlir::isa<mlir::MemRefType>(output.getType()))
      continue;
    mlir::Value root = resolveDDRMovementRoot(output);
    if (!root)
      return false;
    writableRoots.insert(root);
  }

  bool valid = true;
  module.walk([&](mlir::Operation *operation) {
    if (!valid)
      return mlir::WalkResult::interrupt();
    if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(operation)) {
      mlir::Value root = resolveDDRMovementRoot(rdma.getSource());
      valid = root && readableRoots.contains(root);
    } else if (auto wdma = mlir::dyn_cast<InstrWDMAOp>(operation)) {
      mlir::Value root = resolveDDRMovementRoot(wdma.getDest());
      valid = root && writableRoots.contains(root);
    }
    return valid ? mlir::WalkResult::advance() : mlir::WalkResult::interrupt();
  });
  return valid;
}

namespace {

static mlir::LogicalResult
verifyAcceptedRankModule(mlir::ModuleOp module, const ExecutionConfig &config,
                         int64_t logicalRank, TransportContract transport) {
  if (logicalRank < 0 || logicalRank >= config.getRankCount())
    return module.emitOpError("logical rank is outside ExecutionConfig");
  if (mlir::failed(verifyExactExecutionConfig(module, config)) ||
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
      allowed = mlir::isa<TargetTopologyOp, ExecutionMeshOp, TileRegionOp,
                          TileYieldOp>(operation) ||
                name.starts_with("wafer.instr.");
    if (!allowed) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation)) {
      if (transport != TransportContract::DirectDTE || !send.getBinding()) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    }
    if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation)) {
      if (transport != TransportContract::DirectDTE || !recv.getBinding()) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    }
    if (mlir::isa<InstrDTEWaitOp>(operation) &&
        transport != TransportContract::DirectDTE) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }

    auto verifyType = [](mlir::Type type) {
      return !mlir::isa<mlir::BaseMemRefType>(type) || isWaferMemRefType(type);
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
      auto memory = getWaferMemoryAttr(alloc.getType());
      if (!memory ||
          (memory.getSpace() == MemorySpace::SPM &&
           !alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName)) ||
          (memory.getSpace() == MemorySpace::DDR &&
           !alloc->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName))) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  if (!illegal)
    return mlir::success();
  if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(illegal))
    return illegal->emitOpError(
        "does not satisfy the accepted executable transport contract");
  return illegal->emitOpError(
      "is not legal in an accepted static-rank executable");
}

static std::optional<frontend::ProgramRankSlice>
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

static mlir::FailureOr<std::vector<RankProgramBinding>>
buildRankProgramBindings(
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
                        binding.programIndex,
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
                        -1, parameter.name, parameter.dtype,
                        parameter.distribution, parameter.globalShape,
                        parameter.localShape, std::move(*slice)});
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
                        constant.position,
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

struct PreTargetCoordinatedExecutable {
  PreTargetCoordinatedExecutable(
      std::vector<RankExecutable> ranks,
      RuntimeLaunchContract runtimeLaunchContract,
      analysis::WholeCardInstructionProgramCost resourceCost)
      : ranks(std::move(ranks)),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        resourceCost(std::move(resourceCost)) {}

  std::vector<RankExecutable> ranks;
  RuntimeLaunchContract runtimeLaunchContract;
  analysis::WholeCardInstructionProgramCost resourceCost;
};

static mlir::FailureOr<PreTargetCoordinatedExecutable>
runPreTargetAdmission(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules,
    llvm::ArrayRef<std::shared_ptr<const std::string>> selectedTileIR,
    const CoordinatedScheduleActionIdentity &actionIdentity,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, std::string &failureGate,
    unsigned rankPipelineParallelism) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "acceptance", "coordinated-executable-admission", "pre-target");

  const size_t rankCount = modules.size();
  if (rankCount == 0 ||
      rankCount != static_cast<size_t>(executionConfig.getRankCount()) ||
      rankCount != selectedTileIR.size() ||
      program.logicalRankCount != executionConfig.getRankCount()) {
    failureGate = "rank-domain";
    return mlir::failure();
  }
  const bool fixedSlot =
      actionIdentity.bufferingKind == CoordinatedBufferingKind::StaticFixedSlot;
  if ((fixedSlot && actionIdentity.bufferingPlanOrdinal == 0) ||
      (!fixedSlot && actionIdentity.bufferingPlanOrdinal != 0) ||
      (actionIdentity.workerPlacementKind ==
               CoordinatedWorkerPlacementKind::DisjointComponents &&
       actionIdentity.workerPlacementPlanOrdinal == 0) ||
      (actionIdentity.workerPlacementKind ==
               CoordinatedWorkerPlacementKind::Unplaced &&
       actionIdentity.workerPlacementPlanOrdinal != 0)) {
    failureGate = "action-identity";
    return mlir::failure();
  }

  mlir::MLIRContext *ownerContext = nullptr;
  llvm::SmallVector<mlir::ModuleOp, 16> moduleViews;
  moduleViews.reserve(rankCount);
  for (mlir::OwningOpRef<mlir::ModuleOp> &owned : modules) {
    if (!owned) {
      failureGate = "rank-materialization";
      return mlir::failure();
    }
    mlir::ModuleOp module = *owned;
    if (!ownerContext)
      ownerContext = module.getContext();
    if (module.getContext() != ownerContext ||
        mlir::failed(verifyExactExecutionConfig(module, executionConfig)) ||
        mlir::failed(mlir::verify(module))) {
      failureGate = "rank-verifier";
      return mlir::failure();
    }
    moduleViews.push_back(module);
  }

  const TargetMemoryPolicy memory =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default).memory;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "acceptance-phase", "coordinated-executable-admission",
        "ddr-planning");
    std::vector<uint8_t> failedRanks(rankCount);
    runBoundedRankModulePipelines(
        moduleViews,
        [&](size_t rank) {
          mlir::ModuleOp module = moduleViews[rank];
          failedRanks[rank] =
              mlir::failed(planDDRMemoryModule(
                  module, memory.ddrAlignmentBytes, memory.ddrCapacityBytes,
                  memory.ddrLargestContiguousBytes,
                  memory.ddrBandwidthLimitBytes)) ||
              mlir::failed(mlir::verify(module));
        },
        rankPipelineParallelism == 0 ? kMaximumBoundedRankPipelineWorkers
                                     : rankPipelineParallelism);
    if (llvm::is_contained(failedRanks, uint8_t{1})) {
      failureGate = "whole-variant-ddr";
      return mlir::failure();
    }
  }

  {
    wafer::support::ScopedCompileTimingSpan timing(
        "acceptance-phase", "coordinated-executable-admission",
        "index-lowering");
    std::vector<uint8_t> failedRanks(rankCount);
    runBoundedRankModulePipelines(
        moduleViews,
        [&](size_t rank) {
          mlir::ModuleOp module = moduleViews[rank];
          failedRanks[rank] =
              mlir::failed(lowerAcceptedRankIndexExpressions(module)) ||
              mlir::failed(mlir::verify(module));
        },
        rankPipelineParallelism == 0 ? kMaximumBoundedRankPipelineWorkers
                                     : rankPipelineParallelism);
    if (llvm::is_contained(failedRanks, uint8_t{1})) {
      failureGate = "accepted-index-lowering";
      return mlir::failure();
    }
  }

  if (fixedSlot) {
    std::string specializationFailure;
    if (mlir::failed(wafer::specializePeriodicDirectDTESites(
            moduleViews, &specializationFailure))) {
      if (!specializationFailure.empty())
        moduleViews.front().emitError(specializationFailure);
      failureGate = "direct-dte-site-specialization";
      return mlir::failure();
    }
    for (mlir::ModuleOp module : moduleViews)
      scheduleIndependentInstructionsByReadyOrder(module.getOperation());
  }

  mlir::FailureOr<TransportContract> transport;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "acceptance-phase", "coordinated-executable-admission", "transport");
    transport = acceptDirectDTETransport(moduleViews);
  }
  if (mlir::failed(transport)) {
    failureGate = "direct-dte";
    return mlir::failure();
  }

  mlir::FailureOr<analysis::WholeCardInstructionProgramCost> resourceCost;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "acceptance-phase", "coordinated-executable-admission",
        "whole-card-resources");
    resourceCost = acceptWholeVariantResources(moduleViews, executionConfig);
  }
  if (mlir::failed(resourceCost)) {
    failureGate = "whole-card-resources";
    return mlir::failure();
  }

  std::vector<RankExecutable> ranks;
  ranks.reserve(rankCount);
  for (size_t rank = 0; rank < rankCount; ++rank) {
    mlir::ModuleOp module = *modules[rank];
    if (mlir::failed(verifyAcceptedRankModule(
            module, executionConfig, static_cast<int64_t>(rank), *transport))) {
      failureGate = "accepted-rank-verifier";
      return mlir::failure();
    }
    llvm::Expected<AcceptedCallClosure> closure =
        analyzeAcceptedCallClosure(module);
    if (!closure) {
      llvm::consumeError(closure.takeError());
      failureGate = "accepted-call-closure";
      return mlir::failure();
    }
    mlir::FailureOr<std::vector<RankProgramBinding>> bindings =
        buildRankProgramBindings(program, static_cast<int64_t>(rank), module);
    if (mlir::failed(bindings)) {
      failureGate = "rank-resource-projection";
      return mlir::failure();
    }
    llvm::StringRef tileEvidence =
        selectedTileIR[rank] ? llvm::StringRef(*selectedTileIR[rank])
                             : llvm::StringRef{};
    ranks.push_back(ExecutableBundleBuilder::makeRank(
        static_cast<int64_t>(rank), std::move(modules[rank]),
        closure->entry.getSymName(), std::move(*bindings), *transport,
        tileEvidence));
  }

  const uint64_t programSlotCount = ranks.front().getProgramBindings().size();
  if (llvm::any_of(ranks, [&](const RankExecutable &rank) {
        return rank.getProgramBindings().size() != programSlotCount;
      })) {
    failureGate = "runtime-launch-contract";
    return mlir::failure();
  }
  llvm::Expected<RuntimeLaunchContract> runtimeLaunchContract =
      formAcceptedRuntimeLaunchContract(executionConfig, moduleViews,
                                        programSlotCount);
  if (!runtimeLaunchContract) {
    llvm::consumeError(runtimeLaunchContract.takeError());
    failureGate = "runtime-launch-contract";
    return mlir::failure();
  }

  return PreTargetCoordinatedExecutable(
      std::move(ranks), std::move(*runtimeLaunchContract),
      std::move(*resourceCost));
}

static mlir::FailureOr<AcceptedWholeVariant>
runTargetGate(PreTargetCoordinatedExecutable candidate,
              const ExecutionConfig &executionConfig,
              WholeVariantSelectionStatistics *statistics,
              std::string &failureGate, unsigned rankPipelineParallelism) {
  if (statistics)
    ++statistics->targetGateInvocations;

  const bool transportPreparedBeforeEntry =
      llvm::is_contained(candidate.runtimeLaunchContract.getPhases(),
                         RuntimeLaunchPhaseRole::Prepare);
  llvm::SmallVector<mlir::ModuleOp, 16> rankModules;
  rankModules.reserve(candidate.ranks.size());
  for (const RankExecutable &rank : candidate.ranks)
    rankModules.push_back(rank.getModule());

  enum class TargetRankFailure : uint8_t { None, Preparation, Lowering };
  std::vector<TargetRankFailure> rankFailures(candidate.ranks.size());
  const unsigned workers = runBoundedRankModulePipelines(
      rankModules,
      [&](size_t rankIndex) {
        RankExecutable &rank = candidate.ranks[rankIndex];
        mlir::FailureOr<PreparedTargetRank> prepared = prepareTargetABI(
            rank, executionConfig, transportPreparedBeforeEntry);
        if (mlir::failed(prepared)) {
          rankFailures[rankIndex] = TargetRankFailure::Preparation;
          return;
        }
        if (mlir::failed(lowerToTargetLLVM(*prepared)) ||
            mlir::failed(
                verifyLoweredKernelABI(*prepared, rank.getEntrySymbol())))
          rankFailures[rankIndex] = TargetRankFailure::Lowering;
      },
      rankPipelineParallelism == 0 ? kMaximumBoundedRankPipelineWorkers
                                   : rankPipelineParallelism);
  if (statistics) {
    statistics->targetRankGateInvocations += candidate.ranks.size();
    statistics->maximumRankPipelineWorkers =
        std::max<uint64_t>(statistics->maximumRankPipelineWorkers, workers);
  }
  for (TargetRankFailure failure : rankFailures) {
    if (failure == TargetRankFailure::Preparation) {
      failureGate = "target-abi-preparation";
      return mlir::failure();
    }
    if (failure == TargetRankFailure::Lowering) {
      failureGate = "target-abi-lowering";
      return mlir::failure();
    }
  }
  return AcceptedWholeVariant(std::move(candidate.ranks),
                              std::move(candidate.runtimeLaunchContract),
                              std::move(candidate.resourceCost));
}

} // namespace

mlir::FailureOr<AcceptedWholeVariant> admitCoordinatedExecutable(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> rankModules,
    llvm::ArrayRef<std::shared_ptr<const std::string>> selectedTileIR,
    const CoordinatedScheduleActionIdentity &actionIdentity,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    std::string *failureGate, WholeVariantSelectionStatistics *statistics,
    unsigned rankPipelineParallelism) {
  auto fail = [&](llvm::StringRef gate, llvm::StringRef message) {
    if (failureGate)
      *failureGate = gate.str();
    if (!message.empty())
      diagnostics << "wafer-compile: " << message << '\n';
    return mlir::FailureOr<AcceptedWholeVariant>(mlir::failure());
  };

  std::string gate = "unknown";
  if (statistics)
    ++statistics->preTargetAttempts;
  mlir::FailureOr<PreTargetCoordinatedExecutable> preTarget =
      runPreTargetAdmission(std::move(rankModules), selectedTileIR,
                            actionIdentity, program, executionConfig, gate,
                            rankPipelineParallelism);
  if (mlir::failed(preTarget))
    return fail(gate,
                "coordinated executable candidate failed admission gate '" +
                    gate + "'");
  if (statistics)
    ++statistics->preTargetAccepted;

  mlir::FailureOr<AcceptedWholeVariant> accepted =
      runTargetGate(std::move(*preTarget), executionConfig, statistics, gate,
                    rankPipelineParallelism);
  if (mlir::failed(accepted))
    return fail(gate,
                "coordinated executable candidate failed admission gate '" +
                    gate + "'");
  if (statistics)
    ++statistics->admittedExecutableCount;
  if (failureGate)
    failureGate->clear();
  return accepted;
}

} // namespace wafer::compiler::detail
