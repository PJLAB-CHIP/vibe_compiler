//===- WholeCardExecutableAdmission.cpp - Whole-card exact gate --------===//

#include "WholeCardExecutableAdmission.h"

#include "AcceptedCallClosure.h"
#include "BoundedTileExecutor.h"
#include "CompilationInternal.h"
#include "DirectDTETransport.h"
#include "ExecutableBundleInternal.h"
#include "TargetArtifactInternal.h"
#include "WholeCardResourceAcceptance.h"

#include "Wafer/Analysis/SingleExecutionRegionFlow.h"
#include "Wafer/IR/Target/PhysicalTopology.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

llvm::StringRef WholeCardAdmissionFailure::getDiagnosticLabel() const {
  switch (kind) {
  case WholeCardAdmissionFailureKind::None:
    return "none";
  case WholeCardAdmissionFailureKind::Contract:
    return "admission-contract";
  case WholeCardAdmissionFailureKind::PhysicalTileDomain:
    return "physical-tile-domain";
  case WholeCardAdmissionFailureKind::PhysicalTileMaterialization:
    return "physical-tile-materialization";
  case WholeCardAdmissionFailureKind::PhysicalTileVerification:
    return "physical-tile-verifier";
  case WholeCardAdmissionFailureKind::DDRPlanning:
    return "whole-card-ddr";
  case WholeCardAdmissionFailureKind::IndexLowering:
    return "accepted-index-lowering";
  case WholeCardAdmissionFailureKind::DirectDTETransport:
    return "direct-dte";
  case WholeCardAdmissionFailureKind::WholeCardResources:
    return "whole-card-resources";
  case WholeCardAdmissionFailureKind::AcceptedPhysicalTileVerification:
    return "accepted-physical-tile-verifier";
  case WholeCardAdmissionFailureKind::AcceptedCallClosure:
    return "accepted-call-closure";
  case WholeCardAdmissionFailureKind::PartitionResourceProjection:
    return "partition-resource-projection";
  case WholeCardAdmissionFailureKind::RuntimeLaunchContract:
    return "runtime-launch-contract";
  case WholeCardAdmissionFailureKind::TargetABIPreparation:
    return "target-abi-preparation";
  case WholeCardAdmissionFailureKind::TargetABILowering:
    return "target-abi-lowering";
  }
  llvm_unreachable("unknown whole-card admission failure kind");
}

bool WholeCardAdmissionFailure::isProvenExactRejection() const {
  // The current admission kinds identify the stage that failed, not a proof
  // class. Each stage still combines unsupported IR, verifier/internal errors
  // and (in some cases) exact resource rejection. Until those producers return
  // typed proof evidence, none of these coarse kinds may prune search.
  return false;
}

namespace {

static mlir::LogicalResult
lowerAcceptedPhysicalTileIndexExpressions(mlir::ModuleOp module) {
  return runPassPipeline(module, "accepted-tile-index-lowering",
                         wafer::addLowerAffineControlAndIndexingPass);
}

static mlir::LogicalResult
assignAcceptedPhysicalTileDDROffsets(mlir::ModuleOp module,
                                     const TargetMemoryPolicy &memory) {
  PlanDDRMemoryPassOptions options;
  options.ddrAlignmentBytes = memory.ddrAlignmentBytes;
  options.ddrCapacityBytes = memory.ddrCapacityBytes;
  options.ddrLargestContiguousBytes = memory.ddrLargestContiguousBytes;
  options.ddrBandwidthLimitBytes = memory.ddrBandwidthLimitBytes;
  return runPassPipeline(module, "accepted-tile-ddr-planning",
                         [&](mlir::OpPassManager &manager) {
                           wafer::addAssignDDROffsetsPass(manager, options);
                         });
}

static llvm::Expected<RuntimeLaunchContract> formAcceptedRuntimeLaunchContract(
    const ExecutionConfig &executionConfig,
    llvm::ArrayRef<mlir::ModuleOp> acceptedTileModules,
    uint64_t programSlotCount) {
  constexpr std::array main{RuntimeLaunchPhaseRole::Main};
  constexpr std::array prepareMain{RuntimeLaunchPhaseRole::Prepare,
                                   RuntimeLaunchPhaseRole::Main};
  bool requiresRuntimePrepare = false;
  for (mlir::ModuleOp module : acceptedTileModules)
    module.walk([&](mlir::Operation *operation) {
      if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(operation))
        requiresRuntimePrepare = true;
    });

  if (executionConfig.getRuntimeLaunchKind() == RuntimeLaunchKind::Model) {
    if (executionConfig.getPhysicalTileCount() != 16 || requiresRuntimePrepare)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "model runtime launch requires 16 physical Tiles without a prepare "
          "phase");
    return RuntimeLaunchContract::createModel(ModelEntryABI::Tx81ModelBootParam,
                                              main);
  }
  if (executionConfig.getPhysicalTileCount() != 16)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel runtime launch requires the complete 16-Tile domain");
  const KernelLaunchForm form = requiresRuntimePrepare
                                    ? KernelLaunchForm::Cluster
                                    : KernelLaunchForm::Grid;
  const uint64_t packetBytes = requiresRuntimePrepare
                                   ? kTx81ClusterKernelArgumentBytesMax
                                   : kTx81KernelArgumentBytesMax;
  const uint64_t directSlotsPerTile =
      packetBytes / sizeof(uint64_t) /
      static_cast<uint64_t>(executionConfig.getPhysicalTileCount());
  // Keep ordinary and profile compilation on the same entry ABI. Target ABI
  // preparation can append one default-DDR workspace, one profile record and,
  // for cluster launches, one transport-status slot after program-boundary
  // bindings have been fixed.
  const uint64_t possibleCompilerSlots =
      2 + static_cast<uint64_t>(requiresRuntimePrepare);
  const KernelEntryABI entryABI =
      programSlotCount > directSlotsPerTile ||
              possibleCompilerSlots > directSlotsPerTile - programSlotCount
          ? KernelEntryABI::TileRowPointerTable
          : KernelEntryABI::TileMajorPointerTable;
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
    // rotating-buffer loop may carry multiple subviews of the same returned DDR
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
    if (mlir::Value entry =
            analysis::getSingleExecutionRegionEntryOperand(argument))
      return resolveDDRMovementRootImpl(entry, std::move(seen));
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
  if (result)
    if (mlir::Value exit =
            analysis::getSingleExecutionRegionExitOperand(result))
      return resolveDDRMovementRootImpl(exit, std::move(seen));
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

bool hasBoundaryOnlyDDRMovementEvidence(const PhysicalTileExecutable &tile) {
  mlir::ModuleOp module = tile.getModule();
  mlir::func::FuncOp entry =
      module.lookupSymbol<mlir::func::FuncOp>(tile.getEntrySymbol());
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

static mlir::LogicalResult verifyAcceptedPhysicalTileModule(
    mlir::ModuleOp module, const ExecutionConfig &config,
    PhysicalCardId physicalCardId, PhysicalTileId physicalTileId,
    TransportContract transport) {
  if (physicalCardId != PhysicalCardId(0))
    return module.emitOpError(
        "physical card is outside the current single-card configuration");
  if (physicalTileId.getValue() < 0 ||
      physicalTileId.getValue() >= config.getPhysicalTileCount())
    return module.emitOpError("physical Tile is outside ExecutionConfig");
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
      "is not legal in an accepted physical Tile executable");
}

static std::optional<frontend::ProgramPartitionSlice>
findPartitionSlice(llvm::ArrayRef<frontend::ProgramPartitionSlice> slices,
                   int64_t partitionId) {
  const frontend::ProgramPartitionSlice *match = nullptr;
  for (const frontend::ProgramPartitionSlice &slice : slices) {
    if (slice.partitionId != partitionId)
      continue;
    if (match)
      return std::nullopt;
    match = &slice;
  }
  if (!match)
    return std::nullopt;
  return *match;
}

static mlir::FailureOr<std::vector<ProgramResourceBinding>>
buildProgramResourceBindings(
    const frontend::FrontendProgramVerificationResult &program,
    int64_t partitionId, mlir::ModuleOp diagnosticAnchor) {
  std::vector<ProgramResourceBinding> bindings;
  auto appendBoundary = [&](const frontend::ProgramBoundaryBinding &binding,
                            ProgramResourceRole role) -> mlir::LogicalResult {
    std::optional<frontend::ProgramPartitionSlice> slice =
        findPartitionSlice(binding.partitionSlices, partitionId);
    if (!slice)
      return diagnosticAnchor.emitOpError(
          "typed program boundary does not contain exactly one partition "
          "slice");
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
    std::optional<frontend::ProgramPartitionSlice> slice =
        findPartitionSlice(parameter.partitionSlices, partitionId);
    if (!slice) {
      diagnosticAnchor.emitOpError(
          "typed parameter metadata does not contain exactly one partition "
          "slice");
      return mlir::failure();
    }
    bindings.push_back({ProgramResourceRole::Parameter, parameter.argumentIndex,
                        -1, parameter.name, parameter.dtype,
                        parameter.distribution, parameter.globalShape,
                        parameter.localShape, std::move(*slice)});
  }
  for (const frontend::ProgramConstantBinding &constant : program.constants) {
    frontend::ProgramPartitionSlice slice;
    slice.partitionId = partitionId;
    slice.replicaId = partitionId;
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

static mlir::FailureOr<llvm::SmallVector<PhysicalTileId, 16>>
deriveSingleCardPhysicalTileDomain(llvm::ArrayRef<mlir::ModuleOp> modules,
                                   const ExecutionConfig &executionConfig) {
  if (modules.empty())
    return mlir::failure();

  llvm::SmallVector<PhysicalTileId, 16> expectedTileIds;
  for (size_t moduleIndex = 0; moduleIndex < modules.size(); ++moduleIndex) {
    mlir::ModuleOp module = modules[moduleIndex];
    std::string reason;
    mlir::FailureOr<PhysicalTopology> topology =
        PhysicalTopology::create(module, &reason);
    if (mlir::failed(topology)) {
      module.emitOpError("cannot derive the physical Tile executable domain: ")
          << reason;
      return mlir::failure();
    }
    if (topology->getCardCount() != 1 ||
        topology->getTilesPerCard() != executionConfig.getPhysicalTileCount()) {
      module.emitOpError(
          "physical topology does not match the single-card ExecutionConfig");
      return mlir::failure();
    }
    std::optional<llvm::ArrayRef<PhysicalTileId>> available =
        topology->getAvailableTileIds(PhysicalCardId(0));
    if (!available ||
        available->size() !=
            static_cast<size_t>(executionConfig.getPhysicalTileCount()) ||
        available->size() != modules.size()) {
      module.emitOpError(
          "physical Tile executable domain must cover all and only available "
          "Tiles");
      return mlir::failure();
    }

    if (moduleIndex == 0) {
      expectedTileIds.assign(available->begin(), available->end());
      continue;
    }
    if (!std::equal(available->begin(), available->end(),
                    expectedTileIds.begin(), expectedTileIds.end())) {
      module.emitOpError(
          "physical Tile modules do not agree on the available Tile domain");
      return mlir::failure();
    }
  }
  return expectedTileIds;
}

struct PreTargetWholeCardExecutable {
  PreTargetWholeCardExecutable(
      std::vector<PhysicalTileExecutable> tiles,
      RuntimeLaunchContract runtimeLaunchContract,
      analysis::WholeCardInstructionProgramCost resourceCost)
      : tiles(std::move(tiles)),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        resourceCost(std::move(resourceCost)) {}

  std::vector<PhysicalTileExecutable> tiles;
  RuntimeLaunchContract runtimeLaunchContract;
  analysis::WholeCardInstructionProgramCost resourceCost;
};

static mlir::FailureOr<PreTargetWholeCardExecutable> runPreTargetAdmission(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    WholeCardAdmissionFailureKind &failureKind,
    unsigned tilePipelineParallelism) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "acceptance", "whole-card-executable-admission", "pre-target");

  const size_t tileCount = modules.size();
  if (tileCount == 0 ||
      tileCount !=
          static_cast<size_t>(executionConfig.getPhysicalTileCount()) ||
      program.numPartitions != executionConfig.getNumPartitions()) {
    failureKind = WholeCardAdmissionFailureKind::PhysicalTileDomain;
    return mlir::failure();
  }
  mlir::MLIRContext *ownerContext = nullptr;
  llvm::SmallVector<mlir::ModuleOp, 16> moduleViews;
  moduleViews.reserve(tileCount);
  for (mlir::OwningOpRef<mlir::ModuleOp> &owned : modules) {
    if (!owned) {
      failureKind = WholeCardAdmissionFailureKind::PhysicalTileMaterialization;
      return mlir::failure();
    }
    mlir::ModuleOp module = *owned;
    if (!ownerContext)
      ownerContext = module.getContext();
    if (module.getContext() != ownerContext ||
        mlir::failed(verifyExactExecutionConfig(module, executionConfig)) ||
        mlir::failed(mlir::verify(module))) {
      failureKind = WholeCardAdmissionFailureKind::PhysicalTileVerification;
      return mlir::failure();
    }
    moduleViews.push_back(module);
  }
  mlir::FailureOr<llvm::SmallVector<PhysicalTileId, 16>> physicalTileIds =
      deriveSingleCardPhysicalTileDomain(moduleViews, executionConfig);
  if (mlir::failed(physicalTileIds)) {
    failureKind = WholeCardAdmissionFailureKind::PhysicalTileDomain;
    return mlir::failure();
  }

  const TargetMemoryPolicy memory =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default).memory;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "acceptance-phase", "whole-card-executable-admission", "ddr-planning");
    std::vector<uint8_t> failedTiles(tileCount);
    runBoundedTileModulePipelines(
        moduleViews,
        [&](size_t tile) {
          mlir::ModuleOp module = moduleViews[tile];
          failedTiles[tile] = mlir::failed(assignAcceptedPhysicalTileDDROffsets(
                                  module, memory)) ||
                              mlir::failed(mlir::verify(module));
        },
        tilePipelineParallelism == 0 ? kMaximumBoundedTilePipelineWorkers
                                     : tilePipelineParallelism);
    if (llvm::is_contained(failedTiles, uint8_t{1})) {
      failureKind = WholeCardAdmissionFailureKind::DDRPlanning;
      return mlir::failure();
    }
  }

  {
    wafer::support::ScopedCompileTimingSpan timing(
        "acceptance-phase", "whole-card-executable-admission",
        "index-lowering");
    std::vector<uint8_t> failedTiles(tileCount);
    runBoundedTileModulePipelines(
        moduleViews,
        [&](size_t tile) {
          mlir::ModuleOp module = moduleViews[tile];
          failedTiles[tile] =
              mlir::failed(lowerAcceptedPhysicalTileIndexExpressions(module)) ||
              mlir::failed(mlir::verify(module));
        },
        tilePipelineParallelism == 0 ? kMaximumBoundedTilePipelineWorkers
                                     : tilePipelineParallelism);
    if (llvm::is_contained(failedTiles, uint8_t{1})) {
      failureKind = WholeCardAdmissionFailureKind::IndexLowering;
      return mlir::failure();
    }
  }

  mlir::FailureOr<TransportContract> transport;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "acceptance-phase", "whole-card-executable-admission", "transport");
    transport = acceptDirectDTETransport(moduleViews);
  }
  if (mlir::failed(transport)) {
    failureKind = WholeCardAdmissionFailureKind::DirectDTETransport;
    return mlir::failure();
  }

  mlir::FailureOr<analysis::WholeCardInstructionProgramCost> resourceCost;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "acceptance-phase", "whole-card-executable-admission",
        "whole-card-resources");
    resourceCost = acceptWholeCardResources(moduleViews, *physicalTileIds,
                                            executionConfig);
  }
  if (mlir::failed(resourceCost)) {
    failureKind = WholeCardAdmissionFailureKind::WholeCardResources;
    return mlir::failure();
  }

  std::vector<PhysicalTileExecutable> tiles;
  tiles.reserve(tileCount);
  const PhysicalCardId physicalCardId(0);
  constexpr int64_t kSingleCardPartitionId = 0;
  for (size_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
    mlir::ModuleOp module = *modules[tileIndex];
    const PhysicalTileId physicalTileId = (*physicalTileIds)[tileIndex];
    if (mlir::failed(verifyAcceptedPhysicalTileModule(
            module, executionConfig, physicalCardId, physicalTileId,
            *transport))) {
      failureKind =
          WholeCardAdmissionFailureKind::AcceptedPhysicalTileVerification;
      return mlir::failure();
    }
    llvm::Expected<AcceptedCallClosure> closure =
        analyzeAcceptedCallClosure(module);
    if (!closure) {
      llvm::consumeError(closure.takeError());
      failureKind = WholeCardAdmissionFailureKind::AcceptedCallClosure;
      return mlir::failure();
    }
    // Card-level GSPMD has one partition in the current producer. Every
    // physical Tile therefore consumes the same logical partition-0 boundary
    // projection; tile_id is never substituted for partition_id.
    mlir::FailureOr<std::vector<ProgramResourceBinding>> bindings =
        buildProgramResourceBindings(program, kSingleCardPartitionId, module);
    if (mlir::failed(bindings)) {
      failureKind = WholeCardAdmissionFailureKind::PartitionResourceProjection;
      return mlir::failure();
    }
    tiles.push_back(ExecutableBundleBuilder::makePhysicalTile(
        physicalCardId, physicalTileId,
        LaunchSlotId(static_cast<int64_t>(tileIndex)),
        std::move(modules[tileIndex]), closure->entry.getSymName(),
        std::move(*bindings), *transport));
  }

  const uint64_t programSlotCount = tiles.front().getProgramBindings().size();
  if (llvm::any_of(tiles, [&](const PhysicalTileExecutable &tile) {
        return tile.getProgramBindings().size() != programSlotCount;
      })) {
    failureKind = WholeCardAdmissionFailureKind::RuntimeLaunchContract;
    return mlir::failure();
  }
  llvm::Expected<RuntimeLaunchContract> runtimeLaunchContract =
      formAcceptedRuntimeLaunchContract(executionConfig, moduleViews,
                                        programSlotCount);
  if (!runtimeLaunchContract) {
    llvm::consumeError(runtimeLaunchContract.takeError());
    failureKind = WholeCardAdmissionFailureKind::RuntimeLaunchContract;
    return mlir::failure();
  }

  return PreTargetWholeCardExecutable(std::move(tiles),
                                      std::move(*runtimeLaunchContract),
                                      std::move(*resourceCost));
}

static mlir::FailureOr<AcceptedWholeCardExecutable>
runTargetGate(PreTargetWholeCardExecutable candidate,
              const ExecutionConfig &executionConfig,
              WholeCardSynthesisStatistics *statistics,
              WholeCardAdmissionFailureKind &failureKind,
              unsigned tilePipelineParallelism) {
  if (statistics)
    ++statistics->targetGateInvocations;

  const bool transportPreparedBeforeEntry =
      llvm::is_contained(candidate.runtimeLaunchContract.getPhases(),
                         RuntimeLaunchPhaseRole::Prepare);
  llvm::SmallVector<mlir::ModuleOp, 16> tileModules;
  tileModules.reserve(candidate.tiles.size());
  for (const PhysicalTileExecutable &tile : candidate.tiles)
    tileModules.push_back(tile.getModule());

  enum class TargetTileFailure : uint8_t { None, Preparation, Lowering };
  std::vector<TargetTileFailure> tileFailures(candidate.tiles.size());
  const unsigned workers = runBoundedTileModulePipelines(
      tileModules,
      [&](size_t tileIndex) {
        PhysicalTileExecutable &tile = candidate.tiles[tileIndex];
        mlir::FailureOr<PreparedPhysicalTile> prepared = prepareTargetABI(
            tile, executionConfig, transportPreparedBeforeEntry);
        if (mlir::failed(prepared)) {
          tileFailures[tileIndex] = TargetTileFailure::Preparation;
          return;
        }
        if (mlir::failed(lowerToTargetLLVM(*prepared)) ||
            mlir::failed(
                verifyLoweredKernelABI(*prepared, tile.getEntrySymbol())))
          tileFailures[tileIndex] = TargetTileFailure::Lowering;
      },
      tilePipelineParallelism == 0 ? kMaximumBoundedTilePipelineWorkers
                                   : tilePipelineParallelism);
  if (statistics) {
    statistics->targetTileGateInvocations += candidate.tiles.size();
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);
  }
  for (TargetTileFailure failure : tileFailures) {
    if (failure == TargetTileFailure::Preparation) {
      failureKind = WholeCardAdmissionFailureKind::TargetABIPreparation;
      return mlir::failure();
    }
    if (failure == TargetTileFailure::Lowering) {
      failureKind = WholeCardAdmissionFailureKind::TargetABILowering;
      return mlir::failure();
    }
  }
  return AcceptedWholeCardExecutable(std::move(candidate.tiles),
                                     std::move(candidate.runtimeLaunchContract),
                                     std::move(candidate.resourceCost));
}

} // namespace

mlir::FailureOr<AcceptedWholeCardExecutable> admitWholeCardExecutable(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> physicalTileModules,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeCardAdmissionFailure &failure,
    WholeCardSynthesisStatistics *statistics,
    unsigned tilePipelineParallelism) {
  failure = {};
  auto fail = [&](WholeCardAdmissionFailureKind kind, llvm::StringRef message) {
    failure.kind = kind;
    failure.detail = message.str();
    if (!message.empty())
      diagnostics << "wafer-compile: " << message << '\n';
    return mlir::FailureOr<AcceptedWholeCardExecutable>(mlir::failure());
  };

  WholeCardAdmissionFailureKind failureKind =
      WholeCardAdmissionFailureKind::None;
  if (statistics)
    ++statistics->preTargetAttempts;
  mlir::FailureOr<PreTargetWholeCardExecutable> preTarget =
      runPreTargetAdmission(std::move(physicalTileModules), program,
                            executionConfig, failureKind,
                            tilePipelineParallelism);
  if (mlir::failed(preTarget) &&
      failureKind == WholeCardAdmissionFailureKind::None)
    return fail(WholeCardAdmissionFailureKind::Contract,
                "whole-card pre-target admission failed without identifying "
                "its rejecting gate");
  if (mlir::failed(preTarget)) {
    WholeCardAdmissionFailure classified{failureKind, {}};
    std::string message =
        "whole-card executable candidate failed admission gate '" +
        classified.getDiagnosticLabel().str() + "'";
    return fail(failureKind, message);
  }
  if (statistics)
    ++statistics->preTargetAccepted;

  mlir::FailureOr<AcceptedWholeCardExecutable> accepted =
      runTargetGate(std::move(*preTarget), executionConfig, statistics,
                    failureKind, tilePipelineParallelism);
  if (mlir::failed(accepted) &&
      failureKind == WholeCardAdmissionFailureKind::None)
    return fail(WholeCardAdmissionFailureKind::Contract,
                "whole-card target admission failed without identifying its "
                "rejecting gate");
  if (mlir::failed(accepted)) {
    WholeCardAdmissionFailure classified{failureKind, {}};
    std::string message =
        "whole-card executable candidate failed admission gate '" +
        classified.getDiagnosticLabel().str() + "'";
    return fail(failureKind, message);
  }
  if (statistics)
    ++statistics->admittedExecutableCount;
  return accepted;
}

} // namespace wafer::compiler::detail
