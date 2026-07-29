//===- WholeVariantCoordinator.cpp - All-rank candidate commit ----------===//

#include "WholeVariantCoordinator.h"

#include "AcceptedCallClosure.h"
#include "DirectDTETransport.h"
#include "ExecutableBundleInternal.h"
#include "StaticFixedSlotQualification.h"
#include "TargetArtifactInternal.h"
#include "WholeVariantAttemptPlan.h"
#include "WholeVariantResourceAcceptance.h"

#include "Wafer/Analysis/NoCProfitabilityAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetSchedulingCapability.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/SoftwarePipelining.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

constexpr size_t kWholeVariantParetoLimit = 16;
constexpr size_t kReportedAttemptLimit = 8;

static llvm::Expected<RuntimeLaunchContract> formAcceptedRuntimeLaunchContract(
    const ExecutionConfig &executionConfig,
    llvm::ArrayRef<mlir::ModuleOp> acceptedRankModules) {
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
    return RuntimeLaunchContract::createModel(
        ModelEntryABI::Tx81ModelBootParamV1, main);
  }
  if (executionConfig.getRankCount() == 1) {
    if (requiresRuntimePrepare)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "rank-one kernel runtime launch does not support a prepare phase");
    return RuntimeLaunchContract::createKernel(
        KernelLaunchForm::PerRank, KernelEntryABI::RankLocalPointerBlockV1,
        main);
  }
  if (requiresRuntimePrepare)
    return RuntimeLaunchContract::createKernel(
        KernelLaunchForm::Cluster, KernelEntryABI::RankMajorPointerTableV1,
        prepareMain);
  return RuntimeLaunchContract::createKernel(
      KernelLaunchForm::Grid, KernelEntryABI::RankMajorPointerTableV1, main);
}

static std::set<DTEProtocolPhase>
observeCollectivePhases(const RankExecutable &rank) {
  std::set<DTEProtocolPhase> phases;
  rank.getModule().walk(
      [&](InstrDTESendOp op) { phases.insert(op.getMessage().getPhase()); });
  rank.getModule().walk(
      [&](InstrDTERecvOp op) { phases.insert(op.getMessage().getPhase()); });
  return phases;
}

static bool
rankMatchesCollectiveCharacterization(const RankExecutable &rank,
                                      WholeVariantSelectionMode mode) {
  std::set<DTEProtocolPhase> expected;
  switch (mode) {
  case WholeVariantSelectionMode::CharacterizeAllGatherDirect:
    expected = {DTEProtocolPhase::AllGatherDirect};
    break;
  case WholeVariantSelectionMode::CharacterizeAllGatherRing:
    expected = {DTEProtocolPhase::AllGatherRing};
    break;
  case WholeVariantSelectionMode::CharacterizeReduceScatterDirect:
    expected = {DTEProtocolPhase::ReduceScatterDirect};
    break;
  case WholeVariantSelectionMode::CharacterizeReduceScatterRing:
    expected = {DTEProtocolPhase::ReduceScatterRing};
    break;
  case WholeVariantSelectionMode::CharacterizeAllReduceRing:
  case WholeVariantSelectionMode::QualifyNoCResidentAllReduceRing:
    expected = {DTEProtocolPhase::AllReduceRing};
    break;
  case WholeVariantSelectionMode::CharacterizeAllReduceTree:
    expected = {DTEProtocolPhase::AllReduceTreeReduce,
                DTEProtocolPhase::AllReduceTreeBroadcast};
    break;
  case WholeVariantSelectionMode::Production:
  case WholeVariantSelectionMode::ReservedBaseline:
  case WholeVariantSelectionMode::QualifyStaticFixedSlot:
  case WholeVariantSelectionMode::QualifyWorkerPlacement:
  case WholeVariantSelectionMode::QualifyNoCResidentFixedSlotWorker:
    return false;
  }
  return observeCollectivePhases(rank) == expected;
}

static bool
matchesCollectiveCharacterization(llvm::ArrayRef<RankExecutable> ranks,
                                  WholeVariantSelectionMode mode) {
  if (ranks.empty())
    return false;
  for (const RankExecutable &rank : ranks)
    if (!rankMatchesCollectiveCharacterization(rank, mode))
      return false;
  return true;
}

/// Resolve the storage root named by one DDR movement operand. Qualification
/// consumes only current IR: tile-region boundaries, exact SCF loop-carried
/// recurrences, transparent memref casts, and ViewLike aliases are explicit
/// storage-preserving edges. Any other producer or block boundary is
/// intentionally unknown and fails closed.
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

static bool hasBoundaryOnlyDDRMovement(llvm::ArrayRef<RankExecutable> ranks) {
  return !ranks.empty() && llvm::all_of(ranks, [](const RankExecutable &rank) {
    return hasBoundaryOnlyDDRMovementEvidence(rank);
  });
}

static bool
hasAllRankDirectDTEAndBidirectionalTuple(llvm::ArrayRef<RankExecutable> ranks) {
  if (ranks.empty())
    return false;
  bool tupleHasSend = false;
  bool tupleHasRecv = false;
  for (const RankExecutable &rank : ranks) {
    bool rankHasSend = false;
    bool rankHasRecv = false;
    rank.getModule().walk([&](mlir::Operation *operation) {
      rankHasSend |= mlir::isa<InstrDTESendOp>(operation);
      rankHasRecv |= mlir::isa<InstrDTERecvOp>(operation);
    });
    if (!rankHasSend && !rankHasRecv)
      return false;
    tupleHasSend |= rankHasSend;
    tupleHasRecv |= rankHasRecv;
  }
  return tupleHasSend && tupleHasRecv;
}

static bool hasMultipleActualNCCWorkers(const RankExecutable &rank) {
  std::set<NCCWorker> workers;
  rank.getModule().walk([&](mlir::Operation *operation) {
    if (std::optional<NCCWorker> worker = getNCCIssueWorker(operation))
      workers.insert(*worker);
  });
  return workers.size() >= 2 && llvm::any_of(workers, [](NCCWorker worker) {
           return worker != NCCWorker::Worker0;
         });
}

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

/// Disposable complete-rank tuple that has passed every whole-variant gate
/// through final cost collection, but has not yet passed target ABI/LLVM
/// lowering. It is never exposed as an accepted artifact or winner.
struct PreTargetWholeVariant {
  PreTargetWholeVariant(std::vector<RankExecutable> ranks,
                        RuntimeLaunchContract runtimeLaunchContract,
                        analysis::WholeCardInstructionProgramCost resourceCost)
      : ranks(std::move(ranks)),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        resourceCost(std::move(resourceCost)) {}

  std::vector<RankExecutable> ranks;
  RuntimeLaunchContract runtimeLaunchContract;
  analysis::WholeCardInstructionProgramCost resourceCost;
  std::vector<int64_t> selectedStableOrdinals;
  std::vector<wafer::RankArtifactKind> selectedArtifactKinds;
  std::vector<bool> selectedReservedBaselines;
  std::vector<wafer::RankBufferingKind> selectedBufferingKinds;
  std::vector<uint32_t> selectedBufferingPlanOrdinals;
  std::vector<wafer::RankWorkerPlacementKind> selectedWorkerPlacementKinds;
  std::vector<uint32_t> selectedWorkerPlacementPlanOrdinals;
  /// Recomputable target-contract result for scheduling mechanisms introduced
  /// after the reserved serial/storage frontier. It is invocation-local
  /// analysis, never persisted beside the accepted IR.
  bool hasSchedulingCapabilityQuery = false;
  TargetSchedulingProfitabilityEvidence schedulingProfitability;
};

static bool hasMultipleNCCWorkers(mlir::ModuleOp module) {
  llvm::SmallDenseSet<uint32_t, 4> workers;
  module.walk([&](mlir::Operation *operation) {
    if (std::optional<NCCWorker> worker = getNCCIssueWorker(operation))
      workers.insert(static_cast<uint32_t>(*worker));
  });
  return workers.size() > 1;
}

static mlir::LogicalResult classifySchedulingCapability(
    llvm::ArrayRef<mlir::ModuleOp> modules,
    wafer::RankBufferingKind bufferingKind,
    wafer::RankWorkerPlacementKind workerPlacementKind,
    TargetProfileId targetProfile, bool &hasQuery,
    TargetSchedulingProfitabilityEvidence &profitability) {
  hasQuery = false;
  profitability = {};
  if (modules.empty())
    return mlir::failure();

  TargetSchedulingMechanism mechanism =
      TargetSchedulingMechanism::StaticFixedSlot;
  bool anyCrossWorker = false;
  bool anyDTE = false;
  for (mlir::ModuleOp module : modules) {
    anyCrossWorker |= hasMultipleNCCWorkers(module);
    module.walk([&](mlir::Operation *operation) {
      anyDTE |= mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation);
    });
  }
  const bool explicitlyWorkerPlaced =
      workerPlacementKind != wafer::RankWorkerPlacementKind::Unplaced;
  if (anyCrossWorker != explicitlyWorkerPlaced)
    return mlir::failure();

  if (bufferingKind == wafer::RankBufferingKind::StaticFixedSlot) {
    hasQuery = true;
  } else {
    if (workerPlacementKind == wafer::RankWorkerPlacementKind::Unplaced) {
      return mlir::success();
    }
    hasQuery = true;
    mechanism = anyDTE ? TargetSchedulingMechanism::DirectDTEOverlap
                       : TargetSchedulingMechanism::WorkerPlacement;
  }

  llvm::Expected<TargetSchedulingCapabilityRegistry> registry =
      getTargetSchedulingCapabilityRegistry(targetProfile);
  if (!registry) {
    llvm::consumeError(registry.takeError());
    return mlir::failure();
  }

  bool allOverlapQualified = true;
  bool allDrainQualified = true;
  for (mlir::ModuleOp module : modules) {
    llvm::Expected<TargetSchedulingWindowQuery> query =
        analyzeTargetSchedulingWindow(module, targetProfile, mechanism);
    if (!query) {
      llvm::consumeError(query.takeError());
      return mlir::failure();
    }
    if (workerPlacementKind != wafer::RankWorkerPlacementKind::Unplaced &&
        query->workerRelation !=
            TargetSchedulingWorkerRelation::CrossNCCWorkers &&
        query->workerRelation != TargetSchedulingWorkerRelation::MixedNCCAndDTE)
      return mlir::failure();
    llvm::Expected<TargetSchedulingWindowDecision> decision =
        registry->query(*query);
    if (!decision) {
      llvm::consumeError(decision.takeError());
      return mlir::failure();
    }
    if (decision->legality != TargetSchedulingCapabilityState::Supported)
      return mlir::failure();
    allOverlapQualified &= decision->profitability.overlap ==
                           TargetSchedulingOverlapEvidence::QualifiedOverlap;
    allDrainQualified &= decision->profitability.drain ==
                         TargetSchedulingDrainEvidence::QualifiedDrainElision;
  }
  profitability.overlap =
      allOverlapQualified ? TargetSchedulingOverlapEvidence::QualifiedOverlap
                          : TargetSchedulingOverlapEvidence::Unknown;
  profitability.drain =
      allDrainQualified ? TargetSchedulingDrainEvidence::QualifiedDrainElision
                        : TargetSchedulingDrainEvidence::Unknown;
  return mlir::success();
}

static bool
matchesStaticFixedSlotQualification(const PreTargetWholeVariant &candidate) {
  const size_t rankCount = candidate.ranks.size();
  return rankCount != 0 &&
         candidate.selectedReservedBaselines.size() == rankCount &&
         candidate.selectedBufferingKinds.size() == rankCount &&
         candidate.selectedBufferingPlanOrdinals.size() == rankCount &&
         llvm::none_of(candidate.selectedReservedBaselines,
                       [](bool reserved) { return reserved; }) &&
         llvm::all_of(candidate.selectedBufferingKinds,
                      [](wafer::RankBufferingKind kind) {
                        return kind ==
                               wafer::RankBufferingKind::StaticFixedSlot;
                      }) &&
         llvm::all_of(candidate.selectedBufferingPlanOrdinals,
                      [](uint32_t ordinal) { return ordinal > 0; }) &&
         llvm::all_of(candidate.ranks, [](const RankExecutable &rank) {
           return hasStaticFixedSlotQualificationEvidence(rank);
         });
}

static bool
matchesWorkerPlacementQualification(const PreTargetWholeVariant &candidate) {
  const size_t rankCount = candidate.ranks.size();
  return rankCount != 0 &&
         candidate.selectedReservedBaselines.size() == rankCount &&
         candidate.selectedWorkerPlacementKinds.size() == rankCount &&
         candidate.selectedWorkerPlacementPlanOrdinals.size() == rankCount &&
         llvm::none_of(candidate.selectedReservedBaselines,
                       [](bool reserved) { return reserved; }) &&
         llvm::all_of(
             candidate.selectedWorkerPlacementKinds,
             [](wafer::RankWorkerPlacementKind kind) {
               return kind ==
                      wafer::RankWorkerPlacementKind::DisjointComponents;
             }) &&
         llvm::all_of(candidate.selectedWorkerPlacementPlanOrdinals,
                      [](uint32_t ordinal) { return ordinal > 0; }) &&
         llvm::all_of(candidate.ranks, [](const RankExecutable &rank) {
           return hasMultipleActualNCCWorkers(rank);
         });
}

static mlir::FailureOr<PreTargetWholeVariant> tryPreTargetCombination(
    llvm::ArrayRef<size_t> candidateIndices,
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, std::string &failureGate) {
  if (candidateIndices.size() != frontiers.size() || candidateIndices.empty()) {
    failureGate = "rank-candidate-correspondence";
    return mlir::failure();
  }
  std::optional<int64_t> stableOrdinal;
  std::optional<wafer::RankArtifactKind> artifactKind;
  std::optional<wafer::RankBufferingKind> bufferingKind;
  std::optional<uint32_t> bufferingPlanOrdinal;
  std::optional<wafer::RankWorkerPlacementKind> workerPlacementKind;
  std::optional<uint32_t> workerPlacementPlanOrdinal;
  for (auto [rank, candidateIndex] : llvm::enumerate(candidateIndices)) {
    if (candidateIndex >= frontiers[rank].size()) {
      failureGate = "rank-candidate-correspondence";
      return mlir::failure();
    }
    int64_t current = frontiers[rank][candidateIndex].stableOrdinal;
    wafer::RankArtifactKind currentArtifactKind =
        frontiers[rank][candidateIndex].artifactKind;
    wafer::RankBufferingKind currentBufferingKind =
        frontiers[rank][candidateIndex].bufferingKind;
    uint32_t currentBufferingPlanOrdinal =
        frontiers[rank][candidateIndex].bufferingPlanOrdinal;
    wafer::RankWorkerPlacementKind currentWorkerPlacementKind =
        frontiers[rank][candidateIndex].workerPlacementKind;
    uint32_t currentWorkerPlacementPlanOrdinal =
        frontiers[rank][candidateIndex].workerPlacementPlanOrdinal;
    if ((stableOrdinal && current != *stableOrdinal) ||
        (artifactKind && currentArtifactKind != *artifactKind) ||
        (bufferingKind && currentBufferingKind != *bufferingKind) ||
        (bufferingPlanOrdinal &&
         currentBufferingPlanOrdinal != *bufferingPlanOrdinal) ||
        (workerPlacementKind &&
         currentWorkerPlacementKind != *workerPlacementKind) ||
        (workerPlacementPlanOrdinal &&
         currentWorkerPlacementPlanOrdinal != *workerPlacementPlanOrdinal)) {
      failureGate = "rank-candidate-correspondence";
      return mlir::failure();
    }
    stableOrdinal = current;
    artifactKind = currentArtifactKind;
    bufferingKind = currentBufferingKind;
    bufferingPlanOrdinal = currentBufferingPlanOrdinal;
    workerPlacementKind = currentWorkerPlacementKind;
    workerPlacementPlanOrdinal = currentWorkerPlacementPlanOrdinal;
  }

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  llvm::SmallVector<mlir::ModuleOp, 16> moduleViews;
  modules.reserve(candidateIndices.size());
  moduleViews.reserve(candidateIndices.size());
  for (auto [rank, candidateIndex] : llvm::enumerate(candidateIndices)) {
    const RankVariantCandidate &candidate = frontiers[rank][candidateIndex];
    if (!candidate.module) {
      failureGate = "rank-candidate-materialization";
      return mlir::failure();
    }
    modules.push_back(
        mlir::cast<mlir::ModuleOp>(candidate.module.get()->clone()));
    mlir::ModuleOp module = *modules.back();
    if (mlir::failed(verifyExactExecutionConfig(module, executionConfig)) ||
        mlir::failed(mlir::verify(module))) {
      failureGate = "rank-verifier";
      return mlir::failure();
    }
    moduleViews.push_back(module);
  }

  bool hasSchedulingCapabilityQuery = false;
  TargetSchedulingProfitabilityEvidence schedulingProfitability;
  if (mlir::failed(classifySchedulingCapability(
          moduleViews, *bufferingKind, *workerPlacementKind,
          executionConfig.getTargetProfileId(), hasSchedulingCapabilityQuery,
          schedulingProfitability))) {
    failureGate = "target-scheduling-capability";
    return mlir::failure();
  }

  // DDR placement is intentionally absent from rank-frontier entries. Apply
  // it only to this disposable complete tuple so a failed late gate cannot
  // leak offsets into another combination or back into candidate generation.
  const TargetMemoryPolicy memory =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default).memory;
  for (mlir::ModuleOp module : moduleViews) {
    if (mlir::failed(planDDRMemoryModule(
            module, memory.ddrAlignmentBytes, memory.ddrCapacityBytes,
            memory.ddrLargestContiguousBytes, memory.ddrBandwidthLimitBytes)) ||
        mlir::failed(mlir::verify(module))) {
      failureGate = "whole-variant-ddr";
      return mlir::failure();
    }
  }

  if (*bufferingKind == wafer::RankBufferingKind::StaticFixedSlot) {
    std::string specializationFailure;
    if (mlir::failed(wafer::specializePeriodicDirectDTESites(
            moduleViews, &specializationFailure))) {
      if (!specializationFailure.empty())
        moduleViews.front().emitError(specializationFailure);
      failureGate = "direct-dte-site-specialization";
      return mlir::failure();
    }
  }

  mlir::FailureOr<TransportContract> transport =
      acceptDirectDTETransport(moduleViews);
  if (mlir::failed(transport)) {
    failureGate = "direct-dte";
    return mlir::failure();
  }
  llvm::Expected<RuntimeLaunchContract> runtimeLaunchContract =
      formAcceptedRuntimeLaunchContract(executionConfig, moduleViews);
  if (!runtimeLaunchContract) {
    llvm::consumeError(runtimeLaunchContract.takeError());
    failureGate = "runtime-launch-contract";
    return mlir::failure();
  }
  mlir::FailureOr<analysis::WholeCardInstructionProgramCost> resourceCost =
      acceptWholeVariantResources(moduleViews, executionConfig);
  if (mlir::failed(resourceCost)) {
    failureGate = "whole-card-resources";
    return mlir::failure();
  }

  std::vector<RankExecutable> ranks;
  ranks.reserve(modules.size());
  for (size_t rank = 0; rank < modules.size(); ++rank) {
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
    std::string entrySymbol = closure->entry.getSymName().str();
    ranks.push_back(ExecutableBundleBuilder::makeRank(
        static_cast<int64_t>(rank), std::move(modules[rank]), entrySymbol,
        std::move(*bindings), *transport));
  }

  PreTargetWholeVariant candidate(std::move(ranks),
                                  std::move(*runtimeLaunchContract),
                                  std::move(*resourceCost));
  candidate.selectedStableOrdinals.reserve(candidateIndices.size());
  candidate.selectedArtifactKinds.reserve(candidateIndices.size());
  candidate.selectedReservedBaselines.reserve(candidateIndices.size());
  candidate.selectedBufferingKinds.reserve(candidateIndices.size());
  candidate.selectedBufferingPlanOrdinals.reserve(candidateIndices.size());
  candidate.selectedWorkerPlacementKinds.reserve(candidateIndices.size());
  candidate.selectedWorkerPlacementPlanOrdinals.reserve(
      candidateIndices.size());
  for (auto [rank, candidateIndex] : llvm::enumerate(candidateIndices)) {
    candidate.selectedStableOrdinals.push_back(
        frontiers[rank][candidateIndex].stableOrdinal);
    candidate.selectedArtifactKinds.push_back(
        frontiers[rank][candidateIndex].artifactKind);
    candidate.selectedReservedBaselines.push_back(
        frontiers[rank][candidateIndex].reservedBaseline);
    candidate.selectedBufferingKinds.push_back(
        frontiers[rank][candidateIndex].bufferingKind);
    candidate.selectedBufferingPlanOrdinals.push_back(
        frontiers[rank][candidateIndex].bufferingPlanOrdinal);
    candidate.selectedWorkerPlacementKinds.push_back(
        frontiers[rank][candidateIndex].workerPlacementKind);
    candidate.selectedWorkerPlacementPlanOrdinals.push_back(
        frontiers[rank][candidateIndex].workerPlacementPlanOrdinal);
  }
  candidate.hasSchedulingCapabilityQuery = hasSchedulingCapabilityQuery;
  candidate.schedulingProfitability = schedulingProfitability;
  return candidate;
}

static mlir::FailureOr<AcceptedWholeVariant> runTargetGate(
    PreTargetWholeVariant candidate, const ExecutionConfig &executionConfig,
    WholeVariantSelectionStatistics *statistics, std::string &failureGate) {
  if (statistics)
    ++statistics->targetGateInvocations;

  // Target ABI and target-call legality are whole-variant gates, not a later
  // opportunity to replace one rank after the remaining domain was accepted.
  // Lower on owned clones and discard the results; the target-artifact stage
  // will translate the exact committed rank modules once more for publication.
  const bool transportPreparedBeforeEntry =
      llvm::is_contained(candidate.runtimeLaunchContract.getPhases(),
                         RuntimeLaunchPhaseRole::Prepare);
  for (RankExecutable &rank : candidate.ranks) {
    if (statistics)
      ++statistics->targetRankGateInvocations;
    mlir::FailureOr<PreparedTargetRank> prepared =
        prepareTargetABI(rank, executionConfig, transportPreparedBeforeEntry);
    if (mlir::failed(prepared)) {
      failureGate = "target-abi-preparation";
      return mlir::failure();
    }
    if (mlir::failed(lowerToTargetLLVM(*prepared)) ||
        mlir::failed(
            verifyLoweredKernelABI(*prepared, rank.getEntrySymbol()))) {
      failureGate = "target-abi-lowering";
      return mlir::failure();
    }
  }

  AcceptedWholeVariant accepted(std::move(candidate.ranks),
                                std::move(candidate.runtimeLaunchContract),
                                std::move(candidate.resourceCost));
  accepted.selectedStableOrdinals = std::move(candidate.selectedStableOrdinals);
  accepted.selectedArtifactKinds = std::move(candidate.selectedArtifactKinds);
  accepted.selectedReservedBaselines =
      std::move(candidate.selectedReservedBaselines);
  accepted.selectedBufferingKinds = std::move(candidate.selectedBufferingKinds);
  accepted.selectedBufferingPlanOrdinals =
      std::move(candidate.selectedBufferingPlanOrdinals);
  accepted.selectedWorkerPlacementKinds =
      std::move(candidate.selectedWorkerPlacementKinds);
  accepted.selectedWorkerPlacementPlanOrdinals =
      std::move(candidate.selectedWorkerPlacementPlanOrdinals);
  return accepted;
}

enum class ParetoOrder {
  Equivalent,
  LeftDominates,
  RightDominates,
  Incomparable,
  Unknown,
};

struct MetricPair {
  const analysis::ScheduleCostMetric *left;
  const analysis::ScheduleCostMetric *right;
};

static ParetoOrder compareKnownMetrics(llvm::ArrayRef<MetricPair> dimensions) {
  bool leftLower = false;
  bool rightLower = false;
  for (const MetricPair &dimension : dimensions) {
    if (!dimension.left->isKnown() || !dimension.right->isKnown())
      return ParetoOrder::Unknown;
    leftLower |= dimension.left->value < dimension.right->value;
    rightLower |= dimension.right->value < dimension.left->value;
  }
  if (!leftLower && !rightLower)
    return ParetoOrder::Equivalent;
  if (leftLower && !rightLower)
    return ParetoOrder::LeftDominates;
  if (!leftLower && rightLower)
    return ParetoOrder::RightDominates;
  return ParetoOrder::Incomparable;
}

static ParetoOrder
compareNCCDrainCost(const analysis::WholeCardInstructionProgramCost &left,
                    const analysis::WholeCardInstructionProgramCost &right) {
  // A narrower participant set is a correctness scope, not a cheap wait.
  // Count every by-worker call on the highest-priority axes: one steady join
  // covering three workers is more expensive than two steady joins covering
  // one worker each. Join-operation counts remain diagnostics/tie-breakers and
  // cannot hide a wider participant mask.
  const MetricPair priority[] = {
      {&left.aggregateSteadyStateNCCParticipantWaitCount,
       &right.aggregateSteadyStateNCCParticipantWaitCount},
      {&left.aggregateNonTerminalNCCParticipantWaitCount,
       &right.aggregateNonTerminalNCCParticipantWaitCount},
      {&left.aggregateNCCParticipantWaitCount,
       &right.aggregateNCCParticipantWaitCount},
      {&left.aggregateIntrinsicNCCDrainCount,
       &right.aggregateIntrinsicNCCDrainCount},
      {&left.aggregateSteadyStateNCCJoinCount,
       &right.aggregateSteadyStateNCCJoinCount},
      {&left.aggregateNonTerminalNCCJoinCount,
       &right.aggregateNonTerminalNCCJoinCount},
      {&left.aggregateNCCJoinCount, &right.aggregateNCCJoinCount},
  };
  for (const MetricPair &dimension : priority) {
    if (!dimension.left->isKnown() || !dimension.right->isKnown())
      return ParetoOrder::Unknown;
    if (dimension.left->value < dimension.right->value)
      return ParetoOrder::LeftDominates;
    if (dimension.left->value > dimension.right->value)
      return ParetoOrder::RightDominates;
  }
  return ParetoOrder::Equivalent;
}

static ParetoOrder compareExternalDDRMovement(
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right) {
  const MetricPair dimensions[] = {
      {&left.aggregateDDRReadBytes, &right.aggregateDDRReadBytes},
      {&left.aggregateDDRWriteBytes, &right.aggregateDDRWriteBytes},
  };
  return compareKnownMetrics(dimensions);
}

static ParetoOrder compareExactWholeVariantCost(
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right) {
  // When external movement is equal, blocking NCC drains remain the primary
  // exact discriminator. When DDR differs, however, a real drain-vs-DDR
  // tradeoff must remain on the Pareto frontier for the target's static
  // policy; include drain facts as ordinary exact dimensions instead of
  // letting either side erase the other before policy selection.
  if (compareExternalDDRMovement(left, right) == ParetoOrder::Equivalent) {
    ParetoOrder drainOrder = compareNCCDrainCost(left, right);
    if (drainOrder != ParetoOrder::Equivalent)
      return drainOrder;
  }

  const MetricPair dimensions[] = {
      {&left.aggregateSteadyStateNCCParticipantWaitCount,
       &right.aggregateSteadyStateNCCParticipantWaitCount},
      {&left.aggregateNonTerminalNCCParticipantWaitCount,
       &right.aggregateNonTerminalNCCParticipantWaitCount},
      {&left.aggregateNCCParticipantWaitCount,
       &right.aggregateNCCParticipantWaitCount},
      {&left.aggregateIntrinsicNCCDrainCount,
       &right.aggregateIntrinsicNCCDrainCount},
      {&left.aggregateSteadyStateNCCJoinCount,
       &right.aggregateSteadyStateNCCJoinCount},
      {&left.aggregateNonTerminalNCCJoinCount,
       &right.aggregateNonTerminalNCCJoinCount},
      {&left.aggregateNCCJoinCount, &right.aggregateNCCJoinCount},
      {&left.aggregateCompute.npuF16Bf16LogicalOps,
       &right.aggregateCompute.npuF16Bf16LogicalOps},
      {&left.aggregateCompute.npuOtherLogicalOps,
       &right.aggregateCompute.npuOtherLogicalOps},
      {&left.aggregateCompute.vectorF16Bf16LogicalOps,
       &right.aggregateCompute.vectorF16Bf16LogicalOps},
      {&left.aggregateCompute.vectorF32LogicalOps,
       &right.aggregateCompute.vectorF32LogicalOps},
      {&left.aggregateCompute.vectorOtherLogicalOps,
       &right.aggregateCompute.vectorOtherLogicalOps},
      {&left.aggregateDDRReadBytes, &right.aggregateDDRReadBytes},
      {&left.aggregateDDRWriteBytes, &right.aggregateDDRWriteBytes},
      {&left.aggregateSPMMovementBytes, &right.aggregateSPMMovementBytes},
      {&left.aggregateNoC.aggregateTransmitBytes,
       &right.aggregateNoC.aggregateTransmitBytes},
      {&left.aggregateNoC.aggregateReceiveBytes,
       &right.aggregateNoC.aggregateReceiveBytes},
      {&left.minimumHopLinkByteDemand, &right.minimumHopLinkByteDemand},
      {&left.aggregateNoC.collectiveTransmitBytes[0],
       &right.aggregateNoC.collectiveTransmitBytes[0]},
      {&left.aggregateNoC.collectiveTransmitBytes[1],
       &right.aggregateNoC.collectiveTransmitBytes[1]},
      {&left.aggregateNoC.collectiveTransmitBytes[2],
       &right.aggregateNoC.collectiveTransmitBytes[2]},
      {&left.aggregateNoC.collectiveTransmitBytes[3],
       &right.aggregateNoC.collectiveTransmitBytes[3]},
      {&left.aggregateNoC.collectiveTransmitBytes[4],
       &right.aggregateNoC.collectiveTransmitBytes[4]},
      {&left.aggregateInstructionCount, &right.aggregateInstructionCount},
      {&left.aggregateEventCount, &right.aggregateEventCount},
      {&left.maximumRankSPMHighWaterBytes, &right.maximumRankSPMHighWaterBytes},
      {&left.summedRankSPMHighWaterBytes, &right.summedRankSPMHighWaterBytes},
  };

  return compareKnownMetrics(dimensions);
}

static ParetoOrder compareExactExecutionResources(
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right) {
  analysis::WholeCardInstructionProgramCost leftExecution = left;
  analysis::WholeCardInstructionProgramCost rightExecution = right;
  leftExecution.maximumRankSPMHighWaterBytes = {};
  leftExecution.summedRankSPMHighWaterBytes = {};
  rightExecution.maximumRankSPMHighWaterBytes = {};
  rightExecution.summedRankSPMHighWaterBytes = {};
  return compareExactWholeVariantCost(leftExecution, rightExecution);
}

static ParetoOrder compareStaticDataflowPolicy(
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right) {
  const analysis::ScheduleCostMetric &leftDepth =
      left.maximumRankDataDependencyDepth;
  const analysis::ScheduleCostMetric &rightDepth =
      right.maximumRankDataDependencyDepth;
  if (!leftDepth.isKnown() || !rightDepth.isKnown())
    return ParetoOrder::Unknown;
  if (leftDepth.value < rightDepth.value)
    return ParetoOrder::LeftDominates;
  if (leftDepth.value > rightDepth.value)
    return ParetoOrder::RightDominates;
  const analysis::ScheduleCostMetric &leftInversions =
      left.aggregateReadyOrderPriorityInversions;
  const analysis::ScheduleCostMetric &rightInversions =
      right.aggregateReadyOrderPriorityInversions;
  if (!leftInversions.isKnown() || !rightInversions.isKnown())
    return ParetoOrder::Unknown;
  if (leftInversions.value < rightInversions.value)
    return ParetoOrder::LeftDominates;
  if (leftInversions.value > rightInversions.value)
    return ParetoOrder::RightDominates;
  return ParetoOrder::Equivalent;
}

static ParetoOrder compareTargetStaticTradeoff(
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right,
    const TargetStaticSelectionPolicy &policy) {
  if (policy.tradeoff == TargetStaticTradeoffPolicy::Conservative)
    return ParetoOrder::Equivalent;

  const llvm::SmallVector<llvm::SmallVector<MetricPair, 8>, 6> priorityClasses =
      {
          {{&left.aggregateDDRReadBytes, &right.aggregateDDRReadBytes},
           {&left.aggregateDDRWriteBytes, &right.aggregateDDRWriteBytes}},
          {{&left.aggregateNoC.aggregateTransmitBytes,
            &right.aggregateNoC.aggregateTransmitBytes},
           {&left.aggregateNoC.aggregateReceiveBytes,
            &right.aggregateNoC.aggregateReceiveBytes},
           {&left.minimumHopLinkByteDemand, &right.minimumHopLinkByteDemand},
           {&left.aggregateNoC.collectiveTransmitBytes[0],
            &right.aggregateNoC.collectiveTransmitBytes[0]},
           {&left.aggregateNoC.collectiveTransmitBytes[1],
            &right.aggregateNoC.collectiveTransmitBytes[1]},
           {&left.aggregateNoC.collectiveTransmitBytes[2],
            &right.aggregateNoC.collectiveTransmitBytes[2]},
           {&left.aggregateNoC.collectiveTransmitBytes[3],
            &right.aggregateNoC.collectiveTransmitBytes[3]},
           {&left.aggregateNoC.collectiveTransmitBytes[4],
            &right.aggregateNoC.collectiveTransmitBytes[4]}},
          {{&left.aggregateSPMMovementBytes, &right.aggregateSPMMovementBytes}},
          {{&left.aggregateInstructionCount, &right.aggregateInstructionCount},
           {&left.aggregateEventCount, &right.aggregateEventCount}},
          {{&left.aggregateCompute.npuF16Bf16LogicalOps,
            &right.aggregateCompute.npuF16Bf16LogicalOps},
           {&left.aggregateCompute.npuOtherLogicalOps,
            &right.aggregateCompute.npuOtherLogicalOps},
           {&left.aggregateCompute.vectorF16Bf16LogicalOps,
            &right.aggregateCompute.vectorF16Bf16LogicalOps},
           {&left.aggregateCompute.vectorF32LogicalOps,
            &right.aggregateCompute.vectorF32LogicalOps},
           {&left.aggregateCompute.vectorOtherLogicalOps,
            &right.aggregateCompute.vectorOtherLogicalOps}},
          {{&left.maximumRankDataDependencyDepth,
            &right.maximumRankDataDependencyDepth},
           {&left.aggregateReadyOrderPriorityInversions,
            &right.aggregateReadyOrderPriorityInversions}},
      };
  for (const llvm::SmallVector<MetricPair, 8> &priorityClass :
       priorityClasses) {
    ParetoOrder order = compareKnownMetrics(priorityClass);
    if (order != ParetoOrder::Equivalent)
      return order;
  }
  return ParetoOrder::Equivalent;
}

static bool isResourceCostPreferredOver(
    const analysis::WholeCardInstructionProgramCost &candidate,
    const analysis::WholeCardInstructionProgramCost &baseline,
    const TargetStaticSelectionPolicy &policy) {
  // `ExternalMovementFirst` is the target's explicit policy for a strict DDR
  // reduction. A NoC-resident owner-load candidate can require an exact
  // NCC-to-DTE handoff completion while still eliminating the other ranks'
  // DDR transactions; that real cross-resource tradeoff must not be erased by
  // the drain tie-breaker used when external movement is equal.
  if (policy.tradeoff == TargetStaticTradeoffPolicy::ExternalMovementFirst) {
    ParetoOrder ddrOrder = compareExternalDDRMovement(candidate, baseline);
    if (ddrOrder == ParetoOrder::LeftDominates)
      return true;
    if (ddrOrder == ParetoOrder::RightDominates ||
        ddrOrder == ParetoOrder::Unknown)
      return false;
  }

  ParetoOrder order = compareExactWholeVariantCost(candidate, baseline);
  if (order == ParetoOrder::LeftDominates)
    return true;
  if (order == ParetoOrder::Equivalent)
    return compareStaticDataflowPolicy(candidate, baseline) ==
           ParetoOrder::LeftDominates;
  if (order != ParetoOrder::Incomparable)
    return false;
  // Accepted high-water is a capacity fact, not a calibrated performance
  // quantity. When all exact execution-resource dimensions are no worse and
  // at least one is lower, the static target policy accepts the known
  // high-water tradeoff instead of requiring an unavailable timing model.
  if (compareExactExecutionResources(candidate, baseline) ==
      ParetoOrder::LeftDominates)
    return true;
  return compareTargetStaticTradeoff(candidate, baseline, policy) ==
         ParetoOrder::LeftDominates;
}

static bool isPreferredOver(const AcceptedWholeVariant &candidate,
                            const AcceptedWholeVariant &baseline,
                            const TargetStaticSelectionPolicy &policy) {
  return isResourceCostPreferredOver(candidate.resourceCost,
                                     baseline.resourceCost, policy);
}

static bool hasStrictQualifiedDrainReduction(
    const analysis::WholeCardInstructionProgramCost &candidate,
    const analysis::WholeCardInstructionProgramCost &baseline) {
  const MetricPair participantDrainDimensions[] = {
      {&candidate.aggregateSteadyStateNCCParticipantWaitCount,
       &baseline.aggregateSteadyStateNCCParticipantWaitCount},
      {&candidate.aggregateNonTerminalNCCParticipantWaitCount,
       &baseline.aggregateNonTerminalNCCParticipantWaitCount},
      {&candidate.aggregateNCCParticipantWaitCount,
       &baseline.aggregateNCCParticipantWaitCount},
      {&candidate.aggregateIntrinsicNCCDrainCount,
       &baseline.aggregateIntrinsicNCCDrainCount},
  };
  bool strictlyLower = false;
  for (const MetricPair &dimension : participantDrainDimensions) {
    if (!dimension.left->isKnown() || !dimension.right->isKnown() ||
        dimension.left->value > dimension.right->value)
      return false;
    strictlyLower |= dimension.left->value < dimension.right->value;
  }
  return strictlyLower;
}

static bool hasNormalProductionSchedulingEvidence(
    const PreTargetWholeVariant &candidate,
    const AcceptedWholeVariant &reservedBaseline) {
  if (!candidate.hasSchedulingCapabilityQuery)
    return true;
  if (candidate.schedulingProfitability.overlap ==
      TargetSchedulingOverlapEvidence::QualifiedOverlap)
    return true;
  return candidate.schedulingProfitability.drain ==
             TargetSchedulingDrainEvidence::QualifiedDrainElision &&
         hasStrictQualifiedDrainReduction(candidate.resourceCost,
                                          reservedBaseline.resourceCost);
}

struct ParetoCandidateView {
  const analysis::WholeCardInstructionProgramCost *resourceCost = nullptr;
  const std::vector<int64_t> *selectedStableOrdinals = nullptr;
  const std::vector<wafer::RankArtifactKind> *selectedArtifactKinds = nullptr;
  const std::vector<bool> *selectedReservedBaselines = nullptr;
  const std::vector<wafer::RankBufferingKind> *selectedBufferingKinds = nullptr;
  const std::vector<uint32_t> *selectedBufferingPlanOrdinals = nullptr;
  const std::vector<wafer::RankWorkerPlacementKind>
      *selectedWorkerPlacementKinds = nullptr;
  const std::vector<uint32_t> *selectedWorkerPlacementPlanOrdinals = nullptr;
};

template <typename VariantT>
static ParetoCandidateView getParetoCandidateView(const VariantT &candidate) {
  return {&candidate.resourceCost,
          &candidate.selectedStableOrdinals,
          &candidate.selectedArtifactKinds,
          &candidate.selectedReservedBaselines,
          &candidate.selectedBufferingKinds,
          &candidate.selectedBufferingPlanOrdinals,
          &candidate.selectedWorkerPlacementKinds,
          &candidate.selectedWorkerPlacementPlanOrdinals};
}

static bool hasEarlierStaticPolicyOrder(ParetoCandidateView lhs,
                                        ParetoCandidateView rhs) {
  return std::tie(*lhs.selectedStableOrdinals, *lhs.selectedArtifactKinds,
                  *lhs.selectedBufferingKinds,
                  *lhs.selectedBufferingPlanOrdinals,
                  *lhs.selectedWorkerPlacementKinds,
                  *lhs.selectedWorkerPlacementPlanOrdinals,
                  *lhs.selectedReservedBaselines) <
         std::tie(*rhs.selectedStableOrdinals, *rhs.selectedArtifactKinds,
                  *rhs.selectedBufferingKinds,
                  *rhs.selectedBufferingPlanOrdinals,
                  *rhs.selectedWorkerPlacementKinds,
                  *rhs.selectedWorkerPlacementPlanOrdinals,
                  *rhs.selectedReservedBaselines);
}

struct ParetoInsertionPlan {
  bool retained = false;
  llvm::SmallVector<unsigned, 8> dominatedIndices;
};

static ParetoInsertionPlan
planParetoInsertion(ParetoCandidateView candidate,
                    llvm::ArrayRef<AcceptedWholeVariant> frontier,
                    const TargetStaticSelectionPolicy &selectionPolicy) {
  ParetoInsertionPlan plan;
  for (auto [index, existing] : llvm::enumerate(frontier)) {
    ParetoCandidateView existingView = getParetoCandidateView(existing);
    switch (compareExactWholeVariantCost(*candidate.resourceCost,
                                         *existingView.resourceCost)) {
    case ParetoOrder::Unknown:
    case ParetoOrder::RightDominates:
      return {};
    case ParetoOrder::Equivalent:
      switch (compareStaticDataflowPolicy(*candidate.resourceCost,
                                          *existingView.resourceCost)) {
      case ParetoOrder::LeftDominates:
        plan.dominatedIndices.push_back(index);
        continue;
      case ParetoOrder::RightDominates:
        return {};
      case ParetoOrder::Equivalent:
      case ParetoOrder::Incomparable:
      case ParetoOrder::Unknown:
        break;
      }
      if (!hasEarlierStaticPolicyOrder(candidate, existingView))
        return {};
      plan.dominatedIndices.push_back(index);
      break;
    case ParetoOrder::LeftDominates:
      plan.dominatedIndices.push_back(index);
      break;
    case ParetoOrder::Incomparable:
      break;
    }
  }

  // Simulate the exact existing erase/push/sort/cap sequence so a candidate
  // that would immediately be removed by the 16-entry cap does not pay the
  // target ABI/LLVM cost. The same view comparator is used by actual insertion
  // below, including Unknown, equivalent-dataflow, and static-order behavior.
  struct SimulatedEntry {
    ParetoCandidateView view;
    bool isCandidate = false;
  };
  llvm::SmallVector<SimulatedEntry, kWholeVariantParetoLimit + 1> simulated;
  for (auto [index, existing] : llvm::enumerate(frontier)) {
    if (llvm::is_contained(plan.dominatedIndices, static_cast<unsigned>(index)))
      continue;
    simulated.push_back({getParetoCandidateView(existing), false});
  }
  simulated.push_back({candidate, true});
  llvm::sort(simulated,
             [](const SimulatedEntry &lhs, const SimulatedEntry &rhs) {
               return hasEarlierStaticPolicyOrder(lhs.view, rhs.view);
             });
  if (simulated.size() > kWholeVariantParetoLimit) {
    // The pre-target resource facts already carry every input to the target's
    // static selection policy. Preserve the policy-best member when bounding
    // an otherwise incomparable frontier, then use stable metadata order for
    // the remaining slots.
    unsigned policyBestIndex = 0;
    for (unsigned index = 1; index < simulated.size(); ++index)
      if (isResourceCostPreferredOver(
              *simulated[index].view.resourceCost,
              *simulated[policyBestIndex].view.resourceCost, selectionPolicy))
        policyBestIndex = index;
    unsigned removalIndex = simulated.size() - 1;
    if (removalIndex == policyBestIndex)
      --removalIndex;
    simulated.erase(simulated.begin() + removalIndex);
  }
  plan.retained = llvm::any_of(
      simulated, [](const SimulatedEntry &entry) { return entry.isCandidate; });
  if (!plan.retained)
    plan.dominatedIndices.clear();
  return plan;
}

static bool
wouldRetainParetoCandidate(const PreTargetWholeVariant &candidate,
                           llvm::ArrayRef<AcceptedWholeVariant> frontier,
                           const TargetStaticSelectionPolicy &selectionPolicy) {
  return planParetoInsertion(getParetoCandidateView(candidate), frontier,
                             selectionPolicy)
      .retained;
}

static bool
insertParetoCandidate(AcceptedWholeVariant candidate,
                      llvm::SmallVectorImpl<AcceptedWholeVariant> &frontier,
                      const TargetStaticSelectionPolicy &selectionPolicy) {
  ParetoInsertionPlan plan = planParetoInsertion(
      getParetoCandidateView(candidate), frontier, selectionPolicy);
  if (!plan.retained)
    return false;
  for (unsigned index : llvm::reverse(plan.dominatedIndices))
    frontier.erase(frontier.begin() + index);
  frontier.push_back(std::move(candidate));
  llvm::sort(frontier, [](const AcceptedWholeVariant &lhs,
                          const AcceptedWholeVariant &rhs) {
    return hasEarlierStaticPolicyOrder(getParetoCandidateView(lhs),
                                       getParetoCandidateView(rhs));
  });
  if (frontier.size() > kWholeVariantParetoLimit) {
    unsigned policyBestIndex = 0;
    for (unsigned index = 1; index < frontier.size(); ++index)
      if (isResourceCostPreferredOver(frontier[index].resourceCost,
                                      frontier[policyBestIndex].resourceCost,
                                      selectionPolicy))
        policyBestIndex = index;
    unsigned removalIndex = frontier.size() - 1;
    if (removalIndex == policyBestIndex)
      --removalIndex;
    frontier.erase(frontier.begin() + removalIndex);
  }
  return true;
}

static std::string summarizeAttemptFailure(llvm::ArrayRef<size_t> indices,
                                           llvm::StringRef gate,
                                           llvm::StringRef diagnostics) {
  std::string summary;
  llvm::raw_string_ostream os(summary);
  os << "candidates=[";
  for (auto [rank, index] : llvm::enumerate(indices)) {
    if (rank)
      os << ",";
    os << index;
  }
  os << "] gate=" << gate;
  diagnostics = diagnostics.trim();
  if (!diagnostics.empty()) {
    constexpr size_t maxDiagnosticBytes = 512;
    os << " diagnostic=" << diagnostics.take_front(maxDiagnosticBytes);
    if (diagnostics.size() > maxDiagnosticBytes)
      os << "...";
  }
  return summary;
}

} // namespace

static mlir::FailureOr<AcceptedProductionAndBaseline>
selectAcceptedWholeVariants(
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionMode selectionMode, bool retainReservedBaseline,
    WholeVariantSelectionStatistics *statistics) {
  if (program.logicalRankCount != executionConfig.getRankCount()) {
    diagnostics << "wafer-compile: typed program rank domain does not match "
                   "whole-variant ExecutionConfig\n";
    return mlir::failure();
  }
  std::vector<RankVariantMetadataFrontier> frontierMetadata;
  frontierMetadata.reserve(frontiers.size());
  for (const RankVariantFrontier &frontier : frontiers) {
    RankVariantMetadataFrontier metadata;
    metadata.reserve(frontier.size());
    for (const RankVariantCandidate &candidate : frontier)
      metadata.push_back({candidate.stableOrdinal, candidate.artifactKind,
                          candidate.reservedBaseline, candidate.bufferingKind,
                          candidate.bufferingPlanOrdinal,
                          candidate.workerPlacementKind,
                          candidate.workerPlacementPlanOrdinal});
    frontierMetadata.push_back(std::move(metadata));
  }
  WholeVariantAttemptPlan attemptPlan = buildWholeVariantAttemptPlan(
      frontierMetadata, executionConfig.getRankCount());
  if (attemptPlan.failure == WholeVariantAttemptPlanFailure::CandidateDomain) {
    diagnostics << "wafer-compile: rank scheduling frontiers do not form the "
                   "complete canonical rank domain\n";
    return mlir::failure();
  }
  for (auto [rank, requiredIndices] :
       llvm::enumerate(attemptPlan.requiredModuleIndices))
    for (size_t index : requiredIndices)
      if (rank >= frontiers.size() || index >= frontiers[rank].size() ||
          !frontiers[rank][index].module) {
        diagnostics
            << "wafer-compile: rank scheduling frontiers do not form the "
               "complete canonical rank domain\n";
        return mlir::failure();
      }

  mlir::MLIRContext *context = nullptr;
  for (const RankVariantFrontier &frontier : frontiers)
    for (const RankVariantCandidate &candidate : frontier)
      if (candidate.module && !context)
        context = candidate.module.get().getContext();
      else if (candidate.module &&
               candidate.module.get().getContext() != context) {
        diagnostics << "wafer-compile: rank scheduling frontiers do not share "
                       "the executable-bundle owner context\n";
        return mlir::failure();
      }
  if (!context) {
    diagnostics << "wafer-compile: rank scheduling frontiers do not form the "
                   "complete canonical rank domain\n";
    return mlir::failure();
  }

  if (attemptPlan.failure == WholeVariantAttemptPlanFailure::ReservedBaseline) {
    diagnostics << "wafer-compile: every rank frontier must contain exactly "
                   "one reserved baseline candidate\n";
    return mlir::failure();
  }

  std::set<std::vector<size_t>> attemptedCandidateIndices;

  llvm::SmallVector<std::string, kReportedAttemptLimit> failures;
  auto recordFailure = [&](llvm::ArrayRef<size_t> candidateIndices,
                           llvm::StringRef failureGate,
                           llvm::StringRef capturedDiagnostics) {
    std::string summary = summarizeAttemptFailure(candidateIndices, failureGate,
                                                  capturedDiagnostics);
    if (failures.size() < kReportedAttemptLimit)
      failures.push_back(std::move(summary));
    else
      failures.back() = std::move(summary);
  };
  auto attemptPreTarget = [&](const std::vector<size_t> &candidateIndices)
      -> mlir::FailureOr<PreTargetWholeVariant> {
    if (!attemptedCandidateIndices.insert(candidateIndices).second)
      return mlir::failure();
    std::string capturedDiagnostics;
    std::string failureGate = "unknown";
    mlir::FailureOr<PreTargetWholeVariant> result = mlir::failure();
    {
      mlir::ScopedDiagnosticHandler handler(
          context, [&](mlir::Diagnostic &diagnostic) {
            llvm::raw_string_ostream os(capturedDiagnostics);
            diagnostic.print(os);
            os << "\n";
            return mlir::success();
          });
      result = tryPreTargetCombination(candidateIndices, frontiers, program,
                                       executionConfig, failureGate);
    }
    if (mlir::succeeded(result))
      return result;
    recordFailure(candidateIndices, failureGate, capturedDiagnostics);
    return mlir::failure();
  };
  auto targetGate = [&](const std::vector<size_t> &candidateIndices,
                        PreTargetWholeVariant candidate)
      -> mlir::FailureOr<AcceptedWholeVariant> {
    std::string capturedDiagnostics;
    std::string failureGate = "unknown";
    mlir::FailureOr<AcceptedWholeVariant> result = mlir::failure();
    {
      mlir::ScopedDiagnosticHandler handler(
          context, [&](mlir::Diagnostic &diagnostic) {
            llvm::raw_string_ostream os(capturedDiagnostics);
            diagnostic.print(os);
            os << "\n";
            return mlir::success();
          });
      result = runTargetGate(std::move(candidate), executionConfig, statistics,
                             failureGate);
    }
    if (mlir::succeeded(result))
      return result;
    recordFailure(candidateIndices, failureGate, capturedDiagnostics);
    return mlir::failure();
  };
  auto attemptFullyGated = [&](const std::vector<size_t> &candidateIndices)
      -> mlir::FailureOr<AcceptedWholeVariant> {
    mlir::FailureOr<PreTargetWholeVariant> preTarget =
        attemptPreTarget(candidateIndices);
    if (mlir::failed(preTarget))
      return mlir::failure();
    return targetGate(candidateIndices, std::move(*preTarget));
  };

  // The reserved baseline has its own allowance and must pass every late gate
  // before any optimization budget is consumed. Keep the accepted baseline as
  // the conservative fallback while alternatives are evaluated.
  mlir::FailureOr<AcceptedWholeVariant> baseline =
      attemptFullyGated(attemptPlan.reservedBaselineIndices);
  if (mlir::failed(baseline)) {
    diagnostics << "wafer-compile: reserved all-baseline variant failed "
                   "whole-variant acceptance\n";
    for (const std::string &failure : failures)
      diagnostics << "  - " << failure << "\n";
    return mlir::failure();
  }
  AcceptedWholeVariant baselineAccepted = std::move(*baseline);
  if (selectionMode == WholeVariantSelectionMode::ReservedBaseline)
    return AcceptedProductionAndBaseline{std::move(baselineAccepted),
                                         std::nullopt,
                                         /*productionIsReservedBaseline=*/true};
  const bool characterize =
      isCollectiveCharacterizationSelection(selectionMode);
  const bool qualifyNoCResidentAllReduceRing =
      selectionMode ==
      WholeVariantSelectionMode::QualifyNoCResidentAllReduceRing;
  const bool qualifyStaticFixedSlot =
      selectionMode == WholeVariantSelectionMode::QualifyStaticFixedSlot;
  const bool qualifyWorkerPlacement =
      selectionMode == WholeVariantSelectionMode::QualifyWorkerPlacement;
  const bool qualifyNoCResidentFixedSlotWorker =
      selectionMode ==
      WholeVariantSelectionMode::QualifyNoCResidentFixedSlotWorker;
  const TargetStaticSelectionPolicy selectionPolicy =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default).staticSelection;
  const analysis::TargetScheduleCostPolicy scheduleCostPolicy =
      analysis::getTargetScheduleCostPolicy(
          executionConfig.getTargetProfileId());

  llvm::SmallVector<AcceptedWholeVariant, kWholeVariantParetoLimit>
      paretoFrontier;
  auto explainStaticFixedSlotQualification =
      [](const PreTargetWholeVariant &candidate) {
        for (auto [rank, executable] : llvm::enumerate(candidate.ranks))
          if (llvm::Error error =
                  verifyStaticFixedSlotQualificationEvidence(executable))
            return ("rank " + llvm::Twine(rank) + ": " +
                    llvm::toString(std::move(error)))
                .str();
        return std::string("typed fixed-slot candidate identity is incomplete");
      };
  auto retainPreTarget = [&](const std::vector<size_t> &candidateIndices) {
    mlir::FailureOr<PreTargetWholeVariant> preTarget =
        attemptPreTarget(candidateIndices);
    if (mlir::failed(preTarget))
      return;
    if (characterize &&
        !matchesCollectiveCharacterization(preTarget->ranks, selectionMode))
      return;
    if (qualifyNoCResidentAllReduceRing &&
        (llvm::any_of(preTarget->selectedReservedBaselines,
                      [](bool reserved) { return reserved; }) ||
         !hasBoundaryOnlyDDRMovement(preTarget->ranks)))
      return;
    if (qualifyStaticFixedSlot &&
        !matchesStaticFixedSlotQualification(*preTarget)) {
      recordFailure(candidateIndices, "static-fixed-slot-qualification",
                    explainStaticFixedSlotQualification(*preTarget));
      return;
    }
    if (qualifyWorkerPlacement &&
        !matchesWorkerPlacementQualification(*preTarget)) {
      recordFailure(candidateIndices, "worker-placement-qualification", "");
      return;
    }
    if (qualifyNoCResidentFixedSlotWorker) {
      if (!matchesStaticFixedSlotQualification(*preTarget)) {
        recordFailure(candidateIndices, "static-fixed-slot-qualification",
                      explainStaticFixedSlotQualification(*preTarget));
        return;
      }
      if (!matchesWorkerPlacementQualification(*preTarget)) {
        recordFailure(candidateIndices, "worker-placement-qualification", "");
        return;
      }
      if (!hasBoundaryOnlyDDRMovement(preTarget->ranks)) {
        recordFailure(candidateIndices, "noc-resident-ddr-qualification", "");
        return;
      }
      if (!hasAllRankDirectDTEAndBidirectionalTuple(preTarget->ranks)) {
        recordFailure(candidateIndices, "all-rank-direct-dte-qualification",
                      "");
        return;
      }
    }
    if (selectionMode == WholeVariantSelectionMode::Production &&
        !hasNormalProductionSchedulingEvidence(*preTarget, baselineAccepted))
      return;
    if (selectionMode == WholeVariantSelectionMode::Production) {
      analysis::NoCTradeoffScheduleContext profitabilitySchedule;
      if (preTarget->hasSchedulingCapabilityQuery &&
          preTarget->schedulingProfitability.overlap ==
              TargetSchedulingOverlapEvidence::QualifiedOverlap)
        profitabilitySchedule.candidate =
            analysis::StaticCrossResourceSchedule::PipelinedSteadyState;
      analysis::NoCTradeoffProfitability profitability =
          analysis::analyzeNoCTradeoffProfitability(
              preTarget->resourceCost, baselineAccepted.resourceCost,
              scheduleCostPolicy, profitabilitySchedule);
      if (profitability.decision !=
          analysis::NoCTradeoffDecision::NotApplicable) {
        if (statistics)
          ++statistics->noCProfitabilityEvaluations;
        switch (profitability.decision) {
        case analysis::NoCTradeoffDecision::NotApplicable:
          llvm_unreachable("handled above");
        case analysis::NoCTradeoffDecision::Reject:
          if (statistics)
            ++statistics->noCProfitabilityRejected;
          return;
        case analysis::NoCTradeoffDecision::Indeterminate:
          if (statistics)
            ++statistics->noCProfitabilityIndeterminate;
          return;
        case analysis::NoCTradeoffDecision::EstimatedBenefit:
          if (statistics)
            ++statistics->noCProfitabilityEstimated;
          break;
        case analysis::NoCTradeoffDecision::ProvenBenefit:
          if (statistics)
            ++statistics->noCProfitabilityProven;
          break;
        }
      }
    }
    if (!wouldRetainParetoCandidate(*preTarget, paretoFrontier,
                                    selectionPolicy))
      return;
    mlir::FailureOr<AcceptedWholeVariant> accepted =
        targetGate(candidateIndices, std::move(*preTarget));
    if (mlir::failed(accepted))
      return;
    const bool retained = insertParetoCandidate(
        std::move(*accepted), paretoFrontier, selectionPolicy);
    assert(retained &&
           "target gate cannot change Pareto facts or frontier membership");
  };

  // Replay the exact original bounded Cartesian/coordinated sequence over the
  // original slot indices. Slots belonging only to correspondence-mismatched
  // tuples remain metadata-only and fail before any module is inspected.
  for (const std::vector<size_t> &candidateIndices :
       attemptPlan.optimizedCandidateIndices)
    retainPreTarget(candidateIndices);

  if (characterize) {
    std::optional<AcceptedWholeVariant> selected;
    if (!qualifyNoCResidentAllReduceRing &&
        matchesCollectiveCharacterization(baselineAccepted.ranks,
                                          selectionMode))
      selected.emplace(std::move(baselineAccepted));
    for (AcceptedWholeVariant &candidate : paretoFrontier)
      if (!selected || isPreferredOver(candidate, *selected, selectionPolicy))
        selected.emplace(std::move(candidate));
    if (!selected) {
      diagnostics << "wafer-compile: no fully accepted whole variant matches "
                     "test-only collective characterization alternative '"
                  << getCollectiveCharacterizationAlternative(selectionMode)
                  << "'\n";
      return mlir::failure();
    }
    return AcceptedProductionAndBaseline{
        std::move(*selected), std::nullopt,
        /*productionIsReservedBaseline=*/false};
  }
  if (qualifyStaticFixedSlot || qualifyWorkerPlacement ||
      qualifyNoCResidentFixedSlotWorker) {
    std::optional<AcceptedWholeVariant> selected;
    for (AcceptedWholeVariant &candidate : paretoFrontier)
      if (!selected || isPreferredOver(candidate, *selected, selectionPolicy))
        selected.emplace(std::move(candidate));
    if (!selected) {
      diagnostics << "wafer-compile: no fully accepted whole variant matches "
                     "test-only "
                  << (qualifyNoCResidentFixedSlotWorker
                          ? "NoC-resident static fixed-slot worker-placement"
                      : qualifyStaticFixedSlot ? "static fixed-slot"
                                               : "worker-placement")
                  << " qualification\n";
      for (const std::string &failure : failures)
        diagnostics << "  - " << failure << "\n";
      return mlir::failure();
    }
    return AcceptedProductionAndBaseline{
        std::move(*selected), std::nullopt,
        /*productionIsReservedBaseline=*/false};
  }
  AcceptedWholeVariant *selected = &baselineAccepted;
  for (AcceptedWholeVariant &candidate : paretoFrontier)
    if (isPreferredOver(candidate, *selected, selectionPolicy))
      selected = &candidate;
  if (selected == &baselineAccepted)
    return AcceptedProductionAndBaseline{std::move(baselineAccepted),
                                         std::nullopt,
                                         /*productionIsReservedBaseline=*/true};

  std::optional<AcceptedWholeVariant> retainedBaseline;
  if (retainReservedBaseline)
    retainedBaseline.emplace(std::move(baselineAccepted));
  return AcceptedProductionAndBaseline{std::move(*selected),
                                       std::move(retainedBaseline),
                                       /*productionIsReservedBaseline=*/false};
}

mlir::FailureOr<AcceptedWholeVariant> selectAcceptedWholeVariant(
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionMode selectionMode,
    WholeVariantSelectionStatistics *statistics) {
  mlir::FailureOr<AcceptedProductionAndBaseline> selected =
      selectAcceptedWholeVariants(frontiers, program, executionConfig,
                                  diagnostics, selectionMode,
                                  /*retainReservedBaseline=*/false, statistics);
  if (mlir::failed(selected))
    return mlir::failure();
  return std::move(selected->production);
}

mlir::FailureOr<AcceptedProductionAndBaseline>
selectAcceptedProductionAndBaseline(
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionStatistics *statistics) {
  return selectAcceptedWholeVariants(
      frontiers, program, executionConfig, diagnostics,
      WholeVariantSelectionMode::Production,
      /*retainReservedBaseline=*/true, statistics);
}

} // namespace wafer::compiler::detail
