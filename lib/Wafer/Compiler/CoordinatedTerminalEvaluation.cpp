//===- CoordinatedTerminalEvaluation.cpp - Exact all-rank gate -----------===//

#include "CoordinatedTerminalEvaluation.h"

#include "ScheduledRankFinalization.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/TargetSchedulingCapability.h"
#include "Wafer/Transforms/PhysicalDataflow.h"
#include "Wafer/Transforms/SoftwarePipelining.h"
#include "Wafer/Transforms/WorkerPlacement.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

namespace wafer::compiler::detail {
namespace {

static llvm::SmallVector<mlir::ModuleOp, 16> getModuleViews(
    llvm::ArrayRef<mlir::OwningOpRef<mlir::ModuleOp>> modules) {
  llvm::SmallVector<mlir::ModuleOp, 16> views;
  views.reserve(modules.size());
  for (const mlir::OwningOpRef<mlir::ModuleOp> &module : modules)
    views.push_back(*module);
  return views;
}

static mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>>
cloneCanonicalParents(llvm::ArrayRef<mlir::ModuleOp> parents) {
  if (parents.empty())
    return mlir::failure();
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> clones;
  clones.reserve(parents.size());
  for (mlir::ModuleOp parent : parents) {
    if (!parent || containsTileDataflowOperations(parent.getOperation()))
      return mlir::failure();
    clones.push_back(mlir::cast<mlir::ModuleOp>(parent->clone()));
    clearRankCandidatePhysicalFacts(*clones.back());
    if (mlir::failed(mlir::verify(*clones.back())))
      return mlir::failure();
  }
  return clones;
}

static bool isSupportedSchedulingAction(
    mlir::ModuleOp module, TargetSchedulingMechanism mechanism,
    const TargetSchedulingCapabilityRegistry &registry) {
  llvm::Expected<TargetSchedulingWindowQuery> query =
      analyzeTargetSchedulingWindow(module, mechanism);
  if (!query) {
    llvm::consumeError(query.takeError());
    return false;
  }
  llvm::Expected<TargetSchedulingWindowDecision> decision =
      registry.query(*query);
  if (!decision) {
    llvm::consumeError(decision.takeError());
    return false;
  }
  return decision->legality == TargetSchedulingCapabilityState::Supported;
}

static mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>>
deriveReadyOrderAction(llvm::ArrayRef<mlir::ModuleOp> parents) {
  mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>> clones =
      cloneCanonicalParents(parents);
  if (mlir::failed(clones))
    return mlir::failure();
  bool anyMoved = false;
  for (mlir::OwningOpRef<mlir::ModuleOp> &clone : *clones) {
    anyMoved |= scheduleIndependentInstructionsByReadyOrder(
                    clone->getOperation()) != 0;
    if (mlir::failed(normalizeMinimumNCCJoins(*clone)) ||
        mlir::failed(mlir::verify(*clone)))
      return mlir::failure();
  }
  if (!anyMoved)
    return mlir::failure();
  return clones;
}

static mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>>
deriveFixedSlotAction(llvm::ArrayRef<mlir::ModuleOp> parents,
                      uint32_t structuralOrdinal,
                      const TargetSchedulingCapabilityRegistry &registry) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> siblings;
  siblings.reserve(parents.size());
  for (mlir::ModuleOp parent : parents) {
    llvm::SmallVector<mlir::scf::ForOp, 8> loops;
    parent.walk([&](mlir::scf::ForOp loop) { loops.push_back(loop); });
    if (structuralOrdinal >= loops.size())
      return mlir::failure();

    std::string failureReason;
    mlir::FailureOr<StaticFixedSlotPipelineCandidate> candidate =
        mlir::failure();
    {
      mlir::ScopedDiagnosticHandler handler(
          parent.getContext(),
          [](mlir::Diagnostic &) { return mlir::success(); });
      candidate = deriveStaticFixedSlotPipelineCandidate(
          parent, loops[structuralOrdinal], &failureReason);
    }
    if (mlir::failed(candidate) || candidate->stageCount < 2 ||
        candidate->slotAllocationCount < 2 ||
        !isSupportedSchedulingAction(
            *candidate->module, TargetSchedulingMechanism::StaticFixedSlot,
            registry) ||
        mlir::failed(mlir::verify(*candidate->module)))
      return mlir::failure();
    siblings.push_back(std::move(candidate->module));
  }
  return siblings;
}

static mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>>
deriveWorkerAction(llvm::ArrayRef<mlir::ModuleOp> parents,
                   RankBufferingKind bufferingKind,
                   const TargetSchedulingCapabilityRegistry &registry) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> siblings;
  siblings.reserve(parents.size());
  for (mlir::ModuleOp parent : parents) {
    std::string failureReason;
    mlir::FailureOr<NCCWorkerPlacementCandidate> candidate = mlir::failure();
    {
      mlir::ScopedDiagnosticHandler handler(
          parent.getContext(),
          [](mlir::Diagnostic &) { return mlir::success(); });
      candidate =
          deriveDisjointNCCWorkerPlacementCandidate(parent, &failureReason);
    }
    if (mlir::failed(candidate) ||
        llvm::popcount(candidate->participantMask) < 2)
      return mlir::failure();

    bool hasDirectDTE = false;
    candidate->module->walk([&](WaferInstructionOpInterface instruction) {
      hasDirectDTE |= instruction.getInstructionFamily() == InstrFamily::DTE;
    });
    TargetSchedulingMechanism mechanism =
        bufferingKind == RankBufferingKind::StaticFixedSlot
            ? TargetSchedulingMechanism::StaticFixedSlot
        : hasDirectDTE ? TargetSchedulingMechanism::DirectDTEOverlap
                       : TargetSchedulingMechanism::WorkerPlacement;
    if (!isSupportedSchedulingAction(*candidate->module, mechanism, registry) ||
        mlir::failed(mlir::verify(*candidate->module)))
      return mlir::failure();
    siblings.push_back(std::move(candidate->module));
  }
  return siblings;
}

static uint32_t getCommonStructuralLoopCount(
    llvm::ArrayRef<mlir::ModuleOp> parents) {
  uint32_t common = kMaximumCoordinatedFixedSlotActions;
  for (mlir::ModuleOp parent : parents) {
    uint64_t count = 0;
    parent.walk([&](mlir::scf::ForOp) { ++count; });
    common = std::min<uint32_t>(
        common, static_cast<uint32_t>(std::min<uint64_t>(
                    count, kMaximumCoordinatedFixedSlotActions)));
  }
  return common;
}

static bool chargeActionMaterialization(uint64_t rankCount, uint64_t &work) {
  if (rankCount > std::numeric_limits<uint64_t>::max() - work)
    return false;
  work += rankCount;
  return true;
}

struct TerminalEvaluationWork {
  uint64_t tileLowerings = 0;
  uint64_t actionMaterializations = 0;
  uint64_t rankFinalizations = 0;
  uint64_t exactActionAttempts = 0;
};

static CoordinatedWorkEstimate
getActualTerminalWork(const TerminalEvaluationWork &work, uint64_t rankCount) {
  CoordinatedWorkEstimate actual;
  actual.set(CoordinatedWorkKind::TileToInstrLowering, work.tileLowerings);
  actual.set(CoordinatedWorkKind::TerminalScheduleAction,
             work.actionMaterializations);
  actual.set(CoordinatedWorkKind::SPMAllocationProblem,
             work.rankFinalizations);
  actual.set(CoordinatedWorkKind::DDRAllocationDomain,
             work.exactActionAttempts * rankCount);
  actual.set(CoordinatedWorkKind::TransportValidation,
             work.exactActionAttempts);
  actual.set(CoordinatedWorkKind::ABIValidation,
             work.exactActionAttempts * rankCount);
  return actual;
}

} // namespace

mlir::FailureOr<std::vector<CoordinatedTerminalScheduleAction>>
deriveCoordinatedTerminalScheduleActions(
    llvm::ArrayRef<mlir::ModuleOp> canonicalInstrParents,
    const OptimizationConfig &optimizations, uint64_t *materializationWork) {
  if (materializationWork)
    *materializationWork = 0;
  if (canonicalInstrParents.empty())
    return mlir::failure();

  const uint64_t rankCount = canonicalInstrParents.size();
  uint64_t work = 0;
  mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>> baseline =
      cloneCanonicalParents(canonicalInstrParents);
  if (mlir::failed(baseline) ||
      !chargeActionMaterialization(rankCount, work))
    return mlir::failure();

  std::vector<CoordinatedTerminalScheduleAction> actions;
  actions.reserve(kMaximumCoordinatedTerminalActions);
  actions.push_back({/*stableOrdinal=*/0,
                     RankArtifactKind::Spill,
                     RankBufferingKind::Single,
                     /*bufferingPlanOrdinal=*/0,
                     RankWorkerPlacementKind::Unplaced,
                     /*workerPlacementPlanOrdinal=*/0,
                     std::move(*baseline)});

  llvm::SmallVector<size_t, 2> unplacedOrderParents = {0};
  if (optimizations.isEnabled(OptimizationKind::ReadyOrderScheduling)) {
    llvm::SmallVector<mlir::ModuleOp, 16> parents =
        getModuleViews(actions.front().rankModules);
    if (!chargeActionMaterialization(rankCount, work))
      return mlir::failure();
    mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>> ready =
        deriveReadyOrderAction(parents);
    if (mlir::succeeded(ready)) {
      unplacedOrderParents.push_back(actions.size());
      actions.push_back({static_cast<uint32_t>(actions.size()),
                         RankArtifactKind::SpillReady,
                         RankBufferingKind::Single,
                         /*bufferingPlanOrdinal=*/0,
                         RankWorkerPlacementKind::Unplaced,
                         /*workerPlacementPlanOrdinal=*/0,
                         std::move(*ready)});
    }
  }

  llvm::Expected<TargetSchedulingCapabilityRegistry> registry =
      getTargetSchedulingCapabilityRegistry();
  if (!registry) {
    llvm::consumeError(registry.takeError());
    if (materializationWork)
      *materializationWork = work;
    return actions;
  }

  if (optimizations.isEnabled(OptimizationKind::StaticFixedSlotBuffering)) {
    llvm::SmallVector<size_t, 2> fixedParents = unplacedOrderParents;
    for (size_t parentIndex : fixedParents) {
      llvm::SmallVector<mlir::ModuleOp, 16> parents =
          getModuleViews(actions[parentIndex].rankModules);
      uint32_t loopCount = getCommonStructuralLoopCount(parents);
      for (uint32_t structuralOrdinal = 0; structuralOrdinal < loopCount;
           ++structuralOrdinal) {
        if (!chargeActionMaterialization(rankCount, work))
          return mlir::failure();
        mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>> fixed =
            deriveFixedSlotAction(parents, structuralOrdinal, *registry);
        if (mlir::failed(fixed))
          continue;
        actions.push_back(
            {static_cast<uint32_t>(actions.size()),
             actions[parentIndex].artifactKind,
             RankBufferingKind::StaticFixedSlot,
             structuralOrdinal + 1,
             RankWorkerPlacementKind::Unplaced,
             /*workerPlacementPlanOrdinal=*/0,
             std::move(*fixed)});
      }
    }
  }

  if (optimizations.isEnabled(OptimizationKind::DisjointWorkerPlacement)) {
    const size_t unplacedActionCount = actions.size();
    for (size_t parentIndex = 0; parentIndex < unplacedActionCount;
         ++parentIndex) {
      if (actions[parentIndex].workerPlacementKind !=
          RankWorkerPlacementKind::Unplaced)
        continue;
      llvm::SmallVector<mlir::ModuleOp, 16> parents =
          getModuleViews(actions[parentIndex].rankModules);
      if (!chargeActionMaterialization(rankCount, work))
        return mlir::failure();
      mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>> placed =
          deriveWorkerAction(parents, actions[parentIndex].bufferingKind,
                             *registry);
      if (mlir::failed(placed))
        continue;
      actions.push_back(
          {static_cast<uint32_t>(actions.size()),
           actions[parentIndex].artifactKind,
           actions[parentIndex].bufferingKind,
           actions[parentIndex].bufferingPlanOrdinal,
           RankWorkerPlacementKind::DisjointComponents,
           /*workerPlacementPlanOrdinal=*/1,
           std::move(*placed)});
    }
  }

  if (actions.size() > kMaximumCoordinatedTerminalActions)
    return mlir::failure();
  if (materializationWork)
    *materializationWork = work;
  return actions;
}

mlir::FailureOr<std::vector<FullyGatedCoordinatedVariant>>
evaluateCoordinatedTileVariant(
    const CoordinatedTileVariant &tileVariant,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    const OptimizationConfig &optimizations, CoordinatedWorkLedger &ledger,
    llvm::raw_ostream &diagnostics, CoordinatedTerminalFailure &failure,
    WholeVariantSelectionStatistics *statistics) {
  failure = {};
  const int64_t rankCount = executionConfig.getRankCount();
  if (ledger.getRankCount() != rankCount ||
      mlir::failed(verifyCoordinatedTileVariant(tileVariant, rankCount))) {
    failure.kind = CoordinatedTerminalFailureKind::RankDomain;
    failure.gate = "rank-domain";
    return mlir::failure();
  }

  TerminalEvaluationWork work;
  bool terminalWorkClosed = false;
  auto closeTerminalWork = [&]() {
    if (terminalWorkClosed)
      return mlir::success();
    CoordinatedWorkEstimate actual =
        getActualTerminalWork(work, static_cast<uint64_t>(rankCount));
    if (mlir::failed(ledger.completeTerminalAction(
            tileVariant.terminalReservation, actual)))
      return mlir::failure();
    terminalWorkClosed = true;
    return mlir::success();
  };
  auto closeAndReject = [&]()
      -> mlir::FailureOr<std::vector<FullyGatedCoordinatedVariant>> {
    if (mlir::failed(closeTerminalWork())) {
      failure.kind = CoordinatedTerminalFailureKind::WorkLedger;
      failure.logicalRank = -1;
      failure.gate = "work-ledger";
    }
    return mlir::failure();
  };
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> canonicalOwners;
  llvm::SmallVector<mlir::ModuleOp, 16> canonicalParents;
  canonicalOwners.reserve(tileVariant.ranks.size());
  canonicalParents.reserve(tileVariant.ranks.size());
  for (const CoordinatedRankTileProgram &rank : tileVariant.ranks) {
    canonicalOwners.push_back(
        mlir::cast<mlir::ModuleOp>(rank.module.get()->clone()));
    bool loweredTile = containsTileDataflowOperations(
        canonicalOwners.back()->getOperation());
    work.tileLowerings += loweredTile;
    std::string loweringFailure;
    if (mlir::failed(convertTileRegionToInstrModule(*canonicalOwners.back(),
                                                    &loweringFailure))) {
      failure.kind = CoordinatedTerminalFailureKind::RankFinalization;
      failure.logicalRank = rank.logicalRank;
      failure.gate = "tile-to-instr";
      if (!loweringFailure.empty())
        diagnostics << loweringFailure << '\n';
      return closeAndReject();
    }
    clearRankCandidatePhysicalFacts(*canonicalOwners.back());
    if (containsTileDataflowOperations(canonicalOwners.back()->getOperation()) ||
        mlir::failed(mlir::verify(*canonicalOwners.back()))) {
      failure.kind = CoordinatedTerminalFailureKind::RankFinalization;
      failure.logicalRank = rank.logicalRank;
      failure.gate = "canonical-instr-parent";
      return closeAndReject();
    }
    canonicalParents.push_back(*canonicalOwners.back());
  }

  mlir::FailureOr<std::vector<CoordinatedTerminalScheduleAction>> actions =
      deriveCoordinatedTerminalScheduleActions(
          canonicalParents, optimizations, &work.actionMaterializations);
  if (mlir::failed(actions) || actions->empty()) {
    failure.kind = CoordinatedTerminalFailureKind::RankFinalization;
    failure.gate = "terminal-action-materialization";
    return closeAndReject();
  }

  std::vector<FullyGatedCoordinatedVariant> fullyGated;
  fullyGated.reserve(actions->size());
  bool sawSPMAllocationFailure = false;
  int64_t spmFailureLogicalRank = -1;
  for (CoordinatedTerminalScheduleAction &action : *actions) {
    const bool reservedBaseline =
        tileVariant.reservedBaseline && action.stableOrdinal == 0;
    std::vector<RankVariantCandidate> terminalRanks;
    terminalRanks.reserve(action.rankModules.size());
    bool rankFinalizationFailed = false;
    for (auto [rankIndex, module] : llvm::enumerate(action.rankModules)) {
      ++work.rankFinalizations;
      std::vector<ScheduledRankCandidate> rankFrontier;
      rankFrontier.emplace_back(
          std::move(module), tileVariant.stableSemanticOrdinal,
          action.artifactKind, reservedBaseline, action.bufferingKind,
          action.bufferingPlanOrdinal, action.workerPlacementKind,
          action.workerPlacementPlanOrdinal,
          /*frontierOrderOrdinal=*/action.stableOrdinal,
          tileVariant.ranks[rankIndex].selectedTileIR);
      mlir::FailureOr<std::vector<FinalizedRankCandidate>> finalized =
          mlir::failure();
      RankFinalizationFailure rankFailure;
      std::string capturedDiagnostics;
      {
        mlir::ScopedDiagnosticHandler handler(
            canonicalParents[rankIndex].getContext(),
            [&](mlir::Diagnostic &diagnostic) {
              llvm::raw_string_ostream stream(capturedDiagnostics);
              diagnostic.print(stream);
              stream << '\n';
              return mlir::success();
            });
        finalized = finalizeScheduledRankCandidateFrontier(
            std::move(rankFrontier),
            /*requireReservedBaseline=*/reservedBaseline,
            RankCompletionPolicy::RebuildFromCurrentEffects, &rankFailure);
      }
      if (mlir::failed(finalized) || finalized->size() != 1) {
        diagnostics << capturedDiagnostics;
        failure.kind = rankFailure.kind ==
                               RankFinalizationFailureKind::SPMAllocation
                           ? CoordinatedTerminalFailureKind::SPMAllocation
                           : CoordinatedTerminalFailureKind::RankFinalization;
        failure.logicalRank = tileVariant.ranks[rankIndex].logicalRank;
        failure.gate = "rank-finalization";
        if (failure.kind == CoordinatedTerminalFailureKind::SPMAllocation) {
          sawSPMAllocationFailure = true;
          spmFailureLogicalRank = failure.logicalRank;
        }
        rankFinalizationFailed = true;
        break;
      }
      FinalizedRankCandidate &candidate = finalized->front();
      terminalRanks.push_back(
          {std::move(candidate.module), candidate.stableOrdinal,
           candidate.artifactKind, candidate.reservedBaseline,
           candidate.bufferingKind, candidate.bufferingPlanOrdinal,
           candidate.workerPlacementKind,
           candidate.workerPlacementPlanOrdinal, candidate.selectedTileIR});
    }
    if (rankFinalizationFailed) {
      if (reservedBaseline)
        return closeAndReject();
      continue;
    }

    ++work.exactActionAttempts;
    std::string exactGate;
    mlir::FailureOr<AcceptedWholeVariant> accepted =
        evaluateFullyGatedWholeVariant(std::move(terminalRanks), program,
                                       executionConfig, diagnostics, &exactGate,
                                       statistics);
    if (mlir::failed(accepted)) {
      failure.kind = CoordinatedTerminalFailureKind::WholeVariantExactGate;
      failure.gate = std::move(exactGate);
      if (reservedBaseline)
        return closeAndReject();
      continue;
    }
    fullyGated.emplace_back(tileVariant.stableSemanticOrdinal,
                            reservedBaseline, std::move(*accepted),
                            action.stableOrdinal);
  }

  if (fullyGated.empty()) {
    if (sawSPMAllocationFailure) {
      failure.kind = CoordinatedTerminalFailureKind::SPMAllocation;
      failure.logicalRank = spmFailureLogicalRank;
      failure.gate = "spm-allocation";
    } else if (failure.kind == CoordinatedTerminalFailureKind::None) {
      failure.kind = CoordinatedTerminalFailureKind::WholeVariantExactGate;
      failure.gate = "no-terminal-survivor";
    }
    return closeAndReject();
  }

  if (mlir::failed(closeTerminalWork())) {
    failure.kind = CoordinatedTerminalFailureKind::WorkLedger;
    failure.gate = "work-ledger";
    return mlir::failure();
  }

  failure = {};
  return fullyGated;
}

} // namespace wafer::compiler::detail
