//===- ExecutableBundle.cpp - Static-rank executable bundle -------------===//

#include "ExecutableBundleInternal.h"

#include "AttentionImplementationAlternative.h"
#include "CompilationStatistics.h"
#include "CoordinatedDataflowSearch.h"
#include "CoordinatedExecutableFinalization.h"
#include "CoordinatedVariantSelection.h"
#include "NoCCommunicationAction.h"

#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

static void
printInstructionWorkMetric(llvm::raw_ostream &os,
                           const analysis::ScheduleCostMetric &metric) {
  os << analysis::stringifyScheduleCostKnowledge(metric.knowledge) << ':';
  if (metric.isKnown())
    os << metric.value;
  else
    os << analysis::stringifyScheduleCostReason(metric.reason);
}

static void
printInstructionExecutionCount(llvm::raw_ostream &os, llvm::StringRef name,
                               const analysis::InstructionExecutionCount &count,
                               bool alwaysPrintBounds = false) {
  os << ' ' << name << "_sites=";
  printInstructionWorkMetric(os, count.staticSites);
  os << ' ' << name << "_exact=";
  printInstructionWorkMetric(os, count.exactExecutions);
  auto equal = [](const analysis::ScheduleCostMetric &left,
                  const analysis::ScheduleCostMetric &right) {
    return left.value == right.value && left.knowledge == right.knowledge &&
           left.reason == right.reason;
  };
  if (!alwaysPrintBounds && equal(count.exactExecutions, count.lowerBound) &&
      equal(count.exactExecutions, count.upperBound))
    return;
  os << ' ' << name << "_lower=";
  printInstructionWorkMetric(os, count.lowerBound);
  os << ' ' << name << "_upper=";
  printInstructionWorkMetric(os, count.upperBound);
}

static void
printInstructionProgramWork(llvm::raw_ostream &os,
                            const analysis::InstructionProgramWork &work) {
  printInstructionExecutionCount(os, "instructions", work.instructions,
                                 /*alwaysPrintBounds=*/true);
  printInstructionExecutionCount(os, "events", work.asynchronousEvents);
  printInstructionExecutionCount(os, "rdma", work.rdmaIssues);
  printInstructionExecutionCount(os, "wdma", work.wdmaIssues);
  printInstructionExecutionCount(os, "tdma", work.tdmaIssues);
  printInstructionExecutionCount(os, "ct", work.ctIssues);
  printInstructionExecutionCount(os, "ne", work.neIssues);
  printInstructionExecutionCount(os, "dte", work.dteOperations);
  printInstructionExecutionCount(os, "gather_scatter",
                                 work.gatherScatterOperations);
  printInstructionExecutionCount(os, "dte_send", work.dteSendOperations);
  printInstructionExecutionCount(os, "dte_receive", work.dteReceiveOperations);
  printInstructionExecutionCount(os, "dte_wait", work.dteWaitOperations);
  printInstructionExecutionCount(os, "collective_dte",
                                 work.collectiveDTEIssues);
  printInstructionExecutionCount(os, "peer_dte", work.peerDTEIssues);
  printInstructionExecutionCount(os, "ncc_join", work.nccJoins,
                                 /*alwaysPrintBounds=*/true);
  printInstructionExecutionCount(os, "ncc_join_steady",
                                 work.steadyStateNCCJoins);
  printInstructionExecutionCount(os, "ncc_join_nonterminal",
                                 work.nonTerminalNCCJoins);
  printInstructionExecutionCount(os, "ncc_participant_wait",
                                 work.nccParticipantWaits);
  printInstructionExecutionCount(os, "ncc_participant_wait_steady",
                                 work.steadyStateNCCParticipantWaits);
  printInstructionExecutionCount(os, "ncc_participant_wait_nonterminal",
                                 work.nonTerminalNCCParticipantWaits);
  printInstructionExecutionCount(os, "intrinsic_ncc_drain",
                                 work.intrinsicNCCDrains);
}

static void
printInstructionProgramResources(llvm::raw_ostream &os,
                                 const analysis::InstructionProgramCost &cost) {
  auto print = [&](llvm::StringRef name,
                   const analysis::ScheduleCostMetric &metric) {
    os << ' ' << name << '=';
    printInstructionWorkMetric(os, metric);
  };
  print("ddr_read_bytes", cost.ddrReadBytes);
  print("ddr_write_bytes", cost.ddrWriteBytes);
  print("spm_movement_bytes", cost.spmMovementBytes);
  print("gather_scatter_bytes", cost.gatherScatterBytes);
  print("compute_npu_f16_bf16_ops", cost.compute.npuF16Bf16LogicalOps);
  print("compute_npu_other_ops", cost.compute.npuOtherLogicalOps);
  print("compute_vector_f16_bf16_ops", cost.compute.vectorF16Bf16LogicalOps);
  print("compute_vector_f32_ops", cost.compute.vectorF32LogicalOps);
  print("compute_vector_other_ops", cost.compute.vectorOtherLogicalOps);
  print("noc_transmit_bytes", cost.noc.aggregateTransmitBytes);
  print("noc_receive_bytes", cost.noc.aggregateReceiveBytes);
  print("dependency_depth", cost.dataDependencyDepth);
  print("spm_high_water_bytes", cost.spmHighWaterBytes);
  print("ddr_high_water_bytes", cost.ddrHighWaterBytes);
  print("spm_buffers", cost.compilerOwnedSPMBufferCount);
  print("ddr_buffers", cost.compilerOwnedDDRBufferCount);
}

static void
printAcceptedInstructionWork(llvm::raw_ostream &diagnostics,
                             const detail::AcceptedWholeVariant &accepted) {
  const analysis::WholeCardInstructionProgramCost &cost = accepted.resourceCost;
  for (auto [rank, rankCost] : llvm::enumerate(cost.rankCosts)) {
    diagnostics << "wafer-compile: instruction-work scope=rank rank="
                << accepted.ranks[rank].getLogicalRank();
    printInstructionProgramWork(diagnostics, rankCost.work);
    printInstructionProgramResources(diagnostics, rankCost);
    diagnostics << '\n';
  }
  diagnostics << "wafer-compile: instruction-work scope=all-ranks";
  printInstructionProgramWork(diagnostics, cost.aggregateWork);
  auto printAggregate = [&](llvm::StringRef name,
                            const analysis::ScheduleCostMetric &metric) {
    diagnostics << ' ' << name << '=';
    printInstructionWorkMetric(diagnostics, metric);
  };
  printAggregate("ddr_read_bytes", cost.aggregateDDRReadBytes);
  printAggregate("ddr_write_bytes", cost.aggregateDDRWriteBytes);
  printAggregate("spm_movement_bytes", cost.aggregateSPMMovementBytes);
  printAggregate("gather_scatter_bytes", cost.aggregateGatherScatterBytes);
  printAggregate("compute_npu_f16_bf16_ops",
                 cost.aggregateCompute.npuF16Bf16LogicalOps);
  printAggregate("compute_npu_other_ops",
                 cost.aggregateCompute.npuOtherLogicalOps);
  printAggregate("compute_vector_f16_bf16_ops",
                 cost.aggregateCompute.vectorF16Bf16LogicalOps);
  printAggregate("compute_vector_f32_ops",
                 cost.aggregateCompute.vectorF32LogicalOps);
  printAggregate("compute_vector_other_ops",
                 cost.aggregateCompute.vectorOtherLogicalOps);
  printAggregate("noc_transmit_bytes",
                 cost.aggregateNoC.aggregateTransmitBytes);
  printAggregate("noc_receive_bytes", cost.aggregateNoC.aggregateReceiveBytes);
  printAggregate("spm_high_water_bytes", cost.summedRankSPMHighWaterBytes);
  printAggregate("ddr_high_water_bytes", cost.summedRankDDRHighWaterBytes);
  printAggregate("spm_buffers", cost.aggregateCompilerOwnedSPMBufferCount);
  printAggregate("ddr_buffers", cost.aggregateCompilerOwnedDDRBufferCount);
  diagnostics << '\n';

  diagnostics << "wafer-compile: instruction-work scope=rank-maxima";
  printInstructionProgramWork(diagnostics, cost.maximumRankWork);
  printAggregate("dependency_depth", cost.maximumRankDataDependencyDepth);
  printAggregate("spm_high_water_bytes", cost.maximumRankSPMHighWaterBytes);
  printAggregate("ddr_high_water_bytes", cost.maximumRankDDRHighWaterBytes);
  printAggregate("spm_buffers", cost.maximumRankCompilerOwnedSPMBufferCount);
  printAggregate("ddr_buffers", cost.maximumRankCompilerOwnedDDRBufferCount);
  diagnostics << '\n';
}

static llvm::Expected<ExecutableBundle> buildCoordinatedExecutableBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    const OptimizationConfig &optimizations, llvm::raw_ostream &diagnostics,
    detail::WholeVariantSelectionMode selectionMode,
    std::optional<int64_t> failAfterLogicalRank,
    const detail::CompileClock::time_point &totalStart) {
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    diagnostics << "wafer-compile: " << message << "\n";
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };

  mlir::FailureOr<detail::CoordinatedWorkLedger> ledger =
      detail::CoordinatedWorkLedger::create(
          executionConfig.getRankCount(),
          detail::CoordinatedWorkLedger::kDefaultCapacity,
          /*repairReserve=*/0);
  if (mlir::failed(ledger))
    return fail("cannot reserve coordinated dataflow work ledger");

  const unsigned heavyweightThreadCount =
      llvm::heavyweight_hardware_concurrency().compute_thread_count();
  detail::CoordinatedDataflowSearchConfig searchConfig;
  searchConfig.rankCount = executionConfig.getRankCount();
  searchConfig.candidateParallelism =
      std::max<unsigned>(1, heavyweightThreadCount);
  searchConfig.optimizations = optimizations;
  searchConfig.reservedBaselineOnly =
      selectionMode == detail::WholeVariantSelectionMode::ReservedBaseline;
  detail::AttentionImplementationAlternativeProvider attentionAlternatives;
  searchConfig.implementationAlternativeProviders.push_back(
      &attentionAlternatives);
  detail::NoCCommunicationActionProvider noCCommunicationActions;
  llvm::SmallVector<const detail::CoordinatedCommunicationActionProvider *, 1>
      communicationProviders;
  if (optimizations != OptimizationConfig::none() &&
      selectionMode != detail::WholeVariantSelectionMode::ReservedBaseline)
    communicationProviders.push_back(&noCCommunicationActions);

  const detail::CompileClock::time_point generationStart =
      detail::CompileClock::now();
  detail::CoordinatedStructuredFrontierStatistics structuredStatistics;
  mlir::FailureOr<std::unique_ptr<detail::CoordinatedDataflowSearchSession>>
      searchSession = mlir::failure();
  {
    wafer::support::ScopedCompileTimingSpan generationTiming(
        "stage", "tensor-program-to-executable", "coordinated-tile-frontier");
    searchSession = detail::CoordinatedDataflowSearchSession::create(
        tensorModule, searchConfig, *ledger, &structuredStatistics);
  }
  if (mlir::failed(searchSession))
    return fail("coordinated Tile frontier generation failed");
  const int64_t generationWallMs =
      detail::elapsedCompileMilliseconds(generationStart);
  if (failAfterLogicalRank && *failAfterLogicalRank >= 0 &&
      *failAfterLogicalRank < executionConfig.getRankCount())
    return fail("test-only injected failure after logical rank " +
                std::to_string(*failAfterLogicalRank));

  detail::CoordinatedWorkLedgerSnapshot generationWork = ledger->getSnapshot();
  diagnostics
      << "wafer-compile: compile-stats stage=coordinated-tile-frontier"
      << " wall_ms=" << generationWallMs
      << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
      << " rank_count=" << executionConfig.getRankCount()
      << " candidate_count=" << structuredStatistics.structuralProposalsDerived
      << " structural_pending_peak="
      << structuredStatistics.structuralPendingPeak
      << " structural_equivalent_merged="
      << structuredStatistics.structuralEquivalentProposalsMerged
      << " structural_dominated_pruned="
      << structuredStatistics.structuralDominatedProposalsPruned
      << " structural_coverage_pruned="
      << structuredStatistics.structuralCoverageBeamPruned
      << " implementation_provider_queries="
      << structuredStatistics.implementationAlternativeQueries
      << " implementation_provider_proposals="
      << structuredStatistics.implementationAlternativeProposals
      << " derived_candidates=" << structuredStatistics.derivedCandidates
      << " materialized_candidates="
      << structuredStatistics.materializedCandidates
      << " candidate_materialization_failures="
      << structuredStatistics.candidateMaterializationFailures
      << " promotion_ineligible_rejections="
      << structuredStatistics.promotionIneligibleCandidatesRejected
      << " equivalent_rejections="
      << structuredStatistics.equivalentCandidatesRejected
      << " dominated_rejections="
      << structuredStatistics.dominatedCandidatesRejected
      << " dominated_states_erased="
      << structuredStatistics.dominatedStatesErased
      << " finalization_admission_denied="
      << structuredStatistics.finalizationAdmissionDenied
      << " generation_admission_denied="
      << structuredStatistics.generationAdmissionDenied
      << " structured_connections="
      << structuredStatistics.structuredConnections
      << " connection_proposal_estimate_model="
      << detail::kCoordinatedConnectionProposalEstimateModel
      << " connection_proposal_expansions="
      << structuredStatistics.connectionProposalExpansions
      << " connection_action_materializations="
      << structuredStatistics.connectionActionMaterializations
      << " connection_dp_states_merged="
      << structuredStatistics.connectionDPStatesMerged
      << " connection_dominated_states_pruned="
      << structuredStatistics.connectionDominatedStatesPruned
      << " connection_beam_states_pruned="
      << structuredStatistics.connectionBeamStatesPruned
      << " maximum_connection_states="
      << structuredStatistics.maximumConnectionStates
      << " retained_connection_candidates="
      << structuredStatistics.retainedConnectionCandidates
      << " connection_workers=" << structuredStatistics.connectionWorkerCount
      << " connection_solver="
      << (structuredStatistics.usedGeneralDAGBeam ? "dag-beam"
                                                  : "chain-tree-dp")
      << " frontier_digest=" << (*searchSession)->getStructuralFrontierDigest()
      << " work_capacity=" << generationWork.capacity
      << " work_consumed=" << generationWork.consumed
      << " finalization_reserved=" << generationWork.finalizationReserved
      << " schedule_attempt_capacity="
      << generationWork.scheduleAttemptCapacity
      << " schedule_attempts_reserved="
      << generationWork.scheduleAttemptsReserved
      << " schedule_attempts_consumed="
      << generationWork.scheduleAttemptsConsumed
      << " repair_reserved=" << generationWork.repairReserved
      << " work_unreserved=" << generationWork.unreserved << "\n";

  const detail::CompileClock::time_point finalizationStart =
      detail::CompileClock::now();
  std::vector<detail::AdmittedCoordinatedExecutable> admittedExecutables;
  detail::WholeVariantSelectionStatistics finalizationStatistics;
  int64_t finalizationWallMs = 0;
  {
    wafer::support::ScopedCompileTimingSpan finalizationTiming(
        "stage", "tensor-program-to-executable",
        "coordinated-executable-finalization");
    using FinalizationCursor =
        detail::CoordinatedExecutableFinalizationCursor;
    std::deque<std::unique_ptr<FinalizationCursor>> expansionCursors;
    uint64_t liveCanonicalParents = 0;
    bool structuralSeedsExhausted = false;

    auto updateCursorPeaks = [&](uint64_t additionalCursors = 0,
                                 uint64_t additionalParents = 0) {
      finalizationStatistics.peakLiveFinalizationCursors =
          std::max<uint64_t>(
              finalizationStatistics.peakLiveFinalizationCursors,
              expansionCursors.size() + additionalCursors);
      finalizationStatistics.peakLiveCanonicalInstrParents =
          std::max<uint64_t>(
              finalizationStatistics.peakLiveCanonicalInstrParents,
              liveCanonicalParents + additionalParents);
    };
    auto retainAccepted =
        [&](detail::AdmittedCoordinatedExecutable accepted) -> mlir::LogicalResult {
      admittedExecutables.push_back(std::move(accepted));
      return detail::reduceAdmittedExecutableFrontier(admittedExecutables,
                                                       selectionMode);
    };

    enum class FinalizationLaneDisposition : uint8_t {
      AttemptedAccepted,
      AttemptedRejected,
      Exhausted,
      BudgetExhausted,
    };
    struct FinalizationLaneResult {
      FinalizationLaneDisposition disposition =
          FinalizationLaneDisposition::Exhausted;

      bool attempted() const {
        return disposition == FinalizationLaneDisposition::AttemptedAccepted ||
               disposition == FinalizationLaneDisposition::AttemptedRejected;
      }
      bool accepted() const {
        return disposition == FinalizationLaneDisposition::AttemptedAccepted;
      }
    };
    auto runSeedLane = [&]() -> mlir::FailureOr<FinalizationLaneResult> {
      while (true) {
        mlir::FailureOr<std::unique_ptr<detail::CoordinatedTileVariant>> next =
            (*searchSession)->admitNextActualCandidate();
        if (mlir::failed(next))
          return mlir::failure();
        if (!*next) {
          structuralSeedsExhausted = true;
          return FinalizationLaneResult{
              FinalizationLaneDisposition::Exhausted};
        }
        std::unique_ptr<detail::CoordinatedTileVariant> variant =
            std::move(*next);
        const int64_t semanticOrdinal = variant->stableSemanticOrdinal;
        const bool reservedBaseline = variant->reservedBaseline;
        detail::CoordinatedExecutableAdmissionFailure failure;
        auto cursor = detail::beginCoordinatedExecutableFinalization(
            *variant, program, executionConfig, optimizations, *ledger,
            diagnostics, failure, &finalizationStatistics, selectionMode,
            /*rankPipelineParallelism=*/0, communicationProviders);
        if (mlir::failed(cursor)) {
          if (reservedBaseline ||
              !detail::isRecoverableCoordinatedExecutableSetupFailure(
                  failure))
            return mlir::failure();
          if (mlir::failed((*searchSession)
                               ->completeActiveCandidate(
                                   semanticOrdinal,
                                   detail::CoordinatedActualCandidateDisposition::
                                       ExactRejected,
                                   failure.gate)))
            return mlir::failure();
          diagnostics << "wafer-compile: coordinated executable rejection"
                      << " semantic_ordinal=" << semanticOrdinal
                      << " logical_rank=" << failure.logicalRank
                      << " gate=" << failure.gate << "\n";
          // Tile-to-Instr/setup rejection never started a schedule action and
          // therefore does not rotate the invocation-wide A/B attempt lanes.
          continue;
        }

        updateCursorPeaks(/*additionalCursors=*/1,
                          (*cursor)->getCanonicalParentCount());
        auto step = detail::advanceCoordinatedExecutableFinalization(
            **cursor, variant->finalizationReservation);
        if (mlir::failed(step))
          return mlir::failure();
        FinalizationLaneResult result{
            FinalizationLaneDisposition::AttemptedRejected};
        if (step->kind == detail::CoordinatedExecutableFinalizationStepKind::
                              Accepted) {
          if (!step->admitted ||
              mlir::failed(retainAccepted(std::move(*step->admitted))) ||
              mlir::failed((*searchSession)
                               ->completeActiveCandidate(
                                   semanticOrdinal,
                                   detail::CoordinatedActualCandidateDisposition::
                                       ExactAccepted)))
            return mlir::failure();
          result.disposition =
              FinalizationLaneDisposition::AttemptedAccepted;
          if (!(*cursor)->exactSeedAccepted())
            return mlir::failure();
          if (!(*cursor)->exhausted() &&
              expansionCursors.size() <
                  detail::kMaximumCoordinatedLiveFinalizationCursors) {
            liveCanonicalParents += (*cursor)->getCanonicalParentCount();
            expansionCursors.push_back(std::move(*cursor));
            updateCursorPeaks();
          }
          return result;
        }

        failure = step->failure;
        if (mlir::failed((*searchSession)
                             ->completeActiveCandidate(
                                 semanticOrdinal,
                                 detail::CoordinatedActualCandidateDisposition::
                                     ExactRejected,
                                 failure.gate)))
          return mlir::failure();
        diagnostics << "wafer-compile: coordinated executable rejection"
                    << " semantic_ordinal=" << semanticOrdinal
                    << " logical_rank=" << failure.logicalRank
                    << " gate=" << failure.gate << "\n";
        if (reservedBaseline)
          return mlir::failure();
        return result;
      }
    };

    auto runExpansionLane = [&]()
        -> mlir::FailureOr<FinalizationLaneResult> {
      while (!expansionCursors.empty()) {
        std::unique_ptr<FinalizationCursor> cursor =
            std::move(expansionCursors.front());
        expansionCursors.pop_front();
        if (cursor->exhausted()) {
          liveCanonicalParents -= cursor->getCanonicalParentCount();
          continue;
        }
        std::optional<detail::ExecutableFinalizationReservation> reservation =
            ledger->tryReserveExecutableScheduleAttempt();
        if (!reservation) {
          expansionCursors.push_front(std::move(cursor));
          return FinalizationLaneResult{
              FinalizationLaneDisposition::BudgetExhausted};
        }
        auto step = detail::advanceCoordinatedExecutableFinalization(
            *cursor, *reservation);
        if (mlir::failed(step))
          return mlir::failure();
        if (step->kind == detail::CoordinatedExecutableFinalizationStepKind::
                              Accepted) {
          if (!step->admitted ||
              mlir::failed(retainAccepted(std::move(*step->admitted))))
            return mlir::failure();
        } else if (step->kind ==
                   detail::CoordinatedExecutableFinalizationStepKind::
                       RecoverableRejected) {
          diagnostics << "wafer-compile: coordinated schedule rejection"
                      << " semantic_ordinal="
                      << cursor->getStableSemanticOrdinal()
                      << " logical_rank=" << step->failure.logicalRank
                      << " gate=" << step->failure.gate << "\n";
        }
        if (!cursor->exhausted())
          expansionCursors.push_back(std::move(cursor));
        else
          liveCanonicalParents -= cursor->getCanonicalParentCount();
        updateCursorPeaks();
        return FinalizationLaneResult{
            step->kind ==
                    detail::CoordinatedExecutableFinalizationStepKind::Accepted
                ? FinalizationLaneDisposition::AttemptedAccepted
                : FinalizationLaneDisposition::AttemptedRejected};
      }
      return FinalizationLaneResult{FinalizationLaneDisposition::Exhausted};
    };

    // The mandatory canonical baseline is outside lane rotation. Once it is
    // exact-admitted, both available lanes start with A (a new Tile seed), then
    // alternate A/B after every expensive attempt, including rejection.
    mlir::FailureOr<FinalizationLaneResult> baselineSeed = runSeedLane();
    if (mlir::failed(baselineSeed) || !baselineSeed->accepted())
      return fail("reserved coordinated baseline failed executable admission");
    detail::CoordinatedExecutableFinalizationLaneCoordinator laneCoordinator;
    if (mlir::failed(laneCoordinator.recordMandatoryBaselineAccepted()))
      return fail("coordinated finalization lane baseline accounting failed");
    while (laneCoordinator.canAttempt()) {
      const bool canSeed =
          !structuralSeedsExhausted &&
          expansionCursors.size() <
              detail::kMaximumCoordinatedLiveFinalizationCursors;
      const bool canExpand = !expansionCursors.empty();
      std::optional<detail::CoordinatedExecutableFinalizationLane> lane =
          laneCoordinator.chooseNextLane(canSeed, canExpand);
      if (!lane)
        break;

      const bool chooseSeed =
          *lane == detail::CoordinatedExecutableFinalizationLane::Seed;
      mlir::FailureOr<FinalizationLaneResult> laneResult = mlir::failure();
      if (chooseSeed) {
        laneResult = runSeedLane();
        if (mlir::failed(laneResult))
          return fail("coordinated Tile seed finalization failed");
      } else {
        laneResult = runExpansionLane();
        if (mlir::failed(laneResult))
          return fail("coordinated schedule expansion failed");
      }
      if (laneResult->disposition ==
          FinalizationLaneDisposition::BudgetExhausted)
        break;
      if (!laneResult->attempted()) {
        if (chooseSeed)
          structuralSeedsExhausted = true;
        else if (!canSeed)
          break;
        continue;
      }
      if (mlir::failed(
              laneCoordinator.recordAttempt(*lane, laneResult->accepted())))
        return fail("coordinated finalization lane accounting failed");
    }
    if (laneCoordinator.getAttemptCount() !=
            finalizationStatistics.scheduleMaterializationAttempts ||
        laneCoordinator.getExactAcceptedCount() !=
            finalizationStatistics.scheduleExactActions)
      return fail("coordinated finalization accounting diverged");
    finalizationWallMs = detail::elapsedCompileMilliseconds(finalizationStart);
  }
  if (llvm::count_if(admittedExecutables, [](const auto &variant) {
        return variant.reservedBaseline;
      }) != 1)
    return fail(
        "admitted executable frontier has no unique conservative baseline");

  detail::CoordinatedWorkLedgerSnapshot finalizationWork =
      ledger->getSnapshot();
  if (finalizationWork.finalizationReserved != 0 ||
      finalizationWork.scheduleAttemptsReserved != 0 ||
      finalizationWork.scheduleAttemptsConsumed !=
          finalizationStatistics.scheduleMaterializationAttempts)
    return fail("coordinated executable finalization left open work");
  const std::string finalizationTileFrontierDigest =
      (*searchSession)->getActualAdmissionDigest().str();
  const std::string admittedExecutableFrontierDigest =
      detail::computeAdmittedCoordinatedExecutableFrontierDigest(
          admittedExecutables);
  diagnostics
      << "wafer-compile: compile-stats "
         "stage=coordinated-executable-finalization"
      << " wall_ms=" << finalizationWallMs
      << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
      << " admitted_executables="
      << finalizationStatistics.admittedExecutableCount
      << " schedule_estimate_model="
      << detail::kCoordinatedScheduleActionEstimateModelIdentity
      << " schedule_estimated_actions="
      << finalizationStatistics.scheduleEstimatedActions
      << " schedule_coverage_retained="
      << finalizationStatistics.scheduleCoverageRetainedActions
      << " schedule_materialization_attempts="
      << finalizationStatistics.scheduleMaterializationAttempts
      << " schedule_materialization_failures="
      << finalizationStatistics.scheduleMaterializationFailures
      << " schedule_materialization_backfills="
      << finalizationStatistics.scheduleMaterializationBackfills
      << " schedule_accepted_action_clones="
      << finalizationStatistics.scheduleSuccessfulActionClones
      << " schedule_actual_rank_clones="
      << finalizationStatistics.scheduleActualRankClones
      << " schedule_exact_actions="
      << finalizationStatistics.scheduleExactActions
      << " schedule_seed_attempts="
      << finalizationStatistics.scheduleSeedAttempts
      << " schedule_expansion_attempts="
      << finalizationStatistics.scheduleExpansionAttempts
      << " peak_live_finalization_cursors="
      << finalizationStatistics.peakLiveFinalizationCursors
      << " peak_live_canonical_instr_parents="
      << finalizationStatistics.peakLiveCanonicalInstrParents
      << " peak_live_schedule_action_clones="
      << finalizationStatistics.peakLiveScheduleActionClones
      << " rank_pipeline_workers="
      << finalizationStatistics.maximumRankPipelineWorkers
      << " selection_live=" << admittedExecutables.size()
      << " repair_candidates=0"
      << " actual_candidate_attempts="
      << structuredStatistics.actualCandidateAttempts
      << " materialized_candidates="
      << structuredStatistics.materializedCandidates
      << " candidate_materialization_failures="
      << structuredStatistics.candidateMaterializationFailures
      << " promotion_ineligible_rejections="
      << structuredStatistics.promotionIneligibleCandidatesRejected
      << " equivalent_rejections="
      << structuredStatistics.equivalentCandidatesRejected
      << " dominated_rejections="
      << structuredStatistics.dominatedCandidatesRejected
      << " implementation_provider_materializations="
      << structuredStatistics.implementationAlternativeMaterializations
      << " retained_implementation_provider_candidates="
      << structuredStatistics.retainedImplementationAlternativeCandidates
      << " actual_candidate_admissions="
      << structuredStatistics.actualCandidateAdmissions
      << " actual_rank_clones=" << structuredStatistics.actualRankClones
      << " successful_actual_candidates="
      << structuredStatistics.successfulActualCandidates
      << " exact_failure_backfills="
      << structuredStatistics.exactFailureBackfills
      << " materialization_failure_backfills="
      << structuredStatistics.materializationFailureBackfills
      << " peak_live_actual_candidates="
      << structuredStatistics.peakLiveActualCandidates
      << " structural_pending_at_stop="
      << structuredStatistics.structuralPendingAtStop
      << " tile_frontier_digest=" << finalizationTileFrontierDigest
      << " admitted_executable_frontier_digest="
      << admittedExecutableFrontierDigest
      << " finalization_reserved=" << finalizationWork.finalizationReserved
      << " schedule_attempt_capacity="
      << finalizationWork.scheduleAttemptCapacity
      << " schedule_attempts_reserved="
      << finalizationWork.scheduleAttemptsReserved
      << " schedule_attempts_consumed="
      << finalizationWork.scheduleAttemptsConsumed
      << " work_consumed=" << finalizationWork.consumed << " tile_to_instr="
      << finalizationWork.consumedByKind[static_cast<size_t>(
             detail::CoordinatedWorkKind::TileToInstrLowering)]
      << " spm_problems="
      << finalizationWork.consumedByKind[static_cast<size_t>(
             detail::CoordinatedWorkKind::SPMAllocationProblem)]
      << " ddr_domains="
      << finalizationWork.consumedByKind[static_cast<size_t>(
             detail::CoordinatedWorkKind::DDRAllocationDomain)]
      << " transport_gates="
      << finalizationWork.consumedByKind[static_cast<size_t>(
             detail::CoordinatedWorkKind::TransportValidation)]
      << " abi_gates="
      << finalizationWork.consumedByKind[static_cast<size_t>(
             detail::CoordinatedWorkKind::ABIValidation)]
      << "\n";

  const size_t admittedExecutableCount =
      finalizationStatistics.admittedExecutableCount;
  const detail::CoordinatedWorkLedgerSnapshot selectionWorkBefore =
      ledger->getSnapshot();
  const detail::CompileClock::time_point selectionStart =
      detail::CompileClock::now();
  mlir::FailureOr<detail::AdmittedCoordinatedExecutable> selected =
      mlir::failure();
  {
    wafer::support::ScopedCompileTimingSpan selectionTiming(
        "stage", "tensor-program-to-executable",
        "coordinated-hardware-cost-selection");
    selected = detail::selectAdmittedCoordinatedExecutable(
        std::move(admittedExecutables), selectionMode, diagnostics,
        &finalizationStatistics);
  }
  if (mlir::failed(selected))
    return fail("coordinated whole-rank hardware-cost selection failed");
  const detail::CoordinatedWorkLedgerSnapshot selectionWorkAfter =
      ledger->getSnapshot();
  if (selectionWorkBefore.capacity != selectionWorkAfter.capacity ||
      selectionWorkBefore.consumed != selectionWorkAfter.consumed ||
      selectionWorkBefore.mandatoryGenerationReserved !=
          selectionWorkAfter.mandatoryGenerationReserved ||
      selectionWorkBefore.finalizationReserved !=
          selectionWorkAfter.finalizationReserved ||
      selectionWorkBefore.repairReserved != selectionWorkAfter.repairReserved ||
      selectionWorkBefore.scheduleAttemptCapacity !=
          selectionWorkAfter.scheduleAttemptCapacity ||
      selectionWorkBefore.scheduleAttemptsReserved !=
          selectionWorkAfter.scheduleAttemptsReserved ||
      selectionWorkBefore.scheduleAttemptsConsumed !=
          selectionWorkAfter.scheduleAttemptsConsumed ||
      selectionWorkBefore.unreserved != selectionWorkAfter.unreserved ||
      selectionWorkBefore.consumedByKind != selectionWorkAfter.consumedByKind)
    return fail("coordinated hardware-cost selection mutated the work ledger");
  diagnostics
      << "wafer-compile: compile-stats stage=coordinated-hardware-selection"
      << " wall_ms=" << detail::elapsedCompileMilliseconds(selectionStart)
      << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
      << " admitted_executables=" << admittedExecutableCount
      << " pareto_retained=" << finalizationStatistics.paretoRetainedVariants
      << " work_consumed_delta=0 tile_to_instr_delta=0"
      << " spm_problems_delta=0 ddr_domains_delta=0"
      << " transport_gates_delta=0 abi_gates_delta=0\n";
  detail::AcceptedWholeVariant accepted = std::move(selected->variant);
  printAcceptedInstructionWork(diagnostics, accepted);
  std::vector<RankExecutable> ranks = std::move(accepted.ranks);
  if (ranks.size() != static_cast<size_t>(executionConfig.getRankCount()))
    return fail("executable bundle rank domain is incomplete");
  for (auto [expectedRank, rank] : llvm::enumerate(ranks))
    if (rank.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return fail("executable bundle rank domain is not canonical");

  diagnostics << "wafer-compile: compile-stats stage=executable-bundle"
              << " wall_ms=" << detail::elapsedCompileMilliseconds(totalStart)
              << " peak_rss_kib=" << detail::getCompilePeakRSSKiB() << "\n";
  return ExecutableBundleBuilder::makeBundle(
      executionConfig, std::move(accepted.runtimeLaunchContract), context,
      std::move(ranks));
}

static llvm::Expected<ExecutableBundle> buildExecutableBundleImpl(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLogicalRank,
    detail::WholeVariantSelectionMode selectionMode) {
  const detail::CompileClock::time_point totalStart =
      detail::CompileClock::now();
  wafer::support::ScopedCompileTimingSpan executableBundleTiming(
      "stage", "tensor-program-to-executable", "executable-bundle");
  return buildCoordinatedExecutableBundle(
      context, tensorModule, program, executionConfig, optimizations,
      diagnostics, selectionMode, failAfterLogicalRank, totalStart);
}

llvm::Expected<ExecutableBundle> detail::buildExecutableBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLogicalRank,
    WholeVariantSelectionMode selectionMode) {
  return buildExecutableBundleImpl(context, tensorModule, std::move(program),
                                   executionConfig, optimizations, diagnostics,
                                   failAfterLogicalRank, selectionMode);
}

} // namespace wafer::compiler
