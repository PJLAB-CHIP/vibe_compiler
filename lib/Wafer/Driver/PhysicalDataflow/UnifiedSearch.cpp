//===- UnifiedSearch.cpp - Resumable physical-dataflow traversal ------===//

#include "Wafer/Driver/PhysicalDataflow/UnifiedSearch.h"

#include "Wafer/Planning/PhysicalDataflow/PlanningProfile.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

#include <limits>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {
namespace {

struct SpatialFrame {};

struct StructuralCandidateFrame {
  RegionState state;
};

using FrontierFrame =
    std::variant<SpatialFrame, RegionContinuation, StructuralCandidateFrame>;

bool isTerminal(UnifiedSearchResumeStatus status) {
  return status != UnifiedSearchResumeStatus::Paused;
}

void recordRegionCandidateCounter(size_t index, llvm::StringRef metric,
                                  uint64_t value) {
  std::string name =
      llvm::formatv("region-candidate-{0}-{1}", index, metric).str();
  wafer::support::addCompileCounter("search", name, value);
}

void recordStructuralCandidateMetrics(
    size_t index, const RegionPlan &plan,
    const StructuralCandidateEvaluation &evaluation,
    const std::optional<SearchCostCohort> &cohort) {
  if (index >= 4)
    return;

  uint64_t fusionMerges = 0;
  for (const RegionGroupPlan &group : plan.groups)
    fusionMerges += group.mandatoryRoots.size() - 1;
  recordRegionCandidateCounter(index, "actualizations",
                               evaluation.actualizations);
  recordRegionCandidateCounter(index, "merges", fusionMerges);
  const ActualCandidateStatus status = evaluation.result.status;
  recordRegionCandidateCounter(index, "accepted",
                               status == ActualCandidateStatus::Accepted);
  recordRegionCandidateCounter(index, "exact-rejected",
                               status == ActualCandidateStatus::ExactRejection);
  recordRegionCandidateCounter(index, "unsupported",
                               status == ActualCandidateStatus::Unsupported);
  recordRegionCandidateCounter(index, "indeterminate",
                               status == ActualCandidateStatus::Indeterminate);

  if (status != ActualCandidateStatus::Accepted ||
      !evaluation.result.compilation ||
      !evaluation.result.compilation->executable)
    return;
  const analysis::InstructionProgramAggregateCost &cost =
      evaluation.result.compilation->executable->resourceCost;
  recordRegionCandidateCounter(index, "ddr-read-bytes-known",
                               cost.aggregateDDRReadBytes.isKnown());
  recordRegionCandidateCounter(index, "ddr-write-bytes-known",
                               cost.aggregateDDRWriteBytes.isKnown());
  recordRegionCandidateCounter(index, "instruction-count-known",
                               cost.aggregateInstructionCount.isKnown());
  if (cost.aggregateDDRReadBytes.isKnown())
    recordRegionCandidateCounter(index, "ddr-read-bytes",
                                 cost.aggregateDDRReadBytes.value);
  if (cost.aggregateDDRWriteBytes.isKnown())
    recordRegionCandidateCounter(index, "ddr-write-bytes",
                                 cost.aggregateDDRWriteBytes.value);
  if (cost.aggregateInstructionCount.isKnown())
    recordRegionCandidateCounter(index, "instruction-count",
                                 cost.aggregateInstructionCount.value);

  SearchObjective objective = deriveSearchObjective(cost, cohort);
  const auto *known = std::get_if<KnownSearchObjective>(&objective);
  recordRegionCandidateCounter(index, "objective-known", known != nullptr);
  if (!known)
    return;
  const SearchResourceDurations &durations = known->durations;
  recordRegionCandidateCounter(index, "objective-ne-picoseconds",
                               durations.neF16Bf16Picoseconds);
  recordRegionCandidateCounter(index, "objective-vector-f16-picoseconds",
                               durations.vectorF16Bf16Picoseconds);
  recordRegionCandidateCounter(index, "objective-vector-f32-picoseconds",
                               durations.vectorF32Picoseconds);
  recordRegionCandidateCounter(index, "objective-ddr-picoseconds",
                               durations.ddrPicoseconds);
  recordRegionCandidateCounter(index, "objective-noc-picoseconds",
                               durations.nocPicoseconds);
  recordRegionCandidateCounter(index, "objective-spm-picoseconds",
                               durations.spmMovementPicoseconds);
  recordRegionCandidateCounter(index, "objective-instruction-picoseconds",
                               durations.instructionControlPicoseconds);
  recordRegionCandidateCounter(index, "objective-dte-wait-picoseconds",
                               durations.dteWaitControlPicoseconds);
  recordRegionCandidateCounter(index, "objective-ncc-wait-picoseconds",
                               durations.nccWaitControlPicoseconds);
}

} // namespace

struct UnifiedSearchSession::Impl {
  Impl(PhysicalDataflowPlanningSession &planningSession,
       StructuralCandidateEvaluator &evaluator,
       const UnifiedSearchOptions &options, UnifiedSearchTrace *trace)
      : planningSession(planningSession), evaluator(evaluator),
        termination(options.termination), costCohort(options.costCohort),
        controller(ActualResultControllerOptions{
            options.structuralCandidateCredits, options.costCohort,
            options.exactRejectionCache}),
        profile(options.profile), trace(trace) {
    frontier.emplace_back(SpatialFrame{});
    if (profile) {
      profile->beginSearch();
      profile->observeFrontierDepth(frontier.size());
    }
  }

  template <typename State> void recordPrefix(const State &state) {
    if (trace)
      trace->prefixes.emplace_back(state);
  }

  void fail(llvm::StringRef detail) {
    status = UnifiedSearchResumeStatus::CompilerBug;
    failureDetail = detail.str();
  }

  void pauseIndeterminate(llvm::StringRef detail) {
    status = UnifiedSearchResumeStatus::Indeterminate;
    failureDetail = detail.str();
  }

  void stepSpatial() {
    ++work.successorSteps;
    SpatialExpansionResult expansion = planningSession.resumeSpatial();
    switch (expansion.getKind()) {
    case SpatialExpansionKind::StateQueued: {
      std::optional<SpatialState> state =
          planningSession.takeNextSpatialState();
      if (!state) {
        fail("spatial continuation lost its queued state");
        return;
      }
      recordPrefix(*state);
      frontier.emplace_back(
          planningSession.createRegionContinuation(std::move(*state)));
      return;
    }
    case SpatialExpansionKind::Unsupported:
      sawUnsupportedPrefix = true;
      return;
    case SpatialExpansionKind::Indeterminate:
      pauseIndeterminate(expansion.getDetail());
      return;
    case SpatialExpansionKind::ParentExhausted:
      frontier.pop_back();
      return;
    case SpatialExpansionKind::CompilerBug:
      fail(expansion.getDetail());
      return;
    }
  }

  void stepRegion(RegionContinuation &continuation) {
    ++work.successorSteps;
    std::string detail;
    auto next = planningSession.resumeRegion(continuation, &detail);
    if (mlir::failed(next)) {
      fail(detail.empty() ? "region continuation failed" : detail);
      return;
    }
    if (!*next) {
      frontier.pop_back();
      return;
    }
    recordPrefix(**next);
    frontier.emplace_back(StructuralCandidateFrame{std::move(**next)});
  }

  void stepCandidate(StructuralCandidateFrame frame) {
    RegionState state = std::move(frame.state);
    StructuralCandidateKey key = StructuralCandidateKey::create(state);
    CandidateReservation reservation = controller.reserve(key);
    if (reservation == CandidateReservation::Duplicate) {
      ++work.duplicateCompleteKeys;
      return;
    }
    if (reservation != CandidateReservation::Granted) {
      if (reservation == CandidateReservation::Exhausted) {
        status = UnifiedSearchResumeStatus::CandidateBudgetExhausted;
        return;
      }
      fail("actual-result controller rejected a unique complete key");
      return;
    }
    ++work.structuralStatesActualized;
    StructuralCandidateEvaluation evaluation = evaluator.evaluate(state);
    recordStructuralCandidateMetrics(work.structuralStatesActualized - 1,
                                     state.getRegionPlan(), evaluation,
                                     costCohort);
    work.candidateActualizations += evaluation.actualizations;
    if (!evaluation.domainExhausted) {
      sawIncompleteInnerDomain = true;
      ++work.incompleteInnerDomains;
    }
    const ActualCandidateStatus actualStatus = evaluation.result.status;
    if (profile)
      profile->recordCandidateActualization(
          evaluation.actualizations,
          actualStatus == ActualCandidateStatus::Accepted);
    const std::string actualDetail = evaluation.result.detail;
    if (trace)
      trace->candidates.push_back({key, actualStatus});
    CandidateRecordOutcome recorded =
        controller.record(key, std::move(evaluation.result));
    if (recorded == CandidateRecordOutcome::CompilerBug) {
      fail(actualDetail.empty()
               ? "actual-result controller rejected a typed actual result"
               : actualDetail);
      return;
    }
    if (recorded == CandidateRecordOutcome::Indeterminate) {
      if (!evaluation.domainExhausted)
        return;
      pauseIndeterminate(actualDetail.empty()
                             ? "complete candidate actualization is unknown"
                             : actualDetail);
      return;
    }
    if (recorded == CandidateRecordOutcome::Accepted &&
        termination == SearchTerminationPolicy::FirstAccepted)
      status = UnifiedSearchResumeStatus::AcceptedCheckpoint;
  }

  void step() {
    if (frontier.empty())
      return;
    if (std::holds_alternative<SpatialFrame>(frontier.back())) {
      stepSpatial();
      return;
    }
    if (auto *continuation =
            std::get_if<RegionContinuation>(&frontier.back())) {
      stepRegion(*continuation);
      return;
    }
    if (std::holds_alternative<StructuralCandidateFrame>(frontier.back())) {
      StructuralCandidateFrame frame =
          std::move(std::get<StructuralCandidateFrame>(frontier.back()));
      frontier.pop_back();
      stepCandidate(std::move(frame));
      return;
    }
    fail("current search frontier contains an unknown structural frame");
  }

  UnifiedSearchResumeResult resume(uint64_t credits) {
    if (finalized)
      return {UnifiedSearchResumeStatus::Finished, 0};
    if (isTerminal(status))
      return {status, 0};
    ++work.resumeCalls;
    uint64_t consumed = 0;
    while (consumed < credits) {
      if (frontier.empty()) {
        status = UnifiedSearchResumeStatus::FrontierExhausted;
        return {status, consumed};
      }
      step();
      if (profile)
        profile->observeFrontierDepth(frontier.size());
      ++consumed;
      if (isTerminal(status))
        return {status, consumed};
    }
    if (frontier.empty())
      status = UnifiedSearchResumeStatus::FrontierExhausted;
    return {status, consumed};
  }

  UnifiedSearchResult finish() {
    if (finalized) {
      UnifiedSearchResult result;
      result.control.coverage = SearchControllerCoverage::Failed;
      result.failureDetail = "unified search session was finished twice";
      return result;
    }
    if (status == UnifiedSearchResumeStatus::CompilerBug)
      controller.markCompilerBug();
    const bool exactExhaustion =
        status == UnifiedSearchResumeStatus::FrontierExhausted &&
        !sawUnsupportedPrefix && !sawIncompleteInnerDomain;
    UnifiedSearchResult result;
    result.control =
        controller.finish(exactExhaustion ? SearchFrontierStatus::Exhausted
                                          : SearchFrontierStatus::Incomplete);
    result.planning = planningSession.getWork();
    result.work = work;
    result.frontierExhausted =
        status == UnifiedSearchResumeStatus::FrontierExhausted;
    result.failureDetail = std::move(failureDetail);
    if (result.control.winner) {
      if (trace)
        ++trace->winnerHandoffs;
      if (profile)
        profile->recordWinnerHandoff();
    }
    frontier.clear();
    finalized = true;
    return result;
  }

  PhysicalDataflowPlanningSession &planningSession;
  StructuralCandidateEvaluator &evaluator;
  SearchTerminationPolicy termination;
  std::optional<SearchCostCohort> costCohort;
  ActualResultController controller;
  PlanningProfileSink *profile = nullptr;
  UnifiedSearchTrace *trace = nullptr;
  std::vector<FrontierFrame> frontier;
  UnifiedSearchWork work;
  UnifiedSearchResumeStatus status = UnifiedSearchResumeStatus::Paused;
  bool sawUnsupportedPrefix = false;
  bool sawIncompleteInnerDomain = false;
  bool finalized = false;
  std::string failureDetail;
};

UnifiedSearchSession::UnifiedSearchSession(
    PhysicalDataflowPlanningSession &session,
    StructuralCandidateEvaluator &evaluator,
    const UnifiedSearchOptions &options, UnifiedSearchTrace *trace)
    : impl(std::make_unique<Impl>(session, evaluator, options, trace)) {}

UnifiedSearchSession::~UnifiedSearchSession() = default;

UnifiedSearchResumeResult UnifiedSearchSession::resume(uint64_t credits) {
  return impl->resume(credits);
}

UnifiedSearchResult UnifiedSearchSession::finish() { return impl->finish(); }

UnifiedSearchResult runUnifiedSearch(PhysicalDataflowPlanningSession &session,
                                     StructuralCandidateEvaluator &evaluator,
                                     const UnifiedSearchOptions &options) {
  UnifiedSearchSession search(session, evaluator, options);
  (void)search.resume(options.planningCredits);
  return search.finish();
}

} // namespace wafer::compiler::detail
