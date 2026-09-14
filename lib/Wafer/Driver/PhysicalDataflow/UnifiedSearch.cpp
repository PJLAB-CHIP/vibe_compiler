//===- UnifiedSearch.cpp - Resumable physical-dataflow traversal ------===//

#include "Wafer/Driver/PhysicalDataflow/UnifiedSearch.h"

#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningProfile.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {
namespace {

struct SpatialFrame {};

using FrontierFrame = std::variant<SpatialFrame, RegionContinuation>;

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
    StructuralCandidateEvaluation &evaluation,
    const std::optional<analysis::SearchCostCohort> &cohort) {

  uint64_t fusionMerges = 0;
  for (const RegionGroupPlan &group : plan.groups)
    fusionMerges += group.mandatoryRoots.size() - 1;
  recordRegionCandidateCounter(index, "actualizations",
                               evaluation.actualizations);
  recordRegionCandidateCounter(index, "merges", fusionMerges);
  const ActualCandidateStatus status = evaluation.result->status;
  recordRegionCandidateCounter(index, "accepted",
                               status == ActualCandidateStatus::Accepted);
  recordRegionCandidateCounter(index, "exact-rejected",
                               status == ActualCandidateStatus::ExactRejection);
  recordRegionCandidateCounter(index, "unsupported",
                               status == ActualCandidateStatus::Unsupported);
  recordRegionCandidateCounter(index, "indeterminate",
                               status == ActualCandidateStatus::Indeterminate);

  if (status != ActualCandidateStatus::Accepted ||
      !evaluation.result->compilation ||
      !evaluation.result->compilation->executable)
    return;
  const analysis::InstructionProgramAggregateCost &cost =
      evaluation.result->compilation->executable->resourceCost;
  if (evaluation.result->compilation->physicalIRInventory) {
    const PhysicalDataflowIRInventory &inventory =
        *evaluation.result->compilation->physicalIRInventory;
    recordRegionCandidateCounter(index, "actual-tile-regions",
                                 inventory.tileRegions);
    for (const PhysicalTileIRInventory &tile : inventory.tiles)
      recordRegionCandidateCounter(
          index,
          llvm::formatv("actual-tile-{0}-regions", tile.tile.getValue()).str(),
          tile.regions);
  }
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
  recordRegionCandidateCounter(index, "ddr-segments-known",
                               cost.aggregateDDRSegmentCount.isKnown());
  if (cost.aggregateDDRSegmentCount.isKnown())
    recordRegionCandidateCounter(index, "ddr-segments",
                                 cost.aggregateDDRSegmentCount.value);
  if (cost.maximumTileDDRSegmentCount.isKnown())
    recordRegionCandidateCounter(index, "maximum-tile-ddr-segments",
                                 cost.maximumTileDDRSegmentCount.value);
  if (cost.aggregateInstructionCount.isKnown())
    recordRegionCandidateCounter(index, "instruction-count",
                                 cost.aggregateInstructionCount.value);

  analysis::SearchObjective objective =
      getActualCandidateObjective(*evaluation.result, cohort);
  const auto *known = std::get_if<analysis::KnownSearchObjective>(&objective);
  recordRegionCandidateCounter(index, "objective-known", known != nullptr);
  if (!known)
    return;
  recordRegionCandidateCounter(index, "objective-estimated-picoseconds",
                               known->estimatedDurationPicoseconds);
  recordRegionCandidateCounter(index, "objective-coarse-estimate",
                               known->usesCoarseEstimate);
  const analysis::SearchResourceDurations &durations = known->durations;
  recordRegionCandidateCounter(index, "objective-profile-identity",
                               known->cohort.getPolicy().profileIdentity);
  recordRegionCandidateCounter(
      index, "objective-profile-calibrated",
      known->cohort.getPolicy().profileProvenance ==
          analysis::SearchCostProfileProvenance::CalibratedTarget);
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
  recordRegionCandidateCounter(index, "objective-dte-endpoint-picoseconds",
                               durations.dteEndpointPicoseconds);
  recordRegionCandidateCounter(index, "objective-dte-startup-picoseconds",
                               durations.dteStartupPicoseconds);
  recordRegionCandidateCounter(index, "objective-noc-hop-picoseconds",
                               durations.nocHopPicoseconds);
  recordRegionCandidateCounter(index, "objective-spm-picoseconds",
                               durations.spmMovementPicoseconds);
  recordRegionCandidateCounter(index, "objective-instruction-picoseconds",
                               durations.instructionControlPicoseconds);
  recordRegionCandidateCounter(index, "objective-dte-wait-picoseconds",
                               durations.dteWaitControlPicoseconds);
  recordRegionCandidateCounter(index, "objective-ncc-wait-picoseconds",
                               durations.nccWaitControlPicoseconds);
  recordRegionCandidateCounter(index, "objective-spm-high-water-bytes",
                               durations.spmHighWaterBytes);
  recordRegionCandidateCounter(index, "objective-ddr-high-water-bytes",
                               durations.ddrHighWaterBytes);
  recordRegionCandidateCounter(index, "objective-spm-buffer-count",
                               durations.spmBufferCount);
  recordRegionCandidateCounter(index, "objective-ddr-buffer-count",
                               durations.ddrBufferCount);
}

} // namespace

struct UnifiedSearchSession::Impl {
  struct LocalBest {
    SpatialPlan spatial;
    RegionPlan region;
    analysis::SearchObjective objective;
  };
  struct Branch {
    RegionState state;
    std::unique_ptr<StructuralCandidateSession> session;
    CandidateContinuation next = CandidateContinuation::Explore;
    std::optional<analysis::SearchObjective> objective;
    uint64_t identity = 0;
    uint64_t lastVisit = 0;
    bool evaluated = false;
    CandidateRetention retention = CandidateRetention::Replaceable;
  };

  Impl(PhysicalDataflowPlanningSession &planningSession,
       StructuralCandidateEvaluator &evaluator,
       const UnifiedSearchOptions &options, UnifiedSearchTrace *trace)
      : planningSession(planningSession), evaluator(evaluator),
        termination(options.termination), costCohort(options.costCohort),
        remainingActualizationCredits(options.candidateActualizationCredits),
        retainedBranches(options.retainedBranches),
        maximumRegionRefinementCandidates(
            options.maximumRegionRefinementCandidates),
        controller(ActualResultControllerOptions{
            std::numeric_limits<uint64_t>::max(), options.costCohort,
            options.exactRejectionCache}),
        profile(options.profile), trace(trace) {
    frontier.emplace_back(SpatialFrame{});
    if (profile) {
      profile->beginSearch();
      profile->observeFrontierDepth(frontier.size());
    }
    if (!retainedBranches)
      fail("search retention width must be positive");
  }

  void fail(llvm::StringRef detail) {
    status = UnifiedSearchResumeStatus::CompilerBug;
    failureDetail = detail.str();
  }

  template <typename State> void recordPrefix(const State &state) {
    if (trace)
      trace->prefixes.emplace_back(state);
  }

  // Round-robin parent continuations preserve each raw-domain cursor. A new
  // Spatial family gets its first Region before parameter successors of older
  // families; no Region subtree drains the global budget.
  void generate() {
    FrontierFrame frame = std::move(frontier.front());
    frontier.pop_front();
    ++work.successorSteps;
    if (std::holds_alternative<SpatialFrame>(frame)) {
      SpatialExpansionResult expansion = planningSession.resumeSpatial();
      switch (expansion.getKind()) {
      case SpatialExpansionKind::StateQueued: {
        auto state = planningSession.takeNextSpatialState();
        if (!state) {
          fail("spatial continuation lost its queued state");
          return;
        }
        recordPrefix(*state);
        frontier.emplace_front(
            planningSession.createRegionContinuation(std::move(*state)));
        frontier.emplace_back(SpatialFrame{});
        return;
      }
      case SpatialExpansionKind::Unsupported:
        sawUnsupportedPrefix = true;
        frontier.emplace_back(SpatialFrame{});
        return;
      case SpatialExpansionKind::ParentExhausted:
        return;
      case SpatialExpansionKind::Indeterminate:
        status = UnifiedSearchResumeStatus::Indeterminate;
        failureDetail = expansion.getDetail().str();
        return;
      case SpatialExpansionKind::CompilerBug:
        fail(expansion.getDetail());
        return;
      }
    }
    auto &continuation = std::get<RegionContinuation>(frame);
    if (continuation.needsRefinementProposals(
            maximumRegionRefinementCandidates)) {
      auto local = llvm::find_if(localBest, [&](const LocalBest &best) {
        return best.spatial == continuation.getParent().getPlan();
      });
      if (local != localBest.end() && maximumRegionRefinementCandidates) {
        ++work.localRegionRefinements;
        const auto *incumbent = controller.getIncumbentKey();
        if (!incumbent || !(incumbent->getSpatialPlan() == local->spatial))
          ++work.nonIncumbentRegionRefinements;
        std::string detail;
        if (mlir::failed(planningSession.addRegionRefinementProposals(
                continuation, local->region, maximumRegionRefinementCandidates,
                &detail))) {
          fail(detail);
          return;
        }
      } else if (!maximumRegionRefinementCandidates) {
        planningSession.skipRegionRefinementProposals(continuation);
      }
      // Without a local actual result, ordinary Region traversal continues.
      // A later accepted point can still provide this family's first center.
    }
    std::string detail;
    auto state = planningSession.resumeRegion(continuation, &detail);
    if (mlir::failed(state)) {
      fail(detail.empty() ? "region continuation failed" : detail);
      return;
    }
    if (!*state)
      return;
    recordPrefix(**state);
    pending.emplace(std::move(**state));
    frontier.emplace_back(std::move(continuation));
  }

  bool makeRoom() {
    if (branches.size() < retainedBranches)
      return true;
    std::optional<size_t> worst;
    auto sameStructure = [](const RegionState &lhs, const RegionState &rhs) {
      if (!(lhs.getRegionPlan() == rhs.getRegionPlan()))
        return false;
      const auto &left = lhs.getSpatialPlan().nodes;
      const auto &right = rhs.getSpatialPlan().nodes;
      if (left.size() != right.size())
        return false;
      for (size_t node = 0; node < left.size(); ++node) {
        if (!(left[node].root == right[node].root) ||
            left[node].axes.size() != right[node].axes.size())
          return false;
        for (size_t axis = 0; axis < left[node].axes.size(); ++axis) {
          const auto &a = left[node].axes[axis];
          const auto &b = right[node].axes[axis];
          if (a.iterator != b.iterator || a.scheme != b.scheme ||
              (a.parameter == 1) != (b.parameter == 1))
            return false;
        }
      }
      return true;
    };
    // This grouping affects retention priority only. Distinct placements and
    // numeric choices still have distinct keys and actual evaluations.
    auto redundant = [&](size_t index) {
      if (pending && sameStructure(branches[index].state, *pending))
        return true;
      for (size_t other = 0; other < branches.size(); ++other)
        if (other != index &&
            sameStructure(branches[index].state, branches[other].state))
          return true;
      return false;
    };
    for (size_t i = 0; i < branches.size(); ++i) {
      const Branch &candidate = branches[i];
      // The session protects its actual base comparison and initial coarse
      // repair round. Later pending neighbors do not lock a slot forever.
      if (!candidate.evaluated ||
          candidate.retention != CandidateRetention::Replaceable)
        continue;
      if (!worst) {
        worst = i;
        continue;
      }
      const Branch &previous = branches[*worst];
      if (redundant(i) != redundant(*worst)) {
        if (redundant(i))
          worst = i;
        continue;
      }
      if (!candidate.objective ||
          (previous.objective &&
           analysis::compareSearchObjectives(*candidate.objective,
                                             *previous.objective) ==
               analysis::SearchObjectiveComparison::Worse))
        worst = i;
    }
    if (!worst)
      return false;
    controller.close(StructuralCandidateKey::create(branches[*worst].state),
                     SearchFrontierStatus::Incomplete);
    branches.erase(branches.begin() + *worst);
    ++work.retiredBranches;
    sawIncompleteInnerDomain = true;
    return true;
  }

  void startPending() {
    StructuralCandidateKey key = StructuralCandidateKey::create(*pending);
    CandidateReservation reservation = controller.reserve(key);
    if (reservation == CandidateReservation::Duplicate) {
      ++work.duplicateCompleteKeys;
      pending.reset();
      return;
    }
    if (reservation != CandidateReservation::Granted) {
      fail("actual-result controller rejected a new structural session");
      return;
    }
    // Resolve duplicates before retiring an actual owner. The scheduler only
    // enters here when a slot or an evaluated, non-repair branch is available.
    if (!makeRoom()) {
      fail("reserved structural session has no available retention slot");
      return;
    }
    auto session = evaluator.start(*pending);
    if (!session) {
      fail("structural evaluator returned no owned session");
      return;
    }
    uint64_t identity = work.structuralStatesActualized++;
    branches.push_back({std::move(*pending),
                        std::move(session),
                        CandidateContinuation::Explore,
                        {},
                        identity});
    pending.reset();
    work.peakRetainedBranches =
        std::max<uint64_t>(work.peakRetainedBranches, branches.size());
    evaluate(branches.size() - 1);
  }

  void evaluate(size_t index) {
    Branch &branch = branches[index];
    work.resumedCandidates += branch.evaluated;
    branch.evaluated = true;
    support::ScopedCompileTimingSpan timing(
        "search-candidate", "current-ir", "advance",
        llvm::formatv("structural={0}, attempt={1}", branch.identity,
                      work.candidateActualizations)
            .str());
    StructuralCandidateEvaluation evaluation = branch.session->advance();
    if (evaluation.actualizations > 1 ||
        evaluation.actualizations > remainingActualizationCredits ||
        (evaluation.result && !evaluation.actualizations &&
         evaluation.continuation != CandidateContinuation::Exhausted)) {
      fail("candidate session must advance by at most one actual attempt");
      return;
    }
    if (!evaluation.result) {
      if (evaluation.actualizations) {
        fail("candidate session charged a step without an actual result");
        return;
      }
      if (evaluation.continuation != CandidateContinuation::Exhausted) {
        if (evaluation.retention !=
            CandidateRetention::UnfinishedActualization) {
          fail("yielded candidate must retain its unfinished actual owner");
          return;
        }
        ++work.stageYields;
        branch.next = evaluation.continuation;
        branch.retention = evaluation.retention;
        runningBranch = branch.identity;
        return;
      }
      controller.close(StructuralCandidateKey::create(branch.state),
                       SearchFrontierStatus::Exhausted);
      branches.erase(branches.begin() + index);
      runningBranch.reset();
      return;
    }
    runningBranch.reset();
    branch.lastVisit = ++visit;
    remainingActualizationCredits -= evaluation.actualizations;
    work.candidateActualizations += evaluation.actualizations;
    if (evaluation.actualizations)
      servicePhase = (servicePhase + 1) % 3;
    if (evaluation.actualizations) {
      recordStructuralCandidateMetrics(work.candidateActualizations - 1,
                                       branch.state.getRegionPlan(), evaluation,
                                       costCohort);
      recordRegionCandidateCounter(work.candidateActualizations - 1,
                                   "structural-session", branch.identity);
    }
    const bool exhausted =
        evaluation.continuation == CandidateContinuation::Exhausted;
    const auto actualStatus = evaluation.result->status;
    if (evaluation.result->isAccepted()) {
      if (!hasAccepted) {
        hasAccepted = true;
        servicePhase = 0;
      }
      auto objective = getActualCandidateObjective(*evaluation.result, costCohort);
      if (!branch.objective ||
          analysis::compareSearchObjectives(objective, *branch.objective) ==
              analysis::SearchObjectiveComparison::Better)
        branch.objective = objective;
      auto local = llvm::find_if(localBest, [&](const LocalBest &best) {
        return best.spatial == branch.state.getSpatialPlan();
      });
      if (local == localBest.end())
        localBest.push_back({branch.state.getSpatialPlan(),
                             branch.state.getRegionPlan(),
                             std::move(objective)});
      else if (analysis::compareSearchObjectives(objective, local->objective) ==
               analysis::SearchObjectiveComparison::Better) {
        local->region = branch.state.getRegionPlan();
        local->objective = std::move(objective);
      }
    }
    if (profile)
      profile->recordCandidateActualization(
          evaluation.actualizations,
          actualStatus == ActualCandidateStatus::Accepted);
    auto key = StructuralCandidateKey::create(branch.state);
    if (trace)
      trace->candidates.push_back({key, actualStatus});
    std::string detail = evaluation.result->detail;
    CandidateRecordOutcome recorded = controller.record(
        key, std::move(*evaluation.result),
        exhausted ? CandidateDomainState::Closed : CandidateDomainState::Open);
    if (recorded == CandidateRecordOutcome::CompilerBug) {
      fail(detail.empty() ? "invalid typed actual candidate result" : detail);
      return;
    }
    branch.next = evaluation.continuation;
    branch.retention = evaluation.retention;
    if (!evaluation.actualizations &&
        actualStatus == ActualCandidateStatus::Indeterminate) {
      status = UnifiedSearchResumeStatus::Indeterminate;
      failureDetail = detail;
    }
    if (exhausted)
      branches.erase(branches.begin() + index);
    if (recorded == CandidateRecordOutcome::Accepted &&
        termination == SearchTerminationPolicy::FirstAccepted)
      status = UnifiedSearchResumeStatus::AcceptedCheckpoint;
  }

  void step() {
    if (!remainingActualizationCredits) {
      status = UnifiedSearchResumeStatus::CandidateBudgetExhausted;
      return;
    }
    if (frontier.empty() && !pending && branches.empty()) {
      status = UnifiedSearchResumeStatus::FrontierExhausted;
      return;
    }
    if (runningBranch) {
      for (size_t i = 0; i < branches.size(); ++i)
        if (branches[i].identity == *runningBranch) {
          evaluate(i);
          return;
        }
      fail("yielded actualization lost its owning branch");
      return;
    }
    // Give repair/local improvement explicit service, without consuming the
    // exploration cursor. A yielded stage continues above and is not another
    // service turn. Selection uses actual outcomes and complete tie-breaks.
    const bool preferContinuation = servicePhase < 2;
    if (preferContinuation) {
      const auto wanted = hasAccepted ? CandidateContinuation::Improve
                                      : CandidateContinuation::Repair;
      std::optional<size_t> selected;
      for (size_t index = 0; index < branches.size(); ++index) {
        const auto &branch = branches[index];
        if (branch.next != wanted &&
            (hasAccepted ||
             branch.retention != CandidateRetention::PendingCapacityRepair))
          continue;
        if (!selected ||
            (!hasAccepted && branch.identity < branches[*selected].identity) ||
            (hasAccepted && std::tie(branch.lastVisit, branch.identity) <
                                std::tie(branches[*selected].lastVisit,
                                         branches[*selected].identity)))
          selected = index;
      }
      if (selected) {
        evaluate(*selected);
        return;
      }
    }
    while (!round.empty()) {
      uint64_t identity = round.front();
      round.pop_front();
      for (size_t i = 0; i < branches.size(); ++i)
        if (branches[i].identity == identity) {
          evaluate(i);
          return;
        }
    }
    if (introduceNext) {
      const bool room =
          branches.size() < retainedBranches ||
          llvm::any_of(
              branches,
              [](const Branch &branch) {
                return branch.evaluated &&
                       branch.retention == CandidateRetention::Replaceable;
              });
      if (room) {
        if (!pending && !frontier.empty()) {
          generate();
          return;
        }
        if (pending) {
          introduceNext = false;
          startPending();
          return;
        }
      }
    }
    // One leaf from each live structural family, followed by one new family.
    // Verified-stage yields resume the selected leaf without changing this
    // order or advancing any other family's logical visitation count.
    introduceNext = true;
    for (const auto &branch : branches)
      round.push_back(branch.identity);
  }

  UnifiedSearchResumeResult resume(uint64_t credits) {
    if (finalized)
      return {UnifiedSearchResumeStatus::Finished, 0};
    if (isTerminal(status))
      return {status, 0};
    ++work.resumeCalls;
    uint64_t consumed = 0;
    while (consumed < credits && !isTerminal(status)) {
      step();
      ++consumed;
      if (profile)
        profile->observeFrontierDepth(frontier.size() + branches.size() +
                                      bool(pending));
    }
    return {status, consumed};
  }

  UnifiedSearchResult finish() {
    if (finalized) {
      UnifiedSearchResult result;
      result.failureDetail = "unified search session was finished twice";
      return result;
    }
    if (status == UnifiedSearchResumeStatus::CompilerBug)
      controller.markCompilerBug();
    work.incompleteInnerDomains = work.retiredBranches + branches.size();
    bool exhausted = status == UnifiedSearchResumeStatus::FrontierExhausted;
    UnifiedSearchResult result;
    result.control = controller.finish(exhausted && !sawUnsupportedPrefix &&
                                               !sawIncompleteInnerDomain
                                           ? SearchFrontierStatus::Exhausted
                                           : SearchFrontierStatus::Incomplete);
    result.planning = planningSession.getWork();
    result.work = work;
    result.frontierExhausted = exhausted;
    result.failureDetail = std::move(failureDetail);
    if (result.control.winner) {
      if (trace)
        ++trace->winnerHandoffs;
      if (profile)
        profile->recordWinnerHandoff();
    }
    branches.clear();
    frontier.clear();
    pending.reset();
    finalized = true;
    return result;
  }

  PhysicalDataflowPlanningSession &planningSession;
  StructuralCandidateEvaluator &evaluator;
  SearchTerminationPolicy termination;
  std::optional<analysis::SearchCostCohort> costCohort;
  uint64_t remainingActualizationCredits;
  uint64_t retainedBranches;
  uint64_t maximumRegionRefinementCandidates;
  ActualResultController controller;
  PlanningProfileSink *profile;
  UnifiedSearchTrace *trace;
  std::deque<FrontierFrame> frontier;
  std::optional<RegionState> pending;
  std::vector<Branch> branches;
  std::vector<LocalBest> localBest;
  uint64_t visit = 0;
  unsigned servicePhase = 0;
  bool hasAccepted = false;
  std::deque<uint64_t> round;
  std::optional<uint64_t> runningBranch;
  bool introduceNext = true;
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
