//===- ActualResultController.cpp - Typed accepted-result control -----===//

#include "Wafer/Planning/PhysicalDataflow/Search/ActualResultController.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <limits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

bool addWeightedMetric(uint64_t value, uint64_t weight, uint64_t &total) {
  if (value > std::numeric_limits<uint64_t>::max() / weight)
    return false;
  const uint64_t weighted = value * weight;
  if (total > std::numeric_limits<uint64_t>::max() - weighted)
    return false;
  total += weighted;
  return true;
}

bool hasSPMCapacityRejection(const FullFeasibilityResult &result) {
  return result.compilation &&
         llvm::any_of(result.compilation->tileFailures,
                      [](const CardExecutableTileFailure &failure) {
                        return isProvenExactTileMemoryPlanningFailure(
                            failure.memoryPlanning);
                      });
}

bool isCompletePlanKey(const ClosedSchedulePlan &plan) {
  return !plan.structure.scopes.empty() &&
         !plan.buffers.storageObjects.empty() && !plan.controlOrders.empty();
}

} // namespace

mlir::FailureOr<SearchCostCohort>
SearchCostCohort::create(uint64_t instructionTick, uint64_t ddrReadByteTick,
                         uint64_t ddrWriteByteTick,
                         uint64_t nocMinimumHopByteTick,
                         std::string *failureReason) {
  if (instructionTick == 0 || ddrReadByteTick == 0 || ddrWriteByteTick == 0 ||
      nocMinimumHopByteTick == 0) {
    if (failureReason)
      *failureReason = "search cost cohort rates must be positive";
    return mlir::failure();
  }
  return SearchCostCohort(instructionTick, ddrReadByteTick, ddrWriteByteTick,
                          nocMinimumHopByteTick);
}

SearchObjective
deriveSearchObjective(const analysis::CardInstructionProgramCost &cost,
                      const std::optional<SearchCostCohort> &cohort) {
  if (!cohort)
    return UnknownSearchObjective{SearchObjectiveUnknownReason::NoCohort};
  const std::array<const analysis::ScheduleCostMetric *, 4> metrics{
      &cost.aggregateInstructionCount, &cost.aggregateDDRReadBytes,
      &cost.aggregateDDRWriteBytes, &cost.minimumHopLinkByteDemand};
  if (llvm::any_of(metrics, [](const analysis::ScheduleCostMetric *metric) {
        return !metric->isKnown();
      }))
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::MetricUnavailable};
  const std::array<uint64_t, 4> weights{
      cohort->getInstructionTick(), cohort->getDDRReadByteTick(),
      cohort->getDDRWriteByteTick(), cohort->getNoCMinimumHopByteTick()};
  uint64_t total = 0;
  for (auto [metric, weight] : llvm::zip_equal(metrics, weights))
    if (!addWeightedMetric(metric->value, weight, total))
      return UnknownSearchObjective{
          SearchObjectiveUnknownReason::ArithmeticOverflow};
  return KnownSearchObjective{total, *cohort};
}

SearchObjectiveComparison compareSearchObjectives(const SearchObjective &lhs,
                                                  const SearchObjective &rhs) {
  const auto *left = std::get_if<KnownSearchObjective>(&lhs);
  const auto *right = std::get_if<KnownSearchObjective>(&rhs);
  if (!left || !right || !(left->cohort == right->cohort))
    return SearchObjectiveComparison::Incomparable;
  if (left->ticks < right->ticks)
    return SearchObjectiveComparison::Better;
  if (left->ticks > right->ticks)
    return SearchObjectiveComparison::Worse;
  return SearchObjectiveComparison::Equivalent;
}

CandidateReservation
ActualResultController::reserve(const ClosedSchedulePlan &plan) {
  if (!isCompletePlanKey(plan))
    return CandidateReservation::Invalid;
  if (finished || poisoned || reserved.count(plan) || completed.count(plan))
    return CandidateReservation::Duplicate;
  if (remainingCredits == 0)
    return CandidateReservation::Exhausted;
  --remainingCredits;
  reserved.insert(plan);
  ++statistics.reserved;
  return CandidateReservation::Granted;
}

CandidateRecordOutcome
ActualResultController::record(const ClosedSchedulePlan &plan,
                               FullFeasibilityResult result) {
  if (finished || poisoned || !reserved.erase(plan) || completed.count(plan)) {
    poisoned = true;
    return CandidateRecordOutcome::CompilerBug;
  }
  completed.insert(plan);
  switch (result.status) {
  case FullFeasibilityStatus::Accepted: {
    if (!result.compilation || !result.compilation->isAccepted()) {
      poisoned = true;
      return CandidateRecordOutcome::CompilerBug;
    }
    SearchObjective objective = deriveSearchObjective(
        result.compilation->executable->resourceCost, cohort);
    RetainedSearchCandidate candidate{plan, objective,
                                      std::move(*result.compilation)};
    bool replace = !incumbent;
    if (incumbent) {
      SearchObjectiveComparison comparison =
          compareSearchObjectives(candidate.objective, incumbent->objective);
      switch (comparison) {
      case SearchObjectiveComparison::Better:
        replace = true;
        break;
      case SearchObjectiveComparison::Worse:
        replace = false;
        break;
      case SearchObjectiveComparison::Equivalent:
        replace = candidate.plan < incumbent->plan;
        break;
      case SearchObjectiveComparison::Incomparable:
        sawUnknownOrIncomparable = true;
        replace = candidate.plan < incumbent->plan;
        break;
      }
    }
    if (std::holds_alternative<UnknownSearchObjective>(candidate.objective))
      sawUnknownOrIncomparable = true;
    if (replace)
      incumbent.emplace(std::move(candidate));
    ++statistics.accepted;
    return CandidateRecordOutcome::Accepted;
  }
  case FullFeasibilityStatus::ExactRejection: {
    ExactCompleteRejection rejection;
    rejection.plan = plan;
    rejection.kind = hasSPMCapacityRejection(result)
                         ? ExactCompleteRejectionKind::SPMCapacity
                         : ExactCompleteRejectionKind::ExecutableGate;
    rejection.causalRoots = std::move(result.causalRoots);
    forbidden.insert(std::move(rejection));
    ++statistics.exactRejected;
    return CandidateRecordOutcome::ExactRejection;
  }
  case FullFeasibilityStatus::Unsupported:
    ++statistics.unsupported;
    return CandidateRecordOutcome::Unsupported;
  case FullFeasibilityStatus::Indeterminate:
    ++statistics.indeterminate;
    return CandidateRecordOutcome::Indeterminate;
  case FullFeasibilityStatus::CompilerBug:
    poisoned = true;
    return CandidateRecordOutcome::CompilerBug;
  }
  poisoned = true;
  return CandidateRecordOutcome::CompilerBug;
}

bool ActualResultController::isForbidden(const ClosedSchedulePlan &plan) const {
  return findExactCompleteRejection(plan) != nullptr;
}

const ExactCompleteRejection *
ActualResultController::findExactCompleteRejection(
    const ClosedSchedulePlan &plan) const {
  auto found = forbidden.find(ExactCompleteRejection{plan});
  return found == forbidden.end() ? nullptr : &*found;
}

bool ActualResultController::canPrune(const SearchObjective &lowerBound) const {
  if (!incumbent)
    return false;
  const auto *bound = std::get_if<KnownSearchObjective>(&lowerBound);
  const auto *best = std::get_if<KnownSearchObjective>(&incumbent->objective);
  return bound && best && bound->cohort == best->cohort &&
         bound->ticks > best->ticks;
}

SearchControllerResult ActualResultController::finish(bool frontierExhausted) {
  SearchControllerResult result;
  result.statistics = statistics;
  result.exactCompleteRejections = forbidden.size();
  if (finished || poisoned || !reserved.empty()) {
    result.coverage = SearchControllerCoverage::Failed;
    finished = true;
    incumbent.reset();
    return result;
  }
  finished = true;
  if (!incumbent) {
    result.coverage = frontierExhausted && statistics.unsupported == 0 &&
                              statistics.indeterminate == 0
                          ? SearchControllerCoverage::NoFeasible
                          : SearchControllerCoverage::IncompleteNoCandidate;
    return result;
  }
  const bool incomplete = !frontierExhausted || statistics.unsupported != 0 ||
                          statistics.indeterminate != 0;
  if (incomplete)
    result.coverage = SearchControllerCoverage::FeasiblePartial;
  else if (sawUnknownOrIncomparable)
    result.coverage = SearchControllerCoverage::FeasibleUnranked;
  else
    result.coverage = SearchControllerCoverage::ComparableBest;
  result.winner.emplace(std::move(*incumbent));
  incumbent.reset();
  return result;
}

llvm::StringRef
stringifySearchControllerCoverage(SearchControllerCoverage coverage) {
  switch (coverage) {
  case SearchControllerCoverage::ComparableBest:
    return "comparable-best";
  case SearchControllerCoverage::FeasibleUnranked:
    return "feasible-unranked";
  case SearchControllerCoverage::FeasiblePartial:
    return "feasible-partial";
  case SearchControllerCoverage::NoFeasible:
    return "no-feasible";
  case SearchControllerCoverage::IncompleteNoCandidate:
    return "incomplete-no-candidate";
  case SearchControllerCoverage::Failed:
    return "failed";
  }
  return "unknown";
}

} // namespace wafer::compiler::detail
