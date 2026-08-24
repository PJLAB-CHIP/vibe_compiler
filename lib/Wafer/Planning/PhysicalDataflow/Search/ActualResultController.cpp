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
ActualResultController::reserve(const CompleteCandidateKey &key) {
  if (finished || poisoned) {
    ++statistics.closedReservations;
    return CandidateReservation::Closed;
  }
  if (reserved.count(key) || completed.count(key)) {
    ++statistics.duplicateReservations;
    return CandidateReservation::Duplicate;
  }
  if (remainingCredits == 0) {
    ++statistics.exhaustedReservations;
    return CandidateReservation::Exhausted;
  }
  --remainingCredits;
  reserved.insert(key);
  ++statistics.reserved;
  return CandidateReservation::Granted;
}

CandidateRecordOutcome
ActualResultController::record(const CompleteCandidateKey &key,
                               FullFeasibilityResult result) {
  if (finished || poisoned || !reserved.erase(key) || completed.count(key))
    return failCompilerBug();
  completed.insert(key);
  switch (result.status) {
  case FullFeasibilityStatus::Accepted: {
    if (!result.compilation || !result.compilation->isAccepted() ||
        !result.compilation->executable)
      return failCompilerBug();
    SearchObjective objective = deriveSearchObjective(
        result.compilation->executable->resourceCost, cohort);
    RetainedSearchCandidate candidate{key, objective,
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
        replace = candidate.key < incumbent->key;
        break;
      case SearchObjectiveComparison::Incomparable:
        sawUnknownOrIncomparable = true;
        replace = candidate.key < incumbent->key;
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
    if (!result.compilation || !result.compilation->isProvenExactRejection())
      return failCompilerBug();
    const bool spmCapacity = hasSPMCapacityRejection(result);
    if (spmCapacity && result.causalRoots.empty())
      return failCompilerBug();
    ExactCompleteRejection rejection{
        key,
        spmCapacity ? ExactCompleteRejectionKind::SPMCapacity
                    : ExactCompleteRejectionKind::ExecutableGate,
        std::move(result.causalRoots)};
    if (exactRejectionCache == ExactRejectionCachePolicy::Enabled &&
        !forbidden.insert(std::move(rejection)).second)
      return failCompilerBug();
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
    return failCompilerBug();
  }
  return failCompilerBug();
}

CandidateRecordOutcome ActualResultController::failCompilerBug() {
  if (!poisoned)
    ++statistics.compilerBugs;
  poisoned = true;
  return CandidateRecordOutcome::CompilerBug;
}

void ActualResultController::markCompilerBug() { (void)failCompilerBug(); }

bool ActualResultController::isForbidden(
    const CompleteCandidateKey &key) const {
  return findExactCompleteRejection(key) != nullptr;
}

const ExactCompleteRejection *
ActualResultController::findExactCompleteRejection(
    const CompleteCandidateKey &key) const {
  auto found = forbidden.find(ExactCompleteRejection{
      key, ExactCompleteRejectionKind::ExecutableGate, {}});
  return found == forbidden.end() ? nullptr : &*found;
}

bool ActualResultController::canPrune(
    const SearchLowerBound &lowerBound) const {
  if (!incumbent)
    return false;
  const auto *bound = std::get_if<KnownSearchObjective>(&lowerBound.objective);
  const auto *best = std::get_if<KnownSearchObjective>(&incumbent->objective);
  return bound && best && bound->cohort == best->cohort &&
         bound->ticks > best->ticks;
}

SearchControllerResult
ActualResultController::finish(SearchFrontierStatus frontier) {
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
    result.coverage = frontier == SearchFrontierStatus::Exhausted &&
                              statistics.unsupported == 0 &&
                              statistics.indeterminate == 0
                          ? SearchControllerCoverage::NoFeasible
                          : SearchControllerCoverage::IncompleteNoCandidate;
    return result;
  }
  const bool incomplete = frontier == SearchFrontierStatus::Incomplete ||
                          statistics.unsupported != 0 ||
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
