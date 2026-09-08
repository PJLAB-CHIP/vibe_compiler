//===- ActualResultController.cpp - Typed accepted-result control -----===//

#include "Wafer/Driver/PhysicalDataflow/ActualResultController.h"

#include "Wafer/Analysis/Instr/CostModel.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>

namespace wafer::compiler::detail {
namespace {

bool hasSPMCapacityRejection(const ActualCandidateResult &result) {
  return result.compilation &&
         llvm::any_of(result.compilation->tileFailures,
                      [](const ExecutableTileFailure &failure) {
                        return isProvenExactTileMemoryPlanningFailure(
                            failure.memoryPlanning);
                      });
}

uint64_t getSelectedRegionCount(const RetainedSearchCandidate &candidate) {
  return candidate.key.getRegionPlan().groups.size();
}

bool isNoWorseThanReference(const analysis::SearchObjective &objective,
                            const analysis::SearchObjective &reference) {
  analysis::SearchObjectiveComparison comparison =
      analysis::compareSearchObjectives(objective, reference);
  return comparison == analysis::SearchObjectiveComparison::Better ||
         comparison == analysis::SearchObjectiveComparison::Equivalent;
}

} // namespace

CandidateReservation
ActualResultController::reserve(const StructuralCandidateKey &key) {
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
ActualResultController::record(const StructuralCandidateKey &key,
                               ActualCandidateResult result) {
  if (finished || poisoned || !reserved.erase(key) || completed.count(key))
    return failCompilerBug();
  completed.insert(key);
  switch (result.status) {
  case ActualCandidateStatus::Accepted: {
    if (!result.compilation || !result.compilation->isAccepted() ||
        !result.compilation->executable)
      return failCompilerBug();
    analysis::SearchObjective objective = analysis::deriveSearchObjective(
        result.compilation->executable->resourceCost, cohort);
    RetainedSearchCandidate candidate{key, objective,
                                      std::move(*result.compilation)};
    if (!referenceObjective)
      referenceObjective = candidate.objective;
    bool replace = !incumbent;
    if (incumbent) {
      analysis::SearchObjectiveComparison comparison =
          analysis::compareSearchObjectives(candidate.objective,
                                            incumbent->objective);
      switch (comparison) {
      case analysis::SearchObjectiveComparison::Better:
        replace = true;
        break;
      case analysis::SearchObjectiveComparison::Worse:
        replace = false;
        break;
      case analysis::SearchObjectiveComparison::Equivalent:
        replace = candidate.key < incumbent->key;
        break;
      case analysis::SearchObjectiveComparison::Incomparable:
        sawUnknownOrIncomparable = true;
        if (referenceObjective) {
          const bool candidateQualified =
              isNoWorseThanReference(candidate.objective, *referenceObjective);
          const bool incumbentQualified =
              isNoWorseThanReference(incumbent->objective, *referenceObjective);
          if (candidateQualified != incumbentQualified) {
            replace = candidateQualified;
            break;
          }
          if (candidateQualified) {
            const uint64_t candidateRegions = getSelectedRegionCount(candidate);
            const uint64_t incumbentRegions =
                getSelectedRegionCount(*incumbent);
            if (candidateRegions != incumbentRegions) {
              replace = candidateRegions < incumbentRegions;
              break;
            }
          }
        }
        replace = candidate.key < incumbent->key;
        break;
      }
    }
    if (std::holds_alternative<analysis::UnknownSearchObjective>(
            candidate.objective))
      sawUnknownOrIncomparable = true;
    if (replace)
      incumbent.emplace(std::move(candidate));
    ++statistics.accepted;
    return CandidateRecordOutcome::Accepted;
  }
  case ActualCandidateStatus::ExactRejection: {
    if (!result.compilation || !result.compilation->isProvenExactRejection())
      return failCompilerBug();
    const bool spmCapacity = hasSPMCapacityRejection(result);
    if (spmCapacity &&
        exactRejectionCache == ExactRejectionCachePolicy::Enabled &&
        result.causalRoots.empty())
      return failCompilerBug();
    ExactCompleteRejection rejection{
        key,
        spmCapacity ? ExactCompleteRejectionKind::SPMCapacity
                    : ExactCompleteRejectionKind::ExecutableGate,
        std::move(result.causalRoots)};
    if (exactRejectionCache == ExactRejectionCachePolicy::Enabled) {
      if (!forbidden.insert(std::move(rejection)).second)
        return failCompilerBug();
    }
    ++statistics.exactRejected;
    return CandidateRecordOutcome::ExactRejection;
  }
  case ActualCandidateStatus::Unsupported:
    ++statistics.unsupported;
    return CandidateRecordOutcome::Unsupported;
  case ActualCandidateStatus::Indeterminate:
    ++statistics.indeterminate;
    return CandidateRecordOutcome::Indeterminate;
  case ActualCandidateStatus::CompilerBug:
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
    const StructuralCandidateKey &key) const {
  return findExactCompleteRejection(key) != nullptr;
}

const ExactCompleteRejection *
ActualResultController::findExactCompleteRejection(
    const StructuralCandidateKey &key) const {
  auto found = forbidden.find(ExactCompleteRejection{
      key, ExactCompleteRejectionKind::ExecutableGate, {}});
  return found == forbidden.end() ? nullptr : &*found;
}

bool ActualResultController::canPrune(
    const SearchLowerBound &lowerBound) const {
  if (!incumbent)
    return false;
  const auto *bound =
      std::get_if<analysis::KnownSearchObjective>(&lowerBound.objective);
  const auto *best =
      std::get_if<analysis::KnownSearchObjective>(&incumbent->objective);
  return bound && best && bound->cohort == best->cohort &&
         analysis::compareSearchObjectives(lowerBound.objective,
                                           incumbent->objective) ==
             analysis::SearchObjectiveComparison::Worse;
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

} // namespace wafer::compiler::detail
