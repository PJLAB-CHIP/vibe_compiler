//===- ActualResultController.cpp - Typed accepted-result control -----===//

#include "Wafer/Driver/PhysicalDataflow/ActualResultController.h"

#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Support/CompileTiming.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>
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

} // namespace

analysis::SearchObjective deriveExecutableSearchObjective(
    const ExecutableLoweringResult &executable,
    const std::optional<analysis::SearchCostCohort> &cohort) {
  support::ScopedCompileTimingSpan timing("analysis", "search-objective",
                                         "current-instr");
  support::addCompileCounter("cost", "objective-evaluations", 1);
  llvm::SmallVector<analysis::TileInstructionProgram, 16> programs;
  for (const auto &tile : executable.tiles)
    programs.push_back({tile.getTileId(), tile.getModule()});
  return analysis::deriveSearchObjective(executable.resourceCost, cohort,
                                         programs);
}

const analysis::SearchObjective &getActualCandidateObjective(
    ActualCandidateResult &result,
    const std::optional<analysis::SearchCostCohort> &cohort) {
  assert(result.isAccepted() && "objective requires a current accepted owner");
  if (!result.objective || !(result.objective->cohort == cohort))
    result.objective = ActualCandidateResult::EvaluatedObjective{
        cohort, deriveExecutableSearchObjective(*result.compilation->executable,
                                                cohort)};
  return result.objective->value;
}

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
                               ActualCandidateResult result,
                               CandidateDomainState domain) {
  if (finished || poisoned || !reserved.count(key) || completed.count(key))
    return failCompilerBug();
  if (domain == CandidateDomainState::Closed) {
    reserved.erase(key);
    completed.insert(key);
  }
  switch (result.status) {
  case ActualCandidateStatus::Accepted: {
    if (!result.compilation || !result.compilation->isAccepted() ||
        !result.compilation->executable)
      return failCompilerBug();
    analysis::SearchObjective objective =
        getActualCandidateObjective(result, cohort);
    RetainedSearchCandidate candidate{key, objective,
                                      std::move(*result.compilation)};
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
    if (domain == CandidateDomainState::Closed && spmCapacity &&
        exactRejectionCache == ExactRejectionCachePolicy::Enabled &&
        result.causalRoots.empty())
      return failCompilerBug();
    ExactCompleteRejection rejection{
        key,
        spmCapacity ? ExactCompleteRejectionKind::SPMCapacity
                    : ExactCompleteRejectionKind::ExecutableGate,
        std::move(result.causalRoots)};
    if (domain == CandidateDomainState::Closed &&
        exactRejectionCache == ExactRejectionCachePolicy::Enabled) {
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

void ActualResultController::close(const StructuralCandidateKey &key,
                                   SearchFrontierStatus domain) {
  if (finished || poisoned || !reserved.erase(key)) {
    markCompilerBug();
    return;
  }
  completed.insert(key);
  sawIncompleteDomain |= domain == SearchFrontierStatus::Incomplete;
}

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

SearchControllerResult
ActualResultController::finish(SearchFrontierStatus frontier) {
  SearchControllerResult result;
  result.statistics = statistics;
  result.exactCompleteRejections = forbidden.size();
  if (finished || poisoned ||
      (frontier == SearchFrontierStatus::Exhausted && !reserved.empty())) {
    result.coverage = SearchControllerCoverage::Failed;
    finished = true;
    incumbent.reset();
    return result;
  }
  finished = true;
  if (!incumbent) {
    result.coverage =
        frontier == SearchFrontierStatus::Exhausted && !sawIncompleteDomain &&
                statistics.unsupported == 0 && statistics.indeterminate == 0
            ? SearchControllerCoverage::NoFeasible
            : SearchControllerCoverage::IncompleteNoCandidate;
    return result;
  }
  const bool incomplete = frontier == SearchFrontierStatus::Incomplete ||
                          sawIncompleteDomain || statistics.unsupported != 0 ||
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
