//===- ActualResultController.cpp - Typed accepted-result control -----===//

#include "Wafer/Driver/PhysicalDataflow/ActualResultController.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <limits>
#include <optional>
#include <utility>

namespace wafer::compiler::detail {
namespace {

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::optional<uint64_t> timeForWork(uint64_t work, uint64_t rate) {
  if (rate == 0)
    return std::nullopt;
  if (work == 0)
    return uint64_t{0};
  constexpr uint64_t picosecondsPerSecond = UINT64_C(1000000000000);
  const unsigned __int128 numerator =
      static_cast<unsigned __int128>(work) * picosecondsPerSecond;
  const unsigned __int128 duration =
      (numerator + static_cast<unsigned __int128>(rate) - 1) / rate;
  if (duration > std::numeric_limits<uint64_t>::max())
    return std::nullopt;
  return static_cast<uint64_t>(duration);
}

template <typename Accessor>
std::optional<uint64_t>
maximumTileMetric(const analysis::InstructionProgramAggregateCost &cost,
                  Accessor accessor,
                  const analysis::ScheduleCostMetric &aggregate) {
  if (cost.tileCosts.empty())
    return aggregate.isKnown() ? std::optional<uint64_t>(aggregate.value)
                               : std::nullopt;
  uint64_t maximum = 0;
  for (const analysis::InstructionProgramCost &tile : cost.tileCosts) {
    const analysis::ScheduleCostMetric &metric = accessor(tile);
    if (!metric.isKnown())
      return std::nullopt;
    maximum = std::max(maximum, metric.value);
  }
  return maximum;
}

std::array<uint64_t, 12> asArray(const SearchResourceDurations &durations) {
  return {durations.neF16Bf16Picoseconds,
          durations.vectorF16Bf16Picoseconds,
          durations.vectorF32Picoseconds,
          durations.ddrPicoseconds,
          durations.nocPicoseconds,
          durations.dteEndpointPicoseconds,
          durations.dteStartupPicoseconds,
          durations.nocHopPicoseconds,
          durations.spmMovementPicoseconds,
          durations.instructionControlPicoseconds,
          durations.dteWaitControlPicoseconds,
          durations.nccWaitControlPicoseconds};
}

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

bool isNoWorseThanReference(const SearchObjective &objective,
                            const SearchObjective &reference) {
  SearchObjectiveComparison comparison =
      compareSearchObjectives(objective, reference);
  return comparison == SearchObjectiveComparison::Better ||
         comparison == SearchObjectiveComparison::Equivalent;
}

} // namespace

mlir::FailureOr<SearchCostCohort>
SearchCostCohort::create(const SearchCostPolicy &policy,
                         std::string *failureReason) {
  const std::array<uint64_t, 12> rates{
      policy.ddrNominalBytesPerSecond,
      policy.directionalNoCBytesPerSecond,
      policy.dteEndpointBytesPerSecondEstimate,
      policy.dteMessageStartupPicosecondsEstimate,
      policy.noCHopPicosecondsEstimate,
      policy.instructionFixedPicosecondsEstimate,
      policy.dteWaitedEventPicosecondsEstimate,
      policy.nccParticipantWaitPicosecondsEstimate,
      policy.f16Bf16NPULogicalOpsPerSecondPerTile,
      policy.f16Bf16VectorLogicalOpsPerSecondPerTile,
      policy.f32VectorLogicalOpsPerSecondPerTile,
      policy.spmExplicitMovementBytesPerSecondPerTileEstimate};
  if (policy.profileIdentity == 0 || llvm::is_contained(rates, uint64_t{0})) {
    if (failureReason)
      *failureReason = policy.profileIdentity == 0
                           ? "search cost cohort requires a nonzero profile "
                             "identity"
                           : "search cost cohort requires positive rates for "
                             "every enabled resource term";
    return mlir::failure();
  }
  return SearchCostCohort(policy);
}

SearchObjective
deriveSearchObjective(const analysis::InstructionProgramAggregateCost &cost,
                      const std::optional<SearchCostCohort> &cohort) {
  if (!cohort)
    return UnknownSearchObjective{SearchObjectiveUnknownReason::NoCohort};
  const SearchCostPolicy &policy = cohort->getPolicy();
  if (!cost.aggregateCompute.npuOtherLogicalOps.isKnown() ||
      !cost.aggregateCompute.vectorOtherLogicalOps.isKnown())
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::MetricUnavailable};
  if (cost.aggregateCompute.npuOtherLogicalOps.value != 0 ||
      cost.aggregateCompute.vectorOtherLogicalOps.value != 0)
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::UncalibratedWork};

  auto npu = maximumTileMetric(
      cost,
      [](const analysis::InstructionProgramCost &tile)
          -> const analysis::ScheduleCostMetric & {
        return tile.compute.npuF16Bf16LogicalOps;
      },
      cost.aggregateCompute.npuF16Bf16LogicalOps);
  auto vectorF16 = maximumTileMetric(
      cost,
      [](const analysis::InstructionProgramCost &tile)
          -> const analysis::ScheduleCostMetric & {
        return tile.compute.vectorF16Bf16LogicalOps;
      },
      cost.aggregateCompute.vectorF16Bf16LogicalOps);
  auto vectorF32 = maximumTileMetric(
      cost,
      [](const analysis::InstructionProgramCost &tile)
          -> const analysis::ScheduleCostMetric & {
        return tile.compute.vectorF32LogicalOps;
      },
      cost.aggregateCompute.vectorF32LogicalOps);
  auto spm = maximumTileMetric(
      cost,
      [](const analysis::InstructionProgramCost &tile)
          -> const analysis::ScheduleCostMetric & {
        return tile.spmMovementBytes;
      },
      cost.aggregateSPMMovementBytes);
  auto instructions = maximumTileMetric(
      cost,
      [](const analysis::InstructionProgramCost &tile)
          -> const analysis::ScheduleCostMetric & {
        return tile.instructionCount;
      },
      cost.aggregateInstructionCount);
  auto dteWaits = maximumTileMetric(
      cost,
      [](const analysis::InstructionProgramCost &tile)
          -> const analysis::ScheduleCostMetric & {
        return tile.noc.waitedEventCount;
      },
      cost.aggregateNoC.waitedEventCount);
  auto nccWaits = maximumTileMetric(
      cost,
      [](const analysis::InstructionProgramCost &tile)
          -> const analysis::ScheduleCostMetric & {
        return tile.nccParticipantWaitCount;
      },
      cost.aggregateNCCParticipantWaitCount);
  if (!npu || !vectorF16 || !vectorF32 || !spm || !instructions || !dteWaits ||
      !nccWaits || !cost.aggregateDDRReadBytes.isKnown() ||
      !cost.aggregateDDRWriteBytes.isKnown() ||
      !cost.aggregateNoC.staticIssueSiteCount.isKnown() ||
      !cost.maximumTileNoCTransmitBytes.isKnown() ||
      !cost.maximumTileNoCTransmitMessageCount.isKnown() ||
      !cost.minimumHopMessageDemand.isKnown())
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::MetricUnavailable};

  uint64_t ddrBytes = 0;
  if (!checkedAdd(cost.aggregateDDRReadBytes.value,
                  cost.aggregateDDRWriteBytes.value, ddrBytes))
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::ArithmeticOverflow};
  uint64_t nocBytes = 0;
  if (cost.aggregateNoC.staticIssueSiteCount.value != 0) {
    if (!cost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown())
      return UnknownSearchObjective{
          SearchObjectiveUnknownReason::MetricUnavailable};
    nocBytes = cost.modeledNoCRoute.peakDirectedLinkByteDemand.value;
  }

  const uint64_t dteEndpointBytes = cost.maximumTileNoCTransmitBytes.value;
  const uint64_t dteMessageCount =
      cost.maximumTileNoCTransmitMessageCount.value;
  const uint64_t hopMessageDemand = cost.minimumHopMessageDemand.value;

  SearchResourceDurations durations;
  auto assignTime = [&](uint64_t work, uint64_t rate, uint64_t &destination) {
    std::optional<uint64_t> duration = timeForWork(work, rate);
    if (!duration)
      return false;
    destination = *duration;
    return true;
  };
  if (!assignTime(*npu, policy.f16Bf16NPULogicalOpsPerSecondPerTile,
                  durations.neF16Bf16Picoseconds) ||
      !assignTime(*vectorF16, policy.f16Bf16VectorLogicalOpsPerSecondPerTile,
                  durations.vectorF16Bf16Picoseconds) ||
      !assignTime(*vectorF32, policy.f32VectorLogicalOpsPerSecondPerTile,
                  durations.vectorF32Picoseconds) ||
      !assignTime(ddrBytes, policy.ddrNominalBytesPerSecond,
                  durations.ddrPicoseconds) ||
      !assignTime(nocBytes, policy.directionalNoCBytesPerSecond,
                  durations.nocPicoseconds) ||
      !assignTime(dteEndpointBytes, policy.dteEndpointBytesPerSecondEstimate,
                  durations.dteEndpointPicoseconds) ||
      !checkedMultiply(dteMessageCount,
                       policy.dteMessageStartupPicosecondsEstimate,
                       durations.dteStartupPicoseconds) ||
      !checkedMultiply(hopMessageDemand, policy.noCHopPicosecondsEstimate,
                       durations.nocHopPicoseconds) ||
      !assignTime(*spm, policy.spmExplicitMovementBytesPerSecondPerTileEstimate,
                  durations.spmMovementPicoseconds) ||
      !checkedMultiply(*instructions,
                       policy.instructionFixedPicosecondsEstimate,
                       durations.instructionControlPicoseconds) ||
      !checkedMultiply(*dteWaits, policy.dteWaitedEventPicosecondsEstimate,
                       durations.dteWaitControlPicoseconds) ||
      !checkedMultiply(*nccWaits, policy.nccParticipantWaitPicosecondsEstimate,
                       durations.nccWaitControlPicoseconds))
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::ArithmeticOverflow};
  return KnownSearchObjective{durations, *cohort};
}

SearchObjectiveComparison compareSearchObjectives(const SearchObjective &lhs,
                                                  const SearchObjective &rhs) {
  const auto *left = std::get_if<KnownSearchObjective>(&lhs);
  const auto *right = std::get_if<KnownSearchObjective>(&rhs);
  if (!left || !right || !(left->cohort == right->cohort))
    return SearchObjectiveComparison::Incomparable;
  const std::array<uint64_t, 12> leftTerms = asArray(left->durations);
  const std::array<uint64_t, 12> rightTerms = asArray(right->durations);
  bool noWorse = true;
  bool noBetter = true;
  bool strictlyBetter = false;
  bool strictlyWorse = false;
  for (auto [leftTerm, rightTerm] : llvm::zip_equal(leftTerms, rightTerms)) {
    noWorse &= leftTerm <= rightTerm;
    noBetter &= leftTerm >= rightTerm;
    strictlyBetter |= leftTerm < rightTerm;
    strictlyWorse |= leftTerm > rightTerm;
  }
  if (noWorse && strictlyBetter)
    return SearchObjectiveComparison::Better;
  if (noBetter && strictlyWorse)
    return SearchObjectiveComparison::Worse;
  return leftTerms == rightTerms ? SearchObjectiveComparison::Equivalent
                                 : SearchObjectiveComparison::Incomparable;
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
                               ActualCandidateResult result) {
  if (finished || poisoned || !reserved.erase(key) || completed.count(key))
    return failCompilerBug();
  completed.insert(key);
  switch (result.status) {
  case ActualCandidateStatus::Accepted: {
    if (!result.compilation || !result.compilation->isAccepted() ||
        !result.compilation->executable)
      return failCompilerBug();
    SearchObjective objective = deriveSearchObjective(
        result.compilation->executable->resourceCost, cohort);
    RetainedSearchCandidate candidate{key, objective,
                                      std::move(*result.compilation)};
    if (!referenceObjective)
      referenceObjective = candidate.objective;
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
    if (std::holds_alternative<UnknownSearchObjective>(candidate.objective))
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
  const auto *bound = std::get_if<KnownSearchObjective>(&lowerBound.objective);
  const auto *best = std::get_if<KnownSearchObjective>(&incumbent->objective);
  return bound && best && bound->cohort == best->cohort &&
         compareSearchObjectives(lowerBound.objective, incumbent->objective) ==
             SearchObjectiveComparison::Worse;
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
