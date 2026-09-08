//===- CostModel.cpp - Instruction program performance model ----------===//

#include "Wafer/Analysis/Instr/CostModel.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <utility>

namespace wafer::analysis {
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
maximumTileMetric(const InstructionProgramAggregateCost &cost,
                  Accessor accessor, const ScheduleCostMetric &aggregate) {
  if (cost.tileCosts.empty())
    return aggregate.isKnown() ? std::optional<uint64_t>(aggregate.value)
                               : std::nullopt;
  uint64_t maximum = 0;
  for (const InstructionProgramCost &tile : cost.tileCosts) {
    const ScheduleCostMetric &metric = accessor(tile);
    if (!metric.isKnown())
      return std::nullopt;
    maximum = std::max(maximum, metric.value);
  }
  return maximum;
}

std::variant<uint64_t, SearchObjectiveUnknownReason>
deriveNCCControlTime(const ScheduleCostMetric &joins,
                     const ScheduleCostMetric &participants,
                     const SearchCostPolicy &policy) {
  if (!joins.isKnown() || !participants.isKnown())
    return SearchObjectiveUnknownReason::MetricUnavailable;
  uint64_t calls = 0, waits = 0, total = 0;
  if (!checkedMultiply(joins.value, policy.nccJoinPicosecondsEstimate, calls) ||
      !checkedMultiply(participants.value,
                       policy.nccParticipantWaitPicosecondsEstimate, waits) ||
      !checkedAdd(calls, waits, total))
    return SearchObjectiveUnknownReason::ArithmeticOverflow;
  return total;
}

std::array<uint64_t, 12>
asServiceArray(const SearchResourceDurations &durations) {
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

std::array<uint64_t, 4>
asStorageArray(const SearchResourceDurations &durations) {
  return {durations.spmHighWaterBytes, durations.ddrHighWaterBytes,
          durations.spmBufferCount, durations.ddrBufferCount};
}

} // namespace

mlir::FailureOr<SearchCostCohort>
SearchCostCohort::create(const SearchCostPolicy &policy,
                         std::string *failureReason) {
  const std::array<uint64_t, 14> rates{
      policy.ddrNominalBytesPerSecond,
      policy.directionalNoCBytesPerSecond,
      policy.dteEndpointBytesPerSecondEstimate,
      policy.dteFirstMessagePicosecondsEstimate,
      policy.dteMessageStartupPicosecondsEstimate,
      policy.noCHopPicosecondsEstimate,
      policy.instructionFixedPicosecondsEstimate,
      policy.dteWaitedEventPicosecondsEstimate,
      policy.nccJoinPicosecondsEstimate,
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
deriveSearchObjective(const InstructionProgramAggregateCost &cost,
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
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.compute.npuF16Bf16LogicalOps;
      },
      cost.aggregateCompute.npuF16Bf16LogicalOps);
  auto vectorF16 = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.compute.vectorF16Bf16LogicalOps;
      },
      cost.aggregateCompute.vectorF16Bf16LogicalOps);
  auto vectorF32 = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.compute.vectorF32LogicalOps;
      },
      cost.aggregateCompute.vectorF32LogicalOps);
  auto spm = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.spmMovementBytes;
      },
      cost.aggregateSPMMovementBytes);
  auto instructions = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.instructionCount;
      },
      cost.aggregateInstructionCount);
  auto dteWaits = maximumTileMetric(
      cost,
      [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
        return tile.noc.waitedEventCount;
      },
      cost.aggregateNoC.waitedEventCount);
  if (!npu || !vectorF16 || !vectorF32 || !spm || !instructions || !dteWaits ||
      !cost.aggregateDDRReadBytes.isKnown() ||
      !cost.aggregateDDRWriteBytes.isKnown() ||
      !cost.aggregateNoC.staticIssueSiteCount.isKnown() ||
      !cost.maximumTileNoCTransmitBytes.isKnown() ||
      !cost.maximumTileNoCTransmitMessageCount.isKnown() ||
      !cost.minimumHopMessageDemand.isKnown() ||
      !cost.maximumTileSPMHighWaterBytes.isKnown() ||
      !cost.maximumTileDDRHighWaterBytes.isKnown() ||
      !cost.aggregateCompilerOwnedSPMBufferCount.isKnown() ||
      !cost.aggregateCompilerOwnedDDRBufferCount.isKnown())
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
  if (cost.tileCosts.empty()) {
    auto control =
        deriveNCCControlTime(cost.aggregateNCCJoinCount,
                             cost.aggregateNCCParticipantWaitCount, policy);
    if (const auto *reason =
            std::get_if<SearchObjectiveUnknownReason>(&control))
      return UnknownSearchObjective{*reason};
    durations.nccWaitControlPicoseconds = std::get<uint64_t>(control);
  } else {
    // Join and participant maxima need not belong to the same Tile.
    for (const InstructionProgramCost &tile : cost.tileCosts) {
      auto control = deriveNCCControlTime(tile.nccJoinCount,
                                          tile.nccParticipantWaitCount, policy);
      if (const auto *reason =
              std::get_if<SearchObjectiveUnknownReason>(&control))
        return UnknownSearchObjective{*reason};
      durations.nccWaitControlPicoseconds = std::max(
          durations.nccWaitControlPicoseconds, std::get<uint64_t>(control));
    }
  }
  if (dteMessageCount != 0 &&
      (!checkedMultiply(dteMessageCount - 1,
                        policy.dteMessageStartupPicosecondsEstimate,
                        durations.dteStartupPicoseconds) ||
       !checkedAdd(durations.dteStartupPicoseconds,
                   policy.dteFirstMessagePicosecondsEstimate,
                   durations.dteStartupPicoseconds)))
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::ArithmeticOverflow};
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
      !checkedMultiply(hopMessageDemand, policy.noCHopPicosecondsEstimate,
                       durations.nocHopPicoseconds) ||
      !assignTime(*spm, policy.spmExplicitMovementBytesPerSecondPerTileEstimate,
                  durations.spmMovementPicoseconds) ||
      !checkedMultiply(*instructions,
                       policy.instructionFixedPicosecondsEstimate,
                       durations.instructionControlPicoseconds) ||
      !checkedMultiply(*dteWaits, policy.dteWaitedEventPicosecondsEstimate,
                       durations.dteWaitControlPicoseconds))
    return UnknownSearchObjective{
        SearchObjectiveUnknownReason::ArithmeticOverflow};
  durations.spmHighWaterBytes = cost.maximumTileSPMHighWaterBytes.value;
  durations.ddrHighWaterBytes = cost.maximumTileDDRHighWaterBytes.value;
  durations.spmBufferCount = cost.aggregateCompilerOwnedSPMBufferCount.value;
  durations.ddrBufferCount = cost.aggregateCompilerOwnedDDRBufferCount.value;
  return KnownSearchObjective{durations, *cohort};
}

SearchObjectiveComparison compareSearchObjectives(const SearchObjective &lhs,
                                                  const SearchObjective &rhs) {
  const auto *left = std::get_if<KnownSearchObjective>(&lhs);
  const auto *right = std::get_if<KnownSearchObjective>(&rhs);
  if (!left || !right || !(left->cohort == right->cohort))
    return SearchObjectiveComparison::Incomparable;
  const std::array<uint64_t, 12> leftTerms = asServiceArray(left->durations);
  const std::array<uint64_t, 12> rightTerms = asServiceArray(right->durations);
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
  if (leftTerms != rightTerms)
    return SearchObjectiveComparison::Incomparable;
  const std::array<uint64_t, 4> leftStorage = asStorageArray(left->durations);
  const std::array<uint64_t, 4> rightStorage = asStorageArray(right->durations);
  noWorse = true;
  noBetter = true;
  strictlyBetter = false;
  strictlyWorse = false;
  for (auto [leftTerm, rightTerm] :
       llvm::zip_equal(leftStorage, rightStorage)) {
    noWorse &= leftTerm <= rightTerm;
    noBetter &= leftTerm >= rightTerm;
    strictlyBetter |= leftTerm < rightTerm;
    strictlyWorse |= leftTerm > rightTerm;
  }
  if (noWorse && strictlyBetter)
    return SearchObjectiveComparison::Better;
  if (noBetter && strictlyWorse)
    return SearchObjectiveComparison::Worse;
  return leftStorage == rightStorage ? SearchObjectiveComparison::Equivalent
                                     : SearchObjectiveComparison::Incomparable;
}

} // namespace wafer::analysis
