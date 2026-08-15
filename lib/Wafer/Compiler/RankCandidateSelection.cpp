//===- RankCandidateSelection.cpp - Rank candidate selection -----------===//

#include "RankCandidateSelection.h"

#include "StaticFixedSlotQualification.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/IR/WaferInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

static bool isQualificationSelectionMode(RankCandidateSelectionMode mode) {
  return mode != RankCandidateSelectionMode::Production &&
         mode != RankCandidateSelectionMode::ReservedBaseline;
}

static bool
hasValidAdmittedIdentities(llvm::ArrayRef<EvaluatedRankCandidate> variants,
                           bool requireBaseline) {
  if (variants.empty())
    return false;
  std::optional<size_t> baselineIndex;
  llvm::SmallVector<std::pair<int64_t, uint32_t>, 16> ordinals;
  ordinals.reserve(variants.size());
  for (auto [index, variant] : llvm::enumerate(variants)) {
    std::pair<int64_t, uint32_t> ordinal{variant.stableSemanticOrdinal,
                                         variant.scheduleActionOrdinal};
    if (variant.stableSemanticOrdinal < 0 ||
        llvm::is_contained(ordinals, ordinal))
      return false;
    ordinals.push_back(ordinal);
    if (!variant.reservedBaseline)
      continue;
    if (baselineIndex)
      return false;
    baselineIndex = index;
  }
  return !requireBaseline || baselineIndex.has_value();
}

static bool
hasNoReservedRankIdentity(const EvaluatedRankCandidate &candidate) {
  return !candidate.variant.ranks.empty() && !candidate.reservedBaseline;
}

static bool matchesStaticFixedSlotQualification(
    const EvaluatedRankCandidate &candidate) {
  return hasNoReservedRankIdentity(candidate) &&
         candidate.actionKey.bufferingKind ==
             CoordinatedBufferingKind::StaticFixedSlot &&
         candidate.actionKey.bufferingPlanOrdinal > 0 &&
         llvm::all_of(candidate.variant.ranks, [](const RankExecutable &rank) {
           return hasStaticFixedSlotQualificationEvidence(rank);
         });
}

static bool matchesDirectDTEComputeOverlapQualification(
    const EvaluatedRankCandidate &candidate) {
  if (!matchesStaticFixedSlotQualification(candidate) ||
      candidate.variant.resourceCost.rankCosts.size() !=
          candidate.variant.ranks.size())
    return false;
  return llvm::all_of(
      candidate.variant.resourceCost.rankCosts,
      [](const analysis::InstructionProgramCost &cost) {
        return cost.directDTEComputeOverlapWindowCount.isKnown() &&
               cost.directDTEComputeOverlapWindowCount.value > 0;
      });
}

static bool matchesSerializedDirectDTEComputeQualification(
    const EvaluatedRankCandidate &candidate) {
  if (candidate.actionKey.serializationKind !=
          CoordinatedScheduleSerializationKind::DirectDTEComputeWindows ||
      !matchesStaticFixedSlotQualification(candidate) ||
      candidate.variant.resourceCost.rankCosts.size() !=
          candidate.variant.ranks.size())
    return false;
  return llvm::all_of(
      candidate.variant.resourceCost.rankCosts,
      [](const analysis::InstructionProgramCost &cost) {
        return cost.directDTEComputeOverlapWindowCount.isKnown() &&
               cost.directDTEComputeOverlapWindowCount.value == 0;
      });
}

static bool hasMultipleActualNCCWorkers(const RankExecutable &rank) {
  std::set<NCCWorker> workers;
  rank.getModule().walk([&](mlir::Operation *operation) {
    if (std::optional<NCCWorker> worker = getNCCIssueWorker(operation))
      workers.insert(*worker);
  });
  return workers.size() >= 2 && llvm::any_of(workers, [](NCCWorker worker) {
           return worker != NCCWorker::Worker0;
         });
}

static bool matchesWorkerPlacementQualification(
    const EvaluatedRankCandidate &candidate) {
  return hasNoReservedRankIdentity(candidate) &&
         candidate.actionKey.workerPlacementKind ==
             CoordinatedWorkerPlacementKind::DisjointComponents &&
         candidate.actionKey.workerPlacementPlanOrdinal > 0 &&
         llvm::all_of(candidate.variant.ranks, [](const RankExecutable &rank) {
           return hasMultipleActualNCCWorkers(rank);
         });
}

static bool
hasAllRankTypedNoCAndBidirectionalDomain(llvm::ArrayRef<RankExecutable> ranks) {
  if (ranks.empty())
    return false;
  bool domainHasSend = false;
  bool domainHasRecv = false;
  for (const RankExecutable &rank : ranks) {
    bool rankHasSend = false;
    bool rankHasRecv = false;
    rank.getModule().walk([&](mlir::Operation *operation) {
      rankHasSend |= mlir::isa<InstrDTESendOp>(operation);
      rankHasRecv |= mlir::isa<InstrDTERecvOp>(operation);
    });
    if (!rankHasSend && !rankHasRecv)
      return false;
    domainHasSend |= rankHasSend;
    domainHasRecv |= rankHasRecv;
  }
  return domainHasSend && domainHasRecv;
}

static bool hasBoundaryOnlyDDRMovement(llvm::ArrayRef<RankExecutable> ranks) {
  return !ranks.empty() && llvm::all_of(ranks, [](const RankExecutable &rank) {
    return hasBoundaryOnlyDDRMovementEvidence(rank);
  });
}

static std::set<DTEProtocolPhase>
observeCollectivePhases(const RankExecutable &rank) {
  std::set<DTEProtocolPhase> phases;
  rank.getModule().walk(
      [&](InstrDTESendOp op) { phases.insert(op.getMessage().getPhase()); });
  rank.getModule().walk(
      [&](InstrDTERecvOp op) { phases.insert(op.getMessage().getPhase()); });
  return phases;
}

static std::optional<std::set<DTEProtocolPhase>>
getExpectedCollectivePhases(RankCandidateSelectionMode mode) {
  switch (mode) {
  case RankCandidateSelectionMode::CharacterizeAllGatherDirect:
    return std::set<DTEProtocolPhase>{DTEProtocolPhase::AllGatherDirect};
  case RankCandidateSelectionMode::CharacterizeAllGatherRing:
    return std::set<DTEProtocolPhase>{DTEProtocolPhase::AllGatherRing};
  case RankCandidateSelectionMode::CharacterizeReduceScatterDirect:
    return std::set<DTEProtocolPhase>{DTEProtocolPhase::ReduceScatterDirect};
  case RankCandidateSelectionMode::CharacterizeReduceScatterRing:
    return std::set<DTEProtocolPhase>{DTEProtocolPhase::ReduceScatterRing};
  case RankCandidateSelectionMode::CharacterizeAllReduceRing:
  case RankCandidateSelectionMode::QualifyNoCResidentAllReduceRing:
    return std::set<DTEProtocolPhase>{DTEProtocolPhase::AllReduceRing};
  case RankCandidateSelectionMode::CharacterizeAllReduceTree:
    return std::set<DTEProtocolPhase>{DTEProtocolPhase::AllReduceTreeReduce,
                                      DTEProtocolPhase::AllReduceTreeBroadcast};
  case RankCandidateSelectionMode::Production:
  case RankCandidateSelectionMode::ReservedBaseline:
  case RankCandidateSelectionMode::QualifyStaticFixedSlot:
  case RankCandidateSelectionMode::QualifyDirectDTEComputeOverlap:
  case RankCandidateSelectionMode::SelectSerializedDirectDTEComputeBaseline:
  case RankCandidateSelectionMode::QualifyWorkerPlacement:
  case RankCandidateSelectionMode::QualifyNoCResidentFixedSlotWorker:
    return std::nullopt;
  }
  return std::nullopt;
}

static bool
matchesCollectiveCharacterization(const EvaluatedRankCandidate &candidate,
                                  RankCandidateSelectionMode mode) {
  std::optional<std::set<DTEProtocolPhase>> expected =
      getExpectedCollectivePhases(mode);
  return expected && !candidate.variant.ranks.empty() &&
         llvm::all_of(candidate.variant.ranks, [&](const RankExecutable &rank) {
           return observeCollectivePhases(rank) == *expected;
         });
}

static bool matchesQualification(const EvaluatedRankCandidate &candidate,
                                 RankCandidateSelectionMode mode) {
  switch (mode) {
  case RankCandidateSelectionMode::QualifyStaticFixedSlot:
    return matchesStaticFixedSlotQualification(candidate);
  case RankCandidateSelectionMode::QualifyDirectDTEComputeOverlap:
    return matchesDirectDTEComputeOverlapQualification(candidate);
  case RankCandidateSelectionMode::SelectSerializedDirectDTEComputeBaseline:
    return matchesSerializedDirectDTEComputeQualification(candidate);
  case RankCandidateSelectionMode::QualifyWorkerPlacement:
    return matchesWorkerPlacementQualification(candidate);
  case RankCandidateSelectionMode::QualifyNoCResidentFixedSlotWorker:
    return matchesStaticFixedSlotQualification(candidate) &&
           matchesWorkerPlacementQualification(candidate) &&
           hasBoundaryOnlyDDRMovement(candidate.variant.ranks) &&
           hasAllRankTypedNoCAndBidirectionalDomain(candidate.variant.ranks);
  case RankCandidateSelectionMode::QualifyNoCResidentAllReduceRing:
    return hasNoReservedRankIdentity(candidate) &&
           matchesCollectiveCharacterization(candidate, mode) &&
           hasBoundaryOnlyDDRMovement(candidate.variant.ranks) &&
           hasAllRankTypedNoCAndBidirectionalDomain(candidate.variant.ranks);
  case RankCandidateSelectionMode::CharacterizeAllGatherDirect:
  case RankCandidateSelectionMode::CharacterizeAllGatherRing:
  case RankCandidateSelectionMode::CharacterizeReduceScatterDirect:
  case RankCandidateSelectionMode::CharacterizeReduceScatterRing:
  case RankCandidateSelectionMode::CharacterizeAllReduceRing:
  case RankCandidateSelectionMode::CharacterizeAllReduceTree:
    return matchesCollectiveCharacterization(candidate, mode);
  case RankCandidateSelectionMode::Production:
  case RankCandidateSelectionMode::ReservedBaseline:
    return false;
  }
  return false;
}

static llvm::StringRef
stringifyQualificationSelectionMode(RankCandidateSelectionMode mode) {
  switch (mode) {
  case RankCandidateSelectionMode::QualifyStaticFixedSlot:
    return "static-fixed-slot";
  case RankCandidateSelectionMode::QualifyDirectDTEComputeOverlap:
    return "direct-dte-compute-overlap";
  case RankCandidateSelectionMode::SelectSerializedDirectDTEComputeBaseline:
    return "serialized-direct-dte-compute";
  case RankCandidateSelectionMode::QualifyWorkerPlacement:
    return "worker-placement";
  case RankCandidateSelectionMode::QualifyNoCResidentFixedSlotWorker:
    return "noc-resident-fixed-slot-worker";
  case RankCandidateSelectionMode::CharacterizeAllGatherDirect:
  case RankCandidateSelectionMode::CharacterizeAllGatherRing:
  case RankCandidateSelectionMode::CharacterizeReduceScatterDirect:
  case RankCandidateSelectionMode::CharacterizeReduceScatterRing:
  case RankCandidateSelectionMode::CharacterizeAllReduceRing:
  case RankCandidateSelectionMode::QualifyNoCResidentAllReduceRing:
  case RankCandidateSelectionMode::CharacterizeAllReduceTree:
    return getCollectiveCharacterizationAlternative(mode);
  case RankCandidateSelectionMode::Production:
    return "production";
  case RankCandidateSelectionMode::ReservedBaseline:
    return "reserved-baseline";
  }
  return "";
}

enum class CostOrder : uint8_t {
  Equivalent,
  LeftDominates,
  RightDominates,
  Incomparable,
};

enum class MetricPreference : uint8_t { Lower, Higher };

struct CostOrderAccumulator {
  bool leftBetter = false;
  bool rightBetter = false;
  bool incomparable = false;

  void compare(const analysis::ScheduleCostMetric &left,
               const analysis::ScheduleCostMetric &right,
               MetricPreference preference = MetricPreference::Lower) {
    // Unknown values remain incomparable even when their reason matches: a
    // shared failure mode is not a proof that the hidden quantities are equal.
    // No disposition is ever converted to an implicit zero.
    if (!left.isKnown() || !right.isKnown()) {
      incomparable = true;
      return;
    }
    if (left.value == right.value)
      return;
    bool leftPreferred = left.value < right.value;
    if (preference == MetricPreference::Higher)
      leftPreferred = !leftPreferred;
    leftBetter |= leftPreferred;
    rightBetter |= !leftPreferred;
  }

  CostOrder finish() const {
    if (incomparable || (leftBetter && rightBetter))
      return CostOrder::Incomparable;
    if (leftBetter)
      return CostOrder::LeftDominates;
    if (rightBetter)
      return CostOrder::RightDominates;
    return CostOrder::Equivalent;
  }
};

static void
compareExecutionCount(CostOrderAccumulator &order,
                      const analysis::InstructionExecutionCount &left,
                      const analysis::InstructionExecutionCount &right) {
  order.compare(left.staticSites, right.staticSites);
  order.compare(left.exactExecutions, right.exactExecutions);
  order.compare(left.lowerBound, right.lowerBound);
  order.compare(left.upperBound, right.upperBound);
}

static void compareProgramWork(CostOrderAccumulator &order,
                               const analysis::InstructionProgramWork &left,
                               const analysis::InstructionProgramWork &right) {
  compareExecutionCount(order, left.instructions, right.instructions);
  compareExecutionCount(order, left.asynchronousEvents,
                        right.asynchronousEvents);
  compareExecutionCount(order, left.rdmaIssues, right.rdmaIssues);
  compareExecutionCount(order, left.wdmaIssues, right.wdmaIssues);
  compareExecutionCount(order, left.tdmaIssues, right.tdmaIssues);
  compareExecutionCount(order, left.ctIssues, right.ctIssues);
  compareExecutionCount(order, left.neIssues, right.neIssues);
  compareExecutionCount(order, left.dteOperations, right.dteOperations);
  compareExecutionCount(order, left.gatherScatterOperations,
                        right.gatherScatterOperations);
  compareExecutionCount(order, left.dteSendOperations, right.dteSendOperations);
  compareExecutionCount(order, left.dteReceiveOperations,
                        right.dteReceiveOperations);
  compareExecutionCount(order, left.dteWaitOperations, right.dteWaitOperations);
  compareExecutionCount(order, left.collectiveDTEIssues,
                        right.collectiveDTEIssues);
  compareExecutionCount(order, left.peerDTEIssues, right.peerDTEIssues);
  compareExecutionCount(order, left.nccJoins, right.nccJoins);
  compareExecutionCount(order, left.steadyStateNCCJoins,
                        right.steadyStateNCCJoins);
  compareExecutionCount(order, left.nonTerminalNCCJoins,
                        right.nonTerminalNCCJoins);
  compareExecutionCount(order, left.nccParticipantWaits,
                        right.nccParticipantWaits);
  compareExecutionCount(order, left.steadyStateNCCParticipantWaits,
                        right.steadyStateNCCParticipantWaits);
  compareExecutionCount(order, left.nonTerminalNCCParticipantWaits,
                        right.nonTerminalNCCParticipantWaits);
  compareExecutionCount(order, left.intrinsicNCCDrains,
                        right.intrinsicNCCDrains);
}

static void compareComputeCost(CostOrderAccumulator &order,
                               const analysis::ScheduleComputeCost &left,
                               const analysis::ScheduleComputeCost &right) {
  order.compare(left.npuF16Bf16LogicalOps, right.npuF16Bf16LogicalOps);
  order.compare(left.npuOtherLogicalOps, right.npuOtherLogicalOps);
  order.compare(left.vectorF16Bf16LogicalOps, right.vectorF16Bf16LogicalOps);
  order.compare(left.vectorF32LogicalOps, right.vectorF32LogicalOps);
  order.compare(left.vectorOtherLogicalOps, right.vectorOtherLogicalOps);
}

static void compareNoCCost(CostOrderAccumulator &order,
                           const analysis::ScheduleNoCCost &left,
                           const analysis::ScheduleNoCCost &right) {
  order.compare(left.staticIssueSiteCount, right.staticIssueSiteCount);
  order.compare(left.aggregateTransmitBytes, right.aggregateTransmitBytes);
  order.compare(left.aggregateReceiveBytes, right.aggregateReceiveBytes);
  order.compare(left.transmitMessageCount, right.transmitMessageCount);
  order.compare(left.receiveMessageCount, right.receiveMessageCount);
  order.compare(left.waitOperationCount, right.waitOperationCount);
  order.compare(left.waitedEventCount, right.waitedEventCount);
  for (auto [leftMetric, rightMetric] :
       llvm::zip(left.directionalTransmitBytes, right.directionalTransmitBytes))
    order.compare(leftMetric, rightMetric);
  for (auto [leftMetric, rightMetric] :
       llvm::zip(left.collectiveTransmitBytes, right.collectiveTransmitBytes))
    order.compare(leftMetric, rightMetric);
}

static CostOrder compareExactSelectionCost(
    const analysis::CardInstructionProgramCost &left,
    const analysis::CardInstructionProgramCost &right) {
  CostOrderAccumulator order;
  compareProgramWork(order, left.aggregateWork, right.aggregateWork);
  compareProgramWork(order, left.maximumRankWork, right.maximumRankWork);
  compareComputeCost(order, left.aggregateCompute, right.aggregateCompute);
  compareComputeCost(order, left.maximumRankCompute, right.maximumRankCompute);
  order.compare(left.aggregateDDRReadBytes, right.aggregateDDRReadBytes);
  order.compare(left.aggregateDDRWriteBytes, right.aggregateDDRWriteBytes);
  order.compare(left.aggregateSPMMovementBytes,
                right.aggregateSPMMovementBytes);
  order.compare(left.aggregateGatherScatterBytes,
                right.aggregateGatherScatterBytes);
  order.compare(left.maximumRankSPMMovementBytes,
                right.maximumRankSPMMovementBytes);
  order.compare(left.maximumRankGatherScatterBytes,
                right.maximumRankGatherScatterBytes);
  compareNoCCost(order, left.aggregateNoC, right.aggregateNoC);
  order.compare(left.maximumRankNoCTransmitBytes,
                right.maximumRankNoCTransmitBytes);
  order.compare(left.maximumRankNoCReceiveBytes,
                right.maximumRankNoCReceiveBytes);
  order.compare(left.maximumRankNoCTransmitMessageCount,
                right.maximumRankNoCTransmitMessageCount);
  order.compare(left.maximumRankNoCReceiveMessageCount,
                right.maximumRankNoCReceiveMessageCount);
  order.compare(left.minimumHopLinkByteDemand, right.minimumHopLinkByteDemand);
  order.compare(left.minimumHopMessageDemand, right.minimumHopMessageDemand);
  order.compare(left.directedNoCLinkCount, right.directedNoCLinkCount);
  order.compare(left.idealizedMinimumPeakLinkByteDemand,
                right.idealizedMinimumPeakLinkByteDemand);
  order.compare(left.modeledNoCRoute.peakDirectedLinkByteDemand,
                right.modeledNoCRoute.peakDirectedLinkByteDemand);
  order.compare(left.maximumNoCHopCount, right.maximumNoCHopCount);
  order.compare(left.aggregateInstructionCount,
                right.aggregateInstructionCount);
  order.compare(left.aggregateEventCount, right.aggregateEventCount);
  order.compare(left.aggregateNCCJoinCount, right.aggregateNCCJoinCount);
  order.compare(left.aggregateSteadyStateNCCJoinCount,
                right.aggregateSteadyStateNCCJoinCount);
  order.compare(left.aggregateNonTerminalNCCJoinCount,
                right.aggregateNonTerminalNCCJoinCount);
  order.compare(left.aggregateNCCParticipantWaitCount,
                right.aggregateNCCParticipantWaitCount);
  order.compare(left.aggregateSteadyStateNCCParticipantWaitCount,
                right.aggregateSteadyStateNCCParticipantWaitCount);
  order.compare(left.aggregateNonTerminalNCCParticipantWaitCount,
                right.aggregateNonTerminalNCCParticipantWaitCount);
  order.compare(left.aggregateIntrinsicNCCDrainCount,
                right.aggregateIntrinsicNCCDrainCount);
  order.compare(left.maximumRankDataDependencyDepth,
                right.maximumRankDataDependencyDepth);
  order.compare(left.aggregateReadyOrderPriorityInversions,
                right.aggregateReadyOrderPriorityInversions);
  order.compare(left.aggregateQualifiedOverlapWindowCount,
                right.aggregateQualifiedOverlapWindowCount,
                MetricPreference::Higher);
  order.compare(left.aggregateDirectDTEComputeOverlapWindowCount,
                right.aggregateDirectDTEComputeOverlapWindowCount,
                MetricPreference::Higher);

  // Accepted SPM high-water is a hard-capacity/headroom fact and is
  // intentionally absent from execution Pareto ordering.
  order.compare(left.maximumRankDDRHighWaterBytes,
                right.maximumRankDDRHighWaterBytes);
  order.compare(left.summedRankDDRHighWaterBytes,
                right.summedRankDDRHighWaterBytes);
  order.compare(left.aggregateCompilerOwnedSPMBufferCount,
                right.aggregateCompilerOwnedSPMBufferCount);
  order.compare(left.aggregateCompilerOwnedDDRBufferCount,
                right.aggregateCompilerOwnedDDRBufferCount);
  order.compare(left.maximumRankCompilerOwnedSPMBufferCount,
                right.maximumRankCompilerOwnedSPMBufferCount);
  order.compare(left.maximumRankCompilerOwnedDDRBufferCount,
                right.maximumRankCompilerOwnedDDRBufferCount);

  return order.finish();
}

static analysis::StaticCrossResourceSchedule
getScheduleContext(const analysis::CardInstructionProgramCost &cost) {
  const analysis::ScheduleCostMetric windows[] = {
      cost.aggregateQualifiedOverlapWindowCount,
      cost.aggregateDirectDTEComputeOverlapWindowCount,
  };
  if (llvm::any_of(windows, [](const analysis::ScheduleCostMetric &metric) {
        return metric.isKnown() && metric.value != 0;
      }))
    return analysis::StaticCrossResourceSchedule::PipelinedSteadyState;
  return analysis::StaticCrossResourceSchedule::SequentialPhases;
}

static bool marginClears(uint64_t candidate, uint64_t baseline,
                         uint32_t marginPermille) {
  unsigned __int128 left =
      static_cast<unsigned __int128>(candidate) * (1000ULL + marginPermille);
  unsigned __int128 right = static_cast<unsigned __int128>(baseline) * 1000ULL;
  return left < right;
}

static bool sameMetric(const analysis::ScheduleCostMetric &left,
                       const analysis::ScheduleCostMetric &right) {
  return left.value == right.value && left.knowledge == right.knowledge &&
         left.reason == right.reason;
}

static bool sameInterval(const analysis::StaticDurationInterval &left,
                         const analysis::StaticDurationInterval &right) {
  return sameMetric(left.lowerBoundPicoseconds, right.lowerBoundPicoseconds) &&
         sameMetric(left.nominalPicoseconds, right.nominalPicoseconds) &&
         sameMetric(left.upperBoundPicoseconds, right.upperBoundPicoseconds) &&
         left.nominalAssumptions == right.nominalAssumptions;
}

static bool
sameDurationEstimate(const analysis::ProgramDurationEstimate &left,
                     const analysis::ProgramDurationEstimate &right) {
  return sameInterval(left.ddr, right.ddr) &&
         sameInterval(left.compute, right.compute) &&
         sameInterval(left.noc, right.noc) &&
         sameInterval(left.spm, right.spm) &&
         sameInterval(left.control, right.control) &&
         sameInterval(left.makespan, right.makespan);
}

static bool compareKnownNoWorse(const analysis::ScheduleCostMetric &candidate,
                                const analysis::ScheduleCostMetric &baseline,
                                bool &improved) {
  if (!candidate.isKnown() || !baseline.isKnown() ||
      candidate.value > baseline.value)
    return false;
  improved |= candidate.value < baseline.value;
  return true;
}

static bool isKnownZero(const analysis::ScheduleCostMetric &metric) {
  return metric.isKnown() && metric.value == 0;
}

static bool
accumulateKnownWeightedDelta(const analysis::ScheduleCostMetric &candidate,
                             const analysis::ScheduleCostMetric &baseline,
                             uint64_t weight, unsigned __int128 &benefit,
                             unsigned __int128 &regression) {
  if (!candidate.isKnown() || !baseline.isKnown())
    return false;
  if (candidate.value == baseline.value)
    return true;
  if (weight == 0)
    return false;

  const uint64_t difference = candidate.value < baseline.value
                                  ? baseline.value - candidate.value
                                  : candidate.value - baseline.value;
  const unsigned __int128 term =
      static_cast<unsigned __int128>(difference) * weight;
  unsigned __int128 &destination =
      candidate.value < baseline.value ? benefit : regression;
  constexpr unsigned __int128 maximum = ~static_cast<unsigned __int128>(0);
  if (term > maximum - destination)
    return false;
  destination += term;
  return true;
}

static bool deltaMarginClears(unsigned __int128 benefit,
                              unsigned __int128 regression,
                              uint32_t marginPermille) {
  if (benefit == 0)
    return false;
  if (regression == 0)
    return true;
  constexpr unsigned __int128 maximum = ~static_cast<unsigned __int128>(0);
  const uint64_t regressionFactor = 1000ULL + marginPermille;
  if (regression > maximum / regressionFactor || benefit > maximum / 1000ULL)
    return false;
  return regression * regressionFactor < benefit * 1000ULL;
}

static bool hasKnownPrimitiveWorkPromotion(
    const analysis::CardInstructionProgramCost &candidate,
    const analysis::CardInstructionProgramCost &baseline,
    const analysis::TargetScheduleCostPolicy &policy) {
  if (candidate.rankCosts.empty() ||
      candidate.rankCosts.size() != baseline.rankCosts.size())
    return false;

  // This narrow sub-margin path is only valid under the same explicit
  // sequential schedule contract. Pipelined variants keep the ordinary
  // whole-model margin because dependency and wait placement can affect their
  // overlap. Check both aggregate and rank-local witnesses so inconsistent
  // synthetic cost views cannot manufacture a context change.
  if (!isKnownZero(candidate.aggregateQualifiedOverlapWindowCount) ||
      !isKnownZero(baseline.aggregateQualifiedOverlapWindowCount) ||
      !isKnownZero(candidate.aggregateDirectDTEComputeOverlapWindowCount) ||
      !isKnownZero(baseline.aggregateDirectDTEComputeOverlapWindowCount))
    return false;

  bool improved = false;
  for (auto [candidateRank, baselineRank] :
       llvm::zip_equal(candidate.rankCosts, baseline.rankCosts)) {
    const analysis::ScheduleCostMetric *candidateCompute[] = {
        &candidateRank.compute.npuF16Bf16LogicalOps,
        &candidateRank.compute.npuOtherLogicalOps,
        &candidateRank.compute.vectorF16Bf16LogicalOps,
        &candidateRank.compute.vectorF32LogicalOps,
        &candidateRank.compute.vectorOtherLogicalOps,
    };
    const analysis::ScheduleCostMetric *baselineCompute[] = {
        &baselineRank.compute.npuF16Bf16LogicalOps,
        &baselineRank.compute.npuOtherLogicalOps,
        &baselineRank.compute.vectorF16Bf16LogicalOps,
        &baselineRank.compute.vectorF32LogicalOps,
        &baselineRank.compute.vectorOtherLogicalOps,
    };
    for (auto [candidateMetric, baselineMetric] :
         llvm::zip_equal(candidateCompute, baselineCompute))
      if (!compareKnownNoWorse(*candidateMetric, *baselineMetric, improved))
        return false;

    const analysis::ScheduleCostMetric *candidateMovement[] = {
        &candidateRank.ddrReadBytes,
        &candidateRank.ddrWriteBytes,
        &candidateRank.spmMovementBytes,
    };
    const analysis::ScheduleCostMetric *baselineMovement[] = {
        &baselineRank.ddrReadBytes,
        &baselineRank.ddrWriteBytes,
        &baselineRank.spmMovementBytes,
    };
    for (auto [candidateMetric, baselineMetric] :
         llvm::zip_equal(candidateMovement, baselineMovement))
      if (!compareKnownNoWorse(*candidateMetric, *baselineMetric, improved))
        return false;
    bool gatherGuard = false;
    if (!compareKnownNoWorse(candidateRank.gatherScatterBytes,
                             baselineRank.gatherScatterBytes, gatherGuard))
      return false;

    const analysis::ScheduleCostMetric *candidateNoC[] = {
        &candidateRank.noc.aggregateTransmitBytes,
        &candidateRank.noc.aggregateReceiveBytes,
        &candidateRank.noc.transmitMessageCount,
        &candidateRank.noc.receiveMessageCount,
    };
    const analysis::ScheduleCostMetric *baselineNoC[] = {
        &baselineRank.noc.aggregateTransmitBytes,
        &baselineRank.noc.aggregateReceiveBytes,
        &baselineRank.noc.transmitMessageCount,
        &baselineRank.noc.receiveMessageCount,
    };
    for (auto [candidateMetric, baselineMetric] :
         llvm::zip_equal(candidateNoC, baselineNoC))
      if (!compareKnownNoWorse(*candidateMetric, *baselineMetric, improved))
        return false;

    if (!isKnownZero(candidateRank.qualifiedOverlapWindowCount) ||
        !isKnownZero(baselineRank.qualifiedOverlapWindowCount) ||
        !isKnownZero(candidateRank.directDTEComputeOverlapWindowCount) ||
        !isKnownZero(baselineRank.directDTEComputeOverlapWindowCount))
      return false;

    if (!compareKnownNoWorse(candidateRank.intrinsicNCCDrainCount,
                             baselineRank.intrinsicNCCDrainCount, improved))
      return false;

    // Fixed issue and post-service wait priors describe one control resource.
    // Compare only the changed work, so a single extra completion wait cannot
    // hide a much larger reduction in issued instructions. Cross-kind
    // regressions must still clear the same production uncertainty margin.
    unsigned __int128 controlBenefit = 0;
    unsigned __int128 controlRegression = 0;
    if (!accumulateKnownWeightedDelta(
            candidateRank.instructionCount, baselineRank.instructionCount,
            policy.instructionFixedPicosecondsEstimate, controlBenefit,
            controlRegression) ||
        !accumulateKnownWeightedDelta(candidateRank.noc.waitedEventCount,
                                      baselineRank.noc.waitedEventCount,
                                      policy.dteWaitedEventPicosecondsEstimate,
                                      controlBenefit, controlRegression) ||
        !accumulateKnownWeightedDelta(
            candidateRank.nccParticipantWaitCount,
            baselineRank.nccParticipantWaitCount,
            policy.nccParticipantWaitPicosecondsEstimate, controlBenefit,
            controlRegression))
      return false;
    if (controlRegression != 0 &&
        !deltaMarginClears(controlBenefit, controlRegression,
                           policy.productionBenefitMarginPermille))
      return false;
    improved |= controlBenefit > controlRegression;
  }

  // Duration construction consumes a few all-rank aggregate/max summaries in
  // addition to rankCosts. They are derived duplicates in production, so they
  // may only guard against a regression here and never create benefit.
  bool summaryGuard = false;
  const analysis::ScheduleCostMetric *candidateSummaries[] = {
      &candidate.aggregateDDRReadBytes,
      &candidate.aggregateDDRWriteBytes,
      &candidate.aggregateSPMMovementBytes,
      &candidate.maximumRankSPMMovementBytes,
      &candidate.aggregateGatherScatterBytes,
      &candidate.maximumRankGatherScatterBytes,
      &candidate.aggregateNoC.staticIssueSiteCount,
      &candidate.aggregateNoC.aggregateTransmitBytes,
      &candidate.aggregateNoC.aggregateReceiveBytes,
      &candidate.aggregateNoC.transmitMessageCount,
      &candidate.aggregateNoC.receiveMessageCount,
      &candidate.maximumRankNoCTransmitBytes,
      &candidate.maximumRankNoCReceiveBytes,
      &candidate.maximumRankNoCTransmitMessageCount,
      &candidate.maximumRankNoCReceiveMessageCount,
      &candidate.idealizedMinimumPeakLinkByteDemand,
  };
  const analysis::ScheduleCostMetric *baselineSummaries[] = {
      &baseline.aggregateDDRReadBytes,
      &baseline.aggregateDDRWriteBytes,
      &baseline.aggregateSPMMovementBytes,
      &baseline.maximumRankSPMMovementBytes,
      &baseline.aggregateGatherScatterBytes,
      &baseline.maximumRankGatherScatterBytes,
      &baseline.aggregateNoC.staticIssueSiteCount,
      &baseline.aggregateNoC.aggregateTransmitBytes,
      &baseline.aggregateNoC.aggregateReceiveBytes,
      &baseline.aggregateNoC.transmitMessageCount,
      &baseline.aggregateNoC.receiveMessageCount,
      &baseline.maximumRankNoCTransmitBytes,
      &baseline.maximumRankNoCReceiveBytes,
      &baseline.maximumRankNoCTransmitMessageCount,
      &baseline.maximumRankNoCReceiveMessageCount,
      &baseline.idealizedMinimumPeakLinkByteDemand,
  };
  for (auto [candidateMetric, baselineMetric] :
       llvm::zip_equal(candidateSummaries, baselineSummaries))
    if (!compareKnownNoWorse(*candidateMetric, *baselineMetric, summaryGuard))
      return false;

  bool routeImproved = false;
  const analysis::ScheduleCostMetric *candidateRoute[] = {
      &candidate.minimumHopLinkByteDemand,
      &candidate.minimumHopMessageDemand,
      &candidate.modeledNoCRoute.peakDirectedLinkByteDemand,
      &candidate.maximumNoCHopCount,
  };
  const analysis::ScheduleCostMetric *baselineRoute[] = {
      &baseline.minimumHopLinkByteDemand,
      &baseline.minimumHopMessageDemand,
      &baseline.modeledNoCRoute.peakDirectedLinkByteDemand,
      &baseline.maximumNoCHopCount,
  };
  for (auto [candidateMetric, baselineMetric] :
       llvm::zip_equal(candidateRoute, baselineRoute))
    if (!compareKnownNoWorse(*candidateMetric, *baselineMetric, routeImproved))
      return false;
  if (!candidate.directedNoCLinkCount.isKnown() ||
      !baseline.directedNoCLinkCount.isKnown() ||
      candidate.directedNoCLinkCount.value !=
          baseline.directedNoCLinkCount.value)
    return false;
  improved |= routeImproved;
  return improved;
}

struct EligibleCandidate {
  size_t index = 0;
  RankSelectionBasis basis = RankSelectionBasis::EstimatedBenefit;
  analysis::ProgramDurationEstimate duration;
};

static std::optional<EligibleCandidate> evaluatePromotion(
    size_t index, llvm::ArrayRef<RankCandidateCostView> variants,
    const analysis::ProgramDurationEstimate &baselineDuration,
    const analysis::TargetScheduleCostPolicy &policy,
    const analysis::CardInstructionProgramCost &baselineCost) {
  const RankCandidateCostView &candidate = variants[index];
  EligibleCandidate result;
  result.index = index;
  result.duration = analysis::estimateProgramDuration(
      *candidate.resourceCost, policy,
      getScheduleContext(*candidate.resourceCost));

  const analysis::ScheduleCostMetric &candidateUpper =
      result.duration.makespan.upperBoundPicoseconds;
  const analysis::ScheduleCostMetric &baselineLower =
      baselineDuration.makespan.lowerBoundPicoseconds;
  if (candidateUpper.isKnown() && baselineLower.isKnown() &&
      marginClears(candidateUpper.value, baselineLower.value,
                   policy.productionBenefitMarginPermille)) {
    result.basis = RankSelectionBasis::ProvenBenefit;
    return result;
  }

  const analysis::ScheduleCostMetric &candidateNominal =
      result.duration.makespan.nominalPicoseconds;
  const analysis::ScheduleCostMetric &baselineNominal =
      baselineDuration.makespan.nominalPicoseconds;
  if (!candidateNominal.isKnown() || !baselineNominal.isKnown() ||
      !marginClears(candidateNominal.value, baselineNominal.value,
                    policy.productionBenefitMarginPermille)) {
    if (candidateNominal.isKnown() && baselineNominal.isKnown() &&
        candidateNominal.value < baselineNominal.value &&
        hasKnownPrimitiveWorkPromotion(*candidate.resourceCost, baselineCost,
                                       policy)) {
      result.basis = RankSelectionBasis::EstimatedBenefit;
      return result;
    }

    return std::nullopt;
  }
  result.basis = RankSelectionBasis::EstimatedBenefit;
  return result;
}

enum class EligibleOrder : uint8_t { Left, Right, Equivalent, Incomparable };

static EligibleOrder
compareEligible(const EligibleCandidate &left, const EligibleCandidate &right,
                llvm::ArrayRef<RankCandidateCostView> variants) {
  const analysis::ScheduleCostMetric &leftNominal =
      left.duration.makespan.nominalPicoseconds;
  const analysis::ScheduleCostMetric &rightNominal =
      right.duration.makespan.nominalPicoseconds;
  if (leftNominal.isKnown() != rightNominal.isKnown())
    return EligibleOrder::Incomparable;
  if (leftNominal.isKnown() && leftNominal.value != rightNominal.value)
    return leftNominal.value < rightNominal.value ? EligibleOrder::Left
                                                  : EligibleOrder::Right;

  if (sameDurationEstimate(left.duration, right.duration)) {
    auto leftOrdinal = std::tie(variants[left.index].stableSemanticOrdinal,
                                variants[left.index].scheduleActionOrdinal);
    auto rightOrdinal = std::tie(variants[right.index].stableSemanticOrdinal,
                                 variants[right.index].scheduleActionOrdinal);
    if (leftOrdinal == rightOrdinal)
      return EligibleOrder::Equivalent;
    return leftOrdinal < rightOrdinal ? EligibleOrder::Left
                                      : EligibleOrder::Right;
  }

  switch (compareExactSelectionCost(*variants[left.index].resourceCost,
                                    *variants[right.index].resourceCost)) {
  case CostOrder::LeftDominates:
    return EligibleOrder::Left;
  case CostOrder::RightDominates:
    return EligibleOrder::Right;
  case CostOrder::Equivalent:
  case CostOrder::Incomparable:
    return EligibleOrder::Incomparable;
  }
  llvm_unreachable("unhandled exact cost order");
}

} // namespace

mlir::FailureOr<RankCandidateSelection> selectRankCandidateByCost(
    llvm::ArrayRef<RankCandidateCostView> variants,
    RankCandidateSelectionMode selectionMode) {
  if (variants.empty() ||
      (selectionMode != RankCandidateSelectionMode::Production &&
       selectionMode != RankCandidateSelectionMode::ReservedBaseline))
    return mlir::failure();

  std::optional<size_t> baselineIndex;
  llvm::SmallVector<std::pair<int64_t, uint32_t>, 16> ordinals;
  ordinals.reserve(variants.size());
  for (auto [index, variant] : llvm::enumerate(variants)) {
    std::pair<int64_t, uint32_t> ordinal{variant.stableSemanticOrdinal,
                                         variant.scheduleActionOrdinal};
    if (!variant.resourceCost || variant.stableSemanticOrdinal < 0 ||
        llvm::is_contained(ordinals, ordinal))
      return mlir::failure();
    ordinals.push_back(ordinal);
    if (variant.reservedBaseline) {
      if (baselineIndex)
        return mlir::failure();
      baselineIndex = index;
    }
  }
  if (!baselineIndex)
    return mlir::failure();

  RankCandidateSelection plan;
  plan.baselineIndex = *baselineIndex;
  plan.selectedIndex = *baselineIndex;
  const analysis::TargetScheduleCostPolicy policy =
      analysis::getTargetScheduleCostPolicy();
  plan.selectedDuration = analysis::estimateProgramDuration(
      *variants[*baselineIndex].resourceCost, policy,
      getScheduleContext(*variants[*baselineIndex].resourceCost));

  // Pareto membership is computed from the complete set, independent of C3
  // completion order. The reserved baseline remains present as the fallback
  // even when a fully known candidate strictly dominates its work vector.
  for (size_t candidateIndex = 0; candidateIndex < variants.size();
       ++candidateIndex) {
    bool dominated = false;
    if (candidateIndex != *baselineIndex) {
      for (size_t otherIndex = 0; otherIndex < variants.size(); ++otherIndex) {
        if (candidateIndex == otherIndex)
          continue;
        if (compareExactSelectionCost(*variants[otherIndex].resourceCost,
                                      *variants[candidateIndex].resourceCost) ==
            CostOrder::LeftDominates) {
          dominated = true;
          break;
        }
      }
    }
    if (!dominated)
      plan.paretoIndices.push_back(candidateIndex);
  }
  llvm::sort(plan.paretoIndices, [&](size_t left, size_t right) {
    return std::tie(variants[left].stableSemanticOrdinal,
                    variants[left].scheduleActionOrdinal) <
           std::tie(variants[right].stableSemanticOrdinal,
                    variants[right].scheduleActionOrdinal);
  });

  if (selectionMode == RankCandidateSelectionMode::ReservedBaseline)
    return plan;

  std::optional<EligibleCandidate> selected;
  bool selectedIsAmbiguous = false;
  for (size_t candidateIndex : plan.paretoIndices) {
    if (candidateIndex == *baselineIndex)
      continue;
    std::optional<EligibleCandidate> eligible =
        evaluatePromotion(candidateIndex, variants, plan.selectedDuration,
                          policy, *variants[*baselineIndex].resourceCost);
    if (!eligible)
      continue;
    if (!selected) {
      selected = std::move(*eligible);
      selectedIsAmbiguous = false;
      continue;
    }
    switch (compareEligible(*eligible, *selected, variants)) {
    case EligibleOrder::Left:
      selected = std::move(*eligible);
      selectedIsAmbiguous = false;
      break;
    case EligibleOrder::Right:
      break;
    case EligibleOrder::Equivalent:
      break;
    case EligibleOrder::Incomparable:
      selectedIsAmbiguous = true;
      break;
    }
  }

  // Cost-incomparable candidates with the same model score cannot use stable
  // order as a substitute for hardware basis. Fall back atomically.
  if (!selected || selectedIsAmbiguous)
    return plan;
  plan.selectedIndex = selected->index;
  plan.basis = selected->basis;
  plan.selectedDuration = std::move(selected->duration);
  return plan;
}

mlir::FailureOr<RankQualificationSelection>
selectRankQualificationCandidate(
    llvm::ArrayRef<EvaluatedRankCandidate> variants,
    RankCandidateSelectionMode selectionMode) {
  if (!isQualificationSelectionMode(selectionMode) ||
      !hasValidAdmittedIdentities(variants, /*requireBaseline=*/true))
    return mlir::failure();

  RankQualificationSelection plan;
  for (auto [index, variant] : llvm::enumerate(variants))
    if (matchesQualification(variant, selectionMode))
      plan.matchingIndices.push_back(index);
  if (plan.matchingIndices.empty())
    return mlir::failure();
  llvm::sort(plan.matchingIndices, [&](size_t left, size_t right) {
    return std::tie(variants[left].stableSemanticOrdinal,
                    variants[left].scheduleActionOrdinal) <
           std::tie(variants[right].stableSemanticOrdinal,
                    variants[right].scheduleActionOrdinal);
  });
  plan.selectedIndex = plan.matchingIndices.front();
  return plan;
}

mlir::LogicalResult reduceEvaluatedRankCandidates(
    std::vector<EvaluatedRankCandidate> &variants,
    RankCandidateSelectionMode selectionMode,
    const analysis::CardInstructionProgramCost *externalBaselineCost) {
  if (variants.empty())
    return mlir::success();
  if (externalBaselineCost &&
      selectionMode != RankCandidateSelectionMode::Production)
    return mlir::failure();
  if (isQualificationSelectionMode(selectionMode))
    return mlir::success(
        hasValidAdmittedIdentities(variants, /*requireBaseline=*/false));
  if (selectionMode != RankCandidateSelectionMode::Production &&
      selectionMode != RankCandidateSelectionMode::ReservedBaseline)
    return mlir::failure();

  std::optional<size_t> baselineIndex;
  llvm::SmallVector<std::pair<int64_t, uint32_t>, 16> ordinals;
  ordinals.reserve(variants.size());
  for (auto [variantIndex, variant] : llvm::enumerate(variants)) {
    std::pair<int64_t, uint32_t> ordinal{variant.stableSemanticOrdinal,
                                         variant.scheduleActionOrdinal};
    if (variant.stableSemanticOrdinal < 0 ||
        llvm::is_contained(ordinals, ordinal))
      return mlir::failure();
    ordinals.push_back(ordinal);
    if (variant.reservedBaseline) {
      if (baselineIndex)
        return mlir::failure();
      baselineIndex = variantIndex;
    }
  }

  llvm::SmallVector<bool, 16> permanentlyIneligible(variants.size(), false);
  const analysis::CardInstructionProgramCost *promotionBaselineCost =
      baselineIndex ? &variants[*baselineIndex].variant.resourceCost
                    : externalBaselineCost;
  if (baselineIndex || promotionBaselineCost) {
    if (selectionMode == RankCandidateSelectionMode::ReservedBaseline) {
      for (size_t index = 0; index < variants.size(); ++index)
        permanentlyIneligible[index] = index != *baselineIndex;
    } else {
      llvm::SmallVector<RankCandidateCostView, 16> views;
      views.reserve(variants.size());
      for (const EvaluatedRankCandidate &variant : variants)
        views.push_back(
            {variant.stableSemanticOrdinal, variant.reservedBaseline,
             &variant.variant.resourceCost, variant.scheduleActionOrdinal});
      const analysis::TargetScheduleCostPolicy policy =
          analysis::getTargetScheduleCostPolicy();
      const analysis::ProgramDurationEstimate baselineDuration =
          analysis::estimateProgramDuration(
              *promotionBaselineCost, policy,
              getScheduleContext(*promotionBaselineCost));
      for (size_t index = 0; index < variants.size(); ++index) {
        if (baselineIndex && index == *baselineIndex)
          continue;
        permanentlyIneligible[index] =
            !evaluatePromotion(index, views, baselineDuration, policy,
                               *promotionBaselineCost)
                 .has_value();
      }
    }
  }

  llvm::SmallVector<size_t, 16> retained;
  for (size_t candidateIndex = 0; candidateIndex < variants.size();
       ++candidateIndex) {
    if (baselineIndex && candidateIndex == *baselineIndex) {
      retained.push_back(candidateIndex);
      continue;
    }
    if (permanentlyIneligible[candidateIndex])
      continue;
    bool remove = false;
    const auto candidateOrdinal =
        std::tie(variants[candidateIndex].stableSemanticOrdinal,
                 variants[candidateIndex].scheduleActionOrdinal);
    for (size_t otherIndex = 0; otherIndex < variants.size(); ++otherIndex) {
      if (candidateIndex == otherIndex)
        continue;
      CostOrder order = compareExactSelectionCost(
          variants[otherIndex].variant.resourceCost,
          variants[candidateIndex].variant.resourceCost);
      if (order == CostOrder::LeftDominates) {
        remove = true;
        break;
      }
      if (order != CostOrder::Equivalent)
        continue;
      if ((baselineIndex && otherIndex == *baselineIndex) ||
          std::tie(variants[otherIndex].stableSemanticOrdinal,
                   variants[otherIndex].scheduleActionOrdinal) <
              candidateOrdinal) {
        remove = true;
        break;
      }
    }
    if (!remove)
      retained.push_back(candidateIndex);
  }

  llvm::sort(retained, [&](size_t left, size_t right) {
    return std::tie(variants[left].stableSemanticOrdinal,
                    variants[left].scheduleActionOrdinal) <
           std::tie(variants[right].stableSemanticOrdinal,
                    variants[right].scheduleActionOrdinal);
  });
  std::vector<EvaluatedRankCandidate> reduced;
  reduced.reserve(retained.size());
  for (size_t index : retained)
    reduced.push_back(std::move(variants[index]));
  variants = std::move(reduced);
  return mlir::success();
}

mlir::FailureOr<EvaluatedRankCandidate> selectEvaluatedRankCandidate(
    std::vector<EvaluatedRankCandidate> variants,
    RankCandidateSelectionMode selectionMode, llvm::raw_ostream &diagnostics,
    RankCandidateSearchStatistics *statistics) {
  if (isQualificationSelectionMode(selectionMode)) {
    mlir::FailureOr<RankQualificationSelection> qualification =
        selectRankQualificationCandidate(variants, selectionMode);
    if (mlir::failed(qualification)) {
      diagnostics << "wafer-compile: no exact-admitted coordinated executable "
                     "matches qualification="
                  << stringifyQualificationSelectionMode(selectionMode) << '\n';
      return mlir::failure();
    }
    const EvaluatedRankCandidate &winner =
        variants[qualification->selectedIndex];
    diagnostics
        << "wafer-compile: coordinated whole-rank qualification"
        << " mode=" << stringifyQualificationSelectionMode(selectionMode)
        << " admitted_executables=" << variants.size()
        << " qualification_matched=" << qualification->matchingIndices.size()
        << " winner_ordinal=" << winner.stableSemanticOrdinal
        << " schedule_action_ordinal=" << winner.scheduleActionOrdinal
        << " reserved_baseline=" << (winner.reservedBaseline ? "true" : "false")
        << " implementation_alternative="
        << (winner.implementationAlternativeOrigin ? "true" : "false") << '\n';
    return std::move(variants[qualification->selectedIndex]);
  }

  llvm::SmallVector<RankCandidateCostView, 16> views;
  views.reserve(variants.size());
  for (const EvaluatedRankCandidate &variant : variants)
    views.push_back({variant.stableSemanticOrdinal, variant.reservedBaseline,
                     &variant.variant.resourceCost,
                     variant.scheduleActionOrdinal});
  mlir::FailureOr<RankCandidateSelection> plan =
      selectRankCandidateByCost(views, selectionMode);
  if (mlir::failed(plan))
    return mlir::failure();
  if (statistics)
    statistics->paretoRetainedVariants += plan->paretoIndices.size();

  const EvaluatedRankCandidate &winner = variants[plan->selectedIndex];
  diagnostics << "wafer-compile: coordinated whole-rank selection"
              << " admitted_executables=" << variants.size()
              << " pareto_retained=" << plan->paretoIndices.size()
              << " winner_ordinal=" << winner.stableSemanticOrdinal
              << " schedule_action_ordinal=" << winner.scheduleActionOrdinal
              << " reserved_baseline="
              << (winner.reservedBaseline ? "true" : "false")
              << " implementation_alternative="
              << (winner.implementationAlternativeOrigin ? "true" : "false")
              << " basis=" << stringifyRankSelectionBasis(plan->basis)
              << " nominal_ps=";
  const analysis::ScheduleCostMetric &nominal =
      plan->selectedDuration.makespan.nominalPicoseconds;
  if (nominal.isKnown())
    diagnostics << nominal.value;
  else
    diagnostics << "unknown("
                << analysis::stringifyScheduleCostReason(nominal.reason) << ')';
  diagnostics << '\n';
  return std::move(variants[plan->selectedIndex]);
}

llvm::StringRef
stringifyRankSelectionBasis(RankSelectionBasis basis) {
  switch (basis) {
  case RankSelectionBasis::ReservedBaseline:
    return "reserved-baseline";
  case RankSelectionBasis::EstimatedBenefit:
    return "estimated-benefit";
  case RankSelectionBasis::ProvenBenefit:
    return "proven-benefit";
  }
  llvm_unreachable("unknown coordinated hardware selection basis");
}

} // namespace wafer::compiler::detail
