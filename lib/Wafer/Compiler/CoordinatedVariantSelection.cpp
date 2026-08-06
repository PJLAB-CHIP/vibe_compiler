//===- CoordinatedVariantSelection.cpp - Final all-rank choice ---------===//

#include "CoordinatedVariantSelection.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

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
    // Equal non-Known dispositions carry the same uncertainty. Different
    // knowledge/reason pairs are not ordered and never become implicit zero.
    if (left.knowledge != right.knowledge || left.reason != right.reason) {
      incomparable = true;
      return;
    }
    if (!left.isKnown())
      return;
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
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right) {
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

static bool
metricIsNoWorse(const analysis::ScheduleCostMetric &candidate,
                const analysis::ScheduleCostMetric &baseline,
                MetricPreference preference = MetricPreference::Lower) {
  if (candidate.knowledge != baseline.knowledge ||
      candidate.reason != baseline.reason)
    return false;
  if (!candidate.isKnown() || candidate.value == baseline.value)
    return true;
  return preference == MetricPreference::Lower
             ? candidate.value < baseline.value
             : candidate.value > baseline.value;
}

/// The duration model deliberately has no fabricated SPM/local-movement rate,
/// dependency-depth latency, or buffer-descriptor latency. Those dimensions
/// may improve, but a regression cannot be hidden behind a DDR/NoC point
/// estimate until calibration exists.
static bool hasNoUncalibratedRegression(
    const analysis::WholeCardInstructionProgramCost &candidate,
    const analysis::WholeCardInstructionProgramCost &baseline) {
  if (!metricIsNoWorse(candidate.aggregateSPMMovementBytes,
                       baseline.aggregateSPMMovementBytes) ||
      !metricIsNoWorse(candidate.aggregateGatherScatterBytes,
                       baseline.aggregateGatherScatterBytes) ||
      !metricIsNoWorse(candidate.maximumRankSPMMovementBytes,
                       baseline.maximumRankSPMMovementBytes) ||
      !metricIsNoWorse(candidate.maximumRankGatherScatterBytes,
                       baseline.maximumRankGatherScatterBytes) ||
      !metricIsNoWorse(candidate.maximumRankDataDependencyDepth,
                       baseline.maximumRankDataDependencyDepth) ||
      !metricIsNoWorse(candidate.aggregateReadyOrderPriorityInversions,
                       baseline.aggregateReadyOrderPriorityInversions) ||
      !metricIsNoWorse(candidate.aggregateEventCount,
                       baseline.aggregateEventCount) ||
      !metricIsNoWorse(candidate.maximumRankDDRHighWaterBytes,
                       baseline.maximumRankDDRHighWaterBytes) ||
      !metricIsNoWorse(candidate.summedRankDDRHighWaterBytes,
                       baseline.summedRankDDRHighWaterBytes) ||
      !metricIsNoWorse(candidate.aggregateCompilerOwnedSPMBufferCount,
                       baseline.aggregateCompilerOwnedSPMBufferCount) ||
      !metricIsNoWorse(candidate.aggregateCompilerOwnedDDRBufferCount,
                       baseline.aggregateCompilerOwnedDDRBufferCount) ||
      !metricIsNoWorse(candidate.maximumRankCompilerOwnedSPMBufferCount,
                       baseline.maximumRankCompilerOwnedSPMBufferCount) ||
      !metricIsNoWorse(candidate.maximumRankCompilerOwnedDDRBufferCount,
                       baseline.maximumRankCompilerOwnedDDRBufferCount))
    return false;
  return true;
}

static analysis::StaticCrossResourceSchedule
getScheduleContext(const analysis::WholeCardInstructionProgramCost &cost) {
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
sameDurationEstimate(const analysis::WholeCardResourceDurationEstimate &left,
                     const analysis::WholeCardResourceDurationEstimate &right) {
  return sameInterval(left.ddr, right.ddr) &&
         sameInterval(left.compute, right.compute) &&
         sameInterval(left.noc, right.noc) &&
         sameInterval(left.spm, right.spm) &&
         sameInterval(left.control, right.control) &&
         sameInterval(left.makespan, right.makespan);
}

struct EligibleCandidate {
  size_t index = 0;
  CoordinatedHardwareSelectionEvidence evidence =
      CoordinatedHardwareSelectionEvidence::EstimatedBenefit;
  analysis::WholeCardResourceDurationEstimate duration;
};

static std::optional<EligibleCandidate> evaluatePromotion(
    size_t index, llvm::ArrayRef<CoordinatedVariantCostView> variants,
    const analysis::WholeCardResourceDurationEstimate &baselineDuration,
    const analysis::TargetScheduleCostPolicy &policy, size_t baselineIndex) {
  const CoordinatedVariantCostView &candidate = variants[index];
  const CoordinatedVariantCostView &baseline = variants[baselineIndex];
  if (!hasNoUncalibratedRegression(*candidate.resourceCost,
                                   *baseline.resourceCost))
    return std::nullopt;

  EligibleCandidate result;
  result.index = index;
  result.duration = analysis::estimateWholeCardResourceDuration(
      *candidate.resourceCost, policy,
      getScheduleContext(*candidate.resourceCost));

  const analysis::ScheduleCostMetric &candidateUpper =
      result.duration.makespan.upperBoundPicoseconds;
  const analysis::ScheduleCostMetric &baselineLower =
      baselineDuration.makespan.lowerBoundPicoseconds;
  if (candidateUpper.isKnown() && baselineLower.isKnown() &&
      marginClears(candidateUpper.value, baselineLower.value,
                   policy.productionBenefitMarginPermille)) {
    result.evidence = CoordinatedHardwareSelectionEvidence::ProvenBenefit;
    return result;
  }

  const analysis::ScheduleCostMetric &candidateNominal =
      result.duration.makespan.nominalPicoseconds;
  const analysis::ScheduleCostMetric &baselineNominal =
      baselineDuration.makespan.nominalPicoseconds;
  if (!candidateNominal.isKnown() || !baselineNominal.isKnown() ||
      !marginClears(candidateNominal.value, baselineNominal.value,
                    policy.productionBenefitMarginPermille))
    return std::nullopt;
  result.evidence = CoordinatedHardwareSelectionEvidence::EstimatedBenefit;
  return result;
}

enum class EligibleOrder : uint8_t { Left, Right, Equivalent, Incomparable };

static EligibleOrder
compareEligible(const EligibleCandidate &left, const EligibleCandidate &right,
                llvm::ArrayRef<CoordinatedVariantCostView> variants) {
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
    auto leftOrdinal =
        std::tie(variants[left.index].stableSemanticOrdinal,
                 variants[left.index].terminalActionOrdinal);
    auto rightOrdinal =
        std::tie(variants[right.index].stableSemanticOrdinal,
                 variants[right.index].terminalActionOrdinal);
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

mlir::FailureOr<CoordinatedVariantSelectionPlan>
planCoordinatedVariantSelection(
    llvm::ArrayRef<CoordinatedVariantCostView> variants,
    WholeVariantSelectionMode selectionMode) {
  if (variants.empty() ||
      (selectionMode != WholeVariantSelectionMode::Production &&
       selectionMode != WholeVariantSelectionMode::ReservedBaseline))
    return mlir::failure();

  std::optional<size_t> baselineIndex;
  llvm::SmallVector<std::pair<int64_t, uint32_t>, 16> ordinals;
  ordinals.reserve(variants.size());
  for (auto [index, variant] : llvm::enumerate(variants)) {
    std::pair<int64_t, uint32_t> ordinal{variant.stableSemanticOrdinal,
                                         variant.terminalActionOrdinal};
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

  CoordinatedVariantSelectionPlan plan;
  plan.baselineIndex = *baselineIndex;
  plan.selectedIndex = *baselineIndex;
  const analysis::TargetScheduleCostPolicy policy =
      analysis::getTargetScheduleCostPolicy();
  plan.selectedDuration = analysis::estimateWholeCardResourceDuration(
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
                    variants[left].terminalActionOrdinal) <
           std::tie(variants[right].stableSemanticOrdinal,
                    variants[right].terminalActionOrdinal);
  });

  if (selectionMode == WholeVariantSelectionMode::ReservedBaseline)
    return plan;

  std::optional<EligibleCandidate> selected;
  bool selectedIsAmbiguous = false;
  for (size_t candidateIndex : plan.paretoIndices) {
    if (candidateIndex == *baselineIndex)
      continue;
    std::optional<EligibleCandidate> eligible =
        evaluatePromotion(candidateIndex, variants, plan.selectedDuration,
                          policy, *baselineIndex);
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
  // order as a substitute for hardware evidence. Fall back atomically.
  if (!selected || selectedIsAmbiguous)
    return plan;
  plan.selectedIndex = selected->index;
  plan.evidence = selected->evidence;
  plan.selectedDuration = std::move(selected->duration);
  return plan;
}

mlir::FailureOr<FullyGatedCoordinatedVariant>
selectFullyGatedCoordinatedVariant(
    std::vector<FullyGatedCoordinatedVariant> variants,
    WholeVariantSelectionMode selectionMode, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionStatistics *statistics) {
  llvm::SmallVector<CoordinatedVariantCostView, 16> views;
  views.reserve(variants.size());
  for (const FullyGatedCoordinatedVariant &variant : variants)
    views.push_back({variant.stableSemanticOrdinal, variant.reservedBaseline,
                     &variant.variant.resourceCost,
                     variant.terminalActionOrdinal});
  mlir::FailureOr<CoordinatedVariantSelectionPlan> plan =
      planCoordinatedVariantSelection(views, selectionMode);
  if (mlir::failed(plan))
    return mlir::failure();
  if (statistics)
    statistics->paretoRetainedVariants += plan->paretoIndices.size();

  const FullyGatedCoordinatedVariant &winner = variants[plan->selectedIndex];
  diagnostics << "wafer-compile: coordinated whole-rank selection"
              << " fully_gated=" << variants.size()
              << " pareto_retained=" << plan->paretoIndices.size()
              << " winner_ordinal=" << winner.stableSemanticOrdinal
              << " terminal_action_ordinal="
              << winner.terminalActionOrdinal
              << " reserved_baseline=" << winner.reservedBaseline
              << " evidence="
              << stringifyCoordinatedHardwareSelectionEvidence(plan->evidence)
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

llvm::StringRef stringifyCoordinatedHardwareSelectionEvidence(
    CoordinatedHardwareSelectionEvidence evidence) {
  switch (evidence) {
  case CoordinatedHardwareSelectionEvidence::ReservedBaseline:
    return "reserved-baseline";
  case CoordinatedHardwareSelectionEvidence::EstimatedBenefit:
    return "estimated-benefit";
  case CoordinatedHardwareSelectionEvidence::ProvenBenefit:
    return "proven-benefit";
  }
  llvm_unreachable("unknown coordinated hardware selection evidence");
}

} // namespace wafer::compiler::detail
