//===- NoCProfitabilityAnalysis.cpp - Whole-card NoC tradeoff -----------===//

#include "Wafer/Analysis/NoCProfitabilityAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <optional>

namespace wafer::analysis {
namespace {

constexpr uint64_t kPicosecondsPerSecond = 1'000'000'000'000ULL;

static StaticDurationAssumptionMask
assumptions(std::initializer_list<StaticDurationAssumption> values) {
  StaticDurationAssumptionMask result = 0;
  for (StaticDurationAssumption value : values)
    result |= staticDurationAssumptionMask(value);
  return result;
}

static ScheduleCostMetric unknown(ScheduleCostReason reason) {
  return {0, ScheduleCostKnowledge::Unknown, reason};
}

static ScheduleCostMetric overflow() {
  return {0, ScheduleCostKnowledge::Overflow,
          ScheduleCostReason::ArithmeticOverflow};
}

static unsigned knowledgeSeverity(ScheduleCostKnowledge knowledge) {
  switch (knowledge) {
  case ScheduleCostKnowledge::Known:
    return 0;
  case ScheduleCostKnowledge::Unknown:
    return 1;
  case ScheduleCostKnowledge::Unsupported:
    return 2;
  case ScheduleCostKnowledge::Overflow:
    return 3;
  }
  llvm_unreachable("unhandled schedule cost knowledge");
}

static ScheduleCostMetric mergeNonKnown(ScheduleCostMetric lhs,
                                        ScheduleCostMetric rhs) {
  if (lhs.isKnown())
    return rhs;
  if (rhs.isKnown())
    return lhs;
  if (knowledgeSeverity(rhs.knowledge) > knowledgeSeverity(lhs.knowledge))
    return rhs;
  return lhs;
}

static ScheduleCostMetric addMetric(ScheduleCostMetric lhs,
                                    ScheduleCostMetric rhs) {
  if (!lhs.isKnown() || !rhs.isKnown())
    return mergeNonKnown(lhs, rhs);
  if (rhs.value > std::numeric_limits<uint64_t>::max() - lhs.value)
    return overflow();
  lhs.value += rhs.value;
  return lhs;
}

static ScheduleCostMetric maxMetric(ScheduleCostMetric lhs,
                                    ScheduleCostMetric rhs) {
  if (!lhs.isKnown() || !rhs.isKnown())
    return mergeNonKnown(lhs, rhs);
  lhs.value = std::max(lhs.value, rhs.value);
  return lhs;
}

static ScheduleCostMetric
sumMetrics(llvm::ArrayRef<ScheduleCostMetric> metrics) {
  ScheduleCostMetric result;
  for (ScheduleCostMetric metric : metrics)
    result = addMetric(result, metric);
  return result;
}

static ScheduleCostMetric
maxMetrics(llvm::ArrayRef<ScheduleCostMetric> metrics) {
  ScheduleCostMetric result;
  for (ScheduleCostMetric metric : metrics)
    result = maxMetric(result, metric);
  return result;
}

static ScheduleCostMetric scaleMetric(ScheduleCostMetric metric,
                                      uint64_t multiplier) {
  if (!metric.isKnown())
    return metric;
  unsigned __int128 product =
      static_cast<unsigned __int128>(metric.value) * multiplier;
  if (product > std::numeric_limits<uint64_t>::max())
    return overflow();
  metric.value = static_cast<uint64_t>(product);
  return metric;
}

static ScheduleCostMetric timeForWork(ScheduleCostMetric work,
                                      uint64_t unitsPerSecond) {
  if (!work.isKnown())
    return work;
  if (work.value == 0)
    return {};
  if (unitsPerSecond == 0)
    return unknown(ScheduleCostReason::MissingPerformanceCalibration);
  unsigned __int128 numerator =
      static_cast<unsigned __int128>(work.value) * kPicosecondsPerSecond;
  unsigned __int128 duration =
      (numerator + unitsPerSecond - 1) / unitsPerSecond;
  if (duration > std::numeric_limits<uint64_t>::max())
    return overflow();
  return {static_cast<uint64_t>(duration)};
}

static ScheduleCostMetric timeForWork(ScheduleCostMetric work,
                                      std::optional<uint64_t> unitsPerSecond) {
  if (!work.isKnown())
    return work;
  if (work.value == 0)
    return {};
  if (!unitsPerSecond)
    return unknown(ScheduleCostReason::MissingPerformanceCalibration);
  return timeForWork(work, *unitsPerSecond);
}

static std::optional<uint64_t> validatedSustainedRate(
    std::optional<uint64_t> rate,
    std::optional<uint64_t> documentedMaximum = std::nullopt) {
  if (!rate || *rate == 0 || (documentedMaximum && *rate > *documentedMaximum))
    return std::nullopt;
  return rate;
}

static ScheduleCostMetric
timeForOccurrences(ScheduleCostMetric occurrences,
                   std::optional<uint64_t> picosecondsPerOccurrence) {
  if (!occurrences.isKnown())
    return occurrences;
  if (occurrences.value == 0)
    return {};
  if (!picosecondsPerOccurrence)
    return unknown(ScheduleCostReason::MissingPerformanceCalibration);
  return scaleMetric(occurrences, *picosecondsPerOccurrence);
}

static ScheduleCostMetric requireSupportedZero(ScheduleCostMetric metric) {
  if (!metric.isKnown())
    return metric;
  if (metric.value != 0)
    return unknown(ScheduleCostReason::MissingPerformanceCalibration);
  return {};
}

static StaticDurationInterval
estimateDDR(const WholeCardInstructionProgramCost &cost,
            const TargetScheduleCostPolicy &policy) {
  ScheduleCostMetric bytes =
      addMetric(cost.aggregateDDRReadBytes, cost.aggregateDDRWriteBytes);
  return {
      timeForWork(bytes, policy.cardDDRBytesPerSecond),
      timeForWork(bytes, policy.cardDDRNominalBytesPerSecond),
      timeForWork(bytes, validatedSustainedRate(
                             policy.cardDDRSustainedBytesPerSecondLowerBound,
                             policy.cardDDRBytesPerSecond)),
      assumptions({StaticDurationAssumption::DDROperatingPoint}),
  };
}

static ScheduleCostMetric
estimateRankCompute(const InstructionProgramCost &rank,
                    const TargetScheduleCostPolicy &policy,
                    bool conservativeUpperBound) {
  ScheduleCostMetric unsupported =
      maxMetric(requireSupportedZero(rank.compute.npuOtherLogicalOps),
                requireSupportedZero(rank.compute.vectorOtherLogicalOps));
  if (!unsupported.isKnown())
    return unsupported;

  ScheduleCostMetric npu =
      conservativeUpperBound
          ? timeForWork(
                rank.compute.npuF16Bf16LogicalOps,
                validatedSustainedRate(
                    policy.f16Bf16NPULogicalOpsPerSecondPerTileLowerBound,
                    policy.f16Bf16NPULogicalOpsPerSecondPerTile))
          : timeForWork(rank.compute.npuF16Bf16LogicalOps,
                        policy.f16Bf16NPULogicalOpsPerSecondPerTile);
  ScheduleCostMetric vectorF16 =
      conservativeUpperBound
          ? timeForWork(
                rank.compute.vectorF16Bf16LogicalOps,
                validatedSustainedRate(
                    policy.f16Bf16VectorLogicalOpsPerSecondPerTileLowerBound,
                    policy.f16Bf16VectorLogicalOpsPerSecondPerTile))
          : timeForWork(rank.compute.vectorF16Bf16LogicalOps,
                        policy.f16Bf16VectorLogicalOpsPerSecondPerTile);
  ScheduleCostMetric vectorF32 =
      conservativeUpperBound
          ? timeForWork(
                rank.compute.vectorF32LogicalOps,
                validatedSustainedRate(
                    policy.f32VectorLogicalOpsPerSecondPerTileLowerBound,
                    policy.f32VectorLogicalOpsPerSecondPerTile))
          : timeForWork(rank.compute.vectorF32LogicalOps,
                        policy.f32VectorLogicalOpsPerSecondPerTile);

  // A lower bound allows independent engines to overlap. A conservative upper
  // bound serializes their work because no exact group duration is established.
  if (conservativeUpperBound)
    return sumMetrics({npu, vectorF16, vectorF32});
  return maxMetrics({npu, vectorF16, vectorF32});
}

static StaticDurationInterval
estimateCompute(const WholeCardInstructionProgramCost &cost,
                const TargetScheduleCostPolicy &policy) {
  if (cost.rankCosts.empty()) {
    InstructionProgramCost aggregateAsOneRank;
    aggregateAsOneRank.compute = cost.aggregateCompute;
    ScheduleCostMetric nominal = sumMetrics(
        {timeForWork(aggregateAsOneRank.compute.npuF16Bf16LogicalOps,
                     policy.f16Bf16NPULogicalOpsPerSecondPerTile),
         timeForWork(aggregateAsOneRank.compute.vectorF16Bf16LogicalOps,
                     policy.f16Bf16VectorLogicalOpsPerSecondPerTile),
         timeForWork(aggregateAsOneRank.compute.vectorF32LogicalOps,
                     policy.f32VectorLogicalOpsPerSecondPerTile),
         requireSupportedZero(aggregateAsOneRank.compute.npuOtherLogicalOps),
         requireSupportedZero(
             aggregateAsOneRank.compute.vectorOtherLogicalOps)});
    ScheduleCostMetric upper = estimateRankCompute(
        aggregateAsOneRank, policy, /*conservativeUpperBound=*/true);
    // An aggregate-only synthetic caller has no rank distribution, so zero is
    // the only generally valid compute lower bound. Treating all aggregate
    // work as one rank is an explicit conservative point/upper fallback.
    return {{},
            nominal,
            upper,
            assumptions({StaticDurationAssumption::ComputePeakReference})};
  }
  ScheduleCostMetric lower;
  ScheduleCostMetric nominal;
  ScheduleCostMetric upper;
  for (const InstructionProgramCost &rank : cost.rankCosts) {
    ScheduleCostMetric rankLower =
        estimateRankCompute(rank, policy, /*conservativeUpperBound=*/false);
    ScheduleCostMetric rankNominal =
        sumMetrics({timeForWork(rank.compute.npuF16Bf16LogicalOps,
                                policy.f16Bf16NPULogicalOpsPerSecondPerTile),
                    timeForWork(rank.compute.vectorF16Bf16LogicalOps,
                                policy.f16Bf16VectorLogicalOpsPerSecondPerTile),
                    timeForWork(rank.compute.vectorF32LogicalOps,
                                policy.f32VectorLogicalOpsPerSecondPerTile),
                    requireSupportedZero(rank.compute.npuOtherLogicalOps),
                    requireSupportedZero(rank.compute.vectorOtherLogicalOps)});
    ScheduleCostMetric rankUpper =
        estimateRankCompute(rank, policy, /*conservativeUpperBound=*/true);
    lower = maxMetric(lower, rankLower);
    nominal = maxMetric(nominal, rankNominal);
    // A proof-oriented whole-card upper bound cannot assume that rank-local
    // compute phases overlap: NoC/data dependencies may serialize them.
    upper = addMetric(upper, rankUpper);
  }
  return {lower, nominal, upper,
          assumptions({StaticDurationAssumption::ComputePeakReference})};
}

static StaticDurationInterval
estimateNoC(const WholeCardInstructionProgramCost &cost,
            const TargetScheduleCostPolicy &policy) {
  const ScheduleCostMetric workFacts[] = {
      cost.aggregateNoC.staticIssueSiteCount,
      cost.aggregateNoC.aggregateTransmitBytes,
      cost.aggregateNoC.aggregateReceiveBytes,
      cost.aggregateNoC.transmitMessageCount,
      cost.aggregateNoC.receiveMessageCount,
      cost.aggregateNoC.waitOperationCount,
      cost.aggregateNoC.waitedEventCount,
      cost.minimumHopLinkByteDemand,
      cost.minimumHopMessageDemand,
      cost.idealizedMinimumPeakLinkByteDemand,
      cost.modeledNoCRoute.peakDirectedLinkByteDemand,
      cost.maximumNoCHopCount,
      cost.maximumRankNoCTransmitBytes,
      cost.maximumRankNoCReceiveBytes,
      cost.maximumRankNoCTransmitMessageCount,
      cost.maximumRankNoCReceiveMessageCount,
  };
  bool allKnown = true;
  bool allZero = true;
  ScheduleCostMetric mostSevereNonKnown;
  for (ScheduleCostMetric fact : workFacts) {
    if (!fact.isKnown())
      mostSevereNonKnown = mergeNonKnown(mostSevereNonKnown, fact);
    allKnown &= fact.isKnown();
    allZero &= fact.isKnown() && fact.value == 0;
  }
  if (allKnown && allZero)
    return {};
  if (!allKnown)
    return {{}, mostSevereNonKnown, mostSevereNonKnown};

  // The route-independent ideal balancing floor remains a true lower bound at
  // the documented maximum link rate. The nominal model is deliberately more
  // concrete: a deterministic shortest path derived from typed topology
  // exposes hot directed links, while each endpoint serializes its own
  // message startup and payload stream. The physical route and startup value
  // are point-model assumptions, so this result is Estimated rather than a
  // proof.
  ScheduleCostMetric lower =
      timeForWork(cost.idealizedMinimumPeakLinkByteDemand,
                  policy.directionalNoCBytesPerSecond);
  ScheduleCostMetric link =
      timeForWork(cost.modeledNoCRoute.peakDirectedLinkByteDemand,
                  policy.directionalNoCBytesPerSecond);
  ScheduleCostMetric transmitEndpoint = addMetric(
      timeForWork(cost.maximumRankNoCTransmitBytes,
                  policy.dteEndpointBytesPerSecondEstimate),
      timeForOccurrences(cost.maximumRankNoCTransmitMessageCount,
                         policy.dteMessageStartupPicosecondsEstimate));
  ScheduleCostMetric receiveEndpoint = addMetric(
      timeForWork(cost.maximumRankNoCReceiveBytes,
                  policy.dteEndpointBytesPerSecondEstimate),
      timeForOccurrences(cost.maximumRankNoCReceiveMessageCount,
                         policy.dteMessageStartupPicosecondsEstimate));
  ScheduleCostMetric routeFill = timeForOccurrences(
      cost.maximumNoCHopCount, policy.noCHopPicosecondsEstimate);
  ScheduleCostMetric nominal = addMetric(
      maxMetrics({link, transmitEndpoint, receiveEndpoint}), routeFill);

  std::optional<uint64_t> directionalRate = validatedSustainedRate(
      policy.directionalNoCSustainedBytesPerSecondLowerBound,
      policy.directionalNoCBytesPerSecond);
  std::optional<uint64_t> endpointRate =
      validatedSustainedRate(policy.dteEndpointBytesPerSecondLowerBound);
  if (!directionalRate || !endpointRate ||
      !policy.dteMessageStartupPicosecondsUpperBound ||
      !policy.noCHopPicosecondsUpperBound ||
      !policy.noCRouteDilationUpperBound ||
      *policy.noCRouteDilationUpperBound == 0)
    return {lower, nominal,
            unknown(ScheduleCostReason::MissingPerformanceCalibration),
            assumptions({StaticDurationAssumption::ModeledShortestPathRoute,
                         StaticDurationAssumption::DTEEndpointRatePrior,
                         StaticDurationAssumption::DTEMessageStartupPrior,
                         StaticDurationAssumption::NoCHopPrior})};

  ScheduleCostMetric routedBytes = scaleMetric(
      cost.minimumHopLinkByteDemand, *policy.noCRouteDilationUpperBound);
  ScheduleCostMetric routed = timeForWork(routedBytes, directionalRate);
  // The nominal reference uses endpoint maxima to expose normal parallel
  // pressure. The upper bound deliberately serializes every endpoint byte and
  // message because cross-rank forwarding dependencies may prevent that
  // parallelism.
  ScheduleCostMetric endpointTransmit =
      timeForWork(cost.aggregateNoC.aggregateTransmitBytes, endpointRate);
  ScheduleCostMetric endpointReceive =
      timeForWork(cost.aggregateNoC.aggregateReceiveBytes, endpointRate);
  ScheduleCostMetric endpointMessages =
      addMetric(cost.aggregateNoC.transmitMessageCount,
                cost.aggregateNoC.receiveMessageCount);
  ScheduleCostMetric startup = scaleMetric(
      endpointMessages, *policy.dteMessageStartupPicosecondsUpperBound);
  ScheduleCostMetric dilatedMessageHops = scaleMetric(
      cost.minimumHopMessageDemand, *policy.noCRouteDilationUpperBound);
  ScheduleCostMetric conservativeRouteFill = timeForOccurrences(
      dilatedMessageHops, policy.noCHopPicosecondsUpperBound);
  ScheduleCostMetric upper =
      sumMetrics({routed, endpointTransmit, endpointReceive, startup,
                  conservativeRouteFill});
  return {lower, nominal, upper,
          assumptions({StaticDurationAssumption::ModeledShortestPathRoute,
                       StaticDurationAssumption::DTEEndpointRatePrior,
                       StaticDurationAssumption::DTEMessageStartupPrior,
                       StaticDurationAssumption::NoCHopPrior})};
}

static StaticDurationInterval
estimateSPM(const WholeCardInstructionProgramCost &cost,
            const TargetScheduleCostPolicy &policy) {
  ScheduleCostMetric maximumRankMovement = cost.maximumRankSPMMovementBytes;
  if (cost.rankCosts.empty() && maximumRankMovement.isKnown() &&
      maximumRankMovement.value == 0)
    maximumRankMovement = cost.aggregateSPMMovementBytes;
  ScheduleCostMetric nominal = maximumRankMovement;
  if (nominal.isKnown() && nominal.value != 0)
    nominal = unknown(ScheduleCostReason::MissingPerformanceCalibration);
  return {{},
          nominal,
          timeForWork(cost.aggregateSPMMovementBytes,
                      validatedSustainedRate(
                          policy.spmBytesPerSecondPerTileLowerBound)),
          /*nominalAssumptions=*/0};
}

static StaticDurationInterval
estimateControl(const WholeCardInstructionProgramCost &cost,
                const TargetScheduleCostPolicy &policy) {
  ScheduleCostMetric nominal;
  for (const InstructionProgramCost &rank : cost.rankCosts) {
    ScheduleCostMetric rankNominal = sumMetrics(
        {timeForOccurrences(rank.instructionCount,
                            policy.instructionFixedPicosecondsEstimate),
         timeForOccurrences(rank.noc.waitedEventCount,
                            policy.dteWaitedEventPicosecondsEstimate),
         timeForOccurrences(rank.nccParticipantWaitCount,
                            policy.nccParticipantWaitPicosecondsEstimate)});
    nominal = maxMetric(nominal, rankNominal);
  }
  if (cost.rankCosts.empty())
    nominal = sumMetrics(
        {timeForOccurrences(cost.aggregateInstructionCount,
                            policy.instructionFixedPicosecondsEstimate),
         timeForOccurrences(cost.aggregateNoC.waitedEventCount,
                            policy.dteWaitedEventPicosecondsEstimate),
         timeForOccurrences(cost.aggregateNCCParticipantWaitCount,
                            policy.nccParticipantWaitPicosecondsEstimate)});

  ScheduleCostMetric instructionUpper =
      timeForOccurrences(cost.aggregateInstructionCount,
                         policy.instructionFixedPicosecondsUpperBound);
  ScheduleCostMetric dteWaitUpper =
      timeForOccurrences(cost.aggregateNoC.waitedEventCount,
                         policy.dteWaitedEventPicosecondsUpperBound);
  ScheduleCostMetric nccWaitUpper =
      timeForOccurrences(cost.aggregateNCCParticipantWaitCount,
                         policy.nccParticipantWaitPicosecondsUpperBound);
  return {{},
          nominal,
          sumMetrics({instructionUpper, dteWaitUpper, nccWaitUpper}),
          assumptions({StaticDurationAssumption::ControlIssuePrior})};
}

static bool knownStrictlyLower(ScheduleCostMetric candidate,
                               ScheduleCostMetric baseline) {
  return candidate.isKnown() && baseline.isKnown() &&
         candidate.value < baseline.value;
}

static bool marginClears(uint64_t candidateUpper, uint64_t baselineLower,
                         uint32_t marginPermille) {
  unsigned __int128 lhs = static_cast<unsigned __int128>(candidateUpper) *
                          (1000ULL + marginPermille);
  unsigned __int128 rhs =
      static_cast<unsigned __int128>(baselineLower) * 1000ULL;
  return lhs < rhs;
}

} // namespace

WholeCardResourceDurationEstimate
estimateWholeCardResourceDuration(const WholeCardInstructionProgramCost &cost,
                                  const TargetScheduleCostPolicy &policy,
                                  StaticCrossResourceSchedule schedule) {
  WholeCardResourceDurationEstimate result;
  result.ddr = estimateDDR(cost, policy);
  result.compute = estimateCompute(cost, policy);
  result.noc = estimateNoC(cost, policy);
  result.spm = estimateSPM(cost, policy);
  result.control = estimateControl(cost, policy);
  result.makespan.lowerBoundPicoseconds = maxMetrics(
      {result.ddr.lowerBoundPicoseconds, result.compute.lowerBoundPicoseconds,
       result.noc.lowerBoundPicoseconds, result.control.lowerBoundPicoseconds});
  ScheduleCostMetric service;
  switch (schedule) {
  case StaticCrossResourceSchedule::SequentialPhases:
    // DDR, inter-tile communication and compute are charged in dependency
    // order when current IR has no qualified recurring overlap schedule.
    service = sumMetrics({result.ddr.nominalPicoseconds,
                          result.compute.nominalPicoseconds,
                          result.noc.nominalPicoseconds});
    break;
  case StaticCrossResourceSchedule::PipelinedSteadyState:
    // Explicit multi-buffer/fixed-slot evidence permits the standard
    // steady-state resource-envelope model used by tile pipelines.
    service = maxMetrics({result.ddr.nominalPicoseconds,
                          result.compute.nominalPicoseconds,
                          result.noc.nominalPicoseconds});
    break;
  }
  result.makespan.nominalPicoseconds =
      addMetric(service, result.control.nominalPicoseconds);
  result.makespan.nominalAssumptions =
      result.ddr.nominalAssumptions | result.compute.nominalAssumptions |
      result.noc.nominalAssumptions | result.control.nominalAssumptions |
      staticDurationAssumptionMask(
          schedule == StaticCrossResourceSchedule::SequentialPhases
              ? StaticDurationAssumption::SequentialPhaseModel
              : StaticDurationAssumption::QualifiedPipelineModel);
  result.makespan.upperBoundPicoseconds = sumMetrics(
      {result.ddr.upperBoundPicoseconds, result.compute.upperBoundPicoseconds,
       result.noc.upperBoundPicoseconds, result.control.upperBoundPicoseconds});
  return result;
}

NoCTradeoffProfitability analyzeNoCTradeoffProfitability(
    const WholeCardInstructionProgramCost &candidate,
    const WholeCardInstructionProgramCost &baseline,
    const TargetScheduleCostPolicy &policy,
    NoCTradeoffScheduleContext schedule) {
  NoCTradeoffProfitability result;
  result.baseline =
      estimateWholeCardResourceDuration(baseline, policy, schedule.baseline);
  result.candidate =
      estimateWholeCardResourceDuration(candidate, policy, schedule.candidate);

  ScheduleCostMetric candidateDDR = addMetric(candidate.aggregateDDRReadBytes,
                                              candidate.aggregateDDRWriteBytes);
  ScheduleCostMetric baselineDDR = addMetric(baseline.aggregateDDRReadBytes,
                                             baseline.aggregateDDRWriteBytes);

  const ScheduleCostMetric candidateNoCFacts[] = {
      candidate.aggregateNoC.staticIssueSiteCount,
      candidate.aggregateNoC.aggregateTransmitBytes,
      candidate.aggregateNoC.aggregateReceiveBytes,
      candidate.aggregateNoC.transmitMessageCount,
      candidate.aggregateNoC.receiveMessageCount,
      candidate.aggregateNoC.waitOperationCount,
      candidate.aggregateNoC.waitedEventCount,
      candidate.minimumHopLinkByteDemand,
      candidate.minimumHopMessageDemand,
      candidate.idealizedMinimumPeakLinkByteDemand,
      candidate.modeledNoCRoute.peakDirectedLinkByteDemand,
      candidate.maximumNoCHopCount,
      candidate.maximumRankNoCTransmitBytes,
      candidate.maximumRankNoCReceiveBytes,
      candidate.maximumRankNoCTransmitMessageCount,
      candidate.maximumRankNoCReceiveMessageCount,
  };
  const ScheduleCostMetric baselineNoCFacts[] = {
      baseline.aggregateNoC.staticIssueSiteCount,
      baseline.aggregateNoC.aggregateTransmitBytes,
      baseline.aggregateNoC.aggregateReceiveBytes,
      baseline.aggregateNoC.transmitMessageCount,
      baseline.aggregateNoC.receiveMessageCount,
      baseline.aggregateNoC.waitOperationCount,
      baseline.aggregateNoC.waitedEventCount,
      baseline.minimumHopLinkByteDemand,
      baseline.minimumHopMessageDemand,
      baseline.idealizedMinimumPeakLinkByteDemand,
      baseline.modeledNoCRoute.peakDirectedLinkByteDemand,
      baseline.maximumNoCHopCount,
      baseline.maximumRankNoCTransmitBytes,
      baseline.maximumRankNoCReceiveBytes,
      baseline.maximumRankNoCTransmitMessageCount,
      baseline.maximumRankNoCReceiveMessageCount,
  };
  const bool noCFactsKnown =
      llvm::all_of(
          candidateNoCFacts,
          [](ScheduleCostMetric metric) { return metric.isKnown(); }) &&
      llvm::all_of(baselineNoCFacts,
                   [](ScheduleCostMetric metric) { return metric.isKnown(); });
  // A known NoC-free candidate stays under the ordinary selector even if an
  // unrelated DDR dimension is Unknown. A candidate that retains an existing
  // collective is still NoC-dependent: equality against a NoC-bearing
  // baseline must not bypass the makespan gate.
  if (candidate.aggregateNoC.staticIssueSiteCount.isKnown() &&
      candidate.aggregateNoC.staticIssueSiteCount.value == 0)
    return result;
  if (!candidateDDR.isKnown() || !baselineDDR.isKnown()) {
    result.decision = NoCTradeoffDecision::Indeterminate;
    result.reason = NoCTradeoffReason::UnknownCostFact;
    return result;
  }
  if (!knownStrictlyLower(candidateDDR, baselineDDR))
    return result;
  if (!noCFactsKnown) {
    result.decision = NoCTradeoffDecision::Indeterminate;
    result.reason = NoCTradeoffReason::UnknownCostFact;
    return result;
  }

  const ScheduleCostMetric candidateNominal =
      result.candidate.makespan.nominalPicoseconds;
  const ScheduleCostMetric baselineNominal =
      result.baseline.makespan.nominalPicoseconds;
  if (!candidateNominal.isKnown() || !baselineNominal.isKnown()) {
    result.decision = NoCTradeoffDecision::Indeterminate;
    result.reason = NoCTradeoffReason::UnknownCostFact;
    return result;
  }
  if (candidateNominal.value >= baselineNominal.value) {
    result.decision = NoCTradeoffDecision::Reject;
    result.reason = NoCTradeoffReason::NominalMakespanNotImproved;
    return result;
  }
  if (!marginClears(candidateNominal.value, baselineNominal.value,
                    policy.productionBenefitMarginPermille)) {
    result.decision = NoCTradeoffDecision::Reject;
    result.reason = NoCTradeoffReason::InsufficientBenefitMargin;
    return result;
  }

  const ScheduleCostMetric candidateUpper =
      result.candidate.makespan.upperBoundPicoseconds;
  const ScheduleCostMetric baselineLower =
      result.baseline.makespan.lowerBoundPicoseconds;
  if (candidateUpper.isKnown() && baselineLower.isKnown() &&
      marginClears(candidateUpper.value, baselineLower.value,
                   policy.productionBenefitMarginPermille)) {
    result.decision = NoCTradeoffDecision::ProvenBenefit;
    result.reason = NoCTradeoffReason::ConservativeBoundsProveBenefit;
    return result;
  }

  result.decision = NoCTradeoffDecision::EstimatedBenefit;
  result.reason = NoCTradeoffReason::EstimatedModelClearsMargin;
  return result;
}

llvm::StringRef stringifyNoCTradeoffDecision(NoCTradeoffDecision decision) {
  switch (decision) {
  case NoCTradeoffDecision::NotApplicable:
    return "not-applicable";
  case NoCTradeoffDecision::Reject:
    return "reject";
  case NoCTradeoffDecision::Indeterminate:
    return "indeterminate";
  case NoCTradeoffDecision::EstimatedBenefit:
    return "estimated-benefit";
  case NoCTradeoffDecision::ProvenBenefit:
    return "proven-benefit";
  }
  llvm_unreachable("unhandled NoC tradeoff decision");
}

llvm::StringRef stringifyNoCTradeoffReason(NoCTradeoffReason reason) {
  switch (reason) {
  case NoCTradeoffReason::NoCrossResourceTradeoff:
    return "no-cross-resource-tradeoff";
  case NoCTradeoffReason::UnknownCostFact:
    return "unknown-cost-fact";
  case NoCTradeoffReason::NominalMakespanNotImproved:
    return "nominal-makespan-not-improved";
  case NoCTradeoffReason::InsufficientBenefitMargin:
    return "insufficient-benefit-margin";
  case NoCTradeoffReason::EstimatedModelClearsMargin:
    return "estimated-model-clears-margin";
  case NoCTradeoffReason::ConservativeBoundsProveBenefit:
    return "conservative-bounds-prove-benefit";
  }
  llvm_unreachable("unhandled NoC tradeoff reason");
}

} // namespace wafer::analysis
