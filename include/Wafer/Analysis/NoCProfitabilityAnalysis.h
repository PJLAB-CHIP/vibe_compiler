//===- NoCProfitabilityAnalysis.h - Whole-card NoC tradeoff ----*- C++ -*-===//

#ifndef WAFER_ANALYSIS_NOCPROFITABILITYANALYSIS_H
#define WAFER_ANALYSIS_NOCPROFITABILITYANALYSIS_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace wafer::analysis {

/// Assumptions used only to produce a point estimate. They are deliberately
/// separate from ScheduleCostKnowledge: an exact byte/message count may be
/// Known while the route or service rate used to time it is Estimated.
enum class StaticDurationAssumption : uint32_t {
  None = 0,
  DDROperatingPoint = 1u << 0,
  ComputePeakReference = 1u << 1,
  ModeledShortestPathRoute = 1u << 2,
  DTEEndpointRatePrior = 1u << 3,
  DTEMessageStartupPrior = 1u << 4,
  NoCHopPrior = 1u << 5,
  ControlIssuePrior = 1u << 6,
  SequentialPhaseModel = 1u << 7,
  QualifiedPipelineModel = 1u << 8,
  SPMServiceRatePrior = 1u << 9,
};

using StaticDurationAssumptionMask = uint32_t;

constexpr StaticDurationAssumptionMask
staticDurationAssumptionMask(StaticDurationAssumption assumption) {
  return static_cast<StaticDurationAssumptionMask>(assumption);
}

/// A duration interval derived from one complete-rank final instruction
/// program. Picoseconds keep small payloads distinguishable without implying
/// cycle accuracy.
struct StaticDurationInterval {
  ScheduleCostMetric lowerBoundPicoseconds;
  ScheduleCostMetric nominalPicoseconds;
  ScheduleCostMetric upperBoundPicoseconds;
  StaticDurationAssumptionMask nominalAssumptions = 0;
};

/// Resource-constrained duration facts for one whole-card variant. Nominal
/// values are analytical point estimates. Lower/upper fields remain genuine
/// hardware/calibrated bounds and alone may establish a proof.
struct WholeCardResourceDurationEstimate {
  StaticDurationInterval ddr;
  StaticDurationInterval compute;
  StaticDurationInterval noc;
  StaticDurationInterval spm;
  StaticDurationInterval control;
  /// The lower bound is the maximum calibrated resource floor, the nominal
  /// reference follows the supplied current-IR schedule context, and the
  /// conservative upper bound serializes calibrated external-resource
  /// envelopes. Explicit SPM movement is a simultaneous local-port envelope:
  /// its nominal point is assumption-marked, and its optional conservative
  /// bound participates without adding DDR-visible movement a second time.
  StaticDurationInterval makespan;
};

/// How the current final instruction schedule permits independent engines to
/// overlap. This is supplied from current-IR scheduling/capability evidence;
/// it is not inferred from a workload or operation name.
enum class StaticCrossResourceSchedule : uint8_t {
  /// No steady-state overlap contract: DDR, NoC and compute service phases are
  /// charged in dependency order. SPM remains a simultaneous port constraint.
  SequentialPhases,
  /// Explicit multi-buffer/fixed-slot scheduling plus target capability
  /// evidence permits a steady-state resource-envelope estimate.
  PipelinedSteadyState,
};

struct NoCTradeoffScheduleContext {
  StaticCrossResourceSchedule baseline =
      StaticCrossResourceSchedule::SequentialPhases;
  StaticCrossResourceSchedule candidate =
      StaticCrossResourceSchedule::SequentialPhases;
};

enum class NoCTradeoffDecision : uint8_t {
  /// The candidate does not exchange lower whole-card DDR traffic for higher
  /// or retained NoC-dependent execution, so the ordinary exact/static
  /// selector remains responsible.
  NotApplicable,
  /// The nominal reference does not show an opportunity or does not clear the
  /// required production margin.
  Reject,
  /// A nominal decision cannot be formed because at least one required exact
  /// work fact or point-model input is structurally unavailable. Missing
  /// conservative proof calibration by itself does not cause this state.
  Indeterminate,
  /// Exact final-IR work plus versioned target point parameters clear the
  /// production margin. This is usable by normal production but is not
  /// reported as a calibrated proof.
  EstimatedBenefit,
  /// The candidate upper bound plus margin is below the baseline lower bound.
  ProvenBenefit,
};

enum class NoCTradeoffReason : uint8_t {
  NoCrossResourceTradeoff,
  UnknownCostFact,
  NominalMakespanNotImproved,
  InsufficientBenefitMargin,
  EstimatedModelClearsMargin,
  ConservativeBoundsProveBenefit,
};

struct NoCTradeoffProfitability {
  NoCTradeoffDecision decision = NoCTradeoffDecision::NotApplicable;
  NoCTradeoffReason reason = NoCTradeoffReason::NoCrossResourceTradeoff;
  WholeCardResourceDurationEstimate baseline;
  WholeCardResourceDurationEstimate candidate;
};

/// Build duration intervals from exact final-IR work and versioned target
/// parameters. Missing calibration remains Unknown and is never replaced by
/// zero.
WholeCardResourceDurationEstimate estimateWholeCardResourceDuration(
    const WholeCardInstructionProgramCost &cost,
    const TargetScheduleCostPolicy &policy,
    StaticCrossResourceSchedule schedule =
        StaticCrossResourceSchedule::SequentialPhases);

/// Compare a candidate against its already accepted reserved baseline. This
/// analysis is pure and invocation-local; it neither mutates nor annotates IR.
NoCTradeoffProfitability analyzeNoCTradeoffProfitability(
    const WholeCardInstructionProgramCost &candidate,
    const WholeCardInstructionProgramCost &baseline,
    const TargetScheduleCostPolicy &policy,
    NoCTradeoffScheduleContext schedule = {});

llvm::StringRef stringifyNoCTradeoffDecision(NoCTradeoffDecision decision);
llvm::StringRef stringifyNoCTradeoffReason(NoCTradeoffReason reason);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_NOCPROFITABILITYANALYSIS_H
