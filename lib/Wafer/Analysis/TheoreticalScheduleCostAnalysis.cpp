//===- TheoreticalScheduleCostAnalysis.cpp - Numeric theoretical cost -===//

#include "Wafer/Analysis/TheoreticalScheduleCostAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

namespace wafer::analysis {
namespace {

constexpr uint64_t kPicosecondsPerSecond = 1'000'000'000'000ULL;

static bool enabled(StaticDurationTermMask terms, StaticDurationTerm term) {
  return (terms & staticDurationTermMask(term)) != 0;
}

static void disable(StaticDurationTermMask &terms, StaticDurationTerm term) {
  terms &= ~staticDurationTermMask(term);
}

static uint64_t saturatingAdd(uint64_t lhs, uint64_t rhs) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return std::numeric_limits<uint64_t>::max();
  return lhs + rhs;
}

static uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  if (product > std::numeric_limits<uint64_t>::max())
    return std::numeric_limits<uint64_t>::max();
  return static_cast<uint64_t>(product);
}

static uint64_t timeForWork(uint64_t work, uint64_t unitsPerSecond) {
  if (work == 0 || unitsPerSecond == 0)
    return 0;
  const unsigned __int128 numerator =
      static_cast<unsigned __int128>(work) * kPicosecondsPerSecond;
  const unsigned __int128 duration =
      (numerator + unitsPerSecond - 1) / unitsPerSecond;
  if (duration > std::numeric_limits<uint64_t>::max())
    return std::numeric_limits<uint64_t>::max();
  return static_cast<uint64_t>(duration);
}

static bool
allKnown(llvm::ArrayRef<const ScheduleCostMetric *> requiredMetrics) {
  return llvm::all_of(requiredMetrics, [](const ScheduleCostMetric *metric) {
    return metric && metric->isKnown();
  });
}

template <typename MetricAccessor>
static bool allTileMetricsKnown(const WholeCardInstructionProgramCost &cost,
                                MetricAccessor accessor,
                                const ScheduleCostMetric &aggregateFallback) {
  if (cost.tileCosts.empty())
    return aggregateFallback.isKnown();
  return llvm::all_of(cost.tileCosts, [&](const InstructionProgramCost &tile) {
    return accessor(tile).isKnown();
  });
}

static bool isKnownNoCFree(const WholeCardInstructionProgramCost &cost) {
  return cost.aggregateNoC.staticIssueSiteCount.isKnown() &&
         cost.aggregateNoC.staticIssueSiteCount.value == 0;
}

static StaticDurationTermMask
getParameterEnabledTerms(const TargetScheduleCostPolicy &policy) {
  StaticDurationTermMask terms = 0;
  auto addWhen = [&](bool condition, StaticDurationTerm term) {
    if (condition)
      terms |= staticDurationTermMask(term);
  };
  addWhen(policy.cardDDRNominalBytesPerSecond != 0, StaticDurationTerm::DDR);
  addWhen(policy.f16Bf16NPULogicalOpsPerSecondPerTile != 0,
          StaticDurationTerm::NPUF16Bf16);
  addWhen(policy.f16Bf16VectorLogicalOpsPerSecondPerTile != 0,
          StaticDurationTerm::VectorF16Bf16);
  addWhen(policy.f32VectorLogicalOpsPerSecondPerTile != 0,
          StaticDurationTerm::VectorF32);
  addWhen(policy.directionalNoCBytesPerSecond != 0,
          StaticDurationTerm::NoCLink);
  addWhen(policy.dteEndpointBytesPerSecondEstimate != 0,
          StaticDurationTerm::NoCTransmitEndpoint);
  addWhen(policy.dteEndpointBytesPerSecondEstimate != 0,
          StaticDurationTerm::NoCReceiveEndpoint);
  addWhen(policy.dteMessageStartupPicosecondsEstimate != 0,
          StaticDurationTerm::NoCMessageStartup);
  addWhen(policy.noCHopPicosecondsEstimate != 0, StaticDurationTerm::NoCHop);
  addWhen(policy.spmExplicitMovementBytesPerSecondPerTileEstimate != 0,
          StaticDurationTerm::SPMMovement);
  addWhen(policy.instructionFixedPicosecondsEstimate != 0,
          StaticDurationTerm::InstructionControl);
  addWhen(policy.dteWaitedEventPicosecondsEstimate != 0,
          StaticDurationTerm::DTEWaitControl);
  addWhen(policy.nccParticipantWaitPicosecondsEstimate != 0,
          StaticDurationTerm::NCCWaitControl);
  return terms;
}

static bool hasScheduleResource(StaticScheduleResourceMask resources,
                                StaticScheduleResource resource) {
  return (resources & staticScheduleResourceMask(resource)) != 0;
}

static void
disableUnavailableResourceTerms(StaticDurationTermMask &terms,
                                const WholeCardInstructionProgramCost &cost,
                                StaticScheduleResourceMask resources) {
  if (hasScheduleResource(resources, StaticScheduleResource::DDR) &&
      !allKnown({&cost.aggregateDDRReadBytes, &cost.aggregateDDRWriteBytes}))
    disable(terms, StaticDurationTerm::DDR);

  if (hasScheduleResource(resources, StaticScheduleResource::Compute)) {
    if (!allTileMetricsKnown(
            cost,
            [](const InstructionProgramCost &tile)
                -> const ScheduleCostMetric & {
              return tile.compute.npuF16Bf16LogicalOps;
            },
            cost.aggregateCompute.npuF16Bf16LogicalOps))
      disable(terms, StaticDurationTerm::NPUF16Bf16);
    if (!allTileMetricsKnown(
            cost,
            [](const InstructionProgramCost &tile)
                -> const ScheduleCostMetric & {
              return tile.compute.vectorF16Bf16LogicalOps;
            },
            cost.aggregateCompute.vectorF16Bf16LogicalOps))
      disable(terms, StaticDurationTerm::VectorF16Bf16);
    if (!allTileMetricsKnown(
            cost,
            [](const InstructionProgramCost &tile)
                -> const ScheduleCostMetric & {
              return tile.compute.vectorF32LogicalOps;
            },
            cost.aggregateCompute.vectorF32LogicalOps))
      disable(terms, StaticDurationTerm::VectorF32);
  }

  if (hasScheduleResource(resources, StaticScheduleResource::NoC) &&
      !isKnownNoCFree(cost)) {
    if (!cost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown())
      disable(terms, StaticDurationTerm::NoCLink);
    if (!cost.maximumTileNoCTransmitBytes.isKnown())
      disable(terms, StaticDurationTerm::NoCTransmitEndpoint);
    if (!cost.maximumTileNoCReceiveBytes.isKnown())
      disable(terms, StaticDurationTerm::NoCReceiveEndpoint);
    if (!allKnown({&cost.maximumTileNoCTransmitMessageCount,
                   &cost.maximumTileNoCReceiveMessageCount}))
      disable(terms, StaticDurationTerm::NoCMessageStartup);
    if (!cost.maximumNoCHopCount.isKnown())
      disable(terms, StaticDurationTerm::NoCHop);
  }

  if (hasScheduleResource(resources, StaticScheduleResource::SPMMovement)) {
    const ScheduleCostMetric &spm = cost.tileCosts.empty()
                                        ? cost.aggregateSPMMovementBytes
                                        : cost.maximumTileSPMMovementBytes;
    if (!spm.isKnown())
      disable(terms, StaticDurationTerm::SPMMovement);
  }
}

static void
disableUnavailableControlTerms(StaticDurationTermMask &terms,
                               const WholeCardInstructionProgramCost &cost) {
  if (!allTileMetricsKnown(
          cost,
          [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
            return tile.instructionCount;
          },
          cost.aggregateInstructionCount))
    disable(terms, StaticDurationTerm::InstructionControl);
  if (!allTileMetricsKnown(
          cost,
          [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
            return tile.noc.waitedEventCount;
          },
          cost.aggregateNoC.waitedEventCount))
    disable(terms, StaticDurationTerm::DTEWaitControl);
  if (!allTileMetricsKnown(
          cost,
          [](const InstructionProgramCost &tile) -> const ScheduleCostMetric & {
            return tile.nccParticipantWaitCount;
          },
          cost.aggregateNCCParticipantWaitCount))
    disable(terms, StaticDurationTerm::NCCWaitControl);
}

static uint64_t estimateCompute(const WholeCardInstructionProgramCost &cost,
                                const TargetScheduleCostPolicy &policy,
                                StaticDurationTermMask terms) {
  auto estimateTile = [&](const ScheduleComputeCost &compute) {
    uint64_t duration = 0;
    if (enabled(terms, StaticDurationTerm::NPUF16Bf16))
      duration = saturatingAdd(
          duration, timeForWork(compute.npuF16Bf16LogicalOps.value,
                                policy.f16Bf16NPULogicalOpsPerSecondPerTile));
    if (enabled(terms, StaticDurationTerm::VectorF16Bf16))
      duration = saturatingAdd(
          duration,
          timeForWork(compute.vectorF16Bf16LogicalOps.value,
                      policy.f16Bf16VectorLogicalOpsPerSecondPerTile));
    if (enabled(terms, StaticDurationTerm::VectorF32))
      duration = saturatingAdd(
          duration, timeForWork(compute.vectorF32LogicalOps.value,
                                policy.f32VectorLogicalOpsPerSecondPerTile));
    return duration;
  };

  if (cost.tileCosts.empty())
    return estimateTile(cost.aggregateCompute);
  uint64_t maximum = 0;
  for (const InstructionProgramCost &tile : cost.tileCosts)
    maximum = std::max(maximum, estimateTile(tile.compute));
  return maximum;
}

static uint64_t estimateControl(const WholeCardInstructionProgramCost &cost,
                                const TargetScheduleCostPolicy &policy,
                                StaticDurationTermMask terms) {
  auto estimateTile = [&](const ScheduleCostMetric &instructions,
                          const ScheduleCostMetric &dteWaits,
                          const ScheduleCostMetric &nccWaits) {
    uint64_t duration = 0;
    if (enabled(terms, StaticDurationTerm::InstructionControl))
      duration = saturatingAdd(
          duration,
          saturatingMultiply(instructions.value,
                             policy.instructionFixedPicosecondsEstimate));
    if (enabled(terms, StaticDurationTerm::DTEWaitControl))
      duration = saturatingAdd(
          duration,
          saturatingMultiply(dteWaits.value,
                             policy.dteWaitedEventPicosecondsEstimate));
    if (enabled(terms, StaticDurationTerm::NCCWaitControl))
      duration = saturatingAdd(
          duration,
          saturatingMultiply(nccWaits.value,
                             policy.nccParticipantWaitPicosecondsEstimate));
    return duration;
  };

  if (cost.tileCosts.empty())
    return estimateTile(cost.aggregateInstructionCount,
                        cost.aggregateNoC.waitedEventCount,
                        cost.aggregateNCCParticipantWaitCount);
  uint64_t maximum = 0;
  for (const InstructionProgramCost &tile : cost.tileCosts)
    maximum = std::max(maximum, estimateTile(tile.instructionCount,
                                             tile.noc.waitedEventCount,
                                             tile.nccParticipantWaitCount));
  return maximum;
}

static const StaticDurationEstimate &
getResourceEstimate(const WholeCardResourceDurationEstimate &estimate,
                    StaticScheduleResource resource) {
  switch (resource) {
  case StaticScheduleResource::DDR:
    return estimate.ddr;
  case StaticScheduleResource::Compute:
    return estimate.compute;
  case StaticScheduleResource::NoC:
    return estimate.noc;
  case StaticScheduleResource::SPMMovement:
    return estimate.spm;
  }
  llvm_unreachable("unhandled static schedule resource");
}

static StaticDurationEstimate &
getResourceEstimate(WholeCardResourceDurationEstimate &estimate,
                    StaticScheduleResource resource) {
  return const_cast<StaticDurationEstimate &>(getResourceEstimate(
      static_cast<const WholeCardResourceDurationEstimate &>(estimate),
      resource));
}

static constexpr std::array<StaticScheduleResource, 4> kScheduleResources = {
    StaticScheduleResource::DDR, StaticScheduleResource::Compute,
    StaticScheduleResource::NoC, StaticScheduleResource::SPMMovement};

using ResourceEstimateMap =
    llvm::DenseMap<const WholeCardInstructionProgramCost *,
                   WholeCardResourceDurationEstimate>;

static uint64_t estimateSequentialWork(const StaticScheduleWork &work,
                                       const ResourceEstimateMap &estimates) {
  const auto estimateIt = estimates.find(work.cost);
  assert(estimateIt != estimates.end() && "validated work must be estimated");
  uint64_t duration = 0;
  for (StaticScheduleResource resource : kScheduleResources) {
    if (hasScheduleResource(work.resources, resource))
      duration = saturatingAdd(
          duration,
          getResourceEstimate(estimateIt->second, resource).picoseconds);
  }
  return duration;
}

static uint64_t estimateSequentialBranch(const StaticScheduleBranch &branch,
                                         const ResourceEstimateMap &estimates) {
  uint64_t duration = 0;
  for (const StaticScheduleWork &work : branch.dependentWork)
    duration = saturatingAdd(duration, estimateSequentialWork(work, estimates));
  return duration;
}

static uint64_t estimateIndependentStage(const ResourceEstimateMap &estimates,
                                         const StaticScheduleStage &stage) {
  uint64_t duration = 0;
  for (const StaticScheduleBranch &branch : stage.independentBranches)
    duration = std::max(duration, estimateSequentialBranch(branch, estimates));
  return duration;
}

static uint64_t
estimateDependentStages(const ResourceEstimateMap &estimates,
                        llvm::ArrayRef<StaticScheduleStage> stages) {
  uint64_t duration = 0;
  for (const StaticScheduleStage &stage : stages)
    duration =
        saturatingAdd(duration, estimateIndependentStage(estimates, stage));
  return duration;
}

static uint64_t estimateSchedulePlan(const ResourceEstimateMap &estimates,
                                     const StaticSchedulePlan &plan) {
  uint64_t duration = 0;
  for (const StaticScheduleStep &step : plan.getSteps()) {
    switch (step.kind) {
    case StaticScheduleStep::Kind::Stage:
      duration = saturatingAdd(duration,
                               estimateIndependentStage(estimates, step.stage));
      break;
    case StaticScheduleStep::Kind::BufferedPipeline: {
      const uint64_t prologue =
          estimateDependentStages(estimates, step.pipeline.prologue);
      const uint64_t initiationInterval =
          estimateIndependentStage(estimates, step.pipeline.steady);
      const uint64_t steady =
          saturatingMultiply(initiationInterval, step.pipeline.steadyWaveCount);
      const uint64_t epilogue =
          estimateDependentStages(estimates, step.pipeline.epilogue);
      duration = saturatingAdd(
          duration, saturatingAdd(prologue, saturatingAdd(steady, epilogue)));
      break;
    }
    }
  }
  return duration;
}

static bool hasBufferedPipeline(const StaticSchedulePlan &plan) {
  return llvm::any_of(plan.getSteps(), [](const StaticScheduleStep &step) {
    return step.kind == StaticScheduleStep::Kind::BufferedPipeline;
  });
}

static WholeCardResourceDurationEstimate
estimateResourceDurations(const WholeCardInstructionProgramCost &cost,
                          const TargetScheduleCostPolicy &policy,
                          StaticDurationTermMask terms) {
  WholeCardResourceDurationEstimate result;
  result.enabledTerms = terms;

  if (enabled(terms, StaticDurationTerm::DDR)) {
    const uint64_t bytes = saturatingAdd(cost.aggregateDDRReadBytes.value,
                                         cost.aggregateDDRWriteBytes.value);
    result.ddr.picoseconds =
        timeForWork(bytes, policy.cardDDRNominalBytesPerSecond);
    result.ddr.assumptions = staticDurationAssumptionMask(
        StaticDurationAssumption::DDROperatingPoint);
  }

  result.compute.picoseconds = estimateCompute(cost, policy, terms);
  if (enabled(terms, StaticDurationTerm::NPUF16Bf16) ||
      enabled(terms, StaticDurationTerm::VectorF16Bf16) ||
      enabled(terms, StaticDurationTerm::VectorF32))
    result.compute.assumptions = staticDurationAssumptionMask(
        StaticDurationAssumption::ComputePeakReference);

  if (!isKnownNoCFree(cost)) {
    const uint64_t link =
        enabled(terms, StaticDurationTerm::NoCLink)
            ? timeForWork(cost.modeledNoCRoute.peakDirectedLinkByteDemand.value,
                          policy.directionalNoCBytesPerSecond)
            : 0;
    uint64_t transmitEndpoint =
        enabled(terms, StaticDurationTerm::NoCTransmitEndpoint)
            ? timeForWork(cost.maximumTileNoCTransmitBytes.value,
                          policy.dteEndpointBytesPerSecondEstimate)
            : 0;
    uint64_t receiveEndpoint =
        enabled(terms, StaticDurationTerm::NoCReceiveEndpoint)
            ? timeForWork(cost.maximumTileNoCReceiveBytes.value,
                          policy.dteEndpointBytesPerSecondEstimate)
            : 0;
    if (enabled(terms, StaticDurationTerm::NoCMessageStartup)) {
      transmitEndpoint = saturatingAdd(
          transmitEndpoint,
          saturatingMultiply(cost.maximumTileNoCTransmitMessageCount.value,
                             policy.dteMessageStartupPicosecondsEstimate));
      receiveEndpoint = saturatingAdd(
          receiveEndpoint,
          saturatingMultiply(cost.maximumTileNoCReceiveMessageCount.value,
                             policy.dteMessageStartupPicosecondsEstimate));
    }
    const uint64_t routeFill =
        enabled(terms, StaticDurationTerm::NoCHop)
            ? saturatingMultiply(cost.maximumNoCHopCount.value,
                                 policy.noCHopPicosecondsEstimate)
            : 0;
    result.noc.picoseconds = saturatingAdd(
        std::max({link, transmitEndpoint, receiveEndpoint}), routeFill);
  }
  if (enabled(terms, StaticDurationTerm::NoCLink))
    result.noc.assumptions |= staticDurationAssumptionMask(
        StaticDurationAssumption::ModeledShortestPathRoute);
  if (enabled(terms, StaticDurationTerm::NoCTransmitEndpoint) ||
      enabled(terms, StaticDurationTerm::NoCReceiveEndpoint))
    result.noc.assumptions |= staticDurationAssumptionMask(
        StaticDurationAssumption::DTEEndpointRatePrior);
  if (enabled(terms, StaticDurationTerm::NoCMessageStartup))
    result.noc.assumptions |= staticDurationAssumptionMask(
        StaticDurationAssumption::DTEMessageStartupPrior);
  if (enabled(terms, StaticDurationTerm::NoCHop))
    result.noc.assumptions |=
        staticDurationAssumptionMask(StaticDurationAssumption::NoCHopPrior);

  if (enabled(terms, StaticDurationTerm::SPMMovement)) {
    const ScheduleCostMetric &movement = cost.tileCosts.empty()
                                             ? cost.aggregateSPMMovementBytes
                                             : cost.maximumTileSPMMovementBytes;
    result.spm.picoseconds =
        timeForWork(movement.value,
                    policy.spmExplicitMovementBytesPerSecondPerTileEstimate);
    result.spm.assumptions = staticDurationAssumptionMask(
        StaticDurationAssumption::SPMServiceRatePrior);
  }

  result.control.picoseconds = estimateControl(cost, policy, terms);
  if (enabled(terms, StaticDurationTerm::InstructionControl) ||
      enabled(terms, StaticDurationTerm::DTEWaitControl) ||
      enabled(terms, StaticDurationTerm::NCCWaitControl))
    result.control.assumptions = staticDurationAssumptionMask(
        StaticDurationAssumption::ControlIssuePrior);

  return result;
}

static void accumulateWorkResourceDurations(
    WholeCardResourceDurationEstimate &result, const StaticScheduleWork &work,
    const ResourceEstimateMap &estimates, uint64_t repetition = 1) {
  const auto estimateIt = estimates.find(work.cost);
  assert(estimateIt != estimates.end() && "validated work must be estimated");
  for (StaticScheduleResource resource : kScheduleResources) {
    if (!hasScheduleResource(work.resources, resource))
      continue;
    StaticDurationEstimate &target = getResourceEstimate(result, resource);
    const StaticDurationEstimate &source =
        getResourceEstimate(estimateIt->second, resource);
    target.picoseconds = saturatingAdd(
        target.picoseconds, saturatingMultiply(source.picoseconds, repetition));
    target.assumptions |= source.assumptions;
  }
}

static void accumulateStageResourceDurations(
    WholeCardResourceDurationEstimate &result, const StaticScheduleStage &stage,
    const ResourceEstimateMap &estimates, uint64_t repetition = 1) {
  for (const StaticScheduleBranch &branch : stage.independentBranches)
    for (const StaticScheduleWork &work : branch.dependentWork)
      accumulateWorkResourceDurations(result, work, estimates, repetition);
}

static WholeCardResourceDurationEstimate
estimatePlan(const StaticSchedulePlan &plan, StaticDurationTermMask terms,
             const ResourceEstimateMap &estimates) {
  WholeCardResourceDurationEstimate result;
  result.enabledTerms = terms;
  for (const StaticScheduleStep &step : plan.getSteps()) {
    switch (step.kind) {
    case StaticScheduleStep::Kind::Stage:
      accumulateStageResourceDurations(result, step.stage, estimates);
      break;
    case StaticScheduleStep::Kind::BufferedPipeline:
      for (const StaticScheduleStage &stage : step.pipeline.prologue)
        accumulateStageResourceDurations(result, stage, estimates);
      accumulateStageResourceDurations(result, step.pipeline.steady, estimates,
                                       step.pipeline.steadyWaveCount);
      for (const StaticScheduleStage &stage : step.pipeline.epilogue)
        accumulateStageResourceDurations(result, stage, estimates);
      break;
    }
  }

  const auto controlIt = estimates.find(&plan.getControlCost());
  assert(controlIt != estimates.end() &&
         "validated control cost must be estimated");
  result.control = controlIt->second.control;
  result.makespan.picoseconds = saturatingAdd(
      estimateSchedulePlan(estimates, plan), result.control.picoseconds);
  result.makespan.assumptions =
      result.ddr.assumptions | result.compute.assumptions |
      result.noc.assumptions | result.spm.assumptions |
      result.control.assumptions |
      staticDurationAssumptionMask(
          StaticDurationAssumption::ExplicitSchedulePlan);
  if (hasBufferedPipeline(plan))
    result.makespan.assumptions |= staticDurationAssumptionMask(
        StaticDurationAssumption::BufferedPipelinePlan);
  return result;
}

static bool validateWork(const StaticScheduleWork &work,
                         StaticScheduleResourceMask &referencedResources) {
  if (!work.cost || work.resources == 0 ||
      (work.resources & ~allStaticScheduleResources()) != 0)
    return false;
  referencedResources |= work.resources;
  return true;
}

static bool validateStage(const StaticScheduleStage &stage,
                          StaticScheduleResourceMask &referencedResources) {
  if (stage.independentBranches.empty())
    return false;
  for (const StaticScheduleBranch &branch : stage.independentBranches) {
    if (branch.dependentWork.empty() ||
        !llvm::all_of(branch.dependentWork,
                      [&](const StaticScheduleWork &work) {
                        return validateWork(work, referencedResources);
                      }))
      return false;
  }
  return true;
}

static bool validateStages(llvm::ArrayRef<StaticScheduleStage> stages,
                           StaticScheduleResourceMask &referencedResources) {
  return llvm::all_of(stages, [&](const StaticScheduleStage &stage) {
    return validateStage(stage, referencedResources);
  });
}

template <typename Callback>
static void forEachStageWork(const StaticScheduleStage &stage,
                             Callback callback) {
  for (const StaticScheduleBranch &branch : stage.independentBranches)
    for (const StaticScheduleWork &work : branch.dependentWork)
      callback(work);
}

template <typename Callback>
static void forEachPlanWork(const StaticSchedulePlan &plan, Callback callback) {
  for (const StaticScheduleStep &step : plan.getSteps()) {
    switch (step.kind) {
    case StaticScheduleStep::Kind::Stage:
      forEachStageWork(step.stage, callback);
      break;
    case StaticScheduleStep::Kind::BufferedPipeline:
      for (const StaticScheduleStage &stage : step.pipeline.prologue)
        forEachStageWork(stage, callback);
      forEachStageWork(step.pipeline.steady, callback);
      for (const StaticScheduleStage &stage : step.pipeline.epilogue)
        forEachStageWork(stage, callback);
      break;
    }
  }
}

} // namespace

StaticScheduleStep StaticScheduleStep::forStage(StaticScheduleStage stage) {
  StaticScheduleStep step;
  step.kind = Kind::Stage;
  step.stage = std::move(stage);
  return step;
}

StaticScheduleStep
StaticScheduleStep::forBufferedPipeline(StaticBufferedPipeline pipeline) {
  StaticScheduleStep step;
  step.kind = Kind::BufferedPipeline;
  step.pipeline = std::move(pipeline);
  return step;
}

std::optional<StaticSchedulePlan>
StaticSchedulePlan::create(const WholeCardInstructionProgramCost &controlCost,
                           llvm::ArrayRef<StaticScheduleStep> steps) {
  if (steps.empty())
    return std::nullopt;

  StaticScheduleResourceMask referencedResources = 0;
  for (const StaticScheduleStep &step : steps) {
    switch (step.kind) {
    case StaticScheduleStep::Kind::Stage:
      if (!validateStage(step.stage, referencedResources))
        return std::nullopt;
      break;
    case StaticScheduleStep::Kind::BufferedPipeline:
      if (step.pipeline.steadyWaveCount == 0 ||
          !validateStage(step.pipeline.steady, referencedResources) ||
          !validateStages(step.pipeline.prologue, referencedResources) ||
          !validateStages(step.pipeline.epilogue, referencedResources))
        return std::nullopt;
      break;
    }
  }
  if (referencedResources != allStaticScheduleResources())
    return std::nullopt;

  return StaticSchedulePlan(
      controlCost,
      llvm::SmallVector<StaticScheduleStep, 8>(steps.begin(), steps.end()));
}

StaticSchedulePlan StaticSchedulePlan::getConservative(
    const WholeCardInstructionProgramCost &cost) {
  llvm::SmallVector<StaticScheduleStep, 8> steps;
  for (StaticScheduleResource resource : kScheduleResources) {
    StaticScheduleBranch branch;
    branch.dependentWork.push_back(
        {&cost, staticScheduleResourceMask(resource)});
    StaticScheduleStage stage;
    stage.independentBranches.push_back(std::move(branch));
    steps.push_back(StaticScheduleStep::forStage(std::move(stage)));
  }
  return StaticSchedulePlan(cost, std::move(steps));
}

llvm::SmallVector<WholeCardResourceDurationEstimate, 16>
estimateStaticSchedulePlanDurations(
    llvm::ArrayRef<const StaticSchedulePlan *> plans,
    const TargetScheduleCostPolicy &policy) {
  StaticDurationTermMask terms = getParameterEnabledTerms(policy);
  for (const StaticSchedulePlan *plan : plans) {
    if (!plan)
      return {};
    disableUnavailableControlTerms(terms, plan->getControlCost());
    forEachPlanWork(*plan, [&](const StaticScheduleWork &work) {
      disableUnavailableResourceTerms(terms, *work.cost, work.resources);
    });
  }

  ResourceEstimateMap resourceEstimates;
  auto addEstimate = [&](const WholeCardInstructionProgramCost &cost) {
    if (resourceEstimates.count(&cost) == 0)
      resourceEstimates.try_emplace(
          &cost, estimateResourceDurations(cost, policy, terms));
  };
  for (const StaticSchedulePlan *plan : plans) {
    addEstimate(plan->getControlCost());
    forEachPlanWork(*plan, [&](const StaticScheduleWork &work) {
      addEstimate(*work.cost);
    });
  }

  llvm::SmallVector<WholeCardResourceDurationEstimate, 16> results;
  results.reserve(plans.size());
  for (const StaticSchedulePlan *plan : plans)
    results.push_back(estimatePlan(*plan, terms, resourceEstimates));
  return results;
}

} // namespace wafer::analysis
