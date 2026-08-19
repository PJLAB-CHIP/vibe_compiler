//===- TheoreticalScheduleCostAnalysisTest.cpp -------------------------===//

#include "Wafer/Analysis/ScheduleCost/TheoreticalScheduleCostAnalysis.h"

#include "gtest/gtest.h"

#include <initializer_list>
#include <limits>

namespace {

using wafer::analysis::InstructionProgramCost;
using wafer::analysis::ScheduleCostKnowledge;
using wafer::analysis::ScheduleCostReason;
using wafer::analysis::StaticBufferedPipeline;
using wafer::analysis::StaticDurationAssumption;
using wafer::analysis::staticDurationAssumptionMask;
using wafer::analysis::StaticDurationTerm;
using wafer::analysis::staticDurationTermMask;
using wafer::analysis::StaticScheduleBranch;
using wafer::analysis::StaticSchedulePlan;
using wafer::analysis::StaticScheduleResource;
using wafer::analysis::StaticScheduleResourceMask;
using wafer::analysis::staticScheduleResourceMask;
using wafer::analysis::StaticScheduleStage;
using wafer::analysis::StaticScheduleStep;
using wafer::analysis::StaticScheduleWork;
using wafer::analysis::TargetScheduleCostPolicy;
using wafer::analysis::CardInstructionProgramCost;
using wafer::analysis::ProgramDurationEstimate;

static CardInstructionProgramCost
makeCost(uint64_t ddrReadBytes, uint64_t ddrWriteBytes,
         uint64_t nocTransmitBytes = 0, uint64_t nocReceiveBytes = 0,
         uint64_t transmitMessages = 0, uint64_t receiveMessages = 0,
         uint64_t peakLinkBytes = 0) {
  CardInstructionProgramCost cost;
  cost.tileCosts.push_back(InstructionProgramCost{});
  cost.aggregateDDRReadBytes.value = ddrReadBytes;
  cost.aggregateDDRWriteBytes.value = ddrWriteBytes;
  cost.aggregateNoC.staticIssueSiteCount.value =
      static_cast<uint64_t>(transmitMessages != 0) +
      static_cast<uint64_t>(receiveMessages != 0);
  cost.aggregateNoC.aggregateTransmitBytes.value = nocTransmitBytes;
  cost.aggregateNoC.aggregateReceiveBytes.value = nocReceiveBytes;
  cost.aggregateNoC.transmitMessageCount.value = transmitMessages;
  cost.aggregateNoC.receiveMessageCount.value = receiveMessages;
  cost.modeledNoCRoute.peakDirectedLinkByteDemand.value = peakLinkBytes;
  cost.maximumNoCHopCount.value = peakLinkBytes == 0 ? 0 : 1;
  cost.maximumTileNoCTransmitBytes.value = nocTransmitBytes;
  cost.maximumTileNoCReceiveBytes.value = nocReceiveBytes;
  cost.maximumTileNoCTransmitMessageCount.value = transmitMessages;
  cost.maximumTileNoCReceiveMessageCount.value = receiveMessages;
  return cost;
}

static TargetScheduleCostPolicy defaultPolicy() {
  return TargetScheduleCostPolicy{};
}

static ProgramDurationEstimate
estimateCost(const CardInstructionProgramCost &cost,
             const TargetScheduleCostPolicy &policy) {
  StaticSchedulePlan plan = StaticSchedulePlan::getConservative(cost);
  const StaticSchedulePlan *plans[] = {&plan};
  auto estimates =
      wafer::analysis::estimateStaticSchedulePlanDurations(plans, policy);
  EXPECT_EQ(estimates.size(), 1u);
  if (estimates.empty())
    return {};
  return estimates.front();
}

static ProgramDurationEstimate
estimatePlan(const StaticSchedulePlan &plan,
             const TargetScheduleCostPolicy &policy) {
  const StaticSchedulePlan *plans[] = {&plan};
  auto estimates =
      wafer::analysis::estimateStaticSchedulePlanDurations(plans, policy);
  EXPECT_EQ(estimates.size(), 1u);
  if (estimates.empty())
    return {};
  return estimates.front();
}

static StaticScheduleResourceMask
resources(std::initializer_list<StaticScheduleResource> values) {
  StaticScheduleResourceMask result = 0;
  for (StaticScheduleResource value : values)
    result |= staticScheduleResourceMask(value);
  return result;
}

static StaticScheduleBranch
branch(std::initializer_list<StaticScheduleWork> dependentWork) {
  StaticScheduleBranch result;
  result.dependentWork.append(dependentWork.begin(), dependentWork.end());
  return result;
}

static StaticScheduleWork work(const CardInstructionProgramCost &cost,
                               StaticScheduleResourceMask resourceMask) {
  return {&cost, resourceMask};
}

static StaticScheduleStage
stage(std::initializer_list<StaticScheduleBranch> branches) {
  StaticScheduleStage result;
  result.independentBranches.append(branches.begin(), branches.end());
  return result;
}

static StaticScheduleStep
stageStep(std::initializer_list<StaticScheduleBranch> branches) {
  return StaticScheduleStep::forStage(stage(branches));
}

static StaticScheduleStep
singleWorkStageStep(const CardInstructionProgramCost &cost,
                    StaticScheduleResourceMask resourceMask) {
  return stageStep({branch({work(cost, resourceMask)})});
}

static StaticSchedulePlan
plan(const CardInstructionProgramCost &controlCost,
     std::initializer_list<StaticScheduleStep> steps) {
  auto result = StaticSchedulePlan::create(controlCost, steps);
  EXPECT_TRUE(result.has_value());
  return result ? std::move(*result)
                : StaticSchedulePlan::getConservative(controlCost);
}

TEST(TheoreticalScheduleCostAnalysisTest, UsesOneSharedCardDDRRate) {
  CardInstructionProgramCost cost =
      makeCost(/*ddrReadBytes=*/16 * 4096, /*ddrWriteBytes=*/0);
  auto estimate = estimateCost(cost, defaultPolicy());

  EXPECT_EQ(estimate.ddr.picoseconds, 436'907u);
  EXPECT_EQ(estimate.makespan.picoseconds, 436'907u);
  EXPECT_NE(estimate.enabledTerms &
                staticDurationTermMask(StaticDurationTerm::DDR),
            0u);
  EXPECT_EQ(estimate.ddr.assumptions &
                staticDurationAssumptionMask(
                    StaticDurationAssumption::DDROperatingPoint),
            staticDurationAssumptionMask(
                StaticDurationAssumption::DDROperatingPoint));
}

TEST(TheoreticalScheduleCostAnalysisTest,
     CohortDisablesUnavailableWorkTermForEveryCandidate) {
  CardInstructionProgramCost baseline = makeCost(1'000'000, 0);
  CardInstructionProgramCost candidate = makeCost(100'000, 0);
  candidate.aggregateDDRReadBytes = {0, ScheduleCostKnowledge::Unavailable,
                                     ScheduleCostReason::DynamicLoopTripCount};
  baseline.tileCosts.front().compute.vectorF16Bf16LogicalOps.value = 64'000;
  candidate.tileCosts.front().compute.vectorF16Bf16LogicalOps.value = 32'000;

  StaticSchedulePlan baselinePlan =
      StaticSchedulePlan::getConservative(baseline);
  StaticSchedulePlan candidatePlan =
      StaticSchedulePlan::getConservative(candidate);
  const StaticSchedulePlan *plans[] = {&baselinePlan, &candidatePlan};
  auto estimates = wafer::analysis::estimateStaticSchedulePlanDurations(
      plans, defaultPolicy());

  ASSERT_EQ(estimates.size(), 2u);
  const auto ddrTerm = staticDurationTermMask(StaticDurationTerm::DDR);
  EXPECT_EQ(estimates[0].enabledTerms & ddrTerm, 0u);
  EXPECT_EQ(estimates[1].enabledTerms & ddrTerm, 0u);
  EXPECT_EQ(estimates[0].ddr.picoseconds, 0u);
  EXPECT_EQ(estimates[1].ddr.picoseconds, 0u);
  EXPECT_LT(estimates[1].makespan.picoseconds,
            estimates[0].makespan.picoseconds);
}

TEST(TheoreticalScheduleCostAnalysisTest,
     MissingOptionalPointParameterDisablesOneTermWithoutGuessing) {
  CardInstructionProgramCost cost =
      makeCost(/*ddrReadBytes=*/0, /*ddrWriteBytes=*/0,
               /*nocTransmitBytes=*/128'000,
               /*nocReceiveBytes=*/128'000,
               /*transmitMessages=*/1, /*receiveMessages=*/1,
               /*peakLinkBytes=*/128'000);
  TargetScheduleCostPolicy policy = defaultPolicy();
  policy.dteMessageStartupPicosecondsEstimate = 0;

  auto estimate = estimateCost(cost, policy);

  EXPECT_EQ(estimate.enabledTerms &
                staticDurationTermMask(StaticDurationTerm::NoCMessageStartup),
            0u);
  // Endpoint and link serialization are both 1 us; one 1 ns route-fill prior
  // is added after their shared envelope.
  EXPECT_EQ(estimate.noc.picoseconds, 1'001'000u);
  EXPECT_EQ(estimate.makespan.picoseconds, 1'001'000u);
}

TEST(TheoreticalScheduleCostAnalysisTest, DependentChainSumsResourceStages) {
  CardInstructionProgramCost cost =
      makeCost(/*ddrReadBytes=*/150'000, /*ddrWriteBytes=*/0);
  cost.tileCosts.front().compute.npuF16Bf16LogicalOps.value = 8'000'000;
  cost.aggregateSPMMovementBytes.value = 256'000;
  cost.maximumTileSPMMovementBytes.value = 256'000;

  StaticSchedulePlan chain = StaticSchedulePlan::getConservative(cost);
  auto estimate = estimatePlan(chain, defaultPolicy());

  EXPECT_EQ(estimate.ddr.picoseconds, 1'000'000u);
  EXPECT_EQ(estimate.compute.picoseconds, 1'000'000u);
  EXPECT_EQ(estimate.spm.picoseconds, 1'000'000u);
  EXPECT_EQ(estimate.makespan.picoseconds, 3'000'000u);
}

TEST(TheoreticalScheduleCostAnalysisTest,
     IndependentBranchesInOneStageUseTheLongestBranch) {
  CardInstructionProgramCost shortBranch = makeCost(0, 0);
  CardInstructionProgramCost longBranch = makeCost(0, 0);
  CardInstructionProgramCost zeroWork = makeCost(0, 0);
  shortBranch.tileCosts.front().compute.npuF16Bf16LogicalOps.value = 8'000'000;
  longBranch.tileCosts.front().compute.npuF16Bf16LogicalOps.value = 24'000'000;

  StaticSchedulePlan independent =
      plan(zeroWork,
           {stageStep(
                {branch({work(shortBranch,
                              resources({StaticScheduleResource::Compute}))}),
                 branch({work(longBranch,
                              resources({StaticScheduleResource::Compute}))})}),
            singleWorkStageStep(
                zeroWork, resources({StaticScheduleResource::DDR,
                                     StaticScheduleResource::NoC,
                                     StaticScheduleResource::SPMMovement}))});
  auto estimate = estimatePlan(independent, defaultPolicy());

  EXPECT_EQ(estimate.compute.picoseconds, 4'000'000u);
  EXPECT_EQ(estimate.makespan.picoseconds, 3'000'000u);
}

TEST(TheoreticalScheduleCostAnalysisTest,
     BufferedMovementUsesExplicitPrologueSteadyAndEpilogue) {
  CardInstructionProgramCost cost = makeCost(150'000, 0);
  cost.tileCosts.front().compute.npuF16Bf16LogicalOps.value = 16'000'000;

  llvm::SmallVector<StaticScheduleStep, 16> unbufferedSteps;
  for (unsigned wave = 0; wave < 4; ++wave) {
    unbufferedSteps.push_back(
        singleWorkStageStep(cost, resources({StaticScheduleResource::DDR})));
    unbufferedSteps.push_back(singleWorkStageStep(
        cost, resources({StaticScheduleResource::Compute})));
  }
  unbufferedSteps.push_back(singleWorkStageStep(
      cost, resources({StaticScheduleResource::NoC,
                       StaticScheduleResource::SPMMovement})));
  auto unbufferedPlan = StaticSchedulePlan::create(cost, unbufferedSteps);
  ASSERT_TRUE(unbufferedPlan.has_value());

  StaticBufferedPipeline pipeline;
  pipeline.prologue.push_back(
      stage({branch({work(cost, resources({StaticScheduleResource::DDR}))})}));
  pipeline.steady = stage(
      {branch({work(cost, resources({StaticScheduleResource::DDR}))}),
       branch({work(cost, resources({StaticScheduleResource::Compute}))})});
  pipeline.steadyWaveCount = 3;
  pipeline.epilogue.push_back(stage(
      {branch({work(cost, resources({StaticScheduleResource::Compute}))})}));
  StaticSchedulePlan bufferedPlan =
      plan(cost, {StaticScheduleStep::forBufferedPipeline(std::move(pipeline)),
                  singleWorkStageStep(
                      cost, resources({StaticScheduleResource::NoC,
                                       StaticScheduleResource::SPMMovement}))});

  auto unbuffered = estimatePlan(*unbufferedPlan, defaultPolicy());
  auto buffered = estimatePlan(bufferedPlan, defaultPolicy());

  EXPECT_EQ(unbuffered.ddr.picoseconds, 4'000'000u);
  EXPECT_EQ(unbuffered.compute.picoseconds, 8'000'000u);
  EXPECT_EQ(unbuffered.makespan.picoseconds, 12'000'000u);
  EXPECT_EQ(buffered.makespan.picoseconds, 9'000'000u);
  EXPECT_NE(buffered.makespan.assumptions &
                staticDurationAssumptionMask(
                    StaticDurationAssumption::BufferedPipelinePlan),
            0u);
}

TEST(TheoreticalScheduleCostAnalysisTest,
     InvalidPlanCannotSilentlyDropAResourceOrUseAnEmptySteadyState) {
  CardInstructionProgramCost cost = makeCost(0, 0);
  auto missingResource = StaticSchedulePlan::create(
      cost,
      {singleWorkStageStep(cost, resources({StaticScheduleResource::DDR,
                                            StaticScheduleResource::Compute,
                                            StaticScheduleResource::NoC}))});
  EXPECT_FALSE(missingResource.has_value());

  StaticBufferedPipeline emptySteady;
  emptySteady.steady = stage(
      {branch({work(cost, wafer::analysis::allStaticScheduleResources())})});
  auto zeroWave = StaticSchedulePlan::create(
      cost, {StaticScheduleStep::forBufferedPipeline(std::move(emptySteady))});
  EXPECT_FALSE(zeroWave.has_value());
}

TEST(TheoreticalScheduleCostAnalysisTest,
     DurationAndPipelineArithmeticRemainNumericOnOverflow) {
  CardInstructionProgramCost cost = makeCost(0, 0);
  cost.tileCosts.front().compute.npuF16Bf16LogicalOps.value =
      std::numeric_limits<uint64_t>::max();
  TargetScheduleCostPolicy policy = defaultPolicy();
  policy.f16Bf16NPULogicalOpsPerSecondPerTile = 1;

  StaticBufferedPipeline pipeline;
  pipeline.steady = stage(
      {branch({work(cost, wafer::analysis::allStaticScheduleResources())})});
  pipeline.steadyWaveCount = std::numeric_limits<uint64_t>::max();
  StaticSchedulePlan overflowPlan = plan(
      cost, {StaticScheduleStep::forBufferedPipeline(std::move(pipeline))});
  auto estimate = estimatePlan(overflowPlan, policy);

  EXPECT_EQ(estimate.compute.picoseconds, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(estimate.makespan.picoseconds,
            std::numeric_limits<uint64_t>::max());
}

} // namespace
