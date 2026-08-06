//===- NoCProfitabilityAnalysisTest.cpp ---------------------------------===//

#include "Wafer/Analysis/NoCProfitabilityAnalysis.h"

#include "gtest/gtest.h"

#include <limits>

namespace {

using wafer::analysis::InstructionProgramCost;
using wafer::analysis::NoCTradeoffDecision;
using wafer::analysis::NoCTradeoffReason;
using wafer::analysis::ScheduleCostKnowledge;
using wafer::analysis::ScheduleCostMetric;
using wafer::analysis::ScheduleCostReason;
using wafer::analysis::StaticCrossResourceSchedule;
using wafer::analysis::StaticDurationAssumption;
using wafer::analysis::staticDurationAssumptionMask;
using wafer::analysis::TargetScheduleCostPolicy;
using wafer::analysis::WholeCardInstructionProgramCost;

static WholeCardInstructionProgramCost
makeCost(uint64_t ddrReadBytes, uint64_t ddrWriteBytes,
         uint64_t nocTransmitBytes = 0, uint64_t nocReceiveBytes = 0,
         uint64_t transmitMessages = 0, uint64_t receiveMessages = 0,
         uint64_t minimumHopLinkBytes = 0,
         uint64_t maximumRankTransmitBytes = 0,
         uint64_t maximumRankReceiveBytes = 0) {
  WholeCardInstructionProgramCost cost;
  cost.rankCosts.push_back(InstructionProgramCost{});
  cost.aggregateDDRReadBytes.value = ddrReadBytes;
  cost.aggregateDDRWriteBytes.value = ddrWriteBytes;
  cost.aggregateNoC.staticIssueSiteCount.value =
      static_cast<uint64_t>(transmitMessages != 0) +
      static_cast<uint64_t>(receiveMessages != 0);
  cost.aggregateNoC.aggregateTransmitBytes.value = nocTransmitBytes;
  cost.aggregateNoC.aggregateReceiveBytes.value = nocReceiveBytes;
  cost.aggregateNoC.transmitMessageCount.value = transmitMessages;
  cost.aggregateNoC.receiveMessageCount.value = receiveMessages;
  cost.minimumHopLinkByteDemand.value = minimumHopLinkBytes;
  // Synthetic fixtures use one-hop messages unless a test overwrites the
  // exact message-hop work explicitly.
  cost.minimumHopMessageDemand.value = transmitMessages;
  cost.modeledNoCRoute.peakDirectedLinkByteDemand.value = minimumHopLinkBytes;
  cost.maximumNoCHopCount.value = minimumHopLinkBytes == 0 ? 0 : 1;
  cost.maximumRankNoCTransmitBytes.value = maximumRankTransmitBytes;
  cost.maximumRankNoCReceiveBytes.value = maximumRankReceiveBytes;
  cost.maximumRankNoCTransmitMessageCount.value = transmitMessages;
  cost.maximumRankNoCReceiveMessageCount.value = receiveMessages;
  return cost;
}

static TargetScheduleCostPolicy defaultPolicy() {
  return wafer::analysis::getTargetScheduleCostPolicy();
}

TEST(NoCProfitabilityAnalysisTest, UsesOneSharedWholeCardDDRRate) {
  WholeCardInstructionProgramCost cost =
      makeCost(/*ddrReadBytes=*/16 * 4096, /*ddrWriteBytes=*/0);
  auto estimate =
      wafer::analysis::estimateWholeCardResourceDuration(cost, defaultPolicy());
  ASSERT_TRUE(estimate.ddr.nominalPicoseconds.isKnown());
  // ceil(65536 bytes * 1e12 / 150e9), once for the complete 16-rank domain.
  EXPECT_EQ(estimate.ddr.nominalPicoseconds.value, 436'907u);
  ASSERT_TRUE(estimate.ddr.lowerBoundPicoseconds.isKnown());
  EXPECT_EQ(estimate.ddr.lowerBoundPicoseconds.value, 327'680u);
  EXPECT_EQ(estimate.ddr.upperBoundPicoseconds.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(estimate.ddr.upperBoundPicoseconds.reason,
            ScheduleCostReason::MissingPerformanceCalibration);
}

TEST(NoCProfitabilityAnalysisTest,
     DurationKnowledgePreservesOverflowOverUnknown) {
  WholeCardInstructionProgramCost cost = makeCost(0, 0);
  cost.aggregateDDRReadBytes = {0, ScheduleCostKnowledge::Unknown,
                                ScheduleCostReason::DynamicLoopTripCount};
  cost.rankCosts.front().compute.npuF16Bf16LogicalOps.value =
      std::numeric_limits<uint64_t>::max();
  TargetScheduleCostPolicy policy = defaultPolicy();
  policy.f16Bf16NPULogicalOpsPerSecondPerTile = 1;

  auto estimate =
      wafer::analysis::estimateWholeCardResourceDuration(cost, policy);
  EXPECT_EQ(estimate.ddr.nominalPicoseconds.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(estimate.compute.nominalPicoseconds.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(estimate.makespan.nominalPicoseconds.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(estimate.makespan.nominalPicoseconds.reason,
            ScheduleCostReason::ArithmeticOverflow);
}

TEST(NoCProfitabilityAnalysisTest,
     NominalMakespanOnlyOverlapsResourcesForPipelinedSchedule) {
  WholeCardInstructionProgramCost cost =
      makeCost(/*ddrReadBytes=*/150'000, /*ddrWriteBytes=*/0,
               /*nocTransmitBytes=*/128'000,
               /*nocReceiveBytes=*/128'000,
               /*transmitMessages=*/1, /*receiveMessages=*/1,
               /*minimumHopLinkBytes=*/128'000,
               /*maximumRankTransmitBytes=*/128'000,
               /*maximumRankReceiveBytes=*/128'000);
  auto sequential = wafer::analysis::estimateWholeCardResourceDuration(
      cost, defaultPolicy(), StaticCrossResourceSchedule::SequentialPhases);
  auto pipelined = wafer::analysis::estimateWholeCardResourceDuration(
      cost, defaultPolicy(), StaticCrossResourceSchedule::PipelinedSteadyState);
  ASSERT_TRUE(sequential.ddr.nominalPicoseconds.isKnown());
  ASSERT_TRUE(sequential.noc.nominalPicoseconds.isKnown());
  ASSERT_TRUE(sequential.makespan.nominalPicoseconds.isKnown());
  ASSERT_TRUE(pipelined.makespan.nominalPicoseconds.isKnown());
  EXPECT_EQ(sequential.ddr.nominalPicoseconds.value, 1'000'000u);
  // One 128 KB endpoint stream costs 1 us of serialization plus the
  // versioned 10 us per-message and one-hop point priors.
  EXPECT_EQ(sequential.noc.nominalPicoseconds.value, 11'001'000u);
  EXPECT_EQ(sequential.makespan.nominalPicoseconds.value, 12'001'000u);
  EXPECT_EQ(pipelined.makespan.nominalPicoseconds.value, 11'001'000u);
  EXPECT_EQ(sequential.makespan.upperBoundPicoseconds.knowledge,
            ScheduleCostKnowledge::Unknown);
}

TEST(NoCProfitabilityAnalysisTest,
     SPMMovementHasNoHistoricalFlatDurationInMakespan) {
  WholeCardInstructionProgramCost cost = makeCost(0, 0);
  cost.aggregateSPMMovementBytes.value = 4096;
  cost.maximumRankSPMMovementBytes.value = 4096;

  auto estimate =
      wafer::analysis::estimateWholeCardResourceDuration(cost, defaultPolicy());
  EXPECT_EQ(estimate.spm.nominalPicoseconds.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(estimate.spm.nominalPicoseconds.reason,
            ScheduleCostReason::MissingPerformanceCalibration);
  ASSERT_TRUE(estimate.makespan.nominalPicoseconds.isKnown());
  EXPECT_EQ(estimate.makespan.nominalPicoseconds.value, 0u);
}

TEST(NoCProfitabilityAnalysisTest,
     KnownWorkKeepsModeledRouteOutOfTheNoCLowerBound) {
  WholeCardInstructionProgramCost cost =
      makeCost(/*ddrReadBytes=*/0, /*ddrWriteBytes=*/0,
               /*nocTransmitBytes=*/65'536,
               /*nocReceiveBytes=*/65'536,
               /*transmitMessages=*/1, /*receiveMessages=*/1,
               /*minimumHopLinkBytes=*/65'536,
               /*maximumRankTransmitBytes=*/65'536,
               /*maximumRankReceiveBytes=*/65'536);
  cost.idealizedMinimumPeakLinkByteDemand.value = 4096;
  cost.maximumNoCHopCount.value = 3;

  auto estimate =
      wafer::analysis::estimateWholeCardResourceDuration(cost, defaultPolicy());
  ASSERT_TRUE(estimate.noc.lowerBoundPicoseconds.isKnown());
  ASSERT_TRUE(estimate.noc.nominalPicoseconds.isKnown());
  // The route-independent lower bound uses only the ideal peak-link floor.
  EXPECT_EQ(estimate.noc.lowerBoundPicoseconds.value, 32'000u);
  EXPECT_GT(estimate.noc.nominalPicoseconds.value,
            estimate.noc.lowerBoundPicoseconds.value);

  const auto routeAssumption = staticDurationAssumptionMask(
      StaticDurationAssumption::ModeledShortestPathRoute);
  EXPECT_EQ(estimate.noc.nominalAssumptions & routeAssumption, routeAssumption);
  EXPECT_EQ(estimate.makespan.nominalAssumptions & routeAssumption,
            routeAssumption);
}

TEST(NoCProfitabilityAnalysisTest,
     DefaultModelPriorRejectsFourKiBFanoutStartupOverhead) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/16 * 4096,
               /*ddrWriteBytes=*/16 * 4096);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/4096, /*ddrWriteBytes=*/16 * 4096,
               /*nocTransmitBytes=*/15 * 4096,
               /*nocReceiveBytes=*/15 * 4096,
               /*transmitMessages=*/15, /*receiveMessages=*/15,
               /*minimumHopLinkBytes=*/48 * 4096,
               /*maximumRankTransmitBytes=*/15 * 4096,
               /*maximumRankReceiveBytes=*/4096);

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::Reject);
  EXPECT_EQ(result.reason, NoCTradeoffReason::NominalMakespanNotImproved);
  ASSERT_TRUE(result.candidate.makespan.nominalPicoseconds.isKnown());
  ASSERT_TRUE(result.baseline.makespan.nominalPicoseconds.isKnown());
  EXPECT_GE(result.candidate.makespan.nominalPicoseconds.value,
            result.baseline.makespan.nominalPicoseconds.value);
}

TEST(NoCProfitabilityAnalysisTest, RejectsNominallyWorseNoCTradeoff) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/16 * 4096,
               /*ddrWriteBytes=*/16 * 4096);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/4096, /*ddrWriteBytes=*/16 * 4096,
               /*nocTransmitBytes=*/256 * 1024,
               /*nocReceiveBytes=*/256 * 1024,
               /*transmitMessages=*/15, /*receiveMessages=*/15,
               /*minimumHopLinkBytes=*/512 * 1024,
               /*maximumRankTransmitBytes=*/256 * 1024,
               /*maximumRankReceiveBytes=*/64 * 1024);

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::Reject);
  EXPECT_EQ(result.reason, NoCTradeoffReason::NominalMakespanNotImproved);
}

TEST(NoCProfitabilityAnalysisTest,
     LargeDDRBoundDistributedWorkClearsEstimatedModelMargin) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/1'000'000'000,
               /*ddrWriteBytes=*/0);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/100'000'000, /*ddrWriteBytes=*/0,
               /*nocTransmitBytes=*/100'000'000,
               /*nocReceiveBytes=*/100'000'000,
               /*transmitMessages=*/128, /*receiveMessages=*/128,
               /*minimumHopLinkBytes=*/100'000'000,
               /*maximumRankTransmitBytes=*/6'250'000,
               /*maximumRankReceiveBytes=*/6'250'000);

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::EstimatedBenefit);
  EXPECT_EQ(result.reason, NoCTradeoffReason::EstimatedModelClearsMargin);
  ASSERT_TRUE(result.candidate.makespan.nominalPicoseconds.isKnown());
  ASSERT_TRUE(result.baseline.makespan.nominalPicoseconds.isKnown());
  EXPECT_LT(result.candidate.makespan.nominalPicoseconds.value,
            result.baseline.makespan.nominalPicoseconds.value);
  EXPECT_EQ(result.candidate.noc.upperBoundPicoseconds.knowledge,
            ScheduleCostKnowledge::Unknown);
}

TEST(NoCProfitabilityAnalysisTest,
     NominalImprovementBelowProductionMarginIsRejected) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/1'000'000'000, /*ddrWriteBytes=*/0);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/850'000'000, /*ddrWriteBytes=*/0,
               /*nocTransmitBytes=*/1, /*nocReceiveBytes=*/1,
               /*transmitMessages=*/1, /*receiveMessages=*/1,
               /*minimumHopLinkBytes=*/1,
               /*maximumRankTransmitBytes=*/1,
               /*maximumRankReceiveBytes=*/1);

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  ASSERT_TRUE(result.candidate.makespan.nominalPicoseconds.isKnown());
  ASSERT_TRUE(result.baseline.makespan.nominalPicoseconds.isKnown());
  EXPECT_LT(result.candidate.makespan.nominalPicoseconds.value,
            result.baseline.makespan.nominalPicoseconds.value);
  EXPECT_EQ(result.decision, NoCTradeoffDecision::Reject);
  EXPECT_EQ(result.reason, NoCTradeoffReason::InsufficientBenefitMargin);
}

TEST(NoCProfitabilityAnalysisTest,
     LooseConservativeBoundsDoNotMislabelAnEstimatedWinnerAsProven) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/1'000'000'000, /*ddrWriteBytes=*/0);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/100'000'000, /*ddrWriteBytes=*/0,
               /*nocTransmitBytes=*/100'000'000,
               /*nocReceiveBytes=*/100'000'000,
               /*transmitMessages=*/1, /*receiveMessages=*/1,
               /*minimumHopLinkBytes=*/100'000'000,
               /*maximumRankTransmitBytes=*/100'000'000,
               /*maximumRankReceiveBytes=*/0);
  TargetScheduleCostPolicy policy = defaultPolicy();
  policy.cardDDRSustainedBytesPerSecondLowerBound = 1'000'000'000ULL;
  policy.directionalNoCSustainedBytesPerSecondLowerBound = 1'000'000'000ULL;
  policy.dteEndpointBytesPerSecondLowerBound = 1'000'000'000ULL;
  policy.dteMessageStartupPicosecondsUpperBound = 10'000'000ULL;
  policy.noCHopPicosecondsUpperBound = 1'000ULL;
  policy.noCRouteDilationUpperBound = 4;

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, policy);
  ASSERT_TRUE(result.candidate.makespan.upperBoundPicoseconds.isKnown());
  ASSERT_TRUE(result.baseline.makespan.lowerBoundPicoseconds.isKnown());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::EstimatedBenefit);
  EXPECT_EQ(result.reason, NoCTradeoffReason::EstimatedModelClearsMargin);
}

TEST(NoCProfitabilityAnalysisTest,
     ConservativeNoCUpperChargesEveryDilatedMessageHop) {
  WholeCardInstructionProgramCost cost =
      makeCost(/*ddrReadBytes=*/0, /*ddrWriteBytes=*/0,
               /*nocTransmitBytes=*/16, /*nocReceiveBytes=*/16,
               /*transmitMessages=*/4, /*receiveMessages=*/4,
               /*minimumHopLinkBytes=*/48,
               /*maximumRankTransmitBytes=*/16,
               /*maximumRankReceiveBytes=*/16);
  cost.minimumHopMessageDemand.value = 12;
  cost.maximumNoCHopCount.value = 3;
  TargetScheduleCostPolicy zeroHopPolicy = defaultPolicy();
  zeroHopPolicy.directionalNoCSustainedBytesPerSecondLowerBound =
      128'000'000'000ULL;
  zeroHopPolicy.dteEndpointBytesPerSecondLowerBound = 128'000'000'000ULL;
  zeroHopPolicy.dteMessageStartupPicosecondsUpperBound = 0;
  zeroHopPolicy.noCHopPicosecondsUpperBound = 0;
  zeroHopPolicy.noCRouteDilationUpperBound = 2;
  TargetScheduleCostPolicy nonzeroHopPolicy = zeroHopPolicy;
  nonzeroHopPolicy.noCHopPicosecondsUpperBound = 5;

  auto zeroHop =
      wafer::analysis::estimateWholeCardResourceDuration(cost, zeroHopPolicy);
  auto nonzeroHop = wafer::analysis::estimateWholeCardResourceDuration(
      cost, nonzeroHopPolicy);
  ASSERT_TRUE(zeroHop.noc.upperBoundPicoseconds.isKnown());
  ASSERT_TRUE(nonzeroHop.noc.upperBoundPicoseconds.isKnown());
  // 12 exact message-hops * dilation 2 * 5 ps/hop.
  EXPECT_EQ(nonzeroHop.noc.upperBoundPicoseconds.value -
                zeroHop.noc.upperBoundPicoseconds.value,
            120u);
}

TEST(NoCProfitabilityAnalysisTest,
     ExistingCollectiveUsesTheSameEstimatedDDRAdmission) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/1'000'000'000,
               /*ddrWriteBytes=*/1'000'000'000,
               /*nocTransmitBytes=*/100'000'000,
               /*nocReceiveBytes=*/100'000'000,
               /*transmitMessages=*/128, /*receiveMessages=*/128,
               /*minimumHopLinkBytes=*/100'000'000,
               /*maximumRankTransmitBytes=*/6'250'000,
               /*maximumRankReceiveBytes=*/6'250'000);
  WholeCardInstructionProgramCost candidate = baseline;
  candidate.aggregateDDRReadBytes.value = 500'000'000;
  candidate.aggregateDDRWriteBytes.value = 500'000'000;

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::EstimatedBenefit);
  EXPECT_EQ(result.reason, NoCTradeoffReason::EstimatedModelClearsMargin);
}

TEST(NoCProfitabilityAnalysisTest,
     CompleteSyntheticBoundsCanProveAConservativeBenefit) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/1'000'000'000,
               /*ddrWriteBytes=*/0);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/100'000'000, /*ddrWriteBytes=*/0,
               /*nocTransmitBytes=*/100'000'000,
               /*nocReceiveBytes=*/100'000'000,
               /*transmitMessages=*/1, /*receiveMessages=*/1,
               /*minimumHopLinkBytes=*/100'000'000,
               /*maximumRankTransmitBytes=*/100'000'000,
               /*maximumRankReceiveBytes=*/0);
  TargetScheduleCostPolicy policy = defaultPolicy();
  policy.cardDDRSustainedBytesPerSecondLowerBound = 150'000'000'000ULL;
  policy.directionalNoCSustainedBytesPerSecondLowerBound = 128'000'000'000ULL;
  policy.dteEndpointBytesPerSecondLowerBound = 128'000'000'000ULL;
  policy.dteMessageStartupPicosecondsUpperBound = 0;
  policy.noCHopPicosecondsUpperBound = 0;
  policy.noCRouteDilationUpperBound = 1;
  policy.instructionFixedPicosecondsUpperBound = 0;
  policy.dteWaitedEventPicosecondsUpperBound = 0;
  policy.nccParticipantWaitPicosecondsUpperBound = 0;
  candidate.aggregateInstructionCount.value = 2;
  candidate.aggregateNoC.waitedEventCount.value = 1;

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, policy);
  EXPECT_EQ(result.decision, NoCTradeoffDecision::ProvenBenefit);
  EXPECT_EQ(result.reason, NoCTradeoffReason::ConservativeBoundsProveBenefit);
  ASSERT_TRUE(result.candidate.makespan.upperBoundPicoseconds.isKnown());
  ASSERT_TRUE(result.baseline.makespan.lowerBoundPicoseconds.isKnown());
}

TEST(NoCProfitabilityAnalysisTest,
     UnknownNoCWorkFailsClosedInsteadOfBypassingTheGuard) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/65'536, /*ddrWriteBytes=*/65'536);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/4096, /*ddrWriteBytes=*/65'536);
  candidate.aggregateNoC.staticIssueSiteCount = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.aggregateNoC.aggregateTransmitBytes = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnknownResourceBytes};
  candidate.aggregateNoC.transmitMessageCount = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.minimumHopLinkByteDemand = {0, ScheduleCostKnowledge::Unknown,
                                        ScheduleCostReason::UnresolvedNoCRoute};

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::Indeterminate);
  EXPECT_EQ(result.reason, NoCTradeoffReason::UnknownCostFact);
}

TEST(NoCProfitabilityAnalysisTest,
     NoCWorkKnowledgePreservesOverflowOverEarlierUnknown) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/65'536, /*ddrWriteBytes=*/65'536);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/4096, /*ddrWriteBytes=*/65'536,
               /*nocTransmitBytes=*/4096, /*nocReceiveBytes=*/4096,
               /*transmitMessages=*/1, /*receiveMessages=*/1,
               /*minimumHopLinkBytes=*/4096,
               /*maximumRankTransmitBytes=*/4096,
               /*maximumRankReceiveBytes=*/4096);
  candidate.aggregateNoC.staticIssueSiteCount = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.modeledNoCRoute.peakDirectedLinkByteDemand = {
      0, ScheduleCostKnowledge::Overflow,
      ScheduleCostReason::ArithmeticOverflow};

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::Indeterminate);
  EXPECT_EQ(result.reason, NoCTradeoffReason::UnknownCostFact);
  EXPECT_EQ(result.candidate.noc.nominalPicoseconds.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(result.candidate.noc.nominalPicoseconds.reason,
            ScheduleCostReason::ArithmeticOverflow);
}

TEST(NoCProfitabilityAnalysisTest, LeavesPureLocalDDRReductionToTheSelector) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/65'536, /*ddrWriteBytes=*/65'536);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/4096, /*ddrWriteBytes=*/65'536);
  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::NotApplicable);
  EXPECT_EQ(result.reason, NoCTradeoffReason::NoCrossResourceTradeoff);
}

TEST(NoCProfitabilityAnalysisTest,
     DoesNotCaptureNonNoCCandidateWithUnknownDDRWork) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/4096, /*ddrWriteBytes=*/4096);
  WholeCardInstructionProgramCost candidate = baseline;
  candidate.aggregateDDRReadBytes = {0, ScheduleCostKnowledge::Unknown,
                                     ScheduleCostReason::DynamicLoopTripCount};

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::NotApplicable);
  EXPECT_EQ(result.reason, NoCTradeoffReason::NoCrossResourceTradeoff);
}

TEST(NoCProfitabilityAnalysisTest,
     ExactNoDTESiteFactWinsOverUnrelatedUnknownNoCMultiplicity) {
  WholeCardInstructionProgramCost baseline =
      makeCost(/*ddrReadBytes=*/65'536, /*ddrWriteBytes=*/65'536);
  WholeCardInstructionProgramCost candidate =
      makeCost(/*ddrReadBytes=*/4096, /*ddrWriteBytes=*/65'536);
  candidate.aggregateNoC.aggregateTransmitBytes = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.aggregateNoC.transmitMessageCount = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.minimumHopLinkByteDemand = {0, ScheduleCostKnowledge::Unknown,
                                        ScheduleCostReason::UnresolvedNoCRoute};

  auto result = wafer::analysis::analyzeNoCTradeoffProfitability(
      candidate, baseline, defaultPolicy());
  EXPECT_EQ(result.decision, NoCTradeoffDecision::NotApplicable);
  EXPECT_EQ(result.reason, NoCTradeoffReason::NoCrossResourceTradeoff);
}

} // namespace
