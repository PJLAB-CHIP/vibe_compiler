//===- CostModelTest.cpp - Instruction program performance model -------===//

#include "Wafer/Analysis/Instr/CostModel.h"

#include "gtest/gtest.h"

#include <array>
#include <limits>
#include <utility>

namespace {

using namespace wafer::analysis;

// Unit rates form a bounded arithmetic oracle. Real-scale current-IR and
// product coverage lives in ScheduleCostAnalysis and compiler/no-card tests.
SearchCostPolicy unitCostPolicy() {
  SearchCostPolicy policy;
  policy.ddrNominalBytesPerSecond = UINT64_C(1000000000000);
  policy.directionalNoCBytesPerSecond = UINT64_C(1000000000000);
  policy.dteEndpointBytesPerSecondEstimate = UINT64_C(1000000000000);
  policy.dteMessageStartupPicosecondsEstimate = 1;
  policy.dteFirstMessagePicosecondsEstimate = 1;
  policy.noCHopPicosecondsEstimate = 1;
  policy.instructionFixedPicosecondsEstimate = 1;
  policy.dteWaitedEventPicosecondsEstimate = 1;
  policy.nccJoinPicosecondsEstimate = 1;
  policy.nccParticipantWaitPicosecondsEstimate = 1;
  policy.f16Bf16NPULogicalOpsPerSecondPerTile = UINT64_C(1000000000000);
  policy.f16Bf16VectorLogicalOpsPerSecondPerTile = UINT64_C(1000000000000);
  policy.f32VectorLogicalOpsPerSecondPerTile = UINT64_C(1000000000000);
  policy.spmExplicitMovementBytesPerSecondPerTileEstimate =
      UINT64_C(1000000000000);
  return policy;
}

TEST(CostModelTest, ExplicitCohortDerivesResourceTermsUnknownAndOverflow) {
  std::string failureReason;
  SearchCostPolicy policy = unitCostPolicy();
  auto cohort = SearchCostCohort::create(policy, &failureReason);
  ASSERT_TRUE(mlir::succeeded(cohort)) << failureReason;
  policy.ddrNominalBytesPerSecond = 0;
  EXPECT_TRUE(mlir::failed(SearchCostCohort::create(policy, &failureReason)));
  policy = unitCostPolicy();
  policy.profileIdentity = 0;
  EXPECT_TRUE(mlir::failed(SearchCostCohort::create(policy, &failureReason)));

  InstructionProgramAggregateCost cost;
  cost.aggregateInstructionCount.value = 1;
  cost.aggregateDDRReadBytes.value = 2;
  cost.aggregateDDRWriteBytes.value = 3;
  cost.aggregateNoC.staticIssueSiteCount.value = 1;
  cost.modeledNoCRoute.peakDirectedLinkByteDemand.value = 4;
  cost.maximumTileNoCTransmitBytes.value = 10;
  cost.maximumTileNoCTransmitMessageCount.value = 2;
  cost.minimumHopMessageDemand.value = 3;
  SearchObjective known = deriveSearchObjective(cost, *cohort);
  const auto *knownValue = std::get_if<KnownSearchObjective>(&known);
  ASSERT_NE(knownValue, nullptr);
  EXPECT_EQ(knownValue->durations.instructionControlPicoseconds, 1u);
  EXPECT_EQ(knownValue->durations.ddrPicoseconds, 5u);
  EXPECT_EQ(knownValue->durations.nocPicoseconds, 4u);
  EXPECT_EQ(knownValue->durations.dteEndpointPicoseconds, 10u);
  EXPECT_EQ(knownValue->durations.dteStartupPicoseconds, 2u);
  EXPECT_EQ(knownValue->durations.nocHopPicoseconds, 3u);
  EXPECT_TRUE(std::holds_alternative<UnknownSearchObjective>(
      deriveSearchObjective(cost, std::nullopt)));
  cost.minimumHopMessageDemand.knowledge = ScheduleCostKnowledge::Unavailable;
  EXPECT_EQ(
      std::get<UnknownSearchObjective>(deriveSearchObjective(cost, *cohort))
          .reason,
      SearchObjectiveUnknownReason::MetricUnavailable);
  cost.minimumHopMessageDemand.knowledge = ScheduleCostKnowledge::Known;
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Unavailable;
  EXPECT_EQ(
      std::get<UnknownSearchObjective>(deriveSearchObjective(cost, *cohort))
          .reason,
      SearchObjectiveUnknownReason::MetricUnavailable);
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Known;
  cost.aggregateInstructionCount.value = std::numeric_limits<uint64_t>::max();
  SearchCostPolicy overflowPolicy = unitCostPolicy();
  overflowPolicy.instructionFixedPicosecondsEstimate = 2;
  auto overflowCohort = *SearchCostCohort::create(overflowPolicy);
  EXPECT_EQ(std::get<UnknownSearchObjective>(
                deriveSearchObjective(cost, overflowCohort))
                .reason,
            SearchObjectiveUnknownReason::ArithmeticOverflow);
}

TEST(CostModelTest, DTEStartupSeparatesFirstMessageAndChecksCohortAndOverflow) {
  // Bounded arithmetic oracle; actual multi-Tile IR is covered by the
  // communication candidate tests and the guarded board timing probe.
  SearchCostPolicy policy;
  auto cohort = *SearchCostCohort::create(policy);
  InstructionProgramAggregateCost cost;
  for (auto [count, expected] : std::array<std::pair<uint64_t, uint64_t>, 4>{
           {{0, 0}, {1, 13'000'000}, {15, 34'000'000}, {32, 59'500'000}}}) {
    SCOPED_TRACE(count);
    cost.maximumTileNoCTransmitMessageCount.value = count;
    SearchObjective objective = deriveSearchObjective(cost, cohort);
    const auto *known = std::get_if<KnownSearchObjective>(&objective);
    ASSERT_NE(known, nullptr);
    EXPECT_EQ(known->durations.dteStartupPicoseconds, expected);
  }
  SearchObjective reference = deriveSearchObjective(cost, cohort);
  ++policy.dteFirstMessagePicosecondsEstimate;
  auto otherCohort = *SearchCostCohort::create(policy);
  EXPECT_EQ(compareSearchObjectives(reference,
                                    deriveSearchObjective(cost, otherCohort)),
            SearchObjectiveComparison::Incomparable);
  policy.dteFirstMessagePicosecondsEstimate = 0;
  EXPECT_TRUE(mlir::failed(SearchCostCohort::create(policy)));

  policy = unitCostPolicy();
  policy.dteFirstMessagePicosecondsEstimate =
      std::numeric_limits<uint64_t>::max();
  cost.maximumTileNoCTransmitMessageCount.value = 2;
  EXPECT_EQ(std::get<UnknownSearchObjective>(
                deriveSearchObjective(cost, *SearchCostCohort::create(policy)))
                .reason,
            SearchObjectiveUnknownReason::ArithmeticOverflow);
  policy = unitCostPolicy();
  policy.dteMessageStartupPicosecondsEstimate = 2;
  cost.maximumTileNoCTransmitMessageCount.value =
      std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(std::get<UnknownSearchObjective>(
                deriveSearchObjective(cost, *SearchCostCohort::create(policy)))
                .reason,
            SearchObjectiveUnknownReason::ArithmeticOverflow);
}
TEST(CostModelTest, NCCControlCountsCallsAndParticipantsOnTheSameTile) {
  SearchCostPolicy policy;
  auto cohort = *SearchCostCohort::create(policy);
  InstructionProgramAggregateCost cost;
  for (uint64_t participants : {2, 3}) {
    cost.aggregateNCCJoinCount.value = 1;
    cost.aggregateNCCParticipantWaitCount.value = participants;
    SearchObjective combined = deriveSearchObjective(cost, cohort);
    EXPECT_EQ(std::get<KnownSearchObjective>(combined)
                  .durations.nccWaitControlPicoseconds,
              participants == 2 ? 230'000u : 275'000u);
    cost.aggregateNCCJoinCount.value = participants;
    SearchObjective split = deriveSearchObjective(cost, cohort);
    EXPECT_EQ(std::get<KnownSearchObjective>(split)
                  .durations.nccWaitControlPicoseconds,
              participants == 2 ? 370'000u : 555'000u);
    EXPECT_EQ(compareSearchObjectives(combined, split),
              SearchObjectiveComparison::Better);
  }
  cost.tileCosts.resize(2);
  cost.tileCosts[0].nccJoinCount.value = 2;
  cost.tileCosts[0].nccParticipantWaitCount.value = 2;
  cost.tileCosts[1].nccJoinCount.value = 1;
  cost.tileCosts[1].nccParticipantWaitCount.value = 3;
  SearchObjective known = deriveSearchObjective(cost, cohort);
  EXPECT_EQ(
      std::get<KnownSearchObjective>(known).durations.nccWaitControlPicoseconds,
      370'000u);
  ++policy.nccJoinPicosecondsEstimate;
  EXPECT_EQ(compareSearchObjectives(
                known,
                deriveSearchObjective(cost, *SearchCostCohort::create(policy))),
            SearchObjectiveComparison::Incomparable);
  cost.tileCosts[0].nccJoinCount.knowledge = ScheduleCostKnowledge::Unavailable;
  EXPECT_EQ(
      std::get<UnknownSearchObjective>(deriveSearchObjective(cost, cohort))
          .reason,
      SearchObjectiveUnknownReason::MetricUnavailable);
  cost.tileCosts[0].nccJoinCount.knowledge = ScheduleCostKnowledge::Known;
  cost.tileCosts[0].nccJoinCount.value = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(
      std::get<UnknownSearchObjective>(deriveSearchObjective(cost, cohort))
          .reason,
      SearchObjectiveUnknownReason::ArithmeticOverflow);
}
TEST(CostModelTest, SameInstructionCountKeepsNEAndVectorTradeoffIncomparable) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  auto makeCost = [](uint64_t ne, uint64_t vector) {
    InstructionProgramAggregateCost cost;
    cost.aggregateInstructionCount.value = 10;
    cost.aggregateCompute.npuF16Bf16LogicalOps.value = ne;
    cost.aggregateCompute.vectorF16Bf16LogicalOps.value = vector;
    return cost;
  };

  SearchObjective neHeavy = deriveSearchObjective(makeCost(1024, 0), cohort);
  SearchObjective vectorHeavy =
      deriveSearchObjective(makeCost(0, 1024), cohort);
  EXPECT_EQ(compareSearchObjectives(neHeavy, vectorHeavy),
            SearchObjectiveComparison::Incomparable);

  SearchObjective lessOfBoth =
      deriveSearchObjective(makeCost(1024, 1024), cohort);
  SearchObjective moreOfBoth =
      deriveSearchObjective(makeCost(1025, 1031), cohort);
  EXPECT_EQ(compareSearchObjectives(lessOfBoth, moreOfBoth),
            SearchObjectiveComparison::Better);
  const auto *known = std::get_if<KnownSearchObjective>(&lessOfBoth);
  ASSERT_NE(known, nullptr);
  EXPECT_EQ(known->durations.neF16Bf16Picoseconds, 1024u);
  EXPECT_EQ(known->durations.vectorF16Bf16Picoseconds, 1024u);
  EXPECT_EQ(known->durations.instructionControlPicoseconds, 10u);
}

} // namespace
