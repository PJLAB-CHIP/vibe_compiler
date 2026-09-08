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

void expectCoarse(const SearchObjective &objective, bool saturated = false) {
  const auto *known = std::get_if<KnownSearchObjective>(&objective);
  ASSERT_NE(known, nullptr);
  EXPECT_TRUE(known->usesCoarseEstimate);
  EXPECT_GT(known->estimatedDurationPicoseconds, 0u);
  if (saturated) {
    EXPECT_EQ(known->estimatedDurationPicoseconds,
              std::numeric_limits<uint64_t>::max());
  }
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
  EXPECT_EQ(knownValue->estimatedDurationPicoseconds, 21u);
  EXPECT_TRUE(std::holds_alternative<UnknownSearchObjective>(
      deriveSearchObjective(cost, std::nullopt)));
  cost.minimumHopMessageDemand.knowledge = ScheduleCostKnowledge::Unavailable;
  expectCoarse(deriveSearchObjective(cost, *cohort));
  cost.minimumHopMessageDemand.knowledge = ScheduleCostKnowledge::Known;
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Unavailable;
  expectCoarse(deriveSearchObjective(cost, *cohort));
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Known;
  cost.aggregateInstructionCount.value = std::numeric_limits<uint64_t>::max();
  SearchCostPolicy overflowPolicy = unitCostPolicy();
  overflowPolicy.instructionFixedPicosecondsEstimate = 2;
  auto overflowCohort = *SearchCostCohort::create(overflowPolicy);
  expectCoarse(deriveSearchObjective(cost, overflowCohort), true);
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
  expectCoarse(deriveSearchObjective(cost, *SearchCostCohort::create(policy)),
               true);
  policy = unitCostPolicy();
  policy.dteMessageStartupPicosecondsEstimate = 2;
  cost.maximumTileNoCTransmitMessageCount.value =
      std::numeric_limits<uint64_t>::max();
  expectCoarse(deriveSearchObjective(cost, *SearchCostCohort::create(policy)),
               true);
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
  expectCoarse(deriveSearchObjective(cost, cohort));
  cost.tileCosts[0].nccJoinCount.knowledge = ScheduleCostKnowledge::Known;
  cost.tileCosts[0].nccJoinCount.value = std::numeric_limits<uint64_t>::max();
  expectCoarse(deriveSearchObjective(cost, cohort), true);
}
TEST(CostModelTest, ResourceTradeoffsUseScalarTime) {
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
            SearchObjectiveComparison::Equivalent);

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

TEST(CostModelTest, DDRAndPeerCanBothWinWithTheSameProfile) {
  auto cohort = *SearchCostCohort::create(SearchCostPolicy{});
  InstructionProgramAggregateCost ddr, peer;
  peer.maximumTileNoCTransmitMessageCount.value = 1;
  peer.maximumTileNoCTransmitBytes.value = 2048;
  ddr.aggregateDDRReadBytes.value = 2048;
  EXPECT_EQ(compareSearchObjectives(deriveSearchObjective(ddr, cohort),
                                    deriveSearchObjective(peer, cohort)),
            SearchObjectiveComparison::Better);
  ddr.aggregateDDRReadBytes.value = 16 * 1024 * 1024;
  EXPECT_EQ(compareSearchObjectives(deriveSearchObjective(ddr, cohort),
                                    deriveSearchObjective(peer, cohort)),
            SearchObjectiveComparison::Worse);
}

TEST(CostModelTest, CombineWorkOnEachTileBeforeTakingTheMaximum) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  InstructionProgramAggregateCost split;
  split.tileCosts.resize(2);
  split.tileCosts[0].compute.npuF16Bf16LogicalOps.value = 1024;
  split.tileCosts[1].compute.vectorF32LogicalOps.value = 1031;
  auto joined = split;
  joined.tileCosts[0].compute.vectorF32LogicalOps.value = 1031;
  joined.tileCosts[1].compute.vectorF32LogicalOps.value = 0;
  auto left = deriveSearchObjective(split, cohort);
  auto right = deriveSearchObjective(joined, cohort);
  EXPECT_EQ(std::get<KnownSearchObjective>(left).estimatedDurationPicoseconds,
            1031u);
  EXPECT_EQ(std::get<KnownSearchObjective>(right).estimatedDurationPicoseconds,
            2055u);
  EXPECT_EQ(compareSearchObjectives(left, right),
            SearchObjectiveComparison::Better);
}

TEST(CostModelTest, SharedDDRAndPayloadSerializationAreCountedOnce) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  InstructionProgramAggregateCost cost;
  cost.tileCosts.resize(2);
  for (auto &tile : cost.tileCosts) {
    tile.noc.aggregateTransmitBytes.value = 1024;
    tile.ddrReadBytes.value = 1024;
  }
  cost.aggregateNoC.staticIssueSiteCount.value = 2;
  cost.aggregateDDRReadBytes.value = 2048;
  cost.maximumTileNoCTransmitBytes.value = 1024;
  cost.modeledNoCRoute.peakDirectedLinkByteDemand.value = 1031;
  auto result = deriveSearchObjective(cost, cohort);
  EXPECT_EQ(std::get<KnownSearchObjective>(result).estimatedDurationPicoseconds,
            3079u); // Shared DDR 2048 plus payload bottleneck 1031.
}

TEST(CostModelTest, MissingWorkStillRanksAndStorageCannotVetoFasterTime) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  InstructionProgramAggregateCost cost;
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Unavailable;
  cost.aggregateWork.instructions.upperBound.knowledge =
      ScheduleCostKnowledge::Unavailable;
  cost.aggregateWork.instructions.staticSites.value = 3;
  auto coarse = deriveSearchObjective(cost, cohort);
  expectCoarse(coarse);
  EXPECT_EQ(std::get<KnownSearchObjective>(coarse).estimatedDurationPicoseconds,
            39'000'000u);
  cost.aggregateWork.instructions.upperBound = {5};
  auto larger = deriveSearchObjective(cost, cohort);
  EXPECT_EQ(compareSearchObjectives(coarse, larger),
            SearchObjectiveComparison::Better);

  InstructionProgramAggregateCost fast, slow;
  fast.aggregateInstructionCount.value = 1024;
  fast.maximumTileSPMHighWaterBytes.value = 100000;
  slow.aggregateInstructionCount.value = 1025;
  EXPECT_EQ(compareSearchObjectives(deriveSearchObjective(fast, cohort),
                                    deriveSearchObjective(slow, cohort)),
            SearchObjectiveComparison::Better);
  fast.aggregateCompute.npuOtherLogicalOps.value = 1024;
  expectCoarse(deriveSearchObjective(fast, cohort));
}

} // namespace
