//===- CoordinatedVariantSelectionTest.cpp ------------------------------===//

#include "../../lib/Wafer/Compiler/CoordinatedVariantSelection.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <vector>

namespace {

using wafer::analysis::ScheduleCostKnowledge;
using wafer::analysis::ScheduleCostReason;
using wafer::analysis::WholeCardInstructionProgramCost;
using wafer::compiler::detail::CoordinatedHardwareSelectionEvidence;
using wafer::compiler::detail::CoordinatedVariantCostView;
using wafer::compiler::detail::CoordinatedVariantSelectionPlan;
using wafer::compiler::detail::WholeVariantSelectionMode;

static WholeCardInstructionProgramCost makeCost(uint64_t ddrReadBytes) {
  WholeCardInstructionProgramCost cost;
  cost.aggregateDDRReadBytes.value = ddrReadBytes;
  return cost;
}

static mlir::FailureOr<CoordinatedVariantSelectionPlan>
plan(std::vector<CoordinatedVariantCostView> views,
     WholeVariantSelectionMode mode = WholeVariantSelectionMode::Production) {
  return wafer::compiler::detail::planCoordinatedVariantSelection(views, mode);
}

TEST(CoordinatedVariantSelectionTest,
     SelectsOnlyACompleteTargetModelMarginWinner) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  std::vector<CoordinatedVariantCostView> views = {
      {0, true, &baseline},
      {7, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
  EXPECT_EQ(result->evidence,
            CoordinatedHardwareSelectionEvidence::EstimatedBenefit);
  ASSERT_TRUE(result->selectedDuration.makespan.nominalPicoseconds.isKnown());
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(CoordinatedVariantSelectionTest,
     ReservedModeCommitsTheUniqueBaselineWithoutPromotion) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(1);
  std::vector<CoordinatedVariantCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views, WholeVariantSelectionMode::ReservedBaseline);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
  EXPECT_EQ(result->evidence,
            CoordinatedHardwareSelectionEvidence::ReservedBaseline);
}

TEST(CoordinatedVariantSelectionTest,
     UncalibratedSPMMovementRegressionCannotHideBehindDDRBenefit) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  candidate.aggregateSPMMovementBytes.value = 1;
  candidate.maximumRankSPMMovementBytes.value = 1;
  std::vector<CoordinatedVariantCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(CoordinatedVariantSelectionTest,
     AcceptedSPMHighWaterIsCapacityOnlyNotExecutionCost) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  baseline.maximumRankSPMHighWaterBytes.value = 1024;
  baseline.summedRankSPMHighWaterBytes.value = 1024;
  candidate.maximumRankSPMHighWaterBytes.value = 2048;
  candidate.summedRankSPMHighWaterBytes.value = 2048;
  std::vector<CoordinatedVariantCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
}

TEST(CoordinatedVariantSelectionTest,
     DifferentUnknownDispositionsAreIncomparableAndCannotPromote) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  baseline.maximumRankDataDependencyDepth = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.maximumRankDataDependencyDepth = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::DynamicLoopTripCount};
  std::vector<CoordinatedVariantCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(CoordinatedVariantSelectionTest,
     SameUnknownDispositionDoesNotBecomeZeroOrBlockKnownImprovement) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  baseline.maximumRankDataDependencyDepth = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.maximumRankDataDependencyDepth =
      baseline.maximumRankDataDependencyDepth;
  std::vector<CoordinatedVariantCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
}

TEST(CoordinatedVariantSelectionTest,
     StableOrdinalBreaksOnlyACompleteHardwareTupleTie) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost first = makeCost(100'000'000);
  WholeCardInstructionProgramCost second = first;
  std::vector<CoordinatedVariantCostView> views = {
      {0, true, &baseline},
      {9, false, &first},
      {3, false, &second},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(views[result->selectedIndex].stableSemanticOrdinal, 3);
}

TEST(CoordinatedVariantSelectionTest,
     EqualNominalButCostIncomparableCandidatesFallBackToBaseline) {
  WholeCardInstructionProgramCost baseline = makeCost(2'000'000'000);
  WholeCardInstructionProgramCost ddrCandidate = makeCost(100'000'000);
  WholeCardInstructionProgramCost computeCandidate = makeCost(0);
  // ceil(5,333,333,336 * 1e12 / 8e12) == the 100 MB DDR nominal.
  computeCandidate.aggregateCompute.npuF16Bf16LogicalOps.value = 5'333'333'336;
  std::vector<CoordinatedVariantCostView> views = {
      {0, true, &baseline},
      {1, false, &ddrCandidate},
      {2, false, &computeCandidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
}

TEST(CoordinatedVariantSelectionTest,
     DecisionIsIndependentOfFullyGatedCompletionOrder) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost worse = makeCost(400'000'000);
  WholeCardInstructionProgramCost winner = makeCost(100'000'000);
  std::vector<CoordinatedVariantCostView> first = {
      {0, true, &baseline}, {8, false, &worse}, {4, false, &winner}};
  std::vector<CoordinatedVariantCostView> second = {
      {4, false, &winner}, {0, true, &baseline}, {8, false, &worse}};

  auto firstResult = plan(first);
  auto secondResult = plan(second);
  ASSERT_TRUE(mlir::succeeded(firstResult));
  ASSERT_TRUE(mlir::succeeded(secondResult));
  EXPECT_EQ(first[firstResult->selectedIndex].stableSemanticOrdinal, 4);
  EXPECT_EQ(second[secondResult->selectedIndex].stableSemanticOrdinal, 4);
}

TEST(CoordinatedVariantSelectionTest, RejectsMissingOrDuplicateBaseline) {
  WholeCardInstructionProgramCost cost = makeCost(1);
  EXPECT_TRUE(mlir::failed(plan({{0, false, &cost}})));
  EXPECT_TRUE(mlir::failed(plan({{0, true, &cost}, {1, true, &cost}})));
}

TEST(CoordinatedVariantSelectionTest,
     UsesTerminalActionOrdinalWithinOneStructuredVariant) {
  WholeCardInstructionProgramCost cost = makeCost(1);
  auto distinctActions =
      plan({{4, true, &cost, 0}, {4, false, &cost, 1}});
  ASSERT_TRUE(mlir::succeeded(distinctActions));
  EXPECT_EQ(distinctActions->selectedIndex, 0u);

  EXPECT_TRUE(mlir::failed(
      plan({{4, true, &cost, 0}, {4, false, &cost, 0}})));
}

} // namespace
