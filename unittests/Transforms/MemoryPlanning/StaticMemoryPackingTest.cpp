#include "MemoryPlanning/StaticMemoryPacking.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace {

using namespace wafer::memory_planning::detail;

StaticPackingDemand makeDemand(int64_t sizeBytes, int64_t alignmentBytes,
                               unsigned stableOrdinal, int64_t lifetimeSpan = 0,
                               int64_t allocationEvent = 0) {
  return StaticPackingDemand{sizeBytes, alignmentBytes, lifetimeSpan,
                             allocationEvent, stableOrdinal};
}

StaticPackingProblem makeValidationProblem() {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{0, 16};
  problem.demands = {makeDemand(8, 4, 10), makeDemand(8, 4, 20)};
  problem.conflicts = {PackingConflict{0, 1}};
  return problem;
}

// Size-first first-fit places demands 2 and 3 at [0, 3), then demand 0 at
// [3, 4), leaving no range for demand 1.  A legal placement exists:
//   2 -> [0, 3), 3 -> [1, 4), 0 -> [0, 1), 1 -> [3, 4).
StaticPackingProblem makeFourDemandFirstFitCounterexample() {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{0, 4};
  problem.demands = {makeDemand(1, 1, 0), makeDemand(1, 1, 1),
                     makeDemand(3, 1, 2), makeDemand(3, 1, 3)};
  problem.conflicts = {PackingConflict{0, 1}, PackingConflict{0, 3},
                       PackingConflict{1, 2}};
  return problem;
}

TEST(StaticMemoryPackingTest, ProblemValidatorAcceptsCanonicalProblem) {
  EXPECT_FALSE(validatePackingProblem(makeValidationProblem()));
}

TEST(StaticMemoryPackingTest,
     BuilderGivesEmptySegmentDemandsIndependentPrivateActivity) {
  LifetimeDemand first;
  first.sizeBytes = 8;
  first.alignmentBytes = 4;
  first.stableOrdinal = 20;
  first.allocationPoint.event = 7;
  LifetimeDemand second = first;
  second.stableOrdinal = 10;
  second.allocationPoint.event = 9;

  StaticPackingProblem problem =
      buildStaticPackingProblem({first, second}, ArenaRange{3, 16});
  ASSERT_EQ(problem.demands.size(), 2u);
  EXPECT_TRUE(problem.conflicts.empty());
  EXPECT_EQ(problem.demands[0].lifetimeSpan, 0);
  EXPECT_EQ(problem.demands[1].lifetimeSpan, 0);

  PackingResult result = packStaticMemory(problem);
  ASSERT_EQ(result.status, PackingStatus::Feasible);
  ASSERT_EQ(result.placements.size(), 2u);
  EXPECT_FALSE(validatePlacements(problem, result.placements));
  EXPECT_EQ(result.placements[0].offsetBytes, 4);
  EXPECT_EQ(result.placements[1].offsetBytes, 4);
}

TEST(StaticMemoryPackingTest,
     ProblemValidatorRejectsInvalidArenaDemandAndEdge) {
  StaticPackingProblem problem = makeValidationProblem();
  problem.arena = ArenaRange{-1, 16};
  std::optional<PackingValidationFailure> failure =
      validatePackingProblem(problem);
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::InvalidArena);

  problem = makeValidationProblem();
  problem.demands[0].sizeBytes = -1;
  failure = validatePackingProblem(problem);
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::InvalidDemand);
  EXPECT_EQ(failure->demandIndex, 0u);

  problem = makeValidationProblem();
  problem.demands[1].alignmentBytes = 0;
  failure = validatePackingProblem(problem);
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::InvalidDemand);
  EXPECT_EQ(failure->demandIndex, 1u);

  problem = makeValidationProblem();
  problem.demands[1].stableOrdinal = problem.demands[0].stableOrdinal;
  failure = validatePackingProblem(problem);
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::InvalidDemand);
  EXPECT_EQ(failure->demandIndex, 1u);

  problem = makeValidationProblem();
  problem.conflicts = {PackingConflict{1, 0}};
  failure = validatePackingProblem(problem);
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::InvalidConflict);

  problem = makeValidationProblem();
  problem.conflicts.push_back(PackingConflict{0, 1});
  failure = validatePackingProblem(problem);
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::InvalidConflict);
}

TEST(StaticMemoryPackingTest, PlacementValidatorAcceptsCompleteLegalPlacement) {
  StaticPackingProblem problem = makeValidationProblem();
  llvm::SmallVector<Placement, 2> placements = {Placement{0, 0, 8},
                                                Placement{1, 8, 16}};
  EXPECT_FALSE(validatePlacements(problem, placements));
}

TEST(StaticMemoryPackingTest,
     PlacementValidatorRejectsMissingDuplicateAndConflictingPlacement) {
  StaticPackingProblem problem = makeValidationProblem();
  std::optional<PackingValidationFailure> failure =
      validatePlacements(problem, {Placement{0, 0, 8}});
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::MissingPlacement);
  EXPECT_EQ(failure->demandIndex, 1u);

  failure =
      validatePlacements(problem, {Placement{0, 0, 8}, Placement{0, 8, 16}});
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::DuplicatePlacement);
  EXPECT_EQ(failure->demandIndex, 0u);

  failure =
      validatePlacements(problem, {Placement{0, 0, 8}, Placement{1, 0, 8}});
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::ConflictingRanges);
  EXPECT_EQ(failure->demandIndex, 1u);
}

TEST(StaticMemoryPackingTest,
     PlacementValidatorRejectsMalformedRangeAndAbsoluteMisalignment) {
  StaticPackingProblem problem = makeValidationProblem();
  std::optional<PackingValidationFailure> failure =
      validatePlacements(problem, {Placement{0, 0, 7}, Placement{1, 8, 16}});
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::EndMismatch);
  EXPECT_EQ(failure->demandIndex, 0u);

  failure =
      validatePlacements(problem, {Placement{0, 2, 10}, Placement{1, 8, 16}});
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::Misaligned);
  EXPECT_EQ(failure->demandIndex, 0u);

  failure =
      validatePlacements(problem, {Placement{0, 0, 8}, Placement{1, 12, 20}});
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::OutOfRange);
  EXPECT_EQ(failure->demandIndex, 1u);

  problem.arena = ArenaRange{0, std::numeric_limits<int64_t>::max()};
  problem.demands = {makeDemand(8, 1, 0)};
  problem.conflicts.clear();
  failure = validatePlacements(
      problem, {Placement{0, std::numeric_limits<int64_t>::max() - 3,
                          std::numeric_limits<int64_t>::max()}});
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->kind, PackingValidationFailureKind::RangeOverflow);
  EXPECT_EQ(failure->demandIndex, 0u);
}

TEST(StaticMemoryPackingTest, DefaultBudgetUsesWideDeterministicFormula) {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{0, 64};
  problem.demands = {makeDemand(8, 1, 0), makeDemand(8, 1, 1),
                     makeDemand(8, 1, 2)};
  problem.conflicts = {PackingConflict{0, 1}, PackingConflict{1, 2}};

  EXPECT_EQ(defaultPackingSearchNodeBudget(problem),
            kBasePackingSearchNodes + uint64_t{3} * 64 + uint64_t{2} * 16);
  EXPECT_GT(defaultPackingSearchNodeBudget(problem), uint64_t{1000000});
  EXPECT_LE(defaultPackingSearchNodeBudget(problem),
            kMaxDefaultPackingSearchNodes);
}

TEST(StaticMemoryPackingTest, FirstFitUsesAbsoluteAlignmentForNonzeroBase) {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{3, 32};
  problem.demands = {makeDemand(7, 8, 0)};

  PackingResult result = packFirstFit(problem);
  ASSERT_EQ(result.status, PackingStatus::Feasible);
  ASSERT_EQ(result.backend, PackingBackend::FirstFitFallback);
  ASSERT_EQ(result.placements.size(), 1u);
  EXPECT_EQ(result.placements.front().offsetBytes, 8);
  EXPECT_EQ(result.placements.front().endBytes, 15);
  EXPECT_FALSE(validatePlacements(problem, result.placements));
}

TEST(StaticMemoryPackingTest, FirstFitCanRejectAFeasibleFourDemandProblem) {
  StaticPackingProblem problem = makeFourDemandFirstFitCounterexample();
  PackingResult result = packFirstFit(problem);

  EXPECT_EQ(result.status, PackingStatus::HeuristicNoFit);
  EXPECT_EQ(result.backend, PackingBackend::FirstFitFallback);
  EXPECT_FALSE(result.succeeded());
  EXPECT_TRUE(result.placements.empty());

  llvm::SmallVector<Placement, 4> witness = {
      Placement{0, 0, 1}, Placement{1, 3, 4}, Placement{2, 0, 3},
      Placement{3, 1, 4}};
  EXPECT_FALSE(validatePlacements(problem, witness));
}

} // namespace
