#include "MemoryPlanning/MiniMallocPacking.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {

using namespace wafer::memory_planning::detail;

StaticPackingDemand makeDemand(int64_t sizeBytes, int64_t alignmentBytes,
                               unsigned stableOrdinal, int64_t lifetimeSpan = 0,
                               int64_t allocationEvent = 0) {
  return StaticPackingDemand{sizeBytes, alignmentBytes, lifetimeSpan,
                             allocationEvent, stableOrdinal};
}

StaticPackingProblem makeFourDemandFirstFitCounterexample() {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{0, 4};
  problem.demands = {makeDemand(1, 1, 10), makeDemand(1, 1, 20),
                     makeDemand(3, 1, 30), makeDemand(3, 1, 40)};
  problem.conflicts = {PackingConflict{0, 1}, PackingConflict{0, 3},
                       PackingConflict{1, 2}};
  return problem;
}

StaticPackingProblem makeCycleConflictProblem() {
  // An induced four-cycle is not an interval graph.  It exercises the exact
  // pairwise-conflict boundary used for path-qualified multi-segment demands.
  StaticPackingProblem problem;
  problem.arena = ArenaRange{0, 8};
  problem.demands = {makeDemand(4, 4, 10), makeDemand(4, 4, 20),
                     makeDemand(4, 4, 30), makeDemand(4, 4, 40)};
  problem.conflicts = {PackingConflict{0, 1}, PackingConflict{0, 3},
                       PackingConflict{1, 2}, PackingConflict{2, 3}};
  return problem;
}

bool byteRangesOverlap(int64_t lhsBegin, int64_t lhsEnd, int64_t rhsBegin,
                       int64_t rhsEnd) {
  if (lhsBegin == lhsEnd || rhsBegin == rhsEnd)
    return false;
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

bool demandsConflict(const StaticPackingProblem &problem, unsigned lhs,
                     unsigned rhs) {
  if (lhs > rhs)
    std::swap(lhs, rhs);
  return std::any_of(problem.conflicts.begin(), problem.conflicts.end(),
                     [&](const PackingConflict &conflict) {
                       return conflict.lhsDemandIndex == lhs &&
                              conflict.rhsDemandIndex == rhs;
                     });
}

bool hasExhaustivePacking(const StaticPackingProblem &problem) {
  constexpr int64_t kUnassigned = std::numeric_limits<int64_t>::min();

  std::vector<std::vector<int64_t>> candidates(problem.demands.size());
  for (auto [demandIndex, demand] : llvm::enumerate(problem.demands)) {
    for (int64_t offset = problem.arena.begin; offset <= problem.arena.end;
         ++offset) {
      if (offset % demand.alignmentBytes != 0 ||
          demand.sizeBytes > problem.arena.end - offset)
        continue;
      candidates[demandIndex].push_back(offset);
    }
    if (candidates[demandIndex].empty())
      return false;
  }

  // Candidate-count and conflict-degree ordering keeps the exhaustive oracle
  // bounded without sharing MiniMalloc's ordering or placement algorithm.
  std::vector<unsigned> order(problem.demands.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](unsigned lhs, unsigned rhs) {
    if (candidates[lhs].size() != candidates[rhs].size())
      return candidates[lhs].size() < candidates[rhs].size();
    auto degree = [&](unsigned index) {
      return std::count_if(problem.conflicts.begin(), problem.conflicts.end(),
                           [&](const PackingConflict &conflict) {
                             return conflict.lhsDemandIndex == index ||
                                    conflict.rhsDemandIndex == index;
                           });
    };
    if (degree(lhs) != degree(rhs))
      return degree(lhs) > degree(rhs);
    return lhs < rhs;
  });

  std::vector<int64_t> assigned(problem.demands.size(), kUnassigned);
  auto search = [&](auto &self, size_t orderIndex) -> bool {
    if (orderIndex == order.size())
      return true;
    unsigned demandIndex = order[orderIndex];
    const StaticPackingDemand &demand = problem.demands[demandIndex];
    for (int64_t offset : candidates[demandIndex]) {
      int64_t end = offset + demand.sizeBytes;
      bool legal = true;
      for (unsigned otherIndex = 0; otherIndex < problem.demands.size();
           ++otherIndex) {
        if (assigned[otherIndex] == kUnassigned ||
            !demandsConflict(problem, demandIndex, otherIndex))
          continue;
        if (byteRangesOverlap(offset, end, assigned[otherIndex],
                              assigned[otherIndex] +
                                  problem.demands[otherIndex].sizeBytes)) {
          legal = false;
          break;
        }
      }
      if (!legal)
        continue;
      assigned[demandIndex] = offset;
      if (self(self, orderIndex + 1))
        return true;
      assigned[demandIndex] = kUnassigned;
    }
    return false;
  };
  return search(search, 0);
}

std::map<unsigned, int64_t>
offsetsByStableOrdinal(const StaticPackingProblem &problem,
                       const PackingResult &result) {
  std::map<unsigned, int64_t> offsets;
  for (const Placement &placement : result.placements) {
    offsets.emplace(problem.demands[placement.demandIndex].stableOrdinal,
                    placement.offsetBytes);
  }
  return offsets;
}

std::vector<unsigned> placementOrdinals(const StaticPackingProblem &problem,
                                        const PackingResult &result) {
  std::vector<unsigned> ordinals;
  for (const Placement &placement : result.placements)
    ordinals.push_back(problem.demands[placement.demandIndex].stableOrdinal);
  return ordinals;
}

StaticPackingProblem
permuteProblem(const StaticPackingProblem &problem,
               llvm::ArrayRef<unsigned> newIndexToOldIndex) {
  StaticPackingProblem permuted;
  permuted.arena = problem.arena;
  llvm::SmallVector<unsigned, 8> oldIndexToNewIndex(problem.demands.size());
  for (auto [newIndex, oldIndex] : llvm::enumerate(newIndexToOldIndex)) {
    permuted.demands.push_back(problem.demands[oldIndex]);
    oldIndexToNewIndex[oldIndex] = newIndex;
  }
  for (const PackingConflict &conflict : problem.conflicts) {
    unsigned lhs = oldIndexToNewIndex[conflict.lhsDemandIndex];
    unsigned rhs = oldIndexToNewIndex[conflict.rhsDemandIndex];
    if (lhs > rhs)
      std::swap(lhs, rhs);
    permuted.conflicts.push_back(PackingConflict{lhs, rhs});
  }
  llvm::sort(permuted.conflicts,
             [](const PackingConflict &lhs, const PackingConflict &rhs) {
               return std::pair<unsigned, unsigned>{lhs.lhsDemandIndex,
                                                    lhs.rhsDemandIndex} <
                      std::pair<unsigned, unsigned>{rhs.lhsDemandIndex,
                                                    rhs.rhsDemandIndex};
             });
  return permuted;
}

TEST(MiniMallocPackingTest, MiniMallocSolvesTheFourDemandCounterexample) {
  StaticPackingProblem problem = makeFourDemandFirstFitCounterexample();

  PackingResult result = packStaticMemory(problem);
  ASSERT_EQ(result.status, PackingStatus::Feasible);
  EXPECT_GT(result.searchNodes, 0u);
  EXPECT_FALSE(validatePlacements(problem, result.placements));
}

TEST(MiniMallocPackingTest, ZeroBudgetReturnsResourceExhausted) {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{0, 8};
  problem.demands = {makeDemand(4, 4, 0), makeDemand(4, 4, 1)};

  PackingResult result = packStaticMemory(problem, /*searchNodeBudget=*/0);
  ASSERT_EQ(result.status, PackingStatus::ResourceExhausted);
  EXPECT_EQ(result.searchNodes, 0u);
  EXPECT_TRUE(result.placements.empty());
}

TEST(MiniMallocPackingTest, ProvenInfeasibleCarriesExactCliqueEvidence) {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{0, 5};
  problem.demands = {makeDemand(3, 1, 0), makeDemand(3, 1, 1)};
  problem.conflicts = {PackingConflict{0, 1}};

  PackingResult result = packStaticMemory(problem);
  EXPECT_EQ(result.status, PackingStatus::ProvenInfeasible);
  EXPECT_TRUE(result.placements.empty());
  ASSERT_EQ(result.capacityConflictDemandIndices.size(), 2u);
  EXPECT_EQ(result.capacityConflictDemandIndices[0], 0u);
  EXPECT_EQ(result.capacityConflictDemandIndices[1], 1u);
  EXPECT_TRUE(result.individuallyOversizedDemandIndices.empty());
}

TEST(MiniMallocPackingTest,
     CliqueCoverExposesCapacityCertificateWithoutSearch) {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{3, 10};
  problem.demands = {makeDemand(2, 1, 0), makeDemand(2, 1, 1),
                     makeDemand(2, 1, 2), makeDemand(2, 1, 3)};
  for (unsigned lhs = 0; lhs < problem.demands.size(); ++lhs)
    for (unsigned rhs = lhs + 1; rhs < problem.demands.size(); ++rhs)
      problem.conflicts.push_back(PackingConflict{lhs, rhs});

  PackingResult result = solveWithMiniMalloc(problem, /*searchNodeBudget=*/0);
  EXPECT_EQ(result.status, PackingStatus::ProvenInfeasible);
  EXPECT_EQ(result.searchNodes, 0u);
  EXPECT_TRUE(result.placements.empty());
  ASSERT_EQ(result.capacityConflictDemandIndices.size(), 4u);
  EXPECT_TRUE(result.individuallyOversizedDemandIndices.empty());
  uint64_t certificateBytes = 0;
  for (unsigned index : result.capacityConflictDemandIndices)
    certificateBytes += static_cast<uint64_t>(problem.demands[index].sizeBytes);
  EXPECT_GT(certificateBytes,
            static_cast<uint64_t>(problem.arena.end - problem.arena.begin));
}

TEST(MiniMallocPackingTest,
     ReportsEveryIndependentlyOversizedDemandWithoutClaimingOneClique) {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{3, 10};
  problem.demands = {makeDemand(8, 1, 30), makeDemand(2, 1, 20),
                     makeDemand(9, 1, 10)};

  PackingResult result = solveWithMiniMalloc(problem, /*searchNodeBudget=*/0);
  EXPECT_EQ(result.status, PackingStatus::ProvenInfeasible);
  ASSERT_EQ(result.individuallyOversizedDemandIndices.size(), 2u);
  EXPECT_EQ(result.individuallyOversizedDemandIndices[0], 2u);
  EXPECT_EQ(result.individuallyOversizedDemandIndices[1], 0u);
  ASSERT_EQ(result.capacityConflictDemandIndices.size(), 1u);
  EXPECT_EQ(result.capacityConflictDemandIndices.front(), 2u);
}

TEST(MiniMallocPackingTest,
     InterleavedComponentsReuseNonzeroBaseWithoutCrossCoupling) {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{3, 11};
  problem.demands = {makeDemand(2, 2, 10), makeDemand(2, 2, 20),
                     makeDemand(2, 2, 30), makeDemand(2, 2, 40)};
  problem.conflicts = {PackingConflict{0, 2}, PackingConflict{1, 3}};

  PackingResult result = solveWithMiniMalloc(problem, 100);
  ASSERT_EQ(result.status, PackingStatus::Feasible);
  ASSERT_FALSE(validatePlacements(problem, result.placements));
  std::map<unsigned, int64_t> offsets = offsetsByStableOrdinal(problem, result);
  EXPECT_EQ(offsets.at(10), offsets.at(20));
  EXPECT_EQ(offsets.at(30), offsets.at(40));
  EXPECT_NE(offsets.at(10), offsets.at(30));
}

TEST(MiniMallocPackingTest, RepeatedSolveHasDeterministicPlacementAndWork) {
  StaticPackingProblem problem = makeFourDemandFirstFitCounterexample();
  PackingResult first = packStaticMemory(problem);
  PackingResult second = packStaticMemory(problem);

  ASSERT_EQ(first.status, PackingStatus::Feasible);
  ASSERT_EQ(second.status, PackingStatus::Feasible);
  EXPECT_EQ(offsetsByStableOrdinal(problem, first),
            offsetsByStableOrdinal(problem, second));
  EXPECT_EQ(first.searchNodes, second.searchNodes);
}

TEST(MiniMallocPackingTest, StableOrdinalMakesInputPermutationCanonical) {
  StaticPackingProblem original = makeFourDemandFirstFitCounterexample();
  StaticPackingProblem permuted = permuteProblem(original, {2, 0, 3, 1});
  ASSERT_FALSE(validatePackingProblem(permuted));

  PackingResult originalResult = packStaticMemory(original);
  PackingResult permutedResult = packStaticMemory(permuted);
  ASSERT_EQ(originalResult.status, PackingStatus::Feasible);
  ASSERT_EQ(permutedResult.status, PackingStatus::Feasible);
  EXPECT_EQ(offsetsByStableOrdinal(original, originalResult),
            offsetsByStableOrdinal(permuted, permutedResult));
  EXPECT_EQ(originalResult.searchNodes, permutedResult.searchNodes);
}

TEST(MiniMallocPackingTest,
     ZeroOnlyPlacementAndFailureOriginsArePermutationCanonical) {
  StaticPackingProblem original;
  original.arena = ArenaRange{1, 16};
  original.demands = {makeDemand(0, 4, 30), makeDemand(0, 2, 10),
                      makeDemand(0, 3, 20)};
  StaticPackingProblem permuted = permuteProblem(original, {2, 0, 1});

  PackingResult originalResult = solveWithMiniMalloc(original, 0);
  PackingResult permutedResult = solveWithMiniMalloc(permuted, 0);
  ASSERT_EQ(originalResult.status, PackingStatus::Feasible);
  ASSERT_EQ(permutedResult.status, PackingStatus::Feasible);
  EXPECT_EQ(placementOrdinals(original, originalResult),
            (std::vector<unsigned>{10, 20, 30}));
  EXPECT_EQ(placementOrdinals(original, originalResult),
            placementOrdinals(permuted, permutedResult));
  EXPECT_EQ(offsetsByStableOrdinal(original, originalResult),
            offsetsByStableOrdinal(permuted, permutedResult));
  EXPECT_EQ(originalResult.searchNodes, 0u);
  EXPECT_EQ(permutedResult.searchNodes, 0u);

  original.arena = ArenaRange{1, 1};
  permuted = permuteProblem(original, {2, 0, 1});
  originalResult = solveWithMiniMalloc(original, 0);
  permutedResult = solveWithMiniMalloc(permuted, 0);
  ASSERT_EQ(originalResult.status, PackingStatus::ProvenInfeasible);
  ASSERT_EQ(permutedResult.status, PackingStatus::ProvenInfeasible);
  ASSERT_TRUE(originalResult.demandIndex);
  ASSERT_TRUE(permutedResult.demandIndex);
  EXPECT_EQ(original.demands[*originalResult.demandIndex].stableOrdinal, 10u);
  EXPECT_EQ(permuted.demands[*permutedResult.demandIndex].stableOrdinal, 10u);
  EXPECT_TRUE(originalResult.placements.empty());
  EXPECT_TRUE(permutedResult.placements.empty());
}

TEST(MiniMallocPackingTest, AdapterRejectsInvalidProblemBeforeEncoding) {
  StaticPackingProblem badAlignment;
  badAlignment.arena = ArenaRange{0, 8};
  badAlignment.demands = {makeDemand(0, 0, 0)};
  PackingResult result = solveWithMiniMalloc(badAlignment, 10);
  EXPECT_EQ(result.status, PackingStatus::InvalidProblem);
  EXPECT_EQ(result.demandIndex, 0u);

  StaticPackingProblem badEdge;
  badEdge.arena = ArenaRange{0, 8};
  badEdge.demands = {makeDemand(4, 1, 0)};
  badEdge.conflicts = {PackingConflict{0, 1}};
  result = solveWithMiniMalloc(badEdge, 10);
  EXPECT_EQ(result.status, PackingStatus::InvalidProblem);
  EXPECT_TRUE(result.placements.empty());
}

TEST(MiniMallocPackingTest, ExactConflictGraphSupportsMultiSegmentRelations) {
  StaticPackingProblem problem = makeCycleConflictProblem();
  PackingResult result = packStaticMemory(problem);

  ASSERT_EQ(result.status, PackingStatus::Feasible);
  ASSERT_FALSE(validatePlacements(problem, result.placements));
  std::map<unsigned, int64_t> offsets = offsetsByStableOrdinal(problem, result);
  ASSERT_EQ(offsets.size(), 4u);
  EXPECT_EQ(offsets.at(10), offsets.at(30));
  EXPECT_EQ(offsets.at(20), offsets.at(40));
  EXPECT_NE(offsets.at(10), offsets.at(20));
}

TEST(MiniMallocPackingTest,
     ZeroSizeDemandsDoNotBlockAlignedPositiveDemandReuse) {
  StaticPackingProblem problem;
  problem.arena = ArenaRange{1, 11};
  problem.demands = {makeDemand(0, 1, 10), makeDemand(0, 4, 20),
                     makeDemand(5, 4, 30), makeDemand(5, 4, 40)};
  problem.conflicts = {PackingConflict{0, 2}, PackingConflict{0, 3},
                       PackingConflict{1, 2}, PackingConflict{1, 3}};

  ASSERT_TRUE(hasExhaustivePacking(problem));
  PackingResult result = solveWithMiniMalloc(problem, uint64_t{1} << 20);
  ASSERT_EQ(result.status, PackingStatus::Feasible);
  ASSERT_FALSE(validatePlacements(problem, result.placements));
  std::map<unsigned, int64_t> offsets = offsetsByStableOrdinal(problem, result);
  EXPECT_EQ(offsets.at(30), 4);
  EXPECT_EQ(offsets.at(40), 4);
}

TEST(MiniMallocPackingTest,
     AlignmentBoundaryDistinguishesInfeasibleFromSearchExhaustion) {
  StaticPackingProblem zeroOnly;
  zeroOnly.arena = ArenaRange{std::numeric_limits<int64_t>::max(),
                              std::numeric_limits<int64_t>::max()};
  zeroOnly.demands = {makeDemand(0, 2, 0)};
  PackingResult result = packStaticMemory(zeroOnly);
  EXPECT_EQ(result.status, PackingStatus::ProvenInfeasible);
  EXPECT_TRUE(result.placements.empty());

  StaticPackingProblem exhausted;
  exhausted.arena = ArenaRange{std::numeric_limits<int64_t>::max() - 2,
                               std::numeric_limits<int64_t>::max()};
  exhausted.demands = {makeDemand(1, 4, 0)};
  result = packStaticMemory(exhausted, /*searchNodeBudget=*/0);
  EXPECT_EQ(result.status, PackingStatus::ResourceExhausted);
  EXPECT_EQ(result.searchNodes, 0u);
  EXPECT_TRUE(result.placements.empty());
}

TEST(MiniMallocPackingTest,
     SeededArbitraryGraphsMatchIndependentExhaustiveOracle) {
  constexpr unsigned kSeed = 0x5A17C0DE;
  constexpr unsigned kCaseCount = 256;
  constexpr uint64_t kSearchNodeBudget = uint64_t{1} << 20;
  constexpr int64_t kAlignments[] = {1, 2, 3, 4, 5};
  std::mt19937 random(kSeed);

  bool sawFeasible = false;
  bool sawInfeasible = false;
  bool sawNonzeroArenaBase = false;
  bool sawNontrivialAlignment = false;
  bool sawZeroSizeDemand = false;
  for (unsigned caseIndex = 0; caseIndex < kCaseCount; ++caseIndex) {
    StaticPackingProblem problem;
    const unsigned demandCount = 1 + random() % 5;
    const int64_t arenaSize = 2 + random() % 8;
    // Exercise absolute alignment both at zero and at deliberately unaligned,
    // nonzero arena bases.
    problem.arena.begin = caseIndex % 4 == 0 ? 0 : 1 + random() % 7;
    problem.arena.end = problem.arena.begin + arenaSize;
    for (unsigned demandIndex = 0; demandIndex < demandCount; ++demandIndex) {
      int64_t alignment = kAlignments[random() % std::size(kAlignments)];
      int64_t size = random() % (arenaSize + 3);
      problem.demands.push_back(
          makeDemand(size, alignment, /*stableOrdinal=*/100 + demandIndex * 7));
      sawNontrivialAlignment |= alignment > 1;
      sawZeroSizeDemand |= size == 0;
    }
    for (unsigned lhs = 0; lhs < demandCount; ++lhs) {
      for (unsigned rhs = lhs + 1; rhs < demandCount; ++rhs) {
        if (random() % 2 != 0)
          problem.conflicts.push_back(PackingConflict{lhs, rhs});
      }
    }
    sawNonzeroArenaBase |= problem.arena.begin != 0;

    SCOPED_TRACE(::testing::Message()
                 << "seed=" << kSeed << " case=" << caseIndex
                 << " demands=" << demandCount
                 << " conflicts=" << problem.conflicts.size() << " arena=["
                 << problem.arena.begin << "," << problem.arena.end << ")");
    ASSERT_FALSE(validatePackingProblem(problem));
    bool oracleFeasible = hasExhaustivePacking(problem);
    PackingResult result = solveWithMiniMalloc(problem, kSearchNodeBudget);
    ASSERT_NE(result.status, PackingStatus::ResourceExhausted);
    if (oracleFeasible) {
      sawFeasible = true;
      ASSERT_EQ(result.status, PackingStatus::Feasible);
      EXPECT_FALSE(validatePlacements(problem, result.placements));
    } else {
      sawInfeasible = true;
      EXPECT_EQ(result.status, PackingStatus::ProvenInfeasible);
      EXPECT_TRUE(result.placements.empty());
    }
  }

  EXPECT_TRUE(sawFeasible);
  EXPECT_TRUE(sawInfeasible);
  EXPECT_TRUE(sawNonzeroArenaBase);
  EXPECT_TRUE(sawNontrivialAlignment);
  EXPECT_TRUE(sawZeroSizeDemand);
}

} // namespace
