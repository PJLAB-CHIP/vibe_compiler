//===- WholeVariantAttemptPlanTest.cpp ----------------------------------===//

#include "../../lib/Wafer/Compiler/WholeVariantAttemptPlan.h"
#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using wafer::RankArtifactKind;
using wafer::RankBufferingKind;
using wafer::RankWorkerPlacementKind;
using wafer::compiler::detail::RankVariantMetadataFrontier;
using wafer::compiler::detail::RankVariantSlotMetadata;
using wafer::compiler::detail::WholeVariantAttemptPlan;
using wafer::compiler::detail::WholeVariantAttemptPlanFailure;

constexpr size_t kLegacyProductLimit = 64;
constexpr size_t kCoordinatedLimit = 64;

using CandidateOrder = std::vector<std::vector<size_t>>;
using CorrespondenceKey =
    std::tuple<int64_t, RankArtifactKind, RankBufferingKind, uint32_t,
               RankWorkerPlacementKind, uint32_t>;

struct Combination {
  std::vector<size_t> positions;
};

struct WorseCombination {
  bool operator()(const Combination &lhs, const Combination &rhs) const {
    return lhs.positions > rhs.positions;
  }
};

struct LegacyAttemptReference {
  std::vector<size_t> reservedBaselineIndices;
  std::vector<std::vector<size_t>> optimizedCandidateIndices;
  size_t productPositionCount = 0;
  size_t productUniqueAttemptCount = 0;
  size_t coordinatedAttemptCount = 0;
};

static std::vector<size_t>
indicesForPositions(const std::vector<size_t> &positions,
                    const CandidateOrder &order) {
  std::vector<size_t> indices;
  for (size_t rank = 0; rank < positions.size(); ++rank)
    indices.push_back(order[rank][positions[rank]]);
  return indices;
}

/// Independent copy of the bounded coordinator enumeration. Tests compare
/// every original slot index, not only the correspondence key or final
/// candidate count.
static LegacyAttemptReference enumerateLegacyAttempts(
    const std::vector<RankVariantMetadataFrontier> &frontiers) {
  LegacyAttemptReference reference;
  CandidateOrder order(frontiers.size());
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    const RankVariantMetadataFrontier &frontier = frontiers[rank];
    for (size_t index = 0; index < frontier.size(); ++index) {
      order[rank].push_back(index);
      if (frontier[index].reservedBaseline)
        reference.reservedBaselineIndices.push_back(index);
    }
    std::sort(order[rank].begin(), order[rank].end(),
              [&](size_t lhs, size_t rhs) {
                const RankVariantSlotMetadata &left = frontier[lhs];
                const RankVariantSlotMetadata &right = frontier[rhs];
                return std::tie(left.stableOrdinal, left.artifactKind,
                                left.bufferingKind, left.bufferingPlanOrdinal,
                                left.workerPlacementKind,
                                left.workerPlacementPlanOrdinal, lhs) <
                       std::tie(right.stableOrdinal, right.artifactKind,
                                right.bufferingKind, right.bufferingPlanOrdinal,
                                right.workerPlacementKind,
                                right.workerPlacementPlanOrdinal, rhs);
              });
  }

  std::set<std::vector<size_t>> attempted;
  attempted.insert(reference.reservedBaselineIndices);
  std::set<std::vector<size_t>> enqueued;
  std::priority_queue<Combination, std::vector<Combination>, WorseCombination>
      queue;
  std::vector<size_t> initial(frontiers.size(), 0);
  queue.push({initial});
  enqueued.insert(initial);
  while (!queue.empty() &&
         reference.productPositionCount < kLegacyProductLimit) {
    Combination combination = queue.top();
    queue.pop();
    ++reference.productPositionCount;
    std::vector<size_t> indices =
        indicesForPositions(combination.positions, order);
    if (attempted.insert(indices).second)
      reference.optimizedCandidateIndices.push_back(std::move(indices));
    for (size_t rank = 0; rank < combination.positions.size(); ++rank) {
      std::vector<size_t> neighbor = combination.positions;
      if (++neighbor[rank] >= order[rank].size())
        continue;
      if (enqueued.insert(neighbor).second)
        queue.push({std::move(neighbor)});
    }
  }
  reference.productUniqueAttemptCount =
      reference.optimizedCandidateIndices.size();

  std::set<CorrespondenceKey> keys;
  for (const RankVariantSlotMetadata &candidate : frontiers.front())
    keys.insert({candidate.stableOrdinal, candidate.artifactKind,
                 candidate.bufferingKind, candidate.bufferingPlanOrdinal,
                 candidate.workerPlacementKind,
                 candidate.workerPlacementPlanOrdinal});
  std::vector<std::vector<size_t>> coordinatedCandidates;
  for (CorrespondenceKey key : keys) {
    std::vector<size_t> indices;
    bool complete = true;
    for (const RankVariantMetadataFrontier &frontier : frontiers) {
      std::optional<size_t> match;
      for (size_t index = 0; index < frontier.size(); ++index) {
        const RankVariantSlotMetadata &candidate = frontier[index];
        if (candidate.stableOrdinal != std::get<0>(key) ||
            candidate.artifactKind != std::get<1>(key) ||
            candidate.bufferingKind != std::get<2>(key) ||
            candidate.bufferingPlanOrdinal != std::get<3>(key) ||
            candidate.workerPlacementKind != std::get<4>(key) ||
            candidate.workerPlacementPlanOrdinal != std::get<5>(key))
          continue;
        if (!match || index < *match)
          match = index;
      }
      if (!match) {
        complete = false;
        break;
      }
      indices.push_back(*match);
    }
    if (!complete || attempted.count(indices))
      continue;
    coordinatedCandidates.push_back(std::move(indices));
  }
  auto appendCoordinated = [&](size_t index) {
    std::vector<size_t> indices = std::move(coordinatedCandidates[index]);
    if (!attempted.insert(indices).second)
      return;
    ++reference.coordinatedAttemptCount;
    reference.optimizedCandidateIndices.push_back(std::move(indices));
  };
  if (coordinatedCandidates.size() <= kCoordinatedLimit) {
    for (size_t index = 0; index < coordinatedCandidates.size(); ++index)
      appendCoordinated(index);
  } else {
    const size_t span = coordinatedCandidates.size() - 1;
    const size_t intervals = kCoordinatedLimit - 1;
    const size_t quotient = span / intervals;
    const size_t remainder = span % intervals;
    for (size_t sample = 0; sample < kCoordinatedLimit; ++sample)
      appendCoordinated(sample * quotient + (sample * remainder) / intervals);
  }
  return reference;
}

static bool hasCompleteCorrespondence(
    const std::vector<size_t> &indices,
    const std::vector<RankVariantMetadataFrontier> &frontiers) {
  std::optional<CorrespondenceKey> key;
  for (size_t rank = 0; rank < indices.size(); ++rank) {
    const RankVariantSlotMetadata &candidate = frontiers[rank][indices[rank]];
    CorrespondenceKey current{
        candidate.stableOrdinal,       candidate.artifactKind,
        candidate.bufferingKind,       candidate.bufferingPlanOrdinal,
        candidate.workerPlacementKind, candidate.workerPlacementPlanOrdinal};
    if (key && current != *key)
      return false;
    key = current;
  }
  return key.has_value();
}

static std::vector<std::vector<size_t>> requiredByLegacyAttempts(
    const LegacyAttemptReference &reference,
    const std::vector<RankVariantMetadataFrontier> &frontiers) {
  std::vector<std::vector<bool>> required(frontiers.size());
  for (size_t rank = 0; rank < frontiers.size(); ++rank)
    required[rank].assign(frontiers[rank].size(), false);
  for (size_t rank = 0; rank < reference.reservedBaselineIndices.size(); ++rank)
    required[rank][reference.reservedBaselineIndices[rank]] = true;
  for (const std::vector<size_t> &indices :
       reference.optimizedCandidateIndices) {
    if (!hasCompleteCorrespondence(indices, frontiers))
      continue;
    for (size_t rank = 0; rank < indices.size(); ++rank)
      required[rank][indices[rank]] = true;
  }
  std::vector<std::vector<size_t>> result(frontiers.size());
  for (size_t rank = 0; rank < frontiers.size(); ++rank)
    for (size_t index = 0; index < frontiers[rank].size(); ++index)
      if (required[rank][index])
        result[rank].push_back(index);
  return result;
}

static RankVariantSlotMetadata
slot(int64_t ordinal, RankArtifactKind kind = RankArtifactKind::Spill,
     bool reservedBaseline = false,
     RankBufferingKind bufferingKind = RankBufferingKind::Single,
     uint32_t bufferingPlanOrdinal = 0,
     RankWorkerPlacementKind workerPlacementKind =
         RankWorkerPlacementKind::Unplaced,
     uint32_t workerPlacementPlanOrdinal = 0) {
  return {ordinal,
          kind,
          reservedBaseline,
          bufferingKind,
          bufferingPlanOrdinal,
          workerPlacementKind,
          workerPlacementPlanOrdinal};
}

TEST(WholeVariantAttemptPlanTest,
     RankOneKeepsOriginalSlotsAndDeduplicatesBaselineAttempt) {
  std::vector<RankVariantMetadataFrontier> frontiers(1);
  frontiers[0] = {slot(9), slot(3, RankArtifactKind::Spill, true),
                  slot(1, RankArtifactKind::Resident), slot(9), slot(7)};

  LegacyAttemptReference reference = enumerateLegacyAttempts(frontiers);
  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 1);
  ASSERT_TRUE(plan.isValid());
  EXPECT_EQ(plan.reservedBaselineIndices, reference.reservedBaselineIndices);
  EXPECT_EQ(plan.optimizedCandidateIndices,
            reference.optimizedCandidateIndices);
  EXPECT_EQ(plan.requiredModuleIndices,
            requiredByLegacyAttempts(reference, frontiers));
  EXPECT_EQ(plan.requiredModuleIndices[0],
            (std::vector<size_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(plan.getRequiredModuleCount(), 5u);
  EXPECT_EQ(std::count(plan.optimizedCandidateIndices.begin(),
                       plan.optimizedCandidateIndices.end(),
                       plan.reservedBaselineIndices),
            0);
}

TEST(WholeVariantAttemptPlanTest,
     MatchesBoundedQuantilesWithMissingReorderedAndDuplicateKeys) {
  constexpr size_t rankCount = 16;
  std::vector<RankVariantMetadataFrontier> frontiers(rankCount);
  for (size_t rank = 0; rank < rankCount; ++rank) {
    RankVariantMetadataFrontier &frontier = frontiers[rank];
    for (int64_t ordinal = 0; ordinal < 96; ++ordinal) {
      if (rank == 7 && ordinal != 0 && ordinal % 13 == 0)
        continue;
      RankArtifactKind kind = ordinal % 3 == 0 ? RankArtifactKind::Resident
                                               : RankArtifactKind::Spill;
      frontier.push_back(slot(ordinal, kind, ordinal == 0));
      if ((rank == 3 || rank == 9) && (ordinal == 7 || ordinal == 19))
        frontier.push_back(slot(ordinal, kind));
    }
    size_t rotation = (rank * 11) % frontier.size();
    std::rotate(frontier.begin(), frontier.begin() + rotation, frontier.end());
    if (rank % 2)
      std::reverse(frontier.begin(), frontier.end());
  }

  LegacyAttemptReference reference = enumerateLegacyAttempts(frontiers);
  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 16);
  ASSERT_TRUE(plan.isValid());
  EXPECT_EQ(reference.productPositionCount, 64u);
  EXPECT_EQ(reference.coordinatedAttemptCount, 64u);
  EXPECT_EQ(plan.boundedProductPositionCount, 64u);
  EXPECT_EQ(plan.boundedProductUniqueAttemptCount,
            reference.productUniqueAttemptCount);
  EXPECT_EQ(plan.coordinatedAttemptCount, 64u);
  EXPECT_EQ(plan.reservedBaselineIndices, reference.reservedBaselineIndices);
  ASSERT_GE(plan.optimizedCandidateIndices.size(),
            reference.optimizedCandidateIndices.size());
  EXPECT_EQ(std::vector<std::vector<size_t>>(
                plan.optimizedCandidateIndices.begin(),
                plan.optimizedCandidateIndices.begin() +
                    reference.optimizedCandidateIndices.size()),
            reference.optimizedCandidateIndices);
  std::vector<std::vector<size_t>> expectedRequired =
      requiredByLegacyAttempts(reference, frontiers);
  auto requireBand = [&](const std::vector<std::vector<size_t>> &band) {
    for (const std::vector<size_t> &candidateIndices : band)
      for (auto [rank, index] : llvm::enumerate(candidateIndices))
        if (!llvm::is_contained(expectedRequired[rank], index)) {
          expectedRequired[rank].push_back(index);
          llvm::sort(expectedRequired[rank]);
        }
  };
  requireBand(plan.workerPlacedCandidateIndices);
  requireBand(plan.fixedSlotCandidateIndices);
  requireBand(plan.genericCandidateIndices);
  EXPECT_EQ(plan.requiredModuleIndices, expectedRequired);

  size_t allSlots = 0;
  for (const RankVariantMetadataFrontier &frontier : frontiers)
    allSlots += frontier.size();
  EXPECT_LT(plan.getRequiredModuleCount(), allSlots);

  // Ordinal 13 is absent from rank 7. It may keep arbitrary original slots in
  // every other rank, but no such slot can be materialized.
  for (size_t rank = 0; rank < rankCount; ++rank) {
    if (rank == 7)
      continue;
    for (size_t index = 0; index < frontiers[rank].size(); ++index)
      if (frontiers[rank][index].stableOrdinal == 13)
        EXPECT_EQ(std::count(plan.requiredModuleIndices[rank].begin(),
                             plan.requiredModuleIndices[rank].end(), index),
                  0);
  }
}

TEST(WholeVariantAttemptPlanTest,
     QuantileSamplingIncludesBothEndsAndBoundsUnvisitedKeyGap) {
  RankVariantMetadataFrontier frontier;
  frontier.push_back(slot(0, RankArtifactKind::Spill,
                          /*reservedBaseline=*/true));
  for (int64_t ordinal = 1; ordinal < 256; ++ordinal)
    frontier.push_back(slot(ordinal, RankArtifactKind::Resident));
  std::vector<RankVariantMetadataFrontier> frontiers{std::move(frontier)};

  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 1);
  ASSERT_TRUE(plan.isValid());
  ASSERT_EQ(plan.coordinatedAttemptCount, kCoordinatedLimit);
  ASSERT_LE(plan.boundedProductUniqueAttemptCount,
            plan.optimizedCandidateIndices.size());

  llvm::ArrayRef<std::vector<size_t>> coordinated(
      plan.optimizedCandidateIndices.data() +
          plan.boundedProductUniqueAttemptCount,
      plan.coordinatedAttemptCount);
  ASSERT_EQ(coordinated.size(), kCoordinatedLimit);
  std::vector<int64_t> ordinals;
  for (const std::vector<size_t> &indices : coordinated) {
    ASSERT_EQ(indices.size(), 1u);
    ordinals.push_back(frontiers[0][indices.front()].stableOrdinal);
  }
  EXPECT_EQ(ordinals.front(), 64);
  EXPECT_EQ(ordinals.back(), 255);
  for (auto [lhs, rhs] : llvm::zip(ordinals, llvm::drop_begin(ordinals)))
    EXPECT_LE(rhs - lhs, 4);
}

TEST(WholeVariantAttemptPlanTest,
     ImportsMismatchedReservedBaselinesAndAttemptableCompleteTuple) {
  constexpr size_t rankCount = 16;
  std::vector<RankVariantMetadataFrontier> frontiers(rankCount);
  for (size_t rank = 0; rank < rankCount; ++rank) {
    // Baseline keys intentionally disagree. The coordinator must still see
    // and diagnose the original baseline tuple before optimized work.
    frontiers[rank].push_back(
        slot(1000 + static_cast<int64_t>(rank), RankArtifactKind::Spill, true));
    frontiers[rank].push_back(slot(5, RankArtifactKind::Resident));
    frontiers[rank].push_back(
        slot(2000 + static_cast<int64_t>(rank), RankArtifactKind::Spill));
  }

  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 16);
  ASSERT_TRUE(plan.isValid());
  ASSERT_EQ(plan.requiredModuleIndices.size(), rankCount);
  for (size_t rank = 0; rank < rankCount; ++rank)
    EXPECT_EQ(plan.requiredModuleIndices[rank], (std::vector<size_t>{0, 1}));
  EXPECT_EQ(plan.getRequiredModuleCount(), 32u);
}

TEST(WholeVariantAttemptPlanTest,
     FixedSlotBufferingIsAnIndependentCorrespondenceDimension) {
  std::vector<RankVariantMetadataFrontier> frontiers(2);
  frontiers[0] = {
      slot(0, RankArtifactKind::Spill, true),
      slot(5, RankArtifactKind::Spill, false,
           RankBufferingKind::StaticFixedSlot, 7),
  };
  frontiers[1] = {
      slot(0, RankArtifactKind::Spill, true),
      slot(5, RankArtifactKind::Spill, false, RankBufferingKind::Single),
      slot(5, RankArtifactKind::Spill, false,
           RankBufferingKind::StaticFixedSlot, 7),
  };

  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 2);
  ASSERT_TRUE(plan.isValid());
  EXPECT_EQ(plan.fixedSlotCandidateIndices,
            (std::vector<std::vector<size_t>>{{1, 2}}));
  EXPECT_EQ(plan.requiredModuleIndices[0], (std::vector<size_t>{0, 1}));
  EXPECT_EQ(plan.requiredModuleIndices[1], (std::vector<size_t>{0, 2}));
}

TEST(WholeVariantAttemptPlanTest,
     DistinctFixedSlotPlansCannotCorrespondByKindAlone) {
  std::vector<RankVariantMetadataFrontier> frontiers(2);
  frontiers[0] = {
      slot(0, RankArtifactKind::Spill, true),
      slot(5, RankArtifactKind::Spill, false,
           RankBufferingKind::StaticFixedSlot, 3),
  };
  frontiers[1] = {
      slot(0, RankArtifactKind::Spill, true),
      slot(5, RankArtifactKind::Spill, false,
           RankBufferingKind::StaticFixedSlot, 4),
  };

  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 2);
  ASSERT_TRUE(plan.isValid());
  EXPECT_TRUE(plan.fixedSlotCandidateIndices.empty());
  EXPECT_EQ(plan.requiredModuleIndices[0], (std::vector<size_t>{0}));
  EXPECT_EQ(plan.requiredModuleIndices[1], (std::vector<size_t>{0}));
}

TEST(WholeVariantAttemptPlanTest,
     WorkerPlacementIsAnIndependentCorrespondenceDimension) {
  std::vector<RankVariantMetadataFrontier> frontiers(2);
  frontiers[0] = {
      slot(0, RankArtifactKind::Spill, true),
      slot(5, RankArtifactKind::Spill, false, RankBufferingKind::Single, 0,
           RankWorkerPlacementKind::DisjointComponents, 1),
  };
  frontiers[1] = {
      slot(0, RankArtifactKind::Spill, true),
      slot(5, RankArtifactKind::Spill),
      slot(5, RankArtifactKind::Spill, false, RankBufferingKind::Single, 0,
           RankWorkerPlacementKind::DisjointComponents, 1),
  };

  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 2);
  ASSERT_TRUE(plan.isValid());
  EXPECT_EQ(plan.workerPlacedCandidateIndices,
            (std::vector<std::vector<size_t>>{{1, 2}}));
  EXPECT_EQ(plan.requiredModuleIndices[0], (std::vector<size_t>{0, 1}));
  EXPECT_EQ(plan.requiredModuleIndices[1], (std::vector<size_t>{0, 2}));
}

TEST(WholeVariantAttemptPlanTest,
     DistinctWorkerPlacementPlansCannotCorrespondByKindAlone) {
  std::vector<RankVariantMetadataFrontier> frontiers(2);
  frontiers[0] = {
      slot(0, RankArtifactKind::Spill, true),
      slot(5, RankArtifactKind::Spill, false, RankBufferingKind::Single, 0,
           RankWorkerPlacementKind::DisjointComponents, 3),
  };
  frontiers[1] = {
      slot(0, RankArtifactKind::Spill, true),
      slot(5, RankArtifactKind::Spill, false, RankBufferingKind::Single, 0,
           RankWorkerPlacementKind::DisjointComponents, 4),
  };

  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 2);
  ASSERT_TRUE(plan.isValid());
  EXPECT_TRUE(plan.workerPlacedCandidateIndices.empty());
  EXPECT_EQ(plan.requiredModuleIndices[0], (std::vector<size_t>{0}));
  EXPECT_EQ(plan.requiredModuleIndices[1], (std::vector<size_t>{0}));
}

TEST(WholeVariantAttemptPlanTest,
     BuriedCompleteWorkerSeedIsBoundedAndRequiredForOwnerImport) {
  using wafer::compiler::detail::SerializedRankVariantCandidate;
  using wafer::compiler::detail::SerializedRankVariantFrontier;

  constexpr size_t rankCount = 2;
  constexpr size_t genericCount = 160;
  constexpr size_t workerCount =
      wafer::kWorkerPlacementRankFrontierAdmissionLimit + 2;
  constexpr size_t fixedCount = wafer::kFixedSlotRankFrontierAdmissionLimit + 2;
  std::vector<RankVariantMetadataFrontier> metadata(rankCount);
  for (size_t rank = 0; rank < rankCount; ++rank) {
    metadata[rank].push_back(slot(0, RankArtifactKind::Spill, true));
    for (size_t index = 1; index <= genericCount; ++index)
      metadata[rank].push_back(
          slot(static_cast<int64_t>(index), RankArtifactKind::Resident));
    for (size_t worker = 0; worker < workerCount; ++worker)
      metadata[rank].push_back(
          slot(1000 + static_cast<int64_t>(worker),
               RankArtifactKind::SpillReady, false, RankBufferingKind::Single,
               0, RankWorkerPlacementKind::DisjointComponents,
               static_cast<uint32_t>(worker) + 1 +
                   (worker >= wafer::kWorkerPlacementRankFrontierAdmissionLimit
                        ? static_cast<uint32_t>(rank) * 100
                        : 0)));
    for (size_t fixed = 0; fixed < fixedCount; ++fixed)
      metadata[rank].push_back(
          slot(2000 + static_cast<int64_t>(fixed), RankArtifactKind::Resident,
               false, RankBufferingKind::StaticFixedSlot,
               static_cast<uint32_t>(fixed) + 1 +
                   (fixed >= wafer::kFixedSlotRankFrontierAdmissionLimit
                        ? static_cast<uint32_t>(rank) * 100
                        : 0)));
  }

  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(metadata,
                                                            rankCount);
  ASSERT_TRUE(plan.isValid());
  EXPECT_LE(1 + plan.optimizedCandidateIndices.size(),
            WholeVariantAttemptPlan::kMaximumAttemptCount);
  ASSERT_EQ(plan.workerPlacedCandidateIndices.size(),
            wafer::kWorkerPlacementRankFrontierAdmissionLimit);
  for (size_t worker = 0;
       worker < wafer::kWorkerPlacementRankFrontierAdmissionLimit; ++worker)
    EXPECT_EQ(plan.workerPlacedCandidateIndices[worker],
              (std::vector<size_t>{genericCount + 1 + worker,
                                   genericCount + 1 + worker}));
  ASSERT_EQ(plan.fixedSlotCandidateIndices.size(),
            wafer::kFixedSlotRankFrontierAdmissionLimit);
  for (size_t fixed = 0; fixed < wafer::kFixedSlotRankFrontierAdmissionLimit;
       ++fixed)
    EXPECT_EQ(plan.fixedSlotCandidateIndices[fixed],
              (std::vector<size_t>{genericCount + 1 + workerCount + fixed,
                                   genericCount + 1 + workerCount + fixed}));
  ASSERT_EQ(plan.genericCandidateIndices.size(), 8u);
  const size_t genericSpan = genericCount - 1;
  const size_t genericIntervals = plan.genericCandidateIndices.size() - 1;
  const size_t genericQuotient = genericSpan / genericIntervals;
  const size_t genericRemainder = genericSpan % genericIntervals;
  for (size_t generic = 0; generic < plan.genericCandidateIndices.size();
       ++generic) {
    const size_t sampled = generic * genericQuotient +
                           (generic * genericRemainder) / genericIntervals;
    EXPECT_EQ(plan.genericCandidateIndices[generic],
              (std::vector<size_t>{sampled + 1, sampled + 1}));
  }

  WholeVariantAttemptPlan repeated =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(metadata,
                                                            rankCount);
  EXPECT_EQ(repeated.workerPlacedCandidateIndices,
            plan.workerPlacedCandidateIndices);
  EXPECT_EQ(repeated.fixedSlotCandidateIndices, plan.fixedSlotCandidateIndices);
  EXPECT_EQ(repeated.genericCandidateIndices, plan.genericCandidateIndices);

  std::vector<SerializedRankVariantFrontier> serialized(rankCount);
  for (size_t rank = 0; rank < rankCount; ++rank) {
    std::set<size_t> required(plan.requiredModuleIndices[rank].begin(),
                              plan.requiredModuleIndices[rank].end());
    for (auto [index, candidate] : llvm::enumerate(metadata[rank])) {
      SerializedRankVariantCandidate slot;
      slot.moduleData = std::make_shared<const std::string>(
          required.count(index) ? "module {}" : "unrequired invalid MLIR");
      slot.stableOrdinal = candidate.stableOrdinal;
      slot.artifactKind = candidate.artifactKind;
      slot.reservedBaseline = candidate.reservedBaseline;
      slot.bufferingKind = candidate.bufferingKind;
      slot.bufferingPlanOrdinal = candidate.bufferingPlanOrdinal;
      slot.workerPlacementKind = candidate.workerPlacementKind;
      slot.workerPlacementPlanOrdinal = candidate.workerPlacementPlanOrdinal;
      serialized[rank].push_back(std::move(slot));
    }
  }

  mlir::MLIRContext ownerContext;
  auto imported =
      wafer::compiler::detail::importRankVariantFrontiersIntoOwnerContext(
          ownerContext, serialized, rankCount);
  if (!imported) {
    ADD_FAILURE() << llvm::toString(imported.takeError());
    return;
  }
  for (const auto &frontier : *imported)
    for (size_t worker = 0; worker < workerCount; ++worker)
      EXPECT_EQ(static_cast<bool>(frontier[genericCount + 1 + worker].module),
                worker < wafer::kWorkerPlacementRankFrontierAdmissionLimit);
  for (const auto &frontier : *imported) {
    for (const std::vector<size_t> &generic : plan.genericCandidateIndices)
      ASSERT_TRUE(frontier[generic.front()].module);
    for (size_t fixed = 0; fixed < fixedCount; ++fixed)
      EXPECT_EQ(static_cast<bool>(
                    frontier[genericCount + 1 + workerCount + fixed].module),
                fixed < wafer::kFixedSlotRankFrontierAdmissionLimit);
  }
}

TEST(WholeVariantAttemptPlanTest,
     PhysicalBandQuantilesKeepTailCompoundTupleAttemptable) {
  constexpr size_t rankCount = 2;
  constexpr size_t workerCount =
      wafer::kWorkerPlacementRankFrontierAdmissionLimit * 3;
  std::vector<RankVariantMetadataFrontier> frontiers(rankCount);
  for (size_t rank = 0; rank < rankCount; ++rank) {
    frontiers[rank].push_back(slot(0, RankArtifactKind::Spill, true));
    for (size_t worker = 0; worker + 1 < workerCount; ++worker)
      frontiers[rank].push_back(
          slot(1000 + static_cast<int64_t>(worker),
               RankArtifactKind::SpillReady, false, RankBufferingKind::Single,
               0, RankWorkerPlacementKind::DisjointComponents,
               static_cast<uint32_t>(worker) + 1));
    frontiers[rank].push_back(slot(5000, RankArtifactKind::Resident, false,
                                   RankBufferingKind::StaticFixedSlot, 7,
                                   RankWorkerPlacementKind::DisjointComponents,
                                   99));
  }

  WholeVariantAttemptPlan plan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers,
                                                            rankCount);
  ASSERT_TRUE(plan.isValid());
  ASSERT_EQ(plan.workerPlacedCandidateIndices.size(),
            wafer::kWorkerPlacementRankFrontierAdmissionLimit);
  const std::vector<size_t> tail(rankCount, workerCount);
  EXPECT_EQ(plan.workerPlacedCandidateIndices.back(), tail);
  EXPECT_NE(std::find(plan.optimizedCandidateIndices.begin(),
                      plan.optimizedCandidateIndices.end(), tail),
            plan.optimizedCandidateIndices.end());
  for (size_t rank = 0; rank < rankCount; ++rank)
    EXPECT_NE(std::find(plan.requiredModuleIndices[rank].begin(),
                        plan.requiredModuleIndices[rank].end(), workerCount),
              plan.requiredModuleIndices[rank].end());
}

TEST(WholeVariantAttemptPlanTest,
     IncompleteWorkerSeedRemainsMetadataOnlyDuringOwnerImport) {
  using wafer::compiler::detail::SerializedRankVariantCandidate;
  using wafer::compiler::detail::SerializedRankVariantFrontier;

  auto makeSerialized = [](int64_t ordinal, bool baseline, uint32_t workerPlan,
                           llvm::StringRef moduleData) {
    SerializedRankVariantCandidate candidate;
    candidate.moduleData =
        std::make_shared<const std::string>(moduleData.str());
    candidate.stableOrdinal = ordinal;
    candidate.artifactKind = RankArtifactKind::Spill;
    candidate.reservedBaseline = baseline;
    if (!baseline) {
      candidate.workerPlacementKind =
          RankWorkerPlacementKind::DisjointComponents;
      candidate.workerPlacementPlanOrdinal = workerPlan;
    }
    return candidate;
  };

  std::vector<SerializedRankVariantFrontier> serialized(2);
  serialized[0].push_back(makeSerialized(0, true, 0, "module {}"));
  serialized[0].push_back(
      makeSerialized(5, false, 1, "unrequired invalid MLIR"));
  serialized[1].push_back(makeSerialized(0, true, 0, "module {}"));
  serialized[1].push_back(
      makeSerialized(5, false, 2, "unrequired invalid MLIR"));

  mlir::MLIRContext ownerContext;
  auto imported =
      wafer::compiler::detail::importRankVariantFrontiersIntoOwnerContext(
          ownerContext, serialized, /*expectedRankCount=*/2);
  if (!imported) {
    ADD_FAILURE() << llvm::toString(imported.takeError());
    return;
  }
  for (const auto &frontier : *imported) {
    ASSERT_TRUE(frontier.front().module);
    EXPECT_FALSE(frontier.back().module);
  }
}

TEST(WholeVariantAttemptPlanTest,
     OwnerImportParsesOnlyAttemptPlanRequiredOriginalSlots) {
  using wafer::compiler::detail::SerializedRankVariantCandidate;
  using wafer::compiler::detail::SerializedRankVariantFrontier;

  auto serialized = [](llvm::StringRef moduleData, int64_t stableOrdinal,
                       RankArtifactKind artifactKind, bool reservedBaseline,
                       RankBufferingKind bufferingKind =
                           RankBufferingKind::Single,
                       uint32_t bufferingPlanOrdinal = 0,
                       RankWorkerPlacementKind workerPlacementKind =
                           RankWorkerPlacementKind::Unplaced,
                       uint32_t workerPlacementPlanOrdinal = 0) {
    SerializedRankVariantCandidate candidate;
    candidate.moduleData =
        std::make_shared<const std::string>(moduleData.str());
    candidate.stableOrdinal = stableOrdinal;
    candidate.artifactKind = artifactKind;
    candidate.reservedBaseline = reservedBaseline;
    candidate.bufferingKind = bufferingKind;
    candidate.bufferingPlanOrdinal = bufferingPlanOrdinal;
    candidate.workerPlacementKind = workerPlacementKind;
    candidate.workerPlacementPlanOrdinal = workerPlacementPlanOrdinal;
    return candidate;
  };

  std::vector<SerializedRankVariantFrontier> frontiers(2);
  frontiers[0] = {
      serialized("not valid MLIR and must remain unparsed", 10,
                 RankArtifactKind::Spill, false),
      serialized("module {}", 0, RankArtifactKind::Spill, true),
      serialized("module {}", 1, RankArtifactKind::Resident, false,
                 RankBufferingKind::StaticFixedSlot, 7,
                 RankWorkerPlacementKind::DisjointComponents, 1),
  };
  frontiers[1] = {
      serialized("also not valid MLIR and must remain unparsed", 20,
                 RankArtifactKind::Spill, false),
      serialized("module {}", 0, RankArtifactKind::Spill, true),
      serialized("module {}", 1, RankArtifactKind::Resident, false,
                 RankBufferingKind::StaticFixedSlot, 7,
                 RankWorkerPlacementKind::DisjointComponents, 1),
  };

  mlir::MLIRContext ownerContext;
  auto imported =
      wafer::compiler::detail::importRankVariantFrontiersIntoOwnerContext(
          ownerContext, frontiers, /*expectedRankCount=*/2);
  if (!imported) {
    ADD_FAILURE() << llvm::toString(imported.takeError());
    return;
  }

  ASSERT_EQ(imported->size(), 2u);
  size_t materializedModuleCount = 0;
  for (size_t rank = 0; rank < imported->size(); ++rank) {
    const wafer::compiler::detail::RankVariantFrontier &frontier =
        (*imported)[rank];
    ASSERT_EQ(frontier.size(), 3u);

    // The original mismatched slot and all metadata remain visible to the
    // fixed attempt walk, but its deliberately invalid text was never parsed.
    EXPECT_FALSE(frontier[0].module);
    EXPECT_EQ(frontier[0].stableOrdinal,
              rank == 0 ? static_cast<int64_t>(10) : static_cast<int64_t>(20));
    EXPECT_EQ(frontier[0].artifactKind, RankArtifactKind::Spill);
    EXPECT_FALSE(frontier[0].reservedBaseline);

    ASSERT_TRUE(frontier[1].module);
    EXPECT_EQ(frontier[1].stableOrdinal, 0);
    EXPECT_EQ(frontier[1].artifactKind, RankArtifactKind::Spill);
    EXPECT_TRUE(frontier[1].reservedBaseline);
    ASSERT_TRUE(frontier[2].module);
    EXPECT_EQ(frontier[2].stableOrdinal, 1);
    EXPECT_EQ(frontier[2].artifactKind, RankArtifactKind::Resident);
    EXPECT_FALSE(frontier[2].reservedBaseline);
    EXPECT_EQ(frontier[2].bufferingKind, RankBufferingKind::StaticFixedSlot);
    EXPECT_EQ(frontier[2].bufferingPlanOrdinal, 7u);
    EXPECT_EQ(frontier[2].workerPlacementKind,
              RankWorkerPlacementKind::DisjointComponents);
    EXPECT_EQ(frontier[2].workerPlacementPlanOrdinal, 1u);
    materializedModuleCount += frontier[1].module ? 1 : 0;
    materializedModuleCount += frontier[2].module ? 1 : 0;
  }
  EXPECT_EQ(materializedModuleCount, 4u);
}

TEST(WholeVariantAttemptPlanTest, RequiredInvalidOwnerImportModuleStillFails) {
  using wafer::compiler::detail::SerializedRankVariantCandidate;
  using wafer::compiler::detail::SerializedRankVariantFrontier;

  SerializedRankVariantCandidate required;
  required.moduleData = std::make_shared<const std::string>("not valid MLIR");
  required.stableOrdinal = 0;
  required.reservedBaseline = true;
  std::vector<SerializedRankVariantFrontier> frontiers(1);
  frontiers.front().push_back(std::move(required));

  mlir::MLIRContext ownerContext;
  auto imported =
      wafer::compiler::detail::importRankVariantFrontiersIntoOwnerContext(
          ownerContext, frontiers, /*expectedRankCount=*/1);
  ASSERT_FALSE(static_cast<bool>(imported));
  EXPECT_NE(llvm::toString(imported.takeError())
                .find("failed to import a lowered scheduling candidate"),
            std::string::npos);
}

TEST(WholeVariantAttemptPlanTest,
     RequiredOwnerImportAcceptsInvocationLocalMLIRBytecode) {
  using wafer::compiler::detail::SerializedRankVariantCandidate;
  using wafer::compiler::detail::SerializedRankVariantFrontier;

  mlir::MLIRContext workerContext;
  mlir::OwningOpRef<mlir::ModuleOp> workerModule =
      mlir::parseSourceString<mlir::ModuleOp>("module {}", &workerContext);
  ASSERT_TRUE(workerModule);

  SerializedRankVariantCandidate required;
  required.stableOrdinal = 0;
  required.reservedBaseline = true;
  std::string moduleData;
  llvm::raw_string_ostream bytecode(moduleData);
  ASSERT_TRUE(mlir::succeeded(
      mlir::writeBytecodeToFile(workerModule.get().getOperation(), bytecode)));
  bytecode.flush();
  required.moduleData =
      std::make_shared<const std::string>(std::move(moduleData));

  std::vector<SerializedRankVariantFrontier> frontiers(1);
  frontiers.front().push_back(std::move(required));
  mlir::MLIRContext ownerContext;
  auto imported =
      wafer::compiler::detail::importRankVariantFrontiersIntoOwnerContext(
          ownerContext, frontiers, /*expectedRankCount=*/1);
  if (!imported) {
    ADD_FAILURE() << llvm::toString(imported.takeError());
    return;
  }
  ASSERT_EQ(imported->size(), 1u);
  ASSERT_EQ(imported->front().size(), 1u);
  EXPECT_TRUE(imported->front().front().module);
}

TEST(WholeVariantAttemptPlanTest,
     CompactionKeepsPostImportCandidateReachableAfterQuantileReplanning) {
  using wafer::compiler::detail::RankVariantCandidate;
  using wafer::compiler::detail::SerializedRankVariantCandidate;
  using wafer::compiler::detail::SerializedRankVariantFrontier;

  constexpr size_t originalCandidateCount = 256;
  RankVariantMetadataFrontier metadata;
  metadata.reserve(originalCandidateCount);
  for (size_t index = 0; index < originalCandidateCount; ++index)
    metadata.push_back(
        slot(static_cast<int64_t>(index),
             index == 0 ? RankArtifactKind::Spill : RankArtifactKind::Resident,
             /*reservedBaseline=*/index == 0));
  WholeVariantAttemptPlan importPlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan({metadata}, 1);
  ASSERT_TRUE(importPlan.isValid());

  std::vector<bool> required(originalCandidateCount, false);
  for (size_t index : importPlan.requiredModuleIndices.front())
    required[index] = true;
  std::vector<SerializedRankVariantFrontier> serialized(1);
  for (size_t index = 0; index < originalCandidateCount; ++index) {
    SerializedRankVariantCandidate candidate;
    candidate.moduleData = std::make_shared<const std::string>(
        required[index] ? "module {}" : "deliberately invalid unrequired MLIR");
    candidate.stableOrdinal = static_cast<int64_t>(index);
    candidate.artifactKind =
        index == 0 ? RankArtifactKind::Spill : RankArtifactKind::Resident;
    candidate.reservedBaseline = index == 0;
    serialized.front().push_back(std::move(candidate));
  }

  mlir::MLIRContext ownerContext;
  auto imported =
      wafer::compiler::detail::importRankVariantFrontiersIntoOwnerContext(
          ownerContext, serialized, /*expectedRankCount=*/1);
  if (!imported) {
    ADD_FAILURE() << llvm::toString(imported.takeError());
    return;
  }
  wafer::compiler::detail::compactImportedRankVariantFrontiers(*imported);
  ASSERT_LT(imported->front().size(), originalCandidateCount);
  EXPECT_TRUE(llvm::all_of(imported->front(),
                           [](const RankVariantCandidate &candidate) {
                             return static_cast<bool>(candidate.module);
                           }));

  auto appendedModule =
      mlir::parseSourceString<mlir::ModuleOp>("module {}", &ownerContext);
  ASSERT_TRUE(appendedModule);
  imported->front().push_back({std::move(appendedModule), 1000,
                               RankArtifactKind::Resident,
                               /*reservedBaseline=*/false});

  std::vector<RankVariantMetadataFrontier> replanningMetadata(1);
  for (const RankVariantCandidate &candidate : imported->front())
    replanningMetadata.front().push_back(
        {candidate.stableOrdinal, candidate.artifactKind,
         candidate.reservedBaseline, candidate.bufferingKind,
         candidate.bufferingPlanOrdinal, candidate.workerPlacementKind,
         candidate.workerPlacementPlanOrdinal});
  WholeVariantAttemptPlan replanned =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(
          replanningMetadata, /*expectedRankCount=*/1);
  ASSERT_TRUE(replanned.isValid());
  EXPECT_TRUE(llvm::any_of(
      replanned.optimizedCandidateIndices,
      [&](const std::vector<size_t> &indices) {
        return indices.size() == 1 &&
               imported->front()[indices.front()].stableOrdinal == 1000;
      }));
}

TEST(WholeVariantAttemptPlanTest,
     MalformedMetadataConservativelyImportsEveryOriginalSlot) {
  std::vector<RankVariantMetadataFrontier> duplicateBaseline(16);
  for (RankVariantMetadataFrontier &frontier : duplicateBaseline) {
    frontier = {slot(0, RankArtifactKind::Spill, true),
                slot(1, RankArtifactKind::Resident)};
  }
  duplicateBaseline[5][1].reservedBaseline = true;
  WholeVariantAttemptPlan duplicatePlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(duplicateBaseline,
                                                            16);
  EXPECT_EQ(duplicatePlan.failure,
            WholeVariantAttemptPlanFailure::ReservedBaseline);
  EXPECT_EQ(duplicatePlan.getRequiredModuleCount(), 32u);
  for (const std::vector<size_t> &indices : duplicatePlan.requiredModuleIndices)
    EXPECT_EQ(indices, (std::vector<size_t>{0, 1}));

  std::vector<RankVariantMetadataFrontier> invalidOrdinal = duplicateBaseline;
  invalidOrdinal[5][1].reservedBaseline = false;
  invalidOrdinal[9][1].stableOrdinal = -1;
  WholeVariantAttemptPlan invalidPlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(invalidOrdinal, 16);
  EXPECT_EQ(invalidPlan.failure,
            WholeVariantAttemptPlanFailure::CandidateDomain);
  EXPECT_EQ(invalidPlan.getRequiredModuleCount(), 32u);

  std::vector<RankVariantMetadataFrontier> pipelinedBaseline(1);
  pipelinedBaseline[0] = {
      slot(0, RankArtifactKind::Spill, true, RankBufferingKind::StaticFixedSlot,
           /*bufferingPlanOrdinal=*/1),
  };
  WholeVariantAttemptPlan pipelinedBaselinePlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(pipelinedBaseline,
                                                            1);
  EXPECT_EQ(pipelinedBaselinePlan.failure,
            WholeVariantAttemptPlanFailure::ReservedBaseline);
  EXPECT_EQ(pipelinedBaselinePlan.getRequiredModuleCount(), 1u);

  std::vector<RankVariantMetadataFrontier> nonCanonicalSingle(1);
  nonCanonicalSingle[0] = {
      slot(0, RankArtifactKind::Spill, true, RankBufferingKind::Single,
           /*bufferingPlanOrdinal=*/9),
  };
  WholeVariantAttemptPlan nonCanonicalSinglePlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(nonCanonicalSingle,
                                                            1);
  EXPECT_EQ(nonCanonicalSinglePlan.failure,
            WholeVariantAttemptPlanFailure::CandidateDomain);
  EXPECT_EQ(nonCanonicalSinglePlan.getRequiredModuleCount(), 1u);

  std::vector<RankVariantMetadataFrontier> nonCanonicalFixed(1);
  nonCanonicalFixed[0] = {
      slot(0, RankArtifactKind::Spill, true, RankBufferingKind::StaticFixedSlot,
           /*bufferingPlanOrdinal=*/0),
  };
  WholeVariantAttemptPlan nonCanonicalFixedPlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(nonCanonicalFixed,
                                                            1);
  EXPECT_EQ(nonCanonicalFixedPlan.failure,
            WholeVariantAttemptPlanFailure::CandidateDomain);
  EXPECT_EQ(nonCanonicalFixedPlan.getRequiredModuleCount(), 1u);

  std::vector<RankVariantMetadataFrontier> workerPlacedBaseline(1);
  workerPlacedBaseline[0] = {
      slot(0, RankArtifactKind::Spill, true, RankBufferingKind::Single, 0,
           RankWorkerPlacementKind::DisjointComponents,
           /*workerPlacementPlanOrdinal=*/1),
  };
  WholeVariantAttemptPlan workerPlacedBaselinePlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(
          workerPlacedBaseline, 1);
  EXPECT_EQ(workerPlacedBaselinePlan.failure,
            WholeVariantAttemptPlanFailure::ReservedBaseline);
  EXPECT_EQ(workerPlacedBaselinePlan.getRequiredModuleCount(), 1u);

  std::vector<RankVariantMetadataFrontier> nonCanonicalUnplaced(1);
  nonCanonicalUnplaced[0] = {
      slot(0, RankArtifactKind::Spill, true, RankBufferingKind::Single, 0,
           RankWorkerPlacementKind::Unplaced,
           /*workerPlacementPlanOrdinal=*/9),
  };
  WholeVariantAttemptPlan nonCanonicalUnplacedPlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(
          nonCanonicalUnplaced, 1);
  EXPECT_EQ(nonCanonicalUnplacedPlan.failure,
            WholeVariantAttemptPlanFailure::CandidateDomain);
  EXPECT_EQ(nonCanonicalUnplacedPlan.getRequiredModuleCount(), 1u);

  std::vector<RankVariantMetadataFrontier> nonCanonicalWorkerPlaced(1);
  nonCanonicalWorkerPlaced[0] = {
      slot(0, RankArtifactKind::Spill, true, RankBufferingKind::Single, 0,
           RankWorkerPlacementKind::DisjointComponents,
           /*workerPlacementPlanOrdinal=*/0),
  };
  WholeVariantAttemptPlan nonCanonicalWorkerPlacedPlan =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(
          nonCanonicalWorkerPlaced, 1);
  EXPECT_EQ(nonCanonicalWorkerPlacedPlan.failure,
            WholeVariantAttemptPlanFailure::CandidateDomain);
  EXPECT_EQ(nonCanonicalWorkerPlacedPlan.getRequiredModuleCount(), 1u);
}

TEST(WholeVariantAttemptPlanTest, IsDeterministicAcrossRepeatedPlanning) {
  std::vector<RankVariantMetadataFrontier> frontiers(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank)
    frontiers[rank] = {slot(4, RankArtifactKind::Resident),
                       slot(0, RankArtifactKind::Spill, true),
                       slot(rank == 15 ? 8 : 7, RankArtifactKind::SpillReady),
                       slot(4, RankArtifactKind::Resident)};

  WholeVariantAttemptPlan first =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 16);
  WholeVariantAttemptPlan second =
      wafer::compiler::detail::buildWholeVariantAttemptPlan(frontiers, 16);
  ASSERT_TRUE(first.isValid());
  ASSERT_TRUE(second.isValid());
  EXPECT_EQ(first.reservedBaselineIndices, second.reservedBaselineIndices);
  EXPECT_EQ(first.optimizedCandidateIndices, second.optimizedCandidateIndices);
  EXPECT_EQ(first.workerPlacedCandidateIndices,
            second.workerPlacedCandidateIndices);
  EXPECT_EQ(first.fixedSlotCandidateIndices, second.fixedSlotCandidateIndices);
  EXPECT_EQ(first.genericCandidateIndices, second.genericCandidateIndices);
  EXPECT_EQ(first.requiredModuleIndices, second.requiredModuleIndices);
  EXPECT_EQ(first.boundedProductPositionCount,
            second.boundedProductPositionCount);
  EXPECT_EQ(first.boundedProductUniqueAttemptCount,
            second.boundedProductUniqueAttemptCount);
  EXPECT_EQ(first.coordinatedAttemptCount, second.coordinatedAttemptCount);
}

} // namespace
