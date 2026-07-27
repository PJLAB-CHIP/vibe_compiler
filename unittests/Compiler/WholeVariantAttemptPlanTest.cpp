//===- WholeVariantAttemptPlanTest.cpp ----------------------------------===//

#include "../../lib/Wafer/Compiler/WholeVariantAttemptPlan.h"
#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using wafer::RankArtifactKind;
using wafer::compiler::detail::RankVariantMetadataFrontier;
using wafer::compiler::detail::RankVariantSlotMetadata;
using wafer::compiler::detail::WholeVariantAttemptPlan;
using wafer::compiler::detail::WholeVariantAttemptPlanFailure;

constexpr size_t kLegacyProductLimit = 64;
constexpr size_t kLegacyCoordinatedLimit = 64;

using CandidateOrder = std::vector<std::vector<size_t>>;
using CorrespondenceKey = std::pair<int64_t, RankArtifactKind>;

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

/// Independent copy of the eager coordinator enumeration that existed before
/// context-transfer filtering. Tests compare every original slot index, not
/// only the correspondence key or final candidate count.
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
                return std::tie(left.stableOrdinal, left.artifactKind, lhs) <
                       std::tie(right.stableOrdinal, right.artifactKind, rhs);
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
    keys.insert({candidate.stableOrdinal, candidate.artifactKind});
  for (CorrespondenceKey key : keys) {
    if (reference.coordinatedAttemptCount >= kLegacyCoordinatedLimit)
      break;
    std::vector<size_t> indices;
    bool complete = true;
    for (const RankVariantMetadataFrontier &frontier : frontiers) {
      std::optional<size_t> match;
      for (size_t index = 0; index < frontier.size(); ++index) {
        const RankVariantSlotMetadata &candidate = frontier[index];
        if (candidate.stableOrdinal != key.first ||
            candidate.artifactKind != key.second)
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
    if (!complete || !attempted.insert(indices).second)
      continue;
    ++reference.coordinatedAttemptCount;
    reference.optimizedCandidateIndices.push_back(std::move(indices));
  }
  return reference;
}

static bool hasCompleteCorrespondence(
    const std::vector<size_t> &indices,
    const std::vector<RankVariantMetadataFrontier> &frontiers) {
  std::optional<CorrespondenceKey> key;
  for (size_t rank = 0; rank < indices.size(); ++rank) {
    const RankVariantSlotMetadata &candidate = frontiers[rank][indices[rank]];
    CorrespondenceKey current{candidate.stableOrdinal, candidate.artifactKind};
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
     bool reservedBaseline = false) {
  return {ordinal, kind, reservedBaseline};
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
     MatchesLegacyLimitsWithMissingReorderedAndDuplicateKeys) {
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
  EXPECT_EQ(plan.optimizedCandidateIndices,
            reference.optimizedCandidateIndices);
  EXPECT_EQ(plan.requiredModuleIndices,
            requiredByLegacyAttempts(reference, frontiers));

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
     OwnerImportParsesOnlyAttemptPlanRequiredOriginalSlots) {
  using wafer::compiler::detail::SerializedRankVariantCandidate;
  using wafer::compiler::detail::SerializedRankVariantFrontier;

  auto serialized = [](llvm::StringRef moduleText, int64_t stableOrdinal,
                       RankArtifactKind artifactKind, bool reservedBaseline) {
    SerializedRankVariantCandidate candidate;
    candidate.moduleText = moduleText.str();
    candidate.stableOrdinal = stableOrdinal;
    candidate.artifactKind = artifactKind;
    candidate.reservedBaseline = reservedBaseline;
    return candidate;
  };

  std::vector<SerializedRankVariantFrontier> frontiers(2);
  frontiers[0] = {
      serialized("not valid MLIR and must remain unparsed", 10,
                 RankArtifactKind::Spill, false),
      serialized("module {}", 0, RankArtifactKind::Spill, true),
      serialized("module {}", 1, RankArtifactKind::Resident, false),
  };
  frontiers[1] = {
      serialized("also not valid MLIR and must remain unparsed", 20,
                 RankArtifactKind::Spill, false),
      serialized("module {}", 0, RankArtifactKind::Spill, true),
      serialized("module {}", 1, RankArtifactKind::Resident, false),
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
    materializedModuleCount += frontier[1].module ? 1 : 0;
    materializedModuleCount += frontier[2].module ? 1 : 0;
  }
  EXPECT_EQ(materializedModuleCount, 4u);
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
  EXPECT_EQ(first.requiredModuleIndices, second.requiredModuleIndices);
  EXPECT_EQ(first.boundedProductPositionCount,
            second.boundedProductPositionCount);
  EXPECT_EQ(first.boundedProductUniqueAttemptCount,
            second.boundedProductUniqueAttemptCount);
  EXPECT_EQ(first.coordinatedAttemptCount, second.coordinatedAttemptCount);
}

} // namespace
