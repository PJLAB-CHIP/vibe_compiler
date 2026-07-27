//===- WholeVariantAttemptPlan.cpp - Invocation-local tuple plan --------===//

#include "WholeVariantAttemptPlan.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

constexpr size_t kWholeVariantVisitLimit = 64;
constexpr size_t kCoordinatedPolicyVisitLimit = 64;

using CandidateOrder = std::vector<std::vector<size_t>>;
using CorrespondenceKey = std::pair<int64_t, wafer::RankArtifactKind>;

struct Combination {
  std::vector<size_t> positions;
};

struct WorseCombination {
  bool operator()(const Combination &lhs, const Combination &rhs) const {
    return lhs.positions > rhs.positions;
  }
};

static void
requireEveryModule(llvm::ArrayRef<RankVariantMetadataFrontier> frontiers,
                   WholeVariantAttemptPlan &plan) {
  plan.requiredModuleIndices.clear();
  plan.requiredModuleIndices.resize(frontiers.size());
  for (auto [rank, frontier] : llvm::enumerate(frontiers)) {
    plan.requiredModuleIndices[rank].resize(frontier.size());
    for (size_t index = 0; index < frontier.size(); ++index)
      plan.requiredModuleIndices[rank][index] = index;
  }
}

static std::vector<size_t> getCandidateIndices(llvm::ArrayRef<size_t> positions,
                                               const CandidateOrder &order) {
  std::vector<size_t> indices;
  indices.reserve(positions.size());
  for (size_t rank = 0; rank < positions.size(); ++rank)
    indices.push_back(order[rank][positions[rank]]);
  return indices;
}

static bool hasCompleteCorrespondence(
    llvm::ArrayRef<size_t> candidateIndices,
    llvm::ArrayRef<RankVariantMetadataFrontier> frontiers) {
  if (candidateIndices.size() != frontiers.size() || candidateIndices.empty())
    return false;
  std::optional<CorrespondenceKey> key;
  for (auto [rank, candidateIndex] : llvm::enumerate(candidateIndices)) {
    if (candidateIndex >= frontiers[rank].size())
      return false;
    const RankVariantSlotMetadata &candidate = frontiers[rank][candidateIndex];
    CorrespondenceKey current{candidate.stableOrdinal, candidate.artifactKind};
    if (key && current != *key)
      return false;
    key = current;
  }
  return true;
}

} // namespace

size_t WholeVariantAttemptPlan::getRequiredModuleCount() const {
  size_t count = 0;
  for (const std::vector<size_t> &indices : requiredModuleIndices)
    count += indices.size();
  return count;
}

WholeVariantAttemptPlan buildWholeVariantAttemptPlan(
    llvm::ArrayRef<RankVariantMetadataFrontier> frontiers,
    int64_t expectedRankCount) {
  WholeVariantAttemptPlan plan;
  if (expectedRankCount <= 0 ||
      frontiers.size() != static_cast<size_t>(expectedRankCount)) {
    plan.failure = WholeVariantAttemptPlanFailure::CandidateDomain;
    requireEveryModule(frontiers, plan);
    return plan;
  }

  CandidateOrder order(frontiers.size());
  for (auto [rank, frontier] : llvm::enumerate(frontiers)) {
    if (frontier.empty()) {
      plan.failure = WholeVariantAttemptPlanFailure::CandidateDomain;
      requireEveryModule(frontiers, plan);
      return plan;
    }
    order[rank].resize(frontier.size());
    for (size_t index = 0; index < frontier.size(); ++index) {
      if (frontier[index].stableOrdinal < 0) {
        plan.failure = WholeVariantAttemptPlanFailure::CandidateDomain;
        requireEveryModule(frontiers, plan);
        return plan;
      }
      order[rank][index] = index;
    }
    llvm::sort(order[rank], [&](size_t lhs, size_t rhs) {
      const RankVariantSlotMetadata &left = frontier[lhs];
      const RankVariantSlotMetadata &right = frontier[rhs];
      return std::tie(left.stableOrdinal, left.artifactKind, lhs) <
             std::tie(right.stableOrdinal, right.artifactKind, rhs);
    });
  }

  plan.reservedBaselineIndices.reserve(frontiers.size());
  for (const RankVariantMetadataFrontier &frontier : frontiers) {
    std::optional<size_t> baseline;
    for (auto [index, candidate] : llvm::enumerate(frontier)) {
      if (!candidate.reservedBaseline)
        continue;
      if (baseline) {
        plan.failure = WholeVariantAttemptPlanFailure::ReservedBaseline;
        requireEveryModule(frontiers, plan);
        return plan;
      }
      baseline = index;
    }
    if (!baseline) {
      plan.failure = WholeVariantAttemptPlanFailure::ReservedBaseline;
      requireEveryModule(frontiers, plan);
      return plan;
    }
    plan.reservedBaselineIndices.push_back(*baseline);
  }

  std::set<std::vector<size_t>> enqueuedPositions;
  std::set<std::vector<size_t>> attemptedCandidateIndices;
  attemptedCandidateIndices.insert(plan.reservedBaselineIndices);
  std::priority_queue<Combination, std::vector<Combination>, WorseCombination>
      queue;
  std::vector<size_t> initial(frontiers.size(), 0);
  queue.push({initial});
  enqueuedPositions.insert(initial);

  size_t visited = 0;
  while (!queue.empty() && visited < kWholeVariantVisitLimit) {
    Combination combination = queue.top();
    queue.pop();
    ++visited;
    std::vector<size_t> candidateIndices =
        getCandidateIndices(combination.positions, order);
    if (attemptedCandidateIndices.insert(candidateIndices).second)
      plan.optimizedCandidateIndices.push_back(std::move(candidateIndices));

    for (size_t rank = 0; rank < combination.positions.size(); ++rank) {
      std::vector<size_t> neighbor = combination.positions;
      if (++neighbor[rank] >= order[rank].size())
        continue;
      if (!enqueuedPositions.insert(neighbor).second)
        continue;
      queue.push({std::move(neighbor)});
    }
  }
  plan.boundedProductPositionCount = visited;
  plan.boundedProductUniqueAttemptCount = plan.optimizedCandidateIndices.size();

  std::set<CorrespondenceKey> correspondenceKeys;
  for (const RankVariantSlotMetadata &candidate : frontiers.front())
    correspondenceKeys.insert(
        {candidate.stableOrdinal, candidate.artifactKind});
  size_t coordinatedVisited = 0;
  for (CorrespondenceKey key : correspondenceKeys) {
    if (coordinatedVisited >= kCoordinatedPolicyVisitLimit)
      break;
    std::vector<size_t> candidateIndices;
    candidateIndices.reserve(frontiers.size());
    bool complete = true;
    for (const RankVariantMetadataFrontier &frontier : frontiers) {
      std::optional<size_t> match;
      for (auto [index, candidate] : llvm::enumerate(frontier)) {
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
      candidateIndices.push_back(*match);
    }
    if (!complete || !attemptedCandidateIndices.insert(candidateIndices).second)
      continue;
    ++coordinatedVisited;
    plan.optimizedCandidateIndices.push_back(std::move(candidateIndices));
  }
  plan.coordinatedAttemptCount = coordinatedVisited;

  std::vector<std::vector<bool>> required(frontiers.size());
  for (auto [rank, frontier] : llvm::enumerate(frontiers))
    required[rank].assign(frontier.size(), false);
  for (auto [rank, index] : llvm::enumerate(plan.reservedBaselineIndices))
    required[rank][index] = true;
  for (const std::vector<size_t> &candidateIndices :
       plan.optimizedCandidateIndices) {
    if (!hasCompleteCorrespondence(candidateIndices, frontiers))
      continue;
    for (auto [rank, index] : llvm::enumerate(candidateIndices))
      required[rank][index] = true;
  }

  plan.requiredModuleIndices.resize(frontiers.size());
  for (auto [rank, frontier] : llvm::enumerate(frontiers))
    for (size_t index = 0; index < frontier.size(); ++index)
      if (required[rank][index])
        plan.requiredModuleIndices[rank].push_back(index);
  return plan;
}

} // namespace wafer::compiler::detail
