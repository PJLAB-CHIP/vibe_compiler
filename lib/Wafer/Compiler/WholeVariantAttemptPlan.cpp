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

constexpr size_t kWholeVariantVisitLimit =
    WholeVariantAttemptPlan::kCartesianVisitLimit;
constexpr size_t kCoordinatedPolicyVisitLimit =
    WholeVariantAttemptPlan::kCoordinatedVisitLimit;
constexpr size_t kWorkerPlacedVisitLimit =
    wafer::kWorkerPlacementRankFrontierAdmissionLimit;
constexpr size_t kFixedSlotVisitLimit =
    wafer::kFixedSlotRankFrontierAdmissionLimit;
constexpr size_t kGenericCorrespondenceVisitLimit =
    WholeVariantAttemptPlan::kGenericCorrespondenceVisitLimit;

using CandidateOrder = std::vector<std::vector<size_t>>;
using CorrespondenceKey =
    std::tuple<int64_t, wafer::RankArtifactKind, wafer::RankBufferingKind,
               uint32_t, wafer::RankWorkerPlacementKind, uint32_t>;

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
    CorrespondenceKey current{
        candidate.stableOrdinal,       candidate.artifactKind,
        candidate.bufferingKind,       candidate.bufferingPlanOrdinal,
        candidate.workerPlacementKind, candidate.workerPlacementPlanOrdinal};
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
      bool canonicalBufferingIdentity =
          (frontier[index].bufferingKind == wafer::RankBufferingKind::Single) ==
          (frontier[index].bufferingPlanOrdinal == 0);
      bool canonicalWorkerPlacementIdentity =
          (frontier[index].workerPlacementKind ==
           wafer::RankWorkerPlacementKind::Unplaced) ==
          (frontier[index].workerPlacementPlanOrdinal == 0);
      if (frontier[index].stableOrdinal < 0 || !canonicalBufferingIdentity ||
          !canonicalWorkerPlacementIdentity) {
        plan.failure = WholeVariantAttemptPlanFailure::CandidateDomain;
        requireEveryModule(frontiers, plan);
        return plan;
      }
      order[rank][index] = index;
    }
    llvm::sort(order[rank], [&](size_t lhs, size_t rhs) {
      const RankVariantSlotMetadata &left = frontier[lhs];
      const RankVariantSlotMetadata &right = frontier[rhs];
      return std::tie(left.stableOrdinal, left.artifactKind, left.bufferingKind,
                      left.bufferingPlanOrdinal, left.workerPlacementKind,
                      left.workerPlacementPlanOrdinal, lhs) <
             std::tie(right.stableOrdinal, right.artifactKind,
                      right.bufferingKind, right.bufferingPlanOrdinal,
                      right.workerPlacementKind,
                      right.workerPlacementPlanOrdinal, rhs);
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
      if (candidate.bufferingKind != wafer::RankBufferingKind::Single ||
          candidate.workerPlacementKind !=
              wafer::RankWorkerPlacementKind::Unplaced) {
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
        {candidate.stableOrdinal, candidate.artifactKind,
         candidate.bufferingKind, candidate.bufferingPlanOrdinal,
         candidate.workerPlacementKind, candidate.workerPlacementPlanOrdinal});
  std::vector<std::vector<size_t>> coordinatedCandidates;
  for (CorrespondenceKey key : correspondenceKeys) {
    std::vector<size_t> candidateIndices;
    candidateIndices.reserve(frontiers.size());
    bool complete = true;
    for (const RankVariantMetadataFrontier &frontier : frontiers) {
      std::optional<size_t> match;
      for (auto [index, candidate] : llvm::enumerate(frontier)) {
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
      candidateIndices.push_back(*match);
    }
    if (!complete || attemptedCandidateIndices.count(candidateIndices))
      continue;
    coordinatedCandidates.push_back(std::move(candidateIndices));
  }

  // Cover the complete correspondence domain at deterministic, evenly spaced
  // quantiles. A prefix-only cap permanently starves every post-frontier
  // candidate once an upstream producer contributes 64 keys, irrespective of
  // its actual resource cost; a head/tail split merely moves that starvation
  // into the middle. Quantile coverage retains the same total budget, always
  // includes both endpoints, and bounds the unvisited key gap without
  // inferring candidate semantics from names or workload-specific roles.
  auto appendCoordinated = [&](size_t index) {
    std::vector<size_t> candidateIndices =
        std::move(coordinatedCandidates[index]);
    if (!attemptedCandidateIndices.insert(candidateIndices).second)
      return;
    plan.optimizedCandidateIndices.push_back(std::move(candidateIndices));
    ++plan.coordinatedAttemptCount;
  };
  if (coordinatedCandidates.size() <= kCoordinatedPolicyVisitLimit) {
    for (size_t index = 0; index < coordinatedCandidates.size(); ++index)
      appendCoordinated(index);
  } else {
    const size_t span = coordinatedCandidates.size() - 1;
    const size_t intervals = kCoordinatedPolicyVisitLimit - 1;
    const size_t quotient = span / intervals;
    const size_t remainder = span % intervals;
    for (size_t sample = 0; sample < kCoordinatedPolicyVisitLimit; ++sample) {
      // Equivalent to floor(sample * span / intervals), expressed without a
      // potentially overflowing multiplication by the candidate count.
      const size_t index = sample * quotient + (sample * remainder) / intervals;
      appendCoordinated(index);
    }
  }

  // Physical realizations have the same orthogonal admission bands used by
  // RankFrontierAdmissionState: a worker-placed fixed-slot tuple belongs to
  // the worker band, otherwise fixed-slot has its own band. Ordinary
  // Single+Unplaced tuples have an independent bounded correspondence band so
  // a saturated pair of physical bands cannot starve generic NoC seeds.
  //
  // First collect the complete tuple domain for each band, then sample that
  // domain at deterministic quantiles. Prefix truncation would permanently
  // starve every high-stable-ordinal physical realization once an upstream
  // stage filled the same band. Quantiles retain the exact cap while including
  // both ends and bounding every unvisited gap, without inspecting artifact
  // roles or workload-specific semantics.
  std::vector<std::vector<size_t>> workerPlacedCandidates;
  std::vector<std::vector<size_t>> fixedSlotCandidates;
  std::vector<std::vector<size_t>> genericCandidates;
  for (CorrespondenceKey key : correspondenceKeys) {
    const bool workerPlaced =
        std::get<4>(key) ==
            wafer::RankWorkerPlacementKind::DisjointComponents &&
        std::get<5>(key) != 0;
    const bool fixedSlot =
        std::get<2>(key) == wafer::RankBufferingKind::StaticFixedSlot &&
        std::get<3>(key) != 0;
    const bool generic =
        std::get<2>(key) == wafer::RankBufferingKind::Single &&
        std::get<3>(key) == 0 &&
        std::get<4>(key) == wafer::RankWorkerPlacementKind::Unplaced &&
        std::get<5>(key) == 0;
    if (!workerPlaced && !fixedSlot && !generic)
      continue;
    std::vector<size_t> candidateIndices;
    candidateIndices.reserve(frontiers.size());
    bool complete = true;
    for (const RankVariantMetadataFrontier &frontier : frontiers) {
      std::optional<size_t> match;
      for (auto [index, candidate] : llvm::enumerate(frontier)) {
        if (candidate.reservedBaseline ||
            candidate.stableOrdinal != std::get<0>(key) ||
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
      candidateIndices.push_back(*match);
    }
    if (!complete)
      continue;
    if (workerPlaced)
      workerPlacedCandidates.push_back(std::move(candidateIndices));
    else if (fixedSlot)
      fixedSlotCandidates.push_back(std::move(candidateIndices));
    else
      genericCandidates.push_back(std::move(candidateIndices));
  }
  auto appendPhysicalBand = [&](std::vector<std::vector<size_t>> &candidates,
                                size_t limit,
                                std::vector<std::vector<size_t>> &band) {
    if (candidates.empty() || limit == 0)
      return;
    auto appendCandidate = [&](size_t index) {
      std::vector<size_t> candidateIndices = std::move(candidates[index]);
      band.push_back(candidateIndices);
      if (attemptedCandidateIndices.insert(candidateIndices).second)
        plan.optimizedCandidateIndices.push_back(std::move(candidateIndices));
    };
    if (candidates.size() <= limit) {
      for (size_t index = 0; index < candidates.size(); ++index)
        appendCandidate(index);
      return;
    }
    if (limit == 1) {
      appendCandidate(0);
      return;
    }
    const size_t span = candidates.size() - 1;
    const size_t intervals = limit - 1;
    const size_t quotient = span / intervals;
    const size_t remainder = span % intervals;
    for (size_t sample = 0; sample < limit; ++sample) {
      const size_t index = sample * quotient + (sample * remainder) / intervals;
      appendCandidate(index);
    }
  };
  appendPhysicalBand(workerPlacedCandidates, kWorkerPlacedVisitLimit,
                     plan.workerPlacedCandidateIndices);
  appendPhysicalBand(fixedSlotCandidates, kFixedSlotVisitLimit,
                     plan.fixedSlotCandidateIndices);
  appendPhysicalBand(genericCandidates, kGenericCorrespondenceVisitLimit,
                     plan.genericCandidateIndices);

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
