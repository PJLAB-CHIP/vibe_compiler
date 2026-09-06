//===- CollectiveAlgorithms.cpp - Generic collective algorithms ----------===//

#include "Wafer/Planning/PhysicalDataflow/CollectiveAlgorithms.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace wafer::compiler::detail {

mlir::FailureOr<llvm::SmallVector<uint64_t, 16>> buildMinimumHopRing(
    llvm::ArrayRef<uint64_t> participants, uint64_t maximumParticipants,
    llvm::function_ref<std::optional<uint64_t>(uint64_t, uint64_t)>
        distanceOracle) {
  // Held--Karp is exact but exponential.  The bound is an algorithmic
  // resource contract, not a target-card limit; larger collectives must use a
  // different bounded algorithm rather than exhausting compiler memory.
  constexpr size_t kMaximumExactParticipants = 20;
  if (participants.size() < 2 || participants.size() > maximumParticipants ||
      participants.size() > kMaximumExactParticipants || !distanceOracle)
    return mlir::failure();
  const size_t count = participants.size();
  llvm::SmallVector<uint64_t, 20> orderedParticipants(participants.begin(),
                                                       participants.end());
  std::sort(orderedParticipants.begin(), orderedParticipants.end());
  if (std::adjacent_find(orderedParticipants.begin(),
                         orderedParticipants.end()) !=
      orderedParticipants.end())
    return mlir::failure();
  std::vector<std::vector<uint64_t>> distances(count,
                                               std::vector<uint64_t>(count));
  for (size_t lhs = 0; lhs < count; ++lhs)
    for (size_t rhs = 0; rhs < count; ++rhs) {
      if (lhs == rhs)
        continue;
      std::optional<uint64_t> distance =
          distanceOracle(orderedParticipants[lhs], orderedParticipants[rhs]);
      if (!distance)
        return mlir::failure();
      distances[lhs][rhs] = *distance;
    }
  bool symmetric = true;
  for (size_t lhs = 0; lhs < count && symmetric; ++lhs)
    for (size_t rhs = lhs + 1; rhs < count; ++rhs)
      if (distances[lhs][rhs] != distances[rhs][lhs]) {
        symmetric = false;
        break;
      }

  const size_t movable = count - 1;
  const size_t stateCount = size_t{1} << movable;
  const uint64_t infinity = std::numeric_limits<uint64_t>::max();
  std::vector<uint64_t> costs(stateCount * movable, infinity);
  std::vector<int32_t> predecessors(stateCount * movable, -1);
  auto cell = [&](size_t mask, size_t node) {
    return mask * movable + node - 1;
  };
  for (size_t node = 1; node < count; ++node)
    costs[cell(size_t{1} << (node - 1), node)] = distances[0][node];
  for (size_t mask = 1; mask < stateCount; ++mask)
    for (size_t node = 1; node < count; ++node) {
      const size_t nodeBit = size_t{1} << (node - 1);
      if ((mask & nodeBit) == 0)
        continue;
      const size_t previousMask = mask ^ nodeBit;
      if (previousMask == 0)
        continue;
      for (size_t previous = 1; previous < count; ++previous) {
        if ((previousMask & (size_t{1} << (previous - 1))) == 0)
          continue;
        uint64_t previousCost = costs[cell(previousMask, previous)];
        if (previousCost == infinity ||
            distances[previous][node] > infinity - previousCost)
          continue;
        uint64_t candidate = previousCost + distances[previous][node];
        uint64_t &best = costs[cell(mask, node)];
        int32_t &bestPrevious = predecessors[cell(mask, node)];
        if (candidate < best ||
            (candidate == best &&
             (bestPrevious < 0 ||
              orderedParticipants[previous] <
                  orderedParticipants[static_cast<size_t>(bestPrevious)]))) {
          best = candidate;
          bestPrevious = static_cast<int16_t>(previous);
        }
      }
    }

  const size_t fullMask = stateCount - 1;
  uint64_t bestCycle = infinity;
  int32_t bestEnd = -1;
  for (size_t node = 1; node < count; ++node) {
    uint64_t path = costs[cell(fullMask, node)];
    if (path == infinity || distances[node][0] > infinity - path)
      continue;
    uint64_t cycle = path + distances[node][0];
    if (cycle < bestCycle ||
        (cycle == bestCycle &&
         (bestEnd < 0 ||
          orderedParticipants[node] <
              orderedParticipants[static_cast<size_t>(bestEnd)]))) {
      bestCycle = cycle;
      bestEnd = static_cast<int16_t>(node);
    }
  }
  if (bestEnd < 0)
    return mlir::failure();

  llvm::SmallVector<size_t, 16> indices(count);
  indices.front() = 0;
  size_t mask = fullMask;
  int32_t current = bestEnd;
  for (size_t position = count - 1; position > 0; --position) {
    if (current <= 0)
      return mlir::failure();
    indices[position] = static_cast<size_t>(current);
    int32_t previous = predecessors[cell(mask, static_cast<size_t>(current))];
    mask ^= size_t{1} << (static_cast<size_t>(current) - 1);
    current = previous;
  }
  if (mask != 0)
    return mlir::failure();

  llvm::SmallVector<uint64_t, 16> ring;
  for (size_t index : indices)
    ring.push_back(orderedParticipants[index]);
  // Reversing a directed cycle is generally a different cost.  It is only a
  // valid canonicalization when the supplied oracle proved symmetry.
  if (symmetric) {
    llvm::SmallVector<uint64_t, 16> reversed(ring.size());
    reversed.front() = ring.front();
    for (size_t index = 1; index < ring.size(); ++index)
      reversed[index] = ring[ring.size() - index];
    if (std::lexicographical_compare(reversed.begin(), reversed.end(),
                                     ring.begin(), ring.end()))
      ring = std::move(reversed);
  }
  return ring;
}

} // namespace wafer::compiler::detail
