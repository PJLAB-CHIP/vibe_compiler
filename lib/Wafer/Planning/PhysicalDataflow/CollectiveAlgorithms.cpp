//===- CollectiveAlgorithms.cpp - Generic collective algorithms ----------===//

#include "Wafer/Planning/PhysicalDataflow/CollectiveAlgorithms.h"

#include <algorithm>
#include <limits>
#include <set>
#include <vector>

namespace wafer::compiler::detail {

mlir::FailureOr<llvm::SmallVector<uint64_t, 16>> buildMinimumHopRing(
    llvm::ArrayRef<uint64_t> participants, uint64_t maximumParticipants,
    llvm::function_ref<std::optional<uint64_t>(uint64_t, uint64_t)>
        distanceOracle) {
  if (participants.size() < 2 || participants.size() > maximumParticipants ||
      participants.size() >= std::numeric_limits<size_t>::digits ||
      !distanceOracle)
    return mlir::failure();
  std::set<uint64_t> uniqueParticipants;
  for (uint64_t participant : participants)
    if (participant > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        !uniqueParticipants.insert(participant).second)
      return mlir::failure();

  const size_t count = participants.size();
  std::vector<std::vector<uint64_t>> distances(count,
                                               std::vector<uint64_t>(count));
  for (size_t lhs = 0; lhs < count; ++lhs)
    for (size_t rhs = 0; rhs < count; ++rhs) {
      if (lhs == rhs)
        continue;
      std::optional<uint64_t> distance =
          distanceOracle(participants[lhs], participants[rhs]);
      if (!distance)
        return mlir::failure();
      distances[lhs][rhs] = *distance;
    }

  const size_t movable = count - 1;
  const size_t stateCount = size_t{1} << movable;
  const uint64_t infinity = std::numeric_limits<uint64_t>::max();
  std::vector<uint64_t> costs(stateCount * movable, infinity);
  std::vector<int16_t> predecessors(stateCount * movable, -1);
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
        int16_t &bestPrevious = predecessors[cell(mask, node)];
        if (candidate < best ||
            (candidate == best &&
             (bestPrevious < 0 ||
              participants[previous] <
                  participants[static_cast<size_t>(bestPrevious)]))) {
          best = candidate;
          bestPrevious = static_cast<int16_t>(previous);
        }
      }
    }

  const size_t fullMask = stateCount - 1;
  uint64_t bestCycle = infinity;
  int16_t bestEnd = -1;
  for (size_t node = 1; node < count; ++node) {
    uint64_t path = costs[cell(fullMask, node)];
    if (path == infinity || distances[node][0] > infinity - path)
      continue;
    uint64_t cycle = path + distances[node][0];
    if (cycle < bestCycle ||
        (cycle == bestCycle &&
         (bestEnd < 0 ||
          participants[node] < participants[static_cast<size_t>(bestEnd)]))) {
      bestCycle = cycle;
      bestEnd = static_cast<int16_t>(node);
    }
  }
  if (bestEnd < 0)
    return mlir::failure();

  llvm::SmallVector<size_t, 16> indices(count);
  indices.front() = 0;
  size_t mask = fullMask;
  int16_t current = bestEnd;
  for (size_t position = count - 1; position > 0; --position) {
    if (current <= 0)
      return mlir::failure();
    indices[position] = static_cast<size_t>(current);
    int16_t previous = predecessors[cell(mask, static_cast<size_t>(current))];
    mask ^= size_t{1} << (static_cast<size_t>(current) - 1);
    current = previous;
  }
  if (mask != 0)
    return mlir::failure();

  llvm::SmallVector<uint64_t, 16> ring;
  for (size_t index : indices)
    ring.push_back(participants[index]);
  llvm::SmallVector<uint64_t, 16> reversed(ring.size());
  reversed.front() = ring.front();
  for (size_t index = 1; index < ring.size(); ++index)
    reversed[index] = ring[ring.size() - index];
  if (std::lexicographical_compare(reversed.begin(), reversed.end(),
                                   ring.begin(), ring.end()))
    ring = std::move(reversed);
  return ring;
}

} // namespace wafer::compiler::detail
