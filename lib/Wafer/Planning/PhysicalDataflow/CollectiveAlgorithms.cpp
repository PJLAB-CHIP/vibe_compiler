//===- CollectiveAlgorithms.cpp - Generic collective algorithms ----------===//

#include "Wafer/Planning/PhysicalDataflow/CollectiveAlgorithms.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

namespace wafer::compiler::detail {

mlir::FailureOr<llvm::SmallVector<DimensionOrderedAllToAllStep, 64>>
buildDimensionOrderedAllToAll(llvm::ArrayRef<uint64_t> rowMajorParticipants,
                              uint64_t rows, uint64_t columns) {
  if (rows == 0 || columns == 0 || rows > std::numeric_limits<size_t>::max() /
                                      columns ||
      rowMajorParticipants.size() !=
          static_cast<size_t>(rows * columns))
    return mlir::failure();
  llvm::SmallVector<uint64_t, 64> participants(rowMajorParticipants.begin(),
                                               rowMajorParticipants.end());
  std::sort(participants.begin(), participants.end());
  if (std::adjacent_find(participants.begin(), participants.end()) !=
      participants.end())
    return mlir::failure();

  llvm::SmallVector<DimensionOrderedAllToAllStep, 64> steps;
  for (uint64_t sourceIndex = 0; sourceIndex < rows * columns;
       ++sourceIndex) {
    const uint64_t source = rowMajorParticipants[sourceIndex];
    const uint64_t sourceRow = sourceIndex / columns;
    for (uint64_t destinationIndex = 0; destinationIndex < rows * columns;
         ++destinationIndex) {
      if (sourceIndex == destinationIndex)
        continue;
      const uint64_t destination = rowMajorParticipants[destinationIndex];
      const uint64_t destinationRow = destinationIndex / columns;
      const uint64_t destinationColumn = destinationIndex % columns;
      const uint64_t relayIndex = sourceRow * columns + destinationColumn;
      const uint64_t relay = rowMajorParticipants[relayIndex];
      if (relay != source)
        steps.push_back({source, relay, relay, /*dimension=*/0,
                         static_cast<uint32_t>(destinationColumn)});
      if (relay != destination)
        steps.push_back({relay, relay, destination, /*dimension=*/1,
                         static_cast<uint32_t>(destinationRow)});
    }
  }
  return steps;
}

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
          bestPrevious = static_cast<int32_t>(previous);
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
      bestEnd = static_cast<int32_t>(node);
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

mlir::FailureOr<llvm::SmallVector<BroadcastTreeEdge, 16>>
buildMinimumHopBroadcastTree(
    llvm::ArrayRef<uint64_t> participants,
    llvm::function_ref<std::optional<uint64_t>(uint64_t, uint64_t)>
        distanceOracle) {
  if (participants.size() < 2 || !distanceOracle)
    return mlir::failure();
  llvm::SmallVector<uint64_t, 16> ordered(participants);
  std::sort(ordered.begin(), ordered.end());
  if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end())
    return mlir::failure();
  const size_t count = ordered.size();
  std::vector<bool> reached(count, false);
  std::vector<uint64_t> degree(count, 0), depth(count, 0);
  reached[0] = true;
  llvm::SmallVector<BroadcastTreeEdge, 16> result;
  while (result.size() + 1 < count) {
    std::optional<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>>
        best;
    size_t source = 0, dest = 0;
    for (size_t a = 0; a < count; ++a)
      if (reached[a])
        for (size_t b = 0; b < count; ++b)
          if (!reached[b]) {
            auto forward = distanceOracle(ordered[a], ordered[b]);
            auto backward = distanceOracle(ordered[b], ordered[a]);
            if (!forward || !backward || *forward != *backward)
              return mlir::failure();
            auto score = std::make_tuple(*forward, degree[a], depth[a],
                                         ordered[a], ordered[b]);
            if (!best || score < *best) {
              best = score;
              source = a;
              dest = b;
            }
          }
    if (!best)
      return mlir::failure();
    result.push_back({ordered[source], ordered[dest]});
    reached[dest] = true;
    ++degree[source];
    depth[dest] = depth[source] + 1;
  }
  return result;
}

} // namespace wafer::compiler::detail
