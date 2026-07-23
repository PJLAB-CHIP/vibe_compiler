//===- CollectiveTopologyAnalysis.cpp - Collective topology facts -------===//

#include "Wafer/Analysis/CollectiveTopologyAnalysis.h"
#include "Wafer/Analysis/ExecutionTopologyAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::analysis {
namespace {

constexpr size_t kMaximumExactRingGroupSize = 16;
constexpr size_t kMaximumExactTreeGroupSize = 16;
constexpr uint64_t kInfiniteCost = std::numeric_limits<uint64_t>::max();

static bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static mlir::FailureOr<ExecutionTopologyAnalysis>
getTopology(mlir::Operation *anchor, llvm::ArrayRef<int64_t> rankGroup) {
  if (!anchor || rankGroup.empty())
    return mlir::failure();
  mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(anchor);
  if (!module)
    module = anchor->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return mlir::failure();
  llvm::SmallVector<int64_t, 16> seen;
  seen.reserve(rankGroup.size());
  for (int64_t rank : rankGroup) {
    if (rank < 0 || llvm::is_contained(seen, rank))
      return mlir::failure();
    seen.push_back(rank);
  }
  mlir::FailureOr<ExecutionTopologyAnalysis> topology =
      ExecutionTopologyAnalysis::create(module);
  if (mlir::failed(topology) ||
      mlir::failed(topology->getShortestHopMatrix(rankGroup)))
    return mlir::failure();
  return topology;
}

static bool
cycleRanksLexicographicallyLess(llvm::ArrayRef<int64_t> lhsGroupIndices,
                                llvm::ArrayRef<int64_t> rhsGroupIndices,
                                llvm::ArrayRef<int64_t> rankGroup) {
  for (auto [lhs, rhs] : llvm::zip(lhsGroupIndices, rhsGroupIndices)) {
    int64_t lhsRank = rankGroup[static_cast<size_t>(lhs)];
    int64_t rhsRank = rankGroup[static_cast<size_t>(rhs)];
    if (lhsRank != rhsRank)
      return lhsRank < rhsRank;
  }
  return false;
}

struct OrderedTreeState {
  bool valid = false;
  uint64_t edgeCost = 0;
  uint64_t maximumRootDistance = 0;
  uint64_t summedRootDistance = 0;
  int16_t leftRoot = -1;
  int16_t rightRoot = -1;
};

static bool isBetterOrderedTreeState(const OrderedTreeState &candidate,
                                     const OrderedTreeState &current,
                                     llvm::ArrayRef<int64_t> rankGroup) {
  if (!current.valid)
    return true;
  if (candidate.edgeCost != current.edgeCost)
    return candidate.edgeCost < current.edgeCost;
  if (candidate.maximumRootDistance != current.maximumRootDistance)
    return candidate.maximumRootDistance < current.maximumRootDistance;
  if (candidate.summedRootDistance != current.summedRootDistance)
    return candidate.summedRootDistance < current.summedRootDistance;
  auto childRank = [&](int16_t child) {
    return child < 0 ? std::numeric_limits<int64_t>::min()
                     : rankGroup[static_cast<size_t>(child)];
  };
  return std::make_tuple(childRank(candidate.leftRoot),
                         childRank(candidate.rightRoot)) <
         std::make_tuple(childRank(current.leftRoot),
                         childRank(current.rightRoot));
}

} // namespace

mlir::FailureOr<CollectiveRingOrder>
buildMinimumHopCollectiveRingOrder(mlir::Operation *anchor,
                                   llvm::ArrayRef<int64_t> rankGroup) {
  mlir::FailureOr<ExecutionTopologyAnalysis> topology =
      getTopology(anchor, rankGroup);
  if (mlir::failed(topology) || rankGroup.size() < 2 ||
      rankGroup.size() > kMaximumExactRingGroupSize)
    return mlir::failure();

  mlir::FailureOr<llvm::SmallVector<uint64_t, 16>> distances =
      topology->getShortestHopMatrix(rankGroup);
  if (mlir::failed(distances))
    return mlir::failure();

  const size_t nodeCount = rankGroup.size();
  const size_t movableCount = nodeCount - 1;
  const size_t stateCount = size_t{1} << movableCount;
  if (stateCount > std::numeric_limits<size_t>::max() / movableCount)
    return mlir::failure();
  const size_t cellCount = stateCount * movableCount;
  std::vector<uint64_t> costs(cellCount, kInfiniteCost);
  std::vector<int16_t> predecessors(cellCount, -1);
  auto distance = [&](size_t lhs, size_t rhs) {
    return (*distances)[lhs * nodeCount + rhs];
  };
  auto cell = [&](size_t mask, size_t node) {
    return mask * movableCount + (node - 1);
  };

  for (size_t node = 1; node < nodeCount; ++node)
    costs[cell(size_t{1} << (node - 1), node)] = distance(0, node);

  for (size_t mask = 1; mask < stateCount; ++mask) {
    for (size_t node = 1; node < nodeCount; ++node) {
      const size_t nodeBit = size_t{1} << (node - 1);
      if ((mask & nodeBit) == 0)
        continue;
      const size_t previousMask = mask ^ nodeBit;
      if (previousMask == 0)
        continue;
      uint64_t best = kInfiniteCost;
      int16_t bestPredecessor = -1;
      for (size_t predecessor = 1; predecessor < nodeCount; ++predecessor) {
        if ((previousMask & (size_t{1} << (predecessor - 1))) == 0)
          continue;
        uint64_t previous = costs[cell(previousMask, predecessor)];
        uint64_t candidate = 0;
        if (previous == kInfiniteCost ||
            !checkedAdd(previous, distance(predecessor, node), candidate))
          continue;
        if (candidate < best ||
            (candidate == best &&
             (bestPredecessor < 0 ||
              rankGroup[predecessor] <
                  rankGroup[static_cast<size_t>(bestPredecessor)]))) {
          best = candidate;
          bestPredecessor = static_cast<int16_t>(predecessor);
        }
      }
      costs[cell(mask, node)] = best;
      predecessors[cell(mask, node)] = bestPredecessor;
    }
  }

  const size_t fullMask = stateCount - 1;
  uint64_t bestCycleCost = kInfiniteCost;
  int16_t bestEnd = -1;
  for (size_t node = 1; node < nodeCount; ++node) {
    uint64_t path = costs[cell(fullMask, node)];
    uint64_t cycle = 0;
    if (path == kInfiniteCost || !checkedAdd(path, distance(node, 0), cycle))
      continue;
    if (cycle < bestCycleCost ||
        (cycle == bestCycleCost &&
         (bestEnd < 0 ||
          rankGroup[node] < rankGroup[static_cast<size_t>(bestEnd)]))) {
      bestCycleCost = cycle;
      bestEnd = static_cast<int16_t>(node);
    }
  }
  if (bestEnd < 0)
    return mlir::failure();

  CollectiveRingOrder result;
  result.groupIndices.resize(nodeCount);
  result.groupIndices.front() = 0;
  size_t mask = fullMask;
  int16_t current = bestEnd;
  for (size_t position = nodeCount - 1; position > 0; --position) {
    if (current <= 0)
      return mlir::failure();
    result.groupIndices[position] = current;
    int16_t previous = predecessors[cell(mask, static_cast<size_t>(current))];
    mask ^= size_t{1} << (static_cast<size_t>(current) - 1);
    current = previous;
  }
  if (mask != 0)
    return mlir::failure();

  llvm::SmallVector<int64_t, 16> reversed(result.groupIndices.size());
  reversed.front() = 0;
  for (size_t index = 1; index < result.groupIndices.size(); ++index)
    reversed[index] = result.groupIndices[result.groupIndices.size() - index];
  if (cycleRanksLexicographicallyLess(reversed, result.groupIndices, rankGroup))
    result.groupIndices = std::move(reversed);
  return result;
}

mlir::FailureOr<CollectiveTree>
buildMinimumHopCollectiveTree(mlir::Operation *anchor,
                              llvm::ArrayRef<int64_t> rankGroup) {
  if (rankGroup.size() < 2 || rankGroup.size() > kMaximumExactTreeGroupSize)
    return mlir::failure();
  mlir::FailureOr<ExecutionTopologyAnalysis> topology =
      getTopology(anchor, rankGroup);
  if (mlir::failed(topology))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<uint64_t, 16>> distances =
      topology->getShortestHopMatrix(rankGroup);
  if (mlir::failed(distances))
    return mlir::failure();

  const size_t nodeCount = rankGroup.size();
  auto distance = [&](size_t lhs, size_t rhs) {
    return (*distances)[lhs * nodeCount + rhs];
  };

  // Any binary tree whose in-order traversal is [begin, end] has a root that
  // separates two independent, contiguous ordered subtrees. This interval DP
  // therefore considers every legal ordered binary tree while keeping the
  // reduction's rank_group operand order explicit.
  std::vector<OrderedTreeState> states(nodeCount * nodeCount * nodeCount);
  auto state = [&](size_t begin, size_t end,
                   size_t root) -> OrderedTreeState & {
    return states[(begin * nodeCount + end) * nodeCount + root];
  };
  for (size_t node = 0; node < nodeCount; ++node)
    state(node, node, node).valid = true;

  for (size_t length = 2; length <= nodeCount; ++length) {
    for (size_t begin = 0; begin + length <= nodeCount; ++begin) {
      const size_t end = begin + length - 1;
      for (size_t root = begin; root <= end; ++root) {
        llvm::SmallVector<int16_t, 16> leftRoots;
        llvm::SmallVector<int16_t, 16> rightRoots;
        if (root == begin)
          leftRoots.push_back(-1);
        else
          for (size_t child = begin; child < root; ++child)
            leftRoots.push_back(static_cast<int16_t>(child));
        if (root == end)
          rightRoots.push_back(-1);
        else
          for (size_t child = root + 1; child <= end; ++child)
            rightRoots.push_back(static_cast<int16_t>(child));

        OrderedTreeState &best = state(begin, end, root);
        for (int16_t leftRoot : leftRoots) {
          for (int16_t rightRoot : rightRoots) {
            OrderedTreeState candidate;
            candidate.valid = true;
            candidate.leftRoot = leftRoot;
            candidate.rightRoot = rightRoot;
            auto includeChild = [&](int16_t childRoot, size_t childBegin,
                                    size_t childEnd) {
              if (childRoot < 0)
                return true;
              const OrderedTreeState &child =
                  state(childBegin, childEnd, static_cast<size_t>(childRoot));
              if (!child.valid)
                return false;
              const uint64_t edge =
                  distance(root, static_cast<size_t>(childRoot));
              uint64_t childEdgeCost = 0;
              if (!checkedAdd(child.edgeCost, edge, childEdgeCost) ||
                  !checkedAdd(candidate.edgeCost, childEdgeCost,
                              candidate.edgeCost))
                return false;
              uint64_t branchMaximum = 0;
              if (!checkedAdd(child.maximumRootDistance, edge, branchMaximum))
                return false;
              candidate.maximumRootDistance =
                  std::max(candidate.maximumRootDistance, branchMaximum);
              uint64_t edgeDistanceSum = 0;
              uint64_t branchSum = 0;
              const uint64_t childCount = childEnd - childBegin + 1;
              if (!checkedMultiply(edge, childCount, edgeDistanceSum) ||
                  !checkedAdd(child.summedRootDistance, edgeDistanceSum,
                              branchSum) ||
                  !checkedAdd(candidate.summedRootDistance, branchSum,
                              candidate.summedRootDistance))
                return false;
              return true;
            };
            if ((leftRoot >= 0 && !includeChild(leftRoot, begin, root - 1)) ||
                (rightRoot >= 0 && !includeChild(rightRoot, root + 1, end)))
              continue;
            if (isBetterOrderedTreeState(candidate, best, rankGroup))
              best = candidate;
          }
        }
        if (!best.valid)
          return mlir::failure();
      }
    }
  }

  int64_t bestRoot = -1;
  const OrderedTreeState *bestState = nullptr;
  for (size_t root = 0; root < nodeCount; ++root) {
    const OrderedTreeState &candidate = state(0, nodeCount - 1, root);
    if (!candidate.valid)
      continue;
    if (!bestState || candidate.edgeCost < bestState->edgeCost ||
        (candidate.edgeCost == bestState->edgeCost &&
         (candidate.maximumRootDistance < bestState->maximumRootDistance ||
          (candidate.maximumRootDistance == bestState->maximumRootDistance &&
           (candidate.summedRootDistance < bestState->summedRootDistance ||
            (candidate.summedRootDistance == bestState->summedRootDistance &&
             rankGroup[root] < rankGroup[static_cast<size_t>(bestRoot)])))))) {
      bestRoot = static_cast<int64_t>(root);
      bestState = &candidate;
    }
  }
  if (bestRoot < 0 || !bestState)
    return mlir::failure();

  CollectiveTree result;
  result.rootGroupIndex = bestRoot;
  result.parentGroupIndices.assign(nodeCount, -1);
  result.childGroupIndices.resize(nodeCount);
  result.depths.assign(nodeCount, -1);
  struct PendingSubtree {
    size_t begin;
    size_t end;
    size_t root;
    int64_t parent;
    int64_t depth;
  };
  llvm::SmallVector<PendingSubtree, 16> pending{
      {0, nodeCount - 1, static_cast<size_t>(bestRoot), -1, 0}};
  for (size_t cursor = 0; cursor < pending.size(); ++cursor) {
    PendingSubtree current = pending[cursor];
    if (result.depths[current.root] >= 0)
      return mlir::failure();
    result.parentGroupIndices[current.root] = current.parent;
    result.depths[current.root] = current.depth;
    const OrderedTreeState &currentState =
        state(current.begin, current.end, current.root);
    if (!currentState.valid)
      return mlir::failure();
    if (currentState.leftRoot >= 0) {
      const size_t child = static_cast<size_t>(currentState.leftRoot);
      if (child < current.begin || child >= current.root)
        return mlir::failure();
      result.childGroupIndices[current.root].push_back(currentState.leftRoot);
      pending.push_back({current.begin, current.root - 1, child,
                         static_cast<int64_t>(current.root),
                         current.depth + 1});
    }
    if (currentState.rightRoot >= 0) {
      const size_t child = static_cast<size_t>(currentState.rightRoot);
      if (child <= current.root || child > current.end)
        return mlir::failure();
      result.childGroupIndices[current.root].push_back(currentState.rightRoot);
      pending.push_back({current.root + 1, current.end, child,
                         static_cast<int64_t>(current.root),
                         current.depth + 1});
    }
  }
  if (llvm::any_of(result.depths, [](int64_t depth) { return depth < 0; }))
    return mlir::failure();
  return result;
}

} // namespace wafer::analysis
