//===- SpatialPlanReference.cpp - Independent tiny spatial oracle -------===//

#include "TestSupport/Planning/SpatialPlanReference.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

namespace wafer::test {
namespace {

using namespace compiler::detail;
using IntervalVector = std::vector<std::pair<int64_t, int64_t>>;

IntervalVector buildIntervals(int64_t extent, IteratorPartitionScheme scheme,
                              int64_t parameter) {
  IntervalVector intervals;
  if (extent <= 0 || parameter <= 0)
    return intervals;
  if (scheme == IteratorPartitionScheme::BalancedParts) {
    if (parameter > extent)
      return intervals;
    int64_t offset = 0;
    for (int64_t part = 0; part < parameter; ++part) {
      int64_t size = extent / parameter + (part < extent % parameter ? 1 : 0);
      intervals.emplace_back(offset, size);
      offset += size;
    }
    return intervals;
  }
  int64_t offset = 0;
  while (offset < extent) {
    const int64_t size = std::min(parameter, extent - offset);
    intervals.emplace_back(offset, size);
    offset += size;
  }
  return intervals;
}

std::vector<IteratorPartition> getAxisChoices(const ReferenceSpatialRoot &root,
                                              size_t iterator,
                                              size_t tileCount) {
  std::vector<IteratorPartition> choices;
  std::set<IntervalVector> seen;
  const int64_t extent = root.iteratorExtents[iterator];
  const bool partitionable = root.partitionableIterators[iterator] != 0;
  const int64_t maximumParts =
      partitionable ? std::min<int64_t>(extent, tileCount) : 1;
  for (int64_t parts = 1; parts <= maximumParts; ++parts) {
    IntervalVector intervals =
        buildIntervals(extent, IteratorPartitionScheme::BalancedParts, parts);
    seen.insert(intervals);
    choices.push_back({static_cast<uint32_t>(iterator),
                       IteratorPartitionScheme::BalancedParts, parts});
  }
  if (!partitionable)
    return choices;
  for (int64_t size = 1; size <= extent; ++size) {
    IntervalVector intervals =
        buildIntervals(extent, IteratorPartitionScheme::UniformExtent, size);
    if (intervals.empty() || intervals.size() > tileCount ||
        !seen.insert(intervals).second)
      continue;
    choices.push_back({static_cast<uint32_t>(iterator),
                       IteratorPartitionScheme::UniformExtent, size});
  }
  return choices;
}

bool keyValueRequirementHolds(const ReferenceSpatialRoot &root,
                              llvm::ArrayRef<int64_t> counts) {
  size_t cells = 1;
  for (uint8_t iterator : root.keyValueReductionIterators)
    cells *= static_cast<size_t>(counts[iterator]);
  switch (root.keyValueRequirement) {
  case ReferenceKeyValueRequirement::None:
    return true;
  case ReferenceKeyValueRequirement::SingleCell:
    return cells == 1;
  case ReferenceKeyValueRequirement::MultipleCells:
    return cells > 1;
  }
  return false;
}

void enumerateEmbeddings(llvm::ArrayRef<TileId> tiles, size_t count,
                         llvm::SmallVectorImpl<TileId> &current,
                         std::vector<llvm::SmallVector<TileId, 4>> &result) {
  if (current.size() == count) {
    result.emplace_back(current.begin(), current.end());
    return;
  }
  for (TileId tile : tiles) {
    if (llvm::is_contained(current, tile))
      continue;
    current.push_back(tile);
    enumerateEmbeddings(tiles, count, current, result);
    current.pop_back();
  }
}

std::vector<ReductionGroupId>
getReductionGroups(const ReferenceSpatialRoot &root,
                   llvm::ArrayRef<int64_t> counts) {
  bool partitionedReduction = false;
  for (size_t iterator = 0; iterator < counts.size(); ++iterator)
    partitionedReduction |=
        root.reductionIterators[iterator] != 0 && counts[iterator] > 1;
  if (!partitionedReduction || root.reductionResultGroupCount == 0)
    return {};
  std::vector<ReductionGroupId> groups;
  for (uint32_t result = 0; result < root.reductionResultGroupCount; ++result) {
    llvm::ArrayRef<uint8_t> resultParallel = root.resultParallelIterators;
    if (!root.resultParallelIteratorsByGroup.empty()) {
      if (result >= root.resultParallelIteratorsByGroup.size())
        return {};
      resultParallel = root.resultParallelIteratorsByGroup[result];
    }
    llvm::SmallVector<int64_t, 4> parallelCounts;
    for (size_t iterator = 0; iterator < counts.size(); ++iterator)
      if (resultParallel[iterator])
        parallelCounts.push_back(counts[iterator]);
    size_t coordinateCount = 1;
    for (int64_t count : parallelCounts)
      coordinateCount *= static_cast<size_t>(count);
    for (size_t linear = 0; linear < coordinateCount; ++linear) {
      size_t remainder = linear;
      ReductionGroupId group;
      group.root = root.root;
      group.resultGroup = result;
      group.parallelCoordinate.resize(parallelCounts.size());
      for (size_t reverse = 0; reverse < parallelCounts.size(); ++reverse) {
        size_t axis = parallelCounts.size() - reverse - 1;
        group.parallelCoordinate[axis] = static_cast<uint32_t>(
            remainder % static_cast<size_t>(parallelCounts[axis]));
        remainder /= static_cast<size_t>(parallelCounts[axis]);
      }
      groups.push_back(std::move(group));
    }
  }
  std::sort(groups.begin(), groups.end());
  return groups;
}

void enumerateMerges(
    llvm::ArrayRef<ReductionGroupId> groups, llvm::ArrayRef<TileId> tiles,
    size_t group, llvm::SmallVectorImpl<MergePlacement> &current,
    std::vector<llvm::SmallVector<MergePlacement, 4>> &result) {
  if (group == groups.size()) {
    result.emplace_back(current.begin(), current.end());
    return;
  }
  for (TileId tile : tiles) {
    current.push_back({groups[group], tile});
    enumerateMerges(groups, tiles, group + 1, current, result);
    current.pop_back();
  }
}

void enumerateAxes(const ReferenceSpatialRoot &root,
                   llvm::ArrayRef<TileId> tiles, size_t iterator,
                   llvm::SmallVectorImpl<IteratorPartition> &axes,
                   llvm::SmallVectorImpl<int64_t> &counts,
                   std::vector<NodeSpatialPlan> &plans) {
  if (iterator != root.iteratorExtents.size()) {
    for (const IteratorPartition &choice :
         getAxisChoices(root, iterator, tiles.size())) {
      size_t count = buildIntervals(root.iteratorExtents[iterator],
                                    choice.scheme, choice.parameter)
                         .size();
      size_t cells = count;
      for (int64_t previous : counts)
        cells *= static_cast<size_t>(previous);
      if (cells > tiles.size())
        continue;
      axes.push_back(choice);
      counts.push_back(static_cast<int64_t>(count));
      enumerateAxes(root, tiles, iterator + 1, axes, counts, plans);
      counts.pop_back();
      axes.pop_back();
    }
    return;
  }
  if (!keyValueRequirementHolds(root, counts))
    return;
  size_t cells = 1;
  for (int64_t count : counts)
    cells *= static_cast<size_t>(count);
  std::vector<llvm::SmallVector<TileId, 4>> embeddings;
  llvm::SmallVector<TileId, 4> embedding;
  enumerateEmbeddings(tiles, cells, embedding, embeddings);
  std::vector<ReductionGroupId> groups = getReductionGroups(root, counts);
  bool partitionedReduction = false;
  for (size_t index = 0; index < counts.size(); ++index)
    partitionedReduction |= root.reductionIterators[index] && counts[index] > 1;
  if (partitionedReduction && groups.empty())
    return;
  std::vector<llvm::SmallVector<MergePlacement, 4>> merges;
  llvm::SmallVector<MergePlacement, 4> merge;
  enumerateMerges(groups, tiles, 0, merge, merges);
  for (const auto &selectedEmbedding : embeddings)
    for (const auto &selectedMerges : merges) {
      NodeSpatialPlan plan;
      plan.root = root.root;
      plan.axes.assign(axes.begin(), axes.end());
      plan.embedding.assign(selectedEmbedding.begin(), selectedEmbedding.end());
      plan.reductionMerges.assign(selectedMerges.begin(), selectedMerges.end());
      plans.push_back(std::move(plan));
    }
}

void enumeratePrograms(llvm::ArrayRef<std::vector<NodeSpatialPlan>> nodePlans,
                       size_t node, SpatialPlan &current,
                       std::set<SpatialPlan> &result) {
  if (node == nodePlans.size()) {
    result.insert(current);
    return;
  }
  for (const NodeSpatialPlan &plan : nodePlans[node]) {
    current.nodes.push_back(plan);
    enumeratePrograms(nodePlans, node + 1, current, result);
    current.nodes.pop_back();
  }
}

} // namespace

std::set<SpatialPlan>
enumerateReferenceSpatialPlans(llvm::ArrayRef<ReferenceSpatialRoot> roots,
                               llvm::ArrayRef<TileId> availableTiles) {
  llvm::SmallVector<TileId, 4> tiles(availableTiles.begin(),
                                     availableTiles.end());
  llvm::sort(tiles, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
  std::vector<std::vector<NodeSpatialPlan>> nodePlans;
  for (const ReferenceSpatialRoot &root : roots) {
    if (root.iteratorExtents.size() != root.partitionableIterators.size() ||
        root.iteratorExtents.size() != root.reductionIterators.size() ||
        root.iteratorExtents.size() != root.resultParallelIterators.size())
      return {};
    if (!root.resultParallelIteratorsByGroup.empty() &&
        (root.resultParallelIteratorsByGroup.size() !=
             root.reductionResultGroupCount ||
         llvm::any_of(root.resultParallelIteratorsByGroup,
                      [&](llvm::ArrayRef<uint8_t> iterators) {
                        return iterators.size() != root.iteratorExtents.size();
                      })))
      return {};
    std::vector<NodeSpatialPlan> plans;
    llvm::SmallVector<IteratorPartition, 4> axes;
    llvm::SmallVector<int64_t, 4> counts;
    enumerateAxes(root, tiles, 0, axes, counts, plans);
    nodePlans.push_back(std::move(plans));
  }
  std::set<SpatialPlan> result;
  SpatialPlan current;
  enumeratePrograms(nodePlans, 0, current, result);
  return result;
}

} // namespace wafer::test
