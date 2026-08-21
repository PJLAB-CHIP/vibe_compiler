//===- SpatialPlan.cpp - Compact spatial planning schema -----------------===//

#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

mlir::LogicalResult fail(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
  return mlir::failure();
}

template <typename T>
mlir::FailureOr<T> failValue(std::string *failureReason,
                             llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
  return mlir::failure();
}

bool isKnownAnchorKind(SemanticRootAnchorKind kind) {
  switch (kind) {
  case SemanticRootAnchorKind::FunctionResult:
  case SemanticRootAnchorKind::EffectBoundary:
    return true;
  }
  return false;
}

bool isKnownPathRelation(SemanticRootPathRelation relation) {
  switch (relation) {
  case SemanticRootPathRelation::SSAUseDef:
  case SemanticRootPathRelation::RegionBranch:
  case SemanticRootPathRelation::EffectDependency:
    return true;
  }
  return false;
}

mlir::LogicalResult validateRootKey(const SemanticRootKey &root,
                                    std::string *failureReason) {
  if (!isKnownAnchorKind(root.anchorKind))
    return fail(failureReason, "semantic root has unknown anchor kind");
  for (const SemanticRootPathStep &step : root.path) {
    if (!isKnownPathRelation(step.relation))
      return fail(failureReason, "semantic root has unknown path relation");
  }
  return mlir::success();
}

bool tileLess(TileId lhs, TileId rhs) {
  return lhs.getValue() < rhs.getValue();
}

bool containsTile(llvm::ArrayRef<TileId> tiles, TileId tile) {
  return llvm::any_of(tiles,
                      [tile](TileId available) { return available == tile; });
}

template <typename T>
bool lexicographicalLess(llvm::ArrayRef<T> lhs, llvm::ArrayRef<T> rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end());
}

bool tileArrayLess(llvm::ArrayRef<TileId> lhs, llvm::ArrayRef<TileId> rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), tileLess);
}

mlir::FailureOr<llvm::SmallVector<IteratorInterval, 4>>
buildIntervals(int64_t extent, const IteratorPartition &partition,
               size_t maximumIntervals, std::string *failureReason) {
  mlir::FailureOr<int64_t> intervalCount =
      getIteratorPartitionIntervalCount(extent, partition, failureReason);
  if (mlir::failed(intervalCount))
    return mlir::failure();
  if (static_cast<uint64_t>(*intervalCount) > maximumIntervals)
    return failValue<llvm::SmallVector<IteratorInterval, 4>>(
        failureReason, "partition interval count exceeds available Tiles");

  llvm::SmallVector<IteratorInterval, 4> intervals;
  switch (partition.scheme) {
  case IteratorPartitionScheme::BalancedParts: {
    const int64_t parts = partition.parameter;
    const int64_t quotient = extent / parts;
    const int64_t remainder = extent % parts;
    int64_t offset = 0;
    intervals.reserve(parts);
    for (int64_t part = 0; part < parts; ++part) {
      const int64_t size = quotient + (part < remainder ? 1 : 0);
      intervals.push_back({offset, size});
      offset += size;
    }
    break;
  }
  case IteratorPartitionScheme::UniformExtent: {
    const int64_t tileSize = partition.parameter;
    for (int64_t offset = 0; offset < extent; offset += tileSize)
      intervals.push_back({offset, std::min(tileSize, extent - offset)});
    break;
  }
  default:
    return failValue<llvm::SmallVector<IteratorInterval, 4>>(
        failureReason, "partition uses unknown scheme");
  }
  return intervals;
}

mlir::LogicalResult
validateCanonicalPartition(int64_t extent, const IteratorPartition &partition,
                           llvm::ArrayRef<IteratorInterval> intervals,
                           size_t maximumIntervals,
                           std::string *failureReason) {
  if (partition.scheme != IteratorPartitionScheme::UniformExtent)
    return mlir::success();
  IteratorPartition balanced{partition.iterator,
                             IteratorPartitionScheme::BalancedParts,
                             static_cast<int64_t>(intervals.size())};
  mlir::FailureOr<llvm::SmallVector<IteratorInterval, 4>> balancedIntervals =
      buildIntervals(extent, balanced, maximumIntervals, failureReason);
  if (mlir::failed(balancedIntervals))
    return mlir::failure();
  if (llvm::equal(*balancedIntervals, intervals))
    return fail(failureReason,
                "UniformExtent duplicates canonical BalancedParts intervals");
  return mlir::success();
}

mlir::FailureOr<size_t>
getCellCount(llvm::ArrayRef<llvm::SmallVector<IteratorInterval, 4>> axes,
             size_t maximumCells, std::string *failureReason) {
  size_t count = 1;
  for (llvm::ArrayRef<IteratorInterval> intervals : axes) {
    if (intervals.empty())
      return failValue<size_t>(failureReason,
                               "partition produced no intervals");
    if (count > maximumCells / intervals.size())
      return failValue<size_t>(failureReason,
                               "spatial cell count exceeds available Tiles");
    count *= intervals.size();
  }
  if (count == 0 || count > maximumCells)
    return failValue<size_t>(failureReason,
                             "spatial cell count exceeds available Tiles");
  return count;
}

mlir::LogicalResult validateMergePlacements(
    const SemanticRootKey &root, llvm::ArrayRef<MergePlacement> placements,
    llvm::ArrayRef<TileId> availableTiles, std::string *failureReason) {
  for (size_t index = 0; index < placements.size(); ++index) {
    const MergePlacement &placement = placements[index];
    if (!(placement.group.root == root))
      return fail(failureReason,
                  "reduction group belongs to another semantic root");
    if (!containsTile(availableTiles, placement.tile))
      return fail(failureReason, "merge placement uses unavailable Tile");
    if (index != 0 && !(placements[index - 1].group < placement.group))
      return fail(failureReason,
                  "reduction merge groups must be unique and sorted");
  }
  return mlir::success();
}

mlir::LogicalResult validateReductionGroupPlacements(
    const SemanticRootKey &root,
    llvm::ArrayRef<ReductionGroupPlacement> placements,
    llvm::ArrayRef<TileId> availableTiles, std::string *failureReason) {
  for (size_t index = 0; index < placements.size(); ++index) {
    const ReductionGroupPlacement &placement = placements[index];
    if (!(placement.group.root == root))
      return fail(failureReason,
                  "assignment reduction group belongs to another root");
    if (!containsTile(availableTiles, placement.mergeTile))
      return fail(failureReason,
                  "assignment reduction group uses unavailable merge Tile");
    if (index != 0 && !(placements[index - 1].group < placement.group))
      return fail(failureReason,
                  "assignment reduction groups must be unique and sorted");
  }
  return mlir::success();
}

void enumerateCells(const SemanticRootKey &root,
                    llvm::ArrayRef<llvm::SmallVector<IteratorInterval, 4>> axes,
                    llvm::ArrayRef<TileId> embedding,
                    llvm::SmallVectorImpl<ExecutionShard> &shards) {
  for (size_t cell = 0; cell < embedding.size(); ++cell) {
    size_t remainder = cell;
    LogicalShardId id;
    id.root = root;
    id.coordinate.resize(axes.size());
    ExecutionShard shard;
    shard.tile = embedding[cell];
    shard.iterationDomain.resize(axes.size());
    for (size_t reverse = 0; reverse < axes.size(); ++reverse) {
      const size_t axis = axes.size() - reverse - 1;
      const size_t coordinate = remainder % axes[axis].size();
      remainder /= axes[axis].size();
      id.coordinate[axis] = static_cast<uint32_t>(coordinate);
      shard.iterationDomain[axis] = axes[axis][coordinate];
    }
    shard.shard = std::move(id);
    shards.push_back(std::move(shard));
  }
}

} // namespace

bool operator<(const IteratorPartition &lhs, const IteratorPartition &rhs) {
  return std::tie(lhs.iterator, lhs.scheme, lhs.parameter) <
         std::tie(rhs.iterator, rhs.scheme, rhs.parameter);
}

bool operator<(const MergePlacement &lhs, const MergePlacement &rhs) {
  if (lhs.group != rhs.group)
    return lhs.group < rhs.group;
  return tileLess(lhs.tile, rhs.tile);
}

bool operator<(const NodeSpatialPlan &lhs, const NodeSpatialPlan &rhs) {
  if (lhs.root != rhs.root)
    return lhs.root < rhs.root;
  if (lhs.axes != rhs.axes)
    return lexicographicalLess<IteratorPartition>(lhs.axes, rhs.axes);
  if (lhs.embedding != rhs.embedding)
    return tileArrayLess(lhs.embedding, rhs.embedding);
  return lexicographicalLess<MergePlacement>(lhs.reductionMerges,
                                             rhs.reductionMerges);
}

bool operator<(const SpatialPlan &lhs, const SpatialPlan &rhs) {
  return lexicographicalLess<NodeSpatialPlan>(lhs.nodes, rhs.nodes);
}

mlir::FailureOr<int64_t>
getIteratorPartitionIntervalCount(int64_t extent,
                                  const IteratorPartition &partition,
                                  std::string *failureReason) {
  if (extent <= 0)
    return failValue<int64_t>(failureReason,
                              "iterator extent must be positive");
  if (partition.parameter <= 0 || partition.parameter > extent)
    return failValue<int64_t>(failureReason,
                              "partition parameter is outside iterator extent");
  switch (partition.scheme) {
  case IteratorPartitionScheme::BalancedParts:
    return partition.parameter;
  case IteratorPartitionScheme::UniformExtent:
    return 1 + (extent - 1) / partition.parameter;
  }
  return failValue<int64_t>(failureReason, "partition uses unknown scheme");
}

mlir::FailureOr<SpatialPlanningProblem>
SpatialPlanningProblem::create(llvm::ArrayRef<NodeIterationSpace> inputNodes,
                               llvm::ArrayRef<TileId> inputTiles,
                               std::string *failureReason) {
  if (inputNodes.empty())
    return failValue<SpatialPlanningProblem>(failureReason,
                                             "spatial problem has no roots");
  if (inputTiles.empty())
    return failValue<SpatialPlanningProblem>(failureReason,
                                             "spatial problem has no Tiles");

  llvm::SmallVector<NodeIterationSpace, 16> nodes(inputNodes.begin(),
                                                  inputNodes.end());
  llvm::sort(nodes,
             [](const NodeIterationSpace &lhs, const NodeIterationSpace &rhs) {
               return lhs.root < rhs.root;
             });
  for (size_t index = 0; index < nodes.size(); ++index) {
    if (mlir::failed(validateRootKey(nodes[index].root, failureReason)))
      return mlir::failure();
    if (index != 0 && nodes[index - 1].root == nodes[index].root)
      return failValue<SpatialPlanningProblem>(
          failureReason, "spatial problem contains duplicate root");
    if (llvm::any_of(nodes[index].iteratorExtents,
                     [](int64_t extent) { return extent <= 0; }))
      return failValue<SpatialPlanningProblem>(
          failureReason, "spatial problem has non-positive iterator extent");
  }

  llvm::SmallVector<TileId, 16> availableTiles(inputTiles.begin(),
                                               inputTiles.end());
  llvm::sort(availableTiles, tileLess);
  for (size_t index = 0; index < availableTiles.size(); ++index) {
    if (availableTiles[index].getValue() < 0)
      return failValue<SpatialPlanningProblem>(failureReason,
                                               "Tile identity is negative");
    if (index != 0 && availableTiles[index - 1] == availableTiles[index])
      return failValue<SpatialPlanningProblem>(
          failureReason, "spatial problem contains duplicate Tile");
  }
  return SpatialPlanningProblem(std::move(nodes), std::move(availableTiles));
}

mlir::LogicalResult
validateSpatialPlanStructure(const SpatialPlanningProblem &problem,
                             const SpatialPlan &plan,
                             std::string *failureReason) {
  if (plan.nodes.size() != problem.getNodes().size())
    return fail(failureReason,
                "SpatialPlan must cover all-and-only problem roots");
  for (auto [problemNode, nodePlan] :
       llvm::zip_equal(problem.getNodes(), plan.nodes)) {
    if (!(problemNode.root == nodePlan.root))
      return fail(failureReason,
                  "SpatialPlan roots must use canonical problem order");
    if (nodePlan.axes.size() != problemNode.iteratorExtents.size())
      return fail(failureReason,
                  "SpatialPlan must specify every iterator exactly once");

    llvm::SmallVector<llvm::SmallVector<IteratorInterval, 4>, 4> axes;
    axes.reserve(nodePlan.axes.size());
    for (size_t iterator = 0; iterator < nodePlan.axes.size(); ++iterator) {
      const IteratorPartition &partition = nodePlan.axes[iterator];
      if (partition.iterator != iterator)
        return fail(failureReason,
                    "SpatialPlan iterator partitions must be ordered");
      mlir::FailureOr<llvm::SmallVector<IteratorInterval, 4>> intervals =
          buildIntervals(problemNode.iteratorExtents[iterator], partition,
                         problem.getAvailableTiles().size(), failureReason);
      if (mlir::failed(intervals) ||
          mlir::failed(validateCanonicalPartition(
              problemNode.iteratorExtents[iterator], partition, *intervals,
              problem.getAvailableTiles().size(), failureReason)))
        return mlir::failure();
      axes.push_back(std::move(*intervals));
    }
    mlir::FailureOr<size_t> cellCount =
        getCellCount(axes, problem.getAvailableTiles().size(), failureReason);
    if (mlir::failed(cellCount))
      return mlir::failure();
    if (nodePlan.embedding.size() != *cellCount)
      return fail(failureReason,
                  "SpatialPlan embedding size must equal logical cell count");
    llvm::SmallVector<TileId, 16> sortedEmbedding(nodePlan.embedding.begin(),
                                                  nodePlan.embedding.end());
    llvm::sort(sortedEmbedding, tileLess);
    for (size_t index = 0; index < sortedEmbedding.size(); ++index) {
      if (!containsTile(problem.getAvailableTiles(), sortedEmbedding[index]))
        return fail(failureReason,
                    "SpatialPlan embedding uses unavailable Tile");
      if (index != 0 && sortedEmbedding[index - 1] == sortedEmbedding[index])
        return fail(failureReason, "SpatialPlan embedding must be injective");
    }
    if (mlir::failed(validateMergePlacements(
            nodePlan.root, nodePlan.reductionMerges,
            problem.getAvailableTiles(), failureReason)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::FailureOr<SpatialAssignment>
closeSpatialPlanStructure(const SpatialPlanningProblem &problem,
                          const SpatialPlan &plan, std::string *failureReason) {
  if (mlir::failed(validateSpatialPlanStructure(problem, plan, failureReason)))
    return mlir::failure();

  SpatialAssignment assignment;
  assignment.nodes.reserve(plan.nodes.size());
  for (auto [problemNode, nodePlan] :
       llvm::zip_equal(problem.getNodes(), plan.nodes)) {
    llvm::SmallVector<llvm::SmallVector<IteratorInterval, 4>, 4> axes;
    axes.reserve(nodePlan.axes.size());
    for (size_t iterator = 0; iterator < nodePlan.axes.size(); ++iterator) {
      mlir::FailureOr<llvm::SmallVector<IteratorInterval, 4>> intervals =
          buildIntervals(problemNode.iteratorExtents[iterator],
                         nodePlan.axes[iterator],
                         problem.getAvailableTiles().size(), failureReason);
      if (mlir::failed(intervals))
        return mlir::failure();
      axes.push_back(std::move(*intervals));
    }

    NodeExecutionPartition nodeAssignment;
    nodeAssignment.root = nodePlan.root;
    enumerateCells(nodePlan.root, axes, nodePlan.embedding,
                   nodeAssignment.shards);
    for (const MergePlacement &merge : nodePlan.reductionMerges)
      nodeAssignment.reductionGroups.push_back({merge.group, merge.tile});
    assignment.nodes.push_back(std::move(nodeAssignment));
  }
  if (mlir::failed(validateSpatialAssignmentStructure(problem, assignment,
                                                      failureReason)))
    return mlir::failure();
  return assignment;
}

mlir::LogicalResult
validateSpatialAssignmentStructure(const SpatialPlanningProblem &problem,
                                   const SpatialAssignment &assignment,
                                   std::string *failureReason) {
  if (assignment.nodes.size() != problem.getNodes().size())
    return fail(failureReason,
                "SpatialAssignment must cover all-and-only problem roots");
  for (auto [problemNode, nodeAssignment] :
       llvm::zip_equal(problem.getNodes(), assignment.nodes)) {
    if (!(problemNode.root == nodeAssignment.root))
      return fail(failureReason,
                  "SpatialAssignment roots must use canonical problem order");
    if (nodeAssignment.shards.empty())
      return fail(failureReason, "SpatialAssignment node has no shards");

    llvm::SmallVector<std::map<uint32_t, IteratorInterval>, 4> axisIntervals(
        problemNode.iteratorExtents.size());
    std::set<std::vector<uint32_t>> coordinates;
    llvm::SmallVector<TileId, 16> tiles;
    for (const ExecutionShard &shard : nodeAssignment.shards) {
      if (!(shard.shard.root == nodeAssignment.root))
        return fail(failureReason, "shard belongs to another semantic root");
      if (shard.shard.coordinate.size() != problemNode.iteratorExtents.size() ||
          shard.iterationDomain.size() != problemNode.iteratorExtents.size())
        return fail(failureReason,
                    "shard coordinate/domain rank does not match root");
      if (!containsTile(problem.getAvailableTiles(), shard.tile))
        return fail(failureReason, "shard uses unavailable Tile");
      if (llvm::is_contained(tiles, shard.tile))
        return fail(failureReason,
                    "SpatialAssignment embedding must be injective");
      tiles.push_back(shard.tile);
      std::vector<uint32_t> coordinate(shard.shard.coordinate.begin(),
                                       shard.shard.coordinate.end());
      if (!coordinates.insert(coordinate).second)
        return fail(failureReason,
                    "SpatialAssignment has duplicate logical coordinate");
      for (size_t axis = 0; axis < coordinate.size(); ++axis) {
        const IteratorInterval interval = shard.iterationDomain[axis];
        const int64_t extent = problemNode.iteratorExtents[axis];
        if (interval.offset < 0 || interval.size <= 0 ||
            interval.size > extent || interval.offset > extent - interval.size)
          return fail(failureReason,
                      "shard interval is outside iterator extent");
        auto [position, inserted] =
            axisIntervals[axis].emplace(coordinate[axis], interval);
        if (!inserted && !(position->second == interval))
          return fail(failureReason,
                      "same logical coordinate has conflicting interval");
      }
    }

    llvm::SmallVector<size_t, 4> axisCounts;
    axisCounts.reserve(axisIntervals.size());
    for (auto [axis, intervals] : llvm::enumerate(axisIntervals)) {
      if (intervals.empty())
        return fail(failureReason, "assignment axis has no intervals");
      int64_t end = 0;
      uint32_t expectedCoordinate = 0;
      for (const auto &[coordinate, interval] : intervals) {
        if (coordinate != expectedCoordinate++ || interval.offset != end)
          return fail(failureReason,
                      "assignment axis intervals are not dense and contiguous");
        end = interval.getEnd();
      }
      if (end != problemNode.iteratorExtents[axis])
        return fail(failureReason,
                    "assignment axis intervals do not cover full extent");
      axisCounts.push_back(intervals.size());
    }
    size_t product = 1;
    for (size_t count : axisCounts) {
      if (product > problem.getAvailableTiles().size() / count)
        return fail(failureReason,
                    "assignment cell count exceeds available Tiles");
      product *= count;
    }
    if (product != nodeAssignment.shards.size())
      return fail(failureReason,
                  "assignment omits a Cartesian logical coordinate");
    llvm::SmallVector<unsigned char, 16> visited(product, 0);
    for (const std::vector<uint32_t> &coordinate : coordinates) {
      size_t linear = 0;
      for (auto [axis, value] : llvm::enumerate(coordinate)) {
        if (value >= axisCounts[axis])
          return fail(failureReason,
                      "assignment coordinate is outside partition domain");
        linear = linear * axisCounts[axis] + value;
      }
      if (visited[linear] != 0)
        return fail(failureReason,
                    "assignment coordinate linearization is duplicated");
      visited[linear] = 1;
    }
    if (llvm::any_of(visited,
                     [](unsigned char present) { return present == 0; }))
      return fail(failureReason,
                  "assignment omits a Cartesian logical coordinate");
    if (mlir::failed(validateReductionGroupPlacements(
            nodeAssignment.root, nodeAssignment.reductionGroups,
            problem.getAvailableTiles(), failureReason)))
      return mlir::failure();
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
