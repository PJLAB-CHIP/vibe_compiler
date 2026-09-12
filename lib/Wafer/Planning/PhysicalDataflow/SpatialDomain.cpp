//===- SpatialDomain.cpp - Complete spatial plan domain ----------------===//

#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPartitionPropagation.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

SpatialDomainFailure fail(SpatialDomainFailureKind kind, llvm::StringRef detail,
                          std::optional<SemanticRootKey> root = std::nullopt) {
  return SpatialDomainFailure{kind, std::move(root), detail.str()};
}

bool tileLess(TileId lhs, TileId rhs) {
  return lhs.getValue() < rhs.getValue();
}

bool containsTile(llvm::ArrayRef<TileId> tiles, TileId tile) {
  return llvm::is_contained(tiles, tile);
}

std::optional<size_t> checkedProduct(llvm::ArrayRef<int64_t> values,
                                     size_t limit) {
  size_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || static_cast<uint64_t>(value) > limit ||
        result > limit / static_cast<size_t>(value))
      return std::nullopt;
    result *= static_cast<size_t>(value);
  }
  return result;
}

std::optional<size_t> getCellCount(llvm::ArrayRef<int64_t> extents,
                                   llvm::ArrayRef<IteratorPartition> partitions,
                                   size_t tileCount) {
  if (extents.size() != partitions.size())
    return std::nullopt;
  llvm::SmallVector<int64_t, 8> counts;
  counts.reserve(extents.size());
  for (auto [extent, partition] : llvm::zip_equal(extents, partitions)) {
    mlir::FailureOr<int64_t> count =
        getIteratorPartitionIntervalCount(extent, partition);
    if (mlir::failed(count))
      return std::nullopt;
    counts.push_back(*count);
  }
  return checkedProduct(counts, tileCount);
}

mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
getIntervalCounts(llvm::ArrayRef<int64_t> extents,
                  llvm::ArrayRef<IteratorPartition> partitions,
                  std::string *failureReason = nullptr) {
  if (extents.size() != partitions.size())
    return mlir::failure();
  llvm::SmallVector<int64_t, 8> counts;
  counts.reserve(extents.size());
  for (auto [extent, partition] : llvm::zip_equal(extents, partitions)) {
    mlir::FailureOr<int64_t> count =
        getIteratorPartitionIntervalCount(extent, partition, failureReason);
    if (mlir::failed(count))
      return mlir::failure();
    counts.push_back(*count);
  }
  return counts;
}

bool intervalsEqual(int64_t extent, const IteratorPartition &lhs,
                    const IteratorPartition &rhs, size_t maximumIntervals) {
  mlir::FailureOr<llvm::SmallVector<IteratorInterval, 4>> lhsIntervals =
      getIteratorPartitionIntervals(extent, lhs, maximumIntervals);
  mlir::FailureOr<llvm::SmallVector<IteratorInterval, 4>> rhsIntervals =
      getIteratorPartitionIntervals(extent, rhs, maximumIntervals);
  return mlir::succeeded(lhsIntervals) && mlir::succeeded(rhsIntervals) &&
         *lhsIntervals == *rhsIntervals;
}

bool canPartitionIterator(const SpatialRootDomainFacts &root, size_t iterator) {
  return iterator < root.iteratorExtents.size() &&
         (root.partitionableParallelIterators.test(iterator) ||
          root.partitionableReductionIterators.test(iterator));
}

IteratorPartition firstPartition(size_t iterator) {
  return IteratorPartition{static_cast<uint32_t>(iterator),
                           IteratorPartitionScheme::BalancedParts, 1};
}

std::optional<IteratorPartition>
nextPartition(const SpatialRootDomainFacts &root, size_t iterator,
              const IteratorPartition &current, size_t maximumIntervals) {
  if (iterator >= root.iteratorExtents.size() || current.iterator != iterator ||
      maximumIntervals == 0 ||
      maximumIntervals >
          static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  const int64_t extent = root.iteratorExtents[iterator];
  if (!canPartitionIterator(root, iterator))
    return std::nullopt;

  if (current.scheme == IteratorPartitionScheme::BalancedParts) {
    const int64_t maximumParts =
        std::min<int64_t>(extent, static_cast<int64_t>(maximumIntervals));
    if (current.parameter < maximumParts)
      return IteratorPartition{static_cast<uint32_t>(iterator),
                               IteratorPartitionScheme::BalancedParts,
                               current.parameter + 1};
  }

  int64_t start = 1;
  if (current.scheme == IteratorPartitionScheme::BalancedParts) {
    start = 1 + (extent - 1) / static_cast<int64_t>(maximumIntervals);
  } else if (current.scheme == IteratorPartitionScheme::UniformExtent) {
    start = current.parameter + 1;
  } else {
    return std::nullopt;
  }
  for (int64_t size = start; size <= extent; ++size) {
    IteratorPartition uniform{static_cast<uint32_t>(iterator),
                              IteratorPartitionScheme::UniformExtent, size};
    mlir::FailureOr<int64_t> count =
        getIteratorPartitionIntervalCount(extent, uniform);
    if (mlir::failed(count) || *count > static_cast<int64_t>(maximumIntervals))
      continue;
    IteratorPartition balanced{static_cast<uint32_t>(iterator),
                               IteratorPartitionScheme::BalancedParts, *count};
    if (intervalsEqual(extent, uniform, balanced, maximumIntervals))
      continue;
    return uniform;
  }
  return std::nullopt;
}

AttentionSpatialConstraintViolation
checkAttention(const SpatialRootDomainFacts &root,
               llvm::ArrayRef<IteratorPartition> partitions) {
  if (!root.attention)
    return AttentionSpatialConstraintViolation::None;
  return checkAttentionSpatialConstraints(*root.attention, root.iteratorExtents,
                                          partitions);
}

bool hasAllowedPartitionKinds(const SpatialRootDomainFacts &root,
                              llvm::ArrayRef<IteratorPartition> partitions,
                              size_t maximumIntervals) {
  if (partitions.size() != root.iteratorExtents.size())
    return false;
  for (auto [iterator, partition] : llvm::enumerate(partitions)) {
    if (partition.iterator != iterator)
      return false;
    mlir::FailureOr<int64_t> count = getIteratorPartitionIntervalCount(
        root.iteratorExtents[iterator], partition);
    if (mlir::failed(count) || *count <= 0 ||
        *count > static_cast<int64_t>(maximumIntervals) ||
        (*count > 1 && !canPartitionIterator(root, iterator)))
      return false;
    if (partition.scheme == IteratorPartitionScheme::UniformExtent) {
      IteratorPartition balanced{static_cast<uint32_t>(iterator),
                                 IteratorPartitionScheme::BalancedParts,
                                 *count};
      if (intervalsEqual(root.iteratorExtents[iterator], partition, balanced,
                         maximumIntervals))
        return false;
    }
  }
  return checkAttention(root, partitions) ==
         AttentionSpatialConstraintViolation::None;
}

bool axesAreValid(const SpatialRootDomainFacts &root,
                  llvm::ArrayRef<IteratorPartition> partitions,
                  size_t tileCount) {
  if (!hasAllowedPartitionKinds(root, partitions, tileCount))
    return false;
  return getCellCount(root.iteratorExtents, partitions, tileCount).has_value();
}

bool advanceAxes(const SpatialRootDomainFacts &root, size_t tileCount,
                 llvm::SmallVectorImpl<IteratorPartition> &partitions) {
  if (partitions.size() != root.iteratorExtents.size())
    return false;
  while (true) {
    bool advanced = false;
    for (size_t reverse = 0; reverse < partitions.size(); ++reverse) {
      const size_t iterator = partitions.size() - reverse - 1;
      std::optional<IteratorPartition> next =
          nextPartition(root, iterator, partitions[iterator], tileCount);
      if (!next) {
        partitions[iterator] = firstPartition(iterator);
        continue;
      }
      partitions[iterator] = *next;
      for (size_t suffix = iterator + 1; suffix < partitions.size(); ++suffix)
        partitions[suffix] = firstPartition(suffix);
      advanced = true;
      break;
    }
    if (!advanced)
      return false;
    if (axesAreValid(root, partitions, tileCount))
      return true;
  }
}

std::optional<llvm::SmallVector<IteratorPartition, 4>>
getFirstAxes(const SpatialRootDomainFacts &root, size_t tileCount) {
  llvm::SmallVector<IteratorPartition, 4> axes;
  for (size_t iterator = 0; iterator < root.iteratorExtents.size(); ++iterator)
    axes.push_back(firstPartition(iterator));
  if (axesAreValid(root, axes, tileCount))
    return axes;
  if (advanceAxes(root, tileCount, axes))
    return axes;
  return std::nullopt;
}

bool nextDistinctTileSequence(llvm::ArrayRef<TileId> available,
                              llvm::SmallVectorImpl<TileId> &embedding) {
  for (size_t reverse = 0; reverse < embedding.size(); ++reverse) {
    const size_t position = embedding.size() - reverse - 1;
    auto current = llvm::find(available, embedding[position]);
    if (current == available.end())
      return false;
    for (auto next = std::next(current); next != available.end(); ++next) {
      if (containsTile(llvm::ArrayRef<TileId>(embedding).take_front(position),
                       *next))
        continue;
      embedding[position] = *next;
      llvm::SmallVector<TileId, 16> used(embedding.begin(),
                                         embedding.begin() + position + 1);
      size_t suffix = position + 1;
      if (suffix == embedding.size())
        return true;
      for (TileId tile : available) {
        if (containsTile(used, tile))
          continue;
        embedding[suffix++] = tile;
        used.push_back(tile);
        if (suffix == embedding.size())
          return true;
      }
      return suffix == embedding.size();
    }
  }
  return false;
}

bool nextMergePlacement(llvm::ArrayRef<TileId> available,
                        llvm::SmallVectorImpl<MergePlacement> &placements) {
  for (size_t reverse = 0; reverse < placements.size(); ++reverse) {
    const size_t index = placements.size() - reverse - 1;
    auto current = llvm::find(available, placements[index].tile);
    if (current == available.end())
      return false;
    if (++current == available.end()) {
      placements[index].tile = available.front();
      continue;
    }
    placements[index].tile = *current;
    for (size_t suffix = index + 1; suffix < placements.size(); ++suffix)
      placements[suffix].tile = available.front();
    return true;
  }
  return false;
}

llvm::SmallVector<MergePlacement, 8>
getFirstMergePlacements(llvm::ArrayRef<ReductionGroupId> groups,
                        llvm::ArrayRef<TileId> available) {
  llvm::SmallVector<MergePlacement, 8> placements;
  placements.reserve(groups.size());
  for (const ReductionGroupId &group : groups)
    placements.push_back({group, available.front()});
  return placements;
}

std::optional<NodeSpatialPlan>
getFirstNodePlan(const SpatialRootDomainFacts &root,
                 llvm::ArrayRef<TileId> available) {
  std::optional<llvm::SmallVector<IteratorPartition, 4>> axes =
      getFirstAxes(root, available.size());
  if (!axes)
    return std::nullopt;
  std::optional<size_t> cells =
      getCellCount(root.iteratorExtents, *axes, available.size());
  if (!cells)
    return std::nullopt;
  mlir::FailureOr<llvm::SmallVector<ReductionGroupId, 8>> groups =
      deriveSpatialReductionGroups(root, *axes);
  if (mlir::failed(groups))
    return std::nullopt;
  NodeSpatialPlan plan;
  plan.root = root.root;
  plan.axes = std::move(*axes);
  plan.embedding.assign(available.begin(), available.begin() + *cells);
  plan.reductionMerges = getFirstMergePlacements(*groups, available);
  return plan;
}

bool nodePlanContains(const SpatialRootDomainFacts &root,
                      llvm::ArrayRef<TileId> available,
                      const NodeSpatialPlan &plan) {
  if (!(plan.root == root.root) ||
      !axesAreValid(root, plan.axes, available.size()))
    return false;
  std::optional<size_t> cells =
      getCellCount(root.iteratorExtents, plan.axes, available.size());
  if (!cells || plan.embedding.size() != *cells)
    return false;
  llvm::SmallVector<TileId, 16> seen;
  for (TileId tile : plan.embedding) {
    if (!containsTile(available, tile) || containsTile(seen, tile))
      return false;
    seen.push_back(tile);
  }
  mlir::FailureOr<llvm::SmallVector<ReductionGroupId, 8>> groups =
      deriveSpatialReductionGroups(root, plan.axes);
  if (mlir::failed(groups) || groups->size() != plan.reductionMerges.size())
    return false;
  for (auto [group, placement] : llvm::zip_equal(*groups, plan.reductionMerges))
    if (placement.group != group || !containsTile(available, placement.tile))
      return false;
  return true;
}

std::optional<NodeSpatialPlan> getNextNodePlan(
    const SpatialRootDomainFacts &root, llvm::ArrayRef<TileId> available,
    const NodeSpatialPlan &current, SpatialSuccessorDomain successorDomain) {
  if (!nodePlanContains(root, available, current))
    return std::nullopt;
  NodeSpatialPlan next = current;
  if (successorDomain == SpatialSuccessorDomain::AllPlacements) {
    if (nextMergePlacement(available, next.reductionMerges))
      return next;
    for (MergePlacement &placement : next.reductionMerges)
      placement.tile = available.front();
    if (nextDistinctTileSequence(available, next.embedding))
      return next;
  }
  if (!advanceAxes(root, available.size(), next.axes))
    return std::nullopt;
  std::optional<size_t> cells =
      getCellCount(root.iteratorExtents, next.axes, available.size());
  if (!cells)
    return std::nullopt;
  next.embedding.assign(available.begin(), available.begin() + *cells);
  mlir::FailureOr<llvm::SmallVector<ReductionGroupId, 8>> groups =
      deriveSpatialReductionGroups(root, next.axes);
  if (mlir::failed(groups))
    return std::nullopt;
  next.reductionMerges = getFirstMergePlacements(*groups, available);
  return next;
}

uint64_t getInternalHopWork(const TargetTopology &topology, CardId cardId,
                            llvm::ArrayRef<TileId> tiles) {
  uint64_t work = 0;
  for (size_t lhs = 0; lhs < tiles.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < tiles.size(); ++rhs) {
      std::optional<uint64_t> distance =
          topology.getOnCardShortestHopDistance(cardId, tiles[lhs], tiles[rhs]);
      if (!distance || work > std::numeric_limits<uint64_t>::max() - *distance)
        return std::numeric_limits<uint64_t>::max();
      work += *distance;
    }
  return work;
}

bool tileVectorLess(llvm::ArrayRef<TileId> lhs, llvm::ArrayRef<TileId> rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), tileLess);
}

llvm::SmallVector<TileId, 16>
getCompactEmbedding(const TargetTopology &topology, CardId cardId,
                    llvm::ArrayRef<TileId> available, size_t count) {
  llvm::SmallVector<llvm::SmallVector<TileId, 16>, 64> candidates;
  auto append = [&](llvm::SmallVector<TileId, 16> candidate) {
    if (candidate.size() != count)
      return;
    if (llvm::any_of(candidate, [&](TileId tile) {
          return !containsTile(available, tile);
        }))
      return;
    llvm::sort(candidate, [&](TileId lhs, TileId rhs) {
      std::optional<TileCoordinate> left = topology.getTileCoordinate(lhs);
      std::optional<TileCoordinate> right = topology.getTileCoordinate(rhs);
      if (!left || !right)
        return tileLess(lhs, rhs);
      return std::tuple(left->y, left->x, lhs.getValue()) <
             std::tuple(right->y, right->x, rhs.getValue());
    });
    if (!llvm::is_contained(candidates, candidate))
      candidates.push_back(std::move(candidate));
  };

  if (count == 0 || count > available.size())
    return {};
  append(llvm::SmallVector<TileId, 16>(available.begin(),
                                       available.begin() + count));
  llvm::ArrayRef<int64_t> grid = topology.getTileGrid();
  if (grid.size() == 2) {
    for (int64_t height = 1; height <= grid[0]; ++height)
      for (int64_t width = 1; width <= grid[1]; ++width) {
        if (static_cast<size_t>(height * width) != count)
          continue;
        for (int64_t y = 0; y <= grid[0] - height; ++y)
          for (int64_t x = 0; x <= grid[1] - width; ++x) {
            llvm::SmallVector<TileId, 16> rectangle;
            for (int64_t dy = 0; dy < height; ++dy)
              for (int64_t dx = 0; dx < width; ++dx) {
                std::optional<TileId> tile =
                    topology.getTileId({y + dy, x + dx});
                if (tile && topology.isTileAvailable(cardId, *tile))
                  rectangle.push_back(*tile);
              }
            append(std::move(rectangle));
          }
      }
  }
  for (TileId start : available) {
    llvm::SmallVector<TileId, 16> queue{start};
    llvm::SmallVector<TileId, 16> visited{start};
    for (size_t cursor = 0; cursor < queue.size() && visited.size() < count;
         ++cursor) {
      mlir::FailureOr<llvm::SmallVector<TileId, 4>> neighbors =
          topology.getOnCardNeighbors(cardId, queue[cursor]);
      if (mlir::failed(neighbors))
        break;
      for (TileId neighbor : *neighbors) {
        if (containsTile(visited, neighbor))
          continue;
        visited.push_back(neighbor);
        queue.push_back(neighbor);
        if (visited.size() == count)
          break;
      }
    }
    append(std::move(visited));
  }
  llvm::sort(candidates, [&](const auto &lhs, const auto &rhs) {
    uint64_t lhsWork = getInternalHopWork(topology, cardId, lhs);
    uint64_t rhsWork = getInternalHopWork(topology, cardId, rhs);
    if (lhsWork != rhsWork)
      return lhsWork < rhsWork;
    return tileVectorLess(lhs, rhs);
  });
  return candidates.empty() ? llvm::SmallVector<TileId, 16>{}
                            : candidates.front();
}

enum class BalancedAxisSelection : uint8_t {
  Constructive,
  AllPartitionable,
};

struct OperandProjection {
  llvm::SmallBitVector iterators;
  uint64_t bytes = 0;
};

std::optional<llvm::SmallVector<OperandProjection, 4>>
getReadOperandProjections(mlir::Operation *operation) {
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!linalg)
    return std::nullopt;
  llvm::SmallVector<OperandProjection, 4> result;
  for (mlir::OpOperand &operand : linalg->getOpOperands()) {
    if (!linalg.payloadUsesValueFromOperand(&operand))
      continue;
    mlir::AffineMap map = linalg.getMatchingIndexingMap(&operand);
    if (!map.isProjectedPermutation())
      return std::nullopt;
    mlir::Type element = operand.get().getType();
    uint64_t elements = 1;
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(element)) {
      if (!shaped.hasStaticShape())
        return std::nullopt;
      for (int64_t extent : shaped.getShape()) {
        if (extent <= 0)
          return std::nullopt;
        elements =
            llvm::SaturatingMultiply(elements, static_cast<uint64_t>(extent));
      }
      element = shaped.getElementType();
    }
    if (!element.isIntOrFloat())
      return std::nullopt;
    OperandProjection projection;
    projection.bytes = llvm::SaturatingMultiply(
        elements,
        static_cast<uint64_t>((element.getIntOrFloatBitWidth() + 7) / 8));
    projection.iterators.resize(map.getNumDims(), false);
    for (mlir::AffineExpr expression : map.getResults())
      projection.iterators.set(
          mlir::cast<mlir::AffineDimExpr>(expression).getPosition());
    result.push_back(std::move(projection));
  }
  return result;
}

// This ranks explicit source-domain partitions, before any movement exists.
// It is neither an actual traffic inventory nor a memory-admission estimate.
uint64_t getRepeatedOperandBytes(llvm::ArrayRef<OperandProjection> operands,
                                 llvm::ArrayRef<IteratorPartition> axes) {
  uint64_t bytes = 0;
  for (const OperandProjection &operand : operands) {
    uint64_t repeated = operand.bytes;
    for (auto [iterator, axis] : llvm::enumerate(axes))
      if (!operand.iterators.test(iterator))
        repeated = llvm::SaturatingMultiply(
            repeated, static_cast<uint64_t>(axis.parameter));
    bytes = llvm::SaturatingAdd(bytes, repeated);
  }
  return bytes;
}

bool canPartitionForProposal(const SpatialRootDomainFacts &root,
                             size_t iterator, BalancedAxisSelection selection) {
  if (selection == BalancedAxisSelection::AllPartitionable)
    return canPartitionIterator(root, iterator);
  if (root.partitionableParallelIterators.test(iterator))
    return true;
  return root.attention &&
         root.attention->keyValuePartition ==
             AttentionKeyValuePartitionRequirement::MultipleIntervals &&
         llvm::is_contained(root.attention->keyValueReductionIterators,
                            static_cast<unsigned>(iterator));
}

void findMaximumBalancedAxes(const SpatialRootDomainFacts &root,
                             BalancedAxisSelection selection, size_t tileCount,
                             size_t iterator, size_t cells,
                             llvm::SmallVectorImpl<IteratorPartition> &current,
                             llvm::SmallVectorImpl<IteratorPartition> &best,
                             size_t &bestCells,
                             llvm::ArrayRef<OperandProjection> operands) {
  if (iterator == root.iteratorExtents.size()) {
    if (checkAttention(root, current) !=
        AttentionSpatialConstraintViolation::None)
      return;
    bool prefer = cells > bestCells;
    if (cells == bestCells) {
      uint64_t currentBytes = getRepeatedOperandBytes(operands, current);
      uint64_t bestBytes = getRepeatedOperandBytes(operands, best);
      prefer = currentBytes < bestBytes ||
               (currentBytes == bestBytes &&
                std::lexicographical_compare(best.begin(), best.end(),
                                             current.begin(), current.end()));
    }
    if (prefer) {
      best.assign(current.begin(), current.end());
      bestCells = cells;
    }
    return;
  }
  const size_t maximumFactor =
      canPartitionForProposal(root, iterator, selection)
          ? std::min<size_t>(
                static_cast<size_t>(root.iteratorExtents[iterator]),
                tileCount / cells)
          : 1;
  for (size_t factor = 1; factor <= maximumFactor; ++factor) {
    current.push_back({static_cast<uint32_t>(iterator),
                       IteratorPartitionScheme::BalancedParts,
                       static_cast<int64_t>(factor)});
    findMaximumBalancedAxes(root, selection, tileCount, iterator + 1,
                            cells * factor, current, best, bestCells, operands);
    current.pop_back();
  }
}

std::optional<llvm::SmallVector<IteratorPartition, 4>> getMaximumBalancedAxes(
    const SpatialRootDomainFacts &root, size_t tileCount,
    BalancedAxisSelection selection = BalancedAxisSelection::AllPartitionable,
    llvm::ArrayRef<OperandProjection> operands = {}) {
  if (tileCount == 0)
    return std::nullopt;
  llvm::SmallVector<IteratorPartition, 4> current;
  llvm::SmallVector<IteratorPartition, 4> best;
  size_t bestCells = 0;
  findMaximumBalancedAxes(root, selection, tileCount, 0, 1, current, best,
                          bestCells, operands);
  if (bestCells == 0)
    return std::nullopt;
  return best;
}

std::optional<llvm::SmallVector<IteratorPartition, 4>>
getFirstUniformAxes(const SpatialRootDomainFacts &root, size_t tileCount) {
  std::optional<llvm::SmallVector<IteratorPartition, 4>> current =
      getFirstAxes(root, tileCount);
  while (current) {
    if (llvm::any_of(*current, [](const IteratorPartition &partition) {
          return partition.scheme == IteratorPartitionScheme::UniformExtent;
        }))
      return current;
    if (!advanceAxes(root, tileCount, *current))
      break;
  }
  return std::nullopt;
}

std::optional<TileId> getFirstContributor(
    const SpatialRootDomainFacts &root, llvm::ArrayRef<IteratorPartition> axes,
    llvm::ArrayRef<TileId> embedding, const ReductionGroupId &group) {
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> counts =
      getIntervalCounts(root.iteratorExtents, axes);
  if (mlir::failed(counts) ||
      group.resultGroup >= root.resultParallelIteratorsByGroup.size())
    return std::nullopt;
  const llvm::SmallBitVector &resultParallel =
      root.resultParallelIteratorsByGroup[group.resultGroup];
  for (size_t cell = 0; cell < embedding.size(); ++cell) {
    size_t remainder = cell;
    llvm::SmallVector<uint32_t, 8> coordinate(counts->size());
    for (size_t reverse = 0; reverse < counts->size(); ++reverse) {
      size_t iterator = counts->size() - reverse - 1;
      coordinate[iterator] =
          remainder % static_cast<size_t>((*counts)[iterator]);
      remainder /= static_cast<size_t>((*counts)[iterator]);
    }
    llvm::SmallVector<uint32_t, 4> parallel;
    for (int iterator = resultParallel.find_first(); iterator >= 0;
         iterator = resultParallel.find_next(iterator))
      parallel.push_back(coordinate[iterator]);
    if (parallel == group.parallelCoordinate)
      return embedding[cell];
  }
  return std::nullopt;
}

} // namespace

const SpatialRootDomainFacts *
SpatialDomainProblem::findRoot(const SemanticRootKey &root) const {
  auto found = llvm::lower_bound(
      roots, root,
      [](const SpatialRootDomainFacts &facts,
         const SemanticRootKey &candidate) { return facts.root < candidate; });
  return found == roots.end() || found->root != root ? nullptr : &*found;
}

mlir::FailureOr<SpatialPlan> buildGraphCoherentSpatialProposal(
    const SpatialDomainProblem &problem, const SpatialPlan &plan,
    const SpatialAssignment &assignment,
    const analysis::ExactDemandProof &demand, std::string *failureReason) {
  const SpatialPlanningProblem &structure = problem.getStructuralProblem();
  if (mlir::failed(
          validateSpatialPlanStructure(structure, plan, failureReason)))
    return mlir::failure();

  std::map<LogicalShardId, TileId> tilesByShard;
  std::map<SemanticRootKey, const NodeExecutionPartition *> partitions;
  for (const NodeExecutionPartition &partition : assignment.nodes) {
    if (!partitions.try_emplace(partition.root, &partition).second) {
      if (failureReason)
        *failureReason =
            "graph-coherent proposal has duplicate assignment roots";
      return mlir::failure();
    }
    for (const ExecutionShard &shard : partition.shards)
      if (!tilesByShard.try_emplace(shard.shard, shard.tile).second) {
        if (failureReason)
          *failureReason =
              "graph-coherent proposal has duplicate logical shards";
        return mlir::failure();
      }
  }

  std::map<analysis::DemandDestination, TileId> destinations;
  for (const auto &[shard, tile] : tilesByShard)
    destinations.emplace(shard, tile);
  for (const auto &node : assignment.nodes)
    for (const auto &merge : node.reductionGroups)
      destinations.emplace(merge.group, merge.mergeTile);
  using TileVotes = std::map<int64_t, uint64_t>;
  std::map<LogicalShardId, TileVotes> votes;
  for (const analysis::DependencyDemand &dependency : demand.dependencyDemands)
    for (const analysis::DestinationDemand &destination :
         dependency.perDestination) {
      auto destinationTile = destinations.find(destination.destination);
      if (destinationTile == destinations.end() ||
          destinationTile->second != destination.destinationTile) {
        if (failureReason)
          *failureReason =
              "graph-coherent proposal has stale destination ownership";
        return mlir::failure();
      }
      for (const analysis::SourceDemand &source : destination.sources)
        for (const analysis::OwnerIntersection &owner :
             source.eligibleFinalOwners) {
          if (!owner.ownerShard || owner.domain.isEmpty())
            continue;
          auto ownerTile = tilesByShard.find(*owner.ownerShard);
          if (ownerTile == tilesByShard.end() ||
              ownerTile->second != owner.tile) {
            if (failureReason)
              *failureReason =
                  "graph-coherent proposal has stale producer ownership";
            return mlir::failure();
          }
          uint64_t &weight =
              votes[*owner.ownerShard][destination.destinationTile.getValue()];
          if (weight == std::numeric_limits<uint64_t>::max()) {
            if (failureReason)
              *failureReason = "graph-coherent proposal vote count overflows";
            return mlir::failure();
          }
          ++weight;
        }
    }

  llvm::ArrayRef<TileId> available = structure.getAvailableTiles();
  SpatialPlan result = plan;
  for (NodeSpatialPlan &node : result.nodes) {
    auto partition = partitions.find(node.root);
    if (partition == partitions.end() ||
        partition->second->shards.size() != node.embedding.size() ||
        node.embedding.size() > available.size()) {
      if (failureReason)
        *failureReason =
            "graph-coherent proposal plan/assignment roots do not match";
      return mlir::failure();
    }
    const size_t shardCount = node.embedding.size();
    const size_t tileCount = available.size();
    bool hasVote = false;
    for (auto [index, shard] : llvm::enumerate(partition->second->shards)) {
      if (shard.shard.root != node.root ||
          shard.tile != node.embedding[index]) {
        if (failureReason)
          *failureReason =
              "graph-coherent proposal plan/assignment embedding is stale";
        return mlir::failure();
      }
      hasVote |= votes.count(shard.shard) != 0;
    }
    if (!hasVote)
      continue;

    const int64_t primaryScale = static_cast<int64_t>(shardCount) + 1;
    std::vector<std::vector<int64_t>> costs(
        shardCount + 1, std::vector<int64_t>(tileCount + 1));
    for (size_t row = 1; row <= shardCount; ++row) {
      const ExecutionShard &shard = partition->second->shards[row - 1];
      auto shardVotes = votes.find(shard.shard);
      for (size_t column = 1; column <= tileCount; ++column) {
        uint64_t weight = 0;
        if (shardVotes != votes.end()) {
          auto found =
              shardVotes->second.find(available[column - 1].getValue());
          if (found != shardVotes->second.end())
            weight = found->second;
        }
        const uint64_t maximumWeight = static_cast<uint64_t>(
            std::numeric_limits<int64_t>::max() / primaryScale);
        if (weight > maximumWeight) {
          if (failureReason)
            *failureReason = "graph-coherent proposal score overflows";
          return mlir::failure();
        }
        int64_t score = static_cast<int64_t>(weight) * primaryScale;
        score += node.embedding[row - 1] == available[column - 1];
        costs[row][column] = -score;
      }
    }

    // Rectangular Hungarian assignment. Rows use canonical logical-shard
    // order and columns use sorted physical Tile IDs; equal scores therefore
    // have a complete, deterministic semantic tie-break.
    const int64_t infinity = std::numeric_limits<int64_t>::max() / 4;
    std::vector<int64_t> rowPotential(shardCount + 1);
    std::vector<int64_t> columnPotential(tileCount + 1);
    std::vector<size_t> matchedRow(tileCount + 1);
    std::vector<size_t> predecessor(tileCount + 1);
    for (size_t row = 1; row <= shardCount; ++row) {
      matchedRow[0] = row;
      size_t column = 0;
      std::vector<int64_t> minimum(tileCount + 1, infinity);
      std::vector<bool> used(tileCount + 1, false);
      do {
        used[column] = true;
        const size_t currentRow = matchedRow[column];
        int64_t delta = infinity;
        size_t nextColumn = 0;
        for (size_t candidate = 1; candidate <= tileCount; ++candidate) {
          if (used[candidate])
            continue;
          const int64_t reduced = costs[currentRow][candidate] -
                                  rowPotential[currentRow] -
                                  columnPotential[candidate];
          if (reduced < minimum[candidate]) {
            minimum[candidate] = reduced;
            predecessor[candidate] = column;
          }
          if (minimum[candidate] < delta) {
            delta = minimum[candidate];
            nextColumn = candidate;
          }
        }
        if (nextColumn == 0 || delta == infinity) {
          if (failureReason)
            *failureReason =
                "graph-coherent proposal has no injective Tile embedding";
          return mlir::failure();
        }
        for (size_t candidate = 0; candidate <= tileCount; ++candidate) {
          if (used[candidate]) {
            rowPotential[matchedRow[candidate]] += delta;
            columnPotential[candidate] -= delta;
          } else {
            minimum[candidate] -= delta;
          }
        }
        column = nextColumn;
      } while (matchedRow[column] != 0);
      do {
        const size_t previous = predecessor[column];
        matchedRow[column] = matchedRow[previous];
        column = previous;
      } while (column != 0);
    }

    llvm::SmallVector<TileId, 16> embedding(shardCount, TileId(0));
    for (size_t column = 1; column <= tileCount; ++column)
      if (matchedRow[column] != 0)
        embedding[matchedRow[column] - 1] = available[column - 1];
    node.embedding = std::move(embedding);
  }

  if (mlir::failed(
          validateSpatialPlanStructure(structure, result, failureReason)))
    return mlir::failure();
  return result;
}

SpatialDomainProblemResult
buildSpatialDomainProblem(const StructuredDAGAnalysis &dag,
                          llvm::ArrayRef<TileId> availableTiles) {
  if (availableTiles.size() > std::numeric_limits<uint32_t>::max())
    return {{},
            fail(SpatialDomainFailureKind::BrokenContract,
                 "spatial Tile domain is not representable")};
  std::string detail;
  mlir::FailureOr<SemanticRootAnalysis> semanticRoots =
      SemanticRootAnalysis::create(dag, &detail);
  if (mlir::failed(semanticRoots))
    return {{}, fail(SpatialDomainFailureKind::BrokenContract, detail)};

  llvm::SmallVector<SpatialRootDomainFacts, 16> roots;
  llvm::SmallVector<NodeIterationSpace, 16> spaces;
  roots.reserve(semanticRoots->getRoots().size());
  spaces.reserve(semanticRoots->getRoots().size());
  for (const SemanticRootBinding &binding : semanticRoots->getRoots()) {
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(binding.operation);
    auto destination =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(binding.operation);
    if (!tiling || !destination)
      return {{},
              fail(SpatialDomainFailureKind::UnsupportedSemantics,
                   "structured root lacks tiling or destination semantics",
                   binding.key)};
    SpatialRootDomainFacts facts;
    facts.root = binding.key;
    auto node = llvm::find_if(dag.getNodes(), [&](const StructuredDAGNode &n) {
      return n.operation == binding.operation;
    });
    if (node == dag.getNodes().end())
      return {{},
              fail(SpatialDomainFailureKind::BrokenContract,
                   "semantic root is absent from the structured DAG",
                   binding.key)};
    facts.node = node->id;

    llvm::SmallVector<mlir::utils::IteratorType, 8> iteratorTypes =
        tiling.getLoopIteratorTypes();
    llvm::SmallVector<mlir::AffineMap, 4> resultMaps;
    if (auto linalg =
            mlir::dyn_cast<mlir::linalg::LinalgOp>(binding.operation)) {
      facts.iteratorExtents = linalg.getStaticLoopRanges();
      for (int64_t result = 0; result < destination.getNumDpsInits(); ++result)
        resultMaps.push_back(linalg.getMatchingIndexingMap(
            destination.getDpsInitOperand(result)));
    } else if (auto attention =
                   mlir::dyn_cast<LinalgExtAttentionOp>(binding.operation)) {
      facts.iteratorExtents = attention.getStaticLoopRanges();
      resultMaps.push_back(attention.getOutputMap());
      mlir::FailureOr<AttentionSpatialConstraints> constraints =
          deriveAttentionSpatialConstraints(attention);
      if (mlir::failed(constraints))
        return {{},
                fail(SpatialDomainFailureKind::BrokenContract,
                     "attention spatial constraints are malformed",
                     binding.key)};
      facts.attention = std::move(*constraints);
    } else {
      return {{},
              fail(SpatialDomainFailureKind::UnsupportedSemantics,
                   "structured root lacks typed indexing semantics",
                   binding.key)};
    }
    if (facts.iteratorExtents.size() != iteratorTypes.size() ||
        facts.iteratorExtents.size() > std::numeric_limits<uint32_t>::max() ||
        resultMaps.empty() ||
        resultMaps.size() > std::numeric_limits<uint32_t>::max() ||
        llvm::any_of(facts.iteratorExtents,
                     [](int64_t extent) { return extent <= 0; }))
      return {{},
              fail(SpatialDomainFailureKind::UnsupportedSemantics,
                   "structured root requires positive static iterators",
                   binding.key)};

    facts.partitionableParallelIterators.resize(iteratorTypes.size(), false);
    facts.partitionableReductionIterators.resize(iteratorTypes.size(), false);
    for (auto [iterator, type] : llvm::enumerate(iteratorTypes)) {
      if (type == mlir::utils::IteratorType::parallel)
        facts.iteratorKinds.push_back(SpatialIteratorKind::Parallel);
      else if (type == mlir::utils::IteratorType::reduction)
        facts.iteratorKinds.push_back(SpatialIteratorKind::Reduction);
      else
        return {{},
                fail(SpatialDomainFailureKind::UnsupportedSemantics,
                     "structured root has an unsupported iterator kind",
                     binding.key)};
    }
    bool firstResultMap = true;
    bool projectedResultMaps = true;
    for (mlir::AffineMap map : resultMaps) {
      if (!map || map.getNumDims() != iteratorTypes.size() ||
          map.getNumSymbols() != 0)
        return {{},
                fail(SpatialDomainFailureKind::UnsupportedSemantics,
                     "structured result map has an invalid loop domain",
                     binding.key)};
      llvm::SmallBitVector resultParallel(iteratorTypes.size(), false);
      for (mlir::AffineExpr expression : map.getResults()) {
        auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!dimension || dimension.getPosition() >= iteratorTypes.size()) {
          // A non-projected result index cannot prove which result elements a
          // parallel or reduction partition owns. Keep the root in the domain
          // as one unpartitioned cell instead of failing the complete spatial
          // domain: its typed indexing semantics stay exact and analyzable,
          // and the unpartitioned point is an ordinary domain member.
          projectedResultMaps = false;
          resultParallel.reset();
          break;
        }
        if (iteratorTypes[dimension.getPosition()] ==
            mlir::utils::IteratorType::parallel) {
          resultParallel.set(dimension.getPosition());
        }
      }
      if (firstResultMap) {
        facts.partitionableParallelIterators = resultParallel;
        firstResultMap = false;
      } else {
        facts.partitionableParallelIterators &= resultParallel;
      }
      facts.resultParallelIteratorsByGroup.push_back(std::move(resultParallel));
    }

    if (mlir::isa<mlir::PartialReductionOpInterface>(binding.operation)) {
      for (auto [iterator, kind] : llvm::enumerate(facts.iteratorKinds))
        if (kind == SpatialIteratorKind::Reduction)
          facts.partitionableReductionIterators.set(iterator);
      facts.reductionResultGroupCount =
          static_cast<uint32_t>(facts.resultParallelIteratorsByGroup.size());
    }
    if (auto coupled = mlir::dyn_cast<WaferCoupledReductionOpInterface>(
            binding.operation)) {
      CoupledReductionDescription description =
          coupled.getCoupledReductionDescription();
      if (description.components.empty())
        return {{},
                fail(SpatialDomainFailureKind::BrokenContract,
                     "coupled reduction has no state components", binding.key)};
      for (unsigned iterator : description.reductionIterators) {
        if (iterator >= iteratorTypes.size() ||
            iteratorTypes[iterator] != mlir::utils::IteratorType::reduction)
          return {{},
                  fail(SpatialDomainFailureKind::BrokenContract,
                       "coupled reduction iterator domain is malformed",
                       binding.key)};
        facts.partitionableReductionIterators.set(iterator);
      }
      facts.reductionResultGroupCount = 1;
      facts.resultParallelIteratorsByGroup = {
          facts.partitionableParallelIterators};
    }
    if (!projectedResultMaps) {
      // Every result index must be a projected dimension before any partition
      // can own its elements exactly. Withholding reduction partitioning too
      // keeps this root at the single unpartitioned cell rather than inventing
      // contribution groups it cannot justify.
      facts.partitionableParallelIterators.reset();
      facts.partitionableReductionIterators.reset();
      facts.reductionResultGroupCount = 0;
    }

    NodeIterationSpace space;
    space.root = facts.root;
    space.iteratorExtents.assign(facts.iteratorExtents.begin(),
                                 facts.iteratorExtents.end());
    spaces.push_back(std::move(space));
    roots.push_back(std::move(facts));
  }
  mlir::FailureOr<SpatialPlanningProblem> structural =
      SpatialPlanningProblem::create(spaces, availableTiles, &detail);
  if (mlir::failed(structural))
    return {{}, fail(SpatialDomainFailureKind::BrokenContract, detail)};

  llvm::SmallVector<uint32_t, 16> rootIndexByNode(
      dag.getNodes().size(), std::numeric_limits<uint32_t>::max());
  for (auto [index, root] : llvm::enumerate(roots)) {
    if (root.node >= rootIndexByNode.size() ||
        index > std::numeric_limits<uint32_t>::max())
      return {{},
              fail(SpatialDomainFailureKind::BrokenContract,
                   "spatial root index is not representable")};
    rootIndexByNode[root.node] = static_cast<uint32_t>(index);
  }
  llvm::SmallVector<llvm::SmallVector<uint32_t, 4>, 4> components;
  for (const StructuredDAGDependencyComponent &component :
       dag.getObservableDependencyComponents()) {
    llvm::SmallVector<uint32_t, 4> rootsInComponent;
    for (StructuredDAGNodeID node : component.nodes) {
      if (node >= rootIndexByNode.size() ||
          rootIndexByNode[node] == std::numeric_limits<uint32_t>::max())
        return {{},
                fail(SpatialDomainFailureKind::BrokenContract,
                     "spatial component has no semantic root")};
      rootsInComponent.push_back(rootIndexByNode[node]);
    }
    llvm::sort(rootsInComponent);
    components.push_back(std::move(rootsInComponent));
  }
  return {SpatialDomainProblem(std::move(*semanticRoots),
                               std::move(*structural), std::move(roots),
                               std::move(components)),
          {}};
}

mlir::FailureOr<llvm::SmallVector<ReductionGroupId, 8>>
deriveSpatialReductionGroups(const SpatialRootDomainFacts &root,
                             llvm::ArrayRef<IteratorPartition> partitions,
                             std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> counts =
      getIntervalCounts(root.iteratorExtents, partitions, failureReason);
  if (mlir::failed(counts))
    return mlir::failure();
  bool partitionsReduction = false;
  for (auto [iterator, kind] : llvm::enumerate(root.iteratorKinds)) {
    if (kind != SpatialIteratorKind::Reduction || (*counts)[iterator] == 1)
      continue;
    if (!root.partitionableReductionIterators.test(iterator)) {
      if (failureReason)
        *failureReason =
            "spatial reduction lacks complete source-owned mechanics";
      return mlir::failure();
    }
    partitionsReduction = true;
  }
  if (!partitionsReduction)
    return llvm::SmallVector<ReductionGroupId, 8>{};
  if (root.reductionResultGroupCount == 0) {
    if (failureReason)
      *failureReason = "spatial reduction has no result group contract";
    return mlir::failure();
  }

  if (root.resultParallelIteratorsByGroup.size() !=
      root.reductionResultGroupCount) {
    if (failureReason)
      *failureReason = "spatial reduction result-group maps are incomplete";
    return mlir::failure();
  }

  llvm::SmallVector<ReductionGroupId, 8> groups;
  for (uint32_t resultGroup = 0; resultGroup < root.reductionResultGroupCount;
       ++resultGroup) {
    const llvm::SmallBitVector &resultParallel =
        root.resultParallelIteratorsByGroup[resultGroup];
    llvm::SmallVector<int64_t, 4> parallelCounts;
    for (int iterator = resultParallel.find_first(); iterator >= 0;
         iterator = resultParallel.find_next(iterator))
      parallelCounts.push_back((*counts)[iterator]);
    std::optional<size_t> coordinateCount =
        checkedProduct(parallelCounts, std::numeric_limits<size_t>::max());
    if (!coordinateCount)
      return mlir::failure();
    for (size_t linear = 0; linear < *coordinateCount; ++linear) {
      size_t remainder = linear;
      ReductionGroupId group;
      group.root = root.root;
      group.resultGroup = resultGroup;
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
  llvm::sort(groups);
  return groups;
}

SpatialPlanDomainResult buildSpatialPlanDomain(const StructuredDAGAnalysis &dag,
                                               const TargetTopology &topology,
                                               CardId cardId) {
  std::optional<llvm::ArrayRef<TileId>> available =
      topology.getAvailableTileIds(cardId);
  if (!available || available->empty())
    return {{},
            fail(SpatialDomainFailureKind::BrokenContract,
                 "spatial domain has no available Tile endpoints")};
  SpatialDomainProblemResult problem =
      buildSpatialDomainProblem(dag, *available);
  if (!problem.succeeded())
    return {{}, std::move(problem.failure)};
  SpatialPlanDomain domain(std::move(*problem.problem), topology, cardId);
  SpatialPlan first = domain.getFirstPlan();
  if (!domain.contains(first))
    return {{},
            fail(SpatialDomainFailureKind::UnsupportedSemantics,
                 "spatial domain has no legal complete plan")};
  return {std::move(domain), {}};
}

SpatialPlan SpatialPlanDomain::getFirstPlan() const {
  SpatialPlan plan;
  llvm::ArrayRef<TileId> available =
      problem.getStructuralProblem().getAvailableTiles();
  for (const SpatialRootDomainFacts &root : problem.getRoots()) {
    std::optional<NodeSpatialPlan> node = getFirstNodePlan(root, available);
    if (!node)
      return {};
    plan.nodes.push_back(std::move(*node));
  }
  return plan;
}

bool SpatialPlanDomain::contains(const SpatialPlan &plan) const {
  if (mlir::failed(
          validateSpatialPlanStructure(problem.getStructuralProblem(), plan)) ||
      plan.nodes.size() != problem.getRoots().size())
    return false;
  llvm::ArrayRef<TileId> available =
      problem.getStructuralProblem().getAvailableTiles();
  return llvm::all_of(llvm::zip_equal(problem.getRoots(), plan.nodes),
                      [&](auto values) {
                        const auto &[root, node] = values;
                        return nodePlanContains(root, available, node);
                      });
}

SpatialPlanSuccessor
SpatialPlanDomain::getNextPlan(const SpatialPlan &plan,
                               SpatialSuccessorDomain successorDomain) const {
  if (!contains(plan))
    return {SpatialPlanSuccessorKind::Failure,
            {},
            fail(SpatialDomainFailureKind::BrokenContract,
                 "spatial successor input is outside its typed domain")};
  SpatialPlan next = plan;
  llvm::ArrayRef<TileId> available =
      problem.getStructuralProblem().getAvailableTiles();
  for (size_t reverse = 0; reverse < next.nodes.size(); ++reverse) {
    size_t index = next.nodes.size() - reverse - 1;
    std::optional<NodeSpatialPlan> successor =
        getNextNodePlan(problem.getRoots()[index], available, next.nodes[index],
                        successorDomain);
    if (!successor)
      continue;
    next.nodes[index] = std::move(*successor);
    for (size_t suffix = index + 1; suffix < next.nodes.size(); ++suffix) {
      std::optional<NodeSpatialPlan> first =
          getFirstNodePlan(problem.getRoots()[suffix], available);
      if (!first)
        return {SpatialPlanSuccessorKind::Failure,
                {},
                fail(SpatialDomainFailureKind::BrokenContract,
                     "spatial suffix has no first plan")};
      next.nodes[suffix] = std::move(*first);
    }
    return {SpatialPlanSuccessorKind::Successor, std::move(next), {}};
  }
  return {SpatialPlanSuccessorKind::End, {}, {}};
}

mlir::FailureOr<SpatialAssignment>
SpatialPlanDomain::close(const SpatialPlan &plan,
                         std::string *failureReason) const {
  if (!contains(plan)) {
    if (failureReason)
      *failureReason = "SpatialPlan is outside the complete spatial domain";
    return mlir::failure();
  }
  return closeSpatialPlanStructure(problem.getStructuralProblem(), plan,
                                   failureReason);
}

SpatialDomainEvaluation
SpatialPlanDomain::evaluate(const StructuredDAGAnalysis &dag,
                            const SpatialPlan &plan,
                            const analysis::IndexRelationLimits &limits) const {
  SpatialDomainEvaluation evaluation;
  std::string detail;
  mlir::FailureOr<SpatialAssignment> assignment = close(plan, &detail);
  if (mlir::failed(assignment)) {
    evaluation.failure = fail(SpatialDomainFailureKind::BrokenContract, detail);
    return evaluation;
  }
  mlir::FailureOr<DemandPlanningSession> session =
      DemandPlanningSession::create(dag, limits, &detail);
  if (mlir::failed(session)) {
    evaluation.failure = fail(SpatialDomainFailureKind::BrokenContract, detail);
    return evaluation;
  }
  evaluation.assignment = std::move(*assignment);
  evaluation.demand = session->query(*evaluation.assignment);
  session->close();
  return evaluation;
}

llvm::SmallVector<SpatialPlan, 4> SpatialPlanDomain::getProposals() const {
  llvm::SmallVector<SpatialPlan, 4> proposals;
  auto append = [&](SpatialPlan plan) {
    if (contains(plan) && !llvm::is_contained(proposals, plan))
      proposals.push_back(std::move(plan));
  };

  llvm::ArrayRef<TileId> available =
      problem.getStructuralProblem().getAvailableTiles();
  auto buildNode =
      [&](const SpatialRootDomainFacts &root,
          llvm::SmallVector<IteratorPartition, 4> axes,
          llvm::ArrayRef<TileId> tilePool) -> std::optional<NodeSpatialPlan> {
    std::optional<size_t> cells =
        getCellCount(root.iteratorExtents, axes, tilePool.size());
    if (!cells)
      return std::nullopt;
    NodeSpatialPlan node;
    node.root = root.root;
    node.axes = std::move(axes);
    node.embedding = getCompactEmbedding(topology, cardId, tilePool, *cells);
    mlir::FailureOr<llvm::SmallVector<ReductionGroupId, 8>> groups =
        deriveSpatialReductionGroups(root, node.axes);
    if (node.embedding.size() != *cells || mlir::failed(groups))
      return std::nullopt;
    for (const ReductionGroupId &group : *groups) {
      std::optional<TileId> contributor =
          getFirstContributor(root, node.axes, node.embedding, group);
      if (!contributor)
        return std::nullopt;
      node.reductionMerges.push_back({group, *contributor});
    }
    return node;
  };

  // The first point retains maximum parallelism while reducing replicated
  // reads of current operands. The original constructive point follows it;
  // neither proposal removes any raw-domain member or decides actual cost.
  SpatialPlan reuse;
  bool complete = true;
  for (const SpatialRootDomainFacts &root : problem.getRoots()) {
    const auto *binding = problem.getSemanticRoots().find(root.root);
    auto operands =
        binding ? getReadOperandProjections(binding->operation) : std::nullopt;
    if (operands &&
        llvm::any_of(*operands, [&](const OperandProjection &operand) {
          return operand.iterators.size() != root.iteratorExtents.size();
        }))
      operands.reset();
    std::optional<llvm::SmallVector<IteratorPartition, 4>> axes =
        getMaximumBalancedAxes(
            root, available.size(), BalancedAxisSelection::Constructive,
            operands ? llvm::ArrayRef<OperandProjection>(*operands)
                     : llvm::ArrayRef<OperandProjection>{});
    auto node =
        axes ? buildNode(root, std::move(*axes), available) : std::nullopt;
    if (!node) {
      complete = false;
      break;
    }
    reuse.nodes.push_back(std::move(*node));
  }
  if (complete)
    append(std::move(reuse));

  // The original constructive point uses as many parallel partitions as the
  // target admits, while leaving optional reductions local. Flash decoding's
  // required key/value partition remains explicit. This is an ordinary exact
  // domain member; later proposals still cover reduction-parallel points.
  SpatialPlan constructive;
  complete = true;
  for (const SpatialRootDomainFacts &root : problem.getRoots()) {
    std::optional<llvm::SmallVector<IteratorPartition, 4>> axes =
        getMaximumBalancedAxes(root, available.size(),
                               BalancedAxisSelection::Constructive);
    std::optional<NodeSpatialPlan> node =
        axes ? buildNode(root, std::move(*axes), available) : std::nullopt;
    if (!node) {
      complete = false;
      break;
    }
    constructive.nodes.push_back(std::move(*node));
  }
  if (complete)
    append(std::move(constructive));

  SpatialPlan maximum;
  complete = true;
  for (const SpatialRootDomainFacts &root : problem.getRoots()) {
    std::optional<llvm::SmallVector<IteratorPartition, 4>> axes =
        getMaximumBalancedAxes(root, available.size());
    std::optional<NodeSpatialPlan> node =
        axes ? buildNode(root, std::move(*axes), available) : std::nullopt;
    if (!node) {
      complete = false;
      break;
    }
    maximum.nodes.push_back(std::move(*node));
  }
  if (complete)
    append(maximum);

  // Independent observable components receive disjoint topology-shaped Tile
  // pools when such a complete point exists. Nodes within one dependency
  // component reuse that pool, preserving co-location opportunities.
  llvm::ArrayRef<llvm::SmallVector<uint32_t, 4>> components =
      problem.getComponents();
  if (components.size() > 1 && components.size() <= available.size()) {
    llvm::SmallVector<llvm::SmallVector<TileId, 16>, 4> componentPools;
    llvm::SmallVector<TileId, 16> remaining(available.begin(), available.end());
    complete = true;
    for (size_t component = 0; component < components.size(); ++component) {
      size_t remainingComponents = components.size() - component;
      size_t count = remaining.size() / remainingComponents;
      llvm::SmallVector<TileId, 16> pool =
          getCompactEmbedding(topology, cardId, remaining, count);
      if (pool.size() != count) {
        complete = false;
        break;
      }
      componentPools.push_back(pool);
      llvm::erase_if(remaining,
                     [&](TileId tile) { return containsTile(pool, tile); });
    }
    SpatialPlan disjoint;
    if (complete) {
      for (size_t rootIndex = 0; rootIndex < problem.getRoots().size();
           ++rootIndex) {
        auto component =
            llvm::find_if(components, [&](llvm::ArrayRef<uint32_t> roots) {
              return llvm::is_contained(roots,
                                        static_cast<uint32_t>(rootIndex));
            });
        if (component == components.end()) {
          complete = false;
          break;
        }
        size_t componentIndex =
            static_cast<size_t>(std::distance(components.begin(), component));
        llvm::ArrayRef<TileId> pool = componentPools[componentIndex];
        const SpatialRootDomainFacts root = problem.getRoots()[rootIndex];
        std::optional<llvm::SmallVector<IteratorPartition, 4>> axes =
            getMaximumBalancedAxes(root, pool.size());
        std::optional<NodeSpatialPlan> node =
            axes ? buildNode(root, std::move(*axes), pool) : std::nullopt;
        if (!node) {
          complete = false;
          break;
        }
        disjoint.nodes.push_back(std::move(*node));
      }
    }
    if (complete)
      append(std::move(disjoint));
  }

  SpatialPlan uniformBoundary;
  complete = true;
  for (const SpatialRootDomainFacts &root : problem.getRoots()) {
    std::optional<llvm::SmallVector<IteratorPartition, 4>> axes =
        getFirstUniformAxes(root, available.size());
    std::optional<NodeSpatialPlan> node =
        axes ? buildNode(root, std::move(*axes), available) : std::nullopt;
    if (!node) {
      complete = false;
      break;
    }
    uniformBoundary.nodes.push_back(std::move(*node));
  }
  if (complete)
    append(std::move(uniformBoundary));
  append(getFirstPlan());
  return proposals;
}

mlir::FailureOr<llvm::SmallVector<SpatialPlan, 8>>
SpatialPlanDomain::getGraphCoherentProposals(
    const StructuredDAGAnalysis &dag,
    const analysis::IndexRelationLimits &limits) const {
  llvm::SmallVector<SpatialPlan, 8> result;
  auto append = [&](SpatialPlan plan) {
    if (contains(plan) && !llvm::is_contained(result, plan))
      result.push_back(std::move(plan));
  };
  auto proposals = getProposals();
  // Coordinated reuse keeps the earliest entry. Raw reuse and the fixed-policy
  // coordinate remain separate seeds; propagation never removes either choice.
  if (!proposals.empty()) {
    const SpatialPlan seed = proposals.front();
    for (auto order : {SpatialPropagationOrder::ProducersFirst,
                       SpatialPropagationOrder::ConsumersFirst}) {
      auto coordinated =
          propagateSpatialPartitions(*this, dag, seed, limits, order);
      if (mlir::failed(coordinated))
        return mlir::failure();
      append(*coordinated);
      if (!llvm::is_contained(proposals, *coordinated))
        proposals.push_back(std::move(*coordinated));
    }
  }
  if (!proposals.empty())
    append(proposals.front());
  auto canonical = buildCanonicalSpatialAssignment(
      dag, problem.getStructuralProblem().getAvailableTiles());
  if (mlir::failed(canonical))
    return mlir::failure();
  append(std::move(canonical->plan));
  for (const SpatialPlan &raw : proposals) {
    SpatialDomainEvaluation evaluation = evaluate(dag, raw, limits);
    if (evaluation.failure)
      return mlir::failure();
    const analysis::ExactDemandProof *proof =
        evaluation.demand ? analysis::getExactDemandProof(*evaluation.demand)
                          : nullptr;
    if (evaluation.assignment && proof) {
      std::string detail;
      mlir::FailureOr<SpatialPlan> coherent = buildGraphCoherentSpatialProposal(
          problem, raw, *evaluation.assignment, *proof, &detail);
      if (mlir::failed(coherent))
        return mlir::failure();
      append(std::move(*coherent));
    }
    append(raw);
  }
  return result;
}

} // namespace wafer::compiler::detail
