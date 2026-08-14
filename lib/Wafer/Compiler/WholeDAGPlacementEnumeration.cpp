//===- WholeDAGPlacementEnumeration.cpp - Joint node placement ------------===//

#include "WholeDAGPlacementEnumeration.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace wafer::compiler::detail {
namespace {

void setFailure(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
}

uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  return product > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(product);
}

uint64_t saturatingAdd(uint64_t lhs, uint64_t rhs) {
  return rhs > std::numeric_limits<uint64_t>::max() - lhs
             ? std::numeric_limits<uint64_t>::max()
             : lhs + rhs;
}

bool tileVectorLess(llvm::ArrayRef<PhysicalTileId> lhs,
                    llvm::ArrayRef<PhysicalTileId> rhs) {
  return std::lexicographical_compare(
      lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
      [](PhysicalTileId left, PhysicalTileId right) {
        return left.getValue() < right.getValue();
      });
}

size_t getOverlap(llvm::ArrayRef<PhysicalTileId> lhs,
                  llvm::ArrayRef<PhysicalTileId> rhs) {
  size_t overlap = 0;
  for (PhysicalTileId tile : lhs)
    overlap += llvm::is_contained(rhs, tile);
  return overlap;
}

struct TopologyTileGroup {
  llvm::SmallVector<PhysicalTileId, 16> tiles;
  uint64_t internalHopWork = 0;
};

bool sameGroup(const TopologyTileGroup &lhs, const TopologyTileGroup &rhs) {
  return lhs.tiles == rhs.tiles;
}

uint64_t getInternalHopWork(const PhysicalTopology &topology,
                            PhysicalCardId cardId,
                            llvm::ArrayRef<PhysicalTileId> tiles) {
  // Pairwise topology distance is independent of Tile ID traversal order and
  // distinguishes a compact 2x2 rectangle from a 1x4 strip of the same size.
  uint64_t result = 0;
  for (size_t lhs = 0; lhs < tiles.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < tiles.size(); ++rhs) {
      std::optional<uint64_t> hops =
          topology.getOnCardShortestHopDistance(cardId, tiles[lhs], tiles[rhs]);
      if (!hops)
        return std::numeric_limits<uint64_t>::max();
      result = saturatingAdd(result, *hops);
    }
  return result;
}

llvm::SmallVector<TopologyTileGroup, 128>
deriveTopologyGroups(const PhysicalTopology &topology, PhysicalCardId cardId,
                     llvm::ArrayRef<PhysicalTileId> availableTiles) {
  llvm::SmallVector<TopologyTileGroup, 128> groups;
  if (availableTiles.empty())
    return groups;
  auto append = [&](llvm::ArrayRef<PhysicalTileId> tiles) {
    if (tiles.empty())
      return;
    TopologyTileGroup group;
    group.tiles.append(tiles.begin(), tiles.end());
    llvm::sort(group.tiles, [&](PhysicalTileId lhs, PhysicalTileId rhs) {
      std::optional<PhysicalTileCoordinate> left =
          topology.getTileCoordinate(lhs);
      std::optional<PhysicalTileCoordinate> right =
          topology.getTileCoordinate(rhs);
      if (!left || !right)
        return lhs.getValue() < rhs.getValue();
      return std::tuple(left->y, left->x, lhs.getValue()) <
             std::tuple(right->y, right->x, rhs.getValue());
    });
    if (llvm::any_of(groups, [&](const TopologyTileGroup &existing) {
          return sameGroup(existing, group);
        }))
      return;
    group.internalHopWork = getInternalHopWork(topology, cardId, group.tiles);
    if (group.internalHopWork != std::numeric_limits<uint64_t>::max())
      groups.push_back(std::move(group));
  };

  llvm::ArrayRef<int64_t> grid = topology.getTileGrid();
  if (grid.size() != 2 || grid[0] <= 0 || grid[1] <= 0)
    return {};
  // Enumerate physical rectangles by coordinate. On the target 4x4 mesh this
  // is the complete set of 100 axis-aligned translations, including every
  // singleton and all four 2x2 squares. A rectangle containing an unavailable
  // endpoint is not a physical participant group.
  for (int64_t height = 1; height <= grid[0]; ++height) {
    for (int64_t width = 1; width <= grid[1]; ++width) {
      for (int64_t y = 0; y <= grid[0] - height; ++y) {
        for (int64_t x = 0; x <= grid[1] - width; ++x) {
          llvm::SmallVector<PhysicalTileId, 16> rectangle;
          bool available = true;
          for (int64_t dy = 0; dy < height && available; ++dy) {
            for (int64_t dx = 0; dx < width; ++dx) {
              std::optional<PhysicalTileId> tile =
                  topology.getTileId(PhysicalTileCoordinate{y + dy, x + dx});
              if (!tile || !topology.isTileAvailable(cardId, *tile)) {
                available = false;
                break;
              }
              rectangle.push_back(*tile);
            }
          }
          if (available)
            append(rectangle);
        }
      }
    }
  }
  llvm::sort(groups,
             [](const TopologyTileGroup &lhs, const TopologyTileGroup &rhs) {
               if (lhs.tiles.size() != rhs.tiles.size())
                 return lhs.tiles.size() > rhs.tiles.size();
               if (lhs.internalHopWork != rhs.internalHopWork)
                 return lhs.internalHopWork < rhs.internalHopWork;
               return tileVectorLess(lhs.tiles, rhs.tiles);
             });
  return groups;
}

struct StaticSpatialAxis {
  unsigned iteratorDimension = 0;
  unsigned resultDimension = 0;
  uint64_t extent = 0;
  llvm::SmallVector<uint32_t, 4> basePartitionFactors;
};

std::optional<llvm::SmallVector<StaticSpatialAxis, 4>>
getNodeSpatialAxes(const CardDAGNode &node) {
  if (!node.operation || node.operation->getNumResults() != 1)
    return std::nullopt;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(
      node.operation->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() <= 0)
    return std::nullopt;
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(node.operation);
  if (!tiling)
    return std::nullopt;
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      tiling.getLoopIteratorTypes();
  if (iteratorTypes.empty())
    return std::nullopt;

  llvm::SmallVector<StaticSpatialAxis, 4> axes;
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(node.operation)) {
    mlir::AffineMap resultMap =
        linalg.getIndexingMapMatchingResult(node.operation->getResult(0));
    if (!resultMap || resultMap.getNumDims() != iteratorTypes.size() ||
        resultMap.getNumResults() != static_cast<unsigned>(type.getRank()))
      return std::nullopt;
    for (auto [resultDimension, expression] :
         llvm::enumerate(resultMap.getResults())) {
      auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!iterator || iterator.getPosition() >= iteratorTypes.size() ||
          iteratorTypes[iterator.getPosition()] !=
              mlir::utils::IteratorType::parallel)
        continue;
      const int64_t extent = type.getShape()[resultDimension];
      if (extent <= 0)
        continue;
      StaticSpatialAxis axis;
      axis.iteratorDimension = iterator.getPosition();
      axis.resultDimension = static_cast<unsigned>(resultDimension);
      axis.extent = static_cast<uint64_t>(extent);
      axis.basePartitionFactors.assign(iteratorTypes.size(), 1);
      axes.push_back(std::move(axis));
    }
  } else if (static_cast<size_t>(type.getRank()) == iteratorTypes.size()) {
    for (auto [dimension, extent] : llvm::enumerate(type.getShape())) {
      if (extent <= 0 ||
          iteratorTypes[dimension] != mlir::utils::IteratorType::parallel)
        continue;
      StaticSpatialAxis axis;
      axis.iteratorDimension = static_cast<unsigned>(dimension);
      axis.resultDimension = static_cast<unsigned>(dimension);
      axis.extent = static_cast<uint64_t>(extent);
      axis.basePartitionFactors.assign(iteratorTypes.size(), 1);
      axes.push_back(std::move(axis));
    }
  }
  llvm::sort(
      axes, [](const StaticSpatialAxis &lhs, const StaticSpatialAxis &rhs) {
        return std::tuple(std::numeric_limits<uint64_t>::max() - lhs.extent,
                          lhs.iteratorDimension, lhs.resultDimension) <
               std::tuple(std::numeric_limits<uint64_t>::max() - rhs.extent,
                          rhs.iteratorDimension, rhs.resultDimension);
      });
  if (axes.empty())
    return std::nullopt;
  return axes;
}

bool samePlacement(const WholeDAGNodePlacement &lhs,
                   const WholeDAGNodePlacement &rhs) {
  return lhs.spatialIteratorDimension == rhs.spatialIteratorDimension &&
         lhs.iteratorPartitionFactors == rhs.iteratorPartitionFactors &&
         lhs.shardDimension == rhs.shardDimension && lhs.tiles == rhs.tiles;
}

bool placementLess(const WholeDAGNodePlacement &lhs,
                   const WholeDAGNodePlacement &rhs) {
  if (lhs.spatialIteratorDimension != rhs.spatialIteratorDimension)
    return lhs.spatialIteratorDimension < rhs.spatialIteratorDimension;
  if (lhs.iteratorPartitionFactors != rhs.iteratorPartitionFactors)
    return std::lexicographical_compare(lhs.iteratorPartitionFactors.begin(),
                                        lhs.iteratorPartitionFactors.end(),
                                        rhs.iteratorPartitionFactors.begin(),
                                        rhs.iteratorPartitionFactors.end());
  if (lhs.shardDimension != rhs.shardDimension)
    return lhs.shardDimension < rhs.shardDimension;
  return tileVectorLess(lhs.tiles, rhs.tiles);
}

static size_t getPlacementHash(const WholeDAGNodePlacement &placement) {
  llvm::hash_code hash = llvm::hash_combine(
      placement.spatialIteratorDimension, placement.shardDimension,
      placement.iteratorPartitionFactors.size(), placement.tiles.size());
  for (uint32_t factor : placement.iteratorPartitionFactors)
    hash = llvm::hash_combine(hash, factor);
  for (PhysicalTileId tile : placement.tiles)
    hash = llvm::hash_combine(hash, tile.getValue());
  return static_cast<size_t>(hash);
}

/// Exact query-local memoization of one closed DAG edge's logical demand.
/// Demand legality observes only each side's shard dimension and participant
/// count. Tile identity, layout, bytes, residency, and transport are downstream
/// decisions and therefore cannot affect this cache key.
class EdgeTransitionLegalityCache {
public:
  explicit EdgeTransitionLegalityCache(const CardDAGAnalysis &dag)
      : planner(dag) {}

  bool lookupOrCompute(const CardDAGAnalysis &dag, CardDAGEdgeID edgeID,
                       const WholeDAGNodePlacement &producer,
                       const WholeDAGNodePlacement &consumer) {
    (void)dag;
    TransitionRelation relation = getRelation(producer, consumer);
    llvm::hash_code relationHash = llvm::hash_combine(
        edgeID, relation.producerShardDimension,
        relation.consumerShardDimension, relation.producerParticipants,
        relation.consumerParticipants);
    const size_t hash = static_cast<size_t>(relationHash);
    llvm::SmallVector<Entry, 1> &bucket = buckets[hash];
    auto found = llvm::find_if(bucket, [&](const Entry &entry) {
      return entry.edgeID == edgeID && entry.relation == relation;
    });
    if (found != bucket.end())
      return found->legal;
    std::string ignoredFailure;
    const bool legal = mlir::succeeded(
        planner.derive(edgeID, producer, consumer, &ignoredFailure));
    bucket.push_back(Entry{edgeID, std::move(relation), legal});
    return legal;
  }

private:
  struct TransitionRelation {
    unsigned producerShardDimension = 0;
    unsigned consumerShardDimension = 0;
    uint32_t producerParticipants = 0;
    uint32_t consumerParticipants = 0;
    bool operator==(const TransitionRelation &other) const {
      return producerShardDimension == other.producerShardDimension &&
             consumerShardDimension == other.consumerShardDimension &&
             producerParticipants == other.producerParticipants &&
             consumerParticipants == other.consumerParticipants;
    }
  };

  static TransitionRelation getRelation(const WholeDAGNodePlacement &producer,
                                        const WholeDAGNodePlacement &consumer) {
    TransitionRelation relation;
    relation.producerShardDimension = producer.shardDimension;
    relation.consumerShardDimension = consumer.shardDimension;
    relation.producerParticipants =
        static_cast<uint32_t>(producer.tiles.size());
    relation.consumerParticipants =
        static_cast<uint32_t>(consumer.tiles.size());
    return relation;
  }

  struct Entry {
    CardDAGEdgeID edgeID;
    TransitionRelation relation;
    bool legal;
  };
  std::unordered_map<size_t, llvm::SmallVector<Entry, 1>> buckets;
  WholeDAGEdgeDemandPlanner planner;
};

uint64_t getNodeElementWork(const CardDAGNode &node) {
  auto type = node.operation && node.operation->getNumResults() == 1
                  ? mlir::dyn_cast<mlir::RankedTensorType>(
                        node.operation->getResult(0).getType())
                  : mlir::RankedTensorType{};
  if (!type || !type.hasStaticShape())
    return std::numeric_limits<uint64_t>::max();
  uint64_t result = 1;
  for (int64_t extent : type.getShape()) {
    if (extent <= 0)
      return std::numeric_limits<uint64_t>::max();
    result = saturatingMultiply(result, static_cast<uint64_t>(extent));
  }
  return result;
}

struct PlacementPathNode {
  std::shared_ptr<const PlacementPathNode> parent;
  WholeDAGNodePlacement placement;
  size_t size = 0;
};

struct PartialPlacementState {
  std::shared_ptr<const PlacementPathNode> path;
  size_t identityHash = 0;
  uint64_t computeWorkLowerBound = 0;
  uint64_t transitionPenalty = 0;
  uint64_t topologyCompactnessWork = 0;
  uint32_t localEdges = 0;
  uint32_t sameGroupRemapEdges = 0;
  uint32_t partialEdges = 0;
  uint32_t disjointEdges = 0;
  uint32_t distinctGroups = 0;
  uint32_t utilizedTiles = 0;
  uint32_t componentOverlapPairs = 0;
  uint32_t parallelComponents = 1;
};

static size_t getPlacementCount(const PartialPlacementState &state) {
  return state.path ? state.path->size : 0;
}

static const WholeDAGNodePlacement *
findPlacement(const PartialPlacementState &state, CardDAGNodeID node) {
  for (const PlacementPathNode *current = state.path.get(); current;
       current = current->parent.get())
    if (current->placement.node == node)
      return &current->placement;
  return nullptr;
}

static llvm::SmallVector<WholeDAGNodePlacement, 16>
materializePlacements(const PartialPlacementState &state) {
  llvm::SmallVector<WholeDAGNodePlacement, 16> result(getPlacementCount(state));
  for (const PlacementPathNode *current = state.path.get(); current;
       current = current->parent.get())
    result[current->placement.node] = current->placement;
  return result;
}

enum class PlacementRelationClass : uint8_t {
  ExactLocal,
  SameGroupRemap,
  GenuinePartialOverlap,
  Disjoint,
};

PlacementRelationClass
classifyPlacementRelation(const WholeDAGNodePlacement &producer,
                          const WholeDAGNodePlacement &consumer) {
  if (samePlacement(producer, consumer))
    return PlacementRelationClass::ExactLocal;
  if (producer.tiles == consumer.tiles)
    return PlacementRelationClass::SameGroupRemap;
  if (getOverlap(producer.tiles, consumer.tiles) != 0)
    return PlacementRelationClass::GenuinePartialOverlap;
  return PlacementRelationClass::Disjoint;
}

uint32_t countDistinctGroups(llvm::ArrayRef<WholeDAGNodePlacement> placements) {
  uint32_t result = 0;
  for (size_t index = 0; index < placements.size(); ++index)
    if (llvm::none_of(llvm::ArrayRef(placements).take_front(index),
                      [&](const WholeDAGNodePlacement &previous) {
                        return previous.tiles == placements[index].tiles;
                      }) &&
        result != std::numeric_limits<uint32_t>::max())
      ++result;
  return result;
}

uint32_t countUtilizedTiles(llvm::ArrayRef<WholeDAGNodePlacement> placements) {
  llvm::SmallVector<int64_t, 16> tiles;
  for (const WholeDAGNodePlacement &placement : placements)
    for (PhysicalTileId tile : placement.tiles)
      tiles.push_back(tile.getValue());
  llvm::sort(tiles);
  tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
  return static_cast<uint32_t>(
      std::min<size_t>(tiles.size(), std::numeric_limits<uint32_t>::max()));
}

struct ComponentConcurrency {
  uint32_t overlapPairs = 0;
  uint32_t parallelComponents = 1;
};

ComponentConcurrency
getComponentConcurrency(const CardDAGAnalysis &dag,
                        llvm::ArrayRef<WholeDAGNodePlacement> placements) {
  llvm::ArrayRef<CardDAGDependencyComponent> components =
      dag.getObservableDependencyComponents();
  if (!dag.supportsIndependentComponentPlacement() || components.size() < 2)
    return {};

  llvm::SmallVector<llvm::SmallVector<PhysicalTileId, 16>, 4> componentTiles(
      components.size());
  for (const WholeDAGNodePlacement &placement : placements) {
    auto component = llvm::find_if(
        components, [&](const CardDAGDependencyComponent &candidate) {
          return llvm::is_contained(candidate.nodes, placement.node);
        });
    if (component == components.end())
      continue;
    const size_t index = std::distance(components.begin(), component);
    componentTiles[index].append(placement.tiles.begin(),
                                 placement.tiles.end());
  }
  uint32_t activeComponents = 0;
  for (auto &tiles : componentTiles) {
    llvm::sort(tiles, [](PhysicalTileId lhs, PhysicalTileId rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
    if (!tiles.empty() &&
        activeComponents != std::numeric_limits<uint32_t>::max())
      ++activeComponents;
  }

  ComponentConcurrency result;
  for (size_t left = 0; left < componentTiles.size(); ++left) {
    if (componentTiles[left].empty())
      continue;
    for (size_t right = left + 1; right < componentTiles.size(); ++right) {
      if (componentTiles[right].empty())
        continue;
      if (llvm::any_of(componentTiles[left],
                       [&](PhysicalTileId tile) {
                         return llvm::is_contained(componentTiles[right], tile);
                       }) &&
          result.overlapPairs != std::numeric_limits<uint32_t>::max())
        ++result.overlapPairs;
    }
  }
  result.parallelComponents =
      result.overlapPairs == 0 ? std::max<uint32_t>(1, activeComponents) : 1;
  return result;
}

uint64_t getGroupTransitionPenalty(const PhysicalTopology &topology,
                                   PhysicalCardId cardId,
                                   const WholeDAGNodePlacement &producer,
                                   const WholeDAGNodePlacement &consumer) {
  if (samePlacement(producer, consumer))
    return 0;
  const uint64_t producerCount = producer.tiles.size();
  const uint64_t consumerCount = consumer.tiles.size();
  const uint64_t overlap = getOverlap(producer.tiles, consumer.tiles);
  uint64_t penalty = saturatingAdd(producerCount, consumerCount);
  const uint64_t sharedEndpoints = saturatingMultiply(2, overlap);
  penalty = penalty >= sharedEndpoints ? penalty - sharedEndpoints : 0;
  // A shard-axis redistribution on the same physical group is not local and
  // must not look free to the pre-materialization candidate enumeration.
  if (producer.tiles == consumer.tiles)
    penalty = saturatingAdd(penalty, std::max<uint64_t>(1, consumerCount));
  for (PhysicalTileId destination : consumer.tiles) {
    uint64_t nearest = std::numeric_limits<uint64_t>::max();
    for (PhysicalTileId source : producer.tiles) {
      std::optional<uint64_t> hops =
          topology.getOnCardShortestHopDistance(cardId, source, destination);
      if (hops)
        nearest = std::min(nearest, *hops);
    }
    if (nearest == std::numeric_limits<uint64_t>::max())
      return nearest;
    penalty = saturatingAdd(penalty, nearest);
  }
  return penalty;
}

llvm::SmallVector<WholeDAGNodePlacement, 32>
deriveNodePlacementDomain(const CardDAGNode &node,
                          llvm::ArrayRef<TopologyTileGroup> groups) {
  struct PlacementOption {
    WholeDAGNodePlacement placement;
    uint64_t internalHopWork = 0;
  };
  std::optional<llvm::SmallVector<StaticSpatialAxis, 4>> axes =
      getNodeSpatialAxes(node);
  if (!axes)
    return {};
  llvm::SmallVector<PlacementOption, 128> options;
  for (const StaticSpatialAxis &axis : *axes) {
    for (const TopologyTileGroup &group : groups) {
      if (group.tiles.empty() || group.tiles.size() > axis.extent)
        continue;
      PlacementOption option;
      option.placement.node = node.id;
      option.placement.spatialIteratorDimension = axis.iteratorDimension;
      option.placement.iteratorPartitionFactors = axis.basePartitionFactors;
      option.placement.iteratorPartitionFactors[axis.iteratorDimension] =
          static_cast<uint32_t>(group.tiles.size());
      option.placement.shardDimension = axis.resultDimension;
      option.placement.tiles = group.tiles;
      option.internalHopWork = group.internalHopWork;
      options.push_back(std::move(option));
    }
  }
  llvm::sort(options,
             [](const PlacementOption &lhs, const PlacementOption &rhs) {
               if (lhs.placement.tiles.size() != rhs.placement.tiles.size())
                 return lhs.placement.tiles.size() > rhs.placement.tiles.size();
               if (lhs.internalHopWork != rhs.internalHopWork)
                 return lhs.internalHopWork < rhs.internalHopWork;
               return placementLess(lhs.placement, rhs.placement);
             });
  llvm::SmallVector<WholeDAGNodePlacement, 32> result;
  result.reserve(options.size());
  for (PlacementOption &option : options)
    if (result.empty() || !samePlacement(result.back(), option.placement))
      result.push_back(std::move(option.placement));
  return result;
}

llvm::SmallVector<WholeDAGNodePlacement, 32>
deriveNodeOptions(const CardDAGAnalysis &dag, const CardDAGNode &node,
                  const PartialPlacementState &state,
                  llvm::ArrayRef<TopologyTileGroup> groups,
                  EdgeTransitionLegalityCache &transitionCache,
                  WholeDAGPlacementEnumerationStatistics &statistics) {
  struct PlacementOption {
    WholeDAGNodePlacement placement;
    uint64_t internalHopWork = 0;
    uint32_t unusedTiles = 0;
  };

  llvm::SmallVector<PlacementOption, 128> options;
  std::optional<llvm::SmallVector<StaticSpatialAxis, 4>> axes =
      getNodeSpatialAxes(node);
  if (!axes)
    return {};
  llvm::DenseSet<int64_t> utilized;
  for (const PlacementPathNode *current = state.path.get(); current;
       current = current->parent.get())
    for (PhysicalTileId tile : current->placement.tiles)
      utilized.insert(tile.getValue());
  for (const StaticSpatialAxis &axis : *axes) {
    for (const TopologyTileGroup &group : groups) {
      if (group.tiles.empty() || group.tiles.size() > axis.extent)
        continue;
      PlacementOption option;
      option.placement.node = node.id;
      option.placement.spatialIteratorDimension = axis.iteratorDimension;
      option.placement.iteratorPartitionFactors = axis.basePartitionFactors;
      option.placement.iteratorPartitionFactors[axis.iteratorDimension] =
          static_cast<uint32_t>(group.tiles.size());
      option.placement.shardDimension = axis.resultDimension;
      option.placement.tiles = group.tiles;
      option.internalHopWork = group.internalHopWork;
      option.unusedTiles = static_cast<uint32_t>(
          llvm::count_if(group.tiles, [&](PhysicalTileId tile) {
            return !utilized.contains(tile.getValue());
          }));
      options.push_back(std::move(option));
    }
  }
  auto optionLess = [](const PlacementOption &lhs, const PlacementOption &rhs) {
    if (lhs.placement.tiles.size() != rhs.placement.tiles.size())
      return lhs.placement.tiles.size() > rhs.placement.tiles.size();
    if (lhs.unusedTiles != rhs.unusedTiles)
      return lhs.unusedTiles > rhs.unusedTiles;
    if (lhs.internalHopWork != rhs.internalHopWork)
      return lhs.internalHopWork < rhs.internalHopWork;
    return placementLess(lhs.placement, rhs.placement);
  };
  llvm::sort(options, optionLess);

  llvm::SmallVector<int8_t, 128> legality(options.size(), -1);
  auto isLegal = [&](size_t optionIndex) {
    if (legality[optionIndex] >= 0)
      return legality[optionIndex] != 0;
    const WholeDAGNodePlacement &candidate = options[optionIndex].placement;
    bool legal = true;
    for (CardDAGEdgeID edgeID : node.incomingEdges) {
      const CardDAGEdge *edge = dag.getEdge(edgeID);
      const WholeDAGNodePlacement *producer =
          edge ? findPlacement(state, edge->producer) : nullptr;
      if (!edge || !producer) {
        legal = false;
        break;
      }
      if (!transitionCache.lookupOrCompute(dag, edgeID, *producer, candidate)) {
        legal = false;
        break;
      }
    }
    legality[optionIndex] = legal ? 1 : 0;
    if (!legal)
      statistics.rejectedTransitions =
          saturatingAdd(statistics.rejectedTransitions, 1);
    return legal;
  };

  llvm::SmallVector<WholeDAGNodePlacement, 32> legal;
  legal.reserve(options.size());
  for (size_t index = 0; index < options.size(); ++index)
    if (isLegal(index))
      legal.push_back(options[index].placement);
  // An isolated scalar component has exactly one participant and no physical
  // relation or movement.  While an unused Tile exists, assigning it to an
  // already occupied Tile can only serialize otherwise independent work.
  // All unused identities have the same empty relation/resource signature, so
  // retain the canonical first one as an exact query-local equivalence class.
  if (dag.supportsIndependentComponentPlacement() &&
      node.incomingEdges.empty() && node.outgoingEdges.empty() &&
      llvm::all_of(*axes, [](const StaticSpatialAxis &axis) {
        return axis.extent == 1;
      })) {
    auto unused =
        llvm::find_if(legal, [&](const WholeDAGNodePlacement &option) {
          return llvm::none_of(option.tiles, [&](PhysicalTileId tile) {
            return utilized.contains(tile.getValue());
          });
        });
    if (unused != legal.end())
      return {*unused};
  }
  // `options` is already in deterministic cost/identity order.  Returning all
  // legal entries is intentional: relation class, iterator axis, participant
  // count and physical translation are semantic search coordinates, not
  // diversity samples.
  return legal;
}

PartialPlacementState extendState(const CardDAGAnalysis &dag,
                                  const PhysicalTopology &topology,
                                  PhysicalCardId cardId,
                                  const PartialPlacementState &parent,
                                  WholeDAGNodePlacement placement) {
  PartialPlacementState result = parent;
  const CardDAGNode &node = dag.getNodes()[placement.node];
  const uint64_t work = getNodeElementWork(node);
  result.computeWorkLowerBound = saturatingAdd(
      result.computeWorkLowerBound,
      placement.tiles.empty() ? std::numeric_limits<uint64_t>::max()
                              : work / placement.tiles.size() +
                                    (work % placement.tiles.size() != 0));
  result.topologyCompactnessWork =
      saturatingAdd(result.topologyCompactnessWork,
                    getInternalHopWork(topology, cardId, placement.tiles));
  for (CardDAGEdgeID edgeID : node.incomingEdges) {
    const CardDAGEdge *edge = dag.getEdge(edgeID);
    const WholeDAGNodePlacement *producer =
        edge ? findPlacement(parent, edge->producer) : nullptr;
    if (!edge || !producer)
      continue;
    switch (classifyPlacementRelation(*producer, placement)) {
    case PlacementRelationClass::ExactLocal:
      if (result.localEdges != std::numeric_limits<uint32_t>::max())
        ++result.localEdges;
      break;
    case PlacementRelationClass::SameGroupRemap:
      if (result.sameGroupRemapEdges != std::numeric_limits<uint32_t>::max())
        ++result.sameGroupRemapEdges;
      break;
    case PlacementRelationClass::GenuinePartialOverlap:
      if (result.partialEdges != std::numeric_limits<uint32_t>::max())
        ++result.partialEdges;
      break;
    case PlacementRelationClass::Disjoint:
      if (result.disjointEdges != std::numeric_limits<uint32_t>::max())
        ++result.disjointEdges;
      break;
    }
    result.transitionPenalty = saturatingAdd(
        result.transitionPenalty,
        getGroupTransitionPenalty(topology, cardId, *producer, placement));
  }
  auto path = std::make_shared<PlacementPathNode>();
  path->parent = parent.path;
  path->placement = std::move(placement);
  path->size = getPlacementCount(parent) + 1;
  result.path = std::move(path);
  result.identityHash = static_cast<size_t>(llvm::hash_combine(
      parent.identityHash, getPlacementHash(result.path->placement)));
  llvm::SmallVector<WholeDAGNodePlacement, 16> placements =
      materializePlacements(result);
  result.distinctGroups = countDistinctGroups(placements);
  result.utilizedTiles = countUtilizedTiles(placements);
  const ComponentConcurrency concurrency =
      getComponentConcurrency(dag, placements);
  result.componentOverlapPairs = concurrency.overlapPairs;
  result.parallelComponents = concurrency.parallelComponents;
  return result;
}

bool samePartialPlacement(const PartialPlacementState &lhs,
                          const PartialPlacementState &rhs) {
  if (getPlacementCount(lhs) != getPlacementCount(rhs))
    return false;
  const PlacementPathNode *left = lhs.path.get();
  const PlacementPathNode *right = rhs.path.get();
  while (left && right) {
    if (left->placement.node != right->placement.node ||
        !samePlacement(left->placement, right->placement))
      return false;
    left = left->parent.get();
    right = right->parent.get();
  }
  return left == nullptr && right == nullptr;
}

bool partialStateLess(const PartialPlacementState &lhs,
                      const PartialPlacementState &rhs) {
  const auto lhsScore =
      std::tuple(lhs.computeWorkLowerBound, lhs.transitionPenalty,
                 lhs.topologyCompactnessWork,
                 std::numeric_limits<uint32_t>::max() - lhs.utilizedTiles,
                 std::numeric_limits<uint32_t>::max() - lhs.distinctGroups,
                 lhs.componentOverlapPairs,
                 std::numeric_limits<uint32_t>::max() - lhs.parallelComponents,
                 lhs.disjointEdges, lhs.partialEdges, lhs.sameGroupRemapEdges);
  const auto rhsScore =
      std::tuple(rhs.computeWorkLowerBound, rhs.transitionPenalty,
                 rhs.topologyCompactnessWork,
                 std::numeric_limits<uint32_t>::max() - rhs.utilizedTiles,
                 std::numeric_limits<uint32_t>::max() - rhs.distinctGroups,
                 rhs.componentOverlapPairs,
                 std::numeric_limits<uint32_t>::max() - rhs.parallelComponents,
                 rhs.disjointEdges, rhs.partialEdges, rhs.sameGroupRemapEdges);
  if (lhsScore != rhsScore)
    return lhsScore < rhsScore;
  if (lhs.identityHash != rhs.identityHash)
    return lhs.identityHash < rhs.identityHash;
  llvm::SmallVector<WholeDAGNodePlacement, 16> lhsPlacements =
      materializePlacements(lhs);
  llvm::SmallVector<WholeDAGNodePlacement, 16> rhsPlacements =
      materializePlacements(rhs);
  for (auto [left, right] : llvm::zip_equal(lhsPlacements, rhsPlacements)) {
    if (samePlacement(left, right))
      continue;
    return placementLess(left, right);
  }
  return false;
}

static bool isObservableRoot(const CardDAGAnalysis &dag, CardDAGNodeID node) {
  return llvm::any_of(dag.getObservableOutputRootNodes(),
                      [&](llvm::ArrayRef<CardDAGNodeID> roots) {
                        return llvm::is_contained(roots, node);
                      });
}

static bool remainsOnLiveBoundary(const CardDAGAnalysis &dag,
                                  CardDAGNodeID node, size_t placedNodes) {
  if (isObservableRoot(dag, node))
    return true;
  const CardDAGNode *current = dag.getNode(node);
  return current &&
         llvm::any_of(current->outgoingEdges, [&](CardDAGEdgeID edgeID) {
           const CardDAGEdge *edge = dag.getEdge(edgeID);
           return edge && edge->consumer >= placedNodes;
         });
}

static bool sameLiveBoundary(const CardDAGAnalysis &dag,
                             const PartialPlacementState &lhs,
                             const PartialPlacementState &rhs) {
  if (getPlacementCount(lhs) != getPlacementCount(rhs))
    return false;
  const size_t placedNodes = getPlacementCount(lhs);
  for (CardDAGNodeID node = 0; node < placedNodes; ++node) {
    if (!remainsOnLiveBoundary(dag, node, placedNodes))
      continue;
    const WholeDAGNodePlacement *left = findPlacement(lhs, node);
    const WholeDAGNodePlacement *right = findPlacement(rhs, node);
    if (!left || !right || !samePlacement(*left, *right))
      return false;
  }
  return true;
}

static size_t getLiveBoundaryHash(const CardDAGAnalysis &dag,
                                  const PartialPlacementState &state) {
  const size_t placedNodes = getPlacementCount(state);
  llvm::hash_code hash = llvm::hash_combine(placedNodes);
  for (CardDAGNodeID node = 0; node < placedNodes; ++node) {
    if (!remainsOnLiveBoundary(dag, node, placedNodes))
      continue;
    const WholeDAGNodePlacement *placement = findPlacement(state, node);
    if (placement)
      hash = llvm::hash_combine(hash, node, getPlacementHash(*placement));
  }
  return static_cast<size_t>(hash);
}

/// A closed pure prefix can affect a future transition only through its live
/// SSA boundary and through the accumulated planner/resource terms below.
/// Requiring an identical live boundary and component-wise improvement keeps
/// this a strict dominance proof; unlike the former beam it has no width and
/// never removes incomparable states to obtain a representative sample.
static bool strictlyDominatesPartialState(const CardDAGAnalysis &dag,
                                          const PartialPlacementState &lhs,
                                          const PartialPlacementState &rhs) {
  if (!sameLiveBoundary(dag, lhs, rhs))
    return false;
  const bool noWorse =
      lhs.computeWorkLowerBound <= rhs.computeWorkLowerBound &&
      lhs.transitionPenalty <= rhs.transitionPenalty &&
      lhs.topologyCompactnessWork <= rhs.topologyCompactnessWork &&
      lhs.componentOverlapPairs <= rhs.componentOverlapPairs &&
      lhs.sameGroupRemapEdges <= rhs.sameGroupRemapEdges &&
      lhs.partialEdges <= rhs.partialEdges &&
      lhs.disjointEdges <= rhs.disjointEdges &&
      lhs.localEdges >= rhs.localEdges &&
      lhs.parallelComponents >= rhs.parallelComponents;
  if (!noWorse)
    return false;
  return lhs.computeWorkLowerBound < rhs.computeWorkLowerBound ||
         lhs.transitionPenalty < rhs.transitionPenalty ||
         lhs.topologyCompactnessWork < rhs.topologyCompactnessWork ||
         lhs.componentOverlapPairs < rhs.componentOverlapPairs ||
         lhs.sameGroupRemapEdges < rhs.sameGroupRemapEdges ||
         lhs.partialEdges < rhs.partialEdges ||
         lhs.disjointEdges < rhs.disjointEdges ||
         lhs.localEdges > rhs.localEdges ||
         lhs.parallelComponents > rhs.parallelComponents;
}

llvm::SmallVector<PartialPlacementState, 24> retainNondominatedPlacementStates(
    const CardDAGAnalysis &dag,
    llvm::SmallVector<PartialPlacementState, 128> candidates) {
  llvm::sort(candidates, partialStateLess);
  candidates.erase(
      std::unique(candidates.begin(), candidates.end(), samePartialPlacement),
      candidates.end());
  std::unordered_map<size_t, llvm::SmallVector<PartialPlacementState, 4>>
      boundaryBuckets;
  for (PartialPlacementState &candidate : candidates) {
    llvm::SmallVector<PartialPlacementState, 4> &bucket =
        boundaryBuckets[getLiveBoundaryHash(dag, candidate)];
    if (llvm::any_of(bucket, [&](const PartialPlacementState &existing) {
          return sameLiveBoundary(dag, existing, candidate) &&
                 strictlyDominatesPartialState(dag, existing, candidate);
        }))
      continue;
    llvm::erase_if(bucket, [&](const PartialPlacementState &existing) {
      return sameLiveBoundary(dag, candidate, existing) &&
             strictlyDominatesPartialState(dag, candidate, existing);
    });
    bucket.push_back(std::move(candidate));
  }
  llvm::SmallVector<PartialPlacementState, 24> result;
  for (auto &[hash, bucket] : boundaryBuckets) {
    (void)hash;
    result.append(std::make_move_iterator(bucket.begin()),
                  std::make_move_iterator(bucket.end()));
  }
  llvm::sort(result, partialStateLess);
  return result;
}

/// Maps a shard axis of a support-op operand through the op to its result.
/// The sharded axis keeps a contiguous output-domain slice only through
/// axis-preserving views (insert/extract slices, shape-preserving casts) and
/// expand/collapse groups whose leading members all have extent 1, so the
/// shard is a clean slice of exactly one group axis.  A shard that would
/// straddle several group axes (a strided or concatenated region) is not a
/// single output-domain shard dimension and fails closed.
static mlir::FailureOr<unsigned>
mapAxisForwardThroughSupportOp(mlir::Operation *op, unsigned operandAxis,
                               uint64_t shardCount) {
  if (mlir::isa<mlir::tensor::ExtractSliceOp, mlir::tensor::InsertSliceOp>(op))
    return operandAxis;
  if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op)) {
    // Operand axis a expands to result axes reassociation[a] with extents
    // [e1..ek].  Shard k of the operand axis is the row-major range
    // [k*w, (k+1)*w).  That range stays a single slice of group member j
    // when every member before j has extent 1 (otherwise the range repeats
    // as several disjoint slices) and the extent product of the members
    // after j divides w (so the range starts on a whole trailing page).
    // Prefer the leftmost such member.
    // getReassociationIndices() hands out an ArrayRef into a per-call buffer;
    // copy the groups into locals before anything else can re-enter it.
    llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation =
        expand.getReassociationIndices();
    auto operandType =
        mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (operandAxis >= reassociation.size() ||
        operandAxis >= static_cast<unsigned>(operandType.getRank()))
      return mlir::failure();
    llvm::ArrayRef<int64_t> group = reassociation[operandAxis];
    const int64_t extent = operandType.getShape()[operandAxis];
    if (extent <= 0 || shardCount == 0 ||
        extent % static_cast<int64_t>(shardCount) != 0)
      return mlir::failure();
    const int64_t shardSize = extent / static_cast<int64_t>(shardCount);
    llvm::SmallVector<int64_t, 4> memberExtents;
    for (int64_t member : group) {
      if (member < 0 || member >= resultType.getRank())
        return mlir::failure();
      const int64_t memberExtent = resultType.getShape()[member];
      if (memberExtent <= 0)
        return mlir::failure();
      memberExtents.push_back(memberExtent);
    }
    uint64_t leadingExtentProduct = 1;
    uint64_t trailingExtentProduct = 1;
    for (int64_t memberExtent : llvm::drop_begin(memberExtents))
      trailingExtentProduct *= memberExtent;
    for (auto [index, pair] :
         llvm::enumerate(llvm::zip(group, memberExtents))) {
      auto [member, memberExtent] = pair;
      if (leadingExtentProduct == 1 &&
          shardSize % static_cast<int64_t>(trailingExtentProduct) == 0)
        return mlir::FailureOr<unsigned>(static_cast<unsigned>(member));
      leadingExtentProduct *= memberExtent;
      if (index + 1 < memberExtents.size())
        trailingExtentProduct /= memberExtents[index + 1];
    }
    return mlir::failure();
  }
  if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op)) {
    // Operand axes reassociation[resultAxis] collapse into result axis
    // resultAxis.  The sharded operand axis maps to a single contiguous
    // collapsed range only when every leading member of its group has extent
    // 1; otherwise the shard repeats as several disjoint collapsed ranges.
    auto operandType =
        mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (operandAxis >= static_cast<unsigned>(operandType.getRank()))
      return mlir::failure();
    const int64_t extent = operandType.getShape()[operandAxis];
    if (extent <= 0 || shardCount == 0 ||
        extent % static_cast<int64_t>(shardCount) != 0)
      return mlir::failure();
    llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation =
        collapse.getReassociationIndices();
    for (auto [resultAxis, group] : llvm::enumerate(reassociation)) {
      if (!llvm::is_contained(group, static_cast<int64_t>(operandAxis)))
        continue;
      for (int64_t member : group) {
        if (member == static_cast<int64_t>(operandAxis))
          break;
        if (operandType.getShape()[member] != 1)
          return mlir::failure();
      }
      return mlir::FailureOr<unsigned>(static_cast<unsigned>(resultAxis));
    }
    return mlir::failure();
  }
  // Any other pure support op keeps the shard axis only when operand and
  // result are identical ranked shapes (e.g. tensor.cast).
  auto operandType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (operandType && resultType && operandType.hasStaticShape() &&
      resultType.hasStaticShape() &&
      operandType.getRank() == resultType.getRank() &&
      operandType.getShape() == resultType.getShape())
    return operandAxis;
  return mlir::failure();
}

/// Collects the pure support chain from a structured root result to an
/// observable function result, recording ops in root-to-output order.  The
/// walk follows def-use only (acyclic by construction) and backtracks over the
/// operand paths of a pure support op until the path reaches the root result.
static mlir::LogicalResult
collectOutputSupportChain(mlir::Value value, mlir::Operation *rootOp,
                          llvm::SmallVectorImpl<mlir::Operation *> &chain) {
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return mlir::failure();
  mlir::Operation *owner = result.getOwner();
  if (owner == rootOp)
    // The placed shard axis is defined against result 0 of the root.
    return result.getResultNumber() == 0 ? mlir::success() : mlir::failure();
  if (!owner || owner->getNumResults() != 1 || !mlir::isMemoryEffectFree(owner))
    return mlir::failure();
  for (mlir::Value operand : owner->getOperands()) {
    if (mlir::succeeded(collectOutputSupportChain(operand, rootOp, chain))) {
      chain.push_back(owner);
      return mlir::success();
    }
  }
  return mlir::failure();
}

/// Maps the placed shard axis of an observable output root through the pure
/// support chain to the axis of the function result it produces.
static mlir::FailureOr<unsigned>
mapRootShardAxisToOutput(const CardDAGAnalysis &dag, unsigned outputIndex,
                         const CardDAGNode &root, unsigned rootAxis,
                         uint64_t shardCount) {
  mlir::func::ReturnOp returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      dag.getFunction().getBody().front().getTerminator());
  if (!returnOp || outputIndex >= returnOp.getNumOperands())
    return mlir::failure();
  llvm::SmallVector<mlir::Operation *, 4> chain;
  if (mlir::failed(collectOutputSupportChain(returnOp.getOperand(outputIndex),
                                             root.operation, chain)))
    return mlir::failure();
  unsigned axis = rootAxis;
  for (mlir::Operation *op : chain) {
    mlir::FailureOr<unsigned> mapped =
        mapAxisForwardThroughSupportOp(op, axis, shardCount);
    if (mlir::failed(mapped))
      return mlir::failure();
    axis = *mapped;
  }
  return axis;
}

mlir::FailureOr<llvm::SmallVector<WholeDAGObservablePlacement, 4>>
deriveObservablePlacements(const CardDAGAnalysis &dag,
                           llvm::ArrayRef<WholeDAGNodePlacement> placements) {
  llvm::SmallVector<WholeDAGObservablePlacement, 4> outputs;
  outputs.reserve(dag.getFunction().getNumResults());
  mlir::func::ReturnOp returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      dag.getFunction().getBody().front().getTerminator());
  if (!returnOp ||
      returnOp.getNumOperands() != dag.getFunction().getNumResults())
    return mlir::failure();
  for (auto [outputIndex, roots] :
       llvm::enumerate(dag.getObservableOutputRootNodes())) {
    if (roots.empty() || roots.front() >= placements.size())
      return mlir::failure();
    const WholeDAGNodePlacement &owner = placements[roots.front()];
    if (llvm::any_of(roots, [&](CardDAGNodeID root) {
          return root >= placements.size() ||
                 !samePlacement(owner, placements[root]);
        }))
      return mlir::failure();
    // The placed shard axis names an axis of the root result; map it through
    // the pure support chain to the observable output domain.  Every root of
    // the result must map to the same output axis, and the domain must be able
    // to hold the active Tile count.  An inexpressible chain fails closed.
    std::optional<unsigned> outputAxis;
    for (CardDAGNodeID rootID : roots) {
      const CardDAGNode *rootNode = dag.getNode(rootID);
      if (!rootNode || !rootNode->operation)
        return mlir::failure();
      mlir::FailureOr<unsigned> mapped = mapRootShardAxisToOutput(
          dag, static_cast<unsigned>(outputIndex), *rootNode,
          owner.shardDimension, owner.tiles.size());
      if (mlir::failed(mapped))
        return mlir::failure();
      if (outputAxis && *outputAxis != *mapped)
        return mlir::failure();
      outputAxis = *mapped;
    }
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(
        returnOp.getOperand(outputIndex).getType());
    if (!outputType || !outputType.hasStaticShape() ||
        *outputAxis >= static_cast<unsigned>(outputType.getRank()) ||
        static_cast<uint64_t>(outputType.getShape()[*outputAxis]) <
            owner.tiles.size())
      return mlir::failure();
    outputs.push_back(WholeDAGObservablePlacement{
        static_cast<uint32_t>(outputIndex), *outputAxis, owner.tiles});
  }
  return outputs;
}

std::optional<CardDAGEdgeID>
resolveStrategyEdge(const CardDAGAnalysis &dag,
                    const SpatialEdgeStrategy &strategy) {
  for (const CardDAGEdge &edge : dag.getEdges()) {
    const CardDAGNode *producer = dag.getNode(edge.producer);
    const CardDAGNode *consumer = dag.getNode(edge.consumer);
    if (producer && consumer && producer->operation == strategy.producer &&
        consumer->operation == strategy.consumer &&
        edge.producerResult == strategy.producerResult &&
        edge.consumerOperand == strategy.consumerOperand)
      return edge.id;
  }
  return std::nullopt;
}

std::optional<uint64_t>
getStrategyElementBytes(const SpatialEdgeStrategy &strategy) {
  if (!strategy.producer ||
      strategy.producerResult >= strategy.producer->getNumResults())
    return std::nullopt;
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(
      strategy.producer->getResult(strategy.producerResult).getType());
  if (!shaped)
    return std::nullopt;
  unsigned bits = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(shaped.getElementType()))
    bits = integer.getWidth();
  else if (auto floating =
               mlir::dyn_cast<mlir::FloatType>(shaped.getElementType()))
    bits = floating.getWidth();
  if (bits == 0 || bits % 8 != 0)
    return std::nullopt;
  return bits / 8;
}

std::optional<uint64_t> getDomainBytes(llvm::ArrayRef<int64_t> sizes,
                                       uint64_t elementBytes) {
  uint64_t elements = 1;
  for (int64_t size : sizes) {
    if (size <= 0)
      return std::nullopt;
    elements = saturatingMultiply(elements, static_cast<uint64_t>(size));
  }
  return saturatingMultiply(elements, elementBytes);
}

std::optional<llvm::SmallVector<WholeDAGLocalResidency, 32>>
derivePlacementResidencies(const CardDAGAnalysis &dag,
                           const WholeDAGEdgeStrategyPlan &plan) {
  std::map<std::pair<CardDAGEdgeID, int64_t>, uint64_t> bytesByEdgeTile;
  for (const SpatialEdgeStrategy &strategy : plan.strategies) {
    std::optional<CardDAGEdgeID> edge = resolveStrategyEdge(dag, strategy);
    std::optional<uint64_t> elementBytes = getStrategyElementBytes(strategy);
    if (!edge || !elementBytes)
      return std::nullopt;
    if (strategy.action == SpatialEdgeAction::CoupledFusion ||
        strategy.action == SpatialEdgeAction::LocalShardResidency) {
      std::optional<uint64_t> bytes =
          getDomainBytes(strategy.producerSizes, *elementBytes);
      if (!bytes || *bytes == 0)
        return std::nullopt;
      bytesByEdgeTile[{*edge, strategy.destinationTile.getValue()}] =
          saturatingAdd(
              bytesByEdgeTile[{*edge, strategy.destinationTile.getValue()}],
              *bytes);
      continue;
    }
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      if (fragment.kind != SpatialEdgeFragmentKind::Resident)
        continue;
      std::optional<uint64_t> bytes =
          getDomainBytes(fragment.sizes, *elementBytes);
      if (!bytes || *bytes == 0)
        return std::nullopt;
      bytesByEdgeTile[{*edge, strategy.destinationTile.getValue()}] =
          saturatingAdd(
              bytesByEdgeTile[{*edge, strategy.destinationTile.getValue()}],
              *bytes);
    }
  }
  llvm::SmallVector<WholeDAGLocalResidency, 32> result;
  for (const auto &[key, bytes] : bytesByEdgeTile)
    result.push_back(
        WholeDAGLocalResidency{key.first, PhysicalTileId(key.second), bytes});
  return result;
}

std::optional<llvm::SmallVector<WholeDAGPeerMovement, 32>>
derivePeerMovements(const PhysicalTopology &topology, PhysicalCardId cardId,
                    const CardDAGAnalysis &dag,
                    const WholeDAGEdgeStrategyPlan &plan) {
  llvm::SmallVector<WholeDAGPeerMovement, 32> movements;
  for (const SpatialEdgeStrategy &strategy : plan.strategies) {
    std::optional<CardDAGEdgeID> edge = resolveStrategyEdge(dag, strategy);
    if (!edge)
      return std::nullopt;
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      if (fragment.kind != SpatialEdgeFragmentKind::Peer)
        continue;
      if (fragment.bytes == 0 ||
          fragment.sourceTile == strategy.destinationTile)
        return std::nullopt;
      mlir::FailureOr<llvm::SmallVector<PhysicalTileDirectedLink, 8>> route =
          topology.getCanonicalOnCardPath(cardId, fragment.sourceTile,
                                          strategy.destinationTile);
      if (mlir::failed(route) || route->empty())
        return std::nullopt;
      movements.push_back(WholeDAGPeerMovement{
          *edge, fragment.sourceTile, strategy.destinationTile, fragment.bytes,
          /*bufferCount=*/1, std::move(*route)});
    }
  }
  return movements;
}

uint64_t
getTopologyHopByteWork(llvm::ArrayRef<WholeDAGPeerMovement> movements) {
  uint64_t work = 0;
  for (const WholeDAGPeerMovement &movement : movements)
    work = saturatingAdd(
        work, saturatingMultiply(movement.duration, movement.route.size()));
  return work;
}

/// Canonical resource-graph identity for a complete placement.  Physical Tile
/// numbers are assigned query-local canonical labels in first semantic-use
/// order.  Edge domains, layout facts, fragments and every routed link remain
/// in the key, so two states compare equal only when the scheduler and later
/// physical materialization see the same structure modulo a consistent
/// renaming of otherwise interchangeable Tile identities.
static std::vector<uint64_t> getResourceRenamingSignature(
    llvm::ArrayRef<WholeDAGNodePlacement> placements,
    llvm::ArrayRef<WholeDAGObservablePlacement> outputs,
    const WholeDAGEdgeStrategyPlan &edgePlan,
    llvm::ArrayRef<WholeDAGPeerMovement> movements) {
  std::map<int64_t, uint64_t> labels;
  auto label = [&](PhysicalTileId tile) {
    auto [found, inserted] = labels.try_emplace(tile.getValue(), labels.size());
    (void)inserted;
    return found->second;
  };
  std::vector<uint64_t> signature;
  auto appendSigned = [&](int64_t value) {
    signature.push_back(static_cast<uint64_t>(value));
  };
  signature.push_back(placements.size());
  for (const WholeDAGNodePlacement &placement : placements) {
    signature.push_back(placement.node);
    signature.push_back(placement.spatialIteratorDimension);
    signature.push_back(placement.shardDimension);
    signature.push_back(placement.iteratorPartitionFactors.size());
    for (uint32_t factor : placement.iteratorPartitionFactors)
      signature.push_back(factor);
    signature.push_back(placement.tiles.size());
    for (PhysicalTileId tile : placement.tiles)
      signature.push_back(label(tile));
  }
  signature.push_back(outputs.size());
  for (const WholeDAGObservablePlacement &output : outputs) {
    signature.push_back(output.outputIndex);
    signature.push_back(output.shardDimension);
    signature.push_back(output.tiles.size());
    for (PhysicalTileId tile : output.tiles)
      signature.push_back(label(tile));
  }
  signature.push_back(edgePlan.strategies.size());
  for (const SpatialEdgeStrategy &strategy : edgePlan.strategies) {
    signature.push_back(strategy.producerResult);
    signature.push_back(strategy.consumerOperand);
    signature.push_back(static_cast<uint8_t>(strategy.action));
    signature.push_back(strategy.hasLayoutAssignment);
    signature.push_back(static_cast<uint32_t>(strategy.producerLayout));
    signature.push_back(static_cast<uint32_t>(strategy.consumerLayout));
    signature.push_back(strategy.bufferCount);
    signature.push_back(label(strategy.sourceTile));
    signature.push_back(label(strategy.destinationTile));
    signature.push_back(strategy.producerOffsets.size());
    for (int64_t value : strategy.producerOffsets)
      appendSigned(value);
    signature.push_back(strategy.producerSizes.size());
    for (int64_t value : strategy.producerSizes)
      appendSigned(value);
    signature.push_back(strategy.consumerOffsets.size());
    for (int64_t value : strategy.consumerOffsets)
      appendSigned(value);
    signature.push_back(strategy.consumerSizes.size());
    for (int64_t value : strategy.consumerSizes)
      appendSigned(value);
    signature.push_back(strategy.fragments.size());
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      signature.push_back(static_cast<uint8_t>(fragment.kind));
      signature.push_back(label(fragment.sourceTile));
      signature.push_back(fragment.bytes);
      signature.push_back(fragment.communicationId);
      signature.push_back(fragment.offsets.size());
      for (int64_t value : fragment.offsets)
        appendSigned(value);
      signature.push_back(fragment.sizes.size());
      for (int64_t value : fragment.sizes)
        appendSigned(value);
      appendSigned(fragment.payloadSlice);
    }
  }
  signature.push_back(movements.size());
  for (const WholeDAGPeerMovement &movement : movements) {
    signature.push_back(movement.edge);
    signature.push_back(label(movement.sourceTile));
    signature.push_back(label(movement.destinationTile));
    signature.push_back(movement.duration);
    signature.push_back(movement.bufferCount);
    signature.push_back(movement.route.size());
    for (const PhysicalTileDirectedLink &link : movement.route) {
      signature.push_back(label(link.source));
      signature.push_back(label(link.destination));
    }
  }
  return signature;
}

static size_t hashResourceRenamingSignature(llvm::ArrayRef<uint64_t> values) {
  llvm::hash_code hash = llvm::hash_combine(values.size());
  for (uint64_t value : values)
    hash = llvm::hash_combine(hash, value);
  return static_cast<size_t>(hash);
}

bool sameCandidatePlacement(const WholeDAGPlacementCandidate &lhs,
                            const WholeDAGPlacementCandidate &rhs) {
  if (lhs.nodePlacements.size() != rhs.nodePlacements.size())
    return false;
  return llvm::all_of(
      llvm::zip_equal(lhs.nodePlacements, rhs.nodePlacements), [](auto values) {
        auto [left, right] = values;
        return left.node == right.node && samePlacement(left, right);
      });
}

bool placementVectorLess(llvm::ArrayRef<WholeDAGNodePlacement> lhs,
                         llvm::ArrayRef<WholeDAGNodePlacement> rhs) {
  for (auto [left, right] : llvm::zip_equal(lhs, rhs)) {
    if (samePlacement(left, right))
      continue;
    return placementLess(left, right);
  }
  return lhs.size() < rhs.size();
}

bool candidateLess(const WholeDAGPlacementCandidate &lhs,
                   const WholeDAGPlacementCandidate &rhs) {
  const auto lhsScore = std::tuple(
      lhs.schedule.makespan, lhs.schedule.peakLiveSPMBytes,
      lhs.edgePlan.totalPeerBytes, lhs.topologyHopByteWork,
      lhs.topologyCompactnessWork, countPeerFragments(lhs.edgePlan),
      std::numeric_limits<uint32_t>::max() - lhs.distinctTileGroupCount,
      std::numeric_limits<uint32_t>::max() - lhs.parallelComponentCount,
      lhs.disjointEdgeCount, lhs.partialOverlapEdgeCount,
      lhs.sameGroupRemapEdgeCount);
  const auto rhsScore = std::tuple(
      rhs.schedule.makespan, rhs.schedule.peakLiveSPMBytes,
      rhs.edgePlan.totalPeerBytes, rhs.topologyHopByteWork,
      rhs.topologyCompactnessWork, countPeerFragments(rhs.edgePlan),
      std::numeric_limits<uint32_t>::max() - rhs.distinctTileGroupCount,
      std::numeric_limits<uint32_t>::max() - rhs.parallelComponentCount,
      rhs.disjointEdgeCount, rhs.partialOverlapEdgeCount,
      rhs.sameGroupRemapEdgeCount);
  if (lhsScore != rhsScore)
    return lhsScore < rhsScore;
  return placementVectorLess(lhs.nodePlacements, rhs.nodePlacements);
}

} // namespace

mlir::FailureOr<WholeDAGPlacementSearchDomain>
deriveWholeDAGPlacementSearchDomain(const CardDAGAnalysis &dag,
                                    const PhysicalTopology &topology,
                                    PhysicalCardId cardId,
                                    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (dag.getNodes().empty()) {
    setFailure(failureReason, "whole-DAG placement domain requires a DAG");
    return mlir::failure();
  }
  std::optional<llvm::ArrayRef<PhysicalTileId>> availableTiles =
      topology.getAvailableTileIds(cardId);
  if (!availableTiles || availableTiles->empty()) {
    setFailure(failureReason,
               "whole-DAG placement domain requires available Tiles");
    return mlir::failure();
  }
  llvm::SmallVector<TopologyTileGroup, 128> groups =
      deriveTopologyGroups(topology, cardId, *availableTiles);
  if (groups.empty()) {
    setFailure(failureReason,
               "whole-DAG placement domain has no connected Tile group");
    return mlir::failure();
  }
  WholeDAGPlacementSearchDomain domain;
  domain.placementGroupCount = groups.size();
  domain.nodeOptions.reserve(dag.getNodes().size());
  for (const CardDAGNode &node : dag.getNodes()) {
    llvm::SmallVector<WholeDAGNodePlacement, 32> options =
        deriveNodePlacementDomain(node, groups);
    if (options.empty()) {
      setFailure(failureReason,
                 "whole-DAG placement domain cannot place a DAG node");
      return mlir::failure();
    }
    domain.nodeOptions.push_back(std::move(options));
  }
  return domain;
}

class WholeDAGPlacementEvaluator::Impl {
public:
  Impl(const CardDAGAnalysis &dag, const PhysicalTopology &topology,
       PhysicalCardId cardId)
      : dag(dag), topology(topology), cardId(cardId), edgePlanner(dag) {}

  const CardDAGAnalysis &dag;
  const PhysicalTopology &topology;
  PhysicalCardId cardId;
  WholeDAGEdgeStrategyPlanner edgePlanner;

  mlir::FailureOr<const WholeDAGEdgeStrategyPlan *>
  getEdgePlan(const CardDAGEdge &edge, const WholeDAGNodePlacement &producer,
              const WholeDAGNodePlacement &consumer,
              std::string *failureReason) {
    const size_t hash = static_cast<size_t>(llvm::hash_combine(
        edge.id, getPlacementHash(producer), getPlacementHash(consumer)));
    llvm::SmallVector<EdgePlanEntry, 1> &bucket = edgePlans[hash];
    auto found = llvm::find_if(bucket, [&](const EdgePlanEntry &entry) {
      return entry.edge == edge.id && samePlacement(entry.producer, producer) &&
             samePlacement(entry.consumer, consumer);
    });
    if (found != bucket.end()) {
      if (!found->legal) {
        setFailure(failureReason, found->failureReason);
        return mlir::failure();
      }
      return &found->plan;
    }

    EdgePlanEntry entry;
    entry.edge = edge.id;
    entry.producer = producer;
    entry.consumer = consumer;
    mlir::FailureOr<WholeDAGEdgeStrategyPlan> plan =
        edgePlanner.derive(edge.id, producer, consumer, &entry.failureReason);
    entry.legal = mlir::succeeded(plan);
    if (entry.legal)
      entry.plan = std::move(*plan);
    bucket.push_back(std::move(entry));
    EdgePlanEntry &stored = bucket.back();
    if (!stored.legal) {
      setFailure(failureReason, stored.failureReason);
      return mlir::failure();
    }
    return &stored.plan;
  }

private:
  struct EdgePlanEntry {
    CardDAGEdgeID edge = 0;
    WholeDAGNodePlacement producer;
    WholeDAGNodePlacement consumer;
    WholeDAGEdgeStrategyPlan plan;
    std::string failureReason;
    bool legal = false;
  };
  std::unordered_map<size_t, llvm::SmallVector<EdgePlanEntry, 1>> edgePlans;
};

WholeDAGPlacementEvaluator::WholeDAGPlacementEvaluator(
    const CardDAGAnalysis &dag, const PhysicalTopology &topology,
    PhysicalCardId cardId)
    : impl(std::make_unique<Impl>(dag, topology, cardId)) {}

WholeDAGPlacementEvaluator::~WholeDAGPlacementEvaluator() = default;
WholeDAGPlacementEvaluator::WholeDAGPlacementEvaluator(
    WholeDAGPlacementEvaluator &&) noexcept = default;
WholeDAGPlacementEvaluator &WholeDAGPlacementEvaluator::operator=(
    WholeDAGPlacementEvaluator &&) noexcept = default;

mlir::FailureOr<WholeDAGPlacementCandidate>
WholeDAGPlacementEvaluator::evaluate(
    llvm::ArrayRef<WholeDAGNodePlacement> requestedPlacements,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  const CardDAGAnalysis &dag = impl->dag;
  const PhysicalTopology &topology = impl->topology;
  const PhysicalCardId cardId = impl->cardId;
  if (requestedPlacements.size() != dag.getNodes().size()) {
    setFailure(failureReason,
               "whole-DAG placement evaluation requires every DAG node");
    return mlir::failure();
  }
  std::optional<llvm::ArrayRef<PhysicalTileId>> availableTiles =
      topology.getAvailableTileIds(cardId);
  if (!availableTiles || availableTiles->empty()) {
    setFailure(failureReason,
               "whole-DAG placement evaluation requires available Tiles");
    return mlir::failure();
  }

  PartialPlacementState state;
  for (auto [node, placement] :
       llvm::zip_equal(dag.getNodes(), requestedPlacements)) {
    if (placement.node != node.id || placement.tiles.empty()) {
      setFailure(failureReason,
                 "whole-DAG placement evaluation has an invalid node option");
      return mlir::failure();
    }
    state = extendState(dag, topology, cardId, state, placement);
  }
  llvm::SmallVector<WholeDAGNodePlacement, 16> placements =
      materializePlacements(state);
  mlir::FailureOr<llvm::SmallVector<WholeDAGObservablePlacement, 4>> outputs =
      deriveObservablePlacements(dag, placements);
  if (mlir::failed(outputs)) {
    setFailure(failureReason,
               "whole-DAG placement has inconsistent observable roots");
    return mlir::failure();
  }
  // Reuse the placement-independent exact relation proof across the complete
  // factorized query.  The per-placement fragment carrier is still rebuilt
  // here because its resident/peer split, payload domains and Tile endpoints
  // are observable by routing, residency and actual materialization.
  WholeDAGEdgeStrategyPlan edgePlan;
  for (const CardDAGEdge &edge : dag.getEdges()) {
    mlir::FailureOr<const WholeDAGEdgeStrategyPlan *> edgeResult =
        impl->getEdgePlan(edge, placements[edge.producer],
                          placements[edge.consumer], failureReason);
    if (mlir::failed(edgeResult))
      return mlir::failure();
    const WholeDAGEdgeStrategyPlan &cachedEdgePlan = **edgeResult;
    if (cachedEdgePlan.totalPeerBytes >
        std::numeric_limits<uint64_t>::max() - edgePlan.totalPeerBytes) {
      setFailure(failureReason, "whole-DAG placement peer bytes overflow");
      return mlir::failure();
    }
    edgePlan.totalPeerBytes += cachedEdgePlan.totalPeerBytes;
    edgePlan.strategies.append(cachedEdgePlan.strategies.begin(),
                               cachedEdgePlan.strategies.end());
  }
  std::optional<llvm::SmallVector<WholeDAGPeerMovement, 32>> movements =
      derivePeerMovements(topology, cardId, dag, edgePlan);
  std::optional<llvm::SmallVector<WholeDAGLocalResidency, 32>> residencies =
      derivePlacementResidencies(dag, edgePlan);
  if (!movements || !residencies) {
    setFailure(failureReason,
               "whole-DAG placement cannot derive exact edge resources");
    return mlir::failure();
  }
  mlir::FailureOr<WholeDAGCandidateSchedule> schedule =
      scheduleWholeDAGCandidate(dag, *availableTiles, placements, *residencies,
                                failureReason, *movements,
                                /*localMovements=*/{},
                                /*enforceSPMCapacity=*/false);
  if (mlir::failed(schedule))
    return mlir::failure();

  WholeDAGPlacementCandidate candidate;
  candidate.nodePlacements = schedule->nodePlacements;
  candidate.outputPlacements = std::move(*outputs);
  candidate.schedule = std::move(*schedule);
  candidate.edgePlan = std::move(edgePlan);
  candidate.topologyHopByteWork = getTopologyHopByteWork(*movements);
  candidate.topologyCompactnessWork = state.topologyCompactnessWork;
  candidate.distinctTileGroupCount = state.distinctGroups;
  candidate.parallelComponentCount = state.parallelComponents;
  candidate.localEdgeCount = state.localEdges;
  candidate.sameGroupRemapEdgeCount = state.sameGroupRemapEdges;
  candidate.partialOverlapEdgeCount = state.partialEdges;
  candidate.disjointEdgeCount = state.disjointEdges;
  return candidate;
}

mlir::FailureOr<WholeDAGPlacementCandidate> evaluateWholeDAGPlacement(
    const CardDAGAnalysis &dag, const PhysicalTopology &topology,
    PhysicalCardId cardId,
    llvm::ArrayRef<WholeDAGNodePlacement> requestedPlacements,
    std::string *failureReason) {
  WholeDAGPlacementEvaluator evaluator(dag, topology, cardId);
  return evaluator.evaluate(requestedPlacements, failureReason);
}

mlir::FailureOr<llvm::SmallVector<WholeDAGPlacementCandidate, 12>>
enumerateWholeDAGPlacements(const CardDAGAnalysis &dag,
                            const PhysicalTopology &topology,
                            PhysicalCardId cardId,
                            WholeDAGPlacementEnumerationStatistics *statistics,
                            std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  WholeDAGPlacementEnumerationStatistics localStatistics;
  WholeDAGPlacementEnumerationStatistics &resultStatistics =
      statistics ? *statistics : localStatistics;
  resultStatistics = {};
  if (dag.getNodes().empty()) {
    setFailure(failureReason, "whole-DAG placement enumeration requires a DAG");
    return mlir::failure();
  }
  std::optional<llvm::ArrayRef<PhysicalTileId>> availableTiles =
      topology.getAvailableTileIds(cardId);
  if (!availableTiles || availableTiles->empty()) {
    setFailure(failureReason,
               "whole-DAG placement enumeration requires available Tiles");
    return mlir::failure();
  }
  if (dag.getObservableOutputRootNodes().size() !=
          dag.getFunction().getNumResults() ||
      llvm::any_of(
          dag.getObservableOutputRootNodes(),
          [](llvm::ArrayRef<CardDAGNodeID> roots) { return roots.empty(); }))
    return llvm::SmallVector<WholeDAGPlacementCandidate, 12>{};

  llvm::SmallVector<TopologyTileGroup, 128> groups =
      deriveTopologyGroups(topology, cardId, *availableTiles);
  resultStatistics.placementGroupCount = groups.size();
  if (groups.empty()) {
    setFailure(failureReason,
               "whole-DAG placement enumeration has no connected Tile group");
    return mlir::failure();
  }

  llvm::SmallVector<PartialPlacementState, 24> retainedStates(1);
  EdgeTransitionLegalityCache transitionCache(dag);
  for (const CardDAGNode &node : dag.getNodes()) {
    resultStatistics.expandedStates =
        saturatingAdd(resultStatistics.expandedStates, retainedStates.size());
    llvm::SmallVector<PartialPlacementState, 128> expanded;
    for (const PartialPlacementState &state : retainedStates) {
      llvm::SmallVector<WholeDAGNodePlacement, 32> options = deriveNodeOptions(
          dag, node, state, groups, transitionCache, resultStatistics);
      for (WholeDAGNodePlacement &option : options)
        expanded.push_back(
            extendState(dag, topology, cardId, state, std::move(option)));
    }
    if (expanded.empty()) {
      setFailure(failureReason,
                 "whole-DAG placement enumeration cannot place a DAG node");
      return mlir::failure();
    }
    retainedStates =
        retainNondominatedPlacementStates(dag, std::move(expanded));
  }

  // Preserve one topology-generic operator-pipeline representative: when the
  // card has at least one endpoint per structured node, bind consecutive DAG
  // nodes to distinct singleton Tiles.  This is not a workload special case;
  // it is the finite opposite of maximal co-location and gives exact
  // materialization a redistribution state whose receive demand is a single
  // complete fragment.  The ordinary edge-plan/scheduler gates below still
  // decide whether the state is legal.
  if (dag.getNodes().size() <= availableTiles->size()) {
    PartialPlacementState singletonPipeline;
    bool complete = true;
    for (const CardDAGNode &node : dag.getNodes()) {
      std::optional<llvm::SmallVector<StaticSpatialAxis, 4>> axes =
          getNodeSpatialAxes(node);
      if (!axes || axes->empty()) {
        complete = false;
        break;
      }
      const StaticSpatialAxis &axis = axes->front();
      WholeDAGNodePlacement placement;
      placement.node = node.id;
      placement.shardDimension = axis.resultDimension;
      placement.tiles.push_back((*availableTiles)[node.id]);
      placement.spatialIteratorDimension = axis.iteratorDimension;
      placement.iteratorPartitionFactors = axis.basePartitionFactors;
      singletonPipeline = extendState(dag, topology, cardId, singletonPipeline,
                                      std::move(placement));
    }
    if (complete &&
        llvm::none_of(retainedStates, [&](const PartialPlacementState &state) {
          return samePartialPlacement(state, singletonPipeline);
        }))
      retainedStates.push_back(std::move(singletonPipeline));
  }

  llvm::SmallVector<WholeDAGPlacementCandidate, 24> accepted;
  std::unordered_map<size_t, llvm::SmallVector<size_t, 1>>
      resourceRenamingClasses;
  WholeDAGEdgeStrategyPlanner finalEdgePlanner(dag);
  for (PartialPlacementState &state : retainedStates) {
    llvm::SmallVector<WholeDAGNodePlacement, 16> statePlacements =
        materializePlacements(state);
    mlir::FailureOr<llvm::SmallVector<WholeDAGObservablePlacement, 4>> outputs =
        deriveObservablePlacements(dag, statePlacements);
    if (mlir::failed(outputs)) {
      resultStatistics.rejectedTransitions =
          saturatingAdd(resultStatistics.rejectedTransitions, 1);
      continue;
    }
    std::string ignoredFailure;
    mlir::FailureOr<WholeDAGEdgeStrategyPlan> edgePlan =
        finalEdgePlanner.derive(statePlacements, &ignoredFailure);
    if (mlir::failed(edgePlan)) {
      resultStatistics.rejectedTransitions =
          saturatingAdd(resultStatistics.rejectedTransitions, 1);
      continue;
    }
    std::optional<llvm::SmallVector<WholeDAGPeerMovement, 32>> movements =
        derivePeerMovements(topology, cardId, dag, *edgePlan);
    std::optional<llvm::SmallVector<WholeDAGLocalResidency, 32>> residencies =
        derivePlacementResidencies(dag, *edgePlan);
    if (!movements || !residencies) {
      resultStatistics.rejectedTransitions =
          saturatingAdd(resultStatistics.rejectedTransitions, 1);
      continue;
    }
    std::vector<uint64_t> resourceSignature = getResourceRenamingSignature(
        statePlacements, *outputs, *edgePlan, *movements);
    const size_t resourceHash =
        hashResourceRenamingSignature(resourceSignature);
    llvm::SmallVector<size_t, 1> &resourceBucket =
        resourceRenamingClasses[resourceHash];
    std::optional<size_t> resourceEquivalent;
    for (size_t representativeIndex : resourceBucket) {
      const WholeDAGPlacementCandidate &representative =
          accepted[representativeIndex];
      std::optional<llvm::SmallVector<WholeDAGPeerMovement, 32>>
          representativeMovements = derivePeerMovements(
              topology, cardId, dag, representative.edgePlan);
      if (representativeMovements &&
          getResourceRenamingSignature(
              representative.nodePlacements, representative.outputPlacements,
              representative.edgePlan,
              *representativeMovements) == resourceSignature) {
        resourceEquivalent = representativeIndex;
        break;
      }
    }
    if (resourceEquivalent) {
      resultStatistics.resourceRenamingEquivalentStates =
          saturatingAdd(resultStatistics.resourceRenamingEquivalentStates, 1);
      // Resource-renamed physical translations are exactly equivalent for
      // the common calendar, but the representative itself is observable in
      // the selected IR. Keep the lexicographically canonical placement,
      // independent of unordered state-bucket iteration order.
      const WholeDAGPlacementCandidate &representative =
          accepted[*resourceEquivalent];
      if (state.topologyCompactnessWork >
              representative.topologyCompactnessWork ||
          (state.topologyCompactnessWork ==
               representative.topologyCompactnessWork &&
           !placementVectorLess(statePlacements,
                                representative.nodePlacements)))
        continue;
    }
    // Placement is evaluated before a temporal tile and edge-retention action
    // exist.  Feed the exact spatial resident fragments into the common event
    // state so search never exercises a zero LiveSPM path, but do not use
    // the full spatial shard as a hard capacity rejection.  Temporal/layout/
    // buffer transitions immediately reschedule with their selected windows
    // and enforce the search-time proven capacity bound.
    mlir::FailureOr<WholeDAGCandidateSchedule> schedule =
        scheduleWholeDAGCandidate(dag, *availableTiles, statePlacements,
                                  *residencies, &ignoredFailure, *movements,
                                  /*localMovements=*/{},
                                  /*enforceSPMCapacity=*/false);
    if (mlir::failed(schedule)) {
      resultStatistics.rejectedTransitions =
          saturatingAdd(resultStatistics.rejectedTransitions, 1);
      continue;
    }

    WholeDAGPlacementCandidate candidate;
    candidate.nodePlacements = schedule->nodePlacements;
    candidate.outputPlacements = std::move(*outputs);
    candidate.schedule = std::move(*schedule);
    candidate.edgePlan = std::move(*edgePlan);
    candidate.topologyHopByteWork = getTopologyHopByteWork(*movements);
    candidate.topologyCompactnessWork = state.topologyCompactnessWork;
    candidate.distinctTileGroupCount = state.distinctGroups;
    candidate.parallelComponentCount = state.parallelComponents;
    candidate.localEdgeCount = state.localEdges;
    candidate.sameGroupRemapEdgeCount = state.sameGroupRemapEdges;
    candidate.partialOverlapEdgeCount = state.partialEdges;
    candidate.disjointEdgeCount = state.disjointEdges;
    if (resourceEquivalent) {
      accepted[*resourceEquivalent] = std::move(candidate);
    } else if (llvm::none_of(accepted, [&](const auto &existing) {
                 return sameCandidatePlacement(existing, candidate);
               })) {
      accepted.push_back(std::move(candidate));
      resourceBucket.push_back(accepted.size() - 1);
    }
  }

  llvm::sort(accepted, candidateLess);
  llvm::SmallVector<WholeDAGPlacementCandidate, 12> result;
  auto retainFirst = [&](auto predicate) {
    auto found = llvm::find_if(accepted, predicate);
    if (found == accepted.end())
      return;
    result.push_back(std::move(*found));
    accepted.erase(found);
  };
  retainFirst([](const WholeDAGPlacementCandidate &candidate) {
    return countPeerFragments(candidate.edgePlan) == 0;
  });
  retainFirst([](const WholeDAGPlacementCandidate &candidate) {
    return candidate.sameGroupRemapEdgeCount != 0;
  });
  retainFirst([](const WholeDAGPlacementCandidate &candidate) {
    return candidate.partialOverlapEdgeCount != 0;
  });
  retainFirst([](const WholeDAGPlacementCandidate &candidate) {
    return candidate.disjointEdgeCount != 0;
  });
  retainFirst([](const WholeDAGPlacementCandidate &candidate) {
    return candidate.parallelComponentCount > 1;
  });
  retainFirst([&](const WholeDAGPlacementCandidate &candidate) {
    return candidate.nodePlacements.size() == dag.getNodes().size() &&
           candidate.distinctTileGroupCount == dag.getNodes().size() &&
           llvm::all_of(candidate.nodePlacements, [](const auto &placement) {
             return placement.tiles.size() == 1;
           });
  });
  retainFirst([](const WholeDAGPlacementCandidate &candidate) {
    return candidate.distinctTileGroupCount >= 3;
  });
  while (!accepted.empty()) {
    result.push_back(std::move(accepted.front()));
    accepted.erase(accepted.begin());
  }
  llvm::sort(result, candidateLess);
  return result;
}

} // namespace wafer::compiler::detail
