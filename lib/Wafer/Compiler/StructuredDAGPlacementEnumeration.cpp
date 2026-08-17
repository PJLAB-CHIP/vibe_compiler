//===- StructuredDAGPlacementEnumeration.cpp - Joint node placement
//------------===//

#include "StructuredDAGPlacementEnumeration.h"

#include "StructuredDAGExactDemandQuery.h"

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

bool tileVectorLess(llvm::ArrayRef<TileId> lhs, llvm::ArrayRef<TileId> rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), [](TileId left, TileId right) {
                                        return left.getValue() <
                                               right.getValue();
                                      });
}

size_t getOverlap(llvm::ArrayRef<TileId> lhs, llvm::ArrayRef<TileId> rhs) {
  size_t overlap = 0;
  for (TileId tile : lhs)
    overlap += llvm::is_contained(rhs, tile);
  return overlap;
}

struct TopologyTileGroup {
  llvm::SmallVector<TileId, 16> tiles;
  uint64_t internalHopWork = 0;
};

bool sameGroup(const TopologyTileGroup &lhs, const TopologyTileGroup &rhs) {
  return lhs.tiles == rhs.tiles;
}

uint64_t getInternalHopWork(const TargetTopology &topology, CardId cardId,
                            llvm::ArrayRef<TileId> tiles) {
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
deriveTopologyGroups(const TargetTopology &topology, CardId cardId,
                     llvm::ArrayRef<TileId> availableTiles) {
  llvm::SmallVector<TopologyTileGroup, 128> groups;
  if (availableTiles.empty())
    return groups;
  auto append = [&](llvm::ArrayRef<TileId> tiles) {
    if (tiles.empty())
      return;
    TopologyTileGroup group;
    group.tiles.append(tiles.begin(), tiles.end());
    llvm::sort(group.tiles, [&](TileId lhs, TileId rhs) {
      std::optional<TileCoordinate> left = topology.getTileCoordinate(lhs);
      std::optional<TileCoordinate> right = topology.getTileCoordinate(rhs);
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
          llvm::SmallVector<TileId, 16> rectangle;
          bool available = true;
          for (int64_t dy = 0; dy < height && available; ++dy) {
            for (int64_t dx = 0; dx < width; ++dx) {
              std::optional<TileId> tile =
                  topology.getTileId(TileCoordinate{y + dy, x + dx});
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
getNodeSpatialAxes(const StructuredDAGNode &node) {
  if (!node.operation || node.operation->getNumResults() == 0)
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

bool samePlacement(const StructuredDAGNodePlacement &lhs,
                   const StructuredDAGNodePlacement &rhs) {
  return lhs.spatialIteratorDimension == rhs.spatialIteratorDimension &&
         lhs.iteratorPartitionFactors == rhs.iteratorPartitionFactors &&
         lhs.shardDimension == rhs.shardDimension && lhs.tiles == rhs.tiles;
}

bool placementLess(const StructuredDAGNodePlacement &lhs,
                   const StructuredDAGNodePlacement &rhs) {
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

static size_t getPlacementHash(const StructuredDAGNodePlacement &placement) {
  llvm::hash_code hash = llvm::hash_combine(
      placement.spatialIteratorDimension, placement.shardDimension,
      placement.iteratorPartitionFactors.size(), placement.tiles.size());
  for (uint32_t factor : placement.iteratorPartitionFactors)
    hash = llvm::hash_combine(hash, factor);
  for (TileId tile : placement.tiles)
    hash = llvm::hash_combine(hash, tile.getValue());
  return static_cast<size_t>(hash);
}

/// Exact query-local legality of one closed DAG edge transition. The memo key
/// observes the complete placement pair through a proven projection: for the
/// balanced placement domain the complete consumer iteration domain, per-Tile
/// ownership domains and roles are a deterministic function of
/// (extent, shard dimension, participant count), and Tile identity merely
/// relabels the same interval collection, so the verdict is invariant under
/// Tile relabeling. The memo therefore keys on the domain-determining fields
/// instead of a placement legality bool, and the stored value stays typed:
/// only Satisfied and ProvenLogicalInfeasible are legality conclusions;
/// UnsupportedSemanticRelation and IndeterminateFailure stop the owning
/// enumeration path and are never cached as placement-illegal.
class EdgeTransitionLegalityCache {
public:
  EdgeTransitionLegalityCache(const StructuredDAGAnalysis &dag,
                              analysis::IREpoch epoch)
      : dag(dag), query(dag, epoch) {}

  analysis::ExactDemandStatus lookupOrCompute(
      StructuredDAGEdgeID edgeID, const StructuredDAGNodePlacement &producer,
      const StructuredDAGNodePlacement &consumer, std::string *failureReason) {
    TransitionRelation relation = getRelation(producer, consumer);
    const size_t hash = static_cast<size_t>(llvm::hash_combine(
        edgeID, relation.producerShardDimension,
        relation.consumerShardDimension, relation.producerParticipants,
        relation.consumerParticipants));
    llvm::SmallVector<Entry, 1> &bucket = buckets[hash];
    auto found = llvm::find_if(bucket, [&](const Entry &entry) {
      return entry.edgeID == edgeID && entry.relation == relation;
    });
    if (found != bucket.end()) {
      if (failureReason)
        *failureReason = found->detail;
      return found->status;
    }
    Entry entry;
    entry.edgeID = edgeID;
    entry.relation = relation;
    mlir::FailureOr<analysis::LogicalShardTrial> trial = buildEdgeShardTrial(
        dag, producer, consumer, query.getEpoch(), &entry.detail);
    if (mlir::failed(trial))
      entry.status = analysis::ExactDemandStatus::IndeterminateFailure;
    else {
      analysis::ExactDemandResult demand = query.query(edgeID, *trial);
      entry.status = demand.status;
      if (demand.detail.size() > entry.detail.size())
        entry.detail = std::move(demand.detail);
    }
    const analysis::ExactDemandStatus status = entry.status;
    const std::string detail = entry.detail;
    bucket.push_back(std::move(entry));
    if (failureReason)
      *failureReason = detail;
    return status;
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

  static TransitionRelation
  getRelation(const StructuredDAGNodePlacement &producer,
              const StructuredDAGNodePlacement &consumer) {
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
    StructuredDAGEdgeID edgeID = 0;
    TransitionRelation relation;
    analysis::ExactDemandStatus status =
        analysis::ExactDemandStatus::IndeterminateFailure;
    std::string detail;
  };
  std::unordered_map<size_t, llvm::SmallVector<Entry, 1>> buckets;
  const StructuredDAGAnalysis &dag;
  StructuredDAGExactDemandQuery query;
};

uint64_t getNodeElementWork(const StructuredDAGNode &node) {
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
  StructuredDAGNodePlacement placement;
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

static const StructuredDAGNodePlacement *
findPlacement(const PartialPlacementState &state, StructuredDAGNodeID node) {
  for (const PlacementPathNode *current = state.path.get(); current;
       current = current->parent.get())
    if (current->placement.node == node)
      return &current->placement;
  return nullptr;
}

static llvm::SmallVector<StructuredDAGNodePlacement, 16>
materializePlacements(const PartialPlacementState &state) {
  llvm::SmallVector<StructuredDAGNodePlacement, 16> result(
      getPlacementCount(state));
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
classifyPlacementRelation(const StructuredDAGNodePlacement &producer,
                          const StructuredDAGNodePlacement &consumer) {
  if (samePlacement(producer, consumer))
    return PlacementRelationClass::ExactLocal;
  if (producer.tiles == consumer.tiles)
    return PlacementRelationClass::SameGroupRemap;
  if (getOverlap(producer.tiles, consumer.tiles) != 0)
    return PlacementRelationClass::GenuinePartialOverlap;
  return PlacementRelationClass::Disjoint;
}

uint32_t
countDistinctGroups(llvm::ArrayRef<StructuredDAGNodePlacement> placements) {
  uint32_t result = 0;
  for (size_t index = 0; index < placements.size(); ++index)
    if (llvm::none_of(llvm::ArrayRef(placements).take_front(index),
                      [&](const StructuredDAGNodePlacement &previous) {
                        return previous.tiles == placements[index].tiles;
                      }) &&
        result != std::numeric_limits<uint32_t>::max())
      ++result;
  return result;
}

uint32_t
countUtilizedTiles(llvm::ArrayRef<StructuredDAGNodePlacement> placements) {
  llvm::SmallVector<int64_t, 16> tiles;
  for (const StructuredDAGNodePlacement &placement : placements)
    for (TileId tile : placement.tiles)
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
getComponentConcurrency(const StructuredDAGAnalysis &dag,
                        llvm::ArrayRef<StructuredDAGNodePlacement> placements) {
  llvm::ArrayRef<StructuredDAGDependencyComponent> components =
      dag.getObservableDependencyComponents();
  if (!dag.supportsIndependentComponentPlacement() || components.size() < 2)
    return {};

  llvm::SmallVector<llvm::SmallVector<TileId, 16>, 4> componentTiles(
      components.size());
  for (const StructuredDAGNodePlacement &placement : placements) {
    auto component = llvm::find_if(
        components, [&](const StructuredDAGDependencyComponent &candidate) {
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
    llvm::sort(tiles, [](TileId lhs, TileId rhs) {
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
                       [&](TileId tile) {
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

uint64_t getGroupTransitionPenalty(const TargetTopology &topology,
                                   CardId cardId,
                                   const StructuredDAGNodePlacement &producer,
                                   const StructuredDAGNodePlacement &consumer) {
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
  for (TileId destination : consumer.tiles) {
    uint64_t nearest = std::numeric_limits<uint64_t>::max();
    for (TileId source : producer.tiles) {
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

llvm::SmallVector<StructuredDAGNodePlacement, 32>
deriveNodePlacementDomain(const StructuredDAGNode &node,
                          llvm::ArrayRef<TopologyTileGroup> groups) {
  struct PlacementOption {
    StructuredDAGNodePlacement placement;
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
  llvm::SmallVector<StructuredDAGNodePlacement, 32> result;
  result.reserve(options.size());
  for (PlacementOption &option : options)
    if (result.empty() || !samePlacement(result.back(), option.placement))
      result.push_back(std::move(option.placement));
  return result;
}

llvm::SmallVector<StructuredDAGNodePlacement, 32>
deriveNodeOptions(const StructuredDAGAnalysis &dag,
                  const StructuredDAGNode &node,
                  const PartialPlacementState &state,
                  llvm::ArrayRef<TopologyTileGroup> groups,
                  EdgeTransitionLegalityCache &transitionCache,
                  StructuredDAGPlacementEnumerationStatistics &statistics,
                  StructuredDAGPlacementLegality *legality = nullptr) {
  struct PlacementOption {
    StructuredDAGNodePlacement placement;
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
    for (TileId tile : current->placement.tiles)
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
      option.unusedTiles =
          static_cast<uint32_t>(llvm::count_if(group.tiles, [&](TileId tile) {
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

  llvm::SmallVector<int8_t, 128> optionLegality(options.size(), -1);
  bool aborted = false;
  auto isLegal = [&](size_t optionIndex) {
    if (optionLegality[optionIndex] >= 0)
      return optionLegality[optionIndex] != 0;
    const StructuredDAGNodePlacement &candidate =
        options[optionIndex].placement;
    for (StructuredDAGEdgeID edgeID : node.incomingEdges) {
      const StructuredDAGEdge *edge = dag.getEdge(edgeID);
      const StructuredDAGNodePlacement *producer =
          edge ? findPlacement(state, edge->producer) : nullptr;
      if (!edge || !producer) {
        optionLegality[optionIndex] = 0;
        return false;
      }
      std::string edgeFailure;
      analysis::ExactDemandStatus status = transitionCache.lookupOrCompute(
          edgeID, *producer, candidate, &edgeFailure);
      if (status == analysis::ExactDemandStatus::Satisfied)
        continue;
      if (status == analysis::ExactDemandStatus::ProvenLogicalInfeasible) {
        optionLegality[optionIndex] = 0;
        return false;
      }
      // UnsupportedSemanticRelation and IndeterminateFailure apply to the
      // semantics, not to this option: they stop the whole enumeration path
      // and are never cached or counted as a placement rejection.
      if (legality) {
        legality->status = status;
        legality->detail = std::move(edgeFailure);
      }
      aborted = true;
      optionLegality[optionIndex] = 0;
      return false;
    }
    optionLegality[optionIndex] = 1;
    return true;
  };

  llvm::SmallVector<StructuredDAGNodePlacement, 32> legal;
  legal.reserve(options.size());
  for (size_t index = 0; index < options.size(); ++index) {
    if (isLegal(index)) {
      legal.push_back(options[index].placement);
    } else if (!aborted) {
      statistics.rejectedTransitions =
          saturatingAdd(statistics.rejectedTransitions, 1);
    } else {
      return {};
    }
  }
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
        llvm::find_if(legal, [&](const StructuredDAGNodePlacement &option) {
          return llvm::none_of(option.tiles, [&](TileId tile) {
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

PartialPlacementState extendState(const StructuredDAGAnalysis &dag,
                                  const TargetTopology &topology, CardId cardId,
                                  const PartialPlacementState &parent,
                                  StructuredDAGNodePlacement placement) {
  PartialPlacementState result = parent;
  const StructuredDAGNode &node = dag.getNodes()[placement.node];
  const uint64_t work = getNodeElementWork(node);
  result.computeWorkLowerBound = saturatingAdd(
      result.computeWorkLowerBound,
      placement.tiles.empty() ? std::numeric_limits<uint64_t>::max()
                              : work / placement.tiles.size() +
                                    (work % placement.tiles.size() != 0));
  result.topologyCompactnessWork =
      saturatingAdd(result.topologyCompactnessWork,
                    getInternalHopWork(topology, cardId, placement.tiles));
  for (StructuredDAGEdgeID edgeID : node.incomingEdges) {
    const StructuredDAGEdge *edge = dag.getEdge(edgeID);
    const StructuredDAGNodePlacement *producer =
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
  llvm::SmallVector<StructuredDAGNodePlacement, 16> placements =
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
  llvm::SmallVector<StructuredDAGNodePlacement, 16> lhsPlacements =
      materializePlacements(lhs);
  llvm::SmallVector<StructuredDAGNodePlacement, 16> rhsPlacements =
      materializePlacements(rhs);
  for (auto [left, right] : llvm::zip_equal(lhsPlacements, rhsPlacements)) {
    if (samePlacement(left, right))
      continue;
    return placementLess(left, right);
  }
  return false;
}

static bool isObservableRoot(const StructuredDAGAnalysis &dag,
                             StructuredDAGNodeID node) {
  return llvm::any_of(dag.getObservableOutputRootNodes(),
                      [&](llvm::ArrayRef<StructuredDAGNodeID> roots) {
                        return llvm::is_contained(roots, node);
                      });
}

static bool remainsOnLiveBoundary(const StructuredDAGAnalysis &dag,
                                  StructuredDAGNodeID node,
                                  size_t placedNodes) {
  if (isObservableRoot(dag, node))
    return true;
  const StructuredDAGNode *current = dag.getNode(node);
  return current &&
         llvm::any_of(current->outgoingEdges, [&](StructuredDAGEdgeID edgeID) {
           const StructuredDAGEdge *edge = dag.getEdge(edgeID);
           return edge && edge->consumer >= placedNodes;
         });
}

static bool sameLiveBoundary(const StructuredDAGAnalysis &dag,
                             const PartialPlacementState &lhs,
                             const PartialPlacementState &rhs) {
  if (getPlacementCount(lhs) != getPlacementCount(rhs))
    return false;
  const size_t placedNodes = getPlacementCount(lhs);
  for (StructuredDAGNodeID node = 0; node < placedNodes; ++node) {
    if (!remainsOnLiveBoundary(dag, node, placedNodes))
      continue;
    const StructuredDAGNodePlacement *left = findPlacement(lhs, node);
    const StructuredDAGNodePlacement *right = findPlacement(rhs, node);
    if (!left || !right || !samePlacement(*left, *right))
      return false;
  }
  return true;
}

static size_t getLiveBoundaryHash(const StructuredDAGAnalysis &dag,
                                  const PartialPlacementState &state) {
  const size_t placedNodes = getPlacementCount(state);
  llvm::hash_code hash = llvm::hash_combine(placedNodes);
  for (StructuredDAGNodeID node = 0; node < placedNodes; ++node) {
    if (!remainsOnLiveBoundary(dag, node, placedNodes))
      continue;
    const StructuredDAGNodePlacement *placement = findPlacement(state, node);
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
static bool strictlyDominatesPartialState(const StructuredDAGAnalysis &dag,
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
    const StructuredDAGAnalysis &dag,
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
mapRootShardAxisToOutput(const StructuredDAGAnalysis &dag, unsigned outputIndex,
                         const StructuredDAGNode &root, unsigned rootAxis,
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

mlir::FailureOr<llvm::SmallVector<StructuredDAGObservablePlacement, 4>>
deriveObservablePlacements(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> placements) {
  llvm::SmallVector<StructuredDAGObservablePlacement, 4> outputs;
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
    const StructuredDAGNodePlacement &owner = placements[roots.front()];
    if (llvm::any_of(roots, [&](StructuredDAGNodeID root) {
          return root >= placements.size() ||
                 !samePlacement(owner, placements[root]);
        }))
      return mlir::failure();
    // The placed shard axis names an axis of the root result; map it through
    // the pure support chain to the observable output domain.  Every root of
    // the result must map to the same output axis, and the domain must be able
    // to hold the active Tile count.  An inexpressible chain fails closed.
    std::optional<unsigned> outputAxis;
    for (StructuredDAGNodeID rootID : roots) {
      const StructuredDAGNode *rootNode = dag.getNode(rootID);
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
    outputs.push_back(StructuredDAGObservablePlacement{
        static_cast<uint32_t>(outputIndex), *outputAxis, owner.tiles});
  }
  return outputs;
}

std::optional<StructuredDAGEdgeID>
resolveStrategyEdge(const StructuredDAGAnalysis &dag,
                    const SpatialEdgeStrategy &strategy) {
  for (const StructuredDAGEdge &edge : dag.getEdges()) {
    const StructuredDAGNode *producer = dag.getNode(edge.producer);
    const StructuredDAGNode *consumer = dag.getNode(edge.consumer);
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

std::optional<llvm::SmallVector<StructuredDAGLocalResidency, 32>>
derivePlacementResidencies(const StructuredDAGAnalysis &dag,
                           const StructuredDAGEdgeStrategyPlan &plan) {
  std::map<std::pair<StructuredDAGEdgeID, int64_t>, uint64_t> bytesByEdgeTile;
  for (const SpatialEdgeStrategy &strategy : plan.strategies) {
    std::optional<StructuredDAGEdgeID> edge =
        resolveStrategyEdge(dag, strategy);
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
  llvm::SmallVector<StructuredDAGLocalResidency, 32> result;
  for (const auto &[key, bytes] : bytesByEdgeTile)
    result.push_back(
        StructuredDAGLocalResidency{key.first, TileId(key.second), bytes});
  return result;
}

std::optional<llvm::SmallVector<StructuredDAGPeerMovement, 32>>
derivePeerMovements(const TargetTopology &topology, CardId cardId,
                    const StructuredDAGAnalysis &dag,
                    const StructuredDAGEdgeStrategyPlan &plan) {
  llvm::SmallVector<StructuredDAGPeerMovement, 32> movements;
  for (const SpatialEdgeStrategy &strategy : plan.strategies) {
    std::optional<StructuredDAGEdgeID> edge =
        resolveStrategyEdge(dag, strategy);
    if (!edge)
      return std::nullopt;
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      if (fragment.kind != SpatialEdgeFragmentKind::Peer)
        continue;
      if (fragment.bytes == 0 ||
          fragment.sourceTile == strategy.destinationTile)
        return std::nullopt;
      mlir::FailureOr<llvm::SmallVector<TileLink, 8>> route =
          topology.getCanonicalOnCardPath(cardId, fragment.sourceTile,
                                          strategy.destinationTile);
      if (mlir::failed(route) || route->empty())
        return std::nullopt;
      movements.push_back(StructuredDAGPeerMovement{
          *edge, fragment.sourceTile, strategy.destinationTile, fragment.bytes,
          /*bufferCount=*/1, std::move(*route)});
    }
  }
  return movements;
}

uint64_t
getTopologyHopByteWork(llvm::ArrayRef<StructuredDAGPeerMovement> movements) {
  uint64_t work = 0;
  for (const StructuredDAGPeerMovement &movement : movements)
    work = saturatingAdd(
        work, saturatingMultiply(movement.duration, movement.route.size()));
  return work;
}

/// Canonical resource-graph identity for a complete placement.  Tile
/// numbers are assigned query-local canonical labels in first semantic-use
/// order.  Edge domains, layout facts, fragments and every routed link remain
/// in the key, so two states compare equal only when the scheduler and later
/// physical materialization see the same structure modulo a consistent
/// renaming of otherwise interchangeable Tile identities.
static std::vector<uint64_t> getResourceRenamingSignature(
    llvm::ArrayRef<StructuredDAGNodePlacement> placements,
    llvm::ArrayRef<StructuredDAGObservablePlacement> outputs,
    const StructuredDAGEdgeStrategyPlan &edgePlan,
    llvm::ArrayRef<StructuredDAGPeerMovement> movements) {
  std::map<int64_t, uint64_t> labels;
  auto label = [&](TileId tile) {
    auto [found, inserted] = labels.try_emplace(tile.getValue(), labels.size());
    (void)inserted;
    return found->second;
  };
  std::vector<uint64_t> signature;
  auto appendSigned = [&](int64_t value) {
    signature.push_back(static_cast<uint64_t>(value));
  };
  signature.push_back(placements.size());
  for (const StructuredDAGNodePlacement &placement : placements) {
    signature.push_back(placement.node);
    signature.push_back(placement.spatialIteratorDimension);
    signature.push_back(placement.shardDimension);
    signature.push_back(placement.iteratorPartitionFactors.size());
    for (uint32_t factor : placement.iteratorPartitionFactors)
      signature.push_back(factor);
    signature.push_back(placement.tiles.size());
    for (TileId tile : placement.tiles)
      signature.push_back(label(tile));
  }
  signature.push_back(outputs.size());
  for (const StructuredDAGObservablePlacement &output : outputs) {
    signature.push_back(output.outputIndex);
    signature.push_back(output.shardDimension);
    signature.push_back(output.tiles.size());
    for (TileId tile : output.tiles)
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
  for (const StructuredDAGPeerMovement &movement : movements) {
    signature.push_back(movement.edge);
    signature.push_back(label(movement.sourceTile));
    signature.push_back(label(movement.destinationTile));
    signature.push_back(movement.duration);
    signature.push_back(movement.bufferCount);
    signature.push_back(movement.route.size());
    for (const TileLink &link : movement.route) {
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

bool sameCandidatePlacement(const StructuredDAGPlacementCandidate &lhs,
                            const StructuredDAGPlacementCandidate &rhs) {
  if (lhs.nodePlacements.size() != rhs.nodePlacements.size())
    return false;
  return llvm::all_of(
      llvm::zip_equal(lhs.nodePlacements, rhs.nodePlacements), [](auto values) {
        auto [left, right] = values;
        return left.node == right.node && samePlacement(left, right);
      });
}

bool placementVectorLess(llvm::ArrayRef<StructuredDAGNodePlacement> lhs,
                         llvm::ArrayRef<StructuredDAGNodePlacement> rhs) {
  for (auto [left, right] : llvm::zip_equal(lhs, rhs)) {
    if (samePlacement(left, right))
      continue;
    return placementLess(left, right);
  }
  return lhs.size() < rhs.size();
}

bool candidateLess(const StructuredDAGPlacementCandidate &lhs,
                   const StructuredDAGPlacementCandidate &rhs) {
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

mlir::FailureOr<StructuredDAGPlacementSearchDomain>
deriveStructuredDAGPlacementSearchDomain(const StructuredDAGAnalysis &dag,
                                         const TargetTopology &topology,
                                         CardId cardId,
                                         std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (dag.getNodes().empty()) {
    setFailure(failureReason, "structured-DAG placement domain requires a DAG");
    return mlir::failure();
  }
  std::optional<llvm::ArrayRef<TileId>> availableTiles =
      topology.getAvailableTileIds(cardId);
  if (!availableTiles || availableTiles->empty()) {
    setFailure(failureReason,
               "structured-DAG placement domain requires available Tiles");
    return mlir::failure();
  }
  llvm::SmallVector<TopologyTileGroup, 128> groups =
      deriveTopologyGroups(topology, cardId, *availableTiles);
  if (groups.empty()) {
    setFailure(failureReason,
               "structured-DAG placement domain has no connected Tile group");
    return mlir::failure();
  }
  StructuredDAGPlacementSearchDomain domain;
  domain.placementGroupCount = groups.size();
  domain.nodeOptions.reserve(dag.getNodes().size());
  for (const StructuredDAGNode &node : dag.getNodes()) {
    llvm::SmallVector<StructuredDAGNodePlacement, 32> options =
        deriveNodePlacementDomain(node, groups);
    if (options.empty()) {
      setFailure(failureReason,
                 "structured-DAG placement domain cannot place a DAG node");
      return mlir::failure();
    }
    domain.nodeOptions.push_back(std::move(options));
  }
  return domain;
}

class StructuredDAGPlacementEvaluator::Impl {
public:
  Impl(const StructuredDAGAnalysis &dag, const TargetTopology &topology,
       CardId cardId, analysis::IREpoch epoch)
      : dag(dag), topology(topology), cardId(cardId), demandQuery(dag, epoch) {}

  const StructuredDAGAnalysis &dag;
  const TargetTopology &topology;
  CardId cardId;
  StructuredDAGExactDemandQuery demandQuery;

  struct EdgePlanEntry {
    StructuredDAGEdgeID edge = 0;
    StructuredDAGNodePlacement producer;
    StructuredDAGNodePlacement consumer;
    analysis::ExactDemandResult demand;
    StructuredDAGEdgeStrategyPlan plan;
    bool carrierComplete = false;
    std::string failureReason;
  };

  /// Per-edge memo: the typed demand verdict decides legality; the canonical
  /// carrier plan remains a modeling input for movements, residency,
  /// schedule and cost, and its failure never deletes the placement.
  const EdgePlanEntry *
  getEdgeEntry(const StructuredDAGEdge &edge,
               const StructuredDAGNodePlacement &producer,
               const StructuredDAGNodePlacement &consumer) {
    const size_t hash = static_cast<size_t>(llvm::hash_combine(
        edge.id, getPlacementHash(producer), getPlacementHash(consumer)));
    llvm::SmallVector<EdgePlanEntry, 1> &bucket = edgePlans[hash];
    auto found = llvm::find_if(bucket, [&](const EdgePlanEntry &entry) {
      return entry.edge == edge.id && samePlacement(entry.producer, producer) &&
             samePlacement(entry.consumer, consumer);
    });
    if (found != bucket.end())
      return &*found;

    EdgePlanEntry entry;
    entry.edge = edge.id;
    entry.producer = producer;
    entry.consumer = consumer;
    std::string trialFailure;
    mlir::FailureOr<analysis::LogicalShardTrial> trial = buildEdgeShardTrial(
        dag, producer, consumer, demandQuery.getEpoch(), &trialFailure);
    if (mlir::failed(trial)) {
      entry.demand.status = analysis::ExactDemandStatus::IndeterminateFailure;
      entry.demand.detail = std::move(trialFailure);
    } else {
      entry.demand = demandQuery.query(edge.id, *trial);
    }
    if (entry.demand.status == analysis::ExactDemandStatus::Satisfied) {
      // The carrier plan is assembled from the same typed verdict and trial:
      // the logical boundary is derived exactly once per placement pair.
      StructuredDAGEdgeDemandPlan demandPlan;
      if (mlir::succeeded(assembleStructuredDAGEdgeDemandPlan(
              dag, edge, producer, consumer, *trial, entry.demand, &demandPlan,
              &entry.failureReason))) {
        mlir::FailureOr<StructuredDAGEdgeStrategyPlan> plan =
            lowerStructuredDAGEdgeDemandPlanToCanonicalStrategies(
                dag, demandPlan, &entry.failureReason);
        if (mlir::succeeded(plan)) {
          entry.plan = std::move(*plan);
          entry.carrierComplete = true;
        }
      }
    }
    bucket.push_back(std::move(entry));
    return &bucket.back();
  }

private:
  std::unordered_map<size_t, llvm::SmallVector<EdgePlanEntry, 1>> edgePlans;
};

StructuredDAGPlacementEvaluator::StructuredDAGPlacementEvaluator(
    const StructuredDAGAnalysis &dag, const TargetTopology &topology,
    CardId cardId, analysis::IREpoch epoch)
    : impl(std::make_unique<Impl>(dag, topology, cardId, epoch)) {}

StructuredDAGPlacementEvaluator::~StructuredDAGPlacementEvaluator() = default;
StructuredDAGPlacementEvaluator::StructuredDAGPlacementEvaluator(
    StructuredDAGPlacementEvaluator &&) noexcept = default;
StructuredDAGPlacementEvaluator &StructuredDAGPlacementEvaluator::operator=(
    StructuredDAGPlacementEvaluator &&) noexcept = default;

mlir::FailureOr<StructuredDAGPlacementCandidate>
StructuredDAGPlacementEvaluator::evaluate(
    llvm::ArrayRef<StructuredDAGNodePlacement> requestedPlacements,
    std::string *failureReason, StructuredDAGPlacementLegality *legality) {
  if (failureReason)
    failureReason->clear();
  auto reportIndeterminate = [&](llvm::StringRef detail) {
    if (legality) {
      legality->status = analysis::ExactDemandStatus::IndeterminateFailure;
      legality->detail = detail.str();
    }
  };
  const StructuredDAGAnalysis &dag = impl->dag;
  const TargetTopology &topology = impl->topology;
  const CardId cardId = impl->cardId;
  if (requestedPlacements.size() != dag.getNodes().size()) {
    setFailure(failureReason,
               "structured-DAG placement evaluation requires every DAG node");
    reportIndeterminate("structured-DAG placement evaluation requires every "
                        "DAG node");
    return mlir::failure();
  }
  std::optional<llvm::ArrayRef<TileId>> availableTiles =
      topology.getAvailableTileIds(cardId);
  if (!availableTiles || availableTiles->empty()) {
    setFailure(failureReason,
               "structured-DAG placement evaluation requires available Tiles");
    reportIndeterminate(
        "structured-DAG placement evaluation requires available Tiles");
    return mlir::failure();
  }

  PartialPlacementState state;
  for (auto [node, placement] :
       llvm::zip_equal(dag.getNodes(), requestedPlacements)) {
    if (placement.node != node.id || placement.tiles.empty()) {
      setFailure(
          failureReason,
          "structured-DAG placement evaluation has an invalid node option");
      reportIndeterminate(
          "structured-DAG placement evaluation has an invalid node option");
      return mlir::failure();
    }
    state = extendState(dag, topology, cardId, state, placement);
  }
  llvm::SmallVector<StructuredDAGNodePlacement, 16> placements =
      materializePlacements(state);
  mlir::FailureOr<llvm::SmallVector<StructuredDAGObservablePlacement, 4>>
      outputs = deriveObservablePlacements(dag, placements);
  if (mlir::failed(outputs)) {
    setFailure(failureReason,
               "structured-DAG placement has inconsistent observable roots");
    reportIndeterminate(
        "structured-DAG placement has inconsistent observable roots");
    return mlir::failure();
  }
  // Typed logical gate per edge; the placement-independent relation proof is
  // reused across the complete factorized query.  The per-placement fragment
  // carrier is rebuilt here for modeling only: its resident/peer split,
  // payload domains and Tile endpoints are observable by routing, residency
  // and actual materialization, but a carrier failure never deletes the
  // placement.
  StructuredDAGEdgeStrategyPlan edgePlan;
  bool carrierComplete = true;
  for (const StructuredDAGEdge &edge : dag.getEdges()) {
    const Impl::EdgePlanEntry *entry = impl->getEdgeEntry(
        edge, placements[edge.producer], placements[edge.consumer]);
    if (entry->demand.status != analysis::ExactDemandStatus::Satisfied) {
      if (legality) {
        legality->status = entry->demand.status;
        legality->detail = entry->demand.detail;
      }
      setFailure(failureReason,
                 entry->demand.detail.empty()
                     ? "structured-DAG placement edge demand is not provable"
                     : entry->demand.detail);
      return mlir::failure();
    }
    if (!entry->carrierComplete) {
      carrierComplete = false;
      continue;
    }
    if (entry->plan.totalPeerBytes >
        std::numeric_limits<uint64_t>::max() - edgePlan.totalPeerBytes) {
      setFailure(failureReason, "structured-DAG placement peer bytes overflow");
      return mlir::failure();
    }
    edgePlan.totalPeerBytes += entry->plan.totalPeerBytes;
    edgePlan.strategies.append(entry->plan.strategies.begin(),
                               entry->plan.strategies.end());
  }
  std::optional<llvm::SmallVector<StructuredDAGPeerMovement, 32>> movements =
      derivePeerMovements(topology, cardId, dag, edgePlan);
  std::optional<llvm::SmallVector<StructuredDAGLocalResidency, 32>>
      residencies = derivePlacementResidencies(dag, edgePlan);
  if (!movements || !residencies) {
    setFailure(failureReason,
               "structured-DAG placement cannot derive exact edge resources");
    reportIndeterminate(
        "structured-DAG placement cannot derive exact edge resources");
    return mlir::failure();
  }
  mlir::FailureOr<StructuredDAGCandidateSchedule> schedule =
      scheduleStructuredDAGCandidate(dag, *availableTiles, placements,
                                     *residencies, failureReason, *movements,
                                     /*localMovements=*/{},
                                     /*enforceSPMCapacity=*/false);
  if (mlir::failed(schedule)) {
    reportIndeterminate(
        "structured-DAG placement cannot schedule the candidate");
    return mlir::failure();
  }

  StructuredDAGPlacementCandidate candidate;
  candidate.nodePlacements = schedule->nodePlacements;
  candidate.outputPlacements = std::move(*outputs);
  candidate.schedule = std::move(*schedule);
  candidate.edgePlan = std::move(edgePlan);
  candidate.edgeCarrierComplete = carrierComplete;
  candidate.topologyHopByteWork = getTopologyHopByteWork(*movements);
  candidate.topologyCompactnessWork = state.topologyCompactnessWork;
  candidate.distinctTileGroupCount = state.distinctGroups;
  candidate.parallelComponentCount = state.parallelComponents;
  candidate.localEdgeCount = state.localEdges;
  candidate.sameGroupRemapEdgeCount = state.sameGroupRemapEdges;
  candidate.partialOverlapEdgeCount = state.partialEdges;
  candidate.disjointEdgeCount = state.disjointEdges;
  if (legality) {
    legality->status = analysis::ExactDemandStatus::Satisfied;
    legality->detail.clear();
  }
  return candidate;
}

mlir::FailureOr<StructuredDAGPlacementCandidate> evaluateStructuredDAGPlacement(
    const StructuredDAGAnalysis &dag, const TargetTopology &topology,
    CardId cardId,
    llvm::ArrayRef<StructuredDAGNodePlacement> requestedPlacements,
    std::string *failureReason, StructuredDAGPlacementLegality *legality) {
  StructuredDAGPlacementEvaluator evaluator(dag, topology, cardId);
  return evaluator.evaluate(requestedPlacements, failureReason, legality);
}

mlir::FailureOr<llvm::SmallVector<StructuredDAGPlacementCandidate, 12>>
enumerateStructuredDAGPlacements(
    const StructuredDAGAnalysis &dag, const TargetTopology &topology,
    CardId cardId, StructuredDAGPlacementEnumerationStatistics *statistics,
    std::string *failureReason, StructuredDAGPlacementLegality *legality) {
  if (failureReason)
    failureReason->clear();
  StructuredDAGPlacementEnumerationStatistics localStatistics;
  StructuredDAGPlacementEnumerationStatistics &resultStatistics =
      statistics ? *statistics : localStatistics;
  resultStatistics = {};
  if (dag.getNodes().empty()) {
    setFailure(failureReason,
               "structured-DAG placement enumeration requires a DAG");
    return mlir::failure();
  }
  std::optional<llvm::ArrayRef<TileId>> availableTiles =
      topology.getAvailableTileIds(cardId);
  if (!availableTiles || availableTiles->empty()) {
    setFailure(failureReason,
               "structured-DAG placement enumeration requires available Tiles");
    return mlir::failure();
  }
  if (dag.getObservableOutputRootNodes().size() !=
          dag.getFunction().getNumResults() ||
      llvm::any_of(dag.getObservableOutputRootNodes(),
                   [](llvm::ArrayRef<StructuredDAGNodeID> roots) {
                     return roots.empty();
                   }))
    return llvm::SmallVector<StructuredDAGPlacementCandidate, 12>{};

  llvm::SmallVector<TopologyTileGroup, 128> groups =
      deriveTopologyGroups(topology, cardId, *availableTiles);
  resultStatistics.placementGroupCount = groups.size();
  if (groups.empty()) {
    setFailure(
        failureReason,
        "structured-DAG placement enumeration has no connected Tile group");
    return mlir::failure();
  }

  llvm::SmallVector<PartialPlacementState, 24> retainedStates(1);
  EdgeTransitionLegalityCache transitionCache(dag, analysis::IREpoch::mint());
  for (const StructuredDAGNode &node : dag.getNodes()) {
    resultStatistics.expandedStates =
        saturatingAdd(resultStatistics.expandedStates, retainedStates.size());
    llvm::SmallVector<PartialPlacementState, 128> expanded;
    for (const PartialPlacementState &state : retainedStates) {
      StructuredDAGPlacementLegality optionLegality;
      llvm::SmallVector<StructuredDAGNodePlacement, 32> options =
          deriveNodeOptions(dag, node, state, groups, transitionCache,
                            resultStatistics, &optionLegality);
      if (optionLegality.status != analysis::ExactDemandStatus::Satisfied) {
        if (legality)
          *legality = std::move(optionLegality);
        setFailure(failureReason,
                   optionLegality.detail.empty()
                       ? "structured-DAG placement enumeration hit an "
                         "unsupported or indeterminate edge demand"
                       : optionLegality.detail);
        return mlir::failure();
      }
      for (StructuredDAGNodePlacement &option : options)
        expanded.push_back(
            extendState(dag, topology, cardId, state, std::move(option)));
    }
    if (expanded.empty()) {
      setFailure(
          failureReason,
          "structured-DAG placement enumeration cannot place a DAG node");
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
    for (const StructuredDAGNode &node : dag.getNodes()) {
      std::optional<llvm::SmallVector<StaticSpatialAxis, 4>> axes =
          getNodeSpatialAxes(node);
      if (!axes || axes->empty()) {
        complete = false;
        break;
      }
      const StaticSpatialAxis &axis = axes->front();
      StructuredDAGNodePlacement placement;
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

  llvm::SmallVector<StructuredDAGPlacementCandidate, 24> accepted;
  std::unordered_map<size_t, llvm::SmallVector<size_t, 1>>
      resourceRenamingClasses;
  StructuredDAGExactDemandQuery demandQuery(dag, analysis::IREpoch::mint());
  for (PartialPlacementState &state : retainedStates) {
    llvm::SmallVector<StructuredDAGNodePlacement, 16> statePlacements =
        materializePlacements(state);
    mlir::FailureOr<llvm::SmallVector<StructuredDAGObservablePlacement, 4>>
        outputs = deriveObservablePlacements(dag, statePlacements);
    if (mlir::failed(outputs)) {
      resultStatistics.rejectedTransitions =
          saturatingAdd(resultStatistics.rejectedTransitions, 1);
      continue;
    }
    // Typed logical gate: only a proven partition/relation/ownership
    // contradiction rejects the trial; unsupported semantics and
    // indeterminate failures stop the whole enumeration path.
    std::string trialFailure;
    mlir::FailureOr<analysis::LogicalShardTrial> trial = buildLogicalShardTrial(
        dag, statePlacements, demandQuery.getEpoch(), &trialFailure);
    if (mlir::failed(trial)) {
      if (legality) {
        legality->status = analysis::ExactDemandStatus::IndeterminateFailure;
        legality->detail = std::move(trialFailure);
      }
      setFailure(failureReason, trialFailure);
      return mlir::failure();
    }
    bool logicallyInfeasible = false;
    llvm::SmallVector<analysis::ExactDemandResult, 8> edgeDemands;
    edgeDemands.reserve(dag.getEdges().size());
    for (const StructuredDAGEdge &edge : dag.getEdges()) {
      analysis::ExactDemandResult demand = demandQuery.query(edge.id, *trial);
      if (demand.status == analysis::ExactDemandStatus::Satisfied) {
        edgeDemands.push_back(std::move(demand));
        continue;
      }
      if (demand.status ==
          analysis::ExactDemandStatus::ProvenLogicalInfeasible) {
        logicallyInfeasible = true;
        break;
      }
      if (legality) {
        legality->status = demand.status;
        legality->detail = std::move(demand.detail);
      }
      setFailure(failureReason,
                 "structured-DAG placement enumeration hit an unsupported or "
                 "indeterminate edge demand");
      return mlir::failure();
    }
    if (logicallyInfeasible) {
      resultStatistics.rejectedTransitions =
          saturatingAdd(resultStatistics.rejectedTransitions, 1);
      continue;
    }
    // The canonical carrier remains a modeling input for movements,
    // residency, schedule and cost; its failure is a physical-assignment
    // gap, never a placement rejection. It is assembled from the same
    // typed trial and verdicts: the logical boundary is derived exactly
    // once per complete placement.
    StructuredDAGEdgeDemandPlan demandPlan;
    bool carrierAssembled = true;
    for (auto [edge, demand] : llvm::zip_equal(dag.getEdges(), edgeDemands)) {
      if (mlir::failed(assembleStructuredDAGEdgeDemandPlan(
              dag, edge, statePlacements[edge.producer],
              statePlacements[edge.consumer], *trial, demand, &demandPlan,
              /*failureReason=*/nullptr))) {
        carrierAssembled = false;
        break;
      }
    }
    std::string ignoredFailure;
    mlir::FailureOr<StructuredDAGEdgeStrategyPlan> edgePlan;
    if (carrierAssembled)
      edgePlan = lowerStructuredDAGEdgeDemandPlanToCanonicalStrategies(
          dag, demandPlan, &ignoredFailure);
    if (!carrierAssembled || mlir::failed(edgePlan)) {
      resultStatistics.edgeCarrierIncompleteTransitions =
          saturatingAdd(resultStatistics.edgeCarrierIncompleteTransitions, 1);
      continue;
    }
    std::optional<llvm::SmallVector<StructuredDAGPeerMovement, 32>> movements =
        derivePeerMovements(topology, cardId, dag, *edgePlan);
    std::optional<llvm::SmallVector<StructuredDAGLocalResidency, 32>>
        residencies = derivePlacementResidencies(dag, *edgePlan);
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
      const StructuredDAGPlacementCandidate &representative =
          accepted[representativeIndex];
      std::optional<llvm::SmallVector<StructuredDAGPeerMovement, 32>>
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
      const StructuredDAGPlacementCandidate &representative =
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
    mlir::FailureOr<StructuredDAGCandidateSchedule> schedule =
        scheduleStructuredDAGCandidate(dag, *availableTiles, statePlacements,
                                       *residencies, &ignoredFailure,
                                       *movements,
                                       /*localMovements=*/{},
                                       /*enforceSPMCapacity=*/false);
    if (mlir::failed(schedule)) {
      resultStatistics.rejectedTransitions =
          saturatingAdd(resultStatistics.rejectedTransitions, 1);
      continue;
    }

    StructuredDAGPlacementCandidate candidate;
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
  llvm::SmallVector<StructuredDAGPlacementCandidate, 12> result;
  auto retainFirst = [&](auto predicate) {
    auto found = llvm::find_if(accepted, predicate);
    if (found == accepted.end())
      return;
    result.push_back(std::move(*found));
    accepted.erase(found);
  };
  retainFirst([](const StructuredDAGPlacementCandidate &candidate) {
    return countPeerFragments(candidate.edgePlan) == 0;
  });
  retainFirst([](const StructuredDAGPlacementCandidate &candidate) {
    return candidate.sameGroupRemapEdgeCount != 0;
  });
  retainFirst([](const StructuredDAGPlacementCandidate &candidate) {
    return candidate.partialOverlapEdgeCount != 0;
  });
  retainFirst([](const StructuredDAGPlacementCandidate &candidate) {
    return candidate.disjointEdgeCount != 0;
  });
  retainFirst([](const StructuredDAGPlacementCandidate &candidate) {
    return candidate.parallelComponentCount > 1;
  });
  retainFirst([&](const StructuredDAGPlacementCandidate &candidate) {
    return candidate.nodePlacements.size() == dag.getNodes().size() &&
           candidate.distinctTileGroupCount == dag.getNodes().size() &&
           llvm::all_of(candidate.nodePlacements, [](const auto &placement) {
             return placement.tiles.size() == 1;
           });
  });
  retainFirst([](const StructuredDAGPlacementCandidate &candidate) {
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
