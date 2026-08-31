//===- BoundaryMovement.cpp - Close physical Tile boundaries ----------===//

#include "BoundaryMovement.h"

#include "StructuredToTile.h"
#include "Wafer/IR/Topology/TargetTopology.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/DirectDTE.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

struct InputPlan {
  unsigned index = 0;
  mlir::Value originalOperand;
  mlir::BlockArgument originalArgument;
  llvm::SmallVector<mlir::bufferization::ToMemrefOp, 2> bridges;
  std::optional<unsigned> peerRelation;
  bool unused = false;
};

struct ResultPlan {
  unsigned index = 0;
  std::optional<unsigned> outputIndex;
  mlir::OpResult originalResult;
  mlir::bufferization::ToTensorOp bridge;
  mlir::Value spmValue;
  mlir::Value ddrDestination;
  mlir::memref::SubViewOp obsoleteOutputSubview;
  mlir::bufferization::ToMemrefOp publicationBridge;
  mlir::bufferization::ToMemrefOp returnBridge;
  mlir::memref::CopyOp outputCopy;
  llvm::SmallVector<unsigned, 2> peerRelations;
  bool hasSameTileConsumer = false;
};

struct RegionPlan {
  TileRegionOp operation;
  TileModuleOp tileModule;
  llvm::SmallVector<InputPlan, 8> inputs;
  llvm::SmallVector<ResultPlan, 8> results;
};

struct PeerPlan {
  unsigned relationIndex = 0;
  TileRegionOp sourceRegion;
  unsigned sourceResult = 0;
  mlir::Value sourceSPM;
  TileRegionOp destinationRegion;
  unsigned destinationInput = 0;
  mlir::MemRefType destinationSPMType;
  llvm::SmallVector<mlir::Value, 2> destinationSPMCarriers;
  llvm::SmallVector<mlir::memref::SubViewOp, 2> destinationSubviews;
  llvm::SmallVector<int64_t, 4> sourceWindowOffsets;
  llvm::SmallVector<int64_t, 4> sourceWindowSizes;
  llvm::SmallVector<int64_t, 4> sourceWindowStrides;
  uint64_t sourceTile = 0;
  uint64_t destinationTile = 0;
  uint64_t bytes = 0;
  uint64_t transportSourceTile = 0;
  int64_t communicationId = 0;
  int64_t protocolRound = 0;
  int64_t scheduleComponent = -1;
  unsigned destinationRegionRank = 0;
  unsigned sourceTileOrder = 0;
  unsigned destinationTileOrder = 0;
  bool useSharedDDR = false;
  bool scheduledPeer = false;
  std::optional<unsigned> relaySourceRelation;
};

struct PayloadGroup {
  llvm::SmallVector<unsigned, 8> peers;
  uint64_t sourceTile = 0;
  llvm::SmallVector<uint64_t, 16> participants;
  llvm::SmallVector<mlir::Operation *, 16> regions;
};

struct CommunicationComponent {
  llvm::SmallVector<unsigned, 16> groups;
  llvm::SmallVector<uint64_t, 16> participants;
};

static mlir::Operation *findFirstBufferConsumer(mlir::Value buffer,
                                                mlir::Block &block);
static mlir::LogicalResult findLastBufferWrite(mlir::Value buffer,
                                               mlir::Block &block,
                                               StorageRootMemo &memo,
                                               mlir::Operation *&last,
                                               std::string &detail);

static bool hasSameFanoutPayload(const PeerPlan &lhs, const PeerPlan &rhs) {
  return lhs.sourceRegion == rhs.sourceRegion &&
         lhs.sourceResult == rhs.sourceResult &&
         lhs.sourceSPM == rhs.sourceSPM &&
         lhs.destinationSPMType == rhs.destinationSPMType &&
         lhs.sourceWindowOffsets == rhs.sourceWindowOffsets &&
         lhs.sourceWindowSizes == rhs.sourceWindowSizes &&
         lhs.sourceWindowStrides == rhs.sourceWindowStrides &&
         lhs.bytes == rhs.bytes;
}

static mlir::Operation *getOperation(TileRegionOp operation) {
  return operation.getOperation();
}

struct FanoutHolder {
  uint64_t tile = 0;
  unsigned tileOrder = 0;
  TileRegionOp region;
  std::optional<unsigned> peerIndex;
  uint64_t sends = 0;
};

struct FanoutCandidate {
  unsigned holderIndex = 0;
  unsigned peerIndex = 0;
  uint64_t senderLoad = 0;
  uint64_t hops = 0;
  uint64_t sourceTile = 0;
  uint64_t destinationTile = 0;
  unsigned destinationOrder = 0;
};

static llvm::SmallVector<PayloadGroup, 16>
buildPayloadGroups(llvm::ArrayRef<PeerPlan> peers) {
  llvm::SmallVector<PayloadGroup, 16> groups;
  for (auto [peerIndex, peer] : llvm::enumerate(peers)) {
    auto group = llvm::find_if(groups, [&](const PayloadGroup &candidate) {
      return !candidate.peers.empty() &&
             hasSameFanoutPayload(peers[candidate.peers.front()], peer);
    });
    if (group == groups.end()) {
      PayloadGroup created;
      created.peers.push_back(static_cast<unsigned>(peerIndex));
      created.sourceTile = peer.sourceTile;
      created.participants.push_back(peer.sourceTile);
      created.participants.push_back(peer.destinationTile);
      created.regions.push_back(getOperation(peer.sourceRegion));
      created.regions.push_back(getOperation(peer.destinationRegion));
      groups.push_back(std::move(created));
      continue;
    }
    group->peers.push_back(static_cast<unsigned>(peerIndex));
    if (!llvm::is_contained(group->participants, peer.destinationTile))
      group->participants.push_back(peer.destinationTile);
    if (!llvm::is_contained(group->regions,
                            getOperation(peer.destinationRegion)))
      group->regions.push_back(getOperation(peer.destinationRegion));
  }
  for (PayloadGroup &group : groups)
    llvm::sort(group.participants);
  return groups;
}

static llvm::SmallVector<CommunicationComponent, 8>
buildCommunicationComponents(llvm::ArrayRef<PayloadGroup> groups,
                             llvm::ArrayRef<PeerPlan> peers) {
  (void)peers;
  llvm::SmallVector<unsigned, 16> parents(groups.size());
  for (auto [index, parent] : llvm::enumerate(parents))
    parent = static_cast<unsigned>(index);
  std::function<unsigned(unsigned)> find = [&](unsigned index) {
    if (parents[index] != index)
      parents[index] = find(parents[index]);
    return parents[index];
  };
  auto unite = [&](unsigned lhs, unsigned rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (rhs < lhs)
      std::swap(lhs, rhs);
    parents[rhs] = lhs;
  };
  for (unsigned lhs = 0; lhs < groups.size(); ++lhs)
    for (unsigned rhs = lhs + 1; rhs < groups.size(); ++rhs) {
      bool sharesRegion =
          llvm::any_of(groups[lhs].regions, [&](mlir::Operation *region) {
            return llvm::is_contained(groups[rhs].regions, region);
          });
      if (sharesRegion)
        unite(lhs, rhs);
    }

  llvm::SmallVector<CommunicationComponent, 8> components;
  llvm::DenseMap<unsigned, unsigned> componentByRoot;
  for (unsigned group = 0; group < groups.size(); ++group) {
    unsigned root = find(group);
    auto found = componentByRoot.find(root);
    if (found == componentByRoot.end()) {
      componentByRoot.try_emplace(root,
                                  static_cast<unsigned>(components.size()));
      components.emplace_back();
      found = componentByRoot.find(root);
    }
    CommunicationComponent &component = components[found->second];
    component.groups.push_back(group);
    for (uint64_t tile : groups[group].participants)
      if (!llvm::is_contained(component.participants, tile))
        component.participants.push_back(tile);
  }
  for (CommunicationComponent &component : components) {
    llvm::sort(component.groups);
    llvm::sort(component.participants);
  }
  return components;
}

static bool componentUsesSharedDDR(const CommunicationComponent &component,
                                   llvm::ArrayRef<PayloadGroup> groups,
                                   llvm::ArrayRef<PeerPlan> peers) {
  return llvm::all_of(component.groups, [&](unsigned groupIndex) {
    return llvm::all_of(groups[groupIndex].peers, [&](unsigned peerIndex) {
      return peers[peerIndex].useSharedDDR;
    });
  });
}

static mlir::LogicalResult breakCrossComponentRegionOrderCycles(
    llvm::ArrayRef<CommunicationComponent> components,
    llvm::ArrayRef<PayloadGroup> groups, llvm::MutableArrayRef<PeerPlan> peers,
    BoundaryMovementStatistics &statistics) {
  const unsigned count = static_cast<unsigned>(components.size());
  llvm::SmallVector<llvm::DenseSet<unsigned>, 8> successors(count);
  llvm::SmallVector<llvm::SmallVector<TileRegionOp, 8>, 8> regions(count);
  for (auto [componentIndex, component] : llvm::enumerate(components))
    for (unsigned groupIndex : component.groups)
      for (unsigned peerIndex : groups[groupIndex].peers)
        for (TileRegionOp region : {peers[peerIndex].sourceRegion,
                                    peers[peerIndex].destinationRegion})
          if (!llvm::is_contained(regions[componentIndex], region))
            regions[componentIndex].push_back(region);

  for (unsigned lhs = 0; lhs < count; ++lhs)
    for (unsigned rhs = lhs + 1; rhs < count; ++rhs)
      for (TileRegionOp lhsRegion : regions[lhs])
        for (TileRegionOp rhsRegion : regions[rhs]) {
          auto lhsTile = lhsRegion->getParentOfType<TileModuleOp>();
          auto rhsTile = rhsRegion->getParentOfType<TileModuleOp>();
          if (!lhsTile || lhsTile != rhsTile || lhsRegion == rhsRegion)
            continue;
          if (lhsRegion->getBlock() != rhsRegion->getBlock()) {
            // There is no current-IR ordering proof across different entry
            // blocks on one Tile. Treat both phase orders as possible so one
            // component is materialized through an explicit DDR boundary.
            successors[lhs].insert(rhs);
            successors[rhs].insert(lhs);
            continue;
          }
          (lhsRegion->isBeforeInBlock(rhsRegion) ? successors[lhs]
                                                 : successors[rhs])
              .insert(lhsRegion->isBeforeInBlock(rhsRegion) ? rhs : lhs);
        }

  auto findCycle = [&]() -> llvm::SmallVector<unsigned, 8> {
    llvm::SmallVector<uint8_t, 8> state(count, 0);
    llvm::SmallVector<unsigned, 8> stack;
    llvm::SmallVector<int64_t, 8> position(count, -1);
    llvm::SmallVector<unsigned, 8> cycle;
    std::function<bool(unsigned)> visit = [&](unsigned node) {
      state[node] = 1;
      position[node] = static_cast<int64_t>(stack.size());
      stack.push_back(node);
      llvm::SmallVector<unsigned, 8> ordered(successors[node].begin(),
                                             successors[node].end());
      llvm::sort(ordered);
      for (unsigned next : ordered) {
        if (componentUsesSharedDDR(components[next], groups, peers))
          continue;
        if (state[next] == 0) {
          if (visit(next))
            return true;
          continue;
        }
        if (state[next] == 1) {
          cycle.append(stack.begin() + position[next], stack.end());
          return true;
        }
      }
      stack.pop_back();
      position[node] = -1;
      state[node] = 2;
      return false;
    };
    for (unsigned node = 0; node < count; ++node)
      if (!componentUsesSharedDDR(components[node], groups, peers) &&
          state[node] == 0 && visit(node))
        break;
    return cycle;
  };

  while (true) {
    llvm::SmallVector<unsigned, 8> cycle = findCycle();
    if (cycle.empty())
      return mlir::success();
    auto bytes = [&](unsigned componentIndex) {
      uint64_t total = 0;
      for (unsigned groupIndex : components[componentIndex].groups)
        for (unsigned peerIndex : groups[groupIndex].peers)
          if (peers[peerIndex].bytes >
              std::numeric_limits<uint64_t>::max() - total)
            return std::numeric_limits<uint64_t>::max();
          else
            total += peers[peerIndex].bytes;
      return total;
    };
    unsigned selected =
        *llvm::min_element(cycle, [&](unsigned lhs, unsigned rhs) {
          return std::make_tuple(bytes(lhs), lhs) <
                 std::make_tuple(bytes(rhs), rhs);
        });
    for (unsigned groupIndex : components[selected].groups)
      for (unsigned peerIndex : groups[groupIndex].peers)
        peers[peerIndex].useSharedDDR = true;
    ++statistics.noCutDDRComponents;
  }
}

static mlir::FailureOr<llvm::SmallVector<uint64_t, 16>>
buildMinimumHopRing(const TargetTopology &topology,
                    llvm::ArrayRef<uint64_t> participants) {
  if (participants.size() < 2 || participants.size() > 16)
    return mlir::failure();
  const size_t count = participants.size();
  std::vector<std::vector<uint64_t>> distances(count,
                                               std::vector<uint64_t>(count, 0));
  for (size_t lhs = 0; lhs < count; ++lhs)
    for (size_t rhs = 0; rhs < count; ++rhs) {
      if (lhs == rhs)
        continue;
      std::optional<uint64_t> distance = topology.getOnCardShortestHopDistance(
          CardId(0), TileId(participants[lhs]), TileId(participants[rhs]));
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

static mlir::LogicalResult buildRegionTopologicalRanks(
    mlir::ModuleOp module, llvm::MutableArrayRef<PeerPlan> peers,
    llvm::DenseMap<mlir::Operation *, unsigned> &ranks, std::string &detail) {
  struct RegionNode {
    TileRegionOp region;
    uint64_t tile = 0;
    unsigned tileOrder = 0;
  };
  llvm::SmallVector<RegionNode, 64> nodes;
  for (TileModuleOp tile : module.getOps<TileModuleOp>()) {
    unsigned tileOrder = 0;
    tile.walk([&](TileRegionOp region) {
      nodes.push_back(RegionNode{
          region, static_cast<uint64_t>(tile.getTileIdAttr().getInt()),
          tileOrder++});
    });
  }
  llvm::sort(nodes, [](const RegionNode &lhs, const RegionNode &rhs) {
    return std::tie(lhs.tile, lhs.tileOrder) <
           std::tie(rhs.tile, rhs.tileOrder);
  });
  llvm::DenseMap<mlir::Operation *, unsigned> nodeIndices;
  for (auto [index, node] : llvm::enumerate(nodes))
    nodeIndices.try_emplace(node.region.getOperation(),
                            static_cast<unsigned>(index));

  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 64> successors(
      nodes.size());
  llvm::SmallVector<unsigned, 64> indegrees(nodes.size(), 0);
  llvm::DenseSet<uint64_t> edges;
  auto addEdge = [&](TileRegionOp source,
                     TileRegionOp destination) -> mlir::LogicalResult {
    auto sourceIndex = nodeIndices.find(source.getOperation());
    auto destinationIndex = nodeIndices.find(destination.getOperation());
    if (sourceIndex == nodeIndices.end() ||
        destinationIndex == nodeIndices.end() ||
        sourceIndex->second == destinationIndex->second)
      return mlir::failure();
    uint64_t key = (static_cast<uint64_t>(sourceIndex->second) << 32) |
                   destinationIndex->second;
    if (!edges.insert(key).second)
      return mlir::success();
    successors[sourceIndex->second].push_back(destinationIndex->second);
    ++indegrees[destinationIndex->second];
    return mlir::success();
  };

  for (size_t index = 1; index < nodes.size(); ++index)
    if (nodes[index - 1].tile == nodes[index].tile &&
        mlir::failed(addEdge(nodes[index - 1].region, nodes[index].region))) {
      detail = "same-Tile Region execution order is malformed";
      return mlir::failure();
    }

  for (PeerPlan &peer : peers) {
    if (peer.useSharedDDR)
      continue;
    peer.sourceTileOrder = 0;
    peer.destinationTileOrder = static_cast<unsigned>(peer.destinationTile + 1);
    if (peer.scheduledPeer)
      continue;
    if (mlir::failed(addEdge(peer.sourceRegion, peer.destinationRegion))) {
      detail = "cross-Tile relation cannot be placed in the current Region "
               "dependency graph";
      return mlir::failure();
    }
  }

  llvm::SmallVector<unsigned, 64> ready;
  for (auto [index, indegree] : llvm::enumerate(indegrees))
    if (indegree == 0)
      ready.push_back(static_cast<unsigned>(index));
  unsigned nextRank = 0;
  while (!ready.empty()) {
    auto next = llvm::min_element(ready, [&](unsigned lhs, unsigned rhs) {
      return std::tie(nodes[lhs].tile, nodes[lhs].tileOrder) <
             std::tie(nodes[rhs].tile, nodes[rhs].tileOrder);
    });
    unsigned nodeIndex = *next;
    ready.erase(next);
    ranks.try_emplace(nodes[nodeIndex].region.getOperation(), nextRank++);
    llvm::sort(successors[nodeIndex]);
    for (unsigned successor : successors[nodeIndex])
      if (--indegrees[successor] == 0)
        ready.push_back(successor);
  }
  if (ranks.size() != nodes.size()) {
    detail = "current cross-Tile relations and same-Tile Region order form a "
             "cycle before communication materialization";
    return mlir::failure();
  }
  return mlir::success();
}

static std::optional<unsigned>
getCompleteExchangeLaneCount(const CommunicationComponent &component,
                             llvm::ArrayRef<PayloadGroup> groups,
                             llvm::ArrayRef<PeerPlan> peers) {
  if (component.participants.size() < 2)
    return std::nullopt;
  llvm::DenseMap<uint64_t, unsigned> sourceCounts;
  for (unsigned groupIndex : component.groups) {
    const PayloadGroup &group = groups[groupIndex];
    if (group.participants != component.participants ||
        group.peers.size() + 1 != component.participants.size())
      return std::nullopt;
    ++sourceCounts[group.sourceTile];
    llvm::DenseSet<uint64_t> destinations;
    for (unsigned peerIndex : group.peers) {
      const PeerPlan &peer = peers[peerIndex];
      if (peer.sourceTile != group.sourceTile ||
          !destinations.insert(peer.destinationTile).second)
        return std::nullopt;
    }
    for (uint64_t participant : component.participants)
      if (participant != group.sourceTile &&
          !destinations.contains(participant))
        return std::nullopt;
  }
  unsigned lanes = 0;
  for (uint64_t participant : component.participants) {
    unsigned count = sourceCounts.lookup(participant);
    if (count == 0 || (lanes != 0 && count != lanes))
      return std::nullopt;
    lanes = count;
  }
  return lanes;
}

static bool
hasOneCommunicationRegionPerTile(const CommunicationComponent &component,
                                 llvm::ArrayRef<PayloadGroup> groups,
                                 llvm::ArrayRef<PeerPlan> peers) {
  llvm::DenseMap<uint64_t, mlir::Operation *> regionByTile;
  auto add = [&](uint64_t tile, TileRegionOp region) {
    mlir::Operation *operation = getOperation(region);
    auto [position, inserted] = regionByTile.try_emplace(tile, operation);
    return inserted || position->second == operation;
  };
  for (unsigned groupIndex : component.groups)
    for (unsigned peerIndex : groups[groupIndex].peers) {
      const PeerPlan &peer = peers[peerIndex];
      if (!add(peer.sourceTile, peer.sourceRegion) ||
          !add(peer.destinationTile, peer.destinationRegion))
        return false;
    }
  return regionByTile.size() == component.participants.size();
}

static bool hasCurrentCommunicationCut(const CommunicationComponent &component,
                                       llvm::ArrayRef<PayloadGroup> groups,
                                       llvm::ArrayRef<PeerPlan> peers) {
  if (!hasOneCommunicationRegionPerTile(component, groups, peers))
    return false;
  for (uint64_t tile : component.participants) {
    TileRegionOp region;
    for (unsigned groupIndex : component.groups)
      for (unsigned peerIndex : groups[groupIndex].peers) {
        const PeerPlan &peer = peers[peerIndex];
        if (peer.sourceTile == tile)
          region = peer.sourceRegion;
        if (peer.destinationTile == tile)
          region = peer.destinationRegion;
      }
    if (!region || !region.getBody().hasOneBlock())
      return false;
    mlir::Block &block = region.getBody().front();
    StorageRootMemo storageRoots;
    mlir::Operation *lastProducer = nullptr;
    mlir::Operation *firstConsumer = block.getTerminator();
    llvm::DenseSet<mlir::Value> sources;
    llvm::DenseSet<mlir::Value> destinations;
    std::string detail;
    for (unsigned groupIndex : component.groups)
      for (unsigned peerIndex : groups[groupIndex].peers) {
        const PeerPlan &peer = peers[peerIndex];
        if (peer.sourceTile == tile && sources.insert(peer.sourceSPM).second) {
          mlir::Operation *producer = nullptr;
          if (mlir::failed(findLastBufferWrite(peer.sourceSPM, block,
                                               storageRoots, producer, detail)))
            return false;
          if (producer &&
              (!lastProducer || lastProducer->isBeforeInBlock(producer)))
            lastProducer = producer;
        }
        if (peer.destinationTile != tile)
          continue;
        for (mlir::Value carrier : peer.destinationSPMCarriers) {
          if (!destinations.insert(carrier).second)
            continue;
          mlir::Operation *consumer = findFirstBufferConsumer(carrier, block);
          if (!consumer)
            return false;
          if (consumer->isBeforeInBlock(firstConsumer))
            firstConsumer = consumer;
        }
      }
    if (sources.empty() || destinations.empty() ||
        (lastProducer && (lastProducer == firstConsumer ||
                          firstConsumer->isBeforeInBlock(lastProducer))))
      return false;
  }
  return true;
}

static bool hasBidirectionalParticipant(const CommunicationComponent &component,
                                        llvm::ArrayRef<PayloadGroup> groups,
                                        llvm::ArrayRef<PeerPlan> peers) {
  llvm::DenseSet<uint64_t> sources;
  llvm::DenseSet<uint64_t> destinations;
  for (unsigned groupIndex : component.groups)
    for (unsigned peerIndex : groups[groupIndex].peers) {
      sources.insert(peers[peerIndex].sourceTile);
      destinations.insert(peers[peerIndex].destinationTile);
    }
  return llvm::any_of(
      sources, [&](uint64_t tile) { return destinations.contains(tile); });
}

static mlir::LogicalResult scheduleRingComponent(
    const CommunicationComponent &component,
    llvm::ArrayRef<PayloadGroup> groups, llvm::MutableArrayRef<PeerPlan> peers,
    unsigned laneCount, int64_t componentId, const TargetTopology &topology,
    BoundaryMovementStatistics &statistics, std::string &detail) {
  auto ring = buildMinimumHopRing(topology, component.participants);
  if (mlir::failed(ring)) {
    detail = "complete exchange has no bounded minimum-hop ring";
    return mlir::failure();
  }
  llvm::DenseMap<uint64_t, unsigned> ringPositions;
  for (auto [position, tile] : llvm::enumerate(*ring))
    ringPositions.try_emplace(tile, static_cast<unsigned>(position));

  llvm::DenseMap<uint64_t, llvm::SmallVector<unsigned, 4>> groupsBySource;
  for (unsigned groupIndex : component.groups)
    groupsBySource[groups[groupIndex].sourceTile].push_back(groupIndex);
  for (uint64_t source : component.participants) {
    llvm::SmallVector<unsigned, 4> &sourceGroups = groupsBySource[source];
    llvm::sort(sourceGroups, [&](unsigned lhs, unsigned rhs) {
      const PeerPlan &left = peers[groups[lhs].peers.front()];
      const PeerPlan &right = peers[groups[rhs].peers.front()];
      if (left.sourceRegion != right.sourceRegion)
        return left.sourceRegion->isBeforeInBlock(right.sourceRegion);
      return std::tie(left.sourceResult, left.relationIndex) <
             std::tie(right.sourceResult, right.relationIndex);
    });
  }
  for (unsigned lane = 0; lane < laneCount; ++lane) {
    for (uint64_t source : component.participants) {
      const PayloadGroup &group = groups[groupsBySource[source][lane]];
      unsigned sourcePosition = ringPositions.lookup(group.sourceTile);
      unsigned communication = peers[group.peers.front()].relationIndex;
      for (unsigned peerIndex : llvm::drop_begin(group.peers))
        communication = std::min(communication, peers[peerIndex].relationIndex);
      std::optional<unsigned> previousRelation;
      for (unsigned distance = 1; distance < ring->size(); ++distance) {
        uint64_t destinationTile =
            (*ring)[(sourcePosition + distance) % ring->size()];
        auto relation = llvm::find_if(group.peers, [&](unsigned peerIndex) {
          return peers[peerIndex].destinationTile == destinationTile;
        });
        if (relation == group.peers.end()) {
          detail = "complete exchange ring cannot find one destination edge";
          return mlir::failure();
        }
        PeerPlan &peer = peers[*relation];
        peer.scheduledPeer = true;
        peer.transportSourceTile =
            (*ring)[(sourcePosition + distance - 1) % ring->size()];
        peer.communicationId = communication;
        peer.protocolRound =
            static_cast<int64_t>(lane * (ring->size() - 1) + distance - 1);
        peer.scheduleComponent = componentId;
        peer.relaySourceRelation = previousRelation;
        previousRelation = peer.relationIndex;
      }
    }
  }
  ++statistics.ringComponents;
  statistics.ringRounds += laneCount * (ring->size() - 1);
  return mlir::success();
}

static mlir::LogicalResult scheduleSparseComponent(
    const CommunicationComponent &component,
    llvm::ArrayRef<PayloadGroup> groups, llvm::MutableArrayRef<PeerPlan> peers,
    int64_t componentId, const TargetTopology &topology,
    BoundaryMovementStatistics &statistics, std::string &detail) {
  llvm::SmallVector<unsigned, 64> remaining;
  for (unsigned groupIndex : component.groups)
    llvm::append_range(remaining, groups[groupIndex].peers);
  int64_t round = 0;
  while (!remaining.empty()) {
    llvm::sort(remaining, [&](unsigned lhs, unsigned rhs) {
      const PeerPlan &left = peers[lhs];
      const PeerPlan &right = peers[rhs];
      uint64_t leftHops =
          topology
              .getOnCardShortestHopDistance(CardId(0), TileId(left.sourceTile),
                                            TileId(left.destinationTile))
              .value_or(std::numeric_limits<uint64_t>::max());
      uint64_t rightHops =
          topology
              .getOnCardShortestHopDistance(CardId(0), TileId(right.sourceTile),
                                            TileId(right.destinationTile))
              .value_or(std::numeric_limits<uint64_t>::max());
      return std::tie(leftHops, left.sourceTile, left.destinationTile,
                      left.relationIndex) <
             std::tie(rightHops, right.sourceTile, right.destinationTile,
                      right.relationIndex);
    });
    llvm::DenseSet<uint64_t> usedSources;
    llvm::DenseMap<uint64_t, unsigned> receiverCounts;
    llvm::DenseSet<unsigned> selected;
    for (unsigned peerIndex : remaining) {
      PeerPlan &peer = peers[peerIndex];
      std::optional<uint64_t> hops = topology.getOnCardShortestHopDistance(
          CardId(0), TileId(peer.sourceTile), TileId(peer.destinationTile));
      if (!hops) {
        detail = "sparse exchange contains disconnected participants";
        return mlir::failure();
      }
      if (usedSources.contains(peer.sourceTile) ||
          receiverCounts.lookup(peer.destinationTile) >=
              TargetDirectDTEResourceLimits::receiverFSMsPerTile)
        continue;
      usedSources.insert(peer.sourceTile);
      ++receiverCounts[peer.destinationTile];
      selected.insert(peerIndex);
      peer.scheduledPeer = true;
      peer.transportSourceTile = peer.sourceTile;
      peer.communicationId = peer.relationIndex;
      peer.protocolRound = round;
      peer.scheduleComponent = componentId;
      peer.relaySourceRelation.reset();
    }
    if (selected.empty()) {
      detail = "sparse exchange round matching made no progress";
      return mlir::failure();
    }
    remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                                   [&](unsigned peerIndex) {
                                     return selected.contains(peerIndex);
                                   }),
                    remaining.end());
    ++round;
  }
  ++statistics.sparseRoundComponents;
  statistics.sparseRounds += static_cast<uint64_t>(round);
  return mlir::success();
}

static mlir::LogicalResult buildTopologyFanoutChoices(
    mlir::ModuleOp module, llvm::MutableArrayRef<PeerPlan> peers,
    BoundaryMovementStatistics &statistics, std::string &detail) {
  if (peers.empty())
    return mlir::success();

  llvm::SmallVector<PayloadGroup, 16> groups = buildPayloadGroups(peers);
  llvm::SmallVector<CommunicationComponent, 8> components =
      buildCommunicationComponents(groups, peers);
  std::string topologyFailure;
  mlir::FailureOr<TargetTopology> topology =
      TargetTopology::create(module, &topologyFailure);
  if (mlir::failed(topology)) {
    detail =
        "peer movement requires current target topology: " + topologyFailure;
    return mlir::failure();
  }
  if (mlir::failed(breakCrossComponentRegionOrderCycles(components, groups,
                                                        peers, statistics))) {
    detail = "communication component Region order cannot be closed";
    return mlir::failure();
  }
  for (auto [componentIndex, component] : llvm::enumerate(components)) {
    if (componentUsesSharedDDR(component, groups, peers))
      continue;
    if (component.groups.size() == 1)
      continue;
    if (std::optional<unsigned> lanes =
            getCompleteExchangeLaneCount(component, groups, peers);
        lanes && hasCurrentCommunicationCut(component, groups, peers)) {
      if (mlir::failed(
              scheduleRingComponent(component, groups, peers, *lanes,
                                    static_cast<int64_t>(componentIndex),
                                    *topology, statistics, detail)))
        return mlir::failure();
      continue;
    }
    if (hasBidirectionalParticipant(component, groups, peers) &&
        !hasCurrentCommunicationCut(component, groups, peers)) {
      for (unsigned groupIndex : component.groups)
        for (unsigned peerIndex : groups[groupIndex].peers)
          peers[peerIndex].useSharedDDR = true;
      ++statistics.noCutDDRComponents;
      continue;
    }
    if (mlir::failed(scheduleSparseComponent(
            component, groups, peers, static_cast<int64_t>(componentIndex),
            *topology, statistics, detail)))
      return mlir::failure();
  }

  llvm::DenseMap<mlir::Operation *, unsigned> regionRanks;
  if (mlir::failed(
          buildRegionTopologicalRanks(module, peers, regionRanks, detail)))
    return mlir::failure();

  for (PeerPlan &peer : peers)
    peer.destinationRegionRank =
        regionRanks.lookup(peer.destinationRegion.getOperation());

  constexpr CardId cardId(0);
  for (const PayloadGroup &payloadGroup : groups) {
    if (payloadGroup.peers.empty() ||
        peers[payloadGroup.peers.front()].scheduledPeer ||
        peers[payloadGroup.peers.front()].useSharedDDR)
      continue;
    llvm::ArrayRef<unsigned> group = payloadGroup.peers;

    const PeerPlan &root = peers[group.front()];
    llvm::DenseSet<uint64_t> destinationTiles;
    for (unsigned peerIndex : group) {
      const PeerPlan &peer = peers[peerIndex];
      if (peer.sourceTile != root.sourceTile ||
          !destinationTiles.insert(peer.destinationTile).second) {
        detail = "same-payload fanout has duplicate destination Tile "
                 "residency";
        return mlir::failure();
      }
      if (!topology->isTileAvailable(cardId, TileId(peer.sourceTile)) ||
          !topology->isTileAvailable(cardId, TileId(peer.destinationTile))) {
        detail = "same-payload fanout names an unavailable physical Tile";
        return mlir::failure();
      }
    }

    unsigned firstRelation = peers[group.front()].relationIndex;
    for (unsigned peerIndex : llvm::drop_begin(group))
      firstRelation = std::min(firstRelation, peers[peerIndex].relationIndex);
    const int64_t communicationId = static_cast<int64_t>(firstRelation);
    llvm::SmallVector<FanoutHolder, 16> holders;
    holders.push_back(FanoutHolder{root.sourceTile, root.sourceTileOrder,
                                   root.sourceRegion, std::nullopt, 0});
    llvm::SmallVector<unsigned, 16> remaining(group.begin(), group.end());
    llvm::sort(remaining, [&](unsigned lhs, unsigned rhs) {
      return peers[lhs].destinationTile < peers[rhs].destinationTile;
    });

    int64_t round = 0;
    while (!remaining.empty()) {
      llvm::SmallVector<FanoutCandidate, 64> candidates;
      for (auto [holderIndex, holder] : llvm::enumerate(holders))
        for (unsigned peerIndex : remaining) {
          const PeerPlan &destination = peers[peerIndex];
          if (holder.tileOrder >= destination.destinationTileOrder)
            continue;
          auto sourceRank = regionRanks.find(holder.region.getOperation());
          TileRegionOp destinationRegion = destination.destinationRegion;
          auto destinationRank =
              regionRanks.find(destinationRegion.getOperation());
          if (sourceRank == regionRanks.end() ||
              destinationRank == regionRanks.end() ||
              sourceRank->second >= destinationRank->second)
            continue;
          std::optional<uint64_t> hops = topology->getOnCardShortestHopDistance(
              cardId, TileId(holder.tile), TileId(destination.destinationTile));
          if (!hops)
            continue;
          candidates.push_back(FanoutCandidate{
              static_cast<unsigned>(holderIndex), peerIndex, holder.sends,
              *hops, holder.tile, destination.destinationTile,
              destination.destinationTileOrder});
        }
      llvm::sort(candidates, [](const FanoutCandidate &lhs,
                                const FanoutCandidate &rhs) {
        return std::tie(lhs.senderLoad, lhs.destinationOrder, lhs.hops,
                        lhs.sourceTile, lhs.destinationTile) <
               std::tie(rhs.senderLoad, rhs.destinationOrder, rhs.hops,
                        rhs.sourceTile, rhs.destinationTile);
      });

      llvm::DenseSet<unsigned> usedHolders;
      llvm::DenseSet<unsigned> selectedPeers;
      llvm::SmallVector<FanoutCandidate, 16> selected;
      for (const FanoutCandidate &candidate : candidates) {
        if (usedHolders.contains(candidate.holderIndex) ||
            selectedPeers.contains(candidate.peerIndex))
          continue;
        usedHolders.insert(candidate.holderIndex);
        selectedPeers.insert(candidate.peerIndex);
        selected.push_back(candidate);
      }
      if (selected.empty()) {
        detail = "same-payload fanout participants are disconnected in the "
                 "current target topology";
        return mlir::failure();
      }

      llvm::sort(selected,
                 [](const FanoutCandidate &lhs, const FanoutCandidate &rhs) {
                   return std::tie(lhs.sourceTile, lhs.destinationTile) <
                          std::tie(rhs.sourceTile, rhs.destinationTile);
                 });
      for (const FanoutCandidate &choice : selected) {
        PeerPlan &destination = peers[choice.peerIndex];
        FanoutHolder &source = holders[choice.holderIndex];
        destination.transportSourceTile = source.tile;
        destination.communicationId = communicationId;
        destination.protocolRound = round;
        if (source.peerIndex)
          destination.relaySourceRelation =
              peers[*source.peerIndex].relationIndex;
        ++source.sends;
      }
      remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                                     [&](unsigned peerIndex) {
                                       return selectedPeers.contains(peerIndex);
                                     }),
                      remaining.end());
      for (const FanoutCandidate &choice : selected)
        holders.push_back(FanoutHolder{
            peers[choice.peerIndex].destinationTile,
            peers[choice.peerIndex].destinationTileOrder,
            peers[choice.peerIndex].destinationRegion, choice.peerIndex, 0});
      ++round;
    }
    ++statistics.topologyFanoutGroups;
    statistics.topologyFanoutRounds += static_cast<uint64_t>(round);
  }
  return mlir::success();
}

static BoundaryMovementResult fail(BoundaryMovementFailureKind kind,
                                   llvm::StringRef detail) {
  BoundaryMovementResult result;
  result.failure = kind;
  result.detail = detail.str();
  return result;
}

static bool collectSPMBridges(
    mlir::BlockArgument argument,
    llvm::SmallVectorImpl<mlir::bufferization::ToMemrefOp> &bridges) {
  for (mlir::Operation *user : argument.getUsers()) {
    auto current = mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(user);
    if (!current || !isWaferSPMMemRefType(current.getMemref().getType()))
      return false;
    bridges.push_back(current);
  }
  return true;
}

static mlir::bufferization::ToMemrefOp
getCompatibleSPMBridge(mlir::BlockArgument argument) {
  llvm::SmallVector<mlir::bufferization::ToMemrefOp, 2> bridges;
  if (!collectSPMBridges(argument, bridges) || bridges.empty())
    return {};
  mlir::Type type = bridges.front().getMemref().getType();
  if (llvm::any_of(bridges, [&](mlir::bufferization::ToMemrefOp bridge) {
        return bridge.getMemref().getType() != type;
      }))
    return {};
  return bridges.front();
}

static mlir::bufferization::ToTensorOp getYieldBridge(TileRegionOp region,
                                                      unsigned resultIndex) {
  if (!region || region.getBody().empty())
    return {};
  auto yield =
      mlir::dyn_cast<TileYieldOp>(region.getBody().front().getTerminator());
  if (!yield || resultIndex >= yield.getValues().size())
    return {};
  auto bridge = yield.getValues()[resultIndex]
                    .getDefiningOp<mlir::bufferization::ToTensorOp>();
  if (!bridge || !isWaferSPMMemRefType(bridge.getMemref().getType()))
    return {};
  return bridge;
}

static mlir::MemRefType getDDRType(mlir::MemRefType spmType) {
  return mlir::MemRefType::get(spmType.getShape(), spmType.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               MemoryAttr::get(spmType.getContext(),
                                               MemorySpace::DDR,
                                               MemLayout::Tensor));
}

static mlir::MemRefType getOwnedSPMType(mlir::MemRefType type) {
  return mlir::MemRefType::get(type.getShape(), type.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               type.getMemorySpace());
}

static bool hasOnlySubviewUses(mlir::Value value) {
  return !value.use_empty() &&
         llvm::all_of(value.getUsers(), [](mlir::Operation *user) {
           return mlir::isa<mlir::memref::SubViewOp>(user);
         });
}

static void retargetSubviewUsers(mlir::Value oldValue, mlir::Value newValue,
                                 mlir::IRRewriter &rewriter) {
  for (mlir::OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
    auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(use.getOwner());
    if (!subview || use.getOperandNumber() != 0) {
      use.set(newValue);
      continue;
    }
    rewriter.setInsertionPoint(subview);
    auto replacement = rewriter.create<mlir::memref::SubViewOp>(
        subview.getLoc(), newValue, subview.getMixedOffsets(),
        subview.getMixedSizes(), subview.getMixedStrides());
    retargetSubviewUsers(subview.getResult(), replacement.getResult(),
                         rewriter);
    rewriter.eraseOp(subview);
  }
}

static mlir::LogicalResult materializeSubviewLoads(
    mlir::bufferization::ToMemrefOp bridge, mlir::BlockArgument ddrArgument,
    mlir::IRRewriter &rewriter, BoundaryMovementStatistics &statistics) {
  llvm::SmallVector<mlir::memref::SubViewOp, 8> subviews;
  for (mlir::Operation *user : bridge.getMemref().getUsers())
    subviews.push_back(mlir::cast<mlir::memref::SubViewOp>(user));
  for (mlir::memref::SubViewOp subview : subviews) {
    rewriter.setInsertionPoint(subview);
    auto ddrSubview = rewriter.create<mlir::memref::SubViewOp>(
        subview.getLoc(), ddrArgument, subview.getMixedOffsets(),
        subview.getMixedSizes(), subview.getMixedStrides());
    auto oldType = mlir::cast<mlir::MemRefType>(subview.getType());
    auto allocation = rewriter.create<mlir::memref::AllocOp>(
        subview.getLoc(), getOwnedSPMType(oldType));
    rewriter.create<StorageLoadOp>(subview.getLoc(), ddrSubview.getResult(),
                                   allocation.getResult());
    retargetSubviewUsers(subview.getResult(), allocation.getResult(), rewriter);
    rewriter.eraseOp(subview);
    ++statistics.ddrLoads;
  }
  if (!bridge.getMemref().use_empty())
    return mlir::failure();
  rewriter.eraseOp(bridge);
  ++statistics.tensorBridgesRemoved;
  return mlir::success();
}

static TileRegionOp getRegionOwner(mlir::Value value) {
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
    return mlir::dyn_cast<TileRegionOp>(result.getOwner());
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  return argument && argument.getOwner()
             ? mlir::dyn_cast_or_null<TileRegionOp>(
                   argument.getOwner()->getParentOp())
             : TileRegionOp{};
}

static bool logicalTypesMatch(mlir::MemRefType lhs, mlir::MemRefType rhs) {
  return lhs && rhs && lhs.getShape() == rhs.getShape() &&
         lhs.getElementType() == rhs.getElementType();
}

static bool sameStaticSubview(mlir::memref::SubViewOp lhs,
                              mlir::memref::SubViewOp rhs) {
  auto allStatic = [](llvm::ArrayRef<int64_t> values) {
    return llvm::none_of(values, [](int64_t value) {
      return mlir::ShapedType::isDynamic(value);
    });
  };
  return allStatic(lhs.getStaticOffsets()) && allStatic(lhs.getStaticSizes()) &&
         allStatic(lhs.getStaticStrides()) &&
         lhs.getStaticOffsets() == rhs.getStaticOffsets() &&
         lhs.getStaticSizes() == rhs.getStaticSizes() &&
         lhs.getStaticStrides() == rhs.getStaticStrides();
}

static mlir::memref::SubViewOp getCurrentSourceWindow(mlir::Value source) {
  if (auto direct = source.getDefiningOp<mlir::memref::SubViewOp>())
    return direct;
  mlir::memref::SubViewOp result;
  for (mlir::Operation *user : source.getUsers()) {
    auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(user);
    if (!copy || copy.getSource() != source)
      continue;
    auto subview = copy.getTarget().getDefiningOp<mlir::memref::SubViewOp>();
    if (!subview || (result && !sameStaticSubview(result, subview)))
      return {};
    result = subview;
  }
  return result;
}

static std::optional<MemLayout> getLayout(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  MemoryAttr memory = memref ? getWaferMemoryAttr(memref) : MemoryAttr{};
  return memory ? std::optional<MemLayout>(memory.getLayout()) : std::nullopt;
}

static mlir::LogicalResult
preflight(mlir::ModuleOp module, StructuredMaterializationRelations &relations,
          llvm::SmallVectorImpl<RegionPlan> &regions,
          llvm::SmallVectorImpl<PeerPlan> &peers, std::string &detail) {
  llvm::DenseMap<mlir::Value, llvm::SmallVector<unsigned, 2>> peerSource;
  llvm::DenseMap<mlir::Value, unsigned> peerDestination;
  for (auto [index, relation] : llvm::enumerate(relations.boundaryRelations)) {
    peerSource[relation.sourceEndpoint].push_back(index);
    if (!peerDestination.try_emplace(relation.destinationEndpoint, index)
             .second) {
      detail = "one Tile boundary destination has multiple producers";
      return mlir::failure();
    }
  }

  llvm::DenseSet<unsigned> matchedOutputs;
  llvm::DenseMap<mlir::Operation *, unsigned> regionIndex;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
      functionOutputIndices;
  module.walk([&](TileRegionOp region) {
    regionIndex.try_emplace(region, regions.size());
    RegionPlan plan;
    plan.operation = region;
    plan.tileModule = region->getParentOfType<TileModuleOp>();
    regions.push_back(std::move(plan));
  });
  if (regions.empty()) {
    detail = "physical boundary closure found no TileRegion";
    return mlir::failure();
  }

  for (RegionPlan &plan : regions) {
    TileRegionOp region = plan.operation;
    if (!plan.tileModule || !region.getBody().hasOneBlock()) {
      detail = "TileRegion has no physical Tile owner or single-block body";
      return mlir::failure();
    }
    mlir::Block &block = region.getBody().front();
    for (auto [index, inputAndArgument] :
         llvm::enumerate(llvm::zip(region.getInputs(), block.getArguments()))) {
      mlir::Value input = std::get<0>(inputAndArgument);
      mlir::BlockArgument argument = std::get<1>(inputAndArgument);
      if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
        continue;
      llvm::SmallVector<mlir::bufferization::ToMemrefOp, 2> bridges;
      if (!collectSPMBridges(argument, bridges)) {
        detail = "logical TileRegion input has a non-buffer bridge user";
        return mlir::failure();
      }
      auto peer = peerDestination.find(argument);
      if (bridges.empty() && peer != peerDestination.end()) {
        detail = "cross-Tile destination has no current SPM consumer";
        return mlir::failure();
      }
      if (peer != peerDestination.end() && !bridges.empty() &&
          llvm::any_of(bridges, [&](mlir::bufferization::ToMemrefOp bridge) {
            return bridge.getMemref().getType() !=
                   bridges.front().getMemref().getType();
          })) {
        detail =
            "cross-Tile destination requires one exact receive representation";
        return mlir::failure();
      }
      InputPlan inputPlan;
      inputPlan.index = index;
      inputPlan.originalOperand = input;
      inputPlan.originalArgument = argument;
      inputPlan.bridges = std::move(bridges);
      inputPlan.unused = inputPlan.bridges.empty();
      if (peer != peerDestination.end())
        inputPlan.peerRelation = peer->second;
      plan.inputs.push_back(std::move(inputPlan));
    }

    for (auto [index, result] : llvm::enumerate(region.getResults())) {
      if (!mlir::isa<mlir::RankedTensorType>(result.getType()))
        continue;
      mlir::bufferization::ToTensorOp bridge = getYieldBridge(region, index);
      if (!bridge) {
        detail = "logical TileRegion result has no current SPM yield bridge";
        return mlir::failure();
      }
      ResultPlan resultPlan;
      resultPlan.index = index;
      resultPlan.originalResult = result;
      resultPlan.bridge = bridge;
      resultPlan.spmValue = bridge.getMemref();
      if (auto found = peerSource.find(result); found != peerSource.end())
        resultPlan.peerRelations.append(found->second.begin(),
                                        found->second.end());

      for (mlir::Operation *user : result.getUsers()) {
        if (mlir::isa<TileRegionOp>(user)) {
          resultPlan.hasSameTileConsumer = true;
          continue;
        }
        auto toMemref = mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(user);
        if (!toMemref) {
          detail = "logical TileRegion result has an unknown boundary user";
          return mlir::failure();
        }
        if (isWaferDDRMemRefType(toMemref.getMemref().getType())) {
          if (resultPlan.returnBridge) {
            detail = "TileRegion result has multiple return bridges";
            return mlir::failure();
          }
          if (!llvm::all_of(toMemref.getMemref().getUsers(),
                            [](mlir::Operation *bridgeUser) {
                              return mlir::isa<mlir::func::ReturnOp>(
                                  bridgeUser);
                            })) {
            detail = "DDR TileRegion bridge is not an entry return";
            return mlir::failure();
          }
          resultPlan.returnBridge = toMemref;
          continue;
        }
        for (mlir::Operation *bridgeUser : toMemref.getMemref().getUsers()) {
          auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(bridgeUser);
          if (!copy || copy.getSource() != toMemref.getMemref()) {
            detail = "TileRegion output bridge has a non-publication user";
            return mlir::failure();
          }
          bool matched = false;
          for (auto [outputIndex, output] :
               llvm::enumerate(relations.structuralOutputs)) {
            if (copy.getTarget() != output.endpoint)
              continue;
            if (resultPlan.ddrDestination ||
                !matchedOutputs.insert(outputIndex).second) {
              detail = "observable output relation is duplicated";
              return mlir::failure();
            }
            resultPlan.ddrDestination = copy.getTarget();
            resultPlan.outputIndex = output.outputIndex;
            resultPlan.publicationBridge = toMemref;
            resultPlan.outputCopy = copy;
            matched = true;
          }
          if (!matched) {
            detail = "TileRegion publication copy has no output relation";
            return mlir::failure();
          }
        }
      }
      plan.results.push_back(std::move(resultPlan));
    }
    auto function = region->getParentOfType<mlir::func::FuncOp>();
    for (const ResultPlan &result : plan.results)
      if (result.ddrDestination) {
        if (!function || !result.outputIndex) {
          detail = "observable output has no entry function result index";
          return mlir::failure();
        }
        functionOutputIndices[function.getOperation()].push_back(
            *result.outputIndex);
      }
  }
  if (matchedOutputs.size() != relations.structuralOutputs.size()) {
    detail = "not every observable output has one current publication copy";
    return mlir::failure();
  }
  for (auto &[operation, indices] : functionOutputIndices) {
    auto function = mlir::cast<mlir::func::FuncOp>(operation);
    llvm::sort(indices);
    for (auto [expected, index] : llvm::enumerate(indices))
      if (index != expected) {
        detail = "entry observable output indices are not dense and unique";
        return mlir::failure();
      }
    if (!function.getBody().hasOneBlock() ||
        !mlir::isa<mlir::func::ReturnOp>(
            function.getBody().front().getTerminator()) ||
        (function.getNumResults() != 0 &&
         function.getNumResults() != indices.size())) {
      detail = "entry output boundary is not one complete function return";
      return mlir::failure();
    }
  }

  for (auto [relationIndex, relation] :
       llvm::enumerate(relations.boundaryRelations)) {
    auto source = mlir::dyn_cast<mlir::OpResult>(relation.sourceEndpoint);
    auto destination =
        mlir::dyn_cast<mlir::BlockArgument>(relation.destinationEndpoint);
    TileRegionOp sourceRegion = getRegionOwner(relation.sourceEndpoint);
    TileRegionOp destinationRegion =
        getRegionOwner(relation.destinationEndpoint);
    if (!source || !destination || !sourceRegion || !destinationRegion ||
        sourceRegion == destinationRegion ||
        source.getResultNumber() >= sourceRegion.getNumResults() ||
        destination.getArgNumber() >= destinationRegion.getInputs().size()) {
      detail = "cross-Tile relation does not name current Region endpoints";
      return mlir::failure();
    }
    mlir::bufferization::ToTensorOp sourceBridge =
        getYieldBridge(sourceRegion, source.getResultNumber());
    llvm::SmallVector<mlir::bufferization::ToMemrefOp, 2> destinationBridges;
    const bool compatibleDestinationBridges =
        collectSPMBridges(destination, destinationBridges) &&
        !destinationBridges.empty() &&
        static_cast<bool>(getCompatibleSPMBridge(destination));
    auto sourceTile = sourceRegion->getParentOfType<TileModuleOp>();
    auto destinationTile = destinationRegion->getParentOfType<TileModuleOp>();
    mlir::MemRefType sourceType =
        sourceBridge
            ? mlir::cast<mlir::MemRefType>(sourceBridge.getMemref().getType())
            : mlir::MemRefType{};
    mlir::MemRefType destinationType =
        compatibleDestinationBridges
            ? mlir::cast<mlir::MemRefType>(
                  destinationBridges.front().getMemref().getType())
            : mlir::MemRefType{};
    if (!sourceBridge || !compatibleDestinationBridges || !sourceTile ||
        !destinationTile || sourceTile == destinationTile) {
      detail = "cross-Tile relation has no exact typed peer endpoints";
      return mlir::failure();
    }
    llvm::SmallVector<mlir::memref::SubViewOp, 2> destinationSubviews;
    llvm::SmallVector<int64_t, 4> sourceWindowOffsets;
    llvm::SmallVector<int64_t, 4> sourceWindowSizes;
    llvm::SmallVector<int64_t, 4> sourceWindowStrides;
    if (!logicalTypesMatch(sourceType, destinationType)) {
      for (mlir::bufferization::ToMemrefOp bridge : destinationBridges)
        for (mlir::Operation *user : bridge.getMemref().getUsers()) {
          auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(user);
          if (!subview) {
            llvm::raw_string_ostream stream(detail);
            stream << "cross-Tile peer payload has no exact destination "
                      "window: source="
                   << sourceType << " carrier=" << destinationType;
            return mlir::failure();
          }
          destinationSubviews.push_back(subview);
        }
      if (destinationSubviews.empty() ||
          llvm::any_of(llvm::drop_begin(destinationSubviews),
                       [&](mlir::memref::SubViewOp subview) {
                         return !sameStaticSubview(destinationSubviews.front(),
                                                   subview);
                       })) {
        detail = "cross-Tile peer payload destination windows differ";
        return mlir::failure();
      }
      auto destinationSubviewType = mlir::cast<mlir::MemRefType>(
          destinationSubviews.front().getResult().getType());
      if (!logicalTypesMatch(sourceType, destinationSubviewType)) {
        auto sourceSubview = getCurrentSourceWindow(sourceBridge.getMemref());
        llvm::ArrayRef<int64_t> sourceOffsets =
            sourceSubview ? sourceSubview.getStaticOffsets()
                          : llvm::ArrayRef<int64_t>{};
        llvm::ArrayRef<int64_t> sourceSizes =
            sourceSubview ? sourceSubview.getStaticSizes()
                          : llvm::ArrayRef<int64_t>{};
        llvm::ArrayRef<int64_t> sourceStrides =
            sourceSubview ? sourceSubview.getStaticStrides()
                          : llvm::ArrayRef<int64_t>{};
        llvm::ArrayRef<int64_t> destinationOffsets =
            destinationSubviews.front().getStaticOffsets();
        llvm::ArrayRef<int64_t> destinationSizes =
            destinationSubviews.front().getStaticSizes();
        llvm::ArrayRef<int64_t> destinationStrides =
            destinationSubviews.front().getStaticStrides();
        if (!sourceSubview ||
            sourceOffsets.size() != destinationOffsets.size() ||
            sourceSizes.size() != destinationSizes.size() ||
            sourceStrides.size() != destinationStrides.size() ||
            sourceType.getElementType() !=
                destinationSubviewType.getElementType()) {
          llvm::raw_string_ostream stream(detail);
          stream << "cross-Tile peer windows have incompatible ranks or types: "
                    "source_definition=";
          if (mlir::Operation *definition =
                  sourceBridge.getMemref().getDefiningOp())
            stream << definition->getName();
          else
            stream << "block_argument";
          stream << " source=" << sourceType
                 << " destination=" << destinationSubviewType;
          return mlir::failure();
        }
        for (auto [sourceOffset, sourceSize, sourceStride, destinationOffset,
                   destinationSize, destinationStride] :
             llvm::zip_equal(sourceOffsets, sourceSizes, sourceStrides,
                             destinationOffsets, destinationSizes,
                             destinationStrides)) {
          int64_t sourceEnd = 0;
          int64_t destinationEnd = 0;
          if (mlir::ShapedType::isDynamic(sourceOffset) ||
              mlir::ShapedType::isDynamic(sourceSize) ||
              mlir::ShapedType::isDynamic(sourceStride) ||
              mlir::ShapedType::isDynamic(destinationOffset) ||
              mlir::ShapedType::isDynamic(destinationSize) ||
              mlir::ShapedType::isDynamic(destinationStride) ||
              sourceStride != 1 || destinationStride != 1 || sourceSize <= 0 ||
              destinationSize <= 0 ||
              llvm::AddOverflow(sourceOffset, sourceSize, sourceEnd) ||
              llvm::AddOverflow(destinationOffset, destinationSize,
                                destinationEnd) ||
              destinationOffset < sourceOffset || destinationEnd > sourceEnd) {
            llvm::raw_string_ostream stream(detail);
            stream << "cross-Tile peer destination is outside its source "
                      "window: source_offset="
                   << sourceOffset << " source_size=" << sourceSize
                   << " destination_offset=" << destinationOffset
                   << " destination_size=" << destinationSize
                   << " source_type=" << sourceType
                   << " destination_type=" << destinationSubviewType;
            return mlir::failure();
          }
          sourceWindowOffsets.push_back(destinationOffset - sourceOffset);
          sourceWindowSizes.push_back(destinationSize);
          sourceWindowStrides.push_back(1);
        }
      }
      destinationType = getOwnedSPMType(destinationSubviewType);
    } else {
      destinationType = getOwnedSPMType(destinationType);
    }
    std::optional<WaferPhysicalTensorInfo> destinationPhysical =
        computeWaferPhysicalTensorInfo(destinationType);
    if (!destinationPhysical || destinationPhysical->physicalBytes <= 0) {
      detail = "cross-Tile relation has no exact physical payload bytes";
      return mlir::failure();
    }
    PeerPlan peer{
        static_cast<unsigned>(relationIndex),
        sourceRegion,
        source.getResultNumber(),
        sourceBridge.getMemref(),
        destinationRegion,
        destination.getArgNumber(),
        destinationType,
        {},
        std::move(destinationSubviews),
        std::move(sourceWindowOffsets),
        std::move(sourceWindowSizes),
        std::move(sourceWindowStrides),
        static_cast<uint64_t>(sourceTile.getTileIdAttr().getInt()),
        static_cast<uint64_t>(destinationTile.getTileIdAttr().getInt()),
        static_cast<uint64_t>(destinationPhysical->physicalBytes)};
    for (mlir::bufferization::ToMemrefOp bridge : destinationBridges)
      peer.destinationSPMCarriers.push_back(bridge.getMemref());
    peer.transportSourceTile = peer.sourceTile;
    peer.communicationId = static_cast<int64_t>(relationIndex);
    peers.push_back(std::move(peer));
  }
  return mlir::success();
}

static const PeerPlan *findPeer(llvm::ArrayRef<PeerPlan> peers,
                                unsigned relationIndex) {
  auto found = llvm::find_if(peers, [&](const PeerPlan &peer) {
    return peer.relationIndex == relationIndex;
  });
  return found == peers.end() ? nullptr : &*found;
}

static llvm::SmallVector<const PeerPlan *, 4>
findRelayChildren(llvm::ArrayRef<PeerPlan> peers, unsigned relationIndex) {
  llvm::SmallVector<const PeerPlan *, 4> children;
  for (const PeerPlan &peer : peers)
    if (peer.relaySourceRelation == relationIndex)
      children.push_back(&peer);
  llvm::sort(children, [](const PeerPlan *lhs, const PeerPlan *rhs) {
    return std::tie(lhs->protocolRound, lhs->destinationTile) <
           std::tie(rhs->protocolRound, rhs->destinationTile);
  });
  return children;
}

static DTEMessageAttr getMessage(mlir::MLIRContext *context,
                                 const PeerPlan &peer) {
  return DTEMessageAttr::get(context, peer.communicationId, peer.protocolRound,
                             0);
}

struct SharedDDRPeerBinding {
  mlir::Value source;
  mlir::Value destination;
  mlir::MemRefType type;
};

static mlir::FailureOr<llvm::DenseMap<unsigned, SharedDDRPeerBinding>>
materializeSharedDDRPeerBindings(mlir::ModuleOp module,
                                 llvm::ArrayRef<PeerPlan> peers,
                                 std::string &detail) {
  llvm::DenseMap<unsigned, SharedDDRPeerBinding> bindings;
  int64_t nextResourceId = 0;
  for (mlir::memref::GlobalOp global :
       module.getBody()->getOps<mlir::memref::GlobalOp>())
    if (auto resource =
            global->getAttrOfType<DDRResourceAttr>(kWaferDDRResourceAttrName))
      nextResourceId = std::max(nextResourceId, resource.getResourceId() + 1);

  mlir::OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  llvm::SmallVector<llvm::SmallVector<const PeerPlan *, 4>, 16> groups;
  for (const PeerPlan &peer : peers) {
    if (!peer.useSharedDDR)
      continue;
    auto group =
        llvm::find_if(groups, [&](llvm::ArrayRef<const PeerPlan *> candidate) {
          return !candidate.empty() &&
                 hasSameFanoutPayload(*candidate.front(), peer);
        });
    if (group == groups.end())
      groups.push_back({&peer});
    else
      group->push_back(&peer);
  }
  for (llvm::ArrayRef<const PeerPlan *> group : groups) {
    const PeerPlan &representative = *group.front();
    mlir::func::FuncOp sourceFunction =
        representative.sourceRegion->getParentOfType<mlir::func::FuncOp>();
    if (!sourceFunction) {
      detail = "shared DDR group has no source entry function";
      return mlir::failure();
    }
    llvm::DenseSet<mlir::Operation *> destinationFunctions;
    for (const PeerPlan *peer : group) {
      mlir::func::FuncOp destination =
          peer->destinationRegion->getParentOfType<mlir::func::FuncOp>();
      if (!destination || destination == sourceFunction) {
        detail = "shared DDR group has no distinct destination entry";
        return mlir::failure();
      }
      destinationFunctions.insert(destination.getOperation());
    }

    mlir::MemRefType ddrType = getDDRType(representative.destinationSPMType);
    std::string symbol =
        (llvm::Twine("__wafer_shared_ddr_") + llvm::Twine(nextResourceId))
            .str();
    auto global = builder.create<mlir::memref::GlobalOp>(
        module.getLoc(), symbol, builder.getStringAttr("private"), ddrType,
        mlir::Attribute(), /*constant=*/false, mlir::IntegerAttr());
    global->setAttr(kWaferDDRResourceAttrName,
                    DDRResourceAttr::get(builder.getContext(), nextResourceId));
    auto resource = mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
    auto appendArgument = [&](mlir::func::FuncOp function, DDRAccess access) {
      unsigned index = function.getNumArguments();
      auto binding = DDRBindingAttr::get(builder.getContext(), resource,
                                         nextResourceId, access);
      auto attrs = mlir::DictionaryAttr::get(
          builder.getContext(),
          {builder.getNamedAttr(kWaferDDRBindingAttrName, binding)});
      function.insertArgument(index, ddrType, attrs, module.getLoc());
      return function.getArgument(index);
    };
    mlir::Value source;
    llvm::DenseMap<mlir::Operation *, mlir::Value> entryArguments;
    for (TileModuleOp tile : module.getOps<TileModuleOp>()) {
      mlir::func::FuncOp entry;
      for (mlir::func::FuncOp function : tile.getOps<mlir::func::FuncOp>()) {
        if (function.isPrivate() || function.isExternal())
          continue;
        if (entry) {
          detail = "Tile has multiple public entry functions for shared DDR";
          return mlir::failure();
        }
        entry = function;
      }
      if (!entry) {
        detail = "Tile has no public entry function for shared DDR";
        return mlir::failure();
      }
      const bool writes = entry == sourceFunction;
      const bool reads = destinationFunctions.contains(entry.getOperation());
      DDRAccess access = writes && reads ? DDRAccess::ReadWrite
                         : writes        ? DDRAccess::Write
                         : reads         ? DDRAccess::Read
                                         : DDRAccess::None;
      mlir::Value argument = appendArgument(entry, access);
      entryArguments.try_emplace(entry.getOperation(), argument);
      if (writes)
        source = argument;
    }
    if (!source) {
      detail = "shared DDR source is not a public Tile entry";
      return mlir::failure();
    }
    for (const PeerPlan *peer : group) {
      mlir::func::FuncOp destinationFunction =
          peer->destinationRegion->getParentOfType<mlir::func::FuncOp>();
      mlir::Value destination =
          entryArguments.lookup(destinationFunction.getOperation());
      if (!destination ||
          !bindings
               .try_emplace(peer->relationIndex,
                            SharedDDRPeerBinding{source, destination, ddrType})
               .second) {
        detail = "shared DDR relation has no unique destination binding";
        return mlir::failure();
      }
    }
    ++nextResourceId;
  }
  return bindings;
}

static mlir::FailureOr<mlir::Value>
resolveDDRInput(mlir::Value original,
                const llvm::DenseMap<mlir::Value, mlir::Value> &stagedResults) {
  if (auto mapped = stagedResults.find(original); mapped != stagedResults.end())
    return mapped->second;
  if (auto bridge = original.getDefiningOp<mlir::bufferization::ToTensorOp>()) {
    if (isWaferDDRMemRefType(bridge.getMemref().getType()))
      return bridge.getMemref();
  }
  if (isWaferDDRMemRefType(original.getType()))
    return original;
  return mlir::failure();
}

static void eraseDeadBridges(mlir::ModuleOp module) {
  bool changed = true;
  mlir::IRRewriter rewriter(module.getContext());
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *, 16> dead;
    module.walk([&](mlir::Operation *operation) {
      if (mlir::isa<mlir::bufferization::ToTensorOp,
                    mlir::bufferization::ToMemrefOp>(operation) &&
          operation->use_empty())
        dead.push_back(operation);
    });
    for (mlir::Operation *operation : llvm::reverse(dead)) {
      if (!operation->use_empty())
        continue;
      rewriter.eraseOp(operation);
      changed = true;
    }
  }
}

static mlir::Operation *getBlockAnchor(mlir::Operation *operation,
                                       mlir::Block *block) {
  mlir::Operation *anchor = operation;
  while (anchor && anchor->getBlock() != block)
    anchor = anchor->getParentOp();
  return anchor && anchor->getBlock() == block ? anchor : nullptr;
}

static mlir::Operation *findFirstBufferConsumer(mlir::Value buffer,
                                                mlir::Block &block) {
  mlir::Operation *first = nullptr;
  llvm::SmallVector<mlir::Value, 8> worklist{buffer};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (mlir::Operation *user : value.getUsers()) {
      if (mlir::isa<mlir::ViewLikeOpInterface, mlir::bufferization::ToTensorOp,
                    mlir::bufferization::ToMemrefOp>(user)) {
        for (mlir::Value result : user->getResults())
          if (mlir::isa<mlir::ShapedType>(result.getType()))
            worklist.push_back(result);
        continue;
      }
      mlir::Operation *anchor = getBlockAnchor(user, &block);
      if (!anchor)
        return nullptr;
      if (!first || anchor->isBeforeInBlock(first))
        first = anchor;
    }
  }
  return first;
}

static bool shareStorage(mlir::Value lhs, mlir::Value rhs,
                         StorageRootMemo &memo) {
  const llvm::DenseSet<mlir::Value> &lhsRoots = memo.getStorageRoots(lhs);
  const llvm::DenseSet<mlir::Value> &rhsRoots = memo.getStorageRoots(rhs);
  return llvm::any_of(
      lhsRoots, [&](mlir::Value root) { return rhsRoots.contains(root); });
}

static mlir::LogicalResult findLastBufferWrite(mlir::Value buffer,
                                               mlir::Block &block,
                                               StorageRootMemo &memo,
                                               mlir::Operation *&last,
                                               std::string &detail) {
  last = getBlockAnchor(buffer.getDefiningOp(), &block);
  for (mlir::Operation &operation : block) {
    if (operation.getNumRegions() == 0 &&
        mlir::isa<mlir::ViewLikeOpInterface>(operation))
      continue;
    bool hasBufferOperand =
        llvm::any_of(operation.getOperands(), [&](mlir::Value operand) {
          return mlir::isa<mlir::BaseMemRefType>(operand.getType()) &&
                 shareStorage(buffer, operand, memo);
        });
    std::optional<llvm::SmallVector<mlir::MemoryEffects::EffectInstance>>
        effects = mlir::getEffectsRecursively(&operation);
    if (!effects) {
      if (hasBufferOperand) {
        detail = "communication source has an untyped memory access";
        return mlir::failure();
      }
      continue;
    }
    bool writes = false;
    bool hasValueEffect = false;
    for (const mlir::MemoryEffects::EffectInstance &effect : *effects) {
      mlir::Value value = effect.getValue();
      if (!value || !mlir::isa<mlir::BaseMemRefType>(value.getType()) ||
          !shareStorage(buffer, value, memo))
        continue;
      hasValueEffect = true;
      writes |= mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect());
    }
    if (hasBufferOperand && !hasValueEffect) {
      llvm::raw_string_ostream stream(detail);
      stream << "communication source access has no value-specific effect: "
             << operation.getName() << " source=" << buffer.getType();
      return mlir::failure();
    }
    if (writes)
      last = &operation;
  }
  return mlir::success();
}

static mlir::LogicalResult apply(mlir::ModuleOp module,
                                 llvm::MutableArrayRef<RegionPlan> regions,
                                 llvm::ArrayRef<PeerPlan> peers,
                                 BoundaryMovementStatistics &statistics,
                                 std::string &detail) {
  auto failApply = [&](llvm::StringRef message) {
    detail = message.str();
    return mlir::failure();
  };
  mlir::IRRewriter rewriter(module.getContext());
  mlir::FailureOr<llvm::DenseMap<unsigned, SharedDDRPeerBinding>>
      sharedDDRBindings =
          materializeSharedDDRPeerBindings(module, peers, detail);
  if (mlir::failed(sharedDDRBindings))
    return mlir::failure();
  llvm::DenseMap<mlir::Value, mlir::Value> stagedResults;
  llvm::SmallVector<TileRegionOp, 16> oldRegions;
  llvm::DenseMap<mlir::Operation *,
                 llvm::SmallVector<std::pair<unsigned, mlir::Value>, 2>>
      functionOutputs;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
      outputArguments;

  for (const RegionPlan &plan : regions) {
    mlir::func::FuncOp function =
        plan.operation->getParentOfType<mlir::func::FuncOp>();
    if (!function || function.getBody().empty())
      continue;
    for (const InputPlan &input : plan.inputs) {
      if (!input.peerRelation)
        continue;
      auto argument =
          mlir::dyn_cast<mlir::BlockArgument>(input.originalOperand);
      if (!argument || argument.getOwner() != &function.getBody().front())
        continue;
      outputArguments[function.getOperation()].push_back(
          argument.getArgNumber());
    }
  }
  for (TileModuleOp tile : module.getOps<TileModuleOp>())
    for (mlir::func::FuncOp function : tile.getOps<mlir::func::FuncOp>())
      for (unsigned index = 0; index < function.getNumArguments(); ++index)
        if (function.getArgAttr(index, kWaferCrossTilePlaceholderAttrName))
          outputArguments[function.getOperation()].push_back(index);

  for (RegionPlan &plan : regions) {
    TileRegionOp oldRegion = plan.operation;
    oldRegions.push_back(oldRegion);
    llvm::DenseMap<unsigned, mlir::Value> resultDestinations;
    for (ResultPlan &result : plan.results) {
      if (result.ddrDestination) {
        auto oldSubview =
            result.ddrDestination.getDefiningOp<mlir::memref::SubViewOp>();
        auto outputArgument =
            oldSubview
                ? mlir::dyn_cast<mlir::BlockArgument>(oldSubview.getSource())
                : mlir::BlockArgument{};
        auto function = oldRegion->getParentOfType<mlir::func::FuncOp>();
        if (!oldSubview || !outputArgument || !function ||
            !result.outputIndex ||
            outputArgument.getOwner() != &function.getBody().front())
          return failApply("observable output has no exact temporary arg");
        rewriter.setInsertionPoint(oldRegion);
        auto output = rewriter.create<mlir::memref::AllocOp>(
            oldRegion.getLoc(),
            mlir::cast<mlir::MemRefType>(outputArgument.getType()));
        auto outputSubview = rewriter.create<mlir::memref::SubViewOp>(
            oldSubview.getLoc(), output.getResult(),
            oldSubview.getMixedOffsets(), oldSubview.getMixedSizes(),
            oldSubview.getMixedStrides());
        result.obsoleteOutputSubview = oldSubview;
        result.ddrDestination = outputSubview.getResult();
        functionOutputs[function.getOperation()].push_back(
            {*result.outputIndex, output.getResult()});
        outputArguments[function.getOperation()].push_back(
            outputArgument.getArgNumber());
        resultDestinations.try_emplace(result.index, result.ddrDestination);
        stagedResults.try_emplace(result.originalResult, result.ddrDestination);
        continue;
      }
      if (!result.hasSameTileConsumer)
        continue;
      rewriter.setInsertionPoint(oldRegion);
      auto staging = rewriter.create<mlir::memref::AllocOp>(
          oldRegion.getLoc(),
          getDDRType(mlir::cast<mlir::MemRefType>(result.spmValue.getType())));
      resultDestinations.try_emplace(result.index, staging.getResult());
      stagedResults.try_emplace(result.originalResult, staging.getResult());
      ++statistics.interRegionDDRStages;
    }

    llvm::SmallVector<mlir::Value, 12> newInputs;
    llvm::DenseMap<unsigned, const InputPlan *> inputPlans;
    for (InputPlan &input : plan.inputs) {
      inputPlans.try_emplace(input.index, &input);
    }
    for (auto [index, input] : llvm::enumerate(oldRegion.getInputs())) {
      if (!mlir::isa<mlir::RankedTensorType>(input.getType())) {
        newInputs.push_back(input);
        continue;
      }
      const InputPlan *inputPlan = inputPlans.lookup(index);
      if (!inputPlan)
        return failApply("logical input has no prepared boundary binding");
      if (inputPlan->peerRelation) {
        const PeerPlan *peer = findPeer(peers, *inputPlan->peerRelation);
        if (!peer)
          return failApply("peer input has no prepared movement");
        if (peer->useSharedDDR) {
          auto binding = sharedDDRBindings->find(peer->relationIndex);
          if (binding == sharedDDRBindings->end())
            return failApply("shared DDR peer input has no actual binding");
          newInputs.push_back(binding->second.destination);
        }
        continue;
      }
      if (inputPlan->unused)
        continue;
      mlir::FailureOr<mlir::Value> ddr =
          resolveDDRInput(inputPlan->originalOperand, stagedResults);
      if (mlir::failed(ddr))
        return failApply("logical input has no current DDR source");
      newInputs.push_back(*ddr);
    }

    llvm::SmallVector<unsigned, 8> destinationResultIndices;
    for (const ResultPlan &result : plan.results) {
      auto destination = resultDestinations.find(result.index);
      if (destination == resultDestinations.end())
        continue;
      newInputs.push_back(destination->second);
      destinationResultIndices.push_back(result.index);
    }
    llvm::SmallVector<mlir::Value, 8> sharedDDRResultSources;
    for (const ResultPlan &result : plan.results)
      for (unsigned relationIndex : result.peerRelations) {
        const PeerPlan *peer = findPeer(peers, relationIndex);
        if (!peer || !peer->useSharedDDR)
          continue;
        auto binding = sharedDDRBindings->find(relationIndex);
        if (binding == sharedDDRBindings->end())
          return failApply("shared DDR peer output has no actual binding");
        if (!llvm::is_contained(sharedDDRResultSources,
                                binding->second.source)) {
          newInputs.push_back(binding->second.source);
          sharedDDRResultSources.push_back(binding->second.source);
        }
      }
    llvm::SmallVector<mlir::Type, 4> scalarResultTypes;
    for (mlir::Value result : oldRegion.getResults())
      if (!mlir::isa<mlir::RankedTensorType>(result.getType()))
        scalarResultTypes.push_back(result.getType());

    rewriter.setInsertionPoint(oldRegion);
    auto newRegion = rewriter.create<TileRegionOp>(
        oldRegion.getLoc(), scalarResultTypes, newInputs);
    newRegion.getBody().takeBody(oldRegion.getBody());
    mlir::Block &block = newRegion.getBody().front();
    llvm::DenseMap<unsigned, mlir::BlockArgument> destinationArguments;
    for (unsigned resultIndex : destinationResultIndices) {
      mlir::Value destination = resultDestinations.lookup(resultIndex);
      mlir::BlockArgument argument =
          block.addArgument(destination.getType(), destination.getLoc());
      destinationArguments.try_emplace(resultIndex, argument);
    }
    llvm::DenseMap<mlir::Value, mlir::BlockArgument> sharedDDRResultArguments;
    for (mlir::Value source : sharedDDRResultSources) {
      mlir::BlockArgument argument =
          block.addArgument(source.getType(), source.getLoc());
      sharedDDRResultArguments.try_emplace(source, argument);
    }

    llvm::SmallVector<unsigned, 8> erasedArguments;
    struct ReceivedPayload {
      const PeerPlan *peer = nullptr;
      mlir::Value allocation;
      mlir::Location location;
    };
    llvm::SmallVector<ReceivedPayload, 8> receivedPayloads;
    for (InputPlan &input : plan.inputs) {
      mlir::BlockArgument argument = block.getArgument(input.index);
      if (input.unused) {
        erasedArguments.push_back(input.index);
        continue;
      }
      if (input.peerRelation) {
        const PeerPlan *peer = findPeer(peers, *input.peerRelation);
        if (!peer || input.bridges.empty())
          return failApply("peer input has no exact movement buffer");
        rewriter.setInsertionPointToStart(&block);
        auto allocation = rewriter.create<mlir::memref::AllocOp>(
            argument.getLoc(), peer->destinationSPMType);
        if (peer->useSharedDDR) {
          auto binding = sharedDDRBindings->find(peer->relationIndex);
          if (binding == sharedDDRBindings->end())
            return failApply("shared DDR peer input has no actual binding");
          argument.setType(binding->second.type);
          rewriter.create<StorageLoadOp>(argument.getLoc(), argument,
                                         allocation.getResult());
          ++statistics.ddrLoads;
          ++statistics.crossTileDDRStages;
        } else {
          erasedArguments.push_back(input.index);
          receivedPayloads.push_back(
              ReceivedPayload{peer, allocation.getResult(), argument.getLoc()});
        }
        for (mlir::memref::SubViewOp subview : peer->destinationSubviews) {
          rewriter.replaceAllUsesWith(subview.getResult(),
                                      allocation.getResult());
          rewriter.eraseOp(subview);
        }
        for (mlir::bufferization::ToMemrefOp bridge : input.bridges) {
          if (peer->destinationSubviews.empty())
            rewriter.replaceAllUsesWith(bridge.getMemref(),
                                        allocation.getResult());
          if (!bridge.getMemref().use_empty())
            return failApply("peer destination carrier still has a live use");
          rewriter.eraseOp(bridge);
          ++statistics.tensorBridgesRemoved;
        }
      } else {
        mlir::FailureOr<mlir::Value> ddr =
            resolveDDRInput(input.originalOperand, stagedResults);
        if (mlir::failed(ddr))
          return failApply("local input has no current DDR source");
        argument.setType((*ddr).getType());
        for (mlir::bufferization::ToMemrefOp bridge : input.bridges) {
          if (hasOnlySubviewUses(bridge.getMemref())) {
            if (mlir::failed(materializeSubviewLoads(bridge, argument, rewriter,
                                                     statistics)))
              return failApply("temporal subview load materialization failed");
            continue;
          }
          rewriter.setInsertionPointToStart(&block);
          auto allocation = rewriter.create<mlir::memref::AllocOp>(
              argument.getLoc(),
              mlir::cast<mlir::MemRefType>(bridge.getMemref().getType()));
          rewriter.create<StorageLoadOp>(argument.getLoc(), argument,
                                         allocation.getResult());
          rewriter.replaceAllUsesWith(bridge.getMemref(),
                                      allocation.getResult());
          rewriter.eraseOp(bridge);
          ++statistics.ddrLoads;
          ++statistics.tensorBridgesRemoved;
        }
      }
    }
    llvm::sort(erasedArguments, std::greater<unsigned>());
    for (unsigned index : erasedArguments)
      block.eraseArgument(index);

    auto yield = mlir::cast<TileYieldOp>(block.getTerminator());
    auto materializeTransferSource =
        [&](const ResultPlan &result,
            const PeerPlan &peer) -> mlir::FailureOr<mlir::Value> {
      mlir::bufferization::ToTensorOp bridge = result.bridge;
      mlir::Value source = bridge.getMemref();
      if (!peer.sourceWindowSizes.empty()) {
        llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
        llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
        llvm::SmallVector<mlir::OpFoldResult, 4> strides;
        for (int64_t value : peer.sourceWindowOffsets)
          offsets.push_back(rewriter.getIndexAttr(value));
        for (int64_t value : peer.sourceWindowSizes)
          sizes.push_back(rewriter.getIndexAttr(value));
        for (int64_t value : peer.sourceWindowStrides)
          strides.push_back(rewriter.getIndexAttr(value));
        source = rewriter
                     .create<mlir::memref::SubViewOp>(
                         result.originalResult.getLoc(), source, offsets, sizes,
                         strides)
                     .getResult();
      }
      if (source.getType() == peer.destinationSPMType)
        return source;
      if (!logicalTypesMatch(mlir::cast<mlir::MemRefType>(source.getType()),
                             peer.destinationSPMType))
        return mlir::failure();
      if (getLayout(source.getType()) != getLayout(peer.destinationSPMType))
        source =
            rewriter
                .create<LayoutMaterializeOp>(result.originalResult.getLoc(),
                                             peer.destinationSPMType, source)
                .getResult();
      return source;
    };

    struct PendingSend {
      const ResultPlan *result = nullptr;
      const PeerPlan *peer = nullptr;
      mlir::Value relaySource;
      mlir::Location location;
    };
    llvm::SmallVector<PendingSend, 16> pendingSends;
    for (const ResultPlan &result : plan.results)
      for (unsigned relationIndex : result.peerRelations) {
        const PeerPlan *peer = findPeer(peers, relationIndex);
        if (!peer)
          return failApply("peer output has no prepared transfer");
        if (!peer->useSharedDDR && !peer->relaySourceRelation)
          pendingSends.push_back(
              PendingSend{&result, peer, {}, result.originalResult.getLoc()});
      }

    for (const ReceivedPayload &received : receivedPayloads)
      for (const PeerPlan *child :
           findRelayChildren(peers, received.peer->relationIndex)) {
        if (received.allocation.getType() != child->destinationSPMType)
          return failApply("peer relay payload representation changed");
        pendingSends.push_back(PendingSend{nullptr, child, received.allocation,
                                           received.location});
      }

    StorageRootMemo storageRoots;
    llvm::sort(receivedPayloads, [](const ReceivedPayload &lhs,
                                    const ReceivedPayload &rhs) {
      return std::tie(lhs.peer->protocolRound, lhs.peer->communicationId,
                      lhs.peer->sourceTile) <
             std::tie(rhs.peer->protocolRound, rhs.peer->communicationId,
                      rhs.peer->sourceTile);
    });
    llvm::sort(
        pendingSends, [](const PendingSend &lhs, const PendingSend &rhs) {
          return std::tie(lhs.peer->protocolRound, lhs.peer->communicationId,
                          lhs.peer->destinationTile) <
                 std::tie(rhs.peer->protocolRound, rhs.peer->communicationId,
                          rhs.peer->destinationTile);
        });
    llvm::SmallVector<int64_t, 4> scheduleComponents;
    for (const ReceivedPayload &received : receivedPayloads)
      if (received.peer->scheduleComponent >= 0 &&
          !llvm::is_contained(scheduleComponents,
                              received.peer->scheduleComponent))
        scheduleComponents.push_back(received.peer->scheduleComponent);
    for (const PendingSend &send : pendingSends)
      if (send.peer->scheduleComponent >= 0 &&
          !llvm::is_contained(scheduleComponents, send.peer->scheduleComponent))
        scheduleComponents.push_back(send.peer->scheduleComponent);
    llvm::sort(scheduleComponents);
    llvm::DenseSet<unsigned> emittedReceives;
    llvm::DenseSet<unsigned> emittedSends;
    for (int64_t component : scheduleComponents) {
      mlir::Operation *firstConsumer = block.getTerminator();
      for (const ReceivedPayload &received : receivedPayloads) {
        if (received.peer->scheduleComponent != component)
          continue;
        mlir::Operation *consumer =
            findFirstBufferConsumer(received.allocation, block);
        if (consumer && consumer->isBeforeInBlock(firstConsumer))
          firstConsumer = consumer;
      }
      mlir::Operation *lastProducer = nullptr;
      llvm::DenseSet<mlir::Value> checkedSources;
      for (const PendingSend &send : pendingSends) {
        if (send.peer->scheduleComponent != component || !send.result)
          continue;
        mlir::bufferization::ToTensorOp bridge = send.result->bridge;
        mlir::Value source = bridge.getMemref();
        if (!checkedSources.insert(source).second)
          continue;
        mlir::Operation *producer = nullptr;
        if (mlir::failed(findLastBufferWrite(source, block, storageRoots,
                                             producer, detail)))
          return mlir::failure();
        if (producer &&
            (!lastProducer || lastProducer->isBeforeInBlock(producer)))
          lastProducer = producer;
      }
      if (lastProducer && (lastProducer == firstConsumer ||
                           firstConsumer->isBeforeInBlock(lastProducer))) {
        llvm::raw_string_ostream stream(detail);
        stream << "scheduled communication component has no current "
                  "producer-to-consumer cut: component="
               << component << " first_consumer=" << firstConsumer->getName()
               << " last_producer=" << lastProducer->getName();
        return mlir::failure();
      }

      int64_t maximumRound = -1;
      for (const ReceivedPayload &received : receivedPayloads)
        if (received.peer->scheduleComponent == component)
          maximumRound = std::max(maximumRound, received.peer->protocolRound);
      for (const PendingSend &send : pendingSends)
        if (send.peer->scheduleComponent == component)
          maximumRound = std::max(maximumRound, send.peer->protocolRound);
      rewriter.setInsertionPoint(firstConsumer);
      for (int64_t round = 0; round <= maximumRound; ++round) {
        for (const ReceivedPayload &received : receivedPayloads) {
          if (received.peer->scheduleComponent != component ||
              received.peer->protocolRound != round)
            continue;
          rewriter.create<CommPeerRecvOp>(
              received.location, received.allocation,
              received.peer->transportSourceTile, received.peer->bytes,
              getMessage(rewriter.getContext(), *received.peer));
          emittedReceives.insert(received.peer->relationIndex);
          ++statistics.peerReceives;
        }
        for (const PendingSend &send : pendingSends) {
          if (send.peer->scheduleComponent != component ||
              send.peer->protocolRound != round)
            continue;
          mlir::Value source = send.relaySource;
          if (send.result) {
            mlir::FailureOr<mlir::Value> materialized =
                materializeTransferSource(*send.result, *send.peer);
            if (mlir::failed(materialized))
              return failApply("peer output layout conversion is not exact");
            source = *materialized;
          }
          rewriter.create<CommPeerSendOp>(
              send.location, source, send.peer->destinationTile,
              send.peer->bytes, getMessage(rewriter.getContext(), *send.peer));
          emittedSends.insert(send.peer->relationIndex);
          ++statistics.peerSends;
          statistics.peerRelaySends += static_cast<bool>(send.relaySource);
        }
      }
    }

    for (const ReceivedPayload &received : receivedPayloads) {
      if (emittedReceives.contains(received.peer->relationIndex))
        continue;
      mlir::Operation *consumer =
          findFirstBufferConsumer(received.allocation, block);
      rewriter.setInsertionPoint(consumer ? consumer : block.getTerminator());
      auto receive = rewriter.create<CommPeerRecvOp>(
          received.location, received.allocation,
          received.peer->transportSourceTile, received.peer->bytes,
          getMessage(rewriter.getContext(), *received.peer));
      ++statistics.peerReceives;
      mlir::Operation *lastIssue = receive.getOperation();
      for (const PendingSend &send : pendingSends) {
        if (send.peer->scheduleComponent >= 0 || send.result ||
            send.relaySource != received.allocation)
          continue;
        rewriter.setInsertionPointAfter(lastIssue);
        auto relay = rewriter.create<CommPeerSendOp>(
            send.location, send.relaySource, send.peer->destinationTile,
            send.peer->bytes, getMessage(rewriter.getContext(), *send.peer));
        lastIssue = relay.getOperation();
        emittedSends.insert(send.peer->relationIndex);
        ++statistics.peerSends;
        ++statistics.peerRelaySends;
      }
    }

    llvm::DenseMap<mlir::Operation *, mlir::Operation *> lastIssueByProducer;
    for (const PendingSend &send : pendingSends) {
      if (!send.result || emittedSends.contains(send.peer->relationIndex))
        continue;
      mlir::bufferization::ToTensorOp bridge = send.result->bridge;
      mlir::Value source = bridge.getMemref();
      mlir::Operation *producer = nullptr;
      if (mlir::failed(findLastBufferWrite(source, block, storageRoots,
                                           producer, detail)))
        return mlir::failure();
      if (mlir::Operation *lastIssue = lastIssueByProducer.lookup(producer))
        rewriter.setInsertionPointAfter(lastIssue);
      else if (producer)
        rewriter.setInsertionPointAfter(producer);
      else
        rewriter.setInsertionPointToStart(&block);
      mlir::FailureOr<mlir::Value> materialized =
          materializeTransferSource(*send.result, *send.peer);
      if (mlir::failed(materialized))
        return failApply("peer output layout conversion is not exact");
      auto issue = rewriter.create<CommPeerSendOp>(
          send.location, *materialized, send.peer->destinationTile,
          send.peer->bytes, getMessage(rewriter.getContext(), *send.peer));
      lastIssueByProducer[producer] = issue.getOperation();
      ++statistics.peerSends;
    }
    rewriter.setInsertionPoint(yield);
    llvm::DenseSet<mlir::Value> storedSharedDDRResults;
    for (const ResultPlan &result : plan.results) {
      for (unsigned relationIndex : result.peerRelations) {
        const PeerPlan *peer = findPeer(peers, relationIndex);
        if (!peer || !peer->useSharedDDR)
          continue;
        mlir::Value sourceBinding =
            sharedDDRBindings->lookup(relationIndex).source;
        if (!storedSharedDDRResults.insert(sourceBinding).second)
          continue;
        auto destination = sharedDDRResultArguments.find(sourceBinding);
        if (destination == sharedDDRResultArguments.end())
          return failApply("shared DDR peer output has no Region argument");
        mlir::FailureOr<mlir::Value> source =
            materializeTransferSource(result, *peer);
        if (mlir::failed(source))
          return failApply("shared DDR output conversion is not exact");
        rewriter.create<StorageStoreOp>(result.originalResult.getLoc(), *source,
                                        destination->second);
        ++statistics.ddrStores;
      }
    }
    for (const ResultPlan &result : plan.results) {
      auto destination = destinationArguments.find(result.index);
      if (destination == destinationArguments.end())
        continue;
      mlir::bufferization::ToTensorOp bridge = result.bridge;
      rewriter.create<StorageStoreOp>(result.originalResult.getLoc(),
                                      bridge.getMemref(), destination->second);
      ++statistics.ddrStores;
    }
    llvm::SmallVector<mlir::Value, 4> scalarYields;
    for (auto [index, value] : llvm::enumerate(yield.getValues()))
      if (!mlir::isa<mlir::RankedTensorType>(
              oldRegion.getResult(index).getType()))
        scalarYields.push_back(value);
    rewriter.modifyOpInPlace(
        yield, [&] { yield.getValuesMutable().assign(scalarYields); });

    unsigned scalarResult = 0;
    for (mlir::Value oldResult : oldRegion.getResults()) {
      if (mlir::isa<mlir::RankedTensorType>(oldResult.getType()))
        continue;
      rewriter.replaceAllUsesWith(oldResult,
                                  newRegion.getResult(scalarResult++));
    }
    for (ResultPlan &result : plan.results) {
      if (result.outputCopy) {
        rewriter.eraseOp(result.outputCopy);
        ++statistics.outputCopiesRemoved;
      }
      if (result.publicationBridge &&
          result.publicationBridge.getMemref().use_empty()) {
        rewriter.eraseOp(result.publicationBridge);
        ++statistics.tensorBridgesRemoved;
      }
      if (result.returnBridge && result.returnBridge.getMemref().use_empty()) {
        rewriter.eraseOp(result.returnBridge);
        ++statistics.tensorBridgesRemoved;
      }
      if (result.bridge && result.bridge->getResult(0).use_empty()) {
        rewriter.eraseOp(result.bridge);
        ++statistics.tensorBridgesRemoved;
      }
      if (result.obsoleteOutputSubview &&
          result.obsoleteOutputSubview.getResult().use_empty())
        rewriter.eraseOp(result.obsoleteOutputSubview);
    }
  }

  for (auto &[operation, outputs] : functionOutputs) {
    auto function = mlir::cast<mlir::func::FuncOp>(operation);
    llvm::sort(outputs, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    llvm::SmallVector<mlir::Value, 2> values;
    llvm::SmallVector<mlir::Type, 2> types;
    values.reserve(outputs.size());
    types.reserve(outputs.size());
    for (const auto &[index, value] : outputs) {
      (void)index;
      values.push_back(value);
      types.push_back(value.getType());
    }
    auto returnOp = mlir::cast<mlir::func::ReturnOp>(
        function.getBody().front().getTerminator());
    rewriter.modifyOpInPlace(
        returnOp, [&] { returnOp.getOperandsMutable().assign(values); });
    rewriter.modifyOpInPlace(function, [&] {
      function.setFunctionType(mlir::FunctionType::get(
          function.getContext(), function.getArgumentTypes(), types));
    });
  }

  eraseDeadBridges(module);
  for (TileRegionOp region : llvm::reverse(oldRegions)) {
    if (!region->use_empty())
      return failApply("old logical TileRegion still has a live use");
    rewriter.eraseOp(region);
  }
  eraseDeadBridges(module);
  for (auto &[operation, indices] : outputArguments) {
    auto function = mlir::cast<mlir::func::FuncOp>(operation);
    llvm::sort(indices, std::greater<unsigned>());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    for (unsigned index : indices) {
      if (index >= function.getNumArguments() ||
          !function.getArgument(index).use_empty())
        return failApply(
            "movement-closed function placeholder still has a live use");
      function.eraseArgument(index);
    }
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult verifyPhysicalTileDataflow(mlir::ModuleOp module) {
  if (!module || mlir::failed(mlir::verify(module)) ||
      mlir::failed(verifyTileRegionStorageBoundaries(module)))
    return mlir::failure();
  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::linalg::LinalgOp, mlir::bufferization::ToTensorOp,
                  mlir::bufferization::ToMemrefOp>(operation) ||
        (mlir::isa<mlir::memref::CopyOp>(operation) &&
         !operation->getParentOfType<TileRegionOp>())) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    auto region = mlir::dyn_cast<TileRegionOp>(operation);
    if (!region)
      return mlir::WalkResult::advance();
    for (mlir::Value input : region.getInputs())
      if (mlir::isa<mlir::ShapedType>(input.getType()) &&
          !isWaferDDRMemRefType(input.getType())) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    for (mlir::Type type : region.getResultTypes())
      if (mlir::isa<mlir::ShapedType>(type) && !isWaferDDRMemRefType(type)) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    return mlir::WalkResult::advance();
  });
  return mlir::success(!illegal);
}

BoundaryMovementResult
materializeTileBoundaryMovement(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations) {
  if (!module || mlir::failed(verifyStructuredComputeLowered(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations)))
    return fail(BoundaryMovementFailureKind::BrokenContract,
                "boundary movement requires structured-compute-lowered "
                "current IR and live endpoint relations");
  llvm::SmallVector<RegionPlan, 16> regions;
  llvm::SmallVector<PeerPlan, 16> peers;
  std::string detail;
  if (mlir::failed(preflight(module, relations, regions, peers, detail)))
    return fail(BoundaryMovementFailureKind::Unsupported, detail);

  BoundaryMovementResult result;
  if (mlir::failed(
          buildTopologyFanoutChoices(module, peers, result.statistics, detail)))
    return fail(BoundaryMovementFailureKind::Unsupported, detail);
  if (mlir::failed(apply(module, regions, peers, result.statistics, detail)))
    return fail(BoundaryMovementFailureKind::CompilerFailure,
                detail.empty()
                    ? "preflighted boundary movement failed while rewriting "
                      "current IR"
                    : detail);
  relations.boundaryRelations.clear();
  relations.structuralOutputs.clear();
  rebuildCurrentBufferOwnerRelations(module, relations);
  if (mlir::failed(verifyPhysicalTileDataflow(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations)))
    return fail(BoundaryMovementFailureKind::CompilerFailure,
                "boundary movement produced invalid physical Tile IR or "
                "buffer relations");
  return result;
}

} // namespace wafer::compiler::detail
