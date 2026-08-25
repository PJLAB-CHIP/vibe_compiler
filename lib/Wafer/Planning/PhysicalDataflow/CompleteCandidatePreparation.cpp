//===- CompleteCandidatePreparation.cpp - Selected candidate stages --===//

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidatePreparation.h"

#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"
#include "Wafer/Analysis/Structured/StructuredNodeUseIndex.h"
#include "Wafer/Planning/PhysicalDataflow/ExecutionStructureMaterialization.h"
#include "Wafer/Planning/PhysicalDataflow/MovementTransferBuilder.h"
#include "Wafer/Planning/PhysicalDataflow/ScheduleMaterialization.h"
#include "Wafer/Planning/PhysicalDataflow/StorageObjectBuilder.h"
#include "Wafer/Support/CompileTiming.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>

namespace wafer::compiler::detail {
namespace {

enum class ActualIssueKind : uint8_t {
  Compute,
  LocalCombine,
  DDRLoad,
  DDRStore,
  PeerSend,
  PeerReceive,
};

struct ActualIssue {
  TileId tile{0};
  mlir::Operation *operation = nullptr;
  ActualIssueKind kind = ActualIssueKind::Compute;
  llvm::SmallVector<uint32_t, 4> nodes;
  StructuredMaterializationRelations *relations = nullptr;
};

std::optional<TileId> getExecutionTile(const ExecutionInstanceId &execution) {
  return std::visit([](const auto &source) { return source.work.tile; },
                    execution.source);
}

std::optional<unsigned> getDDRSourceArgument(mlir::Operation *operation) {
  auto rdma = mlir::dyn_cast<InstrRDMAOp>(operation);
  if (!rdma)
    return std::nullopt;
  StorageRootMemo memo;
  const llvm::DenseSet<mlir::Value> &roots =
      memo.getStorageRoots(rdma.getSource());
  if (roots.size() != 1)
    return std::nullopt;
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(*roots.begin());
  if (!argument)
    return std::nullopt;
  mlir::Operation *parent = argument.getOwner()->getParentOp();
  if (!mlir::isa_and_nonnull<mlir::func::FuncOp>(parent))
    return std::nullopt;
  return argument.getArgNumber();
}

bool hasNode(const ActualIssue &issue, uint32_t node) {
  return llvm::is_contained(issue.nodes, node);
}

std::optional<uint64_t> getStaticTripCount(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
    return std::nullopt;
  const __int128 difference =
      static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
  if (difference <= 0)
    return std::nullopt;
  return 1 +
         (static_cast<uint64_t>(difference) - 1) / static_cast<uint64_t>(*step);
}

mlir::Operation *getLoopBodyOwner(mlir::Operation *operation,
                                  mlir::scf::ForOp loop) {
  while (operation && operation->getBlock() != loop.getBody())
    operation = operation->getParentOp();
  return operation;
}

struct ResultKey {
  TileId tile{0};
  uint32_t node = 0;
  unsigned result = 0;
  StructuredResultIdentityKind identityKind =
      StructuredResultIdentityKind::OperationResult;
  MemLayout layout = MemLayout::Tensor;

  friend bool operator==(const ResultKey &lhs, const ResultKey &rhs) {
    return lhs.tile == rhs.tile && lhs.node == rhs.node &&
           lhs.result == rhs.result && lhs.identityKind == rhs.identityKind &&
           lhs.layout == rhs.layout;
  }

  friend bool operator<(const ResultKey &lhs, const ResultKey &rhs) {
    return std::tuple(lhs.tile.getValue(), lhs.node, lhs.result,
                      lhs.identityKind, lhs.layout) <
           std::tuple(rhs.tile.getValue(), rhs.node, rhs.result,
                      rhs.identityKind, rhs.layout);
  }
};

struct ExistingPeerPair {
  MovementActionId action;
  uint32_t payloadSlice = 0;
  CommPeerSendOp send;
  CommPeerRecvOp receive;
};

struct ExistingExternalLoad {
  MovementActionId action;
  uint32_t payloadSlice = 0;
  StorageLoadOp load;
};

std::optional<unsigned> getTileLoadSourceArgument(StorageLoadOp load) {
  StorageRootMemo memo;
  const llvm::DenseSet<mlir::Value> &roots =
      memo.getStorageRoots(load.getSource());
  if (roots.size() != 1)
    return std::nullopt;
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(*roots.begin());
  if (!argument || !mlir::isa_and_nonnull<mlir::func::FuncOp>(
                       argument.getOwner()->getParentOp()))
    return std::nullopt;
  return argument.getArgNumber();
}

mlir::LogicalResult materializeSelectedPeerGraphs(
    const MovementPlan &movement,
    llvm::ArrayRef<MovementResourceDescription> resources,
    llvm::ArrayRef<CandidateExecutionNodeRelation> executionNodes,
    llvm::MutableArrayRef<CandidateTileDataflowIR> tiles,
    std::string &failureReason) {
  if (movement.peerGraphs.empty())
    return mlir::success();
  std::map<int64_t, CandidateTileDataflowIR *> tilesById;
  for (CandidateTileDataflowIR &tile : tiles)
    if (!tile.owner || !*tile.owner || !tile.relations ||
        !tilesById.try_emplace(tile.tile.getValue(), &tile).second) {
      failureReason = "selected peer graph has a malformed Tile domain";
      return mlir::failure();
    }
  std::map<MovementActionId, const MovementResourceDescription *>
      resourcesByAction;
  for (const MovementResourceDescription &resource : resources)
    if (!resourcesByAction.try_emplace(resource.action, &resource).second) {
      failureReason = "selected peer graph has duplicate movement resources";
      return mlir::failure();
    }
  std::map<RegionExecutionId, uint32_t> nodesByExecution;
  for (const CandidateExecutionNodeRelation &relation : executionNodes)
    nodesByExecution.emplace(relation.execution, relation.structuredNodeId);
  auto findVersionNode =
      [&](const PhysicalVersionId &version) -> std::optional<uint32_t> {
    std::optional<RegionExecutionId> execution;
    if (const auto *result =
            std::get_if<ExecutionResultValueId>(&version.logicalValue))
      execution = result->execution;
    else if (const auto *partial =
                 std::get_if<ReductionPartialValueId>(&version.logicalValue))
      execution = RegionExecutionId(partial->execution);
    else if (const auto *component =
                 std::get_if<CoupledComponentValueId>(&version.logicalValue))
      execution = RegionExecutionId(component->execution);
    auto node =
        execution ? nodesByExecution.find(*execution) : nodesByExecution.end();
    return node == nodesByExecution.end()
               ? std::nullopt
               : std::optional<uint32_t>(node->second);
  };
  auto findActionSourceNode =
      [&](const MovementActionId &action) -> std::optional<uint32_t> {
    if (const auto *transfer = std::get_if<DDRBoundaryTransferId>(&action)) {
      auto plan =
          llvm::find_if(movement.ddrTransfers, [&](const auto &candidate) {
            return candidate.id == *transfer;
          });
      return plan == movement.ddrTransfers.end()
                 ? std::nullopt
                 : findVersionNode(plan->source);
    }
    if (const auto *gather = std::get_if<ReductionGatherId>(&action)) {
      auto plan =
          llvm::find_if(movement.reductionGathers, [&](const auto &candidate) {
            return candidate.id == *gather;
          });
      return plan == movement.reductionGathers.end()
                 ? std::nullopt
                 : findVersionNode(plan->source);
    }
    return std::nullopt;
  };

  auto findPair =
      [&](const MovementActionId &action,
          uint32_t payloadSlice) -> std::optional<ExistingPeerPair> {
    auto resource = resourcesByAction.find(action);
    if (resource == resourcesByAction.end() || !resource->second->sourceTile ||
        !resource->second->destinationTile)
      return std::nullopt;
    auto sourceTile = tilesById.find(resource->second->sourceTile->getValue());
    auto destinationTile =
        tilesById.find(resource->second->destinationTile->getValue());
    if (sourceTile == tilesById.end() || destinationTile == tilesById.end())
      return std::nullopt;
    llvm::SmallVector<CommPeerSendOp, 2> sends;
    llvm::SmallVector<CommPeerRecvOp, 2> receives;
    sourceTile->second->getModule().walk([&](CommPeerSendOp send) {
      if (send.getPeerAttr().getInt() ==
              resource->second->destinationTile->getValue() &&
          send.getMessageAttr().getPayloadSlice() == payloadSlice)
        sends.push_back(send);
    });
    destinationTile->second->getModule().walk([&](CommPeerRecvOp receive) {
      if (receive.getPeerAttr().getInt() ==
              resource->second->sourceTile->getValue() &&
          receive.getMessageAttr().getPayloadSlice() == payloadSlice)
        receives.push_back(receive);
    });
    llvm::SmallVector<ExistingPeerPair, 2> matches;
    for (CommPeerSendOp send : sends)
      for (CommPeerRecvOp receive : receives)
        if (send.getMessageAttr() == receive.getMessageAttr())
          matches.push_back({action, payloadSlice, send, receive});
    return matches.size() == 1
               ? std::optional<ExistingPeerPair>(matches.front())
               : std::nullopt;
  };

  auto findExternalLoad = [&](const MovementActionId &action,
                              uint32_t payloadSlice,
                              llvm::ArrayRef<int64_t> expectedShape)
      -> std::optional<ExistingExternalLoad> {
    const auto *loadId = std::get_if<ExternalLoadId>(&action);
    auto resource = resourcesByAction.find(action);
    if (!loadId || resource == resourcesByAction.end() ||
        resource->second->sourceTile || !resource->second->destinationTile ||
        loadId->destination.fragment.source.kind !=
            analysis::RootBoundaryKind::ProgramInput)
      return std::nullopt;
    auto tile = tilesById.find(resource->second->destinationTile->getValue());
    if (tile == tilesById.end())
      return std::nullopt;
    llvm::SmallVector<StorageLoadOp, 2> matches;
    tile->second->getModule().walk([&](StorageLoadOp load) {
      std::optional<unsigned> argument = getTileLoadSourceArgument(load);
      auto type = mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
      if (argument && *argument == loadId->destination.fragment.source.index &&
          type && llvm::equal(type.getShape(), expectedShape))
        matches.push_back(load);
    });
    return matches.size() == 1
               ? std::optional<ExistingExternalLoad>(ExistingExternalLoad{
                     action, payloadSlice, matches.front()})
               : std::nullopt;
  };

  std::vector<ExistingPeerPair> oldPairs;
  std::vector<ExistingExternalLoad> oldLoads;
  std::vector<SelectedPeerGraphBinding> bindings;
  std::vector<std::unique_ptr<mlir::OpBuilder>> builders;
  for (const PeerTransferGraphPlan &graph : movement.peerGraphs) {
    if (graph.actions.empty()) {
      failureReason = "selected peer graph has no action resource";
      return mlir::failure();
    }
    auto resource = resourcesByAction.find(graph.actions.front());
    if (resource == resourcesByAction.end()) {
      failureReason = "selected peer graph has no action resource";
      return mlir::failure();
    }
    auto normalized =
        analysis::normalizeFiniteExactIndexSet(resource->second->exactDomain);
    if (mlir::failed(normalized)) {
      failureReason = "selected peer graph payload is not a finite box union";
      return mlir::failure();
    }
    std::vector<TileId> terminals;
    for (const MovementActionId &action : graph.actions) {
      auto actionResource = resourcesByAction.find(action);
      if (actionResource == resourcesByAction.end() ||
          !actionResource->second->destinationTile) {
        failureReason = "selected peer action has no terminal Tile";
        return mlir::failure();
      }
      terminals.push_back(*actionResource->second->destinationTile);
    }
    llvm::sort(terminals, [](TileId lhs, TileId rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    for (auto [payloadSlice, box] : llvm::enumerate(normalized->getBoxes())) {
      std::vector<ExistingPeerPair> graphPairs;
      std::vector<ExistingExternalLoad> graphLoads;
      if (graph.kind == PeerTransferGraphKind::ExternalLoadFanout) {
        for (const MovementActionId &action : graph.actions) {
          std::optional<ExistingExternalLoad> load = findExternalLoad(
              action, static_cast<uint32_t>(payloadSlice), box.sizes);
          if (!load) {
            llvm::raw_string_ostream diagnostic(failureReason);
            diagnostic << "selected external fanout has no unique DDR load "
                          "donor; expected_shape=[";
            llvm::interleaveComma(box.sizes, diagnostic);
            diagnostic << "] candidates=[";
            auto actionResource = resourcesByAction.find(action);
            auto tile =
                actionResource == resourcesByAction.end() ||
                        !actionResource->second->destinationTile
                    ? tilesById.end()
                    : tilesById.find(
                          actionResource->second->destinationTile->getValue());
            bool first = true;
            if (tile != tilesById.end())
              tile->second->getModule().walk([&](StorageLoadOp candidate) {
                if (!first)
                  diagnostic << ';';
                first = false;
                std::optional<unsigned> argument =
                    getTileLoadSourceArgument(candidate);
                diagnostic << "arg=";
                if (argument)
                  diagnostic << *argument;
                else
                  diagnostic << "unknown";
                diagnostic << ",type=" << candidate.getDest().getType()
                           << ",roots=[";
                StorageRootMemo memo;
                bool firstRoot = true;
                for (mlir::Value root :
                     memo.getStorageRoots(candidate.getSource())) {
                  if (!firstRoot)
                    diagnostic << ',';
                  firstRoot = false;
                  if (auto blockArgument =
                          mlir::dyn_cast<mlir::BlockArgument>(root)) {
                    diagnostic << "block-arg:" << blockArgument.getArgNumber()
                               << "@";
                    if (mlir::Operation *parent =
                            blockArgument.getOwner()->getParentOp())
                      diagnostic << parent->getName();
                  } else if (mlir::Operation *definition =
                                 root.getDefiningOp()) {
                    diagnostic << definition->getName();
                  } else {
                    diagnostic << "unknown";
                  }
                }
                diagnostic << "],op=";
                candidate->print(diagnostic);
              });
            diagnostic << ']';
            return mlir::failure();
          }
          graphLoads.push_back(*load);
        }
      } else {
        for (const MovementActionId &action : graph.actions) {
          std::optional<ExistingPeerPair> pair =
              findPair(action, static_cast<uint32_t>(payloadSlice));
          if (!pair) {
            failureReason =
                "selected peer action has no unique direct donor pair";
            return mlir::failure();
          }
          graphPairs.push_back(*pair);
        }
      }
      std::set<int64_t> active;
      std::set<int64_t> destinations;
      for (const MovementHop &hop : graph.hops) {
        active.insert(hop.source.getValue());
        active.insert(hop.destination.getValue());
        destinations.insert(hop.destination.getValue());
      }
      std::vector<TileId> roots;
      for (int64_t tile : active)
        if (!destinations.count(tile))
          roots.push_back(TileId(tile));
      if (roots.size() != 1) {
        failureReason = "selected peer graph has no unique root";
        return mlir::failure();
      }
      mlir::Value rootBuffer;
      mlir::Value ddrSource;
      StorageLoadOp rootLoad;
      if (graph.kind == PeerTransferGraphKind::ExternalLoadFanout) {
        if (!graph.ddrRoot) {
          failureReason = "selected external fanout has no DDR root action";
          return mlir::failure();
        }
        for (ExistingExternalLoad &load : graphLoads)
          if (load.action == MovementActionId(*graph.ddrRoot)) {
            rootBuffer = load.load.getDest();
            ddrSource = load.load.getSource();
            rootLoad = load.load;
            break;
          }
      } else {
        for (ExistingPeerPair &pair : graphPairs) {
          auto actionResource = resourcesByAction.find(pair.action);
          if (actionResource->second->sourceTile == roots.front()) {
            if (rootBuffer && rootBuffer != pair.send.getBuffer()) {
              failureReason =
                  "selected peer graph actions do not share one root buffer";
              return mlir::failure();
            }
            rootBuffer = pair.send.getBuffer();
          }
        }
      }
      if (!rootBuffer) {
        failureReason = "selected peer graph has no actual root buffer";
        return mlir::failure();
      }
      std::vector<MovementEndpointBinding> endpoints;
      for (int64_t tileId : active) {
        TileId tile(tileId);
        auto owner = tilesById.find(tileId);
        if (owner == tilesById.end()) {
          failureReason = "selected peer graph references an absent Tile";
          return mlir::failure();
        }
        mlir::Value buffer;
        mlir::Operation *anchor = nullptr;
        if (tile == roots.front()) {
          buffer = rootBuffer;
          anchor = graph.kind == PeerTransferGraphKind::ExternalLoadFanout
                       ? rootLoad.getOperation()
                       : graphPairs.front().send.getOperation();
        } else {
          if (graph.kind == PeerTransferGraphKind::ExternalLoadFanout) {
            for (ExistingExternalLoad &load : graphLoads) {
              auto actionResource = resourcesByAction.find(load.action);
              if (actionResource->second->destinationTile == tile) {
                buffer = load.load.getDest();
                anchor = load.load;
                break;
              }
            }
          } else {
            for (ExistingPeerPair &pair : graphPairs) {
              auto actionResource = resourcesByAction.find(pair.action);
              if (actionResource->second->destinationTile == tile) {
                buffer = pair.receive.getBuffer();
                anchor = pair.receive;
                break;
              }
            }
          }
        }
        if (!buffer) {
          auto functions =
              owner->second->getModule().getOps<mlir::func::FuncOp>();
          if (functions.empty()) {
            failureReason = "relay Tile has no insertion block";
            return mlir::failure();
          }
          mlir::func::FuncOp function = *functions.begin();
          if (function.getBody().empty()) {
            failureReason = "relay Tile has no insertion block";
            return mlir::failure();
          }
          mlir::Block &entry = function.getBody().front();
          mlir::OpBuilder functionBuilder(entry.getTerminator());
          auto region = functionBuilder.create<TileRegionOp>(
              rootBuffer.getLoc(), mlir::TypeRange{}, mlir::ValueRange{});
          region.getBody().push_back(new mlir::Block());
          mlir::OpBuilder regionBuilder =
              mlir::OpBuilder::atBlockEnd(&region.getBody().front());
          auto yield = regionBuilder.create<TileYieldOp>(rootBuffer.getLoc());
          auto builder = std::make_unique<mlir::OpBuilder>(yield);
          auto type = mlir::dyn_cast<mlir::MemRefType>(rootBuffer.getType());
          if (!type) {
            failureReason = "selected peer root is not one memref payload";
            return mlir::failure();
          }
          auto allocation =
              builder->create<mlir::memref::AllocOp>(rootBuffer.getLoc(), type);
          buffer = allocation;
          anchor = yield;
          std::optional<uint32_t> node =
              findActionSourceNode(graph.actions.front());
          if (!node) {
            failureReason = "relay allocation has no typed action owner";
            return mlir::failure();
          }
          owner->second->relations->scratchBuffers.push_back({*node, buffer});
          builders.push_back(std::move(builder));
        }
        auto builder = std::make_unique<mlir::OpBuilder>(anchor);
        endpoints.push_back({tile, buffer, builder.get()});
        builders.push_back(std::move(builder));
      }
      SelectedPeerGraphBinding binding{graph.actions, terminals, endpoints};
      binding.ddrSource = ddrSource;
      binding.payloadSlice = static_cast<uint32_t>(payloadSlice);
      binding.logicalOffsets = box.offsets;
      binding.logicalSizes = box.sizes;
      bindings.push_back(std::move(binding));
      oldPairs.insert(oldPairs.end(), graphPairs.begin(), graphPairs.end());
      oldLoads.insert(oldLoads.end(), graphLoads.begin(), graphLoads.end());
    }
  }
  std::sort(oldPairs.begin(), oldPairs.end(),
            [](const auto &lhs, const auto &rhs) {
              return std::tie(lhs.action, lhs.payloadSlice) <
                     std::tie(rhs.action, rhs.payloadSlice);
            });
  auto duplicate = std::adjacent_find(
      oldPairs.begin(), oldPairs.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.send == rhs.send || lhs.receive == rhs.receive;
      });
  if (duplicate != oldPairs.end()) {
    failureReason = "selected peer graphs reuse one direct donor endpoint";
    return mlir::failure();
  }
  llvm::DenseSet<mlir::Operation *> uniqueOldLoads;
  for (ExistingExternalLoad &load : oldLoads)
    if (!load.load || !uniqueOldLoads.insert(load.load).second) {
      failureReason = "selected external fanout reuses one DDR load donor";
      return mlir::failure();
    }
  auto prepared =
      prepareSelectedPeerGraphs(movement, resources, bindings, &failureReason);
  if (mlir::failed(prepared))
    return mlir::failure();
  auto emitted = emitPreparedSelectedPeerGraphs(*prepared, &failureReason);
  if (mlir::failed(emitted) || mlir::failed(verifyTokenOnlySelectedPeerGraphs(
                                   *prepared, *emitted, &failureReason)))
    return mlir::failure();

  llvm::DenseSet<mlir::Operation *> oldAwaits;
  for (ExistingPeerPair &pair : oldPairs) {
    for (mlir::Value token : {pair.send.getToken(), pair.receive.getToken()})
      for (mlir::Operation *user : token.getUsers()) {
        if (!mlir::isa<mlir::async::AwaitOp>(user)) {
          failureReason =
              "direct peer donor token has a non-await actual consumer";
          return mlir::failure();
        }
        oldAwaits.insert(user);
      }
  }
  for (mlir::Operation *await : oldAwaits)
    await->erase();
  for (ExistingExternalLoad &load : oldLoads)
    load.load.erase();
  for (ExistingPeerPair &pair : oldPairs) {
    pair.send.erase();
    pair.receive.erase();
  }
  return mlir::success();
}

} // namespace

PreparedCandidateEvents prepareCandidateEvents(const EventGraph &eventGraph) {
  PreparedCandidateEvents prepared;
  prepared.events.assign(eventGraph.getEvents().begin(),
                         eventGraph.getEvents().end());
  prepared.hardDependencies.assign(eventGraph.getHardDependencies().begin(),
                                   eventGraph.getHardDependencies().end());
  prepared.completionObligations.assign(
      eventGraph.getCompletionObligations().begin(),
      eventGraph.getCompletionObligations().end());
  return prepared;
}

mlir::LogicalResult CompleteCandidatePreparation::prepareTileDataflow(
    llvm::MutableArrayRef<CandidateTileDataflowIR> tiles,
    CardExecutablePreparationFailure &failure) {
  failure = {};
  if (tiles.empty()) {
    failure.detail = "complete candidate has no Tile dataflow IR";
    return mlir::failure();
  }
  if (mlir::failed(materializeSelectedPeerGraphs(
          movement, movementResources, executionNodes, tiles, failure.detail)))
    return mlir::failure();
  std::map<RegionExecutionId, uint32_t> nodesByExecution;
  for (const CandidateExecutionNodeRelation &relation : executionNodes)
    if (!nodesByExecution
             .try_emplace(relation.execution, relation.structuredNodeId)
             .second) {
      failure.detail = "candidate execution/node relation is duplicated";
      return mlir::failure();
    }
  std::map<PhysicalVersionId, const PhysicalVersionPlan *> versions;
  for (const PhysicalVersionPlan &version : representations.physicalVersions)
    if (!versions.try_emplace(version.id, &version).second) {
      failure.detail = "candidate representation has duplicate versions";
      return mlir::failure();
    }
  std::map<PhysicalVersionId, TileId> versionTiles;
  std::map<StorageObjectId, TileId> objectTiles;
  for (const StorageObjectPlan &object : buffers.storageObjects)
    if (!objectTiles.try_emplace(object.id, object.tile).second) {
      failure.detail = "candidate storage has duplicate objects";
      return mlir::failure();
    }
  for (const PhysicalVersionStorageBinding &binding : buffers.versionBindings) {
    auto object = objectTiles.find(binding.object);
    if (!versions.count(binding.version) || object == objectTiles.end() ||
        !versionTiles.try_emplace(binding.version, object->second).second) {
      failure.detail = "candidate storage/version binding is incomplete";
      return mlir::failure();
    }
  }

  std::map<ResultKey, size_t> expected;
  for (const PhysicalVersionPlan &version : representations.physicalVersions) {
    std::optional<RegionExecutionId> execution;
    unsigned result = 0;
    bool coupledComponent = false;
    if (const auto *value =
            std::get_if<ExecutionResultValueId>(&version.id.logicalValue)) {
      execution = value->execution;
      result = value->result;
    } else if (const auto *value = std::get_if<ReductionPartialValueId>(
                   &version.id.logicalValue)) {
      execution = RegionExecutionId(value->execution);
      result = value->result;
    } else if (const auto *value = std::get_if<CoupledComponentValueId>(
                   &version.id.logicalValue)) {
      execution = RegionExecutionId(value->execution);
      result = static_cast<unsigned>(value->component);
      coupledComponent = true;
    } else {
      continue;
    }
    auto node = nodesByExecution.find(*execution);
    auto tile = versionTiles.find(version.id);
    if (node == nodesByExecution.end() || tile == versionTiles.end()) {
      failure.detail =
          "candidate result version has no unique execution/storage owner";
      return mlir::failure();
    }
    ResultKey key{tile->second, node->second, result,
                  coupledComponent
                      ? StructuredResultIdentityKind::CoupledReductionComponent
                      : StructuredResultIdentityKind::OperationResult,
                  version.encoding};
    if (!coupledComponent && expected.count(key)) {
      failure.detail =
          "candidate result version has no unique execution/storage owner";
      return mlir::failure();
    }
    ++expected[key];
  }

  std::map<ResultKey, size_t> actual;
  for (CandidateTileDataflowIR &tile : tiles) {
    if (!tile.owner || !*tile.owner || !tile.relations) {
      failure.detail = "candidate Tile dataflow relation is stale";
      return mlir::failure();
    }
    for (const StructuredOperationResultBufferRelation &relation :
         tile.relations->operationResultBuffers) {
      auto type = mlir::dyn_cast<mlir::MemRefType>(relation.buffer.getType());
      MemoryAttr memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
      if (!type || !memory || memory.getSpace() != MemorySpace::SPM)
        continue;
      ResultKey key{tile.tile, relation.structuredNodeId, relation.resultIndex,
                    relation.identityKind, memory.getLayout()};
      auto required = expected.find(key);
      if (required != expected.end() && actual[key] < required->second)
        ++actual[key];
    }
  }
  if (actual != expected) {
    llvm::raw_string_ostream diagnostic(failure.detail);
    diagnostic << "actual result/version coverage differs from "
                  "RepresentationPlan; missing=[";
    bool first = true;
    for (const auto &[key, required] : expected) {
      const size_t observed = actual[key];
      if (observed >= required)
        continue;
      if (!first)
        diagnostic << ';';
      first = false;
      diagnostic << "tile=" << key.tile.getValue() << ",node=" << key.node
                 << ",result=" << key.result
                 << ",result-kind=" << static_cast<unsigned>(key.identityKind)
                 << ",layout=" << static_cast<unsigned>(key.layout)
                 << ",required=" << required << ",actual=" << observed;
    }
    diagnostic << ']';
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult CompleteCandidatePreparation::prepareInstructionIR(
    llvm::MutableArrayRef<CandidateInstructionIR> tiles,
    CardExecutablePreparationFailure &failure) {
  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "planning-materialization-phase", "complete-candidate",
      "collect-current-instruction-relations");
  failure = {};
  std::string &failureReason = failure.detail;
  failureReason.clear();
  if (tiles.empty() || !scheduleDomain.contains(schedule)) {
    failureReason = "complete candidate has a stale selected schedule";
    return mlir::failure();
  }

  std::map<int64_t, CandidateInstructionIR *> tilesById;
  for (CandidateInstructionIR &tile : tiles)
    if (!tile.owner || !*tile.owner || !tile.relations ||
        !tilesById.try_emplace(tile.tile.getValue(), &tile).second) {
      failureReason = "complete candidate instruction Tile domain is malformed";
      return mlir::failure();
    }
  std::map<ExecutionInstanceId, uint32_t> nodesByExecution;
  std::map<analysis::RootRegionWorkId, std::set<uint32_t>> nodesByWork;
  for (const CandidateExecutionNodeRelation &relation : executionNodes) {
    const auto *execution =
        std::get_if<ExecutionInstanceId>(&relation.execution);
    if (!execution ||
        !nodesByExecution.try_emplace(*execution, relation.structuredNodeId)
             .second) {
      failureReason =
          "complete candidate execution/node relation is incomplete";
      return mlir::failure();
    }
    analysis::RootRegionWorkId work = std::visit(
        [](const auto &source) { return source.work; }, execution->source);
    nodesByWork[work].insert(relation.structuredNodeId);
  }

  std::map<RegionExecutionId, uint32_t> nodesByRegionExecution;
  for (const CandidateExecutionNodeRelation &relation : executionNodes)
    nodesByRegionExecution.emplace(relation.execution,
                                   relation.structuredNodeId);

  // Capture the current instruction-to-execution identity before selected
  // alias/reuse or rotation intentionally gives several versions one storage
  // root. The instruction operations remain live through those storage-only
  // rewrites, while a post-rewrite root query would lose the distinction
  // between non-overlapping logical owners.
  std::vector<ActualIssue> actualIssues;
  for (CandidateInstructionIR &tile : tiles) {
    mlir::ModuleOp module = tile.getModule();
    StructuredNodeUseIndex nodeUses(*tile.relations);
    module.walk([&](mlir::Operation *operation) {
      std::optional<ActualIssueKind> kind;
      if (mlir::isa<InstrGatherScatterOp>(operation))
        kind = ActualIssueKind::LocalCombine;
      else if (mlir::isa<InstrRDMAOp>(operation))
        kind = ActualIssueKind::DDRLoad;
      else if (mlir::isa<InstrWDMAOp>(operation))
        kind = ActualIssueKind::DDRStore;
      else if (mlir::isa<InstrDTESendOp>(operation))
        kind = ActualIssueKind::PeerSend;
      else if (mlir::isa<InstrDTERecvOp>(operation))
        kind = ActualIssueKind::PeerReceive;
      else if (mlir::isa<WaferNCCIssueOpInterface>(operation))
        kind = ActualIssueKind::Compute;
      if (!kind)
        return;
      llvm::SmallVector<uint32_t, 4> nodes =
          nodeUses.collectNodesUsedBy(operation);
      if (nodes.empty() && tile.regionNodes) {
        TileRegionOp region = operation->getParentOfType<TileRegionOp>();
        if (region) {
          auto relation =
              llvm::find_if(*tile.regionNodes, [&](const auto &candidate) {
                return candidate.region == region.getOperation();
              });
          if (relation != tile.regionNodes->end())
            nodes.assign(relation->structuredNodes.begin(),
                         relation->structuredNodes.end());
        }
      }
      actualIssues.push_back(
          {tile.tile, operation, *kind, std::move(nodes), tile.relations});
    });
  }

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "planning-materialization-phase", "complete-candidate",
      "materialize-selected-storage");

  std::map<PhysicalVersionId, const PhysicalVersionPlan *> versionPlans;
  for (const PhysicalVersionPlan &version : representations.physicalVersions)
    versionPlans.emplace(version.id, &version);
  std::map<StorageObjectId, std::vector<PhysicalVersionStorageBinding>>
      bindingsByObject;
  for (const PhysicalVersionStorageBinding &binding : buffers.versionBindings)
    bindingsByObject[binding.object].push_back(binding);
  std::map<StorageObjectId, TileId> selectedObjectTiles;
  for (const StorageObjectPlan &object : buffers.storageObjects)
    selectedObjectTiles.emplace(object.id, object.tile);
  std::map<PhysicalVersionId, TileId> selectedVersionTiles;
  for (const PhysicalVersionStorageBinding &binding : buffers.versionBindings) {
    auto tile = selectedObjectTiles.find(binding.object);
    if (tile != selectedObjectTiles.end())
      selectedVersionTiles.emplace(binding.version, tile->second);
  }
  auto getVersionResult = [&](const PhysicalVersionId &version)
      -> std::optional<
          std::tuple<uint32_t, unsigned, StructuredResultIdentityKind>> {
    if (const auto *result =
            std::get_if<ExecutionResultValueId>(&version.logicalValue)) {
      auto node = nodesByRegionExecution.find(result->execution);
      if (node != nodesByRegionExecution.end())
        return std::make_tuple(node->second, result->result,
                               StructuredResultIdentityKind::OperationResult);
    }
    if (const auto *partial =
            std::get_if<ReductionPartialValueId>(&version.logicalValue)) {
      auto node =
          nodesByRegionExecution.find(RegionExecutionId(partial->execution));
      if (node != nodesByRegionExecution.end())
        return std::make_tuple(node->second, partial->result,
                               StructuredResultIdentityKind::OperationResult);
    }
    if (const auto *component =
            std::get_if<CoupledComponentValueId>(&version.logicalValue)) {
      auto node =
          nodesByRegionExecution.find(RegionExecutionId(component->execution));
      if (node != nodesByRegionExecution.end())
        return std::make_tuple(
            node->second, static_cast<unsigned>(component->component),
            StructuredResultIdentityKind::CoupledReductionComponent);
    }
    return std::nullopt;
  };
  auto findVersionBuffers = [&](const PhysicalVersionId &version) {
    llvm::SmallVector<mlir::Value, 4> values;
    auto versionPlan = versionPlans.find(version);
    if (versionPlan == versionPlans.end())
      return values;
    if (const auto *boundary =
            std::get_if<BoundaryRegionValueId>(&version.logicalValue)) {
      auto expectedNodes = nodesByWork.find(boundary->work);
      if (boundary->fragment.source.kind !=
              analysis::RootBoundaryKind::ProgramInput ||
          expectedNodes == nodesByWork.end() ||
          expectedNodes->second.size() != 1)
        return values;
      const uint32_t expectedNode = *expectedNodes->second.begin();
      for (CandidateInstructionIR &tile : tiles) {
        if (tile.tile != boundary->work.tile)
          continue;
        tile.getModule().walk([&](InstrRDMAOp rdma) {
          std::optional<unsigned> argument =
              getDDRSourceArgument(rdma.getOperation());
          if (!argument || *argument != boundary->fragment.source.index ||
              !operationUsesStructuredNode(rdma, expectedNode, *tile.relations))
            return;
          auto type =
              mlir::dyn_cast<mlir::MemRefType>(rdma.getDest().getType());
          MemoryAttr memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
          if (!memory || memory.getLayout() != versionPlan->second->encoding)
            return;
          StorageRootMemo memo;
          for (mlir::Value root : memo.getStorageRoots(rdma.getDest()))
            if (!llvm::is_contained(values, root))
              values.push_back(root);
        });
      }
      return values;
    }
    auto result = getVersionResult(version);
    auto selectedTile = selectedVersionTiles.find(version);
    if (!result || selectedTile == selectedVersionTiles.end())
      return values;
    for (CandidateInstructionIR &tile : tiles)
      if (tile.tile == selectedTile->second)
        for (const StructuredOperationResultBufferRelation &relation :
             tile.relations->operationResultBuffers) {
          auto type =
              mlir::dyn_cast<mlir::MemRefType>(relation.buffer.getType());
          MemoryAttr memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
          if (relation.structuredNodeId == std::get<0>(*result) &&
              relation.resultIndex == std::get<1>(*result) &&
              relation.identityKind == std::get<2>(*result) && memory &&
              memory.getLayout() == versionPlan->second->encoding &&
              !llvm::is_contained(values, relation.buffer))
            values.push_back(relation.buffer);
        }
    return values;
  };
  auto retargetRelations = [&](mlir::Value from, mlir::Value to) {
    auto retarget = [&](auto &entries) {
      for (auto &entry : entries)
        if (entry.buffer == from)
          entry.buffer = to;
    };
    for (CandidateInstructionIR &tile : tiles) {
      retarget(tile.relations->operationResultBuffers);
      retarget(tile.relations->operandBuffers);
      retarget(tile.relations->scratchBuffers);
      retarget(tile.relations->outputBuffers);
      retarget(tile.relations->cardDDRBuffers);
      retarget(tile.relations->partialReductionContributions);
      retarget(tile.relations->partialReductionMergeInputs);
    }
  };
  for (const auto &[object, objectBindings] : bindingsByObject) {
    (void)object;
    if (objectBindings.size() < 2 ||
        llvm::none_of(objectBindings, [](const auto &binding) {
          return binding.kind != StorageBindingKind::Fresh;
        }))
      continue;
    llvm::SmallVector<mlir::Value, 8> roots;
    for (const PhysicalVersionStorageBinding &binding : objectBindings) {
      llvm::SmallVector<mlir::Value, 4> values =
          findVersionBuffers(binding.version);
      if (values.empty()) {
        failure.kind = CardExecutablePreparationFailureKind::Unsupported;
        failureReason =
            "selected alias/reuse object has no current result-version root";
        return mlir::failure();
      }
      for (mlir::Value value : values)
        if (!llvm::is_contained(roots, value))
          roots.push_back(value);
    }
    if (roots.size() < 2)
      continue;
    auto type = mlir::dyn_cast<mlir::MemRefType>(roots.front().getType());
    mlir::ModuleOp module;
    for (CandidateInstructionIR &tile : tiles)
      if (tile.getModule()->isAncestor(roots.front().getDefiningOp())) {
        module = tile.getModule();
        break;
      }
    if (!type || !module) {
      failureReason =
          "selected alias/reuse roots have no common typed Instr module";
      return mlir::failure();
    }
    TileRegionOp storageRegion =
        roots.front().getDefiningOp()
            ? roots.front().getDefiningOp()->getParentOfType<TileRegionOp>()
            : TileRegionOp{};
    if (!storageRegion || storageRegion.getBody().empty() ||
        llvm::any_of(roots, [&](mlir::Value value) {
          return value.getType() != type || !value.getDefiningOp() ||
                 !module->isAncestor(value.getDefiningOp()) ||
                 value.getDefiningOp()->getParentOfType<TileRegionOp>() !=
                     storageRegion;
        })) {
      failure.kind = CardExecutablePreparationFailureKind::Unsupported;
      failureReason =
          "selected alias/reuse roots cannot share one Tile-region allocation";
      return mlir::failure();
    }
    mlir::OpBuilder builder =
        mlir::OpBuilder::atBlockBegin(&storageRegion.getBody().front());
    mlir::Value shared =
        builder.create<mlir::memref::AllocOp>(roots.front().getLoc(), type);
    for (mlir::Value root : roots) {
      llvm::SmallVector<mlir::memref::DeallocOp, 2> deallocations;
      for (mlir::Operation *user : root.getUsers())
        if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(user))
          deallocations.push_back(dealloc);
      for (mlir::memref::DeallocOp dealloc : deallocations)
        dealloc.erase();
      root.replaceAllUsesWith(shared);
      retargetRelations(root, shared);
      if (auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
          allocation && allocation->use_empty())
        allocation.erase();
    }
  }

  std::map<PipelineScopeId, std::vector<ExternalStorageStageProof>>
      storageProofsByScope;
  for (const SlotFamilyPlan &family : buffers.slotFamilies) {
    if (family.multiplicity <= 1)
      continue;
    const auto *pipeline = [&]() -> const PipelinedExecutionStructure * {
      for (const ExecutionStructureChoice &choice : structure.scopes)
        if (const auto *candidate =
                std::get_if<PipelinedExecutionStructure>(&choice))
          if (candidate->recurrence == family.occurrence)
            return candidate;
      return nullptr;
    }();
    if (!pipeline || family.id.objects.empty()) {
      failureReason = "rotating storage family has no selected pipeline";
      return mlir::failure();
    }
    for (const StorageObjectId &object : family.id.objects) {
      auto objectBindings = bindingsByObject.find(object);
      auto resource = llvm::find_if(
          storageResources, [&](const StorageResourceDescription &candidate) {
            return candidate.object == object;
          });
      if (objectBindings == bindingsByObject.end() ||
          resource == storageResources.end()) {
        failureReason = "rotating storage object has no binding or resource";
        return mlir::failure();
      }
      llvm::SmallVector<mlir::Value, 4> roots;
      for (const PhysicalVersionStorageBinding &binding :
           objectBindings->second)
        for (mlir::Value value : findVersionBuffers(binding.version))
          if (!llvm::is_contained(roots, value))
            roots.push_back(value);
      if (roots.size() != 1) {
        failure.kind = CardExecutablePreparationFailureKind::Unsupported;
        llvm::raw_string_ostream diagnostic(failureReason);
        diagnostic
            << "rotating storage object has no unique current allocation root"
            << " origin_kind=" << object.origin.index()
            << " versions=" << objectBindings->second.size()
            << " roots=" << roots.size();
        return mlir::failure();
      }
      mlir::Value oldRoot = roots.front();
      auto type = mlir::dyn_cast<mlir::MemRefType>(oldRoot.getType());
      mlir::ModuleOp module;
      auto objectTile = selectedObjectTiles.find(object);
      for (CandidateInstructionIR &tile : tiles)
        if (objectTile != selectedObjectTiles.end() &&
            tile.tile == objectTile->second) {
          module = tile.getModule();
          break;
        }
      llvm::SmallVector<mlir::scf::ForOp, 2> loops;
      if (module)
        module.walk([&](mlir::scf::ForOp loop) {
          bool touches = false;
          loop.walk([&](mlir::Operation *operation) {
            touches |= llvm::any_of(
                operation->getOperands(), [&](mlir::Value operand) {
                  return shareStructuredBufferStorage(operand, oldRoot);
                });
          });
          if (getStaticTripCount(loop) ==
                  std::optional<uint64_t>(
                      pipeline->iteration.steadyTripCount) &&
              touches)
            loops.push_back(loop);
        });
      TileRegionOp storageRegion =
          oldRoot.getDefiningOp()
              ? oldRoot.getDefiningOp()->getParentOfType<TileRegionOp>()
              : TileRegionOp{};
      if (!type || !storageRegion || storageRegion.getBody().empty() ||
          loops.size() != 1 || !storageRegion->isAncestor(loops.front())) {
        failure.kind = CardExecutablePreparationFailureKind::Unsupported;
        llvm::raw_string_ostream diagnostic(failureReason);
        diagnostic
            << "rotating storage has no unique actual steady-loop allocation"
            << " expected_trip=" << pipeline->iteration.steadyTripCount
            << " matching_loops=" << loops.size() << " actual_trips=[";
        uint64_t actualLoopCount = 0;
        if (module)
          module.walk([&](mlir::scf::ForOp loop) {
            ++actualLoopCount;
            std::optional<uint64_t> trip = getStaticTripCount(loop);
            if (trip)
              diagnostic << *trip << ',';
          });
        diagnostic << "] actual_loop_count=" << actualLoopCount;
        return mlir::failure();
      }
      mlir::OpBuilder entryBuilder =
          mlir::OpBuilder::atBlockBegin(&storageRegion.getBody().front());
      std::vector<mlir::Value> slots;
      for (uint32_t slot = 0; slot < family.multiplicity; ++slot)
        slots.push_back(
            entryBuilder.create<mlir::memref::AllocOp>(oldRoot.getLoc(), type));
      PreparedStoragePlan preparedStorage;
      auto objectPlan = llvm::find_if(buffers.storageObjects,
                                      [&](const StorageObjectPlan &candidate) {
                                        return candidate.id == object;
                                      });
      if (objectPlan == buffers.storageObjects.end()) {
        failureReason = "rotating storage object has no selected plan";
        return mlir::failure();
      }
      preparedStorage.objects.push_back(
          {*objectPlan, *resource, family.multiplicity});
      preparedStorage.bindings = objectBindings->second;
      preparedStorage.slotFamilies = {family};
      StorageObjectBuilder storageObjects;
      if (mlir::failed(
              bindPreparedStorageObjects(preparedStorage, {{object, slots}},
                                         storageObjects, &failureReason)))
        return mlir::failure();
      mlir::OpBuilder loopBuilder =
          mlir::OpBuilder::atBlockBegin(loops.front().getBody());
      auto coordinate = buildSteadyOccurrenceCoordinate(
          *pipeline, loops.front(), loopBuilder, &failureReason);
      if (mlir::failed(coordinate))
        return mlir::failure();
      auto selection =
          storageObjects.select(object, pipeline->iteration.recurrenceAxis,
                                *coordinate, loopBuilder, &failureReason);
      if (mlir::failed(selection))
        return mlir::failure();
      for (mlir::Operation *user :
           llvm::make_early_inc_range(oldRoot.getUsers())) {
        if (mlir::isa<mlir::memref::DeallocOp>(user)) {
          user->erase();
          continue;
        }
        mlir::Value replacement;
        if (loops.front()->isAncestor(user)) {
          replacement = selection->value;
        } else if (user->getBlock() == loops.front()->getBlock()) {
          replacement =
              user->isBeforeInBlock(loops.front())
                  ? storageObjects.lookup(object, 0)
                  : storageObjects.lookup(
                        object, pipeline->recurrence.axisOccurrences
                                        [pipeline->iteration.recurrenceAxis] -
                                    1);
        } else {
          failure.kind = CardExecutablePreparationFailureKind::Unsupported;
          failureReason =
              "rotating storage use is outside prefix/steady/tail control";
          return mlir::failure();
        }
        for (mlir::OpOperand &operand : user->getOpOperands())
          if (operand.get() == oldRoot)
            operand.set(replacement);
      }
      auto expandRelations = [&](auto &entries) {
        const size_t originalSize = entries.size();
        for (size_t index = 0; index < originalSize; ++index) {
          if (entries[index].buffer != oldRoot)
            continue;
          auto original = entries[index];
          entries[index].buffer = slots.front();
          for (mlir::Value slot :
               llvm::ArrayRef<mlir::Value>(slots).drop_front()) {
            auto copy = original;
            copy.buffer = slot;
            entries.push_back(std::move(copy));
          }
        }
      };
      for (CandidateInstructionIR &tile : tiles) {
        expandRelations(tile.relations->operationResultBuffers);
        expandRelations(tile.relations->operandBuffers);
        expandRelations(tile.relations->scratchBuffers);
        expandRelations(tile.relations->outputBuffers);
        expandRelations(tile.relations->cardDDRBuffers);
        expandRelations(tile.relations->partialReductionContributions);
        expandRelations(tile.relations->partialReductionMergeInputs);
      }
      if (auto allocation = oldRoot.getDefiningOp<mlir::memref::AllocOp>();
          allocation && allocation->use_empty())
        allocation.erase();
      for (mlir::Value slot : slots)
        storageProofsByScope[pipeline->scope].push_back({slot, object});
    }
  }

  for (const CandidateInstructionIR &tile : tiles)
    if (mlir::failed(mlir::verify(tile.getModule()))) {
      failureReason =
          "selected storage produced verifier-invalid current Instr IR";
      return mlir::failure();
    }

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "planning-materialization-phase", "complete-candidate",
      "bind-selected-events");

  std::map<EventId, const PlannedEvent *> plannedById;
  for (const PlannedEvent &event : events.events)
    if (!plannedById.try_emplace(event.id, &event).second) {
      failureReason = "complete candidate EventGraph has duplicate events";
      return mlir::failure();
    }
  llvm::DenseSet<mlir::Operation *> assignedOperations;
  std::map<EventId, std::vector<mlir::Operation *>> operationsByEvent;

  auto assignUnique = [&](const EventId &event,
                          llvm::ArrayRef<ActualIssue *> candidates) {
    if (candidates.size() != 1 || !candidates.front()->operation ||
        !assignedOperations.insert(candidates.front()->operation).second)
      return false;
    operationsByEvent[event].push_back(candidates.front()->operation);
    return true;
  };
  auto assignAll = [&](const EventId &event,
                       llvm::ArrayRef<ActualIssue *> candidates) {
    llvm::SmallDenseSet<mlir::Operation *, 4> unique;
    if (candidates.empty() ||
        llvm::any_of(candidates, [&](const ActualIssue *candidate) {
          return !candidate || !candidate->operation ||
                 assignedOperations.count(candidate->operation) ||
                 !unique.insert(candidate->operation).second;
        }))
      return false;
    for (ActualIssue *candidate : candidates) {
      assignedOperations.insert(candidate->operation);
      operationsByEvent[event].push_back(candidate->operation);
    }
    return true;
  };

  for (const PlannedEvent &planned : events.events) {
    if (planned.id.kind != PlannedEventKind::ComputeIssue)
      continue;
    if (planned.workerDomain.empty())
      continue;
    const auto *action = std::get_if<ExecutionEventAction>(&planned.id.action);
    auto node = action ? nodesByExecution.find(action->execution)
                       : nodesByExecution.end();
    if (!action || node == nodesByExecution.end() || !planned.tile) {
      failureReason = "compute event has no actual execution/node identity";
      return mlir::failure();
    }
    std::vector<ActualIssue *> matches;
    for (ActualIssue &issue : actualIssues)
      if (issue.kind == ActualIssueKind::Compute &&
          issue.tile == *planned.tile && hasNode(issue, node->second))
        matches.push_back(&issue);
    if (matches.empty()) {
      failureReason = "compute event has no actual typed NCC issue";
      return mlir::failure();
    }
    for (ActualIssue *match : matches) {
      if (!assignedOperations.insert(match->operation).second) {
        failureReason = "one actual NCC issue belongs to several events";
        return mlir::failure();
      }
      operationsByEvent[planned.id].push_back(match->operation);
    }
  }

  auto findWorkNode =
      [&](const analysis::RootRegionWorkId &work) -> std::optional<uint32_t> {
    auto nodes = nodesByWork.find(work);
    return nodes != nodesByWork.end() && nodes->second.size() == 1
               ? std::optional<uint32_t>(*nodes->second.begin())
               : std::nullopt;
  };
  auto findVersionNode =
      [&](const PhysicalVersionId &version) -> std::optional<uint32_t> {
    if (const auto *result =
            std::get_if<ExecutionResultValueId>(&version.logicalValue)) {
      const auto *execution =
          std::get_if<ExecutionInstanceId>(&result->execution);
      auto node = execution ? nodesByExecution.find(*execution)
                            : nodesByExecution.end();
      if (node != nodesByExecution.end())
        return node->second;
    }
    if (const auto *partial =
            std::get_if<ReductionPartialValueId>(&version.logicalValue)) {
      auto node = nodesByExecution.find(partial->execution);
      if (node != nodesByExecution.end())
        return node->second;
    }
    if (const auto *component =
            std::get_if<CoupledComponentValueId>(&version.logicalValue)) {
      auto node = nodesByExecution.find(component->execution);
      if (node != nodesByExecution.end())
        return node->second;
    }
    return std::nullopt;
  };
  auto expectedMovementNode =
      [&](const MovementEventAction &action) -> std::optional<uint32_t> {
    if (const auto *load = std::get_if<ExternalLoadId>(&action.action))
      return (action.phase == MovementEventPhase::DDRLoad ||
              action.phase == MovementEventPhase::LocalCombine)
                 ? findWorkNode(load->destination.work)
                 : std::nullopt;
    if (const auto *transfer =
            std::get_if<DDRBoundaryTransferId>(&action.action)) {
      if (action.phase == MovementEventPhase::DDRLoad ||
          action.phase == MovementEventPhase::LocalCombine)
        return findWorkNode(transfer->destination.work);
      auto plan =
          llvm::find_if(movement.ddrTransfers, [&](const auto &candidate) {
            return candidate.id == *transfer;
          });
      return plan == movement.ddrTransfers.end()
                 ? std::nullopt
                 : findVersionNode(plan->source);
    }
    if (const auto *gather = std::get_if<ReductionGatherId>(&action.action)) {
      auto plan =
          llvm::find_if(movement.reductionGathers, [&](const auto &candidate) {
            return candidate.id == *gather;
          });
      if (plan == movement.reductionGathers.end())
        return std::nullopt;
      if (action.phase == MovementEventPhase::DDRStore)
        return findVersionNode(plan->source);
      auto node = nodesByExecution.find(plan->mergeExecution);
      return node == nodesByExecution.end()
                 ? std::nullopt
                 : std::optional<uint32_t>(node->second);
    }
    if (const auto *publication =
            std::get_if<ResultPublicationId>(&action.action)) {
      PhysicalVersionId version{publication->source};
      return action.phase == MovementEventPhase::DDRStore
                 ? findVersionNode(version)
                 : std::nullopt;
    }
    return std::nullopt;
  };
  auto publicationOutput =
      [&](const MovementEventAction &action) -> std::optional<unsigned> {
    const auto *publication = std::get_if<ResultPublicationId>(&action.action);
    if (!publication || action.phase != MovementEventPhase::DDRStore)
      return std::nullopt;
    SemanticRootKey root = std::visit(
        [](const auto &execution) -> SemanticRootKey {
          using T = std::decay_t<decltype(execution)>;
          if constexpr (std::is_same_v<T, ExecutionInstanceId>)
            return std::visit(
                [](const auto &source) { return source.work.root; },
                execution.source);
          else
            return execution.producer.work.root;
        },
        publication->source.execution);
    return root.anchorKind == SemanticRootAnchorKind::FunctionResult
               ? std::optional<unsigned>(root.anchorIndex)
               : std::nullopt;
  };
  auto issueUsesOutput = [](const ActualIssue &issue, unsigned output) {
    if (!issue.operation || !issue.relations)
      return false;
    for (const SpatialOutputBufferRelation &relation :
         issue.relations->outputBuffers) {
      if (relation.outputIndex != output)
        continue;
      for (mlir::Value operand : issue.operation->getOperands())
        if (mlir::isa<mlir::BaseMemRefType>(operand.getType()) &&
            shareStructuredBufferStorage(operand, relation.buffer))
          return true;
    }
    return false;
  };
  auto issueMatchesReductionGather = [&](const ActualIssue &issue,
                                         const MovementEventAction &action) {
    const auto *gather = std::get_if<ReductionGatherId>(&action.action);
    if (!gather || !issue.operation || !issue.relations)
      return !gather;
    auto plan = llvm::find_if(movement.reductionGathers,
                              [&](const ReductionGatherPlan &candidate) {
                                return candidate.id == *gather;
                              });
    auto sourceTile = plan == movement.reductionGathers.end()
                          ? selectedVersionTiles.end()
                          : selectedVersionTiles.find(plan->source);
    if (plan == movement.reductionGathers.end() ||
        sourceTile == selectedVersionTiles.end())
      return false;
    const unsigned resultIndex = std::visit(
        [](const auto &value) -> unsigned {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, ReductionPartialValueId>)
            return value.result;
          else
            return static_cast<unsigned>(value.component);
        },
        gather->value);
    const StructuredResultIdentityKind resultIdentityKind =
        std::holds_alternative<CoupledComponentValueId>(gather->value)
            ? StructuredResultIdentityKind::CoupledReductionComponent
            : StructuredResultIdentityKind::OperationResult;
    StorageRootMemo memo;
    std::optional<uint32_t> sourceNode = findVersionNode(plan->source);
    auto matchesVersionRelation = [&](mlir::Value endpoint) {
      return sourceNode &&
             llvm::any_of(issue.relations->operationResultBuffers,
                          [&](const auto &relation) {
                            return relation.structuredNodeId == *sourceNode &&
                                   relation.resultIndex == resultIndex &&
                                   relation.identityKind ==
                                       resultIdentityKind &&
                                   shareStructuredBufferStorage(
                                       endpoint, relation.buffer, memo);
                          });
    };
    if (action.phase == MovementEventPhase::DDRLoad) {
      auto rdma = mlir::dyn_cast<InstrRDMAOp>(issue.operation);
      return rdma &&
             (matchesVersionRelation(rdma.getSource()) ||
              matchesVersionRelation(rdma.getDest()) ||
              llvm::any_of(issue.relations->partialReductionMergeInputs,
                           [&](const auto &relation) {
                             return relation.group == gather->group &&
                                    relation.resultIndex == resultIndex &&
                                    relation.sourceTile == sourceTile->second &&
                                    shareStructuredBufferStorage(
                                        rdma.getDest(), relation.buffer, memo);
                           }));
    }
    if (action.phase == MovementEventPhase::DDRStore) {
      auto wdma = mlir::dyn_cast<InstrWDMAOp>(issue.operation);
      return wdma && (matchesVersionRelation(wdma.getSource()) ||
                      matchesVersionRelation(wdma.getDest()) ||
                      llvm::any_of(
                          issue.relations->partialReductionContributions,
                          [&](const auto &relation) {
                            return relation.group == gather->group &&
                                   relation.resultIndex == resultIndex &&
                                   relation.sourceTile == sourceTile->second &&
                                   shareStructuredBufferStorage(
                                       wdma.getSource(), relation.buffer, memo);
                          }));
    }
    if (action.phase == MovementEventPhase::LocalCombine) {
      auto combine = mlir::dyn_cast<InstrGatherScatterOp>(issue.operation);
      if (!combine)
        return false;
      auto matchesPartial = [&](const auto &relation) {
        return relation.group == gather->group &&
               relation.resultIndex == resultIndex &&
               relation.sourceTile == sourceTile->second &&
               (shareStructuredBufferStorage(combine.getSource(),
                                             relation.buffer, memo) ||
                shareStructuredBufferStorage(combine.getDest(), relation.buffer,
                                             memo));
      };
      return matchesVersionRelation(combine.getSource()) ||
             matchesVersionRelation(combine.getDest()) ||
             llvm::any_of(issue.relations->partialReductionContributions,
                          matchesPartial) ||
             llvm::any_of(issue.relations->partialReductionMergeInputs,
                          matchesPartial);
    }
    return false;
  };
  auto issueMatchesBoundaryTransferCombine =
      [&](const ActualIssue &issue, const MovementEventAction &action) {
        const auto *transfer =
            std::get_if<DDRBoundaryTransferId>(&action.action);
        if (!transfer || issue.kind != ActualIssueKind::LocalCombine ||
            !issue.operation || !issue.relations)
          return false;
        auto plan =
            llvm::find_if(movement.ddrTransfers,
                          [&](const DDRBoundaryTransferPlan &candidate) {
                            return candidate.id == *transfer;
                          });
        const auto *result = plan == movement.ddrTransfers.end()
                                 ? nullptr
                                 : std::get_if<ExecutionResultValueId>(
                                       &plan->source.logicalValue);
        std::optional<uint32_t> sourceNode =
            plan == movement.ddrTransfers.end() ? std::nullopt
                                                : findVersionNode(plan->source);
        std::optional<uint32_t> destinationNode =
            plan == movement.ddrTransfers.end()
                ? std::nullopt
                : findWorkNode(plan->id.destination.work);
        if (!result || !sourceNode || !destinationNode ||
            !hasNode(issue, *sourceNode) || !hasNode(issue, *destinationNode))
          return false;
        StorageRootMemo memo;
        return llvm::any_of(
            issue.relations->operationResultBuffers, [&](const auto &relation) {
              if (relation.structuredNodeId != *sourceNode ||
                  relation.resultIndex != result->result ||
                  relation.identityKind !=
                      StructuredResultIdentityKind::OperationResult)
                return false;
              return llvm::any_of(
                  issue.operation->getOperands(), [&](mlir::Value operand) {
                    return mlir::isa<mlir::BaseMemRefType>(operand.getType()) &&
                           shareStructuredBufferStorage(operand,
                                                        relation.buffer, memo);
                  });
            });
      };
  std::set<MovementActionId> selectedPeerActions;
  for (const PeerTransferGraphPlan &graph : movement.peerGraphs)
    selectedPeerActions.insert(graph.actions.begin(), graph.actions.end());
  std::vector<MovementActionId> cardDDRActions;
  for (const DDRBoundaryTransferPlan &transfer : movement.ddrTransfers) {
    auto sourceTile = selectedVersionTiles.find(transfer.source);
    if (!selectedPeerActions.count(MovementActionId(transfer.id)) &&
        sourceTile != selectedVersionTiles.end() &&
        sourceTile->second != transfer.id.destination.work.tile)
      cardDDRActions.emplace_back(transfer.id);
  }
  for (const ReductionGatherPlan &gather : movement.reductionGathers)
    if (!selectedPeerActions.count(MovementActionId(gather.id)))
      cardDDRActions.emplace_back(gather.id);
  llvm::sort(cardDDRActions);
  std::map<MovementActionId, int64_t> cardDDRByAction;
  for (auto [resourceId, action] : llvm::enumerate(cardDDRActions))
    cardDDRByAction.emplace(action, static_cast<int64_t>(resourceId));
  auto issueMatchesCardDDRMovementTag = [&](const ActualIssue &issue,
                                            const MovementEventAction &action) {
    auto resource = cardDDRByAction.find(action.action);
    auto combine =
        mlir::dyn_cast_or_null<InstrGatherScatterOp>(issue.operation);
    CardDDRResourceAttr tagged =
        combine ? combine.getCardDdrResourceAttr() : CardDDRResourceAttr{};
    return resource != cardDDRByAction.end() && tagged &&
           tagged.getResourceId() == resource->second;
  };
  auto issueMatchesCardDDR = [&](const ActualIssue &issue,
                                 const MovementEventAction &action) {
    auto resource = cardDDRByAction.find(action.action);
    if (resource == cardDDRByAction.end() || !issue.operation ||
        !issue.relations)
      return false;
    StorageRootMemo memo;
    return llvm::any_of(
        issue.relations->cardDDRBuffers,
        [&](const CardDDRBufferRelation &relation) {
          return relation.resourceId == resource->second &&
                 llvm::any_of(issue.operation->getOperands(),
                              [&](mlir::Value operand) {
                                return mlir::isa<mlir::BaseMemRefType>(
                                           operand.getType()) &&
                                       shareStructuredBufferStorage(
                                           operand, relation.buffer, memo);
                              });
        });
  };

  for (const PlannedEvent &planned : events.events) {
    const auto *movement = std::get_if<MovementEventAction>(&planned.id.action);
    if (!movement || planned.id.kind != PlannedEventKind::LocalCombine ||
        movement->phase != MovementEventPhase::LocalCombine || !planned.tile)
      continue;
    std::optional<uint32_t> node = expectedMovementNode(*movement);
    std::vector<ActualIssue *> matches;
    for (ActualIssue &issue : actualIssues)
      if (issue.kind == ActualIssueKind::LocalCombine &&
          issue.tile == *planned.tile &&
          !assignedOperations.count(issue.operation) &&
          (!node || issue.nodes.empty() || hasNode(issue, *node)) &&
          issueMatchesReductionGather(issue, *movement))
        matches.push_back(&issue);
    if (matches.empty()) {
      failureReason = "local-combine event has no actual gather-scatter issue";
      return mlir::failure();
    }
    for (ActualIssue *match : matches) {
      if (node && match->nodes.empty()) {
        auto combine = mlir::cast<InstrGatherScatterOp>(match->operation);
        for (mlir::Value buffer : {combine.getSource(), combine.getDest()})
          if (!llvm::any_of(
                  match->relations->operandBuffers,
                  [&](const StructuredOperationBufferRelation &relation) {
                    return relation.structuredNodeId == *node &&
                           relation.buffer == buffer;
                  }))
            match->relations->operandBuffers.push_back({*node, buffer});
        match->nodes.push_back(*node);
      }
      if (!assignedOperations.insert(match->operation).second) {
        failureReason =
            "one actual gather-scatter belongs to several combine events";
        return mlir::failure();
      }
      operationsByEvent[planned.id].push_back(match->operation);
    }
  }

  for (const PlannedEvent &planned : events.events) {
    const auto *movement = std::get_if<MovementEventAction>(&planned.id.action);
    if (!movement || planned.id.kind != PlannedEventKind::MovementIssue ||
        (movement->phase != MovementEventPhase::DDRLoad &&
         movement->phase != MovementEventPhase::DDRStore))
      continue;
    const ActualIssueKind kind = movement->phase == MovementEventPhase::DDRLoad
                                     ? ActualIssueKind::DDRLoad
                                     : ActualIssueKind::DDRStore;
    std::optional<uint32_t> node = expectedMovementNode(*movement);
    std::optional<unsigned> output = publicationOutput(*movement);
    std::vector<ActualIssue *> matches;
    for (ActualIssue &issue : actualIssues) {
      if (!planned.tile || issue.tile != *planned.tile ||
          assignedOperations.count(issue.operation))
        continue;
      if (issue.kind == ActualIssueKind::LocalCombine &&
          cardDDRByAction.count(movement->action)) {
        MovementEventAction combineAction = *movement;
        combineAction.phase = MovementEventPhase::LocalCombine;
        if (issueMatchesCardDDRMovementTag(issue, combineAction) ||
            issueMatchesBoundaryTransferCombine(issue, combineAction) ||
            issueMatchesReductionGather(issue, combineAction)) {
          matches.push_back(&issue);
          continue;
        }
      }
      if (issue.kind != kind)
        continue;
      const bool cardDDRMatch = issueMatchesCardDDR(issue, *movement);
      if (!cardDDRMatch && node && !hasNode(issue, *node) &&
          (!output || !issueUsesOutput(issue, *output)))
        continue;
      if (!cardDDRMatch && !issueMatchesReductionGather(issue, *movement))
        continue;
      if (const auto *load = std::get_if<ExternalLoadId>(&movement->action)) {
        if (load->destination.fragment.source.kind ==
            analysis::RootBoundaryKind::ProgramInput) {
          std::optional<unsigned> argument =
              getDDRSourceArgument(issue.operation);
          if (!argument || *argument != load->destination.fragment.source.index)
            continue;
        }
      }
      matches.push_back(&issue);
    }
    if (!assignAll(planned.id, matches)) {
      llvm::raw_string_ostream diagnostic(failureReason);
      diagnostic << "DDR movement event has no complete actual typed "
                    "instruction occurrence set; tile=";
      if (planned.tile)
        diagnostic << planned.tile->getValue();
      else
        diagnostic << "none";
      diagnostic << ",action=" << stringifyMovementActionKind(movement->action)
                 << ",phase=" << static_cast<unsigned>(movement->phase)
                 << ",payload=" << movement->payloadSlice << ",node=";
      auto expectedCardDDR = cardDDRByAction.find(movement->action);
      if (expectedCardDDR != cardDDRByAction.end())
        diagnostic << "card-ddr:" << expectedCardDDR->second << ':';
      if (node)
        diagnostic << *node;
      else
        diagnostic << "none";
      if (const auto *gather =
              std::get_if<ReductionGatherId>(&movement->action)) {
        diagnostic
            << ",gather-value-kind=" << gather->value.index()
            << ",gather-result="
            << std::visit(
                   [](const auto &value) -> unsigned {
                     using T = std::decay_t<decltype(value)>;
                     if constexpr (std::is_same_v<T, ReductionPartialValueId>)
                       return value.result;
                     else
                       return static_cast<unsigned>(value.component);
                   },
                   gather->value);
      }
      diagnostic << ",resource-issues=[";
      bool firstResourceIssue = true;
      for (ActualIssue &issue : actualIssues) {
        if (!planned.tile || issue.tile != *planned.tile ||
            !issueMatchesCardDDR(issue, *movement))
          continue;
        if (!firstResourceIssue)
          diagnostic << ';';
        firstResourceIssue = false;
        diagnostic << "kind=" << static_cast<unsigned>(issue.kind)
                   << ",assigned=" << assignedOperations.count(issue.operation)
                   << ",op=";
        issue.operation->print(diagnostic);
      }
      diagnostic << ']';
      diagnostic << ",candidates=[";
      bool first = true;
      for (ActualIssue &issue : actualIssues) {
        if (issue.kind != kind || !planned.tile || issue.tile != *planned.tile)
          continue;
        if (!first)
          diagnostic << ';';
        first = false;
        diagnostic << "nodes=[";
        llvm::interleaveComma(issue.nodes, diagnostic);
        diagnostic << "],arg=";
        std::optional<unsigned> argument =
            getDDRSourceArgument(issue.operation);
        if (argument)
          diagnostic << *argument;
        else
          diagnostic << "unknown";
        diagnostic << ",assigned=" << assignedOperations.count(issue.operation)
                   << ",result-relations=[";
        StorageRootMemo relationMemo;
        bool firstRelation = true;
        for (const StructuredOperationResultBufferRelation &relation :
             issue.relations->operationResultBuffers) {
          if (!llvm::any_of(
                  issue.operation->getOperands(), [&](mlir::Value operand) {
                    return mlir::isa<mlir::BaseMemRefType>(operand.getType()) &&
                           shareStructuredBufferStorage(
                               operand, relation.buffer, relationMemo);
                  }))
            continue;
          if (!firstRelation)
            diagnostic << ',';
          firstRelation = false;
          diagnostic << relation.structuredNodeId << ':' << relation.resultIndex
                     << ':' << static_cast<unsigned>(relation.identityKind);
        }
        diagnostic << "],merge-inputs=[";
        firstRelation = true;
        for (const PartialReductionMergeInputBufferRelation &relation :
             issue.relations->partialReductionMergeInputs) {
          if (!llvm::any_of(
                  issue.operation->getOperands(), [&](mlir::Value operand) {
                    return mlir::isa<mlir::BaseMemRefType>(operand.getType()) &&
                           shareStructuredBufferStorage(
                               operand, relation.buffer, relationMemo);
                  }))
            continue;
          if (!firstRelation)
            diagnostic << ',';
          firstRelation = false;
          diagnostic << relation.resultIndex << '@'
                     << relation.sourceTile.getValue();
        }
        diagnostic << "]"
                   << ",op=";
        issue.operation->print(diagnostic);
      }
      diagnostic << ']';
      return mlir::failure();
    }
  }

  // Lowering can expand one structured execution into layout combines and
  // private DDR staging issues. After all selected movement actions have been
  // bound above, an otherwise-unbound issue belongs to the execution event
  // only when its current-IR relations name exactly one structured node and
  // that node has exactly one execution event on this Tile. Cross-node or
  // relation-free issues remain errors. Destination assembly combines start
  // without a node relation and were bound to LocalCombine above.
  auto isPrivateDDRStage = [](const ActualIssue &issue) {
    mlir::Value endpoint;
    if (auto load = mlir::dyn_cast<InstrRDMAOp>(issue.operation))
      endpoint = load.getSource();
    else if (auto store = mlir::dyn_cast<InstrWDMAOp>(issue.operation))
      endpoint = store.getDest();
    else
      return false;
    StorageRootMemo memo;
    const llvm::DenseSet<mlir::Value> &roots = memo.getStorageRoots(endpoint);
    return !roots.empty() && llvm::all_of(roots, [](mlir::Value root) {
      auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
      return allocation && isWaferDDRMemRefType(allocation.getType());
    });
  };
  for (ActualIssue &issue : actualIssues) {
    const bool loweringOwnedIssue =
        issue.kind == ActualIssueKind::LocalCombine ||
        issue.kind == ActualIssueKind::DDRLoad ||
        issue.kind == ActualIssueKind::DDRStore;
    if (!loweringOwnedIssue || assignedOperations.count(issue.operation))
      continue;
    if ((issue.kind == ActualIssueKind::DDRLoad ||
         issue.kind == ActualIssueKind::DDRStore) &&
        !isPrivateDDRStage(issue))
      continue;
    if (auto combine = mlir::dyn_cast<InstrGatherScatterOp>(issue.operation)) {
      llvm::SmallVector<EventId, 2> downstreamEvents;
      TileRegionOp ownerRegion =
          issue.operation->getParentOfType<TileRegionOp>();
      for (const ActualIssue &candidate : actualIssues) {
        if (!assignedOperations.count(candidate.operation) || !ownerRegion ||
            candidate.tile != issue.tile ||
            candidate.operation->getParentOfType<TileRegionOp>() != ownerRegion)
          continue;
        StorageRootMemo memo;
        const bool consumesDestination = llvm::any_of(
            candidate.operation->getOperands(), [&](mlir::Value operand) {
              return mlir::isa<mlir::BaseMemRefType>(operand.getType()) &&
                     shareStructuredBufferStorage(combine.getDest(), operand,
                                                  memo);
            });
        if (!consumesDestination)
          continue;
        for (const auto &[event, operations] : operationsByEvent)
          if (llvm::is_contained(operations, candidate.operation) &&
              !llvm::is_contained(downstreamEvents, event))
            downstreamEvents.push_back(event);
      }
      if (downstreamEvents.size() == 1) {
        assignedOperations.insert(issue.operation);
        operationsByEvent[downstreamEvents.front()].push_back(issue.operation);
        continue;
      }
    }
    std::optional<uint32_t> ownedNode;
    if (auto combine = mlir::dyn_cast<InstrGatherScatterOp>(issue.operation)) {
      StorageRootMemo memo;
      llvm::SmallVector<uint32_t, 4> destinationNodes;
      auto collectDestinationOwner = [&](const auto &relation) {
        if (shareStructuredBufferStorage(combine.getDest(), relation.buffer,
                                         memo) &&
            !llvm::is_contained(destinationNodes, relation.structuredNodeId))
          destinationNodes.push_back(relation.structuredNodeId);
      };
      for (const StructuredOperationBufferRelation &relation :
           issue.relations->scratchBuffers)
        collectDestinationOwner(relation);
      for (const StructuredOperationBufferRelation &relation :
           issue.relations->operandBuffers)
        collectDestinationOwner(relation);
      if (destinationNodes.size() == 1)
        ownedNode = destinationNodes.front();
      if (!ownedNode) {
        llvm::SmallVector<uint32_t, 4> downstreamNodes;
        llvm::SmallVector<mlir::Value, 4> reachableStorage{combine.getDest()};
        llvm::DenseSet<mlir::Operation *> reachedOperations;
        TileRegionOp ownerRegion =
            issue.operation->getParentOfType<TileRegionOp>();
        bool changed = true;
        while (changed) {
          changed = false;
          for (const ActualIssue &candidate : actualIssues) {
            if (candidate.tile != issue.tile || !ownerRegion ||
                candidate.operation->getParentOfType<TileRegionOp>() !=
                    ownerRegion ||
                (candidate.operation->getBlock() ==
                     issue.operation->getBlock() &&
                 !issue.operation->isBeforeInBlock(candidate.operation)) ||
                reachedOperations.contains(candidate.operation))
              continue;
            StorageRootMemo downstreamMemo;
            const bool consumesReachable = llvm::any_of(
                candidate.operation->getOperands(), [&](mlir::Value operand) {
                  return mlir::isa<mlir::BaseMemRefType>(operand.getType()) &&
                         llvm::any_of(reachableStorage,
                                      [&](mlir::Value reachable) {
                                        return shareStructuredBufferStorage(
                                            reachable, operand, downstreamMemo);
                                      });
                });
            if (!consumesReachable)
              continue;
            reachedOperations.insert(candidate.operation);
            changed = true;
            if (candidate.kind == ActualIssueKind::LocalCombine) {
              auto downstream =
                  mlir::cast<InstrGatherScatterOp>(candidate.operation);
              reachableStorage.push_back(downstream.getDest());
              continue;
            }
            if (candidate.kind != ActualIssueKind::Compute)
              continue;
            for (uint32_t node : candidate.nodes)
              if (!llvm::is_contained(downstreamNodes, node))
                downstreamNodes.push_back(node);
          }
        }
        if (downstreamNodes.size() == 1)
          ownedNode = downstreamNodes.front();
      }
    }
    if (!ownedNode && issue.nodes.size() == 1) {
      ownedNode = issue.nodes.front();
    } else if (!ownedNode) {
      TileRegionOp region = issue.operation->getParentOfType<TileRegionOp>();
      auto tile = tilesById.find(issue.tile.getValue());
      if (region && tile != tilesById.end() && tile->second->regionNodes) {
        auto relation = llvm::find_if(
            *tile->second->regionNodes, [&](const auto &candidate) {
              return candidate.region == region.getOperation();
            });
        if (relation != tile->second->regionNodes->end() &&
            relation->structuredNodes.size() == 1 &&
            hasNode(issue, relation->structuredNodes.front()))
          ownedNode = relation->structuredNodes.front();
      }
    }
    if (!ownedNode)
      continue;
    std::vector<EventId> matches;
    for (const PlannedEvent &planned : events.events) {
      if (planned.id.kind != PlannedEventKind::ComputeIssue || !planned.tile ||
          *planned.tile != issue.tile)
        continue;
      const auto *action =
          std::get_if<ExecutionEventAction>(&planned.id.action);
      auto node = action ? nodesByExecution.find(action->execution)
                         : nodesByExecution.end();
      if (node != nodesByExecution.end() && node->second == *ownedNode)
        matches.push_back(planned.id);
    }
    if (matches.size() != 1 ||
        !assignedOperations.insert(issue.operation).second) {
      failureReason = "lowering-owned instruction issue has no unique "
                      "structured execution event";
      return mlir::failure();
    }
    operationsByEvent[matches.front()].push_back(issue.operation);
  }

  struct PeerGraphMessageIdentity {
    const PeerTransferGraphPlan *graph = nullptr;
    int64_t communication = -1;
  };
  std::map<MovementActionId, PeerGraphMessageIdentity> peerGraphMessages;
  std::vector<const PeerTransferGraphPlan *> orderedPeerGraphs;
  for (const PeerTransferGraphPlan &graph : movement.peerGraphs) {
    if (graph.actions.empty()) {
      failureReason = "selected peer graph has no message identity";
      return mlir::failure();
    }
    orderedPeerGraphs.push_back(&graph);
  }
  llvm::sort(orderedPeerGraphs, [](const PeerTransferGraphPlan *lhs,
                                   const PeerTransferGraphPlan *rhs) {
    return lhs->actions < rhs->actions;
  });
  for (auto [communication, graph] : llvm::enumerate(orderedPeerGraphs))
    if (!peerGraphMessages
             .try_emplace(graph->actions.front(),
                          PeerGraphMessageIdentity{
                              graph, static_cast<int64_t>(communication)})
             .second) {
      failureReason = "selected peer graphs have duplicate message anchors";
      return mlir::failure();
    }

  for (const PlannedEvent &planned : events.events) {
    const auto *movement = std::get_if<MovementEventAction>(&planned.id.action);
    if (!movement || planned.id.kind != PlannedEventKind::MovementIssue ||
        (movement->phase != MovementEventPhase::PeerSend &&
         movement->phase != MovementEventPhase::PeerReceive) ||
        !movement->hop || !planned.tile)
      continue;
    const bool send = movement->phase == MovementEventPhase::PeerSend;
    auto graphIdentity = peerGraphMessages.find(movement->action);
    if (graphIdentity == peerGraphMessages.end()) {
      failureReason = "peer movement event has no selected graph identity";
      return mlir::failure();
    }
    auto expectedRound =
        llvm::find(graphIdentity->second.graph->hops, *movement->hop);
    if (expectedRound == graphIdentity->second.graph->hops.end()) {
      failureReason = "peer movement event has no selected graph identity";
      return mlir::failure();
    }
    const int64_t round = static_cast<int64_t>(std::distance(
        graphIdentity->second.graph->hops.begin(), expectedRound));
    std::vector<ActualIssue *> matches;
    for (ActualIssue &issue : actualIssues) {
      if (issue.tile != *planned.tile ||
          assignedOperations.count(issue.operation) ||
          issue.kind !=
              (send ? ActualIssueKind::PeerSend : ActualIssueKind::PeerReceive))
        continue;
      int64_t peer = send ? mlir::cast<InstrDTESendOp>(issue.operation)
                                .getPeerAttr()
                                .getInt()
                          : mlir::cast<InstrDTERecvOp>(issue.operation)
                                .getPeerAttr()
                                .getInt();
      int64_t payload = send ? mlir::cast<InstrDTESendOp>(issue.operation)
                                   .getMessageAttr()
                                   .getPayloadSlice()
                             : mlir::cast<InstrDTERecvOp>(issue.operation)
                                   .getMessageAttr()
                                   .getPayloadSlice();
      const TileId expectedPeer =
          send ? movement->hop->destination : movement->hop->source;
      if (peer == expectedPeer.getValue() &&
          payload == static_cast<int64_t>(movement->payloadSlice) &&
          (send ? mlir::cast<InstrDTESendOp>(issue.operation)
                      .getMessageAttr()
                      .getCommunicationId()
                : mlir::cast<InstrDTERecvOp>(issue.operation)
                      .getMessageAttr()
                      .getCommunicationId()) ==
              graphIdentity->second.communication &&
          (send ? mlir::cast<InstrDTESendOp>(issue.operation)
                      .getMessageAttr()
                      .getRound()
                : mlir::cast<InstrDTERecvOp>(issue.operation)
                      .getMessageAttr()
                      .getRound()) == round)
        matches.push_back(&issue);
    }
    if (!assignUnique(planned.id, matches)) {
      failureReason =
          "peer movement event has no unique actual typed instruction";
      return mlir::failure();
    }
  }

  for (const ActualIssue &issue : actualIssues)
    if (!assignedOperations.count(issue.operation)) {
      llvm::raw_string_ostream diagnostic(failureReason);
      diagnostic << "actual candidate contains an unbound instruction issue: "
                 << issue.operation->getName() << " on Tile "
                 << issue.tile.getValue() << " nodes=[";
      llvm::interleaveComma(issue.nodes, diagnostic);
      diagnostic << "] op=";
      issue.operation->print(diagnostic);
      if (issue.kind == ActualIssueKind::PeerSend ||
          issue.kind == ActualIssueKind::PeerReceive) {
        const bool send = issue.kind == ActualIssueKind::PeerSend;
        DTEMessageAttr message =
            send ? mlir::cast<InstrDTESendOp>(issue.operation).getMessageAttr()
                 : mlir::cast<InstrDTERecvOp>(issue.operation).getMessageAttr();
        const int64_t peer = send ? mlir::cast<InstrDTESendOp>(issue.operation)
                                        .getPeerAttr()
                                        .getInt()
                                  : mlir::cast<InstrDTERecvOp>(issue.operation)
                                        .getPeerAttr()
                                        .getInt();
        diagnostic << " peer-message=[communication="
                   << message.getCommunicationId()
                   << ",round=" << message.getRound()
                   << ",payload=" << message.getPayloadSlice()
                   << ",peer=" << peer << "] selected-graph=[";
        if (message.getCommunicationId() >= 0 &&
            static_cast<size_t>(message.getCommunicationId()) <
                orderedPeerGraphs.size()) {
          const PeerTransferGraphPlan &graph =
              *orderedPeerGraphs[message.getCommunicationId()];
          diagnostic << "actions=" << graph.actions.size() << ",hops=";
          for (auto [round, hop] : llvm::enumerate(graph.hops))
            diagnostic << (round ? "," : "") << round << ':'
                       << hop.source.getValue() << "->"
                       << hop.destination.getValue();
        } else {
          diagnostic << "absent";
        }
        diagnostic << ']';
      }
      TileRegionOp region = issue.operation->getParentOfType<TileRegionOp>();
      diagnostic << " region=" << static_cast<bool>(region)
                 << " explicit_region_relations=[";
      auto tile = tilesById.find(issue.tile.getValue());
      if (tile != tilesById.end() && tile->second->regionNodes) {
        bool first = true;
        for (const auto &relation : *tile->second->regionNodes) {
          if (!first)
            diagnostic << ';';
          first = false;
          diagnostic << "matches="
                     << (region && relation.region == region.getOperation())
                     << ",nodes=[";
          llvm::interleaveComma(relation.structuredNodes, diagnostic);
          diagnostic << ']';
        }
      }
      diagnostic << ']';
      return mlir::failure();
    }

  // Several typed collectors contribute operations to one logical event. Keep
  // the current per-block program order when combining those contributions;
  // collector phase order is not an execution order and must not move layout
  // copies away from their producer/consumer chain.
  for (auto &[event, operations] : operationsByEvent) {
    (void)event;
    llvm::SmallVector<mlir::Block *, 4> blockOrder;
    std::map<mlir::Block *, llvm::SmallVector<mlir::Operation *, 8>> byBlock;
    for (mlir::Operation *operation : operations) {
      mlir::Block *block = operation->getBlock();
      if (!byBlock.count(block))
        blockOrder.push_back(block);
      byBlock[block].push_back(operation);
    }
    operations.clear();
    for (mlir::Block *block : blockOrder) {
      auto &blockOperations = byBlock.at(block);
      llvm::sort(blockOperations,
                 [](mlir::Operation *lhs, mlir::Operation *rhs) {
                   return lhs != rhs && lhs->isBeforeInBlock(rhs);
                 });
      operations.insert(operations.end(), blockOperations.begin(),
                        blockOperations.end());
    }
  }

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "planning-materialization-phase", "complete-candidate",
      "close-event-occurrences");

  using EventBlocks = llvm::SmallVector<mlir::Block *, 4>;
  std::map<EventId, EventBlocks> blocksByEvent;
  auto sameBlocks = [](llvm::ArrayRef<mlir::Block *> lhs,
                       llvm::ArrayRef<mlir::Block *> rhs) {
    return lhs.size() == rhs.size() &&
           llvm::all_of(lhs, [&](mlir::Block *block) {
             return llvm::is_contained(rhs, block);
           });
  };
  auto setEventBlocks = [&](const EventId &event,
                            llvm::ArrayRef<mlir::Block *> blocks) {
    auto [position, inserted] =
        blocksByEvent.try_emplace(event, blocks.begin(), blocks.end());
    return inserted || sameBlocks(position->second, blocks);
  };
  for (const auto &[event, operations] : operationsByEvent) {
    if (operations.empty())
      continue;
    EventBlocks blocks;
    for (mlir::Operation *operation : operations) {
      mlir::Block *block = operation ? operation->getBlock() : nullptr;
      if (!block) {
        failureReason = "one event issue has no current actual block";
        return mlir::failure();
      }
      if (!llvm::is_contained(blocks, block))
        blocks.push_back(block);
    }
    blocksByEvent.emplace(event, std::move(blocks));
  }
  for (const CompletionObligation &completion : events.completionObligations) {
    auto issue = blocksByEvent.find(completion.issue);
    if (issue != blocksByEvent.end() &&
        !setEventBlocks(completion.completion, issue->second)) {
      failureReason =
          "completion event occurrence blocks differ from its actual issue";
      return mlir::failure();
    }
  }
  for (size_t iteration = 0; iteration < events.events.size(); ++iteration) {
    bool changed = false;
    for (const EventDependency &dependency : events.hardDependencies) {
      auto predecessor = blocksByEvent.find(dependency.predecessor);
      auto successor = blocksByEvent.find(dependency.successor);
      const PlannedEvent *predecessorPlan =
          plannedById.at(dependency.predecessor);
      const PlannedEvent *successorPlan = plannedById.at(dependency.successor);
      if (predecessor != blocksByEvent.end() &&
          successor == blocksByEvent.end() &&
          (!predecessorPlan->tile || !successorPlan->tile ||
           predecessorPlan->tile == successorPlan->tile)) {
        blocksByEvent.emplace(dependency.successor, predecessor->second);
        changed = true;
      } else if (successor != blocksByEvent.end() &&
                 predecessor == blocksByEvent.end() &&
                 (!predecessorPlan->tile || !successorPlan->tile ||
                  predecessorPlan->tile == successorPlan->tile)) {
        blocksByEvent.emplace(dependency.predecessor, successor->second);
        changed = true;
      }
    }
    if (!changed)
      break;
  }
  for (const ControlOrder &control : schedule.controlOrders) {
    EventBlocks actualScopeBlocks;
    for (const EventId &event : control.events) {
      auto block = blocksByEvent.find(event);
      if (block == blocksByEvent.end())
        continue;
      for (mlir::Block *occurrence : block->second)
        if (!llvm::is_contained(actualScopeBlocks, occurrence))
          actualScopeBlocks.push_back(occurrence);
    }
    if (actualScopeBlocks.empty()) {
      failureReason = "selected control scope has no actual block";
      return mlir::failure();
    }
    for (const EventId &event : control.events)
      blocksByEvent.try_emplace(event, actualScopeBlocks);
  }

  std::map<mlir::Operation *, EventId> eventsByIssue;
  for (const auto &[event, operations] : operationsByEvent)
    for (mlir::Operation *operation : operations)
      if (mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation))
        eventsByIssue.emplace(operation, event);
  std::map<EventId, CompletionProtocol> completionProtocols;
  for (const CompletionPlacement &placement : schedule.completionPlacements)
    completionProtocols.emplace(placement.issue, placement.protocol);
  llvm::DenseSet<mlir::Value> waitedTokens;
  llvm::SmallVector<InstrDTEWaitOp, 8> oldWaits;
  for (CandidateInstructionIR &tile : tiles) {
    bool oldJoin = false;
    tile.getModule().walk([&](mlir::Operation *operation) {
      oldJoin |= mlir::isa<SyncNCCJoinOp>(operation);
      if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation))
        oldWaits.push_back(wait);
    });
    if (oldJoin) {
      failureReason = "selected instruction input contains an unowned NCC join";
      return mlir::failure();
    }
  }
  for (InstrDTEWaitOp wait : oldWaits)
    for (mlir::Value token : wait.getTokens()) {
      mlir::Operation *issue = token.getDefiningOp();
      auto event = eventsByIssue.find(issue);
      auto protocol = event == eventsByIssue.end()
                          ? completionProtocols.end()
                          : completionProtocols.find(event->second);
      if (!mlir::isa<mlir::async::TokenType>(token.getType()) ||
          event == eventsByIssue.end() ||
          protocol == completionProtocols.end() ||
          protocol->second != CompletionProtocol::DirectDTE ||
          !waitedTokens.insert(token).second ||
          llvm::any_of(token.getUsers(), [&](mlir::Operation *user) {
            return user != wait.getOperation();
          })) {
        failureReason =
            "preexisting DTE wait does not match one selected token event";
        return mlir::failure();
      }
    }
  for (InstrDTEWaitOp wait : oldWaits)
    wait.erase();

  std::vector<ScheduleIRModule> modules;
  std::vector<ScheduleEventIRBinding> bindings;
  for (CandidateInstructionIR &tile : tiles)
    modules.push_back({tile.tile, tile.getModule()});
  for (const PlannedEvent &planned : events.events) {
    auto eventBlocks = blocksByEvent.find(planned.id);
    if (eventBlocks == blocksByEvent.end() || eventBlocks->second.empty()) {
      failureReason = "planned event has no actual control block";
      return mlir::failure();
    }
    std::optional<TileId> ownerTile;
    for (mlir::Block *block : eventBlocks->second) {
      std::optional<TileId> blockTile;
      for (CandidateInstructionIR &tile : tiles) {
        mlir::Operation *parent = block->getParentOp();
        if (parent && tile.getModule()->isAncestor(parent)) {
          if (blockTile) {
            failureReason = "one event block belongs to several Tile modules";
            return mlir::failure();
          }
          blockTile = tile.tile;
        }
      }
      if (!blockTile || (ownerTile && *ownerTile != *blockTile)) {
        failureReason = "one event occurrence set crosses Tile modules";
        return mlir::failure();
      }
      ownerTile = blockTile;
    }
    if (!ownerTile || (planned.tile && *planned.tile != *ownerTile)) {
      failureReason = "planned event is bound to the wrong actual Tile";
      return mlir::failure();
    }
    ScheduleEventIRBinding binding{planned.id, *ownerTile,
                                   eventBlocks->second.front(),
                                   operationsByEvent[planned.id]};
    binding.blocks.assign(eventBlocks->second.begin(),
                          eventBlocks->second.end());
    bindings.push_back(std::move(binding));
  }
  PreparedScheduleMaterializationResult prepared =
      prepareScheduleMaterialization(scheduleDomain, schedule, modules,
                                     bindings);
  if (!prepared.succeeded()) {
    failureReason = prepared.failure ? prepared.failure->detail
                                     : "selected schedule preparation failed";
    return mlir::failure();
  }

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "planning-materialization-phase", "complete-candidate",
      "materialize-selected-schedule");

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> owners;
  owners.reserve(tiles.size());
  for (CandidateInstructionIR &tile : tiles)
    owners.push_back(std::move(*tile.owner));
  MaterializedScheduleResult materialized =
      materializeSchedule(std::move(owners), std::move(*prepared.prepared));
  if (!materialized.succeeded()) {
    failureReason = materialized.failure
                        ? materialized.failure->detail
                        : "selected schedule materialization failed";
    return mlir::failure();
  }
  std::map<EventId, std::vector<mlir::Operation *>> scheduledOperations;
  for (const ScheduleEventIRBinding &binding :
       materialized.materialized->eventBindings)
    scheduledOperations[binding.event] = binding.operations;
  for (const MaterializedCompletionGroup &group :
       materialized.materialized->completionGroups) {
    auto &operations = scheduledOperations[group.boundary.after];
    operations.insert(operations.end(), group.operations.begin(),
                      group.operations.end());
  }
  for (auto [moduleIndex, module] :
       llvm::enumerate(materialized.materialized->modules)) {
    ExecutionStructurePlan localPlan;
    std::vector<ExecutionStructureLoopBinding> loopBindings;
    for (const ExecutionStructureChoice &choice : structure.scopes) {
      const auto *pipeline = std::get_if<PipelinedExecutionStructure>(&choice);
      if (!pipeline)
        continue;
      std::vector<mlir::Operation *> operations;
      for (const EventStageAssignment &assignment : pipeline->eventStages) {
        auto event = scheduledOperations.find(assignment.event);
        if (event != scheduledOperations.end())
          for (mlir::Operation *operation : event->second)
            if (operation && module.get()->isAncestor(operation))
              operations.push_back(operation);
      }
      if (operations.empty())
        continue;
      bool foreignOperation = false;
      for (const EventStageAssignment &assignment : pipeline->eventStages) {
        auto event = scheduledOperations.find(assignment.event);
        if (event == scheduledOperations.end())
          continue;
        foreignOperation |=
            llvm::any_of(event->second, [&](mlir::Operation *operation) {
              return operation && !module.get()->isAncestor(operation);
            });
      }
      if (foreignOperation) {
        failureReason =
            "one selected pipeline scope spans several Tile modules";
        return mlir::failure();
      }
      llvm::SmallVector<mlir::scf::ForOp, 4> loops;
      module->walk([&](mlir::scf::ForOp loop) {
        if (getStaticTripCount(loop) ==
                std::optional<uint64_t>(pipeline->iteration.steadyTripCount) &&
            llvm::all_of(operations, [&](mlir::Operation *operation) {
              return loop->isAncestor(operation);
            }))
          loops.push_back(loop);
      });
      if (loops.size() != 1) {
        failure.kind = CardExecutablePreparationFailureKind::Unsupported;
        failureReason = "selected pipeline has no unique actual steady loop";
        return mlir::failure();
      }
      ExecutionStructureLoopBinding binding;
      binding.scope = pipeline->scope;
      binding.steadyLoop = loops.front();
      auto storageProofs = storageProofsByScope.find(pipeline->scope);
      if (storageProofs != storageProofsByScope.end())
        binding.externalStorageProofs = storageProofs->second;
      llvm::DenseSet<mlir::Operation *> assigned;
      for (const EventStageAssignment &assignment : pipeline->eventStages) {
        ExecutionEventOperationGroup group;
        group.event = assignment.event;
        for (mlir::Operation *operation :
             scheduledOperations[assignment.event]) {
          if (!operation || !module.get()->isAncestor(operation))
            continue;
          mlir::Operation *top = getLoopBodyOwner(operation, loops.front());
          if (!top) {
            failure.kind = CardExecutablePreparationFailureKind::Unsupported;
            failureReason =
                "selected pipeline event is outside its steady-loop body";
            return mlir::failure();
          }
          if (assigned.insert(top).second)
            group.operations.push_back(top);
        }
        binding.eventOrder.push_back(std::move(group));
      }
      if (binding.eventOrder.empty()) {
        failureReason = "selected pipeline has no actual event groups";
        return mlir::failure();
      }
      for (mlir::Operation &operation :
           loops.front().getBody()->without_terminator())
        if (assigned.insert(&operation).second)
          binding.eventOrder.front().operations.push_back(&operation);
      for (ExecutionEventOperationGroup &group : binding.eventOrder)
        llvm::sort(group.operations,
                   [](mlir::Operation *lhs, mlir::Operation *rhs) {
                     return lhs->isBeforeInBlock(rhs);
                   });
      localPlan.scopes.push_back(*pipeline);
      loopBindings.push_back(std::move(binding));
    }
    if (localPlan.scopes.empty())
      continue;
    PreparedExecutionStructureResult preparedStructure =
        prepareExecutionStructureMaterialization(module.get(), localPlan,
                                                 buffers, loopBindings);
    if (!preparedStructure.succeeded()) {
      failure.kind =
          preparedStructure.failure &&
                  preparedStructure.failure->kind ==
                      ExecutionStructureMaterializationFailureKind::Unsupported
              ? CardExecutablePreparationFailureKind::Unsupported
              : CardExecutablePreparationFailureKind::CompilerBug;
      failureReason = preparedStructure.failure
                          ? preparedStructure.failure->detail
                          : "selected execution structure preparation failed";
      return mlir::failure();
    }
    MaterializedExecutionStructureResult structured =
        materializeExecutionStructure(
            std::move(materialized.materialized->modules[moduleIndex]),
            std::move(*preparedStructure.prepared));
    if (!structured.succeeded()) {
      failureReason = structured.failure
                          ? structured.failure->detail
                          : "selected execution structure materialization "
                            "failed";
      return mlir::failure();
    }
    materialized.materialized->modules[moduleIndex] =
        std::move(structured.materialized->module);
  }
  std::map<int64_t, size_t> viewsByTile;
  for (auto [index, tile] : llvm::enumerate(tiles))
    viewsByTile.emplace(tile.tile.getValue(), index);
  for (auto [module, tile] :
       llvm::zip_equal(materialized.materialized->modules,
                       materialized.materialized->moduleTiles)) {
    auto view = viewsByTile.find(tile.getValue());
    if (view == viewsByTile.end()) {
      failureReason = "selected schedule changed the Tile module domain";
      return mlir::failure();
    }
    *tiles[view->second].owner = std::move(module);
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
