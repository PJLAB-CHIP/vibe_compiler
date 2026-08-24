//===- MovementTransferBuilder.cpp - Selected peer transfers ---------===//

#include "Wafer/Planning/PhysicalDataflow/MovementTransferBuilder.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

struct TileIdLess {
  bool operator()(TileId lhs, TileId rhs) const {
    return lhs.getValue() < rhs.getValue();
  }
};

void setFailure(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
}

mlir::LogicalResult fail(std::string *failureReason, llvm::StringRef detail) {
  setFailure(failureReason, detail);
  return mlir::failure();
}

const MovementEndpointBinding *
findEndpoint(llvm::ArrayRef<MovementEndpointBinding> endpoints, TileId tile) {
  auto found = llvm::find_if(
      endpoints, [&](const auto &endpoint) { return endpoint.tile == tile; });
  return found == endpoints.end() ? nullptr : &*found;
}

} // namespace

mlir::FailureOr<PreparedPeerTransfer>
preparePeerTransfer(const MovementActionId &action,
                    const PeerTransferGraphPlan &realization,
                    llvm::ArrayRef<MovementEndpointBinding> endpoints,
                    std::string *failureReason) {
  if (realization.hops.empty()) {
    setFailure(failureReason, "peer transfer requires an explicit non-DDR hop");
    return mlir::failure();
  }
  if (realization.kind == PeerTransferGraphKind::TargetRoutedPeer &&
      realization.hops.size() != 1) {
    setFailure(failureReason,
               "target-routed transfer must contain one endpoint hop");
    return mlir::failure();
  }
  std::vector<MovementActionId> actions = realization.actions;
  if (actions.empty())
    actions.push_back(action);
  if (!llvm::is_sorted(actions) ||
      std::adjacent_find(actions.begin(), actions.end()) != actions.end() ||
      !llvm::is_contained(actions, action)) {
    setFailure(failureReason,
               "peer transfer has an invalid logical action set");
    return mlir::failure();
  }
  std::map<int64_t, const MovementEndpointBinding *> uniqueEndpoints;
  for (const MovementEndpointBinding &endpoint : endpoints)
    if (!endpoint.buffer || !endpoint.builder ||
        !uniqueEndpoints.try_emplace(endpoint.tile.getValue(), &endpoint)
             .second) {
      setFailure(failureReason,
                 "peer transfer endpoints are missing or duplicated");
      return mlir::failure();
    }

  std::set<MovementHop> uniqueHops;
  std::set<TileId, TileIdLess> activeTiles;
  std::map<TileId, TileId, TileIdLess> parents;
  for (const MovementHop &hop : realization.hops) {
    if (hop.source == hop.destination || !uniqueHops.insert(hop).second ||
        !parents.try_emplace(hop.destination, hop.source).second) {
      setFailure(failureReason,
                 "peer transfer graph has a loop, duplicate, or two parents");
      return mlir::failure();
    }
    activeTiles.insert(hop.source);
    activeTiles.insert(hop.destination);
  }
  llvm::SmallVector<TileId, 2> roots;
  for (TileId tile : activeTiles)
    if (!parents.count(tile))
      roots.push_back(tile);
  if (roots.size() != 1) {
    setFailure(failureReason,
               "peer transfer graph does not have one payload root");
    return mlir::failure();
  }
  for (TileId tile : activeTiles) {
    TileId current = tile;
    size_t steps = 0;
    while (current != roots.front()) {
      auto parent = parents.find(current);
      if (parent == parents.end() || ++steps >= activeTiles.size()) {
        setFailure(failureReason,
                   "peer transfer graph is cyclic or disconnected");
        return mlir::failure();
      }
      current = parent->second;
    }
  }

  std::optional<int64_t> physicalBytes;
  std::optional<MemoryAttr> memory;
  for (TileId tile : activeTiles) {
    const MovementEndpointBinding *endpoint = findEndpoint(endpoints, tile);
    auto type =
        endpoint ? mlir::dyn_cast<mlir::MemRefType>(endpoint->buffer.getType())
                 : mlir::MemRefType{};
    std::optional<WaferPhysicalTensorInfo> info =
        type ? computeWaferPhysicalTensorInfo(type) : std::nullopt;
    MemoryAttr currentMemory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
    if (!endpoint || !info || info->physicalBytes <= 0 || !currentMemory ||
        (physicalBytes && *physicalBytes != info->physicalBytes) ||
        (memory && *memory != currentMemory)) {
      setFailure(failureReason,
                 "peer transfer endpoints have incompatible physical buffers");
      return mlir::failure();
    }
    physicalBytes = info->physicalBytes;
    memory = currentMemory;
  }
  if (!physicalBytes || static_cast<uint64_t>(*physicalBytes) >
                            std::numeric_limits<uint32_t>::max()) {
    setFailure(failureReason,
               "peer transfer payload is outside target byte range");
    return mlir::failure();
  }
  return PreparedPeerTransfer{std::move(actions), realization.hops,
                              static_cast<uint64_t>(*physicalBytes)};
}

mlir::FailureOr<EmittedPeerTransfer>
emitPreparedPeerTransfer(const PreparedPeerTransfer &prepared,
                         llvm::ArrayRef<MovementEndpointBinding> endpoints,
                         int64_t communication, int64_t payload,
                         std::string *failureReason) {
  if (communication < 0 || payload < 0 || prepared.hops.empty() ||
      prepared.physicalBytes == 0 ||
      prepared.physicalBytes > std::numeric_limits<uint32_t>::max())
    return fail(failureReason, "prepared peer transfer metadata is invalid");
  EmittedPeerTransfer emitted;
  for (auto [round, hop] : llvm::enumerate(prepared.hops)) {
    const MovementEndpointBinding *source = findEndpoint(endpoints, hop.source);
    const MovementEndpointBinding *destination =
        findEndpoint(endpoints, hop.destination);
    if (!source || !destination || !source->builder || !destination->builder)
      return fail(failureReason, "prepared peer transfer endpoint disappeared");
    auto message =
        DTEMessageAttr::get(source->builder->getContext(), communication,
                            static_cast<int64_t>(round), payload);
    auto send = source->builder->create<CommPeerSendOp>(
        source->buffer.getLoc(),
        mlir::async::TokenType::get(source->builder->getContext()),
        source->buffer,
        source->builder->getI64IntegerAttr(hop.destination.getValue()),
        source->builder->getI64IntegerAttr(
            static_cast<int64_t>(prepared.physicalBytes)),
        message);
    emitted.sendTokens.push_back(send.getToken());
    auto receive = destination->builder->create<CommPeerRecvOp>(
        destination->buffer.getLoc(),
        mlir::async::TokenType::get(destination->builder->getContext()),
        destination->buffer,
        destination->builder->getI64IntegerAttr(hop.source.getValue()),
        destination->builder->getI64IntegerAttr(
            static_cast<int64_t>(prepared.physicalBytes)),
        message);
    emitted.receiveTokens.push_back(receive.getToken());
    emitted.hops.push_back({hop, send.getToken(), receive.getToken()});
  }
  return emitted;
}

mlir::FailureOr<std::vector<PreparedSelectedPeerGraph>>
prepareSelectedPeerGraphs(const MovementPlan &plan,
                          llvm::ArrayRef<MovementResourceDescription> resources,
                          llvm::ArrayRef<SelectedPeerGraphBinding> bindings,
                          std::string *failureReason) {
  std::map<MovementActionId, const MovementResourceDescription *>
      resourcesByAction;
  for (const MovementResourceDescription &resource : resources)
    if (!resourcesByAction.try_emplace(resource.action, &resource).second) {
      setFailure(failureReason,
                 "selected movement resources contain a duplicate action");
      return mlir::failure();
    }
  std::map<std::vector<MovementActionId>, PeerTransferGraphPlan> selected;
  std::set<MovementActionId> movementActions;
  for (const ExternalLoadPlan &load : plan.externalLoads)
    movementActions.insert(MovementActionId(load.id));
  for (const DDRBoundaryTransferPlan &transfer : plan.ddrTransfers)
    movementActions.insert(MovementActionId(transfer.id));
  for (const ReductionGatherPlan &gather : plan.reductionGathers)
    movementActions.insert(MovementActionId(gather.id));
  std::set<MovementActionId> selectedActions;
  for (const PeerTransferGraphPlan &graph : plan.peerGraphs) {
    if (graph.actions.empty() || !llvm::is_sorted(graph.actions) ||
        std::adjacent_find(graph.actions.begin(), graph.actions.end()) !=
            graph.actions.end() ||
        llvm::any_of(graph.actions,
                     [&](const MovementActionId &action) {
                       return !movementActions.count(action) ||
                              !selectedActions.insert(action).second;
                     }) ||
        !selected.try_emplace(graph.actions, graph).second) {
      setFailure(failureReason,
                 "selected MovementPlan has a malformed peer graph set");
      return mlir::failure();
    }
  }

  using BindingKey = std::pair<std::vector<MovementActionId>, uint32_t>;
  std::set<BindingKey> seenBindings;
  size_t expectedBindingCount = 0;
  for (const auto &[actions, graph] : selected) {
    (void)graph;
    auto resource = resourcesByAction.find(actions.front());
    if (resource == resourcesByAction.end()) {
      setFailure(failureReason,
                 "selected peer graph has no exact payload resource");
      return mlir::failure();
    }
    auto normalized =
        analysis::normalizeFiniteExactIndexSet(resource->second->exactDomain);
    if (mlir::failed(normalized) || normalized->getBoxes().empty() ||
        expectedBindingCount > std::numeric_limits<uint32_t>::max() ||
        normalized->getBoxes().size() >
            std::numeric_limits<uint32_t>::max() - expectedBindingCount) {
      setFailure(failureReason,
                 "selected peer payload has no finite piece cover");
      return mlir::failure();
    }
    expectedBindingCount += normalized->getBoxes().size();
  }
  std::vector<PreparedSelectedPeerGraph> result;
  for (const SelectedPeerGraphBinding &binding : bindings) {
    std::vector<TileId> expectedTerminals;
    for (const MovementActionId &action : binding.actions) {
      if (const auto *load = std::get_if<ExternalLoadId>(&action)) {
        expectedTerminals.push_back(load->destination.work.tile);
        continue;
      }
      if (const auto *boundary = std::get_if<DDRBoundaryTransferId>(&action)) {
        expectedTerminals.push_back(boundary->destination.work.tile);
        continue;
      }
      const auto *gatherId = std::get_if<ReductionGatherId>(&action);
      auto gather =
          gatherId ? llvm::find_if(plan.reductionGathers,
                                   [&](const ReductionGatherPlan &candidate) {
                                     return candidate.id == *gatherId;
                                   })
                   : plan.reductionGathers.end();
      if (gather == plan.reductionGathers.end()) {
        setFailure(failureReason,
                   "selected peer graph action has no physical terminal");
        return mlir::failure();
      }
      expectedTerminals.push_back(
          std::visit([](const auto &source) { return source.work.tile; },
                     gather->mergeExecution.source));
    }
    llvm::sort(expectedTerminals, [](TileId lhs, TileId rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    if (!llvm::is_sorted(binding.actions) || binding.actions.empty() ||
        std::adjacent_find(binding.actions.begin(), binding.actions.end()) !=
            binding.actions.end() ||
        !llvm::is_sorted(binding.terminals,
                         [](TileId lhs, TileId rhs) {
                           return lhs.getValue() < rhs.getValue();
                         }) ||
        binding.terminals != expectedTerminals ||
        std::adjacent_find(binding.terminals.begin(),
                           binding.terminals.end()) !=
            binding.terminals.end() ||
        !seenBindings.insert({binding.actions, binding.payloadSlice}).second) {
      setFailure(failureReason,
                 "selected peer graph binding is malformed or duplicated");
      return mlir::failure();
    }
    auto realization = selected.find(binding.actions);
    if (realization == selected.end()) {
      setFailure(failureReason,
                 "selected peer graph binding has no MovementPlan owner");
      return mlir::failure();
    }
    auto payloadResource = resourcesByAction.find(binding.actions.front());
    if (payloadResource == resourcesByAction.end()) {
      setFailure(failureReason,
                 "selected peer graph binding has no payload resource");
      return mlir::failure();
    }
    auto normalized = analysis::normalizeFiniteExactIndexSet(
        payloadResource->second->exactDomain);
    if (mlir::failed(normalized) ||
        binding.payloadSlice >= normalized->getBoxes().size()) {
      setFailure(failureReason,
                 "selected peer graph binding names an unknown payload piece");
      return mlir::failure();
    }
    const analysis::StaticRectangularIndexSet &box =
        normalized->getBoxes()[binding.payloadSlice];
    if (binding.logicalOffsets != box.offsets ||
        binding.logicalSizes != box.sizes ||
        llvm::any_of(binding.actions,
                     [&](const MovementActionId &action) {
                       auto resource = resourcesByAction.find(action);
                       return resource == resourcesByAction.end() ||
                              resource->second->elementType !=
                                  payloadResource->second->elementType ||
                              !resource->second->exactDomain.getPresburgerSet()
                                   .isObviouslyEqual(
                                       payloadResource->second->exactDomain
                                           .getPresburgerSet());
                     }) ||
        llvm::any_of(
            binding.endpoints, [&](const MovementEndpointBinding &endpoint) {
              auto type = endpoint.buffer ? mlir::dyn_cast<mlir::MemRefType>(
                                                endpoint.buffer.getType())
                                          : mlir::MemRefType{};
              return !type ||
                     !llvm::equal(type.getShape(), binding.logicalSizes) ||
                     type.getElementType() !=
                         payloadResource->second->elementType;
            })) {
      setFailure(failureReason,
                 "selected peer graph binding does not match its exact piece");
      return mlir::failure();
    }
    const bool allExternal = llvm::all_of(binding.actions, [](const auto &id) {
      return std::holds_alternative<ExternalLoadId>(id);
    });
    if (allExternal != (realization->second.kind ==
                        PeerTransferGraphKind::ExternalLoadFanout)) {
      setFailure(failureReason,
                 "selected peer graph mixes external and resident sources");
      return mlir::failure();
    }
    std::set<int64_t> active;
    std::set<int64_t> nonLeaves;
    std::set<int64_t> terminals;
    for (TileId terminal : binding.terminals)
      terminals.insert(terminal.getValue());
    for (const MovementHop &hop : realization->second.hops) {
      active.insert(hop.source.getValue());
      active.insert(hop.destination.getValue());
      nonLeaves.insert(hop.source.getValue());
    }
    std::vector<int64_t> roots;
    for (int64_t tile : active)
      if (!llvm::any_of(realization->second.hops, [&](const MovementHop &hop) {
            return hop.destination.getValue() == tile;
          }))
        roots.push_back(tile);
    std::optional<TileId> ddrRoot;
    if (realization->second.kind == PeerTransferGraphKind::ExternalLoadFanout) {
      if (!realization->second.ddrRoot || roots.size() != 1 ||
          roots.front() !=
              realization->second.ddrRoot->destination.work.tile.getValue()) {
        setFailure(failureReason,
                   "external fanout graph has no matching DDR root");
        return mlir::failure();
      }
      ddrRoot = TileId(roots.front());
    } else if (realization->second.ddrRoot || binding.ddrSource) {
      setFailure(failureReason,
                 "non-external peer graph carries a DDR root binding");
      return mlir::failure();
    }
    if (llvm::any_of(
            terminals,
            [&](int64_t terminal) { return !active.count(terminal); }) ||
        llvm::any_of(active, [&](int64_t tile) {
          return !nonLeaves.count(tile) && !terminals.count(tile);
        })) {
      setFailure(failureReason,
                 "selected peer graph has a missing terminal or dead relay");
      return mlir::failure();
    }
    auto prepared =
        preparePeerTransfer(binding.actions.front(), realization->second,
                            binding.endpoints, failureReason);
    if (mlir::failed(prepared))
      return mlir::failure();
    if (ddrRoot) {
      const MovementEndpointBinding *root =
          findEndpoint(binding.endpoints, *ddrRoot);
      auto sourceType =
          binding.ddrSource
              ? mlir::dyn_cast<mlir::MemRefType>(binding.ddrSource.getType())
              : mlir::MemRefType{};
      std::optional<WaferPhysicalTensorInfo> sourceInfo =
          sourceType ? computeWaferPhysicalTensorInfo(sourceType)
                     : std::nullopt;
      if (!binding.ddrSource || !root || !isWaferDDRMemRefType(sourceType) ||
          !sourceInfo || sourceInfo->physicalBytes <= 0 ||
          static_cast<uint64_t>(sourceInfo->physicalBytes) !=
              prepared->physicalBytes) {
        setFailure(failureReason,
                   "external fanout has no exact DDR root payload binding");
        return mlir::failure();
      }
    }
    PreparedSelectedPeerGraph selectedGraph;
    selectedGraph.transfer = std::move(*prepared);
    selectedGraph.endpoints = binding.endpoints;
    selectedGraph.ddrSource = binding.ddrSource;
    selectedGraph.ddrRoot = ddrRoot;
    selectedGraph.payload = binding.payloadSlice;
    result.push_back(std::move(selectedGraph));
  }
  if (seenBindings.size() != expectedBindingCount) {
    setFailure(failureReason,
               "selected MovementPlan has an unbound peer payload graph");
    return mlir::failure();
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.transfer.actions, lhs.payload) <
           std::tie(rhs.transfer.actions, rhs.payload);
  });
  int64_t communication = -1;
  std::vector<MovementActionId> previousActions;
  for (PreparedSelectedPeerGraph &graph : result) {
    if (previousActions != graph.transfer.actions) {
      if (communication == std::numeric_limits<int64_t>::max()) {
        setFailure(failureReason,
                   "selected peer graph identity exceeds the target field");
        return mlir::failure();
      }
      ++communication;
      previousActions = graph.transfer.actions;
    }
    graph.communication = communication;
  }
  return result;
}

mlir::FailureOr<std::vector<EmittedSelectedPeerGraph>>
emitPreparedSelectedPeerGraphs(
    llvm::ArrayRef<PreparedSelectedPeerGraph> prepared,
    std::string *failureReason) {
  std::vector<EmittedSelectedPeerGraph> result;
  result.reserve(prepared.size());
  for (const PreparedSelectedPeerGraph &graph : prepared) {
    mlir::Operation *ddrRootLoad = nullptr;
    if (graph.ddrRoot) {
      const MovementEndpointBinding *root =
          findEndpoint(graph.endpoints, *graph.ddrRoot);
      if (!root || !root->builder || !graph.ddrSource)
        return fail(failureReason,
                    "prepared external fanout root binding disappeared");
      ddrRootLoad = root->builder
                        ->create<StorageLoadOp>(root->buffer.getLoc(),
                                                graph.ddrSource, root->buffer)
                        .getOperation();
    }
    auto emitted = emitPreparedPeerTransfer(graph.transfer, graph.endpoints,
                                            graph.communication, graph.payload,
                                            failureReason);
    if (mlir::failed(emitted))
      return mlir::failure();
    result.push_back(
        {graph.transfer.actions, std::move(*emitted), ddrRootLoad});
  }
  return result;
}

mlir::LogicalResult verifyTokenOnlySelectedPeerGraphs(
    llvm::ArrayRef<PreparedSelectedPeerGraph> prepared,
    llvm::ArrayRef<EmittedSelectedPeerGraph> emitted,
    std::string *failureReason) {
  if (prepared.size() != emitted.size())
    return fail(failureReason,
                "selected peer verifier found an incomplete graph set");
  llvm::DenseSet<mlir::Value> tokens;
  for (auto [expected, actual] : llvm::zip_equal(prepared, emitted)) {
    if (expected.transfer.actions != actual.actions ||
        expected.transfer.hops.size() != actual.transfer.hops.size() ||
        actual.transfer.sendTokens.size() != expected.transfer.hops.size() ||
        actual.transfer.receiveTokens.size() != expected.transfer.hops.size())
      return fail(failureReason,
                  "selected peer verifier found incomplete action or token "
                  "coverage");
    if (expected.ddrRoot.has_value() != (actual.ddrRootLoad != nullptr))
      return fail(failureReason,
                  "selected peer verifier found a missing or extra DDR root");
    if (expected.ddrRoot) {
      const MovementEndpointBinding *root =
          findEndpoint(expected.endpoints, *expected.ddrRoot);
      auto load = mlir::dyn_cast_if_present<StorageLoadOp>(actual.ddrRootLoad);
      if (!root || !load || load.getSource() != expected.ddrSource ||
          load.getDest() != root->buffer)
        return fail(failureReason,
                    "selected peer verifier found a mismatched DDR root load");
    }
    for (auto [round, expectedHop, actualHop] :
         llvm::enumerate(expected.transfer.hops, actual.transfer.hops)) {
      if (!(actualHop.hop == expectedHop) || !actualHop.sendToken ||
          !actualHop.receiveToken ||
          !tokens.insert(actualHop.sendToken).second ||
          !tokens.insert(actualHop.receiveToken).second)
        return fail(failureReason,
                    "selected peer verifier found a duplicate or mismatched "
                    "hop token");
      auto send = actualHop.sendToken.getDefiningOp<CommPeerSendOp>();
      auto receive = actualHop.receiveToken.getDefiningOp<CommPeerRecvOp>();
      if (!send || !receive ||
          send.getPeerAttr().getInt() != expectedHop.destination.getValue() ||
          receive.getPeerAttr().getInt() != expectedHop.source.getValue() ||
          send.getBytesAttr().getInt() !=
              static_cast<int64_t>(expected.transfer.physicalBytes) ||
          receive.getBytesAttr().getInt() !=
              static_cast<int64_t>(expected.transfer.physicalBytes) ||
          send.getMessageAttr() != receive.getMessageAttr() ||
          send.getMessageAttr().getCommunicationId() !=
              expected.communication ||
          send.getMessageAttr().getRound() != static_cast<int64_t>(round) ||
          send.getMessageAttr().getPayloadSlice() != expected.payload)
        return fail(failureReason,
                    "selected peer verifier found a message or endpoint "
                    "mismatch");
      if (llvm::any_of(actualHop.sendToken.getUsers(),
                       [](mlir::Operation *user) {
                         return mlir::isa<mlir::async::AwaitOp>(user);
                       }) ||
          llvm::any_of(actualHop.receiveToken.getUsers(),
                       [](mlir::Operation *user) {
                         return mlir::isa<mlir::async::AwaitOp>(user);
                       }))
        return fail(failureReason,
                    "movement emitter inserted an immediate peer await");
    }
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
