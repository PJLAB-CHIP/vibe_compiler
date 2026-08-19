//===- DataMovementApply.cpp - Apply selected movement ----------------===//

#include "Wafer/Planning/Search/DataMovementApply.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <map>

namespace wafer::compiler::detail {
namespace {

mlir::LogicalResult fail(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

TileModuleOp findTile(mlir::ModuleOp module, TileId tile) {
  TileModuleOp result;
  module.walk([&](TileModuleOp candidate) {
    if (candidate.getTileIdAttr().getInt() == tile.getValue())
      result = candidate;
  });
  return result;
}

mlir::Value
findNodeBuffer(llvm::ArrayRef<StructuredOperationBufferRelation> relations,
               StructuredDAGNodeID node, TileId tile, MemLayout layout,
               llvm::ArrayRef<int64_t> shape) {
  for (const StructuredOperationBufferRelation &relation : relations) {
    if (relation.structuredNodeId != node || !relation.buffer)
      continue;
    mlir::Value buffer = relation.buffer;
    auto owner = buffer.getParentRegion()->getParentOfType<TileModuleOp>();
    auto type = mlir::dyn_cast<mlir::MemRefType>(buffer.getType());
    if (owner && owner.getTileIdAttr().getInt() == tile.getValue() && type &&
        type.getShape() == shape &&
        getWaferMemoryAttr(type).getLayout() == layout)
      return buffer;
  }
  return {};
}

bool isObservableNode(const StructuredDAGAnalysis &dag,
                      StructuredDAGNodeID node) {
  return llvm::any_of(dag.getObservableOutputRootNodes(),
                      [&](llvm::ArrayRef<StructuredDAGNodeID> roots) {
                        return llvm::is_contained(roots, node);
                      });
}

void emitSend(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value buffer,
              TileId peer, uint64_t bytes, int64_t communication, int64_t round,
              int64_t payload) {
  auto message =
      DTEMessageAttr::get(builder.getContext(), communication, round, payload);
  auto send = builder.create<CommPeerSendOp>(
      loc, mlir::async::TokenType::get(builder.getContext()), buffer,
      builder.getI64IntegerAttr(peer.getValue()),
      builder.getI64IntegerAttr(static_cast<int64_t>(bytes)), message);
  builder.create<mlir::async::AwaitOp>(loc, send.getToken());
}

void emitReceive(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value buffer, TileId peer, uint64_t bytes,
                 int64_t communication, int64_t round, int64_t payload) {
  auto message =
      DTEMessageAttr::get(builder.getContext(), communication, round, payload);
  auto receive = builder.create<CommPeerRecvOp>(
      loc, mlir::async::TokenType::get(builder.getContext()), buffer,
      builder.getI64IntegerAttr(peer.getValue()),
      builder.getI64IntegerAttr(static_cast<int64_t>(bytes)), message);
  builder.create<mlir::async::AwaitOp>(loc, receive.getToken());
}

mlir::LogicalResult buildRelay(TileModuleOp tile,
                               const DataMovementFragment &fragment,
                               int64_t communication, int64_t payload,
                               size_t hop, std::string *failureReason) {
  if (hop == 0 || hop >= fragment.route.size())
    return fail(failureReason, "relay hop is outside selected route");
  TileLink incoming = fragment.route[hop - 1];
  TileLink outgoing = fragment.route[hop];
  if (incoming.destination != outgoing.source ||
      tile.getTileIdAttr().getInt() != incoming.destination.getValue())
    return fail(failureReason, "relay route is not contiguous");
  mlir::OpBuilder builder(&tile.getBody().front(),
                          tile.getBody().front().end());
  std::string name =
      (llvm::Twine("relay_message_") + llvm::Twine(communication) +
       "_payload_" + llvm::Twine(payload) + "_hop_" + llvm::Twine(hop))
          .str();
  auto function = mlir::func::FuncOp::create(tile.getLoc(), name,
                                             builder.getFunctionType({}, {}));
  function.setPrivate();
  tile.getBody().front().push_back(function);
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto region = builder.create<TileRegionOp>(tile.getLoc(), mlir::TypeRange{},
                                             mlir::ValueRange{});
  mlir::Block *body = new mlir::Block();
  region.getBody().push_back(body);
  mlir::OpBuilder regionBuilder = mlir::OpBuilder::atBlockBegin(body);
  auto tensorType = mlir::MemRefType::get(
      fragment.sizes, fragment.elementType, mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(tile.getContext(), MemorySpace::SPM,
                      fragment.transportLayout));
  auto allocation =
      regionBuilder.create<mlir::memref::AllocOp>(tile.getLoc(), tensorType);
  emitReceive(regionBuilder, tile.getLoc(), allocation, incoming.source,
              fragment.physicalBytes, communication,
              static_cast<int64_t>(hop - 1), payload);
  emitSend(regionBuilder, tile.getLoc(), allocation, outgoing.destination,
           fragment.physicalBytes, communication, static_cast<int64_t>(hop),
           payload);
  regionBuilder.create<mlir::memref::DeallocOp>(tile.getLoc(), allocation);
  regionBuilder.create<TileYieldOp>(tile.getLoc());
  builder.setInsertionPointAfter(region);
  builder.create<mlir::func::ReturnOp>(tile.getLoc());
  return mlir::success();
}

mlir::LogicalResult
buildMulticastRelay(TileModuleOp tile, const DataMovementFragment &fragment,
                    int64_t communication, int64_t payload, TileId parent,
                    llvm::ArrayRef<TileId> children, int64_t depth,
                    std::string *failureReason) {
  if (children.empty() || depth <= 0)
    return fail(failureReason, "multicast relay has no forwarding work");
  mlir::OpBuilder builder(&tile.getBody().front(),
                          tile.getBody().front().end());
  std::string name =
      (llvm::Twine("multicast_message_") + llvm::Twine(communication) +
       "_payload_" + llvm::Twine(payload) + "_tile_" +
       llvm::Twine(tile.getTileIdAttr().getInt()))
          .str();
  auto function = mlir::func::FuncOp::create(tile.getLoc(), name,
                                             builder.getFunctionType({}, {}));
  function.setPrivate();
  tile.getBody().front().push_back(function);
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto region = builder.create<TileRegionOp>(tile.getLoc(), mlir::TypeRange{},
                                             mlir::ValueRange{});
  mlir::Block *body = new mlir::Block();
  region.getBody().push_back(body);
  mlir::OpBuilder regionBuilder = mlir::OpBuilder::atBlockBegin(body);
  auto type = mlir::MemRefType::get(
      fragment.sizes, fragment.elementType, mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(tile.getContext(), MemorySpace::SPM,
                      fragment.transportLayout));
  auto allocation =
      regionBuilder.create<mlir::memref::AllocOp>(tile.getLoc(), type);
  emitReceive(regionBuilder, tile.getLoc(), allocation, parent,
              fragment.physicalBytes, communication, depth - 1, payload);
  for (TileId child : children)
    emitSend(regionBuilder, tile.getLoc(), allocation, child,
             fragment.physicalBytes, communication, depth, payload);
  regionBuilder.create<mlir::memref::DeallocOp>(tile.getLoc(), allocation);
  regionBuilder.create<TileYieldOp>(tile.getLoc());
  builder.setInsertionPointAfter(region);
  builder.create<mlir::func::ReturnOp>(tile.getLoc());
  return mlir::success();
}

} // namespace

mlir::LogicalResult applySelectedDataMovement(
    mlir::ModuleOp cardModule, const CardProgramAnalysis &program,
    const CardDataMovementAssignment &assignment,
    StructuredMaterializationRelations &relations, std::string *failureReason) {
  auto findLoad = [](mlir::Value buffer) {
    StorageLoadOp result;
    for (StorageLoadOp load : buffer.getParentBlock()->getOps<StorageLoadOp>())
      if (load.getDest() == buffer) {
        result = load;
        break;
      }
    return result;
  };
  auto findStore = [](mlir::Value buffer) {
    StorageStoreOp result;
    for (StorageStoreOp store :
         buffer.getParentBlock()->getOps<StorageStoreOp>())
      if (store.getSource() == buffer) {
        result = store;
        break;
      }
    if (result)
      return result;
    for (mlir::OpOperand &use : buffer.getUses()) {
      auto materialize = mlir::dyn_cast<LayoutMaterializeOp>(use.getOwner());
      if (!materialize || materialize.getSource() != buffer)
        continue;
      for (StorageStoreOp store :
           materialize->getBlock()->getOps<StorageStoreOp>())
        if (store.getSource() == materialize.getResult())
          return store;
    }
    return result;
  };
  auto getSubview =
      [&](mlir::OpBuilder &builder, mlir::Value buffer,
          llvm::ArrayRef<int64_t> offsets,
          llvm::ArrayRef<int64_t> sizes) -> mlir::FailureOr<mlir::Value> {
    auto type = mlir::dyn_cast<mlir::MemRefType>(buffer.getType());
    if (!type || offsets.size() != static_cast<size_t>(type.getRank()) ||
        sizes.size() != static_cast<size_t>(type.getRank()))
      return mlir::failure();
    bool complete = llvm::all_of(
        llvm::zip_equal(offsets, sizes, type.getShape()), [](auto values) {
          auto [offset, size, extent] = values;
          return offset == 0 && size == extent;
        });
    if (complete)
      return buffer;
    llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (int64_t offset : offsets)
      mixedOffsets.push_back(builder.getIndexAttr(offset));
    for (int64_t size : sizes)
      mixedSizes.push_back(builder.getIndexAttr(size));
    for (size_t dimension = 0; dimension < offsets.size(); ++dimension)
      strides.push_back(builder.getIndexAttr(1));
    auto viewType = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferRankReducedResultType(
            sizes, type, mixedOffsets, mixedSizes, strides));
    return builder
        .create<mlir::memref::SubViewOp>(buffer.getLoc(), viewType, buffer,
                                         mixedOffsets, mixedSizes, strides)
        .getResult();
  };

  llvm::DenseMap<mlir::Operation *, StructuredDAGNodeID> removableStores;
  std::map<int64_t, llvm::SmallVector<const DataMovementChoice *, 8>>
      multicastGroups;
  for (const DataMovementChoice &movement : assignment.edges)
    if (movement.multicastGroup >= 0)
      multicastGroups[movement.multicastGroup].push_back(&movement);

  for (const auto &[groupId, choices] : multicastGroups) {
    if (choices.size() < 2 || choices.front()->fragments.size() != 1)
      return fail(failureReason, "multicast group is malformed");
    const DataMovementChoice &firstChoice = *choices.front();
    const DataMovementFragment &fragment = firstChoice.fragments.front();
    const StructuredDAGEdge *edge = program.dag.getEdge(firstChoice.edge);
    if (!edge)
      return fail(failureReason, "multicast edge is unavailable");
    std::map<int64_t, int64_t> parents;
    std::map<int64_t, llvm::SmallVector<TileId, 4>> children;
    llvm::DenseSet<int64_t> destinations;
    for (const DataMovementChoice *choice : choices) {
      destinations.insert(choice->destinationTile.getValue());
      for (const TileLink &link : choice->fragments.front().route) {
        auto [parent, inserted] = parents.emplace(link.destination.getValue(),
                                                  link.source.getValue());
        if (!inserted && parent->second != link.source.getValue())
          return fail(failureReason,
                      "multicast routes do not form one rooted tree");
        auto &next = children[link.source.getValue()];
        if (!llvm::is_contained(next, link.destination))
          next.push_back(link.destination);
      }
    }
    for (auto &[node, next] : children) {
      (void)node;
      llvm::sort(next, [](TileId lhs, TileId rhs) {
        return lhs.getValue() < rhs.getValue();
      });
    }
    std::map<int64_t, int64_t> depths;
    depths[fragment.sourceTile.getValue()] = 0;
    llvm::SmallVector<TileId, 16> queue{fragment.sourceTile};
    for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
      TileId node = queue[cursor];
      for (TileId child : children[node.getValue()]) {
        depths[child.getValue()] = depths[node.getValue()] + 1;
        queue.push_back(child);
      }
    }

    mlir::Value source = findNodeBuffer(
        relations.operationResultBuffers, edge->producer, fragment.sourceTile,
        fragment.sourceLayout, fragment.sourceShape);
    StorageStoreOp sourceStore = source ? findStore(source) : StorageStoreOp{};
    if (!source || !sourceStore)
      return fail(failureReason, "multicast source version is unavailable");
    mlir::OpBuilder sourceBuilder(sourceStore);
    llvm::SmallVector<int64_t, 4> sourceOffsets;
    for (auto [offset, base] :
         llvm::zip_equal(fragment.offsets, fragment.sourceBaseOffsets))
      sourceOffsets.push_back(offset - base);
    auto sourceView =
        getSubview(sourceBuilder, source, sourceOffsets, fragment.sizes);
    if (mlir::failed(sourceView))
      return fail(failureReason, "multicast source subview is not exact");
    mlir::Value transportSource = *sourceView;
    if (fragment.sourceLayout != fragment.transportLayout) {
      auto sourceType = mlir::cast<mlir::MemRefType>(sourceView->getType());
      auto transportType = mlir::MemRefType::get(
          sourceType.getShape(), sourceType.getElementType(),
          mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(source.getContext(), MemorySpace::SPM,
                          fragment.transportLayout));
      transportSource = sourceBuilder
                            .create<LayoutMaterializeOp>(
                                source.getLoc(), transportType, *sourceView)
                            .getResult();
    }
    for (TileId child : children[fragment.sourceTile.getValue()])
      emitSend(sourceBuilder, source.getLoc(), transportSource, child,
               fragment.physicalBytes, edge->id, /*round=*/0, groupId);
    removableStores[sourceStore.getOperation()] = edge->producer;

    for (const DataMovementChoice *choice : choices) {
      const DataMovementFragment &destinationFragment =
          choice->fragments.front();
      mlir::Value destination = findNodeBuffer(
          relations.operandBuffers, edge->consumer, choice->destinationTile,
          choice->consumerLayout, destinationFragment.sizes);
      if (!destination)
        return fail(failureReason,
                    "multicast destination version is unavailable");
      StorageLoadOp load = findLoad(destination);
      LayoutMaterializeOp materialize;
      if (!load) {
        materialize = destination.getDefiningOp<LayoutMaterializeOp>();
        if (materialize)
          load = findLoad(materialize.getSource());
        if (load && materialize) {
          mlir::OpBuilder replacementBuilder(load);
          auto replacement = replacementBuilder.create<mlir::memref::AllocOp>(
              destination.getLoc(),
              mlir::cast<mlir::MemRefType>(destination.getType()));
          destination.replaceAllUsesWith(replacement.getResult());
          for (StructuredOperationBufferRelation &relation :
               relations.operandBuffers)
            if (relation.buffer == destination)
              relation.buffer = replacement.getResult();
          destination = replacement.getResult();
          materialize.erase();
        }
      }
      if (!load)
        return fail(failureReason,
                    "multicast destination has no replaceable load");
      auto parent = parents.find(choice->destinationTile.getValue());
      auto depth = depths.find(choice->destinationTile.getValue());
      if (parent == parents.end() || depth == depths.end())
        return fail(failureReason,
                    "multicast destination is outside selected tree");
      mlir::OpBuilder destinationBuilder(load);
      emitReceive(destinationBuilder, destination.getLoc(), destination,
                  TileId(parent->second), fragment.physicalBytes, edge->id,
                  depth->second - 1, groupId);
      for (TileId child : children[choice->destinationTile.getValue()])
        emitSend(destinationBuilder, destination.getLoc(), destination, child,
                 fragment.physicalBytes, edge->id, depth->second, groupId);
      load.erase();
    }

    for (const auto &[node, next] : children) {
      if (node == fragment.sourceTile.getValue() ||
          destinations.contains(node) || next.empty())
        continue;
      auto parent = parents.find(node);
      auto depth = depths.find(node);
      TileModuleOp relay = findTile(cardModule, TileId(node));
      if (!relay || parent == parents.end() || depth == depths.end() ||
          mlir::failed(buildMulticastRelay(relay, fragment, edge->id, groupId,
                                           TileId(parent->second), next,
                                           depth->second, failureReason)))
        return mlir::failure();
    }
  }

  for (const ReductionGatherChoice &reduction : assignment.reductions) {
    if (reduction.kind != ReductionGatherKind::Peer)
      continue;
    const int64_t communication = (int64_t{1} << 32) +
                                  (static_cast<int64_t>(reduction.node) << 16) +
                                  reduction.resultIndex;
    for (auto [fragmentIndex, fragment] :
         llvm::enumerate(reduction.fragments)) {
      mlir::Value source;
      for (const PartialReductionContributionBufferRelation &relation :
           relations.partialReductionContributions) {
        auto type = mlir::dyn_cast<mlir::MemRefType>(relation.buffer.getType());
        if (relation.structuredNodeId == reduction.node &&
            relation.resultIndex == reduction.resultIndex &&
            relation.sourceTile == fragment.sourceTile && type &&
            llvm::equal(type.getShape(), fragment.sourceShape) &&
            getWaferMemoryAttr(type).getLayout() == fragment.sourceLayout) {
          source = relation.buffer;
          break;
        }
      }
      mlir::Value destination;
      for (const PartialReductionMergeInputBufferRelation &relation :
           relations.partialReductionMergeInputs) {
        auto type = mlir::dyn_cast<mlir::MemRefType>(relation.buffer.getType());
        if (relation.structuredNodeId == reduction.node &&
            relation.resultIndex == reduction.resultIndex &&
            relation.sourceTile == fragment.sourceTile && type &&
            llvm::equal(type.getShape(), fragment.sizes) &&
            getWaferMemoryAttr(type).getLayout() == reduction.mergeLayout) {
          destination = relation.buffer;
          break;
        }
      }
      if (!source || !destination) {
        std::string detail;
        llvm::raw_string_ostream diagnostic(detail);
        diagnostic << "reduction gather has no exact contribution buffers"
                   << "; node=" << reduction.node
                   << ", source_tile=" << fragment.sourceTile.getValue()
                   << ", source_layout="
                   << stringifyMemLayout(fragment.sourceLayout)
                   << ", merge_layout="
                   << stringifyMemLayout(reduction.mergeLayout)
                   << ", contribution_relations="
                   << relations.partialReductionContributions.size()
                   << ", merge_relations="
                   << relations.partialReductionMergeInputs.size();
        diagnostic << ", contribution_types=[";
        llvm::interleaveComma(
            relations.partialReductionContributions, diagnostic,
            [&](const PartialReductionContributionBufferRelation &relation) {
              diagnostic << relation.sourceTile.getValue() << ':'
                         << relation.buffer.getType();
            });
        diagnostic << "], merge_types=[";
        llvm::interleaveComma(
            relations.partialReductionMergeInputs, diagnostic,
            [&](const PartialReductionMergeInputBufferRelation &relation) {
              diagnostic << relation.sourceTile.getValue() << ':'
                         << relation.buffer.getType();
            });
        diagnostic << ']';
        return fail(failureReason, diagnostic.str());
      }
      if (fragment.sourceTile == reduction.mergeTile)
        continue;
      StorageStoreOp sourceStore = findStore(source);
      StorageLoadOp destinationLoad = findLoad(destination);
      if (!sourceStore || !destinationLoad || fragment.route.empty())
        return fail(failureReason,
                    !sourceStore
                        ? "reduction gather contribution has no DDR store"
                        : (!destinationLoad
                               ? "reduction gather input has no DDR load"
                               : "reduction gather route is empty"));
      mlir::OpBuilder sourceBuilder(sourceStore);
      mlir::Value transportSource = source;
      if (fragment.sourceLayout != fragment.transportLayout) {
        auto sourceType = mlir::cast<mlir::MemRefType>(source.getType());
        auto transportType = mlir::MemRefType::get(
            sourceType.getShape(), sourceType.getElementType(),
            mlir::MemRefLayoutAttrInterface{},
            MemoryAttr::get(source.getContext(), MemorySpace::SPM,
                            fragment.transportLayout));
        transportSource = sourceBuilder
                              .create<LayoutMaterializeOp>(
                                  source.getLoc(), transportType, source)
                              .getResult();
      }
      const int64_t payload = reduction.mergeTile.getValue() * 1024 +
                              static_cast<int64_t>(fragmentIndex);
      emitSend(sourceBuilder, source.getLoc(), transportSource,
               fragment.route.front().destination, fragment.physicalBytes,
               communication,
               /*round=*/0, payload);
      mlir::OpBuilder destinationBuilder(destinationLoad);
      emitReceive(destinationBuilder, destination.getLoc(), destination,
                  fragment.route.back().source, fragment.physicalBytes,
                  communication,
                  static_cast<int64_t>(fragment.route.size() - 1), payload);
      destinationLoad.erase();
      sourceStore.erase();
      for (size_t hop = 1; hop < fragment.route.size(); ++hop) {
        TileModuleOp relay =
            findTile(cardModule, fragment.route[hop - 1].destination);
        if (!relay || mlir::failed(buildRelay(relay, fragment, communication,
                                              payload, hop, failureReason)))
          return mlir::failure();
      }
    }
  }

  for (const DataMovementChoice &movement : assignment.edges) {
    if (movement.multicastGroup >= 0)
      continue;
    if (movement.kind == DataMovementKind::Refetch) {
      const StructuredDAGEdge *edge = program.dag.getEdge(movement.edge);
      if (!edge || movement.fragments.size() != 1)
        return fail(failureReason,
                    "refetch requires one exact retained fragment");
      const DataMovementFragment &fragment = movement.fragments.front();
      if (fragment.sourceTile != movement.destinationTile)
        return fail(failureReason, "refetch fragment is not Tile-local");
      mlir::Value source = findNodeBuffer(
          relations.operationResultBuffers, edge->producer, fragment.sourceTile,
          fragment.sourceLayout, fragment.sourceShape);
      mlir::Value destination = findNodeBuffer(
          relations.operandBuffers, edge->consumer, movement.destinationTile,
          movement.consumerLayout, fragment.sizes);
      if (!source || !destination ||
          source.getParentBlock() != destination.getParentBlock())
        return fail(failureReason,
                    "refetch has no same-region source/destination version");
      llvm::SmallVector<mlir::Operation *, 4> consumers;
      for (const StructuredOperationEmissionRelation &relation :
           relations.operationEmissions) {
        if (relation.structuredNodeId != edge->consumer ||
            !relation.operation ||
            relation.operation->getBlock() != source.getParentBlock())
          continue;
        if (llvm::is_contained(relation.operation->getOperands(), destination))
          consumers.push_back(relation.operation);
      }
      if (consumers.empty())
        return fail(failureReason,
                    "refetch destination has no structured consumer use");
      llvm::sort(consumers, [](mlir::Operation *lhs, mlir::Operation *rhs) {
        return lhs->isBeforeInBlock(rhs);
      });
      mlir::OpBuilder builder(consumers.front());
      auto tensor = mlir::cast<mlir::MemRefType>(destination.getType());
      auto ddrType = mlir::MemRefType::get(
          tensor.getShape(), tensor.getElementType(),
          mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(tensor.getContext(), MemorySpace::DDR,
                          MemLayout::Tensor));
      auto stage =
          builder.create<mlir::memref::AllocOp>(destination.getLoc(), ddrType);
      builder.create<StorageStoreOp>(destination.getLoc(), source, stage);
      auto reloaded =
          builder.create<mlir::memref::AllocOp>(destination.getLoc(), tensor);
      builder.create<StorageLoadOp>(destination.getLoc(), stage, reloaded);
      builder.create<mlir::memref::DeallocOp>(destination.getLoc(), stage);
      for (mlir::Operation *consumer : consumers)
        for (mlir::OpOperand &operand : consumer->getOpOperands())
          if (operand.get() == destination)
            operand.set(reloaded);
      for (StructuredOperationBufferRelation &relation :
           relations.operandBuffers)
        if (relation.structuredNodeId == edge->consumer &&
            relation.buffer == destination)
          relation.buffer = reloaded;
      mlir::OpBuilder releaseBuilder(consumers.back());
      releaseBuilder.setInsertionPointAfter(consumers.back());
      releaseBuilder.create<mlir::memref::DeallocOp>(destination.getLoc(),
                                                     reloaded);
      continue;
    }
    if (movement.kind != DataMovementKind::Peer)
      continue;
    const StructuredDAGEdge *edge = program.dag.getEdge(movement.edge);
    if (!edge || movement.fragments.empty())
      return fail(failureReason, "peer apply has no exact edge fragments");

    llvm::SmallVector<int64_t, 4> destinationShape;
    for (const DataMovementFragment &fragment : movement.fragments) {
      if (destinationShape.empty())
        destinationShape.assign(fragment.sizes.size(), 0);
      if (fragment.destinationOffsets.size() != fragment.sizes.size() ||
          destinationShape.size() != fragment.sizes.size())
        return fail(failureReason, "peer destination fragment rank mismatch");
      for (auto [dimension, offset, size] :
           llvm::enumerate(fragment.destinationOffsets, fragment.sizes))
        destinationShape[dimension] =
            std::max(destinationShape[dimension], offset + size);
    }
    mlir::Value destination = findNodeBuffer(
        relations.operandBuffers, edge->consumer, movement.destinationTile,
        movement.consumerLayout, destinationShape);
    if (!destination)
      return fail(failureReason,
                  "peer movement has no exact destination version");

    StorageLoadOp destinationLoad = findLoad(destination);
    LayoutMaterializeOp destinationMaterialize;
    if (!destinationLoad) {
      destinationMaterialize = destination.getDefiningOp<LayoutMaterializeOp>();
      mlir::Value staged = destinationMaterialize
                               ? destinationMaterialize.getSource()
                               : mlir::Value{};
      if (staged)
        destinationLoad = findLoad(staged);
      if (destinationLoad && destinationMaterialize) {
        mlir::OpBuilder replacementBuilder(destinationLoad);
        auto replacement = replacementBuilder.create<mlir::memref::AllocOp>(
            destination.getLoc(),
            mlir::cast<mlir::MemRefType>(destination.getType()));
        destination.replaceAllUsesWith(replacement.getResult());
        for (StructuredOperationBufferRelation &relation :
             relations.operandBuffers)
          if (relation.buffer == destination)
            relation.buffer = replacement.getResult();
        destination = replacement.getResult();
        destinationMaterialize.erase();
      }
    }
    if (!destinationLoad)
      return fail(failureReason,
                  "peer destination has no replaceable DDR load");
    mlir::OpBuilder destinationBuilder(destinationLoad);

    for (auto [fragmentIndex, fragment] : llvm::enumerate(movement.fragments)) {
      if (fragment.sourceTile == movement.destinationTile) {
        auto sourceView =
            getSubview(destinationBuilder, destinationLoad.getSource(),
                       fragment.destinationOffsets, fragment.sizes);
        auto destinationView =
            getSubview(destinationBuilder, destination,
                       fragment.destinationOffsets, fragment.sizes);
        if (mlir::failed(sourceView) || mlir::failed(destinationView))
          return fail(failureReason,
                      "local movement fragment has no exact subview");
        destinationBuilder.create<StorageLoadOp>(destination.getLoc(),
                                                 *sourceView, *destinationView);
        continue;
      }
      if (fragment.route.empty() || fragment.physicalBytes == 0 ||
          fragment.physicalBytes >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          fragment.sourceBaseOffsets.size() != fragment.offsets.size() ||
          fragment.sourceShape.size() != fragment.offsets.size())
        return fail(failureReason, "peer fragment is not representable");

      mlir::Value source = findNodeBuffer(
          relations.operationResultBuffers, edge->producer, fragment.sourceTile,
          fragment.sourceLayout, fragment.sourceShape);
      if (!source)
        return fail(failureReason, "peer movement has no exact source version");
      StorageStoreOp sourceStore = findStore(source);
      if (!sourceStore)
        return fail(failureReason,
                    "peer source has no ordered writeback point");
      mlir::OpBuilder sourceBuilder(sourceStore);
      llvm::SmallVector<int64_t, 4> sourceOffsets;
      for (auto [offset, base] :
           llvm::zip_equal(fragment.offsets, fragment.sourceBaseOffsets)) {
        if (offset < base)
          return fail(failureReason, "peer source fragment precedes its shard");
        sourceOffsets.push_back(offset - base);
      }
      auto sourceView =
          getSubview(sourceBuilder, source, sourceOffsets, fragment.sizes);
      if (mlir::failed(sourceView))
        return fail(failureReason, "peer source subview is not exact");
      mlir::Value transportSource = *sourceView;
      if (fragment.sourceLayout != fragment.transportLayout) {
        auto sourceType =
            mlir::cast<mlir::MemRefType>(transportSource.getType());
        auto transportType = mlir::MemRefType::get(
            sourceType.getShape(), sourceType.getElementType(),
            mlir::MemRefLayoutAttrInterface{},
            MemoryAttr::get(source.getContext(), MemorySpace::SPM,
                            fragment.transportLayout));
        transportSource =
            sourceBuilder
                .create<LayoutMaterializeOp>(source.getLoc(), transportType,
                                             transportSource)
                .getResult();
      }
      const int64_t payload = movement.destinationTile.getValue() * 1024 +
                              static_cast<int64_t>(fragmentIndex);
      emitSend(sourceBuilder, source.getLoc(), transportSource,
               fragment.route.front().destination, fragment.physicalBytes,
               movement.edge, /*round=*/0, payload);

      auto destinationView =
          getSubview(destinationBuilder, destination,
                     fragment.destinationOffsets, fragment.sizes);
      if (mlir::failed(destinationView))
        return fail(failureReason, "peer destination subview is not exact");
      emitReceive(destinationBuilder, destination.getLoc(), *destinationView,
                  fragment.route.back().source, fragment.physicalBytes,
                  movement.edge,
                  static_cast<int64_t>(fragment.route.size() - 1), payload);

      for (size_t hop = 1; hop < fragment.route.size(); ++hop) {
        TileModuleOp relay =
            findTile(cardModule, fragment.route[hop - 1].destination);
        if (!relay || mlir::failed(buildRelay(relay, fragment, movement.edge,
                                              payload, hop, failureReason)))
          return mlir::failure();
      }
      removableStores[sourceStore.getOperation()] = edge->producer;
    }
    destinationLoad.erase();
  }
  for (const auto &[store, producer] : removableStores) {
    bool hasDDRConsumer =
        llvm::any_of(assignment.edges, [&](const DataMovementChoice &movement) {
          const StructuredDAGEdge *edge = program.dag.getEdge(movement.edge);
          return edge && edge->producer == producer &&
                 movement.kind == DataMovementKind::DDR;
        });
    if (!hasDDRConsumer && !isObservableNode(program.dag, producer))
      store->erase();
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
