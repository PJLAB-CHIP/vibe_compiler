//===- SelectedPeerMaterialization.cpp - Selected peer endpoints ---------===//

#include "SelectedEdgeLoweringInternal.h"

#include "EdgeDomain.h"
#include "EdgeFragmentPlanning.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/Twine.h"

#include <limits>
#include <optional>
#include <tuple>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {
namespace {

std::optional<uint64_t> getElementBytes(mlir::Type type) {
  unsigned bits = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    bits = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    bits = floating.getWidth();
  if (bits == 0 || bits % 8 != 0)
    return std::nullopt;
  return bits / 8;
}

std::optional<uint64_t> getDomainElements(llvm::ArrayRef<int64_t> sizes) {
  uint64_t result = 1;
  for (int64_t size : sizes) {
    if (size <= 0 || result > std::numeric_limits<uint64_t>::max() /
                                  static_cast<uint64_t>(size))
      return std::nullopt;
    result *= static_cast<uint64_t>(size);
  }
  return result;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>> getFragmentStreamTileSizes(
    mlir::Operation *producer, unsigned result,
    llvm::ArrayRef<StructuredOpTemporalTile> temporalTiles,
    const SpatialEdgeFragment &fragment, std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> selected =
      getResultTemporalTileSizes(producer, result, temporalTiles,
                                 failureReason);
  if (mlir::failed(selected) || selected->size() != fragment.sizes.size())
    return mlir::failure();
  for (auto [tile, size] : llvm::zip_equal(*selected, fragment.sizes)) {
    if (tile <= 0 || size <= 0)
      return mlir::failure();
    tile = std::min(tile, size);
  }
  return selected;
}

mlir::Value getDDRStageBuffer(mlir::Value tensor) {
  llvm::DenseSet<mlir::Value> visited;
  while (tensor && visited.insert(tensor).second) {
    if (auto slice = tensor.getDefiningOp<mlir::tensor::ExtractSliceOp>()) {
      tensor = slice.getSource();
      continue;
    }
    if (auto toTensor = tensor.getDefiningOp<mlir::bufferization::ToTensorOp>())
      return isWaferDDRMemRefType(toTensor.getMemref().getType())
                 ? toTensor.getMemref()
                 : mlir::Value{};
    return {};
  }
  return {};
}

} // namespace

bool selectedValueReachesDDRStage(SelectedEdgeLoweringState &state,
                                  mlir::Operation *source) {
  llvm::DenseSet<mlir::Value> selectedBuffers;
  for (const CandidateSelectedDDRStage &stage : state.selectedDDRStages)
    selectedBuffers.insert(stage.buffer);
  bool reachesStage = false;
  state.candidate->walk([&](mlir::tensor::InsertSliceOp insert) {
    if (reachesStage)
      return;
    mlir::Value destination = insert.getDest();
    while (auto prior =
               destination.getDefiningOp<mlir::tensor::InsertSliceOp>())
      destination = prior.getDest();
    auto toTensor =
        destination.getDefiningOp<mlir::bufferization::ToTensorOp>();
    auto allocation =
        toTensor ? toTensor.getMemref().getDefiningOp<mlir::memref::AllocOp>()
                 : mlir::memref::AllocOp{};
    if (!allocation || !selectedBuffers.contains(allocation.getResult()))
      return;
    llvm::DenseSet<mlir::Value> visited;
    reachesStage = isInBackwardClosure(insert.getSource(), source, visited);
  });
  return reachesStage;
}

mlir::LogicalResult
materializeSelectedPeerReceives(SelectedEdgeLoweringState &state) {
  TensorProgramScope scope = state.scope;
  TileId currentTile = state.currentTile;
  auto &mappedTemporalTiles = state.mapping.operationTemporalTiles;
  auto &mappedOperationNodes = state.mapping.operationNodes;
  auto &mappedConsumerInputDemands = state.mapping.consumerInputDemands;
  auto &mappedStrategies = state.mapping.strategies;
  auto operationNodes = state.sourceOperationNodes;
  const bool independentDDRStages = state.mapping.independentDDRStages;
  auto &endpoints = state.endpoints;
  auto &materialized = state.materializedSources;
  auto &selectedDDRStages = state.selectedDDRStages;
  auto &preserved = state.preservedOperations;
  auto &producerValues = state.producerValues;
  std::string *failureReason = state.failureReason;
  llvm::SmallVector<MappedStrategy *, 16> peerOrder;
  for (MappedStrategy &mapped : mappedStrategies)
    if (mapped.strategy.action == SpatialEdgeAction::PeerFragments)
      peerOrder.push_back(&mapped);
  llvm::sort(
      peerOrder, [](const MappedStrategy *lhs, const MappedStrategy *rhs) {
        if (lhs->consumer != rhs->consumer)
          return lhs->consumer->isBeforeInBlock(rhs->consumer);
        if (lhs->strategy.consumerOperand != rhs->strategy.consumerOperand)
          return lhs->strategy.consumerOperand < rhs->strategy.consumerOperand;
        if (lhs->producer != rhs->producer)
          return lhs->producer->isBeforeInBlock(rhs->producer);
        return lhs->strategy.destinationTile.getValue() <
               rhs->strategy.destinationTile.getValue();
      });
  // PeerFragments is the only action with an all-and-only fragment assembly.
  for (size_t groupBegin = 0; groupBegin < peerOrder.size();) {
    size_t groupEnd = groupBegin + 1;
    while (groupEnd < peerOrder.size() &&
           peerOrder[groupEnd]->consumer == peerOrder[groupBegin]->consumer &&
           peerOrder[groupEnd]->strategy.consumerOperand ==
               peerOrder[groupBegin]->strategy.consumerOperand)
      ++groupEnd;
    for (size_t peerIndex = groupBegin; peerIndex < groupEnd; ++peerIndex) {
      MappedStrategy &mapped = *peerOrder[peerIndex];
      SpatialEdgeStrategy &strategy = mapped.strategy;
      if (strategy.fragments.empty())
        return reportSelectedEdgeFailure(
            failureReason, "peer edge strategy has no exact fragments");

      uint64_t coveredElements = 0;
      for (size_t index = 0; index < strategy.fragments.size(); ++index) {
        const SpatialEdgeFragment &fragment = strategy.fragments[index];
        if (!isContained(fragment.offsets, fragment.sizes,
                         strategy.producerOffsets, strategy.producerSizes))
          return reportSelectedEdgeFailure(
              failureReason,
              "dependent fragment extends outside its consumer demand");
        std::optional<uint64_t> elements = getDomainElements(fragment.sizes);
        if (!elements ||
            *elements > std::numeric_limits<uint64_t>::max() - coveredElements)
          return reportSelectedEdgeFailure(
              failureReason,
              "dependent fragment coverage is not representable");
        for (size_t previous = 0; previous < index; ++previous)
          if (overlaps(fragment.offsets, fragment.sizes,
                       strategy.fragments[previous].offsets,
                       strategy.fragments[previous].sizes))
            return reportSelectedEdgeFailure(
                failureReason,
                "dependent fragments overlap within one consumer demand");
        coveredElements += *elements;

        if (fragment.kind == SpatialEdgeFragmentKind::Resident) {
          if (fragment.sourceTile != strategy.destinationTile ||
              fragment.bytes != 0 || fragment.communicationId != 0 ||
              fragment.payloadSlice != 0)
            return reportSelectedEdgeFailure(
                failureReason,
                "resident fragment must be local and carry no peer message");
          continue;
        }
        auto producerType = mlir::cast<mlir::RankedTensorType>(
            mapped.producer->getResult(strategy.producerResult).getType());
        std::optional<uint64_t> elementBytes =
            getElementBytes(producerType.getElementType());
        if (!elementBytes ||
            *elements > std::numeric_limits<uint64_t>::max() / *elementBytes ||
            fragment.bytes != *elements * *elementBytes)
          return reportSelectedEdgeFailure(
              failureReason, "fragment has an invalid exact byte count");
        if (fragment.kind == SpatialEdgeFragmentKind::CardDDR) {
          if (!fragment.cardDDRResource || fragment.communicationId != 0 ||
              fragment.payloadSlice != 0 ||
              !mapped.cardDDRBoundaries.count(*fragment.cardDDRResource))
            return reportSelectedEdgeFailure(
                failureReason,
                "card DDR fragment has no typed boundary or carries a peer "
                "message");
          continue;
        }
        if (fragment.kind != SpatialEdgeFragmentKind::Peer ||
            fragment.cardDDRResource ||
            fragment.sourceTile == strategy.destinationTile ||
            fragment.communicationId < 0 || fragment.payloadSlice < 0)
          return reportSelectedEdgeFailure(
              failureReason,
              "peer fragment has invalid endpoint, bytes, or message");
      }
      std::optional<uint64_t> demandedElements =
          getDomainElements(strategy.producerSizes);
      if (!demandedElements || (!strategy.fragmentsDefineProducerDemand &&
                                coveredElements != *demandedElements))
        return reportSelectedEdgeFailure(
            failureReason,
            "dependent fragments do not exactly cover the consumer demand");

      if (strategy.destinationTile == currentTile) {
        auto producerType = mlir::cast<mlir::RankedTensorType>(
            mapped.producer->getResult(strategy.producerResult).getType());
        mlir::OpBuilder builder(mapped.consumer);
        mlir::Location assemblyLoc = mapped.consumer->getLoc();
        mlir::Value assembled;
        mlir::memref::AllocOp independentAssemblyAllocation;
        std::optional<uint32_t> producerNode =
            findStructuredNodeId(mapped.producer, mappedOperationNodes);
        std::optional<uint32_t> consumerNode =
            findStructuredNodeId(mapped.consumer, mappedOperationNodes);
        const bool sameStructuredRoot =
            producerNode && consumerNode && *producerNode == *consumerNode;
        const bool assembleInDDR = independentDDRStages && !sameStructuredRoot;
        if (assembleInDDR) {
          // The logical full-domain assembly may be much larger than SPM.  The
          // baseline stages each exact peer fragment into compiler-owned DDR;
          // downstream temporal waves then load only their demanded windows.
          // Keeping the full carrier in SPM would make temporal refinement
          // ineffective because allocation fails before the consumer wave.
          auto ddrType = mlir::MemRefType::get(
              producerType.getShape(), producerType.getElementType(),
              mlir::MemRefLayoutAttrInterface{},
              MemoryAttr::get(mapped.consumer->getContext(), MemorySpace::DDR,
                              MemLayout::Tensor));
          auto allocation =
              builder.create<mlir::memref::AllocOp>(assemblyLoc, ddrType);
          independentAssemblyAllocation = allocation;
          if (!producerNode)
            return reportSelectedEdgeFailure(
                failureReason,
                "peer DDR stage producer has no structured node identity");
          auto producerMapping =
              llvm::find_if(mappedOperationNodes,
                            [&](const StructuredOperationNodeMapping &mapping) {
                              return mapping.operation == mapped.producer;
                            });
          const bool coupledComponent =
              producerMapping != mappedOperationNodes.end() &&
              !producerMapping->coupledComponentIndices.empty();
          if (coupledComponent &&
              strategy.producerResult >=
                  producerMapping->coupledComponentIndices.size())
            return reportSelectedEdgeFailure(
                failureReason,
                "peer DDR stage has incomplete coupled component identity");
          selectedDDRStages.push_back(CandidateSelectedDDRStage{
              allocation.getResult(), *producerNode,
              coupledComponent
                  ? producerMapping
                        ->coupledComponentIndices[strategy.producerResult]
                  : strategy.producerResult,
              coupledComponent
                  ? StructuredResultIdentityKind::CoupledReductionComponent
                  : StructuredResultIdentityKind::OperationResult});
          auto destination = builder.create<mlir::bufferization::ToTensorOp>(
              assemblyLoc, allocation.getResult(),
              /*restrict=*/true, /*writable=*/true);
          preserved.insert(allocation.getOperation());
          preserved.insert(destination.getOperation());
          assembled = destination.getResult();
        } else {
          auto fullEmpty = builder.create<mlir::tensor::EmptyOp>(
              mapped.consumer->getLoc(), producerType.getShape(),
              producerType.getElementType(), mlir::ValueRange{},
              producerType.getEncoding());
          assembled = fullEmpty.getResult();
        }
        llvm::SmallVector<const SpatialEdgeFragment *, 4> ordered;
        llvm::SmallVector<size_t, 4> streamedReceiveEndpoints;
        std::optional<int64_t> assemblyCardDDRResource;
        for (const SpatialEdgeFragment &fragment : strategy.fragments)
          ordered.push_back(&fragment);
        for (const SpatialEdgeFragment &fragment : strategy.fragments)
          if (fragment.kind == SpatialEdgeFragmentKind::CardDDR &&
              fragment.cardDDRResource &&
              (!assemblyCardDDRResource ||
               *fragment.cardDDRResource < *assemblyCardDDRResource))
            assemblyCardDDRResource = *fragment.cardDDRResource;
        llvm::sort(ordered, [](const auto *lhs, const auto *rhs) {
          if (lhs->offsets != rhs->offsets)
            return lhs->offsets < rhs->offsets;
          if (lhs->sizes != rhs->sizes)
            return lhs->sizes < rhs->sizes;
          return std::tuple(static_cast<uint8_t>(lhs->kind),
                            lhs->sourceTile.getValue()) <
                 std::tuple(static_cast<uint8_t>(rhs->kind),
                            rhs->sourceTile.getValue());
        });
        for (const SpatialEdgeFragment *fragment : ordered) {
          mlir::Value value;
          if (fragment->kind == SpatialEdgeFragmentKind::Resident) {
            if (assembleInDDR) {
              // Keep the resident contribution as a current-SSA window.  The
              // producer is independently materialized later in topological
              // order, and its external fanout rewrite then retargets this
              // slice to the sealed local result without eager recomputation.
              value = createExactSlice(
                  builder, mapped.producer->getLoc(),
                  mapped.producer->getResult(strategy.producerResult),
                  fragment->offsets, fragment->sizes);
            } else {
              mlir::FailureOr<mlir::Value> local = getOrMaterializeSource(
                  scope, mapped.producer, strategy.producerResult,
                  fragment->offsets, fragment->sizes, materialized, preserved,
                  mappedTemporalTiles, failureReason, &mappedOperationNodes);
              if (mlir::failed(local))
                return mlir::failure();
              value = *local;
            }
          } else if (fragment->kind == SpatialEdgeFragmentKind::CardDDR) {
            auto boundary =
                fragment->cardDDRResource
                    ? mapped.cardDDRBoundaries.find(*fragment->cardDDRResource)
                    : mapped.cardDDRBoundaries.end();
            if (boundary == mapped.cardDDRBoundaries.end())
              return reportSelectedEdgeFailure(
                  failureReason,
                  "card DDR fragment has no current boundary argument");
            value = createExactSlice(builder, mapped.producer->getLoc(),
                                     boundary->second, fragment->offsets,
                                     fragment->sizes);
            preserved.insert(value.getDefiningOp());
          } else {
            std::optional<uint32_t> consumerNode =
                findStructuredNodeId(mapped.consumer, mappedOperationNodes);
            if (!consumerNode)
              return reportSelectedEdgeFailure(
                  failureReason,
                  "selected peer receive has no structured consumer identity");
            if (assembleInDDR) {
              mlir::FailureOr<llvm::SmallVector<int64_t, 4>> streamTiles =
                  getFragmentStreamTileSizes(
                      mapped.producer, strategy.producerResult,
                      mappedTemporalTiles, *fragment, failureReason);
              if (mlir::failed(streamTiles))
                return mlir::failure();
              endpoints.push_back(CandidatePeerEndpoint{
                  /*value=*/{}, independentAssemblyAllocation.getResult(),
                  CandidatePeerEndpointKind::Receive, fragment->sourceTile,
                  fragment->bytes, fragment->communicationId,
                  fragment->payloadSlice, mapped.consumerScheduleOrdinal,
                  strategy.consumerOperand, *consumerNode, fragment});
              CandidatePeerEndpoint &endpoint = endpoints.back();
              endpoint.streamOffsets = fragment->offsets;
              endpoint.streamSizes = fragment->sizes;
              endpoint.streamTileSizes = std::move(*streamTiles);
              streamedReceiveEndpoints.push_back(endpoints.size() - 1);
              continue;
            }
            mlir::Location receiveLoc = mapped.consumer->getLoc();
            auto receiveType = mlir::MemRefType::get(
                fragment->sizes, producerType.getElementType(),
                mlir::MemRefLayoutAttrInterface{},
                MemoryAttr::get(mapped.consumer->getContext(), MemorySpace::SPM,
                                MemLayout::Tensor));
            auto allocation =
                builder.create<mlir::memref::AllocOp>(receiveLoc, receiveType);
            auto received = builder.create<mlir::bufferization::ToTensorOp>(
                receiveLoc, allocation.getResult(), /*restrict=*/true,
                /*writable=*/true);
            preserved.insert(allocation.getOperation());
            preserved.insert(received.getOperation());
            value = received.getResult();
            endpoints.push_back(CandidatePeerEndpoint{
                value, allocation.getResult(),
                CandidatePeerEndpointKind::Receive, fragment->sourceTile,
                fragment->bytes, fragment->communicationId,
                fragment->payloadSlice, mapped.consumerScheduleOrdinal,
                strategy.consumerOperand, *consumerNode, fragment});
          }
          mlir::Value inserted =
              insertExactSlice(builder, assemblyLoc, value, assembled,
                               fragment->offsets, fragment->sizes);
          if (!consumerNode)
            return reportSelectedEdgeFailure(
                failureReason,
                "fragment assembly has no structured consumer identity");
          if (llvm::none_of(mappedOperationNodes,
                            [&](const StructuredOperationNodeMapping &mapping) {
                              return mapping.operation ==
                                     inserted.getDefiningOp();
                            }))
            mappedOperationNodes.push_back(
                {inserted.getDefiningOp(), *consumerNode, {}});
          std::optional<int64_t> movementResource =
              fragment->kind == SpatialEdgeFragmentKind::CardDDR
                  ? fragment->cardDDRResource
              : fragment->kind == SpatialEdgeFragmentKind::Resident
                  ? assemblyCardDDRResource
                  : std::nullopt;
          if (movementResource) {
            inserted.getDefiningOp()->setAttr(
                kWaferCardDDRMovementAttrName,
                CardDDRResourceAttr::get(builder.getContext(),
                                         *movementResource));
          }
          if (assembleInDDR) {
            // Every exact fragment writes a disjoint slice of the same
            // compiler-owned writable DDR allocation. Do not thread those
            // stores through one functional tensor.insert_slice chain: that
            // chain makes all receive buffers appear simultaneously live
            // until the final value, defeating the baseline's fragment-wise
            // staging contract. Preserve each store root independently; the
            // sealed read-only view below is the sole downstream carrier.
            preserved.insert(inserted.getDefiningOp());
          } else {
            assembled = inserted;
          }
        }
        if (assembleInDDR) {
          // Reopen the complete exact allocation as a read-only tensor
          // boundary so consumer waves load only their demanded windows and
          // cannot clone fragment stores into those waves.
          auto sealed = builder.create<mlir::bufferization::ToTensorOp>(
              assemblyLoc, independentAssemblyAllocation.getResult(),
              /*restrict=*/false, /*writable=*/false);
          preserved.insert(sealed.getOperation());
          assembled = sealed.getResult();
          for (size_t endpoint : streamedReceiveEndpoints)
            endpoints[endpoint].value = assembled;
        } else {
          // PeerFragments is an explicit physical edge action.  Seal its exact
          // SPM assembly so downstream tiling cannot recompute the producer.
          auto materializedDest = builder.create<mlir::tensor::EmptyOp>(
              mapped.consumer->getLoc(), producerType.getShape(),
              producerType.getElementType(), mlir::ValueRange{},
              producerType.getEncoding());
          assembled =
              builder
                  .create<mlir::bufferization::MaterializeInDestinationOp>(
                      mapped.consumer->getLoc(), producerType, assembled,
                      materializedDest.getResult(), /*restrict=*/false,
                      /*writable=*/false)
                  .getResult();
          preserved.insert(assembled.getDefiningOp());
        }
        if (mapped.requiresConsumerInputReconstruction)
          producerValues.push_back(ProducerValue{
              mapped.producer, strategy.producerResult, mapped.consumer,
              strategy.consumerOperand, strategy.destinationTile, assembled});
        else
          mapped.consumer->setOperand(strategy.consumerOperand, assembled);
      }
    }
    llvm::ArrayRef<MappedStrategy *> group(peerOrder.data() + groupBegin,
                                           groupEnd - groupBegin);
    auto supportAnchor = llvm::find_if(group, [&](MappedStrategy *mapped) {
      return mapped->requiresConsumerInputReconstruction &&
             mapped->strategy.destinationTile == currentTile;
    });
    if (supportAnchor != group.end() &&
        mlir::failed(reconstructConsumerInput(
            (*supportAnchor)->consumer,
            (*supportAnchor)->strategy.consumerOperand, currentTile,
            mappedConsumerInputDemands, producerValues, failureReason)))
      return mlir::failure();
    for (const CandidatePeerEndpoint &endpoint : endpoints) {
      if (endpoint.kind != CandidatePeerEndpointKind::Receive ||
          !endpoint.value.use_empty())
        continue;
      auto groupNode = llvm::find_if(
          operationNodes, [&](const StructuredOperationNodeMapping &entry) {
            return entry.operation == peerOrder[groupBegin]->sourceConsumer;
          });
      return reportSelectedEdgeFailure(
          failureReason,
          (llvm::Twine("selected receive became unused while materializing "
                       "peer consumer group ") +
           (groupNode != operationNodes.end()
                ? llvm::Twine(groupNode->structuredNodeId)
                : llvm::Twine("unknown")) +
           " (communication_id=" + llvm::Twine(endpoint.communicationId) +
           ", payload_slice=" + llvm::Twine(endpoint.payloadSlice) + ")")
              .str());
    }
    groupBegin = groupEnd;
  }
  return mlir::success();
}

mlir::LogicalResult
requireLiveSelectedReceives(SelectedEdgeLoweringState &state,
                            llvm::StringRef stage) {
  for (const CandidatePeerEndpoint &endpoint : state.endpoints) {
    if (endpoint.kind != CandidatePeerEndpointKind::Receive ||
        !endpoint.value.use_empty())
      continue;
    std::string detail;
    llvm::raw_string_ostream diagnostic(detail);
    diagnostic << "selected receive became unused during " << stage
               << " (communication_id=" << endpoint.communicationId
               << ", payload_slice=" << endpoint.payloadSlice << ')';
    return reportSelectedEdgeFailure(state.failureReason, diagnostic.str());
  }
  return mlir::success();
}

mlir::LogicalResult
verifySelectedReceiveOwners(SelectedEdgeLoweringState &state) {
  auto &endpoints = state.endpoints;
  auto &mappedStrategies = state.mapping.strategies;
  std::string *failureReason = state.failureReason;
  for (const CandidatePeerEndpoint &endpoint : endpoints) {
    if (endpoint.kind != CandidatePeerEndpointKind::Receive ||
        !endpoint.selectedFragment)
      continue;
    auto owner = llvm::find_if(mappedStrategies, [&](MappedStrategy &mapped) {
      return llvm::any_of(mapped.strategy.fragments,
                          [&](const SpatialEdgeFragment &fragment) {
                            return &fragment == endpoint.selectedFragment;
                          });
    });
    if (owner == mappedStrategies.end())
      return reportSelectedEdgeFailure(
          failureReason, "selected receive has no mapped edge owner");
    llvm::DenseSet<mlir::Value> visited;
    if (!isInBackwardClosure(
            owner->consumer->getOperand(owner->strategy.consumerOperand),
            endpoint.value.getDefiningOp(), visited) &&
        !selectedValueReachesDDRStage(state, endpoint.value.getDefiningOp()))
      return reportSelectedEdgeFailure(
          failureReason,
          (llvm::Twine("selected receive is outside its consumer operand "
                       "closure before output traversal (communication_id=") +
           llvm::Twine(endpoint.communicationId) + ")")
              .str());
  }
  return mlir::success();
}

mlir::LogicalResult
materializeSelectedPeerSends(SelectedEdgeLoweringState &state) {
  TensorProgramScope scope = state.scope;
  TileId currentTile = state.currentTile;
  auto &mappedTemporalTiles = state.mapping.operationTemporalTiles;
  auto &mappedOperationNodes = state.mapping.operationNodes;
  auto &mappedStrategies = state.mapping.strategies;
  const bool independentDDRStages = state.mapping.independentDDRStages;
  auto &endpoints = state.endpoints;
  auto &materialized = state.materializedSources;
  auto &preserved = state.preservedOperations;
  std::string *failureReason = state.failureReason;
  for (MappedStrategy &mapped : mappedStrategies) {
    SpatialEdgeStrategy &strategy = mapped.strategy;
    if (strategy.action != SpatialEdgeAction::PeerFragments)
      continue;
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      if (fragment.kind != SpatialEdgeFragmentKind::Peer ||
          fragment.sourceTile != currentTile)
        continue;
      mlir::FailureOr<mlir::Value> value = getOrMaterializeSource(
          scope, mapped.producer, strategy.producerResult, fragment.offsets,
          fragment.sizes, materialized, preserved, mappedTemporalTiles,
          failureReason, &mappedOperationNodes);
      if (mlir::failed(value))
        return mlir::failure();
      auto compactType =
          mlir::dyn_cast<mlir::RankedTensorType>(value->getType());
      if (!compactType || !compactType.hasStaticShape() ||
          !llvm::equal(compactType.getShape(), fragment.sizes))
        return reportSelectedEdgeFailure(
            failureReason,
            "selected peer send value differs from its exact fragment");
      mlir::Operation *definition = value->getDefiningOp();
      if (!definition || definition->getBlock() != &scope.getBody())
        return reportSelectedEdgeFailure(
            failureReason, "selected peer send value is not scope-local");
      mlir::OpBuilder builder(definition);
      builder.setInsertionPointAfter(definition);
      std::optional<uint32_t> producerNode =
          findStructuredNodeId(mapped.producer, mappedOperationNodes);
      if (!producerNode)
        return reportSelectedEdgeFailure(
            failureReason, "selected peer send has no structured producer "
                           "identity");
      if (independentDDRStages) {
        mlir::Value stageBuffer = getDDRStageBuffer(*value);
        mlir::FailureOr<llvm::SmallVector<int64_t, 4>> streamTiles =
            getFragmentStreamTileSizes(mapped.producer, strategy.producerResult,
                                       mappedTemporalTiles, fragment,
                                       failureReason);
        if (!stageBuffer || mlir::failed(streamTiles))
          return reportSelectedEdgeFailure(
              failureReason,
              "selected peer send has no exact DDR stage or temporal tile");
        endpoints.push_back(CandidatePeerEndpoint{
            *value, stageBuffer, CandidatePeerEndpointKind::Send,
            strategy.destinationTile, fragment.bytes, fragment.communicationId,
            fragment.payloadSlice, mapped.consumerScheduleOrdinal,
            strategy.consumerOperand, *producerNode, &fragment});
        CandidatePeerEndpoint &endpoint = endpoints.back();
        endpoint.streamOffsets = fragment.offsets;
        endpoint.streamSizes = fragment.sizes;
        endpoint.streamTileSizes = std::move(*streamTiles);
        continue;
      }
      llvm::SmallVector<int64_t, 4> zeroOffsets(fragment.sizes.size(), 0);
      mlir::Value endpointValue =
          createExactSlice(builder, mapped.producer->getLoc(), *value,
                           zeroOffsets, fragment.sizes);
      preserved.insert(endpointValue.getDefiningOp());
      endpoints.push_back(CandidatePeerEndpoint{
          endpointValue, /*carrierBuffer=*/{}, CandidatePeerEndpointKind::Send,
          strategy.destinationTile, fragment.bytes, fragment.communicationId,
          fragment.payloadSlice, mapped.consumerScheduleOrdinal,
          strategy.consumerOperand, *producerNode, &fragment});
    }
  }
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
