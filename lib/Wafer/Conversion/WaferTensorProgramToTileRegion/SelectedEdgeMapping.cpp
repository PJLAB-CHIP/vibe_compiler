//===- SelectedEdgeMapping.cpp - Map selected edges to candidate ----===//

#include "SelectedEdgeMapping.h"

#include "EdgeDomain.h"
#include "EdgeFragmentPlanning.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <map>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {
namespace {

mlir::LogicalResult failResult(std::string *failureReason,
                               llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

bool sameEdge(const MappedStrategy &lhs, const MappedStrategy &rhs) {
  return lhs.producer == rhs.producer &&
         lhs.strategy.producerResult == rhs.strategy.producerResult &&
         lhs.consumer == rhs.consumer &&
         lhs.strategy.consumerOperand == rhs.strategy.consumerOperand;
}

} // namespace

mlir::FailureOr<SelectedEdgeProgramMapping> mapSelectedEdgesToCandidate(
    mlir::Block &sourceBody, mlir::IRMapping &cloneMapping,
    TensorProgramScope scope, TileId currentTile,
    SpatialDataflowMaterializationMode materializationMode,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<SpatialEdgeMaterializationFacts> edgeFacts,
    llvm::ArrayRef<analysis::DependencyDemand> operandDemands,
    std::string *failureReason) {
  SelectedEdgeProgramMapping result;
  llvm::DenseMap<mlir::Operation *, uint64_t> sourcePositions;
  uint64_t sourcePosition = 0;
  for (mlir::Operation &operation : sourceBody.without_terminator())
    sourcePositions.try_emplace(&operation, sourcePosition++);
  llvm::DenseSet<uint32_t> distinctNodeIds;
  bool hasSharedNodeIdentity = false;
  for (const StructuredOperationNodeMapping &node : operationNodes)
    hasSharedNodeIdentity |=
        !distinctNodeIds.insert(node.structuredNodeId).second;
  llvm::DenseSet<mlir::Operation *> seenTemporalOperations;
  for (const StructuredOpTemporalTile &tile : operationTemporalTiles) {
    if (!tile.operation ||
        !seenTemporalOperations.insert(tile.operation).second)
      return failResult(
          failureReason,
          "structured temporal mapping contains a null or duplicate operation");
    mlir::Operation *mapped = cloneMapping.lookupOrNull(tile.operation);
    if (!mapped)
      return failResult(
          failureReason,
          "structured temporal mapping operation is outside source module");
    result.operationTemporalTiles.push_back(StructuredOpTemporalTile{
        mapped, tile.iteratorTileSizes, tile.waveLoopOrder});
  }

  llvm::DenseSet<mlir::Operation *> seenNodeOperations;
  result.operationNodes.reserve(operationNodes.size());
  for (const StructuredOperationNodeMapping &node : operationNodes) {
    if (!node.operation || !seenNodeOperations.insert(node.operation).second)
      return failResult(
          failureReason,
          "structured operation-node mapping contains a null or duplicate "
          "operation");
    mlir::Operation *mapped = cloneMapping.lookupOrNull(node.operation);
    if (!mapped)
      return failResult(failureReason,
                        "structured operation-node mapping is outside source "
                        "module");
    result.operationNodes.push_back(
        {mapped, node.structuredNodeId, node.coupledComponentIndices});
  }

  result.consumerInputDemands.reserve(operandDemands.size());
  for (const analysis::DependencyDemand &demand : operandDemands) {
    if (!demand.consumerOperation)
      return failResult(
          failureReason,
          "operand materialization requires a typed demand recipe");
    analysis::DependencyDemand mapped = demand;
    mapped.consumerOperation =
        cloneMapping.lookupOrNull(demand.consumerOperation);
    if (!mapped.consumerOperation)
      return failResult(failureReason,
                        "operand demand consumer is outside source module");
    for (analysis::DestinationDemand &recipe : mapped.perDestination) {
      for (analysis::TensorTransform &step : recipe.reconstruction.steps) {
        step.operation = cloneMapping.lookupOrNull(step.operation);
        if (!step.operation)
          return failResult(failureReason,
                            "tensor-operand step is outside source module");
      }
      for (analysis::SourceDemand &source : recipe.sources) {
        if (auto *structured =
                std::get_if<analysis::StructuredResultSource>(&source.source)) {
          structured->operation =
              cloneMapping.lookupOrNull(structured->operation);
          if (!structured->operation)
            return failResult(
                failureReason,
                "tensor-operand boundary is outside source module");
        } else if (auto *constant =
                       std::get_if<analysis::ConstantSource>(&source.source)) {
          constant->operation = cloneMapping.lookupOrNull(constant->operation);
          if (!constant->operation)
            return failResult(
                failureReason,
                "tensor constant boundary is outside source module");
        }
      }
    }
    result.consumerInputDemands.push_back(std::move(mapped));
  }

  result.independentDDRStages =
      materializationMode ==
      SpatialDataflowMaterializationMode::IndependentDDRStages;
  llvm::SmallVector<SpatialEdgeStrategy, 16> normalizedEdgeStrategies(
      edgeStrategies.begin(), edgeStrategies.end());

  llvm::SmallVector<SpatialEdgeMaterializationFacts, 16> effectiveEdgeFacts;
  if (edgeFacts.empty()) {
    mlir::FailureOr<llvm::SmallVector<SpatialEdgeMaterializationFacts, 16>>
        derived = deriveSpatialEdgeMaterializationFacts(
            sourceBody, normalizedEdgeStrategies, operandDemands,
            failureReason);
    if (mlir::failed(derived))
      return mlir::failure();
    effectiveEdgeFacts = std::move(*derived);
  } else {
    if (edgeFacts.size() != normalizedEdgeStrategies.size())
      return failResult(
          failureReason,
          "edge materialization facts do not cover normalized strategies");
    effectiveEdgeFacts.assign(edgeFacts.begin(), edgeFacts.end());
  }

  result.strategies.reserve(normalizedEdgeStrategies.size());
  std::map<int64_t, mlir::BlockArgument> cardDDRArguments;
  std::map<int64_t, mlir::RankedTensorType> cardDDRTypes;
  for (const SpatialEdgeStrategy &strategy : normalizedEdgeStrategies) {
    mlir::Operation *producer = cloneMapping.lookupOrNull(strategy.producer);
    auto type =
        producer && strategy.producerResult < producer->getNumResults()
            ? mlir::dyn_cast<mlir::RankedTensorType>(
                  producer->getResult(strategy.producerResult).getType())
            : mlir::RankedTensorType{};
    auto recordType = [&](int64_t resourceId) {
      if (resourceId < 0 || !type || !type.hasStaticShape())
        return false;
      auto [resource, inserted] = cardDDRTypes.try_emplace(resourceId, type);
      return inserted || resource->second == type;
    };
    if (strategy.cardDDRResource && !recordType(*strategy.cardDDRResource))
      return failResult(failureReason,
                        "card DDR resource has inconsistent producer types");
    for (const SpatialEdgeFragment &fragment : strategy.fragments)
      if (fragment.cardDDRResource && !recordType(*fragment.cardDDRResource))
        return failResult(failureReason,
                          "card DDR fragment has inconsistent producer types");
  }
  for (const auto &[resourceId, type] : cardDDRTypes) {
    const unsigned argumentIndex = scope.getFunction().getNumArguments();
    std::string symbol =
        (llvm::Twine("card_ddr_") + llvm::Twine(resourceId)).str();
    auto binding = CardDDRBindingAttr::get(
        scope.getContext(),
        mlir::FlatSymbolRefAttr::get(scope.getContext(), symbol), resourceId,
        CardDDRAccess::None);
    scope.getFunction().insertArgument(
        argumentIndex, type,
        mlir::DictionaryAttr::get(
            scope.getContext(),
            {mlir::NamedAttribute(
                mlir::StringAttr::get(scope.getContext(),
                                      kWaferCardDDRBindingAttrName),
                binding)}),
        scope.getLoc());
    cardDDRArguments.emplace(resourceId,
                             scope.getFunction().getArgument(argumentIndex));
  }
  for (auto [strategyIndex, strategy] :
       llvm::enumerate(normalizedEdgeStrategies)) {
    if (!isSpatialEdgeStrategyIncidentOnTile(strategy, currentTile))
      continue;
    MappedStrategy mapped;
    mapped.strategy = strategy;
    mapped.sourceProducer = strategy.producer;
    mapped.sourceConsumer = strategy.consumer;
    mapped.producer = cloneMapping.lookupOrNull(strategy.producer);
    mapped.consumer = cloneMapping.lookupOrNull(strategy.consumer);
    mapped.consumerScheduleOrdinal =
        effectiveEdgeFacts[strategyIndex].consumerScheduleOrdinal;
    if (hasSharedNodeIdentity &&
        strategy.action == SpatialEdgeAction::PeerFragments) {
      auto producerPosition = sourcePositions.find(strategy.producer);
      if (producerPosition == sourcePositions.end())
        return failResult(failureReason,
                          "shared-root peer producer is outside source order");
      // A selected semantic root may contain several structured action ops.
      // Order its cross-root messages by producer readiness; consumer-first
      // order can otherwise place a downstream send before the receive that
      // makes its producer executable and create an artificial event cycle.
      mapped.consumerScheduleOrdinal = producerPosition->second;
    }
    mapped.strategy.producer = mapped.producer;
    mapped.strategy.consumer = mapped.consumer;
    if (mlir::failed(validateEdge(mapped.producer, strategy.producerResult,
                                  mapped.consumer, strategy.consumerOperand,
                                  scope.getBody(), failureReason)))
      return mlir::failure();
    mapped.requiresConsumerInputReconstruction =
        effectiveEdgeFacts[strategyIndex].requiresConsumerInputReconstruction;
    if (mapped.requiresConsumerInputReconstruction &&
        strategy.action != SpatialEdgeAction::RegionCut &&
        strategy.action != SpatialEdgeAction::PeerFragments &&
        strategy.action != SpatialEdgeAction::CardDDRTransfer)
      return failResult(failureReason, "pure tensor input-chain dependencies "
                                       "require RegionCut, card DDR, or exact "
                                       "peer fragments");
    auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
        mapped.producer->getResult(strategy.producerResult).getType());
    auto consumerType = mlir::dyn_cast<mlir::RankedTensorType>(
        mapped.consumer->getResult(0).getType());
    auto selectedConsumerResultType =
        mapped.consumer->getNumResults() == 1
            ? mlir::dyn_cast<mlir::RankedTensorType>(
                  mapped.consumer->getResult(0).getType())
            : mlir::RankedTensorType{};
    const bool validZeroRankDomain =
        selectedConsumerResultType &&
        selectedConsumerResultType.getRank() == 0 &&
        strategy.consumerOffsets.empty() && strategy.consumerSizes.empty();
    if (!validZeroRankDomain &&
        (strategy.consumerOffsets.empty() || strategy.consumerSizes.empty()))
      return failResult(
          failureReason,
          "selected edge has no proof-derived consumer result domain");
    const SpatialEdgeStrategy &validated = mapped.strategy;
    if (mlir::failed(validateStaticDomain(
            producerType, validated.producerOffsets, validated.producerSizes,
            failureReason, "edge producer demand")) ||
        mlir::failed(validateStaticDomain(
            consumerType, validated.consumerOffsets, validated.consumerSizes,
            failureReason, "edge consumer demand")))
      return mlir::failure();
    if (llvm::any_of(result.strategies, [&](const MappedStrategy &other) {
          return sameEdge(mapped, other) && mapped.strategy.destinationTile ==
                                                other.strategy.destinationTile;
        }))
      return failResult(failureReason,
                        "edge mapping duplicates one destination strategy");

    if (strategy.action != SpatialEdgeAction::PeerFragments &&
        strategy.action != SpatialEdgeAction::CardDDRTransfer &&
        (!strategy.fragments.empty() || strategy.fragmentsDefineProducerDemand))
      return failResult(failureReason,
                        "non-fragment edge strategy cannot carry fragments");
    if (strategy.fragmentsDefineProducerDemand &&
        !mapped.requiresConsumerInputReconstruction)
      return failResult(failureReason,
                        "fragment-union producer demand requires a typed "
                        "consumer-input reconstruction recipe");
    if (strategy.action != SpatialEdgeAction::PeerFragments &&
        strategy.action != SpatialEdgeAction::CardDDRTransfer &&
        strategy.sourceTile != strategy.destinationTile)
      return failResult(failureReason,
                        "local edge strategy requires one Tile placement");
    if (strategy.action == SpatialEdgeAction::CardDDRTransfer &&
        (!strategy.cardDDRResource || strategy.fragments.empty()))
      return failResult(
          failureReason,
          "card DDR transfer requires one resource and exact pieces");
    if (strategy.action != SpatialEdgeAction::CardDDRTransfer &&
        strategy.cardDDRResource)
      return failResult(failureReason,
                        "non-card-DDR edge carries a card resource identity");
    const unsigned elementBits =
        producerType.getElementType().getIntOrFloatBitWidth();
    if (elementBits == 0 || elementBits % 8 != 0)
      return failResult(failureReason,
                        "edge fragment element type is not byte-sized");
    auto bindCardDDR = [&](int64_t resourceId, bool reads, bool writes) {
      auto boundary = cardDDRArguments.find(resourceId);
      if (boundary == cardDDRArguments.end())
        return false;
      auto existing = scope.getFunction().getArgAttrOfType<CardDDRBindingAttr>(
          boundary->second.getArgNumber(), kWaferCardDDRBindingAttrName);
      if (!existing || existing.getResourceId() != resourceId)
        return false;
      CardDDRAccess access = reads && writes ? CardDDRAccess::ReadWrite
                             : reads         ? CardDDRAccess::Read
                                             : CardDDRAccess::Write;
      CardDDRAccess merged =
          existing.getAccess() == CardDDRAccess::None ? access
          : existing.getAccess() == access            ? access
                                           : CardDDRAccess::ReadWrite;
      scope.getFunction().setArgAttr(
          boundary->second.getArgNumber(), kWaferCardDDRBindingAttrName,
          CardDDRBindingAttr::get(scope.getContext(), existing.getResource(),
                                  existing.getResourceId(), merged));
      mapped.cardDDRBoundaries.try_emplace(resourceId, boundary->second);
      return true;
    };
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      std::optional<int64_t> resourceId = fragment.cardDDRResource;
      if (!resourceId && strategy.action == SpatialEdgeAction::CardDDRTransfer)
        resourceId = strategy.cardDDRResource;
      if (strategy.action == SpatialEdgeAction::CardDDRTransfer &&
          resourceId != strategy.cardDDRResource)
        return failResult(
            failureReason,
            "card DDR transfer piece disagrees with its uniform resource");
      if (fragment.kind != SpatialEdgeFragmentKind::CardDDR) {
        if (fragment.cardDDRResource)
          return failResult(
              failureReason,
              "non-card-DDR fragment carries a card resource identity");
        continue;
      }
      uint64_t elements = 1;
      for (int64_t size : fragment.sizes) {
        if (size <= 0 || elements > std::numeric_limits<uint64_t>::max() /
                                        static_cast<uint64_t>(size))
          return failResult(
              failureReason,
              "card DDR transfer piece has invalid static volume");
        elements *= static_cast<uint64_t>(size);
      }
      const uint64_t elementBytes = elementBits / 8;
      if (!resourceId ||
          elements > std::numeric_limits<uint64_t>::max() / elementBytes ||
          fragment.bytes != elements * elementBytes ||
          fragment.communicationId != 0 || fragment.payloadSlice != 0 ||
          !bindCardDDR(*resourceId, currentTile == strategy.destinationTile,
                       currentTile == fragment.sourceTile))
        return failResult(
            failureReason,
            "card DDR transfer piece has invalid resource, bytes, or access");
    }
    if (strategy.action == SpatialEdgeAction::CardDDRTransfer) {
      auto boundary = mapped.cardDDRBoundaries.find(*strategy.cardDDRResource);
      if (boundary == mapped.cardDDRBoundaries.end() ||
          llvm::any_of(strategy.fragments, [](const auto &fragment) {
            return fragment.kind != SpatialEdgeFragmentKind::CardDDR;
          }))
        return failResult(
            failureReason,
            "card DDR transfer does not have one uniform fragment carrier");
      mapped.cardDDRBoundary = boundary->second;
    }
    result.strategies.push_back(std::move(mapped));
  }

  if (result.strategies.empty())
    return failResult(failureReason,
                      "edge-action lowering has no action on this Tile");

  if (result.independentDDRStages &&
      !llvm::all_of(result.strategies, [](const MappedStrategy &mapped) {
        return mapped.strategy.action == SpatialEdgeAction::RegionCut ||
               mapped.strategy.action == SpatialEdgeAction::CardDDRTransfer ||
               mapped.strategy.action == SpatialEdgeAction::PeerFragments;
      }))
    return failResult(
        failureReason,
        "independent DDR stages require RegionCut, card DDR, or exact "
        "cross-Tile fragment actions");

  return result;
}

} // namespace wafer::tensor_program_to_tile_region
