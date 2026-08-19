//===- SelectedEdgeMapping.cpp - Map selected edges to candidate ----===//

#include "SelectedEdgeMapping.h"

#include "EdgeDomain.h"
#include "EdgeFragmentPlanning.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "llvm/ADT/STLExtras.h"

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
    llvm::ArrayRef<analysis::ConsumerInputDemand> operandDemands,
    std::string *failureReason) {
  SelectedEdgeProgramMapping result;
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
  llvm::DenseSet<uint32_t> seenNodeIds;
  result.operationNodes.reserve(operationNodes.size());
  for (const StructuredOperationNodeMapping &node : operationNodes) {
    if (!node.operation || !seenNodeOperations.insert(node.operation).second ||
        !seenNodeIds.insert(node.structuredNodeId).second)
      return failResult(
          failureReason,
          "structured operation-node mapping contains a null or duplicate "
          "entry");
    mlir::Operation *mapped = cloneMapping.lookupOrNull(node.operation);
    if (!mapped)
      return failResult(failureReason,
                        "structured operation-node mapping is outside source "
                        "module");
    result.operationNodes.push_back({mapped, node.structuredNodeId});
  }

  result.consumerInputDemands.reserve(operandDemands.size());
  for (const analysis::ConsumerInputDemand &demand : operandDemands) {
    if (demand.status != analysis::ExactDemandStatus::Satisfied ||
        !demand.consumer)
      return failResult(
          failureReason,
          "operand materialization requires a satisfied typed demand recipe");
    analysis::ConsumerInputDemand mapped = demand;
    mapped.consumer = cloneMapping.lookupOrNull(demand.consumer);
    if (!mapped.consumer)
      return failResult(failureReason,
                        "operand demand consumer is outside source module");
    for (analysis::ConsumerInputReconstruction &recipe :
         mapped.perDestination) {
      for (analysis::TensorTransform &step : recipe.steps) {
        step.operation = cloneMapping.lookupOrNull(step.operation);
        if (!step.operation)
          return failResult(failureReason,
                            "tensor-operand step is outside source module");
      }
      for (analysis::ProducerValueRequirement &boundary : recipe.boundaries) {
        boundary.producer = cloneMapping.lookupOrNull(boundary.producer);
        if (!boundary.producer)
          return failResult(failureReason,
                            "tensor-operand boundary is outside source module");
      }
    }
    result.consumerInputDemands.push_back(std::move(mapped));
  }

  result.independentDDRStages =
      materializationMode ==
      SpatialDataflowMaterializationMode::IndependentDDRStages;
  llvm::SmallVector<SpatialEdgeStrategy, 16> normalizedEdgeStrategies(
      edgeStrategies.begin(), edgeStrategies.end());
  if (result.independentDDRStages &&
      mlir::failed(splitIndependentPeerFragmentsAtTemporalWaves(
          normalizedEdgeStrategies, operationTemporalTiles, failureReason)))
    return mlir::failure();

  result.strategies.reserve(normalizedEdgeStrategies.size());
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
    if (edgeFacts.empty()) {
      auto consumerPosition =
          llvm::find_if(sourceBody, [&](mlir::Operation &operation) {
            return &operation == mapped.sourceConsumer;
          });
      if (consumerPosition == sourceBody.end())
        return failResult(
            failureReason,
            "edge strategy consumer is outside the pristine program body");
      mapped.consumerScheduleOrdinal = static_cast<uint64_t>(
          std::distance(sourceBody.begin(), consumerPosition));
    } else {
      mapped.consumerScheduleOrdinal =
          edgeFacts[strategyIndex].consumerScheduleOrdinal;
    }
    mapped.strategy.producer = mapped.producer;
    mapped.strategy.consumer = mapped.consumer;
    if (mlir::failed(validateEdge(mapped.producer, strategy.producerResult,
                                  mapped.consumer, strategy.consumerOperand,
                                  scope.getBody(), failureReason,
                                  /*dependencyAlreadyValidated=*/
                                  !edgeFacts.empty())))
      return mlir::failure();
    if (edgeFacts.empty()) {
      mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>>
          producerToConsumerChain = traceProducerToConsumerChain(
              mapped.producer, strategy.producerResult, mapped.consumer,
              strategy.consumerOperand, failureReason);
      if (mlir::failed(producerToConsumerChain))
        return mlir::failure();
      mapped.hasProducerToConsumerChain = !producerToConsumerChain->empty();
    } else {
      mapped.hasProducerToConsumerChain =
          edgeFacts[strategyIndex].hasProducerToConsumerChain;
    }
    if (mapped.hasProducerToConsumerChain &&
        strategy.action != SpatialEdgeAction::RegionCut &&
        strategy.action != SpatialEdgeAction::PeerFragments)
      return failResult(failureReason, "pure tensor input-chain dependencies "
                                       "require RegionCut or exact peer "
                                       "fragments");
    auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
        mapped.producer->getResult(strategy.producerResult).getType());
    auto consumerType = mlir::dyn_cast<mlir::RankedTensorType>(
        mapped.consumer->getResult(0).getType());
    if (strategy.consumerOffsets.empty() || strategy.consumerSizes.empty()) {
      if (!strategy.consumerOffsets.empty() ||
          !strategy.consumerSizes.empty() ||
          mlir::failed(deriveConsumerDomainFromProducerDemand(
              mapped.producer, strategy.producerResult, mapped.consumer,
              strategy.consumerOperand, strategy.producerOffsets,
              strategy.producerSizes, mapped.strategy.consumerOffsets,
              mapped.strategy.consumerSizes, failureReason)))
        return mlir::failure();
    }
    const SpatialEdgeStrategy &validated = mapped.strategy;
    if (mlir::failed(validateStaticDomain(
            producerType, validated.producerOffsets, validated.producerSizes,
            failureReason, "edge producer demand")) ||
        mlir::failed(validateStaticDomain(
            consumerType, validated.consumerOffsets, validated.consumerSizes,
            failureReason, "edge consumer demand")) ||
        (!mapped.hasProducerToConsumerChain &&
         mlir::failed(validateDemandIndexRelation(
             mapped.consumer, validated.consumerOperand,
             validated.consumerOffsets, validated.consumerSizes,
             validated.producerOffsets, validated.producerSizes,
             failureReason))))
      return mlir::failure();
    if (llvm::any_of(result.strategies, [&](const MappedStrategy &other) {
          return sameEdge(mapped, other) && mapped.strategy.destinationTile ==
                                                other.strategy.destinationTile;
        }))
      return failResult(failureReason,
                        "edge mapping duplicates one destination strategy");

    if (strategy.bufferCount == 0 || strategy.bufferCount > 3)
      return failResult(failureReason,
                        "edge strategy buffer count is outside [1, 3]");
    if (strategy.action == SpatialEdgeAction::LocalPhysicalConversion &&
        (!strategy.hasLayoutAssignment ||
         strategy.producerLayout == strategy.consumerLayout))
      return failResult(
          failureReason,
          "local physical conversion requires distinct typed layouts");
    if (strategy.action != SpatialEdgeAction::PeerFragments &&
        (!strategy.fragments.empty() || strategy.fragmentsDefineProducerDemand))
      return failResult(failureReason,
                        "non-peer edge strategy cannot carry fragments");
    if (strategy.fragmentsDefineProducerDemand &&
        !mapped.hasProducerToConsumerChain)
      return failResult(failureReason,
                        "fragment-union producer demand requires a typed "
                        "producer-to-consumer tensor chain");
    if (strategy.action != SpatialEdgeAction::PeerFragments &&
        strategy.sourceTile != strategy.destinationTile)
      return failResult(failureReason,
                        "local edge strategy requires one Tile placement");
    result.strategies.push_back(std::move(mapped));
  }

  if (result.strategies.empty())
    return failResult(failureReason,
                      "edge-action lowering has no action on this Tile");

  if (result.independentDDRStages &&
      !llvm::all_of(result.strategies, [](const MappedStrategy &mapped) {
        return mapped.strategy.bufferCount == 1 &&
               (mapped.strategy.action == SpatialEdgeAction::RegionCut ||
                mapped.strategy.action == SpatialEdgeAction::PeerFragments);
      }))
    return failResult(
        failureReason,
        "independent DDR stages require single-buffer RegionCut or exact "
        "cross-Tile fragment actions");

  return result;
}

} // namespace wafer::tensor_program_to_tile_region
