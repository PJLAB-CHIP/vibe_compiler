//===- MappingValidation.cpp - CardModule mapping validation --------===//

#include "Internal.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::tensor_program_to_card_module {

mlir::FailureOr<TileMaterializationPreparation>
prepareTileMaterialization(const TileMaterializationSourcePreparation &source,
                           const TileMapping &mapping,
                           std::string *failureReason) {
  TileMaterializationPreparation preparation;
  mlir::func::FuncOp sourceProgram = source.sourceProgram;
  preparation.sourceModule = source.sourceModule;
  preparation.sourceProgram = sourceProgram;
  preparation.sourceArgumentCount = source.sourceArgumentCount;
  preparation.availableTiles = source.availableTiles;
  preparation.outputDomains = source.outputDomains;

  auto belongsToSource = [&](mlir::Operation *operation) {
    return operation &&
           operation->getParentOfType<mlir::ModuleOp>() == source.sourceModule;
  };

  preparation.operationNodes.reserve(source.operationNodes.size());
  for (const StructuredOperationNodeMapping &node : source.operationNodes) {
    if (!belongsToSource(node.operation))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured operation-node mapping is outside the tensor "
          "program");
    preparation.operationNodes.push_back(node);
  }

  preparation.consumerInputDemands.reserve(mapping.operandDemands.size());
  for (const analysis::ConsumerInputDemand &demand : mapping.operandDemands) {
    if (demand.status != analysis::ExactDemandStatus::Satisfied ||
        !demand.consumer)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card operand demand must be one satisfied current-IR recipe");
    if (!belongsToSource(demand.consumer))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason, "card operand demand consumer is outside the program");
    for (const analysis::ConsumerInputReconstruction &recipe :
         demand.perDestination) {
      for (const analysis::TensorTransform &step : recipe.steps) {
        if (!belongsToSource(step.operation))
          return failCardModuleValue<TileMaterializationPreparation>(
              failureReason,
              "card tensor-operand step is outside the tensor program");
      }
      for (const analysis::ProducerValueRequirement &boundary :
           recipe.boundaries) {
        if (!belongsToSource(boundary.producer))
          return failCardModuleValue<TileMaterializationPreparation>(
              failureReason,
              "card tensor-operand boundary is outside the tensor program");
      }
    }
    preparation.consumerInputDemands.push_back(demand);
  }

  llvm::DenseSet<mlir::Operation *> temporalSources;
  preparation.operationTemporalTiles.reserve(
      mapping.operationTemporalTiles.size());
  for (const StructuredOpTemporalTile &tile : mapping.operationTemporalTiles) {
    if (!tile.operation || !temporalSources.insert(tile.operation).second)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal mapping is null or duplicated");
    if (!belongsToSource(tile.operation))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal mapping is outside the tensor program");
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(tile.operation);
    if (!tiling ||
        tile.iteratorTileSizes.size() != tiling.getLoopIteratorTypes().size() ||
        llvm::any_of(tile.iteratorTileSizes,
                     [](int64_t size) { return size <= 0; }))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal tile does not match its iteration "
          "domain");
    llvm::SmallVector<int64_t, 4> ranges;
    if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(tile.operation)) {
      ranges = linalg.getStaticLoopRanges();
    } else if (tile.operation->getNumResults() == 1) {
      auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
          tile.operation->getResult(0).getType());
      if (resultType && resultType.hasStaticShape() &&
          resultType.getRank() ==
              static_cast<int64_t>(tile.iteratorTileSizes.size()))
        ranges.assign(resultType.getShape().begin(),
                      resultType.getShape().end());
    }
    if (ranges.size() != tile.iteratorTileSizes.size() ||
        llvm::any_of(llvm::zip_equal(ranges, tile.iteratorTileSizes),
                     [](auto values) {
                       auto [range, tileSize] = values;
                       return mlir::ShapedType::isDynamic(range) ||
                              range <= 0 || tileSize > range;
                     }))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal tile is outside its static iteration "
          "domain");
    llvm::SmallVector<uint32_t, 4> sortedOrder = tile.waveLoopOrder;
    llvm::sort(sortedOrder);
    if (std::adjacent_find(sortedOrder.begin(), sortedOrder.end()) !=
            sortedOrder.end() ||
        llvm::any_of(sortedOrder, [&](uint32_t dimension) {
          return dimension >= tile.iteratorTileSizes.size();
        }))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal wave-loop order is malformed");
    preparation.operationTemporalTiles.push_back(StructuredOpTemporalTile{
        tile.operation, tile.iteratorTileSizes, tile.waveLoopOrder});
  }
  for (mlir::Operation &operation :
       sourceProgram.getBody().front().without_terminator()) {
    if (!mlir::isa<mlir::DestinationStyleOpInterface>(&operation) ||
        !mlir::isa<mlir::TilingInterface>(&operation))
      continue;
    if (!temporalSources.contains(&operation))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal mapping must cover every scheduled "
          "operation exactly once");
  }

  preparation.edgeStrategies.reserve(mapping.edgeStrategies.size());
  for (const SpatialEdgeStrategy &strategy : mapping.edgeStrategies) {
    if (!belongsToSource(strategy.producer) ||
        !belongsToSource(strategy.consumer))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card edge strategy is outside the source tensor program");
    preparation.edgeStrategies.push_back(strategy);
  }

  // Every selected structured producer/data-input dependency is an exact
  // direct SSA edge or a statically provable unary pure producer-to-consumer
  // tensor chain. DPS init operands are initialization state rather than data
  // edges and remain governed by the consumer's typed lowering.
  llvm::DenseSet<std::pair<mlir::Operation *, unsigned>>
      coveredStructuredInputs;
  llvm::DenseMap<mlir::Operation *, uint64_t> schedulingOrdinals;
  uint64_t schedulingOrdinal = 0;
  for (mlir::Operation &operation :
       preparation.sourceProgram.getBody().front().without_terminator())
    schedulingOrdinals.try_emplace(&operation, schedulingOrdinal++);
  preparation.edgeFacts.reserve(mapping.edgeStrategies.size());
  for (auto [edgeIndex, strategy] : llvm::enumerate(mapping.edgeStrategies)) {
    SpatialEdgeStrategy &mapped = preparation.edgeStrategies[edgeIndex];
    if (!strategy.producer || !strategy.consumer ||
        strategy.producerResult >= strategy.producer->getNumResults() ||
        strategy.consumerOperand >= strategy.consumer->getNumOperands())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card edge strategy must name an exact structured SSA dependency");
    mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>>
        producerToConsumerChain =
            traceProducerToConsumerChain(mapped.producer, mapped.producerResult,
                                         mapped.consumer,
                                         mapped.consumerOperand, failureReason);
    auto ordinal = schedulingOrdinals.find(mapped.consumer);
    if (mlir::failed(producerToConsumerChain) ||
        ordinal == schedulingOrdinals.end())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card edge strategy must name an exact structured SSA dependency");
    if (mapped.consumerOffsets.empty() || mapped.consumerSizes.empty()) {
      if (!mapped.consumerOffsets.empty() || !mapped.consumerSizes.empty() ||
          mlir::failed(deriveSpatialEdgeConsumerResultDomain(
              mapped, mapped.consumerOffsets, mapped.consumerSizes,
              failureReason)))
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card edge strategy has no exact consumer result domain");
    }
    preparation.edgeFacts.push_back(
        {/*hasProducerToConsumerChain=*/!producerToConsumerChain->empty(),
         /*consumerScheduleOrdinal=*/ordinal->second});
    auto dps =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(strategy.consumer);
    const bool isDataInput =
        dps &&
        llvm::any_of(dps.getDpsInputOperands(), [&](mlir::OpOperand *operand) {
          return operand->getOperandNumber() == strategy.consumerOperand;
        });
    bool isInitInput = false;
    if (dps)
      for (int64_t index = 0; index < dps.getNumDpsInits(); ++index)
        if (dps.getDpsInitOperand(index)->getOperandNumber() ==
            strategy.consumerOperand)
          isInitInput = true;
    if (!isDataInput && !isInitInput)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card edge strategy must target a structured input or init");
    coveredStructuredInputs.insert(
        {strategy.consumer, strategy.consumerOperand});
  }
  if (!preparation.consumerInputDemands.empty()) {
    llvm::BitVector matchedStrategies(preparation.edgeStrategies.size());
    for (const analysis::ConsumerInputDemand &operandDemand :
         preparation.consumerInputDemands) {
      for (const analysis::ConsumerInputReconstruction &recipe :
           operandDemand.perDestination) {
        for (const analysis::ProducerValueRequirement &boundary :
             recipe.boundaries) {
          if (!boundary.requiredDomain)
            return failCardModuleValue<TileMaterializationPreparation>(
                failureReason,
                "card producer value requirement has no exact required domain");
          std::optional<size_t> matchingStrategy;
          for (auto [strategyIndex, strategy] :
               llvm::enumerate(preparation.edgeStrategies)) {
            if (strategy.producer != boundary.producer ||
                strategy.producerResult != boundary.producerResult ||
                strategy.consumer != operandDemand.consumer ||
                strategy.consumerOperand != operandDemand.consumerOperand ||
                strategy.destinationTile != recipe.destinationTile)
              continue;
            if (matchingStrategy)
              return failCardModuleValue<TileMaterializationPreparation>(
                  failureReason, "card producer value requirement has "
                                 "duplicate physical carriers");
            matchingStrategy = strategyIndex;
          }
          if (boundary.requiredDomain->isIntegerEmpty()) {
            if (matchingStrategy)
              return failCardModuleValue<TileMaterializationPreparation>(
                  failureReason, "exact-empty producer value requirement has a "
                                 "physical carrier");
            continue;
          }
          if (!matchingStrategy)
            return failCardModuleValue<TileMaterializationPreparation>(
                failureReason,
                "nonempty producer value requirement has no physical carrier");
          mlir::FailureOr<mlir::presburger::PresburgerSet> carrierDemand =
              getExactStrategyDemand(
                  preparation.edgeStrategies[*matchingStrategy], failureReason);
          if (mlir::failed(carrierDemand) ||
              !carrierDemand->isEqual(*boundary.requiredDomain))
            return failCardModuleValue<TileMaterializationPreparation>(
                failureReason,
                "physical carrier does not all-and-only cover tensor input");
          matchedStrategies.set(*matchingStrategy);
        }
      }
    }
    if (mapping.materializationMode ==
            SpatialDataflowMaterializationMode::IndependentDDRStages &&
        matchedStrategies.count() != preparation.edgeStrategies.size())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "baseline physical carrier has no grouped boundary demand");
  }
  for (mlir::Operation &operation :
       sourceProgram.getBody().front().without_terminator()) {
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(&operation);
    if (!dps || !mlir::isa<mlir::TilingInterface>(&operation))
      continue;
    llvm::SmallVector<mlir::OpOperand *, 4> dependencyOperands;
    for (mlir::OpOperand *operand : dps.getDpsInputOperands())
      dependencyOperands.push_back(operand);
    for (mlir::OpOperand *operand : dependencyOperands) {
      auto producerResult = mlir::dyn_cast<mlir::OpResult>(operand->get());
      mlir::Operation *producer =
          producerResult ? producerResult.getOwner() : nullptr;
      if (!producer || producer->getBlock() != operation.getBlock() ||
          !mlir::isa<mlir::TilingInterface>(producer) ||
          !mlir::isa<mlir::DestinationStyleOpInterface>(producer))
        continue;
      if (!coveredStructuredInputs.contains(
              {&operation, operand->getOperandNumber()}))
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card spatial mapping is missing a direct structured dependency "
            "action");
    }
  }

  llvm::DenseSet<int64_t> availableTileValues;
  for (TileId tileId : preparation.availableTiles)
    availableTileValues.insert(tileId.getValue());
  for (const SpatialEdgeStrategy &strategy : mapping.edgeStrategies) {
    if (!availableTileValues.contains(strategy.destinationTile.getValue()) ||
        (strategy.action != SpatialEdgeAction::PeerFragments &&
         !availableTileValues.contains(strategy.sourceTile.getValue())))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason, "card edge strategy names an unavailable Tile");
    for (const SpatialEdgeFragment &fragment : strategy.fragments)
      if (!availableTileValues.contains(fragment.sourceTile.getValue()))
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason, "card edge fragment names an unavailable Tile");
  }

  if (mapping.outputs.size() != preparation.outputDomains.size())
    return failCardModuleValue<TileMaterializationPreparation>(
        failureReason,
        "card spatial mapping must cover every function result exactly once");
  preparation.outputMappings.assign(preparation.outputDomains.size(), nullptr);
  for (const OutputTileMapping &output : mapping.outputs) {
    if (output.outputIndex >= preparation.outputDomains.size())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping output index is outside function results");
    if (preparation.outputMappings[output.outputIndex])
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason, "card spatial mapping output index is duplicated");
    if (output.shardDimension &&
        *output.shardDimension >=
            preparation.outputDomains[output.outputIndex].size())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping shard dimension is outside output domain");
    if (output.activeTileIds.empty())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason, "card spatial mapping output has no active Tiles");
    if (output.temporalTileSizes.size() !=
        preparation.outputDomains[output.outputIndex].size())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping temporal tile rank differs from output");
    for (auto [tileSize, extent] :
         llvm::zip_equal(output.temporalTileSizes,
                         preparation.outputDomains[output.outputIndex]))
      if (tileSize <= 0 || tileSize > extent)
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card spatial mapping temporal tile is outside output domain");

    llvm::DenseSet<int64_t> outputTiles;
    for (TileId tileId : output.activeTileIds) {
      if (!availableTileValues.contains(tileId.getValue()))
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason, "card spatial mapping names an unavailable Tile");
      if (!outputTiles.insert(tileId.getValue()).second)
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card spatial mapping output contains a duplicate Tile");
    }
    if (!output.shardDimension && output.activeTileIds.size() != 1)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "unpartitioned card output requires exactly one active Tile");
    const int64_t extent =
        output.shardDimension
            ? preparation
                  .outputDomains[output.outputIndex][*output.shardDimension]
            : 1;
    if (output.activeTileIds.size() > static_cast<size_t>(extent))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping output has more active Tiles than nonempty "
          "shards");
    preparation.outputMappings[output.outputIndex] = &output;
  }
  if (llvm::is_contained(preparation.outputMappings, nullptr))
    return failCardModuleValue<TileMaterializationPreparation>(
        failureReason,
        "card spatial mapping must cover every function result exactly once");
  return preparation;
}

} // namespace wafer::tensor_program_to_card_module
