//===- SelectedEdgeActions.cpp - Local selected edge actions ------------===//

#include "SelectedEdgeLoweringInternal.h"

#include "EdgeFragmentPlanning.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

mlir::LogicalResult
materializeSelectedDirectActions(SelectedEdgeLoweringState &state) {
  TensorProgramScope scope = state.scope;
  TileId currentTile = state.currentTile;
  auto outputShards = state.outputShards;
  auto &mappedTemporalTiles = state.mapping.operationTemporalTiles;
  auto &mappedOperationNodes = state.mapping.operationNodes;
  auto &mappedStrategies = state.mapping.strategies;
  const bool independentDDRStages = state.mapping.independentDDRStages;
  auto &materialized = state.materializedSources;
  auto &materializedRegionCutSpills = state.regionCutSpills;
  auto &selectedDDRStages = state.selectedDDRStages;
  auto &preserved = state.preservedOperations;
  std::string *failureReason = state.failureReason;
  for (MappedStrategy &mapped : mappedStrategies) {
    SpatialEdgeStrategy &strategy = mapped.strategy;
    if (strategy.destinationTile != currentTile)
      continue;
    switch (strategy.action) {
    case SpatialEdgeAction::CoupledFusion: {
      // A full-domain consumer does not manufacture an extract_slice, so the
      // ordinary producer-fusion walk has no trigger.  When the common state
      // selected a finer iterator traversal for this producer, materialize
      // that exact traversal explicitly and keep its assembled SSA value in
      // the same region.  Otherwise the selected temporal dimension would be
      // silently replaced by the untiled source operation.
      if (!hasSplitSelectedOutputTraversal(scope, outputShards,
                                           mapped.consumer) &&
          isOneFullTemporalWave(mapped.consumer, mappedTemporalTiles) &&
          !isOneFullTemporalWaveForResultDemand(
              mapped.producer, strategy.producerResult, strategy.producerSizes,
              mappedTemporalTiles)) {
        mlir::FailureOr<mlir::Value> value = getOrMaterializeSource(
            scope, mapped.producer, strategy.producerResult,
            strategy.producerOffsets, strategy.producerSizes, materialized,
            preserved, mappedTemporalTiles, failureReason,
            &mappedOperationNodes);
        if (mlir::failed(value))
          return mlir::failure();
        auto fullType = mlir::dyn_cast<mlir::RankedTensorType>(
            mapped.producer->getResult(strategy.producerResult).getType());
        if (!fullType)
          return reportSelectedEdgeFailure(
              failureReason, "coupled producer result is not a ranked tensor");
        if (isFullStaticResultDomain(mapped.producer, strategy.producerResult,
                                     strategy.producerOffsets,
                                     strategy.producerSizes) &&
            value->getType() == fullType) {
          // A full selected window is already in the consumer's global index
          // space.  Re-embedding it through tensor.empty/insert_slice creates
          // a second full-size physical allocation without changing any
          // index relation.
          mapped.consumer->setOperand(strategy.consumerOperand, *value);
        } else {
          // `getOrMaterializeSource` returns the compact selected producer
          // window.  The consumer still owns global indexing maps, so expose
          // that window at its global result offsets before consumer tiling.
          mlir::OpBuilder builder(mapped.consumer);
          auto empty = builder.create<mlir::tensor::EmptyOp>(
              mapped.consumer->getLoc(), fullType.getShape(),
              fullType.getElementType(), mlir::ValueRange{},
              fullType.getEncoding());
          mlir::Value embedded = insertExactSlice(
              builder, mapped.consumer->getLoc(), *value, empty.getResult(),
              strategy.producerOffsets, strategy.producerSizes);
          mapped.consumer->setOperand(strategy.consumerOperand, embedded);
        }
      }
      break;
    }
    case SpatialEdgeAction::LocalShardResidency:
      if (mlir::failed(materializeLocalShardResidency(mapped, failureReason)))
        return mlir::failure();
      break;
    case SpatialEdgeAction::SpillReload: {
      if (independentDDRStages)
        break;
      if (mlir::failed(materializeSpill(scope, mapped, materialized, preserved,
                                        materializedRegionCutSpills,
                                        selectedDDRStages, mappedTemporalTiles,
                                        failureReason, &mappedOperationNodes)))
        return mlir::failure();
    } break;
    case SpatialEdgeAction::RegionCut: {
      if (independentDDRStages)
        break;
      if (mlir::failed(materializeSpill(scope, mapped, materialized, preserved,
                                        materializedRegionCutSpills,
                                        selectedDDRStages, mappedTemporalTiles,
                                        failureReason, &mappedOperationNodes)))
        return mlir::failure();
      break;
    }
    case SpatialEdgeAction::Recompute:
      if (mlir::failed(materializeRecompute(
              scope, mapped, materialized, preserved, mappedTemporalTiles,
              mappedOperationNodes, failureReason)))
        return mlir::failure();
      break;
    case SpatialEdgeAction::PeerFragments:
      break;
    case SpatialEdgeAction::LocalPhysicalConversion:
      if (mlir::failed(materializeLocalShardResidency(mapped, failureReason)))
        return mlir::failure();
      break;
    }
  }
  return mlir::success();
}

mlir::LogicalResult
materializeSelectedSourceStages(SelectedEdgeLoweringState &state) {
  TensorProgramScope scope = state.scope;
  TileId currentTile = state.currentTile;
  auto outputShards = state.outputShards;
  auto &mappedTemporalTiles = state.mapping.operationTemporalTiles;
  auto &mappedOperationNodes = state.mapping.operationNodes;
  auto &mappedStrategies = state.mapping.strategies;
  const bool independentDDRStages = state.mapping.independentDDRStages;
  auto &materialized = state.materializedSources;
  auto &materializedRegionCutSpills = state.regionCutSpills;
  auto &selectedDDRStages = state.selectedDDRStages;
  auto &preserved = state.preservedOperations;
  std::string *failureReason = state.failureReason;
  if (independentDDRStages) {
    struct SourceOnlyDomain {
      mlir::Operation *operation = nullptr;
      llvm::SmallVector<int64_t, 4> offsets;
      llvm::SmallVector<int64_t, 4> sizes;
    };
    llvm::SmallVector<SourceOnlyDomain, 4> sourceOnlyDomains;
    auto appendSourceDomain = [&](mlir::Operation *operation,
                                  llvm::ArrayRef<int64_t> offsets,
                                  llvm::ArrayRef<int64_t> sizes) {
      if (!operation || offsets.size() != sizes.size())
        return;
      auto existing =
          llvm::find_if(sourceOnlyDomains, [&](const SourceOnlyDomain &domain) {
            return domain.operation == operation &&
                   llvm::ArrayRef(domain.offsets) == offsets &&
                   llvm::ArrayRef(domain.sizes) == sizes;
          });
      if (existing == sourceOnlyDomains.end())
        sourceOnlyDomains.push_back(SourceOnlyDomain{
            operation, llvm::to_vector(offsets), llvm::to_vector(sizes)});
    };
    for (MappedStrategy &mapped : mappedStrategies) {
      const bool hasLocalIncoming =
          llvm::any_of(mappedStrategies, [&](const MappedStrategy &incoming) {
            return incoming.consumer == mapped.producer &&
                   incoming.strategy.destinationTile == currentTile;
          });
      if (hasLocalIncoming)
        continue;
      if (mapped.strategy.action == SpatialEdgeAction::PeerFragments) {
        for (const SpatialEdgeFragment &fragment : mapped.strategy.fragments)
          if (fragment.sourceTile == currentTile)
            appendSourceDomain(mapped.producer, fragment.offsets,
                               fragment.sizes);
      } else if (mapped.strategy.sourceTile == currentTile) {
        appendSourceDomain(mapped.producer, mapped.strategy.producerOffsets,
                           mapped.strategy.producerSizes);
      }
    }
    llvm::sort(sourceOnlyDomains,
               [](const SourceOnlyDomain &lhs, const SourceOnlyDomain &rhs) {
                 if (lhs.operation != rhs.operation)
                   return lhs.operation->isBeforeInBlock(rhs.operation);
                 if (lhs.offsets != rhs.offsets)
                   return lhs.offsets < rhs.offsets;
                 return lhs.sizes < rhs.sizes;
               });
    for (size_t groupBegin = 0; groupBegin < sourceOnlyDomains.size();) {
      size_t groupEnd = groupBegin + 1;
      while (groupEnd < sourceOnlyDomains.size() &&
             sourceOnlyDomains[groupEnd].operation ==
                 sourceOnlyDomains[groupBegin].operation)
        ++groupEnd;
      mlir::Operation *operation = sourceOnlyDomains[groupBegin].operation;
      auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
          operation->getResult(0).getType());
      if (!resultType || !resultType.hasStaticShape())
        return reportSelectedEdgeFailure(
            failureReason,
            "source-only baseline op requires one static ranked result");
      auto ddrType = mlir::MemRefType::get(
          resultType.getShape(), resultType.getElementType(),
          mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(operation->getContext(), MemorySpace::DDR,
                          MemLayout::Tensor));
      mlir::OpBuilder builder(operation);
      auto allocation =
          builder.create<mlir::memref::AllocOp>(operation->getLoc(), ddrType);
      auto destination = builder.create<mlir::bufferization::ToTensorOp>(
          operation->getLoc(), allocation.getResult(),
          /*restrict=*/true, /*writable=*/true);
      preserved.insert(allocation.getOperation());
      preserved.insert(destination.getOperation());
      mlir::Value stored = destination.getResult();
      for (size_t index = groupBegin; index < groupEnd; ++index) {
        SourceOnlyDomain &domain = sourceOnlyDomains[index];
        mlir::FailureOr<mlir::Value> updated =
            materializeCandidateRootTileIntoDestination(
                scope, operation, domain.offsets, domain.sizes,
                mappedTemporalTiles, stored, failureReason,
                &mappedOperationNodes);
        if (mlir::failed(updated))
          return mlir::failure();
        stored = *updated;
      }
      mlir::Operation *storedDefinition = stored.getDefiningOp();
      if (!storedDefinition)
        return reportSelectedEdgeFailure(
            failureReason,
            "source-only baseline store has no materialized definition");
      preserved.insert(storedDefinition);
      builder.setInsertionPointAfter(storedDefinition);
      auto sealed = builder.create<mlir::bufferization::ToTensorOp>(
          operation->getLoc(), allocation.getResult(), /*restrict=*/false,
          /*writable=*/false);
      preserved.insert(sealed.getOperation());
      for (size_t index = groupBegin; index < groupEnd; ++index) {
        SourceOnlyDomain &domain = sourceOnlyDomains[index];
        mlir::Value compact =
            createExactSlice(builder, operation->getLoc(), sealed.getResult(),
                             domain.offsets, domain.sizes);
        preserved.insert(compact.getDefiningOp());
        auto cached = llvm::find_if(
            materialized, [&](const MaterializedSource &candidate) {
              return candidate.producer == operation && candidate.result == 0 &&
                     candidate.offsets == domain.offsets &&
                     candidate.sizes == domain.sizes;
            });
        if (cached == materialized.end())
          materialized.push_back(MaterializedSource{
              operation, /*result=*/0, domain.offsets, domain.sizes, compact});
        else
          cached->value = compact;
      }

      llvm::DenseSet<mlir::Value> visitedValues;
      llvm::DenseSet<mlir::Operation *> dependencyOperations;
      llvm::SmallVector<mlir::Value, 16> worklist{stored};
      while (!worklist.empty()) {
        mlir::Value value = worklist.pop_back_val();
        if (!value || !visitedValues.insert(value).second)
          continue;
        mlir::Operation *definition = value.getDefiningOp();
        if (!definition || !dependencyOperations.insert(definition).second)
          continue;
        worklist.append(definition->operand_begin(), definition->operand_end());
      }
      for (mlir::OpOperand &use :
           llvm::make_early_inc_range(operation->getResult(0).getUses()))
        if (!dependencyOperations.contains(use.getOwner()))
          use.set(sealed.getResult());
      groupBegin = groupEnd;
    }
  }
  return mlir::success();
}

mlir::LogicalResult
materializeSelectedConsumerStages(SelectedEdgeLoweringState &state) {
  TensorProgramScope scope = state.scope;
  TileId currentTile = state.currentTile;
  auto outputShards = state.outputShards;
  auto &mappedTemporalTiles = state.mapping.operationTemporalTiles;
  auto &mappedOperationNodes = state.mapping.operationNodes;
  auto &mappedStrategies = state.mapping.strategies;
  const bool independentDDRStages = state.mapping.independentDDRStages;
  auto &materialized = state.materializedSources;
  auto &materializedRegionCutSpills = state.regionCutSpills;
  auto &selectedDDRStages = state.selectedDDRStages;
  auto &preserved = state.preservedOperations;
  std::string *failureReason = state.failureReason;
  if (independentDDRStages) {
    // Every selected baseline consumer is an independent op-wave, including
    // consumers in an observable output closure. Materializing only internal
    // consumers would leave the final output traversal free to fuse and
    // recompute the original functional closure across explicit edge actions.
    // Search-policy CoupledFusion/LocalShardResidency candidates deliberately
    // do not enter this baseline-only path. Their ordinary output traversal
    // remains the materialization owner; only CoupledFusion is later accepted
    // as an actual producer-fusion witness.
    llvm::SmallVector<MappedStrategy *, 16> consumerOrder;
    for (MappedStrategy &mapped : mappedStrategies)
      if (mapped.strategy.destinationTile == currentTile)
        consumerOrder.push_back(&mapped);
    llvm::sort(consumerOrder, [](const MappedStrategy *lhs,
                                 const MappedStrategy *rhs) {
      if (lhs->consumer != rhs->consumer)
        return lhs->consumer->isBeforeInBlock(rhs->consumer);
      if (lhs->strategy.consumerOffsets != rhs->strategy.consumerOffsets)
        return lhs->strategy.consumerOffsets < rhs->strategy.consumerOffsets;
      return lhs->strategy.consumerSizes < rhs->strategy.consumerSizes;
    });
    llvm::SmallVector<MaterializedSource, 16> independentConsumers;
    llvm::SmallVector<MappedStrategy *, 16> deferredRegionCuts;
    for (MappedStrategy *mappedPointer : consumerOrder) {
      MappedStrategy &mapped = *mappedPointer;
      SpatialEdgeStrategy &strategy = mapped.strategy;
      if (llvm::any_of(independentConsumers,
                       [&](const MaterializedSource &existing) {
                         return existing.producer == mapped.consumer &&
                                existing.result == 0 &&
                                existing.offsets == strategy.consumerOffsets &&
                                existing.sizes == strategy.consumerSizes;
                       }))
        continue;
      // A baseline RegionCut is the explicit input boundary of this op. It
      // must be created only after its producer's own incoming boundaries and
      // independent traversal are available; eager whole-function lowering
      // would cache a pre-boundary recursively fused producer closure.
      for (MappedStrategy &incoming : mappedStrategies) {
        if (incoming.consumer != mapped.consumer ||
            incoming.strategy.destinationTile != currentTile ||
            (incoming.strategy.action != SpatialEdgeAction::RegionCut &&
             incoming.strategy.action != SpatialEdgeAction::SpillReload) ||
            llvm::is_contained(deferredRegionCuts, &incoming))
          continue;
        if (mlir::failed(materializeSpill(
                scope, incoming, materialized, preserved,
                materializedRegionCutSpills, selectedDDRStages,
                mappedTemporalTiles, failureReason, &mappedOperationNodes)))
          return mlir::failure();
        deferredRegionCuts.push_back(&incoming);
      }
      // Materialize the independent op-wave directly into its compiler-owned
      // DDR stage. Building a compact traversal value first would still need
      // to assemble the complete spatial shard in a tensor.empty, then copy
      // that full SPM object into DDR. The temporal coordinate would shrink
      // the compute leaves while leaving one or two full-shard SPM buffers
      // unchanged, so the deterministic capacity controller could never
      // legalize the actual Tile entry. The destination-backed traversal
      // carries the DDR tensor through its wave loops and only materializes
      // each configured leaf in SPM.
      auto fullType = mlir::dyn_cast<mlir::RankedTensorType>(
          mapped.consumer->getResult(0).getType());
      if (!fullType || !fullType.hasStaticShape())
        return reportSelectedEdgeFailure(
            failureReason,
            "independent consumer materialization requires one static ranked "
            "result domain");
      mlir::OpBuilder builder(mapped.consumer);
      auto ddrType = mlir::MemRefType::get(
          fullType.getShape(), fullType.getElementType(),
          mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(mapped.consumer->getContext(), MemorySpace::DDR,
                          MemLayout::Tensor));
      auto allocation = builder.create<mlir::memref::AllocOp>(
          mapped.consumer->getLoc(), ddrType);
      auto destination = builder.create<mlir::bufferization::ToTensorOp>(
          mapped.consumer->getLoc(), allocation.getResult(),
          /*restrict=*/true, /*writable=*/true);
      mlir::FailureOr<mlir::Value> stored =
          materializeCandidateRootTileIntoDestination(
              scope, mapped.consumer, strategy.consumerOffsets,
              strategy.consumerSizes, mappedTemporalTiles,
              destination.getResult(), failureReason, &mappedOperationNodes);
      if (mlir::failed(stored))
        return mlir::failure();
      mlir::Operation *storedDefinition = stored->getDefiningOp();
      if (!storedDefinition || storedDefinition->getBlock() != &scope.getBody())
        return reportSelectedEdgeFailure(
            failureReason,
            "independent consumer traversal did not produce a scope-local "
            "DDR stage");
      builder.setInsertionPointAfter(storedDefinition);
      auto sealed = builder.create<mlir::bufferization::ToTensorOp>(
          mapped.consumer->getLoc(), allocation.getResult(),
          /*restrict=*/false, /*writable=*/false);
      preserved.insert(allocation.getOperation());
      preserved.insert(destination.getOperation());
      preserved.insert(storedDefinition);
      preserved.insert(sealed.getOperation());
      mlir::Value selectedResult = sealed.getResult();
      mlir::Value cachedSelectedResult =
          createExactSlice(builder, mapped.consumer->getLoc(), selectedResult,
                           strategy.consumerOffsets, strategy.consumerSizes);
      preserved.insert(cachedSelectedResult.getDefiningOp());
      independentConsumers.push_back(MaterializedSource{
          mapped.consumer, /*result=*/0,
          llvm::SmallVector<int64_t, 4>(strategy.consumerOffsets),
          llvm::SmallVector<int64_t, 4>(strategy.consumerSizes),
          cachedSelectedResult});
      auto cached =
          llvm::find_if(materialized, [&](const MaterializedSource &candidate) {
            return candidate.producer == mapped.consumer &&
                   candidate.result == 0 &&
                   candidate.offsets == strategy.consumerOffsets &&
                   candidate.sizes == strategy.consumerSizes;
          });
      if (cached == materialized.end())
        materialized.push_back(independentConsumers.back());
      else
        cached->value = cachedSelectedResult;
      if (selectedResult.getType() == mapped.consumer->getResult(0).getType()) {
        // Some legal destination-style traversals retain the original result as
        // an initialization dependency.  Do not rewrite uses inside the
        // selected value's own backward slice; rewrite every external fanout so
        // the new value becomes the common execution path without creating a
        // cyclic SSA definition.
        llvm::DenseSet<mlir::Value> visitedValues;
        llvm::DenseSet<mlir::Operation *> dependencyOperations;
        llvm::SmallVector<mlir::Value, 16> worklist{selectedResult};
        while (!worklist.empty()) {
          mlir::Value value = worklist.pop_back_val();
          if (!value || !visitedValues.insert(value).second)
            continue;
          mlir::Operation *definition = value.getDefiningOp();
          if (!definition || !dependencyOperations.insert(definition).second)
            continue;
          worklist.append(definition->operand_begin(),
                          definition->operand_end());
        }
        for (mlir::OpOperand &use : llvm::make_early_inc_range(
                 mapped.consumer->getResult(0).getUses()))
          if (!dependencyOperations.contains(use.getOwner()))
            use.set(selectedResult);
      }
    }
  }
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
