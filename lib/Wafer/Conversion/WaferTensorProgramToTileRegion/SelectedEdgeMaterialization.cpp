//===- SelectedEdgeMaterialization.cpp - Apply selected edge actions ===//

#include "SelectedEdgeMaterialization.h"
#include "EdgeFragmentPlanning.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

mlir::LogicalResult failResult(std::string *failureReason,
                               llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

} // namespace

mlir::Value createExactSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes);

mlir::FailureOr<mlir::Value> getOrMaterializeSource(
    TensorProgramScope scope, mlir::Operation *producer,
    unsigned producerResult, llvm::ArrayRef<int64_t> offsets,
    llvm::ArrayRef<int64_t> sizes,
    llvm::SmallVectorImpl<MaterializedSource> &materialized,
    llvm::DenseSet<mlir::Operation *> &preserved,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  auto existing = llvm::find_if(materialized, [&](const auto &candidate) {
    return candidate.producer == producer &&
           candidate.result == producerResult &&
           llvm::equal(candidate.offsets, offsets) &&
           llvm::equal(candidate.sizes, sizes);
  });
  if (existing != materialized.end())
    return existing->value;
  auto containing = llvm::find_if(materialized, [&](const auto &candidate) {
    return candidate.producer == producer &&
           candidate.result == producerResult &&
           isContained(offsets, sizes, candidate.offsets, candidate.sizes);
  });
  if (containing != materialized.end()) {
    mlir::Operation *definition = containing->value.getDefiningOp();
    if (!definition || definition->getBlock() != &scope.getBody())
      return fail<mlir::Value>(
          failureReason, "materialized producer domain is not available in the "
                         "tensor-program body");
    llvm::SmallVector<int64_t, 4> relativeOffsets;
    relativeOffsets.reserve(offsets.size());
    for (auto [offset, containerOffset] :
         llvm::zip_equal(offsets, containing->offsets))
      relativeOffsets.push_back(offset - containerOffset);
    mlir::OpBuilder builder(definition);
    builder.setInsertionPointAfter(definition);
    mlir::Value slice = createExactSlice(
        builder, producer->getLoc(), containing->value, relativeOffsets, sizes);
    preserved.insert(slice.getDefiningOp());
    materialized.push_back(MaterializedSource{producer, producerResult,
                                              llvm::to_vector(offsets),
                                              llvm::to_vector(sizes), slice});
    return slice;
  }
  // A full-domain, one-wave op is already the exact actual traversal.  Keep
  // that SSA result in place instead of cloning it through the generic tile
  // materializer: query-local peer endpoints refer to its current operands,
  // and replacing them with untracked cloned tensor.empty values would sever
  // the selected receive/send relation before TileRegion conversion.
  if (isFullStaticResultDomain(producer, producerResult, offsets, sizes) &&
      isOneFullTemporalWaveClosure(producer, operationTemporalTiles)) {
    preserved.insert(producer);
    mlir::Value value = producer->getResult(producerResult);
    materialized.push_back(MaterializedSource{producer, producerResult,
                                              llvm::to_vector(offsets),
                                              llvm::to_vector(sizes), value});
    return value;
  }
  mlir::FailureOr<mlir::Value> tiled = materializeCandidateRootTileValue(
      scope, producer, /*outputIndex=*/0, offsets, sizes,
      operationTemporalTiles, failureReason, operationNodes);
  if (mlir::failed(tiled))
    return mlir::failure();
  mlir::Operation *defining = tiled->getDefiningOp();
  if (!defining)
    return fail<mlir::Value>(
        failureReason,
        "selected producer tile did not materialize an operation result");
  preserved.insert(defining);
  materialized.push_back(MaterializedSource{producer, producerResult,
                                            llvm::to_vector(offsets),
                                            llvm::to_vector(sizes), *tiled});
  return *tiled;
}

void eraseDeadExcept(TensorProgramScope scope,
                     const llvm::DenseSet<mlir::Operation *> &preserved) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *, 32> operations;
    for (mlir::Operation &operation : scope.getBody().without_terminator())
      operations.push_back(&operation);
    for (mlir::Operation *operation : llvm::reverse(operations)) {
      if (preserved.contains(operation) || !mlir::isOpTriviallyDead(operation))
        continue;
      operation->erase();
      changed = true;
    }
  }
}

bool isInBackwardClosure(mlir::Value value, mlir::Operation *needle,
                         llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  if (definition == needle)
    return true;
  if (llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
        return isInBackwardClosure(operand, needle, visited);
      }))
    return true;

  // Region-bearing SSA definitions such as scf.for/scf.if derive each op
  // result from the corresponding region terminator operand. Lexically
  // captured values used only by a steady-state loop body are not operands of
  // the parent op, so following only definition operands loses a real dataflow
  // dependency. Follow the yielded value for this exact result number; the
  // visited set closes loop-carried cycles.
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return false;
  const unsigned resultNumber = result.getResultNumber();
  for (mlir::Region &region : definition->getRegions())
    for (mlir::Block &block : region) {
      mlir::Operation *terminator = block.getTerminator();
      if (terminator && resultNumber < terminator->getNumOperands() &&
          isInBackwardClosure(terminator->getOperand(resultNumber), needle,
                              visited))
        return true;
    }
  return false;
}

bool isInSelectedOutputClosure(TensorProgramScope scope,
                               llvm::ArrayRef<SpatialOutputShard> outputShards,
                               mlir::Operation *operation) {
  mlir::func::ReturnOp returnOp = scope.getReturn();
  for (const SpatialOutputShard &shard : outputShards) {
    if (shard.outputIndex >= returnOp.getNumOperands())
      continue;
    llvm::DenseSet<mlir::Value> visited;
    if (isInBackwardClosure(returnOp.getOperand(shard.outputIndex), operation,
                            visited))
      return true;
  }
  return false;
}

/// Returns whether an ordinary selected output traversal already supplies a
/// repeated consumer-driven wave containing `operation`.  In that case a
/// coupled producer must stay on the consumer's recursive tiling path: eagerly
/// assembling the producer's whole spatial shard would turn one fused wave
/// pipeline into two sequential traversals and erase the selected temporal
/// relationship between the edge endpoints.
bool hasSplitSelectedOutputTraversal(
    TensorProgramScope scope, llvm::ArrayRef<SpatialOutputShard> outputShards,
    mlir::Operation *operation) {
  mlir::func::ReturnOp returnOp = scope.getReturn();
  for (const SpatialOutputShard &shard : outputShards) {
    if (shard.outputIndex >= returnOp.getNumOperands() ||
        shard.temporalTileSizes == shard.sizes)
      continue;
    llvm::DenseSet<mlir::Value> visited;
    if (isInBackwardClosure(returnOp.getOperand(shard.outputIndex), operation,
                            visited))
      return true;
  }
  return false;
}

mlir::Value createExactSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes) {
  auto sourceType = mlir::cast<mlir::RankedTensorType>(source.getType());
  auto sliceType = mlir::RankedTensorType::get(
      sizes, sourceType.getElementType(), sourceType.getEncoding());
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  for (auto [offset, size] : llvm::zip_equal(offsets, sizes)) {
    mixedOffsets.push_back(builder.getIndexAttr(offset));
    mixedSizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }
  return builder
      .create<mlir::tensor::ExtractSliceOp>(loc, sliceType, source,
                                            mixedOffsets, mixedSizes, strides)
      .getResult();
}

mlir::Value insertExactSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source, mlir::Value destination,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  for (auto [offset, size] : llvm::zip_equal(offsets, sizes)) {
    mixedOffsets.push_back(builder.getIndexAttr(offset));
    mixedSizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }
  return builder
      .create<mlir::tensor::InsertSliceOp>(loc, source, destination,
                                           mixedOffsets, mixedSizes, strides)
      .getResult();
}

std::optional<uint32_t> findStructuredNodeId(
    mlir::Operation *operation,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes) {
  auto found = llvm::find_if(
      operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
        return mapping.operation == operation;
      });
  if (found == operationNodes.end())
    return std::nullopt;
  return found->structuredNodeId;
}

mlir::LogicalResult materializeLocalShardResidency(MappedStrategy &mapped,
                                                   std::string *failureReason) {
  SpatialEdgeStrategy &strategy = mapped.strategy;
  auto producerType = mlir::cast<mlir::RankedTensorType>(
      mapped.producer->getResult(strategy.producerResult).getType());
  mlir::OpBuilder builder(mapped.consumer);
  mlir::Value slice =
      createExactSlice(builder, mapped.producer->getLoc(),
                       mapped.producer->getResult(strategy.producerResult),
                       strategy.producerOffsets, strategy.producerSizes);
  auto empty = builder.create<mlir::tensor::EmptyOp>(
      mapped.consumer->getLoc(), producerType.getShape(),
      producerType.getElementType(), mlir::ValueRange{},
      producerType.getEncoding());
  mlir::Value staged = insertExactSlice(
      builder, mapped.consumer->getLoc(), slice, empty.getResult(),
      strategy.producerOffsets, strategy.producerSizes);
  mapped.consumer->setOperand(strategy.consumerOperand, staged);
  return mlir::success();
}

mlir::LogicalResult materializeSpill(
    TensorProgramScope scope, MappedStrategy &mapped,
    llvm::SmallVectorImpl<MaterializedSource> &materialized,
    llvm::DenseSet<mlir::Operation *> &preserved,
    llvm::SmallVectorImpl<MaterializedRegionCutSpill>
        &materializedRegionCutSpills,
    llvm::SmallVectorImpl<CandidateSelectedDDRStage> &selectedDDRStages,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  auto wireStoredProducerToConsumer = [&](mlir::Value storedProducer) {
    if (mapped.requiresConsumerInputReconstruction) {
      // A support DAG may join several independently spilled structured
      // producers (for example tensor.insert_slice assembly). Rebind the
      // complete consumer operand once every selected producer spill exists;
      // doing it edge-by-edge would leave the other producer's SPM value in
      // the cloned tensor transform.
      return mlir::success();
    }
    mapped.consumer->setOperand(mapped.strategy.consumerOperand,
                                storedProducer);
    return mlir::success();
  };

  SpatialEdgeStrategy &strategy = mapped.strategy;
  if (strategy.action == SpatialEdgeAction::RegionCut) {
    auto existing =
        llvm::find_if(materializedRegionCutSpills,
                      [&](const MaterializedRegionCutSpill &spill) {
                        return spill.producer == mapped.producer &&
                               spill.result == strategy.producerResult &&
                               spill.offsets == strategy.producerOffsets &&
                               spill.sizes == strategy.producerSizes;
                      });
    if (existing != materializedRegionCutSpills.end()) {
      return wireStoredProducerToConsumer(existing->value);
    }
  }
  auto producerType = mlir::cast<mlir::RankedTensorType>(
      mapped.producer->getResult(strategy.producerResult).getType());
  mlir::Location loc = mapped.consumer->getLoc();
  auto ddrType = mlir::MemRefType::get(
      producerType.getShape(), producerType.getElementType(),
      mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(mapped.consumer->getContext(), MemorySpace::DDR,
                      MemLayout::Tensor));
  mlir::OpBuilder builder(mapped.consumer);
  auto allocation = builder.create<mlir::memref::AllocOp>(loc, ddrType);
  // This allocation is the selected edge action itself, not an incidental
  // source-program side effect. Keep it explicit when a Tile owns only a
  // subset of observable outputs; the partial-output purity check can then
  // distinguish selected storage from unknown source effects.
  preserved.insert(allocation.getOperation());
  auto destination = builder.create<mlir::bufferization::ToTensorOp>(
      loc, allocation.getResult(), /*restrict=*/true, /*writable=*/true);
  preserved.insert(destination.getOperation());
  mlir::FailureOr<mlir::Value> slice = getOrMaterializeSource(
      scope, mapped.producer, strategy.producerResult, strategy.producerOffsets,
      strategy.producerSizes, materialized, preserved, operationTemporalTiles,
      failureReason, operationNodes);
  if (mlir::failed(slice))
    return mlir::failure();
  mlir::Value stored =
      insertExactSlice(builder, loc, *slice, destination.getResult(),
                       strategy.producerOffsets, strategy.producerSizes);
  if (strategy.action == SpatialEdgeAction::RegionCut) {
    // The functional insert result still exposes its producer slice to the
    // generic tensor tiler. If it is wired directly to the consumer, temporal
    // consumer materialization can clone the insert (and therefore the spill)
    // into every consumer wave, silently composing two supposedly independent
    // op stages. Keep the exact store alive, then reopen the same compiler-
    // owned DDR allocation as a read-only SSA boundary. TileRegion lowering
    // sees the store before the new view and materializes later consumer loads
    // without a tensor-level producer path to fuse across the cut.
    preserved.insert(stored.getDefiningOp());
    auto sealed = builder.create<mlir::bufferization::ToTensorOp>(
        loc, allocation.getResult(), /*restrict=*/false,
        /*writable=*/false);
    preserved.insert(sealed.getOperation());
    mlir::Value sealedSlice =
        createExactSlice(builder, loc, sealed.getResult(),
                         strategy.producerOffsets, strategy.producerSizes);
    preserved.insert(sealedSlice.getDefiningOp());

    // One independently executed producer may fan out to several later op
    // stages or peer sends. They must all read the stored DDR value;
    // retaining a cached SPM traversal for a later fanout would keep that SPM
    // value live across this cut. Preserve only the current store's backward
    // slice, and redirect every already-materialized external fanout plus the
    // exact source cache to the compact read-only DDR view.
    llvm::DenseSet<mlir::Value> visitedValues;
    llvm::DenseSet<mlir::Operation *> storeDependencies;
    llvm::SmallVector<mlir::Value, 16> worklist{stored};
    while (!worklist.empty()) {
      mlir::Value value = worklist.pop_back_val();
      if (!value || !visitedValues.insert(value).second)
        continue;
      mlir::Operation *definition = value.getDefiningOp();
      if (!definition || !storeDependencies.insert(definition).second)
        continue;
      worklist.append(definition->operand_begin(), definition->operand_end());
    }

    // A previously materialized fanout (most notably a resident contribution
    // to a peer-fragment DDR assembly) can lexically precede this deferred
    // RegionCut. Replacing that use with the fresh sealed view without moving
    // the selected store would create a use-before-definition and, more
    // importantly, put the fanout on the wrong side of the explicit op-stage
    // boundary. Place the complete static DDR store/reopen chain immediately
    // before the earliest such use. The original producer slice already
    // dominates every one of its uses, so this preserves SSA and makes the
    // concrete operation order match the selected baseline schedule.
    auto getTopLevelInScope = [&](mlir::Operation *operation) {
      while (operation && operation->getBlock() != &scope.getBody())
        operation = operation->getParentOp();
      return operation;
    };
    mlir::Operation *earliestExternalFanout = nullptr;
    for (mlir::OpOperand &use : slice->getUses()) {
      if (storeDependencies.contains(use.getOwner()))
        continue;
      mlir::Operation *root = getTopLevelInScope(use.getOwner());
      if (!root)
        continue;
      if (!earliestExternalFanout ||
          root->isBeforeInBlock(earliestExternalFanout))
        earliestExternalFanout = root;
    }
    mlir::Operation *sealedSliceDefinition = sealedSlice.getDefiningOp();
    if (earliestExternalFanout &&
        sealedSliceDefinition->getBlock() ==
            earliestExternalFanout->getBlock() &&
        !sealedSliceDefinition->isBeforeInBlock(earliestExternalFanout)) {
      allocation->moveBefore(earliestExternalFanout);
      destination->moveAfter(allocation);
      stored.getDefiningOp()->moveAfter(destination);
      sealed->moveAfter(stored.getDefiningOp());
      sealedSliceDefinition->moveAfter(sealed);
    }
    for (mlir::OpOperand &use : llvm::make_early_inc_range(slice->getUses()))
      if (!storeDependencies.contains(use.getOwner()))
        use.set(sealedSlice);
    for (MaterializedSource &cached : materialized) {
      if (cached.producer != mapped.producer ||
          cached.result != strategy.producerResult ||
          !isContained(cached.offsets, cached.sizes, strategy.producerOffsets,
                       strategy.producerSizes))
        continue;
      if (cached.offsets == strategy.producerOffsets &&
          cached.sizes == strategy.producerSizes) {
        cached.value = sealedSlice;
        continue;
      }
      cached.value = createExactSlice(builder, loc, sealed.getResult(),
                                      cached.offsets, cached.sizes);
      preserved.insert(cached.value.getDefiningOp());
    }
    stored = sealed.getResult();
    materializedRegionCutSpills.push_back(MaterializedRegionCutSpill{
        mapped.producer, strategy.producerResult, strategy.producerOffsets,
        strategy.producerSizes, stored, allocation.getResult()});
    if (!operationNodes)
      return failResult(
          failureReason,
          "selected DDR stage requires structured producer identity");
    std::optional<uint32_t> producerNode =
        findStructuredNodeId(mapped.producer, *operationNodes);
    if (!producerNode)
      return failResult(
          failureReason,
          "selected DDR stage producer has no structured node identity");
    selectedDDRStages.push_back(
        CandidateSelectedDDRStage{allocation.getResult(), *producerNode});
  }
  if (mlir::failed(wireStoredProducerToConsumer(stored)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult materializeRecompute(
    TensorProgramScope scope, MappedStrategy &mapped,
    llvm::SmallVectorImpl<MaterializedSource> &materialized,
    llvm::DenseSet<mlir::Operation *> &preserved,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason) {
  if (!mlir::isMemoryEffectFree(mapped.producer))
    return failResult(failureReason,
                      "recompute requires a pure current-SSA producer");
  // The producer op-wave remains an actual scheduled obligation. Materialize
  // its selected domain first, then wire an independent cloned producer into
  // the consumer traversal. Across a CardModule this is the concrete
  // distinction between retention and recomputation, even when a focused
  // single-Tile test places both obligations on one endpoint.
  mlir::FailureOr<mlir::Value> original = getOrMaterializeSource(
      scope, mapped.producer, mapped.strategy.producerResult,
      mapped.strategy.producerOffsets, mapped.strategy.producerSizes,
      materialized, preserved, operationTemporalTiles, failureReason,
      &operationNodes);
  if (mlir::failed(original))
    return mlir::failure();
  mlir::OpBuilder builder(mapped.consumer);
  mlir::IRMapping mapping;
  mlir::Operation *clone = builder.clone(*mapped.producer, mapping);
  if (!clone || clone->getNumResults() != mapped.producer->getNumResults())
    return failResult(failureReason,
                      "recompute could not clone the exact producer");
  auto sourceNode = llvm::find_if(
      operationNodes, [&](const StructuredOperationNodeMapping &entry) {
        return entry.operation == mapped.producer;
      });
  if (sourceNode != operationNodes.end())
    operationNodes.push_back({clone, sourceNode->structuredNodeId});
  mapped.consumer->setOperand(mapped.strategy.consumerOperand,
                              clone->getResult(mapped.strategy.producerResult));
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
