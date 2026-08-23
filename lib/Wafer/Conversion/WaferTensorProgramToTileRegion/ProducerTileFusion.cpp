//===- ProducerTileFusion.cpp - Structured producer tile fusion ----------===//

#include "ProducerTileFusionInternal.h"

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <deque>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

void recordStructuredOperationNodeMaterialization(
    mlir::Operation *source, mlir::Operation *materialized,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (!source || !materialized || !operationNodes)
    return;
  llvm::SmallVector<uint32_t, 2> nodeIds;
  for (const StructuredOperationNodeMapping &mapping : *operationNodes)
    if (mapping.operation == source)
      nodeIds.push_back(mapping.structuredNodeId);
  for (uint32_t nodeId : nodeIds)
    if (!llvm::any_of(*operationNodes, [&](const auto &mapping) {
          return mapping.operation == materialized &&
                 mapping.structuredNodeId == nodeId;
        }))
      operationNodes->push_back({materialized, nodeId});
}

/// Pushes a non-rank-reducing slice through a static expand_shape when every
/// reassociation group only inserts unit dimensions.  This is an exact view
/// rewrite: the resulting source slice has the same linear element interval,
/// and expanding that tile recreates the original slice type.  More general
/// rectangular slices of a linearized multi-dimensional group are deliberately
/// left as view barriers because they need a separate contiguity proof.
static std::optional<mlir::tensor::ExtractSliceOp>
bubbleSliceThroughUnitExpand(mlir::IRRewriter &rewriter,
                             mlir::tensor::ExtractSliceOp slice,
                             mlir::tensor::ExpandShapeOp expand) {
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(expand.getSrc().getType());
  auto expandedType =
      mlir::dyn_cast<mlir::RankedTensorType>(expand.getResult().getType());
  auto tileType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
  if (!sourceType || !expandedType || !tileType ||
      !sourceType.hasStaticShape() || !expandedType.hasStaticShape() ||
      !tileType.hasStaticShape() ||
      tileType.getRank() != expandedType.getRank() ||
      slice.getMixedOffsets().size() !=
          static_cast<size_t>(expandedType.getRank()) ||
      slice.getMixedSizes().size() !=
          static_cast<size_t>(expandedType.getRank()) ||
      !slice.hasUnitStride())
    return std::nullopt;

  llvm::SmallVector<mlir::OpFoldResult, 6> sourceOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 6> sourceSizes;
  llvm::SmallVector<mlir::OpFoldResult, 6> sourceStrides;
  llvm::SmallVector<int64_t, 6> sourceTileShape;
  sourceOffsets.reserve(sourceType.getRank());
  sourceSizes.reserve(sourceType.getRank());
  sourceStrides.reserve(sourceType.getRank());
  sourceTileShape.reserve(sourceType.getRank());
  llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation =
      expand.getReassociationIndices();
  if (reassociation.size() != static_cast<size_t>(sourceType.getRank()))
    return std::nullopt;

  for (auto [sourceDim, group] : llvm::enumerate(reassociation)) {
    std::optional<unsigned> nonUnitExpandedDim;
    for (int64_t expandedDim : group) {
      if (expandedDim < 0 || expandedDim >= expandedType.getRank())
        return std::nullopt;
      if (expandedType.getDimSize(expandedDim) != 1) {
        if (nonUnitExpandedDim)
          return std::nullopt;
        nonUnitExpandedDim = static_cast<unsigned>(expandedDim);
        continue;
      }
      std::optional<int64_t> offset =
          mlir::getConstantIntValue(slice.getMixedOffsets()[expandedDim]);
      std::optional<int64_t> size =
          mlir::getConstantIntValue(slice.getMixedSizes()[expandedDim]);
      if (!offset || *offset != 0 || !size || *size != 1)
        return std::nullopt;
    }

    int64_t tileExtent = 1;
    mlir::OpFoldResult offset = rewriter.getIndexAttr(0);
    if (nonUnitExpandedDim) {
      tileExtent = tileType.getDimSize(*nonUnitExpandedDim);
      offset = slice.getMixedOffsets()[*nonUnitExpandedDim];
      if (sourceType.getDimSize(sourceDim) !=
          expandedType.getDimSize(*nonUnitExpandedDim))
        return std::nullopt;
    } else if (sourceType.getDimSize(sourceDim) != 1) {
      return std::nullopt;
    }
    sourceOffsets.push_back(offset);
    sourceSizes.push_back(rewriter.getIndexAttr(tileExtent));
    sourceStrides.push_back(rewriter.getIndexAttr(1));
    sourceTileShape.push_back(tileExtent);
  }

  auto sourceTileType = mlir::RankedTensorType::get(
      sourceTileShape, sourceType.getElementType(), sourceType.getEncoding());
  rewriter.setInsertionPoint(slice);
  auto sourceSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
      slice.getLoc(), sourceTileType, expand.getSrc(), sourceOffsets,
      sourceSizes, sourceStrides);
  auto tileExpand = rewriter.create<mlir::tensor::ExpandShapeOp>(
      slice.getLoc(), tileType, sourceSlice.getResult(), reassociation);
  rewriter.replaceOp(slice, tileExpand.getResult());
  return sourceSlice;
}

/// Pushes a non-rank-reducing slice through a static collapse_shape when each
/// reassociation group removes only unit dimensions.  This is the inverse of
/// `bubbleSliceThroughUnitExpand`: the requested collapsed tile maps to one
/// exact rectangular source tile, and collapsing that tile recreates the
/// original result type.  Groups that flatten multiple non-unit dimensions
/// are left intact because a rectangular collapsed interval is not generally
/// a rectangular source slice.
static std::optional<mlir::tensor::ExtractSliceOp>
bubbleSliceThroughUnitCollapse(mlir::IRRewriter &rewriter,
                               mlir::tensor::ExtractSliceOp slice,
                               mlir::tensor::CollapseShapeOp collapse) {
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(collapse.getSrc().getType());
  auto collapsedType =
      mlir::dyn_cast<mlir::RankedTensorType>(collapse.getResult().getType());
  auto tileType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
  if (!sourceType || !collapsedType || !tileType ||
      !sourceType.hasStaticShape() || !collapsedType.hasStaticShape() ||
      !tileType.hasStaticShape() ||
      tileType.getRank() != collapsedType.getRank() ||
      slice.getMixedOffsets().size() !=
          static_cast<size_t>(collapsedType.getRank()) ||
      slice.getMixedSizes().size() !=
          static_cast<size_t>(collapsedType.getRank()) ||
      !slice.hasUnitStride())
    return std::nullopt;

  llvm::SmallVector<mlir::OpFoldResult, 6> sourceOffsets(
      sourceType.getRank(), rewriter.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult, 6> sourceSizes(
      sourceType.getRank(), rewriter.getIndexAttr(1));
  llvm::SmallVector<mlir::OpFoldResult, 6> sourceStrides(
      sourceType.getRank(), rewriter.getIndexAttr(1));
  llvm::SmallVector<int64_t, 6> sourceTileShape(sourceType.getRank(), 1);
  llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation =
      collapse.getReassociationIndices();
  if (reassociation.size() != static_cast<size_t>(collapsedType.getRank()))
    return std::nullopt;

  for (auto [collapsedDim, group] : llvm::enumerate(reassociation)) {
    std::optional<unsigned> nonUnitSourceDim;
    for (int64_t sourceDim : group) {
      if (sourceDim < 0 || sourceDim >= sourceType.getRank())
        return std::nullopt;
      if (sourceType.getDimSize(sourceDim) == 1)
        continue;
      if (nonUnitSourceDim)
        return std::nullopt;
      nonUnitSourceDim = static_cast<unsigned>(sourceDim);
    }

    if (!nonUnitSourceDim) {
      std::optional<int64_t> offset =
          mlir::getConstantIntValue(slice.getMixedOffsets()[collapsedDim]);
      std::optional<int64_t> size =
          mlir::getConstantIntValue(slice.getMixedSizes()[collapsedDim]);
      if (!offset || *offset != 0 || !size || *size != 1 ||
          collapsedType.getDimSize(collapsedDim) != 1)
        return std::nullopt;
      continue;
    }

    const unsigned sourceDim = *nonUnitSourceDim;
    if (sourceType.getDimSize(sourceDim) !=
        collapsedType.getDimSize(collapsedDim))
      return std::nullopt;
    sourceOffsets[sourceDim] = slice.getMixedOffsets()[collapsedDim];
    sourceSizes[sourceDim] = slice.getMixedSizes()[collapsedDim];
    sourceTileShape[sourceDim] = tileType.getDimSize(collapsedDim);
  }

  auto sourceTileType = mlir::RankedTensorType::get(
      sourceTileShape, sourceType.getElementType(), sourceType.getEncoding());
  rewriter.setInsertionPoint(slice);
  auto sourceSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
      slice.getLoc(), sourceTileType, collapse.getSrc(), sourceOffsets,
      sourceSizes, sourceStrides);
  auto tileCollapse = rewriter.create<mlir::tensor::CollapseShapeOp>(
      slice.getLoc(), tileType, sourceSlice.getResult(), reassociation);
  rewriter.replaceOp(slice, tileCollapse.getResult());
  return sourceSlice;
}

/// `tileAndFuseProducerOfSlice` may tie a tensor-semantics DPS producer tile
/// to a slice of the producer's original result.  That value is a convenient
/// reconstruction destination, but it is not the producer's semantic init:
/// retaining it keeps the full untiled producer live and a reduction-like DPS
/// op would consume the already-computed result a second time.  Rebind the
/// tiled result to the exact same slice of the original tied init.  The slice
/// remains in the ordinary upstream fusion worklist, so a fill/empty or any
/// other typed producer keeps its original SSA producer relationship.
static mlir::LogicalResult
rebaseFusedDPSInit(mlir::scf::SCFFuseProducerOfSliceResult &fused,
                   std::string *failureReason) {
  auto originalDps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
      fused.origProducer.getOwner());
  auto tiledResult =
      mlir::dyn_cast<mlir::OpResult>(fused.tiledAndFusedProducer);
  auto tiledDps = tiledResult
                      ? mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
                            tiledResult.getOwner())
                      : mlir::DestinationStyleOpInterface{};
  if (!originalDps || !tiledDps)
    return mlir::success();

  mlir::OpOperand *originalInit =
      originalDps.getDpsInitOperand(fused.origProducer.getResultNumber());
  mlir::OpOperand *tiledInit =
      tiledDps.getDpsInitOperand(tiledResult.getResultNumber());
  if (!originalInit || !tiledInit)
    return mlir::success();

  mlir::Value originalResult = fused.origProducer;
  mlir::Value originalInitValue = originalInit->get();
  mlir::Value tiledInitValue = tiledInit->get();
  if (tiledInitValue == originalResult) {
    if (originalInitValue.getType() != tiledInitValue.getType()) {
      setFailureReason(failureReason,
                       "fused DPS result and original init types differ");
      return mlir::failure();
    }
    tiledInit->set(originalInitValue);
    return mlir::success();
  }

  auto initSlice = tiledInitValue.getDefiningOp<mlir::tensor::ExtractSliceOp>();
  if (!initSlice || initSlice.getSource() != originalResult)
    return mlir::success();
  if (originalInitValue.getType() != originalResult.getType()) {
    setFailureReason(failureReason,
                     "fused DPS init cannot use the result tile relation");
    return mlir::failure();
  }
  initSlice->setOperand(0, originalInitValue);
  if (!llvm::is_contained(fused.generatedSlices, initSlice.getOperation()))
    fused.generatedSlices.push_back(initSlice.getOperation());
  return mlir::success();
}

/// A consumer window that cannot be expressed through a non-unit reshape
/// (the unit expand/collapse bubble failed) still demands the value behind
/// the reshape.  If that value is the result of a structured producer with a
/// configured temporal tile, materialize the producer through its temporal
/// traversal at its scope-body position and replace the producer result with
/// the assembled value: every leaf then reads its demanded window from the
/// shared assembled value instead of retaining the full untiled producer.
/// Returns true when the boundary value was assembled and the producer result
/// replaced, false when the view chain does not lead to a configured
/// structured producer (the window keeps reading the retained producer).
static mlir::FailureOr<bool> tryAssembleBoundaryProducerValue(
    mlir::IRRewriter &rewriter, TensorProgramScope scope,
    mlir::tensor::ExtractSliceOp slice,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
    std::string *failureReason) {
  mlir::Value value = slice.getSource();
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(definition)) {
      value = expand.getSrc();
      continue;
    }
    if (auto collapse =
            mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(definition)) {
      value = collapse.getSrc();
      continue;
    }
    break;
  }
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || definition->getBlock() != &scope.getBody())
    return false;
  auto structured = mlir::dyn_cast<mlir::linalg::LinalgOp>(definition);
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!structured || !resultType)
    return false;
  if (llvm::none_of(operationTemporalTiles,
                    [&](const StructuredOpTemporalTile &tile) {
                      return tile.operation == structured.getOperation();
                    }))
    return false;
  llvm::SmallVector<mlir::OpFoldResult, 4> fullOffsets(
      resultType.getRank(), rewriter.getIndexAttr(0));
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(structured.getOperation());
  mlir::FailureOr<mlir::Value> assembled =
      materializeConfiguredStructuredTraversal(
          rewriter, scope, structured.getOperation(), structured, fullOffsets,
          resultType.getShape(), /*loops=*/{}, operationTemporalTiles,
          nestedTemporalTiles, failureReason);
  if (mlir::failed(assembled))
    return mlir::failure();
  value.replaceAllUsesWith(*assembled);
  return true;
}

static bool haveEquivalentFoldResults(llvm::ArrayRef<mlir::OpFoldResult> lhs,
                                      llvm::ArrayRef<mlir::OpFoldResult> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip_equal(lhs, rhs), [](auto values) {
           auto [left, right] = values;
           if (left == right)
             return true;
           std::optional<int64_t> leftConstant =
               mlir::getConstantIntValue(left);
           std::optional<int64_t> rightConstant =
               mlir::getConstantIntValue(right);
           return leftConstant && rightConstant &&
                  *leftConstant == *rightConstant;
         });
}

static mlir::LogicalResult fuseProducerSlices(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    TensorProgramBody body, std::optional<TensorProgramScope> schedulingScope,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
    mlir::OpBuilder::Listener *insertionListener, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes,
    llvm::SmallVectorImpl<MaterializedCoupledProducerTile>
        *sharedMaterializedProducerTiles) {
  // With a structured loop nest, a fused tile is created in a nested block and
  // therefore cannot be mistaken for another untiled scope producer.  The
  // direct untiled API has no enclosing loop, so remember the finite set
  // of source producers that existed before fusion.  Otherwise slices
  // generated from a newly tiled clone can recursively fuse that clone again.
  llvm::DenseSet<mlir::Operation *> sourceProducers;
  for (mlir::Operation &operation : body.getBody().without_terminator())
    if (&operation != tiledConsumer &&
        mlir::isa<mlir::TilingInterface>(&operation))
      sourceProducers.insert(&operation);

  struct PendingProducerSlice {
    mlir::tensor::ExtractSliceOp slice;
    mlir::Operation *sourceConsumer = nullptr;
    mlir::Operation *sourceProducer = nullptr;
    llvm::SmallVector<unsigned, 2> sourceConsumerOperandNumbers;
  };
  std::deque<PendingProducerSlice> worklist;
  llvm::SmallVector<PendingProducerSlice, 8> seenRelations;
  llvm::SmallVector<PendingProducerSlice, 4> pendingSlices;
  llvm::SmallVector<MaterializedCoupledProducerTile, 4> localProducerTiles;
  llvm::SmallVectorImpl<MaterializedCoupledProducerTile>
      &materializedCoupledTiles =
          sharedMaterializedProducerTiles ? *sharedMaterializedProducerTiles
                                          : localProducerTiles;
  auto enqueueSlices = [&](llvm::ArrayRef<mlir::Operation *> operations,
                           mlir::Operation *downstreamProducer,
                           llvm::ArrayRef<mlir::Operation *> tiledConsumers) {
    for (mlir::Operation *operation : operations) {
      auto slice =
          mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(operation);
      mlir::Operation *sourceProducer =
          slice ? slice.getSource().getDefiningOp() : nullptr;
      // Generated operand slices must move strictly upstream through the
      // original source graph.  Some TilingInterface implementations also
      // report a destination/result slice whose source is the producer just
      // fused. Re-enqueuing that slice would clone the same producer forever
      // when there is no enclosing loop block to separate tiled clones.
      if (slice && downstreamProducer && sourceProducer &&
          sourceProducer->getBlock() == downstreamProducer->getBlock() &&
          !sourceProducer->isBeforeInBlock(downstreamProducer))
        continue;
      if (!slice)
        continue;
      llvm::SmallVector<unsigned, 2> operandNumbers;
      for (mlir::OpOperand &use : slice.getResult().getUses()) {
        if (!llvm::is_contained(tiledConsumers, use.getOwner()))
          continue;
        operandNumbers.push_back(use.getOperandNumber());
      }
      llvm::sort(operandNumbers);
      operandNumbers.erase(
          std::unique(operandNumbers.begin(), operandNumbers.end()),
          operandNumbers.end());
      if (operandNumbers.empty())
        continue;
      // tileAndFuseProducerOfSlice may reuse the same extract_slice operation
      // while retargeting it to the next producer upstream.  A slice is only
      // duplicate work when its exact source-consumer operand relation and its
      // current source producer have already been visited; remembering only
      // the operation pair conflates two operands of the same consumer.
      PendingProducerSlice relation{slice, downstreamProducer, sourceProducer,
                                    operandNumbers};
      auto seen = llvm::find_if(
          seenRelations, [&](const PendingProducerSlice &existing) {
            return existing.slice == relation.slice &&
                   existing.sourceConsumer == relation.sourceConsumer &&
                   existing.sourceProducer == relation.sourceProducer &&
                   llvm::equal(existing.sourceConsumerOperandNumbers,
                               relation.sourceConsumerOperandNumbers);
          });
      if (seen != seenRelations.end())
        continue;
      seenRelations.push_back(relation);
      auto equivalent =
          llvm::find_if(pendingSlices, [&](PendingProducerSlice &existing) {
            return existing.sourceConsumer == downstreamProducer &&
                   llvm::equal(existing.sourceConsumerOperandNumbers,
                               operandNumbers) &&
                   existing.slice->getBlock() == slice->getBlock() &&
                   existing.slice->isBeforeInBlock(slice) &&
                   existing.slice.getSource() == slice.getSource() &&
                   existing.slice.getType() == slice.getType() &&
                   llvm::equal(existing.slice.getMixedOffsets(),
                               slice.getMixedOffsets()) &&
                   llvm::equal(existing.slice.getMixedSizes(),
                               slice.getMixedSizes()) &&
                   llvm::equal(existing.slice.getMixedStrides(),
                               slice.getMixedStrides());
          });
      if (equivalent != pendingSlices.end()) {
        slice.getResult().replaceAllUsesWith(equivalent->slice.getResult());
        if (slice->use_empty())
          slice->erase();
        continue;
      }
      pendingSlices.push_back(relation);
      worklist.push_back(std::move(relation));
    }
  };

  // Tiling a consumer with the same SSA producer on more than one operand can
  // create equivalent extract_slice operations.  Keep distinct operand
  // identities separate so their connection actions remain independent;
  // only slices for the same exact endpoint relation are folded.
  llvm::SmallVector<mlir::Operation *, 4> initialSlices;
  for (mlir::Value operand : tiledConsumer->getOperands())
    if (auto slice = operand.getDefiningOp<mlir::tensor::ExtractSliceOp>())
      initialSlices.push_back(slice.getOperation());
  llvm::SmallVector<mlir::Operation *, 1> initialTiledConsumers{tiledConsumer};
  enqueueSlices(initialSlices, sourceConsumer, initialTiledConsumers);

  mlir::IRRewriter rewriter(body.getContext(), insertionListener);
  while (!worklist.empty()) {
    PendingProducerSlice pendingSlice = worklist.front();
    worklist.pop_front();
    mlir::tensor::ExtractSliceOp slice = pendingSlice.slice;
    auto pending =
        llvm::find_if(pendingSlices, [&](const PendingProducerSlice &existing) {
          return existing.slice == pendingSlice.slice &&
                 existing.sourceConsumer == pendingSlice.sourceConsumer &&
                 llvm::equal(existing.sourceConsumerOperandNumbers,
                             pendingSlice.sourceConsumerOperandNumbers);
        });
    if (pending != pendingSlices.end())
      pendingSlices.erase(pending);
    // Tiling a consumer of an existing view creates an extract_slice of that
    // extract_slice. Compose the exact relation first so fusion sees the
    // actual structured producer. Each fanout branch can then materialize a
    // consumer-compatible producer version from current SSA/indexing facts
    // instead of forcing the full producer through DDR.
    while (
        auto parent =
            slice.getSource().getDefiningOp<mlir::tensor::ExtractSliceOp>()) {
      llvm::SmallVector<mlir::OpFoldResult> offsets;
      llvm::SmallVector<mlir::OpFoldResult> sizes;
      llvm::SmallVector<mlir::OpFoldResult> strides;
      // Composition may materialize affine arithmetic. Keep it in the same
      // loop scope as the consumer slice whose dynamic offsets it uses.
      rewriter.setInsertionPoint(slice);
      if (mlir::failed(mlir::affine::mergeOffsetsSizesAndStrides(
              rewriter, slice.getLoc(), parent, slice, parent.getDroppedDims(),
              offsets, sizes, strides))) {
        setFailureReason(failureReason,
                         "candidate producer view composition failed");
        return mlir::failure();
      }
      auto composed = rewriter.create<mlir::tensor::ExtractSliceOp>(
          slice.getLoc(), slice.getType(), parent.getSource(), offsets, sizes,
          strides);
      rewriter.replaceOp(slice, composed.getResult());
      slice = composed;
    }
    if (auto expand =
            slice.getSource().getDefiningOp<mlir::tensor::ExpandShapeOp>()) {
      std::optional<mlir::tensor::ExtractSliceOp> sourceSlice =
          bubbleSliceThroughUnitExpand(rewriter, slice, expand);
      if (sourceSlice)
        slice = *sourceSlice;
    }
    if (auto collapse =
            slice.getSource().getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
      std::optional<mlir::tensor::ExtractSliceOp> sourceSlice =
          bubbleSliceThroughUnitCollapse(rewriter, slice, collapse);
      if (sourceSlice)
        slice = *sourceSlice;
    }
    // Padding defines values outside its source domain.  Treat it as an
    // explicit semantic materialization boundary: the ordinary pad lowering
    // creates that full initialized value, and this exact consumer slice then
    // selects its demanded window.  Fusing tensor.pad through a slice asks the
    // generic tiler to synthesize dynamic zero-length guards and
    // tensor.generate control flow, which is a different execution
    // representation rather than an operand-window refinement.
    if (slice.getSource().getDefiningOp<mlir::tensor::PadOp>())
      continue;
    // A tiled DPS producer can expose a slice of its original tensor.empty
    // destination while producer fusion walks upstream.  Keeping that slice
    // would retain (and later bufferize) the full untiled destination even
    // though its contents are undefined and every fused tile overwrites its
    // own result.  A fresh tile-local tensor.empty is exactly equivalent and
    // keeps physical storage proportional to the selected tile.  This is a
    // generic tensor-semantics fold; it does not depend on the producer kind
    // or workload.
    if (slice.getSource().getDefiningOp<mlir::tensor::EmptyOp>()) {
      auto tileType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
      if (tileType && tileType.hasStaticShape()) {
        rewriter.setInsertionPoint(slice);
        mlir::Value tileEmpty = rewriter
                                    .create<mlir::tensor::EmptyOp>(
                                        slice.getLoc(), tileType.getShape(),
                                        tileType.getElementType())
                                    .getResult();
        rewriter.replaceOp(slice, tileEmpty);
        continue;
      }
    }
    auto producerResult = mlir::dyn_cast<mlir::OpResult>(slice.getSource());
    // A window that cannot be expressed through a non-unit reshape of a
    // configured structured producer demands the value behind the reshape.
    // Assemble that boundary value through the producer's temporal traversal
    // (once, at the producer's scope-body position) so this and every other
    // leaf window reads the shared assembled value instead of retaining the
    // full untiled producer.
    if (producerResult && !operationTemporalTiles.empty() &&
        (mlir::isa<mlir::tensor::ExpandShapeOp>(producerResult.getOwner()) ||
         mlir::isa<mlir::tensor::CollapseShapeOp>(producerResult.getOwner()))) {
      assert(schedulingScope &&
             "temporal producer assembly requires a scheduling scope");
      mlir::FailureOr<bool> assembled = tryAssembleBoundaryProducerValue(
          rewriter, *schedulingScope, slice, operationTemporalTiles,
          nestedTemporalTiles, failureReason);
      if (mlir::failed(assembled))
        return mlir::failure();
      if (*assembled)
        continue;
    }
    // tensor.insert_slice is handled by an explicit windowing rewrite below
    // rather than through TilingInterface, so it is not part of the ordinary
    // source-producer closure.
    const bool windowedInsertSlice =
        producerResult &&
        mlir::isa<mlir::tensor::InsertSliceOp>(producerResult.getOwner());
    if (!producerResult ||
        producerResult.getOwner()->getBlock() != &body.getBody() ||
        (!windowedInsertSlice &&
         !sourceProducers.contains(producerResult.getOwner()))) {
      continue;
    }
    // A destination slice generated while fusing a DPS producer may point at
    // tensor.empty.  Tiling that placeholder creates another equivalent
    // empty destination slice and can grow the worklist without making the
    // consumer tile more precise.  Empty destinations carry no dataflow to
    // fuse; leave their slice for ordinary dead-support cleanup.
    if (mlir::isa<mlir::tensor::EmptyOp>(producerResult.getOwner())) {
      continue;
    }
    // tensor.insert_slice is not a TilingInterface in this MLIR build, but
    // its exact windowed semantics are explicit: a destination tile plus the
    // intersection of the inserted region with that tile.  Materialize the
    // destination window directly and keep the intersecting input slice on
    // the fusion worklist so attention/GEMM producers of the inserted value
    // stay windowed.  A window that does not intersect the inserted region is
    // a pure copy of the destination and needs no input at all.
    if (auto insert = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(
            producerResult.getOwner())) {
      materializeWindowedInsertSlice(rewriter, slice, insert, producerResult,
                                     enqueueSlices, materializedCoupledTiles);
      continue;
    }
    if (!mlir::isa<mlir::TilingInterface>(producerResult.getOwner())) {
      continue;
    }
    if (mlir::isa<WaferLinalgExtCollectiveOpInterface>(
            producerResult.getOwner())) {
      setFailureReason(
          failureReason,
          "candidate producer fusion requires an internal logical collective "
          "to remain a separate scheduling task");
      return mlir::failure();
    }
    if (!pendingSlice.sourceConsumer ||
        pendingSlice.sourceConsumerOperandNumbers.empty()) {
      continue;
    }
    for (unsigned operandNumber : pendingSlice.sourceConsumerOperandNumbers) {
      if (operandNumber >= pendingSlice.sourceConsumer->getNumOperands()) {
        setFailureReason(
            failureReason,
            "tiled consumer operand identity left the source operation");
        return mlir::failure();
      }
    }

    // Multiple operand uses that request the same producer slice in the same
    // block consume one producer tile/version. This cache avoids cloning the
    // same immutable SSA producer once per equivalent operand use.
    auto reusable = llvm::find_if(
        materializedCoupledTiles,
        [&](const MaterializedCoupledProducerTile &materialized) {
          mlir::Operation *tiledOwner = materialized.tiledValue.getDefiningOp();
          return materialized.producerResult == producerResult &&
                 materialized.block == slice->getBlock() && tiledOwner &&
                 tiledOwner->getBlock() == slice->getBlock() &&
                 tiledOwner->isBeforeInBlock(slice) &&
                 materialized.tileType == slice.getType() &&
                 haveEquivalentFoldResults(materialized.offsets,
                                           slice.getMixedOffsets()) &&
                 haveEquivalentFoldResults(materialized.sizes,
                                           slice.getMixedSizes()) &&
                 haveEquivalentFoldResults(materialized.strides,
                                           slice.getMixedStrides());
        });
    if (reusable != materializedCoupledTiles.end()) {
      slice.getResult().replaceAllUsesWith(reusable->tiledValue);
      if (slice->use_empty())
        rewriter.eraseOp(slice);
      continue;
    }

    if (auto structured =
            mlir::dyn_cast<mlir::linalg::LinalgOp>(producerResult.getOwner())) {
      llvm::SmallVector<StructuredOpTemporalTile, 8> configuredTiles(
          operationTemporalTiles.begin(), operationTemporalTiles.end());
      const StructuredOpNestedTemporalTile *selectedNested = nullptr;
      bool hasNestedProducer = false;
      auto requestedType =
          mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
      for (const StructuredOpNestedTemporalTile &candidate :
           nestedTemporalTiles) {
        if (candidate.producer != structured.getOperation())
          continue;
        hasNestedProducer = true;
        if (!requestedType || candidate.parent != pendingSlice.sourceConsumer ||
            candidate.producerResult != producerResult.getResultNumber() ||
            !llvm::is_contained(pendingSlice.sourceConsumerOperandNumbers,
                                candidate.parentOperand) ||
            !llvm::equal(candidate.requestedResultExtents,
                         requestedType.getShape()))
          continue;
        if (selectedNested) {
          setFailureReason(
              failureReason,
              "nested producer slice matches several temporal classes");
          return mlir::failure();
        }
        selectedNested = &candidate;
      }
      if (hasNestedProducer && !selectedNested) {
        setFailureReason(
            failureReason,
            "nested producer slice has no selected temporal class");
        return mlir::failure();
      }
      if (selectedNested) {
        if (llvm::any_of(configuredTiles,
                         [&](const StructuredOpTemporalTile &tile) {
                           return tile.operation == structured.getOperation();
                         })) {
          setFailureReason(
              failureReason,
              "nested producer also has a top-level temporal assignment");
          return mlir::failure();
        }
        configuredTiles.push_back({structured.getOperation(),
                                   selectedNested->iteratorTileSizes,
                                   selectedNested->waveLoopOrder});
      }
      auto selected = llvm::find_if(
          configuredTiles, [&](const StructuredOpTemporalTile &tile) {
            return tile.operation == structured.getOperation();
          });
      if (selected != configuredTiles.end()) {
        assert(schedulingScope &&
               "temporal producer materialization requires a scheduling "
               "scope");
        mlir::Type tileType = slice.getType();
        llvm::SmallVector<mlir::OpFoldResult, 4> offsets =
            llvm::to_vector(slice.getMixedOffsets());
        llvm::SmallVector<mlir::OpFoldResult, 4> sizes =
            llvm::to_vector(slice.getMixedSizes());
        llvm::SmallVector<mlir::OpFoldResult, 4> strides =
            llvm::to_vector(slice.getMixedStrides());
        rewriter.setInsertionPoint(slice);
        auto requestedType =
            mlir::cast<mlir::RankedTensorType>(slice.getType());
        mlir::FailureOr<mlir::Value> tiled =
            materializeConfiguredStructuredTraversal(
                rewriter, *schedulingScope, structured.getOperation(),
                structured, slice.getMixedOffsets(), requestedType.getShape(),
                loops, configuredTiles, nestedTemporalTiles, failureReason,
                /*outputDestination=*/{}, /*destinationBaseOffsets=*/{},
                operationNodes);
        if (mlir::failed(tiled))
          return mlir::failure();
        materializedCoupledTiles.push_back(MaterializedCoupledProducerTile{
            producerResult, slice->getBlock(), tileType, std::move(offsets),
            std::move(sizes), std::move(strides), *tiled});
        slice.getResult().replaceAllUsesWith(*tiled);
        if (slice->use_empty())
          rewriter.eraseOp(slice);
        continue;
      }
    }

    std::optional<mlir::scf::SCFFuseProducerOfSliceResult> fused =
        mlir::scf::tileAndFuseProducerOfSlice(rewriter, slice, loops);
    if (!fused) {
      setFailureReason(failureReason, "candidate producer tile fusion failed");
      return mlir::failure();
    }
    if (mlir::failed(rebaseFusedDPSInit(*fused, failureReason)))
      return mlir::failure();
    for (mlir::Operation *tiledOperation : fused->tiledOps)
      recordStructuredOperationNodeMaterialization(
          fused->origProducer.getOwner(), tiledOperation, operationNodes);
    materializedCoupledTiles.push_back(MaterializedCoupledProducerTile{
        producerResult, slice->getBlock(), slice.getType(),
        llvm::to_vector(slice.getMixedOffsets()),
        llvm::to_vector(slice.getMixedSizes()),
        llvm::to_vector(slice.getMixedStrides()),
        fused->tiledAndFusedProducer});
    enqueueSlices(fused->generatedSlices, fused->origProducer.getOwner(),
                  fused->tiledOps);
    if (slice->use_empty())
      rewriter.eraseOp(slice);
  }
  return mlir::success();
}

mlir::LogicalResult fuseCandidateProducerSlices(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    TensorProgramScope scope,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
    mlir::OpBuilder::Listener *insertionListener, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  return fuseProducerSlices(tiledConsumer, sourceConsumer, scope, scope, loops,
                            operationTemporalTiles, nestedTemporalTiles,
                            insertionListener, failureReason, operationNodes,
                            /*sharedMaterializedProducerTiles=*/nullptr);
}

mlir::LogicalResult fuseCandidateProducerSlicesWithCache(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    TensorProgramScope scope,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
    mlir::OpBuilder::Listener *insertionListener, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes,
    llvm::SmallVectorImpl<MaterializedCoupledProducerTile>
        &materializedProducerTiles) {
  return fuseProducerSlices(tiledConsumer, sourceConsumer, scope, scope, loops,
                            operationTemporalTiles, nestedTemporalTiles,
                            insertionListener, failureReason, operationNodes,
                            &materializedProducerTiles);
}

void reuseMaterializedProducerTiles(
    llvm::ArrayRef<mlir::Operation *> generatedSlices,
    llvm::ArrayRef<MaterializedCoupledProducerTile> materializedProducerTiles) {
  for (mlir::Operation *operation : generatedSlices) {
    auto slice =
        mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(operation);
    auto producerResult =
        slice ? mlir::dyn_cast<mlir::OpResult>(slice.getSource())
              : mlir::OpResult{};
    if (!slice || !producerResult)
      continue;
    auto reusable = llvm::find_if(
        materializedProducerTiles,
        [&](const MaterializedCoupledProducerTile &materialized) {
          mlir::Operation *tiledOwner = materialized.tiledValue.getDefiningOp();
          return materialized.producerResult == producerResult &&
                 materialized.block == slice->getBlock() && tiledOwner &&
                 tiledOwner->getBlock() == slice->getBlock() &&
                 tiledOwner->isBeforeInBlock(slice) &&
                 materialized.tileType == slice.getType() &&
                 haveEquivalentFoldResults(materialized.offsets,
                                           slice.getMixedOffsets()) &&
                 haveEquivalentFoldResults(materialized.sizes,
                                           slice.getMixedSizes()) &&
                 haveEquivalentFoldResults(materialized.strides,
                                           slice.getMixedStrides());
        });
    if (reusable == materializedProducerTiles.end())
      continue;
    slice.getResult().replaceAllUsesWith(reusable->tiledValue);
    if (slice->use_empty())
      slice.erase();
  }
}

mlir::LogicalResult fuseTensorProgramProducerSlices(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    mlir::func::FuncOp function,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  return fuseProducerSlices(
      tiledConsumer, sourceConsumer, TensorProgramBody(function), std::nullopt,
      loops, /*operationTemporalTiles=*/{}, /*nestedTemporalTiles=*/{},
      /*insertionListener=*/nullptr, failureReason,
      /*operationNodes=*/nullptr,
      /*sharedMaterializedProducerTiles=*/nullptr);
}

void eraseDeadCandidateSupportClosure(
    TensorProgramScope scope,
    llvm::ArrayRef<mlir::Operation *> preservedOperations) {
  // Tiled producer fusion leaves the original untiled producer and its
  // transitive pure support dead. Use the operation effect contract rather
  // than an op-name allowlist so arbitrary shaped views and held-out
  // side-effect-free structured producers disappear with that closure, while
  // unknown or observable effects remain explicit.
  llvm::SmallVector<mlir::Operation *, 8> operations;
  llvm::DenseSet<mlir::Operation *> preserved(preservedOperations.begin(),
                                              preservedOperations.end());
  for (mlir::Operation &op : scope.getBody().without_terminator())
    operations.push_back(&op);
  for (mlir::Operation *op : llvm::reverse(operations)) {
    if (!preserved.contains(op) && mlir::isOpTriviallyDead(op))
      op->erase();
  }
}

void eraseDeadTensorProgramClosure(mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::Operation *, 8> operations;
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator())
    operations.push_back(&operation);
  for (mlir::Operation *operation : llvm::reverse(operations))
    if (mlir::isOpTriviallyDead(operation))
      operation->erase();
}

} // namespace wafer::tensor_program_to_tile_region
