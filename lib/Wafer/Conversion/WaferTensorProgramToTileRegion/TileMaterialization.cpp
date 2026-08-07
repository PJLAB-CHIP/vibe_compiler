//===- TileMaterialization.cpp - Candidate root tile materialization -===//

#include "Internal.h"

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <deque>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

static bool isCandidateOutputDestination(TensorProgramScope scope,
                                         mlir::Value value,
                                         unsigned outputIndex) {
  llvm::DenseSet<mlir::Value> visited;
  mlir::Value current = value;
  while (current && visited.insert(current).second) {
    if (isTensorProgramOutputBoundary(scope, current, outputIndex))
      return true;
    // A joint complete-rank traversal may end at an internal same-region
    // physical version. tensor.empty is the typed destination for that
    // version; bufferization later materializes the corresponding SPM root in
    // the actual Tile clone. This does not make an arbitrary producer result a
    // boundary.
    if (auto empty = current.getDefiningOp<mlir::tensor::EmptyOp>())
      return mlir::isa<mlir::RankedTensorType>(empty.getType());
    if (auto toTensor =
            current.getDefiningOp<mlir::bufferization::ToTensorOp>())
      return toTensor.getWritable() &&
             wafer::isWaferDDRMemRefType(toTensor.getMemref().getType());
    if (auto expand = current.getDefiningOp<mlir::tensor::ExpandShapeOp>()) {
      current = expand.getSrc();
      continue;
    }
    if (auto collapse =
            current.getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
      current = collapse.getSrc();
      continue;
    }
    if (auto cast = current.getDefiningOp<mlir::tensor::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    return false;
  }
  return false;
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

/// `tileAndFuseProducerOfSlice` may tie a tensor-semantics DPS producer tile
/// to a slice of the producer's original result.  That value is a convenient
/// reconstruction destination, but it is not the producer's semantic init:
/// retaining it keeps the full untiled producer live and a reduction-like DPS
/// op would consume the already-computed result a second time.  Rebind the
/// tiled result to the exact same slice of the original tied init.  The slice
/// remains in the ordinary upstream fusion worklist, so a fill/empty or any
/// other typed producer keeps its real SSA provenance.
static mlir::LogicalResult rebaseFusedDPSInit(
    mlir::scf::SCFFuseProducerOfSliceResult &fused,
    std::string *failureReason) {
  auto originalDps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
      fused.origProducer.getOwner());
  auto tiledResult =
      mlir::dyn_cast<mlir::OpResult>(fused.tiledAndFusedProducer);
  auto tiledDps =
      tiledResult
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

  auto initSlice =
      tiledInitValue.getDefiningOp<mlir::tensor::ExtractSliceOp>();
  if (!initSlice || initSlice.getSource() != originalResult)
    return mlir::success();
  if (originalInitValue.getType() != originalResult.getType()) {
    setFailureReason(failureReason,
                     "fused DPS init cannot use the result tile relation");
    return mlir::failure();
  }
  initSlice->setOperand(0, originalInitValue);
  if (!llvm::is_contained(fused.generatedSlices,
                          initSlice.getOperation()))
    fused.generatedSlices.push_back(initSlice.getOperation());
  return mlir::success();
}

mlir::LogicalResult fuseCandidateProducerSlices(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    TensorProgramScope scope,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  // With a structured loop nest, a fused tile is created in a nested block and
  // therefore cannot be mistaken for another untiled scope producer.  The
  // direct single-tile API has no enclosing loop, so remember the finite set
  // of source producers that existed before fusion.  Otherwise slices
  // generated from a newly tiled clone can recursively fuse that clone again.
  llvm::DenseSet<mlir::Operation *> sourceProducers;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    if (&operation != tiledConsumer &&
        mlir::isa<mlir::TilingInterface>(&operation))
      sourceProducers.insert(&operation);

  struct PendingProducerSlice {
    mlir::tensor::ExtractSliceOp slice;
    mlir::Operation *sourceConsumer = nullptr;
    mlir::Operation *sourceProducer = nullptr;
    llvm::SmallVector<unsigned, 2> sourceConsumerOperandNumbers;
  };
  struct MaterializedCoupledProducerTile {
    mlir::OpResult producerResult;
    mlir::Block *block = nullptr;
    mlir::Type tileType;
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    mlir::Value tiledValue;
  };
  std::deque<PendingProducerSlice> worklist;
  llvm::SmallVector<PendingProducerSlice, 8> seenRelations;
  llvm::SmallVector<PendingProducerSlice, 4> pendingSlices;
  llvm::SmallVector<MaterializedCoupledProducerTile, 4>
      materializedCoupledTiles;
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
      auto equivalent = llvm::find_if(
          pendingSlices, [&](PendingProducerSlice &existing) {
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

  mlir::IRRewriter rewriter(scope.getContext());
  while (!worklist.empty()) {
    PendingProducerSlice pendingSlice = worklist.front();
    worklist.pop_front();
    mlir::tensor::ExtractSliceOp slice = pendingSlice.slice;
    auto pending = llvm::find_if(
        pendingSlices, [&](const PendingProducerSlice &existing) {
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
        mlir::Value tileEmpty =
            rewriter
                .create<mlir::tensor::EmptyOp>(
                    slice.getLoc(), tileType.getShape(),
                    tileType.getElementType())
                .getResult();
        rewriter.replaceOp(slice, tileEmpty);
        continue;
      }
    }
    auto producerResult = mlir::dyn_cast<mlir::OpResult>(slice.getSource());
    if (!producerResult ||
        producerResult.getOwner()->getBlock() != &scope.getBody() ||
        !sourceProducers.contains(producerResult.getOwner()))
      continue;
    // A destination slice generated while fusing a DPS producer may point at
    // tensor.empty.  Tiling that placeholder creates another equivalent
    // empty destination slice and can grow the worklist without making the
    // consumer tile more precise.  Empty destinations carry no dataflow to
    // fuse; leave their slice for ordinary dead-support cleanup.
    if (mlir::isa<mlir::tensor::EmptyOp>(producerResult.getOwner()))
      continue;
    if (!mlir::isa<mlir::TilingInterface>(producerResult.getOwner()))
      continue;
    if (mlir::isa<WaferLinalgExtCollectiveOpInterface>(
            producerResult.getOwner())) {
      setFailureReason(
          failureReason,
          "candidate producer fusion requires an internal logical collective "
          "to remain a separate scheduling task");
      return mlir::failure();
    }
    if (!pendingSlice.sourceConsumer ||
        pendingSlice.sourceConsumerOperandNumbers.empty())
      continue;
    std::optional<bool> shouldFuse;
    for (unsigned operandNumber :
         pendingSlice.sourceConsumerOperandNumbers) {
      if (operandNumber >= pendingSlice.sourceConsumer->getNumOperands()) {
        setFailureReason(
            failureReason,
            "tiled consumer operand identity left the source operation");
        return mlir::failure();
      }
      bool current = scope.shouldFuseCandidateConnection(
          producerResult,
          pendingSlice.sourceConsumer->getOpOperand(operandNumber));
      if (shouldFuse && *shouldFuse != current) {
        setFailureReason(
            failureReason,
            "one tiled operand slice spans incompatible connection actions");
        return mlir::failure();
      }
      shouldFuse = current;
    }
    if (!shouldFuse.value_or(false))
      continue;

    // Connection identity is per producer-result/consumer-operand edge, but
    // physical tile identity is per exact producer-result demand.  Two
    // independently accepted Coupled edges that request the same slice in the
    // same block must consume one producer tile/version.  Retaining the edge
    // identities through policy evaluation prevents a Coupled edge from
    // absorbing a Separated edge, while this post-policy cache avoids cloning
    // the same immutable SSA producer once per equivalent operand use.
    auto reusable = llvm::find_if(
        materializedCoupledTiles,
        [&](const MaterializedCoupledProducerTile &materialized) {
          mlir::Operation *tiledOwner =
              materialized.tiledValue.getDefiningOp();
          return materialized.producerResult == producerResult &&
                 materialized.block == slice->getBlock() && tiledOwner &&
                 tiledOwner->getBlock() == slice->getBlock() &&
                 tiledOwner->isBeforeInBlock(slice) &&
                 materialized.tileType == slice.getType() &&
                 llvm::equal(materialized.offsets,
                             slice.getMixedOffsets()) &&
                 llvm::equal(materialized.sizes, slice.getMixedSizes()) &&
                 llvm::equal(materialized.strides,
                             slice.getMixedStrides());
        });
    if (reusable != materializedCoupledTiles.end()) {
      slice.getResult().replaceAllUsesWith(reusable->tiledValue);
      if (slice->use_empty())
        rewriter.eraseOp(slice);
      continue;
    }

    std::optional<mlir::scf::SCFFuseProducerOfSliceResult> fused =
        mlir::scf::tileAndFuseProducerOfSlice(rewriter, slice, loops);
    if (!fused) {
      setFailureReason(failureReason, "candidate producer tile fusion failed");
      return mlir::failure();
    }
    if (mlir::failed(rebaseFusedDPSInit(*fused, failureReason)))
      return mlir::failure();
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

void eraseDeadCandidateSupportClosure(TensorProgramScope scope) {
  // Tiled producer fusion leaves the original untiled producer and its
  // transitive pure support dead. Use the operation effect contract rather
  // than an op-name allowlist so arbitrary shaped views and held-out
  // side-effect-free structured producers disappear with that closure, while
  // unknown or observable effects remain explicit.
  llvm::SmallVector<mlir::Operation *, 8> operations;
  for (mlir::Operation &op : scope.getBody().without_terminator())
    operations.push_back(&op);
  for (mlir::Operation *op : llvm::reverse(operations)) {
    if (mlir::isOpTriviallyDead(op))
      op->erase();
  }
}

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    TensorProgramScope scope, mlir::Operation *root, unsigned outputIndex,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  if (root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires one result");
    return mlir::failure();
  }
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType) {
    setFailureReason(failureReason,
                     "candidate tile materialization result is not ranked");
    return mlir::failure();
  }
  if (mlir::failed(validateCandidateTile(resultType, candidateTileOffsets,
                                         candidateTileSizes, failureReason)))
    return mlir::failure();

  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  mixedOffsets.reserve(candidateTileOffsets.size());
  for (int64_t offset : candidateTileOffsets)
    mixedOffsets.push_back(builder.getIndexAttr(offset));
  llvm::SmallVector<mlir::LoopLikeOpInterface, 0> loops;
  return materializeCandidateRootTileValue(
      builder, scope, root, outputIndex, mixedOffsets, candidateTileSizes,
      candidateReductionTileSizes, loops, failureReason);
}

mlir::FailureOr<mlir::Value> materializeCandidateOperandConsumerTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> operandTileOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> operandTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  if (!root || root->getNumResults() != 1 ||
      !mlir::isa<mlir::TilingInterface>(root)) {
    setFailureReason(
        failureReason,
        "operand-driven candidate requires one tiled consumer result");
    return mlir::failure();
  }

  mlir::FailureOr<OperandTileMaterialization> materialized =
      materializeConsumerFromOperandTile(root, builder, operandNumber,
                                         operandTileOffsets, operandTileSizes,
                                         failureReason);
  if (mlir::failed(materialized))
    return mlir::failure();
  if (materialized->tiledOperations.size() != 1 ||
      materialized->tiledValues.size() != 1 ||
      materialized->tiledOperations.front()->getNumResults() != 1 ||
      materialized->tiledOperations.front()->getResult(0) !=
          materialized->tiledValues.front()) {
    setFailureReason(
        failureReason,
        "operand-driven candidate must materialize one consumer result");
    return mlir::failure();
  }

  auto tiling = mlir::cast<mlir::TilingInterface>(root);
  llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
  llvm::SmallVector<mlir::OpFoldResult> resultSizes;
  if (mlir::failed(tiling.getResultTilePosition(
          builder, /*resultNumber=*/0, materialized->iterationDomain.offsets,
          materialized->iterationDomain.sizes, resultOffsets, resultSizes))) {
    setFailureReason(
        failureReason,
        "operand-driven candidate cannot map its consumer result tile");
    return mlir::failure();
  }

  // A boundary-seeded complete traversal may carry a single output only when
  // every seed tile maps one-to-one onto the same result tile. More general
  // broadcast/permutation cover needs an independent exact-cover proof; it is
  // deliberately rejected here instead of assuming non-overlap.
  if (!llvm::equal(resultOffsets, operandTileOffsets) ||
      !llvm::equal(resultSizes, operandTileSizes)) {
    setFailureReason(
        failureReason,
        "operand-driven candidate requires an exact one-to-one result tile "
        "relation");
    return mlir::failure();
  }

  mlir::Operation *tiledConsumer = materialized->tiledOperations.front();
  if (mlir::failed(fuseCandidateProducerSlices(tiledConsumer, root, scope,
                                               loops, failureReason)))
    return mlir::failure();
  builder.setInsertionPointAfter(tiledConsumer);
  return materialized->tiledValues.front();
}

mlir::FailureOr<mlir::linalg::LinalgOp>
getCandidatePartialReductionComputeRoot(mlir::Operation *root,
                                        std::string *failureReason) {
  if (auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(root))
    return linalg;

  auto allReduce = mlir::dyn_cast_or_null<LinalgExtCollectiveAllReduceOp>(root);
  if (!allReduce || allReduce.getInputs().size() != 1 ||
      allReduce.getOuts().size() != 1 || root->getNumResults() != 1) {
    setFailureReason(
        failureReason,
        "partial-reduction traversal requires a direct reduction or one "
        "typed all-reduce wrapper");
    return mlir::failure();
  }

  mlir::Value input = allReduce.getInputs().front();
  auto producer =
      mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(input.getDefiningOp());
  if (!producer || producer->getBlock() != root->getBlock() ||
      producer->getNumResults() != 1 || producer->getResult(0) != input ||
      !input.hasOneUse()) {
    setFailureReason(
        failureReason,
        "partial-reduction typed all-reduce requires one exact single-use "
        "same-block Linalg producer");
    return mlir::failure();
  }
  return producer;
}

mlir::FailureOr<mlir::Value> materializeCandidatePartialReductionRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  mlir::FailureOr<mlir::linalg::LinalgOp> computeRoot =
      getCandidatePartialReductionComputeRoot(root, failureReason);
  if (mlir::failed(computeRoot))
    return mlir::failure();
  mlir::linalg::LinalgOp linalg = *computeRoot;
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(linalg.getOperation());
  auto partial =
      mlir::dyn_cast<mlir::PartialReductionOpInterface>(linalg.getOperation());
  if (!linalg || !tiling || !partial || linalg.getNumDpsInits() != 1 ||
      linalg->getNumResults() != 1) {
    setFailureReason(
        failureReason,
        "partial-reduction candidate requires one interface-backed result");
    return mlir::failure();
  }
  if (mlir::failed(
          verifyCandidateReductionSplitNumericLegality(linalg, failureReason)))
    return mlir::failure();

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(linalg->getResult(0).getType());
  if (!resultType || candidateTileOffsets.size() != candidateTileSizes.size() ||
      candidateTileSizes.size() != static_cast<size_t>(resultType.getRank()) ||
      llvm::any_of(candidateTileSizes,
                   [](int64_t size) { return size <= 0; })) {
    setFailureReason(failureReason,
                     "partial-reduction candidate tile rank or size mismatch");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 2> reductionLoopDims =
      getReductionLoopDims(linalg);
  llvm::SmallVector<int64_t, 2> effectiveReductionTileSizes(
      candidateReductionTileSizes.begin(), candidateReductionTileSizes.end());
  if (effectiveReductionTileSizes.empty()) {
    llvm::SmallVector<int64_t, 4> loopRanges = linalg.getStaticLoopRanges();
    for (unsigned dim : reductionLoopDims) {
      if (dim >= loopRanges.size() ||
          mlir::ShapedType::isDynamic(loopRanges[dim])) {
        setFailureReason(
            failureReason,
            "partial-reduction candidate requires static reduction ranges");
        return mlir::failure();
      }
      effectiveReductionTileSizes.push_back(loopRanges[dim]);
    }
  }

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      linalg.getIndexingMapsArray();
  unsigned outputMapIndex = static_cast<unsigned>(linalg.getNumDpsInputs());
  if (outputMapIndex >= indexingMaps.size()) {
    setFailureReason(failureReason,
                     "partial-reduction candidate is missing its output map");
    return mlir::failure();
  }
  mlir::FailureOr<llvm::SmallVector<ReductionChunk, 8>> reductionChunks =
      buildReductionChunks(linalg, effectiveReductionTileSizes, failureReason);
  if (mlir::failed(reductionChunks))
    return mlir::failure();

  mlir::Value accumulator;
  for (const ReductionChunk &chunk : *reductionChunks) {
    CandidateLoopTile loopTile;
    if (mlir::failed(buildCandidateLoopTile(
            builder, linalg->getLoc(), linalg, indexingMaps[outputMapIndex],
            candidateTileOffsets, candidateTileSizes, chunk.offsets,
            chunk.sizes, loopTile, failureReason)))
      return mlir::failure();

    mlir::FailureOr<PartialReductionTileMaterialization> materialized =
        accumulator ? materializePartialReductionTile(
                          linalg.getOperation(), builder, loopTile.loopOffsets,
                          loopTile.tileSizes, mlir::ValueRange(accumulator),
                          failureReason)
                    : materializePartialReductionTile(
                          linalg.getOperation(), builder, loopTile.loopOffsets,
                          loopTile.tileSizes, failureReason);
    if (mlir::failed(materialized))
      return mlir::failure();
    if (materialized->partialOperations.size() != 1 ||
        materialized->mergedValues.size() != 1 ||
        materialized->mergeOperations.size() != 1) {
      setFailureReason(
          failureReason,
          "partial-reduction candidate requires one partial and merge result");
      return mlir::failure();
    }

    for (mlir::Operation *operation : materialized->partialOperations)
      if (mlir::failed(fuseCandidateProducerSlices(
              operation, linalg.getOperation(), scope, loops, failureReason)))
        return mlir::failure();
    for (mlir::Operation *operation : materialized->mergeOperations)
      if (mlir::failed(fuseCandidateProducerSlices(
              operation, linalg.getOperation(), scope, loops, failureReason)))
        return mlir::failure();

    builder.setInsertionPointAfter(materialized->mergeOperations.back());
    accumulator = materialized->mergedValues.front();
  }

  if (!accumulator) {
    setFailureReason(
        failureReason,
        "partial-reduction candidate produced no merged result tile");
    return mlir::failure();
  }

  // A local reduction alone remains local. Cross-rank protocol is retained
  // only when the source already yielded a typed all-reduce. Tile that exact
  // collective and replace its generated input slice with the local
  // PartialReductionOpInterface result so the accepted IR has one real SSA
  // chain: local partial/merge -> typed collective -> output tile.
  auto allReduce = mlir::dyn_cast<LinalgExtCollectiveAllReduceOp>(root);
  if (!allReduce)
    return accumulator;

  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  mixedSizes.reserve(candidateTileSizes.size());
  for (int64_t size : candidateTileSizes)
    mixedSizes.push_back(builder.getIndexAttr(size));
  auto collectiveTiling = mlir::cast<mlir::TilingInterface>(root);
  mlir::FailureOr<mlir::TilingResult> tiled =
      collectiveTiling.getTiledImplementation(builder, candidateTileOffsets,
                                              mixedSizes);
  if (mlir::failed(tiled) || tiled->tiledOps.size() != 1 ||
      tiled->tiledValues.size() != 1) {
    setFailureReason(
        failureReason,
        "typed all-reduce rejected the partial-reduction result tile");
    return mlir::failure();
  }
  auto tiledAllReduce =
      mlir::dyn_cast<LinalgExtCollectiveAllReduceOp>(tiled->tiledOps.front());
  if (!tiledAllReduce || tiledAllReduce.getInputs().size() != 1 ||
      tiledAllReduce.getOuts().size() != 1 ||
      tiledAllReduce->getNumResults() != 1 ||
      tiledAllReduce.getInputs().front().getType() != accumulator.getType()) {
    setFailureReason(
        failureReason,
        "typed all-reduce tile does not match the local partial result");
    return mlir::failure();
  }

  mlir::Value unusedInputSlice = tiledAllReduce.getInputs().front();
  tiledAllReduce->setOperand(/*input=*/0, accumulator);
  if (mlir::Operation *slice = unusedInputSlice.getDefiningOp();
      slice && slice->use_empty() &&
      mlir::isa<mlir::tensor::ExtractSliceOp>(slice))
    slice->erase();
  builder.setInsertionPointAfter(tiledAllReduce);
  return tiledAllReduce.getResult(0);
}

static mlir::FailureOr<mlir::Value> materializeCandidateInterfaceRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned outputIndex,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(root);
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(root);
  if (!dps || !tiling || dps.getNumDpsInits() != 1 ||
      root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "candidate interface root requires one DPS output and "
                     "TilingInterface");
    return mlir::failure();
  }
  if (!candidateReductionTileSizes.empty()) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires PartialReductionOpInterface");
    return mlir::failure();
  }
  (void)scope;
  (void)outputIndex;

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType || candidateTileOffsets.size() != candidateTileSizes.size() ||
      candidateTileSizes.size() != static_cast<size_t>(resultType.getRank()) ||
      llvm::any_of(candidateTileSizes,
                   [](int64_t size) { return size <= 0; })) {
    setFailureReason(failureReason,
                     "candidate interface tile rank or size mismatch");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  mixedSizes.reserve(candidateTileSizes.size());
  for (int64_t size : candidateTileSizes)
    mixedSizes.push_back(builder.getIndexAttr(size));

  // Candidate coordinates are expressed in the result domain. Recover an
  // exact iteration-domain tile through TilingInterface result/operand
  // relations and verify the round trip before materialization. Passing
  // result offsets directly to getTiledImplementation without that proof
  // would silently assume an identity indexing map.
  std::optional<OperandTileIterationDomain> iteration;
  auto hasExactResultTile = [&](llvm::ArrayRef<mlir::OpFoldResult> offsets,
                                llvm::ArrayRef<mlir::OpFoldResult> sizes) {
    llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
    llvm::SmallVector<mlir::OpFoldResult> resultSizes;
    return mlir::succeeded(tiling.getResultTilePosition(
               builder, /*resultNumber=*/0, offsets, sizes, resultOffsets,
               resultSizes)) &&
           llvm::equal(resultOffsets, candidateTileOffsets) &&
           llvm::equal(resultSizes, mixedSizes);
  };
  auto retainIteration = [&](llvm::ArrayRef<mlir::OpFoldResult> offsets,
                             llvm::ArrayRef<mlir::OpFoldResult> sizes) {
    if (!hasExactResultTile(offsets, sizes))
      return false;
    iteration.emplace();
    iteration->offsets.assign(offsets.begin(), offsets.end());
    iteration->sizes.assign(sizes.begin(), sizes.end());
    return true;
  };

  // Prefer the interface's explicit result-to-iteration relation. The
  // verified round-trip fallback admits identity iteration/result domains
  // exposed by simpler TilingInterface implementations without assuming that
  // every equal-rank operation is identity-mapped.
  llvm::SmallVector<mlir::OpFoldResult> resultIterationOffsets;
  llvm::SmallVector<mlir::OpFoldResult> resultIterationSizes;
  if (mlir::succeeded(tiling.getIterationDomainTileFromResultTile(
          builder, /*resultNumber=*/0, candidateTileOffsets, mixedSizes,
          resultIterationOffsets, resultIterationSizes)))
    (void)retainIteration(resultIterationOffsets, resultIterationSizes);
  if (!iteration && candidateTileOffsets.size() ==
                        tiling.getLoopIteratorTypes().size())
    (void)retainIteration(candidateTileOffsets, mixedSizes);

  llvm::SmallVector<unsigned, 4> relationOperands;
  if (mlir::OpOperand *resultDestination = dps.getDpsInitOperand(0))
    relationOperands.push_back(resultDestination->getOperandNumber());
  for (unsigned operandNumber = 0; operandNumber < root->getNumOperands();
       ++operandNumber)
    if (!llvm::is_contained(relationOperands, operandNumber))
      relationOperands.push_back(operandNumber);
  for (unsigned operandNumber : relationOperands) {
    if (iteration)
      break;
    std::string relationFailure;
    mlir::FailureOr<OperandTileIterationDomain> candidateIteration =
        mapOperandTileToIterationDomain(
            root, builder, operandNumber, candidateTileOffsets, mixedSizes,
            &relationFailure);
    if (mlir::failed(candidateIteration))
      continue;
    (void)retainIteration(candidateIteration->offsets,
                          candidateIteration->sizes);
  }
  if (!iteration) {
    setFailureReason(
        failureReason,
        "TilingInterface provides no operand relation that exactly covers "
        "the requested result tile");
    return mlir::failure();
  }

  mlir::FailureOr<mlir::TilingResult> tiled =
      tiling.getTiledImplementation(builder, iteration->offsets,
                                    iteration->sizes);
  if (mlir::failed(tiled)) {
    setFailureReason(failureReason,
                     "candidate interface root rejected the requested tile");
    return mlir::failure();
  }
  if (tiled->tiledOps.size() != 1 || tiled->tiledValues.size() != 1 ||
      tiled->tiledOps.front()->getNumResults() != 1 ||
      tiled->tiledOps.front()->getResult(0) != tiled->tiledValues.front()) {
    setFailureReason(
        failureReason,
        "candidate interface root must materialize one tiled op and result");
    return mlir::failure();
  }

  mlir::Operation *tiledRoot = tiled->tiledOps.front();
  if (mlir::failed(fuseCandidateProducerSlices(tiledRoot, root, scope, loops,
                                               failureReason)))
    return mlir::failure();
  builder.setInsertionPointAfter(tiledRoot);
  return tiled->tiledValues.front();
}

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned outputIndex,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  if (classifyCandidateTraversalRoot(root) !=
      CandidateTraversalRootCapability::Tiled) {
    setFailureReason(failureReason,
                     "candidate traversal root does not support tiling");
    return mlir::failure();
  }
  if (!candidateReductionTileSizes.empty())
    return materializeCandidatePartialReductionRootTileValue(
        builder, scope, root, candidateTileOffsets, candidateTileSizes,
        candidateReductionTileSizes, loops, failureReason);
  return materializeCandidateInterfaceRootTileValue(
      builder, scope, root, outputIndex, candidateTileOffsets,
      candidateTileSizes, candidateReductionTileSizes, loops, failureReason);
}

mlir::FailureOr<mlir::Value>
getCandidateOutputBoundary(TensorProgramScope scope, unsigned outputIndex,
                           std::string *failureReason) {
  unsigned inputCount = scope.getInputCount();
  mlir::Block &body = scope.getBody();
  if (inputCount + outputIndex >= body.getNumArguments()) {
    setFailureReason(failureReason,
                     "candidate tile materialization missing output boundary");
    return mlir::failure();
  }
  return body.getArgument(inputCount + outputIndex);
}

mlir::Value
insertCandidateRootTile(mlir::Operation *root, mlir::Value tileValue,
                        mlir::Value outputDestination,
                        llvm::ArrayRef<int64_t> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes) {
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  offsets.reserve(candidateTileOffsets.size());
  for (int64_t offset : candidateTileOffsets)
    offsets.push_back(builder.getIndexAttr(offset));
  return insertCandidateRootTile(builder, root->getLoc(), tileValue,
                                 outputDestination, offsets,
                                 candidateTileSizes);
}

mlir::Value
insertCandidateRootTile(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value tileValue, mlir::Value outputDestination,
                        llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes) {
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  sizes.reserve(candidateTileSizes.size());
  strides.reserve(candidateTileSizes.size());
  for (int64_t size : candidateTileSizes) {
    sizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }

  auto inserted = builder.create<mlir::tensor::InsertSliceOp>(
      loc, tileValue, outputDestination, candidateTileOffsets, sizes, strides);
  return inserted.getResult();
}

mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>>
collectCandidateRoots(TensorProgramScope scope, bool rejectProducerChains,
                      std::string *failureReason) {
  auto returnOp =
      mlir::dyn_cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  if (!returnOp || returnOp.getOperands().empty()) {
    setFailureReason(
        failureReason,
        "candidate tile materialization requires function results");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Operation *, 4> roots;
  llvm::DenseSet<mlir::Operation *> seenRoots;
  for (auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
    mlir::Operation *rootOperation = value.getDefiningOp();
    CandidateTraversalRootCapability capability =
        classifyCandidateTraversalRoot(rootOperation);
    if (capability == CandidateTraversalRootCapability::FullTraversalOnly) {
      setFailureReason(failureReason,
                       "candidate traversal root supports full traversal only");
      return mlir::failure();
    }
    if (capability == CandidateTraversalRootCapability::Unsupported) {
      setFailureReason(failureReason,
                       "candidate traversal root is unsupported");
      return mlir::failure();
    }
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(rootOperation);
    if (rootOperation->getBlock() != &scope.getBody() ||
        !seenRoots.insert(rootOperation).second ||
        rootOperation->getNumResults() != 1 || !dps ||
        dps.getNumDpsInits() != 1 || rootOperation->getResult(0) != value) {
      setFailureReason(failureReason,
                       "candidate multi-output coverage requires distinct "
                       "single-result yielded roots");
      return mlir::failure();
    }

    unsigned matchingYieldUses = 0;
    for (mlir::OpOperand &use : value.getUses()) {
      if (use.getOwner() == returnOp.getOperation() &&
          use.getOperandNumber() == index) {
        ++matchingYieldUses;
        continue;
      }
      setFailureReason(failureReason, "candidate multi-output coverage "
                                      "requires independent yielded roots");
      return mlir::failure();
    }
    if (matchingYieldUses != 1) {
      setFailureReason(failureReason, "candidate multi-output coverage "
                                      "requires independent yielded roots");
      return mlir::failure();
    }

    if (rejectProducerChains) {
      for (mlir::Value input : dps.getDpsInputs()) {
        if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
          continue;
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(input);
        if (blockArg && blockArg.getOwner() == &scope.getBody())
          continue;
        // Static tensor reshapes are shape-only views.  Keeping them in the
        // The scheduling scope does not introduce an independently tiled
        // producer: every
        // root tile still slices the same canonical linear element sequence,
        // and the normal tile-region reshape lowering decides whether the
        // physical layout can alias or must be materialized.  Other producer
        // chains remain rejected because complete traversal does not yet
        // carry their intermediate tensors across output tiles.
        auto isStaticShapeOnlyViewChain = [&](mlir::Value value) {
          mlir::Value current = value;
          while (mlir::Operation *def = current.getDefiningOp()) {
            if (def->getBlock() != &scope.getBody())
              return false;
            if (auto expand =
                    mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(def)) {
              current = expand.getSrc();
              continue;
            }
            if (auto collapse =
                    mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(def)) {
              current = collapse.getSrc();
              continue;
            }
            return false;
          }
          auto sourceArg = mlir::dyn_cast<mlir::BlockArgument>(current);
          return sourceArg && sourceArg.getOwner() == &scope.getBody();
        };
        if (isStaticShapeOnlyViewChain(input))
          continue;
        setFailureReason(
            failureReason,
            "complete candidate traversal does not support tensor producer "
            "chains");
        return mlir::failure();
      }

      mlir::Value init = dps.getDpsInits().front();
      bool hasDirectOutputInit = isTensorProgramOutputBoundary(
          scope, init, static_cast<unsigned>(index));
      auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(rootOperation);
      bool hasReduction = linalg && !getReductionLoopDims(linalg).empty();
      if (!hasDirectOutputInit &&
          (!hasReduction || !init.getDefiningOp<mlir::linalg::FillOp>())) {
        setFailureReason(
            failureReason,
            "complete candidate traversal does not support output producer "
            "chains");
        return mlir::failure();
      }
    }
    roots.push_back(rootOperation);
  }

  return roots;
}

mlir::LogicalResult materializeCandidateTileSlices(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> roots =
      collectCandidateRoots(scope, /*rejectProducerChains=*/false,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();

  if (!candidateReductionTileSizes.empty() &&
      llvm::none_of(*roots, [](mlir::Operation *root) {
        auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(root);
        return linalg && !getReductionLoopDims(linalg).empty();
      })) {
    setFailureReason(failureReason,
                     "candidate reduction split requires a reduction root");
    return mlir::failure();
  }

  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());

  llvm::SmallVector<mlir::Value, 4> insertedValues;
  for (auto [index, root] : llvm::enumerate(*roots)) {
    mlir::FailureOr<mlir::Value> tileValue = materializeCandidateRootTileValue(
        scope, root, static_cast<unsigned>(index), candidateTileOffsets,
        candidateTileSizes, candidateReductionTileSizes, failureReason);
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        scope, static_cast<unsigned>(index), failureReason);
    if (mlir::failed(tileValue) || mlir::failed(outputBoundary))
      return mlir::failure();
    insertedValues.push_back(
        insertCandidateRootTile(root, *tileValue, *outputBoundary,
                                candidateTileOffsets, candidateTileSizes));
  }

  for (auto [index, inserted] : llvm::enumerate(insertedValues))
    returnOp->setOperand(index, inserted);
  for (mlir::Operation *root : *roots)
    root->erase();
  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
