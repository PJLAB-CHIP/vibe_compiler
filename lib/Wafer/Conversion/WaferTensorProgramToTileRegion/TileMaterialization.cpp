//===- TileMaterialization.cpp - Candidate root tile materialization -===//

#include "Internal.h"

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <deque>
#include <functional>
#include <limits>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

static void inheritStructuredOperationNodes(
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

static bool isCandidateOutputDestination(TensorProgramScope scope,
                                         mlir::Value value,
                                         unsigned outputIndex) {
  llvm::DenseSet<mlir::Value> visited;
  mlir::Value current = value;
  while (current && visited.insert(current).second) {
    if (isTensorProgramOutputBoundary(scope, current, outputIndex))
      return true;
    // A structured shard traversal may end at an internal same-region
    // physical encoding. tensor.empty is the typed destination for that
    // encoding; bufferization later materializes the corresponding SPM root in
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

static mlir::FailureOr<llvm::SmallVector<int64_t, 2>>
getConfiguredReductionTileSizes(
    mlir::linalg::LinalgOp operation,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason) {
  auto selected = llvm::find_if(
      operationTemporalTiles, [&](const StructuredOpTemporalTile &tile) {
        return tile.operation == operation.getOperation();
      });
  llvm::SmallVector<unsigned, 2> reductionDims =
      getReductionLoopDims(operation);
  llvm::SmallVector<int64_t, 2> reductionTiles;
  if (selected == operationTemporalTiles.end())
    return reductionTiles;

  llvm::SmallVector<int64_t, 4> loopRanges = operation.getStaticLoopRanges();
  if (selected->iteratorTileSizes.size() != loopRanges.size() ||
      llvm::any_of(loopRanges, [](int64_t extent) {
        return mlir::ShapedType::isDynamic(extent) || extent <= 0;
      })) {
    setFailureReason(
        failureReason,
        "structured temporal tile requires a static iteration domain");
    return mlir::failure();
  }
  for (auto [tileSize, extent] :
       llvm::zip_equal(selected->iteratorTileSizes, loopRanges)) {
    if (tileSize <= 0 || tileSize > extent) {
      setFailureReason(failureReason,
                       "structured temporal tile is outside iteration domain");
      return mlir::failure();
    }
  }
  for (unsigned dimension : reductionDims)
    reductionTiles.push_back(selected->iteratorTileSizes[dimension]);
  return reductionTiles;
}

static mlir::FailureOr<mlir::Value> materializeConfiguredStructuredLeaf(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    mlir::linalg::LinalgOp sourceReduction,
    llvm::ArrayRef<mlir::OpFoldResult> outputOffsets,
    llvm::ArrayRef<int64_t> outputSizes,
    llvm::ArrayRef<mlir::OpFoldResult> reductionOffsets,
    llvm::ArrayRef<int64_t> reductionSizes, mlir::Value accumulator,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  unsigned outputMapIndex =
      static_cast<unsigned>(sourceReduction.getNumDpsInputs());
  llvm::SmallVector<mlir::AffineMap, 4> maps =
      sourceReduction.getIndexingMapsArray();
  if (outputMapIndex >= maps.size()) {
    setFailureReason(failureReason,
                     "reduction temporal tile is missing its output relation");
    return mlir::failure();
  }

  CandidateLoopTile loopTile;
  if (mlir::failed(buildCandidateLoopTile(
          builder, sourceReduction->getLoc(), sourceReduction,
          maps[outputMapIndex], outputOffsets, outputSizes, reductionOffsets,
          reductionSizes, loopTile, failureReason)))
    return mlir::failure();

  auto tiling =
      mlir::cast<mlir::TilingInterface>(sourceReduction.getOperation());
  mlir::FailureOr<mlir::TilingResult> tiled = tiling.getTiledImplementation(
      builder, loopTile.loopOffsets, loopTile.tileSizes);
  if (mlir::failed(tiled) || tiled->tiledOps.size() != 1 ||
      tiled->tiledValues.size() != 1 ||
      tiled->tiledOps.front()->getNumResults() != 1 ||
      tiled->tiledOps.front()->getResult(0) != tiled->tiledValues.front()) {
    setFailureReason(
        failureReason,
        "TilingInterface rejected the configured reduction iterator tile");
    return mlir::failure();
  }

  mlir::Operation *tiledOperation = tiled->tiledOps.front();
  inheritStructuredOperationNodes(sourceReduction.getOperation(),
                                  tiledOperation, operationNodes);
  if (accumulator) {
    auto dps =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(tiledOperation);
    if (!dps || dps.getNumDpsInits() != 1 ||
        dps.getDpsInits().front().getType() != accumulator.getType()) {
      setFailureReason(
          failureReason,
          "reduction temporal accumulator does not match its result tile");
      return mlir::failure();
    }
    mlir::Value replacedInit = dps.getDpsInits().front();
    dps.getDpsInitOperand(0)->set(accumulator);
    if (mlir::Operation *definition = replacedInit.getDefiningOp();
        definition && definition->use_empty() &&
        mlir::isOpTriviallyDead(definition))
      definition->erase();
  }

  llvm::SmallVector<mlir::LoopLikeOpInterface, 4> enclosingLoops(loops.begin(),
                                                                 loops.end());
  if (mlir::failed(fuseCandidateProducerSlices(
          tiledOperation, sourceReduction.getOperation(), scope, enclosingLoops,
          operationTemporalTiles, builder.getListener(), failureReason,
          operationNodes)))
    return mlir::failure();
  builder.setInsertionPointAfter(tiledOperation);
  return tiled->tiledValues.front();
}

/// Materializes a compact sequential accumulator traversal for every
/// configured reduction iterator.  Each split dimension has an explicit
/// prologue, a steady scf.for carrying the typed result tile, and an exact
/// tail.  Nesting those compact traversals enumerates the reduction Cartesian
/// product without host-side chunk expansion.  Every leaf enters ordinary
/// producer fusion, so operand windows come from the leaf's actual indexing
/// relation.
static mlir::FailureOr<mlir::Value> materializeConfiguredReductionProducer(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    mlir::linalg::LinalgOp sourceReduction,
    llvm::ArrayRef<mlir::OpFoldResult> requestedOutputOffsets,
    llvm::ArrayRef<int64_t> requestedOutputSizes,
    llvm::ArrayRef<int64_t> reductionTileSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  llvm::SmallVector<unsigned, 2> reductionDims =
      getReductionLoopDims(sourceReduction);
  llvm::SmallVector<int64_t, 4> loopRanges =
      sourceReduction.getStaticLoopRanges();
  if (reductionDims.size() != reductionTileSizes.size()) {
    setFailureReason(failureReason, "reduction temporal tile rank mismatch");
    return mlir::failure();
  }
  auto selected = llvm::find_if(
      operationTemporalTiles, [&](const StructuredOpTemporalTile &tile) {
        return tile.operation == sourceReduction.getOperation();
      });
  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      sourceReduction.getIndexingMapsArray();
  const unsigned outputMapIndex =
      static_cast<unsigned>(sourceReduction.getNumDpsInputs());
  if (selected == operationTemporalTiles.end() ||
      outputMapIndex >= indexingMaps.size() ||
      requestedOutputSizes.size() !=
          indexingMaps[outputMapIndex].getNumResults()) {
    setFailureReason(
        failureReason,
        "reduction temporal traversal lacks its exact output relation");
    return mlir::failure();
  }
  for (auto [resultDimension, expression] :
       llvm::enumerate(indexingMaps[outputMapIndex].getResults())) {
    auto loopDimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!loopDimension ||
        loopDimension.getPosition() >= selected->iteratorTileSizes.size()) {
      setFailureReason(
          failureReason,
          "reduction output relation is not a projected iterator domain");
      return mlir::failure();
    }
    if (requestedOutputSizes[resultDimension] >
        selected->iteratorTileSizes[loopDimension.getPosition()]) {
      setFailureReason(
          failureReason,
          "configured parallel iterator tile requires an explicit output "
          "sub-traversal");
      return mlir::failure();
    }
  }
  llvm::SmallVector<unsigned, 2> splitOrdinals;
  for (auto [ordinal, dimension] : llvm::enumerate(reductionDims)) {
    if (dimension >= loopRanges.size() || reductionTileSizes[ordinal] <= 0 ||
        reductionTileSizes[ordinal] > loopRanges[dimension]) {
      setFailureReason(failureReason,
                       "reduction temporal tile is outside loop bounds");
      return mlir::failure();
    }
    if (reductionTileSizes[ordinal] == loopRanges[dimension])
      continue;
    splitOrdinals.push_back(static_cast<unsigned>(ordinal));
  }
  if (splitOrdinals.empty()) {
    setFailureReason(failureReason,
                     "configured reduction producer has no temporal split");
    return mlir::failure();
  }

  // The traversal order is [chunk coordinates..., in-tile coordinates...].
  // It is identical to the source reduction's lexicographic order only when
  // every reduction axis preceding the last split axis has a unit in-tile
  // extent. A sole split of the first reduction axis is therefore exact; a
  // rectangular split of two non-unit axes requires source-authorized
  // floating-point reassociation.
  bool preservesSequentialReductionOrder = true;
  const unsigned lastSplitOrdinal = splitOrdinals.back();
  for (unsigned ordinal = 0; ordinal < lastSplitOrdinal; ++ordinal) {
    const int64_t range = loopRanges[reductionDims[ordinal]];
    if (range > 1 && reductionTileSizes[ordinal] != 1) {
      preservesSequentialReductionOrder = false;
      break;
    }
  }
  if (mlir::failed(verifyReductionSplitNumericLegality(
          sourceReduction, preservesSequentialReductionOrder, failureReason)))
    return mlir::failure();

  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      sourceReduction->getResult(0).getType());
  if (!resultType ||
      requestedOutputOffsets.size() !=
          static_cast<size_t>(resultType.getRank()) ||
      requestedOutputSizes.size() !=
          static_cast<size_t>(resultType.getRank()) ||
      llvm::any_of(requestedOutputSizes,
                   [](int64_t size) { return size <= 0; })) {
    setFailureReason(
        failureReason,
        "reduction temporal result window has an invalid static domain");
    return mlir::failure();
  }
  llvm::SmallVector<mlir::OpFoldResult, 2> reductionOffsets(
      reductionDims.size(), builder.getIndexAttr(0));
  llvm::SmallVector<int64_t, 2> chunkSizes(reductionTileSizes.begin(),
                                           reductionTileSizes.end());
  using Traversal = std::function<mlir::FailureOr<mlir::Value>(
      mlir::OpBuilder &, unsigned, mlir::Value,
      llvm::ArrayRef<mlir::LoopLikeOpInterface>)>;
  Traversal traverse =
      [&](mlir::OpBuilder &nestedBuilder, unsigned depth,
          mlir::Value accumulator,
          llvm::ArrayRef<mlir::LoopLikeOpInterface> enclosingLoops)
      -> mlir::FailureOr<mlir::Value> {
    if (depth == splitOrdinals.size())
      return materializeConfiguredStructuredLeaf(
          nestedBuilder, scope, sourceReduction, requestedOutputOffsets,
          requestedOutputSizes, reductionOffsets, chunkSizes, accumulator,
          enclosingLoops, operationTemporalTiles, failureReason,
          operationNodes);

    const unsigned ordinal = splitOrdinals[depth];
    const int64_t range = loopRanges[reductionDims[ordinal]];
    const int64_t tileSize = reductionTileSizes[ordinal];
    const int64_t tailSize = range % tileSize;
    const int64_t mainSize = range - tailSize;
    mlir::Location loc = sourceReduction->getLoc();
    auto materializeAt =
        [&](mlir::OpBuilder &bodyBuilder, mlir::OpFoldResult offset,
            int64_t size, mlir::Value destination,
            llvm::ArrayRef<mlir::LoopLikeOpInterface> nestedLoops)
        -> mlir::FailureOr<mlir::Value> {
      reductionOffsets[ordinal] = offset;
      chunkSizes[ordinal] = size;
      return traverse(bodyBuilder, depth + 1, destination, nestedLoops);
    };

    mlir::FailureOr<mlir::Value> prologue =
        materializeAt(nestedBuilder, nestedBuilder.getIndexAttr(0), tileSize,
                      accumulator, enclosingLoops);
    if (mlir::failed(prologue))
      return mlir::failure();
    mlir::Value current = *prologue;

    if (mainSize > tileSize) {
      auto lower =
          nestedBuilder.create<mlir::arith::ConstantIndexOp>(loc, tileSize);
      auto upper =
          nestedBuilder.create<mlir::arith::ConstantIndexOp>(loc, mainSize);
      auto step =
          nestedBuilder.create<mlir::arith::ConstantIndexOp>(loc, tileSize);
      auto loop = nestedBuilder.create<mlir::scf::ForOp>(
          loc, lower, upper, step, mlir::ValueRange(current));
      llvm::SmallVector<mlir::LoopLikeOpInterface, 4> nestedLoops(
          enclosingLoops.begin(), enclosingLoops.end());
      nestedLoops.push_back(
          mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
      mlir::OpBuilder bodyBuilder =
          mlir::OpBuilder::atBlockBegin(loop.getBody());
      mlir::FailureOr<mlir::Value> steady =
          materializeAt(bodyBuilder, loop.getInductionVar(), tileSize,
                        loop.getRegionIterArgs().front(), nestedLoops);
      if (mlir::failed(steady))
        return mlir::failure();
      bodyBuilder.create<mlir::scf::YieldOp>(loc, *steady);
      nestedBuilder.setInsertionPointAfter(loop);
      current = loop.getResult(0);
    }

    if (tailSize > 0) {
      mlir::FailureOr<mlir::Value> tail =
          materializeAt(nestedBuilder, nestedBuilder.getIndexAttr(mainSize),
                        tailSize, current, enclosingLoops);
      if (mlir::failed(tail))
        return mlir::failure();
      current = *tail;
    }
    return current;
  };
  return traverse(builder, /*depth=*/0, /*accumulator=*/{}, loops);
}

static mlir::FailureOr<mlir::Value> materializeConfiguredComputeTile(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    mlir::linalg::LinalgOp sourceCompute,
    llvm::ArrayRef<mlir::OpFoldResult> outputOffsets,
    llvm::ArrayRef<int64_t> outputSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  mlir::FailureOr<llvm::SmallVector<int64_t, 2>> reductionTiles =
      getConfiguredReductionTileSizes(sourceCompute, operationTemporalTiles,
                                      failureReason);
  if (mlir::failed(reductionTiles))
    return mlir::failure();
  llvm::SmallVector<unsigned, 2> reductionDims =
      getReductionLoopDims(sourceCompute);
  llvm::SmallVector<int64_t, 4> loopRanges =
      sourceCompute.getStaticLoopRanges();
  bool hasReductionSplit =
      reductionTiles->size() == reductionDims.size() &&
      llvm::any_of(llvm::zip_equal(*reductionTiles, reductionDims),
                   [&](auto values) {
                     auto [tileSize, dimension] = values;
                     return dimension < loopRanges.size() &&
                            tileSize < loopRanges[dimension];
                   });
  if (hasReductionSplit)
    return materializeConfiguredReductionProducer(
        builder, scope, sourceCompute, outputOffsets, outputSizes,
        *reductionTiles, loops, operationTemporalTiles, failureReason,
        operationNodes);

  llvm::SmallVector<mlir::OpFoldResult, 2> reductionOffsets(
      reductionDims.size(), builder.getIndexAttr(0));
  llvm::SmallVector<int64_t, 2> reductionSizes;
  reductionSizes.reserve(reductionDims.size());
  for (unsigned dimension : reductionDims)
    reductionSizes.push_back(loopRanges[dimension]);
  return materializeConfiguredStructuredLeaf(
      builder, scope, sourceCompute, outputOffsets, outputSizes,
      reductionOffsets, reductionSizes, /*accumulator=*/{}, loops,
      operationTemporalTiles, failureReason, operationNodes);
}

static mlir::FailureOr<mlir::Value> materializeConfiguredCollectiveTile(
    mlir::OpBuilder &builder, mlir::Operation *root,
    mlir::linalg::LinalgOp sourceCompute,
    llvm::ArrayRef<mlir::OpFoldResult> outputOffsets,
    llvm::ArrayRef<int64_t> outputSizes, mlir::Value localValue,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (root == sourceCompute.getOperation())
    return localValue;
  auto allReduce = mlir::dyn_cast<LinalgExtCollectiveAllReduceOp>(root);
  if (!allReduce) {
    setFailureReason(
        failureReason,
        "configured structured traversal has an unsupported root wrapper");
    return mlir::failure();
  }
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  mixedSizes.reserve(outputSizes.size());
  for (int64_t size : outputSizes)
    mixedSizes.push_back(builder.getIndexAttr(size));
  auto collectiveTiling = mlir::cast<mlir::TilingInterface>(root);
  mlir::FailureOr<mlir::TilingResult> tiled =
      collectiveTiling.getTiledImplementation(builder, outputOffsets,
                                              mixedSizes);
  if (mlir::failed(tiled) || tiled->tiledOps.size() != 1 ||
      tiled->tiledValues.size() != 1) {
    setFailureReason(failureReason,
                     "typed all-reduce rejected the local result tile");
    return mlir::failure();
  }
  auto tiledAllReduce =
      mlir::dyn_cast<LinalgExtCollectiveAllReduceOp>(tiled->tiledOps.front());
  if (!tiledAllReduce || tiledAllReduce.getInputs().size() != 1 ||
      tiledAllReduce.getOuts().size() != 1 ||
      tiledAllReduce->getNumResults() != 1 ||
      tiledAllReduce.getInputs().front().getType() != localValue.getType()) {
    setFailureReason(
        failureReason,
        "typed all-reduce tile does not match the local result tile");
    return mlir::failure();
  }
  inheritStructuredOperationNodes(root, tiledAllReduce.getOperation(),
                                  operationNodes);
  mlir::Value unusedInputSlice = tiledAllReduce.getInputs().front();
  tiledAllReduce->setOperand(/*input=*/0, localValue);
  if (mlir::Operation *slice = unusedInputSlice.getDefiningOp();
      slice && slice->use_empty() &&
      mlir::isa<mlir::tensor::ExtractSliceOp>(slice))
    slice->erase();
  builder.setInsertionPointAfter(tiledAllReduce);
  return tiledAllReduce.getResult(0);
}

static mlir::FailureOr<mlir::OpFoldResult>
addConfiguredTileOffset(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::OpFoldResult base, mlir::OpFoldResult relative) {
  std::optional<int64_t> baseConstant = mlir::getConstantIntValue(base);
  std::optional<int64_t> relativeConstant = mlir::getConstantIntValue(relative);
  if (relativeConstant && *relativeConstant == 0)
    return base;
  if (baseConstant && relativeConstant) {
    const __int128 sum =
        static_cast<__int128>(*baseConstant) + *relativeConstant;
    if (sum < std::numeric_limits<int64_t>::min() ||
        sum > std::numeric_limits<int64_t>::max())
      return mlir::failure();
    return mlir::OpFoldResult(builder.getIndexAttr(static_cast<int64_t>(sum)));
  }
  mlir::Value baseValue =
      mlir::getValueOrCreateConstantIndexOp(builder, loc, base);
  mlir::Value relativeValue =
      mlir::getValueOrCreateConstantIndexOp(builder, loc, relative);
  return mlir::OpFoldResult(
      builder.create<mlir::arith::AddIOp>(loc, baseValue, relativeValue)
          .getResult());
}

static mlir::FailureOr<mlir::Value> materializeConfiguredParallelTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    mlir::linalg::LinalgOp sourceCompute,
    llvm::ArrayRef<mlir::OpFoldResult> requestedOutputOffsets,
    llvm::ArrayRef<int64_t> requestedOutputSizes,
    llvm::ArrayRef<int64_t> parallelTileSizes, unsigned dimension,
    mlir::Value output,
    llvm::ArrayRef<mlir::OpFoldResult> destinationBaseOffsets,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &sourceOffsets,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &localOffsets,
    llvm::SmallVectorImpl<int64_t> &tileSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (dimension == requestedOutputSizes.size()) {
    mlir::FailureOr<mlir::Value> local = materializeConfiguredComputeTile(
        builder, scope, sourceCompute, sourceOffsets, tileSizes, loops,
        operationTemporalTiles, failureReason, operationNodes);
    if (mlir::failed(local))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> wrapped = materializeConfiguredCollectiveTile(
        builder, root, sourceCompute, sourceOffsets, tileSizes, *local,
        failureReason, operationNodes);
    if (mlir::failed(wrapped))
      return mlir::failure();
    return insertCandidateRootTile(builder, root->getLoc(), *wrapped, output,
                                   localOffsets, tileSizes);
  }

  const int64_t range = requestedOutputSizes[dimension];
  const int64_t tileSize = std::min(range, parallelTileSizes[dimension]);
  const int64_t tailSize = range % tileSize;
  const int64_t mainSize = range - tailSize;
  mlir::Location loc = root->getLoc();
  auto materializeAt =
      [&](mlir::OpBuilder &nestedBuilder, mlir::OpFoldResult relativeOffset,
          int64_t currentSize, mlir::Value destination,
          llvm::ArrayRef<mlir::LoopLikeOpInterface> nestedLoops)
      -> mlir::FailureOr<mlir::Value> {
    mlir::FailureOr<mlir::OpFoldResult> sourceOffset = addConfiguredTileOffset(
        nestedBuilder, loc, requestedOutputOffsets[dimension], relativeOffset);
    if (mlir::failed(sourceOffset)) {
      setFailureReason(failureReason,
                       "configured temporal tile offset overflows index");
      return mlir::failure();
    }
    sourceOffsets.push_back(*sourceOffset);
    mlir::OpFoldResult destinationOffset = relativeOffset;
    if (!destinationBaseOffsets.empty()) {
      mlir::FailureOr<mlir::OpFoldResult> rebased = addConfiguredTileOffset(
          nestedBuilder, loc, destinationBaseOffsets[dimension],
          relativeOffset);
      if (mlir::failed(rebased)) {
        setFailureReason(failureReason,
                         "configured destination tile offset overflows index");
        sourceOffsets.pop_back();
        return mlir::failure();
      }
      destinationOffset = *rebased;
    }
    localOffsets.push_back(destinationOffset);
    tileSizes.push_back(currentSize);
    mlir::FailureOr<mlir::Value> result =
        materializeConfiguredParallelTraversal(
            nestedBuilder, scope, root, sourceCompute, requestedOutputOffsets,
            requestedOutputSizes, parallelTileSizes, dimension + 1, destination,
            destinationBaseOffsets, sourceOffsets, localOffsets, tileSizes,
            nestedLoops, operationTemporalTiles, failureReason, operationNodes);
    tileSizes.pop_back();
    localOffsets.pop_back();
    sourceOffsets.pop_back();
    return result;
  };

  mlir::Value currentOutput = output;
  if (tileSize == range)
    return materializeAt(builder, builder.getIndexAttr(0), range, currentOutput,
                         loops);

  mlir::FailureOr<mlir::Value> prologue = materializeAt(
      builder, builder.getIndexAttr(0), tileSize, currentOutput, loops);
  if (mlir::failed(prologue))
    return mlir::failure();
  currentOutput = *prologue;

  if (mainSize > tileSize) {
    auto lower = builder.create<mlir::arith::ConstantIndexOp>(loc, tileSize);
    auto upper = builder.create<mlir::arith::ConstantIndexOp>(loc, mainSize);
    auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, tileSize);
    auto loop = builder.create<mlir::scf::ForOp>(
        loc, lower, upper, step, mlir::ValueRange(currentOutput));
    llvm::SmallVector<mlir::LoopLikeOpInterface, 4> nestedLoops(loops.begin(),
                                                                loops.end());
    nestedLoops.push_back(
        mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
    mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
    mlir::FailureOr<mlir::Value> steady =
        materializeAt(bodyBuilder, loop.getInductionVar(), tileSize,
                      loop.getRegionIterArgs().front(), nestedLoops);
    if (mlir::failed(steady))
      return mlir::failure();
    bodyBuilder.create<mlir::scf::YieldOp>(loc, *steady);
    builder.setInsertionPointAfter(loop);
    currentOutput = loop.getResult(0);
  }

  if (tailSize > 0) {
    mlir::FailureOr<mlir::Value> tail =
        materializeAt(builder, builder.getIndexAttr(mainSize), tailSize,
                      currentOutput, loops);
    if (mlir::failed(tail))
      return mlir::failure();
    currentOutput = *tail;
  }
  return currentOutput;
}

static mlir::FailureOr<mlir::Value> materializeConfiguredStructuredTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    mlir::linalg::LinalgOp sourceCompute,
    llvm::ArrayRef<mlir::OpFoldResult> requestedOutputOffsets,
    llvm::ArrayRef<int64_t> requestedOutputSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason, mlir::Value outputDestination = {},
    llvm::ArrayRef<mlir::OpFoldResult> destinationBaseOffsets = {},
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes =
        nullptr) {
  auto selected = llvm::find_if(
      operationTemporalTiles, [&](const StructuredOpTemporalTile &tile) {
        return tile.operation == sourceCompute.getOperation();
      });
  llvm::SmallVector<mlir::AffineMap, 4> maps =
      sourceCompute.getIndexingMapsArray();
  const unsigned outputMapIndex =
      static_cast<unsigned>(sourceCompute.getNumDpsInputs());
  if (selected == operationTemporalTiles.end() ||
      outputMapIndex >= maps.size() ||
      requestedOutputSizes.size() != maps[outputMapIndex].getNumResults()) {
    setFailureReason(
        failureReason,
        "configured structured traversal lacks its exact output relation");
    return mlir::failure();
  }
  llvm::SmallVector<int64_t, 4> parallelTileSizes;
  parallelTileSizes.reserve(requestedOutputSizes.size());
  bool hasParallelSplit = false;
  for (auto [resultDimension, expression] :
       llvm::enumerate(maps[outputMapIndex].getResults())) {
    auto loopDimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!loopDimension ||
        loopDimension.getPosition() >= selected->iteratorTileSizes.size()) {
      setFailureReason(
          failureReason,
          "configured output relation is not a projected iterator domain");
      return mlir::failure();
    }
    int64_t tileSize = selected->iteratorTileSizes[loopDimension.getPosition()];
    parallelTileSizes.push_back(tileSize);
    hasParallelSplit |= tileSize < requestedOutputSizes[resultDimension];
  }
  if (!hasParallelSplit) {
    mlir::FailureOr<mlir::Value> local = materializeConfiguredComputeTile(
        builder, scope, sourceCompute, requestedOutputOffsets,
        requestedOutputSizes, loops, operationTemporalTiles, failureReason,
        operationNodes);
    if (mlir::failed(local))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> wrapped = materializeConfiguredCollectiveTile(
        builder, root, sourceCompute, requestedOutputOffsets,
        requestedOutputSizes, *local, failureReason, operationNodes);
    if (mlir::failed(wrapped) || !outputDestination)
      return wrapped;
    llvm::ArrayRef<mlir::OpFoldResult> outputOffsets =
        destinationBaseOffsets.empty() ? requestedOutputOffsets
                                       : destinationBaseOffsets;
    return insertCandidateRootTile(builder, root->getLoc(), *wrapped,
                                   outputDestination, outputOffsets,
                                   requestedOutputSizes);
  }

  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      sourceCompute->getResult(0).getType());
  if (!resultType || requestedOutputOffsets.size() !=
                         static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "configured parallel traversal result rank mismatch");
    return mlir::failure();
  }
  mlir::Value output = outputDestination;
  if (!output)
    output = builder
                 .create<mlir::tensor::EmptyOp>(
                     root->getLoc(), requestedOutputSizes,
                     resultType.getElementType(), mlir::ValueRange{},
                     resultType.getEncoding())
                 .getResult();
  llvm::SmallVector<mlir::OpFoldResult, 4> sourceOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> localOffsets;
  llvm::SmallVector<int64_t, 4> tileSizes;
  return materializeConfiguredParallelTraversal(
      builder, scope, root, sourceCompute, requestedOutputOffsets,
      requestedOutputSizes, parallelTileSizes, /*dimension=*/0, output,
      destinationBaseOffsets, sourceOffsets, localOffsets, tileSizes, loops,
      operationTemporalTiles, failureReason, operationNodes);
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
          failureReason);
  if (mlir::failed(assembled))
    return mlir::failure();
  value.replaceAllUsesWith(*assembled);
  return true;
}

// Infer the iteration range [first, last] (inclusive, in induction-variable
// values) that a dynamic window offset can take when it is (a constant shift
// of) an enclosing scf.for induction variable.  A traversal wave window whose
// offset is such a value keeps a static intersection size with an inserted
// extent: the runtime overlap [max(insert, wave), min(insertEnd, wave + size))
// is constant over the whole wave, and the extreme iterations are checked to
// prove it.
struct LoopCarriedWaveRange {
  int64_t first = 0;
  int64_t last = 0;
  int64_t step = 0;
};

static std::optional<LoopCarriedWaveRange>
getLoopCarriedWaveRange(mlir::Value offset) {
  int64_t shift = 0;
  mlir::Value base = offset;
  while (mlir::Operation *def = base.getDefiningOp()) {
    if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(def)) {
      auto lhs = mlir::getConstantIntValue(addi.getLhs());
      auto rhs = mlir::getConstantIntValue(addi.getRhs());
      if (lhs && !rhs) {
        shift += *lhs;
        base = addi.getRhs();
        continue;
      }
      if (rhs && !lhs) {
        shift += *rhs;
        base = addi.getLhs();
        continue;
      }
      return std::nullopt;
    }
    if (auto subi = mlir::dyn_cast<mlir::arith::SubIOp>(def)) {
      auto rhs = mlir::getConstantIntValue(subi.getRhs());
      if (rhs) {
        shift -= *rhs;
        base = subi.getLhs();
        continue;
      }
      return std::nullopt;
    }
    if (auto muli = mlir::dyn_cast<mlir::arith::MulIOp>(def)) {
      auto lhs = mlir::getConstantIntValue(muli.getLhs());
      auto rhs = mlir::getConstantIntValue(muli.getRhs());
      if ((lhs && *lhs != 1) || (rhs && *rhs != 1))
        return std::nullopt;
      base = lhs ? muli.getRhs() : muli.getLhs();
      continue;
    }
    return std::nullopt;
  }
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(base);
  if (!blockArg)
    return std::nullopt;
  auto forOp =
      mlir::dyn_cast<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp || forOp.getInductionVar() != base)
    return std::nullopt;
  auto lb = mlir::getConstantIntValue(forOp.getLowerBound());
  auto ub = mlir::getConstantIntValue(forOp.getUpperBound());
  auto step = mlir::getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0)
    return std::nullopt;
  const int64_t first = *lb + shift;
  const int64_t last = *lb + ((*ub - 1 - *lb) / *step) * *step + shift;
  return LoopCarriedWaveRange{first, last, *step};
}

/// Returns every positive overlap extent that a fixed-size window can have
/// with a fixed insert interval while its offset traverses one arithmetic
/// progression.  The overlap is piecewise linear and changes slope only at
/// the four interval-boundary events below, so sampling the neighboring
/// progression points is complete without expanding every temporal wave.
static llvm::SmallVector<int64_t, 3>
getWaveOverlapExtents(const LoopCarriedWaveRange &wave, int64_t windowSize,
                      int64_t insertLo, int64_t insertHi) {
  llvm::SmallVector<int64_t, 3> extents;
  if (windowSize <= 0 || insertHi <= insertLo || wave.step <= 0 ||
      wave.last < wave.first)
    return extents;
  const int64_t iterationCount = (wave.last - wave.first) / wave.step + 1;
  auto floorDiv = [](int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    int64_t remainder = numerator % denominator;
    if (remainder < 0)
      --quotient;
    return quotient;
  };
  auto sample = [&](int64_t iteration) {
    if (iteration < 0 || iteration >= iterationCount)
      return;
    const int64_t windowLo = wave.first + iteration * wave.step;
    const int64_t windowHi = windowLo + windowSize;
    const int64_t overlap =
        std::min(insertHi, windowHi) - std::max(insertLo, windowLo);
    if (overlap > 0)
      extents.push_back(overlap);
  };
  sample(/*iteration=*/0);
  sample(iterationCount - 1);
  for (int64_t breakpoint :
       {insertLo - windowSize, insertLo, insertHi - windowSize, insertHi}) {
    const int64_t nearest = floorDiv(breakpoint - wave.first, wave.step);
    for (int64_t delta : {-1, 0, 1})
      sample(nearest + delta);
  }
  llvm::sort(extents);
  extents.erase(std::unique(extents.begin(), extents.end()), extents.end());
  return extents;
}

mlir::LogicalResult fuseCandidateProducerSlices(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    TensorProgramScope scope,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    mlir::OpBuilder::Listener *insertionListener, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  // With a structured loop nest, a fused tile is created in a nested block and
  // therefore cannot be mistaken for another untiled scope producer.  The
  // direct untiled API has no enclosing loop, so remember the finite set
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

  mlir::IRRewriter rewriter(scope.getContext(), insertionListener);
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
    if (producerResult &&
        (mlir::isa<mlir::tensor::ExpandShapeOp>(producerResult.getOwner()) ||
         mlir::isa<mlir::tensor::CollapseShapeOp>(producerResult.getOwner()))) {
      mlir::FailureOr<bool> assembled = tryAssembleBoundaryProducerValue(
          rewriter, scope, slice, operationTemporalTiles, failureReason);
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
        producerResult.getOwner()->getBlock() != &scope.getBody() ||
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
      const size_t rank = slice.getMixedOffsets().size();
      llvm::SmallVector<int64_t, 4> insertOffsets;
      llvm::SmallVector<int64_t, 4> insertSizes;
      llvm::SmallVector<int64_t, 4> insertStrides;
      llvm::SmallVector<int64_t, 4> windowOffsets;
      llvm::SmallVector<int64_t, 4> windowSizes;
      insertOffsets.reserve(rank);
      insertSizes.reserve(rank);
      insertStrides.reserve(rank);
      windowOffsets.reserve(rank);
      windowSizes.reserve(rank);
      bool insertFullyStatic = rank == insert.getMixedOffsets().size() &&
                               rank == insert.getMixedSizes().size() &&
                               rank == insert.getMixedStrides().size();
      for (size_t dimension = 0; insertFullyStatic && dimension < rank;
           ++dimension) {
        auto insertOffset =
            mlir::getConstantIntValue(insert.getMixedOffsets()[dimension]);
        auto insertSize =
            mlir::getConstantIntValue(insert.getMixedSizes()[dimension]);
        auto insertStride =
            mlir::getConstantIntValue(insert.getMixedStrides()[dimension]);
        insertFullyStatic &= insertOffset && insertSize && insertStride;
        if (!insertFullyStatic)
          break;
        insertOffsets.push_back(*insertOffset);
        insertSizes.push_back(*insertSize);
        insertStrides.push_back(*insertStride);
      }
      if (!insertFullyStatic)
        continue;
      bool windowFullyStatic = rank == slice.getMixedOffsets().size() &&
                               rank == slice.getMixedSizes().size();
      for (size_t dimension = 0; windowFullyStatic && dimension < rank;
           ++dimension) {
        auto windowOffset =
            mlir::getConstantIntValue(slice.getMixedOffsets()[dimension]);
        auto windowSize =
            mlir::getConstantIntValue(slice.getMixedSizes()[dimension]);
        windowFullyStatic &= windowOffset && windowSize;
        if (!windowFullyStatic)
          break;
        windowOffsets.push_back(*windowOffset);
        windowSizes.push_back(*windowSize);
      }
      if (windowFullyStatic) {
        llvm::SmallVector<int64_t, 4> relativeOffsets(rank);
        llvm::SmallVector<int64_t, 4> intersectSizes(rank);
        bool disjoint = false;
        for (size_t dimension = 0; dimension < rank; ++dimension) {
          const int64_t insertLo = insertOffsets[dimension];
          const int64_t insertHi = insertLo + insertSizes[dimension];
          const int64_t windowLo = windowOffsets[dimension];
          const int64_t windowHi = windowLo + windowSizes[dimension];
          const int64_t intersectLo = std::max(insertLo, windowLo);
          const int64_t intersectHi = std::min(insertHi, windowHi);
          if (intersectHi <= intersectLo) {
            disjoint = true;
            break;
          }
          relativeOffsets[dimension] = intersectLo - insertLo;
          intersectSizes[dimension] = intersectHi - intersectLo;
        }
        rewriter.setInsertionPoint(slice);
        llvm::SmallVector<mlir::OpFoldResult, 4> unitStrides(
            rank, rewriter.getIndexAttr(1));
        mlir::Value destSlice =
            rewriter
                .create<mlir::tensor::ExtractSliceOp>(
                    slice.getLoc(), slice.getType(), insert.getDest(),
                    slice.getMixedOffsets(), slice.getMixedSizes(), unitStrides)
                .getResult();
        mlir::Value tiled = destSlice;
        if (!disjoint) {
          auto sourceType =
              mlir::cast<mlir::RankedTensorType>(insert.getSource().getType());
          auto inputSliceType = mlir::RankedTensorType::get(
              intersectSizes, sourceType.getElementType());
          llvm::SmallVector<mlir::OpFoldResult, 4> inputOffsets;
          llvm::SmallVector<mlir::OpFoldResult, 4> inputSizes;
          inputOffsets.reserve(rank);
          inputSizes.reserve(rank);
          for (size_t dimension = 0; dimension < rank; ++dimension) {
            inputOffsets.push_back(
                rewriter.getIndexAttr(relativeOffsets[dimension]));
            inputSizes.push_back(
                rewriter.getIndexAttr(intersectSizes[dimension]));
          }
          mlir::Value inputSlice =
              rewriter
                  .create<mlir::tensor::ExtractSliceOp>(
                      slice.getLoc(), inputSliceType, insert.getSource(),
                      inputOffsets, inputSizes, unitStrides)
                  .getResult();
          llvm::SmallVector<mlir::OpFoldResult, 4> windowRelativeOffsets;
          llvm::SmallVector<mlir::OpFoldResult, 4> tiledInsertSizes;
          windowRelativeOffsets.reserve(rank);
          tiledInsertSizes.reserve(rank);
          for (size_t dimension = 0; dimension < rank; ++dimension) {
            const int64_t intersectLo =
                std::max(insertOffsets[dimension], windowOffsets[dimension]);
            windowRelativeOffsets.push_back(
                rewriter.getIndexAttr(intersectLo - windowOffsets[dimension]));
            tiledInsertSizes.push_back(
                rewriter.getIndexAttr(intersectSizes[dimension]));
          }
          auto tiledInsert = rewriter.create<mlir::tensor::InsertSliceOp>(
              slice.getLoc(), inputSlice, destSlice, windowRelativeOffsets,
              tiledInsertSizes, insert.getMixedStrides());
          tiled = tiledInsert.getResult();
          // The intersecting input slice continues the fusion walk upstream
          // toward the structured producer of the inserted value.
          enqueueSlices({inputSlice.getDefiningOp()}, insert.getOperation(),
                        {tiledInsert.getOperation()});
        }
        materializedCoupledTiles.push_back(MaterializedCoupledProducerTile{
            producerResult, slice->getBlock(), slice.getType(),
            llvm::to_vector(slice.getMixedOffsets()),
            llvm::to_vector(slice.getMixedSizes()),
            llvm::to_vector(slice.getMixedStrides()), tiled});
        slice.getResult().replaceAllUsesWith(tiled);
        if (slice->use_empty())
          rewriter.eraseOp(slice);
        // The windowed destination may itself be the result of a nested
        // insert (a tiled concatenation).  Continue the walk through that
        // inner insert so every read lands on the innermost producer instead
        // of retaining the chained intermediate value in physical storage.
        // The enqueue must happen after the rewiring above: the destination
        // slice only has users once the original slice's uses have been
        // replaced (a pure destination copy forwards those users directly).
        if (mlir::Operation *destProducer = insert.getDest().getDefiningOp()) {
          llvm::SmallVector<mlir::Operation *, 4> destUsers;
          for (mlir::Operation *user : destSlice.getUsers())
            destUsers.push_back(user);
          enqueueSlices({destSlice.getDefiningOp()}, insert.getOperation(),
                        destUsers);
        }
        continue;
      }
      // A dynamic window dimension (a temporal traversal induction variable)
      // needs a runtime intersection with the inserted region.  Compute the
      // per-dimension overlap [max(insert, window), min(insertEnd, windowEnd))
      // and select between the pure destination copy and the intersecting
      // insert through scf.if.
      //
      // The intersection must keep a static shape: per dimension the inserted
      // extent is a single element (the decode one-row cache write), or the
      // window lies completely inside the inserted extent with a statically
      // aligned start (a windowed read of a full-width cache write), or the
      // inserted extent covers the whole dimension (a full-width cache write
      // read through a traversal wave), or the window offset is an enclosing
      // loop induction variable whose wave has either one constant overlap or
      // a finite prologue/steady/tail overlap class with the inserted extent.
      llvm::SmallVector<int64_t, 4> intersectSizes(rank);
      bool staticIntersect = true;
      std::optional<unsigned> varyingIntersectDimension;
      llvm::SmallVector<int64_t, 3> varyingIntersectExtents;
      llvm::ArrayRef<int64_t> destDimSizes =
          mlir::cast<mlir::RankedTensorType>(insert.getDest().getType())
              .getShape();
      for (size_t dimension = 0; staticIntersect && dimension < rank;
           ++dimension) {
        if (insertSizes[dimension] == 1) {
          intersectSizes[dimension] = 1;
          continue;
        }
        auto windowOffset =
            mlir::getConstantIntValue(slice.getMixedOffsets()[dimension]);
        auto windowSize =
            mlir::getConstantIntValue(slice.getMixedSizes()[dimension]);
        if (!windowSize) {
          staticIntersect = false;
          break;
        }
        if (windowOffset) {
          // The inserted extent covers the whole dimension: the window lies
          // inside it whatever its static offset, so the overlap is the full
          // window extent (a full-width cache write read through a windowed
          // read, with the window starting past the cache's origin).
          if (insertOffsets[dimension] == 0 &&
              insertOffsets[dimension] + insertSizes[dimension] ==
                  destDimSizes[dimension]) {
            intersectSizes[dimension] = *windowSize;
            continue;
          }
          // A static window offset keeps the overlap extent static even when
          // the window straddles the inserted region boundary (a row wave
          // clipped at a concatenation boundary).  The runtime disjoint check
          // below still selects the reachable branch; the intersecting branch
          // only touches the computed overlap.
          const int64_t windowHi = *windowOffset + *windowSize;
          const int64_t intersectLo =
              std::max(insertOffsets[dimension], *windowOffset);
          const int64_t intersectHi = std::min(
              insertOffsets[dimension] + insertSizes[dimension], windowHi);
          const int64_t intersectSize = intersectHi - intersectLo;
          if (intersectSize <= 0) {
            staticIntersect = false;
            break;
          }
          intersectSizes[dimension] = intersectSize;
          continue;
        }
        const int64_t insertHi =
            insertOffsets[dimension] + insertSizes[dimension];
        if (insertOffsets[dimension] == 0 &&
            insertHi == destDimSizes[dimension]) {
          intersectSizes[dimension] = *windowSize;
          continue;
        }
        auto waveRange = getLoopCarriedWaveRange(
            mlir::cast<mlir::Value>(slice.getMixedOffsets()[dimension]));
        if (!waveRange || waveRange->step != *windowSize) {
          staticIntersect = false;
          break;
        }
        llvm::SmallVector<int64_t, 3> extents = getWaveOverlapExtents(
            *waveRange, *windowSize, insertOffsets[dimension], insertHi);
        if (extents.empty()) {
          staticIntersect = false;
          break;
        }
        if (extents.size() > 1) {
          if (varyingIntersectDimension) {
            staticIntersect = false;
            break;
          }
          varyingIntersectDimension = static_cast<unsigned>(dimension);
          varyingIntersectExtents = extents;
        }
        intersectSizes[dimension] = extents.front();
      }
      if (!staticIntersect)
        continue;
      rewriter.setInsertionPoint(slice);
      llvm::SmallVector<mlir::Value, 4> windowLoValues;
      llvm::SmallVector<mlir::Value, 4> intersectLoValues;
      llvm::SmallVector<mlir::Value, 4> intersectSizeValues;
      auto toIndexValue = [&](mlir::OpFoldResult ofr) -> mlir::Value {
        if (auto value = mlir::dyn_cast<mlir::Value>(ofr))
          return value;
        return rewriter.create<mlir::arith::ConstantIndexOp>(
            slice.getLoc(), *mlir::getConstantIntValue(ofr));
      };
      mlir::Value disjoint =
          rewriter.create<mlir::arith::ConstantIntOp>(slice.getLoc(), 0, 1);
      for (size_t dimension = 0; dimension < rank; ++dimension) {
        mlir::Value windowLo = toIndexValue(slice.getMixedOffsets()[dimension]);
        mlir::Value windowHi = toIndexValue(slice.getMixedSizes()[dimension]);
        windowHi = rewriter.create<mlir::arith::AddIOp>(slice.getLoc(),
                                                        windowLo, windowHi);
        const int64_t insertLoValue = insertOffsets[dimension];
        const int64_t insertHiValue =
            insertOffsets[dimension] + insertSizes[dimension];
        mlir::Value insertHi = rewriter.create<mlir::arith::ConstantIndexOp>(
            slice.getLoc(), insertHiValue);
        mlir::Value insertLo = rewriter.create<mlir::arith::ConstantIndexOp>(
            slice.getLoc(), insertLoValue);
        const bool insertCoversDimension =
            insertLoValue == 0 && insertHiValue == destDimSizes[dimension];
        mlir::Value intersectLo = insertCoversDimension
                                      ? windowLo
                                      : rewriter.create<mlir::arith::MaxSIOp>(
                                            slice.getLoc(), insertLo, windowLo);
        mlir::Value intersectHi = insertCoversDimension
                                      ? windowHi
                                      : rewriter.create<mlir::arith::MinSIOp>(
                                            slice.getLoc(), insertHi, windowHi);
        // A window extracted from the destination is necessarily contained
        // in a full-dimension insert.  Do not manufacture an always-false
        // disjoint branch: retaining it hides a straight-line physical loop
        // from exact scheduling and selected buffering.
        if (!insertCoversDimension) {
          mlir::Value emptyDim = rewriter.create<mlir::arith::CmpIOp>(
              slice.getLoc(), mlir::arith::CmpIPredicate::sle, intersectHi,
              intersectLo);
          disjoint = rewriter.create<mlir::arith::OrIOp>(slice.getLoc(),
                                                         disjoint, emptyDim);
        }
        windowLoValues.push_back(windowLo);
        intersectLoValues.push_back(intersectLo);
        intersectSizeValues.push_back(rewriter.create<mlir::arith::SubIOp>(
            slice.getLoc(), intersectHi, intersectLo));
      }
      llvm::SmallVector<mlir::OpFoldResult, 4> unitStrides(
          rank, rewriter.getIndexAttr(1));
      mlir::Value destSlice =
          rewriter
              .create<mlir::tensor::ExtractSliceOp>(
                  slice.getLoc(), slice.getType(), insert.getDest(),
                  slice.getMixedOffsets(), slice.getMixedSizes(), unitStrides)
              .getResult();
      auto materializeStaticIntersection =
          [&](mlir::OpBuilder &branchBuilder,
              llvm::ArrayRef<int64_t> staticIntersectSizes) -> mlir::Value {
        auto sourceType =
            mlir::cast<mlir::RankedTensorType>(insert.getSource().getType());
        auto inputSliceType = mlir::RankedTensorType::get(
            staticIntersectSizes, sourceType.getElementType());
        llvm::SmallVector<mlir::OpFoldResult, 4> inputOffsets;
        llvm::SmallVector<mlir::OpFoldResult, 4> inputSizes;
        llvm::SmallVector<mlir::OpFoldResult, 4> windowRelativeOffsets;
        inputOffsets.reserve(rank);
        inputSizes.reserve(rank);
        windowRelativeOffsets.reserve(rank);
        for (size_t dimension = 0; dimension < rank; ++dimension) {
          mlir::Value intersectLo = intersectLoValues[dimension];
          auto clampRelativeOffset = [&](mlir::Value relative,
                                         int64_t maximum) {
            mlir::Value zero =
                branchBuilder.create<mlir::arith::ConstantIndexOp>(
                    slice.getLoc(), 0);
            mlir::Value upper =
                branchBuilder.create<mlir::arith::ConstantIndexOp>(
                    slice.getLoc(), maximum);
            relative = branchBuilder.create<mlir::arith::MaxSIOp>(
                slice.getLoc(), relative, zero);
            return branchBuilder
                .create<mlir::arith::MinSIOp>(slice.getLoc(), relative, upper)
                .getResult();
          };
          mlir::Value inputRelative =
              branchBuilder
                  .create<mlir::arith::SubIOp>(
                      slice.getLoc(), intersectLo,
                      branchBuilder.create<mlir::arith::ConstantIndexOp>(
                          slice.getLoc(), insertOffsets[dimension]))
                  .getResult();
          inputOffsets.push_back(clampRelativeOffset(
              inputRelative,
              insertSizes[dimension] - staticIntersectSizes[dimension]));
          inputSizes.push_back(
              branchBuilder.getIndexAttr(staticIntersectSizes[dimension]));
          mlir::Value windowRelative =
              branchBuilder
                  .create<mlir::arith::SubIOp>(slice.getLoc(), intersectLo,
                                               windowLoValues[dimension])
                  .getResult();
          windowRelativeOffsets.push_back(clampRelativeOffset(
              windowRelative, slice.getType().getDimSize(dimension) -
                                  staticIntersectSizes[dimension]));
        }
        mlir::Value inputSlice =
            branchBuilder
                .create<mlir::tensor::ExtractSliceOp>(
                    slice.getLoc(), inputSliceType, insert.getSource(),
                    inputOffsets, inputSizes, unitStrides)
                .getResult();
        auto inserted = branchBuilder.create<mlir::tensor::InsertSliceOp>(
            slice.getLoc(), inputSlice, destSlice, windowRelativeOffsets,
            inputSizes, insert.getMixedStrides());
        enqueueSlices({inputSlice.getDefiningOp()}, insert.getOperation(),
                      {inserted.getOperation()});
        return inserted.getResult();
      };
      auto ifOp = rewriter.create<mlir::scf::IfOp>(
          slice.getLoc(), mlir::TypeRange{slice.getType()}, disjoint, true);
      // disjoint branch: the insert does not touch this window, forward the
      // pre-insert destination version unchanged.
      {
        mlir::OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
        mlir::OpBuilder::InsertionGuard guard(thenBuilder);
        thenBuilder.create<mlir::scf::YieldOp>(slice.getLoc(), destSlice);
      }
      {
        // Intersecting branch: every carried wave has one of the statically
        // proven overlap shapes.  When an insertion boundary cuts the final
        // wave (for example a 1023-row cache plus one new row under 64-row
        // traversal), select the finite static shape with nested scf.if
        // rather than retaining the full functional tensor in SPM.
        mlir::OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
        mlir::OpBuilder::InsertionGuard guard(elseBuilder);
        mlir::Value intersected;
        if (!varyingIntersectDimension) {
          intersected =
              materializeStaticIntersection(elseBuilder, intersectSizes);
        } else {
          std::function<mlir::Value(mlir::OpBuilder &, size_t)> buildChoice;
          buildChoice = [&](mlir::OpBuilder &choiceBuilder,
                            size_t extentIndex) -> mlir::Value {
            llvm::SmallVector<int64_t, 4> selectedSizes(intersectSizes);
            selectedSizes[*varyingIntersectDimension] =
                varyingIntersectExtents[extentIndex];
            if (extentIndex + 1 == varyingIntersectExtents.size())
              return materializeStaticIntersection(choiceBuilder,
                                                   selectedSizes);
            auto expected = choiceBuilder.create<mlir::arith::ConstantIndexOp>(
                slice.getLoc(), varyingIntersectExtents[extentIndex]);
            auto matches = choiceBuilder.create<mlir::arith::CmpIOp>(
                slice.getLoc(), mlir::arith::CmpIPredicate::eq,
                intersectSizeValues[*varyingIntersectDimension], expected);
            auto choice = choiceBuilder.create<mlir::scf::IfOp>(
                slice.getLoc(), mlir::TypeRange{slice.getType()}, matches,
                true);
            {
              mlir::OpBuilder thenBuilder = choice.getThenBodyBuilder();
              mlir::OpBuilder::InsertionGuard nestedGuard(thenBuilder);
              thenBuilder.create<mlir::scf::YieldOp>(
                  slice.getLoc(),
                  materializeStaticIntersection(thenBuilder, selectedSizes));
            }
            {
              mlir::OpBuilder nestedElseBuilder = choice.getElseBodyBuilder();
              mlir::OpBuilder::InsertionGuard nestedGuard(nestedElseBuilder);
              nestedElseBuilder.create<mlir::scf::YieldOp>(
                  slice.getLoc(),
                  buildChoice(nestedElseBuilder, extentIndex + 1));
            }
            return choice.getResult(0);
          };
          intersected = buildChoice(elseBuilder, /*extentIndex=*/0);
        }
        elseBuilder.create<mlir::scf::YieldOp>(slice.getLoc(), intersected);
      }
      mlir::Value tiled = ifOp.getResult(0);
      materializedCoupledTiles.push_back(MaterializedCoupledProducerTile{
          producerResult, slice->getBlock(), slice.getType(),
          llvm::to_vector(slice.getMixedOffsets()),
          llvm::to_vector(slice.getMixedSizes()),
          llvm::to_vector(slice.getMixedStrides()), tiled});
      slice.getResult().replaceAllUsesWith(tiled);
      if (slice->use_empty())
        rewriter.eraseOp(slice);
      // The windowed destination may itself be the result of a nested
      // insert (a tiled concatenation).  Continue the walk through that
      // inner insert so every read lands on the innermost producer instead
      // of retaining the chained intermediate value in physical storage.
      // The enqueue must happen after the rewiring above: the destination
      // slice only has users once the original slice's uses have been
      // replaced (a pure destination copy forwards those users directly).
      if (mlir::Operation *destProducer = insert.getDest().getDefiningOp()) {
        llvm::SmallVector<mlir::Operation *, 4> destUsers;
        for (mlir::Operation *user : destSlice.getUsers())
          destUsers.push_back(user);
        enqueueSlices({destSlice.getDefiningOp()}, insert.getOperation(),
                      destUsers);
      }
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
                 llvm::equal(materialized.offsets, slice.getMixedOffsets()) &&
                 llvm::equal(materialized.sizes, slice.getMixedSizes()) &&
                 llvm::equal(materialized.strides, slice.getMixedStrides());
        });
    if (reusable != materializedCoupledTiles.end()) {
      slice.getResult().replaceAllUsesWith(reusable->tiledValue);
      if (slice->use_empty())
        rewriter.eraseOp(slice);
      continue;
    }

    if (auto structured =
            mlir::dyn_cast<mlir::linalg::LinalgOp>(producerResult.getOwner())) {
      auto selected = llvm::find_if(
          operationTemporalTiles, [&](const StructuredOpTemporalTile &tile) {
            return tile.operation == structured.getOperation();
          });
      if (selected != operationTemporalTiles.end()) {
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
                rewriter, scope, structured.getOperation(), structured,
                slice.getMixedOffsets(), requestedType.getShape(), loops,
                operationTemporalTiles, failureReason,
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
      inheritStructuredOperationNodes(fused->origProducer.getOwner(),
                                      tiledOperation, operationNodes);
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

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    TensorProgramScope scope, mlir::Operation *root, unsigned outputIndex,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
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
      loops, operationTemporalTiles, failureReason, operationNodes);
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
  if (mlir::failed(fuseCandidateProducerSlices(
          tiledConsumer, root, scope, loops,
          /*operationTemporalTiles=*/{}, builder.getListener(), failureReason)))
    return mlir::failure();
  builder.setInsertionPointAfter(tiledConsumer);
  return materialized->tiledValues.front();
}

static mlir::FailureOr<mlir::linalg::LinalgOp>
getCandidateReductionComputeRoot(mlir::Operation *root,
                                 std::string *failureReason) {
  if (auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(root))
    return linalg;

  auto allReduce = mlir::dyn_cast_or_null<LinalgExtCollectiveAllReduceOp>(root);
  if (!allReduce || allReduce.getInputs().size() != 1 ||
      allReduce.getOuts().size() != 1 || root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "reduction traversal requires a direct reduction or one "
                     "typed all-reduce wrapper");
    return mlir::failure();
  }

  mlir::Value input = allReduce.getInputs().front();
  auto producer =
      mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(input.getDefiningOp());
  if (!producer || producer->getBlock() != root->getBlock() ||
      producer->getNumResults() != 1 || producer->getResult(0) != input ||
      !input.hasOneUse()) {
    setFailureReason(failureReason,
                     "reduction typed all-reduce requires one exact single-use "
                     "same-block Linalg producer");
    return mlir::failure();
  }
  return producer;
}

mlir::FailureOr<mlir::Value> materializeCandidateRootTileIntoDestination(
    TensorProgramScope scope, mlir::Operation *root,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    mlir::Value destination, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (!root || root->getNumResults() != 1 || !destination) {
    setFailureReason(
        failureReason,
        "destination-backed candidate traversal requires one result");
    return mlir::failure();
  }
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType || destination.getType() != resultType ||
      mlir::failed(validateCandidateTile(resultType, candidateTileOffsets,
                                         candidateTileSizes, failureReason)))
    return mlir::failure();
  mlir::FailureOr<mlir::linalg::LinalgOp> computeRoot =
      getCandidateReductionComputeRoot(root, failureReason);
  if (mlir::failed(computeRoot))
    return mlir::failure();
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  mixedOffsets.reserve(candidateTileOffsets.size());
  for (int64_t offset : candidateTileOffsets)
    mixedOffsets.push_back(builder.getIndexAttr(offset));
  llvm::SmallVector<mlir::LoopLikeOpInterface, 0> loops;
  return materializeConfiguredStructuredTraversal(
      builder, scope, root, *computeRoot, mixedOffsets, candidateTileSizes,
      loops, operationTemporalTiles, failureReason, destination, mixedOffsets,
      operationNodes);
}

static mlir::FailureOr<mlir::Value>
materializeConfiguredStructuredRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  mlir::FailureOr<mlir::linalg::LinalgOp> computeRoot =
      getCandidateReductionComputeRoot(root, failureReason);
  if (mlir::failed(computeRoot))
    return mlir::failure();
  return materializeConfiguredStructuredTraversal(
      builder, scope, root, *computeRoot, candidateTileOffsets,
      candidateTileSizes, loops, operationTemporalTiles, failureReason,
      /*outputDestination=*/{}, /*destinationBaseOffsets=*/{}, operationNodes);
}

static mlir::FailureOr<mlir::Value> materializeCandidateInterfaceRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned outputIndex,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(root);
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(root);
  if (!dps || !tiling || dps.getNumDpsInits() != 1 ||
      root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "candidate interface root requires one DPS output and "
                     "TilingInterface");
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
  if (!iteration &&
      candidateTileOffsets.size() == tiling.getLoopIteratorTypes().size())
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
        mapOperandTileToIterationDomain(root, builder, operandNumber,
                                        candidateTileOffsets, mixedSizes,
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

  mlir::FailureOr<mlir::TilingResult> tiled = tiling.getTiledImplementation(
      builder, iteration->offsets, iteration->sizes);
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
  inheritStructuredOperationNodes(root, tiledRoot, operationNodes);
  if (mlir::failed(fuseCandidateProducerSlices(
          tiledRoot, root, scope, loops, operationTemporalTiles,
          builder.getListener(), failureReason, operationNodes)))
    return mlir::failure();
  builder.setInsertionPointAfter(tiledRoot);
  return tiled->tiledValues.front();
}

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned outputIndex,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (classifyStructuredRoot(root) != StructuredRootCapability::Tiled) {
    setFailureReason(failureReason,
                     "candidate traversal root does not support tiling");
    return mlir::failure();
  }
  mlir::linalg::LinalgOp reductionCompute =
      mlir::dyn_cast<mlir::linalg::LinalgOp>(root);
  if (!reductionCompute && mlir::isa<LinalgExtCollectiveAllReduceOp>(root)) {
    mlir::FailureOr<mlir::linalg::LinalgOp> resolved =
        getCandidateReductionComputeRoot(root, failureReason);
    if (mlir::failed(resolved))
      return mlir::failure();
    reductionCompute = *resolved;
  }
  if (reductionCompute) {
    auto selected = llvm::find_if(
        operationTemporalTiles, [&](const StructuredOpTemporalTile &tile) {
          return tile.operation == reductionCompute.getOperation();
        });
    if (selected != operationTemporalTiles.end())
      return materializeConfiguredStructuredRootTileValue(
          builder, scope, root, candidateTileOffsets, candidateTileSizes, loops,
          operationTemporalTiles, failureReason, operationNodes);
  }
  return materializeCandidateInterfaceRootTileValue(
      builder, scope, root, outputIndex, candidateTileOffsets,
      candidateTileSizes, loops, operationTemporalTiles, failureReason,
      operationNodes);
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

/// Give every pure ranked-tensor function result a structured output anchor.
/// Source programs routinely return a shape view or an insert/extract update
/// rather than the last compute op itself.  Those values are still ordinary
/// SSA dataflow and must not narrow structured-DAG search to workloads whose
/// return happens to be a DPS op.  A generic identity anchor exposes the result
/// domain through TilingInterface; producer fusion must then prove the exact
/// tile relation through the original view/update chain.  Failure to fuse
/// remains a candidate legality failure rather than a full-tensor fallback.
static mlir::LogicalResult materializeCandidateOutputAnchors(
    TensorProgramScope scope, llvm::ArrayRef<SpatialOutputShard> outputShards,
    std::string *failureReason) {
  auto returnOp =
      mlir::dyn_cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  if (!returnOp)
    return mlir::failure();

  mlir::OpBuilder builder(returnOp);
  for (auto [outputIndex, returned] : llvm::enumerate(returnOp.getOperands())) {
    mlir::Operation *definition = returned.getDefiningOp();
    if (classifyStructuredRoot(definition) == StructuredRootCapability::Tiled)
      continue;
    auto toTensor =
        mlir::dyn_cast_or_null<mlir::bufferization::ToTensorOp>(definition);
    auto compilerOwnedAllocation =
        toTensor ? toTensor.getMemref().getDefiningOp<mlir::memref::AllocOp>()
                 : mlir::memref::AllocOp{};
    const bool compilerOwnedDDRBoundary =
        compilerOwnedAllocation &&
        isWaferDDRMemRefType(compilerOwnedAllocation.getType()) &&
        compilerOwnedAllocation.getDynamicSizes().empty() &&
        compilerOwnedAllocation.getSymbolOperands().empty();
    if (!definition || definition->getBlock() != &scope.getBody() ||
        (!mlir::isMemoryEffectFree(definition) && !compilerOwnedDDRBoundary)) {
      setFailureReason(failureReason,
                       "candidate traversal root is unsupported");
      return mlir::failure();
    }
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(returned.getType());
    if (!resultType || !resultType.hasStaticShape()) {
      setFailureReason(
          failureReason,
          "candidate output anchor requires a static ranked tensor");
      return mlir::failure();
    }
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        scope, static_cast<unsigned>(outputIndex), failureReason);
    if (mlir::failed(outputBoundary))
      return mlir::failure();

    mlir::AffineMap identity = mlir::AffineMap::getMultiDimIdentityMap(
        resultType.getRank(), builder.getContext());
    llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
        resultType.getRank(), mlir::utils::IteratorType::parallel);
    auto anchor = builder.create<mlir::linalg::GenericOp>(
        returned.getLoc(), mlir::TypeRange{returned.getType()},
        mlir::ValueRange{returned}, mlir::ValueRange{*outputBoundary},
        llvm::ArrayRef<mlir::AffineMap>{identity, identity}, iteratorTypes,
        [&](mlir::OpBuilder &bodyBuilder, mlir::Location loc,
            mlir::ValueRange arguments) {
          bodyBuilder.create<mlir::linalg::YieldOp>(loc, arguments.front());
        });
    returnOp->setOperand(outputIndex, anchor.getResult(0));
  }
  return mlir::success();
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
  for (auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
    mlir::Operation *rootOperation = value.getDefiningOp();
    StructuredRootCapability capability = classifyStructuredRoot(rootOperation);
    if (capability == StructuredRootCapability::FullTraversalOnly) {
      setFailureReason(failureReason,
                       "candidate traversal root supports full traversal only");
      return mlir::failure();
    }
    if (capability == StructuredRootCapability::Unsupported) {
      setFailureReason(failureReason,
                       "candidate traversal root is unsupported");
      return mlir::failure();
    }
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(rootOperation);
    if (rootOperation->getBlock() != &scope.getBody() ||
        rootOperation->getNumResults() != 1 || !dps ||
        dps.getNumDpsInits() != 1 || rootOperation->getResult(0) != value) {
      setFailureReason(failureReason,
                       "candidate multi-output coverage requires "
                       "single-result yielded roots");
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

/// Materializes one output's selected spatial shard as a compact temporal
/// traversal.  The first full tile is a finite prologue, remaining full-size
/// steady tiles live in structured scf.for loops, and every non-divisible
/// dimension contributes one statically-shaped tail class.  The traversal
/// carries the private full-card scheduling destination through ordinary SSA,
/// so each leaf
/// writes exactly its own subview while the producer closure is fused into
/// that leaf from current TilingInterface relations.
static mlir::FailureOr<mlir::Value> materializeTemporalRootTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned outputIndex, llvm::ArrayRef<int64_t> shardOffsets,
    llvm::ArrayRef<int64_t> shardSizes,
    llvm::ArrayRef<int64_t> temporalTileSizes, unsigned dimension,
    mlir::Value output, llvm::SmallVectorImpl<mlir::OpFoldResult> &tileOffsets,
    llvm::SmallVectorImpl<int64_t> &tileSizes,
    llvm::SmallVectorImpl<mlir::LoopLikeOpInterface> &loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (dimension == shardSizes.size()) {
    mlir::FailureOr<mlir::Value> tile = materializeCandidateRootTileValue(
        builder, scope, root, outputIndex, tileOffsets, tileSizes, loops,
        operationTemporalTiles, failureReason, operationNodes);
    if (mlir::failed(tile))
      return mlir::failure();
    return insertCandidateRootTile(builder, root->getLoc(), *tile, output,
                                   tileOffsets, tileSizes);
  }

  const int64_t shardOffset = shardOffsets[dimension];
  const int64_t shardSize = shardSizes[dimension];
  const int64_t temporalSize = temporalTileSizes[dimension];
  const int64_t tailSize = shardSize % temporalSize;
  const int64_t mainSize = shardSize - tailSize;
  mlir::Value currentOutput = output;

  // A one-wave dimension is still handled by this same traversal owner, but
  // does not need a degenerate loop in the selected IR.
  if (mainSize == shardSize && temporalSize == shardSize) {
    tileOffsets.push_back(builder.getIndexAttr(shardOffset));
    tileSizes.push_back(shardSize);
    mlir::FailureOr<mlir::Value> next = materializeTemporalRootTraversal(
        builder, scope, root, outputIndex, shardOffsets, shardSizes,
        temporalTileSizes, dimension + 1, currentOutput, tileOffsets, tileSizes,
        loops, operationTemporalTiles, failureReason, operationNodes);
    tileSizes.pop_back();
    tileOffsets.pop_back();
    return next;
  }

  // The first full tile is an explicit prologue.  Keeping it outside the
  // steady loop gives downstream lifetime/cost analyses a finite startup
  // class without expanding the remaining waves.
  tileOffsets.push_back(builder.getIndexAttr(shardOffset));
  tileSizes.push_back(temporalSize);
  mlir::FailureOr<mlir::Value> prologue = materializeTemporalRootTraversal(
      builder, scope, root, outputIndex, shardOffsets, shardSizes,
      temporalTileSizes, dimension + 1, currentOutput, tileOffsets, tileSizes,
      loops, operationTemporalTiles, failureReason, operationNodes);
  tileSizes.pop_back();
  tileOffsets.pop_back();
  if (mlir::failed(prologue))
    return mlir::failure();
  currentOutput = *prologue;

  const int64_t steadySize = mainSize - temporalSize;
  if (steadySize > 0) {
    mlir::Location loc = root->getLoc();
    auto lower = builder.create<mlir::arith::ConstantIndexOp>(
        loc, shardOffset + temporalSize);
    auto upper = builder.create<mlir::arith::ConstantIndexOp>(
        loc, shardOffset + mainSize);
    auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, temporalSize);
    auto loop = builder.create<mlir::scf::ForOp>(
        loc, lower, upper, step, mlir::ValueRange(currentOutput));
    tileOffsets.push_back(loop.getInductionVar());
    tileSizes.push_back(temporalSize);
    loops.push_back(mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
    mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
    mlir::FailureOr<mlir::Value> next = materializeTemporalRootTraversal(
        bodyBuilder, scope, root, outputIndex, shardOffsets, shardSizes,
        temporalTileSizes, dimension + 1, loop.getRegionIterArgs().front(),
        tileOffsets, tileSizes, loops, operationTemporalTiles, failureReason,
        operationNodes);
    loops.pop_back();
    tileSizes.pop_back();
    tileOffsets.pop_back();
    if (mlir::failed(next))
      return mlir::failure();
    bodyBuilder.create<mlir::scf::YieldOp>(loc, *next);
    builder.setInsertionPointAfter(loop);
    currentOutput = loop.getResult(0);
  }

  if (tailSize > 0) {
    tileOffsets.push_back(builder.getIndexAttr(shardOffset + mainSize));
    tileSizes.push_back(tailSize);
    mlir::FailureOr<mlir::Value> tail = materializeTemporalRootTraversal(
        builder, scope, root, outputIndex, shardOffsets, shardSizes,
        temporalTileSizes, dimension + 1, currentOutput, tileOffsets, tileSizes,
        loops, operationTemporalTiles, failureReason, operationNodes);
    tileSizes.pop_back();
    tileOffsets.pop_back();
    if (mlir::failed(tail))
      return mlir::failure();
    currentOutput = *tail;
  }
  return currentOutput;
}

mlir::LogicalResult materializeCandidateOutputTileSlices(
    TensorProgramScope scope, llvm::ArrayRef<SpatialOutputShard> outputShards,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes,
    llvm::ArrayRef<mlir::Operation *> preservedOperations) {
  if (mlir::failed(materializeCandidateOutputAnchors(scope, outputShards,
                                                     failureReason)))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> roots =
      collectCandidateRoots(scope, /*rejectProducerChains=*/false,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();

  if (outputShards.empty()) {
    setFailureReason(failureReason,
                     "spatial output materialization requires a shard");
    return mlir::failure();
  }

  llvm::SmallVector<const SpatialOutputShard *, 4> selectedByOutput(
      roots->size(), nullptr);
  for (const SpatialOutputShard &shard : outputShards) {
    if (shard.outputIndex >= roots->size()) {
      setFailureReason(failureReason,
                       "spatial output shard index is outside function "
                       "results");
      return mlir::failure();
    }
    if (selectedByOutput[shard.outputIndex]) {
      setFailureReason(failureReason,
                       "spatial output shard index is duplicated");
      return mlir::failure();
    }
    if (shard.offsets.size() != shard.sizes.size() ||
        shard.sizes.size() != shard.temporalTileSizes.size() ||
        llvm::any_of(llvm::zip_equal(shard.offsets, shard.sizes,
                                     shard.temporalTileSizes),
                     [](auto values) {
                       auto [offset, size, temporalSize] = values;
                       return offset < 0 || size <= 0 || temporalSize <= 0 ||
                              temporalSize > size;
                     })) {
      setFailureReason(
          failureReason,
          "spatial output shard has an invalid temporal tile domain");
      return mlir::failure();
    }
    selectedByOutput[shard.outputIndex] = &shard;
  }

  mlir::Operation *partialOutputEffect = nullptr;
  llvm::DenseSet<mlir::Operation *> preservedEffectOwners(
      preservedOperations.begin(), preservedOperations.end());
  if (outputShards.size() != roots->size())
    for (mlir::Operation &operation : scope.getBody().without_terminator())
      if (!preservedEffectOwners.contains(&operation) &&
          !mlir::isMemoryEffectFree(&operation)) {
        partialOutputEffect = &operation;
        break;
      }
  if (partialOutputEffect) {
    std::string reason =
        "partial output spatial materialization requires a pure tensor "
        "program; first non-pure operation=";
    reason += partialOutputEffect->getName().getStringRef();
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  for (auto [index, root] : llvm::enumerate(*roots)) {
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        scope, static_cast<unsigned>(index), failureReason);
    if (mlir::failed(outputBoundary))
      return mlir::failure();
    const SpatialOutputShard *shard = selectedByOutput[index];
    if (!shard) {
      // This Tile does not own any domain of this observable result.  Keep the
      // full-card ABI result but make its actual body a typed no-store path.
      returnOp->setOperand(index, *outputBoundary);
      continue;
    }
    mlir::OpBuilder builder(root);
    llvm::SmallVector<mlir::OpFoldResult, 4> tileOffsets;
    llvm::SmallVector<int64_t, 4> tileSizes;
    llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
    mlir::FailureOr<mlir::Value> traversed = materializeTemporalRootTraversal(
        builder, scope, root, static_cast<unsigned>(index), shard->offsets,
        shard->sizes, shard->temporalTileSizes,
        /*dimension=*/0, *outputBoundary, tileOffsets, tileSizes, loops,
        operationTemporalTiles, failureReason, operationNodes);
    if (mlir::failed(traversed))
      return mlir::failure();
    returnOp->setOperand(index, *traversed);
  }

  // Replacing every function result leaves the original untiled roots and
  // their producer/consumer closure dead.  Let the ordinary effect-aware
  // reverse walk erase that closure in use-before-def order.  Erasing roots
  // directly is incorrect when one observable result is also the producer of
  // another observable result (or the same value is returned twice).
  eraseDeadCandidateSupportClosure(scope, preservedOperations);
  if (operationNodes) {
    llvm::DenseSet<mlir::Operation *> liveOperations;
    scope.getBody().getParentOp()->walk(
        [&](mlir::Operation *operation) { liveOperations.insert(operation); });
    llvm::erase_if(*operationNodes, [&](const auto &mapping) {
      return !liveOperations.contains(mapping.operation);
    });
  }
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
