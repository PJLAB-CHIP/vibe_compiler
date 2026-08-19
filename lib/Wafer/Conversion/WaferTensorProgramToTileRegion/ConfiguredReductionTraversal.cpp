//===- ConfiguredReductionTraversal.cpp - Reduction wave traversal -----===//

#include "ConfiguredReductionTraversal.h"

#include "Internal.h"
#include "StructuredIterationTile.h"
#include "Wafer/Analysis/Structured/ReductionSemantics.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <functional>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

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

  mlir::FailureOr<StructuredIterationTile> tiled =
      materializeStructuredIterationTile(sourceReduction.getOperation(),
                                         builder, loopTile.loopOffsets,
                                         loopTile.tileSizes, failureReason);
  if (mlir::failed(tiled) || tiled->operations.size() != 1 ||
      tiled->values.size() != 1 ||
      tiled->operations.front()->getNumResults() != 1 ||
      tiled->operations.front()->getResult(0) != tiled->values.front()) {
    setFailureReason(
        failureReason,
        "TilingInterface rejected the configured reduction iterator tile");
    return mlir::failure();
  }

  mlir::Operation *tiledOperation = tiled->operations.front();
  recordStructuredOperationNodeMaterialization(sourceReduction.getOperation(),
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
  return tiled->values.front();
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
  auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(
      sourceReduction.getDpsInits().front().getType());
  if (!isProjectedPermutationWithUnitConstants(indexingMaps[outputMapIndex],
                                               outputType)) {
    setFailureReason(
        failureReason,
        "reduction output relation requires projected dimensions or "
        "constant-zero extent-one positions");
    return mlir::failure();
  }
  for (auto [resultDimension, expression] :
       llvm::enumerate(indexingMaps[outputMapIndex].getResults())) {
    auto loopDimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (mlir::isa<mlir::AffineConstantExpr>(expression)) {
      // A result dimension that does not project any loop iterator is a
      // constant-position extent-one slice of the complete result; its full
      // requested extent is the tile and no sub-traversal exists for it.
      continue;
    }
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
  llvm::SmallVector<unsigned, 2> naturalSplitOrdinals;
  for (auto [ordinal, dimension] : llvm::enumerate(reductionDims)) {
    if (dimension >= loopRanges.size() || reductionTileSizes[ordinal] <= 0 ||
        reductionTileSizes[ordinal] > loopRanges[dimension]) {
      setFailureReason(failureReason,
                       "reduction temporal tile is outside loop bounds");
      return mlir::failure();
    }
    if (reductionTileSizes[ordinal] == loopRanges[dimension])
      continue;
    naturalSplitOrdinals.push_back(static_cast<unsigned>(ordinal));
  }
  if (naturalSplitOrdinals.empty()) {
    setFailureReason(failureReason,
                     "configured reduction producer has no temporal split");
    return mlir::failure();
  }
  llvm::SmallVector<unsigned, 2> splitOrdinals;
  if (selected->waveLoopOrder.empty()) {
    splitOrdinals = naturalSplitOrdinals;
  } else {
    for (uint32_t dimension : selected->waveLoopOrder) {
      auto ordinal = llvm::find(reductionDims, dimension);
      if (ordinal == reductionDims.end())
        continue;
      unsigned index = static_cast<unsigned>(ordinal - reductionDims.begin());
      if (llvm::is_contained(naturalSplitOrdinals, index))
        splitOrdinals.push_back(index);
    }
    if (splitOrdinals.size() != naturalSplitOrdinals.size()) {
      setFailureReason(
          failureReason,
          "reduction wave-loop order omits an active reduction iterator");
      return mlir::failure();
    }
  }

  // The traversal order is [chunk coordinates..., in-tile coordinates...].
  // It is identical to the source reduction's lexicographic order only when
  // every reduction axis preceding the last split axis has a unit in-tile
  // extent. Floating-point reassociation is a supported numeric
  // transformation (the typed comparator owns acceptance, no fast-math flag
  // is consumed); the preserved-order fact still gates integer
  // overflow-flag combinations below.
  bool preservesSequentialReductionOrder =
      splitOrdinals == naturalSplitOrdinals;
  const unsigned lastSplitOrdinal = naturalSplitOrdinals.back();
  for (unsigned ordinal = 0; ordinal < lastSplitOrdinal; ++ordinal) {
    const int64_t range = loopRanges[reductionDims[ordinal]];
    if (range > 1 && reductionTileSizes[ordinal] != 1) {
      preservesSequentialReductionOrder = false;
      break;
    }
  }
  if (mlir::failed(analysis::verifyReductionPartitionLegality(
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

mlir::FailureOr<mlir::Value> materializeConfiguredComputeTile(
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

} // namespace wafer::tensor_program_to_tile_region
