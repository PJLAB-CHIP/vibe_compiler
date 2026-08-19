//===- ConfiguredStructuredTraversal.cpp - Selected temporal traversal
//-----===//

#include "Internal.h"
#include "Wafer/Analysis/Structured/ReductionSemantics.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <functional>
#include <limits>

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
  // extent. Floating-point reassociation is a supported numeric
  // transformation (the typed comparator owns acceptance, no fast-math flag
  // is consumed); the preserved-order fact still gates integer
  // overflow-flag combinations below.
  bool preservesSequentialReductionOrder = true;
  const unsigned lastSplitOrdinal = splitOrdinals.back();
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
  recordStructuredOperationNodeMaterialization(
      root, tiledAllReduce.getOperation(), operationNodes);
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

mlir::FailureOr<mlir::Value> materializeConfiguredStructuredTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    mlir::linalg::LinalgOp sourceCompute,
    llvm::ArrayRef<mlir::OpFoldResult> requestedOutputOffsets,
    llvm::ArrayRef<int64_t> requestedOutputSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason, mlir::Value outputDestination,
    llvm::ArrayRef<mlir::OpFoldResult> destinationBaseOffsets,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
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
  auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(
      sourceCompute.getDpsInits().front().getType());
  if (!isProjectedPermutationWithUnitConstants(maps[outputMapIndex],
                                               outputType)) {
    setFailureReason(
        failureReason,
        "configured output relation requires projected dimensions or "
        "constant-zero extent-one positions");
    return mlir::failure();
  }
  llvm::SmallVector<int64_t, 4> parallelTileSizes;
  parallelTileSizes.reserve(requestedOutputSizes.size());
  bool hasParallelSplit = false;
  for (auto [resultDimension, expression] :
       llvm::enumerate(maps[outputMapIndex].getResults())) {
    auto loopDimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (mlir::isa<mlir::AffineConstantExpr>(expression)) {
      // A result dimension that does not project any loop iterator is a
      // constant-position extent-one slice of the complete result; its full
      // requested extent is the tile and no split exists for it.
      parallelTileSizes.push_back(requestedOutputSizes[resultDimension]);
      continue;
    }
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

} // namespace wafer::tensor_program_to_tile_region
