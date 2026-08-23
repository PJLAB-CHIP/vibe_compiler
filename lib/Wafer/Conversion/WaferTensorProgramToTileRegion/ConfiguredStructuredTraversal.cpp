//===- ConfiguredStructuredTraversal.cpp - Selected temporal traversal ===//

#include "ConfiguredReductionTraversal.h"
#include "Internal.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <limits>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

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
    llvm::ArrayRef<int64_t> parallelTileSizes,
    llvm::ArrayRef<unsigned> dimensionOrder, unsigned depth, mlir::Value output,
    llvm::ArrayRef<mlir::OpFoldResult> destinationBaseOffsets,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &sourceOffsets,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &localOffsets,
    llvm::SmallVectorImpl<int64_t> &tileSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (depth == dimensionOrder.size()) {
    mlir::FailureOr<mlir::Value> local = materializeConfiguredComputeTile(
        builder, scope, sourceCompute, sourceOffsets, tileSizes, loops,
        operationTemporalTiles, nestedTemporalTiles, failureReason,
        operationNodes);
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

  const unsigned dimension = dimensionOrder[depth];
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
    mlir::OpFoldResult oldSourceOffset = sourceOffsets[dimension];
    mlir::OpFoldResult oldLocalOffset = localOffsets[dimension];
    int64_t oldTileSize = tileSizes[dimension];
    sourceOffsets[dimension] = *sourceOffset;
    mlir::OpFoldResult destinationOffset = relativeOffset;
    if (!destinationBaseOffsets.empty()) {
      mlir::FailureOr<mlir::OpFoldResult> rebased = addConfiguredTileOffset(
          nestedBuilder, loc, destinationBaseOffsets[dimension],
          relativeOffset);
      if (mlir::failed(rebased)) {
        setFailureReason(failureReason,
                         "configured destination tile offset overflows index");
        sourceOffsets[dimension] = oldSourceOffset;
        return mlir::failure();
      }
      destinationOffset = *rebased;
    }
    localOffsets[dimension] = destinationOffset;
    tileSizes[dimension] = currentSize;
    mlir::FailureOr<mlir::Value> result =
        materializeConfiguredParallelTraversal(
            nestedBuilder, scope, root, sourceCompute, requestedOutputOffsets,
            requestedOutputSizes, parallelTileSizes, dimensionOrder, depth + 1,
            destination, destinationBaseOffsets, sourceOffsets, localOffsets,
            tileSizes, nestedLoops, operationTemporalTiles, nestedTemporalTiles,
            failureReason, operationNodes);
    tileSizes[dimension] = oldTileSize;
    localOffsets[dimension] = oldLocalOffset;
    sourceOffsets[dimension] = oldSourceOffset;
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
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
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
  llvm::SmallVector<std::optional<unsigned>, 4> resultForIterator(
      selected->iteratorTileSizes.size());
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
    resultForIterator[loopDimension.getPosition()] =
        static_cast<unsigned>(resultDimension);
    parallelTileSizes.push_back(tileSize);
    hasParallelSplit |= tileSize < requestedOutputSizes[resultDimension];
  }
  if (!selected->waveLoopOrder.empty()) {
    llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
        sourceCompute.getIteratorTypesArray();
    llvm::SmallVector<int64_t, 4> loopRanges =
        sourceCompute.getStaticLoopRanges();
    llvm::SmallVector<uint32_t, 4> activeOrder;
    for (uint32_t dimension : selected->waveLoopOrder) {
      if (dimension >= iteratorTypes.size() || dimension >= loopRanges.size())
        return mlir::failure();
      bool active = false;
      if (resultForIterator[dimension]) {
        unsigned resultDimension = *resultForIterator[dimension];
        active = selected->iteratorTileSizes[dimension] <
                 requestedOutputSizes[resultDimension];
      } else if (iteratorTypes[dimension] ==
                 mlir::utils::IteratorType::reduction) {
        active = selected->iteratorTileSizes[dimension] < loopRanges[dimension];
      }
      if (active)
        activeOrder.push_back(dimension);
    }
    llvm::SmallVector<uint32_t, 4> supportedOrder;
    for (uint32_t dimension : activeOrder)
      if (iteratorTypes[dimension] == mlir::utils::IteratorType::parallel)
        supportedOrder.push_back(dimension);
    for (uint32_t dimension : activeOrder)
      if (iteratorTypes[dimension] == mlir::utils::IteratorType::reduction)
        supportedOrder.push_back(dimension);
    if (activeOrder != supportedOrder) {
      setFailureReason(
          failureReason,
          "coupled producer order requires a separate TileRegion traversal");
      return mlir::failure();
    }
  }
  if (!hasParallelSplit) {
    mlir::FailureOr<mlir::Value> local = materializeConfiguredComputeTile(
        builder, scope, sourceCompute, requestedOutputOffsets,
        requestedOutputSizes, loops, operationTemporalTiles,
        nestedTemporalTiles, failureReason, operationNodes);
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
  llvm::SmallVector<unsigned, 4> dimensionOrder;
  if (!selected->waveLoopOrder.empty()) {
    for (uint32_t iteratorDimension : selected->waveLoopOrder) {
      for (auto [resultDimension, expression] :
           llvm::enumerate(maps[outputMapIndex].getResults())) {
        auto mapped = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!mapped || mapped.getPosition() != iteratorDimension ||
            parallelTileSizes[resultDimension] >=
                requestedOutputSizes[resultDimension])
          continue;
        dimensionOrder.push_back(static_cast<unsigned>(resultDimension));
      }
    }
    for (unsigned resultDimension = 0;
         resultDimension < requestedOutputSizes.size(); ++resultDimension)
      if (parallelTileSizes[resultDimension] <
              requestedOutputSizes[resultDimension] &&
          !llvm::is_contained(dimensionOrder, resultDimension)) {
        setFailureReason(
            failureReason,
            "parallel wave-loop order omits an active result iterator");
        return mlir::failure();
      }
  }
  for (unsigned resultDimension = 0;
       resultDimension < requestedOutputSizes.size(); ++resultDimension)
    if (!llvm::is_contained(dimensionOrder, resultDimension))
      dimensionOrder.push_back(resultDimension);

  mlir::Value output = outputDestination;
  if (!output)
    output = builder
                 .create<mlir::tensor::EmptyOp>(
                     root->getLoc(), requestedOutputSizes,
                     resultType.getElementType(), mlir::ValueRange{},
                     resultType.getEncoding())
                 .getResult();
  llvm::SmallVector<mlir::OpFoldResult, 4> sourceOffsets(
      requestedOutputSizes.size());
  llvm::SmallVector<mlir::OpFoldResult, 4> localOffsets(
      requestedOutputSizes.size());
  llvm::SmallVector<int64_t, 4> tileSizes(requestedOutputSizes.size());
  return materializeConfiguredParallelTraversal(
      builder, scope, root, sourceCompute, requestedOutputOffsets,
      requestedOutputSizes, parallelTileSizes, dimensionOrder, /*depth=*/0,
      output, destinationBaseOffsets, sourceOffsets, localOffsets, tileSizes,
      loops, operationTemporalTiles, nestedTemporalTiles, failureReason,
      operationNodes);
}

} // namespace wafer::tensor_program_to_tile_region
