//===- AttentionAlternative.cpp - Attention graph alternative -----------===//

#include "AttentionAlternative.h"
#include "AttentionAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/ProducerTileFusion.h"
#include "Wafer/Planning/PhysicalDataflow/AttentionLinalgOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include <algorithm>
#include <optional>

using namespace wafer;

namespace wafer::tensor_program_alternatives {

using tensor_program_to_tile_region::eraseDeadTensorProgramClosure;
using tensor_program_to_tile_region::fuseTensorProgramProducerSlices;
using namespace wafer::compiler::detail::attention_linalg;
namespace {

static void setFailureReason(std::string *failureReason,
                             llvm::StringRef reason) {
  if (failureReason)
    *failureReason = reason.str();
}

} // namespace

mlir::FailureOr<mlir::Value> materializeAttentionTile(
    mlir::OpBuilder &builder, mlir::func::FuncOp function,
    mlir::Operation *root,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    AttentionProgramAlgorithm implementation,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  mlir::FailureOr<AttentionSemantics> matched =
      analyzeAttentionSemantics(root, failureReason);
  if (mlir::failed(matched))
    return mlir::failure();
  AttentionSemantics &pattern = *matched;
  int64_t keyValueTileSize = 0;
  int64_t splitCount = 1;
  if (implementation == AttentionProgramAlgorithm::Online) {
    if (candidateReductionTileSizes.size() != 1)
      return setFailureReason(failureReason,
                              "online attention requires one K/V block size"),
             mlir::failure();
    keyValueTileSize = candidateReductionTileSizes.front();
  } else {
    if (candidateReductionTileSizes.size() != 2)
      return setFailureReason(
                 failureReason,
                 "split K/V attention requires one K/V block size and one "
                 "partition "
                 "count"),
             mlir::failure();
    keyValueTileSize = candidateReductionTileSizes[0];
    splitCount = candidateReductionTileSizes[1];
    mlir::ModuleOp module = root->getParentOfType<mlir::ModuleOp>();
    mlir::FailureOr<DecodeAttentionSemantics> decode =
        analyzeDecodeAttentionSemantics(module, failureReason);
    if (mlir::failed(decode))
      return mlir::failure();
    if (decode->attention.output.getOperation() !=
            pattern.output.getOperation() ||
        decode->attention.observableOutput.getDefiningOp() != root)
      return setFailureReason(
                 failureReason,
                 "split K/V root is not the proven decode attention "
                 "output"),
             mlir::failure();
  }
  const unsigned rank = pattern.outputType.getRank();
  const unsigned prefixRank = rank - 2;
  if (candidateTileOffsets.size() != rank ||
      candidateTileSizes.size() != rank || keyValueTileSize <= 0 ||
      keyValueTileSize > pattern.reductionExtent || splitCount <= 0 ||
      splitCount > pattern.reductionExtent ||
      (implementation == AttentionProgramAlgorithm::SplitKeyValue &&
       splitCount == 1))
    return setFailureReason(failureReason,
                            "attention recurrence requires the complete "
                            "output domain and "
                            "bounded nonempty K/V partitions"),
           mlir::failure();

  mlir::Location loc = root->getLoc();
  auto accumulationType =
      mlir::cast<mlir::FloatType>(pattern.scoreType.getElementType());
  auto valueStorageType =
      mlir::cast<mlir::FloatType>(pattern.valueType.getElementType());
  auto outputStorageType =
      mlir::cast<mlir::FloatType>(pattern.outputType.getElementType());
  llvm::SmallVector<int64_t, 6> rowShape(candidateTileSizes.begin(),
                                         candidateTileSizes.end() - 1);
  llvm::SmallVector<int64_t, 6> outputShape(candidateTileSizes.begin(),
                                            candidateTileSizes.end());
  mlir::AffineMap rowIdentity = mapForDims(
      builder.getContext(), rowShape.size(), sequence(rowShape.size()));
  mlir::AffineMap outputIdentity = mapForDims(
      builder.getContext(), outputShape.size(), sequence(outputShape.size()));
  llvm::SmallVector<unsigned, 6> rowProjection = sequence(prefixRank + 1);
  mlir::AffineMap outputRowMap =
      mapForDims(builder.getContext(), outputShape.size(), rowProjection);

  struct OnlineAttentionState {
    mlir::Value maximum;
    mlir::Value sum;
    mlir::Value unnormalizedOutput;
  };
  auto materializeBlock =
      [&](mlir::OpBuilder &blockBuilder, mlir::OpFoldResult kOffset,
          int64_t kSize, std::optional<OnlineAttentionState> previousState,
          llvm::MutableArrayRef<mlir::LoopLikeOpInterface> activeLoops)
      -> mlir::FailureOr<OnlineAttentionState> {
    OnlineAttentionState state = previousState.value_or(OnlineAttentionState{});
    llvm::SmallVector<mlir::OpFoldResult, 6> scoreOffsets;
    llvm::SmallVector<int64_t, 6> scoreShape;
    for (unsigned dim = 0; dim < prefixRank + 1; ++dim) {
      scoreOffsets.push_back(candidateTileOffsets[dim]);
      scoreShape.push_back(candidateTileSizes[dim]);
    }
    scoreOffsets.push_back(kOffset);
    scoreShape.push_back(kSize);

    llvm::SmallVector<mlir::OpFoldResult, 6> valueOffsets;
    llvm::SmallVector<int64_t, 6> valueShape;
    for (unsigned dim = 0; dim < prefixRank; ++dim) {
      valueOffsets.push_back(candidateTileOffsets[dim]);
      valueShape.push_back(candidateTileSizes[dim]);
    }
    valueOffsets.push_back(kOffset);
    valueShape.push_back(kSize);
    valueOffsets.push_back(candidateTileOffsets.back());
    valueShape.push_back(candidateTileSizes.back());

    mlir::Value scoreTile = createSlice(blockBuilder, loc, pattern.scores,
                                        scoreOffsets, scoreShape);
    mlir::Value valueTile = createSlice(blockBuilder, loc, pattern.values,
                                        valueOffsets, valueShape);

    mlir::Value localMaxInit = createFill(blockBuilder, loc, rowShape,
                                          accumulationType, pattern.maxInit);
    mlir::Value localMax =
        cloneLinalg(blockBuilder, pattern.rowMax, mlir::ValueRange{scoreTile},
                    localMaxInit);
    mlir::Operation *localMaxOp = localMax.getDefiningOp();

    mlir::Value updatedMax = localMax;
    mlir::Value previousScale;
    if (state.maximum) {
      updatedMax = createPointwise(blockBuilder, loc, {state.maximum, localMax},
                                   {rowIdentity, rowIdentity}, rowShape,
                                   accumulationType, PointwiseKind::Maximum);
      mlir::Value previousDelta =
          createPointwise(blockBuilder, loc, {state.maximum, updatedMax},
                          {rowIdentity, rowIdentity}, rowShape,
                          accumulationType, PointwiseKind::Subtract);
      previousScale =
          createPointwise(blockBuilder, loc, {previousDelta}, {rowIdentity},
                          rowShape, accumulationType, PointwiseKind::Exp);
    }

    // The online recurrence keeps O unnormalized. Each K/V block is shifted by
    // the running maximum, only the previous state is rescaled, and the
    // division by l is delayed until this independent range is complete.
    mlir::Value broadcastMax =
        createPointwise(blockBuilder, loc, {updatedMax},
                        {mapForDims(blockBuilder.getContext(),
                                    scoreShape.size(), rowProjection)},
                        scoreShape, accumulationType, PointwiseKind::Identity);
    mlir::AffineMap scoreIdentity =
        mapForDims(blockBuilder.getContext(), scoreShape.size(),
                   sequence(scoreShape.size()));
    mlir::Value shifted =
        createPointwise(blockBuilder, loc, {scoreTile, broadcastMax},
                        {scoreIdentity, scoreIdentity}, scoreShape,
                        accumulationType, PointwiseKind::Subtract);
    mlir::Value exponent =
        createPointwise(blockBuilder, loc, {shifted}, {scoreIdentity},
                        scoreShape, accumulationType, PointwiseKind::Exp);

    mlir::Value currentSumInit = createFill(blockBuilder, loc, rowShape,
                                            accumulationType, pattern.sumInit);
    mlir::Value currentSum =
        cloneLinalg(blockBuilder, pattern.rowSum, mlir::ValueRange{exponent},
                    currentSumInit);
    mlir::FailureOr<mlir::Value> storedExponent =
        createFloatConvert(blockBuilder, loc, exponent, valueStorageType);
    if (mlir::failed(storedExponent) || valueStorageType != outputStorageType) {
      setFailureReason(
          failureReason,
          "online attention value contraction storage dtypes are not "
          "representable");
      return mlir::failure();
    }
    mlir::Value currentOutputInit = createFill(
        blockBuilder, loc, outputShape, outputStorageType, pattern.outputInit);
    mlir::FailureOr<mlir::Value> storedOutput = createLogicalValueContraction(
        blockBuilder, loc, *storedExponent, valueTile, currentOutputInit);
    if (mlir::failed(storedOutput)) {
      setFailureReason(failureReason,
                       "online attention value contraction is invalid");
      return mlir::failure();
    }
    mlir::FailureOr<mlir::Value> accumulatedOutput =
        createFloatConvert(blockBuilder, loc, *storedOutput, accumulationType);
    if (mlir::failed(accumulatedOutput)) {
      setFailureReason(
          failureReason,
          "online attention output accumulator conversion is invalid");
      return mlir::failure();
    }

    if (mlir::failed(fuseTensorProgramProducerSlices(
            localMaxOp, pattern.rowMax.getOperation(), function, activeLoops,
            failureReason)))
      return mlir::failure();

    if (!state.maximum) {
      state.maximum = updatedMax;
      state.sum = currentSum;
      state.unnormalizedOutput = *accumulatedOutput;
      return state;
    }
    mlir::Value nextSum = createPointwise(
        blockBuilder, loc, {previousScale, state.sum, currentSum},
        {rowIdentity, rowIdentity, rowIdentity}, rowShape, accumulationType,
        PointwiseKind::ScaleAdd);
    mlir::Value nextOutput = createPointwise(
        blockBuilder, loc,
        {previousScale, state.unnormalizedOutput, *accumulatedOutput},
        {outputRowMap, outputIdentity, outputIdentity}, outputShape,
        accumulationType, PointwiseKind::ScaleAdd);
    state.sum = blockBuilder
                    .create<mlir::bufferization::MaterializeInDestinationOp>(
                        loc, nextSum, state.sum)
                    .getResult();
    state.unnormalizedOutput =
        blockBuilder
            .create<mlir::bufferization::MaterializeInDestinationOp>(
                loc, nextOutput, state.unnormalizedOutput)
            .getResult();
    state.maximum =
        blockBuilder
            .create<mlir::bufferization::MaterializeInDestinationOp>(
                loc, updatedMax, state.maximum)
            .getResult();
    return state;
  };

  auto materializeState =
      [&](int64_t kBegin,
          int64_t kEnd) -> mlir::FailureOr<OnlineAttentionState> {
    if (kBegin < 0 || kBegin >= kEnd || kEnd > pattern.reductionExtent) {
      setFailureReason(failureReason,
                       "attention recurrence produced an empty K/V partition");
      return mlir::failure();
    }

    const int64_t firstSize = std::min(keyValueTileSize, kEnd - kBegin);
    mlir::FailureOr<OnlineAttentionState> state = materializeBlock(
        builder, builder.getIndexAttr(kBegin), firstSize, std::nullopt, loops);
    if (mlir::failed(state))
      return mlir::failure();

    const int64_t remainingBegin = kBegin + firstSize;
    const int64_t remainingExtent = kEnd - remainingBegin;
    const int64_t tailSize = remainingExtent % keyValueTileSize;
    const int64_t fullTileEnd = kEnd - tailSize;
    if (remainingBegin < fullTileEnd) {
      auto lower =
          builder.create<mlir::arith::ConstantIndexOp>(loc, remainingBegin);
      auto upper =
          builder.create<mlir::arith::ConstantIndexOp>(loc, fullTileEnd);
      auto step =
          builder.create<mlir::arith::ConstantIndexOp>(loc, keyValueTileSize);
      auto loop = builder.create<mlir::scf::ForOp>(
          loc, lower, upper, step,
          mlir::ValueRange{state->maximum, state->sum,
                           state->unnormalizedOutput});
      llvm::SmallVector<mlir::LoopLikeOpInterface, 5> activeLoops(loops.begin(),
                                                                  loops.end());
      activeLoops.push_back(
          mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
      mlir::OpBuilder bodyBuilder =
          mlir::OpBuilder::atBlockBegin(loop.getBody());
      OnlineAttentionState carried{loop.getRegionIterArgs()[0],
                                   loop.getRegionIterArgs()[1],
                                   loop.getRegionIterArgs()[2]};
      mlir::FailureOr<OnlineAttentionState> next =
          materializeBlock(bodyBuilder, loop.getInductionVar(),
                           keyValueTileSize, carried, activeLoops);
      if (mlir::failed(next))
        return mlir::failure();
      bodyBuilder.create<mlir::scf::YieldOp>(
          loc,
          mlir::ValueRange{next->maximum, next->sum, next->unnormalizedOutput});
      builder.setInsertionPointAfter(loop);
      state = OnlineAttentionState{loop.getResult(0), loop.getResult(1),
                                   loop.getResult(2)};
    }

    if (tailSize > 0) {
      state = materializeBlock(builder, builder.getIndexAttr(fullTileEnd),
                               tailSize, *state, loops);
      if (mlir::failed(state))
        return mlir::failure();
    }
    return state;
  };

  mlir::Value normalized;
  if (implementation == AttentionProgramAlgorithm::Online) {
    mlir::FailureOr<OnlineAttentionState> state =
        materializeState(/*kBegin=*/0, pattern.reductionExtent);
    if (mlir::failed(state))
      return mlir::failure();
    normalized =
        createPointwise(builder, loc, {state->unnormalizedOutput, state->sum},
                        {outputIdentity, outputRowMap}, outputShape,
                        accumulationType, PointwiseKind::Divide);
  } else {
    // Materialize every split before building the merge. The split subgraphs
    // are independent SSA siblings and can therefore be assigned to distinct
    // workers/regions; the final merge is the sole cross-split consumer.
    llvm::SmallVector<mlir::Value, 8> partialLses;
    llvm::SmallVector<mlir::Value, 8> partialOutputs;
    int64_t nextBegin = 0;
    const int64_t baseSize = pattern.reductionExtent / splitCount;
    const int64_t largerSplitCount = pattern.reductionExtent % splitCount;
    for (int64_t split = 0; split < splitCount; ++split) {
      int64_t splitSize = baseSize + (split < largerSplitCount ? 1 : 0);
      int64_t splitEnd = nextBegin + splitSize;
      mlir::FailureOr<OnlineAttentionState> partial =
          materializeState(nextBegin, splitEnd);
      if (mlir::failed(partial))
        return mlir::failure();
      mlir::Value logSum =
          createPointwise(builder, loc, {partial->sum}, {rowIdentity}, rowShape,
                          accumulationType, PointwiseKind::Log);
      partialLses.push_back(createPointwise(
          builder, loc, {partial->maximum, logSum}, {rowIdentity, rowIdentity},
          rowShape, accumulationType, PointwiseKind::Add));
      partialOutputs.push_back(createPointwise(
          builder, loc, {partial->unnormalizedOutput, partial->sum},
          {outputIdentity, outputRowMap}, outputShape, accumulationType,
          PointwiseKind::Divide));
      nextBegin = splitEnd;
    }

    mlir::Value mergedMax = partialLses.front();
    mlir::Value zero = createPointwise(
        builder, loc, {mergedMax, mergedMax}, {rowIdentity, rowIdentity},
        rowShape, accumulationType, PointwiseKind::Subtract);
    mlir::Value mergedWeight =
        createPointwise(builder, loc, {zero}, {rowIdentity}, rowShape,
                        accumulationType, PointwiseKind::Exp);
    mlir::Value mergedOutput = partialOutputs.front();
    for (int64_t split = 1; split < splitCount; ++split) {
      mlir::Value updatedMax =
          createPointwise(builder, loc, {mergedMax, partialLses[split]},
                          {rowIdentity, rowIdentity}, rowShape,
                          accumulationType, PointwiseKind::Maximum);
      mlir::Value previousDelta = createPointwise(
          builder, loc, {mergedMax, updatedMax}, {rowIdentity, rowIdentity},
          rowShape, accumulationType, PointwiseKind::Subtract);
      mlir::Value currentDelta =
          createPointwise(builder, loc, {partialLses[split], updatedMax},
                          {rowIdentity, rowIdentity}, rowShape,
                          accumulationType, PointwiseKind::Subtract);
      mlir::Value previousScale =
          createPointwise(builder, loc, {previousDelta}, {rowIdentity},
                          rowShape, accumulationType, PointwiseKind::Exp);
      mlir::Value currentScale =
          createPointwise(builder, loc, {currentDelta}, {rowIdentity}, rowShape,
                          accumulationType, PointwiseKind::Exp);
      mlir::Value currentZero = createPointwise(
          builder, loc, {partialLses[split], partialLses[split]},
          {rowIdentity, rowIdentity}, rowShape, accumulationType,
          PointwiseKind::Subtract);
      mlir::Value currentWeight =
          createPointwise(builder, loc, {currentZero}, {rowIdentity}, rowShape,
                          accumulationType, PointwiseKind::Exp);
      mergedWeight = createPointwise(
          builder, loc,
          {previousScale, mergedWeight, currentScale, currentWeight},
          {rowIdentity, rowIdentity, rowIdentity, rowIdentity}, rowShape,
          accumulationType, PointwiseKind::WeightedAdd);
      mergedOutput = createPointwise(
          builder, loc,
          {previousScale, mergedOutput, currentScale, partialOutputs[split]},
          {outputRowMap, outputIdentity, outputRowMap, outputIdentity},
          outputShape, accumulationType, PointwiseKind::WeightedAdd);
      mergedMax = updatedMax;
    }
    normalized = createPointwise(builder, loc, {mergedOutput, mergedWeight},
                                 {outputIdentity, outputRowMap}, outputShape,
                                 accumulationType, PointwiseKind::Divide);
  }
  mlir::FailureOr<mlir::Value> storedResult =
      createFloatConvert(builder, loc, normalized, outputStorageType);
  if (mlir::failed(storedResult))
    return setFailureReason(
               failureReason,
               "online attention result conversion is not representable"),
           mlir::failure();
  return *storedResult;
}

mlir::LogicalResult materializeAttentionProgramAlternative(
    mlir::func::FuncOp function, AttentionProgramAlgorithm implementation,
    int64_t keyValueBlockSize, int64_t keyValuePartitionCount,
    std::string *failureReason) {
  if (!function || !function.getBody().hasOneBlock() ||
      function.getNumResults() == 0 || keyValueBlockSize <= 0 ||
      keyValuePartitionCount <= 0 ||
      (implementation == AttentionProgramAlgorithm::Online &&
       keyValuePartitionCount != 1) ||
      (implementation == AttentionProgramAlgorithm::SplitKeyValue &&
       keyValuePartitionCount <= 1)) {
    setFailureReason(failureReason,
                     "attention alternative requires one function body, "
                     "results, and valid recurrence parameters");
    return mlir::failure();
  }
  mlir::ModuleOp module = function->getParentOfType<mlir::ModuleOp>();
  mlir::FailureOr<AttentionSemantics> semantics = mlir::failure();
  if (implementation == AttentionProgramAlgorithm::SplitKeyValue) {
    mlir::FailureOr<DecodeAttentionSemantics> decode =
        analyzeDecodeAttentionSemantics(module, failureReason);
    if (mlir::failed(decode))
      return mlir::failure();
    semantics = decode->attention;
  } else {
    semantics = analyzeAttentionSemantics(module, failureReason);
    if (mlir::failed(semantics))
      return mlir::failure();
  }
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp || semantics->outputIndex >= returnOp.getNumOperands()) {
    setFailureReason(failureReason,
                     "attention output index is outside the function boundary");
    return mlir::failure();
  }
  mlir::Operation *root = semantics->output.getOperation();
  mlir::RankedTensorType resultType = semantics->outputType;
  if (!root || root->getNumResults() != 1 || !resultType ||
      !resultType.hasStaticShape() ||
      keyValueBlockSize > semantics->reductionExtent ||
      keyValuePartitionCount > semantics->reductionExtent) {
    setFailureReason(failureReason,
                     "attention alternative parameters exceed the proven "
                     "current-SSA domain");
    return mlir::failure();
  }
  llvm::SmallVector<int64_t, 2> reductionTileSizes{keyValueBlockSize};
  if (implementation == AttentionProgramAlgorithm::SplitKeyValue)
    reductionTileSizes.push_back(keyValuePartitionCount);
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets(resultType.getRank(),
                                                   builder.getIndexAttr(0));
  llvm::SmallVector<int64_t, 4> sizes(resultType.getShape().begin(),
                                      resultType.getShape().end());
  llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
  mlir::FailureOr<mlir::Value> output = materializeAttentionTile(
      builder, function, root, offsets, sizes, reductionTileSizes,
      implementation, loops, failureReason);
  if (mlir::failed(output))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> physicalOutput = createExactStaticReshape(
      builder, root->getLoc(), *output, semantics->physicalOutputType);
  if (mlir::failed(physicalOutput)) {
    setFailureReason(failureReason,
                     "attention logical output cannot be exactly "
                     "reassociated to its downstream physical type");
    return mlir::failure();
  }
  root->getResult(0).replaceAllUsesWith(*physicalOutput);
  root->erase();
  eraseDeadTensorProgramClosure(function);
  return mlir::success();
}

} // namespace wafer::tensor_program_alternatives
