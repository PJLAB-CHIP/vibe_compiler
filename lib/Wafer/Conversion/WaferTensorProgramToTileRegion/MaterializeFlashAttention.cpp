//===- MaterializeFlashAttention.cpp - Flash strategy materialization ---===//

#include "AttentionSemantics.h"
#include "Internal.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"

#include <limits>

using namespace wafer;

namespace wafer {
#define GEN_PASS_DEF_MATERIALIZEFLASHATTENTION2PASS
#include "Wafer/Transforms/WaferPasses.h.inc"
} // namespace wafer

namespace wafer::tensor_program_to_tile_region {
namespace {

static mlir::AffineMap mapForDims(mlir::MLIRContext *context, unsigned loopRank,
                                  llvm::ArrayRef<unsigned> dims) {
  llvm::SmallVector<mlir::AffineExpr, 6> expressions;
  for (unsigned dim : dims)
    expressions.push_back(mlir::getAffineDimExpr(dim, context));
  return mlir::AffineMap::get(loopRank, 0, expressions, context);
}

static llvm::SmallVector<unsigned, 6> sequence(unsigned count) {
  llvm::SmallVector<unsigned, 6> result;
  for (unsigned index = 0; index < count; ++index)
    result.push_back(index);
  return result;
}

static mlir::Value createEmpty(mlir::OpBuilder &builder, mlir::Location loc,
                               llvm::ArrayRef<int64_t> shape,
                               mlir::Type elementType) {
  return builder.create<mlir::tensor::EmptyOp>(loc, shape, elementType)
      .getResult();
}

static mlir::Value createWritableDDRTensor(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::RankedTensorType type) {
  auto ddrType = mlir::MemRefType::get(
      type.getShape(), type.getElementType(), mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(builder.getContext(), MemorySpace::DDR,
                      MemLayout::Tensor));
  auto allocation = builder.create<mlir::memref::AllocOp>(loc, ddrType);
  return builder
      .create<mlir::bufferization::ToTensorOp>(
          loc, allocation.getResult(), /*restrict=*/true, /*writable=*/true)
      .getResult();
}

static mlir::FailureOr<mlir::Value>
createExactStaticReshape(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value source,
                         mlir::RankedTensorType targetType) {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      !targetType.hasStaticShape() ||
      sourceType.getElementType() != targetType.getElementType() ||
      sourceType.getNumElements() != targetType.getNumElements())
    return mlir::failure();
  if (sourceType == targetType)
    return source;
  std::optional<llvm::SmallVector<mlir::ReassociationIndices>> reassociation =
      mlir::getReassociationIndicesForReshape(sourceType, targetType);
  if (!reassociation || sourceType.getRank() == targetType.getRank())
    return mlir::failure();
  if (sourceType.getRank() > targetType.getRank())
    return builder
        .create<mlir::tensor::CollapseShapeOp>(loc, targetType, source,
                                               *reassociation)
        .getResult();
  return builder
      .create<mlir::tensor::ExpandShapeOp>(loc, targetType, source,
                                           *reassociation)
      .getResult();
}

static mlir::Value getExactStaticTensorViewSource(mlir::Value value) {
  mlir::Operation *operation = value.getDefiningOp();
  mlir::Value source;
  if (auto expand = mlir::dyn_cast_or_null<mlir::tensor::ExpandShapeOp>(
          operation))
    source = expand.getSrc();
  else if (auto collapse =
               mlir::dyn_cast_or_null<mlir::tensor::CollapseShapeOp>(
                   operation))
    source = collapse.getSrc();
  else if (auto cast =
               mlir::dyn_cast_or_null<mlir::tensor::CastOp>(operation))
    source = cast.getSource();
  if (!source || operation->getNumResults() != 1)
    return {};

  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      sourceType.getElementType() != resultType.getElementType() ||
      sourceType.getNumElements() != resultType.getNumElements())
    return {};
  return source;
}

static bool isExactStaticTensorViewChain(mlir::Value source,
                                         mlir::Value observable) {
  llvm::DenseSet<mlir::Value> visited;
  while (observable != source) {
    if (!observable || !visited.insert(observable).second)
      return false;
    observable = getExactStaticTensorViewSource(observable);
  }
  return true;
}

static mlir::Value createFill(mlir::OpBuilder &builder, mlir::Location loc,
                              llvm::ArrayRef<int64_t> shape,
                              mlir::Type elementType, mlir::Value scalar) {
  return builder
      .create<mlir::linalg::FillOp>(
          loc, scalar, createEmpty(builder, loc, shape, elementType))
      .getResult(0);
}

static mlir::Value cloneLinalg(mlir::OpBuilder &builder,
                               mlir::linalg::LinalgOp source,
                               mlir::ValueRange inputs, mlir::Value init) {
  auto resultType = mlir::cast<mlir::RankedTensorType>(init.getType());
  llvm::SmallVector<mlir::Value, 6> operands(inputs.begin(), inputs.end());
  operands.push_back(init);
  mlir::Operation *operation = mlir::clone(
      builder, source.getOperation(), mlir::TypeRange{resultType}, operands);
  return operation->getResult(0);
}

static mlir::FailureOr<mlir::Value>
createFloatConvert(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value input, mlir::FloatType targetElementType) {
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto inputElementType =
      inputType ? mlir::dyn_cast<mlir::FloatType>(inputType.getElementType())
                : mlir::FloatType{};
  if (!inputType || !inputType.hasStaticShape() || !inputElementType)
    return mlir::failure();
  if (inputElementType == targetElementType)
    return input;
  if (inputElementType.getWidth() == targetElementType.getWidth())
    return mlir::failure();
  mlir::Value init =
      createEmpty(builder, loc, inputType.getShape(), targetElementType);
  mlir::AffineMap identity = mlir::AffineMap::getMultiDimIdentityMap(
      inputType.getRank(), builder.getContext());
  llvm::SmallVector<mlir::utils::IteratorType, 6> iterators(
      inputType.getRank(), mlir::utils::IteratorType::parallel);
  auto converted = builder.create<mlir::linalg::GenericOp>(
      loc, mlir::TypeRange{init.getType()}, mlir::ValueRange{input},
      mlir::ValueRange{init},
      llvm::ArrayRef<mlir::AffineMap>{identity, identity}, iterators,
      [&](mlir::OpBuilder &nested, mlir::Location nestedLoc,
          mlir::ValueRange arguments) {
        mlir::Value result =
            inputElementType.getWidth() < targetElementType.getWidth()
                ? nested
                      .create<mlir::arith::ExtFOp>(nestedLoc, targetElementType,
                                                   arguments[0])
                      .getResult()
                : nested
                      .create<mlir::arith::TruncFOp>(
                          nestedLoc, targetElementType, arguments[0])
                      .getResult();
        nested.create<mlir::linalg::YieldOp>(nestedLoc, result);
      });
  return converted.getResult(0);
}

static mlir::FailureOr<mlir::Value>
createLogicalValueContraction(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value probability, mlir::Value values,
                              mlir::Value init) {
  auto probabilityType =
      mlir::dyn_cast<mlir::RankedTensorType>(probability.getType());
  auto valueType = mlir::dyn_cast<mlir::RankedTensorType>(values.getType());
  auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(init.getType());
  if (!probabilityType || !valueType || !outputType ||
      probabilityType.getRank() < 2 ||
      probabilityType.getRank() != valueType.getRank() ||
      probabilityType.getRank() != outputType.getRank() ||
      probabilityType.getElementType() != valueType.getElementType() ||
      probabilityType.getElementType() != outputType.getElementType())
    return mlir::failure();
  if (!probabilityType.hasStaticShape() || !valueType.hasStaticShape() ||
      !outputType.hasStaticShape())
    return mlir::failure();

  const unsigned rank = probabilityType.getRank();
  const unsigned prefixRank = rank - 2;
  for (unsigned dim = 0; dim < prefixRank; ++dim) {
    const int64_t extent = probabilityType.getDimSize(dim);
    if (extent <= 0 || valueType.getDimSize(dim) != extent ||
        outputType.getDimSize(dim) != extent)
      return mlir::failure();
  }
  const int64_t queryExtent = probabilityType.getDimSize(rank - 2);
  const int64_t reductionExtent = probabilityType.getDimSize(rank - 1);
  const int64_t valueExtent = valueType.getDimSize(rank - 1);
  if (queryExtent <= 0 || reductionExtent <= 0 || valueExtent <= 0 ||
      valueType.getDimSize(rank - 2) != reductionExtent ||
      outputType.getDimSize(rank - 2) != queryExtent ||
      outputType.getDimSize(rank - 1) != valueExtent)
    return mlir::failure();

  if (rank == 2)
    return builder
        .create<mlir::linalg::MatmulOp>(
            loc, mlir::ValueRange{probability, values}, mlir::ValueRange{init})
        .getResult(0);
  if (rank == 3)
    return builder
        .create<mlir::linalg::BatchMatmulOp>(
            loc, mlir::ValueRange{probability, values}, mlir::ValueRange{init})
        .getResult(0);

  int64_t batchExtent = 1;
  for (unsigned dim = 0; dim < prefixRank; ++dim) {
    const int64_t extent = probabilityType.getDimSize(dim);
    if (batchExtent > std::numeric_limits<int64_t>::max() / extent)
      return mlir::failure();
    batchExtent *= extent;
  }
  llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation;
  llvm::SmallVector<int64_t, 4> batchGroup;
  for (unsigned dim = 0; dim < prefixRank; ++dim)
    batchGroup.push_back(dim);
  reassociation.push_back(std::move(batchGroup));
  reassociation.push_back({prefixRank});
  reassociation.push_back({prefixRank + 1});
  auto collapsedType = [&](int64_t first, int64_t second) {
    return mlir::RankedTensorType::get({batchExtent, first, second},
                                       probabilityType.getElementType());
  };
  mlir::Value collapsedProbability =
      builder
          .create<mlir::tensor::CollapseShapeOp>(
              loc, collapsedType(queryExtent, reductionExtent), probability,
              reassociation)
          .getResult();
  mlir::Value collapsedValues =
      builder
          .create<mlir::tensor::CollapseShapeOp>(
              loc, collapsedType(reductionExtent, valueExtent), values,
              reassociation)
          .getResult();
  mlir::Value collapsedInit =
      builder
          .create<mlir::tensor::CollapseShapeOp>(
              loc, collapsedType(queryExtent, valueExtent), init, reassociation)
          .getResult();
  mlir::Value collapsedResult =
      builder
          .create<mlir::linalg::BatchMatmulOp>(
              loc, mlir::ValueRange{collapsedProbability, collapsedValues},
              mlir::ValueRange{collapsedInit})
          .getResult(0);
  return builder
      .create<mlir::tensor::ExpandShapeOp>(loc, outputType, collapsedResult,
                                           reassociation)
      .getResult();
}

enum class PointwiseKind {
  Identity,
  Maximum,
  Add,
  Subtract,
  Multiply,
  Exp,
  Log,
  ScaleAdd,
  WeightedAdd,
  Divide
};

static mlir::Value createPointwise(mlir::OpBuilder &builder, mlir::Location loc,
                                   llvm::ArrayRef<mlir::Value> inputs,
                                   llvm::ArrayRef<mlir::AffineMap> maps,
                                   llvm::ArrayRef<int64_t> outputShape,
                                   mlir::Type elementType, PointwiseKind kind) {
  mlir::Value init = createEmpty(builder, loc, outputShape, elementType);
  llvm::SmallVector<mlir::AffineMap, 6> allMaps(maps.begin(), maps.end());
  allMaps.push_back(mapForDims(builder.getContext(), outputShape.size(),
                               sequence(outputShape.size())));
  llvm::SmallVector<mlir::utils::IteratorType, 6> iterators(
      outputShape.size(), mlir::utils::IteratorType::parallel);
  auto generic = builder.create<mlir::linalg::GenericOp>(
      loc, mlir::TypeRange{init.getType()}, inputs, mlir::ValueRange{init},
      allMaps, iterators,
      [&](mlir::OpBuilder &nested, mlir::Location nestedLoc,
          mlir::ValueRange arguments) {
        mlir::Value result;
        switch (kind) {
        case PointwiseKind::Identity:
          result = arguments[0];
          break;
        case PointwiseKind::Maximum:
          result = nested.create<mlir::arith::MaximumFOp>(
              nestedLoc, arguments[0], arguments[1]);
          break;
        case PointwiseKind::Add:
          result = nested.create<mlir::arith::AddFOp>(nestedLoc, arguments[0],
                                                      arguments[1]);
          break;
        case PointwiseKind::Subtract:
          result = nested.create<mlir::arith::SubFOp>(nestedLoc, arguments[0],
                                                      arguments[1]);
          break;
        case PointwiseKind::Multiply:
          result = nested.create<mlir::arith::MulFOp>(nestedLoc, arguments[0],
                                                      arguments[1]);
          break;
        case PointwiseKind::Exp:
          result = nested.create<mlir::math::ExpOp>(nestedLoc, arguments[0]);
          break;
        case PointwiseKind::Log:
          result = nested.create<mlir::math::LogOp>(nestedLoc, arguments[0]);
          break;
        case PointwiseKind::ScaleAdd: {
          mlir::Value scaled = nested.create<mlir::arith::MulFOp>(
              nestedLoc, arguments[0], arguments[1]);
          result = nested.create<mlir::arith::AddFOp>(nestedLoc, scaled,
                                                      arguments[2]);
          break;
        }
        case PointwiseKind::WeightedAdd: {
          mlir::Value left = nested.create<mlir::arith::MulFOp>(
              nestedLoc, arguments[0], arguments[1]);
          mlir::Value right = nested.create<mlir::arith::MulFOp>(
              nestedLoc, arguments[2], arguments[3]);
          result = nested.create<mlir::arith::AddFOp>(nestedLoc, left, right);
          break;
        }
        case PointwiseKind::Divide:
          result = nested.create<mlir::arith::DivFOp>(nestedLoc, arguments[0],
                                                      arguments[1]);
          break;
        }
        nested.create<mlir::linalg::YieldOp>(nestedLoc, result);
      });
  return generic.getResult(0);
}

static mlir::Value createSlice(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value source,
                               llvm::ArrayRef<mlir::OpFoldResult> offsets,
                               llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<mlir::OpFoldResult, 6> mixedSizes;
  llvm::SmallVector<mlir::OpFoldResult, 6> strides;
  for (int64_t size : sizes) {
    mixedSizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }
  auto sourceType = mlir::cast<mlir::RankedTensorType>(source.getType());
  auto sliceType = mlir::RankedTensorType::get(
      sizes, sourceType.getElementType(), sourceType.getEncoding());
  return builder
      .create<mlir::tensor::ExtractSliceOp>(loc, sliceType, source, offsets,
                                            mixedSizes, strides)
      .getResult();
}

} // namespace

} // namespace wafer::tensor_program_to_tile_region

mlir::FailureOr<mlir::Value>
wafer::tensor_program_to_tile_region::materializeCandidateFlashRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    AttentionImplementationKind implementation,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  mlir::FailureOr<AttentionSemantics> matched =
      analyzeAttentionSemantics(root, failureReason);
  if (mlir::failed(matched))
    return mlir::failure();
  AttentionSemantics &pattern = *matched;
  int64_t keyValueTileSize = 0;
  int64_t splitCount = 1;
  if (implementation == AttentionImplementationKind::Online) {
    if (candidateReductionTileSizes.size() != 1)
      return setFailureReason(failureReason,
                              "FlashAttention-2 requires one K/V tile size"),
             mlir::failure();
    keyValueTileSize = candidateReductionTileSizes.front();
  } else {
    if (candidateReductionTileSizes.size() != 2)
      return setFailureReason(
                 failureReason,
                 "FlashDecoding requires one K/V tile size and one split "
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
                 "FlashDecoding root is not the proven decode attention "
                 "output"),
             mlir::failure();
  }
  const unsigned rank = pattern.outputType.getRank();
  const unsigned prefixRank = rank - 2;
  if (candidateTileOffsets.size() != rank ||
      candidateTileSizes.size() != rank || keyValueTileSize <= 0 ||
      keyValueTileSize > pattern.reductionExtent || splitCount <= 0 ||
      splitCount > pattern.reductionExtent ||
      (implementation == AttentionImplementationKind::SplitKV &&
       splitCount == 1))
    return setFailureReason(failureReason,
                            "Flash attention requires exact output tiles and "
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

  struct FlashAttention2State {
    mlir::Value maximum;
    mlir::Value sum;
    mlir::Value unnormalizedOutput;
  };
  auto materializeBlock =
      [&](mlir::OpBuilder &blockBuilder, mlir::OpFoldResult kOffset,
          int64_t kSize, std::optional<FlashAttention2State> previousState,
          llvm::MutableArrayRef<mlir::LoopLikeOpInterface> activeLoops)
      -> mlir::FailureOr<FlashAttention2State> {
    FlashAttention2State state = previousState.value_or(FlashAttention2State{});
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

    // FlashAttention-2 keeps O unnormalized. Each K/V block is shifted by
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
          "FlashAttention-2 value contraction storage dtypes are not "
          "representable");
      return mlir::failure();
    }
    mlir::Value currentOutputInit = createFill(
        blockBuilder, loc, outputShape, outputStorageType, pattern.outputInit);
    mlir::FailureOr<mlir::Value> storedOutput = createLogicalValueContraction(
        blockBuilder, loc, *storedExponent, valueTile, currentOutputInit);
    if (mlir::failed(storedOutput)) {
      setFailureReason(failureReason,
                       "FlashAttention-2 logical value contraction is invalid");
      return mlir::failure();
    }
    mlir::FailureOr<mlir::Value> accumulatedOutput =
        createFloatConvert(blockBuilder, loc, *storedOutput, accumulationType);
    if (mlir::failed(accumulatedOutput)) {
      setFailureReason(
          failureReason,
          "FlashAttention-2 output accumulator conversion is invalid");
      return mlir::failure();
    }

    if (mlir::failed(fuseCandidateProducerSlices(
            localMaxOp, pattern.rowMax.getOperation(), scope, activeLoops,
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
          int64_t kEnd) -> mlir::FailureOr<FlashAttention2State> {
    if (kBegin < 0 || kBegin >= kEnd || kEnd > pattern.reductionExtent) {
      setFailureReason(failureReason,
                       "Flash attention produced an empty K/V partition");
      return mlir::failure();
    }

    const int64_t firstSize = std::min(keyValueTileSize, kEnd - kBegin);
    mlir::FailureOr<FlashAttention2State> state = materializeBlock(
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
      FlashAttention2State carried{loop.getRegionIterArgs()[0],
                                   loop.getRegionIterArgs()[1],
                                   loop.getRegionIterArgs()[2]};
      mlir::FailureOr<FlashAttention2State> next =
          materializeBlock(bodyBuilder, loop.getInductionVar(),
                           keyValueTileSize, carried, activeLoops);
      if (mlir::failed(next))
        return mlir::failure();
      bodyBuilder.create<mlir::scf::YieldOp>(
          loc,
          mlir::ValueRange{next->maximum, next->sum, next->unnormalizedOutput});
      builder.setInsertionPointAfter(loop);
      state = FlashAttention2State{loop.getResult(0), loop.getResult(1),
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
  if (implementation == AttentionImplementationKind::Online) {
    mlir::FailureOr<FlashAttention2State> state =
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
      mlir::FailureOr<FlashAttention2State> partial =
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
               "FlashAttention-2 result conversion is not representable"),
           mlir::failure();
  return *storedResult;
}

namespace wafer::tensor_program_to_tile_region {
namespace {

static mlir::FailureOr<mlir::Value> materializeLoopedFlashTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> outputTileSizes,
    llvm::ArrayRef<int64_t> reductionTileSizes,
    AttentionImplementationKind implementation, unsigned dim,
    mlir::Value output,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets,
    llvm::SmallVectorImpl<int64_t> &sizes,
    llvm::SmallVectorImpl<mlir::LoopLikeOpInterface> &loops,
    std::string *failureReason) {
  if (dim == shape.size()) {
    mlir::FailureOr<mlir::Value> tile = materializeCandidateFlashRootTileValue(
        builder, scope, root, offsets, sizes, reductionTileSizes,
        implementation, loops, failureReason);
    if (mlir::failed(tile))
      return mlir::failure();
    return insertCandidateRootTile(builder, root->getLoc(), *tile, output,
                                   offsets, sizes);
  }

  mlir::Location loc = root->getLoc();
  if (shape[dim] == outputTileSizes[dim]) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(shape[dim]);
    mlir::FailureOr<mlir::Value> complete = materializeLoopedFlashTraversal(
        builder, scope, root, shape, outputTileSizes, reductionTileSizes,
        implementation, dim + 1, output, offsets, sizes, loops,
        failureReason);
    sizes.pop_back();
    offsets.pop_back();
    return complete;
  }
  int64_t tailSize = shape[dim] % outputTileSizes[dim];
  int64_t mainEnd = shape[dim] - tailSize;
  mlir::Value currentOutput = output;
  if (mainEnd > 0) {
    auto lower = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto upper = builder.create<mlir::arith::ConstantIndexOp>(loc, mainEnd);
    auto step =
        builder.create<mlir::arith::ConstantIndexOp>(loc, outputTileSizes[dim]);
    auto loop = builder.create<mlir::scf::ForOp>(
        loc, lower, upper, step, mlir::ValueRange{currentOutput});
    offsets.push_back(loop.getInductionVar());
    sizes.push_back(outputTileSizes[dim]);
    loops.push_back(mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
    mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
    mlir::FailureOr<mlir::Value> next = materializeLoopedFlashTraversal(
        bodyBuilder, scope, root, shape, outputTileSizes, reductionTileSizes,
        implementation, dim + 1, loop.getRegionIterArgs().front(), offsets,
        sizes, loops, failureReason);
    loops.pop_back();
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(next))
      return mlir::failure();
    bodyBuilder.create<mlir::scf::YieldOp>(loc, *next);
    builder.setInsertionPointAfter(loop);
    currentOutput = loop.getResult(0);
  }
  if (tailSize > 0) {
    offsets.push_back(builder.getIndexAttr(mainEnd));
    sizes.push_back(tailSize);
    mlir::FailureOr<mlir::Value> tail = materializeLoopedFlashTraversal(
        builder, scope, root, shape, outputTileSizes, reductionTileSizes,
        implementation, dim + 1, currentOutput, offsets, sizes, loops,
        failureReason);
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(tail))
      return mlir::failure();
    currentOutput = *tail;
  }
  return currentOutput;
}

} // namespace

mlir::LogicalResult materializeCompleteFlashTraversal(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> outputTileSizes,
    llvm::ArrayRef<int64_t> reductionTileSizes,
    AttentionImplementationKind implementation, std::string *failureReason) {
  if (scope.getOutputCount() == 0 || outputTileSizes.empty() ||
      reductionTileSizes.empty()) {
    setFailureReason(failureReason,
                     "Flash traversal requires results and positive typed "
                     "tile parameters");
    return mlir::failure();
  }
  mlir::ModuleOp module =
      scope.getFunction()->getParentOfType<mlir::ModuleOp>();
  mlir::FailureOr<AttentionSemantics> semantics = mlir::failure();
  if (implementation == AttentionImplementationKind::SplitKV) {
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
  if (semantics->outputIndex >= scope.getReturn().getNumOperands()) {
    setFailureReason(failureReason,
                     "Flash attention output index is outside the boundary");
    return mlir::failure();
  }
  mlir::Operation *root = semantics->output.getOperation();
  mlir::RankedTensorType resultType = semantics->outputType;
  if (!root || root->getNumResults() != 1 || !resultType ||
      !resultType.hasStaticShape() ||
      outputTileSizes.size() != static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "Flash traversal result and output tile ranks differ");
    return mlir::failure();
  }
  uint64_t tileCount = 0;
  if (wafer::detail::checkedStaticTileProduct(resultType.getShape(),
                                              outputTileSizes, tileCount) !=
      wafer::detail::CheckedStaticTileProductStatus::Success) {
    setFailureReason(failureReason,
                     "Flash traversal output tile domain is invalid");
    return mlir::failure();
  }

  mlir::OpBuilder builder(root);
  auto returnOp = scope.getReturn();
  bool isDirectPublicOutput = isExactStaticTensorViewChain(
      root->getResult(0), returnOp.getOperand(semantics->outputIndex));
  mlir::Value logicalDestination;
  if (isDirectPublicOutput) {
    mlir::FailureOr<mlir::Value> boundary = getCandidateOutputBoundary(
        scope, semantics->outputIndex, failureReason);
    if (mlir::failed(boundary))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> logicalBoundary = createExactStaticReshape(
        builder, root->getLoc(), *boundary, resultType);
    if (mlir::failed(logicalBoundary)) {
      setFailureReason(
          failureReason,
          "Flash attention output boundary cannot be exactly reassociated "
          "to its logical recurrence domain");
      return mlir::failure();
    }
    logicalDestination = *logicalBoundary;
  } else {
    // An internal attention result has no public output buffer to update.
    // Materialize one ordinary compiler-owned tensor version so dynamic loop
    // offsets remain explicit and the existing residency/search machinery can
    // later keep, spill or fuse the connection based on its actual cost.
    logicalDestination =
        createWritableDDRTensor(builder, root->getLoc(), resultType);
  }
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
  mlir::FailureOr<mlir::Value> output = materializeLoopedFlashTraversal(
      builder, scope, root, resultType.getShape(), outputTileSizes,
      reductionTileSizes, implementation, /*dim=*/0, logicalDestination,
      offsets, sizes, loops, failureReason);
  if (mlir::failed(output))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> physicalOutput = createExactStaticReshape(
      builder, root->getLoc(), *output, semantics->physicalOutputType);
  if (mlir::failed(physicalOutput)) {
    setFailureReason(failureReason,
                     "Flash attention logical output cannot be exactly "
                     "reassociated to its downstream physical type");
    return mlir::failure();
  }
  root->getResult(0).replaceAllUsesWith(*physicalOutput);
  root->erase();
  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region

namespace wafer {
namespace {

static mlir::FailureOr<llvm::SmallVector<int64_t, 6>>
parseFlashOutputTileSizes(llvm::StringRef text) {
  llvm::SmallVector<int64_t, 6> result;
  text = text.trim();
  if (text.empty())
    return mlir::failure();
  while (!text.empty()) {
    auto [part, rest] = text.split(',');
    int64_t value = 0;
    if (part.trim().empty() || part.trim().getAsInteger(10, value) ||
        value <= 0)
      return mlir::failure();
    result.push_back(value);
    text = rest;
  }
  return result;
}

struct MaterializeFlashAttention2Pass
    : public impl::MaterializeFlashAttention2PassBase<
          MaterializeFlashAttention2Pass> {
  using impl::MaterializeFlashAttention2PassBase<
      MaterializeFlashAttention2Pass>::MaterializeFlashAttention2PassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    auto tileSizes = parseFlashOutputTileSizes(outputTileSizes);
    if (mlir::failed(tileSizes) || keyValueTileSize <= 0) {
      module.emitError()
          << "flash_attention_materialization_failed: output and K/V tile "
             "sizes must be positive";
      return signalPassFailure();
    }

    mlir::func::FuncOp function =
        tensor_program_to_tile_region::findSingleStandaloneTensorProgram(
            module);
    if (!function) {
      module.emitError()
          << "flash_attention_materialization_failed: expected one "
             "standalone tensor program";
      return signalPassFailure();
    }
    std::string failureReason;
    tensor_program_to_tile_region::TensorProgramScope scope(function);
    llvm::SmallVector<int64_t, 1> reductionTileSizes{keyValueTileSize};
    if (mlir::failed(
            tensor_program_to_tile_region::materializeCompleteFlashTraversal(
                scope, *tileSizes, reductionTileSizes,
                tensor_program_to_tile_region::AttentionImplementationKind::
                    Online,
                &failureReason))) {
      module.emitError() << "flash_attention_materialization_failed: "
                         << failureReason;
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace wafer
