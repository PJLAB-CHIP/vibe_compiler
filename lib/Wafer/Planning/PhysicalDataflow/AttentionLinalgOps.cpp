//===- AttentionLinalgOps.cpp - Attention Linalg construction ---------===//

#include "Wafer/Planning/PhysicalDataflow/AttentionLinalgOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include <limits>
#include <optional>

namespace wafer::compiler::detail::attention_linalg {

mlir::AffineMap mapForDims(mlir::MLIRContext *context, unsigned loopRank,
                           llvm::ArrayRef<unsigned> dims) {
  llvm::SmallVector<mlir::AffineExpr, 6> expressions;
  for (unsigned dim : dims)
    expressions.push_back(mlir::getAffineDimExpr(dim, context));
  return mlir::AffineMap::get(loopRank, 0, expressions, context);
}

llvm::SmallVector<unsigned, 6> sequence(unsigned count) {
  llvm::SmallVector<unsigned, 6> result;
  for (unsigned index = 0; index < count; ++index)
    result.push_back(index);
  return result;
}

mlir::Value createEmpty(mlir::OpBuilder &builder, mlir::Location loc,
                        llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  return builder.create<mlir::tensor::EmptyOp>(loc, shape, elementType)
      .getResult();
}

mlir::FailureOr<mlir::Value>
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

mlir::Value createFill(mlir::OpBuilder &builder, mlir::Location loc,
                       llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                       mlir::Value scalar) {
  return builder
      .create<mlir::linalg::FillOp>(
          loc, scalar, createEmpty(builder, loc, shape, elementType))
      .getResult(0);
}

mlir::Value cloneLinalg(mlir::OpBuilder &builder, mlir::linalg::LinalgOp source,
                        mlir::ValueRange inputs, mlir::Value init) {
  auto resultType = mlir::cast<mlir::RankedTensorType>(init.getType());
  llvm::SmallVector<mlir::Value, 6> operands(inputs.begin(), inputs.end());
  operands.push_back(init);
  mlir::Operation *operation = mlir::clone(
      builder, source.getOperation(), mlir::TypeRange{resultType}, operands);
  return operation->getResult(0);
}

mlir::FailureOr<mlir::Value>
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

mlir::FailureOr<mlir::Value>
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

mlir::Value createPointwise(mlir::OpBuilder &builder, mlir::Location loc,
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

mlir::Value createSlice(mlir::OpBuilder &builder, mlir::Location loc,
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

} // namespace wafer::compiler::detail::attention_linalg
