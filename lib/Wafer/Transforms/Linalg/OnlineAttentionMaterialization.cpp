//===- OnlineAttentionMaterialization.cpp - Actual online state ----------===//

#include "OnlineAttentionMaterialization.h"

#include "AttentionMath.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace wafer::compiler::detail {
namespace {

mlir::FailureOr<mlir::Value>
materializeAttentionOperandTile(mlir::Value value, mlir::AffineMap map,
                                llvm::ArrayRef<mlir::OpFoldResult> offsets,
                                llvm::ArrayRef<mlir::OpFoldResult> sizes,
                                mlir::OpBuilder &builder) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || map.getNumResults() != static_cast<unsigned>(type.getRank()))
    return mlir::failure();
  llvm::SmallVector<mlir::OpFoldResult, 6> operandOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 6> operandSizes;
  for (mlir::AffineExpr expression : map.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= offsets.size() ||
        dimension.getPosition() >= sizes.size())
      return mlir::failure();
    operandOffsets.push_back(offsets[dimension.getPosition()]);
    operandSizes.push_back(sizes[dimension.getPosition()]);
  }
  llvm::SmallVector<mlir::OpFoldResult, 6> strides(type.getRank(),
                                                   builder.getIndexAttr(1));
  bool isFullTile = true;
  for (auto [dimension, offset, size] :
       llvm::enumerate(operandOffsets, operandSizes)) {
    std::optional<int64_t> constantOffset = mlir::getConstantIntValue(offset);
    std::optional<int64_t> constantSize = mlir::getConstantIntValue(size);
    isFullTile &= constantOffset && *constantOffset == 0 && constantSize &&
                  *constantSize == type.getDimSize(dimension);
  }
  if (isFullTile)
    return value;
  return builder
      .create<mlir::tensor::ExtractSliceOp>(
          value.getLoc(), value, operandOffsets, operandSizes, strides)
      .getResult();
}

mlir::FailureOr<mlir::RankedTensorType>
getAttentionStateType(mlir::AffineMap map,
                      llvm::ArrayRef<mlir::OpFoldResult> sizes,
                      mlir::Type elementType) {
  llvm::SmallVector<int64_t, 4> shape;
  for (mlir::AffineExpr expression : map.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= sizes.size())
      return mlir::failure();
    std::optional<int64_t> size =
        mlir::getConstantIntValue(sizes[dimension.getPosition()]);
    if (!size || *size <= 0)
      return mlir::failure();
    shape.push_back(*size);
  }
  return mlir::RankedTensorType::get(shape, elementType);
}

mlir::Value createFilledTensor(mlir::Location location,
                               mlir::RankedTensorType type, mlir::Value scalar,
                               mlir::OpBuilder &builder) {
  mlir::Value empty = builder
                          .create<mlir::tensor::EmptyOp>(
                              location, type.getShape(), type.getElementType())
                          .getResult();
  return builder.create<mlir::linalg::FillOp>(location, scalar, empty)
      .getResult(0);
}

} // namespace

mlir::FailureOr<OnlineAttentionState> materializeOnlineAttentionTile(
    LinalgExtAttentionOp source, mlir::Value query, mlir::Value key,
    mlir::Value value, mlir::Value scale, mlir::Value mask,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, mlir::OpBuilder &builder) {
  if (!source || !mlir::isa<mlir::RankedTensorType>(query.getType()) ||
      !mlir::isa<mlir::RankedTensorType>(key.getType()) ||
      !mlir::isa<mlir::RankedTensorType>(value.getType()))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> queryTile = materializeAttentionOperandTile(
      query, source.getQueryMap(), offsets, sizes, builder);
  mlir::FailureOr<mlir::Value> keyTile = materializeAttentionOperandTile(
      key, source.getKeyMap(), offsets, sizes, builder);
  mlir::FailureOr<mlir::Value> valueTile = materializeAttentionOperandTile(
      value, source.getValueMap(), offsets, sizes, builder);
  mlir::FailureOr<mlir::Value> maskTile = mlir::failure();
  if (mask)
    maskTile = materializeAttentionOperandTile(mask, *source.getMaskMap(),
                                               offsets, sizes, builder);
  if (mlir::failed(queryTile) || mlir::failed(keyTile) ||
      mlir::failed(valueTile) || (mask && mlir::failed(maskTile)))
    return mlir::failure();

  CoupledReductionDescription description =
      source.getCoupledReductionDescription();
  auto findComponent = [&](CoupledReductionComponentKind kind)
      -> const CoupledReductionComponent * {
    auto found =
        llvm::find_if(description.components, [&](const auto &component) {
          return component.kind == kind;
        });
    return found == description.components.end() ? nullptr : &*found;
  };
  const CoupledReductionComponent *accumulatorComponent =
      findComponent(CoupledReductionComponentKind::Accumulator);
  const CoupledReductionComponent *maximumComponent =
      findComponent(CoupledReductionComponentKind::Maximum);
  const CoupledReductionComponent *sumComponent =
      findComponent(CoupledReductionComponentKind::Sum);
  if (!accumulatorComponent || !maximumComponent || !sumComponent)
    return mlir::failure();
  mlir::FailureOr<mlir::RankedTensorType> accumulatorType =
      getAttentionStateType(accumulatorComponent->indexingMap, sizes,
                            accumulatorComponent->elementType);
  mlir::FailureOr<mlir::RankedTensorType> maximumType = getAttentionStateType(
      maximumComponent->indexingMap, sizes, maximumComponent->elementType);
  mlir::FailureOr<mlir::RankedTensorType> sumType = getAttentionStateType(
      sumComponent->indexingMap, sizes, sumComponent->elementType);
  if (mlir::failed(accumulatorType) || mlir::failed(maximumType) ||
      mlir::failed(sumType))
    return mlir::failure();

  mlir::Location location = source.getLoc();
  mlir::Value accumulatorIdentity = mlir::arith::getIdentityValue(
      mlir::arith::AtomicRMWKind::addf, accumulatorType->getElementType(),
      builder, location);
  mlir::Value maximumIdentity = mlir::arith::getIdentityValue(
      mlir::arith::AtomicRMWKind::maximumf, maximumType->getElementType(),
      builder, location);
  mlir::Value sumIdentity = mlir::arith::getIdentityValue(
      mlir::arith::AtomicRMWKind::addf, sumType->getElementType(), builder,
      location);
  if (!accumulatorIdentity || !maximumIdentity || !sumIdentity)
    return mlir::failure();
  mlir::Value accumulator = createFilledTensor(location, *accumulatorType,
                                               accumulatorIdentity, builder);
  mlir::Value maximum =
      createFilledTensor(location, *maximumType, maximumIdentity, builder);
  mlir::Value sum =
      createFilledTensor(location, *sumType, sumIdentity, builder);

  llvm::SmallVector<mlir::Attribute, 8> maps{
      mlir::AffineMapAttr::get(source.getQueryMap()),
      mlir::AffineMapAttr::get(source.getKeyMap()),
      mlir::AffineMapAttr::get(source.getValueMap()),
      mlir::AffineMapAttr::get(source.getScaleMap())};
  if (mask)
    maps.push_back(mlir::AffineMapAttr::get(*source.getMaskMap()));
  maps.append({mlir::AffineMapAttr::get(accumulatorComponent->indexingMap),
               mlir::AffineMapAttr::get(maximumComponent->indexingMap),
               mlir::AffineMapAttr::get(sumComponent->indexingMap)});
  auto online = builder.create<LinalgExtOnlineAttentionOp>(
      location,
      mlir::TypeRange{accumulator.getType(), maximum.getType(), sum.getType()},
      *queryTile, *keyTile, *valueTile, scale, mask ? *maskTile : mlir::Value{},
      accumulator, maximum, sum, builder.getArrayAttr(maps));
  mlir::IRMapping scoreMapping;
  source.getScoreRegion().cloneInto(&online.getScoreRegion(), scoreMapping);
  return OnlineAttentionState{online.getUpdatedAccumulator(),
                              online.getUpdatedMaximum(),
                              online.getUpdatedSum()};
}

mlir::FailureOr<mlir::Value>
materializeOnlineAttentionFinalize(LinalgExtAttentionOp source,
                                   const OnlineAttentionState &state,
                                   mlir::OpBuilder &builder) {
  if (!state.accumulator || !state.sum)
    return mlir::failure();
  CoupledReductionDescription description =
      source.getCoupledReductionDescription();
  auto sumComponent =
      llvm::find_if(description.components, [](const auto &component) {
        return component.kind == CoupledReductionComponentKind::Sum;
      });
  if (sumComponent == description.components.end())
    return mlir::failure();
  llvm::SmallVector<mlir::AffineMap, 3> maps = mlir::compressUnusedDims(
      {source.getOutputMap(), sumComponent->indexingMap,
       source.getOutputMap()});
  if (maps.size() != 3)
    return mlir::failure();
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
      maps.front().getNumDims(), mlir::utils::IteratorType::parallel);
  auto generic = builder.create<mlir::linalg::GenericOp>(
      source.getLoc(), mlir::TypeRange{state.accumulator.getType()},
      mlir::ValueRange{state.accumulator, state.sum},
      mlir::ValueRange{state.accumulator}, maps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location location,
          mlir::ValueRange arguments) {
        mlir::Value accumulator = castAttentionFloatScalar(
            arguments[0], arguments[1].getType(), nestedBuilder, location);
        mlir::Value normalized =
            accumulator ? nestedBuilder.create<mlir::arith::DivFOp>(
                              location, accumulator, arguments[1])
                        : mlir::Value{};
        mlir::Value result =
            normalized
                ? castAttentionFloatScalar(normalized, arguments[2].getType(),
                                           nestedBuilder, location)
                : mlir::Value{};
        if (!result)
          result = arguments[2];
        nestedBuilder.create<mlir::linalg::YieldOp>(location, result);
      });
  return generic.getResult(0);
}

mlir::FailureOr<OnlineAttentionState> materializeOnlineAttentionStateMerge(
    LinalgExtAttentionOp source, const OnlineAttentionState &left,
    const OnlineAttentionState &right, mlir::OpBuilder &builder) {
  if (!left.accumulator || !left.maximum || !left.sum || !right.accumulator ||
      !right.maximum || !right.sum ||
      left.accumulator.getType() != right.accumulator.getType() ||
      left.maximum.getType() != right.maximum.getType() ||
      left.sum.getType() != right.sum.getType())
    return mlir::failure();
  mlir::Location location = source.getLoc();
  auto rowType = mlir::cast<mlir::RankedTensorType>(left.maximum.getType());
  mlir::AffineMap rowIdentity = mlir::AffineMap::getMultiDimIdentityMap(
      rowType.getRank(), builder.getContext());
  llvm::SmallVector<mlir::utils::IteratorType, 4> rowIterators(
      rowType.getRank(), mlir::utils::IteratorType::parallel);
  llvm::SmallVector<mlir::AffineMap, 5> rowMaps(5, rowIdentity);

  auto maximum = builder.create<mlir::linalg::GenericOp>(
      location, mlir::TypeRange{rowType},
      mlir::ValueRange{left.maximum, right.maximum},
      mlir::ValueRange{left.maximum},
      llvm::ArrayRef<mlir::AffineMap>{rowIdentity, rowIdentity, rowIdentity},
      rowIterators,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
          mlir::ValueRange arguments) {
        mlir::Value value = nestedBuilder.create<mlir::arith::MaximumFOp>(
            nestedLocation, arguments[0], arguments[1]);
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLocation, value);
      });

  auto createScale = [&](mlir::Value localMaximum) {
    return builder.create<mlir::linalg::GenericOp>(
        location, mlir::TypeRange{rowType},
        mlir::ValueRange{localMaximum, maximum.getResult(0)},
        mlir::ValueRange{localMaximum},
        llvm::ArrayRef<mlir::AffineMap>{rowIdentity, rowIdentity, rowIdentity},
        rowIterators,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
            mlir::ValueRange arguments) {
          mlir::Value difference = nestedBuilder.create<mlir::arith::SubFOp>(
              nestedLocation, arguments[0], arguments[1]);
          mlir::Value scale = nestedBuilder.create<mlir::math::ExpOp>(
              nestedLocation, difference);
          nestedBuilder.create<mlir::linalg::YieldOp>(nestedLocation, scale);
        });
  };
  mlir::linalg::GenericOp leftScale = createScale(left.maximum);
  mlir::linalg::GenericOp rightScale = createScale(right.maximum);

  auto sum = builder.create<mlir::linalg::GenericOp>(
      location, mlir::TypeRange{left.sum.getType()},
      mlir::ValueRange{left.sum, right.sum, leftScale.getResult(0),
                       rightScale.getResult(0)},
      mlir::ValueRange{left.sum}, rowMaps, rowIterators,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
          mlir::ValueRange arguments) {
        mlir::Value scaledLeft = nestedBuilder.create<mlir::arith::MulFOp>(
            nestedLocation, arguments[0], arguments[2]);
        mlir::Value scaledRight = nestedBuilder.create<mlir::arith::MulFOp>(
            nestedLocation, arguments[1], arguments[3]);
        mlir::Value value = nestedBuilder.create<mlir::arith::AddFOp>(
            nestedLocation, scaledLeft, scaledRight);
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLocation, value);
      });

  CoupledReductionDescription description =
      source.getCoupledReductionDescription();
  auto accumulatorComponent =
      llvm::find_if(description.components, [](const auto &component) {
        return component.kind == CoupledReductionComponentKind::Accumulator;
      });
  auto maximumComponent =
      llvm::find_if(description.components, [](const auto &component) {
        return component.kind == CoupledReductionComponentKind::Maximum;
      });
  if (accumulatorComponent == description.components.end() ||
      maximumComponent == description.components.end())
    return mlir::failure();
  llvm::SmallVector<mlir::AffineMap, 5> accumulatorMaps =
      mlir::compressUnusedDims(
          {accumulatorComponent->indexingMap, accumulatorComponent->indexingMap,
           maximumComponent->indexingMap, maximumComponent->indexingMap,
           accumulatorComponent->indexingMap});
  llvm::SmallVector<mlir::utils::IteratorType, 4> accumulatorIterators(
      accumulatorMaps.front().getNumDims(),
      mlir::utils::IteratorType::parallel);
  auto accumulator = builder.create<mlir::linalg::GenericOp>(
      location, mlir::TypeRange{left.accumulator.getType()},
      mlir::ValueRange{left.accumulator, right.accumulator,
                       leftScale.getResult(0), rightScale.getResult(0)},
      mlir::ValueRange{left.accumulator}, accumulatorMaps, accumulatorIterators,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
          mlir::ValueRange arguments) {
        mlir::Value leftWeight =
            castAttentionFloatScalar(arguments[2], arguments[0].getType(),
                                     nestedBuilder, nestedLocation);
        mlir::Value rightWeight =
            castAttentionFloatScalar(arguments[3], arguments[1].getType(),
                                     nestedBuilder, nestedLocation);
        mlir::Value scaledLeft = nestedBuilder.create<mlir::arith::MulFOp>(
            nestedLocation, arguments[0], leftWeight);
        mlir::Value scaledRight = nestedBuilder.create<mlir::arith::MulFOp>(
            nestedLocation, arguments[1], rightWeight);
        mlir::Value value = nestedBuilder.create<mlir::arith::AddFOp>(
            nestedLocation, scaledLeft, scaledRight);
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLocation, value);
      });
  return OnlineAttentionState{accumulator.getResult(0), maximum.getResult(0),
                              sum.getResult(0)};
}

} // namespace wafer::compiler::detail
