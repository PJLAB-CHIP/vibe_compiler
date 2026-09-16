//===- ConvertStablehloGatherToTensor.cpp - Preserve indexed slices -----===//

#include "Wafer/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

#include <algorithm>
#include <optional>

namespace wafer {
#define GEN_PASS_DEF_CONVERTSTABLEHLOGATHERTOTENSORPASS
#include "Wafer/Conversion/WaferConversionPasses.h.inc"

namespace {
#ifdef WAFER_ENABLE_STABLEHLO
struct SliceGather {
  int64_t axis;
  mlir::RankedTensorType indicesType;
  mlir::RankedTensorType coordinatesType;
  mlir::RankedTensorType resultType;
  mlir::stablehlo::ReshapeOp resultReshape;
};

std::optional<SliceGather> describeSliceGather(mlir::stablehlo::GatherOp op) {
  auto source = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand().getType());
  auto indices =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getStartIndices().getType());
  auto result = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  if (!source || !indices || !result || !source.hasStaticShape() ||
      !indices.hasStaticShape() || !result.hasStaticShape() ||
      indices.getRank() == 0 || source.getRank() == 0)
    return std::nullopt;
  auto element = mlir::dyn_cast<mlir::IntegerType>(indices.getElementType());
  if (!element ||
      (element.getWidth() != 32 && element.getWidth() != 64))
    return std::nullopt;
  auto dims = op.getDimensionNumbers();
  if (!dims.getOperandBatchingDims().empty() ||
      !dims.getStartIndicesBatchingDims().empty() ||
      dims.getStartIndexMap().size() != 1 ||
      dims.getCollapsedSliceDims() != dims.getStartIndexMap())
    return std::nullopt;
  int64_t axis = dims.getStartIndexMap().front();
  auto sliceSizes = op.getSliceSizes();
  for (auto [dimension, extent] : llvm::enumerate(source.getShape()))
    if (extent <= 0 || sliceSizes[dimension] !=
                           (static_cast<int64_t>(dimension) == axis ? 1 : extent))
      return std::nullopt;
  llvm::SmallVector<int64_t> coordinateShape(indices.getShape());
  if (dims.getIndexVectorDim() == indices.getRank()) {
    coordinateShape.push_back(1);
  } else if (dims.getIndexVectorDim() != indices.getRank() - 1 ||
             indices.getShape().back() != 1) {
    return std::nullopt;
  }
  int64_t batchRank = coordinateShape.size() - 1;
  for (auto [dimension, offset] : llvm::enumerate(dims.getOffsetDims()))
    if (offset != batchRank + static_cast<int64_t>(dimension))
      return std::nullopt;
  auto loweredElement = element.isSignless()
                            ? element
                            : mlir::IntegerType::get(op.getContext(), 64);
  auto loweredIndices = mlir::RankedTensorType::get(indices.getShape(), loweredElement);
  auto coordinates = mlir::RankedTensorType::get(coordinateShape, loweredElement);
  if (mlir::tensor::GatherOp::inferResultType(source, coordinates, {axis},
                                             /*rankReduced=*/true) != result)
    return std::nullopt;
  mlir::stablehlo::ReshapeOp resultReshape;
  if (op.getResult().hasOneUse()) {
    auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(
        *op.getResult().user_begin());
    auto reshapedType = reshape
                            ? mlir::dyn_cast<mlir::RankedTensorType>(reshape.getType())
                            : mlir::RankedTensorType{};
    int64_t sliceRank = source.getRank() - 1;
    if (reshapedType && reshapedType.hasStaticShape() &&
        reshapedType.getRank() > sliceRank &&
        reshapedType.getShape().take_back(sliceRank) ==
            result.getShape().take_back(sliceRank)) {
      llvm::SmallVector<int64_t> reshapedCoordinates(
          reshapedType.getShape().drop_back(sliceRank));
      reshapedCoordinates.push_back(1);
      auto newCoordinates =
          mlir::RankedTensorType::get(reshapedCoordinates, loweredElement);
      if ((newCoordinates == loweredIndices ||
           mlir::getReassociationIndicesForReshape(loweredIndices, newCoordinates)) &&
          mlir::tensor::GatherOp::inferResultType(source, newCoordinates, {axis},
                                                   true) == reshapedType) {
        coordinates = newCoordinates;
        result = reshapedType;
        resultReshape = reshape;
      }
    }
  }
  return SliceGather{axis, loweredIndices, coordinates, result, resultReshape};
}

struct ConvertSliceGather final
    : mlir::OpConversionPattern<mlir::stablehlo::GatherOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::GatherOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto description = describeSliceGather(op);
    if (!description)
      return rewriter.notifyMatchFailure(op, "not a static full-slice gather");
    auto sourceType =
        mlir::cast<mlir::RankedTensorType>(adaptor.getOperand().getType());
    auto element =
        mlir::cast<mlir::IntegerType>(description->indicesType.getElementType());
    int64_t maximum = std::min(
        sourceType.getDimSize(description->axis) - 1,
        llvm::APInt::getSignedMaxValue(element.getWidth()).getSExtValue());
    auto loc = op.getLoc();
    mlir::Value indexValues = adaptor.getStartIndices();
    auto originalIndicesType = mlir::cast<mlir::RankedTensorType>(indexValues.getType());
    auto originalElement = mlir::cast<mlir::IntegerType>(originalIndicesType.getElementType());
    if (originalElement.isUnsigned() && originalElement.getWidth() == 64) {
      // Clamp before signed conversion: ui64 values above INT64_MAX must not
      // become negative and select the first row instead of the last row.
      auto upper = rewriter.create<mlir::stablehlo::ConstantOp>(
          loc, mlir::DenseElementsAttr::get(originalIndicesType,
              rewriter.getIntegerAttr(originalElement,
                                      sourceType.getDimSize(description->axis) - 1)));
      indexValues = rewriter.create<mlir::stablehlo::MinOp>(loc, indexValues, upper);
    }
    if (originalIndicesType != description->indicesType)
      indexValues = rewriter.create<mlir::stablehlo::ConvertOp>(
          loc, description->indicesType, indexValues);
    mlir::Value empty = rewriter.create<mlir::tensor::EmptyOp>(
        loc, description->indicesType.getShape(), element);
    auto identity =
        rewriter.getMultiDimIdentityMap(description->indicesType.getRank());
    auto clamped = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{description->indicesType},
        mlir::ValueRange{indexValues}, mlir::ValueRange{empty},
        llvm::ArrayRef<mlir::AffineMap>{identity, identity},
        llvm::SmallVector<mlir::utils::IteratorType>(
            description->indicesType.getRank(),
            mlir::utils::IteratorType::parallel),
        [&](mlir::OpBuilder &builder, mlir::Location bodyLoc,
            mlir::ValueRange arguments) {
          mlir::Value zero = builder.create<mlir::arith::ConstantOp>(
              bodyLoc, builder.getIntegerAttr(element, 0));
          mlir::Value upper = builder.create<mlir::arith::ConstantOp>(
              bodyLoc, builder.getIntegerAttr(element, maximum));
          mlir::Value lowerClamped = builder.create<mlir::arith::MaxSIOp>(
              bodyLoc, arguments.front(), zero);
          mlir::Value index = builder.create<mlir::arith::MinSIOp>(
              bodyLoc, lowerClamped, upper);
          builder.create<mlir::linalg::YieldOp>(bodyLoc, index);
        });
    mlir::Value coordinates = clamped.getResult(0);
    if (description->coordinatesType != description->indicesType) {
      auto reassociation = mlir::getReassociationIndicesForReshape(
          description->indicesType, description->coordinatesType);
      assert(reassociation && "full-slice gather coordinate reshape was verified");
      if (description->indicesType.getRank() < description->coordinatesType.getRank())
        coordinates = rewriter.create<mlir::tensor::ExpandShapeOp>(
            loc, description->coordinatesType, coordinates, *reassociation);
      else
        coordinates = rewriter.create<mlir::tensor::CollapseShapeOp>(
            loc, description->coordinatesType, coordinates, *reassociation);
    }
    auto gather = rewriter.create<mlir::tensor::GatherOp>(
        loc, description->resultType, adaptor.getOperand(), coordinates,
        llvm::ArrayRef<int64_t>{description->axis}, /*unique=*/false);
    if (description->resultReshape) {
      // Reassociate only the batch coordinates. The actual sole reshape
      // consumer now has identical input/output shapes and folds normally.
      rewriter.modifyOpInPlace(description->resultReshape, [&] {
        description->resultReshape.getOperandMutable().assign(gather.getResult());
      });
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, gather.getResult());
    }
    return mlir::success();
  }
};
#endif

struct ConvertStablehloGatherToTensorPass final
    : impl::ConvertStablehloGatherToTensorPassBase<
          ConvertStablehloGatherToTensorPass> {
  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<mlir::stablehlo::GatherOp>(
        [](mlir::stablehlo::GatherOp op) { return !describeSliceGather(op); });
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertSliceGather>(&getContext());
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))) ||
        mlir::failed(mlir::verify(getOperation())))
      signalPassFailure();
#endif
  }
};
} // namespace
} // namespace wafer
