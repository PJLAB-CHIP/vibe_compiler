//===- AttentionMath.cpp - Shared attention scalar operations ---------===//

#include "AttentionMath.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include <limits>

namespace wafer::compiler::detail {

mlir::Value computeAttentionExponential(mlir::Value value, mlir::Value maximum,
                                        mlir::OpBuilder &builder,
                                        mlir::Location location) {
  mlir::Value difference =
      builder.create<mlir::arith::SubFOp>(location, value, maximum);
  return builder.create<mlir::math::ExpOp>(location, difference);
}

mlir::Value materializeAttentionExponentialMaximum(mlir::Value maximum,
                                                   mlir::OpBuilder &builder,
                                                   mlir::Location location) {
  auto type = mlir::cast<mlir::RankedTensorType>(maximum.getType());
  auto identity = mlir::AffineMap::getMultiDimIdentityMap(type.getRank(),
                                                          builder.getContext());
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      type.getRank(), mlir::utils::IteratorType::parallel);
  auto rows = builder.create<mlir::linalg::GenericOp>(
      location, mlir::TypeRange{type}, mlir::ValueRange{maximum},
      mlir::ValueRange{maximum},
      llvm::ArrayRef<mlir::AffineMap>{identity, identity}, iterators,
      [&](mlir::OpBuilder &nested, mlir::Location loc,
          mlir::ValueRange values) {
        auto negativeInfinity = nested.create<mlir::arith::ConstantOp>(
            loc, nested.getFloatAttr(type.getElementType(),
                                     -std::numeric_limits<double>::infinity()));
        auto zero = nested.create<mlir::arith::ConstantOp>(
            loc, nested.getFloatAttr(type.getElementType(), 0.0));
        auto empty = nested.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, values[0], negativeInfinity);
        auto result =
            nested.create<mlir::arith::SelectOp>(loc, empty, zero, values[0]);
        nested.create<mlir::linalg::YieldOp>(loc, result.getResult());
      });
  // This is an exponent operand, not the maximum carried to the next block.
  return rows.getResult(0);
}

mlir::Value castAttentionFloatScalar(mlir::Value value, mlir::Type targetType,
                                     mlir::OpBuilder &builder,
                                     mlir::Location location) {
  if (value.getType() == targetType)
    return value;
  auto sourceType = mlir::cast<mlir::FloatType>(value.getType());
  auto destinationType = mlir::cast<mlir::FloatType>(targetType);
  if (sourceType.getWidth() < destinationType.getWidth())
    return builder.create<mlir::arith::ExtFOp>(location, targetType, value);
  if (sourceType.getWidth() > destinationType.getWidth())
    return builder.create<mlir::arith::TruncFOp>(location, targetType, value);
  mlir::Type intermediateType = builder.getF32Type();
  if (sourceType.getWidth() >= 32)
    intermediateType = builder.getF64Type();
  mlir::Value extended =
      builder.create<mlir::arith::ExtFOp>(location, intermediateType, value);
  return builder.create<mlir::arith::TruncFOp>(location, targetType, extended);
}

} // namespace wafer::compiler::detail
