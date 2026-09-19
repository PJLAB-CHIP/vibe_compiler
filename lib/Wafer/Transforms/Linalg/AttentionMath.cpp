//===- AttentionMath.cpp - Shared attention scalar operations ---------===//

#include "AttentionMath.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include <limits>

namespace wafer::compiler::detail {

mlir::Value computeAttentionExponential(mlir::Value value, mlir::Value maximum,
                                        mlir::OpBuilder &builder,
                                        mlir::Location location) {
  mlir::Value negativeInfinity = builder.create<mlir::arith::ConstantOp>(
      location, builder.getFloatAttr(value.getType(),
                                     -std::numeric_limits<double>::infinity()));
  mlir::Value empty = builder.create<mlir::arith::CmpFOp>(
      location, mlir::arith::CmpFPredicate::OEQ, value, negativeInfinity);
  mlir::Value difference =
      builder.create<mlir::arith::SubFOp>(location, value, maximum);
  // Empty contributions must not poison a later visible block with -inf - -inf.
  // Finalization owns the source's zero-sum output behavior.
  // exp(-inf) is already zero; masking its argument avoids a second full
  // tensor select after the exponential without changing finite arithmetic.
  difference = builder.create<mlir::arith::SelectOp>(
      location, empty, negativeInfinity, difference);
  return builder.create<mlir::math::ExpOp>(location, difference);
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
