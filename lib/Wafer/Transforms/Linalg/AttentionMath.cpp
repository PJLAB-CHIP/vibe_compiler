//===- AttentionMath.cpp - Shared attention scalar operations ---------===//

#include "AttentionMath.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace wafer::compiler::detail {

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
