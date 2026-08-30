//===- AttentionMath.h - Shared attention scalar operations -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_ATTENTIONMATH_H
#define WAFER_TRANSFORMS_LINALG_ATTENTIONMATH_H

#include "mlir/IR/Builders.h"

namespace wafer::compiler::detail {

mlir::Value castAttentionFloatScalar(mlir::Value value, mlir::Type targetType,
                                     mlir::OpBuilder &builder,
                                     mlir::Location location);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_LINALG_ATTENTIONMATH_H
