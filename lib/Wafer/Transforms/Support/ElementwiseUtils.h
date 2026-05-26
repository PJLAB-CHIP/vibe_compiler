//===- ElementwiseUtils.h - Elementwise generic helpers --------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_ELEMENTWISEUTILS_H
#define WAFER_TRANSFORMS_ELEMENTWISEUTILS_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <optional>

namespace wafer {

std::optional<wafer::ComputeElementwiseKind>
matchElementwiseGeneric(mlir::linalg::GenericOp generic);

bool isLimitedBroadcastElementwiseGeneric(mlir::linalg::GenericOp generic);

} // namespace wafer

#endif // WAFER_TRANSFORMS_ELEMENTWISEUTILS_H
