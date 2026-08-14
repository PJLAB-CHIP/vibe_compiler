#ifndef WAFER_LIB_CONVERSION_STABLEHLOTOLINALG_CONSTANTTENSORFOLDINGINTERNAL_H
#define WAFER_LIB_CONVERSION_STABLEHLOTOLINALG_CONSTANTTENSORFOLDINGINTERNAL_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace wafer::stablehlo_normalization {

mlir::LogicalResult foldConstantTensorOps(mlir::Operation *root);

} // namespace wafer::stablehlo_normalization

#endif
