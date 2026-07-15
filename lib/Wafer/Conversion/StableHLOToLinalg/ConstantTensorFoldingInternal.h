#ifndef WAFER_LIB_CONVERSION_STABLEHLOTOLINALG_CONSTANTTENSORFOLDINGINTERNAL_H
#define WAFER_LIB_CONVERSION_STABLEHLOTOLINALG_CONSTANTTENSORFOLDINGINTERNAL_H

namespace mlir {
class Operation;
}

namespace wafer::stablehlo_normalization {

void foldConstantTensorOps(mlir::Operation *root);

} // namespace wafer::stablehlo_normalization

#endif
