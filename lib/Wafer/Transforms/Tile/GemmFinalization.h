//===- GemmFinalization.h - Native GEMM output formats -----------*- C++ -*-===//
#ifndef WAFER_TRANSFORMS_TILE_GEMMFINALIZATION_H
#define WAFER_TRANSFORMS_TILE_GEMMFINALIZATION_H
#include "mlir/Support/LogicalResult.h"
namespace mlir {
class ModuleOp;
}
namespace wafer {
mlir::LogicalResult foldGemmOutputConversions(mlir::ModuleOp module);
}
#endif
