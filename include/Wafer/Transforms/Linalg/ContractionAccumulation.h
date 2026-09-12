//===- ContractionAccumulation.h - Explicit wide accumulation -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_CONTRACTIONACCUMULATION_H
#define WAFER_TRANSFORMS_LINALG_CONTRACTIONACCUMULATION_H

#include "mlir/Support/LogicalResult.h"

namespace mlir::func {
class FuncOp;
}

namespace wafer {

/// Materializes F32 accumulators and the original output conversion before
/// spatial/temporal partitioning. Existing mixed-precision contractions and
/// noncanonical scalar payloads are unchanged. No input buffer is widened.
mlir::LogicalResult promoteContractionAccumulation(mlir::func::FuncOp function);

} // namespace wafer

#endif
