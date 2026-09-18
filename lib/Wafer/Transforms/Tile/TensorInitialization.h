//===- TensorInitialization.h - Uniform tensor DPS conversion -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_TENSORINITIALIZATION_H
#define WAFER_TRANSFORMS_TILE_TENSORINITIALIZATION_H

#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
class Operation;
class RewriterBase;
} // namespace mlir

namespace wafer {

struct TensorInitializationStatistics {
  uint64_t pads = 0;
  uint64_t generates = 0;
};

/// Materialize the current uniform Pad/Generate computations as DPS before
/// layout assignment. Nonuniform or effectful computations remain for typed
/// admission by the caller. Replacements use its current-relation listener.
mlir::FailureOr<TensorInitializationStatistics>
lowerUniformTensorInitializers(mlir::RewriterBase &rewriter,
                               mlir::Operation *root);

} // namespace wafer

#endif // WAFER_TRANSFORMS_TILE_TENSORINITIALIZATION_H
