//===- AttentionVisibility.h - Selected attention block visibility -*- C++
//-*-===//

#ifndef WAFER_TRANSFORMS_LINALG_ATTENTIONVISIBILITY_H
#define WAFER_TRANSFORMS_LINALG_ATTENTIONVISIBILITY_H

#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
class RewriterBase;
class Operation;
class Value;
} // namespace mlir
namespace wafer {
class TileRegionOp;

/// Prove an index ordering from current SSA, including the last reachable
/// induction value of positive, constant-step loops. Unknown remains false.
bool proveAttentionPositionOrder(mlir::Value lhs, mlir::Value rhs,
                                 bool strict = false);

/// A bounded arithmetic progression of relative positions. Values outside the
/// requested interval have the same mask as the corresponding endpoint.
struct AttentionPositionRange {
  int64_t first;
  int64_t last;
  int64_t step;

  int64_t size() const { return (last - first) / step + 1; }
};

/// Describe lhs-rhs at an actual use, using ancestor branches and loop strides.
/// The requested endpoints denote the all-masked/all-visible saturation points.
AttentionPositionRange getAttentionPositionRange(mlir::Value lhs,
                                                 mlir::Value rhs,
                                                 mlir::Operation *use,
                                                 int64_t lower, int64_t upper);

/// Specialize actual, statically sized online tiles from their SSA positions.
/// Invisible blocks carry their old DPS states without reading Q/K/V. Fully
/// visible blocks have no position mask; boundary blocks retain it.
mlir::LogicalResult materializeAttentionVisibility(mlir::RewriterBase &rewriter,
                                                   TileRegionOp region);
} // namespace wafer

#endif
