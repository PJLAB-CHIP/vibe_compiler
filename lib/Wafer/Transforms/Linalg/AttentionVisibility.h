//===- AttentionVisibility.h - Selected attention block visibility -*- C++
//-*-===//

#ifndef WAFER_TRANSFORMS_LINALG_ATTENTIONVISIBILITY_H
#define WAFER_TRANSFORMS_LINALG_ATTENTIONVISIBILITY_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class RewriterBase;
class Value;
} // namespace mlir
namespace wafer {
class TileRegionOp;

/// Prove an index ordering from current SSA, including the last reachable
/// induction value of positive, constant-step loops. Unknown remains false.
bool proveAttentionPositionOrder(mlir::Value lhs, mlir::Value rhs,
                                 bool strict = false);

/// Specialize actual, statically sized online tiles from their SSA positions.
/// Invisible blocks carry their old DPS states without reading Q/K/V. Fully
/// visible blocks have no position mask; boundary blocks retain it.
mlir::LogicalResult materializeAttentionVisibility(mlir::RewriterBase &rewriter,
                                                   TileRegionOp region);
} // namespace wafer

#endif
