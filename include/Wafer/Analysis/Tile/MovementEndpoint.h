//===- MovementEndpoint.h - Current physical movement endpoints -*- C++ -*-===//
#ifndef WAFER_ANALYSIS_TILE_MOVEMENTENDPOINT_H
#define WAFER_ANALYSIS_TILE_MOVEMENTENDPOINT_H

#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace wafer::analysis {
/// One invocation-local physical translation derived from current SSA.
struct MovementOffsetTerm {
  mlir::Value value;
  int64_t numerator = 0;
  int64_t denominator = 1;
};

struct MovementEndpoint {
  mlir::Value base;
  analysis::IndexRelation viewToBase;
  llvm::SmallVector<MovementOffsetTerm, 4> dynamicOffsets;

  mlir::MemRefType getType() const {
    return mlir::cast<mlir::MemRefType>(base.getType());
  }
};

/// Resolve explicit subviews to their actual physical root. Dynamic blocked
/// offsets require a proven translation; unsupported views return failure.
/// No address, allocation or descriptor is materialized by this query.
mlir::FailureOr<MovementEndpoint> resolveMovementEndpoint(mlir::Value value);
/// Static-only variant for consumers that have no dynamic offset operand.
mlir::FailureOr<MovementEndpoint>
resolveStaticMovementEndpoint(mlir::Value value);
} // namespace wafer::analysis
#endif
