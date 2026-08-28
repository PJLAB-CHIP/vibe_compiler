//===- NCCJoinPlacement.h - Current-IR required NCC joins ----*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_INSTR_NCCJOINPLACEMENT_H
#define WAFER_TRANSFORMS_INSTR_NCCJOINPLACEMENT_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer {

/// Place the minimum required NCC joins from one function's current typed
/// instruction, SSA and memory-effect IR. The caller owns the mutable
/// function and discards it if placement fails.
mlir::LogicalResult placeRequiredNCCJoins(mlir::func::FuncOp function);

/// Apply the same placement to every directly nested function.
mlir::LogicalResult placeRequiredNCCJoins(mlir::ModuleOp module);

/// Erase compiler-derived NCC joins and rebuild them solely from the current
/// Instr worker order, effects, aliases, ranges, tokens and observers.
mlir::LogicalResult rebuildRequiredNCCJoins(mlir::func::FuncOp function);

/// Apply the same rebuild to every directly nested function.
mlir::LogicalResult rebuildRequiredNCCJoins(mlir::ModuleOp module);

/// Rebuild joins required by one disposable isolated TileRegion query. This
/// does not replace function-level production placement.
mlir::LogicalResult
rebuildRequiredNCCJoinsForIsolatedTileRegion(TileRegionOp tileRegion);

namespace detail {

/// Place joins directly in one caller-owned private function. The caller must
/// discard the complete function on failure.
mlir::LogicalResult
placeRequiredNCCJoinsInPrivateFunction(mlir::func::FuncOp function);

} // namespace detail
} // namespace wafer

#endif // WAFER_TRANSFORMS_INSTR_NCCJOINPLACEMENT_H
