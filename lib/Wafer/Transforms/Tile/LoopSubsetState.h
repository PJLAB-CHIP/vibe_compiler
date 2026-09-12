//===- LoopSubsetState.h - Local tensor recurrence state -------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_LOOPSUBSETSTATE_H
#define WAFER_TRANSFORMS_TILE_LOOPSUBSETSTATE_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail {

/// Promote proven invariant tensor subsets before layout assignment. All
/// relations are retargeted through the same rewriter as the current SSA.
mlir::LogicalResult
normalizeLoopSubsetState(mlir::ModuleOp module,
                         StructuredMaterializationRelations &relations);
} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_LOOPSUBSETSTATE_H
