//===- MovementFusion.h - Tile MovementFusion -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_MOVEMENTFUSION_H
#define WAFER_TRANSFORMS_TILE_MOVEMENTFUSION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"

namespace wafer::compiler::detail {
/// Remove private layout staging around a DMA whose blocked physical order
/// already matches the selected SPM layout. No new layout is selected.
void fuseDMALayoutMovements(mlir::ModuleOp module, mlir::IRRewriter &rewriter);

/// Fuse a private transpose followed by layout conversion into one movement.
void fuseTransposeLayoutMovements(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations = {});

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_MOVEMENTFUSION_H
