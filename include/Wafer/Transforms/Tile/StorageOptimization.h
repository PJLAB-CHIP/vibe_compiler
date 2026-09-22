//===- StorageOptimization.h - Tile StorageOptimization -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_STORAGEOPTIMIZATION_H
#define WAFER_TRANSFORMS_TILE_STORAGEOPTIMIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"

namespace wafer::compiler::detail {
/// Keep private scalar reads on the compute path.
void preservePrivateScalarBroadcasts(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations = {});

/// Forward destinations, remove unobserved copies and resolve loop updates.
/// Excluded operations are live bindings in the caller's current IR epoch.
mlir::LogicalResult optimizeStorage(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations = {});

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_STORAGEOPTIMIZATION_H
