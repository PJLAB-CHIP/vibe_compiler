//===- StorageInitialization.h - Private buffer initialization --*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_STORAGEINITIALIZATION_H
#define WAFER_TRANSFORMS_TILE_STORAGEINITIALIZATION_H

#include "llvm/ADT/DenseSet.h"

namespace mlir {
class Operation;
class RewriterBase;
} // namespace mlir

namespace wafer::compiler::detail {

/// Remove unobserved initial contents of actual private Tile storage. Selected
/// pipeline objects are excluded; ownership and memory planning run afterwards.
void eliminateUnusedStorageInitialization(
    mlir::Operation *root, mlir::RewriterBase &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations);

} // namespace wafer::compiler::detail

#endif
