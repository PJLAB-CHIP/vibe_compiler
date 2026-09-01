//===- StructuredBufferRelations.h - Current-IR buffer queries -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_STRUCTUREDBUFFERRELATIONS_H
#define WAFER_TRANSFORMS_TILE_STRUCTUREDBUFFERRELATIONS_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>

namespace wafer::compiler::detail {

/// Tracks result replacements performed by one successful rewrite driver and
/// retargets the current-IR buffer relations in place. It owns no IR and
/// must not outlive either the relations or the rewrite invocation.
class StructuredBufferReplacementListener
    : public mlir::RewriterBase::Listener {
public:
  explicit StructuredBufferReplacementListener(
      StructuredMaterializationRelations &relations);
  ~StructuredBufferReplacementListener() override;

  void notifyOperationReplaced(mlir::Operation *operation,
                               mlir::ValueRange replacements) final;
  void notifyOperationErased(mlir::Operation *operation) final;
  void recordScratchAllocation(mlir::Operation *sourceOperation,
                               mlir::Value allocation);
  void recordLoweredOperation(mlir::Operation *sourceOperation,
                              mlir::Operation *loweredOperation);

  /// Completes one rewrite epoch. Relations to explicitly erased dead private
  /// allocations are discarded because those buffers no longer contribute
  /// executable work; every other untracked erasure remains a failure. Source
  /// operation owners are replaced by explicit live operations recorded by
  /// the lowering patterns.
  bool finalizeAfterRewrite();
  llvm::StringRef getFailureReason() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

/// Checks that every related SSA buffer belongs to `root`'s current IR. The
/// implementation compares opaque value identities against a live-value set
/// and therefore never dereferences a stale relation while reporting failure.
mlir::LogicalResult checkStructuredBufferRelationsCurrent(
    mlir::Operation *root, const StructuredMaterializationRelations &relations);

/// Drops attribution entries whose SSA value was deleted by a completed
/// cleanup/bufferization epoch. This compares opaque identities against the
/// current root and never dereferences a stale value; it does not redirect a
/// missing required relation to another buffer.
void retainCurrentStructuredBufferRelations(
    mlir::Operation *root, StructuredMaterializationRelations &relations);

/// Recomputes every operation/buffer ownership edge from one unchanged
/// current-IR root while preserving the caller-owned structural output and
/// cross-Tile endpoint relations. This is used after a destructive stage
/// rewrite whose source operation owners no longer exist; it never invents a
/// buffer or follows a source-graph identifier.
void rebuildCurrentBufferOwnerRelations(
    mlir::Operation *root, StructuredMaterializationRelations &relations);

/// Query-local memo of per-value storage roots. Values are only valid within
/// one unchanged IR epoch; a caller constructs one memo per validation root
/// and threads it through every buffer query in that scope, turning
/// per-op×per-relation storage-root walks into amortized O(1) lookups.
/// The per-value sets are heap-owned so the returned references stay valid
/// across later map rehashes.
class StorageRootMemo {
public:
  const llvm::DenseSet<mlir::Value> &getStorageRoots(mlir::Value value);

private:
  llvm::DenseMap<mlir::Value, std::unique_ptr<llvm::DenseSet<mlir::Value>>>
      memo;
};

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_STRUCTUREDBUFFERRELATIONS_H
