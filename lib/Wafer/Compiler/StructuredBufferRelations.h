//===- StructuredBufferRelations.h - Current-IR buffer queries -*- C++ -*-===//

#ifndef WAFER_COMPILER_STRUCTUREDBUFFERRELATIONS_H
#define WAFER_COMPILER_STRUCTUREDBUFFERRELATIONS_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>

namespace wafer::compiler::detail {

/// Remaps every live SSA value in `source` through one same-epoch IR clone.
/// Relations outside the cloned scope are omitted; callers must reject a
/// missing relation when their local contract requires complete coverage.
StructuredMaterializationRelations
remapStructuredBufferRelations(const StructuredMaterializationRelations &source,
                               const mlir::IRMapping &mapping);

/// Tracks result replacements performed by one successful rewrite driver and
/// retargets the current-IR buffer relations in place. It owns no IR and
/// must not outlive either the relations or the rewrite invocation.
class StructuredBufferReplacementListener final
    : public mlir::RewriterBase::Listener {
public:
  explicit StructuredBufferReplacementListener(
      StructuredMaterializationRelations &relations);
  ~StructuredBufferReplacementListener() final;

  void notifyOperationReplaced(mlir::Operation *operation,
                               mlir::ValueRange replacements) final;
  void notifyOperationErased(mlir::Operation *operation) final;

  bool preservedAllRelations() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

/// Checks that every related SSA buffer belongs to `root`'s current IR. The
/// implementation compares opaque value identities against a live-value set
/// and therefore never dereferences a stale relation while reporting failure.
mlir::LogicalResult checkStructuredBufferRelationsCurrent(
    mlir::Operation *root, const StructuredMaterializationRelations &relations);

/// Drops query relations whose SSA value was erased by a completed
/// best-effort cleanup. This never infers a replacement: a downstream query
/// that still requires the missing structured witness must fail closed.
void retainCurrentStructuredBufferRelations(
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

/// Returns true when both values reach at least one common storage root through
/// typed view, TileRegion and structured-control-flow forwarding.
bool shareStructuredBufferStorage(mlir::Value lhs, mlir::Value rhs);

/// Memoized variant; the memo must cover the same unchanged IR epoch.
bool shareStructuredBufferStorage(mlir::Value lhs, mlir::Value rhs,
                                  StorageRootMemo &memo);

/// Returns the sorted unique DAG node ids whose current materialized buffers
/// are read, written or forwarded by `operation`.
llvm::SmallVector<uint32_t, 4> collectStructuredNodesUsedByOperation(
    mlir::Operation *operation,
    const StructuredMaterializationRelations &relations);

/// Memoized variant; the memo must cover the same unchanged IR epoch.
llvm::SmallVector<uint32_t, 4> collectStructuredNodesUsedByOperation(
    mlir::Operation *operation,
    const StructuredMaterializationRelations &relations, StorageRootMemo &memo);

bool operationUsesStructuredNode(
    mlir::Operation *operation, uint32_t structuredNodeId,
    const StructuredMaterializationRelations &relations);

/// Memoized variant; the memo must cover the same unchanged IR epoch.
bool operationUsesStructuredNode(
    mlir::Operation *operation, uint32_t structuredNodeId,
    const StructuredMaterializationRelations &relations, StorageRootMemo &memo);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_STRUCTUREDBUFFERRELATIONS_H
