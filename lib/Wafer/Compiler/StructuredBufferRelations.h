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

/// Source relations a strict remap could not map. An unmapped relation means
/// the rewrite consumed a value that still carries attribution evidence; the
/// caller reports these and fails closed instead of probing or planning with
/// incomplete evidence.
struct StructuredRelationRemapIssue {
  llvm::SmallVector<StructuredOperationBufferRelation, 4>
      unmappedResultBuffers;
  llvm::SmallVector<StructuredOperationBufferRelation, 4> unmappedOperandBuffers;
  llvm::SmallVector<SpatialOutputBufferRelation, 4> unmappedOutputBuffers;

  bool empty() const {
    return unmappedResultBuffers.empty() && unmappedOperandBuffers.empty() &&
           unmappedOutputBuffers.empty();
  }
};

/// Strict form of remapStructuredBufferRelations for the probe/final evidence
/// contract: every relation buffer must be mapped through the clone. Any
/// unmapped relation fails the remap and is reported in `issue` for typed
/// diagnostics; an incomplete remap is a contract violation, never "no
/// relation".
mlir::FailureOr<StructuredMaterializationRelations>
remapStructuredBufferRelationsComplete(
    const StructuredMaterializationRelations &source,
    const mlir::IRMapping &mapping,
    StructuredRelationRemapIssue *issue = nullptr);

/// Keeps only relations whose buffer belongs to `root`'s current probe scope:
/// an operand or result of `root`, or a value defined inside one of `root`'s
/// regions. Relations on sibling regions or other scopes are dropped so a
/// strict probe remap sees a complete in-scope evidence set instead of
/// failing closed on a foreign buffer.
StructuredMaterializationRelations
scopeStructuredBufferRelations(mlir::Operation *root,
                               const StructuredMaterializationRelations &relations);

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
