//===- StructuredBufferRelations.h - Current-IR buffer queries -*- C++ -*-===//

#ifndef WAFER_COMPILER_STRUCTUREDBUFFERRELATIONS_H
#define WAFER_COMPILER_STRUCTUREDBUFFERRELATIONS_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
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

/// Returns true when both values reach at least one common storage root through
/// typed view, TileRegion and structured-control-flow forwarding.
bool shareStructuredBufferStorage(mlir::Value lhs, mlir::Value rhs);

/// Returns the sorted unique DAG node ids whose current materialized buffers
/// are read, written or forwarded by `operation`.
llvm::SmallVector<uint32_t, 4> collectStructuredNodesUsedByOperation(
    mlir::Operation *operation,
    const StructuredMaterializationRelations &relations);

bool operationUsesStructuredNode(
    mlir::Operation *operation, uint32_t structuredNodeId,
    const StructuredMaterializationRelations &relations);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_STRUCTUREDBUFFERRELATIONS_H
