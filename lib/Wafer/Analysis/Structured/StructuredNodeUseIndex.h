//===- StructuredNodeUseIndex.h - Structured node use lookup -*- C++ -*-===//

#ifndef WAFER_COMPILER_STRUCTUREDNODEUSEINDEX_H
#define WAFER_COMPILER_STRUCTUREDNODEUSEINDEX_H

#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer::compiler::detail {

/// Indexes current-IR storage roots by their originating structured DAG nodes.
/// The index is valid only while the queried IR and materialization relations
/// are unchanged. It turns a module-wide operation attribution walk into one
/// relation traversal followed by local operand/result lookups.
class StructuredNodeUseIndex {
public:
  explicit StructuredNodeUseIndex(
      const StructuredMaterializationRelations &relations);

  /// Returns the sorted unique structured DAG node ids whose materialized
  /// buffers are read, written or forwarded by `operation`.
  llvm::SmallVector<uint32_t, 4> collectNodesUsedBy(mlir::Operation *operation);

private:
  StorageRootMemo storageRoots;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint32_t, 2>>
      nodesByOperation;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<uint32_t, 2>> nodesByRoot;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_STRUCTUREDNODEUSEINDEX_H
