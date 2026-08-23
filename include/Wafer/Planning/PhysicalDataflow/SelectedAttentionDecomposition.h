//===- SelectedAttentionDecomposition.h - Winner attention IR -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_SELECTEDATTENTIONDECOMPOSITION_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_SELECTEDATTENTIONDECOMPOSITION_H

#include "Wafer/Planning/PhysicalDataflow/AttentionWorkDescription.h"

#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

#include <string>
#include <variant>
#include <vector>

namespace mlir {
class Operation;
class RewriterBase;
} // namespace mlir

namespace wafer::compiler::detail {

struct PreparedAttentionDecomposition {
  CanonicalAttentionWorkCoordinate work;
};

enum class BrokenPreparedAttentionReason : uint8_t {
  DuplicateIdentity,
  InvalidWorkDescription,
};

struct BrokenPreparedAttention {
  BrokenPreparedAttentionReason reason =
      BrokenPreparedAttentionReason::InvalidWorkDescription;
  std::string detail;
};

using PreparedAttentionDecompositionOutcome =
    std::variant<PreparedAttentionDecomposition, BrokenPreparedAttention>;

PreparedAttentionDecompositionOutcome prepareSelectedAttentionDecomposition(
    const CanonicalAttentionWorkCoordinate &work);

struct AttentionActionMaterialization {
  AttentionActionId id;
  /// Current selected-subtree epoch only; not a stable identity or cache key.
  llvm::SmallVector<mlir::Operation *, 3> operations;
  /// All structured operations created while emitting this action. Unlike
  /// `operations`, this is an ownership partition: each created structured op
  /// belongs to exactly one action and appears here exactly once.
  llvm::SmallVector<mlir::Operation *, 8> structuredOperations;
};

struct AttentionValueMaterialization {
  AttentionValueId id;
  /// Current selected-subtree epoch only. One logical resident value may have
  /// several SSA occurrence classes, for example the steady loop body and a
  /// statically shaped tail block.
  llvm::SmallVector<mlir::Value, 2> occurrences;
};

struct AttentionScratchMaterialization {
  AttentionScratchId id;
  /// A statically expanded recurrence may materialize the same planned
  /// scratch role once per block. Every occurrence is owned by the disposable
  /// selected subtree and must match the planned resident type/domain.
  llvm::SmallVector<mlir::Value, 2> occurrences;
};

struct AttentionScopeOperationMaterialization {
  AttentionWorkScopeId scope;
  /// All top-level structured operations created for this scope in the
  /// current selected-subtree epoch. This is the complete operation set used
  /// by the Card materializer; action mappings may intentionally share ops.
  llvm::SmallVector<mlir::Operation *, 16> operations;
};

struct SelectedAttentionRootMaterialization {
  SemanticRootKey root;
  mlir::Value result;
  std::vector<AttentionActionMaterialization> actions;
  std::vector<AttentionValueMaterialization> values;
  std::vector<AttentionScratchMaterialization> scratch;
  std::vector<AttentionScopeOperationMaterialization> scopes;
};

/// Replaces one selected attention op in the caller-owned new subtree. All
/// planning validation belongs to prepare; an emission failure requires the
/// caller to discard its enclosing transaction.
mlir::FailureOr<SelectedAttentionRootMaterialization>
emitSelectedAttentionDecomposition(mlir::RewriterBase &rewriter,
                                   LinalgExtAttentionOp attention,
                                   const AttentionWorkDescription &description,
                                   std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_SELECTEDATTENTIONDECOMPOSITION_H
