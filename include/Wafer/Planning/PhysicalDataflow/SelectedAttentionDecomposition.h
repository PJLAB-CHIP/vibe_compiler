//===- SelectedAttentionDecomposition.h - Winner attention IR -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_SELECTEDATTENTIONDECOMPOSITION_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_SELECTEDATTENTIONDECOMPOSITION_H

#include "Wafer/Planning/PhysicalDataflow/FeasibilityProof.h"

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
  MissingFullProof,
  ActionCoverageMismatch,
  DuplicateIdentity,
  InvalidWorkDescription,
};

struct BrokenPreparedAttention {
  BrokenPreparedAttentionReason reason =
      BrokenPreparedAttentionReason::MissingFullProof;
  std::string detail;
};

using PreparedAttentionDecompositionOutcome =
    std::variant<PreparedAttentionDecomposition, BrokenPreparedAttention>;

PreparedAttentionDecompositionOutcome prepareSelectedAttentionDecomposition(
    const CanonicalAttentionWorkCoordinate &work,
    const FullFeasibilityProof &proof);

struct AttentionActionMaterialization {
  AttentionActionId id;
  /// Current selected-subtree epoch only; not a stable identity or cache key.
  llvm::SmallVector<mlir::Operation *, 3> operations;
};

struct AttentionValueMaterialization {
  AttentionValueId id;
  /// Current selected-subtree epoch only.
  mlir::Value value;
};

struct SelectedAttentionRootMaterialization {
  SemanticRootKey root;
  mlir::Value result;
  std::vector<AttentionActionMaterialization> actions;
  std::vector<AttentionValueMaterialization> values;
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
