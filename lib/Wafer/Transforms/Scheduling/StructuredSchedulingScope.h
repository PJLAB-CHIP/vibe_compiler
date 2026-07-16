//===- StructuredSchedulingScope.h - Tensor scheduling task scope -*- C++ -*-===//
#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer::structured_scheduler {

/// A source-IR scheduling task.  The scope is derived entirely from SSA
/// dataflow and destination-style ties; it is not persisted in the IR.
struct StructuredSchedulingScope {
  llvm::DenseSet<mlir::Operation *> selected;
  llvm::SmallVector<mlir::Operation *, 8> orderedOps;
  llvm::SmallVector<mlir::Value, 8> inputs;
  llvm::SmallVector<mlir::Value, 4> outs;
  llvm::SmallVector<mlir::Value, 4> yieldedValues;
  mlir::Operation *insertionPoint = nullptr;
};

/// Compiler-private, bounded scope-partition choice.  A negative peer limit
/// admits the complete profitable prefix; zero disables independent
/// shared-input peers. Cross-shape dataflow is still checked by complete
/// candidate lowering before commit. Terminal full-traversal-only roots may
/// be cut only by the dedicated bounded recovery policy.
struct ScopeDiscoveryPolicy {
  int64_t maxSharedInputPeers = -1;
  bool allowCrossShapeDataflow = true;
  bool cutTerminalFullTraversalOnlyRoots = false;
};

enum class ScopeSelectionFailure {
  None,
  MissingExternalResult,
  MissingTensorDestination,
  DuplicateBoundary,
  InterleavedExternalUse,
};

bool isEligibleStructuredSchedulingRoot(mlir::Operation *op);

bool isInsideStructuredSchedulingScope(
    mlir::Operation *operation, const StructuredSchedulingScope &scope);

/// Builds one root-seeded task.  Independent same-family roots sharing an
/// external tensor input may be admitted when requested.  A caller can retry
/// without shared-input peers when their combined scope has no legal commit
/// point.
bool buildStructuredSchedulingScope(mlir::Operation *root,
                                    StructuredSchedulingScope &scope,
                                    ScopeSelectionFailure &failure,
                                    const ScopeDiscoveryPolicy &policy,
                                    const llvm::DenseSet<mlir::Operation *>
                                        &forbiddenOperations);

/// Refreshes boundary values after a previously committed task rewired uses
/// in this still-live source scope.  The selected operations and their ABI
/// type/cardinality contract must remain unchanged.
mlir::LogicalResult refreshStructuredSchedulingScopeBoundary(
    StructuredSchedulingScope &scope);

llvm::StringRef
getScopeSelectionFailureMessage(ScopeSelectionFailure failure);

/// Discovers a complete, deterministic set of non-overlapping task scopes.
/// Shared-input admission automatically falls back to the root-local closure
/// when textual interleaving makes the combined commit point illegal.
mlir::LogicalResult discoverStructuredSchedulingScopes(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<StructuredSchedulingScope> &scopes,
    const ScopeDiscoveryPolicy &policy = {});

/// Clones a task behind a stable function boundary.  Entry arguments are
/// ordered as inputs followed by outs.  Function results are the yielded root
/// values, and therefore have the same cardinality as outs.
mlir::OwningOpRef<mlir::ModuleOp> cloneScopeToStandaloneModule(
    const StructuredSchedulingScope &scope,
    llvm::StringRef functionName = "tensor_program_task");

mlir::func::FuncOp findSingleTaskFunction(mlir::ModuleOp module);

} // namespace wafer::structured_scheduler
