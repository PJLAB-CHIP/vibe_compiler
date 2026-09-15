//===- SharedDDRCompletion.h - Shared DDR publication ----*- C++ -*-===//
#ifndef WAFER_TRANSFORMS_INSTR_SHAREDDDRCOMPLETION_H
#define WAFER_TRANSFORMS_INSTR_SHAREDDDRCOMPLETION_H

#include "Wafer/IR/Topology/TargetTopology.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace wafer {
class DDRBindingAttr;
/// Resolve a current data value through typed single-execution aliases to
/// its entry DDR binding. No binding is inferred from names or shapes.
DDRBindingAttr getSharedDDRBinding(mlir::Value value);

enum class SharedDDRCompletionFailure {
  None,
  Unsupported,
  Indeterminate,
  Contract
};
struct SharedDDRCompletionResult {
  SharedDDRCompletionFailure failure = SharedDDRCompletionFailure::None;
  std::string detail;
  bool succeeded() const { return failure == SharedDDRCompletionFailure::None; }
};

enum class CommunicationOrderStatus { Acyclic, Cycle, Unsupported, Contract };
struct CommunicationOrderAnalysis {
  CommunicationOrderStatus status = CommunicationOrderStatus::Acyclic;
  // These are actual operations in this read-only IR epoch. Any mutation
  // invalidates the complete analysis, including these handles.
  llvm::SmallVector<mlir::Operation *, 8> cycle;
  std::string detail;
};

CommunicationOrderAnalysis
analyzeCurrentCommunicationOrder(llvm::ArrayRef<mlir::ModuleOp> modules,
                                 llvm::ArrayRef<TileId> tileIds);

/// Creates the actual publication/acquisition IR. Order is closed by the
/// proposal constructor or verified by materializeSharedDDRCompletion.
SharedDDRCompletionResult
materializeSharedDDRNotifications(llvm::ArrayRef<mlir::ModuleOp> modules);

/// Creates notification IR for one newly materialized DDR resource. The
/// caller must close and verify the complete candidate communication order
/// before passing it to memory/target planning.
SharedDDRCompletionResult
materializeSharedDDRResourceCompletion(llvm::ArrayRef<mlir::ModuleOp> modules,
                                       int64_t resourceId);

/// The caller owns and discards the complete candidate on failure.
SharedDDRCompletionResult
materializeSharedDDRCompletion(llvm::ArrayRef<mlir::ModuleOp> modules,
                               llvm::ArrayRef<TileId> tileIds);
/// Read-only verification of actual publication, acquisition and resource IR.
SharedDDRCompletionResult
verifySharedDDRCompletion(llvm::ArrayRef<mlir::ModuleOp> modules,
                          llvm::ArrayRef<TileId> tileIds);
} // namespace wafer
#endif
