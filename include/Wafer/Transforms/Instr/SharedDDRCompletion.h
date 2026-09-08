//===- SharedDDRCompletion.h - Shared DDR publication ----*- C++ -*-===//
#ifndef WAFER_TRANSFORMS_INSTR_SHAREDDDRCOMPLETION_H
#define WAFER_TRANSFORMS_INSTR_SHAREDDDRCOMPLETION_H

#include "Wafer/IR/Topology/TargetTopology.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include <string>

namespace wafer {
enum class SharedDDRCompletionFailure { None, Unsupported, Contract };
struct SharedDDRCompletionResult {
  SharedDDRCompletionFailure failure = SharedDDRCompletionFailure::None;
  std::string detail;
  bool succeeded() const { return failure == SharedDDRCompletionFailure::None; }
};

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
