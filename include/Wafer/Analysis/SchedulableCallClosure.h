//===- SchedulableCallClosure.h - Rank-local call closure -----*- C++ -*-===//

#ifndef WAFER_ANALYSIS_SCHEDULABLECALLCLOSURE_H
#define WAFER_ANALYSIS_SCHEDULABLECALLCLOSURE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace wafer {

enum class SchedulableCallClosureStatus : uint8_t {
  Success,
  InvalidIR,
  UnsupportedSemantic,
};

enum class SchedulableCallClosureReason : uint8_t {
  NoEntry,
  MultipleEntries,
  EntryBindingMismatch,
  NonPrivateCallee,
  TensorExternalCallWithoutSummary,
  IndirectCall,
  RecursiveCall,
};

struct SchedulableCallClosure {
  SchedulableCallClosureStatus status = SchedulableCallClosureStatus::InvalidIR;
  std::optional<SchedulableCallClosureReason> reason;
  mlir::Operation *diagnosticOperation = nullptr;
  mlir::func::FuncOp entry;
  /// Reachable functions in module symbol-table source order. Each function
  /// appears exactly once; nested regions remain owned by that function.
  llvm::SmallVector<mlir::func::FuncOp, 8> functions;
};

SchedulableCallClosure analyzeSchedulableCallClosure(
    mlir::ModuleOp module,
    std::optional<llvm::StringRef> expectedEntry = std::nullopt);

/// Returns the source-ordered union of the call closures rooted at every
/// defined non-private function. This is for module-wide debug/replay passes
/// and transaction-local candidate modules that deliberately contain several
/// independent programs. Calls between non-private roots remain invalid; each
/// root may only call private helpers. Production artifact binding continues
/// to use analyzeSchedulableCallClosure and therefore remains single-entry.
SchedulableCallClosure analyzeAllSchedulableCallClosures(mlir::ModuleOp module);

llvm::StringRef
describeSchedulableCallClosureReason(SchedulableCallClosureReason reason);

} // namespace wafer

#endif // WAFER_ANALYSIS_SCHEDULABLECALLCLOSURE_H
