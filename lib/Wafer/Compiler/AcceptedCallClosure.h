//===- AcceptedCallClosure.h - Accepted executable call graph -*- C++ -*-===//

#ifndef WAFER_COMPILER_ACCEPTEDCALLCLOSURE_H
#define WAFER_COMPILER_ACCEPTEDCALLCLOSURE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace wafer::compiler::detail {

/// The all-and-only statically resolvable function closure of one accepted
/// rank. Functions retain deterministic module order; `entry` is the unique
/// externally visible function, or the sole private function accepted for
/// compatibility with an otherwise single-function module.
struct AcceptedCallClosure {
  mlir::func::FuncOp entry;
  llvm::SmallVector<mlir::func::FuncOp> functions;
};

/// Selects the typed entry and verifies that the module contains exactly its
/// private, defined, direct, non-recursive call closure. Function boundaries
/// contain only Wafer DDR memrefs; private helpers cannot own DDR allocations.
/// When supplied, `expectedEntry` must agree with the structurally selected
/// entry.
llvm::Expected<AcceptedCallClosure> analyzeAcceptedCallClosure(
    mlir::ModuleOp module,
    std::optional<llvm::StringRef> expectedEntry = std::nullopt);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_ACCEPTEDCALLCLOSURE_H
