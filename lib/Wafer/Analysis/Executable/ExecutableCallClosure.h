//===- ExecutableCallClosure.h - Executable call closure -------*- C++ -*-===//

#ifndef WAFER_COMPILER_EXECUTABLECALLCLOSURE_H
#define WAFER_COMPILER_EXECUTABLECALLCLOSURE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace wafer::compiler::detail {

/// The all-and-only statically resolvable function closure of one
/// Tile executable. Functions retain deterministic module order;
/// `entry` is the unique externally visible function.
struct ExecutableCallClosure {
  mlir::func::FuncOp entry;
  llvm::SmallVector<mlir::func::FuncOp> functions;
};

/// Selects the typed entry and verifies that the module contains exactly its
/// private, defined, direct, non-recursive call closure. Function boundaries
/// contain only Wafer DDR memrefs; private helpers cannot own DDR allocations.
/// When supplied, `expectedEntry` must agree with the structurally selected
/// entry.
llvm::Expected<ExecutableCallClosure> analyzeExecutableCallClosure(
    mlir::ModuleOp module,
    std::optional<llvm::StringRef> expectedEntry = std::nullopt);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_EXECUTABLECALLCLOSURE_H
