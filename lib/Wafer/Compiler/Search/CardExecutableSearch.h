//===- CardExecutableSearch.h - Card executable selection -----*- C++ -*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLESEARCH_H
#define WAFER_COMPILER_CARDEXECUTABLESEARCH_H

#include "Wafer/Compiler/Executable/CardExecutableLowering.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail {

/// Starts the current search session from an immutable TensorProgram and an
/// already accepted move-only baseline. Q50.S contributes the first partial
/// typed domain, but no physical assignment is complete yet, so the exact same
/// executable owner is returned without cloning, lowering, or recompiling IR.
mlir::FailureOr<CardExecutableLoweringResult>
runCardExecutableSearch(mlir::ModuleOp tensorProgram,
                        CardExecutableLoweringResult baseline);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLESEARCH_H
