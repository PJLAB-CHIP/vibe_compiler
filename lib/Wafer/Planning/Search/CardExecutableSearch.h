//===- CardExecutableSearch.h - Card executable selection -----*- C++ -*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLESEARCH_H
#define WAFER_COMPILER_CARDEXECUTABLESEARCH_H

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/CodeGen/Executable/CardExecutableLowering.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail {

/// Starts the current search session from an immutable TensorProgram and an
/// already accepted move-only baseline. Q50.S and Q50.B contribute typed
/// semantic-root and spatial-placement domains, but the remaining physical
/// coordinates are incomplete, so the exact same executable owner is returned
/// without enumerating, cloning, lowering, or recompiling partial states.
mlir::FailureOr<CardExecutableLoweringResult>
runCardExecutableSearch(mlir::ModuleOp tensorProgram,
                        const CardProgramAnalysis &programAnalysis,
                        CardExecutableLoweringResult baseline);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLESEARCH_H
