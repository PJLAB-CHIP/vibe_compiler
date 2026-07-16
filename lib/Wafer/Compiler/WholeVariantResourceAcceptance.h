//===- WholeVariantResourceAcceptance.h - Resource gate -------*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLEVARIANTRESOURCEACCEPTANCE_H
#define WAFER_COMPILER_WHOLEVARIANTRESOURCEACCEPTANCE_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Recomputes the exact resource dimensions of a complete canonical rank
/// domain. This is a read-only acceptance gate: the returned summary is
/// derived from accepted IR and never becomes a parallel schedule artifact.
mlir::FailureOr<analysis::WholeCardInstructionProgramCost>
acceptWholeVariantResources(llvm::ArrayRef<mlir::ModuleOp> rankModules,
                            const ExecutionConfig &executionConfig);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLEVARIANTRESOURCEACCEPTANCE_H
