//===- WholeCardResourceCostValidation.h - Validate resource costs -------===//

#ifndef WAFER_COMPILER_WHOLECARDRESOURCECOSTVALIDATION_H
#define WAFER_COMPILER_WHOLECARDRESOURCECOSTVALIDATION_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Recomputes the exact resource dimensions of a complete canonical rank
/// domain. The returned summary is derived from the supplied IR and is not a
/// second representation of the schedule.
mlir::FailureOr<analysis::WholeCardInstructionProgramCost>
validateWholeCardResourceCost(llvm::ArrayRef<mlir::ModuleOp> rankModules,
                              const ExecutionConfig &executionConfig);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLECARDRESOURCECOSTVALIDATION_H
