//===- ProgramResourceVerification.h - Program resource verification -*-
// C++ -*-===//

#ifndef WAFER_COMPILER_PROGRAMRESOURCEVERIFICATION_H
#define WAFER_COMPILER_PROGRAMRESOURCEVERIFICATION_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Recomputes resource dimensions from a complete canonical Tile
/// domain. Every hard-capacity fact required here must be exact; an unavailable
/// performance-only work term remains raw analysis data and is omitted
/// cohort-wide by the numeric estimator.
mlir::FailureOr<analysis::CardInstructionProgramCost>
verifyProgramResources(llvm::ArrayRef<mlir::ModuleOp> tileModules,
                         llvm::ArrayRef<TileId> tileIds,
                         const ExecutionConfig &executionConfig);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PROGRAMRESOURCEVERIFICATION_H
