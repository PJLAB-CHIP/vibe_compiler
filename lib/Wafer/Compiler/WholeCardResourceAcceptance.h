//===- WholeCardResourceAcceptance.h - Resource gate ----------*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLECARDRESOURCEACCEPTANCE_H
#define WAFER_COMPILER_WHOLECARDRESOURCEACCEPTANCE_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Recomputes resource dimensions from a complete canonical physical Tile
/// domain. Every hard-capacity fact required here must be exact; an unavailable
/// performance-only work term remains raw analysis data and is omitted
/// cohort-wide by the numeric estimator.
mlir::FailureOr<analysis::WholeCardInstructionProgramCost>
acceptWholeCardResources(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules,
                         llvm::ArrayRef<PhysicalTileId> physicalTileIds,
                         const ExecutionConfig &executionConfig);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLECARDRESOURCEACCEPTANCE_H
