//===- StructuredDAGSchedulePlan.h - Accepted-IR candidate plan ----*- C++ -*-===//

#pragma once

#include "CardExecutableCompilation.h"
#include "CardExecutableLowering.h"
#include "StructuredDAGCandidateSchedule.h"

#include "Wafer/Analysis/TheoreticalScheduleCostAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace wafer::compiler::detail {

/// Builds one finite selector plan from the accepted final Instr programs.
/// Every final operation is assigned by current-IR buffer use to one source
/// DAG phase or to the explicit residual phase. Fused operations are counted
/// once in their unique retained downstream node; eliminated pure source nodes
/// have empty phases only when their observable paths are covered downstream.
/// The returned plan points into `phaseCosts` and `executable.resourceCost`;
/// both owners must outlive it.
mlir::FailureOr<analysis::StaticSchedulePlan> buildAcceptedStructuredDAGSchedulePlan(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
    llvm::ArrayRef<AcceptedOperationNodeRelation> operationNodeRelations,
    CardExecutableLoweringResult &executable,
    llvm::SmallVectorImpl<analysis::CardInstructionProgramCost>
        &phaseCosts,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
