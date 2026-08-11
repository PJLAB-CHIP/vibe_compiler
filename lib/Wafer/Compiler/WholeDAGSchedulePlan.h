//===- WholeDAGSchedulePlan.h - Accepted-IR candidate plan ----*- C++ -*-===//

#pragma once

#include "WholeCardExecutableAdmission.h"
#include "WholeDAGCandidateSchedule.h"

#include "Wafer/Analysis/TheoreticalScheduleCostAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToCardProgram/WaferTensorProgramToCardProgram.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace wafer::compiler::detail {

/// Builds one finite selector plan from the accepted final Instr programs.
/// Every final operation is assigned by query-local lineage to one source DAG
/// phase or to the explicit residual phase. Fused operations are owned once by
/// their unique retained downstream lineage; eliminated pure source nodes have
/// empty phases only when their observable paths are covered downstream. The
/// returned plan points into `phaseCosts` and `executable.resourceCost`; both
/// owners must outlive it.
mlir::FailureOr<analysis::StaticSchedulePlan> buildAcceptedWholeDAGSchedulePlan(
    const CardDAGAnalysis &dag,
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
    llvm::ArrayRef<CardProgramSourceOperationLineage> sourceLineage,
    AcceptedWholeCardExecutable &executable,
    llvm::SmallVectorImpl<analysis::WholeCardInstructionProgramCost>
        &phaseCosts,
    std::string *failureReason = nullptr);

/// Removes every query-local source lineage location from an accepted
/// executable while retaining its original fallback source location.
mlir::LogicalResult stripCardProgramSourceOperationLineage(
    AcceptedWholeCardExecutable &executable,
    llvm::ArrayRef<CardProgramSourceOperationLineage> sourceLineage,
    std::string *failureReason = nullptr);

/// Inspection helper used to prove that a publishable artifact contains no
/// query-local lineage pointer.
bool containsCardProgramSourceOperationLineage(
    const AcceptedWholeCardExecutable &executable);

/// Removes query-local observable-output lineage after exact allocation
/// feedback has finished and before the winning executable is published.
mlir::LogicalResult
stripSpatialOutputLineage(AcceptedWholeCardExecutable &executable,
                          llvm::ArrayRef<SpatialOutputLineage> outputLineage,
                          std::string *failureReason = nullptr);

bool containsSpatialOutputLineage(
    const AcceptedWholeCardExecutable &executable);

mlir::LogicalResult stripStructuredOperandDemandLineage(
    AcceptedWholeCardExecutable &executable,
    llvm::ArrayRef<StructuredOperandDemandLineage> operandDemandLineage,
    std::string *failureReason = nullptr);

bool containsStructuredOperandDemandLineage(
    const AcceptedWholeCardExecutable &executable);

} // namespace wafer::compiler::detail
