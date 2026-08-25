//===- CardDataflowConstruction.h - Selected edge construction -*- C++ -*-===//

#pragma once

#include "Wafer/Planning/PhysicalDataflow/CardEdgeCarriers.h"
#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardDataflowConstruction(
    CardMaterializationPlan &assignment, const StructuredDAGAnalysis &dag,
    const MovementPlan &movement,
    llvm::ArrayRef<CoupledComponentResultMapping> components,
    llvm::ArrayRef<StructuredOperationRootMapping> roots,
    std::string *failureReason);

} // namespace wafer::compiler::detail
