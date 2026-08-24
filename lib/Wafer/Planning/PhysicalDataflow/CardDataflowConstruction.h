//===- CardDataflowConstruction.h - Selected edge construction -*- C++ -*-===//

#pragma once

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardDataflowConstruction(
    CardMaterializationPlan &assignment, const StructuredDAGAnalysis &dag,
    const MovementPlan &movement, std::string *failureReason);

} // namespace wafer::compiler::detail
