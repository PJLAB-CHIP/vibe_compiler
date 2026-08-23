//===- CardBaselineDataMovement.h ---------------------------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineDataMovement(
    CardMaterializationPlan &assignment, const StructuredDAGAnalysis &dag,
    std::string *failureReason);

} // namespace wafer::compiler::detail
