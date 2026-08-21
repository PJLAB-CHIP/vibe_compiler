//===- CardBaselineDataMovement.h ---------------------------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Baseline/CardBaselineAssignment.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineDataMovement(
    CardBaselineAssignment &assignment, const StructuredDAGAnalysis &dag,
    std::string *failureReason);

} // namespace wafer::compiler::detail
