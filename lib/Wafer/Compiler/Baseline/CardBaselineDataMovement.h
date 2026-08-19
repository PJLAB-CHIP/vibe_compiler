//===- CardBaselineDataMovement.h ---------------------------*- C++ -*-===//

#pragma once

#include "Wafer/Compiler/Baseline/CardBaselineAssignment.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineDataMovement(
    CardBaselineAssignment &assignment, const StructuredDAGAnalysis &dag,
    analysis::IREpoch epoch, std::string *failureReason);

} // namespace wafer::compiler::detail
