//===- CardBaselineTemporalTiling.h -------------------------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Baseline/CardBaselineAssignment.h"

namespace wafer::compiler::detail {

mlir::LogicalResult setCardBaselineTemporalTiles(
    CardBaselineAssignment &assignment, const CardProgramAnalysis &program,
    std::string *failureReason);

} // namespace wafer::compiler::detail
