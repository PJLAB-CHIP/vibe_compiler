//===- CardBaselineTemporalTiling.h -------------------------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"

namespace wafer::compiler::detail {

mlir::LogicalResult setCardBaselineTemporalTiles(
    CardMaterializationPlan &assignment, const CardProgramAnalysis &program,
    std::string *failureReason);

} // namespace wafer::compiler::detail
