//===- DataMovementApply.h - Apply selected movement ----------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Search/DataMovement.h"

namespace wafer::compiler::detail {

mlir::LogicalResult
applySelectedDataMovement(mlir::ModuleOp cardModule,
                          const CardProgramAnalysis &program,
                          const CardDataMovementAssignment &assignment,
                          StructuredMaterializationRelations &relations,
                          std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
