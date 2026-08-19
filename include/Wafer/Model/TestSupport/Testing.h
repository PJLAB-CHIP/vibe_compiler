//===- Testing.h - Test-only target model seams ----------------*- C++ -*-===//

#ifndef WAFER_MODEL_TESTING_H
#define WAFER_MODEL_TESTING_H

#include "Wafer/Model/SystemC/SystemCTargetModel.h"

namespace wafer::model::testing {

/// Runs the production SystemC model while failing the selected Tile at its
/// completion callback. This is only for source/driver atomicity tests; normal
/// model execution must use executeSystemCTargetModel.
llvm::Expected<TargetModelResult>
executeSystemCTargetModelWithTileCompletionFailure(
    compiler::TargetCallExecutable executable,
    llvm::ArrayRef<TargetModelInputBinding> inputBindings,
    TargetModelKernelBudget budget, TargetModelExecutionPolicy policy,
    int64_t failureLaunchSlot);

} // namespace wafer::model::testing

#endif // WAFER_MODEL_TESTING_H
