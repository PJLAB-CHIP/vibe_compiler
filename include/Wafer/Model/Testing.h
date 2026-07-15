//===- Testing.h - Test-only target model seams ----------------*- C++ -*-===//

#ifndef WAFER_MODEL_TESTING_H
#define WAFER_MODEL_TESTING_H

#include "Wafer/Model/SystemCTargetModel.h"

namespace wafer::model::testing {

/// Runs the production SystemC model while failing the selected rank at its
/// terminal callback. This is only for source/driver atomicity tests; normal
/// model execution must use executeSystemCTargetModel.
llvm::Expected<TargetModelResult> executeSystemCTargetModelWithTerminalFailure(
    compiler::TargetCallExecutable executable,
    llvm::ArrayRef<TargetModelInputBinding> inputBindings,
    TargetModelKernelBudget budget, TargetModelExecutionPolicy policy,
    int64_t failureLogicalRank);

} // namespace wafer::model::testing

#endif // WAFER_MODEL_TESTING_H
