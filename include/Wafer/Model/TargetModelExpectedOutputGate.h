//===- TargetModelExpectedOutputGate.h - Source-backed model gate -*- C++
//-*-===//

#ifndef WAFER_MODEL_TARGETMODELEXPECTEDOUTPUTGATE_H
#define WAFER_MODEL_TARGETMODELEXPECTEDOUTPUTGATE_H

#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Model/SystemCTargetModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::model {

struct TargetModelTensorFileBinding {
  int64_t index = -1;
  std::string path;
};

/// Executes the exact owner-backed target product through the SystemC model
/// and compares every typed output binding against repository-owned NumPy
/// references.  This is the shared source-backed numeric gate used by the
/// production driver and the isolated optimization qualification worker.
llvm::Expected<TargetModelResult> executeAndCompareTargetModelExpectedOutputs(
    const compiler::ExecutableBundle &executableBundle,
    const compiler::TargetLLVMModuleBundle &targetLLVMModuleBundle,
    llvm::StringRef outputProgramDirectory,
    llvm::ArrayRef<TargetModelTensorFileBinding> inputPaths,
    llvm::ArrayRef<TargetModelTensorFileBinding> expectedPaths, double atol,
    double rtol, TargetModelKernelBudget budget,
    TargetModelExecutionPolicy executionPolicy,
    llvm::raw_ostream *numericStatistics = nullptr);

namespace testing {

/// Failure-injection variant used only by the dedicated compiler test binary.
llvm::Expected<TargetModelResult>
executeAndCompareTargetModelExpectedOutputsWithTerminalFailure(
    const compiler::ExecutableBundle &executableBundle,
    const compiler::TargetLLVMModuleBundle &targetLLVMModuleBundle,
    llvm::StringRef outputProgramDirectory,
    llvm::ArrayRef<TargetModelTensorFileBinding> inputPaths,
    llvm::ArrayRef<TargetModelTensorFileBinding> expectedPaths, double atol,
    double rtol, TargetModelKernelBudget budget,
    TargetModelExecutionPolicy executionPolicy, int64_t failureRank,
    llvm::raw_ostream *numericStatistics = nullptr);

} // namespace testing

} // namespace wafer::model

#endif // WAFER_MODEL_TARGETMODELEXPECTEDOUTPUTGATE_H
