//===- TargetModelGate.cpp - Wafer target-model execution gate -----------===//

#include "DriverInternal.h"

#ifdef WAFER_ENABLE_SYSTEMC_MODEL

#include "Wafer/Model/TargetModelExpectedOutputGate.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace wafer::compile_driver {

bool runTargetModelGate(
    const CommandLineOptions &options,
    const wafer::compiler::TargetCompilationProduct &product,
    llvm::ArrayRef<IndexedPath> inputPaths,
    llvm::ArrayRef<IndexedPath> expectedPaths, double atol, double rtol,
    wafer::model::TargetModelKernelBudget budget,
    wafer::model::TargetModelExecutionPolicy executionPolicy) {
  std::vector<wafer::model::TargetModelTensorFileBinding> inputs;
  std::vector<wafer::model::TargetModelTensorFileBinding> expected;
  for (const IndexedPath &binding : inputPaths)
    inputs.push_back({binding.index, binding.path});
  for (const IndexedPath &binding : expectedPaths)
    expected.push_back({binding.index, binding.path});
  llvm::Expected<wafer::model::TargetModelResult> result = [&] {
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
    if (const char *failureRank =
            std::getenv("WAFER_TEST_FAIL_TARGET_MODEL_TERMINAL_RANK")) {
      int64_t parsedFailureRank = -1;
      if (llvm::StringRef(failureRank).getAsInteger(10, parsedFailureRank))
        return llvm::Expected<wafer::model::TargetModelResult>(
            llvm::createStringError(
                "invalid test-only target model terminal failure rank"));
      return wafer::model::testing::
          executeAndCompareTargetModelExpectedOutputsWithTerminalFailure(
              product.getExecutableBundle(),
              product.getTargetLLVMModuleBundle(),
              *options.outputProgramDirectory, inputs, expected, atol, rtol,
              budget, executionPolicy, parsedFailureRank,
              options.modelReportNumericStatistics ? &llvm::outs() : nullptr);
    }
#endif
    return wafer::model::executeAndCompareTargetModelExpectedOutputs(
        product.getExecutableBundle(), product.getTargetLLVMModuleBundle(),
        *options.outputProgramDirectory, inputs, expected, atol, rtol, budget,
        executionPolicy,
        options.modelReportNumericStatistics ? &llvm::outs() : nullptr);
  }();
  if (!result) {
    llvm::errs() << "wafer-compile: " << llvm::toString(result.takeError())
                 << "\n";
    return true;
  }
  llvm::outs() << "wafer-compile: target model outputs matched; ranks="
               << result->completedRankCount
               << " transactions=" << result->issuedTransactionCount
               << " systemc_threads=" << result->systemCThreadProcessCount
               << " final_delta=" << result->finalDeltaCount
               << " formal_commands=" << result->formalNumericCommandCount
               << " managed_reference_commands="
               << result->managedReferenceNumericCommandCount
               << " managed_reference_scalars="
               << result->managedReferenceScalarEvaluationCount
               << " bulk_commands=" << result->bulkNumericCommandCount
               << " bulk_matmuls=" << result->bulkMatmulInvocationCount
               << " bulk_reorders=" << result->bulkReorderInvocationCount
               << " bulk_formal_fmas="
               << result->bulkFormalFusedMultiplyAddCount
               << " scheduler=" << result->schedulerIdentity << "\n";
  for (llvm::StringRef digest : result->bulkAdmissionRecordDigests)
    llvm::outs() << "wafer-compile: target model bulk admission=" << digest
                 << "\n";
  for (llvm::StringRef digest : result->bulkManagedReferenceEnvironmentDigests)
    llvm::outs() << "wafer-compile: target model managed-reference environment="
                 << digest << "\n";
  for (llvm::StringRef digest :
       result->managedReferenceTensorEnvironmentDigests)
    llvm::outs() << "wafer-compile: target model managed-reference tensor "
                    "environment="
                 << digest << "\n";
  for (llvm::StringRef implementation :
       result->managedReferenceTensorImplementations)
    llvm::outs() << "wafer-compile: target model managed-reference tensor "
                    "implementation="
                 << implementation << "\n";
  return false;
}

} // namespace wafer::compile_driver

#endif // WAFER_ENABLE_SYSTEMC_MODEL
