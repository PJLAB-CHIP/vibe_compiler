//===- OneDNNQualificationExecution.cpp - Unqualified execution seam ---===//

#include "OneDNNTensorNumericInternal.h"

namespace wafer::detail {

using namespace wafer::onednn_detail;

llvm::Expected<UnqualifiedOneDNNExecutionResult>
executeOneDNNTensorForQualification(
    const OneDNNExecutionEnvironment &environment,
    const FormalGemmOperation &operation,
    llvm::ArrayRef<OneDNNTensorStorage> inputs,
    const OneDNNTensorStorage &destinationTemplate,
    OneDNNNumericWorkBudget budget) {
  llvm::Expected<OneDNNExecutionEnvironment> current =
      createManagedOneDNNExecutionEnvironment();
  if (!current)
    return current.takeError();
  if (current->getDigest() != environment.getDigest())
    return onednnError(OneDNNTensorNumericErrorCode::EnvironmentMismatch,
                       "requested onednn environment is not current");
  return executeOneDNN(environment, operation, inputs, destinationTemplate,
                       budget);
}

} // namespace wafer::detail
