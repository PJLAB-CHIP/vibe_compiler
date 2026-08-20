//===- BulkQualificationExecution.cpp - Unqualified execution seam ---===//

#include "BulkTensorNumericInternal.h"

namespace wafer::detail {

using namespace wafer::bulk_detail;

llvm::Expected<UnqualifiedBulkExecutionResult>
executeBulkTensorForQualification(const BulkExecutionEnvironment &environment,
                                  const FormalGemmOperation &operation,
                                  llvm::ArrayRef<BulkTensorStorage> inputs,
                                  const BulkTensorStorage &destinationTemplate,
                                  BulkNumericWorkBudget budget) {
  llvm::Expected<BulkExecutionEnvironment> current =
      createManagedBulkExecutionEnvironment();
  if (!current)
    return current.takeError();
  if (current->getDigest() != environment.getDigest())
    return bulkError(BulkTensorNumericErrorCode::EnvironmentMismatch,
                     "requested bulk environment is not current");
  return executeOneDNN(environment, operation, inputs, destinationTemplate,
                       budget);
}

} // namespace wafer::detail
