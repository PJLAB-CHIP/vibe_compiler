//===- QualifiedBulkExecution.cpp - Qualified bulk execution ------------===//

#include "BulkTensorNumericInternal.h"

#include <utility>

namespace wafer {

using namespace bulk_detail;

llvm::Expected<BulkTensorNumericResult>
executeQualifiedBulkTensorNumeric(const BulkExecutionEnvironment &environment,
                                  const QualifiedBulkExecution &execution,
                                  const ResolvedNumericCommand &command,
                                  llvm::ArrayRef<BulkTensorStorage> inputs,
                                  const BulkTensorStorage &destinationTemplate,
                                  BulkNumericWorkBudget budget) {
  if (execution.getEnvironmentDigest() != environment.getDigest())
    return bulkError(
        BulkTensorNumericErrorCode::EnvironmentMismatch,
        "bulk qualification does not match the current environment");
  if (execution.getAdapterDigest() != getBulkAdapterContractDigest() ||
      !command.getSemantics() ||
      execution.getSemanticProfileDigest() !=
          command.getSemantics()->getDigest() ||
      execution.getResolutionDigest() != command.getDigest() ||
      execution.getInputPayloadDigest() !=
          computeBulkTensorPayloadDigest(inputs) ||
      execution.getDestinationTemplateDigest() !=
          computeBulkTensorStorageDigest(destinationTemplate))
    return bulkError(BulkTensorNumericErrorCode::QualificationMismatch,
                     "bulk command, payload or destination template is not "
                     "the frozen qualified row");
  llvm::Expected<detail::UnqualifiedBulkExecutionResult> result =
      detail::executeBulkTensorForQualification(environment, command, inputs,
                                                destinationTemplate, budget);
  if (!result)
    return result.takeError();
  if (result->evidence.implementation !=
          execution.getExpectedImplementation() ||
      result->evidence.resolvedDescriptorDigest !=
          execution.getExpectedResolvedDescriptorDigest())
    return bulkError(BulkTensorNumericErrorCode::QualificationMismatch,
                     "oneDNN implementation or resolved descriptor changed "
                     "from the frozen qualified row");
  if (computeBulkTensorStorageDigest(result->destination) !=
      execution.getExpectedBackendOutputDigest())
    return bulkError(BulkTensorNumericErrorCode::BackendOutputMismatch,
                     "oneDNN output changed from the frozen validated result");
  return BulkTensorNumericResult{std::move(result->destination),
                                 execution.getFormalFlags(),
                                 std::move(result->evidence)};
}

llvm::Expected<BulkTensorNumericResult>
executeManagedReferenceBulkTensorNumeric(
    const BulkExecutionEnvironment &environment,
    const ResolvedNumericCommand &command,
    llvm::ArrayRef<BulkTensorStorage> inputs,
    const BulkTensorStorage &destinationTemplate,
    BulkNumericWorkBudget budget) {
  if (!command.isSupported() ||
      command.getFamily() != NumericCommandFamily::NEGemm ||
      !command.getSemantics())
    return bulkError(BulkTensorNumericErrorCode::UnsupportedResolvedCommand,
                     "managed-reference bulk execution requires a supported "
                     "NE GEMM semantic identity");
  for (const BulkTensorStorage &input : inputs) {
    llvm::Expected<std::vector<RawLogicalValue>> values =
        unpackBulkTensorLogicalValues(input);
    if (!values)
      return values.takeError();
    for (RawLogicalValue value : *values) {
      llvm::Expected<LogicalValueClassification> classification =
          classifyRawLogicalValue(value, NonCanonicalEncodingPolicy::Reject);
      if (!classification)
        return bulkError(BulkTensorNumericErrorCode::InvalidInputEncoding,
                         llvm::toString(classification.takeError()));
      if (classification->valueClass == LogicalValueClass::Infinity ||
          classification->valueClass == LogicalValueClass::QuietNaN ||
          classification->valueClass == LogicalValueClass::SignalingNaN)
        return bulkError(
            BulkTensorNumericErrorCode::InvalidInputEncoding,
            "managed-reference bulk execution admits only finite inputs");
    }
  }
  llvm::Expected<detail::UnqualifiedBulkExecutionResult> result =
      detail::executeBulkTensorForQualification(environment, command, inputs,
                                                destinationTemplate, budget);
  if (!result)
    return result.takeError();
  return BulkTensorNumericResult{
      std::move(result->destination), {}, std::move(result->evidence)};
}

} // namespace wafer
