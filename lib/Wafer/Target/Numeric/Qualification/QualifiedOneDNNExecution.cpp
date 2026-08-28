//===- QualifiedOneDNNExecution.cpp - Qualified onednn execution
//------------===//

#include "OneDNNTensorNumericInternal.h"

#include <utility>

namespace wafer {

using namespace onednn_detail;

llvm::Expected<OneDNNTensorNumericResult> executeQualifiedOneDNNTensorNumeric(
    const OneDNNExecutionEnvironment &environment,
    const QualifiedOneDNNExecution &execution,
    const FormalGemmOperation &operation,
    llvm::ArrayRef<OneDNNTensorStorage> inputs,
    const OneDNNTensorStorage &destinationTemplate,
    OneDNNNumericWorkBudget budget) {
  if (execution.getEnvironmentDigest() != environment.getDigest())
    return onednnError(
        OneDNNTensorNumericErrorCode::EnvironmentMismatch,
        "onednn qualification does not match the current environment");
  llvm::Expected<std::string> problemDigest =
      computeOneDNNGemmProblemDigest(operation);
  if (!problemDigest)
    return problemDigest.takeError();
  if (execution.getAdapterDigest() != getOneDNNAdapterContractDigest() ||
      execution.getProblemDigest() != *problemDigest ||
      execution.getInputPayloadDigest() !=
          computeOneDNNTensorPayloadDigest(inputs) ||
      execution.getDestinationTemplateDigest() !=
          computeOneDNNTensorStorageDigest(destinationTemplate))
    return onednnError(
        OneDNNTensorNumericErrorCode::QualificationMismatch,
        "onednn operation, payload or destination template is not "
        "the frozen qualified row");
  llvm::Expected<detail::UnqualifiedOneDNNExecutionResult> result =
      detail::executeOneDNNTensorForQualification(
          environment, operation, inputs, destinationTemplate, budget);
  if (!result)
    return result.takeError();
  if (result->evidence.implementation !=
          execution.getExpectedImplementation() ||
      result->evidence.resolvedDescriptorDigest !=
          execution.getExpectedResolvedDescriptorDigest())
    return onednnError(OneDNNTensorNumericErrorCode::QualificationMismatch,
                       "oneDNN implementation or resolved descriptor changed "
                       "from the frozen qualified row");
  if (computeOneDNNTensorStorageDigest(result->destination) !=
      execution.getExpectedBackendOutputDigest())
    return onednnError(
        OneDNNTensorNumericErrorCode::BackendOutputMismatch,
        "oneDNN output changed from the frozen validated result");
  return OneDNNTensorNumericResult{std::move(result->destination),
                                   execution.getFormalFlags(),
                                   std::move(result->evidence)};
}

llvm::Expected<OneDNNTensorNumericResult>
executeManagedReferenceOneDNNTensorNumeric(
    const OneDNNExecutionEnvironment &environment,
    const FormalGemmOperation &operation,
    llvm::ArrayRef<OneDNNTensorStorage> inputs,
    const OneDNNTensorStorage &destinationTemplate,
    OneDNNNumericWorkBudget budget) {
  for (const OneDNNTensorStorage &input : inputs) {
    llvm::Expected<std::vector<RawLogicalValue>> values =
        unpackOneDNNTensorLogicalValues(input);
    if (!values)
      return values.takeError();
    for (RawLogicalValue value : *values) {
      llvm::Expected<LogicalValueClassification> classification =
          classifyRawLogicalValue(value, NonCanonicalEncodingPolicy::Reject);
      if (!classification)
        return onednnError(OneDNNTensorNumericErrorCode::InvalidInputEncoding,
                           llvm::toString(classification.takeError()));
      if (classification->valueClass == LogicalValueClass::Infinity ||
          classification->valueClass == LogicalValueClass::QuietNaN ||
          classification->valueClass == LogicalValueClass::SignalingNaN)
        return onednnError(
            OneDNNTensorNumericErrorCode::InvalidInputEncoding,
            "managed-reference onednn execution admits only finite inputs");
    }
  }
  llvm::Expected<detail::UnqualifiedOneDNNExecutionResult> result =
      detail::executeOneDNNTensorForQualification(
          environment, operation, inputs, destinationTemplate, budget);
  if (!result)
    return result.takeError();
  return OneDNNTensorNumericResult{
      std::move(result->destination), {}, std::move(result->evidence)};
}

} // namespace wafer
