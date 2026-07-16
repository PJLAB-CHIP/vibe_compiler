//===- BulkAdmission.cpp - Exact-match bulk admission execution -----===//

#include "BulkTensorNumericInternal.h"

#include <utility>

namespace wafer {

using namespace bulk_detail;

llvm::Expected<BulkTensorNumericResult>
executeAdmittedBulkTensorNumeric(const BulkExecutionEnvironment &environment,
                                 const BulkBackendAdmission &admission,
                                 const ResolvedNumericCommand &command,
                                 llvm::ArrayRef<BulkTensorStorage> inputs,
                                 const BulkTensorStorage &destinationTemplate,
                                 BulkNumericWorkBudget budget) {
  if (admission.getEnvironmentDigest() != environment.getDigest())
    return bulkError(BulkTensorNumericErrorCode::EnvironmentMismatch,
                     "bulk admission does not match the current environment");
  if (admission.getAdapterDigest() != getBulkAdapterIdentityDigest() ||
      !command.getSemantics() ||
      admission.getSemanticProfileDigest() !=
          command.getSemantics()->getDigest() ||
      admission.getResolutionDigest() != command.getDigest() ||
      admission.getInputPayloadDigest() !=
          computeBulkTensorPayloadDigest(inputs) ||
      admission.getDestinationTemplateDigest() !=
          computeBulkTensorStorageDigest(destinationTemplate))
    return bulkError(BulkTensorNumericErrorCode::AdmissionMismatch,
                     "bulk command, payload or destination template is not "
                     "the frozen qualified row");
  llvm::Expected<detail::UnqualifiedBulkExecutionResult> result =
      detail::executeBulkTensorForQualification(environment, command, inputs,
                                                destinationTemplate, budget);
  if (!result)
    return result.takeError();
  if (result->evidence.implementation !=
          admission.getExpectedImplementation() ||
      result->evidence.resolvedDescriptorDigest !=
          admission.getExpectedResolvedDescriptorDigest())
    return bulkError(BulkTensorNumericErrorCode::AdmissionMismatch,
                     "oneDNN implementation or resolved descriptor changed "
                     "from the frozen qualified row");
  if (computeBulkTensorStorageDigest(result->destination) !=
      admission.getExpectedBackendOutputDigest())
    return bulkError(BulkTensorNumericErrorCode::BackendOutputMismatch,
                     "oneDNN output changed from the frozen validated result");
  return BulkTensorNumericResult{std::move(result->destination),
                                 admission.getFormalFlags(),
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
