//===- BulkQualificationCalibration.cpp - Calibration producer ------===//

#include "BulkQualificationInternal.h"

#include "llvm/Support/JSON.h"

#include <utility>

namespace wafer {

using namespace bulk_qualification_detail;

llvm::Error calibrateBulkBackend(const BulkExecutionEnvironment &environment,
                                 llvm::StringRef specPath,
                                 llvm::StringRef outputPath,
                                 FormalNumericWorkBudget formalBudget,
                                 BulkNumericWorkBudget bulkBudget) {
  llvm::Expected<BulkQualificationSpec> spec =
      loadBulkQualificationSpec(specPath);
  if (!spec)
    return spec.takeError();
  llvm::Expected<BulkQualificationCase> testCase =
      materializeBulkQualificationCase(std::move(*spec), bulkBudget);
  if (!testCase)
    return testCase.takeError();
  llvm::Expected<QualificationRun> run = runQualification(
      environment, std::move(*testCase), formalBudget, bulkBudget);
  if (!run)
    return run.takeError();
  llvm::Expected<std::string> problemDigest =
      computeBulkGemmProblemDigest(run->testCase.getOperation());
  if (!problemDigest)
    return problemDigest.takeError();
  llvm::json::Object record{
      {"schema", kCalibrationSchema},
      {"adapter_digest", getBulkAdapterContractDigest()},
      {"spec", specJSON(run->testCase.getSpec())},
      {"spec_digest", run->testCase.getSpec().getDigest()},
      {"value_domain", kValueDomain},
      {"target_comparator", kTargetComparator},
      {"backend_comparator", kBackendComparator},
      {"backend_digest", environment.getBackend().getDigest()},
      {"environment_digest", environment.getDigest()},
      {"environment", environmentJSON(environment)},
      {"environment_record_digest", environmentRecordDigest(environment)},
      {"problem_digest", *problemDigest},
      {"input_payload_digest",
       computeBulkTensorPayloadDigest(run->testCase.getInputs())},
      {"destination_template_digest",
       computeBulkTensorStorageDigest(run->testCase.getDestinationTemplate())},
      {"formal_output_digest", run->formalOutputDigest},
      {"backend_output_digest",
       computeBulkTensorStorageDigest(run->backend.destination)},
      {"raw_exact", run->comparison.rawExact},
      {"maximum_absolute_error", run->comparison.maximumAbsoluteError},
      {"maximum_relative_error", run->comparison.maximumRelativeError},
      {"formal_flags", flagsJSON(run->formal.flags)},
      {"implementation", run->backend.evidence.implementation},
      {"resolved_descriptor_digest",
       run->backend.evidence.resolvedDescriptorDigest},
      {"matmul_invocations",
       static_cast<int64_t>(run->backend.evidence.matmulInvocations)},
      {"reorder_invocations",
       static_cast<int64_t>(run->backend.evidence.reorderInvocations)},
      {"backend_formal_fma",
       static_cast<int64_t>(run->backend.evidence.formalFusedMultiplyAdds)},
  };
  return writeFileNoReplace(outputPath, canonicalJSON(std::move(record)));
}

} // namespace wafer
