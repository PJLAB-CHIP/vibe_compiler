//===- OneDNNQualificationCalibration.cpp - Calibration producer ------===//

#include "OneDNNQualificationInternal.h"

#include "llvm/Support/JSON.h"

#include <utility>

namespace wafer {

using namespace onednn_qualification_detail;

llvm::Error
calibrateOneDNNBackend(const OneDNNExecutionEnvironment &environment,
                       llvm::StringRef specPath, llvm::StringRef outputPath,
                       FormalNumericWorkBudget formalBudget,
                       OneDNNNumericWorkBudget onednnBudget) {
  llvm::Expected<OneDNNQualificationSpec> spec =
      loadOneDNNQualificationSpec(specPath);
  if (!spec)
    return spec.takeError();
  llvm::Expected<OneDNNQualificationCase> testCase =
      materializeOneDNNQualificationCase(std::move(*spec), onednnBudget);
  if (!testCase)
    return testCase.takeError();
  llvm::Expected<QualificationRun> run = runQualification(
      environment, std::move(*testCase), formalBudget, onednnBudget);
  if (!run)
    return run.takeError();
  llvm::Expected<std::string> problemDigest =
      computeOneDNNGemmProblemDigest(run->testCase.getOperation());
  if (!problemDigest)
    return problemDigest.takeError();
  llvm::json::Object record{
      {"schema", kCalibrationSchema},
      {"adapter_digest", getOneDNNAdapterContractDigest()},
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
       computeOneDNNTensorPayloadDigest(run->testCase.getInputs())},
      {"destination_template_digest",
       computeOneDNNTensorStorageDigest(
           run->testCase.getDestinationTemplate())},
      {"formal_output_digest", run->formalOutputDigest},
      {"backend_output_digest",
       computeOneDNNTensorStorageDigest(run->backend.destination)},
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
