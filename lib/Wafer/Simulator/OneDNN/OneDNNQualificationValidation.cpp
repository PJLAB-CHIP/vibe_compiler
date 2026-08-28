//===- OneDNNQualificationValidation.cpp - Record validation -----------===//

#include "OneDNNQualificationInternal.h"

#include "llvm/Support/JSON.h"

#include <utility>

namespace wafer {

using namespace onednn_qualification_detail;

namespace {

llvm::Expected<OneDNNQualificationKind>
parseQualificationKind(llvm::StringRef spelling) {
  if (spelling == "bit-exact")
    return OneDNNQualificationKind::BitExact;
  if (spelling == "profile-bounded")
    return OneDNNQualificationKind::ProfileBounded;
  return invalid("unknown onednn qualification kind " + spelling);
}

} // namespace

llvm::Error validateOneDNNBackend(const OneDNNExecutionEnvironment &environment,
                                  llvm::StringRef policyPath,
                                  llvm::StringRef outputPath,
                                  FormalNumericWorkBudget formalBudget,
                                  OneDNNNumericWorkBudget onednnBudget) {
  llvm::Expected<ParsedJSON> policy = loadCanonicalJSON(policyPath);
  if (!policy)
    return policy.takeError();
  const llvm::json::Object *object = policy->value.getAsObject();
  if (!object)
    return invalid("onednn frozen policy root must be an object");
  if (llvm::Error error = requireFields(*object,
                                        {"schema",
                                         "adapter_digest",
                                         "value_domain",
                                         "target_comparator",
                                         "backend_comparator",
                                         "proof_basis",
                                         "calibration_digest",
                                         "calibration_spec_digest",
                                         "held_out_spec",
                                         "held_out_spec_digest",
                                         "held_out_input_payload_digest",
                                         "held_out_destination_template_digest",
                                         "backend_digest",
                                         "environment_digest",
                                         "environment_record_digest",
                                         "problem_digest",
                                         "qualification_kind",
                                         "maximum_absolute_error",
                                         "maximum_relative_error",
                                         "disjoint_proof"},
                                        "onednn frozen policy"))
    return error;
  llvm::Expected<llvm::StringRef> schema =
      requireString(*object, "schema", "onednn frozen policy");
  if (!schema)
    return schema.takeError();
  if (*schema != kPolicySchema)
    return invalid("onednn frozen policy schema mismatch");
  llvm::Expected<std::string> adapterDigest =
      requireDigest(*object, "adapter_digest", "onednn frozen policy");
  llvm::Expected<std::string> frozenInputDigest = requireDigest(
      *object, "held_out_input_payload_digest", "onednn frozen policy");
  llvm::Expected<std::string> frozenDestinationDigest = requireDigest(
      *object, "held_out_destination_template_digest", "onednn frozen policy");
  llvm::Expected<std::string> calibrationSpecDigest =
      requireDigest(*object, "calibration_spec_digest", "onednn frozen policy");
  llvm::Expected<llvm::StringRef> valueDomain =
      requireString(*object, "value_domain", "onednn frozen policy");
  llvm::Expected<llvm::StringRef> targetComparator =
      requireString(*object, "target_comparator", "onednn frozen policy");
  llvm::Expected<llvm::StringRef> backendComparator =
      requireString(*object, "backend_comparator", "onednn frozen policy");
  llvm::Expected<llvm::StringRef> proofBasis =
      requireString(*object, "proof_basis", "onednn frozen policy");
  if (llvm::Error error = takeExpectedErrors(
          adapterDigest, frozenInputDigest, frozenDestinationDigest,
          calibrationSpecDigest, valueDomain, targetComparator,
          backendComparator, proofBasis))
    return error;
  if (*adapterDigest != getOneDNNAdapterContractDigest() ||
      *valueDomain != kValueDomain || *targetComparator != kTargetComparator ||
      *backendComparator != kBackendComparator || *proofBasis != kProofBasis)
    return invalid("onednn frozen policy adapter/domain/proof mismatch");
  llvm::Expected<std::string> environmentDigest =
      requireDigest(*object, "environment_digest", "onednn frozen policy");
  llvm::Expected<std::string> environmentRecordJSONDigest = requireDigest(
      *object, "environment_record_digest", "onednn frozen policy");
  llvm::Expected<std::string> backendDigest =
      requireDigest(*object, "backend_digest", "onednn frozen policy");
  if (llvm::Error error = takeExpectedErrors(
          environmentDigest, environmentRecordJSONDigest, backendDigest))
    return error;
  if (*environmentDigest != environment.getDigest() ||
      *environmentRecordJSONDigest != environmentRecordDigest(environment) ||
      *backendDigest != environment.getBackend().getDigest())
    return invalid("onednn frozen policy environment/backend mismatch");
  llvm::Expected<llvm::StringRef> proof =
      requireString(*object, "disjoint_proof", "onednn frozen policy");
  if (!proof)
    return proof.takeError();
  if (*proof != "same-domain-distinct-seed-spec-and-payload")
    return invalid("onednn frozen policy disjoint proof mismatch");
  const llvm::json::Object *specObject = object->getObject("held_out_spec");
  if (!specObject)
    return invalid("onednn frozen policy held-out spec must be an object");
  llvm::Expected<OneDNNQualificationSpec> spec = parseSpecObject(*specObject);
  if (!spec)
    return spec.takeError();
  llvm::Expected<std::string> specDigest =
      requireDigest(*object, "held_out_spec_digest", "onednn frozen policy");
  llvm::Expected<std::string> problemDigest =
      requireDigest(*object, "problem_digest", "onednn frozen policy");
  llvm::Expected<llvm::StringRef> kindText =
      requireString(*object, "qualification_kind", "onednn frozen policy");
  llvm::Expected<double> maximumAbsolute = requireFiniteNonnegative(
      *object, "maximum_absolute_error", "onednn frozen policy");
  llvm::Expected<double> maximumRelative = requireFiniteNonnegative(
      *object, "maximum_relative_error", "onednn frozen policy");
  if (llvm::Error error =
          takeExpectedErrors(specDigest, problemDigest, kindText,
                             maximumAbsolute, maximumRelative))
    return error;
  if (*specDigest != spec->getDigest())
    return invalid("onednn frozen policy held-out spec digest mismatch");
  llvm::Expected<OneDNNQualificationKind> kind =
      parseQualificationKind(*kindText);
  if (!kind)
    return kind.takeError();
  if (*kind != OneDNNQualificationKind::ProfileBounded)
    return invalid("finite-corpus policy cannot claim bit-exact proof");
  llvm::Expected<OneDNNQualificationCase> testCase =
      materializeOneDNNQualificationCase(std::move(*spec), onednnBudget);
  if (!testCase)
    return testCase.takeError();
  llvm::Expected<std::string> materializedProblemDigest =
      computeOneDNNGemmProblemDigest(testCase->getOperation());
  if (!materializedProblemDigest)
    return materializedProblemDigest.takeError();
  if (*materializedProblemDigest != *problemDigest ||
      computeOneDNNTensorPayloadDigest(testCase->getInputs()) !=
          *frozenInputDigest ||
      computeOneDNNTensorStorageDigest(testCase->getDestinationTemplate()) !=
          *frozenDestinationDigest)
    return invalid("held-out operation/payload changed from the frozen policy");
  llvm::Expected<QualificationRun> run = runQualification(
      environment, std::move(*testCase), formalBudget, onednnBudget);
  if (!run)
    return run.takeError();
  llvm::Expected<std::string> executedProblemDigest =
      computeOneDNNGemmProblemDigest(run->testCase.getOperation());
  if (!executedProblemDigest)
    return executedProblemDigest.takeError();
  if (*executedProblemDigest != *problemDigest)
    return invalid("held-out GEMM problem changed from the frozen policy");
  if (run->comparison.maximumAbsoluteError > *maximumAbsolute ||
      run->comparison.maximumRelativeError > *maximumRelative ||
      (*kind == OneDNNQualificationKind::BitExact && !run->comparison.rawExact))
    return invalid("held-out result exceeds the frozen comparator envelope");
  llvm::Expected<std::string> calibrationDigest =
      requireDigest(*object, "calibration_digest", "onednn frozen policy");
  if (!calibrationDigest)
    return calibrationDigest.takeError();
  llvm::json::Object finalRecord{
      {"schema", kFinalSchema},
      {"adapter_digest", *adapterDigest},
      {"value_domain", *valueDomain},
      {"target_comparator", *targetComparator},
      {"backend_comparator", *backendComparator},
      {"proof_basis", *proofBasis},
      {"policy_digest", policy->digest},
      {"calibration_digest", *calibrationDigest},
      {"spec_digest", run->testCase.getSpec().getDigest()},
      {"backend_digest", environment.getBackend().getDigest()},
      {"environment_digest", environment.getDigest()},
      {"environment", environmentJSON(environment)},
      {"environment_record_digest", environmentRecordDigest(environment)},
      {"problem_digest", *executedProblemDigest},
      {"input_payload_digest",
       computeOneDNNTensorPayloadDigest(run->testCase.getInputs())},
      {"destination_template_digest",
       computeOneDNNTensorStorageDigest(
           run->testCase.getDestinationTemplate())},
      {"formal_output_digest", run->formalOutputDigest},
      {"backend_output_digest",
       computeOneDNNTensorStorageDigest(run->backend.destination)},
      {"qualification_kind", stringifyOneDNNQualificationKind(*kind)},
      {"maximum_absolute_error", *maximumAbsolute},
      {"maximum_relative_error", *maximumRelative},
      {"observed_absolute_error", run->comparison.maximumAbsoluteError},
      {"observed_relative_error", run->comparison.maximumRelativeError},
      {"raw_exact", run->comparison.rawExact},
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
  return writeFileNoReplace(outputPath, canonicalJSON(std::move(finalRecord)));
}

llvm::Expected<VerifiedOneDNNQualificationRecord>
loadVerifiedOneDNNQualificationRecord(llvm::StringRef path) {
  llvm::Expected<ParsedJSON> parsed = loadCanonicalJSON(path);
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *object = parsed->value.getAsObject();
  if (!object)
    return invalid("onednn qualification record root must be an object");
  if (llvm::Error error = requireFields(*object,
                                        {"schema",
                                         "adapter_digest",
                                         "value_domain",
                                         "target_comparator",
                                         "backend_comparator",
                                         "proof_basis",
                                         "policy_digest",
                                         "calibration_digest",
                                         "spec_digest",
                                         "backend_digest",
                                         "environment_digest",
                                         "environment",
                                         "environment_record_digest",
                                         "problem_digest",
                                         "input_payload_digest",
                                         "destination_template_digest",
                                         "formal_output_digest",
                                         "backend_output_digest",
                                         "qualification_kind",
                                         "maximum_absolute_error",
                                         "maximum_relative_error",
                                         "observed_absolute_error",
                                         "observed_relative_error",
                                         "raw_exact",
                                         "formal_flags",
                                         "implementation",
                                         "resolved_descriptor_digest",
                                         "matmul_invocations",
                                         "reorder_invocations",
                                         "backend_formal_fma"},
                                        "onednn qualification record"))
    return error;
  llvm::Expected<llvm::StringRef> schema =
      requireString(*object, "schema", "onednn qualification record");
  if (!schema)
    return schema.takeError();
  if (*schema != kFinalSchema)
    return invalid("onednn qualification record schema mismatch");
  llvm::Expected<std::string> policyDigest =
      requireDigest(*object, "policy_digest", "onednn qualification record");
  llvm::Expected<std::string> adapterDigest =
      requireDigest(*object, "adapter_digest", "onednn qualification record");
  llvm::Expected<std::string> calibrationDigest = requireDigest(
      *object, "calibration_digest", "onednn qualification record");
  llvm::Expected<std::string> backendDigest =
      requireDigest(*object, "backend_digest", "onednn qualification record");
  llvm::Expected<std::string> formalOutputDigest = requireDigest(
      *object, "formal_output_digest", "onednn qualification record");
  llvm::Expected<std::string> specDigest =
      requireDigest(*object, "spec_digest", "onednn qualification record");
  llvm::Expected<std::string> environmentDigest = requireDigest(
      *object, "environment_digest", "onednn qualification record");
  llvm::Expected<std::string> environmentRecordJSONDigest = requireDigest(
      *object, "environment_record_digest", "onednn qualification record");
  llvm::Expected<std::string> problemDigest =
      requireDigest(*object, "problem_digest", "onednn qualification record");
  llvm::Expected<std::string> inputPayloadDigest = requireDigest(
      *object, "input_payload_digest", "onednn qualification record");
  llvm::Expected<std::string> destinationTemplateDigest = requireDigest(
      *object, "destination_template_digest", "onednn qualification record");
  llvm::Expected<std::string> backendOutputDigest = requireDigest(
      *object, "backend_output_digest", "onednn qualification record");
  llvm::Expected<llvm::StringRef> kindText = requireString(
      *object, "qualification_kind", "onednn qualification record");
  llvm::Expected<llvm::StringRef> implementation =
      requireString(*object, "implementation", "onednn qualification record");
  llvm::Expected<std::string> descriptorDigest = requireDigest(
      *object, "resolved_descriptor_digest", "onednn qualification record");
  llvm::Expected<llvm::StringRef> valueDomain =
      requireString(*object, "value_domain", "onednn qualification record");
  llvm::Expected<llvm::StringRef> targetComparator = requireString(
      *object, "target_comparator", "onednn qualification record");
  llvm::Expected<llvm::StringRef> backendComparator = requireString(
      *object, "backend_comparator", "onednn qualification record");
  llvm::Expected<llvm::StringRef> proofBasis =
      requireString(*object, "proof_basis", "onednn qualification record");
  llvm::Expected<double> maximumAbsolute = requireFiniteNonnegative(
      *object, "maximum_absolute_error", "onednn qualification record");
  llvm::Expected<double> maximumRelative = requireFiniteNonnegative(
      *object, "maximum_relative_error", "onednn qualification record");
  llvm::Expected<double> observedAbsolute = requireFiniteNonnegative(
      *object, "observed_absolute_error", "onednn qualification record");
  llvm::Expected<double> observedRelative = requireFiniteNonnegative(
      *object, "observed_relative_error", "onednn qualification record");
  std::optional<bool> rawExact = object->getBoolean("raw_exact");
  const llvm::json::Object *flagsObject = object->getObject("formal_flags");
  if (llvm::Error error = takeExpectedErrors(
          policyDigest, adapterDigest, calibrationDigest, backendDigest,
          formalOutputDigest, specDigest, environmentDigest,
          environmentRecordJSONDigest, problemDigest, inputPayloadDigest,
          destinationTemplateDigest, backendOutputDigest, kindText,
          implementation, descriptorDigest, valueDomain, targetComparator,
          backendComparator, proofBasis, maximumAbsolute, maximumRelative,
          observedAbsolute, observedRelative))
    return error;
  if (!rawExact || !flagsObject)
    return invalid(
        "onednn qualification record execution binding is incomplete");
  const llvm::json::Value *environmentValue = object->get("environment");
  const llvm::json::Object *environmentObject =
      environmentValue ? environmentValue->getAsObject() : nullptr;
  if (!environmentObject)
    return invalid("onednn qualification environment must be an object");
  if (llvm::Error error = validateEnvironmentJSON(
          *environmentObject, *environmentDigest, *backendDigest))
    return error;
  if (sha256(canonicalJSON(*environmentValue)) != *environmentRecordJSONDigest)
    return invalid("onednn qualification environment record digest mismatch");
  llvm::Expected<OneDNNQualificationKind> kind =
      parseQualificationKind(*kindText);
  llvm::Expected<FormalNumericExceptionFlags> flags = parseFlags(*flagsObject);
  if (llvm::Error error = takeExpectedErrors(kind, flags))
    return error;
  if (*kind != OneDNNQualificationKind::ProfileBounded ||
      *adapterDigest != getOneDNNAdapterContractDigest() ||
      *valueDomain != kValueDomain || *targetComparator != kTargetComparator ||
      *backendComparator != kBackendComparator || *proofBasis != kProofBasis ||
      implementation->empty() || *observedAbsolute > *maximumAbsolute ||
      *observedRelative > *maximumRelative ||
      (*rawExact && *formalOutputDigest != *backendOutputDigest))
    return invalid("onednn qualification record comparator/proof mismatch");
  llvm::Expected<uint64_t> matmul = requireUnsigned(
      *object, "matmul_invocations", "onednn qualification record");
  llvm::Expected<uint64_t> reorder = requireUnsigned(
      *object, "reorder_invocations", "onednn qualification record");
  llvm::Expected<uint64_t> backendFormalFMA = requireUnsigned(
      *object, "backend_formal_fma", "onednn qualification record");
  if (llvm::Error error = takeExpectedErrors(matmul, reorder, backendFormalFMA))
    return error;
  if (*matmul != 1 || *reorder > 1 || *backendFormalFMA != 0)
    return invalid("onednn qualification record does not prove one MatMul and "
                   "zero backend formal FMAs");
  return VerifiedOneDNNQualificationRecord(
      parsed->digest, std::move(*policyDigest), std::move(*adapterDigest),
      std::move(*specDigest), std::move(*environmentDigest),
      std::move(*problemDigest), std::move(*inputPayloadDigest),
      std::move(*destinationTemplateDigest), std::move(*backendOutputDigest),
      implementation->str(), std::move(*descriptorDigest), *kind, *flags);
}

llvm::Expected<QualifiedOneDNNExecution>
VerifiedOneDNNQualificationRecord::qualifyExecution(
    const OneDNNExecutionEnvironment &environment,
    const FormalGemmOperation &operation,
    llvm::ArrayRef<OneDNNTensorStorage> inputs,
    const OneDNNTensorStorage &destinationTemplate) const {
  llvm::Expected<std::string> actualProblemDigest =
      computeOneDNNGemmProblemDigest(operation);
  if (!actualProblemDigest)
    return actualProblemDigest.takeError();
  if (getOneDNNAdapterContractDigest() != adapterDigest ||
      environment.getDigest() != environmentDigest ||
      *actualProblemDigest != problemDigest ||
      computeOneDNNTensorPayloadDigest(inputs) != inputPayloadDigest ||
      computeOneDNNTensorStorageDigest(destinationTemplate) !=
          destinationTemplateDigest)
    return invalid("onednn qualification record does not exact-match the "
                   "environment, operation, payload or destination template");
  return QualifiedOneDNNExecution(
      recordDigest, adapterDigest, problemDigest, inputPayloadDigest,
      destinationTemplateDigest, environmentDigest, expectedBackendOutputDigest,
      implementation, resolvedDescriptorDigest, kind, formalFlags);
}

} // namespace wafer
