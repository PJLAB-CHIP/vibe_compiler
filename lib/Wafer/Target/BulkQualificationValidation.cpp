//===- BulkQualificationValidation.cpp - Validation and admission ----===//

#include "BulkQualificationInternal.h"

#include "llvm/Support/JSON.h"

#include <utility>

namespace wafer {

using namespace bulk_qualification_detail;

namespace {

llvm::Expected<BulkQualificationKind>
parseQualificationKind(llvm::StringRef spelling) {
  if (spelling == "bit-exact")
    return BulkQualificationKind::BitExact;
  if (spelling == "profile-bounded")
    return BulkQualificationKind::ProfileBounded;
  return invalid("unknown bulk qualification kind " + spelling);
}

} // namespace

llvm::Error validateBulkBackend(const BulkExecutionEnvironment &environment,
                                llvm::StringRef policyPath,
                                llvm::StringRef outputPath,
                                FormalNumericWorkBudget formalBudget,
                                BulkNumericWorkBudget bulkBudget) {
  llvm::Expected<ParsedJSON> policy = loadCanonicalJSON(policyPath);
  if (!policy)
    return policy.takeError();
  const llvm::json::Object *object = policy->value.getAsObject();
  if (!object)
    return invalid("bulk frozen policy root must be an object");
  if (llvm::Error error = requireFields(*object,
                                        {"schema",
                                         "adapter_digest",
                                         "semantic_profile_digest",
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
                                         "resolution_digest",
                                         "qualification_kind",
                                         "maximum_absolute_error",
                                         "maximum_relative_error",
                                         "disjoint_proof"},
                                        "bulk frozen policy"))
    return error;
  llvm::Expected<llvm::StringRef> schema =
      requireString(*object, "schema", "bulk frozen policy");
  if (!schema)
    return schema.takeError();
  if (*schema != kPolicySchema)
    return invalid("bulk frozen policy schema mismatch");
  llvm::Expected<std::string> adapterDigest =
      requireDigest(*object, "adapter_digest", "bulk frozen policy");
  llvm::Expected<std::string> semanticDigest =
      requireDigest(*object, "semantic_profile_digest", "bulk frozen policy");
  llvm::Expected<std::string> frozenInputDigest = requireDigest(
      *object, "held_out_input_payload_digest", "bulk frozen policy");
  llvm::Expected<std::string> frozenDestinationDigest = requireDigest(
      *object, "held_out_destination_template_digest", "bulk frozen policy");
  llvm::Expected<std::string> calibrationSpecDigest =
      requireDigest(*object, "calibration_spec_digest", "bulk frozen policy");
  llvm::Expected<llvm::StringRef> valueDomain =
      requireString(*object, "value_domain", "bulk frozen policy");
  llvm::Expected<llvm::StringRef> targetComparator =
      requireString(*object, "target_comparator", "bulk frozen policy");
  llvm::Expected<llvm::StringRef> backendComparator =
      requireString(*object, "backend_comparator", "bulk frozen policy");
  llvm::Expected<llvm::StringRef> proofBasis =
      requireString(*object, "proof_basis", "bulk frozen policy");
  if (llvm::Error error = takeExpectedErrors(
          adapterDigest, semanticDigest, frozenInputDigest,
          frozenDestinationDigest, calibrationSpecDigest, valueDomain,
          targetComparator, backendComparator, proofBasis))
    return error;
  if (*adapterDigest != getBulkAdapterIdentityDigest() ||
      *valueDomain != kValueDomain || *targetComparator != kTargetComparator ||
      *backendComparator != kBackendComparator || *proofBasis != kProofBasis)
    return invalid("bulk frozen policy adapter/domain/proof mismatch");
  llvm::Expected<std::string> environmentDigest =
      requireDigest(*object, "environment_digest", "bulk frozen policy");
  llvm::Expected<std::string> environmentArtifactDigest =
      requireDigest(*object, "environment_record_digest", "bulk frozen policy");
  llvm::Expected<std::string> backendDigest =
      requireDigest(*object, "backend_digest", "bulk frozen policy");
  if (llvm::Error error = takeExpectedErrors(
          environmentDigest, environmentArtifactDigest, backendDigest))
    return error;
  if (*environmentDigest != environment.getDigest() ||
      *environmentArtifactDigest != environmentRecordDigest(environment) ||
      *backendDigest != environment.getBackend().getDigest())
    return invalid("bulk frozen policy environment/backend mismatch");
  llvm::Expected<llvm::StringRef> proof =
      requireString(*object, "disjoint_proof", "bulk frozen policy");
  if (!proof)
    return proof.takeError();
  if (*proof != "same-domain-distinct-seed-spec-and-payload-v2")
    return invalid("bulk frozen policy disjoint proof mismatch");
  const llvm::json::Object *specObject = object->getObject("held_out_spec");
  if (!specObject)
    return invalid("bulk frozen policy held-out spec must be an object");
  llvm::Expected<BulkQualificationSpec> spec = parseSpecObject(*specObject);
  if (!spec)
    return spec.takeError();
  llvm::Expected<std::string> specDigest =
      requireDigest(*object, "held_out_spec_digest", "bulk frozen policy");
  llvm::Expected<std::string> resolutionDigest =
      requireDigest(*object, "resolution_digest", "bulk frozen policy");
  llvm::Expected<llvm::StringRef> kindText =
      requireString(*object, "qualification_kind", "bulk frozen policy");
  llvm::Expected<double> maximumAbsolute = requireFiniteNonnegative(
      *object, "maximum_absolute_error", "bulk frozen policy");
  llvm::Expected<double> maximumRelative = requireFiniteNonnegative(
      *object, "maximum_relative_error", "bulk frozen policy");
  if (llvm::Error error =
          takeExpectedErrors(specDigest, resolutionDigest, kindText,
                             maximumAbsolute, maximumRelative))
    return error;
  if (*specDigest != spec->getDigest())
    return invalid("bulk frozen policy held-out spec digest mismatch");
  llvm::Expected<BulkQualificationKind> kind =
      parseQualificationKind(*kindText);
  if (!kind)
    return kind.takeError();
  if (*kind != BulkQualificationKind::ProfileBounded)
    return invalid("finite-corpus policy cannot claim bit-exact proof");
  llvm::Expected<BulkQualificationCase> testCase =
      materializeBulkQualificationCase(std::move(*spec), bulkBudget);
  if (!testCase)
    return testCase.takeError();
  if (testCase->getCommand().getDigest() != *resolutionDigest ||
      !testCase->getCommand().getSemantics() ||
      testCase->getCommand().getSemantics()->getDigest() != *semanticDigest ||
      computeBulkTensorPayloadDigest(testCase->getInputs()) !=
          *frozenInputDigest ||
      computeBulkTensorStorageDigest(testCase->getDestinationTemplate()) !=
          *frozenDestinationDigest)
    return invalid("held-out command/payload changed from the frozen policy");
  llvm::Expected<QualificationRun> run = runQualification(
      environment, std::move(*testCase), formalBudget, bulkBudget);
  if (!run)
    return run.takeError();
  if (run->testCase.getCommand().getDigest() != *resolutionDigest)
    return invalid("held-out command changed from the frozen resolution");
  if (run->comparison.maximumAbsoluteError > *maximumAbsolute ||
      run->comparison.maximumRelativeError > *maximumRelative ||
      (*kind == BulkQualificationKind::BitExact && !run->comparison.rawExact))
    return invalid("held-out result exceeds the frozen comparator envelope");
  llvm::Expected<std::string> calibrationDigest =
      requireDigest(*object, "calibration_digest", "bulk frozen policy");
  if (!calibrationDigest)
    return calibrationDigest.takeError();
  llvm::json::Object finalRecord{
      {"schema", kFinalSchema},
      {"adapter_digest", *adapterDigest},
      {"semantic_profile_digest", *semanticDigest},
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
      {"resolution_digest", run->testCase.getCommand().getDigest()},
      {"input_payload_digest",
       computeBulkTensorPayloadDigest(run->testCase.getInputs())},
      {"destination_template_digest",
       computeBulkTensorStorageDigest(run->testCase.getDestinationTemplate())},
      {"formal_output_digest", run->formalOutputDigest},
      {"backend_output_digest",
       computeBulkTensorStorageDigest(run->backend.destination)},
      {"qualification_kind", stringifyBulkQualificationKind(*kind)},
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
  return publishNoReplace(outputPath, canonicalJSON(std::move(finalRecord)));
}

llvm::Expected<VerifiedBulkQualificationRecord>
loadVerifiedBulkQualificationRecord(llvm::StringRef path) {
  llvm::Expected<ParsedJSON> parsed = loadCanonicalJSON(path);
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *object = parsed->value.getAsObject();
  if (!object)
    return invalid("bulk qualification record root must be an object");
  if (llvm::Error error = requireFields(*object,
                                        {"schema",
                                         "adapter_digest",
                                         "semantic_profile_digest",
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
                                         "resolution_digest",
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
                                        "bulk qualification record"))
    return error;
  llvm::Expected<llvm::StringRef> schema =
      requireString(*object, "schema", "bulk qualification record");
  if (!schema)
    return schema.takeError();
  if (*schema != kFinalSchema)
    return invalid("bulk qualification record schema mismatch");
  llvm::Expected<std::string> policyDigest =
      requireDigest(*object, "policy_digest", "bulk qualification record");
  llvm::Expected<std::string> adapterDigest =
      requireDigest(*object, "adapter_digest", "bulk qualification record");
  llvm::Expected<std::string> semanticDigest = requireDigest(
      *object, "semantic_profile_digest", "bulk qualification record");
  llvm::Expected<std::string> calibrationDigest =
      requireDigest(*object, "calibration_digest", "bulk qualification record");
  llvm::Expected<std::string> backendDigest =
      requireDigest(*object, "backend_digest", "bulk qualification record");
  llvm::Expected<std::string> formalOutputDigest = requireDigest(
      *object, "formal_output_digest", "bulk qualification record");
  llvm::Expected<std::string> specDigest =
      requireDigest(*object, "spec_digest", "bulk qualification record");
  llvm::Expected<std::string> environmentDigest =
      requireDigest(*object, "environment_digest", "bulk qualification record");
  llvm::Expected<std::string> environmentArtifactDigest = requireDigest(
      *object, "environment_record_digest", "bulk qualification record");
  llvm::Expected<std::string> resolutionDigest =
      requireDigest(*object, "resolution_digest", "bulk qualification record");
  llvm::Expected<std::string> inputPayloadDigest = requireDigest(
      *object, "input_payload_digest", "bulk qualification record");
  llvm::Expected<std::string> destinationTemplateDigest = requireDigest(
      *object, "destination_template_digest", "bulk qualification record");
  llvm::Expected<std::string> backendOutputDigest = requireDigest(
      *object, "backend_output_digest", "bulk qualification record");
  llvm::Expected<llvm::StringRef> kindText =
      requireString(*object, "qualification_kind", "bulk qualification record");
  llvm::Expected<llvm::StringRef> implementation =
      requireString(*object, "implementation", "bulk qualification record");
  llvm::Expected<std::string> descriptorDigest = requireDigest(
      *object, "resolved_descriptor_digest", "bulk qualification record");
  llvm::Expected<llvm::StringRef> valueDomain =
      requireString(*object, "value_domain", "bulk qualification record");
  llvm::Expected<llvm::StringRef> targetComparator =
      requireString(*object, "target_comparator", "bulk qualification record");
  llvm::Expected<llvm::StringRef> backendComparator =
      requireString(*object, "backend_comparator", "bulk qualification record");
  llvm::Expected<llvm::StringRef> proofBasis =
      requireString(*object, "proof_basis", "bulk qualification record");
  llvm::Expected<double> maximumAbsolute = requireFiniteNonnegative(
      *object, "maximum_absolute_error", "bulk qualification record");
  llvm::Expected<double> maximumRelative = requireFiniteNonnegative(
      *object, "maximum_relative_error", "bulk qualification record");
  llvm::Expected<double> observedAbsolute = requireFiniteNonnegative(
      *object, "observed_absolute_error", "bulk qualification record");
  llvm::Expected<double> observedRelative = requireFiniteNonnegative(
      *object, "observed_relative_error", "bulk qualification record");
  std::optional<bool> rawExact = object->getBoolean("raw_exact");
  const llvm::json::Object *flagsObject = object->getObject("formal_flags");
  if (llvm::Error error = takeExpectedErrors(
          policyDigest, adapterDigest, semanticDigest, calibrationDigest,
          backendDigest, formalOutputDigest, specDigest, environmentDigest,
          environmentArtifactDigest, resolutionDigest, inputPayloadDigest,
          destinationTemplateDigest, backendOutputDigest, kindText,
          implementation, descriptorDigest, valueDomain, targetComparator,
          backendComparator, proofBasis, maximumAbsolute, maximumRelative,
          observedAbsolute, observedRelative))
    return error;
  if (!rawExact || !flagsObject)
    return invalid(
        "bulk qualification record admission identity is incomplete");
  const llvm::json::Value *environmentValue = object->get("environment");
  const llvm::json::Object *environmentObject =
      environmentValue ? environmentValue->getAsObject() : nullptr;
  if (!environmentObject)
    return invalid("bulk qualification environment must be an object");
  if (llvm::Error error = validateEnvironmentJSON(
          *environmentObject, *environmentDigest, *backendDigest))
    return error;
  if (sha256(canonicalJSON(*environmentValue)) != *environmentArtifactDigest)
    return invalid("bulk qualification environment record digest mismatch");
  llvm::Expected<BulkQualificationKind> kind =
      parseQualificationKind(*kindText);
  llvm::Expected<FormalNumericExceptionFlags> flags = parseFlags(*flagsObject);
  if (llvm::Error error = takeExpectedErrors(kind, flags))
    return error;
  if (*kind != BulkQualificationKind::ProfileBounded ||
      *adapterDigest != getBulkAdapterIdentityDigest() ||
      *valueDomain != kValueDomain || *targetComparator != kTargetComparator ||
      *backendComparator != kBackendComparator || *proofBasis != kProofBasis ||
      implementation->empty() || *observedAbsolute > *maximumAbsolute ||
      *observedRelative > *maximumRelative ||
      (*rawExact && *formalOutputDigest != *backendOutputDigest))
    return invalid("bulk qualification record comparator/proof mismatch");
  llvm::Expected<uint64_t> matmul = requireUnsigned(
      *object, "matmul_invocations", "bulk qualification record");
  llvm::Expected<uint64_t> reorder = requireUnsigned(
      *object, "reorder_invocations", "bulk qualification record");
  llvm::Expected<uint64_t> backendFormalFMA = requireUnsigned(
      *object, "backend_formal_fma", "bulk qualification record");
  if (llvm::Error error = takeExpectedErrors(matmul, reorder, backendFormalFMA))
    return error;
  if (*matmul != 1 || *reorder > 1 || *backendFormalFMA != 0)
    return invalid("bulk qualification record does not prove one MatMul and "
                   "zero backend formal FMAs");
  return VerifiedBulkQualificationRecord(
      parsed->digest, std::move(*policyDigest), std::move(*adapterDigest),
      std::move(*semanticDigest), std::move(*specDigest),
      std::move(*environmentDigest), std::move(*resolutionDigest),
      std::move(*inputPayloadDigest), std::move(*destinationTemplateDigest),
      std::move(*backendOutputDigest), implementation->str(),
      std::move(*descriptorDigest), *kind, *flags);
}

llvm::Expected<BulkBackendAdmission>
VerifiedBulkQualificationRecord::createAdmission(
    const BulkExecutionEnvironment &environment,
    const ResolvedNumericCommand &command,
    llvm::ArrayRef<BulkTensorStorage> inputs,
    const BulkTensorStorage &destinationTemplate) const {
  if (getBulkAdapterIdentityDigest() != adapterDigest ||
      environment.getDigest() != environmentDigest || !command.getSemantics() ||
      command.getSemantics()->getDigest() != semanticProfileDigest ||
      command.getDigest() != resolutionDigest ||
      computeBulkTensorPayloadDigest(inputs) != inputPayloadDigest ||
      computeBulkTensorStorageDigest(destinationTemplate) !=
          destinationTemplateDigest)
    return invalid("bulk qualification record does not exact-match the "
                   "environment, command, payload or destination template");
  return BulkBackendAdmission(
      recordDigest, adapterDigest, semanticProfileDigest, resolutionDigest,
      inputPayloadDigest, destinationTemplateDigest, environmentDigest,
      expectedBackendOutputDigest, implementation, resolvedDescriptorDigest,
      kind, formalFlags);
}

} // namespace wafer
