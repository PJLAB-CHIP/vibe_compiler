//===- OneDNNQualificationPolicy.cpp - Frozen policy producer ----------===//

#include "OneDNNQualificationInternal.h"

#include "llvm/Support/JSON.h"

#include <cmath>
#include <utility>

namespace wafer {

using namespace onednn_qualification_detail;

namespace {

bool sameDomain(const OneDNNQualificationSpec &lhs,
                const OneDNNQualificationSpec &rhs) {
  return lhs.getFormat() == rhs.getFormat() && lhs.getM() == rhs.getM() &&
         lhs.getK() == rhs.getK() && lhs.getN() == rhs.getN() &&
         lhs.getBatchCount() == rhs.getBatchCount() &&
         lhs.getLHSLayout() == rhs.getLHSLayout() &&
         lhs.getRHSLayout() == rhs.getRHSLayout() &&
         lhs.getDestinationLayout() == rhs.getDestinationLayout();
}

} // namespace

llvm::Error freezeOneDNNBackendPolicy(llvm::StringRef calibrationPath,
                                      llvm::StringRef heldOutSpecPath,
                                      llvm::StringRef outputPath,
                                      OneDNNQualificationTolerance tolerance) {
  if (!std::isfinite(tolerance.maximumAbsoluteError) ||
      !std::isfinite(tolerance.maximumRelativeError) ||
      tolerance.maximumAbsoluteError < 0.0 ||
      tolerance.maximumRelativeError < 0.0)
    return invalid("onednn qualification tolerance must be finite/nonnegative");
  llvm::Expected<ParsedJSON> calibration = loadCanonicalJSON(calibrationPath);
  if (!calibration)
    return calibration.takeError();
  const llvm::json::Object *object = calibration->value.getAsObject();
  if (!object)
    return invalid("onednn calibration root must be an object");
  if (llvm::Error error = requireFields(*object,
                                        {"schema",
                                         "adapter_digest",
                                         "spec",
                                         "spec_digest",
                                         "value_domain",
                                         "target_comparator",
                                         "backend_comparator",
                                         "backend_digest",
                                         "environment_digest",
                                         "environment",
                                         "environment_record_digest",
                                         "problem_digest",
                                         "input_payload_digest",
                                         "destination_template_digest",
                                         "formal_output_digest",
                                         "backend_output_digest",
                                         "raw_exact",
                                         "maximum_absolute_error",
                                         "maximum_relative_error",
                                         "formal_flags",
                                         "implementation",
                                         "resolved_descriptor_digest",
                                         "matmul_invocations",
                                         "reorder_invocations",
                                         "backend_formal_fma"},
                                        "onednn calibration"))
    return error;
  llvm::Expected<llvm::StringRef> schema =
      requireString(*object, "schema", "onednn calibration");
  if (!schema)
    return schema.takeError();
  if (*schema != kCalibrationSchema)
    return invalid("onednn calibration schema mismatch");
  const llvm::json::Object *calibrationSpecObject = object->getObject("spec");
  if (!calibrationSpecObject)
    return invalid("onednn calibration spec must be an object");
  llvm::Expected<OneDNNQualificationSpec> calibrationSpec =
      parseSpecObject(*calibrationSpecObject);
  if (!calibrationSpec)
    return calibrationSpec.takeError();
  llvm::Expected<std::string> calibrationSpecDigest =
      requireDigest(*object, "spec_digest", "onednn calibration");
  llvm::Expected<std::string> adapterDigest =
      requireDigest(*object, "adapter_digest", "onednn calibration");
  llvm::Expected<std::string> inputPayloadDigest =
      requireDigest(*object, "input_payload_digest", "onednn calibration");
  llvm::Expected<std::string> destinationTemplateDigest = requireDigest(
      *object, "destination_template_digest", "onednn calibration");
  llvm::Expected<std::string> formalOutputDigest =
      requireDigest(*object, "formal_output_digest", "onednn calibration");
  llvm::Expected<std::string> backendOutputDigest =
      requireDigest(*object, "backend_output_digest", "onednn calibration");
  llvm::Expected<std::string> descriptorDigest = requireDigest(
      *object, "resolved_descriptor_digest", "onednn calibration");
  llvm::Expected<llvm::StringRef> valueDomain =
      requireString(*object, "value_domain", "onednn calibration");
  llvm::Expected<llvm::StringRef> targetComparator =
      requireString(*object, "target_comparator", "onednn calibration");
  llvm::Expected<llvm::StringRef> backendComparator =
      requireString(*object, "backend_comparator", "onednn calibration");
  llvm::Expected<llvm::StringRef> implementation =
      requireString(*object, "implementation", "onednn calibration");
  llvm::Expected<double> observedAbsolute = requireFiniteNonnegative(
      *object, "maximum_absolute_error", "onednn calibration");
  llvm::Expected<double> observedRelative = requireFiniteNonnegative(
      *object, "maximum_relative_error", "onednn calibration");
  std::optional<bool> rawExact = object->getBoolean("raw_exact");
  const llvm::json::Object *flagsObject = object->getObject("formal_flags");
  llvm::Expected<uint64_t> matmul =
      requireUnsigned(*object, "matmul_invocations", "onednn calibration");
  llvm::Expected<uint64_t> reorder =
      requireUnsigned(*object, "reorder_invocations", "onednn calibration");
  llvm::Expected<uint64_t> backendFormalFMA =
      requireUnsigned(*object, "backend_formal_fma", "onednn calibration");
  if (llvm::Error error = takeExpectedErrors(
          calibrationSpecDigest, adapterDigest, inputPayloadDigest,
          destinationTemplateDigest, formalOutputDigest, backendOutputDigest,
          descriptorDigest, valueDomain, targetComparator, backendComparator,
          implementation, observedAbsolute, observedRelative, matmul, reorder,
          backendFormalFMA))
    return error;
  if (!rawExact || !flagsObject)
    return invalid("onednn calibration comparison identity is incomplete");
  llvm::Expected<FormalNumericExceptionFlags> flags = parseFlags(*flagsObject);
  if (!flags)
    return flags.takeError();
  if (*calibrationSpecDigest != calibrationSpec->getDigest())
    return invalid("onednn calibration spec digest mismatch");
  if (*adapterDigest != getOneDNNAdapterContractDigest() ||
      *valueDomain != kValueDomain || *targetComparator != kTargetComparator ||
      *backendComparator != kBackendComparator || implementation->empty() ||
      implementation->contains_insensitive("ref") || *matmul != 1 ||
      *reorder > 1 || *backendFormalFMA != 0)
    return invalid("onednn calibration adapter/comparator/evidence mismatch");
  if (*rawExact && *formalOutputDigest != *backendOutputDigest)
    return invalid("raw-exact calibration output digests disagree");
  if (*observedAbsolute > tolerance.maximumAbsoluteError ||
      *observedRelative > tolerance.maximumRelativeError)
    return invalid("frozen tolerance is tighter than calibration evidence");

  llvm::Expected<OneDNNQualificationSpec> heldOut =
      loadOneDNNQualificationSpec(heldOutSpecPath);
  if (!heldOut)
    return heldOut.takeError();
  if (!sameDomain(*calibrationSpec, *heldOut) ||
      calibrationSpec->getSeed() == heldOut->getSeed() ||
      calibrationSpec->getDigest() == heldOut->getDigest())
    return invalid("calibration and held-out specs must share one domain but "
                   "have disjoint seeds/digests");
  llvm::Expected<OneDNNQualificationCase> heldOutCase =
      materializeOneDNNQualificationCase(
          *heldOut, OneDNNNumericWorkBudget::create(
                        std::numeric_limits<uint64_t>::max(),
                        std::numeric_limits<uint64_t>::max(),
                        std::numeric_limits<uint64_t>::max()));
  if (!heldOutCase)
    return heldOutCase.takeError();
  const std::string heldOutInputPayloadDigest =
      computeOneDNNTensorPayloadDigest(heldOutCase->getInputs());
  if (*inputPayloadDigest == heldOutInputPayloadDigest)
    return invalid("calibration and held-out input payloads must be disjoint");
  llvm::Expected<std::string> backendDigest =
      requireDigest(*object, "backend_digest", "onednn calibration");
  llvm::Expected<std::string> environmentDigest =
      requireDigest(*object, "environment_digest", "onednn calibration");
  llvm::Expected<std::string> environmentRecordJSONDigest =
      requireDigest(*object, "environment_record_digest", "onednn calibration");
  llvm::Expected<std::string> problemDigest =
      requireDigest(*object, "problem_digest", "onednn calibration");
  if (llvm::Error error =
          takeExpectedErrors(backendDigest, environmentDigest,
                             environmentRecordJSONDigest, problemDigest))
    return error;
  const llvm::json::Value *environmentValue = object->get("environment");
  const llvm::json::Object *environmentObject =
      environmentValue ? environmentValue->getAsObject() : nullptr;
  if (!environmentObject)
    return invalid("onednn calibration environment must be an object");
  if (llvm::Error error = validateEnvironmentJSON(
          *environmentObject, *environmentDigest, *backendDigest))
    return error;
  if (sha256(canonicalJSON(*environmentValue)) != *environmentRecordJSONDigest)
    return invalid("onednn calibration environment record digest mismatch");
  llvm::Expected<std::string> heldOutProblemDigest =
      computeOneDNNGemmProblemDigest(heldOutCase->getOperation());
  if (!heldOutProblemDigest)
    return heldOutProblemDigest.takeError();
  if (*heldOutProblemDigest != *problemDigest)
    return invalid("held-out GEMM problem changed from calibration");
  llvm::json::Object policy{
      {"schema", kPolicySchema},
      {"adapter_digest", *adapterDigest},
      {"value_domain", *valueDomain},
      {"target_comparator", *targetComparator},
      {"backend_comparator", *backendComparator},
      {"proof_basis", kProofBasis},
      {"calibration_digest", calibration->digest},
      {"calibration_spec_digest", calibrationSpec->getDigest()},
      {"held_out_spec", specJSON(*heldOut)},
      {"held_out_spec_digest", heldOut->getDigest()},
      {"held_out_input_payload_digest", heldOutInputPayloadDigest},
      {"held_out_destination_template_digest",
       computeOneDNNTensorStorageDigest(heldOutCase->getDestinationTemplate())},
      {"backend_digest", *backendDigest},
      {"environment_digest", *environmentDigest},
      {"environment_record_digest", *environmentRecordJSONDigest},
      {"problem_digest", *problemDigest},
      {"qualification_kind", stringifyOneDNNQualificationKind(
                                 OneDNNQualificationKind::ProfileBounded)},
      {"maximum_absolute_error", tolerance.maximumAbsoluteError},
      {"maximum_relative_error", tolerance.maximumRelativeError},
      {"disjoint_proof", "same-domain-distinct-seed-spec-and-payload"},
  };
  return writeFileNoReplace(outputPath, canonicalJSON(std::move(policy)));
}

} // namespace wafer
