//===- OptimizationQualificationArchive.cpp - Run/publication records ---===//

#include "Wafer/Support/OptimizationQualificationArchive.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

namespace wafer {
namespace {

enum Tag : uint8_t {
  U16 = 0x02,
  U32 = 0x03,
  U64 = 0x04,
  ClosedEnum = 0x05,
  Digest32 = 0x07,
  Record = 0x09,
  SortedSet = 0x0b,
  Optional = 0x0c,
};

static bool fail(std::string *diagnostic, llvm::StringRef message) {
  if (diagnostic)
    *diagnostic = message.str();
  return false;
}

static bool isZero(const AdoptionDigest &digest) {
  return llvm::all_of(digest, [](uint8_t byte) { return byte == 0; });
}

static void put16(std::vector<uint8_t> &bytes, uint16_t value) {
  bytes.push_back(static_cast<uint8_t>(value >> 8));
  bytes.push_back(static_cast<uint8_t>(value));
}

static void put32(std::vector<uint8_t> &bytes, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static void put64(std::vector<uint8_t> &bytes, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static std::vector<uint8_t> u16(uint16_t value) {
  std::vector<uint8_t> bytes;
  put16(bytes, value);
  return bytes;
}

static std::vector<uint8_t> u32(uint32_t value) {
  std::vector<uint8_t> bytes;
  put32(bytes, value);
  return bytes;
}

static std::vector<uint8_t> u64(uint64_t value) {
  std::vector<uint8_t> bytes;
  put64(bytes, value);
  return bytes;
}

static void field(std::vector<uint8_t> &body, uint16_t number, Tag tag,
                  llvm::ArrayRef<uint8_t> payload) {
  put16(body, number);
  body.push_back(tag);
  put32(body, payload.size());
  body.insert(body.end(), payload.begin(), payload.end());
}

static std::vector<uint8_t> finish(llvm::StringRef domainIncludingNul,
                                   llvm::ArrayRef<uint8_t> body) {
  std::vector<uint8_t> bytes(domainIncludingNul.bytes_begin(),
                             domainIncludingNul.bytes_end());
  put16(bytes, 1);
  put32(bytes, body.size());
  bytes.insert(bytes.end(), body.begin(), body.end());
  return bytes;
}

static AdoptionDigest digestBytes(const std::vector<uint8_t> &bytes) {
  if (bytes.empty())
    return {};
  llvm::SHA256 hash;
  hash.update(bytes);
  return hash.final();
}

static std::vector<uint8_t>
optDigest(const std::optional<AdoptionDigest> &value) {
  if (!value)
    return {0};
  std::vector<uint8_t> bytes = {1, Digest32};
  put32(bytes, value->size());
  bytes.insert(bytes.end(), value->begin(), value->end());
  return bytes;
}

static std::vector<uint8_t> registry(const RegistryRefV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, U32, u32(value.id));
  field(body, 2, U16, u16(value.registrySchema));
  field(body, 3, Digest32, value.registryDigest);
  return body;
}

static std::vector<uint8_t> caseKey(const QualificationCaseKeyV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, registry(value.corpus));
  field(body, 2, U32, u32(value.rankCount));
  field(body, 3, ClosedEnum, u32(static_cast<uint32_t>(value.inputVariant)));
  return body;
}

static std::vector<uint8_t> specBinding(const MechanismSpecBinding &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, encodeMechanismKeyV1(value.mechanismKey));
  field(body, 2, Digest32, value.specDigest);
  return body;
}

static std::vector<uint8_t> inputCase(const QualificationInputCaseV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, caseKey(value.caseKey));
  field(body, 2, Digest32, value.inputSnapshotDigest);
  return body;
}

static std::vector<uint8_t>
invocationBinding(const InvocationTerminalBindingV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Digest32, value.invocationId);
  field(body, 2, Digest32, value.terminalDigest);
  return body;
}

static std::vector<uint8_t>
observationBinding(const MechanismObservationBindingV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, encodeMechanismKeyV1(value.mechanismKey));
  field(body, 2, Digest32, value.observationDigest);
  return body;
}

static std::vector<uint8_t> identity(const QualificationIdentityV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Digest32, value.build);
  field(body, 2, Digest32, value.toolchain);
  field(body, 3, Digest32, value.hostKernelAffinityGovernor);
  field(body, 4, Digest32, value.corpus);
  field(body, 5, Digest32, value.featureConfig);
  return body;
}

static std::vector<uint8_t> policy(const QualificationPolicyRefV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, U32, u32(value.policyId));
  field(body, 2, U16, u16(value.policySchema));
  field(body, 3, Digest32, value.canonicalPolicyDigest);
  return body;
}

static std::vector<uint8_t> reason(const ClosedReasonV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, registry(value.reason));
  field(body, 2, Optional, optDigest(value.detailDigest));
  return body;
}

static std::vector<uint8_t>
optReason(const std::optional<ClosedReasonV1> &value) {
  if (!value)
    return {0};
  std::vector<uint8_t> inner = reason(*value);
  std::vector<uint8_t> bytes = {1, Record};
  put32(bytes, inner.size());
  bytes.insert(bytes.end(), inner.begin(), inner.end());
  return bytes;
}

template <typename Value, typename Encoder>
static std::vector<uint8_t> set(const std::vector<Value> &values,
                                Encoder encode) {
  std::vector<uint8_t> bytes;
  put32(bytes, values.size());
  for (const Value &value : values) {
    std::vector<uint8_t> element = encode(value);
    put32(bytes, element.size());
    bytes.insert(bytes.end(), element.begin(), element.end());
  }
  return bytes;
}

template <typename Value, typename Encoder>
static bool ordered(const std::vector<Value> &values, Encoder encode) {
  std::vector<uint8_t> previous;
  for (const Value &value : values) {
    std::vector<uint8_t> current = encode(value);
    if (!previous.empty() && !(previous < current))
      return false;
    previous = std::move(current);
  }
  return true;
}

static bool sameRegistry(const RegistryRefV1 &lhs, const RegistryRefV1 &rhs) {
  return lhs.id == rhs.id && lhs.registrySchema == rhs.registrySchema &&
         lhs.registryDigest == rhs.registryDigest;
}

static bool sameCase(const QualificationCaseKeyV1 &lhs,
                     const QualificationCaseKeyV1 &rhs) {
  return sameRegistry(lhs.corpus, rhs.corpus) &&
         lhs.rankCount == rhs.rankCount && lhs.inputVariant == rhs.inputVariant;
}

static bool samePolicy(const QualificationPolicyRefV1 &lhs,
                       const QualificationPolicyRefV1 &rhs) {
  return lhs.policyId == rhs.policyId && lhs.policySchema == rhs.policySchema &&
         lhs.canonicalPolicyDigest == rhs.canonicalPolicyDigest;
}

static bool validateInput(const AdoptionQualificationInputV1 &value,
                          std::string *diagnostic) {
  if (value.schemaVersion != 1 || value.specBindings.empty() ||
      value.qualificationCases.empty() ||
      !ordered(value.specBindings, specBinding) ||
      !ordered(value.qualificationCases, inputCase))
    return fail(diagnostic, "qualification input sets are not canonical");
  for (const MechanismSpecBinding &binding : value.specBindings) {
    std::optional<AdoptionSpec> spec = lookupAdoptionSpec(binding.mechanismKey);
    if (!spec || digestAdoptionSpecV1(*spec) != binding.specDigest)
      return fail(diagnostic, "qualification input has a stale spec binding");
  }
  std::vector<AdoptionSpec> allSpecs = getAllAdoptionSpecs();
  if (value.specBindings.size() != allSpecs.size())
    return fail(diagnostic,
                "qualification input does not cover the named pipeline");
  for (size_t index = 0; index < allSpecs.size(); ++index)
    if (value.specBindings[index].mechanismKey !=
            allSpecs[index].mechanismKey ||
        value.specBindings[index].specDigest !=
            digestAdoptionSpecV1(allSpecs[index]))
      return fail(diagnostic,
                  "qualification input spec bindings are not all-and-only");

  std::vector<QualificationCaseKeyV1> expected;
  for (const QualificationCaseKeyV1 &mandatoryCase :
       getCurrentMandatoryQualificationCasesV1())
    for (EquivalentInputVariantV1 variant :
         {EquivalentInputVariantV1::Original,
          EquivalentInputVariantV1::Metamorphic})
      expected.push_back(
          {mandatoryCase.corpus, mandatoryCase.rankCount, variant});
  std::sort(expected.begin(), expected.end(),
            [](const auto &lhs, const auto &rhs) {
              return caseKey(lhs) < caseKey(rhs);
            });
  if (expected.size() != value.qualificationCases.size())
    return fail(diagnostic, "qualification input case domain is incomplete");
  for (size_t index = 0; index < expected.size(); ++index)
    if (!sameCase(expected[index], value.qualificationCases[index].caseKey) ||
        isZero(value.qualificationCases[index].inputSnapshotDigest))
      return fail(diagnostic, "qualification input case is invalid");

  if (value.optimizationProposalDigest) {
    OptimizationQualificationProposal proposal =
        getCurrentOptimizationQualificationProposal();
    if (isZero(*value.optimizationProposalDigest) ||
        *value.optimizationProposalDigest !=
            digestOptimizationQualificationProposalV1(proposal))
      return fail(diagnostic, "qualification input proposal is invalid");
    auto contains = [&](const MechanismSpecBinding &proposalBinding) {
      return llvm::any_of(value.specBindings, [&](const auto &binding) {
        return binding.mechanismKey == proposalBinding.mechanismKey &&
               binding.specDigest == proposalBinding.specDigest;
      });
    };
    if (!llvm::all_of(proposal.fixedBindings, contains) ||
        !llvm::all_of(proposal.cleanupBindings, contains))
      return fail(diagnostic, "qualification input omits a proposal binding");
    for (const MechanismSpecBinding &binding : value.specBindings) {
      std::optional<AdoptionSpec> spec =
          lookupAdoptionSpec(binding.mechanismKey);
      bool isOptionalOptimization =
          spec && (spec->adoptionMode == AdoptionMode::FixedOptimization ||
                   spec->adoptionMode == AdoptionMode::BestEffortCleanup);
      bool inProposal =
          contains(binding) &&
          (llvm::any_of(proposal.fixedBindings,
                        [&](const auto &candidate) {
                          return candidate.mechanismKey == binding.mechanismKey;
                        }) ||
           llvm::any_of(proposal.cleanupBindings, [&](const auto &candidate) {
             return candidate.mechanismKey == binding.mechanismKey;
           }));
      if (isOptionalOptimization != inProposal)
        return fail(diagnostic,
                    "qualification input proposal subset is not exact");
    }
  } else if (llvm::any_of(value.specBindings, [](const auto &binding) {
               std::optional<AdoptionSpec> spec =
                   lookupAdoptionSpec(binding.mechanismKey);
               return spec &&
                      (spec->adoptionMode == AdoptionMode::FixedOptimization ||
                       spec->adoptionMode == AdoptionMode::BestEffortCleanup);
             })) {
    // A general run may include no fixed/cleanup rows. If the named pipeline
    // includes them, a proposal is the only canonical way to freeze membership.
    return fail(diagnostic,
                "qualification input with optimization specs lacks a proposal");
  }
  return true;
}

static bool validateRun(const AdoptionQualificationRunV1 &value,
                        std::string *diagnostic) {
  const QualificationIdentityV1 &id = value.qualificationIdentity;
  if (value.schemaVersion != 1 || isZero(value.qualificationInputDigest) ||
      isZero(id.build) || isZero(id.toolchain) ||
      isZero(id.hostKernelAffinityGovernor) || isZero(id.corpus) ||
      isZero(id.featureConfig) ||
      !samePolicy(value.qualificationPolicy,
                  getCurrentOptimizationSetQualificationPolicyRefV1()))
    return fail(diagnostic, "qualification run identity or policy is invalid");
  return true;
}

static bool validReason(const std::optional<ClosedReasonV1> &value) {
  if (!value)
    return true;
  if (value->detailDigest && isZero(*value->detailDigest))
    return false;
  for (GlobalClosedReasonV1 candidate :
       {GlobalClosedReasonV1::UnsupportedSemantic,
        GlobalClosedReasonV1::ResourceExhausted,
        GlobalClosedReasonV1::InvalidOwnerTerminal,
        GlobalClosedReasonV1::Cancelled,
        GlobalClosedReasonV1::NoDeterministicBenefit,
        GlobalClosedReasonV1::DownstreamGateBlocked,
        GlobalClosedReasonV1::QualificationEvidenceRejected,
        GlobalClosedReasonV1::HostEnvironmentInvalidated,
        GlobalClosedReasonV1::PublicationConflict,
        GlobalClosedReasonV1::PublicationFailure})
    if (isGlobalClosedReasonRefV1(value->reason, candidate))
      return true;
  return false;
}

static bool validateRunTerminal(const AdoptionQualificationRunTerminalV1 &value,
                                std::string *diagnostic) {
  if (value.schemaVersion != 1 || isZero(value.qualificationRunDigest) ||
      !validReason(value.closedReason) ||
      static_cast<uint32_t>(value.outcome) >
          static_cast<uint32_t>(AdoptionQualificationRunOutcomeV1::Invalid))
    return fail(diagnostic, "qualification run terminal is invalid");
  bool completed =
      value.outcome == AdoptionQualificationRunOutcomeV1::CompletedEvidence;
  if (completed != value.resultManifestDigest.has_value() ||
      completed == value.closedReason.has_value() ||
      (value.resultManifestDigest && isZero(*value.resultManifestDigest)))
    return fail(diagnostic, "qualification terminal presence mismatch");
  if (!completed) {
    GlobalClosedReasonV1 expected = GlobalClosedReasonV1::InvalidOwnerTerminal;
    switch (value.outcome) {
    case AdoptionQualificationRunOutcomeV1::CompletedEvidence:
      break;
    case AdoptionQualificationRunOutcomeV1::Cancelled:
      expected = GlobalClosedReasonV1::Cancelled;
      break;
    case AdoptionQualificationRunOutcomeV1::HostEnvironmentInvalidated:
      expected = GlobalClosedReasonV1::HostEnvironmentInvalidated;
      break;
    case AdoptionQualificationRunOutcomeV1::ResourceExhausted:
      expected = GlobalClosedReasonV1::ResourceExhausted;
      break;
    case AdoptionQualificationRunOutcomeV1::Invalid:
      expected = GlobalClosedReasonV1::InvalidOwnerTerminal;
      break;
    }
    if (!isGlobalClosedReasonRefV1(value.closedReason->reason, expected))
      return fail(diagnostic,
                  "qualification run outcome uses the wrong closed reason");
  }
  return true;
}

static bool validateManifest(const AdoptionQualificationResultManifestV1 &value,
                             std::string *diagnostic) {
  if (value.schemaVersion != 1 || isZero(value.qualificationRunDigest) ||
      value.invocationTerminals.empty() || value.observationBindings.empty() ||
      !ordered(value.invocationTerminals, invocationBinding) ||
      !ordered(value.observationBindings, observationBinding))
    return fail(diagnostic, "qualification result manifest is invalid");
  for (const InvocationTerminalBindingV1 &binding : value.invocationTerminals)
    if (isZero(binding.invocationId) || isZero(binding.terminalDigest))
      return fail(diagnostic, "manifest has a zero invocation digest");
  for (const MechanismObservationBindingV1 &binding : value.observationBindings)
    if (!lookupAdoptionSpec(binding.mechanismKey) ||
        isZero(binding.observationDigest))
      return fail(diagnostic, "manifest observation binding is invalid");
  if (value.optimizationBatchObservationDigest &&
      isZero(*value.optimizationBatchObservationDigest))
    return fail(diagnostic, "manifest batch digest is zero");
  return true;
}

static bool validateAttempt(const OptimizationSetPublicationAttemptV1 &value,
                            std::string *diagnostic) {
  if (value.schemaVersion != 1 || isZero(value.qualificationRunDigest) ||
      isZero(value.proposalDigest) ||
      (value.expectedActiveRefDigest && isZero(*value.expectedActiveRefDigest)))
    return fail(diagnostic, "optimization publication attempt is invalid");
  return true;
}

static bool
validatePublicationTerminal(const OptimizationSetPublicationTerminalV1 &value,
                            std::string *diagnostic) {
  if (value.schemaVersion != 1 ||
      isZero(value.optimizationPublicationAttemptDigest) ||
      !validReason(value.closedReason) ||
      static_cast<uint32_t>(value.outcome) >
          static_cast<uint32_t>(
              OptimizationSetPublicationOutcomeV1::PublicationFailed) ||
      (value.batchObservationDigest && isZero(*value.batchObservationDigest)) ||
      (value.candidateSetDigest && isZero(*value.candidateSetDigest)))
    return fail(diagnostic, "optimization publication terminal is invalid");
  switch (value.outcome) {
  case OptimizationSetPublicationOutcomeV1::QualifiedPublished:
    return (value.batchObservationDigest && value.candidateSetDigest &&
            !value.closedReason) ||
           fail(diagnostic, "QualifiedPublished presence mismatch");
  case OptimizationSetPublicationOutcomeV1::RejectedEvidence:
    return (value.batchObservationDigest && !value.candidateSetDigest &&
            value.closedReason &&
            isGlobalClosedReasonRefV1(
                value.closedReason->reason,
                GlobalClosedReasonV1::QualificationEvidenceRejected)) ||
           fail(diagnostic, "RejectedEvidence presence mismatch");
  case OptimizationSetPublicationOutcomeV1::PublicationConflict:
    return (value.batchObservationDigest && value.candidateSetDigest &&
            value.closedReason &&
            isGlobalClosedReasonRefV1(
                value.closedReason->reason,
                GlobalClosedReasonV1::PublicationConflict)) ||
           fail(diagnostic, "PublicationConflict presence mismatch");
  case OptimizationSetPublicationOutcomeV1::PublicationFailed:
    return (value.batchObservationDigest && value.closedReason &&
            isGlobalClosedReasonRefV1(
                value.closedReason->reason,
                GlobalClosedReasonV1::PublicationFailure)) ||
           fail(diagnostic, "PublicationFailed presence mismatch");
  }
  return false;
}

static bool validateActive(const ActiveQualifiedOptimizationSetRefV1 &value,
                           std::string *diagnostic) {
  if (value.schemaVersion != 1 || value.generation == 0 ||
      (value.generation == 1) != !value.parentSetDigest.has_value() ||
      (value.parentSetDigest && isZero(*value.parentSetDigest)) ||
      isZero(value.setDigest) || isZero(value.qualificationRunDigest) ||
      isZero(value.adoptionQualificationRunTerminalDigest) ||
      isZero(value.optimizationPublicationAttemptDigest) ||
      isZero(value.optimizationPublicationTerminalDigest))
    return fail(diagnostic, "active qualified optimization ref is invalid");
  return true;
}

class Reader {
public:
  Reader(llvm::ArrayRef<uint8_t> bytes, std::string *diagnostic)
      : bytes(bytes), diagnostic(diagnostic) {}

  bool byte(uint8_t &value) {
    if (position >= bytes.size())
      return fail(diagnostic, "truncated u8");
    value = bytes[position++];
    return true;
  }
  bool number16(uint16_t &value) {
    if (bytes.size() - position < 2)
      return fail(diagnostic, "truncated u16");
    value = static_cast<uint16_t>((bytes[position] << 8) | bytes[position + 1]);
    position += 2;
    return true;
  }
  bool number32(uint32_t &value) {
    if (bytes.size() - position < 4)
      return fail(diagnostic, "truncated u32");
    value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes[position++];
    return true;
  }
  bool number64(uint64_t &value) {
    if (bytes.size() - position < 8)
      return fail(diagnostic, "truncated u64");
    value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes[position++];
    return true;
  }
  bool take(size_t size, llvm::ArrayRef<uint8_t> &value) {
    if (size > bytes.size() - position)
      return fail(diagnostic, "truncated payload");
    value = bytes.slice(position, size);
    position += size;
    return true;
  }
  bool next(uint16_t expectedNumber, Tag expectedTag,
            llvm::ArrayRef<uint8_t> &payload) {
    uint16_t number;
    uint8_t tag;
    uint32_t size;
    if (!number16(number) || !byte(tag) || !number32(size))
      return false;
    if (number != expectedNumber || tag != expectedTag)
      return fail(diagnostic, "unknown, missing, reordered or mistyped field");
    return take(size, payload);
  }
  bool done() const { return position == bytes.size(); }

private:
  llvm::ArrayRef<uint8_t> bytes;
  size_t position = 0;
  std::string *diagnostic;
};

static bool readU16(llvm::ArrayRef<uint8_t> bytes, uint16_t &value,
                    std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  return reader.number16(value) && reader.done();
}

static bool readU32(llvm::ArrayRef<uint8_t> bytes, uint32_t &value,
                    std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  return reader.number32(value) && reader.done();
}

static bool readU64(llvm::ArrayRef<uint8_t> bytes, uint64_t &value,
                    std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  return reader.number64(value) && reader.done();
}

static bool readDigest(llvm::ArrayRef<uint8_t> bytes, AdoptionDigest &value,
                       std::string *diagnostic) {
  if (bytes.size() != value.size())
    return fail(diagnostic, "digest is not exactly 32 bytes");
  std::copy(bytes.begin(), bytes.end(), value.begin());
  return true;
}

static bool readRegistry(llvm::ArrayRef<uint8_t> bytes, RegistryRefV1 &value,
                         std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  return reader.next(1, U32, payload) &&
         readU32(payload, value.id, diagnostic) &&
         reader.next(2, U16, payload) &&
         readU16(payload, value.registrySchema, diagnostic) &&
         reader.next(3, Digest32, payload) &&
         readDigest(payload, value.registryDigest, diagnostic) && reader.done();
}

static bool readMechanismKey(llvm::ArrayRef<uint8_t> bytes, MechanismKey &value,
                             std::string *diagnostic) {
  constexpr char domain[] = "wafer.mechanism-key";
  if (bytes.size() != sizeof(domain) + 6 ||
      !std::equal(bytes.begin(), bytes.begin() + sizeof(domain), domain))
    return fail(diagnostic, "invalid MechanismKeyV1 bytes");
  Reader reader(bytes.drop_front(sizeof(domain)), diagnostic);
  uint16_t schema;
  return reader.number16(schema) && schema == 1 &&
         reader.number32(value.semanticId) && value.semanticId != 0 &&
         reader.done();
}

static bool readCaseKey(llvm::ArrayRef<uint8_t> bytes,
                        QualificationCaseKeyV1 &value,
                        std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  uint32_t variant;
  if (!reader.next(1, Record, payload) ||
      !readRegistry(payload, value.corpus, diagnostic) ||
      !reader.next(2, U32, payload) ||
      !readU32(payload, value.rankCount, diagnostic) ||
      !reader.next(3, ClosedEnum, payload) ||
      !readU32(payload, variant, diagnostic) || !reader.done() ||
      variant > static_cast<uint32_t>(EquivalentInputVariantV1::Metamorphic))
    return fail(diagnostic, "invalid QualificationCaseKeyV1 bytes");
  value.inputVariant = static_cast<EquivalentInputVariantV1>(variant);
  return true;
}

static bool readOptionalDigest(llvm::ArrayRef<uint8_t> bytes,
                               std::optional<AdoptionDigest> &value,
                               std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  uint8_t present;
  if (!reader.byte(present))
    return false;
  if (present == 0) {
    value.reset();
    return reader.done() || fail(diagnostic, "absent optional has payload");
  }
  uint8_t tag;
  uint32_t size;
  llvm::ArrayRef<uint8_t> payload;
  AdoptionDigest digest;
  if (present != 1 || !reader.byte(tag) || tag != Digest32 ||
      !reader.number32(size) || !reader.take(size, payload) || !reader.done() ||
      !readDigest(payload, digest, diagnostic))
    return fail(diagnostic, "invalid optional digest");
  value = digest;
  return true;
}

static bool readReason(llvm::ArrayRef<uint8_t> bytes, ClosedReasonV1 &value,
                       std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  return reader.next(1, Record, payload) &&
         readRegistry(payload, value.reason, diagnostic) &&
         reader.next(2, Optional, payload) &&
         readOptionalDigest(payload, value.detailDigest, diagnostic) &&
         reader.done();
}

static bool readOptionalReason(llvm::ArrayRef<uint8_t> bytes,
                               std::optional<ClosedReasonV1> &value,
                               std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  uint8_t present;
  if (!reader.byte(present))
    return false;
  if (present == 0) {
    value.reset();
    return reader.done() || fail(diagnostic, "absent optional has payload");
  }
  uint8_t tag;
  uint32_t size;
  llvm::ArrayRef<uint8_t> payload;
  ClosedReasonV1 parsed;
  if (present != 1 || !reader.byte(tag) || tag != Record ||
      !reader.number32(size) || !reader.take(size, payload) || !reader.done() ||
      !readReason(payload, parsed, diagnostic))
    return fail(diagnostic, "invalid optional closed reason");
  value = parsed;
  return true;
}

template <typename Value, typename Parser>
static bool readSet(llvm::ArrayRef<uint8_t> bytes, std::vector<Value> &values,
                    Parser parser, std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  uint32_t count;
  if (!reader.number32(count))
    return false;
  values.clear();
  values.reserve(count);
  std::vector<uint8_t> previous;
  for (uint32_t index = 0; index < count; ++index) {
    uint32_t size;
    llvm::ArrayRef<uint8_t> payload;
    if (!reader.number32(size) || !reader.take(size, payload))
      return false;
    if (!previous.empty() &&
        !std::lexicographical_compare(previous.begin(), previous.end(),
                                      payload.begin(), payload.end()))
      return fail(diagnostic, "set bytes are not sorted unique");
    Value parsed;
    if (!parser(payload, parsed, diagnostic))
      return false;
    values.push_back(std::move(parsed));
    previous.assign(payload.begin(), payload.end());
  }
  return reader.done() || fail(diagnostic, "set has trailing bytes");
}

static bool readSpecBinding(llvm::ArrayRef<uint8_t> bytes,
                            MechanismSpecBinding &value,
                            std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  return reader.next(1, Record, payload) &&
         readMechanismKey(payload, value.mechanismKey, diagnostic) &&
         reader.next(2, Digest32, payload) &&
         readDigest(payload, value.specDigest, diagnostic) && reader.done();
}

static bool readInputCase(llvm::ArrayRef<uint8_t> bytes,
                          QualificationInputCaseV1 &value,
                          std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  return reader.next(1, Record, payload) &&
         readCaseKey(payload, value.caseKey, diagnostic) &&
         reader.next(2, Digest32, payload) &&
         readDigest(payload, value.inputSnapshotDigest, diagnostic) &&
         reader.done();
}

static bool readIdentity(llvm::ArrayRef<uint8_t> bytes,
                         QualificationIdentityV1 &value,
                         std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  return reader.next(1, Digest32, payload) &&
         readDigest(payload, value.build, diagnostic) &&
         reader.next(2, Digest32, payload) &&
         readDigest(payload, value.toolchain, diagnostic) &&
         reader.next(3, Digest32, payload) &&
         readDigest(payload, value.hostKernelAffinityGovernor, diagnostic) &&
         reader.next(4, Digest32, payload) &&
         readDigest(payload, value.corpus, diagnostic) &&
         reader.next(5, Digest32, payload) &&
         readDigest(payload, value.featureConfig, diagnostic) && reader.done();
}

static bool readPolicy(llvm::ArrayRef<uint8_t> bytes,
                       QualificationPolicyRefV1 &value,
                       std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  return reader.next(1, U32, payload) &&
         readU32(payload, value.policyId, diagnostic) &&
         reader.next(2, U16, payload) &&
         readU16(payload, value.policySchema, diagnostic) &&
         reader.next(3, Digest32, payload) &&
         readDigest(payload, value.canonicalPolicyDigest, diagnostic) &&
         reader.done();
}

static bool readInvocationBinding(llvm::ArrayRef<uint8_t> bytes,
                                  InvocationTerminalBindingV1 &value,
                                  std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  return reader.next(1, Digest32, payload) &&
         readDigest(payload, value.invocationId, diagnostic) &&
         reader.next(2, Digest32, payload) &&
         readDigest(payload, value.terminalDigest, diagnostic) && reader.done();
}

static bool readObservationBinding(llvm::ArrayRef<uint8_t> bytes,
                                   MechanismObservationBindingV1 &value,
                                   std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  return reader.next(1, Record, payload) &&
         readMechanismKey(payload, value.mechanismKey, diagnostic) &&
         reader.next(2, Digest32, payload) &&
         readDigest(payload, value.observationDigest, diagnostic) &&
         reader.done();
}

static bool open(const std::vector<uint8_t> &bytes,
                 llvm::StringRef domainIncludingNul,
                 llvm::ArrayRef<uint8_t> &body, std::string *diagnostic) {
  if (bytes.size() < domainIncludingNul.size() + 6 ||
      !std::equal(bytes.begin(), bytes.begin() + domainIncludingNul.size(),
                  domainIncludingNul.begin()))
    return fail(diagnostic, "qualification archive domain mismatch");
  Reader reader(
      llvm::ArrayRef<uint8_t>(bytes).drop_front(domainIncludingNul.size()),
      diagnostic);
  uint16_t schema;
  uint32_t size;
  return reader.number16(schema) && schema == 1 && reader.number32(size) &&
         reader.take(size, body) && reader.done();
}

} // namespace

namespace {
constexpr char inputDomain[] = "wafer.adoption-qualification-input";
constexpr char runDomain[] = "wafer.adoption-qualification-run";
constexpr char runTerminalDomain[] =
    "wafer.adoption-qualification-run-terminal";
constexpr char manifestDomain[] =
    "wafer.adoption-qualification-result-manifest";
constexpr char publicationAttemptDomain[] =
    "wafer.optimization-publication-attempt";
constexpr char publicationTerminalDomain[] =
    "wafer.optimization-publication-terminal";
constexpr char activeDomain[] = "wafer.active-qualified-optimization-set-ref";
} // namespace

std::vector<uint8_t>
encodeAdoptionQualificationInputV1(const AdoptionQualificationInputV1 &value) {
  if (!validateInput(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.schemaVersion));
  field(body, 2, SortedSet, set(value.specBindings, specBinding));
  field(body, 3, SortedSet, set(value.qualificationCases, inputCase));
  field(body, 4, Optional, optDigest(value.optimizationProposalDigest));
  return finish(llvm::StringRef(inputDomain, sizeof(inputDomain)), body);
}

AdoptionDigest
digestAdoptionQualificationInputV1(const AdoptionQualificationInputV1 &value) {
  return digestBytes(encodeAdoptionQualificationInputV1(value));
}

bool decodeCanonicalAdoptionQualificationInputV1(
    const std::vector<uint8_t> &bytes, AdoptionQualificationInputV1 &value,
    std::string *diagnostic) {
  llvm::ArrayRef<uint8_t> body, payload;
  AdoptionQualificationInputV1 parsed;
  Reader reader({}, diagnostic);
  if (!open(bytes, llvm::StringRef(inputDomain, sizeof(inputDomain)), body,
            diagnostic))
    return false;
  reader = Reader(body, diagnostic);
  if (!reader.next(1, U16, payload) ||
      !readU16(payload, parsed.schemaVersion, diagnostic) ||
      !reader.next(2, SortedSet, payload) ||
      !readSet(payload, parsed.specBindings, readSpecBinding, diagnostic) ||
      !reader.next(3, SortedSet, payload) ||
      !readSet(payload, parsed.qualificationCases, readInputCase, diagnostic) ||
      !reader.next(4, Optional, payload) ||
      !readOptionalDigest(payload, parsed.optimizationProposalDigest,
                          diagnostic) ||
      !reader.done() || !validateInput(parsed, diagnostic) ||
      encodeAdoptionQualificationInputV1(parsed) != bytes)
    return fail(diagnostic, "invalid canonical qualification input");
  value = std::move(parsed);
  return true;
}

std::vector<uint8_t>
encodeAdoptionQualificationRunV1(const AdoptionQualificationRunV1 &value) {
  if (!validateRun(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.schemaVersion));
  field(body, 2, Digest32, value.qualificationInputDigest);
  field(body, 3, Record, identity(value.qualificationIdentity));
  field(body, 4, Record, policy(value.qualificationPolicy));
  field(body, 5, U64, u64(value.runSeriesOrdinal));
  field(body, 6, U32, u32(value.attemptOrdinal));
  return finish(llvm::StringRef(runDomain, sizeof(runDomain)), body);
}

AdoptionDigest
digestAdoptionQualificationRunV1(const AdoptionQualificationRunV1 &value) {
  return digestBytes(encodeAdoptionQualificationRunV1(value));
}

bool decodeCanonicalAdoptionQualificationRunV1(
    const std::vector<uint8_t> &bytes, AdoptionQualificationRunV1 &value,
    std::string *diagnostic) {
  llvm::ArrayRef<uint8_t> body, payload;
  if (!open(bytes, llvm::StringRef(runDomain, sizeof(runDomain)), body,
            diagnostic))
    return false;
  AdoptionQualificationRunV1 parsed;
  Reader reader(body, diagnostic);
  if (!reader.next(1, U16, payload) ||
      !readU16(payload, parsed.schemaVersion, diagnostic) ||
      !reader.next(2, Digest32, payload) ||
      !readDigest(payload, parsed.qualificationInputDigest, diagnostic) ||
      !reader.next(3, Record, payload) ||
      !readIdentity(payload, parsed.qualificationIdentity, diagnostic) ||
      !reader.next(4, Record, payload) ||
      !readPolicy(payload, parsed.qualificationPolicy, diagnostic) ||
      !reader.next(5, U64, payload) ||
      !readU64(payload, parsed.runSeriesOrdinal, diagnostic) ||
      !reader.next(6, U32, payload) ||
      !readU32(payload, parsed.attemptOrdinal, diagnostic) || !reader.done() ||
      !validateRun(parsed, diagnostic) ||
      encodeAdoptionQualificationRunV1(parsed) != bytes)
    return fail(diagnostic, "invalid canonical qualification run");
  value = std::move(parsed);
  return true;
}

std::vector<uint8_t> encodeAdoptionQualificationRunTerminalV1(
    const AdoptionQualificationRunTerminalV1 &value) {
  if (!validateRunTerminal(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.schemaVersion));
  field(body, 2, Digest32, value.qualificationRunDigest);
  field(body, 3, ClosedEnum, u32(static_cast<uint32_t>(value.outcome)));
  field(body, 4, Optional, optDigest(value.resultManifestDigest));
  field(body, 5, Optional, optReason(value.closedReason));
  return finish(llvm::StringRef(runTerminalDomain, sizeof(runTerminalDomain)),
                body);
}

AdoptionDigest digestAdoptionQualificationRunTerminalV1(
    const AdoptionQualificationRunTerminalV1 &value) {
  return digestBytes(encodeAdoptionQualificationRunTerminalV1(value));
}

bool decodeCanonicalAdoptionQualificationRunTerminalV1(
    const std::vector<uint8_t> &bytes,
    AdoptionQualificationRunTerminalV1 &value, std::string *diagnostic) {
  llvm::ArrayRef<uint8_t> body, payload;
  if (!open(bytes,
            llvm::StringRef(runTerminalDomain, sizeof(runTerminalDomain)), body,
            diagnostic))
    return false;
  AdoptionQualificationRunTerminalV1 parsed;
  uint32_t outcome;
  Reader reader(body, diagnostic);
  if (!reader.next(1, U16, payload) ||
      !readU16(payload, parsed.schemaVersion, diagnostic) ||
      !reader.next(2, Digest32, payload) ||
      !readDigest(payload, parsed.qualificationRunDigest, diagnostic) ||
      !reader.next(3, ClosedEnum, payload) ||
      !readU32(payload, outcome, diagnostic) ||
      outcome >
          static_cast<uint32_t>(AdoptionQualificationRunOutcomeV1::Invalid) ||
      !reader.next(4, Optional, payload) ||
      !readOptionalDigest(payload, parsed.resultManifestDigest, diagnostic) ||
      !reader.next(5, Optional, payload) ||
      !readOptionalReason(payload, parsed.closedReason, diagnostic) ||
      !reader.done())
    return fail(diagnostic, "invalid qualification run terminal fields");
  parsed.outcome = static_cast<AdoptionQualificationRunOutcomeV1>(outcome);
  if (!validateRunTerminal(parsed, diagnostic) ||
      encodeAdoptionQualificationRunTerminalV1(parsed) != bytes)
    return fail(diagnostic, "invalid canonical qualification run terminal");
  value = std::move(parsed);
  return true;
}

std::vector<uint8_t> encodeAdoptionQualificationResultManifestV1(
    const AdoptionQualificationResultManifestV1 &value) {
  if (!validateManifest(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.schemaVersion));
  field(body, 2, Digest32, value.qualificationRunDigest);
  field(body, 3, SortedSet, set(value.invocationTerminals, invocationBinding));
  field(body, 4, SortedSet, set(value.observationBindings, observationBinding));
  field(body, 5, Optional, optDigest(value.optimizationBatchObservationDigest));
  return finish(llvm::StringRef(manifestDomain, sizeof(manifestDomain)), body);
}

AdoptionDigest digestAdoptionQualificationResultManifestV1(
    const AdoptionQualificationResultManifestV1 &value) {
  return digestBytes(encodeAdoptionQualificationResultManifestV1(value));
}

bool decodeCanonicalAdoptionQualificationResultManifestV1(
    const std::vector<uint8_t> &bytes,
    AdoptionQualificationResultManifestV1 &value, std::string *diagnostic) {
  llvm::ArrayRef<uint8_t> body, payload;
  if (!open(bytes, llvm::StringRef(manifestDomain, sizeof(manifestDomain)),
            body, diagnostic))
    return false;
  AdoptionQualificationResultManifestV1 parsed;
  Reader reader(body, diagnostic);
  if (!reader.next(1, U16, payload) ||
      !readU16(payload, parsed.schemaVersion, diagnostic) ||
      !reader.next(2, Digest32, payload) ||
      !readDigest(payload, parsed.qualificationRunDigest, diagnostic) ||
      !reader.next(3, SortedSet, payload) ||
      !readSet(payload, parsed.invocationTerminals, readInvocationBinding,
               diagnostic) ||
      !reader.next(4, SortedSet, payload) ||
      !readSet(payload, parsed.observationBindings, readObservationBinding,
               diagnostic) ||
      !reader.next(5, Optional, payload) ||
      !readOptionalDigest(payload, parsed.optimizationBatchObservationDigest,
                          diagnostic) ||
      !reader.done() || !validateManifest(parsed, diagnostic) ||
      encodeAdoptionQualificationResultManifestV1(parsed) != bytes)
    return fail(diagnostic, "invalid canonical qualification manifest");
  value = std::move(parsed);
  return true;
}

std::vector<uint8_t> encodeOptimizationSetPublicationAttemptV1(
    const OptimizationSetPublicationAttemptV1 &value) {
  if (!validateAttempt(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.schemaVersion));
  field(body, 2, Digest32, value.qualificationRunDigest);
  field(body, 3, Digest32, value.proposalDigest);
  field(body, 4, Optional, optDigest(value.expectedActiveRefDigest));
  return finish(llvm::StringRef(publicationAttemptDomain,
                                sizeof(publicationAttemptDomain)),
                body);
}

AdoptionDigest digestOptimizationSetPublicationAttemptV1(
    const OptimizationSetPublicationAttemptV1 &value) {
  return digestBytes(encodeOptimizationSetPublicationAttemptV1(value));
}

bool decodeCanonicalOptimizationSetPublicationAttemptV1(
    const std::vector<uint8_t> &bytes,
    OptimizationSetPublicationAttemptV1 &value, std::string *diagnostic) {
  llvm::ArrayRef<uint8_t> body, payload;
  if (!open(bytes,
            llvm::StringRef(publicationAttemptDomain,
                            sizeof(publicationAttemptDomain)),
            body, diagnostic))
    return false;
  OptimizationSetPublicationAttemptV1 parsed;
  Reader reader(body, diagnostic);
  if (!reader.next(1, U16, payload) ||
      !readU16(payload, parsed.schemaVersion, diagnostic) ||
      !reader.next(2, Digest32, payload) ||
      !readDigest(payload, parsed.qualificationRunDigest, diagnostic) ||
      !reader.next(3, Digest32, payload) ||
      !readDigest(payload, parsed.proposalDigest, diagnostic) ||
      !reader.next(4, Optional, payload) ||
      !readOptionalDigest(payload, parsed.expectedActiveRefDigest,
                          diagnostic) ||
      !reader.done() || !validateAttempt(parsed, diagnostic) ||
      encodeOptimizationSetPublicationAttemptV1(parsed) != bytes)
    return fail(diagnostic, "invalid canonical publication attempt");
  value = std::move(parsed);
  return true;
}

std::vector<uint8_t> encodeOptimizationSetPublicationTerminalV1(
    const OptimizationSetPublicationTerminalV1 &value) {
  if (!validatePublicationTerminal(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.schemaVersion));
  field(body, 2, Digest32, value.optimizationPublicationAttemptDigest);
  field(body, 3, ClosedEnum, u32(static_cast<uint32_t>(value.outcome)));
  field(body, 4, Optional, optDigest(value.batchObservationDigest));
  field(body, 5, Optional, optDigest(value.candidateSetDigest));
  field(body, 6, Optional, optReason(value.closedReason));
  return finish(llvm::StringRef(publicationTerminalDomain,
                                sizeof(publicationTerminalDomain)),
                body);
}

AdoptionDigest digestOptimizationSetPublicationTerminalV1(
    const OptimizationSetPublicationTerminalV1 &value) {
  return digestBytes(encodeOptimizationSetPublicationTerminalV1(value));
}

bool decodeCanonicalOptimizationSetPublicationTerminalV1(
    const std::vector<uint8_t> &bytes,
    OptimizationSetPublicationTerminalV1 &value, std::string *diagnostic) {
  llvm::ArrayRef<uint8_t> body, payload;
  if (!open(bytes,
            llvm::StringRef(publicationTerminalDomain,
                            sizeof(publicationTerminalDomain)),
            body, diagnostic))
    return false;
  OptimizationSetPublicationTerminalV1 parsed;
  uint32_t outcome;
  Reader reader(body, diagnostic);
  if (!reader.next(1, U16, payload) ||
      !readU16(payload, parsed.schemaVersion, diagnostic) ||
      !reader.next(2, Digest32, payload) ||
      !readDigest(payload, parsed.optimizationPublicationAttemptDigest,
                  diagnostic) ||
      !reader.next(3, ClosedEnum, payload) ||
      !readU32(payload, outcome, diagnostic) ||
      outcome > static_cast<uint32_t>(
                    OptimizationSetPublicationOutcomeV1::PublicationFailed) ||
      !reader.next(4, Optional, payload) ||
      !readOptionalDigest(payload, parsed.batchObservationDigest, diagnostic) ||
      !reader.next(5, Optional, payload) ||
      !readOptionalDigest(payload, parsed.candidateSetDigest, diagnostic) ||
      !reader.next(6, Optional, payload) ||
      !readOptionalReason(payload, parsed.closedReason, diagnostic) ||
      !reader.done())
    return fail(diagnostic, "invalid publication terminal fields");
  parsed.outcome = static_cast<OptimizationSetPublicationOutcomeV1>(outcome);
  if (!validatePublicationTerminal(parsed, diagnostic) ||
      encodeOptimizationSetPublicationTerminalV1(parsed) != bytes)
    return fail(diagnostic, "invalid canonical publication terminal");
  value = std::move(parsed);
  return true;
}

std::vector<uint8_t> encodeActiveQualifiedOptimizationSetRefV1(
    const ActiveQualifiedOptimizationSetRefV1 &value) {
  if (!validateActive(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.schemaVersion));
  field(body, 2, U64, u64(value.generation));
  field(body, 3, Optional, optDigest(value.parentSetDigest));
  field(body, 4, Digest32, value.setDigest);
  field(body, 5, Digest32, value.qualificationRunDigest);
  field(body, 6, Digest32, value.adoptionQualificationRunTerminalDigest);
  field(body, 7, Digest32, value.optimizationPublicationAttemptDigest);
  field(body, 8, Digest32, value.optimizationPublicationTerminalDigest);
  return finish(llvm::StringRef(activeDomain, sizeof(activeDomain)), body);
}

AdoptionDigest digestActiveQualifiedOptimizationSetRefV1(
    const ActiveQualifiedOptimizationSetRefV1 &value) {
  return digestBytes(encodeActiveQualifiedOptimizationSetRefV1(value));
}

bool decodeCanonicalActiveQualifiedOptimizationSetRefV1(
    const std::vector<uint8_t> &bytes,
    ActiveQualifiedOptimizationSetRefV1 &value, std::string *diagnostic) {
  llvm::ArrayRef<uint8_t> body, payload;
  if (!open(bytes, llvm::StringRef(activeDomain, sizeof(activeDomain)), body,
            diagnostic))
    return false;
  ActiveQualifiedOptimizationSetRefV1 parsed;
  Reader reader(body, diagnostic);
  if (!reader.next(1, U16, payload) ||
      !readU16(payload, parsed.schemaVersion, diagnostic) ||
      !reader.next(2, U64, payload) ||
      !readU64(payload, parsed.generation, diagnostic) ||
      !reader.next(3, Optional, payload) ||
      !readOptionalDigest(payload, parsed.parentSetDigest, diagnostic) ||
      !reader.next(4, Digest32, payload) ||
      !readDigest(payload, parsed.setDigest, diagnostic) ||
      !reader.next(5, Digest32, payload) ||
      !readDigest(payload, parsed.qualificationRunDigest, diagnostic) ||
      !reader.next(6, Digest32, payload) ||
      !readDigest(payload, parsed.adoptionQualificationRunTerminalDigest,
                  diagnostic) ||
      !reader.next(7, Digest32, payload) ||
      !readDigest(payload, parsed.optimizationPublicationAttemptDigest,
                  diagnostic) ||
      !reader.next(8, Digest32, payload) ||
      !readDigest(payload, parsed.optimizationPublicationTerminalDigest,
                  diagnostic) ||
      !reader.done() || !validateActive(parsed, diagnostic) ||
      encodeActiveQualifiedOptimizationSetRefV1(parsed) != bytes)
    return fail(diagnostic, "invalid canonical active optimization ref");
  value = std::move(parsed);
  return true;
}

bool validateCompletedQualificationEvidenceV1(
    const CompletedQualificationEvidenceV1 &value, std::string *diagnostic) {
  if (!validateInput(value.input, diagnostic) ||
      !validateRun(value.run, diagnostic) ||
      !validateManifest(value.resultManifest, diagnostic) ||
      !validateRunTerminal(value.runTerminal, diagnostic))
    return false;

  AdoptionDigest inputDigest = digestAdoptionQualificationInputV1(value.input);
  AdoptionDigest runDigest = digestAdoptionQualificationRunV1(value.run);
  if (value.run.qualificationInputDigest != inputDigest ||
      value.resultManifest.qualificationRunDigest != runDigest ||
      value.runTerminal.qualificationRunDigest != runDigest ||
      value.runTerminal.outcome !=
          AdoptionQualificationRunOutcomeV1::CompletedEvidence ||
      !value.runTerminal.resultManifestDigest ||
      *value.runTerminal.resultManifestDigest !=
          digestAdoptionQualificationResultManifestV1(value.resultManifest))
    return fail(diagnostic,
                "completed qualification run/input/manifest chain is broken");

  bool optimization = value.input.optimizationProposalDigest.has_value();
  std::optional<AdoptionDigest> attemptDigest;
  if (optimization) {
    if (!value.publicationAttempt ||
        !validateAttempt(*value.publicationAttempt, diagnostic) ||
        value.publicationAttempt->qualificationRunDigest != runDigest ||
        value.publicationAttempt->proposalDigest !=
            *value.input.optimizationProposalDigest)
      return fail(
          diagnostic,
          "completed optimization run lacks its exact publication attempt");
    attemptDigest =
        digestOptimizationSetPublicationAttemptV1(*value.publicationAttempt);
  } else if (value.publicationAttempt) {
    return fail(diagnostic,
                "general qualification run carries a publication attempt");
  }

  auto findCase = [&](const QualificationCaseKeyV1 &key)
      -> const QualificationInputCaseV1 * {
    auto it = llvm::find_if(value.input.qualificationCases,
                            [&](const QualificationInputCaseV1 &candidate) {
                              return sameCase(candidate.caseKey, key);
                            });
    return it == value.input.qualificationCases.end() ? nullptr : &*it;
  };
  auto findSpecBinding = [&](MechanismKey key) -> const MechanismSpecBinding * {
    auto it = llvm::find_if(value.input.specBindings,
                            [&](const MechanismSpecBinding &binding) {
                              return binding.mechanismKey == key;
                            });
    return it == value.input.specBindings.end() ? nullptr : &*it;
  };
  auto sameReason = [&](const std::optional<ClosedReasonV1> &lhs,
                        const std::optional<ClosedReasonV1> &rhs) {
    return lhs.has_value() == rhs.has_value() &&
           (!lhs || (sameRegistry(lhs->reason, rhs->reason) &&
                     lhs->detailDigest == rhs->detailDigest));
  };
  auto sameAction = [](const BackendActionEvidenceV1 &lhs,
                       const BackendActionEvidenceV1 &rhs) {
    return lhs.invocationId == rhs.invocationId &&
           lhs.actionOrdinal == rhs.actionOrdinal && lhs.argv == rhs.argv &&
           lhs.observedToolDigest == rhs.observedToolDigest &&
           lhs.observedOutputDigest == rhs.observedOutputDigest &&
           lhs.terminalStatus == rhs.terminalStatus;
  };

  std::vector<InvocationTerminalBindingV1> terminalBindings;
  terminalBindings.reserve(value.invocationTerminals.size());
  AdoptionDigest previousInvocationId{};
  bool havePreviousInvocation = false;
  for (const InvocationTelemetryV1 &terminal : value.invocationTerminals) {
    if (!validateInvocationTelemetryV1(terminal, diagnostic) ||
        terminal.identity.scopeKind !=
            InvocationScopeKindV1::QualificationRun ||
        terminal.identity.scopeDigest != runDigest ||
        !terminal.qualificationCase)
      return fail(diagnostic,
                  "qualification invocation terminal has wrong scope/case");
    const QualificationInputCaseV1 *inputCase =
        findCase(*terminal.qualificationCase);
    const MechanismSpecBinding *specBinding =
        findSpecBinding(terminal.identity.mechanismKey);
    if (!inputCase || !specBinding || isZero(terminal.inputSnapshotDigest) ||
        terminal.specDigest != specBinding->specDigest ||
        terminal.workSummary.workPolicyDigest !=
            digestAdoptionWorkPolicyV1(
                lookupAdoptionSpec(terminal.identity.mechanismKey)
                    ->workPolicyKind))
      return fail(diagnostic, "qualification terminal does not bind case/spec "
                              "or has a zero stage snapshot");
    AdoptionDigest invocationId = digestInvocationIdentityV1(terminal.identity);
    if (havePreviousInvocation && !(previousInvocationId < invocationId))
      return fail(diagnostic,
                  "qualification terminals are not sorted and unique");
    previousInvocationId = invocationId;
    havePreviousInvocation = true;
    terminalBindings.push_back(
        {invocationId, digestInvocationTelemetryV1(terminal)});
  }
  auto sameTerminalBinding = [](const InvocationTerminalBindingV1 &lhs,
                                const InvocationTerminalBindingV1 &rhs) {
    return lhs.invocationId == rhs.invocationId &&
           lhs.terminalDigest == rhs.terminalDigest;
  };
  if (terminalBindings.size() !=
          value.resultManifest.invocationTerminals.size() ||
      !std::equal(terminalBindings.begin(), terminalBindings.end(),
                  value.resultManifest.invocationTerminals.begin(),
                  sameTerminalBinding))
    return fail(diagnostic,
                "result manifest terminal bindings are not fresh all-and-only");

  if (value.observations.size() != value.input.specBindings.size())
    return fail(diagnostic,
                "qualification observations do not cover input specs");
  std::vector<MechanismObservationBindingV1> observationBindings;
  observationBindings.reserve(value.observations.size());
  for (size_t index = 0; index < value.observations.size(); ++index) {
    const QualificationObservationV1 &observation = value.observations[index];
    const MechanismSpecBinding &binding = value.input.specBindings[index];
    if (observation.mechanismKey != binding.mechanismKey ||
        observation.specDigest != binding.specDigest ||
        !validateQualificationObservationV1(observation, diagnostic) ||
        observation.qualificationRunDigest != runDigest ||
        observation.qualificationIdentity.build !=
            value.run.qualificationIdentity.build ||
        observation.qualificationIdentity.toolchain !=
            value.run.qualificationIdentity.toolchain ||
        observation.qualificationIdentity.hostKernelAffinityGovernor !=
            value.run.qualificationIdentity.hostKernelAffinityGovernor ||
        observation.qualificationIdentity.corpus !=
            value.run.qualificationIdentity.corpus ||
        observation.qualificationIdentity.featureConfig !=
            value.run.qualificationIdentity.featureConfig ||
        !samePolicy(observation.qualificationPolicy,
                    value.run.qualificationPolicy))
      return fail(diagnostic,
                  "qualification observation identity/spec/run mismatch");
    std::optional<AdoptionSpec> spec =
        lookupAdoptionSpec(observation.mechanismKey);
    bool rowIsOptionalOptimization =
        spec && (spec->adoptionMode == AdoptionMode::FixedOptimization ||
                 spec->adoptionMode == AdoptionMode::BestEffortCleanup);
    if (rowIsOptionalOptimization &&
        (!attemptDigest ||
         observation.optimizationPublicationAttemptDigest != attemptDigest))
      return fail(diagnostic,
                  "optimization observation publication attempt mismatch");

    std::vector<const InvocationTelemetryV1 *> terminals;
    for (const InvocationTelemetryV1 &terminal : value.invocationTerminals)
      if (terminal.identity.mechanismKey == observation.mechanismKey)
        terminals.push_back(&terminal);

    uint64_t observedInvocationCount = 0;
    uint64_t observedRewriteCount = 0;
    std::map<uint32_t, uint64_t> observedWork;
    for (const InvocationTelemetryV1 *terminal : terminals) {
      if (observedInvocationCount == std::numeric_limits<uint64_t>::max() ||
          terminal->rewriteCount >
              std::numeric_limits<uint64_t>::max() - observedRewriteCount)
        return fail(diagnostic, "terminal aggregate overflowed");
      ++observedInvocationCount;
      observedRewriteCount += terminal->rewriteCount;
      for (const WorkCounterV1 &counter :
           terminal->workSummary.orderedCounters) {
        uint64_t &sum = observedWork[counter.counterId];
        if (counter.value > std::numeric_limits<uint64_t>::max() - sum)
          return fail(diagnostic, "terminal work aggregate overflowed");
        sum += counter.value;
      }
    }
    uint64_t observationInvocations = 0;
    uint64_t observationRewrites = 0;
    std::vector<BackendActionEvidenceV1> observationActions;
    for (const InvocationEvidenceV1 &row : observation.invocationEvidence) {
      if (row.invocationCount >
              std::numeric_limits<uint64_t>::max() - observationInvocations ||
          row.rewriteCount >
              std::numeric_limits<uint64_t>::max() - observationRewrites)
        return fail(diagnostic, "observation aggregate overflowed");
      observationInvocations += row.invocationCount;
      observationRewrites += row.rewriteCount;
      observationActions.insert(observationActions.end(),
                                row.backendActions.begin(),
                                row.backendActions.end());

      uint64_t bucketInvocations = 0;
      uint64_t bucketRewrites = 0;
      std::vector<BackendActionEvidenceV1> bucketActions;
      for (const InvocationTelemetryV1 *terminal : terminals)
        if (terminal->identity.cutPoint == row.cutPoint &&
            sameRegistry(terminal->identity.invocationSite,
                         row.invocationSite) &&
            sameCase(*terminal->qualificationCase, row.caseKey)) {
          ++bucketInvocations;
          bucketRewrites += terminal->rewriteCount;
          bucketActions.insert(bucketActions.end(),
                               terminal->backendActions.begin(),
                               terminal->backendActions.end());
        }
      std::sort(bucketActions.begin(), bucketActions.end(),
                [](const auto &lhs, const auto &rhs) {
                  return std::tie(lhs.invocationId, lhs.actionOrdinal) <
                         std::tie(rhs.invocationId, rhs.actionOrdinal);
                });
      if (bucketInvocations != row.invocationCount ||
          bucketRewrites != row.rewriteCount ||
          bucketActions.size() != row.backendActions.size() ||
          !std::equal(bucketActions.begin(), bucketActions.end(),
                      row.backendActions.begin(), sameAction))
        return fail(
            diagnostic,
            "invocation evidence was not fresh-aggregated from terminals");
    }
    if (observationInvocations != observedInvocationCount ||
        observationRewrites != observedRewriteCount)
      return fail(diagnostic,
                  "observation totals do not close terminal records");

    for (const OutcomeCountV1 &count : observation.outcomeCounts) {
      uint64_t actual = llvm::count_if(terminals, [&](const auto *terminal) {
        return terminal->outcome == count.outcome &&
               sameReason(terminal->terminalReason, count.reason);
      });
      if (actual != count.count)
        return fail(diagnostic,
                    "outcome counts were not derived from terminals");
    }
    if (observation.compileWorkSummary.orderedCounters.size() !=
        observedWork.size())
      return fail(diagnostic, "work summary does not cover terminal counters");
    size_t workIndex = 0;
    for (const auto &[counterId, sum] : observedWork) {
      const WorkCounterV1 &counter =
          observation.compileWorkSummary.orderedCounters[workIndex++];
      if (counter.counterId != counterId || counter.value != sum)
        return fail(diagnostic, "work summary was not derived from terminals");
    }

    observationBindings.push_back(
        {observation.mechanismKey,
         digestQualificationObservationV1(observation)});
  }
  auto sameObservationBinding = [](const MechanismObservationBindingV1 &lhs,
                                   const MechanismObservationBindingV1 &rhs) {
    return lhs.mechanismKey == rhs.mechanismKey &&
           lhs.observationDigest == rhs.observationDigest;
  };
  if (observationBindings.size() !=
          value.resultManifest.observationBindings.size() ||
      !std::equal(observationBindings.begin(), observationBindings.end(),
                  value.resultManifest.observationBindings.begin(),
                  sameObservationBinding))
    return fail(diagnostic,
                "result manifest observations are not fresh all-and-only");

  if (optimization) {
    if (!value.optimizationBatch ||
        !validateOptimizationBatchObservationV1(*value.optimizationBatch,
                                                diagnostic) ||
        value.optimizationBatch->proposalDigest !=
            *value.input.optimizationProposalDigest ||
        value.optimizationBatch->qualificationRunDigest != runDigest ||
        value.optimizationBatch->optimizationPublicationAttemptDigest !=
            *attemptDigest ||
        !value.resultManifest.optimizationBatchObservationDigest ||
        *value.resultManifest.optimizationBatchObservationDigest !=
            digestOptimizationBatchObservationV1(*value.optimizationBatch))
      return fail(diagnostic,
                  "optimization batch/run/manifest binding is incomplete");
  } else if (value.optimizationBatch ||
             value.resultManifest.optimizationBatchObservationDigest) {
    return fail(diagnostic, "general run carries a optimization batch");
  }
  return true;
}

} // namespace wafer
