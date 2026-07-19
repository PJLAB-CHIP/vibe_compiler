//===- OptimizationQualificationEvidence.cpp - Evidence records --------===//

#include "Wafer/Support/OptimizationQualificationEvidence.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <limits>
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
  Bytes = 0x08,
  Record = 0x09,
  Sequence = 0x0a,
  Optional = 0x0c,
};

static bool fail(std::string *diagnostic, llvm::StringRef message) {
  if (diagnostic)
    *diagnostic = message.str();
  return false;
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

static uint16_t get16(llvm::ArrayRef<uint8_t> bytes) {
  return static_cast<uint16_t>((static_cast<uint16_t>(bytes[0]) << 8) |
                               bytes[1]);
}

static uint32_t get32(llvm::ArrayRef<uint8_t> bytes) {
  uint32_t value = 0;
  for (uint8_t byte : bytes.take_front(4))
    value = (value << 8) | byte;
  return value;
}

static uint64_t get64(llvm::ArrayRef<uint8_t> bytes) {
  uint64_t value = 0;
  for (uint8_t byte : bytes.take_front(8))
    value = (value << 8) | byte;
  return value;
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

static void element(std::vector<uint8_t> &sequence,
                    llvm::ArrayRef<uint8_t> payload) {
  put32(sequence, payload.size());
  sequence.insert(sequence.end(), payload.begin(), payload.end());
}

template <typename Value, typename Encoder>
static std::vector<uint8_t> sequence(llvm::ArrayRef<Value> values,
                                     Encoder encode) {
  std::vector<uint8_t> bytes;
  put32(bytes, values.size());
  for (const Value &value : values)
    element(bytes, encode(value));
  return bytes;
}

static AdoptionDigest sha256(llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 hash;
  hash.update(bytes);
  return hash.final();
}

static bool isZero(const AdoptionDigest &digest) {
  return llvm::all_of(digest, [](uint8_t byte) { return byte == 0; });
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

static bool sameConfiguration(const OptimizationConfigurationV1 &lhs,
                              const OptimizationConfigurationV1 &rhs) {
  return lhs.fixedSelection == rhs.fixedSelection &&
         lhs.fixedDisabledKey == rhs.fixedDisabledKey &&
         lhs.cleanupSelection == rhs.cleanupSelection &&
         lhs.cleanupDisabledKey == rhs.cleanupDisabledKey;
}

static bool sameComparisonKey(const OptimizationComparisonKeyV1 &lhs,
                              const OptimizationComparisonKeyV1 &rhs) {
  return sameCase(lhs.caseKey, rhs.caseKey) &&
         lhs.comparisonKind == rhs.comparisonKind &&
         lhs.disabledMechanismKey == rhs.disabledMechanismKey &&
         sameConfiguration(lhs.configurationA, rhs.configurationA) &&
         sameConfiguration(lhs.configurationB, rhs.configurationB);
}

static AdoptionDigest gateRegistryDigest() {
  constexpr char domain[] = "wafer.optimization-qualification-gate-registry";
  constexpr std::array<std::pair<uint32_t, llvm::StringLiteral>, 2> rows = {{
      {1, "equivalent-ir-2x2"},
      {2, "production-all-on"},
  }};
  std::vector<uint8_t> bytes(domain, domain + sizeof(domain));
  put16(bytes, 1);
  put32(bytes, rows.size());
  for (const auto &row : rows) {
    put32(bytes, row.first);
    put32(bytes, row.second.size());
    bytes.insert(bytes.end(), row.second.bytes_begin(), row.second.bytes_end());
  }
  return sha256(bytes);
}

static std::vector<uint8_t> encRegistry(const RegistryRefV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, U32, u32(value.id));
  field(body, 2, U16, u16(value.registrySchema));
  field(body, 3, Digest32, value.registryDigest);
  return body;
}

static std::vector<uint8_t> encCase(const QualificationCaseKeyV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, encRegistry(value.corpus));
  field(body, 2, U32, u32(value.rankCount));
  field(body, 3, ClosedEnum, u32(static_cast<uint32_t>(value.inputVariant)));
  return body;
}

static std::vector<uint8_t> encIdentity(const QualificationIdentityV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Digest32, value.build);
  field(body, 2, Digest32, value.toolchain);
  field(body, 3, Digest32, value.hostKernelAffinityGovernor);
  field(body, 4, Digest32, value.corpus);
  field(body, 5, Digest32, value.featureConfig);
  return body;
}

static std::vector<uint8_t> encPolicy(const QualificationPolicyRefV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, U32, u32(value.policyId));
  field(body, 2, U16, u16(value.policySchema));
  field(body, 3, Digest32, value.canonicalPolicyDigest);
  return body;
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

static std::vector<uint8_t> optU64(const std::optional<uint64_t> &value) {
  if (!value)
    return {0};
  std::vector<uint8_t> bytes = {1, U64};
  put32(bytes, 8);
  put64(bytes, *value);
  return bytes;
}

static std::vector<uint8_t> encReason(const ClosedReasonV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, encRegistry(value.reason));
  field(body, 2, Optional, optDigest(value.detailDigest));
  return body;
}

static std::vector<uint8_t>
optReason(const std::optional<ClosedReasonV1> &value) {
  if (!value)
    return {0};
  std::vector<uint8_t> inner = encReason(*value);
  std::vector<uint8_t> bytes = {1, Record};
  put32(bytes, inner.size());
  bytes.insert(bytes.end(), inner.begin(), inner.end());
  return bytes;
}

static std::vector<uint8_t>
optMechanism(const std::optional<MechanismKey> &value) {
  if (!value)
    return {0};
  std::vector<uint8_t> inner = encodeMechanismKeyV1(*value);
  std::vector<uint8_t> bytes = {1, Record};
  put32(bytes, inner.size());
  bytes.insert(bytes.end(), inner.begin(), inner.end());
  return bytes;
}

static std::vector<uint8_t> encOutcomeCount(const OutcomeCountV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, ClosedEnum, u32(static_cast<uint32_t>(value.outcome)));
  field(body, 2, Optional, optReason(value.reason));
  field(body, 3, U64, u64(value.count));
  return body;
}

static std::vector<uint8_t>
encBackendAction(const BackendActionEvidenceV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Digest32, value.invocationId);
  field(body, 2, U32, u32(value.actionOrdinal));
  std::vector<uint8_t> argv;
  put32(argv, value.argv.size());
  for (const std::string &arg : value.argv) {
    std::vector<uint8_t> bytes;
    put32(bytes, arg.size());
    bytes.insert(bytes.end(), arg.begin(), arg.end());
    element(argv, bytes);
  }
  field(body, 3, Sequence, argv);
  field(body, 4, Digest32, value.observedToolDigest);
  field(body, 5, Digest32, value.observedOutputDigest);
  field(body, 6, ClosedEnum, u32(static_cast<uint32_t>(value.terminalStatus)));
  return body;
}

static std::vector<uint8_t> encInvocation(const InvocationEvidenceV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, encRegistry(value.invocationSite));
  field(body, 2, ClosedEnum, u32(static_cast<uint32_t>(value.cutPoint)));
  field(body, 3, Record, encCase(value.caseKey));
  field(body, 4, U64, u64(value.invocationCount));
  field(body, 5, U64, u64(value.rewriteCount));
  field(body, 6, Sequence,
        sequence<BackendActionEvidenceV1>(value.backendActions,
                                          encBackendAction));
  return body;
}

static std::vector<uint8_t>
encInvocationKey(const InvocationEvidenceV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, encRegistry(value.invocationSite));
  field(body, 2, ClosedEnum, u32(static_cast<uint32_t>(value.cutPoint)));
  field(body, 3, Record, encCase(value.caseKey));
  return body;
}

static std::vector<uint8_t>
encodeConfiguration(const OptimizationConfigurationV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, ClosedEnum, u32(static_cast<uint32_t>(value.fixedSelection)));
  field(body, 2, Optional, optMechanism(value.fixedDisabledKey));
  field(body, 3, ClosedEnum,
        u32(static_cast<uint32_t>(value.cleanupSelection)));
  field(body, 4, Optional, optMechanism(value.cleanupDisabledKey));
  return body;
}

static std::vector<uint8_t>
encComparisonKey(const OptimizationComparisonKeyV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, encCase(value.caseKey));
  field(body, 2, ClosedEnum, u32(static_cast<uint32_t>(value.comparisonKind)));
  field(body, 3, Optional, optMechanism(value.disabledMechanismKey));
  field(body, 4, Record, encodeConfiguration(value.configurationA));
  field(body, 5, Record, encodeConfiguration(value.configurationB));
  return body;
}

static std::vector<uint8_t> encGate(const GateEvidenceV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, encRegistry(value.gate));
  field(body, 2, Record, encCase(value.caseKey));
  field(body, 3, Record, encodeConfiguration(value.configuration));
  field(body, 4, ClosedEnum, u32(static_cast<uint32_t>(value.status)));
  field(body, 5, Digest32, value.evidenceDigest);
  field(body, 6, Optional, optReason(value.terminalReason));
  return body;
}

static std::vector<uint8_t> encGateKey(const GateEvidenceV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Record, encRegistry(value.gate));
  field(body, 2, Record, encCase(value.caseKey));
  field(body, 3, Record, encodeConfiguration(value.configuration));
  return body;
}

static std::vector<uint8_t> encGateBundle(const GateEvidenceBundleV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Sequence,
        sequence<GateEvidenceV1>(value.orderedGateResults, encGate));
  return body;
}

static std::vector<uint8_t> encWorkCounter(const WorkCounterV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, U32, u32(value.counterId));
  field(body, 2, U64, u64(value.value));
  return body;
}

static std::vector<uint8_t> encWork(const WorkSummaryV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Digest32, value.workPolicyDigest);
  field(body, 2, Sequence,
        sequence<WorkCounterV1>(value.orderedCounters, encWorkCounter));
  return body;
}

static std::vector<uint8_t> encMetric(const MetricEvidenceV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, U32, u32(value.metricId));
  field(body, 2, Optional, optU64(value.value));
  return body;
}

static std::vector<uint8_t>
encStaticVector(const ExactStaticVectorEvidenceV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.registrySchema));
  field(body, 2, Digest32, value.registryDigest);
  field(body, 3, Sequence,
        sequence<MetricEvidenceV1>(value.orderedMetrics, encMetric));
  return body;
}

static std::vector<uint8_t>
encStaticComparison(const StaticComparisonEvidenceV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Digest32, value.proposalDigest);
  field(body, 2, Record, encComparisonKey(value.comparisonKey));
  field(body, 3, Record, encStaticVector(value.vectorA));
  field(body, 4, Record, encStaticVector(value.vectorB));
  field(body, 5, ClosedEnum,
        u32(static_cast<uint32_t>(value.comparisonResult)));
  return body;
}

static std::vector<uint8_t>
encStaticComparisonKey(const StaticComparisonEvidenceV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Digest32, value.proposalDigest);
  field(body, 2, Record, encComparisonKey(value.comparisonKey));
  return body;
}

static std::vector<uint8_t> encSample(const ABBASampleV1 &value) {
  std::vector<uint8_t> body;
  field(body, 1, Digest32, value.proposalDigest);
  field(body, 2, Record, encComparisonKey(value.comparisonKey));
  field(body, 3, U32, u32(value.blockIndex));
  field(body, 4, ClosedEnum,
        u32(static_cast<uint32_t>(value.sequencePosition)));
  field(body, 5, U64, u64(value.wallNs));
  field(body, 6, U64, u64(value.peakRssBytes));
  field(body, 7, ClosedEnum, u32(static_cast<uint32_t>(value.processStatus)));
  return body;
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

static OptimizationConfigurationV1 allOnConfiguration() {
  return {OptimizationGroupSelectionV1::AllOn, std::nullopt,
          OptimizationGroupSelectionV1::AllOn, std::nullopt};
}

static OptimizationConfigurationV1 allOffConfiguration() {
  return {OptimizationGroupSelectionV1::AllOff, std::nullopt,
          OptimizationGroupSelectionV1::AllOff, std::nullopt};
}

static OptimizationConfigurationV1
fixedDisabledConfiguration(MechanismKey key) {
  return {OptimizationGroupSelectionV1::DisableOne, key,
          OptimizationGroupSelectionV1::AllOn, std::nullopt};
}

static OptimizationConfigurationV1
cleanupDisabledConfiguration(MechanismKey key) {
  return {OptimizationGroupSelectionV1::AllOn, std::nullopt,
          OptimizationGroupSelectionV1::DisableOne, key};
}

static OptimizationConfigurationV1
equivalentConfiguration(bool cleanupEnabled) {
  return {OptimizationGroupSelectionV1::AllOff, std::nullopt,
          cleanupEnabled ? OptimizationGroupSelectionV1::AllOn
                         : OptimizationGroupSelectionV1::AllOff,
          std::nullopt};
}

static bool validateIdentity(const QualificationIdentityV1 &identity,
                             std::string *diagnostic) {
  if (isZero(identity.build) || isZero(identity.toolchain) ||
      isZero(identity.hostKernelAffinityGovernor) || isZero(identity.corpus) ||
      isZero(identity.featureConfig))
    return fail(diagnostic, "qualification identity contains a zero digest");
  return true;
}

static bool validateReason(const ClosedReasonV1 &reason,
                           std::string *diagnostic) {
  if (reason.reason.id == 0 || reason.reason.registrySchema == 0 ||
      isZero(reason.reason.registryDigest) ||
      (reason.detailDigest && isZero(*reason.detailDigest)))
    return fail(diagnostic, "closed reason is not a valid registry binding");
  bool known = false;
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
    known |= isGlobalClosedReasonRefV1(reason.reason, candidate);
  if (!known)
    return fail(diagnostic, "closed reason registry revision is unknown");
  return true;
}

static bool isMandatoryCase(const QualificationCaseKeyV1 &key,
                            bool allowMetamorphic) {
  if (key.inputVariant != EquivalentInputVariantV1::Original &&
      (!allowMetamorphic ||
       key.inputVariant != EquivalentInputVariantV1::Metamorphic))
    return false;
  return llvm::any_of(getCurrentMandatoryQualificationCasesV1(),
                      [&](const QualificationCaseKeyV1 &candidate) {
                        return sameRegistry(candidate.corpus, key.corpus) &&
                               candidate.rankCount == key.rankCount;
                      });
}

static bool
validateConfiguration(const OptimizationQualificationProposal &proposal,
                      const OptimizationConfigurationV1 &configuration,
                      std::string *diagnostic) {
  auto convert = [](OptimizationGroupSelectionV1 kind)
      -> std::optional<OptimizationGroupSelectionKind> {
    switch (kind) {
    case OptimizationGroupSelectionV1::AllOn:
      return OptimizationGroupSelectionKind::AllOn;
    case OptimizationGroupSelectionV1::AllOff:
      return OptimizationGroupSelectionKind::AllOff;
    case OptimizationGroupSelectionV1::DisableOne:
      return OptimizationGroupSelectionKind::DisableOne;
    }
    return std::nullopt;
  };
  std::optional<OptimizationGroupSelectionKind> fixed =
      convert(configuration.fixedSelection);
  std::optional<OptimizationGroupSelectionKind> cleanup =
      convert(configuration.cleanupSelection);
  if (!fixed || !cleanup)
    return fail(diagnostic, "unknown optimization selection enum");
  OptimizationConfiguration converted{
      {*fixed, configuration.fixedDisabledKey},
      {*cleanup, configuration.cleanupDisabledKey}};
  return validateOptimizationConfiguration(proposal, converted, diagnostic);
}

static bool validateStaticVector(const ExactStaticVectorEvidenceV1 &value,
                                 std::string *diagnostic) {
  RegistryRefV1 registry = getCurrentStaticMetricRegistryRefV1();
  if (value.registrySchema != registry.registrySchema ||
      value.registryDigest != registry.registryDigest ||
      value.orderedMetrics.size() != 4)
    return fail(diagnostic,
                "static vector does not bind the complete metric registry");
  for (size_t index = 0; index < value.orderedMetrics.size(); ++index)
    if (value.orderedMetrics[index].metricId != index + 1)
      return fail(diagnostic,
                  "static metrics are not all-and-only registry order");
  return true;
}

static StaticComparisonResultV1
recomputeStaticResult(const ExactStaticVectorEvidenceV1 &a,
                      const ExactStaticVectorEvidenceV1 &b) {
  bool unknown = false;
  bool better = false;
  for (size_t index = 0; index < a.orderedMetrics.size(); ++index) {
    const std::optional<uint64_t> &av = a.orderedMetrics[index].value;
    const std::optional<uint64_t> &bv = b.orderedMetrics[index].value;
    if (!av || !bv) {
      unknown = true;
      continue;
    }
    if (*bv > *av)
      return StaticComparisonResultV1::BRegressed;
    better |= *bv < *av;
  }
  if (unknown)
    return StaticComparisonResultV1::IncomparableUnknown;
  return better ? StaticComparisonResultV1::BStrictlyBetter
                : StaticComparisonResultV1::BEqual;
}

static bool
validateComparisonShape(const OptimizationQualificationProposal &proposal,
                        const OptimizationComparisonKeyV1 &key,
                        std::string *diagnostic) {
  if (!isMandatoryCase(key.caseKey, false) ||
      key.caseKey.inputVariant != EquivalentInputVariantV1::Original ||
      !validateConfiguration(proposal, key.configurationA, diagnostic) ||
      !validateConfiguration(proposal, key.configurationB, diagnostic))
    return false;
  switch (key.comparisonKind) {
  case OptimizationComparisonKindV1::GlobalAllOffVsAllOn:
    if (key.disabledMechanismKey ||
        !sameConfiguration(key.configurationA, allOffConfiguration()) ||
        !sameConfiguration(key.configurationB, allOnConfiguration()))
      return fail(diagnostic, "invalid global AllOff-vs-AllOn comparison");
    return true;
  case OptimizationComparisonKindV1::FixedDisableOneVsAllOn:
    if (!key.disabledMechanismKey ||
        !sameConfiguration(
            key.configurationA,
            fixedDisabledConfiguration(*key.disabledMechanismKey)) ||
        !sameConfiguration(key.configurationB, allOnConfiguration()))
      return fail(diagnostic, "invalid fixed DisableOne comparison");
    return true;
  case OptimizationComparisonKindV1::CleanupDisableOneVsAllOn:
    if (!key.disabledMechanismKey ||
        !sameConfiguration(
            key.configurationA,
            cleanupDisabledConfiguration(*key.disabledMechanismKey)) ||
        !sameConfiguration(key.configurationB, allOnConfiguration()))
      return fail(diagnostic, "invalid cleanup DisableOne comparison");
    return true;
  }
  return fail(diagnostic, "unknown optimization comparison kind");
}

static bool
validateStaticComparison(const OptimizationQualificationProposal &proposal,
                         const StaticComparisonEvidenceV1 &value,
                         std::string *diagnostic) {
  AdoptionDigest proposalDigest =
      digestOptimizationQualificationProposalV1(proposal);
  if (value.proposalDigest != proposalDigest ||
      !validateComparisonShape(proposal, value.comparisonKey, diagnostic) ||
      !validateStaticVector(value.vectorA, diagnostic) ||
      !validateStaticVector(value.vectorB, diagnostic))
    return false;
  if (recomputeStaticResult(value.vectorA, value.vectorB) !=
      value.comparisonResult)
    return fail(diagnostic,
                "static comparison result does not match metric vectors");
  return true;
}

static bool validateGate(const OptimizationQualificationProposal &proposal,
                         const GateEvidenceV1 &gate, std::string *diagnostic) {
  if (gate.gate.id == 0 || gate.gate.registrySchema == 0 ||
      isZero(gate.gate.registryDigest) ||
      !isMandatoryCase(gate.caseKey, true) ||
      !validateConfiguration(proposal, gate.configuration, diagnostic) ||
      isZero(gate.evidenceDigest))
    return fail(diagnostic, "gate evidence has an invalid binding");
  bool passed = gate.status == GateStatusV1::Passed;
  if (passed == static_cast<bool>(gate.terminalReason))
    return fail(diagnostic, "gate status/reason presence is inconsistent");
  return !gate.terminalReason ||
         validateReason(*gate.terminalReason, diagnostic);
}

static bool
validateGateBundle(const OptimizationQualificationProposal &proposal,
                   const GateEvidenceBundleV1 &bundle, bool requirePassed,
                   std::string *diagnostic) {
  std::vector<uint8_t> previous;
  for (const GateEvidenceV1 &gate : bundle.orderedGateResults) {
    if (!validateGate(proposal, gate, diagnostic) ||
        (requirePassed && gate.status != GateStatusV1::Passed))
      return false;
    std::vector<uint8_t> key = encGateKey(gate);
    if (!previous.empty() && !(previous < key))
      return fail(diagnostic, "gate evidence keys are not sorted and unique");
    previous = std::move(key);
  }
  return true;
}

static bool validateABBA(const OptimizationQualificationProposal &proposal,
                         llvm::ArrayRef<StaticComparisonEvidenceV1> comparisons,
                         llvm::ArrayRef<ABBASampleV1> samples,
                         bool requireAccepted, std::string *diagnostic) {
  OptimizationSetQualificationPolicyV1 policy =
      getCurrentOptimizationSetQualificationPolicyV1();
  uint64_t perComparison = static_cast<uint64_t>(policy.abbaBlockCount) * 4;
  if (comparisons.size() >
          std::numeric_limits<uint64_t>::max() / perComparison ||
      samples.size() != comparisons.size() * perComparison)
    return fail(diagnostic, "ABBA sample count is not exact");
  size_t sampleIndex = 0;
  for (const StaticComparisonEvidenceV1 &comparison : comparisons) {
    std::vector<ABBASample> hostSamples;
    hostSamples.reserve(perComparison);
    for (uint32_t block = 0; block < policy.abbaBlockCount; ++block) {
      for (uint32_t position = 0; position < 4; ++position) {
        const ABBASampleV1 &sample = samples[sampleIndex++];
        if (sample.proposalDigest != comparison.proposalDigest ||
            !sameComparisonKey(sample.comparisonKey,
                               comparison.comparisonKey) ||
            sample.blockIndex != block ||
            static_cast<uint32_t>(sample.sequencePosition) != position ||
            sample.processStatus != ProcessStatusV1::Success)
          return fail(diagnostic,
                      "ABBA samples do not follow exact A-B-B-A sequence");
        hostSamples.push_back({sample.blockIndex, sample.sequencePosition,
                               sample.wallNs, sample.peakRssBytes, true});
      }
    }
    OptimizationSetHostEvaluation evaluation =
        evaluateOptimizationSetABBA(hostSamples);
    if (!evaluation.decision ||
        (requireAccepted && !evaluation.decision->accepted))
      return fail(diagnostic,
                  "ABBA host evidence is invalid or exceeds fixed guard");
  }
  return true;
}

static bool hasSignificantHostBenefit(llvm::ArrayRef<ABBASampleV1> samples) {
  constexpr size_t samplesPerComparison = 5 * 4;
  if (samples.size() % samplesPerComparison != 0)
    return false;
  for (size_t begin = 0; begin < samples.size();
       begin += samplesPerComparison) {
    std::vector<ABBASample> hostSamples;
    hostSamples.reserve(samplesPerComparison);
    for (const ABBASampleV1 &sample :
         samples.slice(begin, samplesPerComparison))
      hostSamples.push_back({sample.blockIndex, sample.sequencePosition,
                             sample.wallNs, sample.peakRssBytes, true});
    OptimizationSetHostEvaluation evaluation =
        evaluateOptimizationSetABBA(hostSamples);
    if (evaluation.decision && evaluation.decision->significantHostBenefit)
      return true;
  }
  return false;
}

static std::vector<OptimizationComparisonKeyV1>
expectedComparisons(OptimizationComparisonKindV1 kind,
                    std::optional<MechanismKey> disabledMechanism) {
  std::vector<OptimizationComparisonKeyV1> keys;
  for (const QualificationCaseKeyV1 &mandatoryCase :
       getCurrentMandatoryQualificationCasesV1()) {
    OptimizationComparisonKeyV1 key;
    key.caseKey = mandatoryCase;
    key.comparisonKind = kind;
    key.disabledMechanismKey = disabledMechanism;
    key.configurationB = allOnConfiguration();
    if (kind == OptimizationComparisonKindV1::GlobalAllOffVsAllOn)
      key.configurationA = allOffConfiguration();
    else if (kind == OptimizationComparisonKindV1::FixedDisableOneVsAllOn)
      key.configurationA = fixedDisabledConfiguration(*disabledMechanism);
    else
      key.configurationA = cleanupDisabledConfiguration(*disabledMechanism);
    keys.push_back(std::move(key));
  }
  std::sort(keys.begin(), keys.end(),
            [](const OptimizationComparisonKeyV1 &lhs,
               const OptimizationComparisonKeyV1 &rhs) {
              return encComparisonKey(lhs) < encComparisonKey(rhs);
            });
  return keys;
}

static bool
validateComparisonDomain(llvm::ArrayRef<StaticComparisonEvidenceV1> comparisons,
                         const OptimizationQualificationProposal &proposal,
                         OptimizationComparisonKindV1 kind,
                         std::optional<MechanismKey> disabledMechanism,
                         std::string *diagnostic) {
  std::vector<OptimizationComparisonKeyV1> expected =
      expectedComparisons(kind, disabledMechanism);
  // An empty proposal makes AllOff and AllOn the same configuration.  There is
  // no optimization comparison to measure in that case; requiring ABBA rows
  // would manufacture performance evidence for an identity comparison.
  if (kind == OptimizationComparisonKindV1::GlobalAllOffVsAllOn &&
      proposal.fixedBindings.empty() && proposal.cleanupBindings.empty())
    expected.clear();
  if (comparisons.size() != expected.size())
    return fail(diagnostic,
                "static comparison rows do not cover mandatory domain");
  std::vector<uint8_t> previous;
  for (size_t index = 0; index < expected.size(); ++index) {
    if (!sameComparisonKey(comparisons[index].comparisonKey, expected[index]))
      return fail(diagnostic,
                  "static comparison rows are not all-and-only domain");
    std::vector<uint8_t> key = encStaticComparisonKey(comparisons[index]);
    if (!previous.empty() && !(previous < key))
      return fail(diagnostic,
                  "static comparison rows are not sorted and unique");
    previous = std::move(key);
  }
  return true;
}

static bool validateWork(const WorkSummaryV1 &work, const AdoptionSpec &spec,
                         std::string *diagnostic) {
  if (work.workPolicyDigest != digestAdoptionWorkPolicyV1(spec.workPolicyKind))
    return fail(diagnostic, "work summary does not bind spec work policy");
  uint32_t previous = 0;
  for (const WorkCounterV1 &counter : work.orderedCounters) {
    if (counter.counterId == 0 || counter.counterId <= previous)
      return fail(diagnostic, "work counters are not sorted and unique");
    previous = counter.counterId;
  }
  return true;
}

static bool validateOutcomeCounts(llvm::ArrayRef<OutcomeCountV1> counts,
                                  uint64_t &total, bool &hasApplied,
                                  std::string *diagnostic) {
  total = 0;
  hasApplied = false;
  std::vector<uint8_t> previous;
  for (const OutcomeCountV1 &count : counts) {
    if (count.count == 0)
      return fail(diagnostic, "outcome count row is zero");
    bool normal = count.outcome == InvocationOutcome::Applied ||
                  count.outcome == InvocationOutcome::NoChange ||
                  count.outcome == InvocationOutcome::NotApplicable;
    if (normal == static_cast<bool>(count.reason))
      return fail(diagnostic, "outcome count reason presence is inconsistent");
    if (count.reason && !validateReason(*count.reason, diagnostic))
      return false;
    if (count.count > std::numeric_limits<uint64_t>::max() - total)
      return fail(diagnostic, "outcome count total overflowed");
    total += count.count;
    hasApplied |= count.outcome == InvocationOutcome::Applied;
    std::vector<uint8_t> key = encOutcomeCount(count);
    // Count is the final field; comparing full records also fixes deterministic
    // bytes while duplicate outcome/reason is rejected explicitly below.
    if (!previous.empty() && !(previous < key))
      return fail(diagnostic, "outcome rows are not canonical order");
    previous = std::move(key);
  }
  for (size_t i = 1; i < counts.size(); ++i)
    if (counts[i - 1].outcome == counts[i].outcome &&
        optReason(counts[i - 1].reason) == optReason(counts[i].reason))
      return fail(diagnostic, "duplicate outcome/reason row");
  return true;
}

static bool
validateInvocationEvidence(llvm::ArrayRef<InvocationEvidenceV1> evidence,
                           const AdoptionSpec &spec, uint64_t &totalInvocations,
                           uint64_t &totalRewrites, uint64_t &totalActions,
                           bool &allActionsSucceeded, std::string *diagnostic) {
  totalInvocations = totalRewrites = totalActions = 0;
  allActionsSucceeded = true;
  std::vector<uint8_t> previous;
  for (const InvocationEvidenceV1 &row : evidence) {
    if (row.cutPoint != spec.cutPoint || !isMandatoryCase(row.caseKey, true) ||
        row.invocationCount == 0)
      return fail(diagnostic, "invocation evidence has invalid cut/case/count");
    std::optional<RegistryRefV1> expectedSite =
        lookupOptimizationInvocationSiteV1(spec.mechanismKey);
    if (!expectedSite || !sameRegistry(row.invocationSite, *expectedSite))
      return fail(diagnostic,
                  "invocation evidence does not bind mechanism site");
    if (row.invocationCount >
            std::numeric_limits<uint64_t>::max() - totalInvocations ||
        row.rewriteCount > std::numeric_limits<uint64_t>::max() - totalRewrites)
      return fail(diagnostic, "invocation aggregate overflowed");
    totalInvocations += row.invocationCount;
    totalRewrites += row.rewriteCount;
    std::vector<uint8_t> key = encInvocationKey(row);
    if (!previous.empty() && !(previous < key))
      return fail(diagnostic,
                  "invocation evidence keys are not sorted and unique");
    previous = std::move(key);

    AdoptionDigest previousInvocation{};
    uint32_t expectedOrdinal = 0;
    bool haveAction = false;
    for (const BackendActionEvidenceV1 &action : row.backendActions) {
      if (isZero(action.invocationId) || action.argv.empty() ||
          isZero(action.observedToolDigest) ||
          isZero(action.observedOutputDigest))
        return fail(diagnostic, "backend action lacks actual evidence");
      if (!haveAction || action.invocationId != previousInvocation) {
        if (haveAction && !(previousInvocation < action.invocationId))
          return fail(diagnostic,
                      "backend actions are not sorted by invocation ID");
        previousInvocation = action.invocationId;
        expectedOrdinal = 0;
        haveAction = true;
      }
      if (action.actionOrdinal != expectedOrdinal++)
        return fail(diagnostic, "backend action ordinals are not contiguous");
      allActionsSucceeded &=
          action.terminalStatus == BackendActionStatusV1::Success;
      if (totalActions == std::numeric_limits<uint64_t>::max())
        return fail(diagnostic, "backend action count overflowed");
      ++totalActions;
    }
  }
  return true;
}

static bool allGatesPassed(const GateEvidenceBundleV1 &bundle) {
  return llvm::all_of(bundle.orderedGateResults,
                      [](const GateEvidenceV1 &gate) {
                        return gate.status == GateStatusV1::Passed;
                      });
}

class Reader {
public:
  Reader(llvm::ArrayRef<uint8_t> bytes, std::string *diagnostic)
      : bytes_(bytes), diagnostic_(diagnostic) {}

  bool field(uint16_t number, Tag tag, llvm::ArrayRef<uint8_t> &payload) {
    if (bytes_.size() - cursor_ < 7)
      return fail(diagnostic_, "missing or truncated evidence field");
    uint16_t actualNumber = get16(bytes_.slice(cursor_, 2));
    uint8_t actualTag = bytes_[cursor_ + 2];
    uint32_t size = get32(bytes_.slice(cursor_ + 3, 4));
    cursor_ += 7;
    if (actualNumber != number || actualTag != tag ||
        size > bytes_.size() - cursor_)
      return fail(diagnostic_,
                  "unknown, reordered, mistyped or truncated evidence field");
    payload = bytes_.slice(cursor_, size);
    cursor_ += size;
    return true;
  }

  bool done() const { return cursor_ == bytes_.size(); }

private:
  llvm::ArrayRef<uint8_t> bytes_;
  size_t cursor_ = 0;
  std::string *diagnostic_;
};

static bool parseEnvelope(const std::vector<uint8_t> &bytes,
                          llvm::StringRef domainIncludingNul,
                          llvm::ArrayRef<uint8_t> &body,
                          std::string *diagnostic) {
  size_t envelope = domainIncludingNul.size() + 6;
  if (bytes.size() < envelope ||
      !std::equal(domainIncludingNul.begin(), domainIncludingNul.end(),
                  bytes.begin()) ||
      get16(llvm::ArrayRef<uint8_t>(bytes).slice(domainIncludingNul.size(),
                                                 2)) != 1 ||
      get32(llvm::ArrayRef<uint8_t>(bytes).slice(domainIncludingNul.size() + 2,
                                                 4)) != bytes.size() - envelope)
    return fail(diagnostic, "invalid evidence envelope");
  body = llvm::ArrayRef<uint8_t>(bytes).drop_front(envelope);
  return true;
}

static bool parseRegistry(llvm::ArrayRef<uint8_t> bytes, RegistryRefV1 &value,
                          std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, U32, payload) || payload.size() != 4)
    return false;
  value.id = get32(payload);
  if (!reader.field(2, U16, payload) || payload.size() != 2)
    return false;
  value.registrySchema = get16(payload);
  if (!reader.field(3, Digest32, payload) || payload.size() != 32 ||
      !reader.done())
    return false;
  std::copy(payload.begin(), payload.end(), value.registryDigest.begin());
  return true;
}

static bool parseCase(llvm::ArrayRef<uint8_t> bytes,
                      QualificationCaseKeyV1 &value, std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, Record, payload) ||
      !parseRegistry(payload, value.corpus, diagnostic) ||
      !reader.field(2, U32, payload) || payload.size() != 4)
    return false;
  value.rankCount = get32(payload);
  if (!reader.field(3, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) >
          static_cast<uint32_t>(EquivalentInputVariantV1::Metamorphic) ||
      !reader.done())
    return false;
  value.inputVariant = static_cast<EquivalentInputVariantV1>(get32(payload));
  return true;
}

static bool parseIdentity(llvm::ArrayRef<uint8_t> bytes,
                          QualificationIdentityV1 &value,
                          std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  AdoptionDigest *digests[] = {&value.build, &value.toolchain,
                               &value.hostKernelAffinityGovernor, &value.corpus,
                               &value.featureConfig};
  for (uint16_t fieldNumber = 1; fieldNumber <= 5; ++fieldNumber) {
    if (!reader.field(fieldNumber, Digest32, payload) || payload.size() != 32)
      return false;
    std::copy(payload.begin(), payload.end(),
              digests[fieldNumber - 1]->begin());
  }
  return reader.done() || fail(diagnostic, "identity has extra fields");
}

static bool parsePolicy(llvm::ArrayRef<uint8_t> bytes,
                        QualificationPolicyRefV1 &value,
                        std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, U32, payload) || payload.size() != 4)
    return false;
  value.policyId = get32(payload);
  if (!reader.field(2, U16, payload) || payload.size() != 2)
    return false;
  value.policySchema = get16(payload);
  if (!reader.field(3, Digest32, payload) || payload.size() != 32 ||
      !reader.done())
    return false;
  std::copy(payload.begin(), payload.end(),
            value.canonicalPolicyDigest.begin());
  return true;
}

static bool parseOptionalDigest(llvm::ArrayRef<uint8_t> bytes,
                                std::optional<AdoptionDigest> &value,
                                std::string *diagnostic) {
  if (bytes.size() == 1 && bytes[0] == 0) {
    value.reset();
    return true;
  }
  if (bytes.size() != 1 + 1 + 4 + 32 || bytes[0] != 1 || bytes[1] != Digest32 ||
      get32(bytes.slice(2, 4)) != 32)
    return fail(diagnostic, "invalid optional digest");
  AdoptionDigest digest;
  std::copy(bytes.begin() + 6, bytes.end(), digest.begin());
  value = digest;
  return true;
}

static bool parseReason(llvm::ArrayRef<uint8_t> bytes, ClosedReasonV1 &value,
                        std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, Record, payload) ||
      !parseRegistry(payload, value.reason, diagnostic) ||
      !reader.field(2, Optional, payload) ||
      !parseOptionalDigest(payload, value.detailDigest, diagnostic) ||
      !reader.done())
    return false;
  return true;
}

static bool parseOptionalReason(llvm::ArrayRef<uint8_t> bytes,
                                std::optional<ClosedReasonV1> &value,
                                std::string *diagnostic) {
  if (bytes.size() == 1 && bytes[0] == 0) {
    value.reset();
    return true;
  }
  if (bytes.size() < 6 || bytes[0] != 1 || bytes[1] != Record ||
      get32(bytes.slice(2, 4)) != bytes.size() - 6)
    return fail(diagnostic, "invalid optional reason");
  ClosedReasonV1 reason;
  if (!parseReason(bytes.drop_front(6), reason, diagnostic))
    return false;
  value = std::move(reason);
  return true;
}

static bool parseMechanism(llvm::ArrayRef<uint8_t> bytes, MechanismKey &value,
                           std::string *diagnostic) {
  constexpr char domain[] = "wafer.mechanism-key";
  if (bytes.size() != sizeof(domain) + 2 + 4 ||
      !std::equal(std::begin(domain), std::end(domain), bytes.begin()) ||
      get16(bytes.slice(sizeof(domain), 2)) != 1)
    return fail(diagnostic, "invalid mechanism key record");
  value.semanticId = get32(bytes.take_back(4));
  return lookupMechanismDescriptor(value).has_value() ||
         fail(diagnostic, "unknown mechanism key");
}

static bool parseOptionalMechanism(llvm::ArrayRef<uint8_t> bytes,
                                   std::optional<MechanismKey> &value,
                                   std::string *diagnostic) {
  if (bytes.size() == 1 && bytes[0] == 0) {
    value.reset();
    return true;
  }
  if (bytes.size() < 6 || bytes[0] != 1 || bytes[1] != Record ||
      get32(bytes.slice(2, 4)) != bytes.size() - 6)
    return fail(diagnostic, "invalid optional mechanism key");
  MechanismKey key;
  if (!parseMechanism(bytes.drop_front(6), key, diagnostic))
    return false;
  value = key;
  return true;
}

template <typename Value, typename Parser>
static bool parseSequence(llvm::ArrayRef<uint8_t> bytes,
                          std::vector<Value> &values, Parser parse,
                          std::string *diagnostic) {
  if (bytes.size() < 4)
    return fail(diagnostic, "truncated evidence sequence");
  uint32_t count = get32(bytes.take_front(4));
  bytes = bytes.drop_front(4);
  values.clear();
  values.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    if (bytes.size() < 4)
      return fail(diagnostic, "truncated evidence sequence element");
    uint32_t size = get32(bytes.take_front(4));
    bytes = bytes.drop_front(4);
    if (size > bytes.size())
      return fail(diagnostic, "truncated evidence sequence payload");
    Value value;
    if (!parse(bytes.take_front(size), value, diagnostic))
      return false;
    values.push_back(std::move(value));
    bytes = bytes.drop_front(size);
  }
  return bytes.empty() || fail(diagnostic, "evidence sequence has extras");
}

static bool parseOutcome(llvm::ArrayRef<uint8_t> bytes, OutcomeCountV1 &value,
                         std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) > static_cast<uint32_t>(InvocationOutcome::Cancelled))
    return false;
  value.outcome = static_cast<InvocationOutcome>(get32(payload));
  if (!reader.field(2, Optional, payload) ||
      !parseOptionalReason(payload, value.reason, diagnostic) ||
      !reader.field(3, U64, payload) || payload.size() != 8 || !reader.done())
    return false;
  value.count = get64(payload);
  return true;
}

static bool parseBackendAction(llvm::ArrayRef<uint8_t> bytes,
                               BackendActionEvidenceV1 &value,
                               std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), value.invocationId.begin());
  if (!reader.field(2, U32, payload) || payload.size() != 4)
    return false;
  value.actionOrdinal = get32(payload);
  if (!reader.field(3, Sequence, payload) || payload.size() < 4)
    return false;
  uint32_t count = get32(payload.take_front(4));
  payload = payload.drop_front(4);
  value.argv.clear();
  for (uint32_t index = 0; index < count; ++index) {
    if (payload.size() < 8)
      return fail(diagnostic, "truncated backend argv element");
    uint32_t outerSize = get32(payload.take_front(4));
    payload = payload.drop_front(4);
    if (outerSize > payload.size() || outerSize < 4 ||
        get32(payload.take_front(4)) != outerSize - 4)
      return fail(diagnostic, "invalid backend argv bytes");
    value.argv.emplace_back(payload.begin() + 4, payload.begin() + outerSize);
    payload = payload.drop_front(outerSize);
  }
  if (!payload.empty())
    return fail(diagnostic, "backend argv has trailing bytes");
  if (!reader.field(4, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), value.observedToolDigest.begin());
  if (!reader.field(5, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), value.observedOutputDigest.begin());
  if (!reader.field(6, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) >
          static_cast<uint32_t>(BackendActionStatusV1::Cancelled) ||
      !reader.done())
    return false;
  value.terminalStatus = static_cast<BackendActionStatusV1>(get32(payload));
  return true;
}

static bool parseInvocation(llvm::ArrayRef<uint8_t> bytes,
                            InvocationEvidenceV1 &value,
                            std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, Record, payload) ||
      !parseRegistry(payload, value.invocationSite, diagnostic) ||
      !reader.field(2, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) > static_cast<uint32_t>(
                           OptimizationCutPoint::DevicePublicationTransaction))
    return false;
  value.cutPoint = static_cast<OptimizationCutPoint>(get32(payload));
  if (!reader.field(3, Record, payload) ||
      !parseCase(payload, value.caseKey, diagnostic) ||
      !reader.field(4, U64, payload) || payload.size() != 8)
    return false;
  value.invocationCount = get64(payload);
  if (!reader.field(5, U64, payload) || payload.size() != 8)
    return false;
  value.rewriteCount = get64(payload);
  return reader.field(6, Sequence, payload) &&
         parseSequence(payload, value.backendActions, parseBackendAction,
                       diagnostic) &&
         reader.done();
}

static bool parseConfiguration(llvm::ArrayRef<uint8_t> bytes,
                               OptimizationConfigurationV1 &value,
                               std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) >
          static_cast<uint32_t>(OptimizationGroupSelectionV1::DisableOne))
    return false;
  value.fixedSelection =
      static_cast<OptimizationGroupSelectionV1>(get32(payload));
  if (!reader.field(2, Optional, payload) ||
      !parseOptionalMechanism(payload, value.fixedDisabledKey, diagnostic) ||
      !reader.field(3, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) >
          static_cast<uint32_t>(OptimizationGroupSelectionV1::DisableOne))
    return false;
  value.cleanupSelection =
      static_cast<OptimizationGroupSelectionV1>(get32(payload));
  return reader.field(4, Optional, payload) &&
         parseOptionalMechanism(payload, value.cleanupDisabledKey,
                                diagnostic) &&
         reader.done();
}

static bool parseComparisonKey(llvm::ArrayRef<uint8_t> bytes,
                               OptimizationComparisonKeyV1 &value,
                               std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, Record, payload) ||
      !parseCase(payload, value.caseKey, diagnostic) ||
      !reader.field(2, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) >
          static_cast<uint32_t>(
              OptimizationComparisonKindV1::CleanupDisableOneVsAllOn))
    return false;
  value.comparisonKind =
      static_cast<OptimizationComparisonKindV1>(get32(payload));
  return reader.field(3, Optional, payload) &&
         parseOptionalMechanism(payload, value.disabledMechanismKey,
                                diagnostic) &&
         reader.field(4, Record, payload) &&
         parseConfiguration(payload, value.configurationA, diagnostic) &&
         reader.field(5, Record, payload) &&
         parseConfiguration(payload, value.configurationB, diagnostic) &&
         reader.done();
}

static bool parseGate(llvm::ArrayRef<uint8_t> bytes, GateEvidenceV1 &value,
                      std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, Record, payload) ||
      !parseRegistry(payload, value.gate, diagnostic) ||
      !reader.field(2, Record, payload) ||
      !parseCase(payload, value.caseKey, diagnostic) ||
      !reader.field(3, Record, payload) ||
      !parseConfiguration(payload, value.configuration, diagnostic) ||
      !reader.field(4, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) > static_cast<uint32_t>(GateStatusV1::Cancelled))
    return false;
  value.status = static_cast<GateStatusV1>(get32(payload));
  if (!reader.field(5, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), value.evidenceDigest.begin());
  return reader.field(6, Optional, payload) &&
         parseOptionalReason(payload, value.terminalReason, diagnostic) &&
         reader.done();
}

static bool parseGateBundle(llvm::ArrayRef<uint8_t> bytes,
                            GateEvidenceBundleV1 &value,
                            std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  return reader.field(1, Sequence, payload) &&
         parseSequence(payload, value.orderedGateResults, parseGate,
                       diagnostic) &&
         reader.done();
}

static bool parseWorkCounter(llvm::ArrayRef<uint8_t> bytes,
                             WorkCounterV1 &value, std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, U32, payload) || payload.size() != 4)
    return false;
  value.counterId = get32(payload);
  if (!reader.field(2, U64, payload) || payload.size() != 8 || !reader.done())
    return false;
  value.value = get64(payload);
  return true;
}

static bool parseWork(llvm::ArrayRef<uint8_t> bytes, WorkSummaryV1 &value,
                      std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), value.workPolicyDigest.begin());
  return reader.field(2, Sequence, payload) &&
         parseSequence(payload, value.orderedCounters, parseWorkCounter,
                       diagnostic) &&
         reader.done();
}

static bool parseOptionalU64(llvm::ArrayRef<uint8_t> bytes,
                             std::optional<uint64_t> &value,
                             std::string *diagnostic) {
  if (bytes.size() == 1 && bytes[0] == 0) {
    value.reset();
    return true;
  }
  if (bytes.size() != 14 || bytes[0] != 1 || bytes[1] != U64 ||
      get32(bytes.slice(2, 4)) != 8)
    return fail(diagnostic, "invalid optional U64");
  value = get64(bytes.drop_front(6));
  return true;
}

static bool parseMetric(llvm::ArrayRef<uint8_t> bytes, MetricEvidenceV1 &value,
                        std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, U32, payload) || payload.size() != 4)
    return false;
  value.metricId = get32(payload);
  return reader.field(2, Optional, payload) &&
         parseOptionalU64(payload, value.value, diagnostic) && reader.done();
}

static bool parseStaticVector(llvm::ArrayRef<uint8_t> bytes,
                              ExactStaticVectorEvidenceV1 &value,
                              std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, U16, payload) || payload.size() != 2)
    return false;
  value.registrySchema = get16(payload);
  if (!reader.field(2, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), value.registryDigest.begin());
  return reader.field(3, Sequence, payload) &&
         parseSequence(payload, value.orderedMetrics, parseMetric,
                       diagnostic) &&
         reader.done();
}

static bool parseStaticComparison(llvm::ArrayRef<uint8_t> bytes,
                                  StaticComparisonEvidenceV1 &value,
                                  std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), value.proposalDigest.begin());
  if (!reader.field(2, Record, payload) ||
      !parseComparisonKey(payload, value.comparisonKey, diagnostic) ||
      !reader.field(3, Record, payload) ||
      !parseStaticVector(payload, value.vectorA, diagnostic) ||
      !reader.field(4, Record, payload) ||
      !parseStaticVector(payload, value.vectorB, diagnostic) ||
      !reader.field(5, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) > static_cast<uint32_t>(
                           StaticComparisonResultV1::IncomparableUnknown) ||
      !reader.done())
    return false;
  value.comparisonResult =
      static_cast<StaticComparisonResultV1>(get32(payload));
  return true;
}

static bool parseSample(llvm::ArrayRef<uint8_t> bytes, ABBASampleV1 &value,
                        std::string *diagnostic) {
  Reader reader(bytes, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.field(1, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), value.proposalDigest.begin());
  if (!reader.field(2, Record, payload) ||
      !parseComparisonKey(payload, value.comparisonKey, diagnostic) ||
      !reader.field(3, U32, payload) || payload.size() != 4)
    return false;
  value.blockIndex = get32(payload);
  if (!reader.field(4, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) > static_cast<uint32_t>(ABBASamplePosition::ARight))
    return false;
  value.sequencePosition = static_cast<ABBASamplePosition>(get32(payload));
  if (!reader.field(5, U64, payload) || payload.size() != 8)
    return false;
  value.wallNs = get64(payload);
  if (!reader.field(6, U64, payload) || payload.size() != 8)
    return false;
  value.peakRssBytes = get64(payload);
  if (!reader.field(7, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) != static_cast<uint32_t>(ProcessStatusV1::Success) ||
      !reader.done())
    return false;
  value.processStatus = ProcessStatusV1::Success;
  return true;
}

} // namespace

RegistryRefV1 getCurrentEquivalentIRGateRefV1() {
  return {1, 1, gateRegistryDigest()};
}

RegistryRefV1 getCurrentProductionAllOnGateRefV1() {
  return {2, 1, gateRegistryDigest()};
}

bool validateQualificationObservationV1(const QualificationObservationV1 &value,
                                        std::string *diagnostic) {
  if (value.schemaVersion != 1)
    return fail(diagnostic, "unsupported qualification observation schema");
  std::optional<AdoptionSpec> spec = lookupAdoptionSpec(value.mechanismKey);
  if (!spec || value.specDigest != digestAdoptionSpecV1(*spec))
    return fail(diagnostic, "observation does not bind an exact adoption spec");
  if (!validateIdentity(value.qualificationIdentity, diagnostic) ||
      !samePolicy(value.qualificationPolicy,
                  getCurrentOptimizationSetQualificationPolicyRefV1()) ||
      isZero(value.qualificationRunDigest) ||
      !validateWork(value.compileWorkSummary, *spec, diagnostic))
    return false;

  bool optimization = spec->adoptionMode == AdoptionMode::FixedOptimization ||
                      spec->adoptionMode == AdoptionMode::BestEffortCleanup;
  OptimizationQualificationProposal proposal =
      getCurrentOptimizationQualificationProposal();
  AdoptionDigest proposalDigest =
      digestOptimizationQualificationProposalV1(proposal);
  if (optimization) {
    if (!value.optimizationProposalDigest ||
        *value.optimizationProposalDigest != proposalDigest ||
        !value.optimizationPublicationAttemptDigest ||
        isZero(*value.optimizationPublicationAttemptDigest))
      return fail(
          diagnostic,
          "optimization observation lacks proposal/publication binding");
  } else if (value.optimizationProposalDigest ||
             value.optimizationPublicationAttemptDigest) {
    return fail(diagnostic,
                "non-optimization observation carries publication binding");
  }

  uint64_t outcomeTotal = 0;
  bool hasApplied = false;
  if (!validateOutcomeCounts(value.outcomeCounts, outcomeTotal, hasApplied,
                             diagnostic))
    return false;
  uint64_t invocationTotal = 0, rewriteTotal = 0, actionTotal = 0;
  bool allActionsSucceeded = true;
  if (!validateInvocationEvidence(value.invocationEvidence, *spec,
                                  invocationTotal, rewriteTotal, actionTotal,
                                  allActionsSucceeded, diagnostic) ||
      outcomeTotal != invocationTotal)
    return fail(diagnostic,
                "outcome and invocation aggregate counts do not close");

  switch (spec->evidenceKind) {
  case InvocationEvidenceKind::Rewrite:
    if (actionTotal != 0)
      return fail(diagnostic, "Rewrite evidence carries backend actions");
    break;
  case InvocationEvidenceKind::BackendAction:
    if (rewriteTotal != 0)
      return fail(diagnostic, "BackendAction evidence carries rewrites");
    break;
  case InvocationEvidenceKind::InvocationOnly:
    if (rewriteTotal != 0 || actionTotal != 0 || hasApplied)
      return fail(diagnostic,
                  "InvocationOnly evidence carries rewrite/action/Applied");
    break;
  }

  bool qualified =
      value.qualificationStatus == QualificationStatusV1::Qualified;
  if (qualified == static_cast<bool>(value.closedReason))
    return fail(diagnostic,
                "qualification status/reason presence is inconsistent");
  if (value.closedReason && !validateReason(*value.closedReason, diagnostic))
    return false;
  if (!validateGateBundle(proposal, value.mechanismGateResults, qualified,
                          diagnostic))
    return false;

  if (optimization) {
    OptimizationComparisonKindV1 kind =
        spec->adoptionMode == AdoptionMode::FixedOptimization
            ? OptimizationComparisonKindV1::FixedDisableOneVsAllOn
            : OptimizationComparisonKindV1::CleanupDisableOneVsAllOn;
    if (!validateComparisonDomain(value.staticComparisons, proposal, kind,
                                  value.mechanismKey, diagnostic))
      return false;
    for (const StaticComparisonEvidenceV1 &comparison : value.staticComparisons)
      if (!validateStaticComparison(proposal, comparison, diagnostic))
        return false;
    if (!validateABBA(proposal, value.staticComparisons, value.abbaSamples,
                      qualified, diagnostic))
      return false;
  } else if (!value.staticComparisons.empty() || !value.abbaSamples.empty()) {
    return fail(
        diagnostic,
        "non-optimization observation carries marginal optimization evidence");
  }

  if (!qualified)
    switch (value.qualificationStatus) {
    case QualificationStatusV1::Unassessed:
      return true;
    case QualificationStatusV1::NoOpObserved:
      if (!isGlobalClosedReasonRefV1(
              value.closedReason->reason,
              GlobalClosedReasonV1::NoDeterministicBenefit) ||
          invocationTotal == 0 ||
          value.mechanismGateResults.orderedGateResults.empty() ||
          !allGatesPassed(value.mechanismGateResults) ||
          llvm::any_of(value.staticComparisons,
                       [](const StaticComparisonEvidenceV1 &comparison) {
                         return comparison.comparisonResult !=
                                StaticComparisonResultV1::BEqual;
                       }) ||
          hasSignificantHostBenefit(value.abbaSamples))
        return fail(diagnostic,
                    "NoOpObserved invariant has benefit/failure/no invocation");
      return true;
    case QualificationStatusV1::DownstreamBlocked:
      if (!isGlobalClosedReasonRefV1(
              value.closedReason->reason,
              GlobalClosedReasonV1::DownstreamGateBlocked) ||
          !llvm::any_of(value.mechanismGateResults.orderedGateResults,
                        [](const GateEvidenceV1 &gate) {
                          return gate.status == GateStatusV1::Failed;
                        }))
        return fail(diagnostic,
                    "DownstreamBlocked lacks its failed downstream gate");
      return true;
    case QualificationStatusV1::Rejected:
      if (isGlobalClosedReasonRefV1(
              value.closedReason->reason,
              GlobalClosedReasonV1::NoDeterministicBenefit) ||
          isGlobalClosedReasonRefV1(
              value.closedReason->reason,
              GlobalClosedReasonV1::DownstreamGateBlocked))
        return fail(diagnostic,
                    "Rejected uses a reason owned by another status");
      return true;
    case QualificationStatusV1::Qualified:
      break;
    }
  if (spec->availability != Availability::Resolved ||
      spec->adoptionMode == AdoptionMode::None || invocationTotal == 0 ||
      value.mechanismGateResults.orderedGateResults.empty() ||
      !allGatesPassed(value.mechanismGateResults))
    return fail(diagnostic, "Qualified invariant lacks live passing evidence");
  switch (spec->evidenceKind) {
  case InvocationEvidenceKind::Rewrite:
    if (rewriteTotal == 0)
      return fail(diagnostic, "Qualified Rewrite evidence has zero rewrites");
    break;
  case InvocationEvidenceKind::BackendAction:
    if (actionTotal == 0 || !allActionsSucceeded)
      return fail(diagnostic,
                  "Qualified BackendAction evidence lacks successful action");
    break;
  case InvocationEvidenceKind::InvocationOnly:
    break;
  }
  if (optimization)
    for (const StaticComparisonEvidenceV1 &comparison : value.staticComparisons)
      if (comparison.comparisonResult !=
              StaticComparisonResultV1::BStrictlyBetter &&
          comparison.comparisonResult != StaticComparisonResultV1::BEqual)
        return fail(
            diagnostic,
            "Qualified optimization evidence has static regression/unknown");
  if (optimization &&
      !llvm::any_of(value.staticComparisons,
                    [](const StaticComparisonEvidenceV1 &comparison) {
                      return comparison.comparisonResult ==
                             StaticComparisonResultV1::BStrictlyBetter;
                    }) &&
      !hasSignificantHostBenefit(value.abbaSamples))
    return fail(
        diagnostic,
        "Qualified optimization mechanism has no downstream/host benefit");
  return true;
}

std::vector<uint8_t>
encodeQualificationObservationV1(const QualificationObservationV1 &value) {
  if (!validateQualificationObservationV1(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.schemaVersion));
  field(body, 2, Record, encodeMechanismKeyV1(value.mechanismKey));
  field(body, 3, Digest32, value.specDigest);
  field(body, 4, Record, encIdentity(value.qualificationIdentity));
  field(body, 5, Record, encPolicy(value.qualificationPolicy));
  field(body, 6, Sequence,
        sequence<OutcomeCountV1>(value.outcomeCounts, encOutcomeCount));
  field(
      body, 7, Sequence,
      sequence<InvocationEvidenceV1>(value.invocationEvidence, encInvocation));
  field(body, 8, Record, encGateBundle(value.mechanismGateResults));
  field(body, 9, Record, encWork(value.compileWorkSummary));
  field(body, 10, Sequence,
        sequence<StaticComparisonEvidenceV1>(value.staticComparisons,
                                             encStaticComparison));
  field(body, 11, Sequence,
        sequence<ABBASampleV1>(value.abbaSamples, encSample));
  field(body, 12, ClosedEnum,
        u32(static_cast<uint32_t>(value.qualificationStatus)));
  field(body, 13, Optional, optReason(value.closedReason));
  field(body, 14, Optional, optDigest(value.optimizationProposalDigest));
  field(body, 15, Digest32, value.qualificationRunDigest);
  field(body, 16, Optional,
        optDigest(value.optimizationPublicationAttemptDigest));
  constexpr char domain[] = "wafer.optimization-qualification-observation";
  return finish(llvm::StringRef(domain, sizeof(domain)), body);
}

AdoptionDigest
digestQualificationObservationV1(const QualificationObservationV1 &value) {
  std::vector<uint8_t> bytes = encodeQualificationObservationV1(value);
  return bytes.empty() ? AdoptionDigest{} : sha256(bytes);
}

bool decodeCanonicalQualificationObservationV1(
    const std::vector<uint8_t> &bytes, QualificationObservationV1 &value,
    std::string *diagnostic) {
  constexpr char domain[] = "wafer.optimization-qualification-observation";
  llvm::ArrayRef<uint8_t> body;
  if (!parseEnvelope(bytes, llvm::StringRef(domain, sizeof(domain)), body,
                     diagnostic))
    return false;
  Reader reader(body, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  QualificationObservationV1 parsed;
  if (!reader.field(1, U16, payload) || payload.size() != 2 ||
      get16(payload) != 1 || !reader.field(2, Record, payload) ||
      !parseMechanism(payload, parsed.mechanismKey, diagnostic) ||
      !reader.field(3, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), parsed.specDigest.begin());
  if (!reader.field(4, Record, payload) ||
      !parseIdentity(payload, parsed.qualificationIdentity, diagnostic) ||
      !reader.field(5, Record, payload) ||
      !parsePolicy(payload, parsed.qualificationPolicy, diagnostic) ||
      !reader.field(6, Sequence, payload) ||
      !parseSequence(payload, parsed.outcomeCounts, parseOutcome, diagnostic) ||
      !reader.field(7, Sequence, payload) ||
      !parseSequence(payload, parsed.invocationEvidence, parseInvocation,
                     diagnostic) ||
      !reader.field(8, Record, payload) ||
      !parseGateBundle(payload, parsed.mechanismGateResults, diagnostic) ||
      !reader.field(9, Record, payload) ||
      !parseWork(payload, parsed.compileWorkSummary, diagnostic) ||
      !reader.field(10, Sequence, payload) ||
      !parseSequence(payload, parsed.staticComparisons, parseStaticComparison,
                     diagnostic) ||
      !reader.field(11, Sequence, payload) ||
      !parseSequence(payload, parsed.abbaSamples, parseSample, diagnostic) ||
      !reader.field(12, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) > static_cast<uint32_t>(QualificationStatusV1::Rejected))
    return false;
  parsed.qualificationStatus =
      static_cast<QualificationStatusV1>(get32(payload));
  if (!reader.field(13, Optional, payload) ||
      !parseOptionalReason(payload, parsed.closedReason, diagnostic) ||
      !reader.field(14, Optional, payload) ||
      !parseOptionalDigest(payload, parsed.optimizationProposalDigest,
                           diagnostic) ||
      !reader.field(15, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(),
            parsed.qualificationRunDigest.begin());
  if (!reader.field(16, Optional, payload) ||
      !parseOptionalDigest(payload, parsed.optimizationPublicationAttemptDigest,
                           diagnostic) ||
      !reader.done() ||
      !validateQualificationObservationV1(parsed, diagnostic) ||
      encodeQualificationObservationV1(parsed) != bytes)
    return false;
  value = std::move(parsed);
  return true;
}

bool validateOptimizationBatchObservationV1(
    const OptimizationBatchObservationV1 &value, std::string *diagnostic) {
  if (value.schemaVersion != 1)
    return fail(diagnostic, "unsupported optimization batch schema");
  OptimizationQualificationProposal proposal =
      getCurrentOptimizationQualificationProposal();
  if (value.proposalDigest !=
          digestOptimizationQualificationProposalV1(proposal) ||
      !validateIdentity(value.qualificationIdentity, diagnostic) ||
      !samePolicy(value.qualificationPolicy,
                  getCurrentOptimizationSetQualificationPolicyRefV1()) ||
      isZero(value.qualificationRunDigest) ||
      isZero(value.optimizationPublicationAttemptDigest))
    return fail(diagnostic,
                "optimization batch identity/digest binding is invalid");
  bool qualified = value.batchStatus == OptimizationBatchStatusV1::Qualified;
  if (qualified == static_cast<bool>(value.closedReason))
    return fail(diagnostic, "batch status/reason presence is inconsistent");
  if (value.closedReason && !validateReason(*value.closedReason, diagnostic))
    return false;

  if (!validateComparisonDomain(
          value.globalStaticComparisons, proposal,
          OptimizationComparisonKindV1::GlobalAllOffVsAllOn, std::nullopt,
          diagnostic))
    return false;
  for (const StaticComparisonEvidenceV1 &comparison :
       value.globalStaticComparisons)
    if (!validateStaticComparison(proposal, comparison, diagnostic))
      return false;
  if (!validateABBA(proposal, value.globalStaticComparisons,
                    value.globalABBASamples, qualified, diagnostic) ||
      !validateGateBundle(proposal, value.equivalentIR2x2GateResults, qualified,
                          diagnostic) ||
      !validateGateBundle(proposal, value.productionAllOnGateResults, qualified,
                          diagnostic))
    return false;

  std::vector<GateEvidenceV1> expectedEquivalent;
  for (const QualificationCaseKeyV1 &mandatoryCase :
       getCurrentMandatoryQualificationCasesV1())
    for (EquivalentInputVariantV1 variant :
         {EquivalentInputVariantV1::Original,
          EquivalentInputVariantV1::Metamorphic})
      for (bool cleanupEnabled : {false, true}) {
        GateEvidenceV1 gate;
        gate.gate = getCurrentEquivalentIRGateRefV1();
        gate.caseKey = mandatoryCase;
        gate.caseKey.inputVariant = variant;
        gate.configuration = equivalentConfiguration(cleanupEnabled);
        expectedEquivalent.push_back(std::move(gate));
      }
  std::sort(expectedEquivalent.begin(), expectedEquivalent.end(),
            [](const GateEvidenceV1 &lhs, const GateEvidenceV1 &rhs) {
              return encGateKey(lhs) < encGateKey(rhs);
            });
  if (value.equivalentIR2x2GateResults.orderedGateResults.size() !=
      expectedEquivalent.size())
    return fail(diagnostic, "Equivalent-IR gate domain is incomplete");
  for (size_t index = 0; index < expectedEquivalent.size(); ++index) {
    const GateEvidenceV1 &actual =
        value.equivalentIR2x2GateResults.orderedGateResults[index];
    if (!sameRegistry(actual.gate, expectedEquivalent[index].gate) ||
        !sameCase(actual.caseKey, expectedEquivalent[index].caseKey) ||
        !sameConfiguration(actual.configuration,
                           expectedEquivalent[index].configuration))
      return fail(diagnostic, "Equivalent-IR gate rows are not all-and-only");
  }

  std::vector<GateEvidenceV1> expectedProduction;
  for (const QualificationCaseKeyV1 &mandatoryCase :
       getCurrentMandatoryQualificationCasesV1()) {
    GateEvidenceV1 gate;
    gate.gate = getCurrentProductionAllOnGateRefV1();
    gate.caseKey = mandatoryCase;
    gate.configuration = allOnConfiguration();
    expectedProduction.push_back(std::move(gate));
  }
  std::sort(expectedProduction.begin(), expectedProduction.end(),
            [](const GateEvidenceV1 &lhs, const GateEvidenceV1 &rhs) {
              return encGateKey(lhs) < encGateKey(rhs);
            });
  if (value.productionAllOnGateResults.orderedGateResults.size() !=
      expectedProduction.size())
    return fail(diagnostic, "production-AllOn gate domain is incomplete");
  for (size_t index = 0; index < expectedProduction.size(); ++index) {
    const GateEvidenceV1 &actual =
        value.productionAllOnGateResults.orderedGateResults[index];
    if (!sameRegistry(actual.gate, expectedProduction[index].gate) ||
        !sameCase(actual.caseKey, expectedProduction[index].caseKey) ||
        !sameConfiguration(actual.configuration,
                           expectedProduction[index].configuration))
      return fail(diagnostic,
                  "production-AllOn gate rows are not all-and-only");
  }

  if (qualified) {
    for (const StaticComparisonEvidenceV1 &comparison :
         value.globalStaticComparisons)
      if (comparison.comparisonResult !=
              StaticComparisonResultV1::BStrictlyBetter &&
          comparison.comparisonResult != StaticComparisonResultV1::BEqual)
        return fail(diagnostic,
                    "Qualified batch has static regression or unknown metric");
  }
  return true;
}

std::vector<uint8_t> encodeOptimizationBatchObservationV1(
    const OptimizationBatchObservationV1 &value) {
  if (!validateOptimizationBatchObservationV1(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  field(body, 1, U16, u16(value.schemaVersion));
  field(body, 2, Digest32, value.proposalDigest);
  field(body, 3, Record, encIdentity(value.qualificationIdentity));
  field(body, 4, Record, encPolicy(value.qualificationPolicy));
  field(body, 5, Sequence,
        sequence<StaticComparisonEvidenceV1>(value.globalStaticComparisons,
                                             encStaticComparison));
  field(body, 6, Sequence,
        sequence<ABBASampleV1>(value.globalABBASamples, encSample));
  field(body, 7, Record, encGateBundle(value.equivalentIR2x2GateResults));
  field(body, 8, Record, encGateBundle(value.productionAllOnGateResults));
  field(body, 9, ClosedEnum, u32(static_cast<uint32_t>(value.batchStatus)));
  field(body, 10, Optional, optReason(value.closedReason));
  field(body, 11, Digest32, value.qualificationRunDigest);
  field(body, 12, Digest32, value.optimizationPublicationAttemptDigest);
  constexpr char domain[] = "wafer.optimization-batch-observation";
  return finish(llvm::StringRef(domain, sizeof(domain)), body);
}

AdoptionDigest digestOptimizationBatchObservationV1(
    const OptimizationBatchObservationV1 &value) {
  std::vector<uint8_t> bytes = encodeOptimizationBatchObservationV1(value);
  return bytes.empty() ? AdoptionDigest{} : sha256(bytes);
}

bool decodeCanonicalOptimizationBatchObservationV1(
    const std::vector<uint8_t> &bytes, OptimizationBatchObservationV1 &value,
    std::string *diagnostic) {
  constexpr char domain[] = "wafer.optimization-batch-observation";
  llvm::ArrayRef<uint8_t> body;
  if (!parseEnvelope(bytes, llvm::StringRef(domain, sizeof(domain)), body,
                     diagnostic))
    return false;
  Reader reader(body, diagnostic);
  llvm::ArrayRef<uint8_t> payload;
  OptimizationBatchObservationV1 parsed;
  if (!reader.field(1, U16, payload) || payload.size() != 2 ||
      get16(payload) != 1 || !reader.field(2, Digest32, payload) ||
      payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(), parsed.proposalDigest.begin());
  if (!reader.field(3, Record, payload) ||
      !parseIdentity(payload, parsed.qualificationIdentity, diagnostic) ||
      !reader.field(4, Record, payload) ||
      !parsePolicy(payload, parsed.qualificationPolicy, diagnostic) ||
      !reader.field(5, Sequence, payload) ||
      !parseSequence(payload, parsed.globalStaticComparisons,
                     parseStaticComparison, diagnostic) ||
      !reader.field(6, Sequence, payload) ||
      !parseSequence(payload, parsed.globalABBASamples, parseSample,
                     diagnostic) ||
      !reader.field(7, Record, payload) ||
      !parseGateBundle(payload, parsed.equivalentIR2x2GateResults,
                       diagnostic) ||
      !reader.field(8, Record, payload) ||
      !parseGateBundle(payload, parsed.productionAllOnGateResults,
                       diagnostic) ||
      !reader.field(9, ClosedEnum, payload) || payload.size() != 4 ||
      get32(payload) >
          static_cast<uint32_t>(OptimizationBatchStatusV1::Rejected))
    return false;
  parsed.batchStatus = static_cast<OptimizationBatchStatusV1>(get32(payload));
  if (!reader.field(10, Optional, payload) ||
      !parseOptionalReason(payload, parsed.closedReason, diagnostic) ||
      !reader.field(11, Digest32, payload) || payload.size() != 32)
    return false;
  std::copy(payload.begin(), payload.end(),
            parsed.qualificationRunDigest.begin());
  if (!reader.field(12, Digest32, payload) || payload.size() != 32 ||
      !reader.done())
    return false;
  std::copy(payload.begin(), payload.end(),
            parsed.optimizationPublicationAttemptDigest.begin());
  if (!validateOptimizationBatchObservationV1(parsed, diagnostic) ||
      encodeOptimizationBatchObservationV1(parsed) != bytes)
    return false;
  value = std::move(parsed);
  return true;
}

} // namespace wafer
