//===- OptimizationQualificationExecution.cpp - Isolated result codec ---===//

#include "Wafer/Support/OptimizationQualificationExecution.h"

#include "Wafer/Support/OptimizationArtifactDigest.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace wafer {
namespace {

constexpr char domain[] = "wafer.optimization-qualification-execution";

static bool fail(std::string *diagnostic, const char *message) {
  if (diagnostic)
    *diagnostic = message;
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

static void putDigest(std::vector<uint8_t> &bytes,
                      const AdoptionDigest &digest) {
  bytes.insert(bytes.end(), digest.begin(), digest.end());
}

static bool take(llvm::ArrayRef<uint8_t> &bytes, size_t count,
                 llvm::ArrayRef<uint8_t> &result, std::string *diagnostic) {
  if (count > bytes.size())
    return fail(diagnostic, "isolated execution result is truncated");
  result = bytes.take_front(count);
  bytes = bytes.drop_front(count);
  return true;
}

static bool takeDigest(llvm::ArrayRef<uint8_t> &bytes, AdoptionDigest &digest,
                       std::string *diagnostic) {
  llvm::ArrayRef<uint8_t> payload;
  if (!take(bytes, digest.size(), payload, diagnostic))
    return false;
  std::copy(payload.begin(), payload.end(), digest.begin());
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

static bool isMandatoryCase(const QualificationCaseKeyV1 &key) {
  if (key.inputVariant != EquivalentInputVariantV1::Original &&
      key.inputVariant != EquivalentInputVariantV1::Metamorphic)
    return false;
  return llvm::any_of(getCurrentMandatoryQualificationCasesV1(),
                      [&](const QualificationCaseKeyV1 &candidate) {
                        return sameRegistry(candidate.corpus, key.corpus) &&
                               candidate.rankCount == key.rankCount;
                      });
}

static bool convertConfiguration(const OptimizationConfigurationV1 &source,
                                 OptimizationConfiguration &target,
                                 std::string *diagnostic) {
  auto convert = [](OptimizationGroupSelectionV1 value)
      -> std::optional<OptimizationGroupSelectionKind> {
    switch (value) {
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
      convert(source.fixedSelection);
  std::optional<OptimizationGroupSelectionKind> cleanup =
      convert(source.cleanupSelection);
  if (!fixed || !cleanup)
    return fail(diagnostic, "isolated execution configuration enum is invalid");
  target = {{*fixed, source.fixedDisabledKey},
            {*cleanup, source.cleanupDisabledKey}};
  return validateOptimizationConfiguration(
      getCurrentOptimizationQualificationProposal(), target, diagnostic);
}

static bool validateStaticMetrics(const ExactStaticVectorEvidenceV1 &metrics,
                                  std::string *diagnostic) {
  RegistryRefV1 registry = getCurrentStaticMetricRegistryRefV1();
  if (metrics.registrySchema != registry.registrySchema ||
      metrics.registryDigest != registry.registryDigest ||
      metrics.orderedMetrics.size() != 4)
    return fail(diagnostic,
                "isolated execution static metrics do not bind the registry");
  for (size_t index = 0; index < metrics.orderedMetrics.size(); ++index)
    if (metrics.orderedMetrics[index].metricId != index + 1)
      return fail(diagnostic,
                  "isolated execution static metrics are not canonical");
  return true;
}

static void putOptionalKey(std::vector<uint8_t> &bytes,
                           const std::optional<MechanismKey> &key) {
  bytes.push_back(key ? 1 : 0);
  if (key)
    put32(bytes, key->semanticId);
}

static bool takeOptionalKey(llvm::ArrayRef<uint8_t> &bytes,
                            std::optional<MechanismKey> &key,
                            std::string *diagnostic) {
  llvm::ArrayRef<uint8_t> marker;
  if (!take(bytes, 1, marker, diagnostic) || marker[0] > 1)
    return fail(diagnostic, "isolated execution optional key is invalid");
  if (marker[0] == 0) {
    key.reset();
    return true;
  }
  llvm::ArrayRef<uint8_t> payload;
  if (!take(bytes, 4, payload, diagnostic))
    return false;
  key = MechanismKey{get32(payload)};
  return lookupMechanismDescriptor(*key).has_value() ||
         fail(diagnostic, "isolated execution mechanism key is unknown");
}

} // namespace

bool validateOptimizationQualificationExecutionResultV1(
    const OptimizationQualificationExecutionResultV1 &value,
    std::string *diagnostic) {
  if (value.schemaVersion != 1 || isZero(value.qualificationRunDigest) ||
      isZero(value.inputSnapshotDigest) || isZero(value.outputArtifactDigest) ||
      (value.targetModelEvidenceDigest &&
       isZero(*value.targetModelEvidenceDigest)))
    return fail(diagnostic,
                "isolated execution identity/artifact digest is invalid");
  if (!isMandatoryCase(value.caseKey))
    return fail(diagnostic, "isolated execution case is not mandatory");
  OptimizationConfiguration converted;
  if (!convertConfiguration(value.configuration, converted, diagnostic))
    return false;
  if (value.requiredTensorNormalizationRepetitions == 0 ||
      value.requiredTensorNormalizationRepetitions > 2)
    return fail(diagnostic,
                "isolated execution normalization repetition is invalid");
  if (!validateStaticMetrics(value.staticMetrics, diagnostic) ||
      value.invocationTerminals.empty())
    return false;
  AdoptionDigest previous{};
  bool havePrevious = false;
  for (const InvocationTelemetryV1 &terminal : value.invocationTerminals) {
    if (!validateInvocationTelemetryV1(terminal, diagnostic) ||
        terminal.identity.scopeKind !=
            InvocationScopeKindV1::QualificationRun ||
        terminal.identity.scopeDigest != value.qualificationRunDigest ||
        !terminal.qualificationCase ||
        !sameCase(*terminal.qualificationCase, value.caseKey) ||
        isZero(terminal.inputSnapshotDigest))
      return fail(diagnostic,
                  "isolated execution terminal scope/case/stage mismatch");
    AdoptionDigest id = digestInvocationIdentityV1(terminal.identity);
    if (havePrevious && !(previous < id))
      return fail(diagnostic,
                  "isolated execution terminals are not sorted and unique");
    previous = id;
    havePrevious = true;
  }
  return true;
}

std::vector<uint8_t> encodeOptimizationQualificationExecutionResultV1(
    const OptimizationQualificationExecutionResultV1 &value) {
  if (!validateOptimizationQualificationExecutionResultV1(value, nullptr))
    return {};
  std::vector<uint8_t> body;
  put16(body, value.schemaVersion);
  putDigest(body, value.qualificationRunDigest);
  putDigest(body, value.inputSnapshotDigest);
  put32(body, value.caseKey.corpus.id);
  put16(body, value.caseKey.corpus.registrySchema);
  putDigest(body, value.caseKey.corpus.registryDigest);
  put32(body, value.caseKey.rankCount);
  put32(body, static_cast<uint32_t>(value.caseKey.inputVariant));
  put32(body, static_cast<uint32_t>(value.configuration.fixedSelection));
  putOptionalKey(body, value.configuration.fixedDisabledKey);
  put32(body, static_cast<uint32_t>(value.configuration.cleanupSelection));
  putOptionalKey(body, value.configuration.cleanupDisabledKey);
  put32(body, value.requiredTensorNormalizationRepetitions);
  putDigest(body, value.outputArtifactDigest);
  put16(body, value.staticMetrics.registrySchema);
  putDigest(body, value.staticMetrics.registryDigest);
  put32(body, value.staticMetrics.orderedMetrics.size());
  for (const MetricEvidenceV1 &metric : value.staticMetrics.orderedMetrics) {
    put32(body, metric.metricId);
    body.push_back(metric.value ? 1 : 0);
    if (metric.value)
      put64(body, *metric.value);
  }
  body.push_back(value.targetModelEvidenceDigest ? 1 : 0);
  if (value.targetModelEvidenceDigest)
    putDigest(body, *value.targetModelEvidenceDigest);
  put32(body, value.invocationTerminals.size());
  for (const InvocationTelemetryV1 &terminal : value.invocationTerminals) {
    std::vector<uint8_t> bytes = encodeInvocationTelemetryV1(terminal);
    put32(body, bytes.size());
    body.insert(body.end(), bytes.begin(), bytes.end());
  }

  std::vector<uint8_t> bytes(std::begin(domain), std::end(domain));
  put16(bytes, 1);
  put32(bytes, body.size());
  bytes.insert(bytes.end(), body.begin(), body.end());
  return bytes;
}

AdoptionDigest digestOptimizationQualificationExecutionResultV1(
    const OptimizationQualificationExecutionResultV1 &value) {
  std::vector<uint8_t> bytes =
      encodeOptimizationQualificationExecutionResultV1(value);
  return bytes.empty()
             ? AdoptionDigest{}
             : digestOptimizationBytesV1(
                   "wafer.optimization-qualification-execution-result-v1",
                   bytes);
}

bool decodeCanonicalOptimizationQualificationExecutionResultV1(
    const std::vector<uint8_t> &bytes,
    OptimizationQualificationExecutionResultV1 &value,
    std::string *diagnostic) {
  constexpr size_t envelopeSize = sizeof(domain) + 2 + 4;
  if (bytes.size() < envelopeSize ||
      !std::equal(std::begin(domain), std::end(domain), bytes.begin()) ||
      get16(llvm::ArrayRef<uint8_t>(bytes).slice(sizeof(domain), 2)) != 1 ||
      get32(llvm::ArrayRef<uint8_t>(bytes).slice(sizeof(domain) + 2, 4)) !=
          bytes.size() - envelopeSize)
    return fail(diagnostic, "isolated execution envelope is invalid");
  llvm::ArrayRef<uint8_t> input(bytes);
  input = input.drop_front(envelopeSize);
  llvm::ArrayRef<uint8_t> payload;
  OptimizationQualificationExecutionResultV1 parsed;
  if (!take(input, 2, payload, diagnostic))
    return false;
  parsed.schemaVersion = get16(payload);
  if (!takeDigest(input, parsed.qualificationRunDigest, diagnostic) ||
      !takeDigest(input, parsed.inputSnapshotDigest, diagnostic) ||
      !take(input, 4, payload, diagnostic))
    return false;
  parsed.caseKey.corpus.id = get32(payload);
  if (!take(input, 2, payload, diagnostic))
    return false;
  parsed.caseKey.corpus.registrySchema = get16(payload);
  if (!takeDigest(input, parsed.caseKey.corpus.registryDigest, diagnostic) ||
      !take(input, 4, payload, diagnostic))
    return false;
  parsed.caseKey.rankCount = get32(payload);
  if (!take(input, 4, payload, diagnostic) ||
      get32(payload) >
          static_cast<uint32_t>(EquivalentInputVariantV1::Metamorphic))
    return fail(diagnostic, "isolated execution input variant is invalid");
  parsed.caseKey.inputVariant =
      static_cast<EquivalentInputVariantV1>(get32(payload));
  if (!take(input, 4, payload, diagnostic) ||
      get32(payload) >
          static_cast<uint32_t>(OptimizationGroupSelectionV1::DisableOne))
    return fail(diagnostic, "isolated execution fixed selection is invalid");
  parsed.configuration.fixedSelection =
      static_cast<OptimizationGroupSelectionV1>(get32(payload));
  if (!takeOptionalKey(input, parsed.configuration.fixedDisabledKey,
                       diagnostic) ||
      !take(input, 4, payload, diagnostic) ||
      get32(payload) >
          static_cast<uint32_t>(OptimizationGroupSelectionV1::DisableOne))
    return fail(diagnostic, "isolated execution cleanup selection is invalid");
  parsed.configuration.cleanupSelection =
      static_cast<OptimizationGroupSelectionV1>(get32(payload));
  if (!takeOptionalKey(input, parsed.configuration.cleanupDisabledKey,
                       diagnostic) ||
      !take(input, 4, payload, diagnostic))
    return false;
  parsed.requiredTensorNormalizationRepetitions = get32(payload);
  if (!takeDigest(input, parsed.outputArtifactDigest, diagnostic) ||
      !take(input, 2, payload, diagnostic))
    return false;
  parsed.staticMetrics.registrySchema = get16(payload);
  if (!takeDigest(input, parsed.staticMetrics.registryDigest, diagnostic) ||
      !take(input, 4, payload, diagnostic))
    return false;
  uint32_t metricCount = get32(payload);
  if (metricCount > 1024)
    return fail(diagnostic, "isolated execution metric count is excessive");
  for (uint32_t index = 0; index < metricCount; ++index) {
    MetricEvidenceV1 metric;
    if (!take(input, 4, payload, diagnostic))
      return false;
    metric.metricId = get32(payload);
    if (!take(input, 1, payload, diagnostic) || payload[0] > 1)
      return fail(diagnostic, "isolated execution optional metric is invalid");
    if (payload[0]) {
      if (!take(input, 8, payload, diagnostic))
        return false;
      metric.value = get64(payload);
    }
    parsed.staticMetrics.orderedMetrics.push_back(metric);
  }
  if (!take(input, 1, payload, diagnostic) || payload[0] > 1)
    return fail(diagnostic,
                "isolated execution optional model digest is invalid");
  if (payload[0]) {
    AdoptionDigest digest;
    if (!takeDigest(input, digest, diagnostic))
      return false;
    parsed.targetModelEvidenceDigest = digest;
  }
  if (!take(input, 4, payload, diagnostic))
    return false;
  uint32_t terminalCount = get32(payload);
  if (terminalCount > 1000000)
    return fail(diagnostic, "isolated execution terminal count is excessive");
  parsed.invocationTerminals.reserve(terminalCount);
  for (uint32_t index = 0; index < terminalCount; ++index) {
    if (!take(input, 4, payload, diagnostic))
      return false;
    uint32_t size = get32(payload);
    if (!take(input, size, payload, diagnostic))
      return false;
    std::vector<uint8_t> terminalBytes(payload.begin(), payload.end());
    InvocationTelemetryV1 terminal;
    if (!decodeCanonicalInvocationTelemetryV1(terminalBytes, terminal,
                                              diagnostic))
      return false;
    parsed.invocationTerminals.push_back(std::move(terminal));
  }
  if (!input.empty() ||
      !validateOptimizationQualificationExecutionResultV1(parsed, diagnostic) ||
      encodeOptimizationQualificationExecutionResultV1(parsed) != bytes)
    return fail(diagnostic,
                "isolated execution result is not canonical all-and-only");
  value = std::move(parsed);
  return true;
}

} // namespace wafer
