//===- OptimizationInvocation.cpp - Canonical invocation records --------===//

#include "Wafer/Support/OptimizationInvocation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>

namespace wafer {
namespace {

enum TypeTag : uint8_t {
  Bool = 0x01,
  U16 = 0x02,
  U32 = 0x03,
  U64 = 0x04,
  ClosedEnum = 0x05,
  TypedId = 0x06,
  Digest32 = 0x07,
  Bytes = 0x08,
  Record = 0x09,
  Sequence = 0x0a,
  SortedSet = 0x0b,
  Optional = 0x0c,
};

static bool fail(std::string *diagnostic, const char *message) {
  if (diagnostic)
    *diagnostic = message;
  return false;
}

static void appendU16(std::vector<uint8_t> &bytes, uint16_t value) {
  bytes.push_back(static_cast<uint8_t>(value >> 8));
  bytes.push_back(static_cast<uint8_t>(value));
}

static void appendU32(std::vector<uint8_t> &bytes, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static void appendU64(std::vector<uint8_t> &bytes, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static std::vector<uint8_t> u16(uint16_t value) {
  std::vector<uint8_t> result;
  appendU16(result, value);
  return result;
}

static std::vector<uint8_t> u32(uint32_t value) {
  std::vector<uint8_t> result;
  appendU32(result, value);
  return result;
}

static std::vector<uint8_t> u64(uint64_t value) {
  std::vector<uint8_t> result;
  appendU64(result, value);
  return result;
}

static void appendField(std::vector<uint8_t> &body, uint16_t number,
                        TypeTag tag, llvm::ArrayRef<uint8_t> payload) {
  appendU16(body, number);
  body.push_back(tag);
  appendU32(body, payload.size());
  body.insert(body.end(), payload.begin(), payload.end());
}

static void appendElement(std::vector<uint8_t> &sequence,
                          llvm::ArrayRef<uint8_t> element) {
  appendU32(sequence, element.size());
  sequence.insert(sequence.end(), element.begin(), element.end());
}

static AdoptionDigest sha256(llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 hash;
  hash.update(bytes);
  return hash.final();
}

static AdoptionDigest invocationSiteRegistryDigest() {
  std::vector<uint8_t> bytes;
  constexpr char domain[] = "wafer.optimization-invocation-site-registry";
  bytes.insert(bytes.end(), domain, domain + sizeof(domain));
  appendU16(bytes, 1);
  std::vector<MechanismDescriptor> descriptors = getAllMechanismDescriptors();
  appendU32(bytes, descriptors.size());
  for (const MechanismDescriptor &descriptor : descriptors) {
    appendU32(bytes, descriptor.key.semanticId);
    std::vector<uint8_t> key = encodeMechanismKeyV1(descriptor.key);
    appendU32(bytes, key.size());
    bytes.insert(bytes.end(), key.begin(), key.end());
    appendU32(bytes, static_cast<uint32_t>(descriptor.cutPoint));
  }
  return sha256(bytes);
}

static AdoptionDigest closedReasonRegistryDigest() {
  std::vector<uint8_t> bytes;
  constexpr char domain[] = "wafer.global-closed-reason-registry";
  bytes.insert(bytes.end(), domain, domain + sizeof(domain));
  appendU16(bytes, 1);
  constexpr std::array<std::pair<uint32_t, llvm::StringLiteral>, 10> rows = {{
      {1, "unsupported-semantic"},
      {2, "resource-exhausted"},
      {3, "invalid-owner-terminal"},
      {4, "cancelled"},
      {5, "no-deterministic-benefit"},
      {6, "downstream-gate-blocked"},
      {7, "qualification-evidence-rejected"},
      {8, "host-environment-invalidated"},
      {9, "publication-conflict"},
      {10, "publication-failure"},
  }};
  appendU32(bytes, rows.size());
  for (const auto &row : rows) {
    appendU32(bytes, row.first);
    appendU32(bytes, row.second.size());
    bytes.insert(bytes.end(), row.second.bytes_begin(), row.second.bytes_end());
  }
  return sha256(bytes);
}

static bool isZeroDigest(const AdoptionDigest &digest) {
  return llvm::all_of(digest, [](uint8_t byte) { return byte == 0; });
}

static std::vector<uint8_t> encodeRegistryRef(const RegistryRefV1 &ref) {
  std::vector<uint8_t> body;
  appendField(body, 1, U32, u32(ref.id));
  appendField(body, 2, U16, u16(ref.registrySchema));
  appendField(body, 3, Digest32, ref.registryDigest);
  return body;
}

static std::vector<uint8_t>
encodeQualificationCase(const QualificationCaseKeyV1 &key) {
  std::vector<uint8_t> body;
  appendField(body, 1, Record, encodeRegistryRef(key.corpus));
  appendField(body, 2, U32, u32(key.rankCount));
  appendField(body, 3, ClosedEnum,
              u32(static_cast<uint32_t>(key.inputVariant)));
  return body;
}

static std::vector<uint8_t>
encodeOptionalCase(const std::optional<QualificationCaseKeyV1> &value) {
  if (!value)
    return {0};
  std::vector<uint8_t> payload = {1, Record};
  std::vector<uint8_t> inner = encodeQualificationCase(*value);
  appendU32(payload, inner.size());
  payload.insert(payload.end(), inner.begin(), inner.end());
  return payload;
}

static std::vector<uint8_t> encodeClosedReason(const ClosedReasonV1 &reason) {
  std::vector<uint8_t> body;
  appendField(body, 1, Record, encodeRegistryRef(reason.reason));
  std::vector<uint8_t> detail = {
      static_cast<uint8_t>(reason.detailDigest ? 1 : 0)};
  if (reason.detailDigest) {
    detail.push_back(Digest32);
    appendU32(detail, reason.detailDigest->size());
    detail.insert(detail.end(), reason.detailDigest->begin(),
                  reason.detailDigest->end());
  }
  appendField(body, 2, Optional, detail);
  return body;
}

static std::vector<uint8_t>
encodeOptionalReason(const std::optional<ClosedReasonV1> &value) {
  if (!value)
    return {0};
  std::vector<uint8_t> payload = {1, Record};
  std::vector<uint8_t> inner = encodeClosedReason(*value);
  appendU32(payload, inner.size());
  payload.insert(payload.end(), inner.begin(), inner.end());
  return payload;
}

static std::vector<uint8_t> encodeWorkCounter(const WorkCounterV1 &counter) {
  std::vector<uint8_t> body;
  appendField(body, 1, U32, u32(counter.counterId));
  appendField(body, 2, U64, u64(counter.value));
  return body;
}

static std::vector<uint8_t> encodeWorkSummary(const WorkSummaryV1 &summary) {
  std::vector<uint8_t> body;
  appendField(body, 1, Digest32, summary.workPolicyDigest);
  std::vector<uint8_t> counters;
  appendU32(counters, summary.orderedCounters.size());
  for (const WorkCounterV1 &counter : summary.orderedCounters)
    appendElement(counters, encodeWorkCounter(counter));
  appendField(body, 2, Sequence, counters);
  return body;
}

static std::vector<uint8_t> encodeRawBytes(llvm::StringRef value) {
  std::vector<uint8_t> bytes;
  appendU32(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
  return bytes;
}

static std::vector<uint8_t>
encodeBackendAction(const BackendActionEvidenceV1 &action) {
  std::vector<uint8_t> body;
  appendField(body, 1, Digest32, action.invocationId);
  appendField(body, 2, U32, u32(action.actionOrdinal));
  std::vector<uint8_t> argv;
  appendU32(argv, action.argv.size());
  for (const std::string &arg : action.argv)
    appendElement(argv, encodeRawBytes(arg));
  appendField(body, 3, Sequence, argv);
  appendField(body, 4, Digest32, action.observedToolDigest);
  appendField(body, 5, Digest32, action.observedOutputDigest);
  appendField(body, 6, ClosedEnum,
              u32(static_cast<uint32_t>(action.terminalStatus)));
  return body;
}

static std::vector<uint8_t> makeStandalone(llvm::StringRef domain,
                                           uint16_t schema,
                                           llvm::ArrayRef<uint8_t> body) {
  std::vector<uint8_t> bytes(domain.bytes_begin(), domain.bytes_end());
  bytes.push_back(0);
  appendU16(bytes, schema);
  appendU32(bytes, body.size());
  bytes.insert(bytes.end(), body.begin(), body.end());
  return bytes;
}

struct Reader {
  llvm::ArrayRef<uint8_t> bytes;
  size_t cursor = 0;
  std::string *diagnostic = nullptr;

  bool takeU8(uint8_t &value) {
    if (cursor == bytes.size())
      return fail(diagnostic, "truncated canonical record");
    value = bytes[cursor++];
    return true;
  }
  bool takeU16(uint16_t &value) {
    if (bytes.size() - cursor < 2)
      return fail(diagnostic, "truncated canonical u16");
    value = static_cast<uint16_t>((bytes[cursor] << 8) | bytes[cursor + 1]);
    cursor += 2;
    return true;
  }
  bool takeU32(uint32_t &value) {
    if (bytes.size() - cursor < 4)
      return fail(diagnostic, "truncated canonical u32");
    value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes[cursor++];
    return true;
  }
  bool takeU64(uint64_t &value) {
    if (bytes.size() - cursor < 8)
      return fail(diagnostic, "truncated canonical u64");
    value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes[cursor++];
    return true;
  }
  bool takeBytes(size_t size, llvm::ArrayRef<uint8_t> &value) {
    if (size > bytes.size() - cursor)
      return fail(diagnostic, "truncated canonical byte payload");
    value = bytes.slice(cursor, size);
    cursor += size;
    return true;
  }
  bool takeField(uint16_t expectedNumber, TypeTag expectedTag,
                 llvm::ArrayRef<uint8_t> &payload) {
    uint16_t number;
    uint8_t tag;
    uint32_t size;
    if (!takeU16(number) || !takeU8(tag) || !takeU32(size))
      return false;
    if (number != expectedNumber || tag != expectedTag)
      return fail(diagnostic,
                  "unknown, missing, reordered or mistyped canonical field");
    return takeBytes(size, payload);
  }
  bool done() const { return cursor == bytes.size(); }
};

static bool parseU16Payload(llvm::ArrayRef<uint8_t> payload, uint16_t &value,
                            std::string *diagnostic) {
  Reader reader{payload, 0, diagnostic};
  return reader.takeU16(value) && reader.done();
}

static bool parseU32Payload(llvm::ArrayRef<uint8_t> payload, uint32_t &value,
                            std::string *diagnostic) {
  Reader reader{payload, 0, diagnostic};
  return reader.takeU32(value) && reader.done();
}

static bool parseU64Payload(llvm::ArrayRef<uint8_t> payload, uint64_t &value,
                            std::string *diagnostic) {
  Reader reader{payload, 0, diagnostic};
  return reader.takeU64(value) && reader.done();
}

static bool parseDigest(llvm::ArrayRef<uint8_t> payload, AdoptionDigest &value,
                        std::string *diagnostic) {
  if (payload.size() != value.size())
    return fail(diagnostic, "canonical digest is not 32 bytes");
  std::copy(payload.begin(), payload.end(), value.begin());
  return true;
}

static bool parseRegistryRef(llvm::ArrayRef<uint8_t> bytes, RegistryRefV1 &ref,
                             std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.takeField(1, U32, payload) ||
      !parseU32Payload(payload, ref.id, diagnostic) ||
      !reader.takeField(2, U16, payload) ||
      !parseU16Payload(payload, ref.registrySchema, diagnostic) ||
      !reader.takeField(3, Digest32, payload) ||
      !parseDigest(payload, ref.registryDigest, diagnostic) || !reader.done())
    return fail(diagnostic, "invalid canonical RegistryRefV1");
  return true;
}

static bool parseQualificationCase(llvm::ArrayRef<uint8_t> bytes,
                                   QualificationCaseKeyV1 &key,
                                   std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  llvm::ArrayRef<uint8_t> payload;
  uint32_t variant;
  if (!reader.takeField(1, Record, payload) ||
      !parseRegistryRef(payload, key.corpus, diagnostic) ||
      !reader.takeField(2, U32, payload) ||
      !parseU32Payload(payload, key.rankCount, diagnostic) ||
      !reader.takeField(3, ClosedEnum, payload) ||
      !parseU32Payload(payload, variant, diagnostic) || !reader.done() ||
      variant > static_cast<uint32_t>(EquivalentInputVariantV1::Metamorphic))
    return fail(diagnostic, "invalid canonical QualificationCaseKeyV1");
  key.inputVariant = static_cast<EquivalentInputVariantV1>(variant);
  return true;
}

static bool parseOptionalCase(llvm::ArrayRef<uint8_t> bytes,
                              std::optional<QualificationCaseKeyV1> &value,
                              std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  uint8_t present;
  if (!reader.takeU8(present))
    return false;
  if (present == 0) {
    value.reset();
    return reader.done() || fail(diagnostic, "absent optional has payload");
  }
  uint8_t tag;
  uint32_t size;
  llvm::ArrayRef<uint8_t> payload;
  if (present != 1 || !reader.takeU8(tag) || tag != Record ||
      !reader.takeU32(size) || !reader.takeBytes(size, payload) ||
      !reader.done())
    return fail(diagnostic, "invalid optional qualification case");
  QualificationCaseKeyV1 parsed;
  if (!parseQualificationCase(payload, parsed, diagnostic))
    return false;
  value = parsed;
  return true;
}

static bool parseClosedReason(llvm::ArrayRef<uint8_t> bytes,
                              ClosedReasonV1 &reason, std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.takeField(1, Record, payload) ||
      !parseRegistryRef(payload, reason.reason, diagnostic) ||
      !reader.takeField(2, Optional, payload) || !reader.done())
    return fail(diagnostic, "invalid canonical ClosedReasonV1");
  Reader optional{payload, 0, diagnostic};
  uint8_t present;
  if (!optional.takeU8(present))
    return false;
  if (present == 0) {
    reason.detailDigest.reset();
    return optional.done() || fail(diagnostic, "absent optional has payload");
  }
  uint8_t tag;
  uint32_t size;
  llvm::ArrayRef<uint8_t> digestBytes;
  AdoptionDigest digest;
  if (present != 1 || !optional.takeU8(tag) || tag != Digest32 ||
      !optional.takeU32(size) || !optional.takeBytes(size, digestBytes) ||
      !optional.done() || !parseDigest(digestBytes, digest, diagnostic))
    return fail(diagnostic, "invalid optional reason detail digest");
  reason.detailDigest = digest;
  return true;
}

static bool parseOptionalReason(llvm::ArrayRef<uint8_t> bytes,
                                std::optional<ClosedReasonV1> &value,
                                std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  uint8_t present;
  if (!reader.takeU8(present))
    return false;
  if (present == 0) {
    value.reset();
    return reader.done() || fail(diagnostic, "absent optional has payload");
  }
  uint8_t tag;
  uint32_t size;
  llvm::ArrayRef<uint8_t> payload;
  if (present != 1 || !reader.takeU8(tag) || tag != Record ||
      !reader.takeU32(size) || !reader.takeBytes(size, payload) ||
      !reader.done())
    return fail(diagnostic, "invalid optional terminal reason");
  ClosedReasonV1 parsed;
  if (!parseClosedReason(payload, parsed, diagnostic))
    return false;
  value = parsed;
  return true;
}

static bool parseWorkSummary(llvm::ArrayRef<uint8_t> bytes,
                             WorkSummaryV1 &summary, std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.takeField(1, Digest32, payload) ||
      !parseDigest(payload, summary.workPolicyDigest, diagnostic) ||
      !reader.takeField(2, Sequence, payload) || !reader.done())
    return fail(diagnostic, "invalid canonical WorkSummaryV1");
  Reader sequence{payload, 0, diagnostic};
  uint32_t count;
  if (!sequence.takeU32(count))
    return false;
  summary.orderedCounters.clear();
  summary.orderedCounters.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    uint32_t size;
    llvm::ArrayRef<uint8_t> counterBytes;
    if (!sequence.takeU32(size) || !sequence.takeBytes(size, counterBytes))
      return false;
    Reader counter{counterBytes, 0, diagnostic};
    WorkCounterV1 value;
    if (!counter.takeField(1, U32, payload) ||
        !parseU32Payload(payload, value.counterId, diagnostic) ||
        !counter.takeField(2, U64, payload) ||
        !parseU64Payload(payload, value.value, diagnostic) || !counter.done())
      return fail(diagnostic, "invalid canonical WorkCounterV1");
    summary.orderedCounters.push_back(value);
  }
  return sequence.done() || fail(diagnostic, "extra work counter bytes");
}

static bool parseRawBytes(llvm::ArrayRef<uint8_t> bytes, std::string &value,
                          std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  uint32_t size;
  llvm::ArrayRef<uint8_t> payload;
  if (!reader.takeU32(size) || !reader.takeBytes(size, payload) ||
      !reader.done())
    return fail(diagnostic, "invalid canonical Bytes value");
  value.assign(reinterpret_cast<const char *>(payload.data()), payload.size());
  return true;
}

static bool parseBackendAction(llvm::ArrayRef<uint8_t> bytes,
                               BackendActionEvidenceV1 &action,
                               std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  llvm::ArrayRef<uint8_t> payload;
  uint32_t status;
  if (!reader.takeField(1, Digest32, payload) ||
      !parseDigest(payload, action.invocationId, diagnostic) ||
      !reader.takeField(2, U32, payload) ||
      !parseU32Payload(payload, action.actionOrdinal, diagnostic) ||
      !reader.takeField(3, Sequence, payload))
    return fail(diagnostic, "invalid canonical BackendActionEvidenceV1");
  Reader argv{payload, 0, diagnostic};
  uint32_t count;
  if (!argv.takeU32(count))
    return false;
  action.argv.clear();
  action.argv.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    uint32_t size;
    llvm::ArrayRef<uint8_t> arg;
    std::string value;
    if (!argv.takeU32(size) || !argv.takeBytes(size, arg) ||
        !parseRawBytes(arg, value, diagnostic))
      return false;
    action.argv.push_back(std::move(value));
  }
  if (!argv.done() || !reader.takeField(4, Digest32, payload) ||
      !parseDigest(payload, action.observedToolDigest, diagnostic) ||
      !reader.takeField(5, Digest32, payload) ||
      !parseDigest(payload, action.observedOutputDigest, diagnostic) ||
      !reader.takeField(6, ClosedEnum, payload) ||
      !parseU32Payload(payload, status, diagnostic) || !reader.done() ||
      status > static_cast<uint32_t>(BackendActionStatusV1::Cancelled))
    return fail(diagnostic, "invalid canonical BackendActionEvidenceV1");
  action.terminalStatus = static_cast<BackendActionStatusV1>(status);
  return true;
}

static bool parseIdentityBody(llvm::ArrayRef<uint8_t> bytes,
                              InvocationIdentityV1 &identity,
                              std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  llvm::ArrayRef<uint8_t> payload;
  uint32_t scopeKind;
  uint32_t cutPoint;
  if (!reader.takeField(1, ClosedEnum, payload) ||
      !parseU32Payload(payload, scopeKind, diagnostic) ||
      !reader.takeField(2, Digest32, payload) ||
      !parseDigest(payload, identity.scopeDigest, diagnostic) ||
      !reader.takeField(3, Record, payload))
    return fail(diagnostic, "invalid canonical InvocationIdentityV1");
  constexpr char keyDomain[] = "wafer.mechanism-key";
  if (payload.size() != sizeof(keyDomain) + 2 + 4 ||
      !std::equal(payload.begin(), payload.begin() + sizeof(keyDomain),
                  keyDomain))
    return fail(diagnostic, "invalid mechanism key owner record");
  Reader keyReader{payload.drop_front(sizeof(keyDomain)), 0, diagnostic};
  uint16_t keySchema;
  if (!keyReader.takeU16(keySchema) || keySchema != 1 ||
      !keyReader.takeU32(identity.mechanismKey.semanticId) ||
      !keyReader.done() || !reader.takeField(4, Record, payload) ||
      !parseRegistryRef(payload, identity.invocationSite, diagnostic) ||
      !reader.takeField(5, ClosedEnum, payload) ||
      !parseU32Payload(payload, cutPoint, diagnostic) ||
      !reader.takeField(6, U64, payload) ||
      !parseU64Payload(payload, identity.invocationOrdinal, diagnostic) ||
      !reader.done() ||
      scopeKind >
          static_cast<uint32_t>(InvocationScopeKindV1::QualificationRun) ||
      cutPoint > static_cast<uint32_t>(
                     OptimizationCutPoint::DevicePublicationTransaction))
    return fail(diagnostic, "invalid canonical InvocationIdentityV1");
  identity.scopeKind = static_cast<InvocationScopeKindV1>(scopeKind);
  identity.cutPoint = static_cast<OptimizationCutPoint>(cutPoint);
  return true;
}

static bool samePreparation(const InvocationPreparationV1 &preparation,
                            const InvocationTelemetryV1 &terminal) {
  return encodeInvocationIdentityV1(preparation.identity) ==
             encodeInvocationIdentityV1(terminal.identity) &&
         encodeOptionalCase(preparation.qualificationCase) ==
             encodeOptionalCase(terminal.qualificationCase) &&
         preparation.specDigest == terminal.specDigest &&
         preparation.inputSnapshotDigest == terminal.inputSnapshotDigest;
}

} // namespace

std::optional<RegistryRefV1>
lookupOptimizationInvocationSiteV1(MechanismKey mechanismKey) {
  if (!lookupMechanismDescriptor(mechanismKey))
    return std::nullopt;
  return RegistryRefV1{mechanismKey.semanticId, 1,
                       invocationSiteRegistryDigest()};
}

std::vector<std::string> auditOptimizationInvocationSiteRegistryV1() {
  std::vector<std::string> diagnostics;
  uint32_t previous = 0;
  for (const MechanismDescriptor &descriptor : getAllMechanismDescriptors()) {
    std::optional<RegistryRefV1> site =
        lookupOptimizationInvocationSiteV1(descriptor.key);
    if (!site)
      diagnostics.push_back("mechanism has no invocation-site registry row");
    else if (site->id <= previous)
      diagnostics.push_back("invocation-site rows are not sorted unique");
    else if (site->registrySchema != 1 ||
             site->registryDigest != invocationSiteRegistryDigest())
      diagnostics.push_back("invocation-site row has registry drift");
    if (site)
      previous = site->id;
  }
  return diagnostics;
}

RegistryRefV1 getGlobalClosedReasonRefV1(GlobalClosedReasonV1 reason) {
  return {static_cast<uint32_t>(reason), 1, closedReasonRegistryDigest()};
}

bool isGlobalClosedReasonRefV1(const RegistryRefV1 &ref,
                               GlobalClosedReasonV1 reason) {
  RegistryRefV1 expected = getGlobalClosedReasonRefV1(reason);
  return ref.id == expected.id &&
         ref.registrySchema == expected.registrySchema &&
         ref.registryDigest == expected.registryDigest;
}

std::vector<uint8_t>
encodeInvocationIdentityV1(const InvocationIdentityV1 &identity) {
  std::vector<uint8_t> body;
  appendField(body, 1, ClosedEnum,
              u32(static_cast<uint32_t>(identity.scopeKind)));
  appendField(body, 2, Digest32, identity.scopeDigest);
  appendField(body, 3, Record, encodeMechanismKeyV1(identity.mechanismKey));
  appendField(body, 4, Record, encodeRegistryRef(identity.invocationSite));
  appendField(body, 5, ClosedEnum,
              u32(static_cast<uint32_t>(identity.cutPoint)));
  appendField(body, 6, U64, u64(identity.invocationOrdinal));
  return body;
}

AdoptionDigest
digestInvocationIdentityV1(const InvocationIdentityV1 &identity) {
  return sha256(makeStandalone("wafer.optimization-invocation", 1,
                               encodeInvocationIdentityV1(identity)));
}

std::vector<uint8_t>
encodeInvocationTelemetryV1(const InvocationTelemetryV1 &telemetry) {
  if (!validateInvocationTelemetryV1(telemetry, nullptr))
    return {};
  std::vector<uint8_t> body;
  appendField(body, 1, U16, u16(telemetry.schemaVersion));
  appendField(body, 2, Record, encodeInvocationIdentityV1(telemetry.identity));
  appendField(body, 3, Optional,
              encodeOptionalCase(telemetry.qualificationCase));
  appendField(body, 4, Digest32, telemetry.specDigest);
  appendField(body, 5, Digest32, telemetry.inputSnapshotDigest);
  appendField(body, 6, ClosedEnum,
              u32(static_cast<uint32_t>(telemetry.outcome)));
  appendField(body, 7, Optional,
              encodeOptionalReason(telemetry.terminalReason));
  appendField(body, 8, U64, u64(telemetry.rewriteCount));
  appendField(body, 9, Record, encodeWorkSummary(telemetry.workSummary));
  std::vector<uint8_t> actions;
  appendU32(actions, telemetry.backendActions.size());
  for (const BackendActionEvidenceV1 &action : telemetry.backendActions)
    appendElement(actions, encodeBackendAction(action));
  appendField(body, 10, Sequence, actions);
  return body;
}

AdoptionDigest
digestInvocationTelemetryV1(const InvocationTelemetryV1 &telemetry) {
  std::vector<uint8_t> body = encodeInvocationTelemetryV1(telemetry);
  if (body.empty())
    return {};
  return sha256(
      makeStandalone("wafer.optimization-invocation-terminal", 1, body));
}

bool validateInvocationPreparationV1(const InvocationPreparationV1 &preparation,
                                     std::string *diagnostic) {
  std::optional<MechanismDescriptor> descriptor =
      lookupMechanismDescriptor(preparation.identity.mechanismKey);
  if (!descriptor)
    return fail(diagnostic, "invocation identity has unknown mechanism key");
  if (descriptor->cutPoint != preparation.identity.cutPoint)
    return fail(diagnostic, "invocation identity has wrong cut point");
  if (preparation.identity.invocationSite.id == 0 ||
      preparation.identity.invocationSite.registrySchema == 0 ||
      isZeroDigest(preparation.identity.invocationSite.registryDigest))
    return fail(diagnostic, "invocation site registry reference is invalid");
  if (isZeroDigest(preparation.identity.scopeDigest) ||
      isZeroDigest(preparation.inputSnapshotDigest))
    return fail(diagnostic,
                "invocation scope or input snapshot digest is zero");
  std::optional<AdoptionSpec> spec =
      lookupAdoptionSpec(preparation.identity.mechanismKey);
  if (!spec || digestAdoptionSpecV1(*spec) != preparation.specDigest)
    return fail(diagnostic,
                "invocation spec digest is not the current exact row");
  if (preparation.identity.scopeKind ==
      InvocationScopeKindV1::QualificationRun) {
    if (!preparation.qualificationCase)
      return fail(diagnostic, "qualification invocation has no case key");
    if (preparation.qualificationCase->corpus.id == 0 ||
        preparation.qualificationCase->corpus.registrySchema == 0 ||
        isZeroDigest(preparation.qualificationCase->corpus.registryDigest) ||
        preparation.qualificationCase->rankCount == 0)
      return fail(diagnostic, "qualification case key is invalid");
  } else if (preparation.qualificationCase) {
    return fail(diagnostic, "non-qualification invocation carried a case key");
  }
  return true;
}

bool validateInvocationTelemetryV1(const InvocationTelemetryV1 &telemetry,
                                   std::string *diagnostic) {
  if (telemetry.schemaVersion != 1)
    return fail(diagnostic, "unsupported invocation telemetry schema");
  InvocationPreparationV1 preparation{
      telemetry.identity, telemetry.qualificationCase, telemetry.specDigest,
      telemetry.inputSnapshotDigest};
  if (!validateInvocationPreparationV1(preparation, diagnostic))
    return false;
  bool successfulOutcome =
      telemetry.outcome == InvocationOutcome::Applied ||
      telemetry.outcome == InvocationOutcome::NoChange ||
      telemetry.outcome == InvocationOutcome::NotApplicable;
  if (successfulOutcome != !telemetry.terminalReason.has_value())
    return fail(diagnostic, "invocation outcome/reason presence mismatch");
  if (telemetry.terminalReason &&
      (telemetry.terminalReason->reason.id == 0 ||
       telemetry.terminalReason->reason.registrySchema == 0 ||
       isZeroDigest(telemetry.terminalReason->reason.registryDigest)))
    return fail(diagnostic, "terminal reason registry reference is invalid");
  if (telemetry.terminalReason) {
    GlobalClosedReasonV1 expected = GlobalClosedReasonV1::InvalidOwnerTerminal;
    switch (telemetry.outcome) {
    case InvocationOutcome::Unsupported:
      expected = GlobalClosedReasonV1::UnsupportedSemantic;
      break;
    case InvocationOutcome::ResourceExhausted:
      expected = GlobalClosedReasonV1::ResourceExhausted;
      break;
    case InvocationOutcome::Invalid:
      expected = GlobalClosedReasonV1::InvalidOwnerTerminal;
      break;
    case InvocationOutcome::Cancelled:
      expected = GlobalClosedReasonV1::Cancelled;
      break;
    case InvocationOutcome::Applied:
    case InvocationOutcome::NoChange:
    case InvocationOutcome::NotApplicable:
      break;
    }
    if (!isGlobalClosedReasonRefV1(telemetry.terminalReason->reason, expected))
      return fail(diagnostic,
                  "invocation outcome uses the wrong closed reason");
  }
  if (isZeroDigest(telemetry.workSummary.workPolicyDigest))
    return fail(diagnostic, "work summary has zero policy digest");
  uint32_t previousCounter = 0;
  for (const WorkCounterV1 &counter : telemetry.workSummary.orderedCounters) {
    if (counter.counterId == 0 || counter.counterId <= previousCounter)
      return fail(diagnostic, "work counters are not sorted unique IDs");
    previousCounter = counter.counterId;
  }

  AdoptionDigest invocationId = digestInvocationIdentityV1(telemetry.identity);
  for (size_t index = 0; index < telemetry.backendActions.size(); ++index) {
    const BackendActionEvidenceV1 &action = telemetry.backendActions[index];
    if (action.invocationId != invocationId || action.actionOrdinal != index)
      return fail(diagnostic, "backend action identity or ordinal mismatch");
    if (action.argv.empty() || isZeroDigest(action.observedToolDigest) ||
        isZeroDigest(action.observedOutputDigest))
      return fail(diagnostic,
                  "backend action lacks actual argv/tool/output evidence");
  }

  InvocationEvidenceKind kind =
      lookupMechanismDescriptor(telemetry.identity.mechanismKey)->evidenceKind;
  switch (kind) {
  case InvocationEvidenceKind::Rewrite:
    if (!telemetry.backendActions.empty())
      return fail(diagnostic, "rewrite invocation carried backend actions");
    if ((telemetry.outcome == InvocationOutcome::Applied) !=
        (telemetry.rewriteCount != 0))
      return fail(diagnostic, "rewrite Applied outcome/count invariant failed");
    break;
  case InvocationEvidenceKind::BackendAction:
    if (telemetry.rewriteCount != 0)
      return fail(diagnostic, "backend action invocation carried rewrites");
    break;
  case InvocationEvidenceKind::InvocationOnly:
    if (telemetry.rewriteCount != 0 || !telemetry.backendActions.empty() ||
        telemetry.outcome == InvocationOutcome::Applied)
      return fail(diagnostic,
                  "invocation-only terminal carried action/rewrite");
    break;
  }
  return true;
}

bool decodeCanonicalInvocationTelemetryV1(const std::vector<uint8_t> &bytes,
                                          InvocationTelemetryV1 &telemetry,
                                          std::string *diagnostic) {
  Reader reader{bytes, 0, diagnostic};
  llvm::ArrayRef<uint8_t> payload;
  uint32_t outcome;
  if (!reader.takeField(1, U16, payload) ||
      !parseU16Payload(payload, telemetry.schemaVersion, diagnostic) ||
      !reader.takeField(2, Record, payload) ||
      !parseIdentityBody(payload, telemetry.identity, diagnostic) ||
      !reader.takeField(3, Optional, payload) ||
      !parseOptionalCase(payload, telemetry.qualificationCase, diagnostic) ||
      !reader.takeField(4, Digest32, payload) ||
      !parseDigest(payload, telemetry.specDigest, diagnostic) ||
      !reader.takeField(5, Digest32, payload) ||
      !parseDigest(payload, telemetry.inputSnapshotDigest, diagnostic) ||
      !reader.takeField(6, ClosedEnum, payload) ||
      !parseU32Payload(payload, outcome, diagnostic) ||
      outcome > static_cast<uint32_t>(InvocationOutcome::Cancelled) ||
      !reader.takeField(7, Optional, payload) ||
      !parseOptionalReason(payload, telemetry.terminalReason, diagnostic) ||
      !reader.takeField(8, U64, payload) ||
      !parseU64Payload(payload, telemetry.rewriteCount, diagnostic) ||
      !reader.takeField(9, Record, payload) ||
      !parseWorkSummary(payload, telemetry.workSummary, diagnostic) ||
      !reader.takeField(10, Sequence, payload) || !reader.done())
    return fail(diagnostic, "invalid canonical InvocationTelemetryV1");
  telemetry.outcome = static_cast<InvocationOutcome>(outcome);

  Reader actions{payload, 0, diagnostic};
  uint32_t count;
  if (!actions.takeU32(count))
    return false;
  telemetry.backendActions.clear();
  telemetry.backendActions.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    uint32_t size;
    llvm::ArrayRef<uint8_t> actionBytes;
    BackendActionEvidenceV1 action;
    if (!actions.takeU32(size) || !actions.takeBytes(size, actionBytes) ||
        !parseBackendAction(actionBytes, action, diagnostic))
      return false;
    telemetry.backendActions.push_back(std::move(action));
  }
  if (!actions.done())
    return fail(diagnostic, "extra backend action bytes");
  if (!validateInvocationTelemetryV1(telemetry, diagnostic))
    return false;
  if (encodeInvocationTelemetryV1(telemetry) != bytes)
    return fail(diagnostic, "invocation terminal bytes are not canonical");
  return true;
}

bool OptimizationInvocationJournalV1::beginInvocation(
    const InvocationPreparationV1 &preparation, std::string *diagnostic) {
  if (!validateInvocationPreparationV1(preparation, diagnostic))
    return false;
  std::lock_guard<std::mutex> lock(mutex_);
  if (sealedScopes_.count(preparation.identity.scopeDigest) != 0)
    return fail(diagnostic, "cannot begin invocation in a sealed scope");
  AdoptionDigest id = digestInvocationIdentityV1(preparation.identity);
  if (!entries_.emplace(id, Entry{preparation, std::nullopt}).second)
    return fail(diagnostic, "duplicate invocation identity begin");
  return true;
}

bool OptimizationInvocationJournalV1::commitInvocationTerminal(
    const InvocationTelemetryV1 &telemetry, std::string *diagnostic) {
  if (!validateInvocationTelemetryV1(telemetry, diagnostic))
    return false;
  std::lock_guard<std::mutex> lock(mutex_);
  if (sealedScopes_.count(telemetry.identity.scopeDigest) != 0)
    return fail(diagnostic, "cannot commit terminal in a sealed scope");
  AdoptionDigest id = digestInvocationIdentityV1(telemetry.identity);
  auto iterator = entries_.find(id);
  if (iterator == entries_.end())
    return fail(diagnostic, "terminal commit has no begin entry");
  if (iterator->second.terminal)
    return fail(diagnostic, "duplicate terminal commit for invocation ID");
  if (!samePreparation(iterator->second.preparation, telemetry))
    return fail(diagnostic, "terminal does not match its prepared invocation");
  iterator->second.terminal = telemetry;
  return true;
}

bool OptimizationInvocationJournalV1::sealScope(
    InvocationScopeKindV1 scopeKind, const AdoptionDigest &scopeDigest,
    std::string *diagnostic) {
  if (isZeroDigest(scopeDigest))
    return fail(diagnostic, "cannot seal a zero-digest scope");
  std::lock_guard<std::mutex> lock(mutex_);
  if (sealedScopes_.count(scopeDigest) != 0)
    return fail(diagnostic, "scope already sealed");
  bool found = false;
  for (const auto &entry : entries_) {
    if (entry.second.preparation.identity.scopeDigest != scopeDigest)
      continue;
    found = true;
    if (entry.second.preparation.identity.scopeKind != scopeKind)
      return fail(diagnostic,
                  "scope digest is shared by different scope kinds");
    if (!entry.second.terminal)
      return fail(diagnostic, "scope has an invocation without terminal");
  }
  if (!found)
    return fail(diagnostic, "cannot seal an empty invocation scope");
  sealedScopes_.emplace(scopeDigest, scopeKind);
  return true;
}

std::vector<InvocationTelemetryV1>
OptimizationInvocationJournalV1::committedTerminals() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<InvocationTelemetryV1> terminals;
  for (const auto &entry : entries_)
    if (entry.second.terminal)
      terminals.push_back(*entry.second.terminal);
  return terminals;
}

} // namespace wafer
