//===- OptimizationQualification.cpp - Exact optimization qualification
//------===//

#include "Wafer/Support/OptimizationQualification.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

namespace wafer {
namespace {

static bool failWith(std::string *diagnostic, const char *message) {
  if (diagnostic)
    *diagnostic = message;
  return false;
}

static bool bindingLess(const MechanismSpecBinding &lhs,
                        const MechanismSpecBinding &rhs) {
  return lhs.mechanismKey < rhs.mechanismKey;
}

static const MechanismSpecBinding *
findBinding(const std::vector<MechanismSpecBinding> &bindings,
            MechanismKey key) {
  auto it = std::lower_bound(
      bindings.begin(), bindings.end(), key,
      [](const MechanismSpecBinding &binding, MechanismKey query) {
        return binding.mechanismKey < query;
      });
  if (it == bindings.end() || it->mechanismKey != key)
    return nullptr;
  return &*it;
}

static bool
validateBindingGroup(const std::vector<MechanismSpecBinding> &bindings,
                     AdoptionMode mode, std::string *diagnostic) {
  uint32_t previous = 0;
  for (const MechanismSpecBinding &binding : bindings) {
    if (binding.mechanismKey.semanticId <= previous)
      return failWith(
          diagnostic,
          "optimization proposal bindings are not sorted and unique");
    previous = binding.mechanismKey.semanticId;
    std::optional<AdoptionSpec> spec = lookupAdoptionSpec(binding.mechanismKey);
    if (!spec)
      return failWith(diagnostic,
                      "optimization proposal binding has no adoption spec");
    if (spec->adoptionMode != mode)
      return failWith(diagnostic,
                      "optimization proposal binding is in the wrong group");
    if (digestAdoptionSpecV1(*spec) != binding.specDigest)
      return failWith(diagnostic,
                      "optimization proposal binding has a stale spec digest");
  }
  return true;
}

static bool
validateGroupSelection(const std::vector<MechanismSpecBinding> &ownBindings,
                       const std::vector<MechanismSpecBinding> &otherBindings,
                       const OptimizationGroupSelection &selection,
                       std::string *diagnostic) {
  switch (selection.kind) {
  case OptimizationGroupSelectionKind::AllOn:
  case OptimizationGroupSelectionKind::AllOff:
    if (selection.disabledKey)
      return failWith(
          diagnostic,
          "AllOn/AllOff optimization selection carried a disabled key");
    return true;
  case OptimizationGroupSelectionKind::DisableOne:
    if (!selection.disabledKey)
      return failWith(diagnostic,
                      "DisableOne optimization selection has no mechanism key");
    if (!findBinding(ownBindings, *selection.disabledKey))
      return failWith(
          diagnostic,
          findBinding(otherBindings, *selection.disabledKey)
              ? "DisableOne mechanism belongs to the other optimization group"
              : "mechanism is absent from the optimization qualification "
                "proposal");
    return true;
  }
  return failWith(diagnostic, "unknown optimization group selection kind");
}

enum QualificationTypeTag : uint8_t {
  QTU16 = 0x02,
  QTU32 = 0x03,
  QTU64 = 0x04,
  QTDigest32 = 0x07,
  QTRecord = 0x09,
  QTSortedSet = 0x0b,
};

static bool readField(const std::vector<uint8_t> &bytes, size_t limit,
                      size_t &cursor, uint16_t expectedNumber,
                      QualificationTypeTag expectedTag, size_t &payloadBegin,
                      size_t &payloadSize, std::string *diagnostic);
static AdoptionDigest
digestRegistry(llvm::StringRef domain,
               llvm::ArrayRef<std::pair<uint32_t, llvm::StringRef>> rows);

constexpr uint16_t currentMandatoryCorpusRegistrySchema = 2;
constexpr uint16_t currentStaticMetricRegistrySchema = 2;
constexpr uint32_t currentOptimizationSetQualificationPolicyId = 1;

static llvm::ArrayRef<std::pair<uint32_t, llvm::StringRef>>
mandatoryCorpusRegistryRows() {
  static const std::pair<uint32_t, llvm::StringRef> rows[] = {
      {1, "wafer-source-vertical-v1:ranks=1,16"},
      {2, "wafer-llama-2-7b-block-v1:ranks=16"},
  };
  return rows;
}

static llvm::ArrayRef<std::pair<uint32_t, llvm::StringRef>>
staticMetricRegistryRows() {
  static const std::pair<uint32_t, llvm::StringRef> rows[] = {
      {1, "ddr-high-water-bytes:u64:minimize"},
      {2, "spm-high-water-bytes:u64:minimize"},
      {3, "aggregate-spm-movement-bytes:u64:minimize"},
      {4, "aggregate-instruction-command-count:u64:minimize"},
  };
  return rows;
}

static AdoptionDigest mandatoryCorpusRegistryDigest() {
  return digestRegistry("wafer.mandatory-qualification-corpus-registry",
                        mandatoryCorpusRegistryRows());
}

static AdoptionDigest staticMetricRegistryDigest() {
  return digestRegistry("wafer.optimization-static-metric-registry",
                        staticMetricRegistryRows());
}

static void appendU16BE(std::vector<uint8_t> &bytes, uint16_t value) {
  bytes.push_back(static_cast<uint8_t>(value >> 8));
  bytes.push_back(static_cast<uint8_t>(value));
}

static void appendU32BE(std::vector<uint8_t> &bytes, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static void appendU64BE(std::vector<uint8_t> &bytes, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static void appendQualificationField(std::vector<uint8_t> &body,
                                     uint16_t number, QualificationTypeTag tag,
                                     const std::vector<uint8_t> &payload) {
  appendU16BE(body, number);
  body.push_back(static_cast<uint8_t>(tag));
  appendU32BE(body, payload.size());
  body.insert(body.end(), payload.begin(), payload.end());
}

static std::vector<uint8_t>
encodeSpecBinding(const MechanismSpecBinding &binding) {
  std::vector<uint8_t> body;
  appendQualificationField(body, 1, QTRecord,
                           encodeMechanismKeyV1(binding.mechanismKey));
  appendQualificationField(body, 2, QTDigest32,
                           std::vector<uint8_t>(binding.specDigest.begin(),
                                                binding.specDigest.end()));
  return body;
}

static std::vector<uint8_t>
encodeObservationBinding(const MechanismObservationBindingV1 &binding) {
  std::vector<uint8_t> body;
  appendQualificationField(body, 1, QTRecord,
                           encodeMechanismKeyV1(binding.mechanismKey));
  appendQualificationField(
      body, 2, QTDigest32,
      std::vector<uint8_t>(binding.observationDigest.begin(),
                           binding.observationDigest.end()));
  return body;
}

static std::vector<uint8_t>
encodeBindingSet(const std::vector<MechanismSpecBinding> &bindings) {
  std::vector<uint8_t> payload;
  appendU32BE(payload, bindings.size());
  for (const MechanismSpecBinding &binding : bindings) {
    std::vector<uint8_t> record = encodeSpecBinding(binding);
    appendU32BE(payload, record.size());
    payload.insert(payload.end(), record.begin(), record.end());
  }
  return payload;
}

static std::vector<uint8_t> encodeObservationBindingSet(
    const std::vector<MechanismObservationBindingV1> &bindings) {
  std::vector<uint8_t> payload;
  appendU32BE(payload, bindings.size());
  for (const MechanismObservationBindingV1 &binding : bindings) {
    std::vector<uint8_t> record = encodeObservationBinding(binding);
    appendU32BE(payload, record.size());
    payload.insert(payload.end(), record.begin(), record.end());
  }
  return payload;
}

static uint16_t readU16BE(const std::vector<uint8_t> &bytes, size_t offset) {
  return static_cast<uint16_t>((static_cast<uint16_t>(bytes[offset]) << 8) |
                               bytes[offset + 1]);
}

static uint32_t readU32BE(const std::vector<uint8_t> &bytes, size_t offset) {
  uint32_t value = 0;
  for (unsigned index = 0; index < 4; ++index)
    value = (value << 8) | bytes[offset + index];
  return value;
}

static uint64_t readU64BE(const std::vector<uint8_t> &bytes, size_t offset) {
  uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index)
    value = (value << 8) | bytes[offset + index];
  return value;
}

static AdoptionDigest digestBytes(llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 hash;
  hash.update(bytes);
  return hash.final();
}

static AdoptionDigest
digestRegistry(llvm::StringRef domain,
               llvm::ArrayRef<std::pair<uint32_t, llvm::StringRef>> rows) {
  std::vector<uint8_t> bytes(domain.bytes_begin(), domain.bytes_end());
  bytes.push_back(0);
  appendU16BE(bytes, 1);
  appendU32BE(bytes, rows.size());
  for (auto [id, definition] : rows) {
    appendU32BE(bytes, id);
    appendU32BE(bytes, definition.size());
    bytes.insert(bytes.end(), definition.bytes_begin(), definition.bytes_end());
  }
  return digestBytes(bytes);
}

static std::vector<uint8_t> encodeRegistryRef(const RegistryRefV1 &ref) {
  std::vector<uint8_t> body;
  std::vector<uint8_t> id;
  appendU32BE(id, ref.id);
  appendQualificationField(body, 1, QTU32, id);
  std::vector<uint8_t> schema;
  appendU16BE(schema, ref.registrySchema);
  appendQualificationField(body, 2, QTU16, schema);
  appendQualificationField(body, 3, QTDigest32,
                           std::vector<uint8_t>(ref.registryDigest.begin(),
                                                ref.registryDigest.end()));
  return body;
}

static bool parseRegistryRef(const std::vector<uint8_t> &bytes, size_t begin,
                             size_t size, RegistryRefV1 &ref,
                             std::string *diagnostic) {
  size_t cursor = begin;
  size_t limit = begin + size;
  size_t payloadBegin = 0, payloadSize = 0;
  if (!readField(bytes, limit, cursor, 1, QTU32, payloadBegin, payloadSize,
                 diagnostic) ||
      payloadSize != 4)
    return failWith(diagnostic, "invalid RegistryRefV1 id field");
  ref.id = readU32BE(bytes, payloadBegin);
  if (!readField(bytes, limit, cursor, 2, QTU16, payloadBegin, payloadSize,
                 diagnostic) ||
      payloadSize != 2)
    return failWith(diagnostic, "invalid RegistryRefV1 schema field");
  ref.registrySchema = readU16BE(bytes, payloadBegin);
  if (!readField(bytes, limit, cursor, 3, QTDigest32, payloadBegin, payloadSize,
                 diagnostic) ||
      payloadSize != ref.registryDigest.size() || cursor != limit)
    return failWith(diagnostic, "invalid RegistryRefV1 digest field");
  std::copy_n(bytes.begin() + payloadBegin, ref.registryDigest.size(),
              ref.registryDigest.begin());
  return true;
}

static std::vector<uint8_t> encodeU32Set(llvm::ArrayRef<uint32_t> values) {
  std::vector<uint8_t> payload;
  appendU32BE(payload, values.size());
  for (uint32_t value : values) {
    appendU32BE(payload, 4);
    appendU32BE(payload, value);
  }
  return payload;
}

static bool parseU32Set(const std::vector<uint8_t> &bytes, size_t begin,
                        size_t size, std::vector<uint32_t> &values,
                        std::string *diagnostic) {
  if (size < 4)
    return failWith(diagnostic, "truncated U32 sorted set");
  size_t cursor = begin;
  size_t limit = begin + size;
  uint32_t count = readU32BE(bytes, cursor);
  cursor += 4;
  values.clear();
  values.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    if (limit - cursor < 8 || readU32BE(bytes, cursor) != 4)
      return failWith(diagnostic, "invalid U32 sorted-set element");
    cursor += 4;
    values.push_back(readU32BE(bytes, cursor));
    cursor += 4;
  }
  return cursor == limit ||
         failWith(diagnostic, "U32 sorted set has trailing bytes");
}

static bool readField(const std::vector<uint8_t> &bytes, size_t limit,
                      size_t &cursor, uint16_t expectedNumber,
                      QualificationTypeTag expectedTag, size_t &payloadBegin,
                      size_t &payloadSize, std::string *diagnostic) {
  if (cursor > limit || limit - cursor < 7)
    return failWith(diagnostic, "missing or truncated proposal field");
  uint16_t number = readU16BE(bytes, cursor);
  auto tag = static_cast<QualificationTypeTag>(bytes[cursor + 2]);
  uint32_t size = readU32BE(bytes, cursor + 3);
  cursor += 7;
  if (number != expectedNumber || tag != expectedTag || size > limit - cursor)
    return failWith(diagnostic,
                    "unknown, reordered, mistyped or truncated proposal field");
  payloadBegin = cursor;
  payloadSize = size;
  cursor += size;
  return true;
}

static bool parseBindingSet(const std::vector<uint8_t> &bytes, size_t begin,
                            size_t size,
                            std::vector<MechanismSpecBinding> &bindings,
                            std::string *diagnostic) {
  if (size < 4)
    return failWith(diagnostic, "truncated proposal binding set");
  size_t cursor = begin;
  size_t limit = begin + size;
  uint32_t count = readU32BE(bytes, cursor);
  cursor += 4;
  bindings.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    if (limit - cursor < 4)
      return failWith(diagnostic, "truncated proposal binding size");
    uint32_t recordSize = readU32BE(bytes, cursor);
    cursor += 4;
    if (recordSize > limit - cursor)
      return failWith(diagnostic, "truncated proposal binding record");
    size_t recordLimit = cursor + recordSize;
    size_t keyBegin = 0, keySize = 0;
    if (!readField(bytes, recordLimit, cursor, 1, QTRecord, keyBegin, keySize,
                   diagnostic))
      return false;
    constexpr char keyOwner[] = "wafer.mechanism-key";
    if (keySize != sizeof(keyOwner) + 2 + 4 ||
        !std::equal(bytes.begin() + keyBegin,
                    bytes.begin() + keyBegin + sizeof(keyOwner), keyOwner) ||
        readU16BE(bytes, keyBegin + sizeof(keyOwner)) != 1)
      return failWith(diagnostic,
                      "proposal binding has invalid mechanism key record");
    MechanismKey key{readU32BE(bytes, keyBegin + sizeof(keyOwner) + 2)};
    size_t digestBegin = 0, digestSize = 0;
    if (!readField(bytes, recordLimit, cursor, 2, QTDigest32, digestBegin,
                   digestSize, diagnostic))
      return false;
    if (digestSize != 32 || cursor != recordLimit)
      return failWith(diagnostic,
                      "proposal binding has invalid digest or extra field");
    AdoptionDigest digest;
    std::copy_n(bytes.begin() + digestBegin, digest.size(), digest.begin());
    bindings.push_back({key, digest});
  }
  if (cursor != limit)
    return failWith(diagnostic,
                    "proposal binding set has trailing or extra records");
  return true;
}

static bool
parseObservationBindingSet(const std::vector<uint8_t> &bytes, size_t begin,
                           size_t size,
                           std::vector<MechanismObservationBindingV1> &bindings,
                           std::string *diagnostic) {
  if (size < 4)
    return failWith(diagnostic, "truncated observation binding set");
  size_t cursor = begin;
  size_t limit = begin + size;
  uint32_t count = readU32BE(bytes, cursor);
  cursor += 4;
  bindings.clear();
  bindings.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    if (limit - cursor < 4)
      return failWith(diagnostic, "truncated observation binding size");
    uint32_t recordSize = readU32BE(bytes, cursor);
    cursor += 4;
    if (recordSize > limit - cursor)
      return failWith(diagnostic, "truncated observation binding record");
    size_t recordLimit = cursor + recordSize;
    size_t keyBegin = 0, keySize = 0;
    if (!readField(bytes, recordLimit, cursor, 1, QTRecord, keyBegin, keySize,
                   diagnostic))
      return false;
    constexpr char keyOwner[] = "wafer.mechanism-key";
    if (keySize != sizeof(keyOwner) + 2 + 4 ||
        !std::equal(bytes.begin() + keyBegin,
                    bytes.begin() + keyBegin + sizeof(keyOwner), keyOwner) ||
        readU16BE(bytes, keyBegin + sizeof(keyOwner)) != 1)
      return failWith(diagnostic,
                      "observation binding has invalid mechanism key");
    MechanismKey key{readU32BE(bytes, keyBegin + sizeof(keyOwner) + 2)};
    size_t digestBegin = 0, digestSize = 0;
    if (!readField(bytes, recordLimit, cursor, 2, QTDigest32, digestBegin,
                   digestSize, diagnostic) ||
        digestSize != 32 || cursor != recordLimit)
      return failWith(diagnostic,
                      "observation binding has invalid digest or field");
    AdoptionDigest digest;
    std::copy_n(bytes.begin() + digestBegin, digest.size(), digest.begin());
    bindings.push_back({key, digest});
  }
  return cursor == limit ||
         failWith(diagnostic, "observation binding set has trailing bytes");
}

static llvm::APInt canonicalUnsigned(llvm::APInt value) {
  unsigned width = std::max(1u, value.getActiveBits());
  return value.zextOrTrunc(width);
}

static llvm::APInt unsignedValue(uint64_t value) {
  return canonicalUnsigned(llvm::APInt(64, value));
}

static llvm::APInt addUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  unsigned width = std::max(lhs.getActiveBits(), rhs.getActiveBits()) + 1;
  return canonicalUnsigned(lhs.zext(width) + rhs.zext(width));
}

static llvm::APInt multiplyUnsigned(const llvm::APInt &lhs,
                                    const llvm::APInt &rhs) {
  if (lhs.isZero() || rhs.isZero())
    return unsignedValue(0);
  unsigned width = lhs.getActiveBits() + rhs.getActiveBits();
  return canonicalUnsigned(lhs.zext(width) * rhs.zext(width));
}

static int compareUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  unsigned width = std::max(lhs.getActiveBits(), rhs.getActiveBits());
  llvm::APInt left = lhs.zextOrTrunc(std::max(1u, width));
  llvm::APInt right = rhs.zextOrTrunc(std::max(1u, width));
  if (left.ult(right))
    return -1;
  if (right.ult(left))
    return 1;
  return 0;
}

static llvm::APInt subtractUnsigned(const llvm::APInt &larger,
                                    const llvm::APInt &smaller) {
  assert(compareUnsigned(larger, smaller) >= 0);
  unsigned width = std::max(larger.getActiveBits(), smaller.getActiveBits());
  return canonicalUnsigned(larger.zextOrTrunc(std::max(1u, width)) -
                           smaller.zextOrTrunc(std::max(1u, width)));
}

static llvm::APInt greatestCommonDivisor(llvm::APInt lhs, llvm::APInt rhs) {
  unsigned width = std::max({1u, lhs.getActiveBits(), rhs.getActiveBits()});
  lhs = lhs.zextOrTrunc(width);
  rhs = rhs.zextOrTrunc(width);
  while (!rhs.isZero()) {
    llvm::APInt remainder = lhs.urem(rhs);
    lhs = std::move(rhs);
    rhs = std::move(remainder);
  }
  return canonicalUnsigned(std::move(lhs));
}

static llvm::APInt divideUnsignedExact(const llvm::APInt &numerator,
                                       const llvm::APInt &denominator) {
  unsigned width =
      std::max({1u, numerator.getActiveBits(), denominator.getActiveBits()});
  llvm::APInt n = numerator.zextOrTrunc(width);
  llvm::APInt d = denominator.zextOrTrunc(width);
  assert(!d.isZero() && n.urem(d).isZero());
  return canonicalUnsigned(n.udiv(d));
}

static std::string unsignedDecimal(const llvm::APInt &value) {
  llvm::SmallString<64> text;
  value.toString(text, /*Radix=*/10, /*Signed=*/false);
  return std::string(text);
}

class BigRational {
public:
  BigRational()
      : numerator_(unsignedValue(0)), denominator_(unsignedValue(1)) {}

  static BigRational fromUnsigned(uint64_t value) {
    return BigRational(/*negative=*/false, unsignedValue(value),
                       unsignedValue(1));
  }

  BigRational operator-() const {
    BigRational result = *this;
    if (!result.numerator_.isZero())
      result.negative_ = !result.negative_;
    return result;
  }

  BigRational operator+(const BigRational &other) const {
    llvm::APInt left = multiplyUnsigned(numerator_, other.denominator_);
    llvm::APInt right = multiplyUnsigned(other.numerator_, denominator_);
    llvm::APInt denominator =
        multiplyUnsigned(denominator_, other.denominator_);
    if (negative_ == other.negative_)
      return BigRational(negative_, addUnsigned(left, right), denominator);
    int comparison = compareUnsigned(left, right);
    if (comparison == 0)
      return BigRational();
    if (comparison > 0)
      return BigRational(negative_, subtractUnsigned(left, right), denominator);
    return BigRational(other.negative_, subtractUnsigned(right, left),
                       denominator);
  }

  BigRational operator-(const BigRational &other) const {
    return *this + (-other);
  }

  BigRational divideBy(uint32_t divisor) const {
    assert(divisor != 0);
    return BigRational(negative_, numerator_,
                       multiplyUnsigned(denominator_, unsignedValue(divisor)));
  }

  BigRational multiplyBy(uint32_t multiplier) const {
    return BigRational(negative_,
                       multiplyUnsigned(numerator_, unsignedValue(multiplier)),
                       denominator_);
  }

  BigRational absolute() const {
    BigRational value = *this;
    value.negative_ = false;
    return value;
  }

  int compare(const BigRational &other) const {
    if (negative_ != other.negative_)
      return negative_ ? -1 : 1;
    llvm::APInt left = multiplyUnsigned(numerator_, other.denominator_);
    llvm::APInt right = multiplyUnsigned(other.numerator_, denominator_);
    int result = compareUnsigned(left, right);
    return negative_ ? -result : result;
  }

  ExactRationalValue value() const {
    std::string numerator = unsignedDecimal(numerator_);
    if (negative_)
      numerator.insert(numerator.begin(), '-');
    return {std::move(numerator), unsignedDecimal(denominator_)};
  }

private:
  BigRational(bool negative, llvm::APInt numerator, llvm::APInt denominator)
      : negative_(negative),
        numerator_(canonicalUnsigned(std::move(numerator))),
        denominator_(canonicalUnsigned(std::move(denominator))) {
    assert(!denominator_.isZero());
    if (numerator_.isZero()) {
      negative_ = false;
      denominator_ = unsignedValue(1);
      return;
    }
    llvm::APInt divisor = greatestCommonDivisor(numerator_, denominator_);
    numerator_ = divideUnsignedExact(numerator_, divisor);
    denominator_ = divideUnsignedExact(denominator_, divisor);
  }

  bool negative_ = false;
  llvm::APInt numerator_;
  llvm::APInt denominator_;
};

static BigRational median(std::vector<BigRational> values) {
  assert(!values.empty());
  std::sort(values.begin(), values.end(),
            [](const BigRational &lhs, const BigRational &rhs) {
              return lhs.compare(rhs) < 0;
            });
  size_t middle = values.size() / 2;
  if ((values.size() & 1) != 0)
    return values[middle];
  return (values[middle - 1] + values[middle]).divideBy(2);
}

template <typename Selector>
static ABBAHostMetricDecision
evaluateMetric(const std::array<std::array<const ABBASample *, 4>, 5> &blocks,
               Selector selector) {
  std::vector<BigRational> baselines;
  std::vector<BigRational> deltas;
  baselines.reserve(blocks.size());
  deltas.reserve(blocks.size());
  for (const auto &block : blocks) {
    BigRational aLeft = BigRational::fromUnsigned(selector(*block[0]));
    BigRational bLeft = BigRational::fromUnsigned(selector(*block[1]));
    BigRational bRight = BigRational::fromUnsigned(selector(*block[2]));
    BigRational aRight = BigRational::fromUnsigned(selector(*block[3]));
    BigRational a = median({aLeft, aRight});
    BigRational b = median({bLeft, bRight});
    baselines.push_back(a);
    deltas.push_back(b - a);
  }

  BigRational baselineMedian = median(baselines);
  BigRational deltaMedian = median(deltas);
  std::vector<BigRational> deviations;
  deviations.reserve(deltas.size());
  for (const BigRational &delta : deltas)
    deviations.push_back((delta - deltaMedian).absolute());
  BigRational mad = median(std::move(deviations));
  BigRational fractionalGuard = baselineMedian.divideBy(20);
  BigRational variabilityGuard = mad.multiplyBy(3);
  BigRational guard = fractionalGuard.compare(variabilityGuard) >= 0
                          ? fractionalGuard
                          : variabilityGuard;
  BigRational zero;
  bool nonRegressed = deltaMedian.compare(guard) <= 0;
  bool significantImprovement =
      deltaMedian.compare(-guard) < 0 && guard.compare(zero) >= 0;
  return {baselineMedian.value(), deltaMedian.value(), mad.value(),
          guard.value(),          nonRegressed,        significantImprovement};
}

} // namespace

OptimizationQualificationProposal
getCurrentOptimizationQualificationProposal() {
  OptimizationQualificationProposal proposal;
  for (const AdoptionSpec &spec : getAllAdoptionSpecs()) {
    std::vector<MechanismSpecBinding> *bindings = nullptr;
    if (spec.adoptionMode == AdoptionMode::FixedOptimization)
      bindings = &proposal.fixedBindings;
    else if (spec.adoptionMode == AdoptionMode::BestEffortCleanup)
      bindings = &proposal.cleanupBindings;
    if (bindings)
      bindings->push_back({spec.mechanismKey, digestAdoptionSpecV1(spec)});
  }
  std::sort(proposal.fixedBindings.begin(), proposal.fixedBindings.end(),
            bindingLess);
  std::sort(proposal.cleanupBindings.begin(), proposal.cleanupBindings.end(),
            bindingLess);
  return proposal;
}

std::vector<uint8_t> encodeOptimizationQualificationProposalV1(
    const OptimizationQualificationProposal &proposal) {
  if (!validateOptimizationQualificationProposal(proposal, nullptr))
    return {};
  std::vector<uint8_t> body;
  std::vector<uint8_t> schema;
  appendU16BE(schema, proposal.schemaVersion);
  appendQualificationField(body, 1, QTU16, schema);
  appendQualificationField(body, 2, QTSortedSet,
                           encodeBindingSet(proposal.fixedBindings));
  appendQualificationField(body, 3, QTSortedSet,
                           encodeBindingSet(proposal.cleanupBindings));

  constexpr char owner[] = "wafer.optimization-qualification-proposal";
  std::vector<uint8_t> bytes(owner, owner + sizeof(owner));
  appendU16BE(bytes, proposal.schemaVersion);
  appendU32BE(bytes, body.size());
  bytes.insert(bytes.end(), body.begin(), body.end());
  return bytes;
}

AdoptionDigest digestOptimizationQualificationProposalV1(
    const OptimizationQualificationProposal &proposal) {
  std::vector<uint8_t> bytes =
      encodeOptimizationQualificationProposalV1(proposal);
  llvm::SHA256 hash;
  hash.update(llvm::ArrayRef<uint8_t>(bytes));
  return hash.final();
}

bool validateCanonicalOptimizationQualificationProposalV1(
    const std::vector<uint8_t> &bytes, std::string *diagnostic) {
  constexpr char owner[] = "wafer.optimization-qualification-proposal";
  constexpr size_t envelopeSize = sizeof(owner) + 2 + 4;
  if (bytes.size() < envelopeSize ||
      !std::equal(bytes.begin(), bytes.begin() + sizeof(owner), owner))
    return failWith(diagnostic,
                    "invalid fixed optimization proposal domain separator");
  if (readU16BE(bytes, sizeof(owner)) != 1 ||
      readU32BE(bytes, sizeof(owner) + 2) != bytes.size() - envelopeSize)
    return failWith(
        diagnostic,
        "invalid fixed optimization proposal envelope size or schema");

  size_t cursor = envelopeSize;
  size_t schemaBegin = 0, schemaSize = 0;
  if (!readField(bytes, bytes.size(), cursor, 1, QTU16, schemaBegin, schemaSize,
                 diagnostic) ||
      schemaSize != 2 || readU16BE(bytes, schemaBegin) != 1)
    return failWith(diagnostic, "invalid proposal schema field");
  size_t fixedBegin = 0, fixedSize = 0;
  if (!readField(bytes, bytes.size(), cursor, 2, QTSortedSet, fixedBegin,
                 fixedSize, diagnostic))
    return false;
  size_t cleanupBegin = 0, cleanupSize = 0;
  if (!readField(bytes, bytes.size(), cursor, 3, QTSortedSet, cleanupBegin,
                 cleanupSize, diagnostic))
    return false;
  if (cursor != bytes.size())
    return failWith(diagnostic,
                    "fixed optimization proposal has extra fields or bytes");

  OptimizationQualificationProposal proposal;
  if (!parseBindingSet(bytes, fixedBegin, fixedSize, proposal.fixedBindings,
                       diagnostic) ||
      !parseBindingSet(bytes, cleanupBegin, cleanupSize,
                       proposal.cleanupBindings, diagnostic) ||
      !validateOptimizationQualificationProposal(proposal, diagnostic))
    return false;
  if (encodeOptimizationQualificationProposalV1(proposal) != bytes)
    return failWith(diagnostic,
                    "fixed optimization proposal is not canonical bytes");
  return true;
}

bool validateOptimizationQualificationProposal(
    const OptimizationQualificationProposal &proposal,
    std::string *diagnostic) {
  if (proposal.schemaVersion != 1)
    return failWith(diagnostic, "unsupported optimization proposal schema");
  if (!validateBindingGroup(proposal.fixedBindings,
                            AdoptionMode::FixedOptimization, diagnostic) ||
      !validateBindingGroup(proposal.cleanupBindings,
                            AdoptionMode::BestEffortCleanup, diagnostic))
    return false;
  for (const MechanismSpecBinding &binding : proposal.fixedBindings)
    if (findBinding(proposal.cleanupBindings, binding.mechanismKey))
      return failWith(diagnostic,
                      "optimization proposal fixed and cleanup groups overlap");
  return true;
}

bool validateQualifiedOptimizationSetV1(
    const QualifiedOptimizationSetV1 &qualifiedSet, std::string *diagnostic) {
  if (qualifiedSet.schemaVersion != 1)
    return failWith(diagnostic,
                    "unsupported qualified optimization set schema");
  auto digestIsZero = [](const AdoptionDigest &digest) {
    return std::all_of(digest.begin(), digest.end(),
                       [](uint8_t byte) { return byte == 0; });
  };
  OptimizationQualificationProposal proposal =
      getCurrentOptimizationQualificationProposal();
  if (!validateOptimizationQualificationProposal(proposal, diagnostic))
    return false;
  if (qualifiedSet.proposalDigest !=
      digestOptimizationQualificationProposalV1(proposal))
    return failWith(diagnostic,
                    "qualified set does not bind the current proposal");
  if (digestIsZero(qualifiedSet.batchObservationDigest))
    return failWith(diagnostic,
                    "qualified set has a zero batch observation digest");

  std::vector<MechanismKey> expected;
  expected.reserve(proposal.fixedBindings.size() +
                   proposal.cleanupBindings.size());
  for (const MechanismSpecBinding &binding : proposal.fixedBindings)
    expected.push_back(binding.mechanismKey);
  for (const MechanismSpecBinding &binding : proposal.cleanupBindings)
    expected.push_back(binding.mechanismKey);
  std::sort(expected.begin(), expected.end());

  if (qualifiedSet.observationBindings.size() != expected.size())
    return failWith(
        diagnostic,
        "qualified set observation bindings do not cover proposal union");
  for (size_t index = 0; index < expected.size(); ++index) {
    const MechanismObservationBindingV1 &binding =
        qualifiedSet.observationBindings[index];
    if (binding.mechanismKey != expected[index])
      return failWith(
          diagnostic,
          "qualified set observation bindings are not all-and-only sorted");
    if (digestIsZero(binding.observationDigest))
      return failWith(diagnostic,
                      "qualified set has a zero observation digest");
  }
  return true;
}

std::vector<uint8_t> encodeQualifiedOptimizationSetV1(
    const QualifiedOptimizationSetV1 &qualifiedSet) {
  if (!validateQualifiedOptimizationSetV1(qualifiedSet, nullptr))
    return {};
  std::vector<uint8_t> body;
  std::vector<uint8_t> schema;
  appendU16BE(schema, qualifiedSet.schemaVersion);
  appendQualificationField(body, 1, QTU16, schema);
  appendQualificationField(
      body, 2, QTDigest32,
      std::vector<uint8_t>(qualifiedSet.proposalDigest.begin(),
                           qualifiedSet.proposalDigest.end()));
  appendQualificationField(
      body, 3, QTDigest32,
      std::vector<uint8_t>(qualifiedSet.batchObservationDigest.begin(),
                           qualifiedSet.batchObservationDigest.end()));
  appendQualificationField(
      body, 4, QTSortedSet,
      encodeObservationBindingSet(qualifiedSet.observationBindings));

  constexpr char owner[] = "wafer.qualified-optimization-set";
  std::vector<uint8_t> bytes(owner, owner + sizeof(owner));
  appendU16BE(bytes, qualifiedSet.schemaVersion);
  appendU32BE(bytes, body.size());
  bytes.insert(bytes.end(), body.begin(), body.end());
  return bytes;
}

AdoptionDigest digestQualifiedOptimizationSetV1(
    const QualifiedOptimizationSetV1 &qualifiedSet) {
  std::vector<uint8_t> bytes = encodeQualifiedOptimizationSetV1(qualifiedSet);
  return bytes.empty() ? AdoptionDigest{} : digestBytes(bytes);
}

bool decodeCanonicalQualifiedOptimizationSetV1(
    const std::vector<uint8_t> &bytes, QualifiedOptimizationSetV1 &qualifiedSet,
    std::string *diagnostic) {
  constexpr char owner[] = "wafer.qualified-optimization-set";
  constexpr size_t envelopeSize = sizeof(owner) + 2 + 4;
  if (bytes.size() < envelopeSize ||
      !std::equal(bytes.begin(), bytes.begin() + sizeof(owner), owner))
    return failWith(diagnostic,
                    "invalid qualified optimization set domain separator");
  if (readU16BE(bytes, sizeof(owner)) != 1 ||
      readU32BE(bytes, sizeof(owner) + 2) != bytes.size() - envelopeSize)
    return failWith(diagnostic, "invalid qualified optimization set envelope");

  QualifiedOptimizationSetV1 parsed;
  size_t cursor = envelopeSize;
  size_t payloadBegin = 0, payloadSize = 0;
  if (!readField(bytes, bytes.size(), cursor, 1, QTU16, payloadBegin,
                 payloadSize, diagnostic) ||
      payloadSize != 2 || readU16BE(bytes, payloadBegin) != 1)
    return failWith(diagnostic,
                    "invalid qualified optimization set schema field");
  if (!readField(bytes, bytes.size(), cursor, 2, QTDigest32, payloadBegin,
                 payloadSize, diagnostic) ||
      payloadSize != 32)
    return failWith(diagnostic,
                    "invalid qualified optimization proposal digest field");
  std::copy_n(bytes.begin() + payloadBegin, parsed.proposalDigest.size(),
              parsed.proposalDigest.begin());
  if (!readField(bytes, bytes.size(), cursor, 3, QTDigest32, payloadBegin,
                 payloadSize, diagnostic) ||
      payloadSize != 32)
    return failWith(diagnostic,
                    "invalid qualified optimization batch digest field");
  std::copy_n(bytes.begin() + payloadBegin,
              parsed.batchObservationDigest.size(),
              parsed.batchObservationDigest.begin());
  if (!readField(bytes, bytes.size(), cursor, 4, QTSortedSet, payloadBegin,
                 payloadSize, diagnostic) ||
      !parseObservationBindingSet(bytes, payloadBegin, payloadSize,
                                  parsed.observationBindings, diagnostic))
    return false;
  if (cursor != bytes.size())
    return failWith(diagnostic,
                    "qualified optimization set has extra fields or bytes");
  if (!validateQualifiedOptimizationSetV1(parsed, diagnostic))
    return false;
  if (encodeQualifiedOptimizationSetV1(parsed) != bytes)
    return failWith(diagnostic,
                    "qualified optimization set is not canonical bytes");
  qualifiedSet = std::move(parsed);
  return true;
}

RegistryRefV1 getCurrentMandatoryCorpusRegistryRefV1() {
  return {1, currentMandatoryCorpusRegistrySchema,
          mandatoryCorpusRegistryDigest()};
}

std::vector<RegistryRefV1> getCurrentMandatoryCorpusRowsV1() {
  std::vector<RegistryRefV1> rows;
  rows.reserve(mandatoryCorpusRegistryRows().size());
  AdoptionDigest digest = mandatoryCorpusRegistryDigest();
  for (auto [id, definition] : mandatoryCorpusRegistryRows()) {
    (void)definition;
    rows.push_back({id, currentMandatoryCorpusRegistrySchema, digest});
  }
  return rows;
}

std::vector<QualificationCaseKeyV1> getCurrentMandatoryQualificationCasesV1() {
  std::vector<RegistryRefV1> corpora = getCurrentMandatoryCorpusRowsV1();
  return {
      {corpora[0], 1, EquivalentInputVariantV1::Original},
      {corpora[0], 16, EquivalentInputVariantV1::Original},
      {corpora[1], 16, EquivalentInputVariantV1::Original},
  };
}

RegistryRefV1 getCurrentStaticMetricRegistryRefV1() {
  return {1, currentStaticMetricRegistrySchema, staticMetricRegistryDigest()};
}

OptimizationSetQualificationPolicyV1
getCurrentOptimizationSetQualificationPolicyV1() {
  OptimizationSetQualificationPolicyV1 policy;
  policy.mandatoryCorpusRegistry = getCurrentMandatoryCorpusRegistryRefV1();
  policy.mandatoryRankCounts = {1, 16};
  policy.staticMetricRegistry = getCurrentStaticMetricRegistryRefV1();
  policy.hostEnvironmentRetryCap = 2;
  policy.totalProcessLaunchCap = 65536;
  policy.totalGatewayInvocationCap = 1000000;
  return policy;
}

QualificationPolicyRefV1 getCurrentOptimizationSetQualificationPolicyRefV1() {
  OptimizationSetQualificationPolicyV1 policy =
      getCurrentOptimizationSetQualificationPolicyV1();
  return {currentOptimizationSetQualificationPolicyId, policy.schemaVersion,
          digestOptimizationSetQualificationPolicyV1(policy)};
}

std::vector<uint8_t> encodeOptimizationSetQualificationPolicyV1(
    const OptimizationSetQualificationPolicyV1 &policy) {
  if (!validateOptimizationSetQualificationPolicyV1(policy, nullptr))
    return {};

  std::vector<uint8_t> body;
  auto appendU32Field = [&](uint16_t number, uint32_t value) {
    std::vector<uint8_t> payload;
    appendU32BE(payload, value);
    appendQualificationField(body, number, QTU32, payload);
  };
  auto appendU64Field = [&](uint16_t number, uint64_t value) {
    std::vector<uint8_t> payload;
    appendU64BE(payload, value);
    appendQualificationField(body, number, QTU64, payload);
  };

  std::vector<uint8_t> schema;
  appendU16BE(schema, policy.schemaVersion);
  appendQualificationField(body, 1, QTU16, schema);
  appendQualificationField(body, 2, QTRecord,
                           encodeRegistryRef(policy.mandatoryCorpusRegistry));
  appendQualificationField(body, 3, QTSortedSet,
                           encodeU32Set(policy.mandatoryRankCounts));
  appendQualificationField(body, 4, QTRecord,
                           encodeRegistryRef(policy.staticMetricRegistry));
  appendU32Field(5, policy.abbaBlockCount);
  appendU32Field(6, policy.wallGuardFractionNumerator);
  appendU32Field(7, policy.wallGuardFractionDenominator);
  appendU32Field(8, policy.madMultiplier);
  appendU32Field(9, policy.hostEnvironmentRetryCap);
  appendU64Field(10, policy.totalProcessLaunchCap);
  appendU64Field(11, policy.totalGatewayInvocationCap);

  constexpr char owner[] = "wafer.optimization-set-qualification-policy";
  std::vector<uint8_t> bytes(owner, owner + sizeof(owner));
  appendU16BE(bytes, policy.schemaVersion);
  appendU32BE(bytes, body.size());
  bytes.insert(bytes.end(), body.begin(), body.end());
  return bytes;
}

AdoptionDigest digestOptimizationSetQualificationPolicyV1(
    const OptimizationSetQualificationPolicyV1 &policy) {
  return digestBytes(encodeOptimizationSetQualificationPolicyV1(policy));
}

bool validateOptimizationSetQualificationPolicyV1(
    const OptimizationSetQualificationPolicyV1 &policy,
    std::string *diagnostic) {
  auto sameRegistryRef = [](const RegistryRefV1 &lhs,
                            const RegistryRefV1 &rhs) {
    return lhs.id == rhs.id && lhs.registrySchema == rhs.registrySchema &&
           lhs.registryDigest == rhs.registryDigest;
  };
  if (policy.schemaVersion != 1)
    return failWith(
        diagnostic,
        "unsupported fixed optimization qualification policy schema");
  if (!sameRegistryRef(policy.mandatoryCorpusRegistry,
                       getCurrentMandatoryCorpusRegistryRefV1()))
    return failWith(diagnostic, "unknown mandatory corpus registry revision");
  if (!sameRegistryRef(policy.staticMetricRegistry,
                       getCurrentStaticMetricRegistryRefV1()))
    return failWith(diagnostic, "unknown static metric registry revision");
  if (policy.mandatoryRankCounts.empty())
    return failWith(diagnostic, "mandatory rank-count set is empty");
  uint32_t previousRank = 0;
  for (uint32_t rankCount : policy.mandatoryRankCounts) {
    if (rankCount == 0 || rankCount <= previousRank)
      return failWith(
          diagnostic,
          "mandatory rank-count set is not sorted, unique and nonzero");
    previousRank = rankCount;
  }
  const std::vector<uint32_t> registeredRankDomain = {1, 16};
  if (policy.mandatoryRankCounts != registeredRankDomain)
    return failWith(
        diagnostic,
        "mandatory rank-count domain does not match the corpus case registry");
  if (policy.abbaBlockCount != 5 || policy.wallGuardFractionNumerator != 1 ||
      policy.wallGuardFractionDenominator != 20 || policy.madMultiplier != 3)
    return failWith(diagnostic,
                    "policy does not match the registered fixed ABBA guard");

  auto checkedMultiply = [](uint64_t lhs, uint64_t rhs, uint64_t &result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
      return false;
    result = lhs * rhs;
    return true;
  };
  auto checkedAdd = [](uint64_t lhs, uint64_t rhs, uint64_t &result) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
      return false;
    result = lhs + rhs;
    return true;
  };

  OptimizationQualificationProposal proposal =
      getCurrentOptimizationQualificationProposal();
  uint64_t caseCount = getCurrentMandatoryQualificationCasesV1().size();
  uint64_t comparisonCount = 0;
  uint64_t sampledLaunchesPerComparison = 0;
  uint64_t sampledLaunches = 0;
  uint64_t mandatoryNonSampledLaunches = 0;
  uint64_t minimumProcessLaunches = 0;
  uint64_t minimumGatewayInvocations = 0;
  if (!checkedAdd(proposal.fixedBindings.size(),
                  proposal.cleanupBindings.size(), comparisonCount) ||
      (!(proposal.fixedBindings.empty() && proposal.cleanupBindings.empty()) &&
       !checkedAdd(comparisonCount, 1, comparisonCount)) ||
      !checkedMultiply(policy.abbaBlockCount, 4,
                       sampledLaunchesPerComparison) ||
      !checkedAdd(sampledLaunchesPerComparison, 2,
                  sampledLaunchesPerComparison) ||
      !checkedMultiply(caseCount, comparisonCount, sampledLaunches) ||
      !checkedMultiply(sampledLaunches, sampledLaunchesPerComparison,
                       sampledLaunches) ||
      // Equivalent-IR 2x2, once/twice idempotence, and final AllOn vertical.
      !checkedMultiply(caseCount, 7, mandatoryNonSampledLaunches) ||
      !checkedAdd(sampledLaunches, mandatoryNonSampledLaunches,
                  minimumProcessLaunches) ||
      !checkedMultiply(minimumProcessLaunches, getAllAdoptionSpecs().size(),
                       minimumGatewayInvocations))
    return failWith(
        diagnostic,
        "mandatory qualification work-bound calculation overflowed");
  if (policy.totalProcessLaunchCap < minimumProcessLaunches)
    return failWith(diagnostic,
                    "process launch cap cannot cover the mandatory run");
  if (policy.totalGatewayInvocationCap < minimumGatewayInvocations)
    return failWith(diagnostic,
                    "gateway invocation cap cannot cover the mandatory run");
  return true;
}

bool decodeCanonicalOptimizationSetQualificationPolicyV1(
    const std::vector<uint8_t> &bytes,
    OptimizationSetQualificationPolicyV1 &policy, std::string *diagnostic) {
  constexpr char owner[] = "wafer.optimization-set-qualification-policy";
  constexpr size_t envelopeSize = sizeof(owner) + 2 + 4;
  if (bytes.size() < envelopeSize ||
      !std::equal(bytes.begin(), bytes.begin() + sizeof(owner), owner))
    return failWith(diagnostic,
                    "invalid qualification policy domain separator");
  if (readU16BE(bytes, sizeof(owner)) != 1 ||
      readU32BE(bytes, sizeof(owner) + 2) != bytes.size() - envelopeSize)
    return failWith(diagnostic,
                    "invalid qualification policy envelope size or schema");

  OptimizationSetQualificationPolicyV1 decoded;
  size_t cursor = envelopeSize;
  size_t begin = 0, size = 0;
  if (!readField(bytes, bytes.size(), cursor, 1, QTU16, begin, size,
                 diagnostic) ||
      size != 2)
    return failWith(diagnostic, "invalid qualification policy schema field");
  decoded.schemaVersion = readU16BE(bytes, begin);
  if (!readField(bytes, bytes.size(), cursor, 2, QTRecord, begin, size,
                 diagnostic) ||
      !parseRegistryRef(bytes, begin, size, decoded.mandatoryCorpusRegistry,
                        diagnostic))
    return false;
  if (!readField(bytes, bytes.size(), cursor, 3, QTSortedSet, begin, size,
                 diagnostic) ||
      !parseU32Set(bytes, begin, size, decoded.mandatoryRankCounts, diagnostic))
    return false;
  if (!readField(bytes, bytes.size(), cursor, 4, QTRecord, begin, size,
                 diagnostic) ||
      !parseRegistryRef(bytes, begin, size, decoded.staticMetricRegistry,
                        diagnostic))
    return false;

  auto readU32Field = [&](uint16_t number, uint32_t &value) {
    if (!readField(bytes, bytes.size(), cursor, number, QTU32, begin, size,
                   diagnostic) ||
        size != 4)
      return false;
    value = readU32BE(bytes, begin);
    return true;
  };
  auto readU64Field = [&](uint16_t number, uint64_t &value) {
    if (!readField(bytes, bytes.size(), cursor, number, QTU64, begin, size,
                   diagnostic) ||
        size != 8)
      return false;
    value = readU64BE(bytes, begin);
    return true;
  };
  if (!readU32Field(5, decoded.abbaBlockCount) ||
      !readU32Field(6, decoded.wallGuardFractionNumerator) ||
      !readU32Field(7, decoded.wallGuardFractionDenominator) ||
      !readU32Field(8, decoded.madMultiplier) ||
      !readU32Field(9, decoded.hostEnvironmentRetryCap) ||
      !readU64Field(10, decoded.totalProcessLaunchCap) ||
      !readU64Field(11, decoded.totalGatewayInvocationCap) ||
      cursor != bytes.size())
    return failWith(diagnostic,
                    "qualification policy has missing or extra fields");
  if (!validateOptimizationSetQualificationPolicyV1(decoded, diagnostic))
    return false;
  if (encodeOptimizationSetQualificationPolicyV1(decoded) != bytes)
    return failWith(diagnostic, "qualification policy is not canonical bytes");
  policy = std::move(decoded);
  return true;
}

OptimizationConfiguration getAllOnOptimizationConfiguration() { return {}; }

OptimizationConfiguration getAllOffOptimizationConfiguration() {
  OptimizationConfiguration configuration;
  configuration.fixed.kind = OptimizationGroupSelectionKind::AllOff;
  configuration.cleanup.kind = OptimizationGroupSelectionKind::AllOff;
  return configuration;
}

OptimizationConfiguration getDisableOneOptimizationConfiguration(
    const OptimizationQualificationProposal &proposal, MechanismKey key,
    std::string *diagnostic) {
  OptimizationConfiguration configuration = getAllOnOptimizationConfiguration();
  if (findBinding(proposal.fixedBindings, key)) {
    configuration.fixed.kind = OptimizationGroupSelectionKind::DisableOne;
    configuration.fixed.disabledKey = key;
    return configuration;
  }
  if (findBinding(proposal.cleanupBindings, key)) {
    configuration.cleanup.kind = OptimizationGroupSelectionKind::DisableOne;
    configuration.cleanup.disabledKey = key;
    return configuration;
  }
  if (diagnostic)
    *diagnostic =
        "mechanism is absent from the optimization qualification proposal";
  return configuration;
}

bool validateOptimizationConfiguration(
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration, std::string *diagnostic) {
  if (!validateOptimizationQualificationProposal(proposal, diagnostic))
    return false;
  return validateGroupSelection(proposal.fixedBindings,
                                proposal.cleanupBindings, configuration.fixed,
                                diagnostic) &&
         validateGroupSelection(proposal.cleanupBindings,
                                proposal.fixedBindings, configuration.cleanup,
                                diagnostic);
}

std::optional<bool> isOptimizationMechanismEnabled(
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration, MechanismKey key,
    std::string *diagnostic) {
  if (!validateOptimizationConfiguration(proposal, configuration, diagnostic))
    return std::nullopt;
  const OptimizationGroupSelection *selection = nullptr;
  if (findBinding(proposal.fixedBindings, key))
    selection = &configuration.fixed;
  else if (findBinding(proposal.cleanupBindings, key))
    selection = &configuration.cleanup;
  else {
    if (diagnostic)
      *diagnostic =
          "mechanism is absent from the optimization qualification proposal";
    return std::nullopt;
  }
  switch (selection->kind) {
  case OptimizationGroupSelectionKind::AllOn:
    return true;
  case OptimizationGroupSelectionKind::AllOff:
    return false;
  case OptimizationGroupSelectionKind::DisableOne:
    return *selection->disabledKey != key;
  }
  if (diagnostic)
    *diagnostic = "unknown optimization group selection kind";
  return std::nullopt;
}

OptimizationSetHostEvaluation
evaluateOptimizationSetABBA(const std::vector<ABBASample> &samples) {
  constexpr size_t blockCount = 5;
  constexpr size_t positionsPerBlock = 4;
  if (samples.size() != blockCount * positionsPerBlock)
    return {std::nullopt,
            "ABBA evidence must contain exactly five four-sample blocks"};

  std::array<std::array<const ABBASample *, positionsPerBlock>, blockCount>
      blocks{};
  for (size_t index = 0; index < samples.size(); ++index) {
    const ABBASample &sample = samples[index];
    uint32_t expectedBlock = index / positionsPerBlock;
    uint8_t expectedPosition = index % positionsPerBlock;
    if (sample.blockIndex != expectedBlock ||
        static_cast<uint8_t>(sample.position) != expectedPosition)
      return {std::nullopt,
              "ABBA samples are missing, duplicated or out of canonical "
              "A-left/B-left/B-right/A-right order"};
    if (!sample.processSucceeded)
      return {std::nullopt,
              "non-success process terminal cannot enter ABBA evidence"};
    blocks[expectedBlock][expectedPosition] = &sample;
  }

  OptimizationSetHostDecision decision;
  decision.wall = evaluateMetric(
      blocks, [](const ABBASample &sample) { return sample.wallNs; });
  decision.peakRss = evaluateMetric(
      blocks, [](const ABBASample &sample) { return sample.peakRssBytes; });
  decision.accepted =
      decision.wall.nonRegressed && decision.peakRss.nonRegressed;
  decision.significantHostBenefit =
      decision.accepted && (decision.wall.significantImprovement ||
                            decision.peakRss.significantImprovement);
  return {std::move(decision), {}};
}

} // namespace wafer
