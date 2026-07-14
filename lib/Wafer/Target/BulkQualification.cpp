//===- BulkQualification.cpp - Three-stage bulk backend admission -------===//

#include "Wafer/Target/BulkQualification.h"

#include "BulkTensorNumericInternal.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace wafer {
namespace {

constexpr llvm::StringLiteral kSpecSchema = "wafer-bulk-qualification-spec-v1";
constexpr llvm::StringLiteral kCalibrationSchema = "wafer-bulk-calibration-v1";
constexpr llvm::StringLiteral kPolicySchema = "wafer-bulk-frozen-policy-v1";
constexpr llvm::StringLiteral kFinalSchema =
    "wafer-bulk-qualification-record-v1";
constexpr llvm::StringLiteral kValueDomain =
    "deterministic-finite-f32-exact-inputs-v1";
constexpr llvm::StringLiteral kTargetComparator = "raw-exact-v1";
constexpr llvm::StringLiteral kBackendComparator = "absolute-relative-v1";
constexpr llvm::StringLiteral kProofBasis =
    "finite-calibration-held-out-exact-payload-v1";

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(llvm::errc::invalid_argument, detail);
}

template <typename... Values>
llvm::Error takeExpectedErrors(Values &...values) {
  llvm::Error errors = llvm::Error::success();
  auto take = [&](auto &value) {
    if (!value)
      errors = llvm::joinErrors(std::move(errors), value.takeError());
  };
  (take(values), ...);
  return errors;
}

std::string sha256(llvm::StringRef payload) {
  llvm::SHA256 hasher;
  hasher.update(payload);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

bool isDigest(llvm::StringRef value) {
  if (!value.starts_with("sha256:") || value.size() != 71)
    return false;
  return llvm::all_of(value.drop_front(7), [](char character) {
    return llvm::isHexDigit(character) &&
           !(character >= 'A' && character <= 'F');
  });
}

void printCanonicalJSON(llvm::raw_ostream &stream,
                        const llvm::json::Value &value) {
  if (const llvm::json::Object *object = value.getAsObject()) {
    std::vector<llvm::StringRef> keys;
    keys.reserve(object->size());
    for (const auto &entry : *object)
      keys.push_back(entry.first);
    llvm::sort(keys);
    stream << '{';
    for (size_t index = 0; index < keys.size(); ++index) {
      if (index != 0)
        stream << ',';
      stream << llvm::json::Value(keys[index]) << ':';
      printCanonicalJSON(stream, *object->get(keys[index]));
    }
    stream << '}';
    return;
  }
  if (const llvm::json::Array *array = value.getAsArray()) {
    stream << '[';
    for (size_t index = 0; index < array->size(); ++index) {
      if (index != 0)
        stream << ',';
      printCanonicalJSON(stream, (*array)[index]);
    }
    stream << ']';
    return;
  }
  stream << value;
}

std::string canonicalJSON(const llvm::json::Value &value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  printCanonicalJSON(stream, value);
  stream << '\n';
  stream.flush();
  return result;
}

std::string canonicalJSON(llvm::json::Object &&object) {
  return canonicalJSON(llvm::json::Value(std::move(object)));
}

struct ParsedJSON {
  llvm::json::Value value;
  std::string canonical;
  std::string digest;
};

llvm::Error validateCanonicalArtifactPath(llvm::StringRef path,
                                          bool mustExist) {
  if (!llvm::sys::path::is_absolute(path) || path.empty())
    return invalid("qualification artifact paths must be absolute");
  llvm::SmallString<512> normalized(path);
  llvm::sys::path::remove_dots(normalized, /*remove_dot_dot=*/true);
  if (normalized != path)
    return invalid("qualification artifact path contains an alias component");
  llvm::SmallString<512> resolved;
  if (mustExist) {
    if (std::error_code error = llvm::sys::fs::real_path(path, resolved))
      return llvm::errorCodeToError(error);
    if (resolved != path)
      return invalid("qualification artifact path resolves through an alias");
    return llvm::Error::success();
  }
  llvm::StringRef parent = llvm::sys::path::parent_path(path);
  if (parent.empty())
    return invalid("qualification output path has no parent directory");
  if (std::error_code error = llvm::sys::fs::real_path(parent, resolved))
    return llvm::errorCodeToError(error);
  if (resolved != parent)
    return invalid("qualification output parent resolves through an alias");
  return llvm::Error::success();
}

llvm::Expected<ParsedJSON> loadCanonicalJSON(llvm::StringRef path) {
  if (llvm::Error error =
          validateCanonicalArtifactPath(path, /*mustExist=*/true))
    return std::move(error);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/true,
                                  /*RequiresNullTerminator=*/true);
  if (!buffer)
    return llvm::errorCodeToError(buffer.getError());
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return invalid("invalid JSON in " + path + ": " +
                   llvm::toString(parsed.takeError()));
  std::string canonical = canonicalJSON(*parsed);
  if ((*buffer)->getBuffer() != canonical)
    return invalid("JSON artifact is not in canonical form: " + path);
  return ParsedJSON{std::move(*parsed), canonical, sha256(canonical)};
}

llvm::Error requireFields(const llvm::json::Object &object,
                          std::initializer_list<llvm::StringRef> fields,
                          llvm::StringRef context) {
  if (object.size() != fields.size())
    return invalid(context + " has unknown, missing or duplicate fields");
  for (llvm::StringRef field : fields)
    if (!object.get(field))
      return invalid(context + " is missing field " + field);
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value || value->empty())
    return invalid(context + " field " + key + " must be a nonempty string");
  return *value;
}

llvm::Expected<uint64_t> requireUnsigned(const llvm::json::Object &object,
                                         llvm::StringRef key,
                                         llvm::StringRef context) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value || *value < 0)
    return invalid(context + " field " + key + " must be a nonnegative int64");
  return static_cast<uint64_t>(*value);
}

llvm::Expected<double>
requireFiniteNonnegative(const llvm::json::Object &object, llvm::StringRef key,
                         llvm::StringRef context) {
  std::optional<double> value = object.getNumber(key);
  if (!value || !std::isfinite(*value) || *value < 0.0)
    return invalid(context + " field " + key +
                   " must be finite and nonnegative");
  return *value;
}

llvm::Expected<std::string> requireDigest(const llvm::json::Object &object,
                                          llvm::StringRef key,
                                          llvm::StringRef context) {
  llvm::Expected<llvm::StringRef> value = requireString(object, key, context);
  if (!value)
    return value.takeError();
  if (!isDigest(*value))
    return invalid(context + " field " + key + " is not lowercase SHA-256");
  return value->str();
}

llvm::Expected<NumericTensorLayout> parseLayout(llvm::StringRef spelling) {
  if (spelling == "tensor")
    return NumericTensorLayout::Tensor;
  if (spelling == "ntensor")
    return NumericTensorLayout::NTensor;
  if (spelling == "cx")
    return NumericTensorLayout::Cx;
  if (spelling == "ncx")
    return NumericTensorLayout::NCx;
  return invalid("unknown numeric tensor layout " + spelling);
}

llvm::json::Object specJSON(const BulkQualificationSpec &spec) {
  return llvm::json::Object{
      {"schema", kSpecSchema},
      {"format", stringifyLogicalFormat(spec.getFormat())},
      {"m", static_cast<int64_t>(spec.getM())},
      {"k", static_cast<int64_t>(spec.getK())},
      {"n", static_cast<int64_t>(spec.getN())},
      {"batch_count", static_cast<int64_t>(spec.getBatchCount())},
      {"lhs_layout", stringifyNumericTensorLayout(spec.getLHSLayout())},
      {"rhs_layout", stringifyNumericTensorLayout(spec.getRHSLayout())},
      {"destination_layout",
       stringifyNumericTensorLayout(spec.getDestinationLayout())},
      {"seed", static_cast<int64_t>(spec.getSeed())},
  };
}

llvm::json::Object
environmentJSON(const BulkExecutionEnvironment &environment) {
  const BulkBackendIdentity &backend = environment.getBackend();
  return llvm::json::Object{
      {"schema", "wafer-bulk-execution-environment-v1"},
      {"backend_name", backend.getName()},
      {"backend_version", backend.getVersion()},
      {"backend_commit", backend.getCommit()},
      {"dependency_record_digest", backend.getDependencyRecordDigest()},
      {"library_digest", backend.getLibraryDigest()},
      {"backend_digest", backend.getDigest()},
      {"host_cpu_name", environment.getHostCPUName()},
      {"host_features_digest", environment.getHostFeaturesDigest()},
      {"host_platform_digest", environment.getHostPlatformDigest()},
      {"effective_isa", environment.getEffectiveISA()},
      {"fenv_round",
       static_cast<int64_t>(environment.getFloatingRoundingMode())},
      {"mxcsr_control", static_cast<int64_t>(environment.getMXCSR())},
      {"thread_runtime", environment.getThreadRuntime()},
      {"worker_count", 1},
      {"max_cpu_isa_policy", "isa-default"},
      {"cpu_isa_hints_policy", "no-hints"},
      {"primitive_cache_capacity", 0},
      {"environment_digest", environment.getDigest()},
  };
}

std::string
environmentRecordDigest(const BulkExecutionEnvironment &environment) {
  return sha256(canonicalJSON(environmentJSON(environment)));
}

llvm::Error validateEnvironmentJSON(const llvm::json::Object &object,
                                    llvm::StringRef expectedEnvironmentDigest,
                                    llvm::StringRef expectedBackendDigest) {
  if (llvm::Error error = requireFields(
          object,
          {"schema", "backend_name", "backend_version", "backend_commit",
           "dependency_record_digest", "library_digest", "backend_digest",
           "host_cpu_name", "host_features_digest", "host_platform_digest",
           "effective_isa", "fenv_round", "mxcsr_control", "thread_runtime",
           "worker_count", "max_cpu_isa_policy", "cpu_isa_hints_policy",
           "primitive_cache_capacity", "environment_digest"},
          "bulk execution environment"))
    return error;
  auto string = [&](llvm::StringRef name) {
    return requireString(object, name, "bulk execution environment");
  };
  llvm::Expected<llvm::StringRef> schema = string("schema");
  llvm::Expected<llvm::StringRef> backendName = string("backend_name");
  llvm::Expected<llvm::StringRef> backendVersion = string("backend_version");
  llvm::Expected<llvm::StringRef> backendCommit = string("backend_commit");
  llvm::Expected<std::string> dependencyDigest = requireDigest(
      object, "dependency_record_digest", "bulk execution environment");
  llvm::Expected<std::string> libraryDigest =
      requireDigest(object, "library_digest", "bulk execution environment");
  llvm::Expected<std::string> backendDigest =
      requireDigest(object, "backend_digest", "bulk execution environment");
  llvm::Expected<llvm::StringRef> cpuName = string("host_cpu_name");
  llvm::Expected<std::string> featuresDigest = requireDigest(
      object, "host_features_digest", "bulk execution environment");
  llvm::Expected<std::string> platformDigest = requireDigest(
      object, "host_platform_digest", "bulk execution environment");
  llvm::Expected<llvm::StringRef> effectiveISA = string("effective_isa");
  std::optional<int64_t> fenvRound = object.getInteger("fenv_round");
  llvm::Expected<uint64_t> mxcsr =
      requireUnsigned(object, "mxcsr_control", "bulk execution environment");
  llvm::Expected<llvm::StringRef> threadRuntime = string("thread_runtime");
  llvm::Expected<uint64_t> workerCount =
      requireUnsigned(object, "worker_count", "bulk execution environment");
  llvm::Expected<llvm::StringRef> maxISA = string("max_cpu_isa_policy");
  llvm::Expected<llvm::StringRef> hints = string("cpu_isa_hints_policy");
  llvm::Expected<uint64_t> cache = requireUnsigned(
      object, "primitive_cache_capacity", "bulk execution environment");
  llvm::Expected<std::string> environmentDigest =
      requireDigest(object, "environment_digest", "bulk execution environment");
  if (llvm::Error error = takeExpectedErrors(
          schema, backendName, backendVersion, backendCommit, dependencyDigest,
          libraryDigest, backendDigest, cpuName, featuresDigest, platformDigest,
          effectiveISA, mxcsr, threadRuntime, workerCount, maxISA, hints, cache,
          environmentDigest))
    return error;
  if (!fenvRound)
    return invalid("bulk execution environment identity is incomplete");
  if (*schema != "wafer-bulk-execution-environment-v1" ||
      *backendName != "oneDNN" || *backendVersion != "3.12" ||
      backendCommit->size() != 40 || cpuName->empty() ||
      effectiveISA->empty() || *threadRuntime != "seq-caller-worker-v1" ||
      *fenvRound != FE_TONEAREST || (*mxcsr & UINT64_C(0x8040)) != 0 ||
      (*mxcsr & UINT64_C(0x6000)) != 0 ||
      (*mxcsr & UINT64_C(0x1f80)) != UINT64_C(0x1f80) || *workerCount != 1 ||
      *maxISA != "isa-default" || *hints != "no-hints" || *cache != 0 ||
      *environmentDigest != expectedEnvironmentDigest ||
      *backendDigest != expectedBackendDigest)
    return invalid("bulk execution environment policy/readback mismatch");
  return llvm::Error::success();
}

llvm::Expected<BulkQualificationSpec>
parseSpecObject(const llvm::json::Object &object) {
  if (llvm::Error error = requireFields(
          object,
          {"schema", "format", "m", "k", "n", "batch_count", "lhs_layout",
           "rhs_layout", "destination_layout", "seed"},
          "bulk qualification spec"))
    return std::move(error);
  llvm::Expected<llvm::StringRef> schema =
      requireString(object, "schema", "bulk qualification spec");
  llvm::Expected<llvm::StringRef> formatText =
      requireString(object, "format", "bulk qualification spec");
  llvm::Expected<uint64_t> m =
      requireUnsigned(object, "m", "bulk qualification spec");
  llvm::Expected<uint64_t> k =
      requireUnsigned(object, "k", "bulk qualification spec");
  llvm::Expected<uint64_t> n =
      requireUnsigned(object, "n", "bulk qualification spec");
  llvm::Expected<uint64_t> batch =
      requireUnsigned(object, "batch_count", "bulk qualification spec");
  llvm::Expected<llvm::StringRef> lhsLayoutText =
      requireString(object, "lhs_layout", "bulk qualification spec");
  llvm::Expected<llvm::StringRef> rhsLayoutText =
      requireString(object, "rhs_layout", "bulk qualification spec");
  llvm::Expected<llvm::StringRef> destinationLayoutText =
      requireString(object, "destination_layout", "bulk qualification spec");
  llvm::Expected<uint64_t> seed =
      requireUnsigned(object, "seed", "bulk qualification spec");
  if (llvm::Error error =
          takeExpectedErrors(schema, formatText, m, k, n, batch, lhsLayoutText,
                             rhsLayoutText, destinationLayoutText, seed))
    return error;
  if (*schema != kSpecSchema)
    return invalid("bulk qualification spec schema mismatch");
  llvm::Expected<LogicalFormat> format = parseLogicalFormat(*formatText);
  llvm::Expected<NumericTensorLayout> lhsLayout = parseLayout(*lhsLayoutText);
  llvm::Expected<NumericTensorLayout> rhsLayout = parseLayout(*rhsLayoutText);
  llvm::Expected<NumericTensorLayout> destinationLayout =
      parseLayout(*destinationLayoutText);
  if (llvm::Error error =
          takeExpectedErrors(format, lhsLayout, rhsLayout, destinationLayout))
    return error;
  return BulkQualificationSpec::create(*format, *m, *k, *n, *batch, *lhsLayout,
                                       *rhsLayout, *destinationLayout, *seed);
}

uint64_t splitMix64(uint64_t &state) {
  state += UINT64_C(0x9e3779b97f4a7c15);
  uint64_t value = state;
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

RawLogicalValue generatedValue(LogicalFormat format, uint64_t &state) {
  static constexpr uint32_t f32Values[] = {
      UINT32_C(0x00000000), UINT32_C(0x3e800000), UINT32_C(0xbe800000),
      UINT32_C(0x3f000000), UINT32_C(0xbf000000), UINT32_C(0x3f800000),
      UINT32_C(0xbf800000), UINT32_C(0x40000000), UINT32_C(0xc0000000),
  };
  static constexpr uint16_t f16Values[] = {
      UINT16_C(0x0000), UINT16_C(0x3400), UINT16_C(0xb400),
      UINT16_C(0x3800), UINT16_C(0xb800), UINT16_C(0x3c00),
      UINT16_C(0xbc00), UINT16_C(0x4000), UINT16_C(0xc000),
  };
  static constexpr uint16_t bf16Values[] = {
      UINT16_C(0x0000), UINT16_C(0x3e80), UINT16_C(0xbe80),
      UINT16_C(0x3f00), UINT16_C(0xbf00), UINT16_C(0x3f80),
      UINT16_C(0xbf80), UINT16_C(0x4000), UINT16_C(0xc000),
  };
  const size_t index = static_cast<size_t>(splitMix64(state) % 9);
  switch (format) {
  case LogicalFormat::F16:
    return {format, f16Values[index]};
  case LogicalFormat::BF16:
    return {format, bf16Values[index]};
  case LogicalFormat::F32:
    return {format, f32Values[index]};
  default:
    llvm_unreachable("qualification generator received unsupported format");
  }
}

std::vector<RawLogicalValue> generateValues(LogicalFormat format,
                                            uint64_t count, uint64_t &state) {
  std::vector<RawLogicalValue> values;
  values.reserve(static_cast<size_t>(count));
  for (uint64_t index = 0; index < count; ++index)
    values.push_back(generatedValue(format, state));
  return values;
}

} // namespace

llvm::Expected<BulkQualificationCase>
materializeBulkQualificationCase(BulkQualificationSpec spec,
                                 BulkNumericWorkBudget bulkBudget) {
  std::vector<uint64_t> lhsShape;
  std::vector<uint64_t> rhsShape;
  std::vector<uint64_t> destinationShape;
  const uint64_t rank = spec.getBatchCount() == 1 ? 2 : 3;
  if (rank == 3) {
    lhsShape.push_back(spec.getBatchCount());
    rhsShape.push_back(spec.getBatchCount());
    destinationShape.push_back(spec.getBatchCount());
  }
  lhsShape.insert(lhsShape.end(), {spec.getM(), spec.getK()});
  rhsShape.insert(rhsShape.end(), {spec.getK(), spec.getN()});
  destinationShape.insert(destinationShape.end(), {spec.getM(), spec.getN()});
  llvm::Expected<NumericTensorKey> lhs =
      NumericTensorKey::create(spec.getFormat(), spec.getLHSLayout(), lhsShape);
  llvm::Expected<NumericTensorKey> rhs =
      NumericTensorKey::create(spec.getFormat(), spec.getRHSLayout(), rhsShape);
  llvm::Expected<NumericTensorKey> destination = NumericTensorKey::create(
      spec.getFormat(), spec.getDestinationLayout(), destinationShape);
  if (llvm::Error error = takeExpectedErrors(lhs, rhs, destination))
    return error;
  llvm::Expected<NumericGemmAxes> axes = getCanonicalNumericGemmAxes(rank);
  if (!axes)
    return axes.takeError();
  llvm::Expected<NumericCommandKey> key = NumericCommandKey::createNEGemm(
      TargetProfileId::waferTx81SingleCardKernelV1(), *lhs, *rhs, *destination,
      spec.getM(), spec.getK(), spec.getN(), spec.getBatchCount(), *axes);
  if (!key)
    return key.takeError();
  llvm::Expected<ResolvedNumericCommand> command = resolveNumericCommand(
      ModelProfileId::formalDeterministicV1(), std::move(*key));
  if (!command)
    return command.takeError();
  if (!command->isSupported())
    return invalid("qualification spec resolves to an unsupported command");

  llvm::Expected<uint64_t> lhsBytes = getBulkTensorPhysicalBytes(*lhs);
  llvm::Expected<uint64_t> rhsBytes = getBulkTensorPhysicalBytes(*rhs);
  llvm::Expected<uint64_t> destinationBytes =
      getBulkTensorPhysicalBytes(*destination);
  if (llvm::Error error =
          takeExpectedErrors(lhsBytes, rhsBytes, destinationBytes))
    return error;
  if (*lhsBytes > bulkBudget.getMaximumTotalBytes() ||
      *rhsBytes > bulkBudget.getMaximumTotalBytes() - *lhsBytes ||
      *destinationBytes >
          bulkBudget.getMaximumTotalBytes() - *lhsBytes - *rhsBytes)
    return invalid(
        "qualification physical tensors exceed the bulk byte budget");

  uint64_t state = spec.getSeed();
  std::vector<RawLogicalValue> lhsValues =
      generateValues(spec.getFormat(), lhs->getElementCount(), state);
  std::vector<RawLogicalValue> rhsValues =
      generateValues(spec.getFormat(), rhs->getElementCount(), state);
  llvm::Expected<BulkTensorStorage> lhsStorage =
      packBulkTensorLogicalValues(*lhs, lhsValues, UINT8_C(0xa5));
  llvm::Expected<BulkTensorStorage> rhsStorage =
      packBulkTensorLogicalValues(*rhs, rhsValues, UINT8_C(0xa5));
  std::vector<RawLogicalValue> destinationZeros(
      static_cast<size_t>(destination->getElementCount()),
      RawLogicalValue{spec.getFormat(), 0});
  llvm::Expected<BulkTensorStorage> destinationTemplate =
      packBulkTensorLogicalValues(*destination, destinationZeros,
                                  UINT8_C(0x5a));
  if (llvm::Error error =
          takeExpectedErrors(lhsStorage, rhsStorage, destinationTemplate))
    return error;
  std::vector<BulkTensorStorage> inputs;
  inputs.push_back(std::move(*lhsStorage));
  inputs.push_back(std::move(*rhsStorage));
  return BulkQualificationCase(std::move(spec), std::move(*command),
                               std::move(inputs),
                               std::move(*destinationTemplate));
}

namespace {

struct Comparison {
  bool rawExact = true;
  double maximumAbsoluteError = 0.0;
  double maximumRelativeError = 0.0;
};

const llvm::fltSemantics &getSemantics(LogicalFormat format) {
  switch (format) {
  case LogicalFormat::F16:
    return llvm::APFloat::IEEEhalf();
  case LogicalFormat::BF16:
    return llvm::APFloat::BFloat();
  case LogicalFormat::F32:
    return llvm::APFloat::IEEEsingle();
  default:
    llvm_unreachable("unsupported qualification floating format");
  }
}

double toDouble(RawLogicalValue value) {
  const LogicalFormatDescriptor &descriptor =
      *findLogicalFormatDescriptor(value.format);
  llvm::APFloat floating(
      getSemantics(value.format),
      llvm::APInt(descriptor.storageBits, value.bits, /*isSigned=*/false));
  return floating.convertToDouble();
}

llvm::Expected<Comparison>
compareOutputs(const FormalTensorNumericResult &formal,
               const BulkTensorStorage &backend) {
  llvm::Expected<std::vector<RawLogicalValue>> backendValues =
      unpackBulkTensorLogicalValues(backend);
  if (!backendValues)
    return backendValues.takeError();
  if (backendValues->size() != formal.values.size())
    return invalid("formal/backend qualification output count mismatch");
  Comparison comparison;
  for (auto [expected, actual] :
       llvm::zip_equal(formal.values, *backendValues)) {
    if (expected.format != actual.format)
      return invalid("formal/backend qualification output format mismatch");
    comparison.rawExact &= expected.bits == actual.bits;
    llvm::Expected<LogicalValueClassification> expectedClass =
        classifyRawLogicalValue(expected, NonCanonicalEncodingPolicy::Reject);
    llvm::Expected<LogicalValueClassification> actualClass =
        classifyRawLogicalValue(actual, NonCanonicalEncodingPolicy::Reject);
    if (llvm::Error error = takeExpectedErrors(expectedClass, actualClass))
      return error;
    if (expectedClass->valueClass != actualClass->valueClass ||
        expectedClass->negative != actualClass->negative)
      return invalid("formal/backend special value classification mismatch");
    if (expectedClass->valueClass == LogicalValueClass::Infinity ||
        expectedClass->valueClass == LogicalValueClass::QuietNaN ||
        expectedClass->valueClass == LogicalValueClass::SignalingNaN)
      continue;
    const double expectedValue = toDouble(expected);
    const double actualValue = toDouble(actual);
    const double absolute = std::abs(actualValue - expectedValue);
    const double relative =
        absolute /
        std::max(std::abs(expectedValue), std::numeric_limits<double>::min());
    comparison.maximumAbsoluteError =
        std::max(comparison.maximumAbsoluteError, absolute);
    comparison.maximumRelativeError =
        std::max(comparison.maximumRelativeError, relative);
  }
  return comparison;
}

llvm::json::Object flagsJSON(FormalNumericExceptionFlags flags) {
  return llvm::json::Object{{"invalid", flags.invalid},
                            {"div_by_zero", flags.divByZero},
                            {"overflow", flags.overflow},
                            {"underflow", flags.underflow},
                            {"inexact", flags.inexact}};
}

llvm::Expected<FormalNumericExceptionFlags>
parseFlags(const llvm::json::Object &object) {
  if (llvm::Error error = requireFields(
          object,
          {"invalid", "div_by_zero", "overflow", "underflow", "inexact"},
          "formal flags"))
    return std::move(error);
  auto get = [&](llvm::StringRef name) -> llvm::Expected<bool> {
    std::optional<bool> value = object.getBoolean(name);
    if (!value)
      return invalid("formal flag " + name + " must be boolean");
    return *value;
  };
  llvm::Expected<bool> invalidFlag = get("invalid");
  llvm::Expected<bool> divByZero = get("div_by_zero");
  llvm::Expected<bool> overflow = get("overflow");
  llvm::Expected<bool> underflow = get("underflow");
  llvm::Expected<bool> inexact = get("inexact");
  if (llvm::Error error = takeExpectedErrors(invalidFlag, divByZero, overflow,
                                             underflow, inexact))
    return error;
  return FormalNumericExceptionFlags{*invalidFlag, *divByZero, *overflow,
                                     *underflow, *inexact};
}

struct QualificationRun {
  BulkQualificationCase testCase;
  FormalTensorNumericResult formal;
  detail::UnqualifiedBulkExecutionResult backend;
  Comparison comparison;
  std::string formalOutputDigest;
};

llvm::Expected<QualificationRun> runQualification(
    const BulkExecutionEnvironment &environment, BulkQualificationCase testCase,
    FormalNumericWorkBudget formalBudget, BulkNumericWorkBudget bulkBudget) {
  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      unpackBulkTensorLogicalValues(testCase.getInputs()[0]);
  llvm::Expected<std::vector<RawLogicalValue>> rhs =
      unpackBulkTensorLogicalValues(testCase.getInputs()[1]);
  if (llvm::Error error = takeExpectedErrors(lhs, rhs))
    return error;
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*lhs, *rhs};
  FormalNumericExecutionContext context;
  llvm::Expected<FormalTensorNumericResult> formal = executeFormalTensorNumeric(
      context, testCase.getCommand(), views, formalBudget);
  if (!formal)
    return formal.takeError();
  llvm::Expected<detail::UnqualifiedBulkExecutionResult> backend =
      detail::executeBulkTensorForQualification(
          environment, testCase.getCommand(), testCase.getInputs(),
          testCase.getDestinationTemplate(), bulkBudget);
  if (!backend)
    return backend.takeError();
  llvm::Expected<Comparison> comparison =
      compareOutputs(*formal, backend->destination);
  if (!comparison)
    return comparison.takeError();
  llvm::Expected<BulkTensorStorage> formalStorage =
      packBulkTensorLogicalValues(testCase.getDestinationTemplate().getKey(),
                                  formal->values, UINT8_C(0x5a));
  if (!formalStorage)
    return formalStorage.takeError();
  return QualificationRun{std::move(testCase), std::move(*formal),
                          std::move(*backend), *comparison,
                          computeBulkTensorStorageDigest(*formalStorage)};
}

llvm::Error publishNoReplace(llvm::StringRef path, llvm::StringRef content) {
  if (llvm::Error error =
          validateCanonicalArtifactPath(path, /*mustExist=*/false))
    return error;
  std::string candidate =
      (path + ".candidate." + llvm::Twine(static_cast<uint64_t>(::getpid())))
          .str();
  int descriptor =
      ::open(candidate.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (descriptor < 0)
    return llvm::errorCodeToError(
        std::error_code(errno, std::generic_category()));
  auto cleanup = llvm::make_scope_exit([&] {
    ::close(descriptor);
    ::unlink(candidate.c_str());
  });
  const char *data = content.data();
  size_t remaining = content.size();
  while (remaining != 0) {
    ssize_t written = ::write(descriptor, data, remaining);
    if (written < 0)
      return llvm::errorCodeToError(
          std::error_code(errno, std::generic_category()));
    data += written;
    remaining -= static_cast<size_t>(written);
  }
  if (::fsync(descriptor) != 0)
    return llvm::errorCodeToError(
        std::error_code(errno, std::generic_category()));
  if (::link(candidate.c_str(), path.str().c_str()) != 0)
    return llvm::errorCodeToError(
        std::error_code(errno, std::generic_category()));
  return llvm::Error::success();
}

llvm::Expected<BulkQualificationKind>
parseQualificationKind(llvm::StringRef spelling) {
  if (spelling == "bit-exact")
    return BulkQualificationKind::BitExact;
  if (spelling == "profile-bounded")
    return BulkQualificationKind::ProfileBounded;
  return invalid("unknown bulk qualification kind " + spelling);
}

bool sameDomain(const BulkQualificationSpec &lhs,
                const BulkQualificationSpec &rhs) {
  return lhs.getFormat() == rhs.getFormat() && lhs.getM() == rhs.getM() &&
         lhs.getK() == rhs.getK() && lhs.getN() == rhs.getN() &&
         lhs.getBatchCount() == rhs.getBatchCount() &&
         lhs.getLHSLayout() == rhs.getLHSLayout() &&
         lhs.getRHSLayout() == rhs.getRHSLayout() &&
         lhs.getDestinationLayout() == rhs.getDestinationLayout();
}

} // namespace

llvm::Expected<BulkQualificationSpec> BulkQualificationSpec::create(
    LogicalFormat format, uint64_t m, uint64_t k, uint64_t n,
    uint64_t batchCount, NumericTensorLayout lhsLayout,
    NumericTensorLayout rhsLayout, NumericTensorLayout destinationLayout,
    uint64_t seed) {
  if (format != LogicalFormat::F16 && format != LogicalFormat::BF16 &&
      format != LogicalFormat::F32)
    return invalid("bulk qualification only accepts f16, bf16 or f32");
  if (m == 0 || k == 0 || n == 0 || batchCount == 0 ||
      m > std::numeric_limits<uint16_t>::max() ||
      k > std::numeric_limits<uint16_t>::max() ||
      n > std::numeric_limits<uint16_t>::max() ||
      batchCount > std::numeric_limits<uint16_t>::max() ||
      seed > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return invalid("bulk qualification dimensions must be positive uint16 and "
                   "seed must fit int64");
  const NumericTensorLayout required =
      batchCount == 1 ? NumericTensorLayout::Cx : NumericTensorLayout::NCx;
  if (lhsLayout != required || rhsLayout != required ||
      destinationLayout != required)
    return invalid("unbatched qualification requires cx; batched requires ncx");
  BulkQualificationSpec provisional(format, m, k, n, batchCount, lhsLayout,
                                    rhsLayout, destinationLayout, seed, "");
  std::string canonical = canonicalJSON(specJSON(provisional));
  return BulkQualificationSpec(format, m, k, n, batchCount, lhsLayout,
                               rhsLayout, destinationLayout, seed,
                               sha256(canonical));
}

llvm::Expected<BulkQualificationSpec>
loadBulkQualificationSpec(llvm::StringRef path) {
  llvm::Expected<ParsedJSON> parsed = loadCanonicalJSON(path);
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *object = parsed->value.getAsObject();
  if (!object)
    return invalid("bulk qualification spec root must be an object");
  llvm::Expected<BulkQualificationSpec> spec = parseSpecObject(*object);
  if (!spec)
    return spec.takeError();
  if (spec->getDigest() != parsed->digest)
    return invalid("bulk qualification spec digest readback mismatch");
  return spec;
}

llvm::Error writeBulkQualificationSpec(const BulkQualificationSpec &spec,
                                       llvm::StringRef path) {
  return publishNoReplace(path, canonicalJSON(specJSON(spec)));
}

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
  llvm::json::Object record{
      {"schema", kCalibrationSchema},
      {"adapter_digest", getBulkAdapterIdentityDigest()},
      {"spec", specJSON(run->testCase.getSpec())},
      {"spec_digest", run->testCase.getSpec().getDigest()},
      {"semantic_profile_digest",
       run->testCase.getCommand().getSemantics()->getDigest()},
      {"value_domain", kValueDomain},
      {"target_comparator", kTargetComparator},
      {"backend_comparator", kBackendComparator},
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
  return publishNoReplace(outputPath, canonicalJSON(std::move(record)));
}

llvm::Error freezeBulkBackendPolicy(llvm::StringRef calibrationPath,
                                    llvm::StringRef heldOutSpecPath,
                                    llvm::StringRef outputPath,
                                    BulkQualificationTolerance tolerance) {
  if (!std::isfinite(tolerance.maximumAbsoluteError) ||
      !std::isfinite(tolerance.maximumRelativeError) ||
      tolerance.maximumAbsoluteError < 0.0 ||
      tolerance.maximumRelativeError < 0.0)
    return invalid("bulk qualification tolerance must be finite/nonnegative");
  llvm::Expected<ParsedJSON> calibration = loadCanonicalJSON(calibrationPath);
  if (!calibration)
    return calibration.takeError();
  const llvm::json::Object *object = calibration->value.getAsObject();
  if (!object)
    return invalid("bulk calibration root must be an object");
  if (llvm::Error error = requireFields(*object,
                                        {"schema",
                                         "adapter_digest",
                                         "spec",
                                         "spec_digest",
                                         "semantic_profile_digest",
                                         "value_domain",
                                         "target_comparator",
                                         "backend_comparator",
                                         "backend_digest",
                                         "environment_digest",
                                         "environment",
                                         "environment_record_digest",
                                         "resolution_digest",
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
                                        "bulk calibration"))
    return error;
  llvm::Expected<llvm::StringRef> schema =
      requireString(*object, "schema", "bulk calibration");
  if (!schema)
    return schema.takeError();
  if (*schema != kCalibrationSchema)
    return invalid("bulk calibration schema mismatch");
  const llvm::json::Object *calibrationSpecObject = object->getObject("spec");
  if (!calibrationSpecObject)
    return invalid("bulk calibration spec must be an object");
  llvm::Expected<BulkQualificationSpec> calibrationSpec =
      parseSpecObject(*calibrationSpecObject);
  if (!calibrationSpec)
    return calibrationSpec.takeError();
  llvm::Expected<std::string> calibrationSpecDigest =
      requireDigest(*object, "spec_digest", "bulk calibration");
  llvm::Expected<std::string> adapterDigest =
      requireDigest(*object, "adapter_digest", "bulk calibration");
  llvm::Expected<std::string> semanticDigest =
      requireDigest(*object, "semantic_profile_digest", "bulk calibration");
  llvm::Expected<std::string> inputPayloadDigest =
      requireDigest(*object, "input_payload_digest", "bulk calibration");
  llvm::Expected<std::string> destinationTemplateDigest =
      requireDigest(*object, "destination_template_digest", "bulk calibration");
  llvm::Expected<std::string> formalOutputDigest =
      requireDigest(*object, "formal_output_digest", "bulk calibration");
  llvm::Expected<std::string> backendOutputDigest =
      requireDigest(*object, "backend_output_digest", "bulk calibration");
  llvm::Expected<std::string> descriptorDigest =
      requireDigest(*object, "resolved_descriptor_digest", "bulk calibration");
  llvm::Expected<llvm::StringRef> valueDomain =
      requireString(*object, "value_domain", "bulk calibration");
  llvm::Expected<llvm::StringRef> targetComparator =
      requireString(*object, "target_comparator", "bulk calibration");
  llvm::Expected<llvm::StringRef> backendComparator =
      requireString(*object, "backend_comparator", "bulk calibration");
  llvm::Expected<llvm::StringRef> implementation =
      requireString(*object, "implementation", "bulk calibration");
  llvm::Expected<double> observedAbsolute = requireFiniteNonnegative(
      *object, "maximum_absolute_error", "bulk calibration");
  llvm::Expected<double> observedRelative = requireFiniteNonnegative(
      *object, "maximum_relative_error", "bulk calibration");
  std::optional<bool> rawExact = object->getBoolean("raw_exact");
  const llvm::json::Object *flagsObject = object->getObject("formal_flags");
  llvm::Expected<uint64_t> matmul =
      requireUnsigned(*object, "matmul_invocations", "bulk calibration");
  llvm::Expected<uint64_t> reorder =
      requireUnsigned(*object, "reorder_invocations", "bulk calibration");
  llvm::Expected<uint64_t> backendFormalFMA =
      requireUnsigned(*object, "backend_formal_fma", "bulk calibration");
  if (llvm::Error error = takeExpectedErrors(
          calibrationSpecDigest, adapterDigest, semanticDigest,
          inputPayloadDigest, destinationTemplateDigest, formalOutputDigest,
          backendOutputDigest, descriptorDigest, valueDomain, targetComparator,
          backendComparator, implementation, observedAbsolute, observedRelative,
          matmul, reorder, backendFormalFMA))
    return error;
  if (!rawExact || !flagsObject)
    return invalid("bulk calibration comparison identity is incomplete");
  llvm::Expected<FormalNumericExceptionFlags> flags = parseFlags(*flagsObject);
  if (!flags)
    return flags.takeError();
  if (*calibrationSpecDigest != calibrationSpec->getDigest())
    return invalid("bulk calibration spec digest mismatch");
  if (*adapterDigest != getBulkAdapterIdentityDigest() ||
      *valueDomain != kValueDomain || *targetComparator != kTargetComparator ||
      *backendComparator != kBackendComparator || implementation->empty() ||
      implementation->contains_insensitive("ref") || *matmul != 1 ||
      *reorder > 1 || *backendFormalFMA != 0)
    return invalid("bulk calibration adapter/comparator/evidence mismatch");
  if (*rawExact && *formalOutputDigest != *backendOutputDigest)
    return invalid("raw-exact calibration output digests disagree");
  if (*observedAbsolute > tolerance.maximumAbsoluteError ||
      *observedRelative > tolerance.maximumRelativeError)
    return invalid("frozen tolerance is tighter than calibration evidence");

  llvm::Expected<BulkQualificationSpec> heldOut =
      loadBulkQualificationSpec(heldOutSpecPath);
  if (!heldOut)
    return heldOut.takeError();
  if (!sameDomain(*calibrationSpec, *heldOut) ||
      calibrationSpec->getSeed() == heldOut->getSeed() ||
      calibrationSpec->getDigest() == heldOut->getDigest())
    return invalid("calibration and held-out specs must share one domain but "
                   "have disjoint seeds/digests");
  llvm::Expected<BulkQualificationCase> heldOutCase =
      materializeBulkQualificationCase(
          *heldOut,
          BulkNumericWorkBudget::create(std::numeric_limits<uint64_t>::max(),
                                        std::numeric_limits<uint64_t>::max(),
                                        std::numeric_limits<uint64_t>::max()));
  if (!heldOutCase)
    return heldOutCase.takeError();
  llvm::Expected<std::string> backendDigest =
      requireDigest(*object, "backend_digest", "bulk calibration");
  llvm::Expected<std::string> environmentDigest =
      requireDigest(*object, "environment_digest", "bulk calibration");
  llvm::Expected<std::string> environmentArtifactDigest =
      requireDigest(*object, "environment_record_digest", "bulk calibration");
  llvm::Expected<std::string> resolutionDigest =
      requireDigest(*object, "resolution_digest", "bulk calibration");
  if (llvm::Error error =
          takeExpectedErrors(backendDigest, environmentDigest,
                             environmentArtifactDigest, resolutionDigest))
    return error;
  const llvm::json::Value *environmentValue = object->get("environment");
  const llvm::json::Object *environmentObject =
      environmentValue ? environmentValue->getAsObject() : nullptr;
  if (!environmentObject)
    return invalid("bulk calibration environment must be an object");
  if (llvm::Error error = validateEnvironmentJSON(
          *environmentObject, *environmentDigest, *backendDigest))
    return error;
  if (sha256(canonicalJSON(*environmentValue)) != *environmentArtifactDigest)
    return invalid("bulk calibration environment record digest mismatch");
  if (heldOutCase->getCommand().getDigest() != *resolutionDigest ||
      !heldOutCase->getCommand().getSemantics() ||
      heldOutCase->getCommand().getSemantics()->getDigest() != *semanticDigest)
    return invalid("held-out command changed from calibration semantics");
  llvm::json::Object policy{
      {"schema", kPolicySchema},
      {"adapter_digest", *adapterDigest},
      {"semantic_profile_digest", *semanticDigest},
      {"value_domain", *valueDomain},
      {"target_comparator", *targetComparator},
      {"backend_comparator", *backendComparator},
      {"proof_basis", kProofBasis},
      {"calibration_digest", calibration->digest},
      {"calibration_spec_digest", calibrationSpec->getDigest()},
      {"held_out_spec", specJSON(*heldOut)},
      {"held_out_spec_digest", heldOut->getDigest()},
      {"held_out_input_payload_digest",
       computeBulkTensorPayloadDigest(heldOutCase->getInputs())},
      {"held_out_destination_template_digest",
       computeBulkTensorStorageDigest(heldOutCase->getDestinationTemplate())},
      {"backend_digest", *backendDigest},
      {"environment_digest", *environmentDigest},
      {"environment_record_digest", *environmentArtifactDigest},
      {"resolution_digest", *resolutionDigest},
      {"qualification_kind",
       stringifyBulkQualificationKind(BulkQualificationKind::ProfileBounded)},
      {"maximum_absolute_error", tolerance.maximumAbsoluteError},
      {"maximum_relative_error", tolerance.maximumRelativeError},
      {"disjoint_proof", "same-domain-distinct-seed-and-digest-v1"},
  };
  return publishNoReplace(outputPath, canonicalJSON(std::move(policy)));
}

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
  if (*proof != "same-domain-distinct-seed-and-digest-v1")
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
      expectedBackendOutputDigest, kind, formalFlags);
}

} // namespace wafer
