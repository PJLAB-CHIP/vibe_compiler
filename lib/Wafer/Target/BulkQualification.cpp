//===- BulkQualification.cpp - Canonical qualification artifacts ----===//

#include "BulkQualificationInternal.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace wafer::bulk_qualification_detail {

static constexpr llvm::StringLiteral kCurrentSpecSchema =
    "wafer-bulk-qualification-spec-v2";

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(llvm::errc::invalid_argument, detail);
}

std::string sha256(llvm::StringRef payload) {
  llvm::SHA256 hasher;
  hasher.update(payload);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static bool isDigest(llvm::StringRef value) {
  if (!value.starts_with("sha256:") || value.size() != 71)
    return false;
  return llvm::all_of(value.drop_front(7), [](char character) {
    return llvm::isHexDigit(character) &&
           !(character >= 'A' && character <= 'F');
  });
}

static void printCanonicalJSON(llvm::raw_ostream &stream,
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

static llvm::Error validateCanonicalArtifactPath(llvm::StringRef path,
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

static llvm::Expected<PhysicalTensorLayout>
parseLayout(llvm::StringRef spelling) {
  if (spelling == "tensor")
    return PhysicalTensorLayout::Tensor;
  if (spelling == "ntensor")
    return PhysicalTensorLayout::NTensor;
  if (spelling == "cx")
    return PhysicalTensorLayout::Cx;
  if (spelling == "ncx")
    return PhysicalTensorLayout::NCx;
  return invalid("unknown numeric tensor layout " + spelling);
}

static std::string physicalHex(llvm::ArrayRef<uint8_t> bytes) {
  return llvm::toHex(bytes, /*LowerCase=*/true);
}

static llvm::Expected<std::vector<uint8_t>>
parsePhysicalHex(llvm::StringRef value, llvm::StringRef field) {
  if (value.empty() || value.size() % 2 != 0 ||
      !llvm::all_of(value, [](char character) {
        return llvm::isHexDigit(character) &&
               !(character >= 'A' && character <= 'F');
      }))
    return invalid(field + " must be nonempty lowercase even-length hex");
  std::vector<uint8_t> bytes;
  bytes.reserve(value.size() / 2);
  auto nibble = [](char character) -> uint8_t {
    if (character >= '0' && character <= '9')
      return static_cast<uint8_t>(character - '0');
    return static_cast<uint8_t>(character - 'a' + 10);
  };
  for (size_t index = 0; index < value.size(); index += 2)
    bytes.push_back(static_cast<uint8_t>((nibble(value[index]) << 4) |
                                         nibble(value[index + 1])));
  return bytes;
}

llvm::json::Object specJSON(const BulkQualificationSpec &spec) {
  llvm::json::Object object{
      {"schema", kCurrentSpecSchema},
      {"format", stringifyLogicalFormat(spec.getFormat())},
      {"m", static_cast<int64_t>(spec.getM())},
      {"k", static_cast<int64_t>(spec.getK())},
      {"n", static_cast<int64_t>(spec.getN())},
      {"batch_count", static_cast<int64_t>(spec.getBatchCount())},
      {"lhs_layout", stringifyPhysicalTensorLayout(spec.getLHSLayout())},
      {"rhs_layout", stringifyPhysicalTensorLayout(spec.getRHSLayout())},
      {"destination_layout",
       stringifyPhysicalTensorLayout(spec.getDestinationLayout())},
      {"seed", static_cast<int64_t>(spec.getSeed())},
  };
  if (spec.hasExplicitPhysicalPayload()) {
    object["lhs_physical"] = physicalHex(spec.getLHSPhysicalPayload());
    object["rhs_physical"] = physicalHex(spec.getRHSPhysicalPayload());
    object["destination_template_physical"] =
        physicalHex(spec.getDestinationTemplatePhysicalPayload());
  }
  return object;
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
  llvm::Expected<llvm::StringRef> schema =
      requireString(object, "schema", "bulk qualification spec");
  if (!schema)
    return schema.takeError();
  if (*schema != kCurrentSpecSchema)
    return invalid("bulk qualification spec schema mismatch");
  const bool explicitPayload = object.get("lhs_physical") ||
                               object.get("rhs_physical") ||
                               object.get("destination_template_physical");
  if (explicitPayload) {
    if (llvm::Error error = requireFields(
            object,
            {"schema", "format", "m", "k", "n", "batch_count", "lhs_layout",
             "rhs_layout", "destination_layout", "seed", "lhs_physical",
             "rhs_physical", "destination_template_physical"},
            "bulk qualification spec"))
      return std::move(error);
  } else if (llvm::Error error = requireFields(
                 object,
                 {"schema", "format", "m", "k", "n", "batch_count",
                  "lhs_layout", "rhs_layout", "destination_layout", "seed"},
                 "bulk qualification spec")) {
    return std::move(error);
  }
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
          takeExpectedErrors(formatText, m, k, n, batch, lhsLayoutText,
                             rhsLayoutText, destinationLayoutText, seed))
    return error;
  llvm::Expected<LogicalFormat> format = parseLogicalFormat(*formatText);
  llvm::Expected<PhysicalTensorLayout> lhsLayout = parseLayout(*lhsLayoutText);
  llvm::Expected<PhysicalTensorLayout> rhsLayout = parseLayout(*rhsLayoutText);
  llvm::Expected<PhysicalTensorLayout> destinationLayout =
      parseLayout(*destinationLayoutText);
  if (llvm::Error error =
          takeExpectedErrors(format, lhsLayout, rhsLayout, destinationLayout))
    return error;
  if (!explicitPayload)
    return BulkQualificationSpec::create(*format, *m, *k, *n, *batch,
                                         *lhsLayout, *rhsLayout,
                                         *destinationLayout, *seed);
  llvm::Expected<llvm::StringRef> lhsHex =
      requireString(object, "lhs_physical", "bulk qualification spec");
  llvm::Expected<llvm::StringRef> rhsHex =
      requireString(object, "rhs_physical", "bulk qualification spec");
  llvm::Expected<llvm::StringRef> destinationHex = requireString(
      object, "destination_template_physical", "bulk qualification spec");
  if (llvm::Error error = takeExpectedErrors(lhsHex, rhsHex, destinationHex))
    return error;
  llvm::Expected<std::vector<uint8_t>> lhsPhysical =
      parsePhysicalHex(*lhsHex, "lhs_physical");
  llvm::Expected<std::vector<uint8_t>> rhsPhysical =
      parsePhysicalHex(*rhsHex, "rhs_physical");
  llvm::Expected<std::vector<uint8_t>> destinationPhysical =
      parsePhysicalHex(*destinationHex, "destination_template_physical");
  if (llvm::Error error =
          takeExpectedErrors(lhsPhysical, rhsPhysical, destinationPhysical))
    return error;
  return BulkQualificationSpec::createWithPhysicalPayload(
      *format, *m, *k, *n, *batch, *lhsLayout, *rhsLayout, *destinationLayout,
      *seed, std::move(*lhsPhysical), std::move(*rhsPhysical),
      std::move(*destinationPhysical));
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

} // namespace wafer::bulk_qualification_detail

namespace wafer {

using namespace bulk_qualification_detail;

llvm::Expected<BulkQualificationSpec> BulkQualificationSpec::create(
    LogicalFormat format, uint64_t m, uint64_t k, uint64_t n,
    uint64_t batchCount, PhysicalTensorLayout lhsLayout,
    PhysicalTensorLayout rhsLayout, PhysicalTensorLayout destinationLayout,
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
  const PhysicalTensorLayout required =
      batchCount == 1 ? PhysicalTensorLayout::Cx : PhysicalTensorLayout::NCx;
  if (lhsLayout != required || rhsLayout != required ||
      destinationLayout != required)
    return invalid("unbatched qualification requires cx; batched requires ncx");
  BulkQualificationSpec provisional(format, m, k, n, batchCount, lhsLayout,
                                    rhsLayout, destinationLayout, seed,
                                    /*explicitPhysicalPayload=*/false, {}, {},
                                    {}, "");
  std::string canonical = canonicalJSON(specJSON(provisional));
  return BulkQualificationSpec(format, m, k, n, batchCount, lhsLayout,
                               rhsLayout, destinationLayout, seed,
                               /*explicitPhysicalPayload=*/false, {}, {}, {},
                               sha256(canonical));
}

llvm::Expected<BulkQualificationSpec>
BulkQualificationSpec::createWithPhysicalPayload(
    LogicalFormat format, uint64_t m, uint64_t k, uint64_t n,
    uint64_t batchCount, PhysicalTensorLayout lhsLayout,
    PhysicalTensorLayout rhsLayout, PhysicalTensorLayout destinationLayout,
    uint64_t seed, std::vector<uint8_t> lhsPhysical,
    std::vector<uint8_t> rhsPhysical,
    std::vector<uint8_t> destinationTemplatePhysical) {
  llvm::Expected<BulkQualificationSpec> generated =
      create(format, m, k, n, batchCount, lhsLayout, rhsLayout,
             destinationLayout, seed);
  if (!generated)
    return generated.takeError();
  std::vector<uint64_t> lhsShape;
  std::vector<uint64_t> rhsShape;
  std::vector<uint64_t> destinationShape;
  if (batchCount > 1) {
    lhsShape.push_back(batchCount);
    rhsShape.push_back(batchCount);
    destinationShape.push_back(batchCount);
  }
  lhsShape.insert(lhsShape.end(), {m, k});
  rhsShape.insert(rhsShape.end(), {k, n});
  destinationShape.insert(destinationShape.end(), {m, n});
  llvm::Expected<NumericTensorKey> lhs =
      NumericTensorKey::create(format, lhsLayout, lhsShape);
  llvm::Expected<NumericTensorKey> rhs =
      NumericTensorKey::create(format, rhsLayout, rhsShape);
  llvm::Expected<NumericTensorKey> destination =
      NumericTensorKey::create(format, destinationLayout, destinationShape);
  if (llvm::Error error = takeExpectedErrors(lhs, rhs, destination))
    return error;
  llvm::Expected<uint64_t> lhsBytes = getBulkTensorPhysicalBytes(*lhs);
  llvm::Expected<uint64_t> rhsBytes = getBulkTensorPhysicalBytes(*rhs);
  llvm::Expected<uint64_t> destinationBytes =
      getBulkTensorPhysicalBytes(*destination);
  if (llvm::Error error =
          takeExpectedErrors(lhsBytes, rhsBytes, destinationBytes))
    return error;
  if (lhsPhysical.size() != *lhsBytes || rhsPhysical.size() != *rhsBytes ||
      destinationTemplatePhysical.size() != *destinationBytes)
    return invalid("explicit qualification payload byte geometry differs "
                   "from its tensor domain");
  BulkQualificationSpec provisional(
      format, m, k, n, batchCount, lhsLayout, rhsLayout, destinationLayout,
      seed, /*explicitPhysicalPayload=*/true, std::move(lhsPhysical),
      std::move(rhsPhysical), std::move(destinationTemplatePhysical), "");
  std::string canonical = canonicalJSON(specJSON(provisional));
  std::string digest = sha256(canonical);
  provisional.digest = std::move(digest);
  return provisional;
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

} // namespace wafer
