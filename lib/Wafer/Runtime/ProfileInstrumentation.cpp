//===- ProfileInstrumentation.cpp - Verified profiler instrumentation readback
//------===//

#include "Wafer/Runtime/ProfileInstrumentation.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Target/TargetCall.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::runtime {
namespace {

constexpr llvm::StringLiteral kPlanSchema = "wafer-profile-plan";
constexpr llvm::StringLiteral kActivationSchema = "wafer-profile-activation";
constexpr llvm::StringLiteral kSiteMapSchema =
    "wafer-profile-target-call-site-map";
constexpr llvm::StringLiteral kSiteBasis =
    "verified-target-llvm-entry-reachable-physical-tile-target-call-preorder";
constexpr uint64_t kCountRecordBytes = WAFER_TX81_PROFILER_MIN_BUFFER_BYTES;
constexpr uint64_t kTraceRecordBytes = WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES;

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

bool exceedsJSONNesting(llvm::StringRef input, uint64_t maximum) {
  uint64_t depth = 0;
  bool inString = false;
  bool escaped = false;
  for (char character : input) {
    if (inString) {
      if (escaped)
        escaped = false;
      else if (character == '\\')
        escaped = true;
      else if (character == '"')
        inString = false;
      continue;
    }
    if (character == '"')
      inString = true;
    else if (character == '{' || character == '[') {
      if (++depth > maximum)
        return true;
    } else if ((character == '}' || character == ']') && depth != 0) {
      --depth;
    }
  }
  return false;
}

llvm::Error requireFields(const llvm::json::Object &object,
                          std::initializer_list<llvm::StringRef> required,
                          std::initializer_list<llvm::StringRef> optional,
                          llvm::StringRef context) {
  llvm::StringSet<> allowed;
  for (llvm::StringRef field : required)
    allowed.insert(field);
  for (llvm::StringRef field : optional)
    allowed.insert(field);
  for (const auto &member : object)
    if (!allowed.contains(member.first))
      return invalid(context + " contains unknown field '" +
                     member.first.str() + "'");
  for (llvm::StringRef field : required)
    if (object.find(field) == object.end())
      return invalid(context + " is missing field '" + field + "'");
  return llvm::Error::success();
}

llvm::Expected<std::string> requireString(const llvm::json::Object &object,
                                          llvm::StringRef field,
                                          llvm::StringRef context,
                                          const PackageParseLimits &limits) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value)
    return invalid(context + "." + field + " must be a string");
  if (value->empty() || value->size() > limits.maxStringBytes)
    return invalid(context + "." + field + " has invalid length");
  return value->str();
}

llvm::Expected<uint64_t> requireUnsigned(const llvm::json::Object &object,
                                         llvm::StringRef field,
                                         llvm::StringRef context) {
  std::optional<int64_t> value = object.getInteger(field);
  if (!value || *value < 0)
    return invalid(context + "." + field + " must be a non-negative integer");
  return static_cast<uint64_t>(*value);
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef field,
             llvm::StringRef context) {
  const llvm::json::Array *array = object.getArray(field);
  if (!array)
    return invalid(context + "." + field + " must be an array");
  return array;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Value &value, llvm::StringRef context) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(context + " must be an object");
  return object;
}

llvm::Error accountRecords(uint64_t amount, uint64_t &total,
                           const PackageParseLimits &limits) {
  if (amount > limits.maxRecords || total > limits.maxRecords - amount)
    return invalid("profile instrumentation exceeds record limit");
  total += amount;
  return llvm::Error::success();
}

struct LoadedJSONDocument {
  llvm::json::Value root;
  std::string digest;
};

llvm::Expected<LoadedJSONDocument>
loadJSONDocument(llvm::StringRef path, llvm::StringRef label,
                 const PackageParseLimits &limits) {
  if (llvm::sys::fs::get_file_type(path, /*Follow=*/false) !=
      llvm::sys::fs::file_type::regular_file)
    return invalid(label + " is not a regular file");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/true,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read " + label + ": " + path);
  llvm::StringRef contents = (*buffer)->getBuffer();
  if (contents.size() > limits.maxJSONBytes)
    return invalid(label + " exceeds JSON byte limit");
  if (exceedsJSONNesting(contents, limits.maxJSONNesting))
    return invalid(label + " exceeds JSON nesting limit");
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(contents);
  if (!parsed)
    return parsed.takeError();
  if (!parsed->getAsObject())
    return invalid(label + " must be a JSON object");
  llvm::SHA256 hasher;
  hasher.update(contents);
  return LoadedJSONDocument{
      std::move(*parsed),
      "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true)};
}

struct RawActivation {
  std::string manifestDigest;
  std::string planDigest;
  std::string siteMapDigest;
};

struct RawCapturePackage {
  ProfileCaptureKind capture = ProfileCaptureKind::Count;
  std::string packageReference;
  std::string manifestDigest;
  std::string recordABI;
  uint64_t recordBytes = 0;
};

struct RawPlan {
  ProfileStaticCostModel staticCostModel;
  std::vector<RawCapturePackage> capturePackages;
};

llvm::Expected<ProfileCaptureKind> parseCaptureKind(llvm::StringRef value,
                                                    llvm::StringRef context) {
  if (value == "count")
    return ProfileCaptureKind::Count;
  if (value == "trace")
    return ProfileCaptureKind::Trace;
  return invalid(context + " is not a supported capture kind");
}

uint64_t expectedRecordBytes(ProfileCaptureKind capture) {
  switch (capture) {
  case ProfileCaptureKind::Count:
    return kCountRecordBytes;
  case ProfileCaptureKind::Trace:
    return kTraceRecordBytes;
  }
  llvm_unreachable("unknown profile capture kind");
}

llvm::Error verifyHeader(const llvm::json::Object &root,
                         llvm::StringRef expectedSchema, bool hasTopology,
                         const PackageParseLimits &limits,
                         llvm::StringRef context);
bool isLowercaseSHA256(llvm::StringRef digest);

llvm::Expected<RawActivation>
parseActivation(const llvm::json::Object &root,
                const PackageParseLimits &limits) {
  if (llvm::Error error =
          requireFields(root, {"schema", "primary_manifest_sha256",
                               "metadata_sha256"},
                        {}, "profile activation"))
    return std::move(error);
  if (llvm::Error error =
          verifyHeader(root, kActivationSchema, /*hasTopology=*/false, limits,
                       "profile activation"))
    return std::move(error);
  llvm::Expected<std::string> primaryDigest = requireString(
      root, "primary_manifest_sha256", "profile activation", limits);
  if (!primaryDigest)
    return primaryDigest.takeError();
  const llvm::json::Object *metadata = root.getObject("metadata_sha256");
  if (!metadata)
    return invalid("profile activation.metadata_sha256 must be an object");
  if (llvm::Error error =
          requireFields(*metadata,
                        {kProfileInstrumentationPlanFileName,
                         kProfileInstrumentationSiteMapFileName},
                        {}, "profile activation.metadata_sha256"))
    return std::move(error);
  llvm::Expected<std::string> plan =
      requireString(*metadata, kProfileInstrumentationPlanFileName,
                    "profile activation.metadata_sha256", limits);
  if (!plan)
    return plan.takeError();
  llvm::Expected<std::string> siteMap =
      requireString(*metadata, kProfileInstrumentationSiteMapFileName,
                    "profile activation.metadata_sha256", limits);
  if (!siteMap)
    return siteMap.takeError();
  if (!isLowercaseSHA256(*primaryDigest) || !isLowercaseSHA256(*plan) ||
      !isLowercaseSHA256(*siteMap))
    return invalid("profile activation contains a malformed SHA-256 digest");
  return RawActivation{std::move(*primaryDigest), std::move(*plan),
                       std::move(*siteMap)};
}

llvm::Error verifyHeader(const llvm::json::Object &root,
                         llvm::StringRef expectedSchema, bool hasTopology,
                         const PackageParseLimits &limits,
                         llvm::StringRef context) {
  llvm::Expected<std::string> schema =
      requireString(root, "schema", context, limits);
  if (!schema)
    return schema.takeError();
  if (*schema != expectedSchema)
    return invalid(context + ".schema is not supported");
  if (hasTopology) {
    llvm::Expected<uint64_t> cardCount =
        requireUnsigned(root, "card_count", context);
    if (!cardCount)
      return cardCount.takeError();
    llvm::Expected<uint64_t> tileCount =
        requireUnsigned(root, "tile_count", context);
    if (!tileCount)
      return tileCount.takeError();
    if (*cardCount != static_cast<uint64_t>(kProfileInstrumentationCardCount) ||
        *tileCount != static_cast<uint64_t>(kProfileInstrumentationTileCount))
      return invalid(context + ".card_count/tile_count must be exactly 1/16");
  }
  return llvm::Error::success();
}

llvm::Expected<ProfileStaticCostModel>
parseStaticCostModel(const llvm::json::Value &value,
                     const PackageParseLimits &limits, uint64_t &totalRecords,
                     llvm::StringRef context);

llvm::Expected<RawPlan> parsePlan(const llvm::json::Object &root,
                                  const PackageParseLimits &limits,
                                  uint64_t &totalRecords) {
  if (llvm::Error error = requireFields(
          root,
          {"schema", "card_count", "tile_count", "site_map",
           "site_key_contract", "static_cost_model", "capture_packages"},
          {}, "profile plan"))
    return std::move(error);
  if (llvm::Error error =
          verifyHeader(root, kPlanSchema, /*hasTopology=*/true, limits,
                       "profile plan"))
    return std::move(error);

  llvm::Expected<std::string> siteMap =
      requireString(root, "site_map", "profile plan", limits);
  if (!siteMap)
    return siteMap.takeError();
  llvm::Expected<std::string> siteKeyContract =
      requireString(root, "site_key_contract", "profile plan", limits);
  if (!siteKeyContract)
    return siteKeyContract.takeError();
  if (*siteMap != kProfileInstrumentationSiteMapFileName)
    return invalid("profile plan metadata references are not canonical");
  if (*siteKeyContract != kProfileSiteKeyContract)
    return invalid("profile plan site-key contract is not supported");

  RawPlan plan;
  const llvm::json::Value *staticCost = root.get("static_cost_model");
  if (!staticCost)
    return invalid("profile plan.static_cost_model is missing");
  llvm::Expected<ProfileStaticCostModel> parsedStaticCost =
      parseStaticCostModel(*staticCost, limits, totalRecords,
                           "profile plan.static_cost_model");
  if (!parsedStaticCost)
    return parsedStaticCost.takeError();
  plan.staticCostModel = std::move(*parsedStaticCost);

  llvm::Expected<const llvm::json::Array *> captures =
      requireArray(root, "capture_packages", "profile plan");
  if (!captures)
    return captures.takeError();
  if ((*captures)->size() != 2)
    return invalid("profile plan must contain exactly two capture packages");
  if (llvm::Error error =
          accountRecords((*captures)->size(), totalRecords, limits))
    return std::move(error);
  const std::array<ProfileCaptureKind, 2> expectedOrder = {
      ProfileCaptureKind::Count, ProfileCaptureKind::Trace};
  for (auto [index, value] : llvm::enumerate(**captures)) {
    std::string context =
        "profile plan.capture_packages[" + std::to_string(index) + "]";
    llvm::Expected<const llvm::json::Object *> object =
        requireObject(value, context);
    if (!object)
      return object.takeError();
    if (llvm::Error error =
            requireFields(**object,
                          {"capture", "package_ref", "manifest_sha256",
                           "record_abi", "record_bytes"},
                          {}, context))
      return std::move(error);
    llvm::Expected<std::string> captureText =
        requireString(**object, "capture", context, limits);
    if (!captureText)
      return captureText.takeError();
    llvm::Expected<ProfileCaptureKind> capture =
        parseCaptureKind(*captureText, context + ".capture");
    if (!capture)
      return capture.takeError();
    llvm::Expected<std::string> reference =
        requireString(**object, "package_ref", context, limits);
    if (!reference)
      return reference.takeError();
    llvm::Expected<std::string> digest =
        requireString(**object, "manifest_sha256", context, limits);
    if (!digest)
      return digest.takeError();
    llvm::Expected<std::string> recordABI =
        requireString(**object, "record_abi", context, limits);
    if (!recordABI)
      return recordABI.takeError();
    llvm::Expected<uint64_t> recordBytes =
        requireUnsigned(**object, "record_bytes", context);
    if (!recordBytes)
      return recordBytes.takeError();
    if (*capture != expectedOrder[index])
      return invalid("profile plan capture packages are not in canonical "
                     "capture order");
    if (*recordBytes != expectedRecordBytes(*capture))
      return invalid(context + ".record_bytes is not the capture contract");
    if (*recordABI != kProfileRecordABI)
      return invalid(context + ".record_abi is not the capture contract");
    plan.capturePackages.push_back(
        {*capture, *reference, *digest, *recordABI, *recordBytes});
  }
  return plan;
}

bool isStaticCostKnowledge(llvm::StringRef value) {
  static constexpr std::array<llvm::StringLiteral, 4> values = {
      "known", "unavailable", "unsupported", "overflow"};
  return llvm::is_contained(values, value);
}

bool isStaticCostReason(llvm::StringRef value) {
  static constexpr std::array<llvm::StringLiteral, 17> values = {
      "none",
      "dynamic-loop-trip-count",
      "invalid-loop-step",
      "conditional-control-flow",
      "unsupported-control-flow",
      "unavailable-physical-geometry",
      "unavailable-resource-bytes",
      "missing-accepted-spm-offset",
      "invalid-accepted-spm-offset",
      "missing-accepted-ddr-offset",
      "invalid-accepted-ddr-offset",
      "unsupported-spm-root",
      "unresolved-noc-route",
      "invalid-execution-topology",
      "unsupported-instruction-semantics",
      "unsupported-compute-type",
      "arithmetic-overflow"};
  return llvm::is_contained(values, value);
}

llvm::Expected<uint64_t> requirePositiveUINT64(const llvm::json::Object &object,
                                               llvm::StringRef field,
                                               llvm::StringRef context) {
  const llvm::json::Value *raw = object.get(field);
  std::optional<uint64_t> value =
      raw ? raw->getAsUINT64() : std::optional<uint64_t>();
  if (!value || *value == 0)
    return invalid(context + "." + field +
                   " must be a positive unsigned integer");
  return *value;
}

llvm::Expected<ProfileStaticCostMetric>
parseStaticCostMetric(const llvm::json::Value &value,
                      const PackageParseLimits &limits,
                      llvm::StringRef context) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, context);
  if (!object)
    return object.takeError();
  if (llvm::Error error = requireFields(
          **object, {"knowledge", "value", "reason"}, {}, context))
    return std::move(error);

  llvm::Expected<std::string> knowledge =
      requireString(**object, "knowledge", context, limits);
  if (!knowledge)
    return knowledge.takeError();
  llvm::Expected<std::string> reason =
      requireString(**object, "reason", context, limits);
  if (!reason)
    return reason.takeError();
  if (!isStaticCostKnowledge(*knowledge))
    return invalid(context + ".knowledge is not supported");
  if (!isStaticCostReason(*reason))
    return invalid(context + ".reason is not supported");

  const llvm::json::Value *rawValue = (*object)->get("value");
  if (!rawValue)
    return invalid(context + ".value is missing");
  ProfileStaticCostMetric result;
  result.knowledge = *knowledge;
  result.reason = *reason;
  if (*knowledge == "known") {
    std::optional<llvm::StringRef> decimal = rawValue->getAsString();
    if (!decimal || decimal->empty() ||
        decimal->size() > limits.maxStringBytes ||
        (decimal->size() > 1 && decimal->front() == '0'))
      return invalid(context +
                     ".value must be a canonical uint64 decimal string");
    uint64_t parsed = 0;
    if (decimal->getAsInteger(10, parsed))
      return invalid(context +
                     ".value must be a canonical uint64 decimal string");
    if (*reason != "none")
      return invalid(context + " known metric must use reason 'none'");
    result.value = parsed;
  } else {
    if (!rawValue->getAsNull())
      return invalid(context + " non-known metric must use a null value");
    if (*reason == "none")
      return invalid(context + " non-known metric must carry a reason");
    if (*knowledge == "overflow" && *reason != "arithmetic-overflow")
      return invalid(context +
                     " overflow metric must use reason 'arithmetic-overflow'");
  }
  return result;
}

llvm::Expected<ProfileStaticDirectionalNoCWork>
parseStaticDirectionalNoCWork(const llvm::json::Value &value,
                              const PackageParseLimits &limits,
                              llvm::StringRef context) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, context);
  if (!object)
    return object.takeError();
  if (llvm::Error error = requireFields(
          **object, {"north", "east", "south", "west"}, {}, context))
    return std::move(error);

  ProfileStaticDirectionalNoCWork result;
  auto parse =
      [&](llvm::StringRef field) -> llvm::Expected<ProfileStaticCostMetric> {
    const llvm::json::Value *metric = (*object)->get(field);
    if (!metric)
      return invalid(context + "." + field + " is missing");
    return parseStaticCostMetric(*metric, limits,
                                 (context + "." + field).str());
  };
  llvm::Expected<ProfileStaticCostMetric> north = parse("north");
  if (!north)
    return north.takeError();
  llvm::Expected<ProfileStaticCostMetric> east = parse("east");
  if (!east)
    return east.takeError();
  llvm::Expected<ProfileStaticCostMetric> south = parse("south");
  if (!south)
    return south.takeError();
  llvm::Expected<ProfileStaticCostMetric> west = parse("west");
  if (!west)
    return west.takeError();
  result.north = std::move(*north);
  result.east = std::move(*east);
  result.south = std::move(*south);
  result.west = std::move(*west);
  return result;
}

llvm::Expected<ProfileStaticTileWork>
parseStaticTileWork(const llvm::json::Value &value,
                    const PackageParseLimits &limits, llvm::StringRef context) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, context);
  if (!object)
    return object.takeError();
  if (llvm::Error error = requireFields(
          **object,
          {"npu_f16_bf16_logical_ops", "npu_other_logical_ops",
           "vector_f16_bf16_logical_ops", "vector_f32_logical_ops",
           "vector_other_logical_ops", "ddr_read_bytes", "ddr_write_bytes",
           "spm_movement_bytes", "noc_transmit_bytes", "noc_receive_bytes",
           "directional_noc_transmit_bytes"},
          {}, context))
    return std::move(error);

  auto parseMetric =
      [&](llvm::StringRef field) -> llvm::Expected<ProfileStaticCostMetric> {
    const llvm::json::Value *metric = (*object)->get(field);
    if (!metric)
      return invalid(context + "." + field + " is missing");
    return parseStaticCostMetric(*metric, limits,
                                 (context + "." + field).str());
  };

  ProfileStaticTileWork result;
#define PARSE_STATIC_METRIC(JSON_NAME, MEMBER)                                 \
  do {                                                                         \
    llvm::Expected<ProfileStaticCostMetric> metric = parseMetric(JSON_NAME);   \
    if (!metric)                                                               \
      return metric.takeError();                                               \
    result.MEMBER = std::move(*metric);                                        \
  } while (false)
  PARSE_STATIC_METRIC("npu_f16_bf16_logical_ops", npuF16Bf16LogicalOps);
  PARSE_STATIC_METRIC("npu_other_logical_ops", npuOtherLogicalOps);
  PARSE_STATIC_METRIC("vector_f16_bf16_logical_ops", vectorF16Bf16LogicalOps);
  PARSE_STATIC_METRIC("vector_f32_logical_ops", vectorF32LogicalOps);
  PARSE_STATIC_METRIC("vector_other_logical_ops", vectorOtherLogicalOps);
  PARSE_STATIC_METRIC("ddr_read_bytes", ddrReadBytes);
  PARSE_STATIC_METRIC("ddr_write_bytes", ddrWriteBytes);
  PARSE_STATIC_METRIC("spm_movement_bytes", spmMovementBytes);
  PARSE_STATIC_METRIC("noc_transmit_bytes", nocTransmitBytes);
  PARSE_STATIC_METRIC("noc_receive_bytes", nocReceiveBytes);
#undef PARSE_STATIC_METRIC

  const llvm::json::Value *directional =
      (*object)->get("directional_noc_transmit_bytes");
  llvm::Expected<ProfileStaticDirectionalNoCWork> directionalWork =
      directional
          ? parseStaticDirectionalNoCWork(
                *directional, limits,
                (context + ".directional_noc_transmit_bytes").str())
          : llvm::Expected<ProfileStaticDirectionalNoCWork>(invalid(
                context + ".directional_noc_transmit_bytes is missing"));
  if (!directionalWork)
    return directionalWork.takeError();
  result.directionalNoCTransmitBytes = std::move(*directionalWork);
  return result;
}

llvm::Expected<ProfileStaticCostRates>
parseStaticCostRates(const llvm::json::Value &value, llvm::StringRef context) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, context);
  if (!object)
    return object.takeError();
  if (llvm::Error error = requireFields(
          **object,
          {"card_ddr_bytes_per_second", "directional_noc_bytes_per_second",
           "f16_bf16_npu_logical_ops_per_second_per_tile",
           "f16_bf16_vector_logical_ops_per_second_per_tile",
           "f32_vector_logical_ops_per_second_per_tile",
           "spm_movement_bytes_per_second"},
          {}, context))
    return std::move(error);

  ProfileStaticCostRates result;
#define PARSE_STATIC_RATE(JSON_NAME, MEMBER)                                   \
  do {                                                                         \
    llvm::Expected<uint64_t> rate =                                            \
        requirePositiveUINT64(**object, JSON_NAME, context);                   \
    if (!rate)                                                                 \
      return rate.takeError();                                                 \
    result.MEMBER = *rate;                                                     \
  } while (false)
  PARSE_STATIC_RATE("card_ddr_bytes_per_second", cardDDRBytesPerSecond);
  PARSE_STATIC_RATE("directional_noc_bytes_per_second",
                    directionalNoCBytesPerSecond);
  PARSE_STATIC_RATE("f16_bf16_npu_logical_ops_per_second_per_tile",
                    f16Bf16NPULogicalOpsPerSecondPerTile);
  PARSE_STATIC_RATE("f16_bf16_vector_logical_ops_per_second_per_tile",
                    f16Bf16VectorLogicalOpsPerSecondPerTile);
  PARSE_STATIC_RATE("f32_vector_logical_ops_per_second_per_tile",
                    f32VectorLogicalOpsPerSecondPerTile);
#undef PARSE_STATIC_RATE

  const llvm::json::Value *spm =
      (*object)->get("spm_movement_bytes_per_second");
  if (!spm || !spm->getAsNull())
    return invalid(context + ".spm_movement_bytes_per_second must be null");
  result.spmMovementBytesPerSecond = std::nullopt;
  return result;
}

llvm::Expected<ProfileStaticCostModel>
parseStaticCostModel(const llvm::json::Value &value,
                     const PackageParseLimits &limits, uint64_t &totalRecords,
                     llvm::StringRef context) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, context);
  if (!object)
    return object.takeError();
  if (llvm::Error error = requireFields(
          **object, {"model", "scope", "rates", "tiles"}, {}, context))
    return std::move(error);
  llvm::Expected<std::string> model =
      requireString(**object, "model", context, limits);
  if (!model)
    return model.takeError();
  llvm::Expected<std::string> scope =
      requireString(**object, "scope", context, limits);
  if (!scope)
    return scope.takeError();
  if (*model != kProfileStaticCostModelName)
    return invalid(context + ".model is not supported");
  if (*scope != kProfileStaticCostModelScope)
    return invalid(context + ".scope is not supported");

  const llvm::json::Value *ratesValue = (*object)->get("rates");
  if (!ratesValue)
    return invalid(context + ".rates is missing");
  llvm::Expected<ProfileStaticCostRates> rates =
      parseStaticCostRates(*ratesValue, (context + ".rates").str());
  if (!rates)
    return rates.takeError();
  llvm::Expected<const llvm::json::Array *> tiles =
      requireArray(**object, "tiles", context);
  if (!tiles)
    return tiles.takeError();
  if ((*tiles)->size() != static_cast<size_t>(kProfileInstrumentationTileCount))
    return invalid(context + ".tiles must contain all and only 16 Tiles");
  if (llvm::Error error =
          accountRecords((*tiles)->size() * 20, totalRecords, limits))
    return std::move(error);

  ProfileStaticCostModel result;
  result.model = std::move(*model);
  result.scope = std::move(*scope);
  result.rates = std::move(*rates);
  result.tiles.reserve((*tiles)->size());
  std::array<bool, kProfileInstrumentationTileCount> seenTileIds{};
  std::array<bool, kProfileInstrumentationTileCount> seenLaunchSlots{};
  for (auto [index, tileValue] : llvm::enumerate(**tiles)) {
    std::string tileContext =
        (context + ".tiles[" + llvm::Twine(index) + "]").str();
    llvm::Expected<const llvm::json::Object *> tileObject =
        requireObject(tileValue, tileContext);
    if (!tileObject)
      return tileObject.takeError();
    if (llvm::Error error = requireFields(
            **tileObject, {"card_id", "tile_id", "launch_slot", "work"}, {},
            tileContext))
      return std::move(error);
    llvm::Expected<uint64_t> cardId =
        requireUnsigned(**tileObject, "card_id", tileContext);
    if (!cardId)
      return cardId.takeError();
    llvm::Expected<uint64_t> tileId =
        requireUnsigned(**tileObject, "tile_id", tileContext);
    if (!tileId)
      return tileId.takeError();
    llvm::Expected<uint64_t> launchSlot =
        requireUnsigned(**tileObject, "launch_slot", tileContext);
    if (!launchSlot)
      return launchSlot.takeError();
    if (*cardId != 0 ||
        *tileId >= static_cast<uint64_t>(kProfileInstrumentationTileCount) ||
        *launchSlot >=
            static_cast<uint64_t>(kProfileInstrumentationTileCount) ||
        seenTileIds[*tileId] || seenLaunchSlots[*launchSlot])
      return invalid(context + ".tiles must contain unique Tiles and a "
                               "unique dense launch-slot domain");
    seenTileIds[*tileId] = true;
    seenLaunchSlots[*launchSlot] = true;
    const llvm::json::Value *workValue = (*tileObject)->get("work");
    if (!workValue)
      return invalid(tileContext + ".work is missing");
    llvm::Expected<ProfileStaticTileWork> work =
        parseStaticTileWork(*workValue, limits, tileContext + ".work");
    if (!work)
      return work.takeError();
    result.tiles.push_back({CardId(static_cast<int64_t>(*cardId)),
                            TileId(static_cast<int64_t>(*tileId)),
                            LaunchSlotId(static_cast<int64_t>(*launchSlot)),
                            std::move(*work)});
  }
  llvm::sort(result.tiles, [](const auto &lhs, const auto &rhs) {
    return lhs.launchSlot.getValue() < rhs.launchSlot.getValue();
  });
  return result;
}

void emitStaticCostMetric(llvm::json::OStream &json,
                          const ProfileStaticCostMetric &metric) {
  json.object([&] {
    json.attribute("knowledge", metric.knowledge);
    if (metric.value)
      json.attribute("value", std::to_string(*metric.value));
    else
      json.attribute("value", llvm::json::Value(nullptr));
    json.attribute("reason", metric.reason);
  });
}

void emitStaticTileWork(llvm::json::OStream &json,
                        const ProfileStaticTileWork &work) {
  auto emitMetric = [&](llvm::StringRef name,
                        const ProfileStaticCostMetric &metric) {
    json.attributeBegin(name);
    emitStaticCostMetric(json, metric);
    json.attributeEnd();
  };
  json.object([&] {
    emitMetric("npu_f16_bf16_logical_ops", work.npuF16Bf16LogicalOps);
    emitMetric("npu_other_logical_ops", work.npuOtherLogicalOps);
    emitMetric("vector_f16_bf16_logical_ops", work.vectorF16Bf16LogicalOps);
    emitMetric("vector_f32_logical_ops", work.vectorF32LogicalOps);
    emitMetric("vector_other_logical_ops", work.vectorOtherLogicalOps);
    emitMetric("ddr_read_bytes", work.ddrReadBytes);
    emitMetric("ddr_write_bytes", work.ddrWriteBytes);
    emitMetric("spm_movement_bytes", work.spmMovementBytes);
    emitMetric("noc_transmit_bytes", work.nocTransmitBytes);
    emitMetric("noc_receive_bytes", work.nocReceiveBytes);
    json.attributeObject("directional_noc_transmit_bytes", [&] {
      emitMetric("north", work.directionalNoCTransmitBytes.north);
      emitMetric("east", work.directionalNoCTransmitBytes.east);
      emitMetric("south", work.directionalNoCTransmitBytes.south);
      emitMetric("west", work.directionalNoCTransmitBytes.west);
    });
  });
}

void emitStaticCostModel(llvm::json::OStream &json,
                         const ProfileStaticCostModel &model) {
  json.object([&] {
    json.attribute("model", model.model);
    json.attribute("scope", model.scope);
    json.attributeObject("rates", [&] {
      json.attribute("card_ddr_bytes_per_second",
                     model.rates.cardDDRBytesPerSecond);
      json.attribute("directional_noc_bytes_per_second",
                     model.rates.directionalNoCBytesPerSecond);
      json.attribute("f16_bf16_npu_logical_ops_per_second_per_tile",
                     model.rates.f16Bf16NPULogicalOpsPerSecondPerTile);
      json.attribute("f16_bf16_vector_logical_ops_per_second_per_tile",
                     model.rates.f16Bf16VectorLogicalOpsPerSecondPerTile);
      json.attribute("f32_vector_logical_ops_per_second_per_tile",
                     model.rates.f32VectorLogicalOpsPerSecondPerTile);
      if (model.rates.spmMovementBytesPerSecond)
        json.attribute("spm_movement_bytes_per_second",
                       *model.rates.spmMovementBytesPerSecond);
      else
        json.attribute("spm_movement_bytes_per_second",
                       llvm::json::Value(nullptr));
    });
    json.attributeArray("tiles", [&] {
      std::vector<const ProfileStaticTileCost *> tiles;
      tiles.reserve(model.tiles.size());
      for (const ProfileStaticTileCost &tile : model.tiles)
        tiles.push_back(&tile);
      llvm::sort(tiles, [](const auto *lhs, const auto *rhs) {
        return lhs->launchSlot.getValue() < rhs->launchSlot.getValue();
      });
      for (const ProfileStaticTileCost *tile : tiles)
        json.object([&] {
          json.attribute("card_id", tile->cardId.getValue());
          json.attribute("tile_id", tile->tileId.getValue());
          json.attribute("launch_slot", tile->launchSlot.getValue());
          json.attributeBegin("work");
          emitStaticTileWork(json, tile->work);
          json.attributeEnd();
        });
    });
  });
}

llvm::Expected<ProfileTSMEngine> parseEngine(llvm::StringRef value,
                                             llvm::StringRef context) {
  if (value == "CT")
    return ProfileTSMEngine::CT;
  if (value == "NE")
    return ProfileTSMEngine::NE;
  if (value == "RDMA")
    return ProfileTSMEngine::RDMA;
  if (value == "WDMA")
    return ProfileTSMEngine::WDMA;
  if (value == "TDMA")
    return ProfileTSMEngine::TDMA;
  if (value == "DIRECT_DTE")
    return ProfileTSMEngine::DirectDTE;
  return invalid(context + " is not a supported profile engine");
}

llvm::Expected<ProfileTargetSiteKind> parseSiteKind(llvm::StringRef value,
                                                    llvm::StringRef context) {
  if (value == "ncc-command")
    return ProfileTargetSiteKind::NCCCommand;
  if (value == "ncc-completion")
    return ProfileTargetSiteKind::NCCCompletion;
  if (value == "direct-dte-control")
    return ProfileTargetSiteKind::DirectDTEControl;
  if (value == "direct-dte-issue")
    return ProfileTargetSiteKind::DirectDTEIssue;
  if (value == "direct-dte-wait")
    return ProfileTargetSiteKind::DirectDTEWait;
  return invalid(context + " is not a supported profile target site kind");
}

std::optional<ProfileTSMEngine>
getDescriptorEngine(const TargetCallDescriptor &descriptor) {
  std::optional<TargetCallTSMEngine> engine =
      getTargetCallTSMEngine(descriptor);
  if (!engine)
    return std::nullopt;
  switch (*engine) {
  case TargetCallTSMEngine::CT:
    return ProfileTSMEngine::CT;
  case TargetCallTSMEngine::NE:
    return ProfileTSMEngine::NE;
  case TargetCallTSMEngine::RDMA:
    return ProfileTSMEngine::RDMA;
  case TargetCallTSMEngine::WDMA:
    return ProfileTSMEngine::WDMA;
  case TargetCallTSMEngine::TDMA:
    return ProfileTSMEngine::TDMA;
  case TargetCallTSMEngine::DirectDTE:
    return ProfileTSMEngine::DirectDTE;
  }
  llvm_unreachable("unknown target-call NCC engine");
}

llvm::Expected<std::optional<uint64_t>>
parseOptionalUnsigned(const llvm::json::Object &object, llvm::StringRef field,
                      llvm::StringRef context) {
  if (object.find(field) == object.end())
    return std::optional<uint64_t>();
  llvm::Expected<uint64_t> value = requireUnsigned(object, field, context);
  if (!value)
    return value.takeError();
  return std::optional<uint64_t>(*value);
}

llvm::Expected<std::optional<ProfileTSMEngine>>
parseOptionalEngine(const llvm::json::Object &object, llvm::StringRef context,
                    const PackageParseLimits &limits) {
  if (object.find("engine") == object.end())
    return std::optional<ProfileTSMEngine>();
  llvm::Expected<std::string> value =
      requireString(object, "engine", context, limits);
  if (!value)
    return value.takeError();
  std::string engineContext = (context + ".engine").str();
  llvm::Expected<ProfileTSMEngine> engine = parseEngine(*value, engineContext);
  if (!engine)
    return engine.takeError();
  return std::optional<ProfileTSMEngine>(*engine);
}

llvm::Expected<ProfileTargetCallSite>
parseSite(const llvm::json::Value &value, uint64_t index,
          const PackageParseLimits &limits, llvm::StringRef tileContext) {
  std::string context =
      (tileContext + ".sites[" + llvm::Twine(index) + "]").str();
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, context);
  if (!object)
    return object.takeError();
  if (llvm::Error error =
          requireFields(**object,
                        {"site_id", "target_call_ordinal", "target_call_symbol",
                         "site_kind", "correlation_key"},
                        {"engine", "function_ordinal", "block_ordinal",
                         "instruction_ordinal"},
                        context))
    return std::move(error);

  ProfileTargetCallSite site;
  llvm::Expected<uint64_t> siteId =
      requireUnsigned(**object, "site_id", context);
  if (!siteId)
    return siteId.takeError();
  llvm::Expected<uint64_t> targetCallOrdinal =
      requireUnsigned(**object, "target_call_ordinal", context);
  if (!targetCallOrdinal)
    return targetCallOrdinal.takeError();
  llvm::Expected<std::string> symbol =
      requireString(**object, "target_call_symbol", context, limits);
  if (!symbol)
    return symbol.takeError();
  llvm::Expected<std::string> siteKindText =
      requireString(**object, "site_kind", context, limits);
  if (!siteKindText)
    return siteKindText.takeError();
  llvm::Expected<ProfileTargetSiteKind> siteKind =
      parseSiteKind(*siteKindText, context + ".site_kind");
  if (!siteKind)
    return siteKind.takeError();
  llvm::Expected<std::optional<ProfileTSMEngine>> engine =
      parseOptionalEngine(**object, context, limits);
  if (!engine)
    return engine.takeError();
  llvm::Expected<std::string> correlationKey =
      requireString(**object, "correlation_key", context, limits);
  if (!correlationKey)
    return correlationKey.takeError();
  llvm::Expected<std::optional<uint64_t>> functionOrdinal =
      parseOptionalUnsigned(**object, "function_ordinal", context);
  if (!functionOrdinal)
    return functionOrdinal.takeError();
  llvm::Expected<std::optional<uint64_t>> blockOrdinal =
      parseOptionalUnsigned(**object, "block_ordinal", context);
  if (!blockOrdinal)
    return blockOrdinal.takeError();
  llvm::Expected<std::optional<uint64_t>> instructionOrdinal =
      parseOptionalUnsigned(**object, "instruction_ordinal", context);
  if (!instructionOrdinal)
    return instructionOrdinal.takeError();

  llvm::ArrayRef<TargetCallDescriptor> descriptors = getTargetCallDescriptors();
  if (*targetCallOrdinal >= descriptors.size())
    return invalid(context + ".target_call_ordinal is outside the registry");
  const TargetCallDescriptor &descriptor = descriptors[*targetCallOrdinal];
  if (descriptor.symbol != *symbol)
    return invalid(context +
                   " target-call registry ordinal/symbol do not agree");
  const ProfileTargetSiteKind descriptorKind =
      getProfileTargetSiteKind(descriptor);
  if (descriptorKind != *siteKind)
    return invalid(context + " target-call registry semantic/site_kind do not "
                             "agree");
  std::optional<ProfileTSMEngine> descriptorEngine =
      getDescriptorEngine(descriptor);
  const bool kindRequiresEngine =
      *siteKind == ProfileTargetSiteKind::NCCCommand ||
      *siteKind == ProfileTargetSiteKind::DirectDTEIssue ||
      *siteKind == ProfileTargetSiteKind::DirectDTEWait;
  if (kindRequiresEngine && (!descriptorEngine || !*engine))
    return invalid(context + " site_kind requires its typed engine");
  if (!kindRequiresEngine && (descriptorEngine || *engine))
    return invalid(context + " site_kind must not carry an engine");
  if (descriptorEngine != *engine)
    return invalid(context + " target-call registry semantic/engine do not "
                             "agree");
  if (*siteKind == ProfileTargetSiteKind::NCCCommand &&
      **engine == ProfileTSMEngine::DirectDTE)
    return invalid(context +
                   " ncc-command must name a CT/NE/RDMA/WDMA/TDMA engine");
  if ((*siteKind == ProfileTargetSiteKind::DirectDTEIssue ||
       *siteKind == ProfileTargetSiteKind::DirectDTEWait) &&
      **engine != ProfileTSMEngine::DirectDTE)
    return invalid(context + " Direct-DTE issue/wait must name DIRECT_DTE");

  site.siteId = *siteId;
  site.targetCallOrdinal = *targetCallOrdinal;
  site.targetCallSymbol = *symbol;
  site.siteKind = *siteKind;
  site.engine = *engine;
  site.correlationKey = *correlationKey;
  site.functionOrdinal = *functionOrdinal;
  site.blockOrdinal = *blockOrdinal;
  site.instructionOrdinal = *instructionOrdinal;
  return site;
}

llvm::Expected<ProfileTileSiteMap>
parseTileSiteMap(const llvm::json::Value &value, uint64_t index,
                 const PackageParseLimits &limits, uint64_t &totalRecords,
                 llvm::StringRef siteMapContext) {
  std::string context =
      (siteMapContext + ".tiles[" + llvm::Twine(index) + "]").str();
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, context);
  if (!object)
    return object.takeError();
  if (llvm::Error error = requireFields(
          **object, {"card_id", "tile_id", "launch_slot", "sites"}, {},
          context))
    return std::move(error);
  llvm::Expected<uint64_t> cardId =
      requireUnsigned(**object, "card_id", context);
  if (!cardId)
    return cardId.takeError();
  llvm::Expected<uint64_t> tileId =
      requireUnsigned(**object, "tile_id", context);
  if (!tileId)
    return tileId.takeError();
  llvm::Expected<uint64_t> launchSlot =
      requireUnsigned(**object, "launch_slot", context);
  if (!launchSlot)
    return launchSlot.takeError();
  if (*cardId != 0 ||
      *tileId >= static_cast<uint64_t>(kProfileInstrumentationTileCount) ||
      *launchSlot >= static_cast<uint64_t>(kProfileInstrumentationTileCount))
    return invalid(context + " Tile/launch-slot binding is outside card0 "
                             "Tile0..15");
  llvm::Expected<const llvm::json::Array *> sites =
      requireArray(**object, "sites", context);
  if (!sites)
    return sites.takeError();
  if (llvm::Error error =
          accountRecords(1 + (*sites)->size(), totalRecords, limits))
    return std::move(error);

  ProfileTileSiteMap result;
  result.cardId = CardId(static_cast<int64_t>(*cardId));
  result.tileId = TileId(static_cast<int64_t>(*tileId));
  result.launchSlot = LaunchSlotId(static_cast<int64_t>(*launchSlot));
  result.sites.reserve((*sites)->size());
  for (auto [siteIndex, siteValue] : llvm::enumerate(**sites)) {
    llvm::Expected<ProfileTargetCallSite> site =
        parseSite(siteValue, siteIndex, limits, context);
    if (!site)
      return site.takeError();
    result.sites.push_back(std::move(*site));
  }
  llvm::sort(result.sites, [](const auto &lhs, const auto &rhs) {
    return lhs.siteId < rhs.siteId;
  });
  llvm::StringSet<> correlationKeys;
  for (auto [siteIndex, site] : llvm::enumerate(result.sites))
    if (site.siteId != siteIndex)
      return invalid(context + " site IDs must be unique and dense from zero");
    else if (!correlationKeys.insert(site.correlationKey).second)
      return invalid(context + " correlation keys must be unique per Tile");
  return result;
}

llvm::Expected<std::vector<ProfileTileSiteMap>>
parseSiteMap(const llvm::json::Object &root, const PackageParseLimits &limits,
             uint64_t &totalRecords) {
  if (llvm::Error error = requireFields(root,
                                        {"schema", "card_count", "tile_count",
                                         "site_basis", "correlation_basis",
                                         "target_call_registry_size", "tiles"},
                                        {}, "profile site map"))
    return std::move(error);
  if (llvm::Error error =
          verifyHeader(root, kSiteMapSchema, /*hasTopology=*/true, limits,
                       "profile site map"))
    return std::move(error);
  llvm::Expected<std::string> basis =
      requireString(root, "site_basis", "profile site map", limits);
  if (!basis)
    return basis.takeError();
  if (*basis != kSiteBasis)
    return invalid("profile site map basis is not supported");
  llvm::Expected<std::string> correlationBasis =
      requireString(root, "correlation_basis", "profile site map", limits);
  if (!correlationBasis)
    return correlationBasis.takeError();
  if (*correlationBasis != kProfileSiteCorrelationBasis)
    return invalid("profile site map correlation basis is not supported");
  llvm::Expected<uint64_t> registrySize =
      requireUnsigned(root, "target_call_registry_size", "profile site map");
  if (!registrySize)
    return registrySize.takeError();
  if (*registrySize != getTargetCallDescriptors().size())
    return invalid("profile site map target-call registry size is stale");
  llvm::Expected<const llvm::json::Array *> tiles =
      requireArray(root, "tiles", "profile site map");
  if (!tiles)
    return tiles.takeError();
  if ((*tiles)->size() != static_cast<size_t>(kProfileInstrumentationTileCount))
    return invalid("profile site map must contain all and only 16 Tiles");
  if (llvm::Error error =
          accountRecords((*tiles)->size(), totalRecords, limits))
    return std::move(error);

  std::vector<ProfileTileSiteMap> result;
  result.reserve((*tiles)->size());
  for (auto [tileIndex, tileValue] : llvm::enumerate(**tiles)) {
    llvm::Expected<ProfileTileSiteMap> tile = parseTileSiteMap(
        tileValue, tileIndex, limits, totalRecords, "profile site map");
    if (!tile)
      return tile.takeError();
    result.push_back(std::move(*tile));
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return lhs.launchSlot.getValue() < rhs.launchSlot.getValue();
  });
  std::array<bool, kProfileInstrumentationTileCount> seenTileIds{};
  for (auto [launchSlot, tile] : llvm::enumerate(result)) {
    const int64_t tileId = tile.tileId.getValue();
    if (tile.cardId != CardId(0) || tileId < 0 ||
        tileId >= kProfileInstrumentationTileCount || seenTileIds[tileId] ||
        tile.launchSlot != LaunchSlotId(static_cast<int64_t>(launchSlot)))
      return invalid("profile site map must contain each Tile and "
                     "launch slot exactly once");
    seenTileIds[tileId] = true;
  }
  return result;
}

bool isLowercaseSHA256(llvm::StringRef digest) {
  if (!digest.consume_front("sha256:") || digest.size() != 64)
    return false;
  return llvm::all_of(digest, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

llvm::Expected<std::string> digestManifest(llvm::StringRef packageRoot) {
  llvm::SmallString<256> manifest(packageRoot);
  llvm::sys::path::append(manifest, kPackageManifestFileName);
  if (llvm::sys::fs::get_file_type(manifest, /*Follow=*/false) !=
      llvm::sys::fs::file_type::regular_file)
    return invalid("profile primary manifest is not a regular file");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(manifest, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read profile primary manifest");
  llvm::SHA256 hasher;
  hasher.update((*buffer)->getBuffer());
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

bool isPathWithin(llvm::StringRef path, llvm::StringRef parent) {
  if (!path.starts_with(parent) || path.size() <= parent.size())
    return false;
  return llvm::sys::path::is_separator(path[parent.size()]);
}

llvm::Expected<std::string>
resolveCapturePackageReference(llvm::StringRef instrumentationRoot,
                               llvm::StringRef reference) {
  if (reference.empty() || llvm::sys::path::is_absolute(reference) ||
      reference.contains('\\'))
    return invalid("profile capture package_ref is not a safe relative path");
  llvm::SmallVector<llvm::StringRef, 4> components;
  reference.split(components, '/', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  if (components.size() != 2 || components[0] != "captures" ||
      (components[1] != "count" && components[1] != "trace"))
    return invalid("profile capture package_ref is not canonical");
  llvm::SmallString<256> candidate(instrumentationRoot);
  llvm::sys::path::append(candidate, reference);
  if (llvm::sys::fs::get_file_type(candidate, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("profile capture package_ref is not a directory");
  llvm::SmallString<256> canonical;
  if (std::error_code error = llvm::sys::fs::real_path(candidate, canonical))
    return llvm::createStringError(
        error, "failed to resolve profile capture package_ref");
  if (!isPathWithin(canonical, instrumentationRoot))
    return invalid(
        "profile capture package_ref escapes the instrumentation root");
  return canonical.str().str();
}

const RawCapturePackage *
findCapturePackage(llvm::ArrayRef<RawCapturePackage> packages,
                   ProfileCaptureKind capture) {
  auto iterator = llvm::find_if(packages, [&](const auto &package) {
    return package.capture == capture;
  });
  return iterator == packages.end() ? nullptr : &*iterator;
}

llvm::Error verifyProfileGraph(const RawPlan &plan,
                               llvm::ArrayRef<ProfileTileSiteMap> siteMap) {
  for (ProfileCaptureKind capture :
       {ProfileCaptureKind::Count, ProfileCaptureKind::Trace}) {
    const RawCapturePackage *capturePackage =
        findCapturePackage(plan.capturePackages, capture);
    if (!capturePackage)
      return invalid("profile plan capture packages are incomplete");
    std::string captureName = stringifyProfileCaptureKind(capture).str();
    std::string reference = "captures/" + captureName;
    if (capturePackage->packageReference != reference)
      return invalid("profile capture package_ref is not canonical");
  }

  if (siteMap.size() != static_cast<size_t>(kProfileInstrumentationTileCount))
    return invalid("profile site map must contain all 16 Tiles");
  return llvm::Error::success();
}

const PackageEntrypointRecord *findEntryForTile(const PackageManifest &manifest,
                                                int64_t tileId) {
  auto iterator = llvm::find_if(manifest.entries, [&](const auto &entry) {
    return entry.cardId == CardId(0) && entry.tileId == TileId(tileId);
  });
  return iterator == manifest.entries.end() ? nullptr : &*iterator;
}

const PackageEntrypointRecord *
findEntryForLaunchSlot(const PackageManifest &manifest,
                       LaunchSlotId launchSlot) {
  auto iterator = llvm::find_if(manifest.entries, [&](const auto &entry) {
    return entry.launchSlot == launchSlot;
  });
  return iterator == manifest.entries.end() ? nullptr : &*iterator;
}

template <typename TileRecord>
bool hasSamePhysicalBinding(const PackageEntrypointRecord &entry,
                            const TileRecord &tile) {
  return entry.cardId == tile.cardId && entry.tileId == tile.tileId &&
         entry.launchSlot == tile.launchSlot;
}

llvm::Error
verifyProfilePhysicalBindings(const ProfiledPackage &profiledPackage,
                              llvm::ArrayRef<ProfileTileSiteMap> siteMap) {
  const PackageManifest &manifest = profiledPackage.getPackage().getManifest();
  const ProfileStaticCostModel &staticCost =
      profiledPackage.getStaticCostModel();
  if (staticCost.tiles.size() != static_cast<size_t>(manifest.tileCount) ||
      siteMap.size() != static_cast<size_t>(manifest.tileCount))
    return invalid("profile Tile bindings are incomplete");
  for (int64_t slot = 0; slot < manifest.tileCount; ++slot) {
    const LaunchSlotId launchSlot(static_cast<uint64_t>(slot));
    const PackageEntrypointRecord *entry =
        findEntryForLaunchSlot(manifest, launchSlot);
    const ProfileStaticTileCost &staticTile = staticCost.tiles[slot];
    const ProfileTileSiteMap &siteTile = siteMap[slot];
    if (!entry || staticTile.launchSlot != launchSlot ||
        siteTile.launchSlot != launchSlot ||
        !hasSamePhysicalBinding(*entry, staticTile) ||
        !hasSamePhysicalBinding(*entry, siteTile))
      return invalid("profile Tile/launch-slot binding differs from the "
                     "profiled package");
  }
  return llvm::Error::success();
}

bool sameTransport(const TransportRequirements &lhs,
                   const TransportRequirements &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  const auto *lhsDirect = std::get_if<DirectDTETransportRequirements>(&lhs);
  const auto *rhsDirect = std::get_if<DirectDTETransportRequirements>(&rhs);
  if (!lhsDirect)
    return true;
  return rhsDirect && lhsDirect->statusABI == rhsDirect->statusABI &&
         lhsDirect->hostWatchdogRequired == rhsDirect->hostWatchdogRequired;
}

const ProfileRecordArgument *
findProfileRecordArgument(const PackageEntrypointRecord &entry) {
  for (const TileEntryArgumentRecord &argument : entry.arguments)
    if (const auto *profile =
            std::get_if<ProfileRecordArgument>(&argument.reference))
      return profile;
  return nullptr;
}

llvm::Error verifyCapturePackageContract(const PackageManifest &execution,
                                         const PackageManifest &capture,
                                         uint64_t recordBytes) {
  if (execution.targetIdentity != capture.targetIdentity ||
      execution.runtimeABI != capture.runtimeABI ||
      execution.launch != capture.launch ||
      execution.moduleFormat != capture.moduleFormat)
    return invalid("profile capture target/ABI contract differs from its "
                   "execution package");
  if (execution.cardCount != 1 || capture.cardCount != 1 ||
      execution.tileCount != kProfileInstrumentationTileCount ||
      capture.tileCount != kProfileInstrumentationTileCount)
    return invalid("profile capture packages must each contain one card and "
                   "16 Tiles");

  // The capture package reuses the same accepted Tile domain: program
  // tensors, target tensors, ports and program data are byte-identical;
  // only the entry-local profile record extends each entry.
  if (execution.programTensors != capture.programTensors ||
      execution.targetTensors != capture.targetTensors ||
      execution.inputs != capture.inputs ||
      execution.outputs != capture.outputs)
    return invalid("profile capture program/target tensor or port contract "
                   "differs from its execution package");
  if (execution.programData.digest != capture.programData.digest ||
      execution.programData.totalBytes != capture.programData.totalBytes ||
      execution.programData.baseAlignment !=
          capture.programData.baseAlignment ||
      execution.programData.relativePath !=
          capture.programData.relativePath)
    return invalid("profile capture program data contract differs from its "
                   "execution package");

  for (int64_t tileId = 0; tileId < kProfileInstrumentationTileCount;
       ++tileId) {
    const PackageEntrypointRecord *executionEntry =
        findEntryForTile(execution, tileId);
    const PackageEntrypointRecord *captureEntry =
        findEntryForTile(capture, tileId);
    if (!executionEntry || !captureEntry ||
        executionEntry->cardId != captureEntry->cardId ||
        executionEntry->tileId != captureEntry->tileId ||
        executionEntry->launchSlot != captureEntry->launchSlot ||
        executionEntry->completion != captureEntry->completion ||
        captureEntry->arguments.size() !=
            executionEntry->arguments.size() + 1 ||
        !sameTransport(executionEntry->transport, captureEntry->transport))
      return invalid("profile capture entry ABI extension is invalid");
    for (size_t index = 0; index < executionEntry->arguments.size(); ++index) {
      const TileEntryArgumentRecord &executionArgument =
          executionEntry->arguments[index];
      const TileEntryArgumentRecord &captureArgument =
          captureEntry->arguments[index];
      if (executionArgument.reference != captureArgument.reference ||
          executionArgument.access != captureArgument.access)
        return invalid("profile capture entry ABI base arguments differ from "
                       "its execution package");
    }
    const ProfileRecordArgument *profile =
        findProfileRecordArgument(*captureEntry);
    if (!profile ||
        profile->recordABI != kProfileRecordABI ||
        profile->bytes != recordBytes ||
        profile->alignment != WAFER_TX81_PROFILER_BUFFER_ALIGNMENT ||
        captureEntry->arguments.back().access != PackageAccessMode::ReadWrite ||
        !std::holds_alternative<ProfileRecordArgument>(
            captureEntry->arguments.back().reference))
      return invalid("profile capture profiler record is not the exact final "
                     "entry argument");
    if (findProfileRecordArgument(*executionEntry))
      return invalid("execution package occupies the reserved profiler "
                     "record argument");
  }
  return llvm::Error::success();
}

llvm::Expected<ProfiledPackage> loadProfiledPackage(
    llvm::StringRef primaryPackageRoot, llvm::StringRef manifestDigest,
    ProfileStaticCostModel staticCostModel, const PackageParseLimits &limits) {
  if (!isLowercaseSHA256(manifestDigest))
    return invalid("profile primary manifest_sha256 is malformed");
  llvm::Expected<std::string> digest = digestManifest(primaryPackageRoot);
  if (!digest)
    return digest.takeError();
  if (*digest != manifestDigest)
    return invalid("profile primary manifest digest mismatch");
  llvm::Expected<VerifiedPackageManifest> package =
      loadVerifiedPackageManifest(primaryPackageRoot, limits);
  if (!package)
    return package.takeError();

  return ProfiledPackage(manifestDigest.str(), std::move(staticCostModel),
                         primaryPackageRoot.str(), std::move(*package));
}

llvm::Expected<ProfileCapturePackage> loadCapturePackage(
    const RawCapturePackage &capture, llvm::StringRef instrumentationRoot,
    const ProfiledPackage &profiledPackage, const PackageParseLimits &limits) {
  if (!isLowercaseSHA256(capture.manifestDigest))
    return invalid("profile capture manifest_sha256 is malformed");
  if (capture.recordABI != kProfileRecordABI)
    return invalid("profile capture record_abi is inconsistent");
  if (capture.recordBytes != expectedRecordBytes(capture.capture))
    return invalid("profile capture record_bytes is inconsistent");
  llvm::Expected<std::string> packageDirectory = resolveCapturePackageReference(
      instrumentationRoot, capture.packageReference);
  if (!packageDirectory)
    return packageDirectory.takeError();
  llvm::Expected<std::string> digest = digestManifest(*packageDirectory);
  if (!digest)
    return digest.takeError();
  if (*digest != capture.manifestDigest)
    return invalid("profile capture manifest digest mismatch");
  llvm::Expected<VerifiedPackageManifest> package =
      loadVerifiedPackageManifest(*packageDirectory, limits);
  if (!package)
    return package.takeError();
  if (llvm::Error error = verifyCapturePackageContract(
          profiledPackage.getPackage().getManifest(), package->getManifest(),
          capture.recordBytes))
    return std::move(error);
  return ProfileCapturePackage(capture.capture, capture.packageReference,
                               capture.manifestDigest, capture.recordABI,
                               capture.recordBytes, *packageDirectory,
                               std::move(*package));
}

} // namespace

ProfileTargetSiteKind
getProfileTargetSiteKind(const TargetCallDescriptor &descriptor) {
  if (std::optional<TargetCallTSMEngine> engine =
          getTargetCallTSMEngine(descriptor)) {
    if (*engine != TargetCallTSMEngine::DirectDTE)
      return ProfileTargetSiteKind::NCCCommand;
    const auto *builtin = std::get_if<TargetCallBuiltin>(&descriptor.semantic);
    if (!builtin)
      llvm_unreachable("Direct-DTE target call is not a builtin");
    if (*builtin == TargetCallBuiltin::DirectDTESendIssue)
      return ProfileTargetSiteKind::DirectDTEIssue;
    if (*builtin == TargetCallBuiltin::DirectDTEWait)
      return ProfileTargetSiteKind::DirectDTEWait;
    llvm_unreachable("unknown Direct-DTE engine target call");
  }

  const auto *builtin = std::get_if<TargetCallBuiltin>(&descriptor.semantic);
  if (!builtin)
    llvm_unreachable(
        "non-builtin target call without an NCC issue-domain engine");
  switch (*builtin) {
  case TargetCallBuiltin::NCCJoin:
    return ProfileTargetSiteKind::NCCCompletion;
  case TargetCallBuiltin::DirectDTEBegin:
  case TargetCallBuiltin::DirectDTEBeginAfterPrepare:
  case TargetCallBuiltin::DirectDTESendPrepare:
  case TargetCallBuiltin::DirectDTERecvPrepare:
  case TargetCallBuiltin::DirectDTEFinish:
    return ProfileTargetSiteKind::DirectDTEControl;
  case TargetCallBuiltin::RDMA:
  case TargetCallBuiltin::WDMA:
  case TargetCallBuiltin::GatherScatter:
  case TargetCallBuiltin::Memset:
  case TargetCallBuiltin::Bit2FP:
  case TargetCallBuiltin::MaskMove:
  case TargetCallBuiltin::Gemm:
  case TargetCallBuiltin::GemmOriented:
  case TargetCallBuiltin::TDMAPad:
  case TargetCallBuiltin::TDMAImg2Col:
  case TargetCallBuiltin::DirectDTESendIssue:
  case TargetCallBuiltin::DirectDTEWait:
    llvm_unreachable("engine target call lost its typed issue domain");
  }
  llvm_unreachable("unknown target-call builtin");
}

ProfiledPackage::ProfiledPackage(std::string manifestDigest,
                                 ProfileStaticCostModel staticCostModel,
                                 std::string packageDirectory,
                                 VerifiedPackageManifest package)
    : manifestDigest(std::move(manifestDigest)),
      staticCostModel(std::move(staticCostModel)),
      packageDirectory(std::move(packageDirectory)),
      package(std::move(package)) {}

ProfileCapturePackage::ProfileCapturePackage(
    ProfileCaptureKind capture, std::string packageReference,
    std::string manifestDigest, std::string recordABI, uint64_t recordBytes,
    std::string packageDirectory, VerifiedPackageManifest package)
    : capture(capture), packageReference(std::move(packageReference)),
      manifestDigest(std::move(manifestDigest)),
      recordABI(std::move(recordABI)), recordBytes(recordBytes),
      packageDirectory(std::move(packageDirectory)),
      package(std::move(package)) {}

llvm::StringRef stringifyProfileCaptureKind(ProfileCaptureKind capture) {
  switch (capture) {
  case ProfileCaptureKind::Count:
    return "count";
  case ProfileCaptureKind::Trace:
    return "trace";
  }
  llvm_unreachable("unknown profile capture kind");
}

llvm::StringRef stringifyProfileTSMEngine(ProfileTSMEngine engine) {
  switch (engine) {
  case ProfileTSMEngine::CT:
    return "CT";
  case ProfileTSMEngine::NE:
    return "NE";
  case ProfileTSMEngine::RDMA:
    return "RDMA";
  case ProfileTSMEngine::WDMA:
    return "WDMA";
  case ProfileTSMEngine::TDMA:
    return "TDMA";
  case ProfileTSMEngine::DirectDTE:
    return "DIRECT_DTE";
  }
  llvm_unreachable("unknown profile TSM engine");
}

llvm::StringRef stringifyProfileTargetSiteKind(ProfileTargetSiteKind kind) {
  switch (kind) {
  case ProfileTargetSiteKind::NCCCommand:
    return "ncc-command";
  case ProfileTargetSiteKind::NCCCompletion:
    return "ncc-completion";
  case ProfileTargetSiteKind::DirectDTEControl:
    return "direct-dte-control";
  case ProfileTargetSiteKind::DirectDTEIssue:
    return "direct-dte-issue";
  case ProfileTargetSiteKind::DirectDTEWait:
    return "direct-dte-wait";
  }
  llvm_unreachable("unknown profile target site kind");
}

void writeProfileStaticCostModel(llvm::json::OStream &json,
                                 const ProfileStaticCostModel &model) {
  emitStaticCostModel(json, model);
}

const ProfileCapturePackage *
VerifiedProfileInstrumentation::findCapture(ProfileCaptureKind capture) const {
  auto iterator = llvm::find_if(captures, [&](const auto &package) {
    return package.getCaptureKind() == capture;
  });
  return iterator == captures.end() ? nullptr : &*iterator;
}

uint64_t VerifiedProfileInstrumentation::getSiteCount() const {
  uint64_t count = 0;
  for (const ProfileTileSiteMap &tile : siteMap)
    count += tile.sites.size();
  return count;
}

llvm::Expected<VerifiedProfileInstrumentation>
loadVerifiedProfileInstrumentation(llvm::StringRef instrumentationRoot,
                                   llvm::StringRef primaryPackageRoot,
                                   const PackageParseLimits &limits) {
  if (llvm::sys::fs::get_file_type(instrumentationRoot, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("profile instrumentation root is not a directory");
  if (llvm::sys::fs::get_file_type(primaryPackageRoot,
                                   /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("selected primary package root is not a directory");
  llvm::SmallString<256> canonicalInstrumentation;
  if (std::error_code error = llvm::sys::fs::real_path(
          instrumentationRoot, canonicalInstrumentation))
    return llvm::createStringError(
        error, "failed to resolve profile instrumentation root");
  llvm::SmallString<256> canonicalPrimary;
  if (std::error_code error =
          llvm::sys::fs::real_path(primaryPackageRoot, canonicalPrimary))
    return llvm::createStringError(
        error, "failed to resolve selected primary package");

  llvm::SmallString<256> activationPath(canonicalInstrumentation);
  llvm::sys::path::append(activationPath,
                          kProfileInstrumentationActivationFileName);
  llvm::SmallString<256> planPath(canonicalInstrumentation);
  llvm::sys::path::append(planPath, kProfileInstrumentationPlanFileName);
  llvm::SmallString<256> siteMapPath(canonicalInstrumentation);
  llvm::sys::path::append(siteMapPath, kProfileInstrumentationSiteMapFileName);

  llvm::Expected<LoadedJSONDocument> activationJSON =
      loadJSONDocument(activationPath, "profile activation", limits);
  if (!activationJSON)
    return activationJSON.takeError();
  llvm::Expected<RawActivation> activation =
      parseActivation(*activationJSON->root.getAsObject(), limits);
  if (!activation)
    return activation.takeError();

  llvm::Expected<std::string> primaryDigest = digestManifest(canonicalPrimary);
  if (!primaryDigest)
    return primaryDigest.takeError();
  if (*primaryDigest != activation->manifestDigest)
    return invalid("profile activation primary manifest digest mismatch");

  llvm::Expected<LoadedJSONDocument> planJSON =
      loadJSONDocument(planPath, "profile plan", limits);
  if (!planJSON)
    return planJSON.takeError();
  llvm::Expected<LoadedJSONDocument> siteMapJSON =
      loadJSONDocument(siteMapPath, "profile site map", limits);
  if (!siteMapJSON)
    return siteMapJSON.takeError();
  if (planJSON->digest != activation->planDigest ||
      siteMapJSON->digest != activation->siteMapDigest)
    return invalid("profile activation metadata digest mismatch");

  uint64_t totalRecords = 0;
  llvm::Expected<RawPlan> plan =
      parsePlan(*planJSON->root.getAsObject(), limits, totalRecords);
  if (!plan)
    return plan.takeError();
  llvm::Expected<std::vector<ProfileTileSiteMap>> siteMap =
      parseSiteMap(*siteMapJSON->root.getAsObject(), limits, totalRecords);
  if (!siteMap)
    return siteMap.takeError();
  if (llvm::Error error = verifyProfileGraph(*plan, *siteMap))
    return std::move(error);

  llvm::Expected<ProfiledPackage> profiledPackage =
      loadProfiledPackage(canonicalPrimary, activation->manifestDigest,
                          std::move(plan->staticCostModel), limits);
  if (!profiledPackage)
    return profiledPackage.takeError();
  if (llvm::Error error =
          verifyProfilePhysicalBindings(*profiledPackage, *siteMap))
    return std::move(error);

  std::vector<ProfileCapturePackage> captures;
  captures.reserve(plan->capturePackages.size());
  for (const RawCapturePackage &rawCapture : plan->capturePackages) {
    llvm::Expected<ProfileCapturePackage> capture = loadCapturePackage(
        rawCapture, canonicalInstrumentation, *profiledPackage, limits);
    if (!capture)
      return capture.takeError();
    captures.push_back(std::move(*capture));
  }

  return VerifiedProfileInstrumentation(
      canonicalInstrumentation.str().str(), std::move(*profiledPackage),
      std::move(captures), std::move(*siteMap));
}

llvm::Expected<std::optional<VerifiedProfileInstrumentation>>
loadSiblingProfileInstrumentationIfPresent(llvm::StringRef primaryPackageRoot,
                                           const PackageParseLimits &limits) {
  std::string instrumentation = primaryPackageRoot.str();
  while (!instrumentation.empty() &&
         llvm::sys::path::is_separator(instrumentation.back()))
    instrumentation.pop_back();
  instrumentation += ".profile";

  llvm::sys::fs::file_status status;
  if (std::error_code error =
          llvm::sys::fs::status(instrumentation, status, /*follow=*/false)) {
    if (error == std::errc::no_such_file_or_directory)
      return std::optional<VerifiedProfileInstrumentation>();
    return llvm::createStringError(
        error, "failed to inspect sibling profile instrumentation");
  }
  if (!llvm::sys::fs::exists(status))
    return std::optional<VerifiedProfileInstrumentation>();
  llvm::Expected<VerifiedProfileInstrumentation> loaded =
      loadVerifiedProfileInstrumentation(instrumentation, primaryPackageRoot,
                                         limits);
  if (!loaded)
    return loaded.takeError();
  return std::optional<VerifiedProfileInstrumentation>(std::move(*loaded));
}

} // namespace wafer::runtime
