//===- ProfileCompanion.cpp - Verified profiler companion readback ------===//

#include "Wafer/Runtime/ProfileCompanion.h"

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
constexpr llvm::StringLiteral kVariantsSchema = "wafer-profile-variants";
constexpr llvm::StringLiteral kActivationSchema = "wafer-profile-activation";
constexpr llvm::StringLiteral kSiteMapSchema =
    "wafer-profile-target-call-site-map";
constexpr llvm::StringLiteral kSiteBasis =
    "verified-target-llvm-entry-reachable-tsm-call-preorder";
constexpr llvm::StringLiteral kSiteIdentity =
    "variant-rank-local-tsm-site-id-and-typed-correlation-key";
constexpr llvm::StringLiteral kProductionWinner = "production-winner";
constexpr llvm::StringLiteral kReservedBaseline = "reserved-baseline";
constexpr llvm::StringLiteral kBaselinePackageDirectory = "baseline-package";
constexpr uint64_t kSummaryRecordBytes = WAFER_TX81_PROFILER_MIN_BUFFER_BYTES;
constexpr uint64_t kCountRecordBytes = WAFER_TX81_PROFILER_MIN_BUFFER_BYTES;
constexpr uint64_t kTraceRecordBytes = 1024 * 1024;

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
    return invalid("profile companion exceeds record limit");
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
  std::string productionManifestDigest;
  std::string planDigest;
  std::string variantsDigest;
  std::string siteMapDigest;
};

struct RawVariant {
  std::string id;
  std::string role;
  std::string packageReference;
  std::string manifestDigest;
  std::optional<std::string> sameAs;
};

struct RawExecutionPackage {
  std::string variantId;
  std::string packageReference;
  std::string manifestDigest;
};

struct RawCapturePackage {
  std::string variantId;
  ProfileCaptureKind capture = ProfileCaptureKind::Summary;
  std::string packageReference;
  std::string manifestDigest;
  uint64_t recordBytes = 0;
};

struct RawPlan {
  std::vector<RawExecutionPackage> executionPackages;
  std::vector<RawCapturePackage> capturePackages;
};

llvm::Expected<ProfileCaptureKind> parseCaptureKind(llvm::StringRef value,
                                                    llvm::StringRef context) {
  if (value == "summary")
    return ProfileCaptureKind::Summary;
  if (value == "count")
    return ProfileCaptureKind::Count;
  if (value == "trace")
    return ProfileCaptureKind::Trace;
  return invalid(context + " is not a supported capture kind");
}

uint64_t expectedRecordBytes(ProfileCaptureKind capture) {
  switch (capture) {
  case ProfileCaptureKind::Summary:
    return kSummaryRecordBytes;
  case ProfileCaptureKind::Count:
    return kCountRecordBytes;
  case ProfileCaptureKind::Trace:
    return kTraceRecordBytes;
  }
  llvm_unreachable("unknown profile capture kind");
}

llvm::Error verifyHeader(const llvm::json::Object &root,
                         llvm::StringRef expectedSchema, bool hasRankCount,
                         const PackageParseLimits &limits,
                         llvm::StringRef context);
bool isLowercaseSHA256(llvm::StringRef digest);

llvm::Expected<RawActivation>
parseActivation(const llvm::json::Object &root,
                const PackageParseLimits &limits) {
  if (llvm::Error error =
          requireFields(root,
                        {"schema", "schema_version",
                         "production_manifest_sha256", "metadata_sha256"},
                        {}, "profile activation"))
    return std::move(error);
  if (llvm::Error error =
          verifyHeader(root, kActivationSchema, /*hasRankCount=*/false, limits,
                       "profile activation"))
    return std::move(error);
  llvm::Expected<std::string> production = requireString(
      root, "production_manifest_sha256", "profile activation", limits);
  if (!production)
    return production.takeError();
  const llvm::json::Object *metadata = root.getObject("metadata_sha256");
  if (!metadata)
    return invalid("profile activation.metadata_sha256 must be an object");
  if (llvm::Error error = requireFields(
          *metadata,
          {kProfileCompanionPlanFileName, kProfileCompanionVariantsFileName,
           kProfileCompanionSiteMapFileName},
          {}, "profile activation.metadata_sha256"))
    return std::move(error);
  llvm::Expected<std::string> plan =
      requireString(*metadata, kProfileCompanionPlanFileName,
                    "profile activation.metadata_sha256", limits);
  if (!plan)
    return plan.takeError();
  llvm::Expected<std::string> variants =
      requireString(*metadata, kProfileCompanionVariantsFileName,
                    "profile activation.metadata_sha256", limits);
  if (!variants)
    return variants.takeError();
  llvm::Expected<std::string> siteMap =
      requireString(*metadata, kProfileCompanionSiteMapFileName,
                    "profile activation.metadata_sha256", limits);
  if (!siteMap)
    return siteMap.takeError();
  if (!isLowercaseSHA256(*production) || !isLowercaseSHA256(*plan) ||
      !isLowercaseSHA256(*variants) || !isLowercaseSHA256(*siteMap))
    return invalid("profile activation contains a malformed SHA-256 digest");
  return RawActivation{std::move(*production), std::move(*plan),
                       std::move(*variants), std::move(*siteMap)};
}

llvm::Error verifyHeader(const llvm::json::Object &root,
                         llvm::StringRef expectedSchema, bool hasRankCount,
                         const PackageParseLimits &limits,
                         llvm::StringRef context) {
  llvm::Expected<std::string> schema =
      requireString(root, "schema", context, limits);
  if (!schema)
    return schema.takeError();
  if (*schema != expectedSchema)
    return invalid(context + ".schema is not supported");
  llvm::Expected<uint64_t> version =
      requireUnsigned(root, "schema_version", context);
  if (!version)
    return version.takeError();
  if (*version != kProfileCompanionSchemaVersion)
    return invalid(context + ".schema_version is not supported");
  if (hasRankCount) {
    llvm::Expected<uint64_t> rankCount =
        requireUnsigned(root, "rank_count", context);
    if (!rankCount)
      return rankCount.takeError();
    if (*rankCount != static_cast<uint64_t>(kProfileCompanionRankCount))
      return invalid(context + ".rank_count must be exactly 16");
  }
  return llvm::Error::success();
}

llvm::Expected<RawPlan> parsePlan(const llvm::json::Object &root,
                                  const PackageParseLimits &limits,
                                  uint64_t &totalRecords) {
  if (llvm::Error error =
          requireFields(root,
                        {"schema", "schema_version", "rank_count",
                         "variant_metadata", "site_map", "site_identity",
                         "execution_packages", "capture_packages"},
                        {}, "profile plan"))
    return std::move(error);
  if (llvm::Error error = verifyHeader(root, kPlanSchema, /*hasRankCount=*/true,
                                       limits, "profile plan"))
    return std::move(error);

  llvm::Expected<std::string> variants =
      requireString(root, "variant_metadata", "profile plan", limits);
  if (!variants)
    return variants.takeError();
  llvm::Expected<std::string> siteMap =
      requireString(root, "site_map", "profile plan", limits);
  if (!siteMap)
    return siteMap.takeError();
  llvm::Expected<std::string> identity =
      requireString(root, "site_identity", "profile plan", limits);
  if (!identity)
    return identity.takeError();
  if (*variants != kProfileCompanionVariantsFileName ||
      *siteMap != kProfileCompanionSiteMapFileName)
    return invalid("profile plan metadata references are not canonical");
  if (*identity != kSiteIdentity)
    return invalid("profile plan site identity is not supported");

  llvm::Expected<const llvm::json::Array *> packages =
      requireArray(root, "execution_packages", "profile plan");
  if (!packages)
    return packages.takeError();
  if ((*packages)->size() != 2)
    return invalid("profile plan must contain exactly two execution packages");
  if (llvm::Error error =
          accountRecords((*packages)->size(), totalRecords, limits))
    return std::move(error);

  RawPlan plan;
  for (auto [index, value] : llvm::enumerate(**packages)) {
    std::string context =
        "profile plan.execution_packages[" + std::to_string(index) + "]";
    llvm::Expected<const llvm::json::Object *> object =
        requireObject(value, context);
    if (!object)
      return object.takeError();
    if (llvm::Error error = requireFields(
            **object, {"variant_id", "package_ref", "manifest_sha256"}, {},
            context))
      return std::move(error);
    llvm::Expected<std::string> id =
        requireString(**object, "variant_id", context, limits);
    if (!id)
      return id.takeError();
    llvm::Expected<std::string> reference =
        requireString(**object, "package_ref", context, limits);
    if (!reference)
      return reference.takeError();
    llvm::Expected<std::string> digest =
        requireString(**object, "manifest_sha256", context, limits);
    if (!digest)
      return digest.takeError();
    plan.executionPackages.push_back({*id, *reference, *digest});
  }

  llvm::Expected<const llvm::json::Array *> captures =
      requireArray(root, "capture_packages", "profile plan");
  if (!captures)
    return captures.takeError();
  if ((*captures)->size() != 6)
    return invalid("profile plan must contain exactly six capture packages");
  if (llvm::Error error =
          accountRecords((*captures)->size(), totalRecords, limits))
    return std::move(error);
  const std::array<std::pair<llvm::StringRef, ProfileCaptureKind>, 6>
      expectedOrder = {{
          {kReservedBaseline, ProfileCaptureKind::Summary},
          {kReservedBaseline, ProfileCaptureKind::Count},
          {kReservedBaseline, ProfileCaptureKind::Trace},
          {kProductionWinner, ProfileCaptureKind::Summary},
          {kProductionWinner, ProfileCaptureKind::Count},
          {kProductionWinner, ProfileCaptureKind::Trace},
      }};
  for (auto [index, value] : llvm::enumerate(**captures)) {
    std::string context =
        "profile plan.capture_packages[" + std::to_string(index) + "]";
    llvm::Expected<const llvm::json::Object *> object =
        requireObject(value, context);
    if (!object)
      return object.takeError();
    if (llvm::Error error =
            requireFields(**object,
                          {"variant_id", "capture", "package_ref",
                           "manifest_sha256", "record_bytes"},
                          {}, context))
      return std::move(error);
    llvm::Expected<std::string> id =
        requireString(**object, "variant_id", context, limits);
    if (!id)
      return id.takeError();
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
    llvm::Expected<uint64_t> recordBytes =
        requireUnsigned(**object, "record_bytes", context);
    if (!recordBytes)
      return recordBytes.takeError();
    if (*id != expectedOrder[index].first ||
        *capture != expectedOrder[index].second)
      return invalid("profile plan capture packages are not in canonical "
                     "variant/capture order");
    if (*recordBytes != expectedRecordBytes(*capture))
      return invalid(context + ".record_bytes is not the capture contract");
    plan.capturePackages.push_back(
        {*id, *capture, *reference, *digest, *recordBytes});
  }
  return plan;
}

llvm::Expected<std::vector<RawVariant>>
parseVariants(const llvm::json::Object &root, const PackageParseLimits &limits,
              uint64_t &totalRecords) {
  if (llvm::Error error = requireFields(
          root, {"schema", "schema_version", "rank_count", "variants"}, {},
          "profile variants"))
    return std::move(error);
  if (llvm::Error error =
          verifyHeader(root, kVariantsSchema, /*hasRankCount=*/true, limits,
                       "profile variants"))
    return std::move(error);
  llvm::Expected<const llvm::json::Array *> variants =
      requireArray(root, "variants", "profile variants");
  if (!variants)
    return variants.takeError();
  if ((*variants)->size() != 2)
    return invalid("profile variants must contain exactly two variants");
  if (llvm::Error error =
          accountRecords((*variants)->size(), totalRecords, limits))
    return std::move(error);

  std::vector<RawVariant> result;
  for (auto [index, value] : llvm::enumerate(**variants)) {
    std::string context =
        "profile variants.variants[" + std::to_string(index) + "]";
    llvm::Expected<const llvm::json::Object *> object =
        requireObject(value, context);
    if (!object)
      return object.takeError();
    if (llvm::Error error = requireFields(
            **object, {"id", "role", "package_ref", "manifest_sha256"},
            {"same_as"}, context))
      return std::move(error);

    RawVariant variant;
    llvm::Expected<std::string> id =
        requireString(**object, "id", context, limits);
    if (!id)
      return id.takeError();
    llvm::Expected<std::string> role =
        requireString(**object, "role", context, limits);
    if (!role)
      return role.takeError();
    llvm::Expected<std::string> reference =
        requireString(**object, "package_ref", context, limits);
    if (!reference)
      return reference.takeError();
    llvm::Expected<std::string> digest =
        requireString(**object, "manifest_sha256", context, limits);
    if (!digest)
      return digest.takeError();
    variant.id = *id;
    variant.role = *role;
    variant.packageReference = *reference;
    variant.manifestDigest = *digest;
    if ((*object)->find("same_as") != (*object)->end()) {
      llvm::Expected<std::string> sameAs =
          requireString(**object, "same_as", context, limits);
      if (!sameAs)
        return sameAs.takeError();
      variant.sameAs = *sameAs;
    }
    result.push_back(std::move(variant));
  }
  return result;
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
  return invalid(context + " is not a supported TsmExecute engine");
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
  }
  llvm_unreachable("unknown target-call TsmExecute engine");
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

llvm::Expected<ProfileTargetCallSite>
parseSite(const llvm::json::Value &value, uint64_t index,
          const PackageParseLimits &limits, llvm::StringRef rankContext) {
  std::string context =
      (rankContext + ".sites[" + llvm::Twine(index) + "]").str();
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, context);
  if (!object)
    return object.takeError();
  if (llvm::Error error = requireFields(
          **object,
          {"site_id", "target_call_ordinal", "target_call_symbol", "engine",
           "correlation_key"},
          {"function_ordinal", "block_ordinal", "instruction_ordinal"},
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
  llvm::Expected<std::string> engineText =
      requireString(**object, "engine", context, limits);
  if (!engineText)
    return engineText.takeError();
  llvm::Expected<ProfileTSMEngine> engine =
      parseEngine(*engineText, context + ".engine");
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
  if (descriptors[*targetCallOrdinal].symbol != *symbol)
    return invalid(context +
                   " target-call registry ordinal/symbol do not agree");
  std::optional<ProfileTSMEngine> descriptorEngine =
      getDescriptorEngine(descriptors[*targetCallOrdinal]);
  if (!descriptorEngine)
    return invalid(context +
                   " names a target call that does not reach TsmExecute");
  if (*descriptorEngine != *engine)
    return invalid(context + " target-call registry semantic/engine do not "
                             "agree");

  site.siteId = *siteId;
  site.targetCallOrdinal = *targetCallOrdinal;
  site.targetCallSymbol = *symbol;
  site.engine = *engine;
  site.correlationKey = *correlationKey;
  site.functionOrdinal = *functionOrdinal;
  site.blockOrdinal = *blockOrdinal;
  site.instructionOrdinal = *instructionOrdinal;
  return site;
}

llvm::Expected<ProfileRankSiteMap>
parseRankSiteMap(const llvm::json::Value &value, uint64_t index,
                 const PackageParseLimits &limits, uint64_t &totalRecords,
                 llvm::StringRef variantContext) {
  std::string context =
      (variantContext + ".ranks[" + llvm::Twine(index) + "]").str();
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, context);
  if (!object)
    return object.takeError();
  if (llvm::Error error =
          requireFields(**object, {"logical_rank", "sites"}, {}, context))
    return std::move(error);
  llvm::Expected<uint64_t> rank =
      requireUnsigned(**object, "logical_rank", context);
  if (!rank)
    return rank.takeError();
  if (*rank >= static_cast<uint64_t>(kProfileCompanionRankCount))
    return invalid(context + ".logical_rank is outside 0..15");
  llvm::Expected<const llvm::json::Array *> sites =
      requireArray(**object, "sites", context);
  if (!sites)
    return sites.takeError();
  if (llvm::Error error =
          accountRecords(1 + (*sites)->size(), totalRecords, limits))
    return std::move(error);

  ProfileRankSiteMap result;
  result.logicalRank = static_cast<int64_t>(*rank);
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
      return invalid(context + " correlation keys must be unique per rank");
  return result;
}

llvm::Expected<std::vector<ProfileVariantSiteMap>>
parseSiteMaps(const llvm::json::Object &root, const PackageParseLimits &limits,
              uint64_t &totalRecords) {
  if (llvm::Error error = requireFields(
          root,
          {"schema", "schema_version", "site_basis", "correlation_basis",
           "target_call_registry_size", "variants"},
          {}, "profile site map"))
    return std::move(error);
  if (llvm::Error error =
          verifyHeader(root, kSiteMapSchema, /*hasRankCount=*/false, limits,
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
  llvm::Expected<const llvm::json::Array *> variants =
      requireArray(root, "variants", "profile site map");
  if (!variants)
    return variants.takeError();
  if ((*variants)->size() != 2)
    return invalid("profile site map must contain exactly two variants");
  if (llvm::Error error =
          accountRecords((*variants)->size(), totalRecords, limits))
    return std::move(error);

  std::vector<ProfileVariantSiteMap> result;
  for (auto [index, value] : llvm::enumerate(**variants)) {
    std::string context =
        "profile site map.variants[" + std::to_string(index) + "]";
    llvm::Expected<const llvm::json::Object *> object =
        requireObject(value, context);
    if (!object)
      return object.takeError();
    if (llvm::Error error = requireFields(**object, {"variant_id"},
                                          {"same_as", "ranks"}, context))
      return std::move(error);
    const bool hasSameAs = (*object)->find("same_as") != (*object)->end();
    const bool hasRanks = (*object)->find("ranks") != (*object)->end();
    if (hasSameAs == hasRanks)
      return invalid(context + " must contain exactly one of same_as or ranks");

    ProfileVariantSiteMap variant;
    llvm::Expected<std::string> id =
        requireString(**object, "variant_id", context, limits);
    if (!id)
      return id.takeError();
    variant.variantId = *id;
    if (hasSameAs) {
      llvm::Expected<std::string> sameAs =
          requireString(**object, "same_as", context, limits);
      if (!sameAs)
        return sameAs.takeError();
      variant.sameAs = *sameAs;
    } else {
      llvm::Expected<const llvm::json::Array *> ranks =
          requireArray(**object, "ranks", context);
      if (!ranks)
        return ranks.takeError();
      if ((*ranks)->size() != static_cast<size_t>(kProfileCompanionRankCount))
        return invalid(context + " must contain all and only 16 ranks");
      variant.ranks.reserve((*ranks)->size());
      for (auto [rankIndex, rankValue] : llvm::enumerate(**ranks)) {
        llvm::Expected<ProfileRankSiteMap> rank = parseRankSiteMap(
            rankValue, rankIndex, limits, totalRecords, context);
        if (!rank)
          return rank.takeError();
        variant.ranks.push_back(std::move(*rank));
      }
      llvm::sort(variant.ranks, [](const auto &lhs, const auto &rhs) {
        return lhs.logicalRank < rhs.logicalRank;
      });
      for (auto [rankIndex, rank] : llvm::enumerate(variant.ranks))
        if (rank.logicalRank != static_cast<int64_t>(rankIndex))
          return invalid(context +
                         " must contain each logical rank exactly once");
    }
    result.push_back(std::move(variant));
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
    return invalid("profile variant manifest is not a regular file");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(manifest, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read profile variant manifest");
  llvm::SHA256 hasher;
  hasher.update((*buffer)->getBuffer());
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

bool isSafePackageReferenceSyntax(llvm::StringRef reference,
                                  bool productionWinner) {
  if (reference.empty() || llvm::sys::path::is_absolute(reference) ||
      reference.contains('\\'))
    return false;
  llvm::SmallVector<llvm::StringRef, 4> components;
  reference.split(components, '/', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  if (productionWinner)
    return components.size() == 2 && components[0] == ".." &&
           !components[1].empty() && components[1] != "." &&
           components[1] != "..";
  return components.size() == 1 && components[0] == kBaselinePackageDirectory;
}

llvm::Expected<std::string>
resolvePackageReference(llvm::StringRef companionRoot,
                        llvm::StringRef reference, bool productionWinner) {
  if (!isSafePackageReferenceSyntax(reference, productionWinner))
    return invalid("profile variant package_ref is not a safe canonical "
                   "relative reference");
  llvm::SmallString<256> candidate(companionRoot);
  llvm::sys::path::append(candidate, reference);
  if (llvm::sys::fs::get_file_type(candidate, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("profile variant package_ref is not a directory");
  llvm::SmallString<256> canonical;
  if (std::error_code error = llvm::sys::fs::real_path(candidate, canonical))
    return llvm::createStringError(error,
                                   "failed to resolve profile package_ref");
  return canonical.str().str();
}

bool isPathWithin(llvm::StringRef path, llvm::StringRef parent) {
  if (!path.starts_with(parent) || path.size() <= parent.size())
    return false;
  return llvm::sys::path::is_separator(path[parent.size()]);
}

llvm::Expected<std::string>
resolveCapturePackageReference(llvm::StringRef companionRoot,
                               llvm::StringRef reference) {
  if (reference.empty() || llvm::sys::path::is_absolute(reference) ||
      reference.contains('\\'))
    return invalid("profile capture package_ref is not a safe relative path");
  llvm::SmallVector<llvm::StringRef, 4> components;
  reference.split(components, '/', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  if (components.size() != 3 || components[0] != "captures" ||
      (components[1] != kProductionWinner &&
       components[1] != kReservedBaseline) ||
      (components[2] != "summary" && components[2] != "count" &&
       components[2] != "trace"))
    return invalid("profile capture package_ref is not canonical");
  llvm::SmallString<256> candidate(companionRoot);
  llvm::sys::path::append(candidate, reference);
  if (llvm::sys::fs::get_file_type(candidate, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("profile capture package_ref is not a directory");
  llvm::SmallString<256> canonical;
  if (std::error_code error = llvm::sys::fs::real_path(candidate, canonical))
    return llvm::createStringError(
        error, "failed to resolve profile capture package_ref");
  if (!isPathWithin(canonical, companionRoot))
    return invalid("profile capture package_ref escapes the companion root");
  return canonical.str().str();
}

const RawVariant *findRawVariant(llvm::ArrayRef<RawVariant> variants,
                                 llvm::StringRef id) {
  auto iterator = llvm::find_if(
      variants, [&](const RawVariant &variant) { return variant.id == id; });
  return iterator == variants.end() ? nullptr : &*iterator;
}

const RawExecutionPackage *
findExecutionPackage(llvm::ArrayRef<RawExecutionPackage> packages,
                     llvm::StringRef id) {
  auto iterator = llvm::find_if(
      packages, [&](const auto &package) { return package.variantId == id; });
  return iterator == packages.end() ? nullptr : &*iterator;
}

const RawCapturePackage *
findCapturePackage(llvm::ArrayRef<RawCapturePackage> packages,
                   llvm::StringRef variantId, ProfileCaptureKind capture) {
  auto iterator = llvm::find_if(packages, [&](const auto &package) {
    return package.variantId == variantId && package.capture == capture;
  });
  return iterator == packages.end() ? nullptr : &*iterator;
}

const ProfileVariantSiteMap *
findRawSiteMap(llvm::ArrayRef<ProfileVariantSiteMap> maps, llvm::StringRef id) {
  auto iterator =
      llvm::find_if(maps, [&](const auto &map) { return map.variantId == id; });
  return iterator == maps.end() ? nullptr : &*iterator;
}

llvm::Error verifyVariantGraph(llvm::ArrayRef<RawVariant> variants,
                               const RawPlan &plan,
                               llvm::ArrayRef<ProfileVariantSiteMap> siteMaps) {
  const RawVariant *winner = findRawVariant(variants, kProductionWinner);
  const RawVariant *baseline = findRawVariant(variants, kReservedBaseline);
  if (!winner || !baseline || winner == baseline)
    return invalid("profile companion must define one production winner and "
                   "one reserved baseline");
  if (winner->role != kProductionWinner || baseline->role != kReservedBaseline)
    return invalid("profile variant id and role do not agree");
  if (winner->sameAs)
    return invalid("production winner cannot alias another variant");
  if (baseline->sameAs && *baseline->sameAs != kProductionWinner)
    return invalid("reserved baseline same_as must name production-winner");
  if (baseline->sameAs &&
      (baseline->packageReference != winner->packageReference ||
       baseline->manifestDigest != winner->manifestDigest))
    return invalid("aliased reserved baseline package identity differs from "
                   "production winner");

  const RawExecutionPackage *winnerExecution =
      findExecutionPackage(plan.executionPackages, kProductionWinner);
  const RawExecutionPackage *baselineExecution =
      findExecutionPackage(plan.executionPackages, kReservedBaseline);
  if (!winnerExecution || !baselineExecution ||
      winnerExecution == baselineExecution)
    return invalid("profile plan execution package IDs are not unique");
  for (const auto &[variant, execution] :
       {std::pair<const RawVariant *, const RawExecutionPackage *>(
            winner, winnerExecution),
        std::pair<const RawVariant *, const RawExecutionPackage *>(
            baseline, baselineExecution)}) {
    if (variant->packageReference != execution->packageReference ||
        variant->manifestDigest != execution->manifestDigest)
      return invalid("profile plan and variant package identities disagree");
  }
  for (ProfileCaptureKind capture :
       {ProfileCaptureKind::Summary, ProfileCaptureKind::Count,
        ProfileCaptureKind::Trace}) {
    const RawCapturePackage *winnerCapture =
        findCapturePackage(plan.capturePackages, kProductionWinner, capture);
    const RawCapturePackage *baselineCapture =
        findCapturePackage(plan.capturePackages, kReservedBaseline, capture);
    if (!winnerCapture || !baselineCapture || winnerCapture == baselineCapture)
      return invalid("profile plan capture package identities are incomplete");
    std::string captureName = stringifyProfileCaptureKind(capture).str();
    std::string winnerReference =
        ("captures/" + kProductionWinner + "/" + captureName).str();
    if (winnerCapture->packageReference != winnerReference)
      return invalid("production capture package_ref is not canonical");
    if (baseline->sameAs) {
      if (baselineCapture->packageReference !=
              winnerCapture->packageReference ||
          baselineCapture->manifestDigest != winnerCapture->manifestDigest ||
          baselineCapture->recordBytes != winnerCapture->recordBytes)
        return invalid("aliased baseline capture identity differs from "
                       "production winner capture");
    } else {
      std::string baselineReference =
          ("captures/" + kReservedBaseline + "/" + captureName).str();
      if (baselineCapture->packageReference != baselineReference)
        return invalid("baseline capture package_ref is not canonical");
    }
  }

  const ProfileVariantSiteMap *winnerMap =
      findRawSiteMap(siteMaps, kProductionWinner);
  const ProfileVariantSiteMap *baselineMap =
      findRawSiteMap(siteMaps, kReservedBaseline);
  if (!winnerMap || !baselineMap || winnerMap == baselineMap)
    return invalid("profile site-map variant IDs are not unique");
  if (winnerMap->sameAs || winnerMap->ranks.size() !=
                               static_cast<size_t>(kProfileCompanionRankCount))
    return invalid("production winner site map must contain all 16 ranks");
  if (baseline->sameAs) {
    if (!baselineMap->sameAs || *baselineMap->sameAs != kProductionWinner ||
        !baselineMap->ranks.empty())
      return invalid("aliased reserved baseline site map is inconsistent");
  } else if (baselineMap->sameAs ||
             baselineMap->ranks.size() !=
                 static_cast<size_t>(kProfileCompanionRankCount)) {
    return invalid("distinct reserved baseline site map must contain all 16 "
                   "ranks");
  }
  return llvm::Error::success();
}

bool sameSemanticResource(const PackageResourceRecord &lhs,
                          const PackageResourceRecord &rhs) {
  return lhs.logicalRank == rhs.logicalRank && lhs.role == rhs.role &&
         lhs.roleIndex == rhs.roleIndex && lhs.type.dtype == rhs.type.dtype &&
         lhs.type.shape == rhs.type.shape && lhs.bytes == rhs.bytes &&
         lhs.alignment == rhs.alignment && lhs.access == rhs.access &&
         lhs.hostVisible == rhs.hostVisible;
}

const PackageResourceRecord *findResource(const PackageManifest &manifest,
                                          ResourceId id) {
  auto iterator = llvm::find_if(manifest.resources, [&](const auto &resource) {
    return resource.id == id;
  });
  return iterator == manifest.resources.end() ? nullptr : &*iterator;
}

const PackageEntrypointRecord *findEntryForRank(const PackageManifest &manifest,
                                                int64_t rank) {
  auto iterator = llvm::find_if(manifest.entries, [&](const auto &entry) {
    return entry.logicalRank == rank;
  });
  return iterator == manifest.entries.end() ? nullptr : &*iterator;
}

bool sameTransport(const PackageManifest &lhsManifest,
                   const TransportRequirements &lhs,
                   const PackageManifest &rhsManifest,
                   const TransportRequirements &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  const auto *lhsDirect = std::get_if<DirectDTETransportRequirements>(&lhs);
  const auto *rhsDirect = std::get_if<DirectDTETransportRequirements>(&rhs);
  if (!lhsDirect)
    return true;
  const PackageResourceRecord *lhsStatus =
      findResource(lhsManifest, lhsDirect->statusResource);
  const PackageResourceRecord *rhsStatus =
      rhsDirect ? findResource(rhsManifest, rhsDirect->statusResource)
                : nullptr;
  return rhsDirect && lhsStatus && rhsStatus &&
         lhsDirect->statusABI == rhsDirect->statusABI &&
         lhsDirect->hostWatchdogRequired == rhsDirect->hostWatchdogRequired &&
         sameSemanticResource(*lhsStatus, *rhsStatus);
}

std::vector<const PackageResourceRecord *>
getHostVisibleResources(const PackageManifest &manifest) {
  std::vector<const PackageResourceRecord *> resources;
  for (const PackageResourceRecord &resource : manifest.resources)
    if (resource.hostVisible)
      resources.push_back(&resource);
  llvm::sort(resources, [](const auto *lhs, const auto *rhs) {
    return std::tuple(lhs->logicalRank, static_cast<int>(lhs->role),
                      lhs->roleIndex) < std::tuple(rhs->logicalRank,
                                                   static_cast<int>(rhs->role),
                                                   rhs->roleIndex);
  });
  return resources;
}

llvm::Expected<std::vector<const PackageResourceRecord *>>
getEntryHostVisibleSlots(const PackageManifest &manifest,
                         const PackageEntrypointRecord &entry) {
  std::vector<const PackageResourceRecord *> result;
  for (const PackageABISlotBinding &slot : entry.slots) {
    const PackageResourceRecord *resource =
        findResource(manifest, slot.resource);
    if (!resource)
      return invalid("profile variant entry references a missing resource");
    if (resource->hostVisible)
      result.push_back(resource);
  }
  return result;
}

llvm::Error verifyPackageContractsMatch(const PackageManifest &reference,
                                        const PackageManifest &candidate) {
  if (reference.targetProfile != candidate.targetProfile ||
      reference.targetIdentity != candidate.targetIdentity ||
      reference.runtimeABI != candidate.runtimeABI ||
      reference.launchABI != candidate.launchABI ||
      reference.moduleFormat != candidate.moduleFormat)
    return invalid("profile variant target/ABI contracts differ");
  if (reference.rankCount != kProfileCompanionRankCount ||
      candidate.rankCount != kProfileCompanionRankCount)
    return invalid("profile variant packages must each contain 16 ranks");
  std::vector<const PackageResourceRecord *> referenceResources =
      getHostVisibleResources(reference);
  std::vector<const PackageResourceRecord *> candidateResources =
      getHostVisibleResources(candidate);
  if (referenceResources.size() != candidateResources.size())
    return invalid("profile variant host-visible resource contracts differ");
  for (auto [lhs, rhs] : llvm::zip(referenceResources, candidateResources))
    if (!sameSemanticResource(*lhs, *rhs))
      return invalid("profile variant host-visible resource contracts differ");

  if (reference.entries.size() != candidate.entries.size())
    return invalid("profile variant entry ABI contracts differ");
  for (int64_t rank = 0; rank < kProfileCompanionRankCount; ++rank) {
    const PackageEntrypointRecord *lhs = findEntryForRank(reference, rank);
    const PackageEntrypointRecord *rhs = findEntryForRank(candidate, rank);
    if (!lhs || !rhs ||
        !sameTransport(reference, lhs->transport, candidate, rhs->transport))
      return invalid("profile variant entry ABI contracts differ");
    llvm::Expected<std::vector<const PackageResourceRecord *>> lhsSlots =
        getEntryHostVisibleSlots(reference, *lhs);
    if (!lhsSlots)
      return lhsSlots.takeError();
    llvm::Expected<std::vector<const PackageResourceRecord *>> rhsSlots =
        getEntryHostVisibleSlots(candidate, *rhs);
    if (!rhsSlots)
      return rhsSlots.takeError();
    if (lhsSlots->size() != rhsSlots->size())
      return invalid("profile variant entry ABI contracts differ");
    for (auto [lhsResource, rhsResource] : llvm::zip(*lhsSlots, *rhsSlots))
      if (!sameSemanticResource(*lhsResource, *rhsResource))
        return invalid("profile variant entry ABI contracts differ");
  }
  return llvm::Error::success();
}

bool isProfilerRecordResource(const PackageResourceRecord &resource,
                              uint64_t recordBytes) {
  return resource.role == PackageResourceRole::Workspace &&
         resource.roleIndex == 1 && resource.type.dtype == "u8" &&
         resource.type.shape ==
             std::vector<int64_t>{static_cast<int64_t>(recordBytes)} &&
         resource.bytes == recordBytes &&
         resource.alignment == WAFER_TX81_PROFILER_BUFFER_ALIGNMENT &&
         resource.access == PackageAccessMode::ReadWrite &&
         !resource.hostVisible;
}

llvm::Error verifyCapturePackageContract(const PackageManifest &execution,
                                         const PackageManifest &capture,
                                         uint64_t recordBytes) {
  if (execution.targetProfile != capture.targetProfile ||
      execution.targetIdentity != capture.targetIdentity ||
      execution.runtimeABI != capture.runtimeABI ||
      execution.launchABI != capture.launchABI ||
      execution.moduleFormat != capture.moduleFormat)
    return invalid("profile capture target/ABI contract differs from its "
                   "execution package");
  if (execution.rankCount != kProfileCompanionRankCount ||
      capture.rankCount != kProfileCompanionRankCount)
    return invalid("profile capture packages must each contain 16 ranks");

  std::vector<const PackageResourceRecord *> executionResources;
  std::vector<const PackageResourceRecord *> captureResources;
  std::array<const PackageResourceRecord *, kProfileCompanionRankCount>
      profilerResources{};
  for (const PackageResourceRecord &resource : execution.resources) {
    if (resource.role == PackageResourceRole::Workspace &&
        resource.roleIndex == 1)
      return invalid("execution package occupies the reserved profiler "
                     "workspace identity");
    executionResources.push_back(&resource);
  }
  for (const PackageResourceRecord &resource : capture.resources) {
    if (resource.role == PackageResourceRole::Workspace &&
        resource.roleIndex == 1) {
      if (!isProfilerRecordResource(resource, recordBytes))
        return invalid("profile capture has an invalid profiler workspace");
      if (resource.logicalRank < 0 ||
          resource.logicalRank >= kProfileCompanionRankCount ||
          profilerResources[resource.logicalRank])
        return invalid("profile capture profiler workspace rank domain is "
                       "not unique");
      profilerResources[resource.logicalRank] = &resource;
      continue;
    }
    captureResources.push_back(&resource);
  }
  if (llvm::any_of(profilerResources,
                   [](const auto *resource) { return resource == nullptr; }))
    return invalid("profile capture must contain one profiler workspace for "
                   "each of 16 ranks");
  auto bySemanticIdentity = [](const auto *lhs, const auto *rhs) {
    return std::tuple(lhs->logicalRank, static_cast<int>(lhs->role),
                      lhs->roleIndex) < std::tuple(rhs->logicalRank,
                                                   static_cast<int>(rhs->role),
                                                   rhs->roleIndex);
  };
  llvm::sort(executionResources, bySemanticIdentity);
  llvm::sort(captureResources, bySemanticIdentity);
  if (executionResources.size() != captureResources.size())
    return invalid("profile capture contains resources other than its exact "
                   "profiler workspace extension");
  for (auto [lhs, rhs] : llvm::zip(executionResources, captureResources))
    if (!sameSemanticResource(*lhs, *rhs))
      return invalid("profile capture base resource contract differs from its "
                     "execution package");

  for (int64_t rank = 0; rank < kProfileCompanionRankCount; ++rank) {
    const PackageEntrypointRecord *executionEntry =
        findEntryForRank(execution, rank);
    const PackageEntrypointRecord *captureEntry =
        findEntryForRank(capture, rank);
    if (!executionEntry || !captureEntry ||
        captureEntry->slots.size() != executionEntry->slots.size() + 1 ||
        !sameTransport(execution, executionEntry->transport, capture,
                       captureEntry->transport))
      return invalid("profile capture entry ABI extension is invalid");
    const PackageABISlotBinding &profilerSlot = captureEntry->slots.back();
    if (profilerSlot.resource != profilerResources[rank]->id ||
        profilerSlot.access != PackageAccessMode::ReadWrite)
      return invalid("profile capture profiler workspace is not the final "
                     "entry slot");
    for (size_t index = 0; index < executionEntry->slots.size(); ++index) {
      const PackageResourceRecord *executionResource =
          findResource(execution, executionEntry->slots[index].resource);
      const PackageResourceRecord *captureResource =
          findResource(capture, captureEntry->slots[index].resource);
      if (!executionResource || !captureResource ||
          executionEntry->slots[index].access !=
              captureEntry->slots[index].access ||
          !sameSemanticResource(*executionResource, *captureResource))
        return invalid("profile capture entry ABI base slots differ from its "
                       "execution package");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<ProfileVariantPackage>
loadVariantPackage(const RawVariant &variant, llvm::StringRef companionRoot,
                   llvm::StringRef productionPackageRoot,
                   const PackageParseLimits &limits) {
  const bool isWinner = variant.id == kProductionWinner;
  const bool referencesProduction = isWinner || variant.sameAs.has_value();
  if (!isLowercaseSHA256(variant.manifestDigest))
    return invalid("profile variant manifest_sha256 is malformed");
  llvm::Expected<std::string> packageDirectory = resolvePackageReference(
      companionRoot, variant.packageReference, referencesProduction);
  if (!packageDirectory)
    return packageDirectory.takeError();
  if (referencesProduction) {
    if (*packageDirectory != productionPackageRoot)
      return invalid("profile production package reference does not name the "
                     "selected ordinary package");
  } else if (!variant.sameAs &&
             !isPathWithin(*packageDirectory, companionRoot)) {
    return invalid("profile reserved baseline escapes the companion root");
  }

  llvm::Expected<std::string> digest = digestManifest(*packageDirectory);
  if (!digest)
    return digest.takeError();
  if (*digest != variant.manifestDigest)
    return invalid("profile variant manifest digest mismatch");
  llvm::Expected<VerifiedPackageManifest> package =
      loadVerifiedPackageManifest(*packageDirectory, limits);
  if (!package)
    return package.takeError();

  ProfileVariantRole role = isWinner ? ProfileVariantRole::ProductionWinner
                                     : ProfileVariantRole::ReservedBaseline;
  return ProfileVariantPackage(variant.id, role, variant.packageReference,
                               variant.manifestDigest, variant.sameAs,
                               *packageDirectory, std::move(*package));
}

llvm::Expected<ProfileCapturePackage>
loadCapturePackage(const RawCapturePackage &capture,
                   llvm::StringRef companionRoot,
                   const ProfileVariantPackage &executionVariant,
                   const PackageParseLimits &limits) {
  if (!isLowercaseSHA256(capture.manifestDigest))
    return invalid("profile capture manifest_sha256 is malformed");
  if (capture.recordBytes != expectedRecordBytes(capture.capture))
    return invalid("profile capture record_bytes is inconsistent");
  llvm::Expected<std::string> packageDirectory =
      resolveCapturePackageReference(companionRoot, capture.packageReference);
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
          executionVariant.getPackage().getManifest(), package->getManifest(),
          capture.recordBytes))
    return std::move(error);
  return ProfileCapturePackage(capture.variantId, capture.capture,
                               capture.packageReference, capture.manifestDigest,
                               capture.recordBytes, *packageDirectory,
                               std::move(*package));
}

} // namespace

ProfileVariantPackage::ProfileVariantPackage(
    std::string id, ProfileVariantRole role, std::string packageReference,
    std::string manifestDigest, std::optional<std::string> sameAs,
    std::string packageDirectory, VerifiedPackageManifest package)
    : id(std::move(id)), role(role),
      packageReference(std::move(packageReference)),
      manifestDigest(std::move(manifestDigest)), sameAs(std::move(sameAs)),
      packageDirectory(std::move(packageDirectory)),
      package(std::move(package)) {}

ProfileCapturePackage::ProfileCapturePackage(std::string variantId,
                                             ProfileCaptureKind capture,
                                             std::string packageReference,
                                             std::string manifestDigest,
                                             uint64_t recordBytes,
                                             std::string packageDirectory,
                                             VerifiedPackageManifest package)
    : variantId(std::move(variantId)), capture(capture),
      packageReference(std::move(packageReference)),
      manifestDigest(std::move(manifestDigest)), recordBytes(recordBytes),
      packageDirectory(std::move(packageDirectory)),
      package(std::move(package)) {}

llvm::StringRef stringifyProfileVariantRole(ProfileVariantRole role) {
  switch (role) {
  case ProfileVariantRole::ProductionWinner:
    return kProductionWinner;
  case ProfileVariantRole::ReservedBaseline:
    return kReservedBaseline;
  }
  llvm_unreachable("unknown profile variant role");
}

llvm::StringRef stringifyProfileCaptureKind(ProfileCaptureKind capture) {
  switch (capture) {
  case ProfileCaptureKind::Summary:
    return "summary";
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
  }
  llvm_unreachable("unknown profile TSM engine");
}

const ProfileVariantPackage *
VerifiedProfileCompanion::findVariant(ProfileVariantRole role) const {
  auto iterator = llvm::find_if(
      variants, [&](const auto &variant) { return variant.getRole() == role; });
  return iterator == variants.end() ? nullptr : &*iterator;
}

const ProfileCapturePackage *
VerifiedProfileCompanion::findCapture(llvm::StringRef variantId,
                                      ProfileCaptureKind capture) const {
  auto iterator = llvm::find_if(captures, [&](const auto &package) {
    return package.getVariantId() == variantId &&
           package.getCaptureKind() == capture;
  });
  return iterator == captures.end() ? nullptr : &*iterator;
}

const ProfileVariantSiteMap *
VerifiedProfileCompanion::findSiteMap(llvm::StringRef variantId) const {
  auto iterator = llvm::find_if(
      siteMaps, [&](const auto &map) { return map.variantId == variantId; });
  return iterator == siteMaps.end() ? nullptr : &*iterator;
}

uint64_t VerifiedProfileCompanion::getSiteCount() const {
  uint64_t count = 0;
  for (const ProfileVariantSiteMap &variant : siteMaps)
    for (const ProfileRankSiteMap &rank : variant.ranks)
      count += rank.sites.size();
  return count;
}

llvm::Expected<VerifiedProfileCompanion>
loadVerifiedProfileCompanion(llvm::StringRef companionRoot,
                             llvm::StringRef productionPackageRoot,
                             const PackageParseLimits &limits) {
  if (llvm::sys::fs::get_file_type(companionRoot, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("profile companion root is not a directory");
  if (llvm::sys::fs::get_file_type(productionPackageRoot,
                                   /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("selected production package root is not a directory");
  llvm::SmallString<256> canonicalCompanion;
  if (std::error_code error =
          llvm::sys::fs::real_path(companionRoot, canonicalCompanion))
    return llvm::createStringError(error,
                                   "failed to resolve profile companion root");
  llvm::SmallString<256> canonicalProduction;
  if (std::error_code error =
          llvm::sys::fs::real_path(productionPackageRoot, canonicalProduction))
    return llvm::createStringError(
        error, "failed to resolve selected production package");

  llvm::SmallString<256> activationPath(canonicalCompanion);
  llvm::sys::path::append(activationPath, kProfileCompanionActivationFileName);
  llvm::SmallString<256> planPath(canonicalCompanion);
  llvm::sys::path::append(planPath, kProfileCompanionPlanFileName);
  llvm::SmallString<256> variantsPath(canonicalCompanion);
  llvm::sys::path::append(variantsPath, kProfileCompanionVariantsFileName);
  llvm::SmallString<256> siteMapPath(canonicalCompanion);
  llvm::sys::path::append(siteMapPath, kProfileCompanionSiteMapFileName);

  llvm::Expected<LoadedJSONDocument> activationJSON =
      loadJSONDocument(activationPath, "profile activation", limits);
  if (!activationJSON)
    return activationJSON.takeError();
  llvm::Expected<RawActivation> activation =
      parseActivation(*activationJSON->root.getAsObject(), limits);
  if (!activation)
    return activation.takeError();

  llvm::Expected<std::string> productionDigest =
      digestManifest(canonicalProduction);
  if (!productionDigest)
    return productionDigest.takeError();
  if (*productionDigest != activation->productionManifestDigest)
    return invalid("profile activation production manifest digest mismatch");

  llvm::Expected<LoadedJSONDocument> planJSON =
      loadJSONDocument(planPath, "profile plan", limits);
  if (!planJSON)
    return planJSON.takeError();
  llvm::Expected<LoadedJSONDocument> variantsJSON =
      loadJSONDocument(variantsPath, "profile variants", limits);
  if (!variantsJSON)
    return variantsJSON.takeError();
  llvm::Expected<LoadedJSONDocument> siteMapJSON =
      loadJSONDocument(siteMapPath, "profile site map", limits);
  if (!siteMapJSON)
    return siteMapJSON.takeError();
  if (planJSON->digest != activation->planDigest ||
      variantsJSON->digest != activation->variantsDigest ||
      siteMapJSON->digest != activation->siteMapDigest)
    return invalid("profile activation metadata digest mismatch");

  uint64_t totalRecords = 0;
  llvm::Expected<RawPlan> plan =
      parsePlan(*planJSON->root.getAsObject(), limits, totalRecords);
  if (!plan)
    return plan.takeError();
  llvm::Expected<std::vector<RawVariant>> rawVariants =
      parseVariants(*variantsJSON->root.getAsObject(), limits, totalRecords);
  if (!rawVariants)
    return rawVariants.takeError();
  llvm::Expected<std::vector<ProfileVariantSiteMap>> siteMaps =
      parseSiteMaps(*siteMapJSON->root.getAsObject(), limits, totalRecords);
  if (!siteMaps)
    return siteMaps.takeError();
  if (llvm::Error error = verifyVariantGraph(*rawVariants, *plan, *siteMaps))
    return std::move(error);

  std::vector<ProfileVariantPackage> packages;
  packages.reserve(rawVariants->size());
  for (const RawVariant &variant : *rawVariants) {
    llvm::Expected<ProfileVariantPackage> package = loadVariantPackage(
        variant, canonicalCompanion, canonicalProduction, limits);
    if (!package)
      return package.takeError();
    packages.push_back(std::move(*package));
  }
  const auto winner = llvm::find_if(packages, [](const auto &variant) {
    return variant.getRole() == ProfileVariantRole::ProductionWinner;
  });
  const auto baseline = llvm::find_if(packages, [](const auto &variant) {
    return variant.getRole() == ProfileVariantRole::ReservedBaseline;
  });
  if (winner == packages.end() || baseline == packages.end())
    return invalid("profile companion variant roles are incomplete");
  if (llvm::Error error =
          verifyPackageContractsMatch(winner->getPackage().getManifest(),
                                      baseline->getPackage().getManifest()))
    return std::move(error);

  std::vector<ProfileCapturePackage> captures;
  captures.reserve(plan->capturePackages.size());
  for (const RawCapturePackage &rawCapture : plan->capturePackages) {
    const auto execution =
        llvm::find_if(packages, [&](const ProfileVariantPackage &variant) {
          return variant.getId() == rawCapture.variantId;
        });
    if (execution == packages.end())
      return invalid("profile capture references an unknown variant");
    llvm::Expected<ProfileCapturePackage> capture =
        loadCapturePackage(rawCapture, canonicalCompanion, *execution, limits);
    if (!capture)
      return capture.takeError();
    captures.push_back(std::move(*capture));
  }

  return VerifiedProfileCompanion(
      canonicalCompanion.str().str(), activation->productionManifestDigest,
      std::move(packages), std::move(captures), std::move(*siteMaps));
}

llvm::Expected<std::optional<VerifiedProfileCompanion>>
loadSiblingProfileCompanionIfPresent(llvm::StringRef productionPackageRoot,
                                     const PackageParseLimits &limits) {
  std::string companion = productionPackageRoot.str();
  while (!companion.empty() && llvm::sys::path::is_separator(companion.back()))
    companion.pop_back();
  companion += ".profile";

  llvm::sys::fs::file_status status;
  if (std::error_code error =
          llvm::sys::fs::status(companion, status, /*follow=*/false)) {
    if (error == std::errc::no_such_file_or_directory)
      return std::optional<VerifiedProfileCompanion>();
    return llvm::createStringError(
        error, "failed to inspect sibling profile companion");
  }
  if (!llvm::sys::fs::exists(status))
    return std::optional<VerifiedProfileCompanion>();
  llvm::Expected<VerifiedProfileCompanion> loaded =
      loadVerifiedProfileCompanion(companion, productionPackageRoot, limits);
  if (!loaded)
    return loaded.takeError();
  return std::optional<VerifiedProfileCompanion>(std::move(*loaded));
}

} // namespace wafer::runtime
