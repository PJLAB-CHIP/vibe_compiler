//===- PackageManifestJson.cpp - Typed manifest JSON decoding ------------===//

#include "PackageManifestInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace wafer::runtime {
namespace {

using detail::invalid;

bool exceedsJSONNesting(llvm::StringRef input, uint64_t maximum) {
  uint64_t depth = 0;
  bool inString = false;
  bool escaped = false;
  for (char character : input) {
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
    } else if (character == '{' || character == '[') {
      if (++depth > maximum)
        return true;
    } else if ((character == '}' || character == ']') && depth != 0) {
      --depth;
    }
  }
  return false;
}

llvm::Error requireExactFields(const llvm::json::Object &object,
                               std::initializer_list<llvm::StringRef> fields,
                               llvm::StringRef context) {
  llvm::StringSet<> expected;
  for (llvm::StringRef field : fields)
    expected.insert(field);
  for (const auto &member : object)
    if (!expected.contains(member.first))
      return invalid(context + " contains unknown field '" +
                     member.first.str() + "'");
  for (llvm::StringRef field : fields)
    if (object.find(field) == object.end())
      return invalid(context + " is missing field '" + field + "'");
  return llvm::Error::success();
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef field,
              llvm::StringRef context) {
  if (const llvm::json::Object *value = object.getObject(field))
    return value;
  return invalid(context + "." + field + " must be an object");
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef field,
             llvm::StringRef context) {
  if (const llvm::json::Array *value = object.getArray(field))
    return value;
  return invalid(context + "." + field + " must be an array");
}

llvm::Expected<std::string> requireString(const llvm::json::Object &object,
                                          llvm::StringRef field,
                                          llvm::StringRef context,
                                          const PackageParseLimits &limits,
                                          bool allowEmpty = false) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value)
    return invalid(context + "." + field + " must be a string");
  if ((!allowEmpty && value->empty()) || value->size() > limits.maxStringBytes)
    return invalid(context + "." + field + " has invalid length");
  return value->str();
}

llvm::Expected<int64_t> requireInteger(const llvm::json::Object &object,
                                       llvm::StringRef field,
                                       llvm::StringRef context) {
  std::optional<int64_t> value = object.getInteger(field);
  if (!value)
    return invalid(context + "." + field + " must be an integer");
  return *value;
}

llvm::Expected<uint64_t> requireUnsigned(const llvm::json::Object &object,
                                         llvm::StringRef field,
                                         llvm::StringRef context) {
  llvm::Expected<int64_t> value = requireInteger(object, field, context);
  if (!value)
    return value.takeError();
  if (*value < 0)
    return invalid(context + "." + field + " must be non-negative");
  return static_cast<uint64_t>(*value);
}

llvm::Expected<bool> requireBoolean(const llvm::json::Object &object,
                                    llvm::StringRef field,
                                    llvm::StringRef context) {
  std::optional<bool> value = object.getBoolean(field);
  if (!value)
    return invalid(context + "." + field + " must be a boolean");
  return *value;
}

llvm::Expected<PackageResourceRole> parseRole(llvm::StringRef role) {
  if (role == "user_input")
    return PackageResourceRole::UserInput;
  if (role == "parameter")
    return PackageResourceRole::Parameter;
  if (role == "constant")
    return PackageResourceRole::Constant;
  if (role == "output")
    return PackageResourceRole::Output;
  if (role == "workspace")
    return PackageResourceRole::Workspace;
  if (role == "transport_status")
    return PackageResourceRole::TransportStatus;
  return invalid("unsupported package resource role '" + role + "'");
}

llvm::Expected<PackageAccessMode> parseAccess(llvm::StringRef access) {
  if (access == "read_only")
    return PackageAccessMode::ReadOnly;
  if (access == "write_only")
    return PackageAccessMode::WriteOnly;
  if (access == "read_write")
    return PackageAccessMode::ReadWrite;
  return invalid("unsupported package access mode '" + access + "'");
}

llvm::Expected<PackageModuleExportRole>
parseModuleExportRole(llvm::StringRef role) {
  if (role == "prepare")
    return PackageModuleExportRole::Prepare;
  if (role == "main")
    return PackageModuleExportRole::Main;
  return invalid("unsupported package module export role '" + role + "'");
}

llvm::Expected<std::vector<RuntimeLaunchPhaseRole>>
parseRuntimeLaunchPhases(const llvm::json::Object &object,
                         llvm::StringRef context,
                         const PackageParseLimits &limits) {
  llvm::Expected<const llvm::json::Array *> phases =
      requireArray(object, "phases", context);
  if (!phases)
    return phases.takeError();
  if ((*phases)->empty() || (*phases)->size() > 2)
    return invalid(context + ".phases must contain one or two ordered roles");

  std::vector<RuntimeLaunchPhaseRole> result;
  result.reserve((*phases)->size());
  for (auto [index, value] : llvm::enumerate(**phases)) {
    std::optional<llvm::StringRef> spelling = value.getAsString();
    const std::string phaseContext =
        (context + ".phases[" + llvm::Twine(index) + "]").str();
    if (!spelling)
      return invalid(phaseContext + " must be a string");
    if (spelling->empty() || spelling->size() > limits.maxStringBytes)
      return invalid(phaseContext + " has invalid length");
    llvm::Expected<RuntimeLaunchPhaseRole> phase =
        parseRuntimeLaunchPhaseRole(*spelling);
    if (!phase)
      return phase.takeError();
    result.push_back(*phase);
  }
  return result;
}

llvm::Expected<RuntimeLaunchContract>
parseRuntimeLaunchContract(const llvm::json::Object &object,
                           llvm::StringRef context,
                           const PackageParseLimits &limits) {
  llvm::Expected<std::string> kindSpelling =
      requireString(object, "kind", context, limits);
  if (!kindSpelling)
    return kindSpelling.takeError();
  llvm::Expected<RuntimeLaunchKind> kind =
      parseRuntimeLaunchKind(*kindSpelling);
  if (!kind)
    return kind.takeError();

  if (*kind == RuntimeLaunchKind::Kernel) {
    if (llvm::Error error = requireExactFields(
            object, {"kind", "form", "entry_abi", "phases"}, context))
      return std::move(error);
    llvm::Expected<std::string> formSpelling =
        requireString(object, "form", context, limits);
    if (!formSpelling)
      return formSpelling.takeError();
    llvm::Expected<KernelLaunchForm> form =
        parseKernelLaunchForm(*formSpelling);
    if (!form)
      return form.takeError();
    llvm::Expected<std::string> entryABISpelling =
        requireString(object, "entry_abi", context, limits);
    if (!entryABISpelling)
      return entryABISpelling.takeError();
    llvm::Expected<KernelEntryABI> entryABI =
        parseKernelEntryABI(*entryABISpelling);
    if (!entryABI)
      return entryABI.takeError();
    llvm::Expected<std::vector<RuntimeLaunchPhaseRole>> phases =
        parseRuntimeLaunchPhases(object, context, limits);
    if (!phases)
      return phases.takeError();
    return RuntimeLaunchContract::createKernel(*form, *entryABI, *phases);
  }

  if (llvm::Error error =
          requireExactFields(object, {"kind", "entry_abi", "phases"}, context))
    return std::move(error);
  llvm::Expected<std::string> entryABISpelling =
      requireString(object, "entry_abi", context, limits);
  if (!entryABISpelling)
    return entryABISpelling.takeError();
  llvm::Expected<ModelEntryABI> entryABI =
      parseModelEntryABI(*entryABISpelling);
  if (!entryABI)
    return entryABI.takeError();
  llvm::Expected<std::vector<RuntimeLaunchPhaseRole>> phases =
      parseRuntimeLaunchPhases(object, context, limits);
  if (!phases)
    return phases.takeError();
  return RuntimeLaunchContract::createModel(*entryABI, *phases);
}

llvm::Expected<PackageResourceRecord>
parseResource(const llvm::json::Value &value, uint64_t index,
              const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "resources[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = requireExactFields(
          *object,
          {"id", "rank", "role", "role_index", "name", "type", "bytes",
           "alignment", "access", "host_visible"},
          context))
    return std::move(error);

  PackageResourceRecord record;
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  llvm::Expected<int64_t> rank = requireInteger(*object, "rank", context);
  if (!rank)
    return rank.takeError();
  llvm::Expected<std::string> roleText =
      requireString(*object, "role", context, limits);
  if (!roleText)
    return roleText.takeError();
  llvm::Expected<int64_t> roleIndex =
      requireInteger(*object, "role_index", context);
  if (!roleIndex)
    return roleIndex.takeError();
  llvm::Expected<std::string> name =
      requireString(*object, "name", context, limits, /*allowEmpty=*/true);
  if (!name)
    return name.takeError();
  llvm::Expected<const llvm::json::Object *> type =
      requireObject(*object, "type", context);
  if (!type)
    return type.takeError();
  llvm::Expected<uint64_t> bytes = requireUnsigned(*object, "bytes", context);
  if (!bytes)
    return bytes.takeError();
  llvm::Expected<uint64_t> alignment =
      requireUnsigned(*object, "alignment", context);
  if (!alignment)
    return alignment.takeError();
  llvm::Expected<std::string> accessText =
      requireString(*object, "access", context, limits);
  if (!accessText)
    return accessText.takeError();
  llvm::Expected<bool> hostVisible =
      requireBoolean(*object, "host_visible", context);
  if (!hostVisible)
    return hostVisible.takeError();
  if (llvm::Error error =
          requireExactFields(**type, {"dtype", "shape"}, context + ".type"))
    return std::move(error);
  llvm::Expected<std::string> dtype =
      requireString(**type, "dtype", context + ".type", limits);
  if (!dtype)
    return dtype.takeError();
  llvm::Expected<const llvm::json::Array *> shape =
      requireArray(**type, "shape", context + ".type");
  if (!shape)
    return shape.takeError();
  if ((*shape)->size() > limits.maxShapeRank)
    return invalid(context + ".type.shape exceeds rank limit");

  llvm::Expected<PackageResourceRole> role = parseRole(*roleText);
  if (!role)
    return role.takeError();
  llvm::Expected<PackageAccessMode> access = parseAccess(*accessText);
  if (!access)
    return access.takeError();

  record.id = ResourceId(*id);
  record.logicalRank = *rank;
  record.role = *role;
  record.roleIndex = *roleIndex;
  record.name = std::move(*name);
  record.type.dtype = std::move(*dtype);
  for (const llvm::json::Value &dimensionValue : **shape) {
    std::optional<int64_t> dimension = dimensionValue.getAsInteger();
    if (!dimension || *dimension < 0)
      return invalid(context +
                     ".type.shape must contain non-negative integers");
    record.type.shape.push_back(*dimension);
  }
  record.bytes = *bytes;
  record.alignment = *alignment;
  record.access = *access;
  record.hostVisible = *hostVisible;
  return record;
}

llvm::Expected<PackageModuleRecord>
parseModuleRecord(const llvm::json::Value &value, uint64_t index,
                  const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "modules[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = requireExactFields(
          *object, {"id", "path", "digest", "format", "exports"}, context))
    return std::move(error);
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  llvm::Expected<std::string> path =
      requireString(*object, "path", context, limits);
  if (!path)
    return path.takeError();
  llvm::Expected<std::string> digest =
      requireString(*object, "digest", context, limits);
  if (!digest)
    return digest.takeError();
  llvm::Expected<std::string> format =
      requireString(*object, "format", context, limits);
  if (!format)
    return format.takeError();
  llvm::Expected<const llvm::json::Array *> exports =
      requireArray(*object, "exports", context);
  if (!exports)
    return exports.takeError();
  if ((*exports)->size() > limits.maxRecords)
    return invalid(context + ".exports exceeds record limit");

  PackageModuleRecord record{ModuleId(*id),
                             std::move(*path),
                             std::move(*digest),
                             std::move(*format),
                             {}};
  for (auto [exportIndex, exportValue] : llvm::enumerate(**exports)) {
    const llvm::json::Object *moduleExport = exportValue.getAsObject();
    std::string exportContext =
        context + ".exports[" + std::to_string(exportIndex) + "]";
    if (!moduleExport)
      return invalid(exportContext + " must be an object");
    if (llvm::Error error = requireExactFields(
            *moduleExport, {"role", "symbol"}, exportContext))
      return std::move(error);
    llvm::Expected<std::string> roleText =
        requireString(*moduleExport, "role", exportContext, limits);
    if (!roleText)
      return roleText.takeError();
    llvm::Expected<PackageModuleExportRole> role =
        parseModuleExportRole(*roleText);
    if (!role)
      return role.takeError();
    llvm::Expected<std::string> symbol =
        requireString(*moduleExport, "symbol", exportContext, limits);
    if (!symbol)
      return symbol.takeError();
    record.exports.push_back({*role, std::move(*symbol)});
  }
  return record;
}

llvm::Expected<PackageCompletionRecord>
parseCompletionRecord(const llvm::json::Value &value, uint64_t index,
                      const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "completions[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error =
          requireExactFields(*object, {"id", "rank", "kind"}, context))
    return std::move(error);
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  llvm::Expected<int64_t> rank = requireInteger(*object, "rank", context);
  if (!rank)
    return rank.takeError();
  llvm::Expected<std::string> kind =
      requireString(*object, "kind", context, limits);
  if (!kind)
    return kind.takeError();
  return PackageCompletionRecord{CompletionId(*id), *rank, std::move(*kind)};
}

llvm::Expected<TransportRequirements>
parseTransportRequirements(const llvm::json::Object &object,
                           llvm::StringRef context,
                           const PackageParseLimits &limits) {
  llvm::Expected<std::string> kind =
      requireString(object, "kind", context, limits);
  if (!kind)
    return kind.takeError();
  if (*kind == "none") {
    if (llvm::Error error = requireExactFields(object, {"kind"}, context))
      return std::move(error);
    return TransportRequirements{NoTransportRequirements{}};
  }
  if (*kind != "direct_dte")
    return invalid(context + " has unsupported transport kind '" + *kind + "'");
  if (llvm::Error error = requireExactFields(
          object,
          {"kind", "status_resource", "status_abi", "host_watchdog_required"},
          context))
    return std::move(error);
  llvm::Expected<uint64_t> statusResource =
      requireUnsigned(object, "status_resource", context);
  if (!statusResource)
    return statusResource.takeError();
  llvm::Expected<std::string> statusABI =
      requireString(object, "status_abi", context, limits);
  if (!statusABI)
    return statusABI.takeError();
  llvm::Expected<bool> watchdog =
      requireBoolean(object, "host_watchdog_required", context);
  if (!watchdog)
    return watchdog.takeError();
  return TransportRequirements{DirectDTETransportRequirements{
      ResourceId(*statusResource), std::move(*statusABI), *watchdog}};
}

llvm::Expected<PackageEntrypointRecord>
parseEntrypointRecord(const llvm::json::Value &value, uint64_t index,
                      const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "entries[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = requireExactFields(
          *object,
          {"id", "rank", "module", "slots", "terminal_completion", "transport"},
          context))
    return std::move(error);
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  llvm::Expected<int64_t> rank = requireInteger(*object, "rank", context);
  if (!rank)
    return rank.takeError();
  llvm::Expected<uint64_t> module = requireUnsigned(*object, "module", context);
  if (!module)
    return module.takeError();
  llvm::Expected<const llvm::json::Array *> slots =
      requireArray(*object, "slots", context);
  if (!slots)
    return slots.takeError();
  llvm::Expected<uint64_t> completion =
      requireUnsigned(*object, "terminal_completion", context);
  if (!completion)
    return completion.takeError();
  llvm::Expected<const llvm::json::Object *> transportObject =
      requireObject(*object, "transport", context);
  if (!transportObject)
    return transportObject.takeError();
  llvm::Expected<TransportRequirements> transport = parseTransportRequirements(
      **transportObject, context + ".transport", limits);
  if (!transport)
    return transport.takeError();
  if ((*slots)->size() > limits.maxRecords)
    return invalid(context + ".slots exceeds record limit");

  PackageEntrypointRecord record;
  record.id = EntryId(*id);
  record.logicalRank = *rank;
  record.module = ModuleId(*module);
  record.terminalCompletion = CompletionId(*completion);
  record.transport = std::move(*transport);
  for (auto [slotIndex, slotValue] : llvm::enumerate(**slots)) {
    const llvm::json::Object *slot = slotValue.getAsObject();
    std::string slotContext =
        context + ".slots[" + std::to_string(slotIndex) + "]";
    if (!slot)
      return invalid(slotContext + " must be an object");
    if (llvm::Error error = requireExactFields(
            *slot, {"ordinal", "resource", "access"}, slotContext))
      return std::move(error);
    llvm::Expected<uint64_t> ordinal =
        requireUnsigned(*slot, "ordinal", slotContext);
    if (!ordinal)
      return ordinal.takeError();
    llvm::Expected<uint64_t> resource =
        requireUnsigned(*slot, "resource", slotContext);
    if (!resource)
      return resource.takeError();
    llvm::Expected<std::string> accessText =
        requireString(*slot, "access", slotContext, limits);
    if (!accessText)
      return accessText.takeError();
    llvm::Expected<PackageAccessMode> access = parseAccess(*accessText);
    if (!access)
      return access.takeError();
    record.slots.push_back({*ordinal, ResourceId(*resource), *access});
  }
  return record;
}

} // namespace

llvm::Expected<PackageManifest>
detail::parseManifest(llvm::StringRef json, const PackageParseLimits &limits) {
  if (json.size() > limits.maxJSONBytes)
    return invalid("package manifest exceeds JSON byte limit");
  if (exceedsJSONNesting(json, limits.maxJSONNesting))
    return invalid("package manifest exceeds JSON nesting limit");
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(json);
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("package manifest must be a JSON object");
  if (llvm::Error error = requireExactFields(
          *root,
          {"schema_version", "program", "target", "rank_count", "resources",
           "modules", "entries", "completions"},
          "manifest"))
    return std::move(error);

  llvm::Expected<uint64_t> schemaVersion =
      requireUnsigned(*root, "schema_version", "manifest");
  if (!schemaVersion)
    return schemaVersion.takeError();
  if (*schemaVersion != kPackageManifestSchemaVersion)
    return invalid("unsupported package manifest schema_version");
  llvm::Expected<const llvm::json::Object *> program =
      requireObject(*root, "program", "manifest");
  if (!program)
    return program.takeError();
  llvm::Expected<const llvm::json::Object *> target =
      requireObject(*root, "target", "manifest");
  if (!target)
    return target.takeError();
  llvm::Expected<int64_t> rankCount =
      requireInteger(*root, "rank_count", "manifest");
  if (!rankCount)
    return rankCount.takeError();
  llvm::Expected<const llvm::json::Array *> resources =
      requireArray(*root, "resources", "manifest");
  if (!resources)
    return resources.takeError();
  llvm::Expected<const llvm::json::Array *> modules =
      requireArray(*root, "modules", "manifest");
  if (!modules)
    return modules.takeError();
  llvm::Expected<const llvm::json::Array *> entries =
      requireArray(*root, "entries", "manifest");
  if (!entries)
    return entries.takeError();
  llvm::Expected<const llvm::json::Array *> completions =
      requireArray(*root, "completions", "manifest");
  if (!completions)
    return completions.takeError();

  uint64_t totalRecords = (*resources)->size() + (*modules)->size() +
                          (*entries)->size() + (*completions)->size();
  if (totalRecords > limits.maxRecords)
    return invalid("package manifest exceeds record limit");
  if (llvm::Error error =
          requireExactFields(**program, {"id"}, "manifest.program"))
    return std::move(error);
  if (llvm::Error error = requireExactFields(
          **target,
          {"profile", "identity", "runtime_abi", "launch", "module_format"},
          "manifest.target"))
    return std::move(error);
  llvm::Expected<uint64_t> programId =
      requireUnsigned(**program, "id", "manifest.program");
  if (!programId)
    return programId.takeError();
  llvm::Expected<std::string> targetProfile =
      requireString(**target, "profile", "manifest.target", limits);
  if (!targetProfile)
    return targetProfile.takeError();
  llvm::Expected<std::string> targetIdentity =
      requireString(**target, "identity", "manifest.target", limits);
  if (!targetIdentity)
    return targetIdentity.takeError();
  llvm::Expected<std::string> runtimeABI =
      requireString(**target, "runtime_abi", "manifest.target", limits);
  if (!runtimeABI)
    return runtimeABI.takeError();
  llvm::Expected<const llvm::json::Object *> launch =
      requireObject(**target, "launch", "manifest.target");
  if (!launch)
    return launch.takeError();
  llvm::Expected<std::string> moduleFormat =
      requireString(**target, "module_format", "manifest.target", limits);
  if (!moduleFormat)
    return moduleFormat.takeError();

  llvm::Expected<TargetProfileId> parsedTargetProfile =
      parseTargetProfileId(*targetProfile);
  if (!parsedTargetProfile)
    return parsedTargetProfile.takeError();
  llvm::Expected<TargetIdentityId> parsedTargetIdentity =
      parseTargetIdentityId(*targetIdentity);
  if (!parsedTargetIdentity)
    return parsedTargetIdentity.takeError();
  llvm::Expected<KernelRuntimeABIId> parsedRuntimeABI =
      parseKernelRuntimeABIId(*runtimeABI);
  if (!parsedRuntimeABI)
    return parsedRuntimeABI.takeError();
  llvm::Expected<RuntimeLaunchContract> parsedLaunch =
      parseRuntimeLaunchContract(**launch, "manifest.target.launch", limits);
  if (!parsedLaunch)
    return parsedLaunch.takeError();
  PackageManifest manifest(*parsedTargetProfile, *parsedTargetIdentity,
                           *parsedRuntimeABI, std::move(*parsedLaunch),
                           *moduleFormat);

  manifest.schemaVersion = static_cast<uint32_t>(*schemaVersion);
  manifest.program = ProgramId(*programId);
  manifest.rankCount = *rankCount;
  for (auto [index, value] : llvm::enumerate(**resources)) {
    llvm::Expected<PackageResourceRecord> record =
        parseResource(value, index, limits);
    if (!record)
      return record.takeError();
    manifest.resources.push_back(std::move(*record));
  }
  for (auto [index, value] : llvm::enumerate(**modules)) {
    llvm::Expected<PackageModuleRecord> record =
        parseModuleRecord(value, index, limits);
    if (!record)
      return record.takeError();
    manifest.modules.push_back(std::move(*record));
  }
  for (auto [index, value] : llvm::enumerate(**entries)) {
    llvm::Expected<PackageEntrypointRecord> record =
        parseEntrypointRecord(value, index, limits);
    if (!record)
      return record.takeError();
    manifest.entries.push_back(std::move(*record));
  }
  for (auto [index, value] : llvm::enumerate(**completions)) {
    llvm::Expected<PackageCompletionRecord> record =
        parseCompletionRecord(value, index, limits);
    if (!record)
      return record.takeError();
    manifest.completions.push_back(std::move(*record));
  }
  return manifest;
}

} // namespace wafer::runtime
