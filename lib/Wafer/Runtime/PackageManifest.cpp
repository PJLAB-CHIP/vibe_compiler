//===- PackageManifest.cpp - Typed Wafer package format ------------------===//

#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::runtime {
namespace {

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

bool isRegularFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

bool isPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool isValidRole(PackageResourceRole role) {
  switch (role) {
  case PackageResourceRole::UserInput:
  case PackageResourceRole::Parameter:
  case PackageResourceRole::Constant:
  case PackageResourceRole::Output:
  case PackageResourceRole::Workspace:
    return true;
  }
  return false;
}

bool isValidAccess(PackageAccessMode access) {
  switch (access) {
  case PackageAccessMode::ReadOnly:
  case PackageAccessMode::WriteOnly:
  case PackageAccessMode::ReadWrite:
    return true;
  }
  return false;
}

bool isValidDigest(llvm::StringRef digest) {
  if (!digest.consume_front("sha256:") || digest.size() != 64)
    return false;
  return llvm::all_of(digest, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

bool isValidRelativePath(llvm::StringRef path) {
  if (path.empty() || llvm::sys::path::is_absolute(path) || path.contains('\\'))
    return false;
  llvm::SmallVector<llvm::StringRef, 8> components;
  path.split(components, '/', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  return llvm::all_of(components, [](llvm::StringRef component) {
    return !component.empty() && component != "." && component != "..";
  });
}

llvm::Expected<std::string> digestFile(llvm::StringRef path) {
  if (!isRegularFile(path))
    return invalid("package module is not a regular file: " + path);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read package module: " + path);
  llvm::SHA256 hasher;
  hasher.update((*buffer)->getBuffer());
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

template <typename Range, typename Projection>
bool hasDenseIds(const Range &records, Projection projection) {
  std::vector<bool> seen(records.size(), false);
  for (const auto &record : records) {
    uint64_t id = projection(record).getValue();
    if (id >= seen.size() || seen[id])
      return false;
    seen[id] = true;
  }
  return llvm::all_of(seen, [](bool value) { return value; });
}

PackageAccessMode expectedAccess(PackageResourceRole role) {
  switch (role) {
  case PackageResourceRole::UserInput:
  case PackageResourceRole::Parameter:
  case PackageResourceRole::Constant:
    return PackageAccessMode::ReadOnly;
  case PackageResourceRole::Output:
    return PackageAccessMode::WriteOnly;
  case PackageResourceRole::Workspace:
    return PackageAccessMode::ReadWrite;
  }
  llvm_unreachable("unknown package resource role");
}

bool expectedHostVisible(PackageResourceRole role) {
  return role != PackageResourceRole::Workspace;
}

const PackageResourceRecord *
findResource(llvm::ArrayRef<PackageResourceRecord> resources, ResourceId id) {
  auto iterator = llvm::find_if(
      resources, [&](const auto &resource) { return resource.id == id; });
  return iterator == resources.end() ? nullptr : &*iterator;
}

const PackageModuleRecord *
findModule(llvm::ArrayRef<PackageModuleRecord> modules, ModuleId id) {
  auto iterator = llvm::find_if(
      modules, [&](const auto &module) { return module.id == id; });
  return iterator == modules.end() ? nullptr : &*iterator;
}

const PackageCompletionRecord *
findCompletion(llvm::ArrayRef<PackageCompletionRecord> completions,
               CompletionId id) {
  auto iterator = llvm::find_if(
      completions, [&](const auto &completion) { return completion.id == id; });
  return iterator == completions.end() ? nullptr : &*iterator;
}

llvm::Error verifyModuleFiles(const PackageManifest &manifest,
                              llvm::StringRef packageRoot) {
  llvm::StringSet<> expectedPaths;
  for (const PackageModuleRecord &module : manifest.modules) {
    llvm::SmallString<256> path(packageRoot);
    llvm::sys::path::append(path, module.relativePath);
    llvm::Expected<std::string> digest = digestFile(path);
    if (!digest)
      return digest.takeError();
    if (*digest != module.digest)
      return invalid("package module digest mismatch: " + module.relativePath);
    expectedPaths.insert(path);
  }

  llvm::SmallString<256> modulesRoot(packageRoot);
  llvm::sys::path::append(modulesRoot, "modules");
  if (llvm::sys::fs::get_file_type(modulesRoot, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("package modules directory is missing or not a directory");

  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(modulesRoot, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return invalid("failed to walk package modules: " + error.message());
    llvm::sys::fs::file_type type = iterator->type();
    if (type == llvm::sys::fs::file_type::directory_file)
      continue;
    if (type != llvm::sys::fs::file_type::regular_file)
      return invalid("package modules contain a non-regular member");
    if (!expectedPaths.contains(iterator->path()))
      return invalid("package modules contain an unreferenced payload: " +
                     iterator->path());
  }
  if (error)
    return invalid("failed to walk package modules: " + error.message());
  return llvm::Error::success();
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
          *object, {"id", "rank", "path", "digest", "format"}, context))
    return std::move(error);
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  llvm::Expected<int64_t> rank = requireInteger(*object, "rank", context);
  if (!rank)
    return rank.takeError();
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
  return PackageModuleRecord{ModuleId(*id), *rank, std::move(*path),
                             std::move(*digest), std::move(*format)};
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

llvm::Expected<PackageEntrypointRecord>
parseEntrypointRecord(const llvm::json::Value &value, uint64_t index,
                      const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "entries[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = requireExactFields(
          *object,
          {"id", "rank", "module", "symbol", "slots", "terminal_completion"},
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
  llvm::Expected<std::string> symbol =
      requireString(*object, "symbol", context, limits);
  if (!symbol)
    return symbol.takeError();
  llvm::Expected<const llvm::json::Array *> slots =
      requireArray(*object, "slots", context);
  if (!slots)
    return slots.takeError();
  llvm::Expected<uint64_t> completion =
      requireUnsigned(*object, "terminal_completion", context);
  if (!completion)
    return completion.takeError();
  if ((*slots)->size() > limits.maxRecords)
    return invalid(context + ".slots exceeds record limit");

  PackageEntrypointRecord record;
  record.id = EntryId(*id);
  record.logicalRank = *rank;
  record.module = ModuleId(*module);
  record.symbol = std::move(*symbol);
  record.terminalCompletion = CompletionId(*completion);
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

llvm::Expected<PackageManifest>
parseManifest(llvm::StringRef json, const PackageParseLimits &limits) {
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

  PackageManifest manifest;
  llvm::Expected<uint64_t> schemaVersion =
      requireUnsigned(*root, "schema_version", "manifest");
  if (!schemaVersion)
    return schemaVersion.takeError();
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
          **target, {"identity", "runtime_abi", "module_format"},
          "manifest.target"))
    return std::move(error);
  llvm::Expected<uint64_t> programId =
      requireUnsigned(**program, "id", "manifest.program");
  if (!programId)
    return programId.takeError();
  llvm::Expected<std::string> targetIdentity =
      requireString(**target, "identity", "manifest.target", limits);
  if (!targetIdentity)
    return targetIdentity.takeError();
  llvm::Expected<std::string> runtimeABI =
      requireString(**target, "runtime_abi", "manifest.target", limits);
  if (!runtimeABI)
    return runtimeABI.takeError();
  llvm::Expected<std::string> moduleFormat =
      requireString(**target, "module_format", "manifest.target", limits);
  if (!moduleFormat)
    return moduleFormat.takeError();

  if (*schemaVersion > std::numeric_limits<uint32_t>::max())
    return invalid("manifest.schema_version exceeds uint32");
  manifest.schemaVersion = static_cast<uint32_t>(*schemaVersion);
  manifest.program = ProgramId(*programId);
  manifest.targetIdentity = std::move(*targetIdentity);
  manifest.runtimeABI = std::move(*runtimeABI);
  manifest.moduleFormat = std::move(*moduleFormat);
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

} // namespace

llvm::StringRef stringifyPackageResourceRole(PackageResourceRole role) {
  switch (role) {
  case PackageResourceRole::UserInput:
    return "user_input";
  case PackageResourceRole::Parameter:
    return "parameter";
  case PackageResourceRole::Constant:
    return "constant";
  case PackageResourceRole::Output:
    return "output";
  case PackageResourceRole::Workspace:
    return "workspace";
  }
  llvm_unreachable("unknown package resource role");
}

llvm::StringRef stringifyPackageAccessMode(PackageAccessMode access) {
  switch (access) {
  case PackageAccessMode::ReadOnly:
    return "read_only";
  case PackageAccessMode::WriteOnly:
    return "write_only";
  case PackageAccessMode::ReadWrite:
    return "read_write";
  }
  llvm_unreachable("unknown package access mode");
}

llvm::Expected<VerifiedPackageManifest>
verifyPackageManifest(PackageManifest manifest, llvm::StringRef packageRoot,
                      const PackageParseLimits &limits) {
  if (manifest.schemaVersion != kPackageManifestSchemaVersion)
    return invalid("unsupported package manifest schema_version");
  if (!manifest.program.isValid() || manifest.program.getValue() != 0)
    return invalid("package program identity is invalid");
  if (manifest.targetIdentity != kSingleCardTargetIdentity ||
      manifest.runtimeABI != kKernelRuntimeABI ||
      manifest.moduleFormat != kRiscv64ELFModuleFormat)
    return invalid("package target/runtime ABI/module format is unsupported");
  if (manifest.rankCount != 1 && manifest.rankCount != 16)
    return invalid("package rank_count must be exactly 1 or 16");
  uint64_t totalRecords = manifest.resources.size() + manifest.modules.size() +
                          manifest.entries.size() + manifest.completions.size();
  if (totalRecords > limits.maxRecords)
    return invalid("package manifest exceeds record limit");
  for (const PackageEntrypointRecord &entry : manifest.entries) {
    if (entry.slots.size() > limits.maxRecords - totalRecords)
      return invalid("package manifest exceeds record limit");
    totalRecords += entry.slots.size();
  }
  if (manifest.modules.size() != static_cast<uint64_t>(manifest.rankCount) ||
      manifest.entries.size() != static_cast<uint64_t>(manifest.rankCount) ||
      manifest.completions.size() != static_cast<uint64_t>(manifest.rankCount))
    return invalid("package rank/module/entry/completion domain is incomplete");
  if (!hasDenseIds(manifest.resources,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.modules,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.entries,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.completions,
                   [](const auto &record) { return record.id; }))
    return invalid("package IDs must be unique dense zero-based domains");

  llvm::sort(manifest.resources,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.modules,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.entries,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.completions,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });

  llvm::DenseSet<std::pair<int64_t, int64_t>> roleIndices;
  for (const PackageResourceRecord &resource : manifest.resources) {
    if (!isValidRole(resource.role) || !isValidAccess(resource.access) ||
        resource.logicalRank < 0 ||
        resource.logicalRank >= manifest.rankCount || resource.roleIndex < 0 ||
        resource.roleIndex > std::numeric_limits<uint32_t>::max() ||
        resource.name.size() > limits.maxStringBytes ||
        resource.type.dtype.empty() ||
        resource.type.dtype.size() > limits.maxStringBytes ||
        resource.type.shape.size() > limits.maxShapeRank ||
        llvm::any_of(resource.type.shape,
                     [](int64_t dimension) { return dimension < 0; }) ||
        resource.bytes == 0 ||
        resource.bytes >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        resource.alignment >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        !isPowerOfTwo(resource.alignment))
      return invalid("package resource has invalid rank/type/size/alignment");
    if (resource.access != expectedAccess(resource.role) ||
        resource.hostVisible != expectedHostVisible(resource.role))
      return invalid("package resource role/access/visibility mismatch");
    int64_t roleKey = static_cast<int64_t>(resource.role) << 32 |
                      static_cast<uint32_t>(resource.roleIndex);
    if (!roleIndices.insert({resource.logicalRank, roleKey}).second)
      return invalid("package resource role/index is duplicated within rank");
  }

  std::vector<bool> seenModuleRank(manifest.rankCount, false);
  llvm::StringSet<> modulePaths;
  for (const PackageModuleRecord &module : manifest.modules) {
    if (module.logicalRank < 0 || module.logicalRank >= manifest.rankCount ||
        seenModuleRank[module.logicalRank])
      return invalid("package module rank domain is not all-and-only");
    seenModuleRank[module.logicalRank] = true;
    if (!isValidRelativePath(module.relativePath) ||
        module.relativePath.size() > limits.maxStringBytes ||
        !llvm::StringRef(module.relativePath).starts_with("modules/") ||
        !modulePaths.insert(module.relativePath).second ||
        module.digest.size() > limits.maxStringBytes ||
        !isValidDigest(module.digest) ||
        module.format.size() > limits.maxStringBytes ||
        module.format != manifest.moduleFormat)
      return invalid("package module path/digest/format is invalid");
  }

  std::vector<bool> seenEntryRank(manifest.rankCount, false);
  std::vector<bool> referencedResources(manifest.resources.size(), false);
  for (PackageEntrypointRecord &entry : manifest.entries) {
    if (entry.logicalRank < 0 || entry.logicalRank >= manifest.rankCount ||
        seenEntryRank[entry.logicalRank] || entry.symbol.empty() ||
        entry.symbol.size() > limits.maxStringBytes)
      return invalid("package entry rank/symbol domain is invalid");
    seenEntryRank[entry.logicalRank] = true;
    const PackageModuleRecord *module =
        findModule(manifest.modules, entry.module);
    const PackageCompletionRecord *completion =
        findCompletion(manifest.completions, entry.terminalCompletion);
    if (!module || module->logicalRank != entry.logicalRank || !completion ||
        completion->logicalRank != entry.logicalRank)
      return invalid("package entry module/completion relation is invalid");
    llvm::sort(entry.slots, [](const auto &lhs, const auto &rhs) {
      return lhs.ordinal < rhs.ordinal;
    });
    for (auto [ordinal, slot] : llvm::enumerate(entry.slots)) {
      if (!isValidAccess(slot.access) || slot.ordinal != ordinal)
        return invalid("package ABI slots must be dense and zero-based");
      const PackageResourceRecord *resource =
          findResource(manifest.resources, slot.resource);
      if (!resource || resource->logicalRank != entry.logicalRank ||
          slot.access != resource->access ||
          referencedResources[resource->id.getValue()])
        return invalid("package ABI slot/resource relation is invalid");
      referencedResources[resource->id.getValue()] = true;
    }
    uint64_t rankResourceCount =
        llvm::count_if(manifest.resources, [&](const auto &resource) {
          return resource.logicalRank == entry.logicalRank;
        });
    if (entry.slots.size() != rankResourceCount)
      return invalid("package entry omits or adds rank resources");
  }
  if (!llvm::all_of(referencedResources, [](bool value) { return value; }))
    return invalid(
        "package resources are not covered all-and-only by ABI slots");

  std::vector<bool> seenCompletionRank(manifest.rankCount, false);
  for (const PackageCompletionRecord &completion : manifest.completions) {
    if (completion.logicalRank < 0 ||
        completion.logicalRank >= manifest.rankCount ||
        seenCompletionRank[completion.logicalRank] ||
        completion.kind != "entry_return")
      return invalid("package terminal completion domain is invalid");
    seenCompletionRank[completion.logicalRank] = true;
  }

  if (packageRoot.empty())
    return invalid("package root must not be empty");
  if (llvm::Error error = verifyModuleFiles(manifest, packageRoot))
    return std::move(error);
  return VerifiedPackageManifest(std::move(manifest));
}

std::string
serializeCanonicalPackageJson(const VerifiedPackageManifest &verified) {
  const PackageManifest &manifest = verified.getManifest();
  std::string storage;
  llvm::raw_string_ostream output(storage);
  llvm::json::OStream json(output, /*IndentSize=*/2);
  json.object([&] {
    json.attribute("schema_version", int64_t(manifest.schemaVersion));
    json.attributeObject("program", [&] {
      json.attribute("id", int64_t(manifest.program.getValue()));
    });
    json.attributeObject("target", [&] {
      json.attribute("identity", manifest.targetIdentity);
      json.attribute("runtime_abi", manifest.runtimeABI);
      json.attribute("module_format", manifest.moduleFormat);
    });
    json.attribute("rank_count", manifest.rankCount);
    json.attributeArray("resources", [&] {
      for (const PackageResourceRecord &resource : manifest.resources)
        json.object([&] {
          json.attribute("id", int64_t(resource.id.getValue()));
          json.attribute("rank", resource.logicalRank);
          json.attribute("role", stringifyPackageResourceRole(resource.role));
          json.attribute("role_index", resource.roleIndex);
          json.attribute("name", resource.name);
          json.attributeObject("type", [&] {
            json.attribute("dtype", resource.type.dtype);
            json.attributeArray("shape", [&] {
              for (int64_t dimension : resource.type.shape)
                json.value(dimension);
            });
          });
          json.attribute("bytes", int64_t(resource.bytes));
          json.attribute("alignment", int64_t(resource.alignment));
          json.attribute("access", stringifyPackageAccessMode(resource.access));
          json.attribute("host_visible", resource.hostVisible);
        });
    });
    json.attributeArray("modules", [&] {
      for (const PackageModuleRecord &module : manifest.modules)
        json.object([&] {
          json.attribute("id", int64_t(module.id.getValue()));
          json.attribute("rank", module.logicalRank);
          json.attribute("path", module.relativePath);
          json.attribute("digest", module.digest);
          json.attribute("format", module.format);
        });
    });
    json.attributeArray("entries", [&] {
      for (const PackageEntrypointRecord &entry : manifest.entries)
        json.object([&] {
          json.attribute("id", int64_t(entry.id.getValue()));
          json.attribute("rank", entry.logicalRank);
          json.attribute("module", int64_t(entry.module.getValue()));
          json.attribute("symbol", entry.symbol);
          json.attributeArray("slots", [&] {
            for (const PackageABISlotBinding &slot : entry.slots)
              json.object([&] {
                json.attribute("ordinal", int64_t(slot.ordinal));
                json.attribute("resource", int64_t(slot.resource.getValue()));
                json.attribute("access",
                               stringifyPackageAccessMode(slot.access));
              });
          });
          json.attribute("terminal_completion",
                         int64_t(entry.terminalCompletion.getValue()));
        });
    });
    json.attributeArray("completions", [&] {
      for (const PackageCompletionRecord &completion : manifest.completions)
        json.object([&] {
          json.attribute("id", int64_t(completion.id.getValue()));
          json.attribute("rank", completion.logicalRank);
          json.attribute("kind", completion.kind);
        });
    });
  });
  output << '\n';
  output.flush();
  return storage;
}

llvm::Expected<VerifiedPackageManifest>
parseCanonicalPackageJson(llvm::StringRef json, llvm::StringRef packageRoot,
                          const PackageParseLimits &limits) {
  llvm::Expected<PackageManifest> manifest = parseManifest(json, limits);
  if (!manifest)
    return manifest.takeError();
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(*manifest), packageRoot, limits);
  if (!verified)
    return verified.takeError();
  if (serializeCanonicalPackageJson(*verified) != json)
    return invalid("package manifest JSON is not canonical");
  return std::move(*verified);
}

llvm::Expected<VerifiedPackageManifest>
loadVerifiedPackageManifest(llvm::StringRef packageRoot,
                            const PackageParseLimits &limits) {
  llvm::SmallString<256> path(packageRoot);
  llvm::sys::path::append(path, kPackageManifestFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read package manifest: " + path);
  return parseCanonicalPackageJson((*buffer)->getBuffer(), packageRoot, limits);
}

llvm::Expected<RuntimeSessionPlan> preflightNoCardRuntimeSession(
    const VerifiedPackageManifest &package, EntryId entryId,
    llvm::ArrayRef<RuntimeInvocationBinding> invocationBindings,
    const RuntimeEnvironment &environment) {
  const PackageManifest &manifest = package.getManifest();
  if (environment.targetIdentity != manifest.targetIdentity ||
      environment.runtimeABI != manifest.runtimeABI ||
      environment.moduleFormat != manifest.moduleFormat)
    return invalid("runtime environment is incompatible with package target");
  auto entryIterator = llvm::find_if(
      manifest.entries, [&](const auto &entry) { return entry.id == entryId; });
  if (entryIterator == manifest.entries.end())
    return invalid("runtime entry ID is not present in package");
  const PackageEntrypointRecord &entry = *entryIterator;
  const PackageModuleRecord *module =
      findModule(manifest.modules, entry.module);
  if (!module)
    return invalid("runtime entry references a missing module");

  std::vector<const RuntimeInvocationBinding *> bindingsByResource(
      manifest.resources.size(), nullptr);
  for (const RuntimeInvocationBinding &binding : invocationBindings) {
    if (!binding.resource.isValid() ||
        binding.resource.getValue() >= bindingsByResource.size() ||
        bindingsByResource[binding.resource.getValue()])
      return invalid(
          "runtime invocation contains duplicate or unknown resource");
    bindingsByResource[binding.resource.getValue()] = &binding;
  }

  RuntimeSessionPlan plan;
  plan.entry = entry.id;
  plan.logicalRank = entry.logicalRank;
  plan.module = module->id;
  plan.modulePath = module->relativePath;
  plan.entrySymbol = entry.symbol;
  plan.terminalCompletion = entry.terminalCompletion;
  for (const PackageABISlotBinding &slot : entry.slots) {
    const PackageResourceRecord *resource =
        findResource(manifest.resources, slot.resource);
    if (!resource)
      return invalid("runtime ABI slot references a missing resource");
    if (resource->bytes > environment.maxResourceBytes)
      return invalid("runtime environment resource capacity is insufficient");
    const RuntimeInvocationBinding *binding =
        bindingsByResource[resource->id.getValue()];
    if (resource->hostVisible) {
      if (!binding || binding->bytes < resource->bytes ||
          binding->alignment < resource->alignment ||
          binding->alignment % resource->alignment != 0 ||
          binding->access != resource->access || !binding->hostVisible)
        return invalid("runtime invocation binding does not satisfy resource");
    } else if (binding) {
      return invalid("runtime invocation must not bind internal workspace");
    }
    plan.resources.push_back({resource->id, resource->role, resource->bytes,
                              resource->alignment, resource->access,
                              binding != nullptr});
    plan.launchOrder.push_back(resource->id);
  }
  for (const RuntimeInvocationBinding &binding : invocationBindings) {
    const PackageResourceRecord *resource =
        findResource(manifest.resources, binding.resource);
    if (!resource || resource->logicalRank != entry.logicalRank ||
        !resource->hostVisible)
      return invalid("runtime invocation contains an extra binding");
  }
  plan.executesBoard = false;
  return plan;
}

} // namespace wafer::runtime
