//===- PackageManifestVerification.cpp - Manifest semantic verification --===//

#include "PackageManifestInternal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace wafer::runtime {
namespace {

using detail::findCompletion;
using detail::findModule;
using detail::findResource;
using detail::invalid;

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
  case PackageResourceRole::TransportStatus:
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
  case PackageResourceRole::TransportStatus:
    return PackageAccessMode::ReadWrite;
  }
  llvm_unreachable("unknown package resource role");
}

bool expectedHostVisible(PackageResourceRole role) {
  return role != PackageResourceRole::Workspace &&
         role != PackageResourceRole::TransportStatus;
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

} // namespace

llvm::Expected<VerifiedPackageManifest>
verifyPackageManifest(PackageManifest manifest, llvm::StringRef packageRoot,
                      const PackageParseLimits &limits) {
  if (manifest.schemaVersion != kPackageManifestSchemaVersion)
    return invalid("unsupported package manifest schema_version");
  if (!manifest.program.isValid() || manifest.program.getValue() != 0)
    return invalid("package program identity is invalid");
  const TargetProfileRecord &targetProfile =
      getTargetProfileRecord(manifest.targetProfile);
  if (manifest.targetIdentity != targetProfile.targetIdentity ||
      manifest.runtimeABI != targetProfile.kernelRuntimeABI ||
      manifest.moduleFormat != targetProfile.moduleFormat)
    return invalid("package target profile mapping is inconsistent");
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

    llvm::SmallVector<const PackageResourceRecord *, 1> statusResources;
    for (const PackageResourceRecord &resource : manifest.resources)
      if (resource.logicalRank == entry.logicalRank &&
          resource.role == PackageResourceRole::TransportStatus)
        statusResources.push_back(&resource);
    if (std::holds_alternative<NoTransportRequirements>(entry.transport)) {
      if (!statusResources.empty())
        return invalid(
            "package transport status exists without Direct DTE requirement");
    } else {
      const auto &requirements =
          std::get<DirectDTETransportRequirements>(entry.transport);
      const PackageResourceRecord *status =
          findResource(manifest.resources, requirements.statusResource);
      if (statusResources.size() != 1 || !status ||
          status != statusResources.front() ||
          status->logicalRank != entry.logicalRank || status->roleIndex != 0 ||
          status->name != "direct_dte_status" || status->type.dtype != "u32" ||
          status->type.shape != std::vector<int64_t>{1} || status->bytes != 4 ||
          status->alignment != 4 ||
          status->access != PackageAccessMode::ReadWrite ||
          status->hostVisible ||
          requirements.statusABI != kDirectDTEStatusABI ||
          !requirements.hostWatchdogRequired)
        return invalid("package Direct DTE transport requirement is invalid");
    }
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

} // namespace wafer::runtime
