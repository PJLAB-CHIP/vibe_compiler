//===- PackageManifestVerification.cpp - Manifest semantic verification --===//

#include "PackageManifestInternal.h"
#include "Wafer/Runtime/Tx81ModelABI.h"

#include "llvm/ADT/DenseMap.h"
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

llvm::Error verifyLaunchABIContract(const PackageManifest &manifest) {
  const bool perRank =
      manifest.launchABI == TargetLaunchABIId::perRankPointerBlockV1();
  const bool kernelGrid = manifest.launchABI ==
                          TargetLaunchABIId::tx81KernelGridPointerTableV1();
  const bool model =
      manifest.launchABI == TargetLaunchABIId::tx81ModelBootParamV1();
  const bool cluster =
      manifest.launchABI ==
      TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1();
  if (!perRank && !kernelGrid && !model && !cluster)
    return invalid("package target launch ABI is not registered");
  if (!perRank &&
      (manifest.rankCount != 16 || manifest.entries.size() != 16))
    return invalid("multi-tile launch ABI requires a complete 16-rank domain");

  std::vector<const PackageEntrypointRecord *> entriesByRank(
      manifest.rankCount, nullptr);
  for (const PackageEntrypointRecord &entry : manifest.entries)
    entriesByRank[entry.logicalRank] = &entry;
  const PackageEntrypointRecord &first = *entriesByRank.front();
  const PackageModuleRecord *firstModule =
      findModule(manifest.modules, first.module);
  if (!firstModule)
    return invalid("launch ABI entry references a missing module");

  llvm::DenseMap<uint64_t, uint64_t> moduleReferenceCount;
  for (const PackageEntrypointRecord &entry : manifest.entries)
    ++moduleReferenceCount[entry.module.getValue()];
  if (perRank || model) {
    if (manifest.modules.size() !=
        static_cast<size_t>(manifest.rankCount))
      return invalid(
          "per-rank/model launch requires one module per logical rank");
    if (llvm::any_of(manifest.modules, [&](const auto &module) {
          return moduleReferenceCount.lookup(module.id.getValue()) != 1;
        }))
      return invalid(
          "per-rank/model entry-to-module mapping is not one-to-one");
  } else {
    if (manifest.modules.size() != 1 ||
        llvm::any_of(manifest.entries, [&](const auto &entry) {
          return entry.module != first.module;
        }))
      return invalid(
          "shared launch ABI requires one module referenced by all ranks");
  }

  auto hasOnlyMain = [](const PackageModuleRecord &module) {
    return module.exports.size() == 1 &&
           module.exports.front().role == PackageModuleExportRole::Main;
  };
  if (cluster) {
    const PackageModuleExportRecord *prepare = detail::findModuleExport(
        *firstModule, PackageModuleExportRole::Prepare);
    const PackageModuleExportRecord *main = detail::findModuleExport(
        *firstModule, PackageModuleExportRole::Main);
    if (firstModule->exports.size() != 2 || !prepare || !main ||
        prepare->symbol == main->symbol)
      return invalid(
          "cluster Direct DTE module requires distinct prepare/main exports");
  } else if (llvm::any_of(manifest.modules, [&](const auto &module) {
               return !hasOnlyMain(module);
             })) {
    return invalid("launch ABI module must expose exactly one typed main");
  }

  if (perRank) {
    for (const PackageEntrypointRecord &entry : manifest.entries) {
      if (entry.slots.size() >
          kTx81KernelArgumentBytesMax / sizeof(uint64_t))
        return invalid(
            "per-rank kernel argument block exceeds the qualified V5.6 "
            "packet limit");
      if (!std::holds_alternative<NoTransportRequirements>(entry.transport))
        return invalid(
            "per-rank launch ABI does not support transport requirements");
    }
    return llvm::Error::success();
  }

  if (kernelGrid || cluster) {
    if (first.slots.empty())
      return invalid("shared launch requires at least one typed ABI slot");
    const uint64_t argumentBytesMax =
        cluster ? kTx81ClusterKernelArgumentBytesMax
                : kTx81KernelArgumentBytesMax;
    if (first.slots.size() >
        argumentBytesMax / sizeof(uint64_t) / 16)
      return invalid(
          "shared rank-major argument table exceeds the qualified V5.6 "
          "packet limit");
  }
  const PackageModuleExportRecord *firstMain = detail::findModuleExport(
      *firstModule, PackageModuleExportRole::Main);
  if (model && (!firstMain || firstMain->symbol.size() >= 128))
    return invalid(
        "model launch entry symbol must fit the 128-byte loader field");

  std::vector<Tx81ModelTensorDescriptor> modelTensors;

  for (int64_t rank = 0; rank < 16; ++rank) {
    const PackageEntrypointRecord &entry = *entriesByRank[rank];
    const PackageModuleRecord *module =
        findModule(manifest.modules, entry.module);
    const PackageModuleExportRecord *main =
        module ? detail::findModuleExport(*module,
                                          PackageModuleExportRole::Main)
               : nullptr;
    if (!module || !main || entry.slots.size() != first.slots.size())
      return invalid("multi-tile launch entries have different slot counts");
    if (model && (!firstMain || main->symbol != firstMain->symbol))
      return invalid("model launch modules have different main symbols");
    if (cluster) {
      const auto *requirements =
          std::get_if<DirectDTETransportRequirements>(&entry.transport);
      const auto *firstRequirements =
          std::get_if<DirectDTETransportRequirements>(&first.transport);
      if (!requirements || !firstRequirements ||
          requirements->statusABI != firstRequirements->statusABI ||
          requirements->hostWatchdogRequired !=
              firstRequirements->hostWatchdogRequired)
        return invalid(
            "cluster Direct DTE ranks have inconsistent transport contracts");
    } else if (!std::holds_alternative<NoTransportRequirements>(
                   entry.transport)) {
      return invalid("multi-tile launch ABI does not support transport slots");
    }
    for (size_t slotIndex = 0; slotIndex < entry.slots.size(); ++slotIndex) {
      const PackageResourceRecord *resource =
          findResource(manifest.resources, entry.slots[slotIndex].resource);
      const PackageResourceRecord *reference = findResource(
          manifest.resources, first.slots[slotIndex].resource);
      if (!resource || !reference || resource->role != reference->role ||
          resource->roleIndex != reference->roleIndex ||
          resource->type.dtype != reference->type.dtype ||
          resource->type.shape != reference->type.shape ||
          resource->bytes != reference->bytes ||
          resource->alignment != reference->alignment ||
          resource->access != reference->access)
        return invalid("multi-tile launch rank slot schemas are inconsistent");
      if (model &&
          (resource->role != PackageResourceRole::UserInput &&
           resource->role != PackageResourceRole::Output))
        return invalid(
            "model launch ABI accepts only user-input and output resources");
      if (model &&
          (resource->type.dtype != "f32" || resource->type.shape.empty() ||
           resource->type.shape.size() > 6 ||
           resource->alignment < alignof(float)))
        return invalid(
            "model launch ABI accepts only aligned rank-1..6 f32 tensors");
      if (model) {
        Tx81ModelTensorClass tensorClass =
            resource->role == PackageResourceRole::UserInput
                ? Tx81ModelTensorClass::Input
                : Tx81ModelTensorClass::Output;
        modelTensors.push_back({tensorClass, rank, slotIndex,
                                /*deviceAddress=*/8, resource->bytes,
                                resource->type.dtype, resource->type.shape});
      }
    }
  }
  if (model) {
    llvm::Expected<Tx81ModelBootParamImage> bootParam =
        buildTx81ModelBootParam(modelTensors,
                                /*dynamicTLVDeviceAddress=*/8);
    if (!bootParam)
      return invalid("model launch BootParam contract is invalid: " +
                     llvm::toString(bootParam.takeError()));
  }
  return llvm::Error::success();
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
  if (!isTargetLaunchABICompatible(manifest.launchABI,
                                   manifest.targetProfile))
    return invalid(
        "package launch ABI is not qualified for its target profile");
  if (manifest.rankCount != 1 && manifest.rankCount != 16)
    return invalid("package rank_count must be exactly 1 or 16");
  if (manifest.launchABI != TargetLaunchABIId::perRankPointerBlockV1() &&
      manifest.rankCount != 16)
    return invalid("multi-tile package launch ABI requires rank_count=16");
  uint64_t totalRecords = manifest.resources.size() + manifest.modules.size() +
                          manifest.entries.size() + manifest.completions.size();
  if (totalRecords > limits.maxRecords)
    return invalid("package manifest exceeds record limit");
  for (const PackageEntrypointRecord &entry : manifest.entries) {
    if (entry.slots.size() > limits.maxRecords - totalRecords)
      return invalid("package manifest exceeds record limit");
    totalRecords += entry.slots.size();
  }
  for (const PackageModuleRecord &module : manifest.modules) {
    if (module.exports.size() > limits.maxRecords - totalRecords)
      return invalid("package manifest exceeds record limit");
    totalRecords += module.exports.size();
  }
  if (manifest.modules.empty() ||
      manifest.entries.size() != static_cast<uint64_t>(manifest.rankCount) ||
      manifest.completions.size() != static_cast<uint64_t>(manifest.rankCount))
    return invalid("package module/entry/completion domain is incomplete");
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

  llvm::StringSet<> modulePaths;
  for (const PackageModuleRecord &module : manifest.modules) {
    if (!isValidRelativePath(module.relativePath) ||
        module.relativePath.size() > limits.maxStringBytes ||
        !llvm::StringRef(module.relativePath).starts_with("modules/") ||
        !modulePaths.insert(module.relativePath).second ||
        module.digest.size() > limits.maxStringBytes ||
        !isValidDigest(module.digest) ||
        module.format.size() > limits.maxStringBytes ||
        module.format != manifest.moduleFormat)
      return invalid("package module path/digest/format is invalid");
    if (module.exports.empty())
      return invalid("package module export set is empty");
    bool seenPrepare = false;
    bool seenMain = false;
    llvm::StringSet<> symbols;
    for (const PackageModuleExportRecord &moduleExport : module.exports) {
      bool *seen = nullptr;
      switch (moduleExport.role) {
      case PackageModuleExportRole::Prepare:
        seen = &seenPrepare;
        break;
      case PackageModuleExportRole::Main:
        seen = &seenMain;
        break;
      }
      if (*seen || moduleExport.symbol.empty() ||
          moduleExport.symbol.size() > limits.maxStringBytes ||
          llvm::StringRef(moduleExport.symbol).contains('\0') ||
          !symbols.insert(moduleExport.symbol).second)
        return invalid("package module export role/symbol is invalid");
      *seen = true;
    }
  }

  std::vector<bool> seenEntryRank(manifest.rankCount, false);
  std::vector<bool> referencedResources(manifest.resources.size(), false);
  llvm::DenseMap<uint64_t, uint64_t> referencedModules;
  for (PackageEntrypointRecord &entry : manifest.entries) {
    if (entry.logicalRank < 0 || entry.logicalRank >= manifest.rankCount ||
        seenEntryRank[entry.logicalRank])
      return invalid("package entry rank domain is invalid");
    seenEntryRank[entry.logicalRank] = true;
    const PackageModuleRecord *module =
        findModule(manifest.modules, entry.module);
    const PackageCompletionRecord *completion =
        findCompletion(manifest.completions, entry.terminalCompletion);
    if (!module || !completion ||
        completion->logicalRank != entry.logicalRank)
      return invalid("package entry module/completion relation is invalid");
    ++referencedModules[entry.module.getValue()];
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
          status->type.dtype != "u32" ||
          status->type.shape != std::vector<int64_t>{1} ||
          status->bytes != kDirectDTEStatusStorageBytes ||
          status->alignment != kDirectDTEStatusStorageAlignment ||
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
  if (llvm::any_of(manifest.modules, [&](const auto &module) {
        return referencedModules.lookup(module.id.getValue()) == 0;
      }))
    return invalid("package contains an unreferenced module");
  if (llvm::Error error = verifyLaunchABIContract(manifest))
    return std::move(error);

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
