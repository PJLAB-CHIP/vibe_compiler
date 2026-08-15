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
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace wafer::runtime {
namespace {

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

bool isProgramBoundaryResource(PackageResourceRole role) {
  return role == PackageResourceRole::UserInput ||
         role == PackageResourceRole::Parameter ||
         role == PackageResourceRole::Constant ||
         role == PackageResourceRole::Output;
}

bool isEntryLocalResource(PackageResourceRole role) {
  return role == PackageResourceRole::Workspace ||
         role == PackageResourceRole::TransportStatus;
}

bool isSameTile(const TileResourceScope &scope,
                        const PackageEntrypointRecord &entry) {
  return scope.cardId == entry.cardId && scope.tileId == entry.tileId;
}

llvm::Error verifyRuntimeLaunchContract(const PackageManifest &manifest) {
  const KernelRuntimeLaunchContract *kernel = manifest.launch.getKernel();
  const ModelRuntimeLaunchContract *model = manifest.launch.getModel();
  const bool kernelGrid = kernel && kernel->form == KernelLaunchForm::Grid;
  const bool cluster = kernel && kernel->form == KernelLaunchForm::Cluster;
  if (manifest.tileCount != 16 || manifest.entries.size() != 16)
    return invalid(
        "grid, cluster and model launches require a complete 16-Tile domain");

  std::vector<const PackageEntrypointRecord *> entriesByLaunchSlot(
      manifest.tileCount, nullptr);
  for (const PackageEntrypointRecord &entry : manifest.entries)
    entriesByLaunchSlot[entry.launchSlot.getValue()] = &entry;
  const PackageEntrypointRecord &first = *entriesByLaunchSlot.front();
  const PackageModuleRecord *firstModule =
      findModule(manifest.modules, first.module);
  if (!firstModule)
    return invalid("runtime launch entry references a missing module");

  llvm::DenseMap<uint64_t, uint64_t> moduleReferenceCount;
  for (const PackageEntrypointRecord &entry : manifest.entries)
    ++moduleReferenceCount[entry.module.getValue()];
  if (model) {
    if (manifest.modules.size() != static_cast<size_t>(manifest.tileCount))
      return invalid(
          "model launch requires one module per Tile");
    if (llvm::any_of(manifest.modules, [&](const auto &module) {
          return moduleReferenceCount.lookup(module.id.getValue()) != 1;
        }))
      return invalid(
          "model entry-to-module mapping is not one-to-one");
  } else {
    if (manifest.modules.size() != 1 ||
        llvm::any_of(manifest.entries, [&](const auto &entry) {
          return entry.module != first.module;
        }))
      return invalid(
          "grid/cluster launch requires one module referenced by all Tiles");
  }

  auto exportRoleForPhase = [](RuntimeLaunchPhaseRole phase) {
    return phase == RuntimeLaunchPhaseRole::Prepare
               ? PackageModuleExportRole::Prepare
               : PackageModuleExportRole::Main;
  };
  for (const PackageModuleRecord &module : manifest.modules) {
    if (module.exports.size() != manifest.launch.getPhases().size())
      return invalid(
          "package module exports do not match the runtime launch phases");
    for (auto [index, phase] : llvm::enumerate(manifest.launch.getPhases()))
      if (module.exports[index].role != exportRoleForPhase(phase))
        return invalid(
            "package module export order does not match the runtime launch "
            "phases");
  }

  if (kernelGrid || cluster) {
    if (first.slots.empty())
      return invalid("shared launch requires at least one typed ABI slot");
    const uint64_t argumentBytesMax = cluster
                                          ? kTx81ClusterKernelArgumentBytesMax
                                          : kTx81KernelArgumentBytesMax;
    if (kernel->entryABI == KernelEntryABI::TileMajorPointerTable) {
      if (first.slots.size() > argumentBytesMax / sizeof(uint64_t) / 16)
        return invalid(
            "shared Tile-major argument table exceeds the qualified V5.6 "
            "packet limit");
    } else if (kernel->entryABI == KernelEntryABI::TileRowPointerTable) {
      if (16 > argumentBytesMax / sizeof(uint64_t))
        return invalid(
            "shared Tile-row pointer table exceeds the qualified V5.6 "
            "packet limit");
    } else {
      return invalid("shared kernel launch has an incompatible entry ABI");
    }
  }
  const PackageModuleExportRecord *firstMain =
      detail::findModuleExport(*firstModule, PackageModuleExportRole::Main);
  if (model && (!firstMain || firstMain->symbol.size() >= 128))
    return invalid(
        "model launch entry symbol must fit the 128-byte loader field");

  std::vector<Tx81ModelTensorDescriptor> modelTensors;
  const size_t firstTransportKind = first.transport.index();
  for (int64_t launchSlot = 0; launchSlot < manifest.tileCount; ++launchSlot) {
    const PackageEntrypointRecord &entry =
        *entriesByLaunchSlot[launchSlot];
    const PackageModuleRecord *module =
        findModule(manifest.modules, entry.module);
    const PackageModuleExportRecord *main =
        module
            ? detail::findModuleExport(*module, PackageModuleExportRole::Main)
            : nullptr;
    if (!module || !main)
      return invalid("runtime launch entry has no typed main export");
    if (entry.transport.index() != firstTransportKind)
      return invalid("runtime launch Tiles have mixed transport contracts");
    if ((kernelGrid || cluster || model) &&
        entry.slots.size() != first.slots.size())
      return invalid("multi-tile launch entries have different slot counts");
    if (model && (!firstMain || main->symbol != firstMain->symbol))
      return invalid("model launch modules have different main symbols");
    if (const auto *requirements =
            std::get_if<DirectDTETransportRequirements>(&entry.transport)) {
      const auto *firstRequirements =
          std::get_if<DirectDTETransportRequirements>(&first.transport);
      if (!firstRequirements ||
          requirements->statusABI != firstRequirements->statusABI ||
          requirements->hostWatchdogRequired !=
              firstRequirements->hostWatchdogRequired)
        return invalid(
            "Direct DTE Tiles have inconsistent transport contracts");
    }
    for (size_t slotIndex = 0; slotIndex < entry.slots.size(); ++slotIndex) {
      const PackageResourceRecord *resource =
          findResource(manifest.resources, entry.slots[slotIndex].resource);
      const PackageResourceRecord *reference =
          findResource(manifest.resources, first.slots[slotIndex].resource);
      if (!resource || !reference)
        return invalid("runtime launch slot references a missing resource");
      if ((kernelGrid || cluster || model) &&
          (resource->role != reference->role ||
           resource->roleIndex != reference->roleIndex ||
           resource->type.dtype != reference->type.dtype ||
           resource->type.shape != reference->type.shape ||
           resource->bytes != reference->bytes ||
           resource->alignment != reference->alignment ||
           resource->access != reference->access))
        return invalid("multi-Tile launch slot schemas are inconsistent");
      if (model && (resource->role != PackageResourceRole::UserInput &&
                    resource->role != PackageResourceRole::Output))
        return invalid(
            "model launch contract accepts only user-input and output "
            "resources");
      if (model &&
          (resource->type.dtype != "f32" || resource->type.shape.empty() ||
           resource->type.shape.size() > 6 ||
           resource->alignment < alignof(float)))
        return invalid(
            "model launch contract accepts only aligned rank-1..6 f32 "
            "tensors");
      if (model) {
        Tx81ModelTensorClass tensorClass =
            resource->role == PackageResourceRole::UserInput
                ? Tx81ModelTensorClass::Input
                : Tx81ModelTensorClass::Output;
        modelTensors.push_back({tensorClass, entry.cardId, entry.tileId,
                                entry.launchSlot, slotIndex,
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
  if (manifest.targetIdentity != TargetIdentityId::waferTx81SingleCard() ||
      manifest.runtimeABI != KernelRuntimeABIId::waferTx81Kernel() ||
      manifest.moduleFormat != kCurrentTargetModuleFormat)
    return invalid("package target identity or runtime ABI is unsupported");
  if (manifest.cardCount != 1)
    return invalid("package card_count must be exactly 1");
  if (manifest.tileCount != 16)
    return invalid("package tile_count must be exactly 16");
  uint64_t totalRecords = manifest.resources.size() + manifest.modules.size() +
                          manifest.entries.size();
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
      manifest.entries.size() != static_cast<uint64_t>(manifest.tileCount))
    return invalid("package module/entry Tile domain is incomplete");
  if (!hasDenseIds(manifest.resources,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.modules,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.entries,
                   [](const auto &record) { return record.id; }))
    return invalid("package IDs must be unique dense zero-based domains");

  llvm::sort(manifest.resources,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.modules,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.entries,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });

  std::set<std::tuple<bool, int64_t, int64_t, int64_t>> roleIndices;
  for (const PackageResourceRecord &resource : manifest.resources) {
    if (!isValidRole(resource.role) || !isValidAccess(resource.access) ||
        resource.roleIndex < 0 ||
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
      return invalid("package resource has invalid scope/type/size/alignment");
    if (resource.access != expectedAccess(resource.role) ||
        resource.hostVisible != expectedHostVisible(resource.role))
      return invalid("package resource role/access/visibility mismatch");
    const auto *cardScope = std::get_if<CardResourceScope>(&resource.scope);
    const auto *tileScope = std::get_if<TileResourceScope>(&resource.scope);
    if ((cardScope && (!isProgramBoundaryResource(resource.role) ||
                       cardScope->cardId != CardId(0))) ||
        (tileScope &&
         (!isEntryLocalResource(resource.role) ||
          tileScope->cardId != CardId(0) ||
          tileScope->tileId.getValue() < 0 ||
          tileScope->tileId.getValue() >= 16)))
      return invalid("package resource role and typed physical scope disagree");
    int64_t roleKey = static_cast<int64_t>(resource.role) << 32 |
                      static_cast<uint32_t>(resource.roleIndex);
    const bool isTile = tileScope != nullptr;
    const int64_t cardId =
        isTile ? tileScope->cardId.getValue() : cardScope->cardId.getValue();
    const int64_t tileId = isTile ? tileScope->tileId.getValue() : -1;
    if (!roleIndices.insert({isTile, cardId, tileId, roleKey}).second)
      return invalid(
          "package resource role/index is duplicated within physical scope");
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

  std::vector<uint64_t> resourceReferenceCounts(manifest.resources.size(), 0);
  for (const PackageEntrypointRecord &entry : manifest.entries)
    for (const PackageABISlotBinding &slot : entry.slots) {
      if (!slot.resource.isValid() ||
          slot.resource.getValue() >= resourceReferenceCounts.size())
        return invalid("package ABI slot references a missing resource");
      ++resourceReferenceCounts[slot.resource.getValue()];
    }

  std::vector<bool> seenLaunchSlots(manifest.tileCount, false);
  llvm::DenseSet<int64_t> seenTileIds;
  llvm::DenseMap<uint64_t, uint64_t> referencedModules;
  for (PackageEntrypointRecord &entry : manifest.entries) {
    if (entry.cardId != CardId(0) || entry.tileId.getValue() < 0 ||
        entry.tileId.getValue() >= 16 || !entry.launchSlot.isValid() ||
        entry.launchSlot.getValue() >=
            static_cast<uint64_t>(manifest.tileCount) ||
        seenLaunchSlots[entry.launchSlot.getValue()] ||
        !seenTileIds.insert(entry.tileId.getValue()).second ||
        entry.completion !=
            PackageEntryCompletionKind::ReturnAfterLocalDrain)
      return invalid("package entry Tile domain is invalid");
    seenLaunchSlots[entry.launchSlot.getValue()] = true;
    const PackageModuleRecord *module =
        findModule(manifest.modules, entry.module);
    if (!module)
      return invalid("package entry module relation is invalid");
    ++referencedModules[entry.module.getValue()];
    llvm::sort(entry.slots, [](const auto &lhs, const auto &rhs) {
      return lhs.ordinal < rhs.ordinal;
    });
    llvm::DenseSet<uint64_t> entryResources;
    for (auto [ordinal, slot] : llvm::enumerate(entry.slots)) {
      if (!isValidAccess(slot.access) || slot.ordinal != ordinal)
        return invalid("package ABI slots must be dense and zero-based");
      const PackageResourceRecord *resource =
          findResource(manifest.resources, slot.resource);
      if (!resource || slot.access != resource->access ||
          !entryResources.insert(resource->id.getValue()).second)
        return invalid("package ABI slot/resource relation is invalid");
      const auto *cardScope = std::get_if<CardResourceScope>(&resource->scope);
      const auto *tileScope = std::get_if<TileResourceScope>(&resource->scope);
      const uint64_t referenceCount =
          resourceReferenceCounts[resource->id.getValue()];
      if ((cardScope &&
           (cardScope->cardId != entry.cardId ||
            referenceCount != static_cast<uint64_t>(manifest.tileCount))) ||
          (tileScope &&
           (!isSameTile(*tileScope, entry) || referenceCount != 1)))
        return invalid(
            "package resource references disagree with typed physical scope");
    }

    llvm::SmallVector<const PackageResourceRecord *, 1> statusResources;
    for (const PackageResourceRecord &resource : manifest.resources)
      if (const auto *scope = std::get_if<TileResourceScope>(&resource.scope);
          scope && isSameTile(*scope, entry) &&
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
          status->roleIndex != 0 || status->type.dtype != "u32" ||
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
  if (llvm::any_of(resourceReferenceCounts,
                   [](uint64_t count) { return count == 0; }))
    return invalid(
        "package resources are not covered all-and-only by ABI slots");
  if (llvm::any_of(manifest.modules, [&](const auto &module) {
        return referencedModules.lookup(module.id.getValue()) == 0;
      }))
    return invalid("package contains an unreferenced module");
  if (llvm::Error error = verifyRuntimeLaunchContract(manifest))
    return std::move(error);

  if (packageRoot.empty())
    return invalid("package root must not be empty");
  if (llvm::Error error = verifyModuleFiles(manifest, packageRoot))
    return std::move(error);
  return VerifiedPackageManifest(std::move(manifest));
}

} // namespace wafer::runtime
