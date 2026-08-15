//===- Package.cpp - Typed compiler package assembly ---------------------===//

#include "PackageInternal.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace wafer::compiler {

struct VerifiedPackageBuilder {
  static VerifiedPackage
  makePackage(llvm::StringRef rootDirectory, ExecutionConfig executionConfig,
              runtime::VerifiedPackageManifest manifest) {
    return VerifiedPackage(rootDirectory, executionConfig, std::move(manifest));
  }
};

namespace {

llvm::Error fail(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-compile: " << message << "\n";
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

bool pathEntryExists(llvm::StringRef path) {
  llvm::sys::fs::file_type type =
      llvm::sys::fs::get_file_type(path, /*Follow=*/false);
  return type != llvm::sys::fs::file_type::file_not_found &&
         type != llvm::sys::fs::file_type::status_error;
}

bool isDirectory(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::directory_file;
}

bool isRegularFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

llvm::Error createDirectory(llvm::StringRef path) {
  if (std::error_code error = llvm::sys::fs::create_directories(path))
    return llvm::createStringError(error, "failed to create package directory");
  return llvm::Error::success();
}

llvm::Error copyDirectory(llvm::StringRef sourceDirectory,
                          llvm::StringRef destinationDirectory) {
  if (!isDirectory(sourceDirectory))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "tensor program is not a directory");
  if (llvm::Error error = createDirectory(destinationDirectory))
    return error;

  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(sourceDirectory, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return llvm::createStringError(error, "failed to walk tensor program");
    llvm::StringRef source = iterator->path();
    llvm::StringRef relative = source;
    if (!relative.consume_front(sourceDirectory))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "failed to derive package member path");
    if (relative.starts_with(llvm::sys::path::get_separator()))
      relative = relative.drop_front();
    llvm::SmallString<256> destination(destinationDirectory);
    llvm::sys::path::append(destination, relative);

    llvm::sys::fs::file_type type = iterator->type();
    if (type == llvm::sys::fs::file_type::directory_file) {
      if (llvm::Error directoryError = createDirectory(destination))
        return directoryError;
      continue;
    }
    if (type != llvm::sys::fs::file_type::regular_file)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "tensor program contains a non-regular package member");
    llvm::SmallString<256> parent(destination);
    llvm::sys::path::remove_filename(parent);
    if (llvm::Error directoryError = createDirectory(parent))
      return directoryError;
    if (std::error_code copyError =
            llvm::sys::fs::copy_file(source, destination))
      return llvm::createStringError(copyError,
                                     "failed to copy tensor program member");
  }
  if (error)
    return llvm::createStringError(error, "failed to walk tensor program");
  return llvm::Error::success();
}

runtime::PackageResourceRole getPackageRole(KernelABISlotRole role) {
  switch (role) {
  case KernelABISlotRole::UserInput:
    return runtime::PackageResourceRole::UserInput;
  case KernelABISlotRole::Parameter:
    return runtime::PackageResourceRole::Parameter;
  case KernelABISlotRole::Constant:
    return runtime::PackageResourceRole::Constant;
  case KernelABISlotRole::Output:
    return runtime::PackageResourceRole::Output;
  case KernelABISlotRole::Workspace:
    return runtime::PackageResourceRole::Workspace;
  case KernelABISlotRole::TransportStatus:
    return runtime::PackageResourceRole::TransportStatus;
  }
  llvm_unreachable("unknown kernel ABI slot role");
}

ProgramResourceRole getProgramRole(KernelABISlotRole role) {
  switch (role) {
  case KernelABISlotRole::UserInput:
    return ProgramResourceRole::UserInput;
  case KernelABISlotRole::Parameter:
    return ProgramResourceRole::Parameter;
  case KernelABISlotRole::Constant:
    return ProgramResourceRole::Constant;
  case KernelABISlotRole::Output:
    return ProgramResourceRole::Output;
  case KernelABISlotRole::Workspace:
  case KernelABISlotRole::TransportStatus:
    llvm_unreachable("workspace has no program resource binding");
  }
  llvm_unreachable("unknown kernel ABI slot role");
}

runtime::PackageAccessMode getAccess(KernelABISlotRole role) {
  switch (role) {
  case KernelABISlotRole::UserInput:
  case KernelABISlotRole::Parameter:
  case KernelABISlotRole::Constant:
    return runtime::PackageAccessMode::ReadOnly;
  case KernelABISlotRole::Output:
    return runtime::PackageAccessMode::WriteOnly;
  case KernelABISlotRole::Workspace:
  case KernelABISlotRole::TransportStatus:
    return runtime::PackageAccessMode::ReadWrite;
  }
  llvm_unreachable("unknown kernel ABI slot role");
}

bool haveSameProgramSlice(const frontend::ProgramPartitionSlice &lhs,
                          const frontend::ProgramPartitionSlice &rhs) {
  return lhs.partitionId == rhs.partitionId && lhs.replicaId == rhs.replicaId &&
         lhs.offsets == rhs.offsets && lhs.sizes == rhs.sizes &&
         lhs.strides == rhs.strides && lhs.payloadPath == rhs.payloadPath;
}

bool haveSameProgramBinding(const ProgramResourceBinding &lhs,
                            const ProgramResourceBinding &rhs) {
  return lhs.role == rhs.role && lhs.index == rhs.index &&
         lhs.programIndex == rhs.programIndex && lhs.dtype == rhs.dtype &&
         lhs.distribution == rhs.distribution &&
         lhs.globalShape == rhs.globalShape &&
         lhs.localShape == rhs.localShape &&
         haveSameProgramSlice(lhs.slice, rhs.slice);
}

bool doesSlotMatchPackageResource(
    const KernelABISlot &slot, const runtime::PackageResourceRecord &resource) {
  return resource.role == getPackageRole(slot.role) &&
         resource.roleIndex == slot.resourceIndex &&
         resource.type.dtype == slot.dtype &&
         resource.type.shape == slot.shape &&
         resource.bytes == static_cast<uint64_t>(slot.byteSize) &&
         resource.alignment == static_cast<uint64_t>(slot.alignment) &&
         resource.access == getAccess(slot.role) && resource.hostVisible;
}

runtime::PackageModuleExportRole getPackageExportRole(TargetExportRole role) {
  switch (role) {
  case TargetExportRole::Prepare:
    return runtime::PackageModuleExportRole::Prepare;
  case TargetExportRole::Main:
    return runtime::PackageModuleExportRole::Main;
  }
  llvm_unreachable("unknown target export role");
}

llvm::StringRef getMainExport(const VerifiedTargetModule &module) {
  for (const VerifiedTargetExport &targetExport : module.getExports())
    if (targetExport.getRole() == TargetExportRole::Main)
      return targetExport.getSymbol();
  return {};
}

llvm::Expected<runtime::PackageManifest>
buildManifest(const CardExecutable &cardExecutable,
              const LinkedTargetModules &targetModules,
              llvm::raw_ostream &diagnostics) {
  const ExecutionConfig &config = cardExecutable.getExecutionConfig();
  if (cardExecutable.getRuntimeLaunchContract() !=
      targetModules.getRuntimeLaunchContract())
    return fail(diagnostics,
                "package runtime launch contract does not match executable "
                "and linked target modules");
  if (targetModules.getExecutionConfig() != config ||
      cardExecutable.getTileExecutables().size() !=
          targetModules.getTileInterfaces().size() ||
      cardExecutable.getTileExecutables().size() !=
          static_cast<size_t>(config.getTileCount()))
    return fail(diagnostics,
                "package Tile domain does not match executable and "
                "linked target modules");

  if (targetModules.getModules().empty())
    return fail(diagnostics, "package target module domain is empty");
  const VerifiedTargetModule &firstTargetModule =
      targetModules.getModules().front();
  if (firstTargetModule.getTargetIdentityId() != config.getTargetIdentityId() ||
      firstTargetModule.getKernelRuntimeABIId() !=
          KernelRuntimeABIId::waferTx81Kernel() ||
      firstTargetModule.getModuleFormat() != kCurrentTargetModuleFormat)
    return fail(diagnostics,
                "package target module facts do not match ExecutionConfig "
                "and target registry");
  runtime::PackageManifest manifest(firstTargetModule.getTargetIdentityId(),
                                    firstTargetModule.getKernelRuntimeABIId(),
                                    targetModules.getRuntimeLaunchContract(),
                                    firstTargetModule.getModuleFormat());
  manifest.program = runtime::ProgramId(0);
  manifest.cardCount = 1;
  manifest.tileCount = config.getTileCount();

  const KernelRuntimeLaunchContract *kernelLaunch =
      targetModules.getRuntimeLaunchContract().getKernel();
  const bool sharedModule = kernelLaunch != nullptr;
  const bool hasPrepare =
      llvm::is_contained(targetModules.getRuntimeLaunchContract().getPhases(),
                         RuntimeLaunchPhaseRole::Prepare);
  if (targetModules.getModules().size() !=
      (sharedModule ? 1u : static_cast<size_t>(config.getTileCount())))
    return fail(diagnostics, "package target module topology does not match "
                             "runtime launch contract");

  const size_t tileCount = cardExecutable.getTileExecutables().size();
  std::vector<const TileExecutable *> tilesByLaunchSlot(tileCount, nullptr);
  std::vector<const VerifiedTargetTileInterface *> interfacesByLaunchSlot(
      tileCount, nullptr);
  std::set<int64_t> tileIds;
  for (const TileExecutable &tile : cardExecutable.getTileExecutables()) {
    const int64_t launchSlot = tile.getLaunchSlotId().getValue();
    if (tile.getCardId() != CardId(0) || tile.getTileId().getValue() < 0 ||
        launchSlot < 0 || launchSlot >= static_cast<int64_t>(tileCount) ||
        !tileIds.insert(tile.getTileId().getValue()).second ||
        tilesByLaunchSlot[launchSlot] != nullptr)
      return fail(diagnostics,
                  "package executable has invalid or duplicate physical "
                  "Tile identity");
    tilesByLaunchSlot[launchSlot] = &tile;
  }
  for (const VerifiedTargetTileInterface &tileInterface :
       targetModules.getTileInterfaces()) {
    const int64_t launchSlot = tileInterface.getLaunchSlotId().getValue();
    if (tileInterface.getCardId() != CardId(0) ||
        tileInterface.getTileId().getValue() < 0 || launchSlot < 0 ||
        launchSlot >= static_cast<int64_t>(tileCount) ||
        interfacesByLaunchSlot[launchSlot] != nullptr)
      return fail(diagnostics,
                  "package target module has invalid or duplicate physical "
                  "Tile interface identity");
    interfacesByLaunchSlot[launchSlot] = &tileInterface;
  }
  if (llvm::is_contained(tilesByLaunchSlot, nullptr) ||
      llvm::is_contained(interfacesByLaunchSlot, nullptr))
    return fail(diagnostics,
                "package launch-slot domain is not dense and complete");

  const std::vector<ProgramResourceBinding> &sharedProgramBindings =
      tilesByLaunchSlot.front()->getProgramBindings();
  for (const TileExecutable *tile : tilesByLaunchSlot) {
    const std::vector<ProgramResourceBinding> &bindings =
        tile->getProgramBindings();
    if (bindings.size() != sharedProgramBindings.size() ||
        !std::equal(bindings.begin(), bindings.end(),
                    sharedProgramBindings.begin(), haveSameProgramBinding))
      return fail(diagnostics, "Tile module boundaries do not agree on the "
                               "typed card-shared resource domain");
  }

  for (auto [expectedModuleId, target] :
       llvm::enumerate(targetModules.getModules())) {
    if (target.getId().getValue() != expectedModuleId ||
        target.getTargetIdentityId() !=
            firstTargetModule.getTargetIdentityId() ||
        target.getKernelRuntimeABIId() !=
            firstTargetModule.getKernelRuntimeABIId() ||
        target.getModuleFormat() != firstTargetModule.getModuleFormat())
      return fail(diagnostics, "package target module domain is not canonical");
    bool seenPrepare = false;
    bool seenMain = false;
    runtime::PackageModuleRecord module;
    module.id = runtime::ModuleId(expectedModuleId);
    module.relativePath = target.getRelativePath().str();
    module.digest = target.getContentDigest().str();
    module.format = target.getModuleFormat().str();
    for (const VerifiedTargetExport &targetExport : target.getExports()) {
      bool *seen = targetExport.getRole() == TargetExportRole::Prepare
                       ? &seenPrepare
                       : &seenMain;
      if (*seen || targetExport.getSymbol().empty())
        return fail(diagnostics,
                    "package target module exports are not canonical");
      *seen = true;
      module.exports.push_back({getPackageExportRole(targetExport.getRole()),
                                targetExport.getSymbol().str()});
    }
    if (!seenMain || seenPrepare != hasPrepare ||
        module.exports.size() != (hasPrepare ? 2u : 1u))
      return fail(
          diagnostics,
          "package target module exports do not match runtime launch contract");
    manifest.modules.push_back(std::move(module));
  }

  uint64_t nextResourceId = 0;
  std::vector<std::optional<runtime::ResourceId>> sharedProgramResources(
      sharedProgramBindings.size());
  std::set<uint64_t> usedModuleIds;
  for (int64_t launchSlot = 0; launchSlot < static_cast<int64_t>(tileCount);
       ++launchSlot) {
    const TileExecutable &tile = *tilesByLaunchSlot[launchSlot];
    const VerifiedTargetTileInterface &tileInterface =
        *interfacesByLaunchSlot[launchSlot];
    const uint64_t moduleId = tileInterface.getModuleId().getValue();
    if (moduleId >= targetModules.getModules().size())
      return fail(diagnostics,
                  "package Tile launch interface references an unknown "
                  "module");
    const VerifiedTargetModule &target = targetModules.getModules()[moduleId];
    if (tile.getCardId() != CardId(0) ||
        tile.getCardId() != tileInterface.getCardId() ||
        tile.getTileId() != tileInterface.getTileId() ||
        tile.getLaunchSlotId() != tileInterface.getLaunchSlotId() ||
        target.getId() != tileInterface.getModuleId() ||
        (sharedModule && moduleId != 0) ||
        (!sharedModule && !usedModuleIds.insert(moduleId).second) ||
        tile.getEntrySymbol() != getMainExport(target))
      return fail(diagnostics,
                  "package Tile/module/entry/profile domain is not canonical");

    std::vector<bool> usedProgramBindings(tile.getProgramBindings().size(),
                                          false);
    runtime::PackageEntrypointRecord entry;
    entry.id = runtime::EntryId(static_cast<uint64_t>(launchSlot));
    entry.cardId = tile.getCardId();
    entry.tileId = tile.getTileId();
    entry.launchSlot = runtime::LaunchSlotId(
        static_cast<uint64_t>(tile.getLaunchSlotId().getValue()));
    entry.module = runtime::ModuleId(moduleId);
    entry.completion =
        runtime::PackageEntryCompletionKind::ReturnAfterLocalDrain;
    std::optional<runtime::ResourceId> transportStatusResource;

    for (const KernelABISlot &slot : tileInterface.getKernelABISlots()) {
      if (slot.ordinal != static_cast<int64_t>(entry.slots.size()) ||
          slot.byteSize <= 0 || slot.alignment <= 0)
        return fail(diagnostics,
                    "package input Kernel ABI slots are not canonical");
      const ProgramResourceBinding *binding = nullptr;
      size_t bindingPosition = 0;
      if (slot.role != KernelABISlotRole::Workspace &&
          slot.role != KernelABISlotRole::TransportStatus) {
        for (auto [position, candidate] :
             llvm::enumerate(tile.getProgramBindings())) {
          if (!detail::doesPackageSlotMatchProgramBinding(slot, candidate))
            continue;
          if (binding)
            return fail(diagnostics,
                        "package program binding is duplicated for ABI slot");
          binding = &candidate;
          bindingPosition = position;
        }
        if (!binding || usedProgramBindings[bindingPosition])
          return fail(diagnostics,
                      "package ABI slot does not match executable resource "
                      "binding");
        usedProgramBindings[bindingPosition] = true;
      } else if (slot.role == KernelABISlotRole::Workspace &&
                 !detail::isValidPackageCompilerManagedSlot(slot)) {
        return fail(diagnostics,
                    "package workspace ABI slot identity is invalid");
      } else if (slot.role == KernelABISlotRole::TransportStatus &&
                 (!detail::isValidPackageCompilerManagedSlot(slot) ||
                  transportStatusResource)) {
        return fail(diagnostics,
                    "package Direct DTE status ABI slot identity is invalid");
      }

      std::optional<runtime::ResourceId> resourceId;
      if (binding && sharedProgramResources[bindingPosition]) {
        resourceId = sharedProgramResources[bindingPosition];
        const runtime::PackageResourceRecord &resource =
            manifest.resources[resourceId->getValue()];
        if (!doesSlotMatchPackageResource(slot, resource))
          return fail(diagnostics, "Tile ABI slots disagree on a typed "
                                   "card-shared program resource");
      } else {
        runtime::PackageResourceRecord resource;
        resource.id = runtime::ResourceId(nextResourceId++);
        resource.scope =
            binding ? runtime::PackageResourceScope(
                          runtime::CardResourceScope{tile.getCardId()})
                    : runtime::PackageResourceScope(runtime::TileResourceScope{
                          tile.getCardId(), tile.getTileId()});
        resource.role = getPackageRole(slot.role);
        resource.roleIndex = slot.resourceIndex;
        resource.name = slot.name;
        resource.type = {slot.dtype, slot.shape};
        resource.bytes = static_cast<uint64_t>(slot.byteSize);
        resource.alignment = static_cast<uint64_t>(slot.alignment);
        resource.access = getAccess(slot.role);
        resource.hostVisible = binding != nullptr;
        resourceId = resource.id;
        manifest.resources.push_back(std::move(resource));
        if (binding)
          sharedProgramResources[bindingPosition] = resourceId;
      }
      if (slot.role == KernelABISlotRole::TransportStatus)
        transportStatusResource = resourceId;
      entry.slots.push_back({static_cast<uint64_t>(slot.ordinal), *resourceId,
                             getAccess(slot.role)});
    }
    if (!llvm::all_of(usedProgramBindings, [](bool used) { return used; }))
      return fail(diagnostics,
                  "package ABI slots omit executable program resource "
                  "bindings");

    if (tile.getTransportContract() == TransportContract::DirectDTE) {
      if (!transportStatusResource)
        return fail(diagnostics,
                    "Direct DTE entry is missing its transport status slot");
      entry.transport = runtime::DirectDTETransportRequirements{
          *transportStatusResource, runtime::kDirectDTEStatusABI.str(), true};
    } else if (transportStatusResource) {
      return fail(diagnostics,
                  "transport status slot exists without Direct DTE contract");
    }

    manifest.entries.push_back(std::move(entry));
  }
  if (!sharedModule &&
      usedModuleIds.size() != targetModules.getModules().size())
    return fail(diagnostics,
                "package per-Tile module domain is not covered exactly once");
  return manifest;
}

llvm::Error copyTargetModules(const LinkedTargetModules &targetModules,
                              llvm::StringRef stagingRoot,
                              llvm::raw_ostream &diagnostics,
                              std::optional<int64_t> failAfterLaunchSlot) {
  for (const VerifiedTargetModule &module : targetModules.getModules()) {
    llvm::SmallString<256> source(targetModules.getRootDirectory());
    llvm::sys::path::append(source, module.getRelativePath());
    if (!isRegularFile(source))
      return fail(diagnostics,
                  "verified target module disappeared before package "
                  "assembly");
    llvm::SmallString<256> destination(stagingRoot);
    llvm::sys::path::append(destination, module.getRelativePath());
    llvm::SmallString<256> parent(destination);
    llvm::sys::path::remove_filename(parent);
    if (llvm::Error error = createDirectory(parent))
      return fail(diagnostics, llvm::toString(std::move(error)));
    if (std::error_code error = llvm::sys::fs::copy_file(source, destination))
      return fail(diagnostics,
                  "failed to copy verified target module: " + error.message());
  }
  if (failAfterLaunchSlot && *failAfterLaunchSlot >= 0 &&
      *failAfterLaunchSlot <
          static_cast<int64_t>(targetModules.getTileInterfaces().size()))
    return fail(diagnostics,
                "test-only injected package failure after launch slot " +
                    std::to_string(*failAfterLaunchSlot));
  return llvm::Error::success();
}

llvm::Error writeManifest(llvm::StringRef stagingRoot,
                          const runtime::VerifiedPackageManifest &manifest) {
  llvm::SmallString<256> path(stagingRoot);
  llvm::sys::path::append(path, runtime::kPackageManifestFileName);
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "failed to open package manifest");
  output << runtime::serializeCanonicalPackageJson(manifest);
  output.close();
  if (output.has_error())
    return llvm::createStringError(llvm::errc::io_error,
                                   "failed to write package manifest");
  return llvm::Error::success();
}

llvm::Error fsyncPath(llvm::StringRef path, bool directory) {
#ifdef __linux__
  std::string storage = path.str();
  int flags = O_RDONLY | O_CLOEXEC;
  if (directory)
    flags |= O_DIRECTORY;
  int descriptor = ::open(storage.c_str(), flags);
  if (descriptor < 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to open package member for fsync");
  int result = ::fsync(descriptor);
  int errorNumber = errno;
  ::close(descriptor);
  if (result != 0)
    return llvm::createStringError(
        std::error_code(errorNumber, std::generic_category()),
        "failed to fsync package member");
  return llvm::Error::success();
#else
  (void)path;
  (void)directory;
  return llvm::createStringError(llvm::errc::not_supported,
                                 "package fsync is unsupported on this host");
#endif
}

llvm::Error fsyncPackageTree(llvm::StringRef root) {
  std::vector<std::string> directories;
  directories.push_back(root.str());
  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(root, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return llvm::createStringError(error, "failed to walk package for fsync");
    if (iterator->type() == llvm::sys::fs::file_type::directory_file) {
      directories.push_back(iterator->path());
      continue;
    }
    if (iterator->type() != llvm::sys::fs::file_type::regular_file)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "package contains unsupported member during fsync");
    if (llvm::Error syncError = fsyncPath(iterator->path(), false))
      return syncError;
  }
  if (error)
    return llvm::createStringError(error, "failed to walk package for fsync");
  for (llvm::StringRef directory : llvm::reverse(directories))
    if (llvm::Error syncError = fsyncPath(directory, true))
      return syncError;
  return llvm::Error::success();
}

bool renameDirectoryNoReplace(llvm::StringRef source,
                              llvm::StringRef destination,
                              llvm::raw_ostream &diagnostics) {
#ifdef __linux__
  std::string sourceStorage = source.str();
  std::string destinationStorage = destination.str();
  if (::syscall(SYS_renameat2, AT_FDCWD, sourceStorage.c_str(), AT_FDCWD,
                destinationStorage.c_str(), RENAME_NOREPLACE) == 0)
    return false;
  diagnostics << "wafer-compile: package_write_failed: "
              << std::error_code(errno, std::generic_category()).message()
              << "\n";
  return true;
#else
  (void)source;
  (void)destination;
  diagnostics << "wafer-compile: package_write_failed: no-replace directory "
                 "rename is unsupported on this host\n";
  return true;
#endif
}

} // namespace

bool detail::doesPackageSlotMatchProgramBinding(
    const KernelABISlot &slot, const ProgramResourceBinding &binding) {
  if (slot.role == KernelABISlotRole::Workspace ||
      slot.role == KernelABISlotRole::TransportStatus)
    return false;
  return binding.role == getProgramRole(slot.role) &&
         binding.index == slot.resourceIndex && binding.dtype == slot.dtype &&
         binding.localShape == slot.shape;
}

bool detail::isValidPackageCompilerManagedSlot(const KernelABISlot &slot) {
  if (slot.layout != MemLayout::Tensor || slot.byteSize <= 0 ||
      slot.alignment <= 0)
    return false;
  if (slot.role == KernelABISlotRole::Workspace) {
    if (slot.dtype != "u8" || slot.shape.size() != 1 ||
        slot.shape.front() != slot.byteSize)
      return false;
    if (slot.resourceIndex == 0)
      return true;
    if (slot.resourceIndex != 1 ||
        slot.alignment != WAFER_TX81_PROFILER_BUFFER_ALIGNMENT)
      return false;
    return slot.byteSize == WAFER_TX81_PROFILER_MIN_BUFFER_BYTES ||
           slot.byteSize == WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES;
  }
  if (slot.role == KernelABISlotRole::TransportStatus)
    return slot.resourceIndex == 0 && slot.dtype == "u32" &&
           slot.shape == std::vector<int64_t>{1} &&
           slot.byteSize == runtime::kDirectDTEStatusStorageBytes &&
           slot.alignment == runtime::kDirectDTEStatusStorageAlignment;
  return false;
}

llvm::Expected<VerifiedPackage>
detail::writePackage(llvm::StringRef tensorProgramDirectory,
                     const CardExecutable &cardExecutable,
                     const LinkedTargetModules &targetModules,
                     llvm::StringRef outputDirectory,
                     llvm::raw_ostream &diagnostics,
                     std::optional<int64_t> failAfterLaunchSlot) {
  if (tensorProgramDirectory.empty() || outputDirectory.empty())
    return fail(diagnostics,
                "package input/output directory must not be empty");
  if (!isDirectory(tensorProgramDirectory))
    return fail(diagnostics, "package tensor-program input is not a directory");
  if (pathEntryExists(outputDirectory))
    return fail(diagnostics, "refusing to replace existing package directory");
  for (llvm::StringRef reserved :
       {llvm::StringRef("modules"),
        llvm::StringRef(runtime::kPackageManifestFileName)}) {
    llvm::SmallString<256> path(tensorProgramDirectory);
    llvm::sys::path::append(path, reserved);
    if (pathEntryExists(path))
      return fail(diagnostics,
                  "tensor program contains reserved package member '" +
                      reserved.str() + "'");
  }

  llvm::SmallString<256> outputParent(outputDirectory);
  llvm::sys::path::remove_filename(outputParent);
  if (outputParent.empty())
    outputParent = ".";
  if (llvm::Error error = createDirectory(outputParent))
    return fail(diagnostics, llvm::toString(std::move(error)));
  llvm::SmallString<256> stagingPrefix(outputParent);
  llvm::sys::path::append(stagingPrefix, ".wafer-package-staging");
  llvm::SmallString<256> stagingRoot;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(stagingPrefix, stagingRoot))
    return fail(diagnostics,
                "failed to create package staging: " + error.message());
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(stagingRoot); });

  if (llvm::Error error = copyDirectory(tensorProgramDirectory, stagingRoot))
    return fail(diagnostics, llvm::toString(std::move(error)));
  if (llvm::Error error = copyTargetModules(targetModules, stagingRoot,
                                            diagnostics, failAfterLaunchSlot))
    return std::move(error);

  llvm::Expected<runtime::PackageManifest> manifest =
      buildManifest(cardExecutable, targetModules, diagnostics);
  if (!manifest)
    return manifest.takeError();
  llvm::Expected<runtime::VerifiedPackageManifest> verified =
      runtime::verifyPackageManifest(std::move(*manifest), stagingRoot);
  if (!verified)
    return fail(diagnostics, "package manifest verification failed: " +
                                 llvm::toString(verified.takeError()));
  if (llvm::Error error = writeManifest(stagingRoot, *verified))
    return fail(diagnostics, llvm::toString(std::move(error)));

  llvm::Expected<runtime::VerifiedPackageManifest> readback =
      runtime::loadVerifiedPackageManifest(stagingRoot);
  if (!readback)
    return fail(diagnostics, "package manifest readback failed: " +
                                 llvm::toString(readback.takeError()));
  const runtime::PackageManifest &readbackManifest = readback->getManifest();
  const ExecutionConfig &executionConfig = cardExecutable.getExecutionConfig();
  const VerifiedTargetModule &targetReadback =
      targetModules.getModules().front();
  if (readbackManifest.cardCount != 1 ||
      readbackManifest.tileCount != executionConfig.getTileCount() ||
      readbackManifest.targetIdentity != targetReadback.getTargetIdentityId() ||
      readbackManifest.runtimeABI != targetReadback.getKernelRuntimeABIId() ||
      readbackManifest.launch != targetModules.getRuntimeLaunchContract() ||
      readbackManifest.moduleFormat != targetReadback.getModuleFormat() ||
      readbackManifest.targetIdentity != executionConfig.getTargetIdentityId())
    return fail(diagnostics,
                "package manifest readback does not match target module facts "
                "and ExecutionConfig");
  if (llvm::Error error = fsyncPackageTree(stagingRoot))
    return fail(diagnostics, llvm::toString(std::move(error)));
  if (renameDirectoryNoReplace(stagingRoot, outputDirectory, diagnostics))
    return llvm::createStringError(llvm::errc::io_error,
                                   "package directory rename failed");
  cleanup.release();
  return VerifiedPackageBuilder::makePackage(
      outputDirectory, cardExecutable.getExecutionConfig(),
      std::move(*readback));
}

} // namespace wafer::compiler
