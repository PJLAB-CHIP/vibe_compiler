//===- Package.cpp - Typed compiler package assembly ---------------------===//

#include "PackageInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cstdint>
#include <optional>
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

struct PackageBundleBuilder {
  static PackageBundle make(llvm::StringRef rootDirectory,
                            ExecutionConfig executionConfig,
                            runtime::VerifiedPackageManifest manifest) {
    return PackageBundle(rootDirectory, executionConfig, std::move(manifest));
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
                                   "grouped program is not a directory");
  if (llvm::Error error = createDirectory(destinationDirectory))
    return error;

  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(sourceDirectory, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return llvm::createStringError(error, "failed to walk grouped program");
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
          "grouped program contains a non-regular package member");
    llvm::SmallString<256> parent(destination);
    llvm::sys::path::remove_filename(parent);
    if (llvm::Error directoryError = createDirectory(parent))
      return directoryError;
    if (std::error_code copyError =
            llvm::sys::fs::copy_file(source, destination))
      return llvm::createStringError(copyError,
                                     "failed to copy grouped program member");
  }
  if (error)
    return llvm::createStringError(error, "failed to walk grouped program");
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

llvm::Expected<runtime::PackageManifest>
buildManifest(const ExecutableBundle &executableBundle,
              const TargetArtifactBundle &targetArtifacts,
              llvm::raw_ostream &diagnostics) {
  const ExecutionConfig &config = executableBundle.getExecutionConfig();
  if (targetArtifacts.getExecutionConfig() != config ||
      executableBundle.getRankExecutables().size() !=
          targetArtifacts.getModules().size() ||
      executableBundle.getRankExecutables().size() !=
          static_cast<size_t>(config.getRankCount()))
    return fail(diagnostics,
                "package rank domain does not match executable and target "
                "artifact bundles");

  const TargetProfileRecord &targetProfile =
      getTargetProfileRecord(config.getTargetProfileId());
  if (targetArtifacts.getModules().empty())
    return fail(diagnostics, "package target module domain is empty");
  const VerifiedTargetModule &firstTargetModule =
      targetArtifacts.getModules().front();
  if (firstTargetModule.getTargetProfileId() != config.getTargetProfileId() ||
      firstTargetModule.getTargetIdentityId() !=
          targetProfile.targetIdentity ||
      firstTargetModule.getKernelRuntimeABIId() !=
          targetProfile.kernelRuntimeABI ||
      firstTargetModule.getModuleFormat() != targetProfile.moduleFormat)
    return fail(diagnostics,
                "package target module facts do not match ExecutionConfig "
                "and target registry");
  runtime::PackageManifest manifest(
      firstTargetModule.getTargetProfileId(),
      firstTargetModule.getTargetIdentityId(),
      firstTargetModule.getKernelRuntimeABIId(),
      firstTargetModule.getModuleFormat());
  manifest.program = runtime::ProgramId(0);
  manifest.rankCount = config.getRankCount();

  uint64_t nextResourceId = 0;
  for (int64_t logicalRank = 0; logicalRank < config.getRankCount();
       ++logicalRank) {
    const RankExecutable &rank =
        executableBundle.getRankExecutables()[logicalRank];
    const VerifiedTargetModule &target =
        targetArtifacts.getModules()[logicalRank];
    if (rank.getLogicalRank() != logicalRank ||
        target.getLogicalRank() != logicalRank ||
        rank.getEntrySymbol() != target.getEntrySymbol() ||
        target.getTargetProfileId() != firstTargetModule.getTargetProfileId() ||
        target.getTargetIdentityId() !=
            firstTargetModule.getTargetIdentityId() ||
        target.getKernelRuntimeABIId() !=
            firstTargetModule.getKernelRuntimeABIId() ||
        target.getModuleFormat() != firstTargetModule.getModuleFormat())
      return fail(diagnostics,
                  "package rank/module/entry/profile domain is not canonical");

    std::vector<bool> usedProgramBindings(rank.getProgramBindings().size(),
                                          false);
    runtime::PackageEntrypointRecord entry;
    entry.id = runtime::EntryId(logicalRank);
    entry.logicalRank = logicalRank;
    entry.module = runtime::ModuleId(logicalRank);
    entry.symbol = target.getEntrySymbol().str();
    entry.terminalCompletion = runtime::CompletionId(logicalRank);
    std::optional<runtime::ResourceId> transportStatusResource;

    for (const KernelABISlot &slot : target.getKernelABISlots()) {
      if (slot.ordinal != static_cast<int64_t>(entry.slots.size()) ||
          slot.byteSize <= 0 || slot.alignment <= 0)
        return fail(diagnostics,
                    "package input Kernel ABI slots are not canonical");
      if (slot.role != KernelABISlotRole::Workspace &&
          slot.role != KernelABISlotRole::TransportStatus) {
        const RankProgramBinding *binding = nullptr;
        size_t bindingPosition = 0;
        for (auto [position, candidate] :
             llvm::enumerate(rank.getProgramBindings())) {
          if (candidate.role != getProgramRole(slot.role) ||
              candidate.index != slot.resourceIndex)
            continue;
          if (binding)
            return fail(diagnostics,
                        "package program binding is duplicated for ABI slot");
          binding = &candidate;
          bindingPosition = position;
        }
        if (!binding || usedProgramBindings[bindingPosition] ||
            binding->dtype != slot.dtype || binding->localShape != slot.shape ||
            binding->name != slot.name)
          return fail(diagnostics,
                      "package ABI slot does not match executable resource "
                      "binding");
        usedProgramBindings[bindingPosition] = true;
      } else if (slot.role == KernelABISlotRole::Workspace &&
                 (slot.resourceIndex != 0 ||
                  slot.name != "default_ddr_arena")) {
        return fail(diagnostics,
                    "package workspace ABI slot identity is invalid");
      } else if (slot.role == KernelABISlotRole::TransportStatus &&
                 (slot.resourceIndex != 0 ||
                  slot.name != "direct_dte_status" || slot.dtype != "u32" ||
                  slot.shape != std::vector<int64_t>{1} ||
                  slot.byteSize != 4 || slot.alignment != 4 ||
                  transportStatusResource)) {
        return fail(diagnostics,
                    "package Direct DTE status ABI slot identity is invalid");
      }

      runtime::PackageResourceRecord resource;
      resource.id = runtime::ResourceId(nextResourceId++);
      resource.logicalRank = logicalRank;
      resource.role = getPackageRole(slot.role);
      resource.roleIndex = slot.resourceIndex;
      resource.name = slot.name;
      resource.type = {slot.dtype, slot.shape};
      resource.bytes = static_cast<uint64_t>(slot.byteSize);
      resource.alignment = static_cast<uint64_t>(slot.alignment);
      resource.access = getAccess(slot.role);
      resource.hostVisible =
          slot.role != KernelABISlotRole::Workspace &&
          slot.role != KernelABISlotRole::TransportStatus;
      if (slot.role == KernelABISlotRole::TransportStatus)
        transportStatusResource = resource.id;
      entry.slots.push_back(
          {static_cast<uint64_t>(slot.ordinal), resource.id, resource.access});
      manifest.resources.push_back(std::move(resource));
    }
    if (!llvm::all_of(usedProgramBindings, [](bool used) { return used; }))
      return fail(diagnostics,
                  "package ABI slots omit executable program resource "
                  "bindings");

    if (rank.getTransportContract() == TransportContract::DirectDTE) {
      if (!transportStatusResource)
        return fail(diagnostics,
                    "Direct DTE entry is missing its transport status slot");
      entry.transport = runtime::DirectDTETransportRequirements{
          *transportStatusResource, runtime::kDirectDTEStatusABI.str(), true};
    } else if (transportStatusResource) {
      return fail(diagnostics,
                  "transport status slot exists without Direct DTE contract");
    }

    manifest.modules.push_back({runtime::ModuleId(logicalRank), logicalRank,
                                target.getRelativePath().str(),
                                target.getContentDigest().str(),
                                target.getModuleFormat().str()});
    manifest.entries.push_back(std::move(entry));
    manifest.completions.push_back(
        {runtime::CompletionId(logicalRank), logicalRank, "entry_return"});
  }
  return manifest;
}

llvm::Error copyTargetModules(const TargetArtifactBundle &targetArtifacts,
                              llvm::StringRef stagingRoot,
                              llvm::raw_ostream &diagnostics,
                              std::optional<int64_t> failAfterLogicalRank) {
  for (const VerifiedTargetModule &module : targetArtifacts.getModules()) {
    llvm::SmallString<256> source(targetArtifacts.getRootDirectory());
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
    if (failAfterLogicalRank &&
        module.getLogicalRank() == *failAfterLogicalRank)
      return fail(diagnostics,
                  "test-only injected package failure after logical rank " +
                      std::to_string(module.getLogicalRank()));
  }
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

bool publishDirectoryNoReplace(llvm::StringRef source,
                               llvm::StringRef destination,
                               llvm::raw_ostream &diagnostics) {
#ifdef __linux__
  std::string sourceStorage = source.str();
  std::string destinationStorage = destination.str();
  if (::syscall(SYS_renameat2, AT_FDCWD, sourceStorage.c_str(), AT_FDCWD,
                destinationStorage.c_str(), RENAME_NOREPLACE) == 0)
    return false;
  diagnostics << "wafer-compile: package_publication_failed: "
              << std::error_code(errno, std::generic_category()).message()
              << "\n";
  return true;
#else
  (void)source;
  (void)destination;
  diagnostics << "wafer-compile: package_publication_failed: atomic "
                 "no-replace publication is unsupported on this host\n";
  return true;
#endif
}

} // namespace

llvm::Expected<PackageBundle>
detail::assemblePackageBundleImpl(llvm::StringRef groupedProgramDirectory,
                                  const ExecutableBundle &executableBundle,
                                  const TargetArtifactBundle &targetArtifacts,
                                  llvm::StringRef outputDirectory,
                                  llvm::raw_ostream &diagnostics,
                                  std::optional<int64_t> failAfterLogicalRank) {
  if (groupedProgramDirectory.empty() || outputDirectory.empty())
    return fail(diagnostics,
                "package input/output directory must not be empty");
  if (!isDirectory(groupedProgramDirectory))
    return fail(diagnostics,
                "package grouped-program input is not a directory");
  if (pathEntryExists(outputDirectory))
    return fail(diagnostics, "refusing to replace existing package directory");
  for (llvm::StringRef reserved :
       {llvm::StringRef("modules"),
        llvm::StringRef(runtime::kPackageManifestFileName)}) {
    llvm::SmallString<256> path(groupedProgramDirectory);
    llvm::sys::path::append(path, reserved);
    if (pathEntryExists(path))
      return fail(diagnostics,
                  "grouped program contains reserved package member '" +
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

  if (llvm::Error error = copyDirectory(groupedProgramDirectory, stagingRoot))
    return fail(diagnostics, llvm::toString(std::move(error)));
  if (llvm::Error error = copyTargetModules(targetArtifacts, stagingRoot,
                                            diagnostics, failAfterLogicalRank))
    return std::move(error);

  llvm::Expected<runtime::PackageManifest> manifest =
      buildManifest(executableBundle, targetArtifacts, diagnostics);
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
  const ExecutionConfig &executionConfig =
      executableBundle.getExecutionConfig();
  const VerifiedTargetModule &targetReadback =
      targetArtifacts.getModules().front();
  if (readbackManifest.rankCount != executionConfig.getRankCount() ||
      readbackManifest.targetProfile != targetReadback.getTargetProfileId() ||
      readbackManifest.targetIdentity !=
          targetReadback.getTargetIdentityId() ||
      readbackManifest.runtimeABI !=
          targetReadback.getKernelRuntimeABIId() ||
      readbackManifest.moduleFormat != targetReadback.getModuleFormat() ||
      readbackManifest.targetProfile !=
          executionConfig.getTargetProfileId())
    return fail(diagnostics,
                "package manifest readback does not match target module facts "
                "and ExecutionConfig");
  if (llvm::Error error = fsyncPackageTree(stagingRoot))
    return fail(diagnostics, llvm::toString(std::move(error)));
  if (publishDirectoryNoReplace(stagingRoot, outputDirectory, diagnostics))
    return llvm::createStringError(llvm::errc::io_error,
                                   "package publication failed");
  cleanup.release();
  return PackageBundleBuilder::make(outputDirectory,
                                    executableBundle.getExecutionConfig(),
                                    std::move(*readback));
}

llvm::Expected<PackageBundle>
assemblePackageBundle(llvm::StringRef groupedProgramDirectory,
                      const ExecutableBundle &executableBundle,
                      const TargetArtifactBundle &targetArtifacts,
                      llvm::StringRef outputDirectory,
                      llvm::raw_ostream &diagnostics) {
  return detail::assemblePackageBundleImpl(
      groupedProgramDirectory, executableBundle, targetArtifacts,
      outputDirectory, diagnostics, std::nullopt);
}

} // namespace wafer::compiler
