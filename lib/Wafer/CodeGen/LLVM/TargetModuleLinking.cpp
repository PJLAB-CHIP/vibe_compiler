//===- TargetModuleLinking.cpp - Link target modules --------------------===//

#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
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

namespace wafer::compiler::detail {

llvm::Error fail(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-compile: " << message << "\n";
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

namespace {

llvm::Error validateRuntimeLaunchContractDomain(
    const TargetLLVMModules &targetLLVMModules) {
  const ExecutionConfig &config = targetLLVMModules.getExecutionConfig();
  const RuntimeLaunchContract &launch =
      targetLLVMModules.getRuntimeLaunchContract();
  if (config.getTileCount() != 16 ||
      targetLLVMModules.getModules().size() != 16)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "multi-Tile runtime launch requires the complete 16-Tile domain");

  const TargetLLVMModule &first = targetLLVMModules.getModules().front();
  {
    const KernelRuntimeLaunchContract &kernel = launch.getKernel();
    if (first.getTileEntryArguments().empty())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "multi-Tile pointer-table launch requires at least one typed ABI "
          "slot");
    const uint64_t packetBytes = kernel.form == KernelLaunchForm::Cluster
                                     ? kTx81ClusterKernelArgumentBytesMax
                                     : kTx81KernelArgumentBytesMax;
    if (kernel.entryABI == KernelEntryABI::TileMajorPointerTable) {
      uint64_t remaining = packetBytes / sizeof(uint64_t);
      for (const auto &module : targetLLVMModules.getModules()) {
        if (module.getTileEntryArguments().size() > remaining)
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "multi-Tile Tile-major argument table exceeds the qualified V5.6 "
              "packet limit");
        remaining -= module.getTileEntryArguments().size();
      }
    } else if (kernel.entryABI == KernelEntryABI::TileRowPointerTable) {
      if (static_cast<uint64_t>(config.getTileCount()) >
          packetBytes / sizeof(uint64_t))
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "multi-Tile Tile-row pointer table exceeds the qualified V5.6 "
            "packet limit");
    } else {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "multi-Tile kernel launch has an incompatible entry ABI");
    }
    const size_t transportStatusSlots = llvm::count_if(
        first.getTileEntryArguments(), [](const TileEntryArgument &slot) {
          return slot.kind == TileEntryArgumentKind::TransportStatus;
        });
    if (transportStatusSlots > 1)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "kernel launch has more than one typed transport status slot");
  }
  for (const TargetLLVMModule &module : targetLLVMModules.getModules()) {
    if (module.getEntrySymbol() != first.getEntrySymbol())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "multi-Tile runtime launch requires one shared entry symbol");
  }
  return validateTileEntryArgumentDomain(targetLLVMModules.getModules());
}

bool targetModulePathEntryExists(llvm::StringRef path) {
  llvm::sys::fs::file_type type =
      llvm::sys::fs::get_file_type(path, /*Follow=*/false);
  return type != llvm::sys::fs::file_type::file_not_found &&
         type != llvm::sys::fs::file_type::status_error;
}

} // namespace

bool isRegularTargetFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

llvm::Error validateRuntimeLaunchContractDomainForTesting(
    const TargetLLVMModules &targetLLVMModules) {
  return validateRuntimeLaunchContractDomain(targetLLVMModules);
}

namespace {

bool isExecutableTargetFile(llvm::StringRef path) {
  return isRegularTargetFile(path) &&
         !llvm::sys::fs::access(path, llvm::sys::fs::AccessMode::Execute);
}

bool isDirectory(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/true) ==
         llvm::sys::fs::file_type::directory_file;
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
  int errorNumber = errno;
  diagnostics << "wafer-compile: target_module_write_failed: "
              << std::error_code(errorNumber, std::generic_category()).message()
              << "\n";
  return true;
#else
  (void)source;
  (void)destination;
  diagnostics << "wafer-compile: target_module_write_failed: atomic "
                 "no-replace directory rename is unsupported on this host\n";
  return true;
#endif
}

} // namespace

} // namespace wafer::compiler::detail

namespace wafer::compiler {

llvm::Expected<LinkedTargetModules> detail::linkTargetLLVMModulesImpl(
    const TargetLLVMModules &targetLLVMModules, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics,
    detail::ProfileCaptureKind profileCapture) {
  if (targetLLVMModules.getModules().size() !=
      static_cast<size_t>(
          targetLLVMModules.getExecutionConfig().getTileCount()))
    return detail::fail(diagnostics, "target LLVM module Tile domain is "
                                     "incomplete");
  if (llvm::Error error =
          detail::validateRuntimeLaunchContractDomain(targetLLVMModules))
    return detail::fail(diagnostics,
                        "runtime launch contract verification failed: " +
                            llvm::toString(std::move(error)));
  if (outputDirectory.empty())
    return detail::fail(diagnostics,
                        "target module directory must not be empty");
  if (detail::targetModulePathEntryExists(outputDirectory))
    return detail::fail(diagnostics,
                        "refusing to replace existing target module directory");
  if (!detail::isExecutableTargetFile(toolchain.getPythonExecutable()))
    return detail::fail(diagnostics,
                        "configured Python executable is not executable");
  if (!detail::isRegularTargetFile(toolchain.getDeviceLinkerScript()))
    return detail::fail(
        diagnostics, "configured device linker script is not a regular file");
  if (!detail::isExecutableTargetFile(toolchain.getLLVMClangXX()))
    return detail::fail(diagnostics,
                        "configured LLVM clang++ is not executable");
  if (!detail::isDirectory(toolchain.getTx8DepsRoot()))
    return detail::fail(diagnostics,
                        "configured TX8 dependency root is not a directory");
  if (!detail::isDirectory(toolchain.getWaferIncludeDir()))
    return detail::fail(
        diagnostics, "configured Wafer include directory is not a directory");
  if (!detail::isRegularTargetFile(toolchain.getWaferCrtSource()))
    return detail::fail(diagnostics,
                        "configured Wafer CRT source is not a regular file");
  if (!detail::isDirectory(toolchain.getWaferCrtIncludeDir()))
    return detail::fail(diagnostics,
                        "configured Wafer CRT include directory is not a "
                        "directory");

  llvm::SmallString<256> outputParent(outputDirectory);
  llvm::sys::path::remove_filename(outputParent);
  if (outputParent.empty())
    outputParent = ".";
  if (std::error_code error = llvm::sys::fs::create_directories(outputParent))
    return detail::fail(diagnostics,
                        "failed to create target module parent directory: " +
                            error.message());
  llvm::SmallString<256> stagingPrefix(outputParent);
  llvm::sys::path::append(stagingPrefix, ".wafer-target-modules-staging");
  llvm::SmallString<256> stagingRoot;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(stagingPrefix, stagingRoot))
    return detail::fail(diagnostics,
                        "failed to create temporary target module directory: " +
                            error.message());
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(stagingRoot); });

  llvm::SmallString<256> modulesDirectory(stagingRoot);
  llvm::sys::path::append(modulesDirectory, "modules");
  llvm::SmallString<256> workDirectory(stagingRoot);
  llvm::sys::path::append(workDirectory, "work");
  if (std::error_code error =
          llvm::sys::fs::create_directories(modulesDirectory))
    return detail::fail(diagnostics,
                        "failed to create staged module directory: " +
                            error.message());
  if (std::error_code error = llvm::sys::fs::create_directories(workDirectory))
    return detail::fail(diagnostics,
                        "failed to create target work directory: " +
                            error.message());

  std::set<int64_t> tileIds;
  std::set<int64_t> launchSlotIds;
  std::optional<CardId> cardId;
  for (const TargetLLVMModule &targetLLVMModule :
       targetLLVMModules.getModules()) {
    if (!cardId)
      cardId = targetLLVMModule.getCardId();
    if (targetLLVMModule.getCardId() != *cardId ||
        targetLLVMModule.getCardId().getValue() < 0 ||
        targetLLVMModule.getTileId().getValue() < 0 ||
        targetLLVMModule.getLaunchSlotId().getValue() < 0 ||
        !tileIds.insert(targetLLVMModule.getTileId().getValue()).second ||
        !launchSlotIds.insert(targetLLVMModule.getLaunchSlotId().getValue())
             .second)
      return detail::fail(
          diagnostics,
          "target LLVM modules have invalid or duplicate physical identity");
    if (llvm::Error error = detail::verifyProfileCaptureTileEntryArguments(
            targetLLVMModule.getTileEntryArguments(), profileCapture))
      return detail::fail(diagnostics,
                          "target LLVM profiler slot verification failed: " +
                              llvm::toString(std::move(error)));
    if (llvm::Error error = detail::verifyProfileTargetModuleInstrumentation(
            targetLLVMModule.getModule(), targetLLVMModule.getEntrySymbol(),
            profileCapture))
      return detail::fail(
          diagnostics,
          "target LLVM profiler instrumentation verification failed: " +
              llvm::toString(std::move(error)));
  }
  for (int64_t launchSlot = 0;
       launchSlot < static_cast<int64_t>(targetLLVMModules.getModules().size());
       ++launchSlot)
    if (launchSlotIds.count(launchSlot) == 0)
      return detail::fail(
          diagnostics,
          "target LLVM module launch-slot domain is not dense and canonical");

  const RuntimeLaunchContract &runtimeLaunchContract =
      targetLLVMModules.getRuntimeLaunchContract();
  const bool hasPrepare = llvm::is_contained(runtimeLaunchContract.getPhases(),
                                             RuntimeLaunchPhaseRole::Prepare);

  auto makeExports = [&](llvm::StringRef mainSymbol) {
    std::vector<VerifiedTargetExport> exports;
    if (hasPrepare)
      exports.push_back(LinkedTargetModulesBuilder::makeExport(
          TargetExportRole::Prepare, detail::kKernelPrepareExportSymbol));
    exports.push_back(LinkedTargetModulesBuilder::makeExport(
        TargetExportRole::Main, mainSymbol));
    return exports;
  };

  auto linkAndReadback =
      [&](const llvm::Module &source, llvm::StringRef entrySymbol,
          llvm::ArrayRef<TileEntryArgument> slots, LaunchSlotId launchSlotId,
          llvm::StringRef workStem, llvm::StringRef modulePath,
          llvm::ArrayRef<VerifiedTargetExport> exports,
          TargetIdentityId targetIdentity, llvm::StringRef expectedFormat)
      -> llvm::Expected<detail::TargetModuleReadback> {
    llvm::SmallString<256> llvmIRPath(workDirectory);
    llvm::sys::path::append(llvmIRPath, workStem + ".ll");
    llvm::SmallString<256> objectPath(workDirectory);
    llvm::sys::path::append(objectPath, workStem + ".o");
    llvm::SmallString<256> crtObjectPath(workDirectory);
    llvm::sys::path::append(crtObjectPath, workStem + ".wafer_crt.o");
    (void)entrySymbol;
    (void)slots;
    (void)launchSlotId;
    llvm::Error writeError = detail::writeTargetLLVMIR(source, llvmIRPath);
    if (writeError)
      return std::move(writeError);
    if (llvm::Error error = detail::runDeviceLink(
            toolchain, llvmIRPath, modulePath, objectPath, crtObjectPath,
            runtimeLaunchContract, profileCapture))
      return std::move(error);
    llvm::Expected<detail::TargetModuleReadback> readback =
        detail::verifyTargetModule(modulePath, exports, targetIdentity,
                                   runtimeLaunchContract);
    if (!readback)
      return readback.takeError();
    if (readback->moduleFormat != expectedFormat)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target module format readback does not match the prepared target "
          "profile");
    return readback;
  };

  std::vector<VerifiedTargetModule> modules;
  modules.reserve(1);
  std::vector<VerifiedTargetTileInterface> tileInterfaces;
  tileInterfaces.reserve(targetLLVMModules.getModules().size());

  {
    const TargetLLVMModule &first = targetLLVMModules.getModules().front();
    llvm::Expected<detail::OwnedTargetLLVMModule> aggregate =
        detail::buildKernelAggregateTargetModule(targetLLVMModules);
    if (!aggregate)
      return detail::fail(diagnostics,
                          "target_module_verification_failed: " +
                              llvm::toString(aggregate.takeError()));
    const std::string stem = "module_00000";
    llvm::SmallString<256> modulePath(modulesDirectory);
    llvm::sys::path::append(modulePath, stem + ".so");
    std::vector<VerifiedTargetExport> exports =
        makeExports(first.getEntrySymbol());
    llvm::Expected<detail::TargetModuleReadback> readback = linkAndReadback(
        *aggregate->module, first.getEntrySymbol(),
        first.getTileEntryArguments(), LaunchSlotId(0), stem, modulePath,
        exports, first.getTargetIdentityId(), first.getModuleFormat());
    if (!readback)
      return detail::fail(diagnostics,
                          "target_module_verification_failed: " +
                              llvm::toString(readback.takeError()));
    const TargetModuleId moduleId(0);
    modules.push_back(LinkedTargetModulesBuilder::makeModule(
        moduleId, "modules/module_00000.so", readback->contentDigest,
        first.getTargetIdentityId(), first.getKernelRuntimeABIId(),
        readback->moduleFormat, std::move(exports)));
    for (const TargetLLVMModule &tile : targetLLVMModules.getModules())
      tileInterfaces.push_back(LinkedTargetModulesBuilder::makeTileInterface(
          tile.getCardId(), tile.getTileId(), tile.getLaunchSlotId(), moduleId,
          tile.getTileEntryArguments()));
  }

  if (std::error_code error = llvm::sys::fs::remove_directories(workDirectory))
    return detail::fail(diagnostics,
                        "failed to remove target work directory: " +
                            error.message());
  if (detail::renameDirectoryNoReplace(stagingRoot, outputDirectory,
                                       diagnostics))
    return llvm::createStringError(llvm::errc::io_error,
                                   "failed to write target module directory");
  cleanup.release();
  return LinkedTargetModulesBuilder::makeModules(
      outputDirectory, targetLLVMModules.getExecutionConfig(),
      targetLLVMModules.getRuntimeLaunchContract(), std::move(modules),
      std::move(tileInterfaces));
}

llvm::Expected<LinkedTargetModules> linkTargetLLVMModules(
    const TargetLLVMModules &targetLLVMModules, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics) {
  return detail::linkTargetLLVMModulesImpl(targetLLVMModules, outputDirectory,
                                           toolchain, diagnostics,
                                           detail::ProfileCaptureKind::None);
}

} // namespace wafer::compiler
