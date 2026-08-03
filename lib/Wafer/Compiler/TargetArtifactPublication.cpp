//===- TargetArtifactPublication.cpp - Atomic target artifact publication ===//

#include "TargetArtifactInternal.h"

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

bool haveSameKernelABISchema(llvm::ArrayRef<KernelABISlot> lhs,
                             llvm::ArrayRef<KernelABISlot> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left.ordinal != right.ordinal || left.role != right.role ||
        left.resourceIndex != right.resourceIndex ||
        left.dtype != right.dtype || left.layout != right.layout ||
        left.shape != right.shape || left.byteSize != right.byteSize ||
        left.alignment != right.alignment)
      return false;
  return true;
}

llvm::Error validateRuntimeLaunchContractDomain(
    const TargetLLVMModuleBundle &targetLLVMModules) {
  const ExecutionConfig &config = targetLLVMModules.getExecutionConfig();
  const RuntimeLaunchContract &launch =
      targetLLVMModules.getRuntimeLaunchContract();
  if (launch.getKind() != config.getRuntimeLaunchKind() ||
      !isRuntimeLaunchContractCompatible(launch, config.getTargetProfileId()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "runtime launch contract does not match the execution configuration");

  const KernelRuntimeLaunchContract *kernel = launch.getKernel();
  if (kernel && kernel->form == KernelLaunchForm::PerRank) {
    if (config.getRankCount() != 1 ||
        targetLLVMModules.getModules().size() != 1)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "rank-local kernel launch requires the rank-one target domain");
    for (const TargetLLVMModule &module : targetLLVMModules.getModules())
      if (module.getKernelABISlots().size() >
          kTx81KernelArgumentBytesMax / sizeof(uint64_t))
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "per-rank kernel argument block exceeds the qualified V5.6 "
            "packet limit");
    return llvm::Error::success();
  }
  if (config.getRankCount() != 16 ||
      targetLLVMModules.getModules().size() != 16)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "multi-tile runtime launch requires the complete 16-rank domain");

  const TargetLLVMModule &first = targetLLVMModules.getModules().front();
  if (kernel) {
    if (first.getKernelABISlots().empty())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "multi-tile pointer-table launch requires at least one typed ABI "
          "slot");
    const uint64_t packetBytes = kernel->form == KernelLaunchForm::Cluster
                                     ? kTx81ClusterKernelArgumentBytesMax
                                     : kTx81KernelArgumentBytesMax;
    if (kernel->entryABI == KernelEntryABI::RankMajorPointerTableV1) {
      if (first.getKernelABISlots().size() >
          packetBytes / sizeof(uint64_t) /
              static_cast<uint64_t>(config.getRankCount()))
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "multi-tile rank-major argument table exceeds the qualified V5.6 "
            "packet limit");
    } else if (kernel->entryABI == KernelEntryABI::RankRowPointerTableV1) {
      if (static_cast<uint64_t>(config.getRankCount()) >
          packetBytes / sizeof(uint64_t))
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "multi-tile rank-row pointer table exceeds the qualified V5.6 "
            "packet limit");
    } else {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "multi-tile kernel launch has an incompatible entry ABI");
    }
    const size_t transportStatusSlots = llvm::count_if(
        first.getKernelABISlots(), [](const KernelABISlot &slot) {
          return slot.role == KernelABISlotRole::TransportStatus;
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
          "multi-tile runtime launch requires one shared entry symbol");
    if (!haveSameKernelABISchema(module.getKernelABISlots(),
                                 first.getKernelABISlots()))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "multi-tile runtime launch requires identical all-rank slot "
          "schemas");
  }
  if (launch.getModel())
    for (const KernelABISlot &slot : first.getKernelABISlots())
      if ((slot.role != KernelABISlotRole::UserInput &&
           slot.role != KernelABISlotRole::Output) ||
          slot.dtype != "f32" || slot.shape.empty() || slot.shape.size() > 6)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "model runtime launch only supports rank-1..6 f32 user-input "
            "and output slots");
  return llvm::Error::success();
}

bool targetArtifactPathEntryExists(llvm::StringRef path) {
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
    const TargetLLVMModuleBundle &targetLLVMModules) {
  return validateRuntimeLaunchContractDomain(targetLLVMModules);
}

namespace {

bool isExecutableTargetFile(llvm::StringRef path) {
  return isRegularTargetFile(path) &&
         !llvm::sys::fs::access(path, llvm::sys::fs::AccessMode::Execute);
}

bool publishTargetArtifactDirectoryNoReplace(llvm::StringRef source,
                                             llvm::StringRef destination,
                                             llvm::raw_ostream &diagnostics) {
#ifdef __linux__
  std::string sourceStorage = source.str();
  std::string destinationStorage = destination.str();
  if (::syscall(SYS_renameat2, AT_FDCWD, sourceStorage.c_str(), AT_FDCWD,
                destinationStorage.c_str(), RENAME_NOREPLACE) == 0)
    return false;
  int errorNumber = errno;
  diagnostics << "wafer-compile: target_publication_failed: "
              << std::error_code(errorNumber, std::generic_category()).message()
              << "\n";
  return true;
#else
  (void)source;
  (void)destination;
  diagnostics << "wafer-compile: target_publication_failed: atomic no-replace "
                 "directory publication is unsupported on this host\n";
  return true;
#endif
}

} // namespace

} // namespace wafer::compiler::detail

namespace wafer::compiler {

llvm::Expected<TargetArtifactBundle>
detail::compileTargetLLVMModuleBundleToTargetArtifactsImpl(
    const TargetLLVMModuleBundle &targetLLVMModules,
    llvm::StringRef outputDirectory, const TargetToolchain &toolchain,
    llvm::raw_ostream &diagnostics, detail::ProfileCaptureKind profileCapture) {
  if (targetLLVMModules.getModules().size() !=
      static_cast<size_t>(
          targetLLVMModules.getExecutionConfig().getRankCount()))
    return detail::fail(diagnostics,
                        "target LLVM bundle rank domain is incomplete");
  if (llvm::Error error =
          detail::validateRuntimeLaunchContractDomain(targetLLVMModules))
    return detail::fail(diagnostics,
                        "runtime launch contract verification failed: " +
                            llvm::toString(std::move(error)));
  if (outputDirectory.empty())
    return detail::fail(diagnostics,
                        "target artifact directory must not be empty");
  if (detail::targetArtifactPathEntryExists(outputDirectory))
    return detail::fail(
        diagnostics, "refusing to replace existing target artifact directory");
  if (!detail::isExecutableTargetFile(toolchain.getPythonExecutable()))
    return detail::fail(diagnostics,
                        "configured Python executable is not executable");
  if (!detail::isRegularTargetFile(toolchain.getDeviceLinkerScript()))
    return detail::fail(
        diagnostics, "configured device linker script is not a regular file");
  if (!detail::isExecutableTargetFile(toolchain.getLLVMClangXX()))
    return detail::fail(diagnostics,
                        "configured LLVM clang++ is not executable");

  llvm::SmallString<256> outputParent(outputDirectory);
  llvm::sys::path::remove_filename(outputParent);
  if (outputParent.empty())
    outputParent = ".";
  if (std::error_code error = llvm::sys::fs::create_directories(outputParent))
    return detail::fail(diagnostics,
                        "failed to create target artifact parent: " +
                            error.message());
  llvm::SmallString<256> stagingPrefix(outputParent);
  llvm::sys::path::append(stagingPrefix, ".wafer-target-artifacts-staging");
  llvm::SmallString<256> stagingRoot;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(stagingPrefix, stagingRoot))
    return detail::fail(diagnostics,
                        "failed to create target artifact staging: " +
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

  for (auto [expectedRank, targetLLVMModule] :
       llvm::enumerate(targetLLVMModules.getModules()))
    if (targetLLVMModule.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return detail::fail(diagnostics,
                          "target LLVM bundle rank domain is not canonical");
    else if (llvm::Error error = detail::verifyProfileCaptureKernelABISlots(
                 targetLLVMModule.getKernelABISlots(), profileCapture))
      return detail::fail(diagnostics,
                          "target LLVM profiler slot verification failed: " +
                              llvm::toString(std::move(error)));
    else if (llvm::Error error =
                 detail::verifyProfileTargetModuleInstrumentation(
                     targetLLVMModule.getModule(),
                     targetLLVMModule.getEntrySymbol(),
                     targetLLVMModule.getTargetProfileId(), profileCapture))
      return detail::fail(
          diagnostics,
          "target LLVM profiler instrumentation verification failed: " +
              llvm::toString(std::move(error)));

  const ExecutionConfig &config = targetLLVMModules.getExecutionConfig();
  const RuntimeLaunchContract &runtimeLaunchContract =
      targetLLVMModules.getRuntimeLaunchContract();
  const KernelRuntimeLaunchContract *kernelLaunch =
      runtimeLaunchContract.getKernel();
  const bool isAggregateKernel =
      kernelLaunch && kernelLaunch->form != KernelLaunchForm::PerRank;
  const bool hasPrepare = llvm::is_contained(runtimeLaunchContract.getPhases(),
                                             RuntimeLaunchPhaseRole::Prepare);

  auto makeExports = [&](llvm::StringRef mainSymbol) {
    std::vector<VerifiedTargetExport> exports;
    if (hasPrepare)
      exports.push_back(TargetArtifactBundleBuilder::makeExport(
          TargetExportRole::Prepare, detail::kKernelPrepareExportSymbol));
    exports.push_back(TargetArtifactBundleBuilder::makeExport(
        TargetExportRole::Main, mainSymbol));
    return exports;
  };

  auto linkAndReadback =
      [&](const llvm::Module &source, llvm::StringRef entrySymbol,
          llvm::ArrayRef<KernelABISlot> slots, int64_t logicalRank,
          llvm::StringRef workStem, llvm::StringRef modulePath,
          llvm::ArrayRef<VerifiedTargetExport> exports,
          TargetProfileId targetProfile, llvm::StringRef expectedFormat,
          bool hasMaterializedEntryABI)
      -> llvm::Expected<detail::TargetModuleReadback> {
    llvm::SmallString<256> llvmIRPath(workDirectory);
    llvm::sys::path::append(llvmIRPath, workStem + ".ll");
    llvm::SmallString<256> objectPath(workDirectory);
    llvm::sys::path::append(objectPath, workStem + ".o");
    llvm::SmallString<256> crtObjectPath(workDirectory);
    llvm::sys::path::append(crtObjectPath, workStem + ".wafer_crt.o");
    llvm::Error writeError =
        hasMaterializedEntryABI
            ? detail::writeTargetLLVMIR(source, llvmIRPath)
            : detail::writeLLVMIR(source, entrySymbol, slots,
                                  runtimeLaunchContract, logicalRank,
                                  config.getRankCount(), llvmIRPath,
                                  profileCapture);
    if (writeError)
      return std::move(writeError);
    if (llvm::Error error = detail::runDeviceLink(
            toolchain, llvmIRPath, modulePath, objectPath, crtObjectPath,
            runtimeLaunchContract, profileCapture))
      return std::move(error);
    llvm::Expected<detail::TargetModuleReadback> readback =
        detail::verifyTargetModule(modulePath, exports, targetProfile,
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
  modules.reserve(isAggregateKernel ? 1
                                    : targetLLVMModules.getModules().size());
  std::vector<VerifiedTargetRankInterface> rankInterfaces;
  rankInterfaces.reserve(targetLLVMModules.getModules().size());

  if (isAggregateKernel) {
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
        *aggregate->module, first.getEntrySymbol(), first.getKernelABISlots(),
        /*logicalRank=*/0, stem, modulePath, exports,
        first.getTargetProfileId(), first.getModuleFormat(),
        /*hasMaterializedEntryABI=*/true);
    if (!readback)
      return detail::fail(diagnostics,
                          "target_module_verification_failed: " +
                              llvm::toString(readback.takeError()));
    const TargetArtifactModuleId moduleId(0);
    modules.push_back(TargetArtifactBundleBuilder::makeModule(
        moduleId, "modules/module_00000.so", readback->contentDigest,
        first.getTargetProfileId(), first.getTargetIdentityId(),
        first.getKernelRuntimeABIId(), readback->moduleFormat,
        std::move(exports)));
    for (const TargetLLVMModule &rank : targetLLVMModules.getModules())
      rankInterfaces.push_back(TargetArtifactBundleBuilder::makeRankInterface(
          rank.getLogicalRank(), moduleId, rank.getKernelABISlots()));
  } else {
    for (auto [expectedRank, targetLLVMModule] :
         llvm::enumerate(targetLLVMModules.getModules())) {
      const TargetArtifactModuleId moduleId(expectedRank);
      const std::string stem =
          llvm::formatv("module_{0:D5}", expectedRank).str();
      llvm::SmallString<256> modulePath(modulesDirectory);
      llvm::sys::path::append(modulePath, stem + ".so");
      std::vector<VerifiedTargetExport> exports =
          makeExports(targetLLVMModule.getEntrySymbol());
      llvm::Expected<detail::TargetModuleReadback> readback = linkAndReadback(
          targetLLVMModule.getModule(), targetLLVMModule.getEntrySymbol(),
          targetLLVMModule.getKernelABISlots(),
          targetLLVMModule.getLogicalRank(), stem, modulePath, exports,
          targetLLVMModule.getTargetProfileId(),
          targetLLVMModule.getModuleFormat(),
          /*hasMaterializedEntryABI=*/false);
      if (!readback)
        return detail::fail(diagnostics,
                            "target_module_verification_failed: " +
                                llvm::toString(readback.takeError()));
      const std::string relativePath =
          (llvm::Twine("modules/") + stem + ".so").str();
      modules.push_back(TargetArtifactBundleBuilder::makeModule(
          moduleId, relativePath, readback->contentDigest,
          targetLLVMModule.getTargetProfileId(),
          targetLLVMModule.getTargetIdentityId(),
          targetLLVMModule.getKernelRuntimeABIId(), readback->moduleFormat,
          std::move(exports)));
      rankInterfaces.push_back(TargetArtifactBundleBuilder::makeRankInterface(
          targetLLVMModule.getLogicalRank(), moduleId,
          targetLLVMModule.getKernelABISlots()));
    }
  }

  if (std::error_code error = llvm::sys::fs::remove_directories(workDirectory))
    return detail::fail(diagnostics,
                        "failed to remove target work directory: " +
                            error.message());
  if (detail::publishTargetArtifactDirectoryNoReplace(
          stagingRoot, outputDirectory, diagnostics))
    return llvm::createStringError(llvm::errc::io_error,
                                   "target artifact publication failed");
  cleanup.release();
  return TargetArtifactBundleBuilder::makeBundle(
      outputDirectory, targetLLVMModules.getExecutionConfig(),
      targetLLVMModules.getRuntimeLaunchContract(), std::move(modules),
      std::move(rankInterfaces));
}

llvm::Expected<TargetArtifactBundle>
compileTargetLLVMModuleBundleToTargetArtifacts(
    const TargetLLVMModuleBundle &targetLLVMModules,
    llvm::StringRef outputDirectory, const TargetToolchain &toolchain,
    llvm::raw_ostream &diagnostics) {
  return detail::compileTargetLLVMModuleBundleToTargetArtifactsImpl(
      targetLLVMModules, outputDirectory, toolchain, diagnostics,
      detail::ProfileCaptureKind::None);
}

namespace detail {

llvm::Expected<TargetArtifactBundle>
compileExecutableBundleToTargetArtifactsImpl(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank) {
  llvm::Expected<TargetLLVMModuleBundle> targetLLVMModules =
      compileExecutableBundleToTargetLLVMModulesImpl(
          executableBundle, diagnostics, failAfterLogicalRank);
  if (!targetLLVMModules)
    return targetLLVMModules.takeError();
  return compileTargetLLVMModuleBundleToTargetArtifacts(
      *targetLLVMModules, outputDirectory, toolchain, diagnostics);
}

} // namespace detail
} // namespace wafer::compiler
