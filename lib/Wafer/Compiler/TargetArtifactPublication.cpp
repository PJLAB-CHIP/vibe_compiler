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
compileTargetLLVMModuleBundleToTargetArtifacts(
    const TargetLLVMModuleBundle &targetLLVMModules,
    llvm::StringRef outputDirectory, const TargetToolchain &toolchain,
    llvm::raw_ostream &diagnostics) {
  if (targetLLVMModules.getModules().size() !=
      static_cast<size_t>(
          targetLLVMModules.getExecutionConfig().getRankCount()))
    return detail::fail(diagnostics,
                        "target LLVM bundle rank domain is incomplete");
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

  std::vector<VerifiedTargetModule> modules;
  modules.reserve(targetLLVMModules.getModules().size());
  for (auto [expectedRank, targetLLVMModule] :
       llvm::enumerate(targetLLVMModules.getModules())) {
    if (targetLLVMModule.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return detail::fail(diagnostics,
                          "target LLVM bundle rank domain is not canonical");
    std::string stem = (llvm::formatv("rank_{0:D5}", expectedRank)).str();
    llvm::SmallString<256> llvmIRPath(workDirectory);
    llvm::sys::path::append(llvmIRPath, stem + ".ll");
    llvm::SmallString<256> objectPath(workDirectory);
    llvm::sys::path::append(objectPath, stem + ".o");
    llvm::SmallString<256> crtObjectPath(workDirectory);
    llvm::sys::path::append(crtObjectPath, stem + ".wafer_crt.o");
    llvm::SmallString<256> modulePath(modulesDirectory);
    llvm::sys::path::append(modulePath, stem + ".so");
    if (llvm::Error error = detail::writeLLVMIR(
            targetLLVMModule.getModule(), targetLLVMModule.getEntrySymbol(),
            targetLLVMModule.getKernelABISlots().size(), llvmIRPath))
      return detail::fail(diagnostics, "target_module_verification_failed: " +
                                           llvm::toString(std::move(error)));
    if (llvm::Error error = detail::runDeviceLink(
            toolchain, llvmIRPath, modulePath, objectPath, crtObjectPath))
      return detail::fail(diagnostics, "target_module_verification_failed: " +
                                           llvm::toString(std::move(error)));
    llvm::Expected<detail::TargetModuleReadback> moduleReadback =
        detail::verifyTargetModule(modulePath,
                                   targetLLVMModule.getEntrySymbol(),
                                   targetLLVMModule.getTargetProfileId());
    if (!moduleReadback)
      return detail::fail(diagnostics,
                          "target_module_verification_failed: " +
                              llvm::toString(moduleReadback.takeError()));
    if (moduleReadback->moduleFormat != targetLLVMModule.getModuleFormat())
      return detail::fail(
          diagnostics, "target module format readback does not match prepared "
                       "target profile for logical rank " +
                           std::to_string(expectedRank));
    std::string relativePath = (llvm::Twine("modules/") + stem + ".so").str();
    modules.push_back(TargetArtifactBundleBuilder::makeModule(
        expectedRank, targetLLVMModule.getEntrySymbol(), relativePath,
        moduleReadback->contentDigest, targetLLVMModule.getTargetProfileId(),
        targetLLVMModule.getTargetIdentityId(),
        targetLLVMModule.getKernelRuntimeABIId(), moduleReadback->moduleFormat,
        targetLLVMModule.getKernelABISlots()));
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
      std::move(modules));
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
