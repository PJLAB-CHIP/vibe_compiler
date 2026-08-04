//===- TargetArtifact.cpp - Typed target artifact facade ----------------===//

#include "TargetArtifactInternal.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>

namespace wafer::compiler {

TargetLLVMModule::TargetLLVMModule(int64_t logicalRank,
                                   llvm::StringRef entrySymbol,
                                   TargetIdentityId targetIdentity,
                                   KernelRuntimeABIId kernelRuntimeABI,
                                   llvm::StringRef moduleFormat,
                                   std::vector<KernelABISlot> kernelABISlots,
                                   std::unique_ptr<llvm::LLVMContext> context,
                                   std::unique_ptr<llvm::Module> module)
    : logicalRank(logicalRank), entrySymbol(entrySymbol.str()),
      targetIdentity(targetIdentity),
      kernelRuntimeABI(kernelRuntimeABI), moduleFormat(moduleFormat.str()),
      kernelABISlots(std::move(kernelABISlots)), context(std::move(context)),
      module(std::move(module)) {}

TargetLLVMModule::~TargetLLVMModule() = default;
TargetLLVMModule::TargetLLVMModule(TargetLLVMModule &&) = default;
TargetLLVMModule &TargetLLVMModule::operator=(TargetLLVMModule &&) = default;

llvm::StringRef TargetLLVMModule::getModuleIdentifier() const {
  return module->getModuleIdentifier();
}

llvm::StringRef TargetLLVMModule::getTargetTriple() const {
  return module->getTargetTriple();
}

const llvm::Module &TargetLLVMModule::getModule() const { return *module; }

TargetLLVMModuleBundle::~TargetLLVMModuleBundle() = default;
TargetLLVMModuleBundle::TargetLLVMModuleBundle(TargetLLVMModuleBundle &&) =
    default;
TargetLLVMModuleBundle &
TargetLLVMModuleBundle::operator=(TargetLLVMModuleBundle &&) = default;

llvm::Expected<TargetToolchain>
TargetToolchain::create(llvm::StringRef pythonExecutable,
                        llvm::StringRef deviceLinkerScript,
                        llvm::StringRef llvmClangXX) {
  if (pythonExecutable.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Python executable must not be empty");
  if (deviceLinkerScript.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "device linker script must not be empty");
  if (llvmClangXX.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "LLVM clang++ must not be empty");
  return TargetToolchain(pythonExecutable, deviceLinkerScript, llvmClangXX);
}

llvm::Expected<TargetArtifactBundle> compileExecutableBundleToTargetArtifacts(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics) {
  return detail::compileExecutableBundleToTargetArtifactsImpl(
      executableBundle, outputDirectory, toolchain, diagnostics, std::nullopt);
}

llvm::Expected<TargetLLVMModuleBundle>
compileExecutableBundleToTargetLLVMModules(
    const ExecutableBundle &executableBundle, llvm::raw_ostream &diagnostics) {
  return detail::compileExecutableBundleToTargetLLVMModulesImpl(
      executableBundle, diagnostics, std::nullopt);
}

} // namespace wafer::compiler
