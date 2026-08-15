//===- TargetLLVMModule.cpp - Target LLVM module translation -----------===//

#include "TargetCodeGenInternal.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <utility>

namespace wafer::compiler {

TargetLLVMModule::TargetLLVMModule(
    CardId cardId, TileId tileId,
    LaunchSlotId launchSlotId, llvm::StringRef entrySymbol,
    TargetIdentityId targetIdentity, KernelRuntimeABIId kernelRuntimeABI,
    llvm::StringRef moduleFormat, std::vector<KernelABISlot> kernelABISlots,
    std::unique_ptr<llvm::LLVMContext> context,
    std::unique_ptr<llvm::Module> module)
    : cardId(cardId), tileId(tileId),
      launchSlotId(launchSlotId), entrySymbol(entrySymbol.str()),
      targetIdentity(targetIdentity), kernelRuntimeABI(kernelRuntimeABI),
      moduleFormat(moduleFormat.str()),
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

TargetLLVMModules::~TargetLLVMModules() = default;
TargetLLVMModules::TargetLLVMModules(TargetLLVMModules &&) = default;
TargetLLVMModules &TargetLLVMModules::operator=(TargetLLVMModules &&) = default;

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

llvm::Expected<TargetLLVMModules>
compileCardExecutableToTargetLLVMModules(
    const CardExecutable &cardExecutable,
    llvm::raw_ostream &diagnostics) {
  return detail::compileCardExecutableToTargetLLVMModulesImpl(
      cardExecutable, diagnostics, std::nullopt);
}

} // namespace wafer::compiler
