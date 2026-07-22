//===- TargetDeviceLink.cpp - Target LLVM emission and device link -------===//

#include "TargetArtifactInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>

namespace wafer::compiler::detail {

namespace {

llvm::Expected<std::unique_ptr<llvm::Module>>
materializeDeviceEntryABI(const llvm::Module &module,
                          llvm::StringRef entrySymbol, size_t slotCount) {
  std::unique_ptr<llvm::Module> deviceModule = llvm::CloneModule(module);
  llvm::Function *body = deviceModule->getFunction(entrySymbol);
  if (!body || body->isDeclaration())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target device entry body is missing");
  llvm::LLVMContext &context = deviceModule->getContext();
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  if (body->isVarArg() || !body->getReturnType()->isVoidTy() ||
      body->arg_size() != slotCount ||
      !llvm::all_of(body->args(), [](const llvm::Argument &argument) {
        return argument.getType()->isIntegerTy(64);
      }))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target device entry body does not match its typed ABI slots");

  body->setName("wafer_device_entry_body");
  body->setLinkage(llvm::GlobalValue::InternalLinkage);
  llvm::FunctionType *entryType = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context),
      {llvm::PointerType::get(context, /*AddressSpace=*/0)},
      /*isVarArg=*/false);
  llvm::Function *entry = llvm::Function::Create(
      entryType, llvm::GlobalValue::ExternalLinkage, entrySymbol,
      *deviceModule);
  entry->getArg(0)->setName("slots");
  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(context, "entry", entry);
  llvm::IRBuilder<> builder(block);
  llvm::SmallVector<llvm::Value *, 16> arguments;
  arguments.reserve(slotCount);
  for (size_t index = 0; index < slotCount; ++index) {
    llvm::Value *slot = builder.CreateInBoundsGEP(
        i64, entry->getArg(0), llvm::ConstantInt::get(i64, index));
    arguments.push_back(builder.CreateLoad(i64, slot));
  }
  builder.CreateCall(body, arguments);
  builder.CreateRetVoid();
  if (llvm::verifyModule(*deviceModule))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target device entry ABI is invalid");
  return deviceModule;
}

} // namespace

llvm::Error writeLLVMIR(const llvm::Module &module,
                        llvm::StringRef entrySymbol, size_t slotCount,
                        llvm::StringRef path) {
  llvm::Expected<std::unique_ptr<llvm::Module>> deviceModule =
      materializeDeviceEntryABI(module, entrySymbol, slotCount);
  if (!deviceModule)
    return deviceModule.takeError();
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "failed to open target LLVM IR");
  (*deviceModule)->print(output, nullptr);
  output.close();
  if (output.has_error())
    return llvm::createStringError(llvm::errc::io_error,
                                   "failed to write target LLVM IR");
  return llvm::Error::success();
}

llvm::Error runDeviceLink(const TargetToolchain &toolchain,
                          llvm::StringRef llvmIR, llvm::StringRef module,
                          llvm::StringRef object, llvm::StringRef crtObject) {
  std::string python = toolchain.getPythonExecutable().str();
  std::string script = toolchain.getDeviceLinkerScript().str();
  std::string clangXX = toolchain.getLLVMClangXX().str();
  std::string llvmIRStorage = llvmIR.str();
  std::string moduleStorage = module.str();
  std::string objectStorage = object.str();
  std::string crtObjectStorage = crtObject.str();
  llvm::SmallVector<llvm::StringRef, 13> arguments = {python,
                                                      script,
                                                      "--llvm-ir",
                                                      llvmIRStorage,
                                                      "--llvm-clangxx",
                                                      clangXX,
                                                      "--output",
                                                      moduleStorage,
                                                      "--object-output",
                                                      objectStorage,
                                                      "--crt-object-output",
                                                      crtObjectStorage};
  int exitCode = llvm::sys::ExecuteAndWait(python, arguments);
  if (exitCode == 0)
    return llvm::Error::success();
  return llvm::createStringError(
      llvm::errc::io_error, "device link failed with exit code %d", exitCode);
}

} // namespace wafer::compiler::detail
