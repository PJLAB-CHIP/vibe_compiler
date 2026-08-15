//===- TargetDeviceLink.cpp - Target LLVM emission and device link -------===//

#include "TargetCodeGenInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cstdint>
#include <limits>
#include <string>
#include <system_error>

namespace wafer::compiler::detail {

namespace {

constexpr int64_t kTx81TileCount = 16;
constexpr uint64_t kModelBootParamHeadBytes = 56;
constexpr uint64_t kModelDynInfoBytes = 72;

llvm::Error validateEntryInputs(llvm::Function &body,
                                llvm::ArrayRef<KernelABISlot> slots,
                                const RuntimeLaunchContract &launch,
                                LaunchSlotId launchSlotId,
                                int64_t tileCount) {
  if (body.isVarArg() || !body.getReturnType()->isVoidTy() ||
      body.arg_size() != slots.size() ||
      !llvm::all_of(body.args(), [](const llvm::Argument &argument) {
        return argument.getType()->isIntegerTy(64);
      }))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target device entry body does not match its typed ABI slots");
  for (auto [index, slot] : llvm::enumerate(slots))
    if (slot.ordinal != static_cast<int64_t>(index))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "target device entry slots are not in canonical ordinal order");

  if (tileCount != kTx81TileCount || launchSlotId.getValue() < 0 ||
      launchSlotId.getValue() >= tileCount)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "multi-Tile target device entry requires the complete 16-Tile "
        "launch-slot domain");
  if (!launch.getModel())
    return llvm::Error::success();
  for (const KernelABISlot &slot : slots)
    if ((slot.role != KernelABISlotRole::UserInput &&
         slot.role != KernelABISlotRole::Output) ||
        slot.dtype != "f32" || slot.shape.empty() || slot.shape.size() > 6)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "model target device entry only supports rank-1..6 f32 user-input "
          "and output ABI slots");
  return llvm::Error::success();
}

void retainModelDynamicExport(llvm::Module &module, llvm::Function &entry,
                              llvm::StringRef entrySymbol) {
  llvm::LLVMContext &context = module.getContext();
  llvm::Constant *nameData =
      llvm::ConstantDataArray::getString(context, entrySymbol, true);
  auto *name = new llvm::GlobalVariable(
      module, nameData->getType(), /*isConstant=*/true,
      llvm::GlobalValue::ExternalLinkage, nameData, "__wafer_model_entry_name");
  name->setSection(".rodata.name");
  name->setAlignment(llvm::Align(1));

  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Constant *zero = llvm::ConstantInt::get(i64, 0);
  llvm::SmallVector<llvm::Constant *, 2> nameIndices = {zero, zero};
  llvm::Constant *nameAddress = llvm::ConstantExpr::getInBoundsGetElementPtr(
      nameData->getType(), name, nameIndices);
  llvm::Type *pointer = llvm::PointerType::get(context, /*AddressSpace=*/0);
  llvm::StructType *recordType = llvm::StructType::get(pointer, pointer);
  llvm::Constant *recordData =
      llvm::ConstantStruct::get(recordType, {&entry, nameAddress});
  auto *record =
      new llvm::GlobalVariable(module, recordType, /*isConstant=*/true,
                               llvm::GlobalValue::ExternalLinkage, recordData,
                               "__wafer_model_export_record");
  record->setSection("ExportedDYNSYMTab");
  record->setAlignment(llvm::Align(8));

  // llvm.used is a link-time retention contract. The section must survive the
  // device linker's --gc-sections because the vendor graph loader discovers
  // this record by section name rather than through an ELF reference.
  llvm::appendToUsed(module, {name, record});
}

llvm::Expected<llvm::SmallVector<uint64_t, 16>>
getModelDescriptorIndices(llvm::ArrayRef<KernelABISlot> slots,
                          LaunchSlotId launchSlotId,
                          int64_t tileCountValue) {
  uint64_t inputCount = 0;
  uint64_t outputCount = 0;
  for (const KernelABISlot &slot : slots) {
    switch (slot.role) {
    case KernelABISlotRole::UserInput:
      ++inputCount;
      break;
    case KernelABISlotRole::Output:
      ++outputCount;
      break;
    case KernelABISlotRole::Parameter:
    case KernelABISlotRole::Constant:
    case KernelABISlotRole::Workspace:
    case KernelABISlotRole::TransportStatus:
      llvm_unreachable("model slot legality was checked before mapping");
    }
  }
  uint64_t perTileCount = inputCount + outputCount;
  if (perTileCount >
      (std::numeric_limits<uint64_t>::max() - kModelBootParamHeadBytes) /
          kModelDynInfoBytes / static_cast<uint64_t>(tileCountValue))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "model target device entry descriptor layout overflows uint64");

  const uint64_t launchSlot = static_cast<uint64_t>(launchSlotId.getValue());
  const uint64_t tileCount = static_cast<uint64_t>(tileCountValue);
  const uint64_t outputBase = tileCount * inputCount;
  uint64_t nextInput = 0;
  uint64_t nextOutput = 0;
  llvm::SmallVector<uint64_t, 16> indices;
  indices.reserve(slots.size());
  for (const KernelABISlot &slot : slots) {
    switch (slot.role) {
    case KernelABISlotRole::UserInput:
      indices.push_back(launchSlot * inputCount + nextInput++);
      break;
    case KernelABISlotRole::Output:
      indices.push_back(outputBase + launchSlot * outputCount + nextOutput++);
      break;
    case KernelABISlotRole::Parameter:
    case KernelABISlotRole::Constant:
    case KernelABISlotRole::Workspace:
    case KernelABISlotRole::TransportStatus:
      llvm_unreachable("model slot legality was checked before mapping");
    }
  }
  return indices;
}

llvm::Expected<std::unique_ptr<llvm::Module>> materializeDeviceEntryABI(
    const llvm::Module &module, llvm::StringRef entrySymbol,
    llvm::ArrayRef<KernelABISlot> slots,
    const RuntimeLaunchContract &runtimeLaunchContract,
    LaunchSlotId launchSlotId, int64_t tileCount) {
  std::unique_ptr<llvm::Module> deviceModule = llvm::CloneModule(module);
  llvm::Function *body = deviceModule->getFunction(entrySymbol);
  if (!body || body->isDeclaration())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target device entry body is missing");
  if (llvm::Error error = validateEntryInputs(
          *body, slots, runtimeLaunchContract, launchSlotId, tileCount))
    return std::move(error);
  llvm::LLVMContext &context = deviceModule->getContext();
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);

  body->setName("wafer_device_entry_body");
  body->setLinkage(llvm::GlobalValue::InternalLinkage);
  llvm::FunctionType *entryType = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context),
      {llvm::PointerType::get(context, /*AddressSpace=*/0)},
      /*isVarArg=*/false);
  llvm::Function *entry =
      llvm::Function::Create(entryType, llvm::GlobalValue::ExternalLinkage,
                             entrySymbol, *deviceModule);
  const bool modelLaunch = runtimeLaunchContract.getModel() != nullptr;
  entry->getArg(0)->setName(modelLaunch ? "boot_parameter" : "slots");
  llvm::BasicBlock *block = llvm::BasicBlock::Create(context, "entry", entry);
  llvm::IRBuilder<> builder(block);
  llvm::SmallVector<llvm::Value *, 16> arguments;
  arguments.reserve(slots.size());
  if (modelLaunch) {
    llvm::Expected<llvm::SmallVector<uint64_t, 16>> descriptorIndices =
        getModelDescriptorIndices(slots, launchSlotId, tileCount);
    if (!descriptorIndices)
      return descriptorIndices.takeError();
    llvm::Type *i8 = llvm::Type::getInt8Ty(context);
    for (uint64_t descriptorIndex : *descriptorIndices) {
      uint64_t byteOffset =
          kModelBootParamHeadBytes + descriptorIndex * kModelDynInfoBytes;
      llvm::Value *descriptor = builder.CreateInBoundsGEP(
          i8, entry->getArg(0), llvm::ConstantInt::get(i64, byteOffset));
      llvm::LoadInst *address = builder.CreateLoad(i64, descriptor);
      address->setAlignment(llvm::Align(8));
      arguments.push_back(address);
    }
  } else {
    for (size_t index = 0; index < slots.size(); ++index) {
      llvm::Value *slot = builder.CreateInBoundsGEP(
          i64, entry->getArg(0), llvm::ConstantInt::get(i64, index));
      arguments.push_back(builder.CreateLoad(i64, slot));
    }
  }
  builder.CreateCall(body, arguments);
  builder.CreateRetVoid();
  if (modelLaunch)
    retainModelDynamicExport(*deviceModule, *entry, entrySymbol);
  if (llvm::verifyModule(*deviceModule))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target device entry ABI is invalid");
  return deviceModule;
}

} // namespace

llvm::Error writeLLVMIR(const llvm::Module &module, llvm::StringRef entrySymbol,
                        llvm::ArrayRef<KernelABISlot> slots,
                        const RuntimeLaunchContract &runtimeLaunchContract,
                        LaunchSlotId launchSlotId, int64_t tileCount,
                        llvm::StringRef path,
                        ProfileCaptureKind profileCapture) {
  if (runtimeLaunchContract.getKernel())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel target modules must use complete-Tile aggregation");
  if (llvm::Error error =
          verifyProfileCaptureKernelABISlots(slots, profileCapture))
    return std::move(error);
  llvm::Expected<std::unique_ptr<llvm::Module>> deviceModule =
      materializeDeviceEntryABI(module, entrySymbol, slots,
                                runtimeLaunchContract, launchSlotId,
                                tileCount);
  if (!deviceModule)
    return deviceModule.takeError();
  return writeTargetLLVMIR(**deviceModule, path);
}

llvm::Error writeTargetLLVMIR(const llvm::Module &module,
                              llvm::StringRef path) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "failed to open target LLVM IR");
  module.print(output, nullptr);
  output.close();
  if (output.has_error())
    return llvm::createStringError(llvm::errc::io_error,
                                   "failed to write target LLVM IR");
  return llvm::Error::success();
}

llvm::Error runDeviceLink(const TargetToolchain &toolchain,
                          llvm::StringRef llvmIR, llvm::StringRef module,
                          llvm::StringRef object, llvm::StringRef crtObject,
                          const RuntimeLaunchContract &runtimeLaunchContract,
                          ProfileCaptureKind profileCapture) {
  std::string python = toolchain.getPythonExecutable().str();
  std::string script = toolchain.getDeviceLinkerScript().str();
  std::string clangXX = toolchain.getLLVMClangXX().str();
  std::string llvmIRStorage = llvmIR.str();
  std::string moduleStorage = module.str();
  std::string objectStorage = object.str();
  std::string crtObjectStorage = crtObject.str();
  std::string loaderABI = "tx8-kcore-loader-v1";
  if (const auto *kernel = runtimeLaunchContract.getKernel()) {
    if (kernel->form == KernelLaunchForm::Grid)
      loaderABI = "tx8-kcore-loader-grid-v1";
    else if (kernel->form == KernelLaunchForm::Cluster)
      loaderABI = "tx8-kcore-loader-cluster-v1";
  }
  llvm::SmallVector<llvm::StringRef, 18> arguments = {python,
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
                                                      crtObjectStorage,
                                                      "--loader-abi",
                                                      loaderABI};
  std::string captureStorage;
  if (profileCapture != ProfileCaptureKind::None) {
    captureStorage = stringifyProfileCaptureKind(profileCapture).str();
    arguments.push_back("--profile-capture");
    arguments.push_back(captureStorage);
  }
  int exitCode = llvm::sys::ExecuteAndWait(python, arguments);
  if (exitCode == 0)
    return llvm::Error::success();
  return llvm::createStringError(
      llvm::errc::io_error, "device link failed with exit code %d", exitCode);
}

} // namespace wafer::compiler::detail
