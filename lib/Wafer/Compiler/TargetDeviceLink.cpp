//===- TargetDeviceLink.cpp - Target LLVM emission and device link -------===//

#include "TargetArtifactInternal.h"

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
                                TargetLaunchABIId launchABI,
                                int64_t logicalRank, int64_t rankCount) {
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

  if (launchABI == TargetLaunchABIId::perRankPointerBlockV1())
    return llvm::Error::success();
  if (rankCount != kTx81TileCount || logicalRank < 0 ||
      logicalRank >= rankCount)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "multi-tile target device entry requires the complete 16-rank domain");
  if (launchABI != TargetLaunchABIId::tx81ModelBootParamV1())
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
                          int64_t logicalRank, int64_t rankCount) {
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
  uint64_t perRankCount = inputCount + outputCount;
  if (perRankCount >
      (std::numeric_limits<uint64_t>::max() - kModelBootParamHeadBytes) /
          kModelDynInfoBytes / static_cast<uint64_t>(rankCount))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "model target device entry descriptor layout overflows uint64");

  const uint64_t rank = static_cast<uint64_t>(logicalRank);
  const uint64_t ranks = static_cast<uint64_t>(rankCount);
  const uint64_t outputBase = ranks * inputCount;
  uint64_t nextInput = 0;
  uint64_t nextOutput = 0;
  llvm::SmallVector<uint64_t, 16> indices;
  indices.reserve(slots.size());
  for (const KernelABISlot &slot : slots) {
    switch (slot.role) {
    case KernelABISlotRole::UserInput:
      indices.push_back(rank * inputCount + nextInput++);
      break;
    case KernelABISlotRole::Output:
      indices.push_back(outputBase + rank * outputCount + nextOutput++);
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
    llvm::ArrayRef<KernelABISlot> slots, TargetLaunchABIId targetLaunchABI,
    int64_t logicalRank, int64_t rankCount) {
  std::unique_ptr<llvm::Module> deviceModule = llvm::CloneModule(module);
  llvm::Function *body = deviceModule->getFunction(entrySymbol);
  if (!body || body->isDeclaration())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target device entry body is missing");
  if (llvm::Error error = validateEntryInputs(*body, slots, targetLaunchABI,
                                              logicalRank, rankCount))
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
  entry->getArg(0)->setName(targetLaunchABI ==
                                    TargetLaunchABIId::tx81ModelBootParamV1()
                                ? "boot_parameter"
                                : "slots");
  llvm::BasicBlock *block = llvm::BasicBlock::Create(context, "entry", entry);
  llvm::IRBuilder<> builder(block);
  llvm::SmallVector<llvm::Value *, 16> arguments;
  arguments.reserve(slots.size());
  if (targetLaunchABI == TargetLaunchABIId::tx81ModelBootParamV1()) {
    llvm::Expected<llvm::SmallVector<uint64_t, 16>> descriptorIndices =
        getModelDescriptorIndices(slots, logicalRank, rankCount);
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
    llvm::Value *rankOffset = llvm::ConstantInt::get(i64, 0);
    if (targetLaunchABI == TargetLaunchABIId::tx81KernelGridPointerTableV1()) {
      llvm::Type *i32 = llvm::Type::getInt32Ty(context);
      llvm::FunctionCallee getPid = deviceModule->getOrInsertFunction(
          "__get_pid", llvm::FunctionType::get(i32, {i32}, false));
      llvm::Value *pid =
          builder.CreateCall(getPid, llvm::ConstantInt::get(i32, 0), "pid.x");
      llvm::Value *pid64 = builder.CreateZExt(pid, i64, "pid.x.i64");
      rankOffset = builder.CreateMul(
          pid64, llvm::ConstantInt::get(i64, slots.size()), "rank.offset");
    }
    for (size_t index = 0; index < slots.size(); ++index) {
      llvm::Value *slotIndex = builder.CreateAdd(
          rankOffset, llvm::ConstantInt::get(i64, index), "slot.index");
      llvm::Value *slot =
          builder.CreateInBoundsGEP(i64, entry->getArg(0), slotIndex);
      arguments.push_back(builder.CreateLoad(i64, slot));
    }
  }
  builder.CreateCall(body, arguments);
  builder.CreateRetVoid();
  if (targetLaunchABI == TargetLaunchABIId::tx81ModelBootParamV1())
    retainModelDynamicExport(*deviceModule, *entry, entrySymbol);
  if (llvm::verifyModule(*deviceModule))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target device entry ABI is invalid");
  return deviceModule;
}

} // namespace

llvm::Error writeLLVMIR(const llvm::Module &module, llvm::StringRef entrySymbol,
                        llvm::ArrayRef<KernelABISlot> slots,
                        TargetLaunchABIId targetLaunchABI, int64_t logicalRank,
                        int64_t rankCount, llvm::StringRef path) {
  if (targetLaunchABI == TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "cluster target modules must use complete-rank aggregation");
  llvm::Expected<std::unique_ptr<llvm::Module>> deviceModule =
      materializeDeviceEntryABI(module, entrySymbol, slots, targetLaunchABI,
                                logicalRank, rankCount);
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
                          TargetLaunchABIId targetLaunchABI) {
  std::string python = toolchain.getPythonExecutable().str();
  std::string script = toolchain.getDeviceLinkerScript().str();
  std::string clangXX = toolchain.getLLVMClangXX().str();
  std::string llvmIRStorage = llvmIR.str();
  std::string moduleStorage = module.str();
  std::string objectStorage = object.str();
  std::string crtObjectStorage = crtObject.str();
  std::string loaderABI = "tx8-kcore-loader-v1";
  if (targetLaunchABI == TargetLaunchABIId::tx81KernelGridPointerTableV1())
    loaderABI = "tx8-kcore-loader-grid-v1";
  else if (targetLaunchABI ==
           TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1())
    loaderABI = "tx8-kcore-loader-cluster-v1";
  llvm::SmallVector<llvm::StringRef, 15> arguments = {python,
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
  int exitCode = llvm::sys::ExecuteAndWait(python, arguments);
  if (exitCode == 0)
    return llvm::Error::success();
  return llvm::createStringError(
      llvm::errc::io_error, "device link failed with exit code %d", exitCode);
}

} // namespace wafer::compiler::detail
