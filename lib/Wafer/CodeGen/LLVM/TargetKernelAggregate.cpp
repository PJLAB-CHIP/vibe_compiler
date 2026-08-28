//===- TargetKernelAggregate.cpp - Aggregate kernel target module --------===//

#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

constexpr int64_t kKernelAggregateTileCount = 16;
constexpr int64_t kKernelAggregateRowLength = 4;

llvm::Error unsupportedLinkConstruct(llvm::StringRef detail) {
  return llvm::createStringError(
      llvm::errc::invalid_argument,
      "kernel target aggregation does not support %s", detail.str().c_str());
}

bool hasSupportedDefinitionLinkage(const llvm::GlobalValue &value) {
  return value.hasExternalLinkage() || value.hasInternalLinkage() ||
         value.hasPrivateLinkage();
}

llvm::Error validateLinkConstructs(const llvm::Module &module) {
  if (!module.getModuleInlineAsm().empty())
    return unsupportedLinkConstruct("module inline assembly");
  if (!module.getComdatSymbolTable().empty())
    return unsupportedLinkConstruct("COMDAT groups");
  if (!module.alias_empty())
    return unsupportedLinkConstruct("global aliases");
  if (!module.ifunc_empty())
    return unsupportedLinkConstruct("indirect functions");

  for (const llvm::Function &function : module.functions()) {
    if (function.isDeclaration())
      continue;
    if (function.hasComdat())
      return unsupportedLinkConstruct("COMDAT functions");
    if (!hasSupportedDefinitionLinkage(function))
      return unsupportedLinkConstruct("non-closed function linkage");
    if (function.getName().starts_with("llvm."))
      return unsupportedLinkConstruct("defined llvm.* functions");
  }
  for (const llvm::GlobalVariable &global : module.globals()) {
    if (!global.hasInitializer())
      continue;
    if (global.hasComdat())
      return unsupportedLinkConstruct("COMDAT globals");
    if (global.isThreadLocal())
      return unsupportedLinkConstruct("thread-local globals");
    if (global.hasAppendingLinkage())
      return unsupportedLinkConstruct("appending globals");
    if (!hasSupportedDefinitionLinkage(global))
      return unsupportedLinkConstruct("non-closed global linkage");
    if (global.getName().starts_with("llvm."))
      return unsupportedLinkConstruct("defined llvm.* globals");
  }
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<llvm::Module>>
importIntoContext(const llvm::Module &source, llvm::LLVMContext &context,
                  LaunchSlotId launchSlotId) {
  llvm::SmallVector<char, 0> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::WriteBitcodeToFile(source, output);
  llvm::StringRef bytes(storage.data(), storage.size());
  llvm::MemoryBufferRef buffer(
      bytes,
      llvm::formatv("wafer.kernel.launch_slot.{0:D5}", launchSlotId.getValue())
          .str());
  llvm::Expected<std::unique_ptr<llvm::Module>> imported =
      llvm::parseBitcodeFile(buffer, context);
  if (!imported)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "failed to import launch slot %lld for kernel aggregation: %s",
        static_cast<long long>(launchSlotId.getValue()),
        llvm::toString(imported.takeError()).c_str());
  return imported;
}

void stripTileMetadata(llvm::Module &module) {
  llvm::SmallVector<llvm::NamedMDNode *, 8> erase;
  for (llvm::NamedMDNode &metadata : module.named_metadata())
    if (metadata.getName().starts_with("wafer.target.") ||
        metadata.getName() == "llvm.ident")
      erase.push_back(&metadata);
  for (llvm::NamedMDNode *metadata : erase)
    module.eraseNamedMetadata(metadata);
}

llvm::Expected<std::string>
scopeLaunchSlotDefinitions(llvm::Module &module, LaunchSlotId launchSlotId,
                           llvm::StringRef entrySymbol) {
  llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "kernel launch-slot entry body is missing");

  const std::string prefix = llvm::formatv("__wafer_kernel_launch_slot_{0:D5}",
                                           launchSlotId.getValue())
                                 .str();
  const std::string bodyName = prefix + "_main_body";
  uint64_t functionOrdinal = 0;
  for (llvm::Function &function : module.functions()) {
    if (function.isDeclaration())
      continue;
    function.setName(
        &function == entry
            ? bodyName
            : llvm::formatv("{0}_function_{1:D5}", prefix, functionOrdinal++)
                  .str());
    // Keep each uniquely scoped definition externally linkable until it has
    // entered the aggregate. LLVM's IR linker may omit unreferenced local
    // definitions, including the launch-slot body that the dispatcher uses.
    function.setLinkage(llvm::GlobalValue::ExternalLinkage);
    function.setVisibility(llvm::GlobalValue::DefaultVisibility);
  }
  uint64_t globalOrdinal = 0;
  for (llvm::GlobalVariable &global : module.globals()) {
    if (!global.hasInitializer())
      continue;
    global.setName(
        llvm::formatv("{0}_global_{1:D5}", prefix, globalOrdinal++).str());
    global.setLinkage(llvm::GlobalValue::ExternalLinkage);
    global.setVisibility(llvm::GlobalValue::DefaultVisibility);
  }
  return bodyName;
}

void internalizeScopedDefinitions(llvm::Module &module) {
  constexpr llvm::StringLiteral prefix = "__wafer_kernel_launch_slot_";
  for (llvm::Function &function : module.functions())
    if (!function.isDeclaration() && function.getName().starts_with(prefix))
      function.setLinkage(llvm::GlobalValue::InternalLinkage);
  for (llvm::GlobalVariable &global : module.globals())
    if (global.hasInitializer() && global.getName().starts_with(prefix))
      global.setLinkage(llvm::GlobalValue::InternalLinkage);
}

llvm::Expected<llvm::Function *>
getOrInsertExactDeclaration(llvm::Module &module, llvm::StringRef symbol,
                            llvm::FunctionType *type) {
  if (llvm::Function *function = module.getFunction(symbol)) {
    if (!function->isDeclaration() || function->getFunctionType() != type)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "kernel runtime declaration '%s' has an incompatible definition or "
          "type",
          symbol.str().c_str());
    return function;
  }
  return llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                symbol, module);
}

llvm::Error createKernelAggregateExports(
    llvm::Module &module, llvm::ArrayRef<std::string> bodyNames,
    llvm::ArrayRef<int64_t> tileIdsByLaunchSlot,
    llvm::StringRef mainSymbol, uint64_t slotsPerTile, KernelEntryABI entryABI,
    bool includePrepare) {
  llvm::LLVMContext &context = module.getContext();
  llvm::Type *voidType = llvm::Type::getVoidTy(context);
  llvm::IntegerType *i32 = llvm::Type::getInt32Ty(context);
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Type *pointer = llvm::PointerType::get(context, /*AddressSpace=*/0);
  llvm::FunctionType *wrapperType =
      llvm::FunctionType::get(voidType, {pointer}, /*isVarArg=*/false);

  if (module.getNamedValue(mainSymbol) ||
      (includePrepare && module.getNamedValue(kKernelPrepareExportSymbol)))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel aggregate export symbol collides with an imported "
        "declaration");

  llvm::FunctionType *pidType =
      llvm::FunctionType::get(i32, {i32}, /*isVarArg=*/false);
  llvm::Expected<llvm::Function *> getPid =
      getOrInsertExactDeclaration(module, "__get_pid", pidType);
  if (!getPid)
    return getPid.takeError();
  if (includePrepare) {
    llvm::FunctionType *initTileType =
        llvm::FunctionType::get(i32, {i32, i32}, /*isVarArg=*/false);
    llvm::Expected<llvm::Function *> initTileId =
        getOrInsertExactDeclaration(module, "init_tile_id", initTileType);
    if (!initTileId)
      return initTileId.takeError();
    llvm::FunctionType *syncType =
        llvm::FunctionType::get(voidType, {i32}, /*isVarArg=*/false);
    llvm::Expected<llvm::Function *> directSyncInit =
        getOrInsertExactDeclaration(module, "direct_sync_init", syncType);
    if (!directSyncInit)
      return directSyncInit.takeError();

    llvm::Function *prepare =
        llvm::Function::Create(wrapperType, llvm::GlobalValue::ExternalLinkage,
                               kKernelPrepareExportSymbol, module);
    prepare->getArg(0)->setName(entryABI == KernelEntryABI::TileRowPointerTable
                                    ? "tile_row_pointers"
                                    : "tile_major_slots");
    llvm::IRBuilder<> prepareBuilder(
        llvm::BasicBlock::Create(context, "entry", prepare));
    llvm::Value *preparePid = prepareBuilder.CreateCall(
        *getPid, llvm::ConstantInt::get(i32, 0), "pid.x");
    prepareBuilder.CreateCall(
        *initTileId,
        {preparePid,
         llvm::ConstantInt::get(
             i32, static_cast<uint64_t>(kKernelAggregateRowLength))});
    prepareBuilder.CreateCall(
        *directSyncInit,
        llvm::ConstantInt::get(
            i32, static_cast<uint64_t>(kKernelAggregateTileCount)));
    prepareBuilder.CreateRetVoid();
  }

  llvm::Function *main = llvm::Function::Create(
      wrapperType, llvm::GlobalValue::ExternalLinkage, mainSymbol, module);
  main->getArg(0)->setName(entryABI == KernelEntryABI::TileRowPointerTable
                               ? "tile_row_pointers"
                               : "tile_major_slots");
  llvm::BasicBlock *entryBlock =
      llvm::BasicBlock::Create(context, "entry", main);
  llvm::BasicBlock *defaultBlock =
      llvm::BasicBlock::Create(context, "pid.default", main);
  llvm::BasicBlock *exitBlock = llvm::BasicBlock::Create(context, "exit", main);
  llvm::IRBuilder<> builder(entryBlock);
  llvm::Value *pid =
      builder.CreateCall(*getPid, llvm::ConstantInt::get(i32, 0), "pid.x");
  llvm::SwitchInst *dispatch =
      builder.CreateSwitch(pid, defaultBlock, kKernelAggregateTileCount);

  for (int64_t launchSlot = 0; launchSlot < kKernelAggregateTileCount;
       ++launchSlot) {
    llvm::Function *body = module.getFunction(bodyNames[launchSlot]);
    if (!body || body->isDeclaration() || body->isVarArg() ||
        !body->getReturnType()->isVoidTy() ||
        body->arg_size() != slotsPerTile ||
        !llvm::all_of(body->args(), [](const llvm::Argument &argument) {
          return argument.getType()->isIntegerTy(64);
        }))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "kernel launch slot %lld body '%s' does not match the common typed "
          "slot schema (present=%d declaration=%d vararg=%d arguments=%llu)",
          static_cast<long long>(launchSlot), bodyNames[launchSlot].c_str(),
          body != nullptr, body ? body->isDeclaration() : 0,
          body ? body->isVarArg() : 0,
          static_cast<unsigned long long>(body ? body->arg_size() : 0));
    const int64_t tileId = tileIdsByLaunchSlot[launchSlot];
    llvm::BasicBlock *slotBlock = llvm::BasicBlock::Create(
        context,
        llvm::formatv("tile.{0}.launch_slot.{1}", tileId,
                      launchSlot)
            .str(),
        main);
    dispatch->addCase(llvm::ConstantInt::get(i32, tileId), slotBlock);
    builder.SetInsertPoint(slotBlock);
    llvm::SmallVector<llvm::Value *, 16> arguments;
    arguments.reserve(slotsPerTile);
    llvm::Value *row = main->getArg(0);
    uint64_t rowBase = static_cast<uint64_t>(launchSlot) * slotsPerTile;
    if (entryABI == KernelEntryABI::TileRowPointerTable) {
      llvm::Value *rowAddress = builder.CreateInBoundsGEP(
          i64, main->getArg(0), llvm::ConstantInt::get(i64, launchSlot),
          llvm::formatv("launch_slot.{0}.row.address", launchSlot).str());
      llvm::LoadInst *rowValue = builder.CreateLoad(
          i64, rowAddress,
          llvm::formatv("launch_slot.{0}.row", launchSlot).str());
      rowValue->setAlignment(llvm::Align(8));
      row = builder.CreateIntToPtr(
          rowValue, pointer,
          llvm::formatv("launch_slot.{0}.slots", launchSlot).str());
      rowBase = 0;
    }
    for (uint64_t slot = 0; slot < slotsPerTile; ++slot) {
      llvm::Value *address = builder.CreateInBoundsGEP(
          i64, row, llvm::ConstantInt::get(i64, rowBase + slot),
          llvm::formatv("launch_slot.{0}.slot.{1}.address", launchSlot, slot)
              .str());
      llvm::LoadInst *value = builder.CreateLoad(
          i64, address,
          llvm::formatv("launch_slot.{0}.slot.{1}", launchSlot, slot).str());
      value->setAlignment(llvm::Align(8));
      arguments.push_back(value);
    }
    builder.CreateCall(body, arguments);
    builder.CreateBr(exitBlock);
  }
  builder.SetInsertPoint(defaultBlock);
  builder.CreateBr(exitBlock);
  builder.SetInsertPoint(exitBlock);
  builder.CreateRetVoid();

  std::string verificationStorage;
  llvm::raw_string_ostream verification(verificationStorage);
  if (llvm::verifyModule(module, &verification))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel aggregate target module verification failed: %s",
        verification.str().c_str());
  return llvm::Error::success();
}

} // namespace

llvm::Expected<OwnedTargetLLVMModule>
buildKernelAggregateTargetModule(const TargetLLVMModules &targetLLVMModules) {
  const ExecutionConfig &config = targetLLVMModules.getExecutionConfig();
  const KernelRuntimeLaunchContract &kernel =
      targetLLVMModules.getRuntimeLaunchContract().getKernel();
  if ((kernel.entryABI != KernelEntryABI::TileMajorPointerTable &&
       kernel.entryABI != KernelEntryABI::TileRowPointerTable) ||
      config.getTileCount() != kKernelAggregateTileCount ||
      targetLLVMModules.getModules().size() != kKernelAggregateTileCount)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel target aggregation requires the complete typed 16-Tile "
        "launch domain");

  const TargetLLVMModule &first = targetLLVMModules.getModules().front();
  const uint64_t packetBytes = kernel.form == KernelLaunchForm::Cluster
                                   ? kTx81ClusterKernelArgumentBytesMax
                                   : kTx81KernelArgumentBytesMax;
  if (first.getTileEntryArguments().empty() ||
      (kernel.entryABI == KernelEntryABI::TileMajorPointerTable &&
       first.getTileEntryArguments().size() >
           packetBytes / sizeof(uint64_t) / kKernelAggregateTileCount) ||
      (kernel.entryABI == KernelEntryABI::TileRowPointerTable &&
       kKernelAggregateTileCount > packetBytes / sizeof(uint64_t)))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel aggregate argument packet exceeds the qualified V5.6 packet "
        "limit");
  const size_t transportStatusSlots =
      llvm::count_if(first.getTileEntryArguments(), [](const TileEntryArgument &slot) {
        return slot.kind == TileEntryArgumentKind::TransportStatus;
      });
  if (transportStatusSlots > 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel aggregate has more than one typed transport status slot");

  std::optional<CardId> cardId;
  std::set<int64_t> tileIds;
  std::vector<const TargetLLVMModule *> modulesByLaunchSlot(
      kKernelAggregateTileCount, nullptr);
  std::vector<int64_t> tileIdsByLaunchSlot(kKernelAggregateTileCount,
                                                   -1);
  for (const TargetLLVMModule &source : targetLLVMModules.getModules()) {
    if (!cardId)
      cardId = source.getCardId();
    const int64_t launchSlot = source.getLaunchSlotId().getValue();
    if (source.getCardId() != *cardId ||
        source.getCardId().getValue() < 0 ||
        source.getTileId().getValue() < 0 ||
        source.getTileId().getValue() >= kKernelAggregateTileCount ||
        launchSlot < 0 || launchSlot >= kKernelAggregateTileCount ||
        !tileIds.insert(source.getTileId().getValue()).second ||
        modulesByLaunchSlot[launchSlot] != nullptr)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "kernel target aggregation has invalid or duplicate physical "
          "identity");
    if (source.getEntrySymbol() != first.getEntrySymbol() ||
        source.getTargetIdentityId() != first.getTargetIdentityId() ||
        source.getKernelRuntimeABIId() != first.getKernelRuntimeABIId() ||
        source.getModuleFormat() != first.getModuleFormat() ||
        source.getModule().getTargetTriple() !=
            first.getModule().getTargetTriple() ||
        source.getModule().getDataLayoutStr() !=
            first.getModule().getDataLayoutStr() ||
        findTileEntryArgumentOrderDifference(source.getTileEntryArguments(),
                                             first.getTileEntryArguments()))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "kernel target Tile domain has inconsistent typed module "
          "facts");
    modulesByLaunchSlot[launchSlot] = &source;
    tileIdsByLaunchSlot[launchSlot] =
        source.getTileId().getValue();
  }
  if (llvm::is_contained(modulesByLaunchSlot, nullptr))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel target launch-slot domain is not dense and complete");

  auto context = std::make_unique<llvm::LLVMContext>();
  auto aggregate =
      std::make_unique<llvm::Module>("wafer.kernel.aggregate", *context);
  aggregate->setTargetTriple(first.getModule().getTargetTriple());
  aggregate->setDataLayout(first.getModule().getDataLayoutStr());
  llvm::Linker linker(*aggregate);
  std::vector<std::string> bodyNames(kKernelAggregateTileCount);

  for (int64_t launchSlot = 0; launchSlot < kKernelAggregateTileCount;
       ++launchSlot) {
    const TargetLLVMModule &source = *modulesByLaunchSlot[launchSlot];
    if (llvm::Error error = validateLinkConstructs(source.getModule()))
      return std::move(error);
    llvm::Expected<std::unique_ptr<llvm::Module>> imported = importIntoContext(
        source.getModule(), *context, source.getLaunchSlotId());
    if (!imported)
      return imported.takeError();
    stripTileMetadata(**imported);
    llvm::Expected<std::string> bodyName = scopeLaunchSlotDefinitions(
        **imported, source.getLaunchSlotId(), source.getEntrySymbol());
    if (!bodyName)
      return bodyName.takeError();
    bodyNames[launchSlot] = std::move(*bodyName);
    (*imported)->setModuleIdentifier(
        llvm::formatv("wafer.kernel.launch_slot.{0:D5}", launchSlot).str());
    (*imported)->setSourceFileName("");
    if (linker.linkInModule(std::move(*imported)))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "kernel target launch-slot modules could "
                                     "not be linked into one closed "
                                     "module");
  }

  internalizeScopedDefinitions(*aggregate);
  const bool hasPrepare = llvm::is_contained(
      targetLLVMModules.getRuntimeLaunchContract().getPhases(),
      RuntimeLaunchPhaseRole::Prepare);
  if (llvm::Error error = createKernelAggregateExports(
          *aggregate, bodyNames, tileIdsByLaunchSlot,
          first.getEntrySymbol(), first.getTileEntryArguments().size(),
          kernel.entryABI, hasPrepare))
    return std::move(error);
  return OwnedTargetLLVMModule{std::move(context), std::move(aggregate)};
}

} // namespace wafer::compiler::detail
