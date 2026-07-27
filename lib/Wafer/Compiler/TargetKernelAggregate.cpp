//===- TargetKernelAggregate.cpp - Aggregate kernel target module --------===//

#include "TargetArtifactInternal.h"

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
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

constexpr int64_t kKernelAggregateRankCount = 16;
constexpr int64_t kKernelAggregateRowLength = 4;

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
                  int64_t logicalRank) {
  llvm::SmallVector<char, 0> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::WriteBitcodeToFile(source, output);
  llvm::StringRef bytes(storage.data(), storage.size());
  llvm::MemoryBufferRef buffer(
      bytes, llvm::formatv("wafer.kernel.rank.{0:D5}", logicalRank).str());
  llvm::Expected<std::unique_ptr<llvm::Module>> imported =
      llvm::parseBitcodeFile(buffer, context);
  if (!imported)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "failed to import logical rank %lld for kernel aggregation: %s",
        static_cast<long long>(logicalRank),
        llvm::toString(imported.takeError()).c_str());
  return imported;
}

void stripRankMetadata(llvm::Module &module) {
  llvm::SmallVector<llvm::NamedMDNode *, 8> erase;
  for (llvm::NamedMDNode &metadata : module.named_metadata())
    if (metadata.getName().starts_with("wafer.target.") ||
        metadata.getName() == "llvm.ident")
      erase.push_back(&metadata);
  for (llvm::NamedMDNode *metadata : erase)
    module.eraseNamedMetadata(metadata);
}

llvm::Expected<std::string> scopeRankDefinitions(llvm::Module &module,
                                                 int64_t logicalRank,
                                                 llvm::StringRef entrySymbol) {
  llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "kernel rank entry body is missing");

  const std::string prefix =
      llvm::formatv("__wafer_kernel_rank_{0:D5}", logicalRank).str();
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
    // definitions, including the rank body that the dispatcher will use.
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
  constexpr llvm::StringLiteral prefix = "__wafer_kernel_rank_";
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

llvm::Error createKernelAggregateExports(llvm::Module &module,
                                         llvm::ArrayRef<std::string> bodyNames,
                                         llvm::StringRef mainSymbol,
                                         uint64_t slotsPerRank,
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
    prepare->getArg(0)->setName("rank_major_slots");
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
            i32, static_cast<uint64_t>(kKernelAggregateRankCount)));
    prepareBuilder.CreateRetVoid();
  }

  llvm::Function *main = llvm::Function::Create(
      wrapperType, llvm::GlobalValue::ExternalLinkage, mainSymbol, module);
  main->getArg(0)->setName("rank_major_slots");
  llvm::BasicBlock *entryBlock =
      llvm::BasicBlock::Create(context, "entry", main);
  llvm::BasicBlock *defaultBlock =
      llvm::BasicBlock::Create(context, "pid.default", main);
  llvm::BasicBlock *exitBlock = llvm::BasicBlock::Create(context, "exit", main);
  llvm::IRBuilder<> builder(entryBlock);
  llvm::Value *pid =
      builder.CreateCall(*getPid, llvm::ConstantInt::get(i32, 0), "pid.x");
  llvm::SwitchInst *dispatch =
      builder.CreateSwitch(pid, defaultBlock, kKernelAggregateRankCount);

  for (int64_t rank = 0; rank < kKernelAggregateRankCount; ++rank) {
    llvm::Function *body = module.getFunction(bodyNames[rank]);
    if (!body || body->isDeclaration() || body->isVarArg() ||
        !body->getReturnType()->isVoidTy() ||
        body->arg_size() != slotsPerRank ||
        !llvm::all_of(body->args(), [](const llvm::Argument &argument) {
          return argument.getType()->isIntegerTy(64);
        }))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "kernel rank %lld body '%s' does not match the common typed slot "
          "schema (present=%d declaration=%d vararg=%d arguments=%llu)",
          static_cast<long long>(rank), bodyNames[rank].c_str(),
          body != nullptr, body ? body->isDeclaration() : 0,
          body ? body->isVarArg() : 0,
          static_cast<unsigned long long>(body ? body->arg_size() : 0));
    llvm::BasicBlock *rankBlock = llvm::BasicBlock::Create(
        context, llvm::formatv("pid.{0}", rank).str(), main);
    dispatch->addCase(llvm::ConstantInt::get(i32, rank), rankBlock);
    builder.SetInsertPoint(rankBlock);
    llvm::SmallVector<llvm::Value *, 16> arguments;
    arguments.reserve(slotsPerRank);
    const uint64_t rowBase = static_cast<uint64_t>(rank) * slotsPerRank;
    for (uint64_t slot = 0; slot < slotsPerRank; ++slot) {
      llvm::Value *address = builder.CreateInBoundsGEP(
          i64, main->getArg(0), llvm::ConstantInt::get(i64, rowBase + slot),
          llvm::formatv("rank.{0}.slot.{1}.address", rank, slot).str());
      llvm::LoadInst *value = builder.CreateLoad(
          i64, address, llvm::formatv("rank.{0}.slot.{1}", rank, slot).str());
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

llvm::Expected<OwnedTargetLLVMModule> buildKernelAggregateTargetModule(
    const TargetLLVMModuleBundle &targetLLVMModules) {
  const ExecutionConfig &config = targetLLVMModules.getExecutionConfig();
  const KernelRuntimeLaunchContract *kernel =
      targetLLVMModules.getRuntimeLaunchContract().getKernel();
  if (!kernel || kernel->form == KernelLaunchForm::PerRank ||
      kernel->entryABI != KernelEntryABI::RankMajorPointerTableV1 ||
      config.getRankCount() != kKernelAggregateRankCount ||
      targetLLVMModules.getModules().size() != kKernelAggregateRankCount)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel target aggregation requires the complete typed 16-rank "
        "launch domain");

  const TargetLLVMModule &first = targetLLVMModules.getModules().front();
  const uint64_t packetBytes = kernel->form == KernelLaunchForm::Cluster
                                   ? kTx81ClusterKernelArgumentBytesMax
                                   : kTx81KernelArgumentBytesMax;
  if (first.getKernelABISlots().empty() ||
      first.getKernelABISlots().size() >
          packetBytes / sizeof(uint64_t) / kKernelAggregateRankCount)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel rank-major argument table exceeds the qualified V5.6 packet "
        "limit");
  const size_t transportStatusSlots =
      llvm::count_if(first.getKernelABISlots(), [](const KernelABISlot &slot) {
        return slot.role == KernelABISlotRole::TransportStatus;
      });
  if (transportStatusSlots > 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel aggregate has more than one typed transport status slot");

  auto context = std::make_unique<llvm::LLVMContext>();
  auto aggregate =
      std::make_unique<llvm::Module>("wafer.kernel.aggregate", *context);
  aggregate->setTargetTriple(first.getModule().getTargetTriple());
  aggregate->setDataLayout(first.getModule().getDataLayoutStr());
  llvm::Linker linker(*aggregate);
  std::vector<std::string> bodyNames;
  bodyNames.reserve(kKernelAggregateRankCount);

  for (int64_t rank = 0; rank < kKernelAggregateRankCount; ++rank) {
    const TargetLLVMModule &source = targetLLVMModules.getModules()[rank];
    if (source.getLogicalRank() != rank ||
        source.getEntrySymbol() != first.getEntrySymbol() ||
        source.getTargetProfileId() != first.getTargetProfileId() ||
        source.getTargetIdentityId() != first.getTargetIdentityId() ||
        source.getKernelRuntimeABIId() != first.getKernelRuntimeABIId() ||
        source.getModuleFormat() != first.getModuleFormat() ||
        source.getModule().getTargetTriple() !=
            first.getModule().getTargetTriple() ||
        source.getModule().getDataLayoutStr() !=
            first.getModule().getDataLayoutStr() ||
        !haveSameKernelABISchema(source.getKernelABISlots(),
                                 first.getKernelABISlots()))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "kernel target rank domain has inconsistent typed module facts");
    if (llvm::Error error = validateLinkConstructs(source.getModule()))
      return std::move(error);
    llvm::Expected<std::unique_ptr<llvm::Module>> imported = importIntoContext(
        source.getModule(), *context, source.getLogicalRank());
    if (!imported)
      return imported.takeError();
    stripRankMetadata(**imported);
    llvm::Expected<std::string> bodyName = scopeRankDefinitions(
        **imported, source.getLogicalRank(), source.getEntrySymbol());
    if (!bodyName)
      return bodyName.takeError();
    bodyNames.push_back(std::move(*bodyName));
    (*imported)->setModuleIdentifier(
        llvm::formatv("wafer.kernel.rank.{0:D5}", rank).str());
    (*imported)->setSourceFileName("");
    if (linker.linkInModule(std::move(*imported)))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "kernel target rank modules could not be linked into one closed "
          "module");
  }

  internalizeScopedDefinitions(*aggregate);
  const bool hasPrepare = llvm::is_contained(
      targetLLVMModules.getRuntimeLaunchContract().getPhases(),
      RuntimeLaunchPhaseRole::Prepare);
  if (llvm::Error error = createKernelAggregateExports(
          *aggregate, bodyNames, first.getEntrySymbol(),
          first.getKernelABISlots().size(), hasPrepare))
    return std::move(error);
  return OwnedTargetLLVMModule{std::move(context), std::move(aggregate)};
}

} // namespace wafer::compiler::detail
