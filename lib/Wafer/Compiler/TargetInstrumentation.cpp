//===- TargetInstrumentation.cpp - Profiling target cloning --------------===//

#include "TargetCodeGenInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SHA256.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

constexpr llvm::StringLiteral kEntryBeginSymbol =
    "wafer_tx81_profile_entry_begin_from_config";
constexpr llvm::StringLiteral kEntryEndSymbol = "wafer_tx81_profile_entry_end";
constexpr llvm::StringLiteral kSiteBeginSymbol =
    "wafer_tx81_profile_site_begin";
constexpr llvm::StringLiteral kSiteEndSymbol = "wafer_tx81_profile_site_end";

struct CollectedProfileTargetCallSite {
  const llvm::CallBase *call = nullptr;
  ProfileTargetCallSite record;
};

struct ProfileIRNumbering {
  llvm::DenseMap<const llvm::Function *, uint64_t> functionOrdinals;
  llvm::DenseMap<const llvm::BasicBlock *, uint64_t> blockOrdinals;
  llvm::DenseMap<const llvm::Instruction *, uint64_t> instructionOrdinals;
};

bool isProfileInstrumentationCall(const llvm::Instruction &instruction) {
  const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
  const llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
  if (!callee)
    return false;
  llvm::StringRef symbol = callee->getName();
  return symbol == kEntryBeginSymbol || symbol == kEntryEndSymbol ||
         symbol == kSiteBeginSymbol || symbol == kSiteEndSymbol;
}

ProfileIRNumbering
buildProfileIRNumbering(llvm::ArrayRef<const llvm::Function *> reachable) {
  ProfileIRNumbering index;
  for (auto [functionOrdinal, function] : llvm::enumerate(reachable)) {
    index.functionOrdinals.try_emplace(function, functionOrdinal);
    for (auto [blockOrdinal, block] : llvm::enumerate(*function)) {
      index.blockOrdinals.try_emplace(&block, blockOrdinal);
      uint64_t instructionOrdinal = 0;
      for (const llvm::Instruction &instruction : block) {
        if (isProfileInstrumentationCall(instruction))
          continue;
        index.instructionOrdinals.try_emplace(&instruction,
                                              instructionOrdinal++);
      }
    }
  }
  return index;
}

llvm::Expected<std::string> getTargetCallArgumentSignature(
    const llvm::CallBase &call, const TargetCallDescriptor &descriptor,
    uint64_t targetCallOrdinal, const ProfileIRNumbering &numbering) {
  std::string storage;
  llvm::raw_string_ostream output(storage);
  output << "target-call=" << targetCallOrdinal;
  for (auto [index, scalar] : llvm::enumerate(descriptor.arguments)) {
    const unsigned width = scalar == TargetCallScalarType::I64 ? 64 : 32;
    output << ";arg" << index << ":i" << width << "=";
    const llvm::Value *operand = call.getArgOperand(index);
    if (!operand->getType()->isIntegerTy(width))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile target call argument does not match its typed descriptor");
    if (const auto *constant = llvm::dyn_cast<llvm::ConstantInt>(operand)) {
      output << "constant:0x"
             << llvm::utohexstr(constant->getZExtValue(),
                                /*LowerCase=*/true);
      continue;
    }
    if (const auto *argument = llvm::dyn_cast<llvm::Argument>(operand)) {
      auto function = numbering.functionOrdinals.find(argument->getParent());
      if (function == numbering.functionOrdinals.end())
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "profile target-call argument is outside the reachable functions");
      output << "argument:function:" << function->second
             << ":index:" << argument->getArgNo();
      continue;
    }
    if (const auto *instruction = llvm::dyn_cast<llvm::Instruction>(operand)) {
      auto function =
          numbering.functionOrdinals.find(instruction->getFunction());
      auto block = numbering.blockOrdinals.find(instruction->getParent());
      auto ordinal = numbering.instructionOrdinals.find(instruction);
      if (function == numbering.functionOrdinals.end() ||
          block == numbering.blockOrdinals.end() ||
          ordinal == numbering.instructionOrdinals.end())
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "profile target-call SSA operand has no assigned IR ordinal");
      output << "instruction:function:" << function->second
             << ":block:" << block->second << ":index:" << ordinal->second
             << ":opcode:" << instruction->getOpcodeName();
      continue;
    }
    if (llvm::isa<llvm::Constant>(operand)) {
      output << "constant-expression:";
      operand->printAsOperand(output, /*PrintType=*/false);
      continue;
    }
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target-call argument has an unsupported SSA producer");
  }
  output.flush();
  llvm::SHA256 hasher;
  hasher.update(storage);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

llvm::Expected<std::vector<CollectedProfileTargetCallSite>>
collectProfileTargetCallSitesImpl(const llvm::Module &module,
                                  llvm::StringRef entrySymbol) {
  const llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target entry is missing or has no definition");

  llvm::SmallVector<const llvm::Function *, 16> reachable = {entry};
  llvm::DenseSet<const llvm::Function *> seen = {entry};
  for (size_t functionIndex = 0; functionIndex < reachable.size();
       ++functionIndex) {
    const llvm::Function *function = reachable[functionIndex];
    for (const llvm::BasicBlock &block : *function)
      for (const llvm::Instruction &instruction : block) {
        const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        const llvm::Function *callee =
            call ? call->getCalledFunction() : nullptr;
        if (callee && !callee->isDeclaration() && seen.insert(callee).second)
          reachable.push_back(callee);
      }
  }

  llvm::ArrayRef<TargetCallDescriptor> descriptors = getTargetCallDescriptors();
  ProfileIRNumbering irNumbering = buildProfileIRNumbering(reachable);
  llvm::StringMap<uint64_t> occurrences;
  std::vector<CollectedProfileTargetCallSite> sites;
  for (auto [functionOrdinal, function] : llvm::enumerate(reachable)) {
    uint64_t blockOrdinal = 0;
    for (const llvm::BasicBlock &block : *function) {
      uint64_t instructionOrdinal = 0;
      for (const llvm::Instruction &instruction : block) {
        const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        const llvm::Function *callee =
            call ? call->getCalledFunction() : nullptr;
        const TargetCallDescriptor *descriptor =
            callee ? findTargetCallDescriptor(callee->getName()) : nullptr;
        if (!descriptor) {
          ++instructionOrdinal;
          continue;
        }
        if (!llvm::isa<llvm::CallInst>(call))
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "profile target call must lower to a direct call");
        if (call->arg_size() != descriptor->arguments.size())
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "profile target call does not match its typed descriptor");

        const uint64_t descriptorOrdinal =
            static_cast<uint64_t>(descriptor - descriptors.data());
        llvm::Expected<std::string> signature = getTargetCallArgumentSignature(
            *call, *descriptor, descriptorOrdinal, irNumbering);
        if (!signature)
          return signature.takeError();
        std::string occurrenceKey =
            llvm::formatv("{0}:{1}", descriptorOrdinal, *signature).str();
        uint64_t occurrence = occurrences[occurrenceKey]++;
        std::string correlationKey =
            llvm::formatv("target-call:{0}:arguments:{1}:occurrence:{2}",
                          descriptorOrdinal, *signature, occurrence)
                .str();
        ProfileTargetCallSite record;
        record.siteId = sites.size();
        record.functionOrdinal = functionOrdinal;
        record.blockOrdinal = blockOrdinal;
        record.instructionOrdinal = instructionOrdinal;
        record.targetCallOrdinal = descriptorOrdinal;
        record.targetCallSymbol = descriptor->symbol;
        record.siteKind = runtime::getProfileTargetSiteKind(*descriptor);
        record.engine = getTargetCallTSMEngine(*descriptor);
        record.correlationKey = std::move(correlationKey);
        sites.push_back({call, std::move(record)});
        ++instructionOrdinal;
      }
      ++blockOrdinal;
    }
  }
  return sites;
}

llvm::Expected<llvm::Function *>
getOrInsertExactDeclaration(llvm::Module &module, llvm::StringRef symbol,
                            llvm::FunctionType *type) {
  if (llvm::Function *function = module.getFunction(symbol)) {
    if (!function->isDeclaration() || function->getFunctionType() != type)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile runtime symbol '%s' has an incompatible definition or type",
          symbol.str().c_str());
    return function;
  }
  return llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                symbol, module);
}

} // namespace

llvm::StringRef stringifyProfileCaptureKind(ProfileCaptureKind capture) {
  switch (capture) {
  case ProfileCaptureKind::None:
    return "none";
  case ProfileCaptureKind::Count:
    return "count";
  case ProfileCaptureKind::Trace:
    return "trace";
  }
  llvm_unreachable("unknown profile capture kind");
}

uint64_t getProfileCaptureRecordBytes(ProfileCaptureKind capture) {
  switch (capture) {
  case ProfileCaptureKind::None:
    return 0;
  case ProfileCaptureKind::Count:
    return WAFER_TX81_PROFILER_MIN_BUFFER_BYTES;
  case ProfileCaptureKind::Trace:
    return WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES;
  }
  llvm_unreachable("unknown profile capture kind");
}

llvm::Error
verifyProfileCaptureKernelABISlots(llvm::ArrayRef<KernelABISlot> slots,
                                   ProfileCaptureKind capture) {
  auto isProfilerSlot = [](const KernelABISlot &slot) {
    return slot.role == KernelABISlotRole::Workspace && slot.resourceIndex == 1;
  };
  const size_t count = llvm::count_if(slots, isProfilerSlot);
  if (capture == ProfileCaptureKind::None) {
    if (count != 0)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "ordinary target ABI contains a profiler workspace slot");
    return llvm::Error::success();
  }
  if (count != 1 || slots.empty() || !isProfilerSlot(slots.back()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target ABI must end with exactly one profiler workspace slot");
  const KernelABISlot &slot = slots.back();
  const uint64_t recordBytes = getProfileCaptureRecordBytes(capture);
  if (slot.ordinal != static_cast<int64_t>(slots.size() - 1) ||
      slot.dtype != "u8" || slot.layout != MemLayout::Tensor ||
      slot.shape != std::vector<int64_t>{static_cast<int64_t>(recordBytes)} ||
      slot.byteSize != static_cast<int64_t>(recordBytes) ||
      slot.alignment != WAFER_TX81_PROFILER_BUFFER_ALIGNMENT)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target ABI profiler workspace facts do not match the record "
        "ABI");
  return llvm::Error::success();
}

llvm::Expected<std::vector<ProfileTargetCallSite>>
collectProfileTargetCallSites(const llvm::Module &module,
                              llvm::StringRef entrySymbol) {
  llvm::Expected<std::vector<CollectedProfileTargetCallSite>> collected =
      collectProfileTargetCallSitesImpl(module, entrySymbol);
  if (!collected)
    return collected.takeError();
  std::vector<ProfileTargetCallSite> sites;
  sites.reserve(collected->size());
  for (CollectedProfileTargetCallSite &site : *collected)
    sites.push_back(std::move(site.record));
  return sites;
}

llvm::Error verifyProfileTargetCallSitesMatch(
    const llvm::Module &productionModule, llvm::StringRef productionEntrySymbol,
    const llvm::Module &traceModule, llvm::StringRef traceEntrySymbol) {
  llvm::Expected<std::vector<ProfileTargetCallSite>> production =
      collectProfileTargetCallSites(productionModule, productionEntrySymbol);
  if (!production)
    return production.takeError();
  llvm::Expected<std::vector<ProfileTargetCallSite>> trace =
      collectProfileTargetCallSites(traceModule, traceEntrySymbol);
  if (!trace)
    return trace.takeError();
  if (production->size() != trace->size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile trace target-call site count differs from final production");

  for (auto [index, pair] : llvm::enumerate(llvm::zip(*production, *trace))) {
    const ProfileTargetCallSite &finalSite = std::get<0>(pair);
    const ProfileTargetCallSite &traceSite = std::get<1>(pair);
    if (finalSite.siteId != index || traceSite.siteId != index ||
        finalSite.functionOrdinal != traceSite.functionOrdinal ||
        finalSite.blockOrdinal != traceSite.blockOrdinal ||
        finalSite.targetCallOrdinal != traceSite.targetCallOrdinal ||
        finalSite.targetCallSymbol != traceSite.targetCallSymbol ||
        finalSite.siteKind != traceSite.siteKind ||
        finalSite.engine != traceSite.engine ||
        finalSite.correlationKey != traceSite.correlationKey)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile trace target-call site numbering differs from final "
          "production at Tile-local site %llu",
          static_cast<unsigned long long>(index));
  }
  return llvm::Error::success();
}

llvm::Error instrumentProfileTargetModule(llvm::Module &module,
                                          llvm::StringRef entrySymbol,
                                          ProfileCaptureKind capture) {
  if (capture == ProfileCaptureKind::None)
    return llvm::Error::success();
  llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration() || entry->arg_empty() ||
      !entry->getArg(entry->arg_size() - 1)->getType()->isIntegerTy(64))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target entry is missing its final i64 record address");

  llvm::Expected<std::vector<CollectedProfileTargetCallSite>> sites =
      collectProfileTargetCallSitesImpl(module, entrySymbol);
  if (!sites)
    return sites.takeError();

  llvm::LLVMContext &context = module.getContext();
  llvm::Type *voidType = llvm::Type::getVoidTy(context);
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  llvm::Expected<llvm::Function *> entryBegin = getOrInsertExactDeclaration(
      module, kEntryBeginSymbol,
      llvm::FunctionType::get(voidType, {i64}, /*isVarArg=*/false));
  if (!entryBegin)
    return entryBegin.takeError();
  llvm::Expected<llvm::Function *> entryEnd = getOrInsertExactDeclaration(
      module, kEntryEndSymbol,
      llvm::FunctionType::get(voidType, {}, /*isVarArg=*/false));
  if (!entryEnd)
    return entryEnd.takeError();

  llvm::IRBuilder<> entryBuilder(
      &*entry->getEntryBlock().getFirstInsertionPt());
  entryBuilder.CreateCall(*entryBegin, entry->getArg(entry->arg_size() - 1));

  llvm::SmallVector<llvm::ReturnInst *, 4> returns;
  for (llvm::BasicBlock &block : *entry)
    if (auto *returnInstruction =
            llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator()))
      returns.push_back(returnInstruction);
  if (returns.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "profile target entry has no return");
  for (llvm::ReturnInst *returnInstruction : returns) {
    llvm::IRBuilder<> returnBuilder(returnInstruction);
    returnBuilder.CreateCall(*entryEnd);
  }

  if (capture == ProfileCaptureKind::Count ||
      capture == ProfileCaptureKind::Trace) {
    llvm::Expected<llvm::Function *> siteBegin = getOrInsertExactDeclaration(
        module, kSiteBeginSymbol,
        llvm::FunctionType::get(voidType, {i32}, /*isVarArg=*/false));
    if (!siteBegin)
      return siteBegin.takeError();
    llvm::Expected<llvm::Function *> siteEnd = getOrInsertExactDeclaration(
        module, kSiteEndSymbol,
        llvm::FunctionType::get(voidType, {i32}, /*isVarArg=*/false));
    if (!siteEnd)
      return siteEnd.takeError();
    for (const CollectedProfileTargetCallSite &site : *sites) {
      auto *call = const_cast<llvm::CallBase *>(site.call);
      llvm::Value *siteId =
          llvm::ConstantInt::get(i32, site.record.siteId, /*isSigned=*/false);
      llvm::IRBuilder<> before(call);
      before.CreateCall(*siteBegin, siteId);
      llvm::Instruction *next = call->getNextNode();
      if (!next)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "profile target call has no continuation");
      llvm::IRBuilder<> after(next);
      after.CreateCall(*siteEnd, siteId);
    }
  }

  std::string verificationStorage;
  llvm::raw_string_ostream verification(verificationStorage);
  if (llvm::verifyModule(module, &verification))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target instrumentation produced invalid LLVM IR: %s",
        verification.str().c_str());
  return llvm::Error::success();
}

llvm::Error
verifyProfileTargetModuleInstrumentation(const llvm::Module &module,
                                         llvm::StringRef entrySymbol,
                                         ProfileCaptureKind capture) {
  auto hasSymbol = [&](llvm::StringRef symbol) {
    return module.getFunction(symbol) != nullptr;
  };
  const bool hasEntryBegin = hasSymbol(kEntryBeginSymbol);
  const bool hasEntryEnd = hasSymbol(kEntryEndSymbol);
  const bool hasSiteBegin = hasSymbol(kSiteBeginSymbol);
  const bool hasSiteEnd = hasSymbol(kSiteEndSymbol);
  if (capture == ProfileCaptureKind::None) {
    if (hasEntryBegin || hasEntryEnd || hasSiteBegin || hasSiteEnd)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "ordinary target module contains profile runtime symbols");
    return llvm::Error::success();
  }
  if (!hasEntryBegin || !hasEntryEnd)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target module is missing entry runtime symbols");

  const llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration() || entry->arg_empty() ||
      !entry->getArg(entry->arg_size() - 1)->getType()->isIntegerTy(64))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target entry is missing its final i64 record address");

  llvm::LLVMContext &context = module.getContext();
  llvm::Type *voidType = llvm::Type::getVoidTy(context);
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  auto requireExactDeclaration =
      [&](llvm::StringRef symbol,
          llvm::FunctionType *type) -> llvm::Expected<const llvm::Function *> {
    const llvm::Function *function = module.getFunction(symbol);
    if (!function || !function->isDeclaration() ||
        function->getFunctionType() != type)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile runtime symbol '%s' has an incompatible definition or type",
          symbol.str().c_str());
    return function;
  };
  auto collectDirectCalls = [&](const llvm::Function &function)
      -> llvm::Expected<llvm::SmallVector<const llvm::CallBase *, 16>> {
    llvm::SmallVector<const llvm::CallBase *, 16> calls;
    for (const llvm::User *user : function.users()) {
      const auto *call = llvm::dyn_cast<llvm::CallBase>(user);
      if (!call || call->getCalledOperand()->stripPointerCasts() != &function)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "profile runtime symbol '%s' has a non-direct-call use",
            function.getName().str().c_str());
      calls.push_back(call);
    }
    return calls;
  };

  llvm::Expected<const llvm::Function *> entryBegin = requireExactDeclaration(
      kEntryBeginSymbol,
      llvm::FunctionType::get(voidType, {i64}, /*isVarArg=*/false));
  if (!entryBegin)
    return entryBegin.takeError();
  llvm::Expected<const llvm::Function *> entryEnd = requireExactDeclaration(
      kEntryEndSymbol,
      llvm::FunctionType::get(voidType, {}, /*isVarArg=*/false));
  if (!entryEnd)
    return entryEnd.takeError();
  llvm::Expected<llvm::SmallVector<const llvm::CallBase *, 16>>
      entryBeginCalls = collectDirectCalls(**entryBegin);
  if (!entryBeginCalls)
    return entryBeginCalls.takeError();
  llvm::Expected<llvm::SmallVector<const llvm::CallBase *, 16>> entryEndCalls =
      collectDirectCalls(**entryEnd);
  if (!entryEndCalls)
    return entryEndCalls.takeError();
  if (entryBeginCalls->size() != 1 ||
      entryBeginCalls->front()->getParent() != &entry->getEntryBlock() ||
      entryBeginCalls->front() !=
          &*entry->getEntryBlock().getFirstInsertionPt() ||
      entryBeginCalls->front()->arg_size() != 1 ||
      entryBeginCalls->front()->getArgOperand(0) !=
          entry->getArg(entry->arg_size() - 1))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target entry must begin with exactly one record binding call");

  llvm::SmallVector<const llvm::ReturnInst *, 4> returns;
  for (const llvm::BasicBlock &block : *entry)
    if (const auto *returnInstruction =
            llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator()))
      returns.push_back(returnInstruction);
  if (returns.empty() || entryEndCalls->size() != returns.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target entry return bracketing is incomplete");
  for (const llvm::ReturnInst *returnInstruction : returns) {
    const auto *endCall = llvm::dyn_cast_or_null<llvm::CallBase>(
        returnInstruction->getPrevNode());
    if (!endCall ||
        endCall->getCalledOperand()->stripPointerCasts() != *entryEnd ||
        endCall->arg_size() != 0)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile target entry return is not immediately bracketed");
  }

  const bool expectsSites = capture == ProfileCaptureKind::Count ||
                            capture == ProfileCaptureKind::Trace;
  if (hasSiteBegin != expectsSites || hasSiteEnd != expectsSites)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile target module site runtime symbols do not match capture kind");
  if (!expectsSites)
    return llvm::Error::success();

  llvm::Expected<const llvm::Function *> siteBegin = requireExactDeclaration(
      kSiteBeginSymbol,
      llvm::FunctionType::get(voidType, {i32}, /*isVarArg=*/false));
  if (!siteBegin)
    return siteBegin.takeError();
  llvm::Expected<const llvm::Function *> siteEnd = requireExactDeclaration(
      kSiteEndSymbol,
      llvm::FunctionType::get(voidType, {i32}, /*isVarArg=*/false));
  if (!siteEnd)
    return siteEnd.takeError();
  llvm::Expected<llvm::SmallVector<const llvm::CallBase *, 16>> siteBeginCalls =
      collectDirectCalls(**siteBegin);
  if (!siteBeginCalls)
    return siteBeginCalls.takeError();
  llvm::Expected<llvm::SmallVector<const llvm::CallBase *, 16>> siteEndCalls =
      collectDirectCalls(**siteEnd);
  if (!siteEndCalls)
    return siteEndCalls.takeError();

  llvm::Expected<std::vector<CollectedProfileTargetCallSite>> sites =
      collectProfileTargetCallSitesImpl(module, entrySymbol);
  if (!sites)
    return sites.takeError();
  if (siteBeginCalls->size() != sites->size() ||
      siteEndCalls->size() != sites->size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile trace site bracketing does not cover the exact typed "
        "target-call site set");
  for (const CollectedProfileTargetCallSite &site : *sites) {
    const auto *beginCall =
        llvm::dyn_cast_or_null<llvm::CallBase>(site.call->getPrevNode());
    const auto *endCall =
        llvm::dyn_cast_or_null<llvm::CallBase>(site.call->getNextNode());
    const auto *beginId =
        beginCall && beginCall->arg_size() == 1
            ? llvm::dyn_cast<llvm::ConstantInt>(beginCall->getArgOperand(0))
            : nullptr;
    const auto *endId =
        endCall && endCall->arg_size() == 1
            ? llvm::dyn_cast<llvm::ConstantInt>(endCall->getArgOperand(0))
            : nullptr;
    if (!beginCall || !endCall ||
        beginCall->getCalledOperand()->stripPointerCasts() != *siteBegin ||
        endCall->getCalledOperand()->stripPointerCasts() != *siteEnd ||
        !beginId || !endId || beginId->getZExtValue() != site.record.siteId ||
        endId->getZExtValue() != site.record.siteId)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile trace site IDs do not densely bracket the exact typed "
          "target-call site set");
  }
  return llvm::Error::success();
}

} // namespace wafer::compiler::detail
