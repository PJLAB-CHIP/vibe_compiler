//===- TargetCallFrontend.cpp - Host target-call execution --------------===//

#include "Wafer/Compiler/TargetCallFrontend.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace wafer::compiler {
namespace {

constexpr llvm::StringLiteral kDispatcherSymbol = "wafer_target_call_dispatch";
constexpr llvm::StringLiteral kEntryThunkSymbol = "wafer_target_entry_thunk";

using HostEntryThunk = void(const uint64_t *);

struct InvocationContext {
  TargetCommandSink *sink = nullptr;
  std::optional<std::string> failure;
};

struct TileInvocationContext {
  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  int64_t tileCount;
  TargetIdentityId targetIdentity;
  InvocationContext *invocation;
  uint64_t nextIssueOrdinal = 0;
  uint64_t issuedCommandCount = 0;
};

struct MaterializedTile {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  HostEntryThunk *entry = nullptr;
};

enum class ExecutableState : uint8_t {
  Ready,
  Running,
  Completed,
  Aborted,
};

enum class TileExecutionState : uint8_t {
  NotStarted,
  Running,
  Completed,
};

extern "C" uint64_t waferTargetCallDispatch(uint64_t contextAddress,
                                            uint32_t descriptorIndex,
                                            const uint64_t *arguments,
                                            uint32_t argumentCount) {
  auto *context = reinterpret_cast<TileInvocationContext *>(
      static_cast<uintptr_t>(contextAddress));
  if (!context || !context->invocation || context->invocation->failure ||
      !context->invocation->sink)
    return 0;
  llvm::ArrayRef<TargetCallDescriptor> descriptors = getTargetCallDescriptors();
  if (descriptorIndex >= descriptors.size()) {
    context->invocation->failure =
        "target-call bridge used an invalid descriptor index";
    return 0;
  }
  const TargetCallDescriptor &descriptor = descriptors[descriptorIndex];
  if (argumentCount != descriptor.arguments.size()) {
    context->invocation->failure =
        "target-call bridge used an invalid argument count";
    return 0;
  }
  llvm::ArrayRef<uint64_t> argumentValues(arguments, argumentCount);
  llvm::Expected<target::TargetCommandPayload> payload =
      decodeTargetCallPayload(descriptor, {context->tileCount},
                              argumentValues);
  if (!payload) {
    context->invocation->failure = llvm::toString(payload.takeError());
    return 0;
  }
  llvm::Expected<std::optional<TargetNCCWorker>> worker =
      decodeTargetCallNCCWorker(descriptor, argumentValues);
  if (!worker) {
    context->invocation->failure = llvm::toString(worker.takeError());
    return 0;
  }
  TargetCommand command{context->cardId, context->tileId,
                        context->launchSlotId, context->nextIssueOrdinal++,
                        std::move(*payload)};
  if (descriptor.issueDomain && *worker)
    command.nccIssueDomain =
        TargetNCCIssueDomain{descriptor.issueDomain->engine, **worker,
                             descriptor.issueDomain->completionBehavior};
  llvm::Expected<uint64_t> issueResult =
      context->invocation->sink->issue(command);
  if (!issueResult) {
    context->invocation->failure = llvm::toString(issueResult.takeError());
    return 0;
  }
  ++context->issuedCommandCount;
  return descriptor.result == TargetCallResultType::I64 ? *issueResult : 0;
}

static bool hasNonZeroAddressSpace(llvm::Type *type) {
  auto *pointer = llvm::dyn_cast<llvm::PointerType>(type);
  return pointer && pointer->getAddressSpace() != 0;
}

static llvm::Error
verifyNativeInstruction(const llvm::Instruction &instruction) {
  if (const auto *result = llvm::dyn_cast<llvm::ReturnInst>(&instruction)) {
    if (!result->getReturnValue() ||
        result->getReturnValue()->getType()->isIntegerTy())
      return llvm::Error::success();
  }
  if (llvm::isa<llvm::BranchInst, llvm::SwitchInst>(instruction))
    return llvm::Error::success();
  if (const auto *phi = llvm::dyn_cast<llvm::PHINode>(&instruction)) {
    if (phi->getType()->isIntegerTy())
      return llvm::Error::success();
  }
  if (const auto *select = llvm::dyn_cast<llvm::SelectInst>(&instruction)) {
    if (select->getType()->isIntegerTy())
      return llvm::Error::success();
  }
  if (const auto *compare = llvm::dyn_cast<llvm::ICmpInst>(&instruction)) {
    if (compare->getOperand(0)->getType()->isIntegerTy())
      return llvm::Error::success();
  }

  if (const auto *binary = llvm::dyn_cast<llvm::BinaryOperator>(&instruction)) {
    if (binary->getType()->isIntegerTy())
      return llvm::Error::success();
  }
  if (const auto *cast = llvm::dyn_cast<llvm::CastInst>(&instruction)) {
    if (cast->getSrcTy()->isIntegerTy() && cast->getDestTy()->isIntegerTy())
      return llvm::Error::success();
  }
  if (const auto *freeze = llvm::dyn_cast<llvm::FreezeInst>(&instruction)) {
    if (freeze->getType()->isIntegerTy())
      return llvm::Error::success();
  }
  if (const auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
    if (llvm::isa<llvm::InlineAsm>(call->getCalledOperand()))
      return llvm::createStringError("target module contains inline assembly");
    llvm::Function *callee = call->getCalledFunction();
    if (!callee)
      return llvm::createStringError("target module contains an indirect call");
    if (callee->isIntrinsic())
      return llvm::createStringError(
          "target module contains unsupported intrinsic @%s",
          callee->getName().str().c_str());
    return llvm::Error::success();
  }

  return llvm::createStringError(
      "target module instruction '%s' is outside the closed native frontend",
      instruction.getOpcodeName());
}

static llvm::Error
verifyDescriptorType(const llvm::Function &function,
                     const TargetCallDescriptor &descriptor) {
  llvm::FunctionType *type = function.getFunctionType();
  if (type->isVarArg() || type->getNumParams() != descriptor.arguments.size())
    return llvm::createStringError(
        "target call @%s has an incompatible argument count",
        function.getName().str().c_str());
  bool resultMatches = descriptor.result == TargetCallResultType::Void
                           ? type->getReturnType()->isVoidTy()
                           : type->getReturnType()->isIntegerTy(64);
  if (!resultMatches)
    return llvm::createStringError(
        "target call @%s has an incompatible result type",
        function.getName().str().c_str());
  for (auto [parameter, scalar] :
       llvm::zip_equal(type->params(), descriptor.arguments)) {
    unsigned width = scalar == TargetCallScalarType::I64 ? 64 : 32;
    if (!parameter->isIntegerTy(width))
      return llvm::createStringError(
          "target call @%s has an incompatible argument width",
          function.getName().str().c_str());
  }
  if (function.getCallingConv() != llvm::CallingConv::C)
    return llvm::createStringError(
        "target call @%s does not use the C calling convention",
        function.getName().str().c_str());
  return llvm::Error::success();
}

static llvm::Error
verifyTargetModuleForHostExecution(const TargetLLVMModule &targetModule) {
  const llvm::Module &module = targetModule.getModule();
  llvm::Function *entry = module.getFunction(targetModule.getEntrySymbol());
  if (!entry || entry->isDeclaration())
    return llvm::createStringError("target entry is missing or undefined");
  llvm::FunctionType *entryType = entry->getFunctionType();
  if (entryType->isVarArg() || !entryType->getReturnType()->isVoidTy() ||
      entryType->getNumParams() != targetModule.getKernelABISlots().size())
    return llvm::createStringError(
        "target entry does not match its fixed Kernel Runtime ABI slots");
  for (llvm::Type *parameter : entryType->params())
    if (!parameter->isIntegerTy(64))
      return llvm::createStringError(
          "target entry ABI contains a non-i64 slot");
  if (entry->getCallingConv() != llvm::CallingConv::C)
    return llvm::createStringError(
        "target entry does not use the C calling convention");

  if (module.getNamedValue(kDispatcherSymbol) ||
      module.getNamedValue(kEntryThunkSymbol))
    return llvm::createStringError(
        "target module collides with a reserved host frontend symbol");
  if (!module.aliases().empty() || !module.ifuncs().empty())
    return llvm::createStringError(
        "target module contains an alias or indirect function");
  if (!module.globals().empty())
    return llvm::createStringError(
        "target module contains a global outside the closed native frontend");

  for (const llvm::Function &function : module) {
    if (function.isTargetIntrinsic())
      return llvm::createStringError(
          "target module contains a target-specific intrinsic");
    if (function.hasFnAttribute("target-cpu") ||
        function.hasFnAttribute("target-features"))
      return llvm::createStringError(
          "target module contains target-specific function attributes");
    if (function.hasPersonalityFn())
      return llvm::createStringError(
          "target module contains an unsupported personality function");
    if (function.isDeclaration() && !function.isIntrinsic()) {
      const TargetCallDescriptor *descriptor =
          findTargetCallDescriptor(function.getName());
      if (!descriptor)
        return llvm::createStringError(
            "target module contains unknown external call @%s",
            function.getName().str().c_str());
      if (llvm::Error error = verifyDescriptorType(function, *descriptor))
        return error;
    }
    for (const llvm::BasicBlock &block : function) {
      for (const llvm::Instruction &instruction : block) {
        if (hasNonZeroAddressSpace(instruction.getType()))
          return llvm::createStringError(
              "target module contains a nonzero result address space");
        for (const llvm::Use &operand : instruction.operands())
          if (hasNonZeroAddressSpace(operand->getType()))
            return llvm::createStringError(
                "target module contains a nonzero operand address space");
        if (llvm::Error error = verifyNativeInstruction(instruction))
          return error;
      }
    }
  }
  return llvm::Error::success();
}

static llvm::Expected<std::pair<std::unique_ptr<llvm::LLVMContext>,
                                std::unique_ptr<llvm::Module>>>
cloneIntoOwnedContext(const llvm::Module &module) {
  llvm::SmallVector<char, 0> bitcode;
  llvm::raw_svector_ostream output(bitcode);
  llvm::WriteBitcodeToFile(module, output);
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBufferCopy(
          llvm::StringRef(bitcode.data(), bitcode.size()), "target-call-clone");
  llvm::Expected<std::unique_ptr<llvm::Module>> parsed =
      llvm::parseBitcodeFile(buffer->getMemBufferRef(), *context);
  if (!parsed)
    return parsed.takeError();
  return std::make_pair(std::move(context), std::move(*parsed));
}

static llvm::Function *declareDispatcher(llvm::Module &module) {
  llvm::LLVMContext &context = module.getContext();
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  llvm::Type *pointer = llvm::PointerType::get(context, 0);
  llvm::FunctionType *type = llvm::FunctionType::get(
      i64, {i64, i32, pointer, i32}, /*isVarArg=*/false);
  return llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                kDispatcherSymbol, module);
}

static llvm::Error defineTargetCallBridges(llvm::Module &module,
                                           TileInvocationContext &context) {
  llvm::Function *dispatcher = declareDispatcher(module);
  llvm::Type *i64 = llvm::Type::getInt64Ty(module.getContext());
  llvm::Type *i32 = llvm::Type::getInt32Ty(module.getContext());
  llvm::ArrayRef<TargetCallDescriptor> descriptors = getTargetCallDescriptors();
  for (auto [descriptorIndex, descriptor] : llvm::enumerate(descriptors)) {
    llvm::Function *function = module.getFunction(descriptor.symbol);
    if (!function)
      continue;
    if (!function->isDeclaration())
      return llvm::createStringError(
          "target call bridge symbol is already defined");
    if (llvm::Error error = verifyDescriptorType(*function, descriptor))
      return error;
    llvm::BasicBlock *block =
        llvm::BasicBlock::Create(module.getContext(), "entry", function);
    llvm::IRBuilder<> builder(block);
    unsigned storageCount = std::max<size_t>(1, descriptor.arguments.size());
    llvm::Value *storage = builder.CreateAlloca(
        i64, llvm::ConstantInt::get(i32, storageCount), "call.arguments");
    for (auto [index, argument] : llvm::enumerate(function->args())) {
      llvm::Value *value = &argument;
      if (descriptor.arguments[index] == TargetCallScalarType::I32)
        value = builder.CreateZExt(value, i64);
      llvm::Value *slot = builder.CreateInBoundsGEP(
          i64, storage, llvm::ConstantInt::get(i64, index));
      builder.CreateStore(value, slot);
    }
    llvm::Value *result = builder.CreateCall(
        dispatcher, {llvm::ConstantInt::get(i64, reinterpret_cast<uintptr_t>(
                                                     std::addressof(context))),
                     llvm::ConstantInt::get(i32, descriptorIndex), storage,
                     llvm::ConstantInt::get(i32, descriptor.arguments.size())});
    if (descriptor.result == TargetCallResultType::Void)
      builder.CreateRetVoid();
    else
      builder.CreateRet(result);
  }
  return llvm::Error::success();
}

static llvm::Error defineEntryThunk(llvm::Module &module,
                                    llvm::StringRef entrySymbol,
                                    size_t slotCount) {
  llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry)
    return llvm::createStringError("target entry disappeared during clone");
  llvm::LLVMContext &context = module.getContext();
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Type *pointer = llvm::PointerType::get(context, 0);
  llvm::FunctionType *type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), {pointer}, /*isVarArg=*/false);
  llvm::Function *thunk = llvm::Function::Create(
      type, llvm::GlobalValue::ExternalLinkage, kEntryThunkSymbol, module);
  llvm::BasicBlock *block = llvm::BasicBlock::Create(context, "entry", thunk);
  llvm::IRBuilder<> builder(block);
  llvm::Value *slots = thunk->getArg(0);
  llvm::SmallVector<llvm::Value *, 16> arguments;
  arguments.reserve(slotCount);
  for (size_t index = 0; index < slotCount; ++index) {
    llvm::Value *slot = builder.CreateInBoundsGEP(
        i64, slots, llvm::ConstantInt::get(i64, index));
    arguments.push_back(builder.CreateLoad(i64, slot));
  }
  builder.CreateCall(entry, arguments);
  builder.CreateRetVoid();
  return llvm::Error::success();
}

static llvm::Error initializeNativeBackend() {
  static std::once_flag flag;
  static std::optional<std::string> initializationFailure;
  std::call_once(flag, [] {
    if (llvm::InitializeNativeTarget()) {
      initializationFailure = "failed to initialize the native LLVM target";
      return;
    }
    if (llvm::InitializeNativeTargetAsmPrinter())
      initializationFailure =
          "failed to initialize the native LLVM assembly printer";
  });
  if (initializationFailure)
    return llvm::createStringError(*initializationFailure);
  return llvm::Error::success();
}

static llvm::Expected<MaterializedTile>
materializeTile(const TargetLLVMModule &targetModule,
                TileInvocationContext &context) {
  if (llvm::Error error = verifyTargetModuleForHostExecution(targetModule))
    return std::move(error);
  llvm::Expected<std::pair<std::unique_ptr<llvm::LLVMContext>,
                           std::unique_ptr<llvm::Module>>>
      owned = cloneIntoOwnedContext(targetModule.getModule());
  if (!owned)
    return owned.takeError();

  llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> jit =
      llvm::orc::LLJITBuilder().create();
  if (!jit)
    return jit.takeError();
  owned->second->setTargetTriple((*jit)->getTargetTriple().str());
  owned->second->setDataLayout((*jit)->getDataLayout());
  if (llvm::Error error = defineTargetCallBridges(*owned->second, context))
    return std::move(error);
  if (llvm::Error error =
          defineEntryThunk(*owned->second, targetModule.getEntrySymbol(),
                           targetModule.getKernelABISlots().size()))
    return std::move(error);
  std::string verificationDiagnostic;
  llvm::raw_string_ostream verificationStream(verificationDiagnostic);
  if (llvm::verifyModule(*owned->second, &verificationStream))
    return llvm::createStringError("host target-call module is invalid: %s",
                                   verificationStream.str().c_str());

  llvm::orc::MangleAndInterner mangle((*jit)->getExecutionSession(),
                                      (*jit)->getDataLayout());
  llvm::orc::SymbolMap symbols;
  symbols[mangle(kDispatcherSymbol)] = llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr::fromPtr(&waferTargetCallDispatch),
      llvm::JITSymbolFlags(llvm::JITSymbolFlags::Exported |
                           llvm::JITSymbolFlags::Callable));
  if (llvm::Error error = (*jit)->getMainJITDylib().define(
          llvm::orc::absoluteSymbols(std::move(symbols))))
    return std::move(error);

  llvm::orc::ThreadSafeModule threadSafeModule(std::move(owned->second),
                                               std::move(owned->first));
  if (llvm::Error error = (*jit)->addIRModule(std::move(threadSafeModule)))
    return std::move(error);
  llvm::Expected<llvm::orc::ExecutorAddr> entry =
      (*jit)->lookup(kEntryThunkSymbol);
  if (!entry)
    return entry.takeError();
  return MaterializedTile{std::move(*jit), entry->toPtr<HostEntryThunk>()};
}

static llvm::Expected<TargetCallInvocationDescriptor>
validateInvocation(const TargetLLVMModules &targetLLVMModules,
                   llvm::ArrayRef<TargetCallTileArguments> arguments) {
  const std::vector<TargetLLVMModule> &modules = targetLLVMModules.getModules();
  if (modules.empty() || modules.size() != arguments.size() ||
      modules.size() !=
          static_cast<size_t>(
              targetLLVMModules.getExecutionConfig().getTileCount()))
    return llvm::createStringError(
        "target-call invocation does not cover the complete Tile "
        "domain");
  TargetCallInvocationDescriptor descriptor{
      targetLLVMModules.getExecutionConfig().getTargetIdentityId(), {}};
  descriptor.tiles.reserve(modules.size());
  std::set<std::pair<int64_t, int64_t>> tileEndpoints;
  for (size_t index = 0; index < modules.size(); ++index) {
    const TargetLLVMModule &module = modules[index];
    const TargetCallTileArguments &tileArguments = arguments[index];
    const LaunchSlotId expectedLaunchSlot(static_cast<int64_t>(index));
    const std::pair<int64_t, int64_t> tileEndpoint = {
        module.getCardId().getValue(),
        module.getTileId().getValue()};
    if (tileEndpoint.first < 0 || tileEndpoint.second < 0 ||
        !tileEndpoints.insert(tileEndpoint).second)
      return llvm::createStringError(
          "target-call invocation contains an invalid or duplicate physical "
          "Tile identity");
    if (module.getLaunchSlotId() != expectedLaunchSlot ||
        tileArguments.launchSlotId != module.getLaunchSlotId() ||
        tileArguments.cardId != module.getCardId() ||
        tileArguments.tileId != module.getTileId())
      return llvm::createStringError(
          "target-call invocation Tile identity or launch-slot order "
          "is not canonical");
    if (tileArguments.slots.size() != module.getKernelABISlots().size())
      return llvm::createStringError(
          "target-call invocation slot count does not match the typed ABI");
    if (module.getTargetIdentityId() != descriptor.targetIdentity)
      return llvm::createStringError(
          "target-call invocation contains inconsistent target identities");
    descriptor.tiles.push_back(
        {module.getCardId(), module.getTileId(),
         module.getLaunchSlotId(), module.getKernelABISlots(),
         tileArguments.slots, module.getTargetIdentityId(),
         module.getKernelRuntimeABIId()});
  }
  return descriptor;
}

} // namespace

struct TargetCallExecutable::Impl {
  Impl(TargetCallInvocationDescriptor descriptor,
       llvm::ArrayRef<TargetCallTileArguments> tileArguments)
      : descriptor(std::move(descriptor)),
        arguments(tileArguments.begin(), tileArguments.end()),
        tileStates(arguments.size(), TileExecutionState::NotStarted) {}

  TargetCallInvocationDescriptor descriptor;
  std::vector<TargetCallTileArguments> arguments;
  InvocationContext invocation;
  std::vector<TileInvocationContext> contexts;
  std::vector<MaterializedTile> materialized;
  std::vector<TileExecutionState> tileStates;
  ExecutableState state = ExecutableState::Ready;
};

TargetCallExecutable::TargetCallExecutable(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

TargetCallExecutable::~TargetCallExecutable() {
  if (impl && impl->state == ExecutableState::Running)
    abort("running target-call executable was destroyed before finish");
}
TargetCallExecutable::TargetCallExecutable(TargetCallExecutable &&) = default;
TargetCallExecutable &
TargetCallExecutable::operator=(TargetCallExecutable &&other) {
  if (this == &other)
    return *this;
  if (impl && impl->state == ExecutableState::Running)
    abort("running target-call executable was replaced before finish");
  impl = std::move(other.impl);
  return *this;
}

const TargetCallInvocationDescriptor &
TargetCallExecutable::getInvocationDescriptor() const {
  assert(impl && "moved-from target-call executable has no descriptor");
  return impl->descriptor;
}

void TargetCallExecutable::abort(llvm::StringRef diagnostic) {
  if (!impl || impl->state == ExecutableState::Completed ||
      impl->state == ExecutableState::Aborted)
    return;
  impl->invocation.failure = diagnostic.str();
  impl->state = ExecutableState::Aborted;
  if (impl->invocation.sink)
    impl->invocation.sink->abort(diagnostic);
}

llvm::Error TargetCallExecutable::begin(TargetCommandSink &sink) {
  if (!impl || impl->state != ExecutableState::Ready)
    return llvm::createStringError(
        "target-call executable cannot begin from its current state");
  impl->invocation.sink = &sink;
  if (llvm::Error error = sink.begin(impl->descriptor)) {
    std::string diagnostic = llvm::toString(std::move(error));
    abort(diagnostic);
    return llvm::createStringError("%s", diagnostic.c_str());
  }
  impl->state = ExecutableState::Running;
  return llvm::Error::success();
}

llvm::Error TargetCallExecutable::executeTile(LaunchSlotId launchSlotId) {
  if (!impl || impl->state != ExecutableState::Running)
    return llvm::createStringError(
        "target-call Tile cannot execute outside a running invocation");
  const int64_t launchSlot = launchSlotId.getValue();
  if (launchSlot < 0 ||
      launchSlot >= static_cast<int64_t>(impl->materialized.size())) {
    std::string diagnostic =
        "target-call launch slot is outside the materialized Tile domain";
    abort(diagnostic);
    return llvm::createStringError("%s", diagnostic.c_str());
  }
  size_t index = static_cast<size_t>(launchSlot);
  if (impl->tileStates[index] != TileExecutionState::NotStarted) {
    std::string diagnostic =
        impl->tileStates[index] == TileExecutionState::Running
            ? "target-call Tile is already running"
            : "target-call Tile is already complete";
    abort(diagnostic);
    return llvm::createStringError("%s", diagnostic.c_str());
  }

  impl->tileStates[index] = TileExecutionState::Running;
  impl->materialized[index].entry(impl->arguments[index].slots.data());
  if (impl->invocation.failure) {
    std::string diagnostic = *impl->invocation.failure;
    abort(diagnostic);
    return llvm::createStringError("%s", diagnostic.c_str());
  }
  const TargetCallTileDescriptor &tile = impl->descriptor.tiles[index];
  if (llvm::Error error = impl->invocation.sink->completeTile(
          tile.cardId, tile.tileId, tile.launchSlotId)) {
    std::string diagnostic = llvm::toString(std::move(error));
    abort(diagnostic);
    return llvm::createStringError("%s", diagnostic.c_str());
  }
  impl->tileStates[index] = TileExecutionState::Completed;
  return llvm::Error::success();
}

llvm::Expected<TargetCallExecutionResult> TargetCallExecutable::finish() {
  if (!impl || impl->state != ExecutableState::Running)
    return llvm::createStringError(
        "target-call executable cannot finish from its current state");
  if (!llvm::all_of(impl->tileStates, [](TileExecutionState state) {
        return state == TileExecutionState::Completed;
      })) {
    std::string diagnostic =
        "target-call executable cannot finish before every Tile completes";
    abort(diagnostic);
    return llvm::createStringError("%s", diagnostic.c_str());
  }
  if (llvm::Error error = impl->invocation.sink->completeInvocation()) {
    std::string diagnostic = llvm::toString(std::move(error));
    abort(diagnostic);
    return llvm::createStringError("%s", diagnostic.c_str());
  }
  impl->state = ExecutableState::Completed;
  uint64_t issuedCommandCount = 0;
  for (const TileInvocationContext &context : impl->contexts)
    issuedCommandCount += context.issuedCommandCount;
  return TargetCallExecutionResult{
      static_cast<int64_t>(impl->materialized.size()), issuedCommandCount};
}

llvm::Expected<TargetCallExecutable>
createTargetCallExecutable(const TargetLLVMModules &targetLLVMModules,
                           llvm::ArrayRef<TargetCallTileArguments> arguments) {
  llvm::Expected<TargetCallInvocationDescriptor> invocation =
      validateInvocation(targetLLVMModules, arguments);
  if (!invocation)
    return invocation.takeError();
  if (llvm::Error error = initializeNativeBackend())
    return std::move(error);

  auto executable = std::make_unique<TargetCallExecutable::Impl>(
      std::move(*invocation), arguments);
  executable->contexts.reserve(targetLLVMModules.getModules().size());
  for (const TargetLLVMModule &module : targetLLVMModules.getModules())
    executable->contexts.push_back(
        {module.getCardId(), module.getTileId(),
         module.getLaunchSlotId(),
         targetLLVMModules.getExecutionConfig().getTileCount(),
         module.getTargetIdentityId(), &executable->invocation});

  executable->materialized.reserve(targetLLVMModules.getModules().size());
  for (size_t index = 0; index < targetLLVMModules.getModules().size();
       ++index) {
    llvm::Expected<MaterializedTile> tile = materializeTile(
        targetLLVMModules.getModules()[index], executable->contexts[index]);
    if (!tile)
      return tile.takeError();
    executable->materialized.push_back(std::move(*tile));
  }
  return TargetCallExecutable(std::move(executable));
}

llvm::Expected<TargetCallExecutionResult>
executeTargetCallFrontend(const TargetLLVMModules &targetLLVMModules,
                          llvm::ArrayRef<TargetCallTileArguments> arguments,
                          TargetCommandSink &sink) {
  llvm::Expected<TargetCallExecutable> executable =
      createTargetCallExecutable(targetLLVMModules, arguments);
  if (!executable)
    return executable.takeError();
  if (llvm::Error error = executable->begin(sink))
    return std::move(error);
  for (const TargetCallTileDescriptor &tile :
       executable->getInvocationDescriptor().tiles)
    if (llvm::Error error = executable->executeTile(tile.launchSlotId))
      return std::move(error);
  return executable->finish();
}

} // namespace wafer::compiler
