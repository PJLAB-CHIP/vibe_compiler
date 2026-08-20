//===- SystemCTargetModelTestSupport.cpp - Source DTE test support --------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "Wafer/CodeGen/Target/TargetCodeGenInternal.h"
#include "Wafer/Target/Core/RuntimeLaunchContract.h"
#include "Wafer/Target/Core/TargetFormat.h"
#include "Wafer/Target/Layout/PhysicalTensorCodec.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::model::test {
namespace {

std::vector<RawLogicalValue> makeTileValues(int64_t tileId) {
  std::vector<RawLogicalValue> values;
  values.reserve(4);
  for (uint64_t index = 0; index < 4; ++index) {
    // Exact finite f32 values in [1, 1.5), unique across the 64 elements.
    const uint64_t element = static_cast<uint64_t>(tileId) * 4 + index;
    values.push_back(
        {LogicalFormat::F32, UINT64_C(0x3f800000) + (element << 15)});
  }
  return values;
}

llvm::Function *getOrDeclareTargetCall(llvm::Module &module,
                                       const TargetCallDescriptor &descriptor) {
  if (llvm::Function *function = module.getFunction(descriptor.symbol))
    return function;
  llvm::SmallVector<llvm::Type *, 16> arguments;
  for (TargetCallScalarType scalar : descriptor.arguments)
    arguments.push_back(scalar == TargetCallScalarType::I64
                            ? llvm::Type::getInt64Ty(module.getContext())
                            : llvm::Type::getInt32Ty(module.getContext()));
  llvm::Type *result = descriptor.result == TargetCallResultType::Void
                           ? llvm::Type::getVoidTy(module.getContext())
                           : llvm::Type::getInt64Ty(module.getContext());
  llvm::Function *function = llvm::Function::Create(
      llvm::FunctionType::get(result, arguments, /*isVarArg=*/false),
      llvm::GlobalValue::ExternalLinkage, descriptor.symbol, module);
  function->setCallingConv(llvm::CallingConv::C);
  return function;
}

llvm::CallInst *emitTargetCall(llvm::IRBuilder<> &builder,
                               const TargetCallDescriptor &descriptor,
                               llvm::ArrayRef<llvm::Value *> arguments) {
  llvm::CallInst *call = builder.CreateCall(
      getOrDeclareTargetCall(*builder.GetInsertBlock()->getModule(),
                             descriptor),
      arguments);
  call->setCallingConv(llvm::CallingConv::C);
  return call;
}

compiler::TileEntryArgument
makeTensorSlot(int64_t ordinal, compiler::TileEntryArgumentKind kind,
               compiler::TileEntryArgumentAccess access, llvm::StringRef name) {
  return {ordinal,           kind, 0,      name.str(), LogicalFormat::F32,
          MemLayout::Tensor, {64}, 64 * 4, 64,         access};
}

compiler::TileEntryArgument makeTransportStatusSlot(int64_t ordinal) {
  return {ordinal,
          compiler::TileEntryArgumentKind::TransportStatus,
          0,
          "direct_dte_status",
          LogicalFormat::U32,
          MemLayout::Tensor,
          {1},
          WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_BYTES,
          WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_ALIGNMENT,
          compiler::TileEntryArgumentAccess::ReadWrite};
}

} // namespace

llvm::Expected<compiler::TargetLLVMModules>
compileDirectDTETargetModules(std::string &diagnosticText,
                              TargetIdentityId targetIdentity) {
  diagnosticText.clear();
  llvm::Expected<compiler::ExecutionConfig> config =
      compiler::ExecutionConfig::createForSingleCard(1);
  if (!config)
    return config.takeError();
  if (config->getTargetIdentityId() != targetIdentity)
    return llvm::createStringError(
        "Direct-DTE model fixture target identity is not current");
  constexpr std::array phases{RuntimeLaunchPhaseRole::Main};
  llvm::Expected<RuntimeLaunchContract> launch =
      RuntimeLaunchContract::createKernel(KernelLaunchForm::Grid,
                                          KernelEntryABI::TileMajorPointerTable,
                                          phases);
  if (!launch)
    return launch.takeError();
  const TargetDataFormatCodeRecord *format =
      findTargetDataFormatCode(LogicalFormat::F32);
  if (!format)
    return llvm::createStringError("current target has no F32 format code");

  const TargetCallDescriptor &begin =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTEBegin);
  const TargetCallDescriptor &rdma =
      getTargetCallDescriptor(TargetCallBuiltin::RDMA);
  const TargetCallDescriptor &receive =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTERecvPrepare);
  const TargetCallDescriptor &send =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTESendPrepare);
  const TargetCallDescriptor &join =
      getTargetCallDescriptor(TargetCallBuiltin::NCCJoin);
  const TargetCallDescriptor &issue =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTESendIssue);
  const TargetCallDescriptor &wait =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTEWait);
  const TargetCallDescriptor &wdma =
      getTargetCallDescriptor(TargetCallBuiltin::WDMA);
  const TargetCallDescriptor &finish =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTEFinish);

  constexpr int64_t tileCount = 16;
  constexpr uint64_t tileBytes = 16;
  constexpr uint64_t sendAddress = UINT64_C(0x10000);
  constexpr uint64_t receiveAddress = UINT64_C(0x11000);
  constexpr uint32_t worker = static_cast<uint32_t>(TargetNCCWorker::Worker0);
  constexpr uint32_t workerMask = uint32_t{1} << worker;
  std::vector<compiler::TargetLLVMModule> modules;
  modules.reserve(tileCount);
  for (int64_t tile = 0; tile < tileCount; ++tile) {
    std::vector<compiler::TileEntryArgument> slots;
    slots.push_back(
        makeTensorSlot(0, compiler::TileEntryArgumentKind::ExternalInput,
                       compiler::TileEntryArgumentAccess::ReadOnly, "input"));
    slots.push_back(
        makeTensorSlot(1, compiler::TileEntryArgumentKind::ExternalOutput,
                       compiler::TileEntryArgumentAccess::WriteOnly, "output"));
    slots.push_back(makeTransportStatusSlot(2));

    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("direct-dte-model", *context);
    llvm::Type *i64 = llvm::Type::getInt64Ty(*context);
    llvm::SmallVector<llvm::Type *, 3> entryArguments(slots.size(), i64);
    llvm::Function *entry = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(*context), entryArguments,
                                /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "main", *module);
    entry->setCallingConv(llvm::CallingConv::C);
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(*context, "entry", entry));
    llvm::Value *inputAddress = builder.CreateAdd(
        entry->getArg(0),
        builder.getInt64(static_cast<uint64_t>(tile) * tileBytes));
    llvm::Value *outputAddress = builder.CreateAdd(
        entry->getArg(1),
        builder.getInt64(static_cast<uint64_t>(tile) * tileBytes));
    emitTargetCall(builder, begin,
                   {entry->getArg(2), builder.getInt32(tileCount)});
    emitTargetCall(
        builder, rdma,
        {inputAddress, builder.getInt64(sendAddress),
         builder.getInt32(tileBytes), builder.getInt32(tileBytes),
         builder.getInt32(0), builder.getInt32(0), builder.getInt32(0),
         builder.getInt32(1), builder.getInt32(1), builder.getInt32(1),
         builder.getInt32(format->dataFormatCode), builder.getInt32(worker)});
    llvm::CallInst *receiveEvent =
        emitTargetCall(builder, receive,
                       {builder.getInt64(receiveAddress),
                        builder.getInt32(tileBytes), builder.getInt32(tile),
                        builder.getInt32(tile ^ 1), builder.getInt32(0)});
    llvm::CallInst *sendEvent = emitTargetCall(
        builder, send,
        {builder.getInt64(sendAddress), builder.getInt64(receiveAddress),
         builder.getInt32(tileBytes), builder.getInt32(tile),
         builder.getInt32(tile ^ 1), builder.getInt32(0), builder.getInt32(0)});
    emitTargetCall(builder, join, {builder.getInt32(workerMask)});
    emitTargetCall(builder, issue, {sendEvent});
    emitTargetCall(builder, wait, {sendEvent});
    emitTargetCall(builder, wait, {receiveEvent});
    emitTargetCall(
        builder, wdma,
        {builder.getInt64(receiveAddress), outputAddress,
         builder.getInt32(tileBytes), builder.getInt32(tileBytes),
         builder.getInt32(0), builder.getInt32(0), builder.getInt32(0),
         builder.getInt32(1), builder.getInt32(1), builder.getInt32(1),
         builder.getInt32(format->dataFormatCode), builder.getInt32(worker)});
    emitTargetCall(builder, join, {builder.getInt32(workerMask)});
    emitTargetCall(builder, finish, {});
    builder.CreateRetVoid();

    std::string verification;
    llvm::raw_string_ostream verificationStream(verification);
    if (llvm::verifyModule(*module, &verificationStream))
      return llvm::createStringError(
          "Direct-DTE model fixture produced invalid LLVM IR: " +
          verificationStream.str());
    modules.push_back(compiler::TargetLLVMModulesBuilder::makeModule(
        CardId(0), TileId(tile), LaunchSlotId(tile), "main", targetIdentity,
        kCurrentKernelRuntimeABI, kCurrentTargetModuleFormat, std::move(slots),
        std::move(context), std::move(module)));
  }
  return compiler::TargetLLVMModulesBuilder::makeModules(
      *config, std::move(*launch), std::move(modules));
}

llvm::Expected<DirectDTEInvocationData> buildDirectDTEInvocationData(
    const compiler::TargetLLVMModules &targetLLVMModules) {
  if (targetLLVMModules.getModules().size() != 16)
    return llvm::createStringError(
        "Direct-DTE test targetLLVMModules must contain exactly 16 ranks");

  DirectDTEInvocationData result;
  result.arguments.reserve(16);
  PhysicalTensorDescriptor tensorKey =
      llvm::cantFail(PhysicalTensorDescriptor::create(
          LogicalFormat::F32, PhysicalTensorLayout::Tensor, {64}));
  std::vector<RawLogicalValue> inputValues;
  std::vector<RawLogicalValue> expectedValues;
  inputValues.reserve(64);
  expectedValues.reserve(64);
  for (int64_t tile = 0; tile < 16; ++tile) {
    std::vector<RawLogicalValue> values = makeTileValues(tile);
    inputValues.insert(inputValues.end(), values.begin(), values.end());
    values = makeTileValues(tile ^ 1);
    expectedValues.insert(expectedValues.end(), values.begin(), values.end());
  }
  llvm::Expected<std::vector<uint8_t>> inputBytes =
      packPhysicalTensorLogicalValues(tensorKey, inputValues, UINT8_C(0));
  llvm::Expected<std::vector<uint8_t>> expectedBytes =
      packPhysicalTensorLogicalValues(tensorKey, expectedValues, UINT8_C(0));
  if (!inputBytes || !expectedBytes) {
    llvm::Error errors = llvm::Error::success();
    if (!inputBytes)
      errors = llvm::joinErrors(std::move(errors), inputBytes.takeError());
    if (!expectedBytes)
      errors = llvm::joinErrors(std::move(errors), expectedBytes.takeError());
    return std::move(errors);
  }
  result.expectedOutputBytes = std::move(*expectedBytes);
  for (const compiler::TargetLLVMModule &module :
       targetLLVMModules.getModules()) {
    const int64_t tile = module.getTileId().getValue();
    if (tile < 0 || tile >= 16)
      return llvm::createStringError("Direct-DTE test target LLVM Tile "
                                     "is outside the canonical domain");
    compiler::TargetCallTileArguments arguments{
        module.getCardId(), module.getTileId(), module.getLaunchSlotId(), {}};
    size_t userInputCount = 0;
    for (const compiler::TileEntryArgument &slot :
         module.getTileEntryArguments()) {
      if (slot.ordinal < 0 || slot.byteSize <= 0 || slot.byteSize >= 0x10000)
        return llvm::createStringError(
            "Direct-DTE test slot is outside its synthetic DDR stride");
      uint64_t base = 0;
      switch (slot.kind) {
      case compiler::TileEntryArgumentKind::ExternalInput:
        base = UINT64_C(0x10000000);
        break;
      case compiler::TileEntryArgumentKind::ExternalOutput:
        base = UINT64_C(0x10010000);
        break;
      case compiler::TileEntryArgumentKind::TransportStatus:
        base = UINT64_C(0x20000000) +
               static_cast<uint64_t>(tile) * UINT64_C(0x10000);
        break;
      case compiler::TileEntryArgumentKind::TargetTensor:
      case compiler::TileEntryArgumentKind::Workspace:
      case compiler::TileEntryArgumentKind::ProfileRecord:
        return llvm::createStringError(
            "Direct-DTE model fixture has an unexpected ABI slot");
      }
      arguments.slots.push_back(base);
      if (slot.kind == compiler::TileEntryArgumentKind::ExternalInput) {
        ++userInputCount;
        if (inputBytes->size() != static_cast<uint64_t>(slot.byteSize))
          return llvm::createStringError(
              "Direct-DTE test input bytes disagree with the typed ABI slot");
        if (tile == 0)
          result.inputBindings.push_back(
              {getTargetModelResourceId(module.getCardId(), module.getTileId(),
                                        slot.kind, slot.resourceIndex),
               *inputBytes});
      }
    }
    if (userInputCount != 1)
      return llvm::createStringError(
          "Direct-DTE model fixture must have one input per Tile ABI");
    result.arguments.push_back(std::move(arguments));
  }
  return result;
}

llvm::Expected<NCCJoinRewriteResult>
rewriteNCCJoinsAfter(compiler::TargetLLVMModules &targetLLVMModules,
                     TargetCallBuiltin anchor) {
  const TargetCallDescriptor &anchorDescriptor =
      getTargetCallDescriptor(anchor);
  const TargetCallDescriptor &joinDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::NCCJoin);
  NCCJoinRewriteResult result;

  for (const compiler::TargetLLVMModule &targetModule :
       targetLLVMModules.getModules()) {
    llvm::Module &module = const_cast<llvm::Module &>(targetModule.getModule());
    llvm::SmallVector<llvm::CallInst *, 8> anchors;
    llvm::SmallVector<llvm::CallInst *, 8> joins;
    llvm::SmallVector<llvm::ReturnInst *, 4> returns;
    for (llvm::Function &function : module)
      for (llvm::BasicBlock &block : function)
        for (llvm::Instruction &instruction : block) {
          if (function.getName() == targetModule.getEntrySymbol())
            if (auto *returnInstruction =
                    llvm::dyn_cast<llvm::ReturnInst>(&instruction))
              returns.push_back(returnInstruction);
          auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
          llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
          if (!callee)
            continue;
          if (callee->getName() == anchorDescriptor.symbol)
            anchors.push_back(call);
          if (callee->getName() == joinDescriptor.symbol)
            joins.push_back(call);
        }

    result.erasedJoinCount += joins.size();
    for (llvm::CallInst *join : joins)
      join->eraseFromParent();
    if (anchors.empty())
      continue;

    llvm::Function *join = module.getFunction(joinDescriptor.symbol);
    if (!join) {
      llvm::FunctionType *joinType =
          llvm::FunctionType::get(llvm::Type::getVoidTy(module.getContext()),
                                  {llvm::Type::getInt32Ty(module.getContext())},
                                  /*isVarArg=*/false);
      join =
          llvm::Function::Create(joinType, llvm::GlobalValue::ExternalLinkage,
                                 joinDescriptor.symbol, module);
      join->setCallingConv(llvm::CallingConv::C);
    }
    for (llvm::CallInst *anchorCall : anchors) {
      llvm::Instruction *next = anchorCall->getNextNode();
      if (!next)
        return llvm::createStringError(
            "Direct-DTE anchor call has no following insertion point");
      llvm::IRBuilder<> builder(next);
      builder.CreateCall(join,
                         {builder.getInt32(uint32_t{1} << static_cast<uint32_t>(
                                               TargetNCCWorker::Worker0))});
      ++result.insertedJoinCount;
    }
    for (llvm::ReturnInst *returnInstruction : returns) {
      llvm::IRBuilder<> builder(returnInstruction);
      builder.CreateCall(join,
                         {builder.getInt32(uint32_t{1} << static_cast<uint32_t>(
                                               TargetNCCWorker::Worker0))});
      ++result.insertedTerminalJoinCount;
    }
  }
  if (result.insertedJoinCount == 0)
    return llvm::createStringError(
        "Direct-DTE targetLLVMModules has no requested join anchor call");
  return result;
}

llvm::Expected<PendingComputeDTERewriteResult>
insertPendingComputeBeforeDTEReceive(
    compiler::TargetLLVMModules &targetLLVMModules,
    PendingComputeDTEAccessMode accessMode) {
  const TargetIdentityId targetIdentity =
      targetLLVMModules.getExecutionConfig().getTargetIdentityId();
  const TargetCallDescriptor &receiveDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTERecvPrepare);
  const TargetCallDescriptor &elementwiseDescriptor =
      getTargetCallDescriptor(TargetElementwiseOperation::Add);
  const TargetCallDescriptor &gemmDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::Gemm);
  const TargetCallDescriptor &joinDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::NCCJoin);
  const TargetCallDescriptor &waitDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTEWait);
  const TargetDataFormatCodeRecord *format =
      findTargetDataFormatCode(LogicalFormat::F32);
  if (!format)
    return llvm::createStringError(
        "current target has no F32 data-format code");
  if (elementwiseDescriptor.arguments.size() != 6 ||
      gemmDescriptor.arguments.size() != 9 ||
      joinDescriptor.arguments.size() != 1)
    return llvm::createStringError(
        "pending-compute test descriptors have unexpected signatures");

  PendingComputeDTERewriteResult result;
  for (const compiler::TargetLLVMModule &targetModule :
       targetLLVMModules.getModules()) {
    llvm::Module &module = const_cast<llvm::Module &>(targetModule.getModule());
    llvm::SmallVector<llvm::CallInst *, 2> receives;
    for (llvm::Function &function : module)
      for (llvm::BasicBlock &block : function)
        for (llvm::Instruction &instruction : block)
          if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
            if (llvm::Function *callee = call->getCalledFunction())
              if (callee->getName() == receiveDescriptor.symbol)
                receives.push_back(call);
    for (llvm::CallInst *receive : receives) {
      auto *receiveDestination =
          llvm::dyn_cast<llvm::ConstantInt>(receive->getArgOperand(0));
      if (!receiveDestination)
        return llvm::createStringError(
            "Direct-DTE receive destination is not a static SPM address");
      const uint64_t lhs = accessMode == PendingComputeDTEAccessMode::Disjoint
                               ? UINT64_C(0x2d0000)
                               : receiveDestination->getZExtValue();
      if (lhs == receiveDestination->getZExtValue())
        ++result.overlappingReadCount;
      constexpr uint64_t rhs = UINT64_C(0x2d1000);
      constexpr uint64_t destination = UINT64_C(0x2d2000);

      llvm::IRBuilder<> builder(receive);
      llvm::CallInst *compute = nullptr;
      if ((targetModule.getTileId().getValue() & 1) == 0) {
        llvm::Function *elementwise =
            getOrDeclareTargetCall(module, elementwiseDescriptor);
        compute = builder.CreateCall(
            elementwise, {builder.getInt64(lhs), builder.getInt64(rhs),
                          builder.getInt64(destination), builder.getInt32(4),
                          builder.getInt32(format->dataFormatCode),
                          builder.getInt32(static_cast<uint32_t>(
                              TargetNCCWorker::Worker0))});
        ++result.elementwiseCount;
      } else {
        llvm::Function *gemm = getOrDeclareTargetCall(module, gemmDescriptor);
        compute = builder.CreateCall(
            gemm,
            {builder.getInt64(lhs), builder.getInt64(rhs),
             builder.getInt64(destination), builder.getInt32(2),
             builder.getInt32(2), builder.getInt32(2), builder.getInt32(1),
             builder.getInt32(format->dataFormatCode),
             builder.getInt32(
                 static_cast<uint32_t>(TargetNCCWorker::Worker0))});
        ++result.gemmCount;
      }
      compute->setCallingConv(llvm::CallingConv::C);

      llvm::Function *join = getOrDeclareTargetCall(module, joinDescriptor);
      llvm::IRBuilder<> setupBuilder(compute);
      llvm::CallInst *setupJoin = setupBuilder.CreateCall(
          join, {setupBuilder.getInt32(uint32_t{1} << static_cast<uint32_t>(
                                           TargetNCCWorker::Worker0))});
      setupJoin->setCallingConv(llvm::CallingConv::C);
      ++result.insertedSetupJoinCount;

      bool foundWait = false;
      for (llvm::Instruction *cursor = compute->getNextNode(); cursor;) {
        llvm::Instruction *next = cursor->getNextNode();
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(cursor)) {
          llvm::Function *callee = call->getCalledFunction();
          if (callee && callee->getName() == waitDescriptor.symbol) {
            foundWait = true;
            break;
          }
          if (callee && callee->getName() == joinDescriptor.symbol) {
            call->eraseFromParent();
            ++result.removedInterveningJoinCount;
          }
        }
        cursor = next;
      }
      if (!foundWait)
        return llvm::createStringError(
            "injected pending compute has no following Direct-DTE wait");

      if (accessMode == PendingComputeDTEAccessMode::OverlapWithPreIssueJoin) {
        llvm::CallInst *joinCall = builder.CreateCall(
            join, {builder.getInt32(uint32_t{1} << static_cast<uint32_t>(
                                        TargetNCCWorker::Worker0))});
        joinCall->setCallingConv(llvm::CallingConv::C);
        ++result.insertedJoinCount;
      }
    }
    std::string verification;
    llvm::raw_string_ostream stream(verification);
    if (llvm::verifyModule(module, &stream))
      return llvm::createStringError(
          "pending-compute Direct-DTE rewrite produced invalid LLVM IR: " +
          stream.str());
  }
  if (result.elementwiseCount == 0 || result.gemmCount == 0)
    return llvm::createStringError(
        "pending-compute Direct-DTE rewrite did not cover both families");
  return result;
}

llvm::Expected<LateJoinDTERewriteResult>
insertPendingComputeWithLateJoin(compiler::TargetLLVMModules &targetLLVMModules,
                                 LateJoinDTEAccessMode accessMode) {
  const TargetIdentityId targetIdentity =
      targetLLVMModules.getExecutionConfig().getTargetIdentityId();

  const TargetCallDescriptor &sendPrepareDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTESendPrepare);
  const TargetCallDescriptor &sendIssueDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTESendIssue);
  const TargetCallDescriptor &receiveDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTERecvPrepare);
  const TargetCallDescriptor &elementwiseDescriptor =
      getTargetCallDescriptor(TargetElementwiseOperation::Add);
  const TargetCallDescriptor &joinDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::NCCJoin);
  const TargetDataFormatCodeRecord *format =
      findTargetDataFormatCode(LogicalFormat::F32);
  if (!format)
    return llvm::createStringError(
        "current target has no F32 data-format code");
  if (elementwiseDescriptor.arguments.size() != 6 ||
      joinDescriptor.arguments.size() != 1)
    return llvm::createStringError(
        "late-join test descriptors have unexpected signatures");

  LateJoinDTERewriteResult result;
  for (const compiler::TargetLLVMModule &targetModule :
       targetLLVMModules.getModules()) {
    llvm::Module &module = const_cast<llvm::Module &>(targetModule.getModule());
    llvm::SmallVector<llvm::CallInst *, 2> sendPrepares;
    llvm::SmallVector<llvm::CallInst *, 2> sendIssues;
    llvm::SmallVector<llvm::CallInst *, 2> receives;
    for (llvm::Function &function : module)
      for (llvm::BasicBlock &block : function)
        for (llvm::Instruction &instruction : block) {
          auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
          llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
          if (!callee)
            continue;
          if (callee->getName() == sendPrepareDescriptor.symbol)
            sendPrepares.push_back(call);
          if (callee->getName() == sendIssueDescriptor.symbol)
            sendIssues.push_back(call);
          if (callee->getName() == receiveDescriptor.symbol)
            receives.push_back(call);
        }
    if (sendPrepares.size() != 1 || sendIssues.size() != 1 ||
        receives.size() != 1)
      return llvm::createStringError(
          "late-join Direct-DTE fixture requires exactly one send prepare, "
          "send issue, and receive prepare per rank");

    llvm::CallInst *sendPrepare = sendPrepares.front();
    llvm::CallInst *sendIssue = sendIssues.front();
    llvm::CallInst *receive = receives.front();
    auto *sendSource =
        llvm::dyn_cast<llvm::ConstantInt>(sendPrepare->getArgOperand(0));
    auto *receiveDestination =
        llvm::dyn_cast<llvm::ConstantInt>(receive->getArgOperand(0));
    if (!sendSource || !receiveDestination)
      return llvm::createStringError(
          "late-join Direct-DTE fixture requires static SPM endpoints");

    constexpr uint64_t safeLHS = UINT64_C(0x2d0000);
    constexpr uint64_t safeRHS = UINT64_C(0x2d1000);
    constexpr uint64_t safeDestination = UINT64_C(0x2d2000);
    uint64_t lhs = safeLHS;
    uint64_t destination = safeDestination;
    llvm::Instruction *computeAnchor = receive;
    switch (accessMode) {
    case LateJoinDTEAccessMode::SourceWrite:
      destination = sendSource->getZExtValue();
      computeAnchor = sendIssue;
      break;
    case LateJoinDTEAccessMode::DestinationRead:
      lhs = receiveDestination->getZExtValue();
      break;
    case LateJoinDTEAccessMode::DestinationWrite:
      destination = receiveDestination->getZExtValue();
      break;
    }

    llvm::Function *elementwise =
        getOrDeclareTargetCall(module, elementwiseDescriptor);
    llvm::Function *join = getOrDeclareTargetCall(module, joinDescriptor);
    llvm::IRBuilder<> builder(computeAnchor);
    llvm::CallInst *compute = builder.CreateCall(
        elementwise,
        {builder.getInt64(lhs), builder.getInt64(safeRHS),
         builder.getInt64(destination), builder.getInt32(4),
         builder.getInt32(format->dataFormatCode),
         builder.getInt32(static_cast<uint32_t>(TargetNCCWorker::Worker0))});
    compute->setCallingConv(llvm::CallingConv::C);
    ++result.insertedComputeCount;

    llvm::IRBuilder<> setupBuilder(compute);
    llvm::CallInst *setupJoin = setupBuilder.CreateCall(
        join, {setupBuilder.getInt32(uint32_t{1} << static_cast<uint32_t>(
                                         TargetNCCWorker::Worker0))});
    setupJoin->setCallingConv(llvm::CallingConv::C);
    ++result.insertedSetupJoinCount;

    bool reachedIssue = false;
    for (llvm::Instruction *cursor = compute->getNextNode(); cursor;) {
      llvm::Instruction *next = cursor->getNextNode();
      if (cursor == sendIssue) {
        reachedIssue = true;
        break;
      }
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(cursor)) {
        llvm::Function *callee = call->getCalledFunction();
        if (callee && callee->getName() == joinDescriptor.symbol) {
          call->eraseFromParent();
          ++result.removedPreIssueJoinCount;
        }
      }
      cursor = next;
    }
    if (!reachedIssue)
      return llvm::createStringError(
          "late-join pending compute does not precede the Direct-DTE issue "
          "in one basic block");

    llvm::Instruction *afterIssue = sendIssue->getNextNode();
    if (!afterIssue)
      return llvm::createStringError(
          "Direct-DTE issue has no late-join insertion point");
    llvm::IRBuilder<> lateJoinBuilder(afterIssue);
    llvm::CallInst *lateJoin = lateJoinBuilder.CreateCall(
        join, {lateJoinBuilder.getInt32(uint32_t{1} << static_cast<uint32_t>(
                                            TargetNCCWorker::Worker0))});
    lateJoin->setCallingConv(llvm::CallingConv::C);
    ++result.insertedLateJoinCount;

    std::string verification;
    llvm::raw_string_ostream stream(verification);
    if (llvm::verifyModule(module, &stream))
      return llvm::createStringError(
          "late-join Direct-DTE rewrite produced invalid LLVM IR: " +
          stream.str());
  }
  if (result.insertedComputeCount != targetLLVMModules.getModules().size() ||
      result.insertedLateJoinCount != targetLLVMModules.getModules().size())
    return llvm::createStringError(
        "late-join Direct-DTE rewrite did not cover every rank");
  return result;
}

} // namespace wafer::model::test
