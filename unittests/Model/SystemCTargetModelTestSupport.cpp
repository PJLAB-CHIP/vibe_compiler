//===- SystemCTargetModelTestSupport.cpp - Source DTE test support --------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/InitAll.h"
#include "Wafer/Target/PhysicalTensorCodec.h"
#include "Wafer/Target/TargetFormat.h"

#include "Wafer/Compiler/ExecutableBundleInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::model::test {
namespace {

frontend::ProgramBoundaryBinding partitionedBoundary(int64_t index) {
  frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = frontend::ProgramDistributionKind::Partitioned;
  binding.globalShape = {64};
  binding.localShape = {4};
  binding.dtype = "f32";
  for (int64_t rank = 0; rank < 16; ++rank) {
    frontend::ProgramRankSlice slice;
    slice.logicalRank = rank;
    slice.replicaId = 0;
    slice.offsets = {rank * 4};
    slice.sizes = {4};
    slice.strides = {1};
    binding.rankSlices.push_back(std::move(slice));
  }
  return binding;
}

std::shared_ptr<mlir::MLIRContext> createCompilerContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  registerAllDialects(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

std::vector<RawLogicalValue> makeRankValues(int64_t logicalRank) {
  std::vector<RawLogicalValue> values;
  values.reserve(4);
  for (uint64_t index = 0; index < 4; ++index) {
    // Exact finite f32 values in [1, 1.5), unique across the 64 elements.
    const uint64_t element = static_cast<uint64_t>(logicalRank) * 4 + index;
    values.push_back(
        {LogicalFormat::F32, UINT64_C(0x3f800000) + (element << 15)});
  }
  return values;
}

} // namespace

llvm::Expected<compiler::TargetLLVMModuleBundle>
buildDirectDTETargetBundle(std::string &diagnosticText,
                           TargetProfileId targetProfile) {
  auto context = createCompilerContext();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64>, policy = "all_available", shape = array<i64: 16>, topology = @default}
  func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
    %out = tensor.empty() : tensor<4xf32>
    %permuted = wafer.linalg_ext.collective.collective_permute
        ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>)
        {source_target_pairs = array<i64: 0, 1, 1, 0, 2, 3, 3, 2,
                                          4, 5, 5, 4, 6, 7, 7, 6,
                                          8, 9, 9, 8, 10, 11, 11, 10,
                                          12, 13, 13, 12, 14, 15, 15, 14>,
         channel_id = 91 : i64} -> tensor<4xf32>
    return %permuted : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError("failed to parse Direct-DTE model module");

  frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 1;
  program.distributedInputs = {partitionedBoundary(0)};
  program.distributedOutputs = {partitionedBoundary(0)};
  llvm::Expected<compiler::ExecutionConfig> config =
      compiler::ExecutionConfig::createForSingleCard(16, targetProfile,
                                                     RuntimeLaunchKind::Kernel);
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  llvm::Expected<compiler::ExecutableBundle> executable =
      compiler::detail::buildExecutableBundle(context, *tensorProgram,
                                              std::move(program), *config,
                                              diagnostics, std::nullopt);
  if (!executable)
    return executable.takeError();
  tensorProgram = nullptr;
  return compiler::compileExecutableBundleToTargetLLVMModules(*executable,
                                                              diagnostics);
}

llvm::Expected<DirectDTEInvocationData>
buildDirectDTEInvocationData(const compiler::TargetLLVMModuleBundle &bundle) {
  if (bundle.getModules().size() != 16)
    return llvm::createStringError(
        "Direct-DTE test bundle must contain exactly 16 ranks");

  DirectDTEInvocationData result;
  result.arguments.reserve(16);
  result.inputBytesByRank.resize(16);
  NumericTensorKey tensorKey = llvm::cantFail(
      NumericTensorKey::create(LogicalFormat::F32, MemLayout::Tensor, {4}));
  for (const compiler::TargetLLVMModule &module : bundle.getModules()) {
    const int64_t rank = module.getLogicalRank();
    if (rank < 0 || rank >= 16)
      return llvm::createStringError(
          "Direct-DTE test bundle rank is outside the canonical domain");
    compiler::TargetCallRankArguments arguments{rank, {}};
    size_t userInputCount = 0;
    for (const compiler::KernelABISlot &slot : module.getKernelABISlots()) {
      if (slot.ordinal < 0 || slot.byteSize <= 0 || slot.byteSize >= 0x10000)
        return llvm::createStringError(
            "Direct-DTE test slot is outside its synthetic DDR stride");
      const uint64_t base =
          UINT64_C(0x10000000) +
          static_cast<uint64_t>(rank) * UINT64_C(0x100000) +
          static_cast<uint64_t>(slot.ordinal) * UINT64_C(0x10000);
      arguments.slots.push_back(base);
      if (slot.role == compiler::KernelABISlotRole::UserInput) {
        ++userInputCount;
        llvm::Expected<std::vector<uint8_t>> bytes =
            packPhysicalTensorLogicalValues(tensorKey, makeRankValues(rank),
                                            UINT8_C(0));
        if (!bytes)
          return bytes.takeError();
        if (bytes->size() != static_cast<uint64_t>(slot.byteSize))
          return llvm::createStringError(
              "Direct-DTE test input bytes disagree with the typed ABI slot");
        result.inputBytesByRank[static_cast<size_t>(rank)] = *bytes;
        result.inputBindings.push_back({rank, slot.ordinal, std::move(*bytes)});
      } else if (slot.role == compiler::KernelABISlotRole::Parameter ||
                 slot.role == compiler::KernelABISlotRole::Constant) {
        return llvm::createStringError(
            "Direct-DTE source vertical unexpectedly gained a read-only slot");
      }
    }
    if (userInputCount != 1)
      return llvm::createStringError(
          "Direct-DTE source vertical must have one user input per rank");
    result.arguments.push_back(std::move(arguments));
  }
  return result;
}

llvm::Expected<NCCJoinRewriteResult>
rewriteNCCJoinsAfter(compiler::TargetLLVMModuleBundle &bundle,
                     TargetCallBuiltin anchor) {
  const TargetProfileId targetProfile =
      bundle.getExecutionConfig().getTargetProfileId();
  const TargetCallDescriptor &anchorDescriptor =
      getTargetCallDescriptor(anchor, targetProfile);
  const TargetCallDescriptor &legacyFenceDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::LocalFence, targetProfile);
  const TargetCallDescriptor &joinDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::NCCJoin, targetProfile);
  NCCJoinRewriteResult result;

  for (const compiler::TargetLLVMModule &targetModule : bundle.getModules()) {
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
          if (callee->getName() == legacyFenceDescriptor.symbol ||
              callee->getName() == joinDescriptor.symbol)
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
      builder.CreateCall(
          join, {builder.getInt32(
                    uint32_t{1} << static_cast<uint32_t>(NCCWorker::Worker0))});
      ++result.insertedJoinCount;
    }
    for (llvm::ReturnInst *returnInstruction : returns) {
      llvm::IRBuilder<> builder(returnInstruction);
      builder.CreateCall(
          join, {builder.getInt32(
                    uint32_t{1} << static_cast<uint32_t>(NCCWorker::Worker0))});
      ++result.insertedTerminalJoinCount;
    }
  }
  if (result.insertedJoinCount == 0)
    return llvm::createStringError(
        "Direct-DTE bundle has no requested join anchor call");
  return result;
}

namespace {

llvm::Function *getOrDeclareTargetCall(llvm::Module &module,
                                       const TargetCallDescriptor &descriptor) {
  if (llvm::Function *function = module.getFunction(descriptor.symbol))
    return function;
  llvm::SmallVector<llvm::Type *, 8> arguments;
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

} // namespace

llvm::Expected<PendingComputeDTERewriteResult>
insertPendingComputeBeforeDTEReceive(compiler::TargetLLVMModuleBundle &bundle,
                                     PendingComputeDTEAccessMode accessMode) {
  const TargetProfileId targetProfile =
      bundle.getExecutionConfig().getTargetProfileId();
  if (targetProfile != TargetProfileId::waferTx81SingleCardKernelV3())
    return llvm::createStringError(
        "pending-compute Direct-DTE rewrite requires the V3 target profile");
  const TargetCallDescriptor &receiveDescriptor = getTargetCallDescriptor(
      TargetCallBuiltin::DirectDTERecvPrepare, targetProfile);
  const TargetCallDescriptor &elementwiseDescriptor =
      getTargetCallDescriptor(InstrElementwiseKind::Add, targetProfile);
  const TargetCallDescriptor &gemmDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::Gemm, targetProfile);
  const TargetCallDescriptor &joinDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::NCCJoin, targetProfile);
  const TargetCallDescriptor &fenceDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::LocalFence, targetProfile);
  const TargetCallDescriptor &waitDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTEWait, targetProfile);
  const TargetDataFormatCodeRecord *format =
      findTargetDataFormatCode(targetProfile, LogicalFormat::F32);
  if (!format)
    return llvm::createStringError(
        "V3 target profile has no F32 data-format code");
  if (elementwiseDescriptor.arguments.size() != 6 ||
      gemmDescriptor.arguments.size() != 9 ||
      joinDescriptor.arguments.size() != 1)
    return llvm::createStringError(
        "pending-compute test descriptors have unexpected V3 signatures");

  PendingComputeDTERewriteResult result;
  for (const compiler::TargetLLVMModule &targetModule : bundle.getModules()) {
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
      if ((targetModule.getLogicalRank() & 1) == 0) {
        llvm::Function *elementwise =
            getOrDeclareTargetCall(module, elementwiseDescriptor);
        compute = builder.CreateCall(
            elementwise,
            {builder.getInt64(lhs), builder.getInt64(rhs),
             builder.getInt64(destination), builder.getInt32(4),
             builder.getInt32(format->dataFormatCode),
             builder.getInt32(static_cast<uint32_t>(NCCWorker::Worker0))});
        ++result.elementwiseCount;
      } else {
        llvm::Function *gemm = getOrDeclareTargetCall(module, gemmDescriptor);
        compute = builder.CreateCall(
            gemm,
            {builder.getInt64(lhs), builder.getInt64(rhs),
             builder.getInt64(destination), builder.getInt32(2),
             builder.getInt32(2), builder.getInt32(2), builder.getInt32(1),
             builder.getInt32(format->dataFormatCode),
             builder.getInt32(static_cast<uint32_t>(NCCWorker::Worker0))});
        ++result.gemmCount;
      }
      compute->setCallingConv(llvm::CallingConv::C);

      llvm::Function *join = getOrDeclareTargetCall(module, joinDescriptor);
      llvm::IRBuilder<> setupBuilder(compute);
      llvm::CallInst *setupJoin = setupBuilder.CreateCall(
          join, {setupBuilder.getInt32(
                    uint32_t{1} << static_cast<uint32_t>(NCCWorker::Worker0))});
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
          if (callee && (callee->getName() == joinDescriptor.symbol ||
                         callee->getName() == fenceDescriptor.symbol)) {
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
                                        NCCWorker::Worker0))});
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
insertPendingComputeWithLateJoin(compiler::TargetLLVMModuleBundle &bundle,
                                 LateJoinDTEAccessMode accessMode) {
  const TargetProfileId targetProfile =
      bundle.getExecutionConfig().getTargetProfileId();
  if (targetProfile != TargetProfileId::waferTx81SingleCardKernelV3())
    return llvm::createStringError(
        "late-join Direct-DTE rewrite requires the V3 target profile");

  const TargetCallDescriptor &sendPrepareDescriptor = getTargetCallDescriptor(
      TargetCallBuiltin::DirectDTESendPrepare, targetProfile);
  const TargetCallDescriptor &sendIssueDescriptor = getTargetCallDescriptor(
      TargetCallBuiltin::DirectDTESendIssue, targetProfile);
  const TargetCallDescriptor &receiveDescriptor = getTargetCallDescriptor(
      TargetCallBuiltin::DirectDTERecvPrepare, targetProfile);
  const TargetCallDescriptor &elementwiseDescriptor =
      getTargetCallDescriptor(InstrElementwiseKind::Add, targetProfile);
  const TargetCallDescriptor &joinDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::NCCJoin, targetProfile);
  const TargetCallDescriptor &fenceDescriptor =
      getTargetCallDescriptor(TargetCallBuiltin::LocalFence, targetProfile);
  const TargetDataFormatCodeRecord *format =
      findTargetDataFormatCode(targetProfile, LogicalFormat::F32);
  if (!format)
    return llvm::createStringError(
        "V3 target profile has no F32 data-format code");
  if (elementwiseDescriptor.arguments.size() != 6 ||
      joinDescriptor.arguments.size() != 1)
    return llvm::createStringError(
        "late-join test descriptors have unexpected V3 signatures");

  LateJoinDTERewriteResult result;
  for (const compiler::TargetLLVMModule &targetModule : bundle.getModules()) {
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
         builder.getInt32(static_cast<uint32_t>(NCCWorker::Worker0))});
    compute->setCallingConv(llvm::CallingConv::C);
    ++result.insertedComputeCount;

    llvm::IRBuilder<> setupBuilder(compute);
    llvm::CallInst *setupJoin = setupBuilder.CreateCall(
        join, {setupBuilder.getInt32(
                  uint32_t{1} << static_cast<uint32_t>(NCCWorker::Worker0))});
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
        if (callee && (callee->getName() == joinDescriptor.symbol ||
                       callee->getName() == fenceDescriptor.symbol)) {
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
        join, {lateJoinBuilder.getInt32(
                  uint32_t{1} << static_cast<uint32_t>(NCCWorker::Worker0))});
    lateJoin->setCallingConv(llvm::CallingConv::C);
    ++result.insertedLateJoinCount;

    std::string verification;
    llvm::raw_string_ostream stream(verification);
    if (llvm::verifyModule(module, &stream))
      return llvm::createStringError(
          "late-join Direct-DTE rewrite produced invalid LLVM IR: " +
          stream.str());
  }
  if (result.insertedComputeCount != bundle.getModules().size() ||
      result.insertedLateJoinCount != bundle.getModules().size())
    return llvm::createStringError(
        "late-join Direct-DTE rewrite did not cover every rank");
  return result;
}

} // namespace wafer::model::test
