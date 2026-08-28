//===- TargetCallExecutionTest.cpp - Host target-call execution tests ---===//

#include "Wafer/Target/Execution/TargetCallExecution.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Target/Core/TargetCall.h"

#include "Wafer/CodeGen/Executable/DeviceExecutableInternal.h"
#include "Wafer/CodeGen/Target/TargetCodeGenInternal.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Program/ProgramData.h"
#include "Wafer/Target/Core/TargetIdentity.h"

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

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

std::vector<uint64_t>
makeDecodableArguments(const wafer::TargetCallDescriptor &descriptor);

llvm::Expected<wafer::compiler::TargetLLVMModules>
compileElementwiseTargetModules(std::string &diagnosticText) {
  diagnosticText.clear();
  auto executionConfig =
      wafer::compiler::ExecutionConfig::createForSingleCard(1);
  if (!executionConfig)
    return executionConfig.takeError();
  auto launch = wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::Grid,
      wafer::KernelEntryABI::TileMajorPointerTable,
      {wafer::RuntimeLaunchPhaseRole::Main});
  if (!launch)
    return launch.takeError();

  const wafer::TargetCallDescriptor &descriptor =
      wafer::getTargetCallDescriptor(wafer::TargetElementwiseOperation::Add);
  std::vector<uint64_t> rawArguments = makeDecodableArguments(descriptor);
  rawArguments[3] = 1;
  const wafer::TargetCallDescriptor &rdma =
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::RDMA);
  const std::vector<uint64_t> rdmaArguments = makeDecodableArguments(rdma);
  std::vector<wafer::compiler::TargetLLVMModule> modules;
  modules.reserve(16);
  for (int64_t tile = 0; tile < 16; ++tile) {
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module =
        std::make_unique<llvm::Module>("target-call-execution", *context);
    module->setTargetTriple("riscv64-unknown-unknown-elf");
    auto getScalarType = [&](wafer::TargetCallScalarType scalar) {
      return scalar == wafer::TargetCallScalarType::I64
                 ? llvm::Type::getInt64Ty(*context)
                 : llvm::Type::getInt32Ty(*context);
    };
    llvm::Function *entry = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(*context),
                                {llvm::Type::getInt64Ty(*context)}, false),
        llvm::GlobalValue::ExternalLinkage, "main", *module);
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(*context, "entry", entry));
    auto emitCall = [&](const wafer::TargetCallDescriptor &call,
                        llvm::ArrayRef<uint64_t> values) {
      llvm::SmallVector<llvm::Type *, 16> argumentTypes;
      llvm::SmallVector<llvm::Value *, 16> arguments;
      for (auto [scalar, value] : llvm::zip_equal(call.arguments, values)) {
        llvm::Type *type = getScalarType(scalar);
        argumentTypes.push_back(type);
        arguments.push_back(llvm::ConstantInt::get(type, value));
      }
      llvm::Type *resultType = call.result == wafer::TargetCallResultType::I64
                                   ? llvm::Type::getInt64Ty(*context)
                                   : llvm::Type::getVoidTy(*context);
      llvm::FunctionCallee callee = module->getOrInsertFunction(
          call.symbol,
          llvm::FunctionType::get(resultType, argumentTypes, false));
      builder.CreateCall(callee, arguments);
    };
    if (tile < 8)
      emitCall(descriptor, rawArguments);
    emitCall(rdma, rdmaArguments);
    builder.CreateRetVoid();
    wafer::compiler::TileEntryArgument slot;
    slot.ordinal = 0;
    slot.kind = wafer::compiler::TileEntryArgumentKind::ExternalInput;
    slot.resourceIndex = 0;
    slot.name = "input";
    slot.dtype = wafer::LogicalFormat::F32;
    slot.layout = wafer::MemLayout::Tensor;
    slot.shape = {8};
    slot.byteSize = 32;
    slot.alignment = 8;
    slot.access = wafer::compiler::TileEntryArgumentAccess::ReadOnly;
    modules.push_back(wafer::compiler::TargetLLVMModulesBuilder::makeModule(
        wafer::CardId(0), wafer::TileId(tile), wafer::LaunchSlotId(tile),
        "main", wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        wafer::kCurrentTargetModuleFormat, {std::move(slot)},
        std::move(context), std::move(module)));
  }
  return wafer::compiler::TargetLLVMModulesBuilder::makeModules(
      *executionConfig, std::move(*launch), std::move(modules));
}

class RecordingSink final : public wafer::compiler::TargetCommandSink {
public:
  llvm::Error begin(const wafer::compiler::TargetCallInvocationDescriptor
                        &descriptor) override {
    began = true;
    invocationTileCount = descriptor.tiles.size();
    tileDescriptors = descriptor.tiles;
    return llvm::Error::success();
  }

  llvm::Expected<uint64_t>
  issue(const wafer::compiler::TargetCommand &command) override {
    if (reenterExecutable && !attemptedReentry) {
      attemptedReentry = true;
      llvm::Error error = reenterExecutable->executeTile(command.launchSlotId);
      if (error)
        reentryDiagnostic = llvm::toString(std::move(error));
    }
    if ((failAtIssue && commands.size() == *failAtIssue) ||
        (failAtLaunchSlot &&
         command.launchSlotId.getValue() == *failAtLaunchSlot)) {
      failedLaunchSlot = command.launchSlotId.getValue();
      return llvm::createStringError("injected command sink failure");
    }
    commands.push_back(command);
    return nextEvent++;
  }

  llvm::Error completeTile(wafer::CardId cardId, wafer::TileId tileId,
                           wafer::LaunchSlotId launchSlotId) override {
    completedCardIds.push_back(cardId.getValue());
    completedTileIds.push_back(tileId.getValue());
    completedLaunchSlots.push_back(launchSlotId.getValue());
    return llvm::Error::success();
  }

  llvm::Error completeInvocation() override {
    if (failCompleteInvocation)
      return llvm::createStringError("injected invocation completion failure");
    invocationCompleted = true;
    return llvm::Error::success();
  }

  void abort(llvm::StringRef diagnostic) override {
    aborted = true;
    abortDiagnostic = diagnostic.str();
    commands.clear();
    completedCardIds.clear();
    completedTileIds.clear();
    completedLaunchSlots.clear();
  }

  bool began = false;
  bool invocationCompleted = false;
  bool aborted = false;
  size_t invocationTileCount = 0;
  uint64_t nextEvent = 0x2000;
  std::optional<size_t> failAtIssue;
  std::optional<int64_t> failAtLaunchSlot;
  bool failCompleteInvocation = false;
  wafer::compiler::TargetCallExecutable *reenterExecutable = nullptr;
  bool attemptedReentry = false;
  std::optional<int64_t> failedLaunchSlot;
  std::string reentryDiagnostic;
  std::string abortDiagnostic;
  std::vector<wafer::compiler::TargetCallTileDescriptor> tileDescriptors;
  std::vector<wafer::compiler::TargetCommand> commands;
  std::vector<int64_t> completedCardIds;
  std::vector<int64_t> completedTileIds;
  std::vector<int64_t> completedLaunchSlots;
};

wafer::compiler::TargetCallTileArguments
makeTileArguments(const wafer::compiler::TargetLLVMModule &module,
                  std::vector<uint64_t> slots) {
  return {module.getCardId(), module.getTileId(), module.getLaunchSlotId(),
          std::move(slots)};
}

std::vector<wafer::compiler::TargetCallTileArguments> makeInvocationArguments(
    const wafer::compiler::TargetLLVMModules &targetLLVMModules,
    uint64_t baseAddress = UINT64_C(0x100000)) {
  std::vector<wafer::compiler::TargetCallTileArguments> arguments;
  arguments.reserve(targetLLVMModules.getModules().size());
  for (const wafer::compiler::TargetLLVMModule &module :
       targetLLVMModules.getModules()) {
    std::vector<uint64_t> slots;
    slots.reserve(module.getTileEntryArguments().size());
    for (const wafer::compiler::TileEntryArgument &slot :
         module.getTileEntryArguments())
      slots.push_back(
          baseAddress +
          static_cast<uint64_t>(module.getLaunchSlotId().getValue()) *
              UINT64_C(0x100000) +
          static_cast<uint64_t>(slot.ordinal) * UINT64_C(0x1000));
    arguments.push_back(makeTileArguments(module, std::move(slots)));
  }
  return arguments;
}

uint64_t supportedF32Code(wafer::TargetFormatEngine engine) {
  const wafer::TargetFormatEncodingRecord *record =
      wafer::findTargetFormatEncoding(engine, wafer::LogicalFormat::F32);
  assert(record);
  return record->dataFormatCode;
}

std::vector<uint64_t>
makeDecodableArguments(const wafer::TargetCallDescriptor &descriptor) {
  std::vector<uint64_t> arguments;
  arguments.reserve(descriptor.arguments.size());
  for (auto [index, scalar] : llvm::enumerate(descriptor.arguments))
    arguments.push_back(scalar == wafer::TargetCallScalarType::I64
                            ? 0x100000 + index * 0x1000
                            : index + 1);
  if (descriptor.issueDomain &&
      descriptor.issueDomain->nccWorkerArgument.has_value())
    arguments[*descriptor.issueDomain->nccWorkerArgument] = 0;

  if (const auto *builtin =
          std::get_if<wafer::TargetCallBuiltin>(&descriptor.semantic)) {
    switch (*builtin) {
    case wafer::TargetCallBuiltin::RDMA:
      arguments[10] = supportedF32Code(wafer::TargetFormatEngine::RDMA);
      break;
    case wafer::TargetCallBuiltin::WDMA:
      arguments[10] = supportedF32Code(wafer::TargetFormatEngine::WDMA);
      break;
    case wafer::TargetCallBuiltin::Memset:
      arguments[3] = supportedF32Code(wafer::TargetFormatEngine::TDMA);
      break;
    case wafer::TargetCallBuiltin::Bit2FP:
      arguments[3] = supportedF32Code(wafer::TargetFormatEngine::CT);
      break;
    case wafer::TargetCallBuiltin::MaskMove:
      arguments[4] = supportedF32Code(wafer::TargetFormatEngine::CT);
      break;
    case wafer::TargetCallBuiltin::Gemm:
      arguments[7] = supportedF32Code(wafer::TargetFormatEngine::NE);
      break;
    case wafer::TargetCallBuiltin::GemmOriented:
      arguments[7] = supportedF32Code(wafer::TargetFormatEngine::NE);
      arguments[8] = 1;
      arguments[9] = 0;
      break;
    case wafer::TargetCallBuiltin::TDMAPad:
      arguments[14] = supportedF32Code(wafer::TargetFormatEngine::TDMA);
      break;
    case wafer::TargetCallBuiltin::TDMAImg2Col:
      arguments[18] = supportedF32Code(wafer::TargetFormatEngine::TDMA);
      break;
    case wafer::TargetCallBuiltin::DirectDTEBegin:
    case wafer::TargetCallBuiltin::DirectDTEBeginAfterPrepare:
      arguments[1] = 16;
      break;
    case wafer::TargetCallBuiltin::DirectDTESendPrepare:
      arguments[6] = 1;
      break;
    case wafer::TargetCallBuiltin::GatherScatter:
    case wafer::TargetCallBuiltin::NCCJoin:
    case wafer::TargetCallBuiltin::DirectDTESendIssue:
    case wafer::TargetCallBuiltin::DirectDTERecvPrepare:
    case wafer::TargetCallBuiltin::DirectDTEWait:
    case wafer::TargetCallBuiltin::DirectDTEFinish:
      break;
    }
    return arguments;
  }
  if (std::holds_alternative<wafer::TargetElementwiseOperation>(
          descriptor.semantic)) {
    const size_t payloadSize =
        arguments.size() -
        (descriptor.issueDomain &&
                 descriptor.issueDomain->nccWorkerArgument.has_value()
             ? 1
             : 0);
    arguments[payloadSize - 1] =
        supportedF32Code(wafer::TargetFormatEngine::CT);
    return arguments;
  }
  if (std::holds_alternative<wafer::TargetReduceOperation>(
          descriptor.semantic)) {
    arguments[7] = supportedF32Code(wafer::TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *operation =
          std::get_if<wafer::TargetConvertOperation>(&descriptor.semantic)) {
    const wafer::TargetConvertRoute *route =
        wafer::findTargetConvertRoute(operation->getOpcode());
    if (route &&
        route->parameterKind == wafer::TargetConvertParameterKind::RoundingMode)
      arguments[4] =
          static_cast<uint8_t>(wafer::TargetRoundingMode::Stochastic);
    return arguments;
  }
  if (const auto *kind = std::get_if<wafer::TargetConvolutionOperation>(
          &descriptor.semantic)) {
    arguments[3] = static_cast<uint32_t>(*kind);
    arguments[30] = supportedF32Code(wafer::TargetFormatEngine::NE);
    return arguments;
  }
  if (const auto *kind =
          std::get_if<wafer::TargetPoolingOperation>(&descriptor.semantic)) {
    bool indexed = *kind == wafer::TargetPoolingOperation::IndexedMaximum ||
                   *kind == wafer::TargetPoolingOperation::IndexedMinimum;
    size_t firstField = indexed ? 3 : 2;
    arguments[firstField] = static_cast<uint32_t>(*kind);
    arguments[firstField + 17] =
        supportedF32Code(wafer::TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *kind =
          std::get_if<wafer::TargetUnpoolingOperation>(&descriptor.semantic)) {
    arguments[2] = static_cast<uint32_t>(*kind);
    arguments[16] = supportedF32Code(wafer::TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *kind =
          std::get_if<wafer::TargetPeripheralOperation>(&descriptor.semantic)) {
    size_t addressCount = 0;
    switch (*kind) {
    case wafer::TargetPeripheralOperation::ArgMaximum:
    case wafer::TargetPeripheralOperation::ArgMinimum:
    case wafer::TargetPeripheralOperation::LookupTable16:
    case wafer::TargetPeripheralOperation::LookupTable32:
      addressCount = 3;
      break;
    case wafer::TargetPeripheralOperation::Bilinear:
    case wafer::TargetPeripheralOperation::ElementMask:
      addressCount = 2;
      break;
    case wafer::TargetPeripheralOperation::Random:
      addressCount = 5;
      break;
    case wafer::TargetPeripheralOperation::Count:
    case wafer::TargetPeripheralOperation::Factorize:
      llvm_unreachable("unregistered peripheral kind");
    }
    arguments[addressCount] = static_cast<uint32_t>(*kind);
    arguments[addressCount + 2] =
        supportedF32Code(wafer::TargetFormatEngine::CT);
  }
  return arguments;
}

template <size_t N>
std::array<uint32_t, N> expectedArray(llvm::ArrayRef<uint64_t> arguments,
                                      size_t start) {
  std::array<uint32_t, N> result{};
  for (size_t index = 0; index < N; ++index)
    result[index] = static_cast<uint32_t>(arguments[start + index]);
  return result;
}

void expectPayloadFields(const wafer::TargetCallDescriptor &descriptor,
                         llvm::ArrayRef<uint64_t> arguments,
                         const wafer::target::TargetCommandPayload &payload) {
  const wafer::TargetCallSemantic &semantic = descriptor.semantic;
  SCOPED_TRACE(descriptor.symbol);
  auto u32 = [&](size_t index) {
    return static_cast<uint32_t>(arguments[index]);
  };
  auto expectFormat = [](wafer::LogicalFormat format) {
    EXPECT_EQ(format, wafer::LogicalFormat::F32);
  };

  if (const auto *builtin = std::get_if<wafer::TargetCallBuiltin>(&semantic)) {
    switch (*builtin) {
    case wafer::TargetCallBuiltin::RDMA:
    case wafer::TargetCallBuiltin::WDMA: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetStridedDMACommand>(
              payload));
      const auto &value =
          std::get<wafer::target::TargetStridedDMACommand>(payload);
      EXPECT_EQ(value.direction,
                *builtin == wafer::TargetCallBuiltin::RDMA
                    ? wafer::target::TargetDMADirection::Read
                    : wafer::target::TargetDMADirection::Write);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.byteCount, u32(2));
      EXPECT_EQ(value.innerBytes, u32(3));
      EXPECT_EQ(value.strides, expectedArray<3>(arguments, 4));
      EXPECT_EQ(value.iterations, expectedArray<3>(arguments, 7));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::GatherScatter: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetGatherScatterCommand>(
              payload));
      const auto &value =
          std::get<wafer::target::TargetGatherScatterCommand>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.byteCount, u32(2));
      EXPECT_EQ(value.innerBytes, u32(3));
      EXPECT_EQ(value.sourceStrides, expectedArray<3>(arguments, 4));
      EXPECT_EQ(value.sourceIterations, expectedArray<3>(arguments, 7));
      EXPECT_EQ(value.destinationStrides, expectedArray<3>(arguments, 10));
      EXPECT_EQ(value.destinationIterations, expectedArray<3>(arguments, 13));
      return;
    }
    case wafer::TargetCallBuiltin::Memset: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetMemsetCommand>(payload));
      const auto &value = std::get<wafer::target::TargetMemsetCommand>(payload);
      EXPECT_EQ(value.destination, arguments[0]);
      EXPECT_EQ(value.value, u32(1));
      EXPECT_EQ(value.elementCount, u32(2));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::Bit2FP: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetBit2FPCommand>(payload));
      const auto &value = std::get<wafer::target::TargetBit2FPCommand>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.elementCount, u32(2));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::MaskMove: {
      ASSERT_TRUE(std::holds_alternative<wafer::target::TargetMaskMoveCommand>(
          payload));
      const auto &value =
          std::get<wafer::target::TargetMaskMoveCommand>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.mask, u32(1));
      EXPECT_EQ(value.destination, arguments[2]);
      EXPECT_EQ(value.elementCount, u32(3));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::Gemm:
    case wafer::TargetCallBuiltin::GemmOriented: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetGemmCommand>(payload));
      const auto &value = std::get<wafer::target::TargetGemmCommand>(payload);
      EXPECT_EQ(value.lhs, arguments[0]);
      EXPECT_EQ(value.rhs, arguments[1]);
      EXPECT_EQ(value.destination, arguments[2]);
      EXPECT_EQ(value.m, u32(3));
      EXPECT_EQ(value.k, u32(4));
      EXPECT_EQ(value.n, u32(5));
      EXPECT_EQ(value.batchCount, u32(6));
      expectFormat(value.format);
      EXPECT_EQ(value.lhsOrientation,
                *builtin == wafer::TargetCallBuiltin::GemmOriented
                    ? wafer::TargetGemmOrientation::Transpose
                    : wafer::TargetGemmOrientation::Normal);
      EXPECT_EQ(value.rhsOrientation, wafer::TargetGemmOrientation::Normal);
      return;
    }
    case wafer::TargetCallBuiltin::TDMAPad:
    case wafer::TargetCallBuiltin::TDMAImg2Col: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetTDMATransformCommand>(
              payload));
      const auto &value =
          std::get<wafer::target::TargetTDMATransformCommand>(payload);
      const bool imageToColumn =
          *builtin == wafer::TargetCallBuiltin::TDMAImg2Col;
      EXPECT_EQ(value.kind,
                imageToColumn
                    ? wafer::target::TargetTDMATransformKind::ImageToColumn
                    : wafer::target::TargetTDMATransformKind::Pad);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.sourceShape, expectedArray<4>(arguments, 2));
      EXPECT_EQ(value.destinationShape, expectedArray<4>(arguments, 6));
      EXPECT_EQ(value.pads, expectedArray<4>(arguments, 10));
      if (imageToColumn) {
        ASSERT_TRUE(value.kernelStrides.has_value());
        EXPECT_EQ(*value.kernelStrides, expectedArray<4>(arguments, 14));
      } else {
        EXPECT_FALSE(value.kernelStrides.has_value());
      }
      expectFormat(value.format);
      return;
    }
    case wafer::TargetCallBuiltin::NCCJoin: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetNCCJoinCommand>(payload));
      const auto &join = std::get<wafer::target::TargetNCCJoinCommand>(payload);
      EXPECT_EQ(join.participantMask, u32(0));
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTEBegin:
    case wafer::TargetCallBuiltin::DirectDTEBeginAfterPrepare: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetDirectDTEBeginCommand>(
              payload));
      const auto &value =
          std::get<wafer::target::TargetDirectDTEBeginCommand>(payload);
      EXPECT_EQ(value.statusAddress, arguments[0]);
      EXPECT_EQ(value.participantCount, u32(1));
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTESendPrepare: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetDirectDTESendCommand>(
              payload));
      const auto &value =
          std::get<wafer::target::TargetDirectDTESendCommand>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.remoteDestination, arguments[1]);
      EXPECT_EQ(value.byteCount, u32(2));
      EXPECT_EQ(value.localTile, u32(3));
      EXPECT_EQ(value.remoteTile, u32(4));
      EXPECT_EQ(value.remoteFSM, u32(5));
      EXPECT_EQ(value.highPerformance, u32(6) != 0);
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTESendIssue: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::target::TargetDirectDTESendIssueCommand>(payload));
      EXPECT_EQ(
          std::get<wafer::target::TargetDirectDTESendIssueCommand>(payload)
              .event,
          arguments[0]);
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTERecvPrepare: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetDirectDTEReceiveCommand>(
              payload));
      const auto &value =
          std::get<wafer::target::TargetDirectDTEReceiveCommand>(payload);
      EXPECT_EQ(value.destination, arguments[0]);
      EXPECT_EQ(value.byteCount, u32(1));
      EXPECT_EQ(value.localTile, u32(2));
      EXPECT_EQ(value.remoteTile, u32(3));
      EXPECT_EQ(value.localFSM, u32(4));
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTEWait: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetDirectDTEWaitCommand>(
              payload));
      EXPECT_EQ(
          std::get<wafer::target::TargetDirectDTEWaitCommand>(payload).event,
          arguments[0]);
      return;
    }
    case wafer::TargetCallBuiltin::DirectDTEFinish:
      EXPECT_TRUE(
          std::holds_alternative<wafer::target::TargetDirectDTEFinishCommand>(
              payload));
      return;
    }
  }

  if (const auto *kind =
          std::get_if<wafer::TargetElementwiseOperation>(&semantic)) {
    ASSERT_TRUE(std::holds_alternative<wafer::target::TargetElementwiseCommand>(
        payload));
    const auto &value =
        std::get<wafer::target::TargetElementwiseCommand>(payload);
    const size_t payloadSize =
        arguments.size() -
        (descriptor.issueDomain &&
                 descriptor.issueDomain->nccWorkerArgument.has_value()
             ? 1
             : 0);
    const bool unary = payloadSize == 4;
    EXPECT_EQ(value.operation, *kind);
    EXPECT_EQ(value.lhs, arguments[0]);
    if (unary)
      EXPECT_FALSE(value.rhs.has_value());
    else {
      ASSERT_TRUE(value.rhs.has_value());
      EXPECT_EQ(*value.rhs, arguments[1]);
    }
    EXPECT_EQ(value.destination, arguments[unary ? 1 : 2]);
    EXPECT_EQ(value.elementCount, u32(unary ? 2 : 3));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind = std::get_if<wafer::TargetReduceOperation>(&semantic)) {
    ASSERT_TRUE(
        std::holds_alternative<wafer::target::TargetReduceCommand>(payload));
    const auto &value = std::get<wafer::target::TargetReduceCommand>(payload);
    EXPECT_EQ(value.operation, *kind);
    EXPECT_EQ(value.source, arguments[0]);
    EXPECT_EQ(value.destination, arguments[1]);
    EXPECT_EQ(value.dimension, u32(2));
    EXPECT_EQ(value.nhwc, expectedArray<4>(arguments, 3));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind =
          std::get_if<wafer::TargetConvertOperation>(&semantic)) {
    ASSERT_TRUE(
        std::holds_alternative<wafer::target::TargetConvertCommand>(payload));
    const auto &value = std::get<wafer::target::TargetConvertCommand>(payload);
    EXPECT_EQ(value.operation, *kind);
    EXPECT_EQ(value.source, arguments[0]);
    EXPECT_EQ(value.destination, arguments[1]);
    EXPECT_EQ(value.elementCount, u32(2));
    const wafer::TargetConvertRoute *route =
        wafer::findTargetConvertRoute(kind->getOpcode());
    ASSERT_NE(route, nullptr);
    switch (route->parameterKind) {
    case wafer::TargetConvertParameterKind::ZeroPoint:
      ASSERT_TRUE(value.parameter.has_value());
      ASSERT_TRUE(value.parameter->getZeroPoint());
      EXPECT_EQ(*value.parameter->getZeroPoint(), u32(3));
      break;
    case wafer::TargetConvertParameterKind::RoundingMode:
      ASSERT_TRUE(value.parameter.has_value());
      ASSERT_TRUE(value.parameter->getRoundingMode());
      EXPECT_EQ(*value.parameter->getRoundingMode(),
                wafer::TargetRoundingMode::Stochastic);
      break;
    case wafer::TargetConvertParameterKind::None:
      EXPECT_FALSE(value.parameter.has_value());
      break;
    }
    return;
  }
  if (const auto *kind =
          std::get_if<wafer::TargetConvolutionOperation>(&semantic)) {
    ASSERT_TRUE(
        std::holds_alternative<wafer::target::TargetConvCommand>(payload));
    const auto &value = std::get<wafer::target::TargetConvCommand>(payload);
    EXPECT_EQ(value.operation, *kind);
    EXPECT_EQ(value.input, arguments[0]);
    EXPECT_EQ(value.weight, arguments[1]);
    EXPECT_EQ(value.destination, arguments[2]);
    EXPECT_EQ(value.inputShape, expectedArray<4>(arguments, 4));
    EXPECT_EQ(value.weightShape, expectedArray<4>(arguments, 8));
    EXPECT_EQ(value.outputShape, expectedArray<4>(arguments, 12));
    EXPECT_EQ(value.pads, expectedArray<4>(arguments, 16));
    EXPECT_EQ(value.unpads, expectedArray<4>(arguments, 20));
    EXPECT_EQ(value.kernelStrides, expectedArray<4>(arguments, 24));
    EXPECT_EQ(value.dilations, expectedArray<2>(arguments, 28));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind =
          std::get_if<wafer::TargetPoolingOperation>(&semantic)) {
    ASSERT_TRUE(
        std::holds_alternative<wafer::target::TargetPoolCommand>(payload));
    const auto &value = std::get<wafer::target::TargetPoolCommand>(payload);
    const bool indexed =
        *kind == wafer::TargetPoolingOperation::IndexedMaximum ||
        *kind == wafer::TargetPoolingOperation::IndexedMinimum;
    const size_t first = indexed ? 3 : 2;
    EXPECT_EQ(value.operation, *kind);
    EXPECT_EQ(value.input, arguments[0]);
    EXPECT_EQ(value.valueDestination, arguments[1]);
    if (indexed) {
      ASSERT_TRUE(value.indexDestination.has_value());
      EXPECT_EQ(*value.indexDestination, arguments[2]);
    } else {
      EXPECT_FALSE(value.indexDestination.has_value());
    }
    EXPECT_EQ(value.sourceShape, expectedArray<4>(arguments, first + 1));
    EXPECT_EQ(value.destinationShape, expectedArray<4>(arguments, first + 5));
    EXPECT_EQ(value.pads, expectedArray<4>(arguments, first + 9));
    EXPECT_EQ(value.kernelStrides, expectedArray<4>(arguments, first + 13));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind =
          std::get_if<wafer::TargetUnpoolingOperation>(&semantic)) {
    ASSERT_TRUE(
        std::holds_alternative<wafer::target::TargetUnpoolCommand>(payload));
    const auto &value = std::get<wafer::target::TargetUnpoolCommand>(payload);
    EXPECT_EQ(value.operation, *kind);
    EXPECT_EQ(value.input, arguments[0]);
    EXPECT_EQ(value.destination, arguments[1]);
    if (*kind == wafer::TargetUnpoolingOperation::Average)
      EXPECT_FALSE(value.indexAddress.has_value());
    else {
      ASSERT_TRUE(value.indexAddress.has_value());
      EXPECT_EQ(*value.indexAddress, u32(3));
    }
    EXPECT_EQ(value.sourceShape, expectedArray<4>(arguments, 4));
    EXPECT_EQ(value.destinationShape, expectedArray<4>(arguments, 8));
    EXPECT_EQ(value.kernelStrides, expectedArray<4>(arguments, 12));
    expectFormat(value.format);
    return;
  }
  if (const auto *kind =
          std::get_if<wafer::TargetPeripheralOperation>(&semantic)) {
    switch (*kind) {
    case wafer::TargetPeripheralOperation::ArgMaximum:
    case wafer::TargetPeripheralOperation::ArgMinimum: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::target::TargetPeripheralArgExtremaCommand>(payload));
      const auto &value =
          std::get<wafer::target::TargetPeripheralArgExtremaCommand>(payload);
      EXPECT_EQ(value.operation, *kind);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.valueDestination, arguments[1]);
      EXPECT_EQ(value.indexDestination, arguments[2]);
      EXPECT_EQ(value.elementCount, u32(4));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetPeripheralOperation::Bilinear: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::target::TargetPeripheralBilinearCommand>(payload));
      const auto &value =
          std::get<wafer::target::TargetPeripheralBilinearCommand>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.elementCount, u32(3));
      expectFormat(value.format);
      EXPECT_EQ(value.sourceShape, expectedArray<4>(arguments, 5));
      EXPECT_EQ(value.destinationShape, expectedArray<4>(arguments, 9));
      return;
    }
    case wafer::TargetPeripheralOperation::LookupTable16:
    case wafer::TargetPeripheralOperation::LookupTable32: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetPeripheralLUTCommand>(
              payload));
      const auto &value =
          std::get<wafer::target::TargetPeripheralLUTCommand>(payload);
      EXPECT_EQ(value.operation, *kind);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.table, arguments[1]);
      EXPECT_EQ(value.destination, arguments[2]);
      EXPECT_EQ(value.elementCount, u32(4));
      expectFormat(value.format);
      EXPECT_EQ(value.tableElementCount, u32(6));
      return;
    }
    case wafer::TargetPeripheralOperation::Random: {
      ASSERT_TRUE(
          std::holds_alternative<wafer::target::TargetPeripheralRandomCommand>(
              payload));
      const auto &value =
          std::get<wafer::target::TargetPeripheralRandomCommand>(payload);
      EXPECT_EQ(value.sources,
                (std::array<uint64_t, 2>{arguments[0], arguments[1]}));
      EXPECT_EQ(
          value.destinations,
          (std::array<uint64_t, 3>{arguments[2], arguments[3], arguments[4]}));
      EXPECT_EQ(value.elementCount, u32(6));
      expectFormat(value.format);
      return;
    }
    case wafer::TargetPeripheralOperation::ElementMask: {
      ASSERT_TRUE(std::holds_alternative<
                  wafer::target::TargetPeripheralElementMaskCommand>(payload));
      const auto &value =
          std::get<wafer::target::TargetPeripheralElementMaskCommand>(payload);
      EXPECT_EQ(value.source, arguments[0]);
      EXPECT_EQ(value.destination, arguments[1]);
      EXPECT_EQ(value.elementCount, u32(3));
      expectFormat(value.format);
      EXPECT_EQ(value.scale, u32(6));
      EXPECT_EQ(value.probability, u32(7));
      EXPECT_EQ(value.roundingMode, u32(8));
      return;
    }
    case wafer::TargetPeripheralOperation::Count:
    case wafer::TargetPeripheralOperation::Factorize:
      FAIL() << "unregistered peripheral semantic";
      return;
    }
  }
  FAIL() << "unknown target-call semantic";
}

TEST(TargetCallRegistryTest, ExactlyCoversTypedTargetCallSurface) {
  llvm::ArrayRef<wafer::TargetCallDescriptor> descriptors =
      wafer::getTargetCallDescriptors();
  ASSERT_EQ(descriptors.size(), 112u);
  llvm::DenseSet<llvm::StringRef> symbols;
  size_t issueDomainCount = 0;
  size_t nccIssueDomainCount = 0;
  size_t argumentNCCIssueDomainCount = 0;
  size_t synchronousWritebackCount = 0;
  size_t directDTEIssueDomainCount = 0;
  for (const wafer::TargetCallDescriptor &descriptor : descriptors) {
    EXPECT_TRUE(llvm::StringRef(descriptor.symbol).starts_with("wafer_tx81_"));
    EXPECT_TRUE(symbols.insert(descriptor.symbol).second);
    EXPECT_EQ(wafer::findTargetCallDescriptor(descriptor.symbol), &descriptor);
    EXPECT_EQ(wafer::findTargetCallDescriptor(descriptor.semantic),
              &descriptor);
    if (!descriptor.issueDomain) {
      EXPECT_FALSE(wafer::getTargetCallTSMEngine(descriptor));
      continue;
    }
    ++issueDomainCount;
    EXPECT_EQ(wafer::getTargetCallTSMEngine(descriptor),
              descriptor.issueDomain->engine);
    if (descriptor.issueDomain->engine ==
        wafer::TargetCallTSMEngine::DirectDTE) {
      ++directDTEIssueDomainCount;
      EXPECT_FALSE(descriptor.issueDomain->nccWorkerArgument);
      EXPECT_EQ(descriptor.issueDomain->completionBehavior,
                wafer::TargetNCCCompletionBehavior::None);
    } else {
      ++nccIssueDomainCount;
      ASSERT_TRUE(descriptor.issueDomain->nccWorkerArgument);
      ++argumentNCCIssueDomainCount;
      EXPECT_EQ(*descriptor.issueDomain->nccWorkerArgument + 1,
                descriptor.arguments.size());
      EXPECT_EQ(descriptor.arguments.back(), wafer::TargetCallScalarType::I32);
      const auto *peripheral =
          std::get_if<wafer::TargetPeripheralOperation>(&descriptor.semantic);
      const bool synchronous =
          peripheral &&
          (*peripheral == wafer::TargetPeripheralOperation::ArgMaximum ||
           *peripheral == wafer::TargetPeripheralOperation::ArgMinimum);
      EXPECT_EQ(
          descriptor.issueDomain->completionBehavior,
          synchronous
              ? wafer::TargetNCCCompletionBehavior::SynchronousWriteback
              : wafer::TargetNCCCompletionBehavior::OrderedAsynchronousIssue);
      synchronousWritebackCount += synchronous;
    }
  }
  EXPECT_EQ(issueDomainCount, 106u);
  EXPECT_EQ(nccIssueDomainCount, 104u);
  EXPECT_EQ(argumentNCCIssueDomainCount, 104u);
  EXPECT_EQ(synchronousWritebackCount, 2u);
  EXPECT_EQ(directDTEIssueDomainCount, 2u);
  EXPECT_EQ(wafer::findTargetCallDescriptor("wafer_tx81_unknown"), nullptr);

  const auto &join =
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::NCCJoin);
  EXPECT_EQ(join.result, wafer::TargetCallResultType::Void);
  ASSERT_EQ(join.arguments.size(), 1u);
  EXPECT_EQ(join.arguments.front(), wafer::TargetCallScalarType::I32);
  EXPECT_FALSE(join.issueDomain);
  const auto &send = wafer::getTargetCallDescriptor(
      wafer::TargetCallBuiltin::DirectDTESendPrepare);
  EXPECT_EQ(send.result, wafer::TargetCallResultType::I64);
  EXPECT_EQ(send.arguments.size(), 7u);
  EXPECT_EQ(send.arguments[0], wafer::TargetCallScalarType::I64);
  EXPECT_EQ(send.arguments[2], wafer::TargetCallScalarType::I32);
  EXPECT_EQ(
      wafer::getTargetCallDescriptor(wafer::TargetElementwiseOperation::Abs)
          .arguments.size(),
      5u);
  EXPECT_EQ(
      wafer::getTargetCallDescriptor(wafer::TargetElementwiseOperation::Add)
          .arguments.size(),
      6u);
  EXPECT_EQ(wafer::getTargetCallDescriptor(
                wafer::TargetConvolutionOperation::Convolution)
                .arguments.size(),
            32u);
  EXPECT_EQ(
      wafer::getTargetCallDescriptor(wafer::TargetPeripheralOperation::Bilinear)
          .arguments.size(),
      18u);
  EXPECT_EQ(
      wafer::getTargetCallDescriptor(wafer::TargetElementwiseOperation::Add)
          .symbol,
      "wafer_tx81_elementwise_add");
  EXPECT_EQ(
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::GemmOriented)
          .symbol,
      "wafer_tx81_gemm_oriented");
}

TEST(TargetCallRegistryTest, UsesCurrentWorkerAwareABI) {
  const auto &rdma =
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::RDMA);
  EXPECT_EQ(rdma.symbol, "wafer_tx81_rdma");
  EXPECT_EQ(rdma.arguments.size(), 12u);
  ASSERT_TRUE(rdma.issueDomain);
  ASSERT_TRUE(rdma.issueDomain->nccWorkerArgument);
  EXPECT_EQ(*rdma.issueDomain->nccWorkerArgument, 11u);
  EXPECT_EQ(wafer::findTargetCallDescriptor(rdma.symbol), &rdma);
  EXPECT_EQ(
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::GemmOriented)
          .symbol,
      "wafer_tx81_gemm_oriented");
  EXPECT_EQ(wafer::getTargetCallDescriptor(
                wafer::TargetCallBuiltin::DirectDTESendIssue)
                .symbol,
            "wafer_tx81_direct_dte_send_issue");
}

TEST(TargetCallRegistryTest, EveryDescriptorDecodesEveryABIField) {
  size_t decoded = 0;
  for (const wafer::TargetCallDescriptor &descriptor :
       wafer::getTargetCallDescriptors()) {
    std::vector<uint64_t> arguments = makeDecodableArguments(descriptor);
    wafer::TargetCallDecodeConfig config{16};
    auto payload =
        wafer::decodeTargetCallPayload(descriptor, config, arguments);
    ASSERT_TRUE(static_cast<bool>(payload))
        << descriptor.symbol << ": " << llvm::toString(payload.takeError());
    expectPayloadFields(descriptor, arguments, *payload);
    ++decoded;
  }
  EXPECT_EQ(decoded, 112u);
}

TEST(TargetCallRegistryTest, DecodesExplicitWorkerOneAndTwo) {
  const wafer::TargetCallDescriptor &descriptor =
      wafer::getTargetCallDescriptor(wafer::TargetElementwiseOperation::Add);
  ASSERT_TRUE(descriptor.issueDomain.has_value());
  ASSERT_TRUE(descriptor.issueDomain->nccWorkerArgument.has_value());
  for (wafer::TargetNCCWorker expected :
       {wafer::TargetNCCWorker::Worker1, wafer::TargetNCCWorker::Worker2}) {
    std::vector<uint64_t> arguments = makeDecodableArguments(descriptor);
    arguments[*descriptor.issueDomain->nccWorkerArgument] =
        static_cast<uint32_t>(expected);
    llvm::Expected<std::optional<wafer::TargetNCCWorker>> worker =
        wafer::decodeTargetCallNCCWorker(descriptor, arguments);
    ASSERT_TRUE(static_cast<bool>(worker))
        << llvm::toString(worker.takeError());
    ASSERT_TRUE(worker->has_value());
    EXPECT_EQ(**worker, expected);
    auto payload = wafer::decodeTargetCallPayload(descriptor, {1}, arguments);
    ASSERT_TRUE(static_cast<bool>(payload))
        << llvm::toString(payload.takeError());
  }
}

TEST(TargetCallRegistryTest, RejectsOutOfRangeExplicitWorker) {
  const wafer::TargetCallDescriptor &descriptor =
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::RDMA);
  std::vector<uint64_t> arguments = makeDecodableArguments(descriptor);
  arguments[*descriptor.issueDomain->nccWorkerArgument] =
      wafer::kTargetNCCWorkerCount;
  llvm::Expected<std::optional<wafer::TargetNCCWorker>> worker =
      wafer::decodeTargetCallNCCWorker(descriptor, arguments);
  ASSERT_FALSE(static_cast<bool>(worker));
  EXPECT_NE(llvm::toString(worker.takeError()).find("worker domain"),
            std::string::npos);
}

TEST(TargetCallRegistryTest, NCCJoinDecoderRejectsInvalidParticipantMasks) {
  const wafer::TargetCallDescriptor &descriptor =
      wafer::getTargetCallDescriptor(wafer::TargetCallBuiltin::NCCJoin);
  wafer::TargetCallDecodeConfig config{1};
  auto empty = wafer::decodeTargetCallPayload(descriptor, config, {0});
  ASSERT_FALSE(static_cast<bool>(empty));
  EXPECT_NE(llvm::toString(empty.takeError()).find("participant mask"),
            std::string::npos);
  const uint64_t outside =
      uint64_t{1} << static_cast<uint32_t>(wafer::kTargetNCCWorkerCount);
  auto outOfRange =
      wafer::decodeTargetCallPayload(descriptor, config, {outside});
  ASSERT_FALSE(static_cast<bool>(outOfRange));
  EXPECT_NE(llvm::toString(outOfRange.takeError()).find("worker domain"),
            std::string::npos);
}

TEST(TargetCallExecutionTest, ExecutesProductionTargetLLVMThroughTypedSink) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  ASSERT_EQ(targetLLVMModules->getModules().size(), 16u);
  std::string originalTriple =
      targetLLVMModules->getModules().front().getTargetTriple().str();
  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);

  RecordingSink sink;
  auto executable = wafer::compiler::createTargetCallExecutable(
      *targetLLVMModules, arguments);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  if (llvm::Error error = executable->begin(sink))
    FAIL() << llvm::toString(std::move(error));
  for (const wafer::compiler::TargetCallTileDescriptor &tile :
       executable->getInvocationDescriptor().tiles)
    if (llvm::Error error = executable->executeTile(tile.launchSlotId))
      FAIL() << llvm::toString(std::move(error));
  auto result = executable->finish();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(sink.began);
  EXPECT_TRUE(sink.invocationCompleted);
  EXPECT_FALSE(sink.aborted);
  EXPECT_EQ(sink.invocationTileCount, 16u);
  ASSERT_EQ(sink.tileDescriptors.size(), 16u);
  EXPECT_EQ(sink.tileDescriptors[0].cardId.getValue(), 0);
  EXPECT_EQ(sink.tileDescriptors[0].tileId.getValue(), 0);
  EXPECT_EQ(sink.tileDescriptors[0].launchSlotId.getValue(), 0);
  ASSERT_EQ(
      sink.tileDescriptors[0].tileEntryArguments.size(),
      targetLLVMModules->getModules().front().getTileEntryArguments().size());
  ASSERT_FALSE(sink.tileDescriptors[0].tileEntryArguments.empty());
  EXPECT_EQ(sink.tileDescriptors[0].slotValues, arguments[0].slots);
  EXPECT_EQ(sink.tileDescriptors[0].tileEntryArguments.front().dtype,
            targetLLVMModules->getModules()
                .front()
                .getTileEntryArguments()
                .front()
                .dtype);
  ASSERT_EQ(sink.completedCardIds.size(), 16u);
  ASSERT_EQ(sink.completedTileIds.size(), 16u);
  ASSERT_EQ(sink.completedLaunchSlots.size(), 16u);
  for (int64_t index = 0; index < 16; ++index) {
    EXPECT_EQ(sink.completedCardIds[static_cast<size_t>(index)], 0);
    EXPECT_EQ(sink.completedTileIds[static_cast<size_t>(index)], index);
    EXPECT_EQ(sink.completedLaunchSlots[static_cast<size_t>(index)], index);
  }
  EXPECT_EQ(result->completedTileCount, 16);
  EXPECT_EQ(result->issuedCommandCount, sink.commands.size());
  ASSERT_FALSE(sink.commands.empty());
  bool sawAdd = false;
  uint64_t totalAddElements = 0;
  std::vector<uint64_t> nextIssueOrdinal(16, 0);
  for (size_t index = 0; index < sink.commands.size(); ++index) {
    const wafer::compiler::TargetCommand &command = sink.commands[index];
    EXPECT_EQ(command.cardId.getValue(), 0);
    EXPECT_EQ(command.tileId.getValue(), command.launchSlotId.getValue());
    const int64_t launchSlot = command.launchSlotId.getValue();
    ASSERT_GE(launchSlot, 0);
    ASSERT_LT(launchSlot, 16);
    EXPECT_EQ(command.issueOrdinal,
              nextIssueOrdinal[static_cast<size_t>(launchSlot)]++);
    if (command.nccIssueDomain) {
      EXPECT_NE(command.nccIssueDomain->engine,
                wafer::TargetCallTSMEngine::DirectDTE);
      EXPECT_EQ(command.nccIssueDomain->worker,
                wafer::TargetNCCWorker::Worker0);
      EXPECT_EQ(command.nccIssueDomain->completionBehavior,
                wafer::TargetNCCCompletionBehavior::OrderedAsynchronousIssue);
    }
    if (const auto *elementwise =
            std::get_if<wafer::target::TargetElementwiseCommand>(
                &command.payload)) {
      sawAdd = true;
      ASSERT_TRUE(command.nccIssueDomain.has_value());
      EXPECT_EQ(command.nccIssueDomain->engine, wafer::TargetCallTSMEngine::CT);
      EXPECT_EQ(elementwise->operation, wafer::TargetElementwiseOperation::Add);
      EXPECT_TRUE(elementwise->rhs.has_value());
      totalAddElements += elementwise->elementCount;
      EXPECT_EQ(elementwise->format, wafer::LogicalFormat::F32);
    }
  }
  EXPECT_TRUE(sawAdd);
  EXPECT_EQ(totalAddElements, 8u);
  EXPECT_EQ(targetLLVMModules->getModules().front().getTargetTriple(),
            originalTriple);
}

TEST(TargetCallExecutionTest, CarriesExplicitWorkerOneAndTwoIntoCommands) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Module &module = const_cast<llvm::Module &>(
      targetLLVMModules->getModules().front().getModule());
  size_t rewritten = 0;
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
        llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
        const wafer::TargetCallDescriptor *descriptor =
            callee ? wafer::findTargetCallDescriptor(callee->getName())
                   : nullptr;
        if (!descriptor || !descriptor->issueDomain ||
            !descriptor->issueDomain->nccWorkerArgument)
          continue;
        wafer::TargetNCCWorker worker =
            descriptor->issueDomain->engine == wafer::TargetCallTSMEngine::CT
                ? wafer::TargetNCCWorker::Worker2
                : wafer::TargetNCCWorker::Worker1;
        call->setArgOperand(
            *descriptor->issueDomain->nccWorkerArgument,
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(module.getContext()),
                                   static_cast<uint32_t>(worker)));
        ++rewritten;
      }
  ASSERT_GT(rewritten, 0u);

  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  bool sawWorker1 = false;
  bool sawWorker2 = false;
  for (const wafer::compiler::TargetCommand &command : sink.commands) {
    if (!command.nccIssueDomain)
      continue;
    sawWorker1 |=
        command.nccIssueDomain->worker == wafer::TargetNCCWorker::Worker1;
    sawWorker2 |=
        command.nccIssueDomain->worker == wafer::TargetNCCWorker::Worker2;
  }
  EXPECT_TRUE(sawWorker1);
  EXPECT_TRUE(sawWorker2);
}

TEST(TargetCallExecutionTest,
     CarriesSynchronousWritebackBehaviorIntoDynamicCommand) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  ASSERT_EQ(targetLLVMModules->getModules().size(), 16u);

  llvm::Module &module = const_cast<llvm::Module &>(
      targetLLVMModules->getModules().front().getModule());
  llvm::Function *entry = module.getFunction(
      targetLLVMModules->getModules().front().getEntrySymbol());
  ASSERT_NE(entry, nullptr);
  llvm::IRBuilder<> builder(entry->getEntryBlock().getTerminator());
  for (wafer::TargetPeripheralOperation kind :
       {wafer::TargetPeripheralOperation::ArgMaximum,
        wafer::TargetPeripheralOperation::ArgMinimum}) {
    const wafer::TargetCallDescriptor &descriptor =
        wafer::getTargetCallDescriptor(kind);
    llvm::SmallVector<llvm::Type *, 10> argumentTypes;
    llvm::SmallVector<llvm::Value *, 10> callArguments;
    std::vector<uint64_t> values = makeDecodableArguments(descriptor);
    ASSERT_EQ(values.size(), descriptor.arguments.size());
    for (auto [type, value] : llvm::zip_equal(descriptor.arguments, values)) {
      if (type == wafer::TargetCallScalarType::I64) {
        argumentTypes.push_back(builder.getInt64Ty());
        callArguments.push_back(builder.getInt64(value));
      } else {
        argumentTypes.push_back(builder.getInt32Ty());
        callArguments.push_back(builder.getInt32(static_cast<uint32_t>(value)));
      }
    }
    llvm::FunctionCallee callee = module.getOrInsertFunction(
        descriptor.symbol,
        llvm::FunctionType::get(builder.getVoidTy(), argumentTypes,
                                /*isVarArg=*/false));
    builder.CreateCall(callee, callArguments);
  }

  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  bool sawArgMax = false;
  bool sawArgMin = false;
  size_t synchronousArgExtremaCount = 0;
  for (const wafer::compiler::TargetCommand &command : sink.commands) {
    const auto *payload =
        std::get_if<wafer::target::TargetPeripheralArgExtremaCommand>(
            &command.payload);
    if (!payload)
      continue;
    sawArgMax |=
        payload->operation == wafer::TargetPeripheralOperation::ArgMaximum;
    sawArgMin |=
        payload->operation == wafer::TargetPeripheralOperation::ArgMinimum;
    ASSERT_TRUE(command.nccIssueDomain.has_value());
    EXPECT_EQ(command.nccIssueDomain->engine, wafer::TargetCallTSMEngine::CT);
    EXPECT_EQ(command.nccIssueDomain->worker, wafer::TargetNCCWorker::Worker0);
    EXPECT_EQ(command.nccIssueDomain->completionBehavior,
              wafer::TargetNCCCompletionBehavior::SynchronousWriteback);
    ++synchronousArgExtremaCount;
  }
  EXPECT_TRUE(sawArgMax);
  EXPECT_TRUE(sawArgMin);
  EXPECT_EQ(synchronousArgExtremaCount, 2u);
}

TEST(TargetCallExecutionTest, SinkFailureAbortsWithoutPartialResult) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  sink.failAtIssue = 0;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(
      llvm::toString(result.takeError()).find("injected command sink failure"),
      std::string::npos);
  EXPECT_TRUE(sink.began);
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.invocationCompleted);
  EXPECT_TRUE(sink.commands.empty());
  EXPECT_TRUE(sink.completedLaunchSlots.empty());
}

TEST(TargetCallExecutionTest, LateTileFailureAbortsTheWholeInvocation) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  sink.failAtLaunchSlot = 1;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(
      llvm::toString(result.takeError()).find("injected command sink failure"),
      std::string::npos);
  ASSERT_TRUE(sink.failedLaunchSlot.has_value());
  EXPECT_GT(*sink.failedLaunchSlot, 0);
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.invocationCompleted);
  EXPECT_TRUE(sink.commands.empty());
  EXPECT_TRUE(sink.completedLaunchSlots.empty());
}

TEST(TargetCallExecutionTest, NativeIllegalInlineAssemblyFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Module &module = const_cast<llvm::Module &>(
      targetLLVMModules->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::IRBuilder<> builder(&entry->getEntryBlock().front());
  llvm::FunctionType *type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module.getContext()), /*isVarArg=*/false);
  llvm::InlineAsm *assembly =
      llvm::InlineAsm::get(type, "", "", /*hasSideEffects=*/true);
  llvm::CallInst *call = builder.CreateCall(assembly);
  auto restore = llvm::make_scope_exit([&] { call->eraseFromParent(); });

  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("inline assembly"),
            std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallExecutionTest, NativeTrapFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Module &module = const_cast<llvm::Module &>(
      targetLLVMModules->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::IRBuilder<> builder(&entry->getEntryBlock().front());
  llvm::Function *trap =
      llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::trap);
  builder.CreateCall(trap);

  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("unsupported intrinsic"),
            std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallExecutionTest, NativeAddressDereferenceFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Module &module = const_cast<llvm::Module &>(
      targetLLVMModules->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::IRBuilder<> builder(&entry->getEntryBlock().front());
  llvm::Type *i64 = llvm::Type::getInt64Ty(module.getContext());
  llvm::Value *syntheticAddress = builder.getInt64(0x100000);
  llvm::Value *pointer = builder.CreateIntToPtr(
      syntheticAddress, llvm::PointerType::get(module.getContext(), 0));
  (void)builder.CreateLoad(i64, pointer);

  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  std::string error = llvm::toString(result.takeError());
  EXPECT_TRUE(error.find("inttoptr") != std::string::npos ||
              error.find("load") != std::string::npos)
      << error;
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallExecutionTest, NativePointerSelectFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Module &module = const_cast<llvm::Module &>(
      targetLLVMModules->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::Type *pointer = llvm::PointerType::get(module.getContext(), 0);
  auto *nullPointer =
      llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointer));
  (void)llvm::SelectInst::Create(
      llvm::ConstantInt::getTrue(module.getContext()), nullPointer, nullPointer,
      "pointer-control", &entry->getEntryBlock().front());

  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("select"),
            std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallExecutionTest, NativePointerCompareFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Module &module = const_cast<llvm::Module &>(
      targetLLVMModules->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  auto *pointer = llvm::PointerType::get(module.getContext(), 0);
  auto *nullPointer = llvm::ConstantPointerNull::get(pointer);
  (void)new llvm::ICmpInst(&entry->getEntryBlock().front(),
                           llvm::ICmpInst::ICMP_EQ, nullPointer, nullPointer,
                           "pointer-control");

  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("icmp"), std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallExecutionTest, NativePointerPhiFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Module &module = const_cast<llvm::Module &>(
      targetLLVMModules->getModules().front().getModule());
  llvm::Function *entry = module.getFunction("main");
  ASSERT_NE(entry, nullptr);
  llvm::BasicBlock *loop =
      llvm::BasicBlock::Create(module.getContext(), "pointer-control", entry);
  llvm::IRBuilder<> builder(loop);
  auto *pointer = llvm::PointerType::get(module.getContext(), 0);
  llvm::PHINode *phi = builder.CreatePHI(pointer, 1);
  builder.CreateBr(loop);
  phi->addIncoming(llvm::ConstantPointerNull::get(pointer), loop);

  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("phi"), std::string::npos);
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallExecutionTest, WrongTargetCallSignatureFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Module &module = const_cast<llvm::Module &>(
      targetLLVMModules->getModules().front().getModule());
  llvm::Function *original = module.getFunction("wafer_tx81_elementwise_add");
  ASSERT_NE(original, nullptr);
  llvm::SmallVector<llvm::Instruction *, 2> calls;
  for (llvm::User *user : original->users())
    calls.push_back(llvm::cast<llvm::Instruction>(user));
  for (llvm::Instruction *call : calls)
    call->eraseFromParent();
  original->eraseFromParent();
  llvm::FunctionType *wrongType = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module.getContext()),
      {llvm::Type::getInt64Ty(module.getContext())}, /*isVarArg=*/false);
  llvm::Function *wrong =
      llvm::Function::Create(wrongType, llvm::GlobalValue::ExternalLinkage,
                             "wafer_tx81_elementwise_add", module);
  (void)wrong;

  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  std::string error = llvm::toString(result.takeError());
  EXPECT_NE(error.find("argument count"), std::string::npos) << error;
  EXPECT_FALSE(sink.began);
}

TEST(TargetCallExecutionTest, SlotMismatchFailsBeforeSinkBegin) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  arguments.front().slots.clear();
  RecordingSink sink;
  auto result =
      wafer::compiler::executeTargetCalls(*targetLLVMModules, arguments, sink);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("slot count"),
            std::string::npos);
  EXPECT_FALSE(sink.began);
  EXPECT_FALSE(sink.aborted);
}

TEST(TargetCallExecutionTest, RejectsTileReentryAndAbortsInvocation) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  auto executable = wafer::compiler::createTargetCallExecutable(
      *targetLLVMModules, arguments);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  sink.reenterExecutable = &*executable;
  ASSERT_FALSE(static_cast<bool>(executable->begin(sink)));
  llvm::Error executionError = executable->executeTile(wafer::LaunchSlotId(0));
  ASSERT_TRUE(static_cast<bool>(executionError));
  EXPECT_NE(llvm::toString(std::move(executionError)).find("already running"),
            std::string::npos);
  EXPECT_TRUE(sink.attemptedReentry);
  EXPECT_NE(sink.reentryDiagnostic.find("already running"), std::string::npos);
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.invocationCompleted);
}

TEST(TargetCallExecutionTest, InvocationCompletionFailureReturnsNoResult) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  sink.failCompleteInvocation = true;
  auto executable = wafer::compiler::createTargetCallExecutable(
      *targetLLVMModules, arguments);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  ASSERT_FALSE(static_cast<bool>(executable->begin(sink)));
  for (const wafer::compiler::TargetCallTileDescriptor &tile :
       executable->getInvocationDescriptor().tiles)
    ASSERT_FALSE(static_cast<bool>(executable->executeTile(tile.launchSlotId)));
  auto result = executable->finish();
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(
      llvm::toString(result.takeError()).find("invocation completion failure"),
      std::string::npos);
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.invocationCompleted);
}

TEST(TargetCallExecutionTest, RunningDestructionAbortsPrivateSinkState) {
  std::string diagnostics;
  auto targetLLVMModules = compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  std::vector<wafer::compiler::TargetCallTileArguments> arguments =
      makeInvocationArguments(*targetLLVMModules);
  RecordingSink sink;
  {
    auto executable = wafer::compiler::createTargetCallExecutable(
        *targetLLVMModules, arguments);
    ASSERT_TRUE(static_cast<bool>(executable))
        << llvm::toString(executable.takeError());
    ASSERT_FALSE(static_cast<bool>(executable->begin(sink)));
  }
  EXPECT_TRUE(sink.aborted);
  EXPECT_FALSE(sink.invocationCompleted);
  EXPECT_NE(sink.abortDiagnostic.find("destroyed before finish"),
            std::string::npos);
}

} // namespace
