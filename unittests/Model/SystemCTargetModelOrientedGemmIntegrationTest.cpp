//===- SystemCTargetModelOrientedGemmIntegrationTest.cpp ----------------===//

#include "Wafer/Model/SystemCTargetModel.h"

#include "Wafer/Compiler/TargetCallFrontend.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Target/PhysicalTensorCodec.h"

#include "Wafer/Compiler/CardExecutableInternal.h"
#include "Wafer/Compiler/CompilationInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;
using namespace wafer::target;

frontend::ProgramBoundaryBinding boundary(int64_t index) {
  frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = frontend::ProgramDistributionKind::Replicated;
  binding.globalShape = {16, 2, 2};
  binding.localShape = {16, 2, 2};
  binding.dtype = "f16";
  frontend::ProgramPartitionSlice slice;
  slice.partitionId = 0;
  slice.replicaId = 0;
  slice.offsets = {0, 0, 0};
  slice.sizes = {16, 2, 2};
  slice.strides = {1, 1, 1};
  binding.partitionSlices.push_back(std::move(slice));
  return binding;
}

std::shared_ptr<mlir::MLIRContext> createCompilerContext() {
  mlir::DialectRegistry registry;
  compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

llvm::Error
rewriteGemmAsOrientationChain(TargetLLVMModules &targetLLVMModules) {
  if (targetLLVMModules.getModules().size() != 16)
    return llvm::createStringError("expected complete Tile domain");
  for (const TargetLLVMModule &immutableTargetModule :
       targetLLVMModules.getModules()) {
    TargetLLVMModule &targetModule =
        const_cast<TargetLLVMModule &>(immutableTargetModule);
    llvm::Module &module = const_cast<llvm::Module &>(targetModule.getModule());
    llvm::Function *entry = module.getFunction(targetModule.getEntrySymbol());
    if (!entry)
      return llvm::createStringError("missing target entry");

    llvm::CallInst *gemm = nullptr;
    bool hasWdma = false;
    for (llvm::Instruction &instruction : llvm::instructions(entry)) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
      if (!call || !call->getCalledFunction())
        continue;
      llvm::StringRef name = call->getCalledFunction()->getName();
      if (name == "wafer_tx81_gemm_v3") {
        if (gemm)
          return llvm::createStringError("expected one GEMM call per Tile");
        gemm = call;
      } else if (name == "wafer_tx81_wdma_v3") {
        hasWdma = true;
      }
    }
    if (!gemm || !hasWdma)
      return llvm::createStringError("missing Tile-local GEMM or WDMA call");

    llvm::Value *lhs = gemm->getArgOperand(0);
    llvm::Value *rhs = gemm->getArgOperand(1);
    llvm::Value *destination = gemm->getArgOperand(2);

    llvm::IRBuilder<> builder(gemm);
    llvm::Type *i64 = builder.getInt64Ty();
    llvm::Type *i32 = builder.getInt32Ty();
    llvm::FunctionType *functionType = llvm::FunctionType::get(
        builder.getVoidTy(),
        {i64, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32}, false);
    llvm::FunctionCallee oriented =
        module.getOrInsertFunction("wafer_tx81_gemm_oriented_v3", functionType);
    auto emit = [&](llvm::Value *callLhs, llvm::Value *callRhs,
                    llvm::Value *callDestination,
                    TargetGemmOrientation lhsOrientation,
                    TargetGemmOrientation rhsOrientation) {
      builder.CreateCall(
          oriented, {callLhs, callRhs, callDestination, gemm->getArgOperand(3),
                     gemm->getArgOperand(4), gemm->getArgOperand(5),
                     gemm->getArgOperand(6), gemm->getArgOperand(7),
                     builder.getInt32(static_cast<uint32_t>(lhsOrientation)),
                     builder.getInt32(static_cast<uint32_t>(rhsOrientation)),
                     gemm->getArgOperand(8)});
    };
    emit(lhs, rhs, destination, TargetGemmOrientation::Normal,
         TargetGemmOrientation::Normal);
    emit(lhs, rhs, destination, TargetGemmOrientation::Normal,
         TargetGemmOrientation::Transpose);
    emit(lhs, rhs, destination, TargetGemmOrientation::Transpose,
         TargetGemmOrientation::Normal);
    emit(lhs, rhs, destination, TargetGemmOrientation::Transpose,
         TargetGemmOrientation::Transpose);
    gemm->eraseFromParent();
  }
  return llvm::Error::success();
}

llvm::Expected<TargetLLVMModules>
compileOrientedGemmTargetModules(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<16x2x2xf16>, %rhs: tensor<16x2x2xf16>)
      -> tensor<16x2x2xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<16x2x2xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<16x2x2xf16>) -> tensor<16x2x2xf16>
    %product = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<16x2x2xf16>, tensor<16x2x2xf16>)
        outs(%init : tensor<16x2x2xf16>) -> tensor<16x2x2xf16>
    return %product : tensor<16x2x2xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError("failed to parse oriented GEMM module");

  frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0), boundary(1)};
  program.distributedOutputs = {boundary(0)};
  llvm::Expected<ExecutionConfig> config =
      ExecutionConfig::createForSingleCard(1, RuntimeLaunchKind::Kernel);
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  llvm::Expected<CardExecutable> executable =
      compiler::detail::buildCardExecutable(
          context, *tensorProgram, std::move(program), *config,
          OptimizationConfig::none(), diagnostics, std::nullopt);
  if (!executable)
    return executable.takeError();
  tensorProgram = nullptr;
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      compileCardExecutableToTargetLLVMModules(*executable, diagnostics);
  if (!targetLLVMModules)
    return targetLLVMModules.takeError();
  if (llvm::Error error = rewriteGemmAsOrientationChain(*targetLLVMModules))
    return std::move(error);
  return std::move(*targetLLVMModules);
}

class RecordingTargetSink final : public TargetCommandSink {
public:
  llvm::Error begin(const TargetCallInvocationDescriptor &) override {
    return llvm::Error::success();
  }
  llvm::Expected<uint64_t> issue(const TargetCommand &command) override {
    commands.push_back(command);
    return nextEvent++;
  }
  llvm::Error completeTile(CardId, TileId, LaunchSlotId) override {
    return llvm::Error::success();
  }
  llvm::Error completeInvocation() override { return llvm::Error::success(); }
  void abort(llvm::StringRef) override {}

  uint64_t nextEvent = 1;
  std::vector<TargetCommand> commands;
};

TEST(SystemCTargetModelOrientedGemmIntegrationTest,
     ExecutesNNNTTNTTThroughCurrentTargetABI) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      compileOrientedGemmTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  ASSERT_EQ(targetLLVMModules->getModules().size(), 16u);

  NumericTensorKey tensorKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F16, PhysicalTensorLayout::Tensor, {16, 2, 2}));
  const std::array<RawLogicalValue, 4> lhsMatrix{
      {{LogicalFormat::F16, UINT64_C(0x3c00)},
       {LogicalFormat::F16, UINT64_C(0x4000)},
       {LogicalFormat::F16, UINT64_C(0x4200)},
       {LogicalFormat::F16, UINT64_C(0x4400)}}};
  const std::array<RawLogicalValue, 4> rhsMatrix{
      {{LogicalFormat::F16, UINT64_C(0x3c00)},
       {LogicalFormat::F16, UINT64_C(0x4000)},
       {LogicalFormat::F16, UINT64_C(0)},
       {LogicalFormat::F16, UINT64_C(0x3c00)}}};
  std::array<std::vector<RawLogicalValue>, 2> logicalInputs;
  for (size_t batch = 0; batch < 16; ++batch) {
    logicalInputs[0].insert(logicalInputs[0].end(), lhsMatrix.begin(),
                            lhsMatrix.end());
    logicalInputs[1].insert(logicalInputs[1].end(), rhsMatrix.begin(),
                            rhsMatrix.end());
  }
  std::vector<TargetCallTileArguments> arguments;
  std::vector<TargetModelInputBinding> inputs;
  for (const TargetLLVMModule &module : targetLLVMModules->getModules()) {
    const int64_t launchSlot = module.getLaunchSlotId().getValue();
    arguments.push_back(
        {module.getCardId(), module.getTileId(), module.getLaunchSlotId(), {}});
    size_t inputCount = 0;
    size_t outputCount = 0;
    for (const KernelABISlot &slot : module.getKernelABISlots()) {
      const bool tileOwned = slot.role == KernelABISlotRole::Workspace ||
                             slot.role == KernelABISlotRole::TransportStatus;
      arguments.back().slots.push_back(
          tileOwned
              ? UINT64_C(0x10000000) +
                    static_cast<uint64_t>(launchSlot) * UINT64_C(0x100000) +
                    static_cast<uint64_t>(slot.ordinal) * UINT64_C(0x10000)
              : (slot.role == KernelABISlotRole::Output
                     ? UINT64_C(0x400000)
                     : UINT64_C(0x100000) +
                           static_cast<uint64_t>(slot.resourceIndex) *
                               UINT64_C(0x100000)));
      if (slot.role == KernelABISlotRole::Output) {
        EXPECT_EQ(slot.shape, (std::vector<int64_t>{16, 2, 2}));
        ++outputCount;
        continue;
      }
      if (slot.role != KernelABISlotRole::UserInput)
        continue;
      ASSERT_LT(slot.resourceIndex, 2);
      EXPECT_EQ(slot.shape, (std::vector<int64_t>{16, 2, 2}));
      std::vector<uint8_t> bytes =
          llvm::cantFail(packPhysicalTensorLogicalValues(
              tensorKey, logicalInputs[static_cast<size_t>(slot.resourceIndex)],
              UINT8_C(0)));
      ASSERT_EQ(bytes.size(), static_cast<uint64_t>(slot.byteSize));
      if (launchSlot == 0)
        inputs.push_back(
            {getTargetModelResourceId(module.getCardId(), module.getTileId(),
                                      slot.role, slot.resourceIndex),
             std::move(bytes)});
      ++inputCount;
    }
    EXPECT_EQ(inputCount, 2u);
    EXPECT_EQ(outputCount, 1u);
  }

  RecordingTargetSink recording;
  llvm::Expected<TargetCallExecutionResult> decoded =
      executeTargetCallFrontend(*targetLLVMModules, arguments, recording);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  std::array<
      std::vector<std::pair<TargetGemmOrientation, TargetGemmOrientation>>, 16>
      orientationsByLaunchSlot;
  for (const TargetCommand &command : recording.commands)
    if (const auto *gemm = std::get_if<TargetGemmCommand>(&command.payload)) {
      const int64_t launchSlot = command.launchSlotId.getValue();
      ASSERT_GE(launchSlot, 0);
      ASSERT_LT(launchSlot, 16);
      orientationsByLaunchSlot[static_cast<size_t>(launchSlot)].emplace_back(
          gemm->lhsOrientation, gemm->rhsOrientation);
    }
  const std::vector<std::pair<TargetGemmOrientation, TargetGemmOrientation>>
      expectedOrder{
          {TargetGemmOrientation::Normal, TargetGemmOrientation::Normal},
          {TargetGemmOrientation::Normal, TargetGemmOrientation::Transpose},
          {TargetGemmOrientation::Transpose, TargetGemmOrientation::Normal},
          {TargetGemmOrientation::Transpose, TargetGemmOrientation::Transpose}};
  for (const auto &orientations : orientationsByLaunchSlot)
    EXPECT_EQ(orientations, expectedOrder);

  llvm::Expected<TargetCallExecutable> frontend =
      createTargetCallExecutable(*targetLLVMModules, arguments);
  ASSERT_TRUE(static_cast<bool>(frontend))
      << llvm::toString(frontend.takeError());
  llvm::Expected<TargetModelResult> result = executeSystemCTargetModel(
      std::move(*frontend), inputs,
      TargetModelKernelBudget::create(
          FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/4096,
                                          /*maximumFusedMultiplyAdds=*/4096),
          /*maximumMovementBytes=*/UINT64_C(1) << 20,
          /*maximumMovementSegments=*/4096));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->outputs.size(), 1u);
  const std::array<uint64_t, 4> expectedMatrix{
      UINT64_C(0x4700), UINT64_C(0x4200), UINT64_C(0x4900), UINT64_C(0x4400)};
  for (const TargetModelOutput &modelOutput : result->outputs) {
    llvm::Expected<std::vector<RawLogicalValue>> output =
        unpackPhysicalTensorLogicalValues(tensorKey, modelOutput.bytes);
    ASSERT_TRUE(static_cast<bool>(output))
        << llvm::toString(output.takeError());
    ASSERT_EQ(output->size(), 16u * 4u);
    for (size_t index = 0; index < output->size(); ++index) {
      EXPECT_EQ((*output)[index].bits, expectedMatrix[index % 4]);
    }
  }
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
