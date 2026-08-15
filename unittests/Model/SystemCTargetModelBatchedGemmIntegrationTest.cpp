//===- SystemCTargetModelBatchedGemmIntegrationTest.cpp -----------------===//

#include "Wafer/Model/SystemCTargetModel.h"

#include "Wafer/Compiler/TargetCallFrontend.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Target/PhysicalTensorCodec.h"

#include "Wafer/Compiler/CompilationInternal.h"
#include "Wafer/Compiler/CardExecutableInternal.h"

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

frontend::ProgramBoundaryBinding
singlePartitionBoundary(int64_t index, llvm::ArrayRef<int64_t> shape) {
  frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = "f16";
  frontend::ProgramPartitionSlice slice;
  slice.partitionId = 0;
  slice.replicaId = 0;
  slice.offsets.assign(shape.size(), 0);
  slice.sizes.assign(shape.begin(), shape.end());
  slice.strides.assign(shape.size(), 1);
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

llvm::Expected<TargetLLVMModules>
compileBatchedGemmTargetModules(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<2x1x128xf16>,
                  %rhs: tensor<2x128x16xf16>) -> tensor<2x1x16xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<2x1x16xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x1x16xf16>) -> tensor<2x1x16xf16>
    %product = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<2x1x128xf16>, tensor<2x128x16xf16>)
        outs(%init : tensor<2x1x16xf16>) -> tensor<2x1x16xf16>
    return %product : tensor<2x1x16xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError("failed to parse batched GEMM model module");

  frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {singlePartitionBoundary(0, {2, 1, 128}),
                               singlePartitionBoundary(1, {2, 128, 16})};
  program.distributedOutputs = {singlePartitionBoundary(0, {2, 1, 16})};
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
  return compileCardExecutableToTargetLLVMModules(*executable,
                                                           diagnostics);
}

class RecordingTargetSink final : public TargetCommandSink {
public:
  llvm::Error begin(const TargetCallInvocationDescriptor &) override {
    began = true;
    return llvm::Error::success();
  }

  llvm::Expected<uint64_t> issue(const TargetCommand &command) override {
    commands.push_back(command);
    return nextEvent++;
  }

  llvm::Error completeTile(CardId, TileId,
                           LaunchSlotId) override {
    return llvm::Error::success();
  }
  llvm::Error completeInvocation() override {
    invocationCompleted = true;
    return llvm::Error::success();
  }
  void abort(llvm::StringRef) override { aborted = true; }

  bool began = false;
  bool invocationCompleted = false;
  bool aborted = false;
  uint64_t nextEvent = 1;
  std::vector<TargetCommand> commands;
};

TEST(SystemCTargetModelBatchedGemmIntegrationTest,
     PreservesImplicitNCxStorageFromSourceThroughTargetModel) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      compileBatchedGemmTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  ASSERT_EQ(targetLLVMModules->getModules().size(), 16u);

  std::vector<TargetCallTileArguments> arguments;
  std::vector<TargetModelInputBinding> inputs;
  NumericTensorKey lhsKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F16, PhysicalTensorLayout::Tensor, {2, 1, 128}));
  NumericTensorKey rhsKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F16, PhysicalTensorLayout::Tensor, {2, 128, 16}));
  NumericTensorKey outputKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F16, PhysicalTensorLayout::Tensor, {2, 1, 16}));

  std::vector<RawLogicalValue> lhs(2 * 128, {LogicalFormat::F16, UINT64_C(0)});
  for (size_t k = 0; k < 128; ++k) {
    lhs[k].bits = k < 64 ? UINT64_C(0x3c00) : UINT64_C(0x4000);
    lhs[128 + k].bits = k < 64 ? UINT64_C(0x4200) : UINT64_C(0x4400);
  }
  std::vector<RawLogicalValue> rhs(2 * 128 * 16,
                                   {LogicalFormat::F16, UINT64_C(0x3c00)});
  const std::array<const NumericTensorKey *, 2> inputKeys{&lhsKey, &rhsKey};
  const std::array<llvm::ArrayRef<RawLogicalValue>, 2> logicalInputs{lhs, rhs};
  for (const TargetLLVMModule &module : targetLLVMModules->getModules()) {
    const int64_t launchSlot = module.getLaunchSlotId().getValue();
    arguments.push_back({module.getCardId(),
                         module.getTileId(),
                         module.getLaunchSlotId(),
                         {}});
    size_t inputCount = 0;
    size_t outputCount = 0;
    for (const KernelABISlot &slot : module.getKernelABISlots()) {
      const bool tileOwned = slot.role == KernelABISlotRole::Workspace ||
                             slot.role == KernelABISlotRole::TransportStatus;
      const uint64_t base =
          tileOwned
              ? UINT64_C(0x10000000) +
                    static_cast<uint64_t>(launchSlot) * UINT64_C(0x100000) +
                    static_cast<uint64_t>(slot.ordinal) * UINT64_C(0x10000)
              : (slot.role == KernelABISlotRole::Output
                     ? UINT64_C(0x400000)
                     : UINT64_C(0x100000) +
                           static_cast<uint64_t>(slot.resourceIndex) *
                               UINT64_C(0x100000));
      arguments.back().slots.push_back(base);
      if (slot.role == KernelABISlotRole::Output) {
        EXPECT_EQ(slot.shape, (std::vector<int64_t>{2, 1, 16}));
        ++outputCount;
        continue;
      }
      if (slot.role != KernelABISlotRole::UserInput)
        continue;
      ASSERT_LT(slot.resourceIndex, 2);
      const size_t inputIndex = static_cast<size_t>(slot.resourceIndex);
      EXPECT_EQ(slot.shape,
                (inputIndex == 0 ? std::vector<int64_t>{2, 1, 128}
                                 : std::vector<int64_t>{2, 128, 16}));
      std::vector<uint8_t> bytes =
          llvm::cantFail(packPhysicalTensorLogicalValues(
              *inputKeys[inputIndex], logicalInputs[inputIndex], UINT8_C(0)));
      ASSERT_EQ(bytes.size(), static_cast<uint64_t>(slot.byteSize));
      if (launchSlot == 0)
        inputs.push_back(
            {getTargetModelResourceId(module.getCardId(),
                                      module.getTileId(), slot.role,
                                      slot.resourceIndex),
             std::move(bytes)});
      ++inputCount;
    }
    EXPECT_EQ(inputCount, 2u);
    EXPECT_EQ(outputCount, 1u);
  }

  // Spatial synthesis shards N across Tiles while preserving each
  // batch-2 operation. The numeric execution below uses a separately prepared
  // frontend from the same complete target targetLLVMModules and explicit Tile
  // triples.
  RecordingTargetSink recordingSink;
  llvm::Expected<TargetCallExecutionResult> decoded =
      executeTargetCallFrontend(*targetLLVMModules, arguments, recordingSink);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_TRUE(recordingSink.began);
  EXPECT_TRUE(recordingSink.invocationCompleted);
  EXPECT_FALSE(recordingSink.aborted);
  std::vector<const TargetGemmCommand *> gemms;
  for (const TargetCommand &command : recordingSink.commands)
    if (const auto *gemm = std::get_if<TargetGemmCommand>(&command.payload))
      gemms.push_back(gemm);
  ASSERT_EQ(gemms.size(), 16u);
  for (const TargetGemmCommand *gemm : gemms) {
    EXPECT_EQ(gemm->batchCount, 2u);
    EXPECT_EQ(gemm->m, 1u);
    EXPECT_EQ(gemm->k, 128u);
    EXPECT_EQ(gemm->n, 1u);
  }

  llvm::Expected<TargetCallExecutable> frontend =
      createTargetCallExecutable(*targetLLVMModules, arguments);
  ASSERT_TRUE(static_cast<bool>(frontend))
      << llvm::toString(frontend.takeError());
  llvm::Expected<TargetModelResult> result =
      executeSystemCTargetModel(std::move(*frontend), inputs,
                                TargetModelKernelBudget::create(
                                    FormalNumericWorkBudget::create(
                                        /*maximumScalarEvaluations=*/4096,
                                        /*maximumFusedMultiplyAdds=*/4096),
                                    /*maximumMovementBytes=*/UINT64_C(1) << 20,
                                    /*maximumMovementSegments=*/4096));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->outputs.size(), 1u);
  // 64 * 1 + 64 * 2 = 192; 64 * 3 + 64 * 4 = 448. Both are
  // exactly representable in f16. A Cx/NCx block-order mismatch instead mixes
  // the two batches at the 64-channel boundary.
  for (const TargetModelOutput &modelOutput : result->outputs) {
    llvm::Expected<std::vector<RawLogicalValue>> output =
        unpackPhysicalTensorLogicalValues(outputKey, modelOutput.bytes);
    ASSERT_TRUE(static_cast<bool>(output))
        << llvm::toString(output.takeError());
    ASSERT_EQ(output->size(), 2u * 16u);
    for (size_t batch = 0; batch < 2; ++batch) {
      const uint64_t expected =
          batch == 0 ? UINT64_C(0x5a00) : UINT64_C(0x5f00);
      for (size_t column = 0; column < 16; ++column) {
        const size_t index = batch * 16 + column;
        EXPECT_EQ((*output)[index].bits, expected);
      }
    }
  }
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
