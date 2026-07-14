//===- SystemCTargetModelIntegrationTest.cpp - SystemC model gate -------===//

#include "Wafer/Model/SystemCTargetModel.h"

#include "Wafer/InitAll.h"
#include "Wafer/Target/PhysicalTensorCodec.h"

#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

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

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

frontend::ProgramBoundaryBinding boundary(int64_t index) {
  frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = frontend::ProgramDistributionKind::Partitioned;
  binding.globalShape = {128};
  binding.localShape = {8};
  binding.dtype = "f32";
  for (int64_t rank = 0; rank < 16; ++rank) {
    frontend::ProgramRankSlice slice;
    slice.logicalRank = rank;
    slice.replicaId = 0;
    slice.offsets = {rank * 8};
    slice.sizes = {8};
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

llvm::Expected<TargetLLVMModuleBundle>
buildElementwiseTargetBundle(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto grouped = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64>, policy = "all_available", shape = array<i64: 16>, topology = @default}
  func.func @main(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>) -> tensor<8xf32> {
    %out = tensor.empty() : tensor<8xf32>
    %group = wafer.group ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%out : tensor<8xf32>) {
    ^bb0(%lhs_arg: tensor<8xf32>, %rhs_arg: tensor<8xf32>,
         %out_arg: tensor<8xf32>):
      %sum = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%lhs_arg, %rhs_arg : tensor<8xf32>, tensor<8xf32>)
          outs(%out_arg : tensor<8xf32>) {
        ^bb0(%a: f32, %b: f32, %old: f32):
          %value = arith.addf %a, %b : f32
          linalg.yield %value : f32
        } -> tensor<8xf32>
      wafer.group.yield %sum : tensor<8xf32>
    } : tensor<8xf32>
    return %group : tensor<8xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!grouped)
    return llvm::createStringError("failed to parse SystemC model module");

  frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0), boundary(1)};
  program.distributedOutputs = {boundary(0)};
  llvm::Expected<ExecutionConfig> config = ExecutionConfig::createForSingleCard(
      16, TargetProfileId::waferTx81SingleCardKernelV1());
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  llvm::Expected<ExecutableBundle> executable =
      wafer::compiler::detail::buildExecutableBundle(
          context, *grouped, std::move(program), *config, diagnostics,
          std::nullopt);
  if (!executable)
    return executable.takeError();
  grouped = nullptr;
  return compileExecutableBundleToTargetLLVMModules(*executable, diagnostics);
}

std::vector<RawLogicalValue> makeF32Values(llvm::ArrayRef<uint64_t> bits) {
  std::vector<RawLogicalValue> values;
  for (uint64_t value : bits)
    values.push_back({LogicalFormat::F32, value});
  return values;
}

uint64_t f32Bits(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<RawLogicalValue> makeSequentialF32Values(int64_t logicalRank,
                                                     float addend) {
  std::vector<RawLogicalValue> values;
  values.reserve(8);
  for (int64_t index = 0; index < 8; ++index) {
    const float value =
        static_cast<float>(logicalRank * 8 + index + 1) + addend;
    values.push_back({LogicalFormat::F32, f32Bits(value)});
  }
  return values;
}

TEST(SystemCTargetModelIntegrationTest,
     ExecutesSourceProducedNumericTransactionsAcrossDeltaCycles) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModuleBundle> bundle =
      buildElementwiseTargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  ASSERT_EQ(bundle->getModules().size(), 16u);

  std::vector<TargetCallRankArguments> arguments;
  std::vector<TargetModelInputBinding> inputs;
  std::vector<std::vector<RawLogicalValue>> expectedByRank;
  NumericTensorKey tensorKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F32, NumericTensorLayout::Tensor, {8}));
  for (const TargetLLVMModule &module : bundle->getModules()) {
    const int64_t rank = module.getLogicalRank();
    arguments.push_back({rank, {}});
    std::vector<std::vector<RawLogicalValue>> logicalInputs{
        makeSequentialF32Values(rank, 0.0f),
        makeF32Values({UINT64_C(0x3f800000), UINT64_C(0x3f800000),
                       UINT64_C(0x3f800000), UINT64_C(0x3f800000),
                       UINT64_C(0x3f800000), UINT64_C(0x3f800000),
                       UINT64_C(0x3f800000), UINT64_C(0x3f800000)}),
    };
    if (rank == 0)
      logicalInputs[1][0].bits = UINT64_C(0x33800000); // 2^-24, RNE tie.
    size_t inputIndex = 0;
    for (const KernelABISlot &slot : module.getKernelABISlots()) {
      const uint64_t base =
          UINT64_C(0x100000) +
          static_cast<uint64_t>(rank) * UINT64_C(0x100000) +
          static_cast<uint64_t>(slot.ordinal) * UINT64_C(0x10000);
      arguments.back().slots.push_back(base);
      if (slot.role == KernelABISlotRole::UserInput) {
        ASSERT_LT(inputIndex, logicalInputs.size());
        std::vector<uint8_t> bytes =
            llvm::cantFail(packPhysicalTensorLogicalValues(
                tensorKey, logicalInputs[inputIndex++], UINT8_C(0)));
        ASSERT_EQ(bytes.size(), static_cast<uint64_t>(slot.byteSize));
        inputs.push_back({rank, slot.ordinal, std::move(bytes)});
      }
    }
    ASSERT_EQ(inputIndex, logicalInputs.size());
    expectedByRank.push_back(makeSequentialF32Values(rank, 1.0f));
    if (rank == 0)
      expectedByRank.back()[0].bits = UINT64_C(0x3f800000);
  }

  llvm::Expected<TargetCallExecutable> frontend =
      prepareTargetCallFrontend(*bundle, arguments);
  ASSERT_TRUE(static_cast<bool>(frontend))
      << llvm::toString(frontend.takeError());
  llvm::Expected<TargetModelResult> result = executeSystemCTargetModel(
      std::move(*frontend), inputs,
      TargetModelKernelBudget::create(FormalNumericWorkBudget::create(
                                          /*maximumScalarEvaluations=*/1024,
                                          /*maximumFusedMultiplyAdds=*/1024),
                                      /*maximumMovementBytes=*/4096,
                                      /*maximumMovementSegments=*/256));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->completedRankCount, 16);
  EXPECT_GE(result->issuedTransactionCount, 16u * 4u);
  EXPECT_GE(result->systemCThreadProcessCount, 17u);
  EXPECT_GT(result->finalDeltaCount, 0u);
  EXPECT_FALSE(result->systemCVersion.empty());
  EXPECT_EQ(result->schedulerIdentity, "untimed-delta-single-issue-domain-v1");
  EXPECT_TRUE(result->numericFlags.inexact);
  EXPECT_FALSE(result->numericFlags.invalid);
  EXPECT_FALSE(result->numericFlags.divByZero);
  EXPECT_FALSE(result->numericFlags.overflow);
  EXPECT_FALSE(result->numericFlags.underflow);
  ASSERT_EQ(result->outputs.size(), 16u);

  for (const TargetModelOutput &modelOutput : result->outputs) {
    ASSERT_GE(modelOutput.logicalRank, 0);
    ASSERT_LT(modelOutput.logicalRank, 16);
    llvm::Expected<std::vector<RawLogicalValue>> output =
        unpackPhysicalTensorLogicalValues(tensorKey, modelOutput.bytes);
    ASSERT_TRUE(static_cast<bool>(output))
        << llvm::toString(output.takeError());
    const std::vector<RawLogicalValue> &expected =
        expectedByRank[static_cast<size_t>(modelOutput.logicalRank)];
    ASSERT_EQ(output->size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index)
      EXPECT_EQ((*output)[index].bits, expected[index].bits);
  }
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
