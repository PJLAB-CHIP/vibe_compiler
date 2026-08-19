//===- SystemCTargetModelIntegrationTest.cpp - SystemC model gate -------===//

#include "Wafer/Model/SystemC/SystemCTargetModel.h"

#include "Wafer/InitWaferDialects.h"
#include "Wafer/Target/Layout/PhysicalTensorCodec.h"

#include "Wafer/CodeGen/Executable/CardExecutableInternal.h"
#include "Wafer/Driver/CompilationInternal.h"

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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
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
  binding.distribution = frontend::ProgramDistributionKind::Replicated;
  binding.globalShape = {8};
  binding.localShape = {8};
  binding.dtype = "f32";
  frontend::ProgramPartitionSlice slice;
  slice.partitionId = 0;
  slice.replicaId = 0;
  slice.offsets = {0};
  slice.sizes = {8};
  slice.strides = {1};
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

struct CompiledElementwiseProgram {
  CardExecutable executable;
  TargetLLVMModules target;
};

llvm::Expected<CompiledElementwiseProgram>
compileElementwiseTargetModules(std::string &diagnosticText) {
  auto context = createCompilerContext();
  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>)
      -> tensor<8xf32> {
    %out = tensor.empty() : tensor<8xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%out : tensor<8xf32>) {
      ^bb0(%a: f32, %b: f32, %old: f32):
        %value = arith.addf %a, %b : f32
        linalg.yield %value : f32
    } -> tensor<8xf32>
    return %sum : tensor<8xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  if (!tensorProgram)
    return llvm::createStringError("failed to parse SystemC model module");

  frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0), boundary(1)};
  program.distributedOutputs = {boundary(0)};
  llvm::Expected<ExecutionConfig> config =
      ExecutionConfig::createForSingleCard(1);
  if (!config)
    return config.takeError();
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::ProgramDataHandoff programData;
  llvm::Expected<CardExecutable> executable =
      wafer::compiler::detail::buildCardExecutable(
          context, *tensorProgram, std::move(program), *config,
          OptimizationConfig::none(), diagnostics, std::nullopt, programData);
  if (!executable)
    return executable.takeError();
  tensorProgram = nullptr;
  llvm::Expected<TargetLLVMModules> target =
      compileCardExecutableToTargetLLVMModules(*executable, diagnostics);
  if (!target)
    return target.takeError();
  return CompiledElementwiseProgram{std::move(*executable), std::move(*target)};
}

uint64_t f32Bits(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<RawLogicalValue> makeSequentialF32Values(int64_t globalOffset,
                                                     int64_t elementCount,
                                                     float addend) {
  std::vector<RawLogicalValue> values;
  values.reserve(static_cast<size_t>(elementCount));
  for (int64_t index = 0; index < elementCount; ++index) {
    const float value = static_cast<float>(globalOffset + index + 1) + addend;
    values.push_back({LogicalFormat::F32, f32Bits(value)});
  }
  return values;
}

const ProgramResourceBinding *findProgramBinding(const TileExecutable &tile,
                                                 const TileEntryArgument &slot) {
  const auto matchesRole = [&](const ProgramResourceBinding &binding) {
    switch (slot.kind) {
    case TileEntryArgumentKind::ExternalInput:
      return binding.role == ProgramResourceRole::UserInput;
    case TileEntryArgumentKind::ExternalOutput:
      return binding.role == ProgramResourceRole::Output;
    case TileEntryArgumentKind::TargetTensor:
      return binding.role == ProgramResourceRole::Parameter ||
             binding.role == ProgramResourceRole::Constant;
    case TileEntryArgumentKind::Workspace:
    case TileEntryArgumentKind::ProfileRecord:
    case TileEntryArgumentKind::TransportStatus:
      return false;
    }
    llvm_unreachable("unknown tile entry argument kind");
  };
  auto match = llvm::find_if(
      tile.getProgramBindings(), [&](const ProgramResourceBinding &binding) {
        return matchesRole(binding) && binding.index == slot.resourceIndex;
      });
  return match == tile.getProgramBindings().end() ? nullptr : &*match;
}

TEST(SystemCTargetModelIntegrationTest,
     ExecutesSourceProducedNumericCommandsAcrossDeltaCycles) {
  std::string diagnostics;
  llvm::Expected<CompiledElementwiseProgram> compiled =
      compileElementwiseTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << diagnostics << llvm::toString(compiled.takeError());
  const auto &tiles = compiled->executable.getTileExecutables();
  const auto &modules = compiled->target.getModules();
  ASSERT_EQ(tiles.size(), 16u);
  ASSERT_EQ(modules.size(), 16u);

  std::vector<TargetCallTileArguments> arguments;
  std::vector<TargetModelInputBinding> inputs;
  NumericTensorKey tensorKey = llvm::cantFail(NumericTensorKey::create(
      LogicalFormat::F32, PhysicalTensorLayout::Tensor, {8}));
  const std::vector<RawLogicalValue> lhs =
      makeSequentialF32Values(/*globalOffset=*/0, /*elementCount=*/8, 0.0f);
  std::vector<RawLogicalValue> rhs(8,
                                   {LogicalFormat::F32, UINT64_C(0x3f800000)});
  rhs.front().bits = UINT64_C(0x33800000); // 2^-24, RNE tie.
  for (auto [tileIndex, module] : llvm::enumerate(modules)) {
    const TileExecutable &tile = tiles[tileIndex];
    ASSERT_EQ(module.getCardId(), tile.getCardId());
    ASSERT_EQ(module.getTileId(), tile.getTileId());
    ASSERT_EQ(module.getLaunchSlotId(), tile.getLaunchSlotId());
    const int64_t launchSlot = module.getLaunchSlotId().getValue();
    arguments.push_back(
        {module.getCardId(), module.getTileId(), module.getLaunchSlotId(), {}});
    size_t userInputCount = 0;
    size_t outputCount = 0;
    for (const TileEntryArgument &slot : module.getTileEntryArguments()) {
      const bool tileOwned = slot.kind == TileEntryArgumentKind::Workspace ||
                             slot.kind == TileEntryArgumentKind::TransportStatus;
      const uint64_t base =
          tileOwned
              ? UINT64_C(0x10000000) +
                    static_cast<uint64_t>(launchSlot) * UINT64_C(0x100000) +
                    static_cast<uint64_t>(slot.ordinal) * UINT64_C(0x10000)
              : (slot.kind == TileEntryArgumentKind::ExternalOutput
                     ? UINT64_C(0x400000)
                     : UINT64_C(0x100000) +
                           static_cast<uint64_t>(slot.resourceIndex) *
                               UINT64_C(0x100000));
      arguments.back().slots.push_back(base);
      const ProgramResourceBinding *binding = findProgramBinding(tile, slot);
      if (!binding)
        continue;
      ASSERT_EQ(slot.shape, binding->localShape);
      ASSERT_EQ(binding->slice.sizes, binding->localShape);
      ASSERT_EQ(binding->slice.offsets.size(), 1u);
      EXPECT_EQ(binding->slice.offsets.front(), 0);
      EXPECT_EQ(slot.shape, (std::vector<int64_t>{8}));
      if (slot.kind == TileEntryArgumentKind::ExternalInput) {
        ASSERT_LT(slot.resourceIndex, 2);
        const llvm::ArrayRef<RawLogicalValue> values =
            slot.resourceIndex == 0 ? llvm::ArrayRef<RawLogicalValue>(lhs)
                                    : llvm::ArrayRef<RawLogicalValue>(rhs);
        std::vector<uint8_t> bytes = llvm::cantFail(
            packPhysicalTensorLogicalValues(tensorKey, values, UINT8_C(0)));
        ASSERT_EQ(bytes.size(), static_cast<uint64_t>(slot.byteSize));
        if (launchSlot == 0)
          inputs.push_back(
              {getTargetModelResourceId(module.getCardId(), module.getTileId(),
                                        slot.kind, slot.resourceIndex),
               std::move(bytes)});
        ++userInputCount;
      } else if (slot.kind == TileEntryArgumentKind::ExternalOutput) {
        ++outputCount;
      }
    }
    EXPECT_EQ(userInputCount, 2u);
    EXPECT_EQ(outputCount, 1u);
  }

  llvm::Expected<TargetCallExecutable> frontend =
      createTargetCallExecutable(compiled->target, arguments);
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
  EXPECT_EQ(result->completedTileCount, 16);
  EXPECT_GE(result->issuedCommandCount, 8u * 4u);
  EXPECT_GE(result->systemCThreadProcessCount, 17u);
  EXPECT_GT(result->finalDeltaCount, 0u);
  EXPECT_FALSE(result->systemCVersion.empty());
  EXPECT_EQ(result->schedulerIdentity, "untimed-delta-worker-aware-ncc");
  EXPECT_TRUE(result->numericFlags.inexact);
  EXPECT_FALSE(result->numericFlags.invalid);
  EXPECT_FALSE(result->numericFlags.divByZero);
  EXPECT_FALSE(result->numericFlags.overflow);
  EXPECT_FALSE(result->numericFlags.underflow);
  ASSERT_EQ(result->outputs.size(), 1u);

  const std::vector<RawLogicalValue> expected = {
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, f32Bits(3.0f)},
      {LogicalFormat::F32, f32Bits(4.0f)},
      {LogicalFormat::F32, f32Bits(5.0f)},
      {LogicalFormat::F32, f32Bits(6.0f)},
      {LogicalFormat::F32, f32Bits(7.0f)},
      {LogicalFormat::F32, f32Bits(8.0f)},
      {LogicalFormat::F32, f32Bits(9.0f)},
  };
  for (const TargetModelOutput &modelOutput : result->outputs) {
    auto module =
        llvm::find_if(modules, [&](const TargetLLVMModule &candidate) {
          return candidate.getCardId() == modelOutput.cardId &&
                 candidate.getTileId() == modelOutput.tileId &&
                 candidate.getLaunchSlotId() == modelOutput.launchSlotId;
        });
    ASSERT_NE(module, modules.end());
    auto outputSlot = llvm::find_if(
        module->getTileEntryArguments(), [&](const TileEntryArgument &slot) {
          return slot.ordinal == modelOutput.slotOrdinal &&
                 slot.kind == TileEntryArgumentKind::ExternalOutput;
        });
    ASSERT_NE(outputSlot, module->getTileEntryArguments().end());
    llvm::Expected<std::vector<RawLogicalValue>> output =
        unpackPhysicalTensorLogicalValues(tensorKey, modelOutput.bytes);
    ASSERT_TRUE(static_cast<bool>(output))
        << llvm::toString(output.takeError());
    ASSERT_EQ(output->size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
      EXPECT_EQ((*output)[index].bits, expected[index].bits);
    }
  }
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
