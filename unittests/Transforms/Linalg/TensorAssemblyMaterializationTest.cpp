//===- TensorAssemblyMaterializationTest.cpp ----------------------------===//

#include "Linalg/TensorAssemblyMaterialization.h"
#include "TestSupport/Transforms/CurrentTensorPipeline.h"
#include "TestSupport/Transforms/TensorSubsetInterpreter.h"
#include "Wafer/Analysis/Instr/ScheduleCostAnalysis.h"
#include "Wafer/CodeGen/DeviceExecutableInternal.h"
#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"
#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Instr/MemoryPlanning.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <limits>
#include <numeric>
#include <vector>

namespace {
using namespace wafer::compiler::detail;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

using TensorInterpreter = wafer::test::TensorSubsetInterpreter;

std::string periodicInput(int64_t extent) {
  std::string text = R"mlir(
    func.func private @observe(tensor<1x32x4xf16>, index)
    func.func private @tail(tensor<1x17x4xf16>)
    func.func private @full(tensor<1xEXTENTx8xf16>)
    func.func @test(%a: tensor<1x512x8xf16>, %b: tensor<1xTAILx8xf16>) {
      %empty = tensor.empty() : tensor<1xEXTENTx8xf16>
      %left = tensor.insert_slice %a into %empty[0, 0, 0] [1, 512, 8] [1, 1, 1]
        : tensor<1x512x8xf16> into tensor<1xEXTENTx8xf16>
      %assembled = tensor.insert_slice %b into %left[0, 512, 0] [1, TAIL, 8] [1, 1, 1]
        : tensor<1xTAILx8xf16> into tensor<1xEXTENTx8xf16>
      %flat = tensor.collapse_shape %assembled [[0, 1, 2]]
        : tensor<1xEXTENTx8xf16> into tensor<FLATxf16>
      %view = tensor.expand_shape %flat [[0, 1, 2]] output_shape [1, ROWS, 4]
        : tensor<FLATxf16> into tensor<1xROWSx4xf16>
      %c0 = arith.constant 0 : index
      %step = arith.constant 31 : index
      %end = arith.constant END : index
      scf.for %i = %c0 to %end step %step {
        %window = tensor.extract_slice %view[0, %i, 0] [1, 32, 4] [1, 1, 1]
          : tensor<1xROWSx4xf16> to tensor<1x32x4xf16>
        func.call @observe(%window, %i) : (tensor<1x32x4xf16>, index) -> ()
      }
      %last = tensor.extract_slice %view[0, LAST, 0] [1, 17, 4] [1, 1, 1]
        : tensor<1xROWSx4xf16> to tensor<1x17x4xf16>
      func.call @tail(%last) : (tensor<1x17x4xf16>) -> ()
      func.call @full(%assembled) : (tensor<1xEXTENTx8xf16>) -> ()
      return
    }
  )mlir";
  for (auto [key, number] :
       {std::pair{"EXTENT", extent}, std::pair{"TAIL", extent - 512},
        std::pair{"FLAT", extent * 8}, std::pair{"ROWS", extent * 2},
        std::pair{"END", extent * 2 - 31},
        std::pair{"LAST", extent * 2 - 17}}) {
    size_t at = 0;
    while ((at = text.find(key, at)) != std::string::npos) {
      auto replacement = std::to_string(number);
      text.replace(at, std::string(key).size(), replacement);
      at += replacement.size();
    }
  }
  return text;
}

TEST(TensorAssemblyMaterializationTest,
     PeriodicMainAndOddTailPreserveEveryElement) {
  for (int64_t extent : {1024, 1025, 1031, 4097}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(periodicInput(extent),
                                                          context.get());
    ASSERT_TRUE(module);
    auto function = module->lookupSymbol<mlir::func::FuncOp>("test");
    TensorInterpreter original;
    original.run(function);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    llvm::SmallVector<mlir::tensor::ExtractSliceOp> reads;
    function.walk(
        [&](mlir::tensor::ExtractSliceOp read) { reads.push_back(read); });
    ASSERT_EQ(reads.size(), 2u);
    mlir::IRRewriter rewriter(context.get());
    TensorInterpreter generated;
    uint64_t templates = 0;
    for (auto read : reads) {
      auto result = materializeTensorSubsetRead(rewriter, read);
      ASSERT_EQ(result.status, TensorSubsetMaterializationStatus::Exact)
          << result.detail << "; work=" << result.work;
      templates += result.blockTemplates;
      for (auto sourceRead : result.materialized->sourceReads) {
        EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(sourceRead.getSource()));
        generated.sourceReads.insert(sourceRead);
      }
      rewriter.replaceOp(read, result.materialized->value);
    }
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    // One original compute/observer loop; all added loops are two-way block
    // subdivision with a compact destination argument, never compute clones.
    unsigned outerLoops = 0, observers = 0;
    function.walk([&](mlir::scf::ForOp loop) {
      if (loop.getNumRegionIterArgs() == 0) {
        ++outerLoops;
        EXPECT_EQ(mlir::getConstantIntValue(loop.getStep()), 31);
      } else {
        EXPECT_EQ(mlir::getConstantIntValue(loop.getLowerBound()), 0);
        EXPECT_EQ(mlir::getConstantIntValue(loop.getUpperBound()), 2);
        EXPECT_EQ(mlir::getConstantIntValue(loop.getStep()), 1);
      }
    });
    function.walk([&](mlir::func::CallOp) { ++observers; });
    EXPECT_EQ(outerLoops, 1u);
    EXPECT_EQ(observers, 3u);
    EXPECT_LE(templates, 80u);
    generated.run(function);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(generated.observations, original.observations);
    EXPECT_GT(generated.dynamicCopies, 0u);
    EXPECT_LT(generated.dynamicCopies, (extent * 2 / 31 + 1) * 32u);
  }
}

TEST(TensorAssemblyMaterializationTest,
     PreflightBudgetLeavesSelectedIRUntouched) {
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(periodicInput(1031),
                                                        context.get());
  ASSERT_TRUE(module);
  mlir::tensor::ExtractSliceOp selected;
  module->walk([&](mlir::tensor::ExtractSliceOp read) {
    if (!selected)
      selected = read;
  });
  auto print = [&] {
    std::string text;
    llvm::raw_string_ostream stream(text);
    module->print(stream);
    return text;
  };
  std::string before = print();
  wafer::analysis::IndexRelationLimits limits;
  limits.maxConstraintWork = 10;
  mlir::IRRewriter rewriter(context.get());
  auto result = materializeTensorSubsetRead(rewriter, selected, limits);
  EXPECT_EQ(result.status,
            TensorSubsetMaterializationStatus::ResourceExhausted);
  EXPECT_EQ(print(), before);
}

TEST(TensorAssemblyMaterializationTest,
     SubsetWindowsReachActualMemoryCostAndLLVM) {
  using namespace wafer;
  using namespace wafer::compiler;
  for (unsigned variant : {0, 1, 2})
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(::testing::Message() << extent << "/" << variant);
      const bool periodic = variant == 0;
      const int64_t sourceChannels = periodic ? 8 : 64;
      const int64_t resultChannels = periodic ? 4 : 64;
      const int64_t rows = extent * sourceChannels / resultChannels;
      auto context = createContext();
      std::string text = R"mlir(
      module { wafer.tile.module card_id = 0 tile_id = 0 {
        func.func @entry(%input: tensor<1xEXTENTxSOURCECxf16> {wafer.program_argument = #wafer.program_argument<0>}) -> tensor<1xROWSxRESULTCxf16> {
          %result = wafer.tile.region(%input : tensor<1xEXTENTxSOURCECxf16>)
              -> (tensor<1xROWSxRESULTCxf16>) {
          ^bb0(%arg: tensor<1xEXTENTxSOURCECxf16>):
            %a = tensor.extract_slice %arg[0, TAIL, 0] [1, 512, SOURCEC] [1, 1, 1]
              : tensor<1xEXTENTxSOURCECxf16> to tensor<1x512xSOURCECxf16>
            %b = tensor.extract_slice %arg[0, 0, 0] [1, TAIL, SOURCEC] [1, 1, 1]
              : tensor<1xEXTENTxSOURCECxf16> to tensor<1xTAILxSOURCECxf16>
            %empty = tensor.empty() : tensor<1xEXTENTxSOURCECxf16>
            %left = tensor.insert_slice %a into %empty[0, 0, 0] [1, 512, SOURCEC] [1, 1, 1]
              : tensor<1x512xSOURCECxf16> into tensor<1xEXTENTxSOURCECxf16>
            %assembled = tensor.insert_slice %b into %left[0, 512, 0] [1, TAIL, SOURCEC] [1, 1, 1]
              : tensor<1xTAILxSOURCECxf16> into tensor<1xEXTENTxSOURCECxf16>
            %flat = tensor.collapse_shape %assembled [[0, 1, 2]]
              : tensor<1xEXTENTxSOURCECxf16> into tensor<FLATxf16>
            %view = tensor.expand_shape %flat [[0, 1, 2]] output_shape [1, ROWS, RESULTC]
              : tensor<FLATxf16> into tensor<1xROWSxRESULTCxf16>
            %c0 = arith.constant 0 : index
            %step = arith.constant 31 : index
            %end = arith.constant END : index
            %output = tensor.empty() : tensor<1xROWSxRESULTCxf16>
            %main = scf.for %i = %c0 to %end step %step
                iter_args(%current = %output) -> tensor<1xROWSxRESULTCxf16> {
              %window = tensor.extract_slice %view[0, %i, 0] [1, 32, RESULTC] [1, 1, 1]
                : tensor<1xROWSxRESULTCxf16> to tensor<1x32xRESULTCxf16>
              %next = tensor.insert_slice %window into %current[0, %i, 0] [1, 32, RESULTC] [1, 1, 1]
                : tensor<1x32xRESULTCxf16> into tensor<1xROWSxRESULTCxf16>
              scf.yield %next : tensor<1xROWSxRESULTCxf16>
            }
            %last = tensor.extract_slice %view[0, LAST, 0] [1, 17, RESULTC] [1, 1, 1]
              : tensor<1xROWSxRESULTCxf16> to tensor<1x17xRESULTCxf16>
            %value = tensor.insert_slice %last into %main[0, LAST, 0] [1, 17, RESULTC] [1, 1, 1]
              : tensor<1x17xRESULTCxf16> into tensor<1xROWSxRESULTCxf16>
            wafer.tile.yield %value : tensor<1xROWSxRESULTCxf16>
          }
          return %result : tensor<1xROWSxRESULTCxf16>
        }
      } }
    )mlir";
      for (auto [key, number] :
           {std::pair{"EXTENT", extent}, std::pair{"TAIL", extent - 512},
            std::pair{"FLAT", extent * sourceChannels}, std::pair{"ROWS", rows},
            std::pair{"END", rows - 31}, std::pair{"LAST", rows - 17},
            std::pair{"SOURCEC", sourceChannels},
            std::pair{"RESULTC", resultChannels}}) {
        size_t at = 0;
        while ((at = text.find(key, at)) != std::string::npos) {
          auto replacement = std::to_string(number);
          text.replace(at, std::string(key).size(), replacement);
          at += replacement.size();
        }
      }
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
      ASSERT_TRUE(module);
      TileRegionOp region;
      llvm::SmallVector<mlir::tensor::ExtractSliceOp> reads;
      module->walk([&](TileRegionOp op) { region = op; });
      region.walk([&](mlir::tensor::ExtractSliceOp read) {
        if (read.getSource().getDefiningOp<mlir::tensor::ExpandShapeOp>())
          reads.push_back(read);
      });
      ASSERT_EQ(reads.size(), 2u);
      StructuredMaterializationRelations relations;
      relations.structuralOutputs.push_back({0, region.getResult(0)});
      mlir::IRRewriter rewriter(context.get());
      for (auto read : reads) {
        auto result = materializeTensorSubsetRead(rewriter, read);
        ASSERT_EQ(result.status, TensorSubsetMaterializationStatus::Exact)
            << result.detail << "; work=" << result.work;
        rewriter.replaceOp(read, result.materialized->value);
      }
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      ExternalBufferLayout external;
      if (variant == 2) {
        external.layout = PhysicalTensorLayout::NCx;
        external.inputArguments.push_back(0);
      }
      auto layout = wafer::test::prepareTensorsAndBufferize(*module, relations,
                                                            100000, external);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      ASSERT_EQ(standalone->size(), 1u);
      auto &tile = standalone->front();
      TileRegionToInstrLoweringSession session(*context);
      llvm::SmallVector<TileRegionOp> regions;
      tile.module->walk([&](TileRegionOp op) { regions.push_back(op); });
      for (auto current : regions)
        ASSERT_TRUE(
            mlir::succeeded(convertTileRegionToInstr(current, session)));
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*tile.module, session)));
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
      TileMemoryPlanningFailure failure;
      auto planned = planTileMemory(std::move(tile.module), &failure);
      ASSERT_TRUE(mlir::succeeded(planned));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(**planned)));
      auto cost = analysis::analyzeInstructionProgramCost(
          (*planned)->getOperation(), getTargetMemoryPolicy());
      EXPECT_TRUE(cost.spmHighWaterBytes.isKnown());
      EXPECT_GT(cost.spmHighWaterBytes.value, 0u);
      EXPECT_TRUE(cost.work.instructions.upperBound.isKnown());
      EXPECT_GT(cost.work.instructions.upperBound.value, 0u);
      ASSERT_TRUE(cost.work.steadyStateNCCJoins.upperBound.isKnown());
      EXPECT_EQ(cost.work.steadyStateNCCJoins.upperBound.value, 0u);
      ASSERT_TRUE(cost.work.nonTerminalNCCJoins.upperBound.isKnown());
      EXPECT_EQ(cost.work.nonTerminalNCCJoins.upperBound.value, 0u);
      ASSERT_TRUE(cost.work.cpuScalarOperations.upperBound.isKnown());
      EXPECT_GT(cost.work.cpuScalarOperations.upperBound.value, 0u);
      unsigned allocations = 0, encodedInputs = 0;
      (*planned)->walk([&](mlir::memref::AllocOp allocation) {
        if (!isWaferSPMMemRefType(allocation.getType()))
          return;
        ++allocations;
        EXPECT_TRUE(
            allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName));
      });
      (*planned)->walk([&](InstrRDMAOp load) {
        auto type = mlir::cast<mlir::MemRefType>(load.getSource().getType());
        encodedInputs += getWaferMemoryAttr(type).getLayout() == MemLayout::NCx;
      });
      EXPECT_GT(allocations, 0u);
      if (variant == 2) {
        EXPECT_GT(encodedInputs, 0u);
      }
      auto memory = getTargetMemoryPolicy();
      ASSERT_TRUE(mlir::succeeded(planDDRMemoryModule(
          **planned, memory.ddrAlignmentBytes, memory.ddrCapacityBytes,
          memory.ddrLargestContiguousBytes, memory.ddrBandwidthLimitBytes)));
      std::vector<ProgramResourceBinding> bindings;
      for (auto role :
           {ProgramResourceRole::UserInput, ProgramResourceRole::Output}) {
        ProgramResourceBinding binding{};
        binding.role = role;
        binding.programTensorId = {role, 0};
        binding.index = binding.programIndex = 0;
        binding.dtype = ProgramElementType::F16;
        binding.localShape =
            role == ProgramResourceRole::UserInput
                ? std::vector<int64_t>{1, extent, sourceChannels}
                : std::vector<int64_t>{1, rows, resultChannels};
        binding.globalShape = binding.localShape;
        bindings.push_back(std::move(binding));
      }
      auto executable = DeviceExecutableBuilder::makeTileExecutable(
          CardId(0), TileId(0), LaunchSlotId(0), std::move(*planned), "entry",
          std::move(bindings), TransportContract::None);
      auto config = ExecutionConfig::createForSingleCard(1);
      ASSERT_TRUE(bool(config));
      auto prepared = prepareTargetABI(executable, *config, false);
      ASSERT_TRUE(mlir::succeeded(prepared));
      ASSERT_TRUE(mlir::succeeded(lowerToTargetLLVM(*prepared)));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*prepared->module)));
      unsigned calls = 0;
      prepared->module->walk([&](mlir::LLVM::CallOp) { ++calls; });
      EXPECT_GT(calls, 0u);
    }
}

} // namespace
