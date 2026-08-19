//===- ScheduleCostAnalysisTest.cpp --------------------------------------===//

#include "Wafer/Analysis/ScheduleCost/ScheduleCostAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace {

using wafer::analysis::InstructionProgramCost;
using wafer::analysis::NoCDirection;
using wafer::analysis::ScheduleCostKnowledge;
using wafer::analysis::ScheduleCostReason;

class ScheduleCostAnalysisTest : public ::testing::Test {
protected:
  ScheduleCostAnalysisTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
    wafer::registerWaferCoreDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  parseWithoutVerification(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get(), /*verifyAfterParse=*/false));
  }

  InstructionProgramCost analyze(mlir::ModuleOp module) {
    return wafer::analysis::analyzeInstructionProgramCost(
        module, wafer::getTargetMemoryPolicy());
  }

  wafer::analysis::CardInstructionProgramCost
  analyzeCardCost(llvm::ArrayRef<mlir::Operation *> roots,
                  const wafer::TargetMemoryPolicy &policy) {
    llvm::SmallVector<wafer::analysis::TileInstructionProgram, 16> programs;
    programs.reserve(roots.size());
    for (auto [tile, root] : llvm::enumerate(roots))
      programs.push_back({wafer::TileId(static_cast<int64_t>(tile)), root});
    return wafer::analysis::analyzeCardInstructionProgramCost(programs, policy);
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeDTETileModule(llvm::StringRef functionArguments,
                    llvm::StringRef instructionBody,
                    bool includeTopology = true) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << "module {\n";
    if (includeTopology) {
      os << "  wafer.target.topology @topology {card_grid = array<i64: 1, 1>, "
            "card_interconnect = \"mesh\", tile_grid = array<i64: 4, 4>, "
            "unavailable_tiles = array<i64>}\n";
    }
    os << "  func.func @main(" << functionArguments << ") {\n"
       << "    %buffer = memref.alloc() "
          "{wafer.spm.offset = #wafer.spm_offset<65536>} "
          ": memref<16xi8, #wafer.memory<spm, tensor>>\n"
       << instructionBody << "\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return includeTopology ? parse(os.str())
                           : parseWithoutVerification(os.str());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(ScheduleCostAnalysisTest, UsesTargetMemoryBoundsForExactHighWater) {
  auto policy = wafer::getTargetMemoryPolicy();
  EXPECT_EQ(policy.spmBase, 65536);
  EXPECT_EQ(policy.spmLimit, 3080192);
}

TEST_F(ScheduleCostAnalysisTest,
       OperationSlicesRetainControlMultiplicityAndConserveRawWork) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %source = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c3 step %c1 {
      wafer.instr.rdma %input to %source
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
          %source, %source into %dest
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::Operation *, 2> movementOperations;
  llvm::SmallVector<mlir::Operation *, 2> computeOperations;
  module->walk([&](mlir::Operation *operation) {
    if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation)) {
      if (allocation.getType().getShape() == llvm::ArrayRef<int64_t>{4}) {
        if (movementOperations.empty())
          movementOperations.push_back(operation);
        else
          computeOperations.push_back(operation);
      }
    } else if (mlir::isa<wafer::InstrRDMAOp>(operation)) {
      movementOperations.push_back(operation);
    } else if (mlir::isa<wafer::InstrElementwiseOp>(operation)) {
      computeOperations.push_back(operation);
    }
  });
  ASSERT_EQ(movementOperations.size(), 2u);
  ASSERT_EQ(computeOperations.size(), 2u);

  llvm::SmallVector<wafer::analysis::TileInstructionProgram, 16> programs;
  llvm::SmallVector<wafer::analysis::TileInstructionProgramSlice, 16>
      movementSlices;
  llvm::SmallVector<wafer::analysis::TileInstructionProgramSlice, 16>
      computeSlices;
  for (int64_t tile = 0; tile < 16; ++tile) {
    programs.push_back({wafer::TileId(tile), module->getOperation()});
    movementSlices.push_back(
        {wafer::TileId(tile), module->getOperation(), movementOperations});
    computeSlices.push_back(
        {wafer::TileId(tile), module->getOperation(), computeOperations});
  }
  const auto policy = wafer::getTargetMemoryPolicy();
  auto whole =
      wafer::analysis::analyzeCardInstructionProgramCost(programs, policy);
  auto movement = wafer::analysis::analyzeCardInstructionProgramCostSlice(
      movementSlices, policy);
  auto compute = wafer::analysis::analyzeCardInstructionProgramCostSlice(
      computeSlices, policy);

  ASSERT_TRUE(movement.aggregateDDRReadBytes.isKnown());
  EXPECT_EQ(movement.aggregateDDRReadBytes.value, 16u * 3u * 8u);
  EXPECT_EQ(compute.aggregateDDRReadBytes.value, 0u);
  ASSERT_TRUE(compute.aggregateCompute.vectorF16Bf16LogicalOps.isKnown());
  EXPECT_EQ(compute.aggregateCompute.vectorF16Bf16LogicalOps.value,
            16u * 3u * 4u);
  EXPECT_EQ(movement.aggregateInstructionCount.value +
                compute.aggregateInstructionCount.value,
            whole.aggregateInstructionCount.value);
  EXPECT_EQ(movement.aggregateCompilerOwnedSPMBufferCount.value +
                compute.aggregateCompilerOwnedSPMBufferCount.value,
            whole.aggregateCompilerOwnedSPMBufferCount.value);
  EXPECT_EQ(std::max(movement.maximumTileSPMHighWaterBytes.value,
                     compute.maximumTileSPMHighWaterBytes.value),
            whole.maximumTileSPMHighWaterBytes.value);
}

TEST_F(ScheduleCostAnalysisTest,
       MultipliesStaticNestedLoopsAndKeepsSPMAddressHighWater) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %output: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c3 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        wafer.instr.rdma %input to %a
            {byte_count = 8 : i64, inner_bytes = 8 : i64,
             src_strides = array<i64: 0, 0, 0>,
             src_iterations = array<i64: 1, 1, 1>}
            : memref<4xf16, #wafer.memory<ddr, tensor>>
           to memref<4xf16, #wafer.memory<spm, tensor>>
        wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
            %a, %a into %b
            : memref<4xf16, #wafer.memory<spm, tensor>>,
              memref<4xf16, #wafer.memory<spm, tensor>>
          into memref<4xf16, #wafer.memory<spm, tensor>>
        wafer.instr.wdma %b to %output
            {byte_count = 8 : i64, inner_bytes = 8 : i64,
             dst_strides = array<i64: 0, 0, 0>,
             dst_iterations = array<i64: 1, 1, 1>}
            : memref<4xf16, #wafer.memory<spm, tensor>>
           to memref<4xf16, #wafer.memory<ddr, tensor>>
        wafer.instr.ncc_join [0]
      }
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.instructionCount.isKnown());
  EXPECT_EQ(cost.instructionCount.value, 24u);
  EXPECT_EQ(cost.work.instructions.staticSites.value, 4u);
  EXPECT_EQ(cost.work.instructions.exactExecutions.value, 24u);
  EXPECT_EQ(cost.work.instructions.lowerBound.value, 24u);
  EXPECT_EQ(cost.work.instructions.upperBound.value, 24u);
  EXPECT_EQ(cost.work.rdmaIssues.staticSites.value, 1u);
  EXPECT_EQ(cost.work.rdmaIssues.exactExecutions.value, 6u);
  EXPECT_EQ(cost.work.wdmaIssues.staticSites.value, 1u);
  EXPECT_EQ(cost.work.wdmaIssues.exactExecutions.value, 6u);
  EXPECT_EQ(cost.work.ctIssues.staticSites.value, 1u);
  EXPECT_EQ(cost.work.ctIssues.exactExecutions.value, 6u);
  EXPECT_EQ(cost.work.nccJoins.staticSites.value, 1u);
  EXPECT_EQ(cost.work.nccJoins.exactExecutions.value, 6u);
  ASSERT_TRUE(cost.ddrReadBytes.isKnown());
  EXPECT_EQ(cost.ddrReadBytes.value, 48u);
  ASSERT_TRUE(cost.ddrWriteBytes.isKnown());
  EXPECT_EQ(cost.ddrWriteBytes.value, 48u);
  ASSERT_TRUE(cost.spmMovementBytes.isKnown());
  EXPECT_EQ(cost.spmMovementBytes.value, 96u);
  ASSERT_TRUE(cost.compute.vectorF16Bf16LogicalOps.isKnown());
  EXPECT_EQ(cost.compute.vectorF16Bf16LogicalOps.value, 24u);
  ASSERT_TRUE(cost.nccJoinCount.isKnown());
  EXPECT_EQ(cost.nccJoinCount.value, 6u);
  ASSERT_TRUE(cost.steadyStateNCCJoinCount.isKnown());
  EXPECT_EQ(cost.steadyStateNCCJoinCount.value, 6u);
  ASSERT_TRUE(cost.nonTerminalNCCJoinCount.isKnown());
  EXPECT_EQ(cost.nonTerminalNCCJoinCount.value, 6u);
  ASSERT_TRUE(cost.nccParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.nccParticipantWaitCount.value, 6u);
  ASSERT_TRUE(cost.steadyStateNCCParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.steadyStateNCCParticipantWaitCount.value, 6u);
  ASSERT_TRUE(cost.nonTerminalNCCParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.nonTerminalNCCParticipantWaitCount.value, 6u);
  ASSERT_TRUE(cost.intrinsicNCCDrainCount.isKnown());
  EXPECT_EQ(cost.intrinsicNCCDrainCount.value, 0u);
  ASSERT_TRUE(cost.spmHighWaterBytes.isKnown());
  EXPECT_EQ(cost.spmHighWaterBytes.value, 264u);
  EXPECT_EQ(cost.compilerOwnedSPMBufferCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest,
       ResolvesStaticLoopBoundsThroughNestedTileRegions) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %bounds:3 = wafer.tile.region(%c0, %c4, %c2 : index, index, index)
        -> (index, index, index) {
    ^bb0(%lower: index, %upper: index, %step: index):
      wafer.tile.yield %lower, %upper, %step : index, index, index
    }
    %result = wafer.tile.region(
        %input, %bounds#0, %bounds#1, %bounds#2
        : memref<4xf16, #wafer.memory<ddr, tensor>>, index, index, index)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%in: memref<4xf16, #wafer.memory<ddr, tensor>>,
         %lower: index, %upper: index, %step: index):
      %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
      scf.for %iteration = %lower to %upper step %step {
        wafer.instr.rdma %in to %spm
            {byte_count = 8 : i64, inner_bytes = 8 : i64,
             src_strides = array<i64: 0, 0, 0>,
             src_iterations = array<i64: 1, 1, 1>}
            : memref<4xf16, #wafer.memory<ddr, tensor>>
           to memref<4xf16, #wafer.memory<spm, tensor>>
      }
      wafer.tile.yield %in
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.instructionCount.isKnown());
  EXPECT_EQ(cost.instructionCount.value, 2u);
  ASSERT_TRUE(cost.work.rdmaIssues.exactExecutions.isKnown());
  EXPECT_EQ(cost.work.rdmaIssues.exactExecutions.value, 2u);
  ASSERT_TRUE(cost.ddrReadBytes.isKnown());
  EXPECT_EQ(cost.ddrReadBytes.value, 16u);
}

TEST_F(ScheduleCostAnalysisTest,
       EvaluatesStaticArithmeticBoundsIntroducedByLoopTransforms) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %pipeline_distance = arith.muli %c2, %c1 : index
    %steady_upper = arith.subi %c8, %pipeline_distance : index
    scf.for %i = %c0 to %steady_upper step %c1 {
      wafer.instr.rdma %input to %buffer
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();

  InstructionProgramCost cost = analyze(*module);

  ASSERT_TRUE(cost.instructionCount.isKnown());
  EXPECT_EQ(cost.instructionCount.value, 6u);
  ASSERT_TRUE(cost.ddrReadBytes.isKnown());
  EXPECT_EQ(cost.ddrReadBytes.value, 48u);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST_F(ScheduleCostAnalysisTest,
       CountsEveryTypedParticipantWaitWithoutScopeDiscount) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    "wafer.instr.ncc_join"() {participants = array<i64: 0, 2>} : () -> ()
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %buffer, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    "wafer.instr.ncc_join"() {participants = array<i64: 1>} : () -> ()
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.nccJoinCount.isKnown());
  EXPECT_EQ(cost.nccJoinCount.value, 2u);
  ASSERT_TRUE(cost.steadyStateNCCJoinCount.isKnown());
  EXPECT_EQ(cost.steadyStateNCCJoinCount.value, 0u);
  ASSERT_TRUE(cost.nonTerminalNCCJoinCount.isKnown());
  EXPECT_EQ(cost.nonTerminalNCCJoinCount.value, 1u);
  ASSERT_TRUE(cost.nccParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.nccParticipantWaitCount.value, 3u);
  ASSERT_TRUE(cost.steadyStateNCCParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.steadyStateNCCParticipantWaitCount.value, 0u);
  ASSERT_TRUE(cost.nonTerminalNCCParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.nonTerminalNCCParticipantWaitCount.value, 2u);
  ASSERT_TRUE(cost.intrinsicNCCDrainCount.isKnown());
  EXPECT_EQ(cost.intrinsicNCCDrainCount.value, 0u);
}

TEST_F(ScheduleCostAnalysisTest,
       CountsSynchronousPeripheralWritebackAsIntrinsicDrain) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %value = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<1xf16, #wafer.memory<spm, tensor>>
    %index = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<1xi32, #wafer.memory<spm, tensor>>
    wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax>
        %input into %value, %index {elem_count = 4 : i64}
        : memref<4xf16, #wafer.memory<spm, tensor>>
      into memref<1xf16, #wafer.memory<spm, tensor>>,
           memref<1xi32, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.nccJoinCount.isKnown());
  EXPECT_EQ(cost.nccJoinCount.value, 0u);
  ASSERT_TRUE(cost.nccParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.nccParticipantWaitCount.value, 1u);
  ASSERT_TRUE(cost.steadyStateNCCParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.steadyStateNCCParticipantWaitCount.value, 0u);
  ASSERT_TRUE(cost.nonTerminalNCCParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.nonTerminalNCCParticipantWaitCount.value, 0u);
  ASSERT_TRUE(cost.intrinsicNCCDrainCount.isKnown());
  EXPECT_EQ(cost.intrinsicNCCDrainCount.value, 1u);
}

TEST_F(ScheduleCostAnalysisTest, CountsNPUAndVectorLogicalOperationsByType) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, cx>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<3x4xf16, #wafer.memory<spm, cx>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<2x4xf16, #wafer.memory<spm, cx>>
    %f32a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %f32b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.gemm %lhs, %rhs into %dst
        {m = 2 : i64, k = 3 : i64, n = 4 : i64}
        : memref<2x3xf16, #wafer.memory<spm, cx>>,
          memref<3x4xf16, #wafer.memory<spm, cx>>
      into memref<2x4xf16, #wafer.memory<spm, cx>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %f32a, %f32a into %f32b
        : memref<4xf32, #wafer.memory<spm, tensor>>,
          memref<4xf32, #wafer.memory<spm, tensor>>
      into memref<4xf32, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.compute.npuF16Bf16LogicalOps.isKnown());
  EXPECT_EQ(cost.compute.npuF16Bf16LogicalOps.value, 48u);
  ASSERT_TRUE(cost.compute.vectorF32LogicalOps.isKnown());
  EXPECT_EQ(cost.compute.vectorF32LogicalOps.value, 4u);
}

TEST_F(ScheduleCostAnalysisTest, ClassifiesConvertWorkFromBothTypedEndpoints) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %bf16 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xbf16, #wafer.memory<spm, tensor>>
    %f16a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %f32 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %i8 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %f16b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %tf32 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66816>}
        : memref<4xtf32, #wafer.memory<spm, tensor>>
    wafer.instr.convert #wafer.instr_convert_kind<bf16_fp16>
        %bf16 into %f16a
        : memref<4xbf16, #wafer.memory<spm, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.convert #wafer.instr_convert_kind<fp16_fp32>
        %f16a into %f32
        : memref<4xf16, #wafer.memory<spm, tensor>>
       to memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.convert #wafer.instr_convert_kind<fp32_int8>
        %f32 into %i8 {rounding_mode = 0 : i64}
        : memref<4xf32, #wafer.memory<spm, tensor>>
       to memref<4xi8, #wafer.memory<spm, tensor>>
    wafer.instr.convert #wafer.instr_convert_kind<int8_fp16>
        %i8 into %f16b {zero_point = 0 : i64}
        : memref<4xi8, #wafer.memory<spm, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.convert #wafer.instr_convert_kind<bf16_tf32>
        %bf16 into %tf32
        : memref<4xbf16, #wafer.memory<spm, tensor>>
       to memref<4xtf32, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.compute.vectorF16Bf16LogicalOps.isKnown());
  EXPECT_EQ(cost.compute.vectorF16Bf16LogicalOps.value, 4u);
  ASSERT_TRUE(cost.compute.vectorF32LogicalOps.isKnown());
  EXPECT_EQ(cost.compute.vectorF32LogicalOps.value, 8u);
  ASSERT_TRUE(cost.compute.vectorOtherLogicalOps.isKnown());
  EXPECT_EQ(cost.compute.vectorOtherLogicalOps.value, 8u);
}

TEST_F(ScheduleCostAnalysisTest, RecomputesHighWaterFromPhysicalLayout) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, cx>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  mlir::memref::AllocOp allocation;
  module->walk([&](mlir::memref::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  std::optional<wafer::WaferPhysicalTensorInfo> physical =
      wafer::computeWaferPhysicalTensorInfo(allocation.getType());
  ASSERT_TRUE(physical);
  ASSERT_GT(physical->physicalBytes, physical->compactBytes);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.spmHighWaterBytes.isKnown());
  EXPECT_EQ(cost.spmHighWaterBytes.value,
            static_cast<uint64_t>(physical->physicalBytes));
}

TEST_F(ScheduleCostAnalysisTest,
       ResolvesRotatingSCFSlotsToEveryExternalAllocation) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %slot0 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %slot1 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %zero = arith.constant 0.0 : f16
    %result:2 = scf.for %index = %c0 to %c3 step %c1
        iter_args(%current = %slot0, %next = %slot1)
        -> (memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>) {
      wafer.instr.fill %current, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      scf.yield %next, %current
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.spmHighWaterBytes.isKnown());
  EXPECT_EQ(cost.spmHighWaterBytes.value, 264u);
}

TEST_F(ScheduleCostAnalysisTest, RejectsRotatingSCFSlotWithAnUnresolvedOrigin) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %unresolved: memref<4xf16, #wafer.memory<spm, tensor>>) {
    %known = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %zero = arith.constant 0.0 : f16
    %result:2 = scf.for %index = %c0 to %c3 step %c1
        iter_args(%current = %known, %next = %unresolved)
        -> (memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>) {
      wafer.instr.fill %current, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      scf.yield %next, %current
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.spmHighWaterBytes.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.spmHighWaterBytes.reason,
            ScheduleCostReason::UnsupportedSPMRoot);
}

TEST_F(ScheduleCostAnalysisTest,
       ResolvesEveryStructuredIfSPMResultToAcceptedAllocations) {
  auto module = parse(R"mlir(
module {
  func.func @main(%condition: i1) {
    %first = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %second = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %selected = scf.if %condition
        -> (memref<4xf16, #wafer.memory<spm, tensor>>) {
      scf.yield %first : memref<4xf16, #wafer.memory<spm, tensor>>
    } else {
      scf.yield %second : memref<4xf16, #wafer.memory<spm, tensor>>
    }
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %selected, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.spmHighWaterBytes.isKnown());
  EXPECT_EQ(cost.spmHighWaterBytes.value, 264u);
  ASSERT_TRUE(cost.compilerOwnedSPMBufferCount.isKnown());
  EXPECT_EQ(cost.compilerOwnedSPMBufferCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest,
       StructuredIfSPMResultStillFailsClosedOnUnresolvedOrigin) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %condition: i1,
      %unresolved: memref<4xf16, #wafer.memory<spm, tensor>>) {
    %known = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %selected = scf.if %condition
        -> (memref<4xf16, #wafer.memory<spm, tensor>>) {
      scf.yield %known : memref<4xf16, #wafer.memory<spm, tensor>>
    } else {
      scf.yield %unresolved : memref<4xf16, #wafer.memory<spm, tensor>>
    }
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %selected, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.spmHighWaterBytes.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.spmHighWaterBytes.reason,
            ScheduleCostReason::UnsupportedSPMRoot);
}

TEST_F(ScheduleCostAnalysisTest,
       KeepsAggregateNoCWhenDirectionalRouteIsUnresolved) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %sent = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %received = wafer.instr.dte_recv %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %sent, %received : !async.token, !async.token
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.noc.aggregateTransmitBytes.isKnown());
  EXPECT_EQ(cost.noc.aggregateTransmitBytes.value, 16u);
  ASSERT_TRUE(cost.noc.aggregateReceiveBytes.isKnown());
  EXPECT_EQ(cost.noc.aggregateReceiveBytes.value, 16u);
  EXPECT_EQ(cost.noc.staticIssueSiteCount.value, 2u);
  EXPECT_EQ(cost.noc.transmitMessageCount.value, 1u);
  EXPECT_EQ(cost.noc.receiveMessageCount.value, 1u);
  EXPECT_EQ(cost.noc.waitOperationCount.value, 1u);
  EXPECT_EQ(cost.noc.waitedEventCount.value, 2u);
  EXPECT_EQ(cost.work.instructions.staticSites.value, 3u);
  EXPECT_EQ(cost.work.instructions.exactExecutions.value, 3u);
  EXPECT_EQ(cost.work.asynchronousEvents.staticSites.value, 2u);
  EXPECT_EQ(cost.work.asynchronousEvents.exactExecutions.value, 2u);
  EXPECT_EQ(cost.work.dteSendOperations.staticSites.value, 1u);
  EXPECT_EQ(cost.work.dteReceiveOperations.staticSites.value, 1u);
  EXPECT_EQ(cost.work.dteWaitOperations.staticSites.value, 1u);
  for (NoCDirection direction : {NoCDirection::North, NoCDirection::South,
                                 NoCDirection::East, NoCDirection::West}) {
    const auto &directional = cost.noc.directional(direction);
    EXPECT_EQ(directional.knowledge, ScheduleCostKnowledge::Unavailable);
    EXPECT_EQ(directional.reason, ScheduleCostReason::UnresolvedNoCRoute);
  }
  ASSERT_TRUE(cost.eventCount.isKnown());
  EXPECT_EQ(cost.eventCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest, DynamicTripCountIsUnavailable) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %n: index,
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      wafer.instr.rdma %input to %buffer
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.instructionCount.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.instructionCount.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_EQ(cost.work.rdmaIssues.staticSites.value, 1u);
  EXPECT_EQ(cost.work.rdmaIssues.exactExecutions.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.work.rdmaIssues.exactExecutions.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_TRUE(cost.work.rdmaIssues.lowerBound.isKnown());
  EXPECT_EQ(cost.work.rdmaIssues.lowerBound.value, 0u);
  EXPECT_EQ(cost.work.rdmaIssues.upperBound.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.work.rdmaIssues.upperBound.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_EQ(cost.ddrReadBytes.knowledge, ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.ddrReadBytes.reason, ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_TRUE(cost.spmHighWaterBytes.isKnown());
  EXPECT_EQ(cost.spmHighWaterBytes.value, 8u);
}

TEST_F(ScheduleCostAnalysisTest,
       ConditionalWorkRetainsStaticSitesAndConservativeBounds) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %condition: i1,
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %output: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %buffer
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    scf.if %condition {
      wafer.instr.rdma %input to %buffer
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
    } else {
      wafer.instr.wdma %buffer to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.work.instructions.staticSites.value, 3u);
  ASSERT_TRUE(cost.work.instructions.exactExecutions.isKnown());
  EXPECT_EQ(cost.work.instructions.exactExecutions.value, 2u);
  EXPECT_EQ(cost.work.instructions.lowerBound.value, 2u);
  EXPECT_EQ(cost.work.instructions.upperBound.value, 2u);
  EXPECT_EQ(cost.work.rdmaIssues.staticSites.value, 2u);
  EXPECT_EQ(cost.work.rdmaIssues.exactExecutions.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.work.rdmaIssues.exactExecutions.reason,
            ScheduleCostReason::ConditionalControlFlow);
  EXPECT_EQ(cost.work.rdmaIssues.lowerBound.value, 1u);
  EXPECT_EQ(cost.work.rdmaIssues.upperBound.value, 2u);
  EXPECT_EQ(cost.work.wdmaIssues.staticSites.value, 1u);
  EXPECT_EQ(cost.work.wdmaIssues.exactExecutions.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.work.wdmaIssues.lowerBound.value, 0u);
  EXPECT_EQ(cost.work.wdmaIssues.upperBound.value, 1u);
  EXPECT_EQ(cost.ddrReadBytes.knowledge, ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.ddrWriteBytes.knowledge, ScheduleCostKnowledge::Unavailable);
  ASSERT_TRUE(cost.spmMovementBytes.isKnown());
  EXPECT_EQ(cost.spmMovementBytes.value, 16u);
}

TEST_F(ScheduleCostAnalysisTest,
       EqualConditionalPathsHaveExactExecutionCostAndTightBounds) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %condition: i1,
      %first: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %second: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    scf.if %condition {
      wafer.instr.rdma %first to %buffer
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
    } else {
      wafer.instr.rdma %second to %buffer
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.work.instructions.staticSites.value, 2u);
  ASSERT_TRUE(cost.work.instructions.exactExecutions.isKnown());
  EXPECT_EQ(cost.work.instructions.exactExecutions.value, 1u);
  EXPECT_EQ(cost.work.instructions.lowerBound.value, 1u);
  EXPECT_EQ(cost.work.instructions.upperBound.value, 1u);
  EXPECT_EQ(cost.work.rdmaIssues.staticSites.value, 2u);
  ASSERT_TRUE(cost.work.rdmaIssues.exactExecutions.isKnown());
  EXPECT_EQ(cost.work.rdmaIssues.exactExecutions.value, 1u);
  EXPECT_EQ(cost.work.rdmaIssues.lowerBound.value, 1u);
  EXPECT_EQ(cost.work.rdmaIssues.upperBound.value, 1u);
  ASSERT_TRUE(cost.ddrReadBytes.isKnown());
  EXPECT_EQ(cost.ddrReadBytes.value, 8u);
  ASSERT_TRUE(cost.ddrWriteBytes.isKnown());
  EXPECT_EQ(cost.ddrWriteBytes.value, 0u);
  ASSERT_TRUE(cost.spmMovementBytes.isKnown());
  EXPECT_EQ(cost.spmMovementBytes.value, 8u);
}

TEST_F(ScheduleCostAnalysisTest,
       CountsGatherScatterCallsAndBytesFromTheFinalInstructionWalk) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %source = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<3x2xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<3x2xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.work.gatherScatterOperations.staticSites.value, 1u);
  EXPECT_EQ(cost.work.gatherScatterOperations.exactExecutions.value, 4u);
  EXPECT_EQ(cost.work.tdmaIssues.exactExecutions.value, 4u);
  EXPECT_EQ(cost.gatherScatterBytes.value, 48u);
  EXPECT_EQ(cost.spmMovementBytes.value, 48u);
}

TEST_F(ScheduleCostAnalysisTest,
       CountsOnePrivateCalleeSiteAcrossMultipleStaticCalls) {
  auto module = parse(R"mlir(
module {
  func.func private @load(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %buffer: memref<4xf16, #wafer.memory<spm, tensor>>) {
    wafer.instr.rdma %input to %buffer
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
  func.func @main(%input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    call @load(%input, %buffer)
        : (memref<4xf16, #wafer.memory<ddr, tensor>>,
           memref<4xf16, #wafer.memory<spm, tensor>>) -> ()
    call @load(%input, %buffer)
        : (memref<4xf16, #wafer.memory<ddr, tensor>>,
           memref<4xf16, #wafer.memory<spm, tensor>>) -> ()
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.work.instructions.staticSites.value, 1u);
  EXPECT_EQ(cost.work.instructions.exactExecutions.value, 2u);
  EXPECT_EQ(cost.work.rdmaIssues.staticSites.value, 1u);
  EXPECT_EQ(cost.work.rdmaIssues.exactExecutions.value, 2u);
  EXPECT_EQ(cost.ddrReadBytes.value, 16u);
}

TEST_F(ScheduleCostAnalysisTest, LoopMultiplicityOverflowIsNotSaturated) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cmax = arith.constant 9223372036854775807 : index
    scf.for %i = %c0 to %cmax step %c1 {
      scf.for %j = %c0 to %c3 step %c1 {
        wafer.instr.rdma %input to %buffer
            {byte_count = 8 : i64, inner_bytes = 8 : i64,
             src_strides = array<i64: 0, 0, 0>,
             src_iterations = array<i64: 1, 1, 1>}
            : memref<4xf16, #wafer.memory<ddr, tensor>>
           to memref<4xf16, #wafer.memory<spm, tensor>>
      }
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.instructionCount.knowledge, ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.instructionCount.reason,
            ScheduleCostReason::ArithmeticOverflow);
  EXPECT_EQ(cost.ddrReadBytes.knowledge, ScheduleCostKnowledge::Overflow);
}

TEST_F(ScheduleCostAnalysisTest, ZeroTripLoopHasNoExecutionCost) {
  auto module = parse(R"mlir(
module {
  func.func private @recursive() {
    func.call @recursive() : () -> ()
    return
  }
  func.func @main() {
    %input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %value = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<1xf16, #wafer.memory<spm, tensor>>
    %index = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<1xi32, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c1 to %c0 step %c1 {
      func.call @recursive() : () -> ()
      wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax>
          %input into %value, %index {elem_count = 4 : i64}
          : memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<1xf16, #wafer.memory<spm, tensor>>,
             memref<1xi32, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.instructionCount.isKnown());
  EXPECT_EQ(cost.instructionCount.value, 0u);
  ASSERT_TRUE(cost.compute.vectorOtherLogicalOps.isKnown());
  EXPECT_EQ(cost.compute.vectorOtherLogicalOps.value, 0u);
}

TEST_F(ScheduleCostAnalysisTest, MissingAcceptedOffsetIsUnavailable) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %buffer, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.spmHighWaterBytes.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.spmHighWaterBytes.reason,
            ScheduleCostReason::MissingAcceptedSPMOffset);
}

TEST_F(ScheduleCostAnalysisTest,
       RecomputesReachableCompilerOwnedSPMHighWaterFromAcceptedOffsets) {
  auto module = parse(R"mlir(
module {
  func.func private @unused() {
    %unused = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
  func.func private @allocate() {
    %reachable = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
  func.func @main() {
    %local = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    func.call @allocate() : () -> ()
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.spmHighWaterBytes.isKnown());
  EXPECT_EQ(cost.spmHighWaterBytes.value, 264u);
  EXPECT_EQ(cost.compilerOwnedSPMBufferCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest,
       RecomputesReachableCompilerOwnedDDRHighWaterFromAcceptedOffsets) {
  auto module = parse(R"mlir(
module {
  func.func private @unused() {
    %unused = memref.alloc()
        {wafer.ddr.offset = #wafer.ddr_offset<1024>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    return
  }
  func.func @main() {
    %first = memref.alloc()
        {wafer.ddr.offset = #wafer.ddr_offset<0>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %second = memref.alloc()
        {wafer.ddr.offset = #wafer.ddr_offset<256>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.ddrHighWaterBytes.isKnown());
  EXPECT_EQ(cost.ddrHighWaterBytes.value, 264u);
  EXPECT_EQ(cost.compilerOwnedDDRBufferCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest, MissingAcceptedDDROffsetIsUnavailable) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.ddrHighWaterBytes.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.ddrHighWaterBytes.reason,
            ScheduleCostReason::MissingAcceptedDDROffset);
}

TEST_F(ScheduleCostAnalysisTest, InvalidAcceptedDDROffsetIsUnsupported) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc()
        {wafer.ddr.offset = #wafer.ddr_offset<-1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.ddrHighWaterBytes.knowledge,
            ScheduleCostKnowledge::Unsupported);
  EXPECT_EQ(cost.ddrHighWaterBytes.reason,
            ScheduleCostReason::InvalidAcceptedDDROffset);
}

TEST_F(ScheduleCostAnalysisTest, UnmodeledComputeSemanticsAreUnsupported) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %value = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<1xf16, #wafer.memory<spm, tensor>>
    %index = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<1xi32, #wafer.memory<spm, tensor>>
    wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax>
        %input into %value, %index {elem_count = 4 : i64}
        : memref<4xf16, #wafer.memory<spm, tensor>>
      into memref<1xf16, #wafer.memory<spm, tensor>>,
           memref<1xi32, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  EXPECT_EQ(cost.compute.vectorOtherLogicalOps.knowledge,
            ScheduleCostKnowledge::Unsupported);
  EXPECT_EQ(cost.compute.vectorOtherLogicalOps.reason,
            ScheduleCostReason::UnsupportedInstructionSemantics);
  EXPECT_TRUE(cost.instructionCount.isKnown());
  EXPECT_EQ(cost.instructionCount.value, 1u);
}

TEST_F(ScheduleCostAnalysisTest,
       AggregatesCardExecutionAndKeepsPrivateSPMDimensions) {
  auto first = parse(R"mlir(
module {
  func.func @main(%input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %buffer
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %buffer, %buffer into %buffer
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
      into memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  auto second = parse(R"mlir(
module {
  func.func @main(%input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %buffer
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);

  llvm::SmallVector<mlir::Operation *, 2> roots{first->getOperation(),
                                                second->getOperation()};
  auto cost = analyzeCardCost(roots, wafer::getTargetMemoryPolicy());
  ASSERT_EQ(cost.tileCosts.size(), 2u);
  ASSERT_TRUE(cost.aggregateDDRReadBytes.isKnown());
  EXPECT_EQ(cost.aggregateDDRReadBytes.value, 16u);
  ASSERT_TRUE(cost.aggregateSPMMovementBytes.isKnown());
  EXPECT_EQ(cost.aggregateSPMMovementBytes.value, 16u);
  ASSERT_TRUE(cost.aggregateInstructionCount.isKnown());
  EXPECT_EQ(cost.aggregateInstructionCount.value, 3u);
  EXPECT_EQ(cost.aggregateWork.instructions.staticSites.value, 3u);
  EXPECT_EQ(cost.aggregateWork.instructions.exactExecutions.value, 3u);
  EXPECT_EQ(cost.maximumTileWork.instructions.staticSites.value, 2u);
  EXPECT_EQ(cost.maximumTileWork.instructions.exactExecutions.value, 2u);
  EXPECT_EQ(cost.aggregateWork.rdmaIssues.exactExecutions.value, 2u);
  EXPECT_EQ(cost.maximumTileWork.rdmaIssues.exactExecutions.value, 1u);
  ASSERT_TRUE(cost.aggregateCompute.vectorF16Bf16LogicalOps.isKnown());
  EXPECT_EQ(cost.aggregateCompute.vectorF16Bf16LogicalOps.value, 4u);
  ASSERT_TRUE(cost.maximumTileSPMHighWaterBytes.isKnown());
  EXPECT_EQ(cost.maximumTileSPMHighWaterBytes.value, 264u);
  ASSERT_TRUE(cost.summedTileSPMHighWaterBytes.isKnown());
  EXPECT_EQ(cost.summedTileSPMHighWaterBytes.value, 272u);
  ASSERT_TRUE(cost.maximumTileDDRHighWaterBytes.isKnown());
  EXPECT_EQ(cost.maximumTileDDRHighWaterBytes.value, 0u);
  ASSERT_TRUE(cost.summedTileDDRHighWaterBytes.isKnown());
  EXPECT_EQ(cost.summedTileDDRHighWaterBytes.value, 0u);
  EXPECT_EQ(cost.aggregateCompilerOwnedSPMBufferCount.value, 2u);
  EXPECT_EQ(cost.maximumTileCompilerOwnedSPMBufferCount.value, 1u);
  EXPECT_EQ(cost.aggregateCompilerOwnedDDRBufferCount.value, 0u);
  EXPECT_EQ(cost.maximumTileCompilerOwnedDDRBufferCount.value, 0u);
}

TEST_F(ScheduleCostAnalysisTest,
       CardAggregationPreservesUnavailableCostReason) {
  auto dynamic = parse(R"mlir(
module {
  func.func @main(
      %n: index,
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      wafer.instr.rdma %input to %buffer
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(dynamic);
  llvm::SmallVector<mlir::Operation *, 1> roots{dynamic->getOperation()};
  auto cost = analyzeCardCost(roots, wafer::getTargetMemoryPolicy());
  EXPECT_EQ(cost.aggregateDDRReadBytes.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.aggregateDDRReadBytes.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_TRUE(cost.maximumTileSPMHighWaterBytes.isKnown());
}

TEST_F(ScheduleCostAnalysisTest, CardTileDistanceChangesMinimumHopDemand) {
  constexpr llvm::StringLiteral nearSend = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 1, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";
  constexpr llvm::StringLiteral farSend = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 12 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 1, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> nearOwners;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> farOwners;
  llvm::SmallVector<mlir::Operation *, 16> nearRoots;
  llvm::SmallVector<mlir::Operation *, 16> farRoots;
  for (int64_t tile = 0; tile < 16; ++tile) {
    nearOwners.push_back(makeDTETileModule("", tile == 0 ? nearSend : ""));
    farOwners.push_back(makeDTETileModule("", tile == 0 ? farSend : ""));
    ASSERT_TRUE(nearOwners.back());
    ASSERT_TRUE(farOwners.back());
    nearRoots.push_back(nearOwners.back()->getOperation());
    farRoots.push_back(farOwners.back()->getOperation());
  }

  const auto policy = wafer::getTargetMemoryPolicy();
  auto nearCost = analyzeCardCost(nearRoots, policy);
  auto farCost = analyzeCardCost(farRoots, policy);
  ASSERT_TRUE(nearCost.aggregateNoC.aggregateTransmitBytes.isKnown());
  ASSERT_TRUE(farCost.aggregateNoC.aggregateTransmitBytes.isKnown());
  EXPECT_EQ(nearCost.aggregateNoC.aggregateTransmitBytes.value, 16u);
  EXPECT_EQ(farCost.aggregateNoC.aggregateTransmitBytes.value, 16u);
  ASSERT_TRUE(nearCost.aggregateNoC.transmitMessageCount.isKnown());
  ASSERT_TRUE(farCost.aggregateNoC.transmitMessageCount.isKnown());
  EXPECT_EQ(nearCost.aggregateNoC.transmitMessageCount.value, 1u);
  EXPECT_EQ(farCost.aggregateNoC.transmitMessageCount.value, 1u);
  EXPECT_EQ(nearCost.maximumTileNoCTransmitBytes.value, 16u);
  EXPECT_EQ(farCost.maximumTileNoCTransmitBytes.value, 16u);
  ASSERT_TRUE(nearCost.minimumHopLinkByteDemand.isKnown());
  ASSERT_TRUE(farCost.minimumHopLinkByteDemand.isKnown());
  EXPECT_EQ(nearCost.minimumHopLinkByteDemand.value, 16u);
  EXPECT_EQ(farCost.minimumHopLinkByteDemand.value, 48u);
  EXPECT_EQ(nearCost.minimumHopMessageDemand.value, 1u);
  EXPECT_EQ(farCost.minimumHopMessageDemand.value, 3u);
  EXPECT_EQ(nearCost.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(farCost.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(nearCost.idealizedMinimumPeakLinkByteDemand.value, 1u);
  EXPECT_EQ(farCost.idealizedMinimumPeakLinkByteDemand.value, 1u);
  ASSERT_TRUE(nearCost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown());
  ASSERT_TRUE(farCost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown());
  EXPECT_EQ(nearCost.modeledNoCRoute.kind,
            wafer::analysis::ModeledNoCRouteKind::CanonicalShortestPath);
  EXPECT_EQ(nearCost.modeledNoCRoute.peakDirectedLinkByteDemand.value, 16u);
  EXPECT_EQ(farCost.modeledNoCRoute.peakDirectedLinkByteDemand.value, 16u);
  EXPECT_EQ(nearCost.maximumNoCHopCount.value, 1u);
  EXPECT_EQ(farCost.maximumNoCHopCount.value, 3u);
}

TEST_F(ScheduleCostAnalysisTest, CardNoCFreeProgramHasExactZeroWork) {
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::Operation *, 16> roots;
  for (int64_t tile = 0; tile < 16; ++tile) {
    owners.push_back(makeDTETileModule("", ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }
  const auto policy = wafer::getTargetMemoryPolicy();
  auto cost = analyzeCardCost(roots, policy);
  ASSERT_TRUE(cost.aggregateNoC.aggregateTransmitBytes.isKnown());
  ASSERT_TRUE(cost.aggregateNoC.aggregateReceiveBytes.isKnown());
  ASSERT_TRUE(cost.aggregateNoC.transmitMessageCount.isKnown());
  ASSERT_TRUE(cost.aggregateNoC.receiveMessageCount.isKnown());
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.value, 0u);
  EXPECT_EQ(cost.aggregateNoC.aggregateReceiveBytes.value, 0u);
  EXPECT_EQ(cost.aggregateNoC.transmitMessageCount.value, 0u);
  EXPECT_EQ(cost.aggregateNoC.receiveMessageCount.value, 0u);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.value, 0u);
  EXPECT_EQ(cost.minimumHopMessageDemand.value, 0u);
  EXPECT_EQ(cost.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(cost.idealizedMinimumPeakLinkByteDemand.value, 0u);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.value, 0u);
  EXPECT_EQ(cost.maximumNoCHopCount.value, 0u);
}

TEST_F(ScheduleCostAnalysisTest, CardExactNoCWorkUsesStaticLoopMultiplicity) {
  constexpr llvm::StringLiteral repeatedSend = R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c3 step %c1 {
      %sent = wafer.instr.dte_send %buffer
          {peer = 2 : i64, bytes = 4 : i64,
           message = #wafer.dte_message<communication = 2, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    }
)mlir";
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::Operation *, 16> roots;
  for (int64_t tile = 0; tile < 16; ++tile) {
    owners.push_back(makeDTETileModule("", tile == 0 ? repeatedSend : ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }
  auto cost = analyzeCardCost(roots, wafer::getTargetMemoryPolicy());
  ASSERT_TRUE(cost.aggregateNoC.aggregateTransmitBytes.isKnown());
  ASSERT_TRUE(cost.aggregateNoC.transmitMessageCount.isKnown());
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.value, 12u);
  EXPECT_EQ(cost.aggregateNoC.transmitMessageCount.value, 3u);
  EXPECT_EQ(cost.aggregateWork.dteSendOperations.exactExecutions.value, 3u);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.value, 24u);
  EXPECT_EQ(cost.minimumHopMessageDemand.value, 6u);
  EXPECT_EQ(cost.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(cost.idealizedMinimumPeakLinkByteDemand.value, 1u);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.value, 12u);
  EXPECT_EQ(cost.maximumNoCHopCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest,
       MaximumTileNoCPressureDistinguishesEqualAggregateWork) {
  constexpr llvm::StringLiteral sendLeft = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 1, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";
  constexpr llvm::StringLiteral sendRight = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 2 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 2, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";
  constexpr llvm::StringLiteral sendBoth = R"mlir(
    %left = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 1, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    %right = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 2, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> balancedOwners;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> hotspotOwners;
  llvm::SmallVector<mlir::Operation *, 16> balancedRoots;
  llvm::SmallVector<mlir::Operation *, 16> hotspotRoots;
  for (int64_t tile = 0; tile < 16; ++tile) {
    llvm::StringRef balancedBody = tile == 1   ? sendLeft
                                   : tile == 3 ? sendRight
                                               : "";
    llvm::StringRef hotspotBody = tile == 1 ? sendBoth : "";
    balancedOwners.push_back(makeDTETileModule("", balancedBody));
    hotspotOwners.push_back(makeDTETileModule("", hotspotBody));
    ASSERT_TRUE(balancedOwners.back());
    ASSERT_TRUE(hotspotOwners.back());
    balancedRoots.push_back(balancedOwners.back()->getOperation());
    hotspotRoots.push_back(hotspotOwners.back()->getOperation());
  }

  const auto policy = wafer::getTargetMemoryPolicy();
  auto balanced = analyzeCardCost(balancedRoots, policy);
  auto hotspot = analyzeCardCost(hotspotRoots, policy);
  ASSERT_TRUE(balanced.aggregateNoC.aggregateTransmitBytes.isKnown());
  ASSERT_TRUE(hotspot.aggregateNoC.aggregateTransmitBytes.isKnown());
  EXPECT_EQ(balanced.aggregateNoC.aggregateTransmitBytes.value, 32u);
  EXPECT_EQ(hotspot.aggregateNoC.aggregateTransmitBytes.value, 32u);
  ASSERT_TRUE(balanced.maximumTileNoCTransmitBytes.isKnown());
  ASSERT_TRUE(hotspot.maximumTileNoCTransmitBytes.isKnown());
  EXPECT_EQ(balanced.maximumTileNoCTransmitBytes.value, 16u);
  EXPECT_EQ(hotspot.maximumTileNoCTransmitBytes.value, 32u);
  EXPECT_EQ(balanced.aggregateNoC.transmitMessageCount.value, 2u);
  EXPECT_EQ(hotspot.aggregateNoC.transmitMessageCount.value, 2u);
  EXPECT_EQ(balanced.minimumHopLinkByteDemand.value, 32u);
  EXPECT_EQ(hotspot.minimumHopLinkByteDemand.value, 32u);
  EXPECT_EQ(balanced.minimumHopMessageDemand.value, 2u);
  EXPECT_EQ(hotspot.minimumHopMessageDemand.value, 2u);
  EXPECT_EQ(balanced.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(hotspot.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(balanced.idealizedMinimumPeakLinkByteDemand.value, 1u);
  EXPECT_EQ(hotspot.idealizedMinimumPeakLinkByteDemand.value, 1u);
  EXPECT_EQ(balanced.modeledNoCRoute.peakDirectedLinkByteDemand.value, 16u);
  EXPECT_EQ(hotspot.modeledNoCRoute.peakDirectedLinkByteDemand.value, 32u);
  EXPECT_EQ(balanced.maximumNoCHopCount.value, 1u);
  EXPECT_EQ(hotspot.maximumNoCHopCount.value, 1u);
}

TEST_F(ScheduleCostAnalysisTest,
       PeerDataflowHasExactP2PTrafficWithoutCollectiveAttribution) {
  constexpr llvm::StringLiteral send = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 2 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 17, round = 0, slice = 3>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %sent : !async.token
)mlir";
  constexpr llvm::StringLiteral recv = R"mlir(
    %received = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 17, round = 0, slice = 3>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %received : !async.token
)mlir";

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::Operation *, 16> roots;
  for (int64_t tile = 0; tile < 16; ++tile) {
    owners.push_back(makeDTETileModule("", tile == 0   ? send
                                           : tile == 2 ? recv
                                                       : ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }

  auto cost = analyzeCardCost(roots, wafer::getTargetMemoryPolicy());
  ASSERT_TRUE(cost.aggregateNoC.aggregateTransmitBytes.isKnown());
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.value, 8u);
  ASSERT_TRUE(cost.aggregateNoC.aggregateReceiveBytes.isKnown());
  EXPECT_EQ(cost.aggregateNoC.aggregateReceiveBytes.value, 8u);
  EXPECT_EQ(cost.aggregateNoC.staticIssueSiteCount.value, 2u);
  EXPECT_EQ(cost.aggregateNoC.transmitMessageCount.value, 1u);
  EXPECT_EQ(cost.aggregateNoC.receiveMessageCount.value, 1u);
  EXPECT_EQ(cost.aggregateNoC.waitOperationCount.value, 2u);
  EXPECT_EQ(cost.aggregateNoC.waitedEventCount.value, 2u);
  EXPECT_EQ(cost.maximumTileNoCTransmitBytes.value, 8u);
  EXPECT_EQ(cost.maximumTileNoCReceiveBytes.value, 8u);
  EXPECT_EQ(cost.maximumTileNoCTransmitMessageCount.value, 1u);
  EXPECT_EQ(cost.maximumTileNoCReceiveMessageCount.value, 1u);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.value, 16u);
  EXPECT_EQ(cost.minimumHopMessageDemand.value, 2u);
  EXPECT_EQ(cost.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(cost.idealizedMinimumPeakLinkByteDemand.value, 1u);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.value, 8u);
  EXPECT_EQ(cost.maximumNoCHopCount.value, 2u);
  ASSERT_TRUE(cost.aggregateEventCount.isKnown());
  EXPECT_EQ(cost.aggregateEventCount.value, 2u);
  ASSERT_TRUE(cost.aggregateInstructionCount.isKnown());
  EXPECT_EQ(cost.aggregateInstructionCount.value, 4u);
}

TEST_F(ScheduleCostAnalysisTest, CardExactNoCWorkPropagatesArithmeticOverflow) {
  constexpr llvm::StringLiteral overflowingSend = R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cmax = arith.constant 9223372036854775807 : index
    scf.for %i = %c0 to %cmax step %c1 {
      %sent = wafer.instr.dte_send %buffer
          {peer = 2 : i64, bytes = 3 : i64,
           message = #wafer.dte_message<communication = 3, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    }
)mlir";
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::Operation *, 16> roots;
  for (int64_t tile = 0; tile < 16; ++tile) {
    owners.push_back(makeDTETileModule("", tile == 0 ? overflowingSend : ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }
  auto cost = analyzeCardCost(roots, wafer::getTargetMemoryPolicy());
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.reason,
            ScheduleCostReason::ArithmeticOverflow);
  ASSERT_TRUE(cost.aggregateNoC.transmitMessageCount.isKnown());
  EXPECT_EQ(cost.aggregateNoC.transmitMessageCount.value,
            9223372036854775807ULL);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.reason,
            ScheduleCostReason::ArithmeticOverflow);
  EXPECT_EQ(cost.minimumHopMessageDemand.value,
            std::numeric_limits<uint64_t>::max() - 1);
  EXPECT_EQ(cost.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(cost.idealizedMinimumPeakLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.maximumNoCHopCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest, CardExactNoCAggregationPropagatesOverflow) {
  constexpr llvm::StringLiteral overflowingSharedLink = R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cmax = arith.constant 9223372036854775807 : index
    scf.for %i = %c0 to %cmax step %c1 {
      %first = wafer.instr.dte_send %buffer
          {peer = 1 : i64, bytes = 2 : i64,
           message = #wafer.dte_message<communication = 31, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
      %second = wafer.instr.dte_send %buffer
          {peer = 1 : i64, bytes = 2 : i64,
           message = #wafer.dte_message<communication = 32, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    }
)mlir";
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::Operation *, 16> roots;
  for (int64_t tile = 0; tile < 16; ++tile) {
    owners.push_back(
        makeDTETileModule("", tile == 0 ? overflowingSharedLink : ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }
  auto cost = analyzeCardCost(roots, wafer::getTargetMemoryPolicy());
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.reason,
            ScheduleCostReason::ArithmeticOverflow);
  ASSERT_TRUE(cost.aggregateNoC.transmitMessageCount.isKnown());
  EXPECT_EQ(cost.aggregateNoC.transmitMessageCount.value,
            std::numeric_limits<uint64_t>::max() - 1);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.minimumHopMessageDemand.value,
            std::numeric_limits<uint64_t>::max() - 1);
  EXPECT_EQ(cost.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(cost.idealizedMinimumPeakLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.maximumNoCHopCount.value, 1u);
}

TEST_F(ScheduleCostAnalysisTest,
       DTEPeerLeafVerifierDoesNotRebuildTargetTopology) {
  constexpr llvm::StringLiteral send = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 4 : i64,
         message = #wafer.dte_message<communication = 4, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";
  auto tile0 = makeDTETileModule("", send, false);
  auto tile1 = makeDTETileModule("", "", false);
  ASSERT_TRUE(tile0);
  ASSERT_TRUE(tile1);
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  // Cross-operation topology is checked once by the CardModule container and
  // again by whole-executable verification. A leaf DTE verifier checks only
  // local fields.
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*tile0)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*tile1)));
}

TEST_F(ScheduleCostAnalysisTest,
       CardExactNoCWorkPreservesUnavailableDynamicMultiplicity) {
  constexpr llvm::StringLiteral dynamicSend = R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %sent = wafer.instr.dte_send %buffer
          {peer = 1 : i64, bytes = 4 : i64,
           message = #wafer.dte_message<communication = 5, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    }
)mlir";
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::Operation *, 16> roots;
  for (int64_t tile = 0; tile < 16; ++tile) {
    owners.push_back(makeDTETileModule(tile == 0 ? "%n: index" : "",
                                       tile == 0 ? dynamicSend : ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }
  auto cost = analyzeCardCost(roots, wafer::getTargetMemoryPolicy());
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_EQ(cost.aggregateNoC.transmitMessageCount.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.aggregateNoC.transmitMessageCount.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_EQ(cost.minimumHopMessageDemand.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.minimumHopMessageDemand.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_EQ(cost.directedNoCLinkCount.value, 48u);
  EXPECT_EQ(cost.idealizedMinimumPeakLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Unavailable);
  EXPECT_EQ(cost.maximumNoCHopCount.value, 1u);
}

} // namespace
