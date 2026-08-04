//===- ScheduleCostAnalysisTest.cpp --------------------------------------===//

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
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
using wafer::analysis::NoCCollectiveKind;
using wafer::analysis::NoCDirection;
using wafer::analysis::ScheduleCostKnowledge;
using wafer::analysis::ScheduleCostReason;

class ScheduleCostAnalysisTest : public ::testing::Test {
protected:
  ScheduleCostAnalysisTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
    wafer::registerAllDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  InstructionProgramCost analyze(mlir::ModuleOp module) {
    return wafer::analysis::analyzeInstructionProgramCost(
        module, wafer::analysis::getTargetScheduleCostPolicy());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeDTERankModule(
      int64_t rankCount, int64_t tileColumns, llvm::StringRef functionArguments,
      llvm::StringRef instructionBody, bool includeTopology = true) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << "module {\n";
    if (includeTopology) {
      os << "  wafer.target.topology @topology {card_grid = array<i64: 1, 1>, "
            "card_interconnect = \"mesh\", tile_grid = array<i64: 1, "
         << tileColumns << ">, unavailable_tiles = array<i64>}\n"
         << "  wafer.execution.mesh @mesh {topology = @topology, axes = "
            "[\"rank\"], shape = array<i64: "
         << rankCount
         << ">, policy = \"all_available\", endpoints = array<i64>}\n";
    }
    os << "  func.func @main(" << functionArguments << ") {\n"
       << "    %buffer = memref.alloc() "
          "{wafer.spm.offset = #wafer.spm_offset<65536>} "
          ": memref<16xi8, #wafer.memory<spm, tensor>>\n"
       << instructionBody << "\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return parse(os.str());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(ScheduleCostAnalysisTest, ExposesOnlyEstablishedTargetRates) {
  auto policy = wafer::analysis::getTargetScheduleCostPolicy();
  EXPECT_EQ(policy.cardDDRBytesPerSecond, 200'000'000'000ULL);
  EXPECT_EQ(policy.cardDDRNominalBytesPerSecond, 150'000'000'000ULL);
  EXPECT_EQ(policy.directionalNoCBytesPerSecond, 128'000'000'000ULL);
  EXPECT_EQ(policy.f16Bf16NPULogicalOpsPerSecondPerTile, 8'000'000'000'000ULL);
  EXPECT_EQ(policy.f16Bf16VectorLogicalOpsPerSecondPerTile, 64'000'000'000ULL);
  EXPECT_EQ(policy.f32VectorLogicalOpsPerSecondPerTile, 32'000'000'000ULL);
  EXPECT_FALSE(policy.cardDDRSustainedBytesPerSecondLowerBound);
  EXPECT_FALSE(policy.directionalNoCSustainedBytesPerSecondLowerBound);
  EXPECT_FALSE(policy.dteEndpointBytesPerSecondLowerBound);
  EXPECT_FALSE(policy.dteMessageStartupPicosecondsUpperBound);
  EXPECT_FALSE(policy.noCHopPicosecondsUpperBound);
  EXPECT_FALSE(policy.noCRouteDilationUpperBound);
  EXPECT_FALSE(policy.instructionFixedPicosecondsUpperBound);
  EXPECT_FALSE(policy.dteWaitedEventPicosecondsUpperBound);
  EXPECT_FALSE(policy.nccParticipantWaitPicosecondsUpperBound);
  EXPECT_EQ(policy.spmAddressBase, 65536u);
  EXPECT_EQ(policy.spmAddressLimit, 3080192u);
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
  ASSERT_TRUE(cost.qualifiedOverlapWindowCount.isKnown());
  EXPECT_EQ(cost.qualifiedOverlapWindowCount.value, 0u);
  ASSERT_TRUE(cost.spmHighWaterBytes.isKnown());
  EXPECT_EQ(cost.spmHighWaterBytes.value, 264u);
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

TEST_F(ScheduleCostAnalysisTest,
       RecognizesQualifiedF16RotatingRDMACTWDMAWindow) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %lhs: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %rhs: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %output: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %lhs0 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %lhs1 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %rhs0 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %rhs1 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %out0 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %out1 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66816>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %result:6 = scf.for %index = %c0 to %c8 step %c1
        iter_args(%lhs_current = %lhs0, %lhs_next = %lhs1,
                  %rhs_current = %rhs0, %rhs_next = %rhs1,
                  %out_current = %out0, %out_next = %out1)
        -> (memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>) {
      wafer.instr.rdma %lhs to %lhs_current
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %rhs to %rhs_current
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %lhs_next, %rhs_next into %out_next
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
         into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %out_current to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      scf.yield %lhs_next, %lhs_current, %rhs_next, %rhs_current,
                %out_next, %out_current
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.qualifiedOverlapWindowCount.isKnown());
  EXPECT_EQ(cost.qualifiedOverlapWindowCount.value, 1u);
  ASSERT_TRUE(cost.directDTEComputeOverlapWindowCount.isKnown());
  EXPECT_EQ(cost.directDTEComputeOverlapWindowCount.value, 0u);
}

TEST_F(ScheduleCostAnalysisTest,
       DirectDTEComputeWitnessRequiresIssueBeforeMatchingWait) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %send0 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %send1 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %compute0 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %compute1 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %steady_upper = arith.subi %c5, %c1 : index
    %result:4 = scf.for %index = %c0 to %steady_upper step %c1
        iter_args(%send_current = %send0, %send_next = %send1,
                  %compute_current = %compute0, %compute_next = %compute1)
        -> (memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>) {
      %sent = wafer.instr.dte_send %send0
          {peer = 1 : i64, bytes = 8 : i64,
           message = #wafer.dte_message<communication = 71, phase = peer_dataflow, round = 0, slice = 0>,
           binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 65792, route_bindings = [], completion = sender_wait_receiver_fsm>}
          : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
      wafer.instr.elementwise <add> %compute0, %compute0
          into %compute0
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
         into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.dte_wait %sent : !async.token
      scf.yield %send_next, %send_current, %compute_next, %compute_current
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.directDTEComputeOverlapWindowCount.isKnown());
  EXPECT_EQ(cost.directDTEComputeOverlapWindowCount.value, 1u);
  ASSERT_TRUE(cost.qualifiedOverlapWindowCount.isKnown());
  EXPECT_EQ(cost.qualifiedOverlapWindowCount.value, 1u);
}

TEST_F(ScheduleCostAnalysisTest,
       DirectDTEComputeWitnessAcceptsSendReadInStraightLineBlock) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %send = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %sent = wafer.instr.dte_send %send
        {peer = 1 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 72, phase = peer_dataflow, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 65792, route_bindings = [], completion = sender_wait_receiver_fsm>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.elementwise <add> %send, %send into %dest
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
       into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %sent : !async.token
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.directDTEComputeOverlapWindowCount.isKnown());
  EXPECT_EQ(cost.directDTEComputeOverlapWindowCount.value, 1u);
}

TEST_F(ScheduleCostAnalysisTest,
       DirectDTEComputeWitnessRejectsConflictingFootprints) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %send = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %recv = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %other = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %sent = wafer.instr.dte_send %send
        {peer = 1 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 73, phase = peer_dataflow, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 65792, route_bindings = [], completion = sender_wait_receiver_fsm>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.elementwise <add> %other, %other into %send
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
       into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %sent : !async.token
    %received = wafer.instr.dte_recv %recv
        {peer = 1 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 74, phase = peer_dataflow, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 1, remote_address_mode = absolute, remote_receiver_address = 65536, route_bindings = [], completion = sender_wait_receiver_fsm>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.elementwise <add> %recv, %recv into %other
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
       into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %received : !async.token
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.directDTEComputeOverlapWindowCount.isKnown());
  EXPECT_EQ(cost.directDTEComputeOverlapWindowCount.value, 0u);
}

TEST_F(ScheduleCostAnalysisTest, RejectsRotatingSCFSlotWithAnUnknownOrigin) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %unknown: memref<4xf16, #wafer.memory<spm, tensor>>) {
    %known = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %zero = arith.constant 0.0 : f16
    %result:2 = scf.for %index = %c0 to %c3 step %c1
        iter_args(%current = %known, %next = %unknown)
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
  EXPECT_EQ(cost.spmHighWaterBytes.knowledge, ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.spmHighWaterBytes.reason,
            ScheduleCostReason::UnsupportedSPMRoot);
}

TEST_F(ScheduleCostAnalysisTest,
       KeepsAggregateNoCWhenDirectionalRouteIsUnresolved) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %sent = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, phase = all_reduce_ring, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %received = wafer.instr.dte_recv %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, phase = all_reduce_ring, round = 0, slice = 0>}
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
  ASSERT_TRUE(cost.noc.collective(NoCCollectiveKind::AllReduce).isKnown());
  EXPECT_EQ(cost.noc.collective(NoCCollectiveKind::AllReduce).value, 16u);
  for (NoCDirection direction : {NoCDirection::North, NoCDirection::South,
                                 NoCDirection::East, NoCDirection::West}) {
    const auto &directional = cost.noc.directional(direction);
    EXPECT_EQ(directional.knowledge, ScheduleCostKnowledge::Unknown);
    EXPECT_EQ(directional.reason, ScheduleCostReason::UnresolvedNoCRoute);
  }
  ASSERT_TRUE(cost.eventCount.isKnown());
  EXPECT_EQ(cost.eventCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest, DynamicTripCountIsTypedUnknown) {
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
  EXPECT_EQ(cost.instructionCount.knowledge, ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.instructionCount.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_EQ(cost.ddrReadBytes.knowledge, ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.ddrReadBytes.reason, ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_TRUE(cost.spmHighWaterBytes.isKnown());
  EXPECT_EQ(cost.spmHighWaterBytes.value, 8u);
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

TEST_F(ScheduleCostAnalysisTest, MissingAcceptedOffsetIsTypedUnknown) {
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
  EXPECT_EQ(cost.spmHighWaterBytes.knowledge, ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.spmHighWaterBytes.reason,
            ScheduleCostReason::MissingAcceptedSPMOffset);
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
       MeasuresStructuralDataDependencyDepthWithoutTimingAssumptions) {
  auto chain = parse(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %d = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %ab = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %abc = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66816>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67072>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <add> %a, %b into %ab
        : memref<4xi8, #wafer.memory<spm, tensor>>,
          memref<4xi8, #wafer.memory<spm, tensor>>
       into memref<4xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <add> %ab, %c into %abc
        : memref<4xi8, #wafer.memory<spm, tensor>>,
          memref<4xi8, #wafer.memory<spm, tensor>>
       into memref<4xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <add> %abc, %d into %out
        : memref<4xi8, #wafer.memory<spm, tensor>>,
          memref<4xi8, #wafer.memory<spm, tensor>>
       into memref<4xi8, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  auto balanced = parse(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %d = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %ab = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %cd = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66816>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67072>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <add> %a, %b into %ab
        : memref<4xi8, #wafer.memory<spm, tensor>>,
          memref<4xi8, #wafer.memory<spm, tensor>>
       into memref<4xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <add> %c, %d into %cd
        : memref<4xi8, #wafer.memory<spm, tensor>>,
          memref<4xi8, #wafer.memory<spm, tensor>>
       into memref<4xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <add> %ab, %cd into %out
        : memref<4xi8, #wafer.memory<spm, tensor>>,
          memref<4xi8, #wafer.memory<spm, tensor>>
       into memref<4xi8, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(chain);
  ASSERT_TRUE(balanced);
  InstructionProgramCost chainCost = analyze(*chain);
  InstructionProgramCost balancedCost = analyze(*balanced);
  ASSERT_TRUE(chainCost.dataDependencyDepth.isKnown());
  ASSERT_TRUE(balancedCost.dataDependencyDepth.isKnown());
  EXPECT_EQ(chainCost.dataDependencyDepth.value, 3u);
  EXPECT_EQ(balancedCost.dataDependencyDepth.value, 2u);
  EXPECT_EQ(chainCost.instructionCount.value,
            balancedCost.instructionCount.value);
  EXPECT_EQ(chainCost.compute.vectorOtherLogicalOps.value,
            balancedCost.compute.vectorOtherLogicalOps.value);
}

TEST_F(ScheduleCostAnalysisTest, TypedJoinOnlyOrdersItsNCCWorkerParticipants) {
  auto module = parse(R"mlir(
module {
  func.func @main(%zero: f32) {
    %worker0 = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %worker1_before = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %worker1_after = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.fill %worker0, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    wafer.instr.fill %worker1_before, %zero
        {worker = #wafer.ncc_worker<worker1>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    wafer.instr.ncc_join [0]
    wafer.instr.fill %worker1_after, %zero
        {worker = #wafer.ncc_worker<worker1>}
        : memref<4xf32, #wafer.memory<spm, tensor>>, f32
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.dataDependencyDepth.isKnown());
  EXPECT_EQ(cost.dataDependencyDepth.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest,
       MeasuresReadyPriorityInversionsWithoutAOverlapModel) {
  auto sourceOrder = parse(R"mlir(
module {
  func.func @main() {
    %compute_source = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %compute_dest = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %dma_dest = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %ddr = memref.alloc()
        : memref<4xi8, #wafer.memory<ddr, tensor>>
    wafer.instr.elementwise <neg> %compute_source into %compute_dest
        : memref<4xi8, #wafer.memory<spm, tensor>>
       into memref<4xi8, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %ddr to %dma_dest
        {byte_count = 4 : i64, inner_bytes = 4 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<4xi8, #wafer.memory<ddr, tensor>>
       to memref<4xi8, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  auto readyOrder = parse(R"mlir(
module {
  func.func @main() {
    %compute_source = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %compute_dest = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %dma_dest = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %ddr = memref.alloc()
        : memref<4xi8, #wafer.memory<ddr, tensor>>
    wafer.instr.rdma %ddr to %dma_dest
        {byte_count = 4 : i64, inner_bytes = 4 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<4xi8, #wafer.memory<ddr, tensor>>
       to memref<4xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <neg> %compute_source into %compute_dest
        : memref<4xi8, #wafer.memory<spm, tensor>>
       into memref<4xi8, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(sourceOrder);
  ASSERT_TRUE(readyOrder);
  InstructionProgramCost sourceCost = analyze(*sourceOrder);
  InstructionProgramCost readyCost = analyze(*readyOrder);
  ASSERT_TRUE(sourceCost.readyOrderPriorityInversions.isKnown());
  ASSERT_TRUE(readyCost.readyOrderPriorityInversions.isKnown());
  EXPECT_EQ(sourceCost.readyOrderPriorityInversions.value, 1u);
  EXPECT_EQ(readyCost.readyOrderPriorityInversions.value, 0u);
  EXPECT_EQ(sourceCost.dataDependencyDepth.value,
            readyCost.dataDependencyDepth.value);
  EXPECT_EQ(sourceCost.instructionCount.value,
            readyCost.instructionCount.value);
}

TEST_F(ScheduleCostAnalysisTest,
       ArgWritebackBarrierContributesStructuralDependencyDepth) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %ddr: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %compute_source = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %compute_dest = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %arg_input = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %arg_value = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<1xf16, #wafer.memory<spm, tensor>>
    %arg_index = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<1xi32, #wafer.memory<spm, tensor>>
    %dma_dest = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66816>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <neg> %compute_source into %compute_dest
        : memref<4xf16, #wafer.memory<spm, tensor>>
       into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax>
        %arg_input into %arg_value, %arg_index {elem_count = 4 : i64}
        : memref<4xf16, #wafer.memory<spm, tensor>>
      into memref<1xf16, #wafer.memory<spm, tensor>>,
           memref<1xi32, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %ddr to %dma_dest
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  InstructionProgramCost cost = analyze(*module);
  ASSERT_TRUE(cost.dataDependencyDepth.isKnown());
  EXPECT_EQ(cost.dataDependencyDepth.value, 3u);
  ASSERT_TRUE(cost.readyOrderPriorityInversions.isKnown());
  EXPECT_EQ(cost.readyOrderPriorityInversions.value, 0u);
}

TEST_F(ScheduleCostAnalysisTest,
       AggregatesWholeCardExecutionAndKeepsPrivateSPMDimensions) {
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
  auto cost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      roots, wafer::analysis::getTargetScheduleCostPolicy());
  ASSERT_EQ(cost.rankCosts.size(), 2u);
  ASSERT_TRUE(cost.aggregateDDRReadBytes.isKnown());
  EXPECT_EQ(cost.aggregateDDRReadBytes.value, 16u);
  ASSERT_TRUE(cost.aggregateSPMMovementBytes.isKnown());
  EXPECT_EQ(cost.aggregateSPMMovementBytes.value, 16u);
  ASSERT_TRUE(cost.aggregateInstructionCount.isKnown());
  EXPECT_EQ(cost.aggregateInstructionCount.value, 3u);
  ASSERT_TRUE(cost.maximumRankDataDependencyDepth.isKnown());
  EXPECT_EQ(cost.maximumRankDataDependencyDepth.value, 2u);
  ASSERT_TRUE(cost.aggregateReadyOrderPriorityInversions.isKnown());
  ASSERT_TRUE(cost.aggregateCompute.vectorF16Bf16LogicalOps.isKnown());
  EXPECT_EQ(cost.aggregateCompute.vectorF16Bf16LogicalOps.value, 4u);
  ASSERT_TRUE(cost.maximumRankSPMHighWaterBytes.isKnown());
  EXPECT_EQ(cost.maximumRankSPMHighWaterBytes.value, 264u);
  ASSERT_TRUE(cost.summedRankSPMHighWaterBytes.isKnown());
  EXPECT_EQ(cost.summedRankSPMHighWaterBytes.value, 272u);
}

TEST_F(ScheduleCostAnalysisTest,
       WholeCardAggregationPreservesUnknownCostReason) {
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
  auto cost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      roots, wafer::analysis::getTargetScheduleCostPolicy());
  EXPECT_EQ(cost.aggregateDDRReadBytes.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.aggregateDDRReadBytes.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_TRUE(cost.maximumRankSPMHighWaterBytes.isKnown());
  EXPECT_EQ(cost.maximumRankDataDependencyDepth.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.maximumRankDataDependencyDepth.reason,
            ScheduleCostReason::UnsupportedControlFlow);
  EXPECT_EQ(cost.aggregateReadyOrderPriorityInversions.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.aggregateReadyOrderPriorityInversions.reason,
            ScheduleCostReason::UnsupportedControlFlow);
}

TEST_F(ScheduleCostAnalysisTest,
       WholeCardMinimumHopDemandDistinguishesEqualPayloadEdges) {
  constexpr llvm::StringLiteral nearSend = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 1, phase = collective_permute, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";
  constexpr llvm::StringLiteral farSend = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 3 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 1, phase = collective_permute, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 4> nearOwners;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 4> farOwners;
  llvm::SmallVector<mlir::Operation *, 4> nearRoots;
  llvm::SmallVector<mlir::Operation *, 4> farRoots;
  for (int64_t rank = 0; rank < 4; ++rank) {
    nearOwners.push_back(
        makeDTERankModule(4, 4, "", rank == 0 ? nearSend : ""));
    farOwners.push_back(makeDTERankModule(4, 4, "", rank == 0 ? farSend : ""));
    ASSERT_TRUE(nearOwners.back());
    ASSERT_TRUE(farOwners.back());
    nearRoots.push_back(nearOwners.back()->getOperation());
    farRoots.push_back(farOwners.back()->getOperation());
  }

  const auto policy = wafer::analysis::getTargetScheduleCostPolicy();
  auto nearCost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      nearRoots, policy);
  auto farCost =
      wafer::analysis::analyzeWholeCardInstructionProgramCost(farRoots, policy);
  ASSERT_TRUE(nearCost.aggregateNoC.aggregateTransmitBytes.isKnown());
  ASSERT_TRUE(farCost.aggregateNoC.aggregateTransmitBytes.isKnown());
  EXPECT_EQ(nearCost.aggregateNoC.aggregateTransmitBytes.value, 16u);
  EXPECT_EQ(farCost.aggregateNoC.aggregateTransmitBytes.value, 16u);
  ASSERT_TRUE(nearCost.minimumHopLinkByteDemand.isKnown());
  ASSERT_TRUE(farCost.minimumHopLinkByteDemand.isKnown());
  EXPECT_EQ(nearCost.minimumHopLinkByteDemand.value, 16u);
  EXPECT_EQ(farCost.minimumHopLinkByteDemand.value, 48u);
  ASSERT_TRUE(nearCost.minimumHopMessageDemand.isKnown());
  ASSERT_TRUE(farCost.minimumHopMessageDemand.isKnown());
  EXPECT_EQ(nearCost.minimumHopMessageDemand.value, 1u);
  EXPECT_EQ(farCost.minimumHopMessageDemand.value, 3u);
  ASSERT_TRUE(nearCost.directedNoCLinkCount.isKnown());
  ASSERT_TRUE(farCost.directedNoCLinkCount.isKnown());
  EXPECT_EQ(nearCost.directedNoCLinkCount.value, 6u);
  EXPECT_EQ(farCost.directedNoCLinkCount.value, 6u);
  ASSERT_TRUE(nearCost.idealizedMinimumPeakLinkByteDemand.isKnown());
  ASSERT_TRUE(farCost.idealizedMinimumPeakLinkByteDemand.isKnown());
  EXPECT_EQ(nearCost.idealizedMinimumPeakLinkByteDemand.value, 3u);
  EXPECT_EQ(farCost.idealizedMinimumPeakLinkByteDemand.value, 8u);
  ASSERT_TRUE(nearCost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown());
  ASSERT_TRUE(farCost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown());
  EXPECT_EQ(nearCost.modeledNoCRoute.kind,
            wafer::analysis::ModeledNoCRouteKind::CanonicalShortestPath);
  EXPECT_EQ(nearCost.modeledNoCRoute.peakDirectedLinkByteDemand.value, 16u);
  EXPECT_EQ(farCost.modeledNoCRoute.peakDirectedLinkByteDemand.value, 16u);
  ASSERT_TRUE(nearCost.maximumNoCHopCount.isKnown());
  ASSERT_TRUE(farCost.maximumNoCHopCount.isKnown());
  EXPECT_EQ(nearCost.maximumNoCHopCount.value, 1u);
  EXPECT_EQ(farCost.maximumNoCHopCount.value, 3u);
}

TEST_F(ScheduleCostAnalysisTest,
       WholeCardNoCFreeProgramKeepsTopologyFactsSeparateFromZeroTraffic) {
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 4> owners;
  llvm::SmallVector<mlir::Operation *, 4> roots;
  for (int64_t rank = 0; rank < 4; ++rank) {
    owners.push_back(makeDTERankModule(4, 4, "", ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }
  const auto policy = wafer::analysis::getTargetScheduleCostPolicy();
  auto cost =
      wafer::analysis::analyzeWholeCardInstructionProgramCost(roots, policy);
  ASSERT_TRUE(cost.directedNoCLinkCount.isKnown());
  EXPECT_EQ(cost.directedNoCLinkCount.value, 6u);
  ASSERT_TRUE(cost.minimumHopLinkByteDemand.isKnown());
  EXPECT_EQ(cost.minimumHopLinkByteDemand.value, 0u);
  ASSERT_TRUE(cost.minimumHopMessageDemand.isKnown());
  EXPECT_EQ(cost.minimumHopMessageDemand.value, 0u);
  ASSERT_TRUE(cost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown());
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.value, 0u);

  auto missing0 = makeDTERankModule(2, 2, "", "", false);
  auto missing1 = makeDTERankModule(2, 2, "", "", false);
  ASSERT_TRUE(missing0);
  ASSERT_TRUE(missing1);
  llvm::SmallVector<mlir::Operation *, 2> missingRoots{
      missing0->getOperation(), missing1->getOperation()};
  auto missingCost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      missingRoots, policy);
  EXPECT_EQ(missingCost.directedNoCLinkCount.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(missingCost.directedNoCLinkCount.reason,
            ScheduleCostReason::InvalidExecutionTopology);
  ASSERT_TRUE(missingCost.minimumHopLinkByteDemand.isKnown());
  EXPECT_EQ(missingCost.minimumHopLinkByteDemand.value, 0u);
  ASSERT_TRUE(missingCost.minimumHopMessageDemand.isKnown());
  EXPECT_EQ(missingCost.minimumHopMessageDemand.value, 0u);
}

TEST_F(ScheduleCostAnalysisTest,
       WholeCardMinimumHopDemandUsesStaticLoopMultiplicity) {
  constexpr llvm::StringLiteral repeatedSend = R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c3 step %c1 {
      %sent = wafer.instr.dte_send %buffer
          {peer = 2 : i64, bytes = 4 : i64,
           message = #wafer.dte_message<communication = 2, phase = collective_permute, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    }
)mlir";
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 3> owners;
  llvm::SmallVector<mlir::Operation *, 3> roots;
  for (int64_t rank = 0; rank < 3; ++rank) {
    owners.push_back(
        makeDTERankModule(3, 3, "", rank == 0 ? repeatedSend : ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }
  auto cost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      roots, wafer::analysis::getTargetScheduleCostPolicy());
  ASSERT_TRUE(cost.minimumHopLinkByteDemand.isKnown());
  EXPECT_EQ(cost.minimumHopLinkByteDemand.value, 24u);
  ASSERT_TRUE(cost.minimumHopMessageDemand.isKnown());
  EXPECT_EQ(cost.minimumHopMessageDemand.value, 6u);
  ASSERT_TRUE(cost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown());
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.value, 12u);
  ASSERT_TRUE(cost.maximumNoCHopCount.isKnown());
  EXPECT_EQ(cost.maximumNoCHopCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest,
       EndpointPressureDistinguishesEqualMeshLinkByteDemand) {
  constexpr llvm::StringLiteral sendLeft = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 1, phase = peer_dataflow, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";
  constexpr llvm::StringLiteral sendRight = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 2 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 2, phase = peer_dataflow, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";
  constexpr llvm::StringLiteral sendBoth = R"mlir(
    %left = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 1, phase = peer_dataflow, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    %right = wafer.instr.dte_send %buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 2, phase = peer_dataflow, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 4> balancedOwners;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 4> hotspotOwners;
  llvm::SmallVector<mlir::Operation *, 4> balancedRoots;
  llvm::SmallVector<mlir::Operation *, 4> hotspotRoots;
  for (int64_t rank = 0; rank < 4; ++rank) {
    llvm::StringRef balancedBody = rank == 1   ? sendLeft
                                   : rank == 3 ? sendRight
                                               : "";
    llvm::StringRef hotspotBody = rank == 1 ? sendBoth : "";
    balancedOwners.push_back(makeDTERankModule(4, 4, "", balancedBody));
    hotspotOwners.push_back(makeDTERankModule(4, 4, "", hotspotBody));
    ASSERT_TRUE(balancedOwners.back());
    ASSERT_TRUE(hotspotOwners.back());
    balancedRoots.push_back(balancedOwners.back()->getOperation());
    hotspotRoots.push_back(hotspotOwners.back()->getOperation());
  }

  const auto policy = wafer::analysis::getTargetScheduleCostPolicy();
  auto balanced = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      balancedRoots, policy);
  auto hotspot = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      hotspotRoots, policy);
  ASSERT_TRUE(balanced.aggregateNoC.aggregateTransmitBytes.isKnown());
  ASSERT_TRUE(hotspot.aggregateNoC.aggregateTransmitBytes.isKnown());
  EXPECT_EQ(balanced.aggregateNoC.aggregateTransmitBytes.value, 32u);
  EXPECT_EQ(hotspot.aggregateNoC.aggregateTransmitBytes.value, 32u);
  ASSERT_TRUE(balanced.minimumHopLinkByteDemand.isKnown());
  ASSERT_TRUE(hotspot.minimumHopLinkByteDemand.isKnown());
  EXPECT_EQ(balanced.minimumHopLinkByteDemand.value, 32u);
  EXPECT_EQ(hotspot.minimumHopLinkByteDemand.value, 32u);
  ASSERT_TRUE(balanced.idealizedMinimumPeakLinkByteDemand.isKnown());
  ASSERT_TRUE(hotspot.idealizedMinimumPeakLinkByteDemand.isKnown());
  EXPECT_EQ(balanced.idealizedMinimumPeakLinkByteDemand.value,
            hotspot.idealizedMinimumPeakLinkByteDemand.value);
  ASSERT_TRUE(balanced.maximumRankNoCTransmitBytes.isKnown());
  ASSERT_TRUE(hotspot.maximumRankNoCTransmitBytes.isKnown());
  EXPECT_EQ(balanced.maximumRankNoCTransmitBytes.value, 16u);
  EXPECT_EQ(hotspot.maximumRankNoCTransmitBytes.value, 32u);
  ASSERT_TRUE(balanced.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown());
  ASSERT_TRUE(hotspot.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown());
  EXPECT_EQ(balanced.modeledNoCRoute.peakDirectedLinkByteDemand.value, 16u);
  EXPECT_EQ(hotspot.modeledNoCRoute.peakDirectedLinkByteDemand.value, 32u);
}

TEST_F(ScheduleCostAnalysisTest,
       PeerDataflowHasExactP2PTrafficWithoutCollectiveAttribution) {
  constexpr llvm::StringLiteral send = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 2 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 17, phase = peer_dataflow, round = 0, slice = 3>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %sent : !async.token
)mlir";
  constexpr llvm::StringLiteral recv = R"mlir(
    %received = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 17, phase = peer_dataflow, round = 0, slice = 3>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %received : !async.token
)mlir";

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 3> owners;
  llvm::SmallVector<mlir::Operation *, 3> roots;
  for (int64_t rank = 0; rank < 3; ++rank) {
    owners.push_back(makeDTERankModule(3, 3, "",
                                       rank == 0   ? send
                                       : rank == 2 ? recv
                                                   : ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }

  auto cost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      roots, wafer::analysis::getTargetScheduleCostPolicy());
  ASSERT_TRUE(cost.aggregateNoC.aggregateTransmitBytes.isKnown());
  EXPECT_EQ(cost.aggregateNoC.aggregateTransmitBytes.value, 8u);
  ASSERT_TRUE(cost.aggregateNoC.aggregateReceiveBytes.isKnown());
  EXPECT_EQ(cost.aggregateNoC.aggregateReceiveBytes.value, 8u);
  EXPECT_EQ(cost.aggregateNoC.staticIssueSiteCount.value, 2u);
  EXPECT_EQ(cost.aggregateNoC.transmitMessageCount.value, 1u);
  EXPECT_EQ(cost.aggregateNoC.receiveMessageCount.value, 1u);
  EXPECT_EQ(cost.aggregateNoC.waitOperationCount.value, 2u);
  EXPECT_EQ(cost.aggregateNoC.waitedEventCount.value, 2u);
  EXPECT_EQ(cost.maximumRankNoCTransmitBytes.value, 8u);
  EXPECT_EQ(cost.maximumRankNoCReceiveBytes.value, 8u);
  EXPECT_EQ(cost.maximumRankNoCTransmitMessageCount.value, 1u);
  EXPECT_EQ(cost.maximumRankNoCReceiveMessageCount.value, 1u);
  ASSERT_TRUE(cost.minimumHopLinkByteDemand.isKnown());
  EXPECT_EQ(cost.minimumHopLinkByteDemand.value, 16u);
  ASSERT_TRUE(cost.minimumHopMessageDemand.isKnown());
  EXPECT_EQ(cost.minimumHopMessageDemand.value, 2u);
  ASSERT_TRUE(cost.aggregateEventCount.isKnown());
  EXPECT_EQ(cost.aggregateEventCount.value, 2u);
  ASSERT_TRUE(cost.aggregateInstructionCount.isKnown());
  EXPECT_EQ(cost.aggregateInstructionCount.value, 4u);
  for (const auto &collective : cost.aggregateNoC.collectiveTransmitBytes) {
    ASSERT_TRUE(collective.isKnown());
    EXPECT_EQ(collective.value, 0u);
  }
}

TEST_F(ScheduleCostAnalysisTest,
       WholeCardMinimumHopDemandPropagatesArithmeticOverflow) {
  constexpr llvm::StringLiteral overflowingSend = R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cmax = arith.constant 9223372036854775807 : index
    scf.for %i = %c0 to %cmax step %c1 {
      %sent = wafer.instr.dte_send %buffer
          {peer = 2 : i64, bytes = 2 : i64,
           message = #wafer.dte_message<communication = 3, phase = collective_permute, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    }
)mlir";
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 3> owners;
  llvm::SmallVector<mlir::Operation *, 3> roots;
  for (int64_t rank = 0; rank < 3; ++rank) {
    owners.push_back(
        makeDTERankModule(3, 3, "", rank == 0 ? overflowingSend : ""));
    ASSERT_TRUE(owners.back());
    roots.push_back(owners.back()->getOperation());
  }
  auto cost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      roots, wafer::analysis::getTargetScheduleCostPolicy());
  EXPECT_EQ(cost.minimumHopLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.reason,
            ScheduleCostReason::ArithmeticOverflow);
  ASSERT_TRUE(cost.modeledNoCRoute.peakDirectedLinkByteDemand.isKnown());
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.value,
            std::numeric_limits<uint64_t>::max() - 1);
  ASSERT_TRUE(cost.maximumNoCHopCount.isKnown());
  EXPECT_EQ(cost.maximumNoCHopCount.value, 2u);
}

TEST_F(ScheduleCostAnalysisTest,
       WholeCardModeledRoutePropagatesSharedLinkOverflow) {
  constexpr llvm::StringLiteral overflowingSharedLink = R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cmax = arith.constant 9223372036854775807 : index
    scf.for %i = %c0 to %cmax step %c1 {
      %first = wafer.instr.dte_send %buffer
          {peer = 1 : i64, bytes = 2 : i64,
           message = #wafer.dte_message<communication = 31, phase = peer_dataflow, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
      %second = wafer.instr.dte_send %buffer
          {peer = 1 : i64, bytes = 2 : i64,
           message = #wafer.dte_message<communication = 32, phase = peer_dataflow, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    }
)mlir";
  auto rank0 = makeDTERankModule(2, 2, "", overflowingSharedLink);
  auto rank1 = makeDTERankModule(2, 2, "", "");
  ASSERT_TRUE(rank0);
  ASSERT_TRUE(rank1);
  llvm::SmallVector<mlir::Operation *, 2> roots{rank0->getOperation(),
                                                rank1->getOperation()};
  auto cost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      roots, wafer::analysis::getTargetScheduleCostPolicy());
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Overflow);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.reason,
            ScheduleCostReason::ArithmeticOverflow);
  ASSERT_TRUE(cost.minimumHopMessageDemand.isKnown());
  EXPECT_EQ(cost.minimumHopMessageDemand.value,
            std::numeric_limits<uint64_t>::max() - 1);
  ASSERT_TRUE(cost.maximumNoCHopCount.isKnown());
  EXPECT_EQ(cost.maximumNoCHopCount.value, 1u);
}

TEST_F(ScheduleCostAnalysisTest,
       WholeCardMinimumHopDemandFailsClosedWithoutTypedTopology) {
  constexpr llvm::StringLiteral send = R"mlir(
    %sent = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 4 : i64,
         message = #wafer.dte_message<communication = 4, phase = collective_permute, round = 0, slice = 0>}
        : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
)mlir";
  auto rank0 = makeDTERankModule(2, 2, "", send, false);
  auto rank1 = makeDTERankModule(2, 2, "", "", false);
  ASSERT_TRUE(rank0);
  ASSERT_TRUE(rank1);
  llvm::SmallVector<mlir::Operation *, 2> roots{rank0->getOperation(),
                                                rank1->getOperation()};
  auto cost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      roots, wafer::analysis::getTargetScheduleCostPolicy());
  EXPECT_EQ(cost.minimumHopLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.reason,
            ScheduleCostReason::InvalidExecutionTopology);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.reason,
            ScheduleCostReason::InvalidExecutionTopology);
  EXPECT_EQ(cost.maximumNoCHopCount.knowledge, ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.maximumNoCHopCount.reason,
            ScheduleCostReason::InvalidExecutionTopology);
}

TEST_F(ScheduleCostAnalysisTest,
       WholeCardMinimumHopDemandPreservesDynamicMultiplicityUnknown) {
  constexpr llvm::StringLiteral dynamicSend = R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %sent = wafer.instr.dte_send %buffer
          {peer = 1 : i64, bytes = 4 : i64,
           message = #wafer.dte_message<communication = 5, phase = collective_permute, round = 0, slice = 0>}
          : memref<16xi8, #wafer.memory<spm, tensor>> -> !async.token
    }
)mlir";
  auto rank0 = makeDTERankModule(2, 2, "%n: index", dynamicSend);
  auto rank1 = makeDTERankModule(2, 2, "", "");
  ASSERT_TRUE(rank0);
  ASSERT_TRUE(rank1);
  llvm::SmallVector<mlir::Operation *, 2> roots{rank0->getOperation(),
                                                rank1->getOperation()};
  auto cost = wafer::analysis::analyzeWholeCardInstructionProgramCost(
      roots, wafer::analysis::getTargetScheduleCostPolicy());
  EXPECT_EQ(cost.minimumHopLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.minimumHopLinkByteDemand.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_EQ(cost.minimumHopMessageDemand.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.minimumHopMessageDemand.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.modeledNoCRoute.peakDirectedLinkByteDemand.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  ASSERT_TRUE(cost.maximumNoCHopCount.isKnown());
  EXPECT_EQ(cost.maximumNoCHopCount.value, 1u);
}

} // namespace
