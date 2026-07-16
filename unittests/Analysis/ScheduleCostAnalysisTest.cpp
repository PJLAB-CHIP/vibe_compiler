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
#include "gtest/gtest.h"

#include <memory>
#include <optional>

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
        module, wafer::analysis::getTargetScheduleCostPolicy(
                    wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(ScheduleCostAnalysisTest, ExposesOnlyEstablishedTargetRates) {
  auto policy = wafer::analysis::getTargetScheduleCostPolicy(
      wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  EXPECT_EQ(policy.cardDDRBytesPerSecond, 200'000'000'000ULL);
  EXPECT_EQ(policy.directionalNoCBytesPerSecond, 128'000'000'000ULL);
  EXPECT_EQ(policy.f16Bf16NPULogicalOpsPerSecondPerTile, 8'000'000'000'000ULL);
  EXPECT_EQ(policy.f16Bf16VectorLogicalOpsPerSecondPerTile, 64'000'000'000ULL);
  EXPECT_EQ(policy.f32VectorLogicalOpsPerSecondPerTile, 32'000'000'000ULL);
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
        wafer.instr.local_fence
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
  ASSERT_TRUE(cost.spmHighWaterBytes.isKnown());
  EXPECT_EQ(cost.spmHighWaterBytes.value, 264u);
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
      roots, wafer::analysis::getTargetScheduleCostPolicy(
                 wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
  ASSERT_EQ(cost.rankCosts.size(), 2u);
  ASSERT_TRUE(cost.aggregateDDRReadBytes.isKnown());
  EXPECT_EQ(cost.aggregateDDRReadBytes.value, 16u);
  ASSERT_TRUE(cost.aggregateSPMMovementBytes.isKnown());
  EXPECT_EQ(cost.aggregateSPMMovementBytes.value, 16u);
  ASSERT_TRUE(cost.aggregateInstructionCount.isKnown());
  EXPECT_EQ(cost.aggregateInstructionCount.value, 3u);
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
      roots, wafer::analysis::getTargetScheduleCostPolicy(
                 wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
  EXPECT_EQ(cost.aggregateDDRReadBytes.knowledge,
            ScheduleCostKnowledge::Unknown);
  EXPECT_EQ(cost.aggregateDDRReadBytes.reason,
            ScheduleCostReason::DynamicLoopTripCount);
  EXPECT_TRUE(cost.maximumRankSPMHighWaterBytes.isKnown());
}

} // namespace
