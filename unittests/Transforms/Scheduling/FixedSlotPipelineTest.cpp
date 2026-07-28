//===- FixedSlotPipelineTest.cpp - Static slot pipeline tests ------------===//

#include "Wafer/Transforms/SoftwarePipelining.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <string>

namespace {

class FixedSlotPipelineTest : public ::testing::Test {
protected:
  FixedSlotPipelineTest() {
    registry.insert<mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
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

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  static llvm::SmallVector<mlir::scf::ForOp, 4>
  collectLoops(mlir::ModuleOp module) {
    llvm::SmallVector<mlir::scf::ForOp, 4> loops;
    module.walk([&](mlir::scf::ForOp loop) { loops.push_back(loop); });
    return loops;
  }

  static std::string threeStageSource(unsigned tripCount,
                                      bool preserveIterArg = false,
                                      unsigned elements = 4) {
    const std::string elementCount = std::to_string(elements);
    const std::string byteCount = std::to_string(elements * 2U);
    std::string source = R"mlir(
module {
  func.func @pipeline() {
    %input = memref.alloc() : memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<ddr, tensor>>
    %output = memref.alloc() : memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<ddr, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %upper = arith.constant )mlir";
    source += std::to_string(tripCount);
    source += R"mlir( : index
)mlir";
    if (preserveIterArg)
      source += R"mlir(    %loop_result = scf.for %iv = %c0 to %upper step %c1
        iter_args(%carried = %c0) -> (index) {
)mlir";
    else
      source += R"mlir(    scf.for %iv = %c0 to %upper step %c1 {
)mlir";
    source += R"mlir(      %loaded = memref.alloc() : memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<spm, tensor>>
      %computed = memref.alloc() : memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %loaded
          {byte_count = )mlir";
    source += byteCount;
    source += R"mlir( : i64, inner_bytes = )mlir";
    source += byteCount;
    source += R"mlir( : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<ddr, tensor>>
         to memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<spm, tensor>>
      %scheduled = arith.addi %iv, %c1
          {wafer.pipeline_test_marker = true} : index
      wafer.instr.elementwise <add> %loaded, %loaded into %computed
          : memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<spm, tensor>>,
            memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<spm, tensor>>
        into memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %computed to %output
          {byte_count = )mlir";
    source += byteCount;
    source += R"mlir( : i64, inner_bytes = )mlir";
    source += byteCount;
    source += R"mlir( : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<spm, tensor>>
         to memref<)mlir";
    source += elementCount;
    source += R"mlir(xf16, #wafer.memory<ddr, tensor>>
)mlir";
    if (preserveIterArg)
      source += R"mlir(      %next = arith.addi %carried, %c1 : index
      scf.yield %next : index
    }
    %used = arith.index_cast %loop_result : index to i64
)mlir";
    else
      source += R"mlir(      scf.yield
    }
)mlir";
    source += R"mlir(    return
  }
}
)mlir";
    return source;
  }

  static std::string twoStageSource(unsigned tripCount) {
    std::string source = R"mlir(
module {
  func.func @pipeline_two_stage() {
    %input = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %upper = arith.constant )mlir";
    source += std::to_string(tripCount);
    source += R"mlir( : index
    scf.for %iv = %c0 to %upper step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %slot
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    return
  }
}
)mlir";
    return source;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(FixedSlotPipelineTest, SingleIterationReturnsIndependentIdentity) {
  auto source = parse(threeStageSource(/*tripCount=*/1));
  ASSERT_TRUE(source);
  mlir::scf::ForOp loop = collectLoops(*source).front();
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, loop, &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_EQ(candidate->stageCount, 1u);
  EXPECT_EQ(candidate->slotAllocationCount, 0u);
  EXPECT_NE(candidate->module->getOperation(), source->getOperation());
  EXPECT_EQ(print(candidate->module->getOperation()), before);
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest, TwoIterationTwoStagePipelineMaterializesSlots) {
  auto source = parse(twoStageSource(/*tripCount=*/2));
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_EQ(candidate->stageCount, 2u);
  EXPECT_EQ(candidate->slotAllocationCount, 2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*candidate->module)));
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest,
       BlockArgumentDDRAndLoopLocalSPMAreProvenDistinct) {
  auto source = parse(R"mlir(
module {
  func.func @block_argument_input(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %slot
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_EQ(candidate->stageCount, 2u);
  EXPECT_EQ(candidate->slotAllocationCount, 2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*candidate->module)));
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest,
       FreshAllocationIsDistinctFromFunctionTensorAdapter) {
  auto source = parse(R"mlir(
module {
  func.func @tensor_adapter(%input: tensor<4xf16>) {
    %adapted = bufferization.to_memref %input read_only
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %output = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %loaded = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      %computed = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %adapted to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %loaded, %loaded into %computed
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %computed to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_EQ(candidate->stageCount, 3u);
  EXPECT_EQ(candidate->slotAllocationCount, 4u);
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest,
       ThreeStageSingleLifetimeMaterializesThreeRotatingSlots) {
  auto source = parse(R"mlir(
module {
  func.func @three_slot_lifetime() {
    %input = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %output = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %slot
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %slot to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_EQ(candidate->stageCount, 3u);
  EXPECT_EQ(candidate->slotAllocationCount, 3u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*candidate->module)));
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest,
       TileRegionSlotsPassFreshSPMPlanningWithDistinctOffsets) {
  auto source = parse(R"mlir(
module {
  func.func @tile_region_pipeline() {
    %input = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %output = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %result = wafer.tile.region(
        %input, %output
        : memref<4xf16, #wafer.memory<ddr, tensor>>,
          memref<4xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%source: memref<4xf16, #wafer.memory<ddr, tensor>>,
         %dest: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %loop_result = scf.for %iv = %c0 to %c4 step %c1
          iter_args(%carried = %dest)
          -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
        %loaded = memref.alloc()
            : memref<4xf16, #wafer.memory<spm, tensor>>
        %computed = memref.alloc()
            : memref<4xf16, #wafer.memory<spm, tensor>>
        wafer.instr.rdma %source to %loaded
            {byte_count = 8 : i64, inner_bytes = 8 : i64,
             src_iterations = array<i64: 1, 1, 1>,
             src_strides = array<i64: 0, 0, 0>}
            : memref<4xf16, #wafer.memory<ddr, tensor>>
           to memref<4xf16, #wafer.memory<spm, tensor>>
        wafer.instr.elementwise <add> %loaded, %loaded into %computed
            : memref<4xf16, #wafer.memory<spm, tensor>>,
              memref<4xf16, #wafer.memory<spm, tensor>>
          into memref<4xf16, #wafer.memory<spm, tensor>>
        wafer.instr.wdma %computed to %carried
            {byte_count = 8 : i64, inner_bytes = 8 : i64,
             dst_iterations = array<i64: 1, 1, 1>,
             dst_strides = array<i64: 0, 0, 0>}
            : memref<4xf16, #wafer.memory<spm, tensor>>
           to memref<4xf16, #wafer.memory<ddr, tensor>>
        scf.yield %carried
            : memref<4xf16, #wafer.memory<ddr, tensor>>
      }
      wafer.tile.yield %loop_result
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.ncc_join [0]
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_EQ(candidate->stageCount, 3u);
  EXPECT_EQ(candidate->slotAllocationCount, 4u);
  EXPECT_EQ(print(source->getOperation()), before);

  mlir::LogicalResult planned = wafer::planSPMMemoryModule(
      *candidate->module, /*spmBase=*/65536, /*spmLimit=*/3080192,
      /*spmAlignment=*/256);
  ASSERT_TRUE(mlir::succeeded(planned));
  std::set<int64_t> slotOffsets;
  candidate->module->walk([&](mlir::memref::AllocOp allocation) {
    if (!wafer::isWaferSPMMemRefType(allocation.getType()))
      return;
    auto offset = allocation->getAttrOfType<wafer::SPMOffsetAttr>(
        wafer::kWaferSPMOffsetAttrName);
    ASSERT_TRUE(offset);
    EXPECT_GE(offset.getOffset(), 65536);
    EXPECT_LT(offset.getOffset(), 3080192);
    slotOffsets.insert(offset.getOffset());
  });
  EXPECT_EQ(slotOffsets.size(), 4u);
}

TEST_F(FixedSlotPipelineTest,
       ThreeStagePipelineHandlesThreeFourAndOddFiveTrips) {
  for (unsigned tripCount : {3U, 4U, 5U}) {
    auto source = parse(threeStageSource(tripCount));
    ASSERT_TRUE(source) << "trip count " << tripCount;
    const std::string before = print(source->getOperation());

    std::string failureReason;
    auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
        *source, collectLoops(*source).front(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(candidate))
        << "trip count " << tripCount << ": " << failureReason;
    EXPECT_EQ(candidate->stageCount, 3u);
    EXPECT_EQ(candidate->slotAllocationCount, 4u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*candidate->module)));
    EXPECT_EQ(print(source->getOperation()), before);

    llvm::SmallVector<mlir::scf::ForOp, 4> candidateLoops =
        collectLoops(*candidate->module);
    ASSERT_EQ(candidateLoops.size(), 1u);
    mlir::scf::ForOp kernel = candidateLoops.front();

    unsigned nestedAllocations = 0;
    unsigned externalSPMAllocations = 0;
    candidate->module->walk([&](mlir::memref::AllocOp allocation) {
      if (allocation->getParentOfType<mlir::scf::ForOp>()) {
        ++nestedAllocations;
        return;
      }
      wafer::MemoryAttr memory =
          wafer::getWaferMemoryAttr(allocation.getType());
      if (memory && memory.getSpace() == wafer::MemorySpace::SPM)
        ++externalSPMAllocations;
    });
    EXPECT_EQ(nestedAllocations, 0u);
    EXPECT_EQ(externalSPMAllocations, 4u);

    unsigned kernelIssues = 0;
    kernel.walk([&](wafer::WaferInstructionOpInterface) { ++kernelIssues; });
    EXPECT_EQ(kernelIssues, 3u);

    bool hasPrologueIssue = false;
    bool hasEpilogueIssue = false;
    candidate->module->walk([&](wafer::WaferInstructionOpInterface issue) {
      mlir::Operation *operation = issue.getOperation();
      if (operation->getParentOfType<mlir::scf::ForOp>())
        return;
      ASSERT_EQ(operation->getBlock(), kernel->getBlock());
      hasPrologueIssue |= operation->isBeforeInBlock(kernel);
      hasEpilogueIssue |= kernel->isBeforeInBlock(operation);
    });
    EXPECT_TRUE(hasPrologueIssue);
    EXPECT_TRUE(hasEpilogueIssue);

    bool retainedScheduledPureOp = false;
    candidate->module->walk([&](mlir::arith::AddIOp operation) {
      retainedScheduledPureOp |=
          operation->hasAttr("wafer.pipeline_test_marker");
    });
    EXPECT_TRUE(retainedScheduledPureOp);

    unsigned joins = 0;
    bool loopJoin = false;
    candidate->module->walk([&](wafer::SyncNCCJoinOp join) {
      ++joins;
      loopJoin |= static_cast<bool>(join->getParentOfType<mlir::scf::ForOp>());
    });
    EXPECT_EQ(joins, 1u);
    EXPECT_FALSE(loopJoin);

    auto yield =
        mlir::cast<mlir::scf::YieldOp>(kernel.getBody()->getTerminator());
    bool hasRotatedSlot = false;
    for (auto [index, yielded] : llvm::enumerate(yield.getOperands())) {
      if (index >= kernel.getRegionIterArgs().size())
        break;
      if (yielded != kernel.getRegionIterArgs()[index])
        hasRotatedSlot = true;
    }
    EXPECT_TRUE(hasRotatedSlot);
  }
}

TEST_F(FixedSlotPipelineTest,
       CanonicalStaticKernelKeepsExactWholePipelineMultiplicity) {
  auto source = parse(threeStageSource(/*tripCount=*/5));
  ASSERT_TRUE(source);

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  auto cost = wafer::analysis::analyzeInstructionProgramCost(
      candidate->module->getOperation(),
      wafer::analysis::getTargetScheduleCostPolicy(
          wafer::TargetProfileId::waferTx81SingleCardKernelV1()));

  ASSERT_TRUE(cost.instructionCount.isKnown());
  EXPECT_EQ(cost.instructionCount.value, 16u);
  ASSERT_TRUE(cost.ddrReadBytes.isKnown());
  EXPECT_EQ(cost.ddrReadBytes.value, 40u);
  ASSERT_TRUE(cost.ddrWriteBytes.isKnown());
  EXPECT_EQ(cost.ddrWriteBytes.value, 40u);
  ASSERT_TRUE(cost.compute.vectorF16Bf16LogicalOps.isKnown());
  EXPECT_EQ(cost.compute.vectorF16Bf16LogicalOps.value, 20u);
  ASSERT_TRUE(cost.nccParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.nccParticipantWaitCount.value, 1u);
  ASSERT_TRUE(cost.steadyStateNCCParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.steadyStateNCCParticipantWaitCount.value, 0u);
}

TEST_F(FixedSlotPipelineTest, PreservesExistingIterArgResultAndUse) {
  auto source =
      parse(threeStageSource(/*tripCount=*/4, /*preserveIterArg=*/true));
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*candidate->module)));
  EXPECT_EQ(print(source->getOperation()), before);

  mlir::arith::IndexCastOp use;
  candidate->module->walk(
      [&](mlir::arith::IndexCastOp operation) { use = operation; });
  ASSERT_TRUE(use);
  auto loopResult = mlir::dyn_cast<mlir::OpResult>(use.getIn());
  ASSERT_TRUE(loopResult);
  EXPECT_TRUE(mlir::isa<mlir::scf::ForOp>(loopResult.getOwner()));
}

TEST_F(FixedSlotPipelineTest,
       RejectsThreeStagesForTwoTripsWithoutMutatingSource) {
  auto source = parse(threeStageSource(/*tripCount=*/2));
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failureReason.find("trip count"), std::string::npos);
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest, RejectsDynamicTripCountAtomically) {
  auto source = parse(R"mlir(
module {
  func.func @dynamic(%upper: index) {
    %input = memref.alloc() : memref<4xf16, #wafer.memory<ddr, tensor>>
    %output = memref.alloc() : memref<4xf16, #wafer.memory<ddr, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c0 to %upper step %c1 {
      %loaded = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
      %computed = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %loaded, %loaded into %computed
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %computed to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failureReason.find("static positive trip"), std::string::npos);
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest, RejectsPlacedInputBeforeSlotCloning) {
  auto source = parse(threeStageSource(/*tripCount=*/3));
  ASSERT_TRUE(source);
  mlir::memref::AllocOp placedAllocation;
  source->walk([&](mlir::memref::AllocOp allocation) {
    if (!placedAllocation &&
        wafer::isWaferSPMMemRefType(allocation.getType()))
      placedAllocation = allocation;
  });
  ASSERT_TRUE(placedAllocation);
  placedAllocation->setAttr(
      wafer::kWaferSPMOffsetAttrName,
      wafer::SPMOffsetAttr::get(context.get(), /*offset=*/65536));
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failureReason.find("requires unplaced input"), std::string::npos)
      << failureReason;
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest, RejectsTripSpanThatOverflowsSCFArithmetic) {
  std::string sourceText = twoStageSource(/*tripCount=*/3);
  const std::string lowerNeedle = "%c0 = arith.constant 0 : index";
  const std::string upperNeedle = "%upper = arith.constant 3 : index";
  sourceText.replace(sourceText.find(lowerNeedle), lowerNeedle.size(),
                     "%c0 = arith.constant -9223372036854775808 : index");
  sourceText.replace(sourceText.find(upperNeedle), upperNeedle.size(),
                     "%upper = arith.constant 9223372036854775807 : index");
  auto source = parse(sourceText);
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failureReason.find("static positive trip"), std::string::npos)
      << failureReason;
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest, RejectsNestedLoopAtomically) {
  auto source = parse(R"mlir(
module {
  func.func @nested() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %outer = %c0 to %c3 step %c1 {
      scf.for %inner = %c0 to %c3 step %c1 {
        scf.yield
      }
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  llvm::SmallVector<mlir::scf::ForOp, 4> loops = collectLoops(*source);
  ASSERT_EQ(loops.size(), 2u);
  mlir::scf::ForOp innerLoop = *llvm::find_if(loops, [](mlir::scf::ForOp loop) {
    return static_cast<bool>(loop->getParentOfType<mlir::scf::ForOp>());
  });
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, innerLoop, &failureReason);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failureReason.find("non-nested"), std::string::npos)
      << failureReason;
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest, RejectsUnknownAliasAndCompletionIsland) {
  auto unknownAlias = parse(R"mlir(
module {
  func.func @unknown_alias(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %output: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %loaded = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
      %computed = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %loaded, %loaded into %computed
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %computed to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(unknownAlias);
  std::string before = print(unknownAlias->getOperation());
  std::string failureReason;
  auto aliasCandidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *unknownAlias, collectLoops(*unknownAlias).front(), &failureReason);
  EXPECT_TRUE(mlir::failed(aliasCandidate));
  EXPECT_NE(failureReason.find("prove two accessed roots distinct"),
            std::string::npos);
  EXPECT_EQ(print(unknownAlias->getOperation()), before);

  std::string completionSource = threeStageSource(/*tripCount=*/3);
  const std::string needle = "      wafer.instr.wdma %computed to %output";
  completionSource.insert(completionSource.find(needle),
                          "      wafer.instr.ncc_join [0]\n");
  auto completion = parse(completionSource);
  ASSERT_TRUE(completion);
  before = print(completion->getOperation());
  failureReason.clear();
  auto completionCandidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *completion, collectLoops(*completion).front(), &failureReason);
  EXPECT_TRUE(mlir::failed(completionCandidate));
  EXPECT_EQ(print(completion->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest,
       RejectsUnknownSameSpaceAliasOfExternalAllocation) {
  auto source = parse(R"mlir(
module {
  func.func @unknown_external_alias() {
    %input = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %output = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %storage = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %alias = builtin.unrealized_conversion_cast %storage
        : memref<4xf16, #wafer.memory<spm, tensor>>
          to memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %loaded = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %loaded, %loaded into %storage
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %alias to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failureReason.find("prove two accessed roots distinct"),
            std::string::npos)
      << failureReason;
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest, PreservesPreexistingIdentityMemrefCast) {
  auto source = parse(R"mlir(
module {
  func.func @identity_cast(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %cast = memref.cast %input
        : memref<4xf16, #wafer.memory<ddr, tensor>>
          to memref<4xf16, #wafer.memory<ddr, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %cast to %slot
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  unsigned identityCastCount = 0;
  candidate->module->walk([&](mlir::memref::CastOp) { ++identityCastCount; });
  EXPECT_EQ(identityCastCount, 1u);
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest,
       RejectsUnrotatedExternalCrossStageWriteHazardAtomically) {
  auto source = parse(R"mlir(
module {
  func.func @external_scratch_hazard() {
    %input = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %output = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %scratch = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %loaded = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %loaded, %loaded into %scratch
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %scratch to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failureReason.find("loop-external cross-stage write hazard"),
            std::string::npos)
      << failureReason;
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest, AllowsExternalReadOnlyRootAcrossStages) {
  auto source = parse(R"mlir(
module {
  func.func @external_read_only() {
    %input = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %read_only = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %output = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %loaded = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      %computed = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %loaded, %read_only into %computed
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %read_only to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      scf.yield
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_EQ(candidate->stageCount, 2u);
  EXPECT_GE(candidate->slotAllocationCount, 2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*candidate->module)));
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FixedSlotPipelineTest, LeavesCapacityRejectionToDownstreamPlanner) {
  auto source =
      parse(threeStageSource(/*tripCount=*/3, /*preserveIterArg=*/false,
                             /*elements=*/2'000'000));
  ASSERT_TRUE(source);

  std::string failureReason;
  auto candidate = wafer::deriveStaticFixedSlotPipelineCandidate(
      *source, collectLoops(*source).front(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_EQ(candidate->slotAllocationCount, 4u);
}

} // namespace
