//===- FullBufferHandoffTest.cpp - Cross-task SPM handoff tests ---------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/InitAll.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

#include <memory>

namespace {

class FullBufferHandoffTest : public ::testing::Test {
protected:
  FullBufferHandoffTest() {
    registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
    wafer::registerAllDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseHandoffModule() {
    return parse(R"mlir(
module {
  func.func @full_buffer_handoff(%token: i1) {
    %spill = memref.alloc()
        : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>
    %produced = wafer.tile.region(%spill
        : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<1x2x2xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%output: memref<1x2x2xf16, #wafer.memory<ddr, tensor>>):
      %resident = memref.alloc()
          : memref<1x2x2xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %resident to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<1x2x2xf16, #wafer.memory<spm, tensor>>
         to memref<1x2x2xf16, #wafer.memory<ddr, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %output
          : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>
    }
    %reshaped = memref.collapse_shape %produced [[0, 1], [2]]
        : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>
       into memref<2x2xf16, #wafer.memory<ddr, tensor>>
    %first = wafer.tile.region(%reshaped, %token
        : memref<2x2xf16, #wafer.memory<ddr, tensor>>, i1) -> (i1) {
    ^bb0(%input: memref<2x2xf16, #wafer.memory<ddr, tensor>>, %pass: i1):
      %loaded = memref.alloc()
          : memref<2x2xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<2x2xf16, #wafer.memory<ddr, tensor>>
         to memref<2x2xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %pass : i1
    }
    %second = wafer.tile.region(%reshaped, %token
        : memref<2x2xf16, #wafer.memory<ddr, tensor>>, i1) -> (i1) {
    ^bb0(%input: memref<2x2xf16, #wafer.memory<ddr, tensor>>, %pass: i1):
      %loaded = memref.alloc()
          : memref<2x2xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<2x2xf16, #wafer.memory<ddr, tensor>>
         to memref<2x2xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %pass : i1
    }
    return
  }
}
)mlir");
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseLocalViewHandoffModule() {
    return parse(R"mlir(
module {
  func.func @local_view_handoff(%token: i1) {
    %spill = memref.alloc()
        : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>
    %produced = wafer.tile.region(%spill
        : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<1x2x2xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%output: memref<1x2x2xf16, #wafer.memory<ddr, tensor>>):
      %resident = memref.alloc()
          : memref<1x2x2xf16, #wafer.memory<spm, tensor>>
      %output_view = memref.subview %output[0, 0, 0] [1, 2, 2] [1, 1, 1]
          : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>
         to memref<1x2x2xf16, strided<[4, 2, 1]>,
                   #wafer.memory<ddr, tensor>>
      wafer.instr.wdma %resident to %output_view
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<1x2x2xf16, #wafer.memory<spm, tensor>>
         to memref<1x2x2xf16, strided<[4, 2, 1]>,
                   #wafer.memory<ddr, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %output
          : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>
    }
    %result = wafer.tile.region(%produced, %token
        : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>, i1) -> (i1) {
    ^bb0(%input: memref<1x2x2xf16, #wafer.memory<ddr, tensor>>, %pass: i1):
      %collapsed = memref.collapse_shape %input [[0, 1], [2]]
          : memref<1x2x2xf16, #wafer.memory<ddr, tensor>>
         into memref<2x2xf16, #wafer.memory<ddr, tensor>>
      %cast = memref.cast %collapsed
          : memref<2x2xf16, #wafer.memory<ddr, tensor>>
         to memref<2x2xf16, strided<[2, 1]>,
                   #wafer.memory<ddr, tensor>>
      %loaded = memref.alloc()
          : memref<2x2xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %cast to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<2x2xf16, strided<[2, 1]>,
                   #wafer.memory<ddr, tensor>>
         to memref<2x2xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %pass : i1
    }
    return
  }
}
)mlir");
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseExternalOutputHandoffModule() {
    return parse(R"mlir(
module {
  func.func @external_output_handoff(
      %spill: memref<4xf16, #wafer.memory<ddr, tensor>>, %token: i1) {
    %produced = wafer.tile.region(%spill
        : memref<4xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%output: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %resident = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %resident to %output
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %output
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    %result = wafer.tile.region(%produced, %token
        : memref<4xf16, #wafer.memory<ddr, tensor>>, i1) -> (i1) {
    ^bb0(%input: memref<4xf16, #wafer.memory<ddr, tensor>>, %pass: i1):
      %loaded = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %loaded
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %pass : i1
    }
    return
  }
}
)mlir");
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseNonTensorHandoffModule() {
    return parse(R"mlir(
module {
  func.func @non_tensor_layout_handoff(%token: i1) {
    %spill = memref.alloc()
        : memref<4x8xf16, #wafer.memory<ddr, cx>>
    %produced = wafer.tile.region(%spill
        : memref<4x8xf16, #wafer.memory<ddr, cx>>)
        -> (memref<4x8xf16, #wafer.memory<ddr, cx>>) {
    ^bb0(%output: memref<4x8xf16, #wafer.memory<ddr, cx>>):
      %resident = memref.alloc()
          : memref<4x8xf16, #wafer.memory<spm, cx>>
      wafer.instr.wdma %resident to %output
          {byte_count = 256 : i64, inner_bytes = 256 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<4x8xf16, #wafer.memory<spm, cx>>
         to memref<4x8xf16, #wafer.memory<ddr, cx>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %output
          : memref<4x8xf16, #wafer.memory<ddr, cx>>
    }
    %result = wafer.tile.region(%produced, %token
        : memref<4x8xf16, #wafer.memory<ddr, cx>>, i1) -> (i1) {
    ^bb0(%input: memref<4x8xf16, #wafer.memory<ddr, cx>>, %pass: i1):
      %loaded = memref.alloc()
          : memref<4x8xf16, #wafer.memory<spm, cx>>
      wafer.instr.rdma %input to %loaded
          {byte_count = 256 : i64, inner_bytes = 256 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4x8xf16, #wafer.memory<ddr, cx>>
         to memref<4x8xf16, #wafer.memory<spm, cx>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %pass : i1
    }
    return
  }
}
)mlir");
  }

  template <typename OpTy, typename RootTy>
  static llvm::SmallVector<OpTy, 4> collectOps(RootTy root) {
    llvm::SmallVector<OpTy, 4> operations;
    root.walk([&](OpTy operation) { operations.push_back(operation); });
    return operations;
  }

  static void
  expectUnmodifiedSpillEdge(mlir::ModuleOp module, mlir::Value originalSpill,
                            wafer::InstrWDMAOp originalWDMA,
                            llvm::ArrayRef<wafer::InstrRDMAOp> originalRDMAs,
                            bool expectCompilerOwnedSpill) {
    llvm::SmallVector<wafer::TileRegionOp, 4> regions =
        collectOps<wafer::TileRegionOp>(module);
    ASSERT_GE(regions.size(), 2u);
    ASSERT_FALSE(regions.front().getInputs().empty());
    mlir::Value currentSpill = regions.front().getInputs().front();
    EXPECT_EQ(currentSpill, originalSpill);
    EXPECT_TRUE(wafer::isWaferDDRMemRefType(currentSpill.getType()));
    EXPECT_TRUE(
        wafer::isWaferDDRMemRefType(regions.front().getResult(0).getType()));
    if (expectCompilerOwnedSpill)
      EXPECT_TRUE(currentSpill.getDefiningOp<mlir::memref::AllocOp>());
    else
      EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(currentSpill));

    llvm::SmallVector<wafer::InstrWDMAOp, 4> wdmas =
        collectOps<wafer::InstrWDMAOp>(module);
    ASSERT_EQ(wdmas.size(), 1u);
    EXPECT_EQ(wdmas.front().getOperation(), originalWDMA.getOperation());
    llvm::SmallVector<wafer::InstrRDMAOp, 4> rdmas =
        collectOps<wafer::InstrRDMAOp>(module);
    ASSERT_EQ(rdmas.size(), originalRDMAs.size());
    for (auto [current, original] : llvm::zip_equal(rdmas, originalRDMAs))
      EXPECT_EQ(current.getOperation(),
                static_cast<mlir::Operation *>(original));
    EXPECT_TRUE(collectOps<wafer::InstrGatherScatterOp>(module).empty());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(FullBufferHandoffTest,
       RejectsStaticReshapeFanoutAcrossTileResidencyDomains) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseHandoffModule();
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  llvm::SmallVector<wafer::TileRegionOp, 4> regions =
      collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 3u);
  mlir::Value spill = regions.front().getInputs().front();
  llvm::SmallVector<wafer::InstrWDMAOp, 4> wdmas =
      collectOps<wafer::InstrWDMAOp>(*module);
  llvm::SmallVector<wafer::InstrRDMAOp, 4> rdmas =
      collectOps<wafer::InstrRDMAOp>(*module);
  ASSERT_EQ(wdmas.size(), 1u);
  ASSERT_EQ(rdmas.size(), 2u);

  EXPECT_EQ(
      wafer::tensor_program_scheduling::promoteFullBufferHandoffs(*module), 0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  expectUnmodifiedSpillEdge(*module, spill, wdmas.front(), rdmas,
                            /*expectCompilerOwnedSpill=*/true);
}

TEST_F(FullBufferHandoffTest,
       RejectsCrossRegionHandoffEvenWithCompletedWDMAWorker) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseHandoffModule();
  ASSERT_TRUE(module);
  llvm::SmallVector<wafer::TileRegionOp, 4> regions =
      collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 3u);
  auto fence = *regions.front()
                    .getBody()
                    .front()
                    .getOps<wafer::SyncNCCJoinOp>()
                    .begin();
  mlir::OpBuilder builder(fence);
  builder.create<wafer::SyncNCCJoinOp>(fence.getLoc(),
                                       wafer::NCCWorker::Worker0);
  fence.erase();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(
      wafer::tensor_program_scheduling::promoteFullBufferHandoffs(*module), 0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(collectOps<wafer::InstrWDMAOp>(*module).size(), 1u);
}

TEST_F(FullBufferHandoffTest,
       RejectsCrossRegionHandoffWithEarlierSameWorkerIssue) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseHandoffModule();
  ASSERT_TRUE(module);
  llvm::SmallVector<wafer::TileRegionOp, 4> regions =
      collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 3u);
  wafer::InstrWDMAOp wdma =
      collectOps<wafer::InstrWDMAOp>(regions.front()).front();
  auto fence = *regions.front()
                    .getBody()
                    .front()
                    .getOps<wafer::SyncNCCJoinOp>()
                    .begin();

  mlir::OpBuilder builder(wdma);
  auto zero = builder.create<mlir::arith::ConstantOp>(
      wdma.getLoc(), builder.getF16FloatAttr(0.0));
  builder.create<wafer::InstrFillOp>(wdma.getLoc(), wdma.getSource(),
                                     zero.getResult(), wafer::FillDomainAttr{},
                                     wafer::NCCWorker::Worker0);
  builder.setInsertionPoint(fence);
  builder.create<wafer::SyncNCCJoinOp>(fence.getLoc(),
                                       wafer::NCCWorker::Worker0);
  fence.erase();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(
      wafer::tensor_program_scheduling::promoteFullBufferHandoffs(*module), 0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  regions = collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 3u);
  EXPECT_EQ(collectOps<wafer::InstrWDMAOp>(*module).size(), 1u);
  EXPECT_EQ(collectOps<wafer::InstrFillOp>(regions.front()).size(), 1u);
}

TEST_F(FullBufferHandoffTest,
       RejectsProducerJoinThatDoesNotCompleteTheWDMAWorker) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseHandoffModule();
  ASSERT_TRUE(module);
  llvm::SmallVector<wafer::TileRegionOp, 4> regions =
      collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 3u);
  auto fence = *regions.front()
                    .getBody()
                    .front()
                    .getOps<wafer::SyncNCCJoinOp>()
                    .begin();
  mlir::OpBuilder builder(fence);
  builder.create<wafer::SyncNCCJoinOp>(fence.getLoc(),
                                       wafer::NCCWorker::Worker1);
  fence.erase();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(
      wafer::tensor_program_scheduling::promoteFullBufferHandoffs(*module), 0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(collectOps<wafer::InstrWDMAOp>(*module).size(), 1u);
}

TEST_F(FullBufferHandoffTest, RejectsPartialCrossRegionHandoffSubset) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseHandoffModule();
  ASSERT_TRUE(module);
  llvm::SmallVector<wafer::InstrRDMAOp, 4> rdmas =
      collectOps<wafer::InstrRDMAOp>(*module);
  ASSERT_EQ(rdmas.size(), 2u);
  mlir::Builder builder(context.get());
  rdmas.back()->setAttr("byte_count", builder.getI64IntegerAttr(4));
  rdmas.back()->setAttr("inner_bytes", builder.getI64IntegerAttr(4));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  llvm::SmallVector<wafer::TileRegionOp, 4> regions =
      collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 3u);
  llvm::SmallVector<mlir::memref::CollapseShapeOp, 4> reshapes =
      collectOps<mlir::memref::CollapseShapeOp>(*module);
  ASSERT_EQ(reshapes.size(), 1u);
  mlir::Value externalReshape = reshapes.front().getResult();

  EXPECT_EQ(
      wafer::tensor_program_scheduling::promoteFullBufferHandoffs(*module), 0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  regions = collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 3u);
  EXPECT_TRUE(wafer::isWaferDDRMemRefType(regions[0].getResult(0).getType()));
  EXPECT_EQ(regions[0].getInputs().size(), 1u);
  EXPECT_EQ(regions[0].getBody().front().getNumArguments(), 1u);
  ASSERT_EQ(regions[0].getNumResults(), 1u);
  EXPECT_EQ(regions[1].getInputs()[0], externalReshape);
  EXPECT_EQ(regions[2].getInputs()[0], externalReshape);
  EXPECT_TRUE(wafer::isWaferDDRMemRefType(
      regions[2].getBody().front().getArgument(0).getType()));
  ASSERT_EQ(collectOps<wafer::InstrWDMAOp>(*module).size(), 1u);
  rdmas = collectOps<wafer::InstrRDMAOp>(*module);
  ASSERT_EQ(rdmas.size(), 2u);
  EXPECT_TRUE(llvm::any_of(rdmas, [](wafer::InstrRDMAOp rdma) {
    return rdma.getByteCountAttr().getInt() == 4;
  }));
  EXPECT_TRUE(collectOps<wafer::InstrGatherScatterOp>(*module).empty());
  EXPECT_EQ(collectOps<mlir::memref::CollapseShapeOp>(*module).size(), 1u);
}

TEST_F(FullBufferHandoffTest,
       RejectsCrossRegionProducerSubviewAndConsumerLocalViewChain) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseLocalViewHandoffModule();
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(
      wafer::tensor_program_scheduling::promoteFullBufferHandoffs(*module), 0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  llvm::SmallVector<wafer::TileRegionOp, 4> regions =
      collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_TRUE(wafer::isWaferDDRMemRefType(regions[0].getResult(0).getType()));
  EXPECT_EQ(regions[1].getInputs()[0], regions[0].getResult(0));
  EXPECT_EQ(collectOps<wafer::InstrWDMAOp>(*module).size(), 1u);
  EXPECT_EQ(collectOps<wafer::InstrRDMAOp>(*module).size(), 1u);
  EXPECT_TRUE(collectOps<wafer::InstrGatherScatterOp>(*module).empty());
  EXPECT_EQ(collectOps<mlir::memref::SubViewOp>(*module).size(), 1u);

  llvm::SmallVector<mlir::memref::CollapseShapeOp, 4> collapses =
      collectOps<mlir::memref::CollapseShapeOp>(*module);
  llvm::SmallVector<mlir::memref::CastOp, 4> casts =
      collectOps<mlir::memref::CastOp>(*module);
  ASSERT_EQ(collapses.size(), 1u);
  ASSERT_EQ(casts.size(), 1u);
  EXPECT_TRUE(wafer::isWaferDDRMemRefType(collapses.front().getType()));
  EXPECT_TRUE(wafer::isWaferDDRMemRefType(casts.front().getType()));
}

TEST_F(FullBufferHandoffTest, RejectsOperationAfterProducerWDMA) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseHandoffModule();
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  llvm::SmallVector<wafer::TileRegionOp, 4> regions =
      collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 3u);
  mlir::Value spill = regions.front().getInputs().front();
  llvm::SmallVector<wafer::InstrWDMAOp, 4> wdmas =
      collectOps<wafer::InstrWDMAOp>(*module);
  llvm::SmallVector<wafer::InstrRDMAOp, 4> rdmas =
      collectOps<wafer::InstrRDMAOp>(*module);
  ASSERT_EQ(wdmas.size(), 1u);
  ASSERT_EQ(rdmas.size(), 2u);
  wafer::InstrWDMAOp originalWDMA = wdmas.front();
  size_t allocationCount = collectOps<mlir::memref::AllocOp>(*module).size();

  mlir::OpBuilder builder(originalWDMA);
  builder.setInsertionPointAfter(originalWDMA);
  auto residentType =
      mlir::cast<mlir::MemRefType>(originalWDMA.getSource().getType());
  builder.create<mlir::memref::AllocOp>(originalWDMA.getLoc(), residentType);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(
      wafer::tensor_program_scheduling::promoteFullBufferHandoffs(*module), 0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  expectUnmodifiedSpillEdge(*module, spill, originalWDMA, rdmas,
                            /*expectCompilerOwnedSpill=*/true);
  EXPECT_EQ(collectOps<mlir::memref::AllocOp>(*module).size(),
            allocationCount + 1);
}

TEST_F(FullBufferHandoffTest, RejectsExternalFunctionOutputSpill) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseExternalOutputHandoffModule();
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  llvm::SmallVector<wafer::TileRegionOp, 4> regions =
      collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 2u);
  mlir::Value spill = regions.front().getInputs().front();
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  EXPECT_EQ(spill, function.getArgument(0));
  llvm::SmallVector<wafer::InstrWDMAOp, 4> wdmas =
      collectOps<wafer::InstrWDMAOp>(*module);
  llvm::SmallVector<wafer::InstrRDMAOp, 4> rdmas =
      collectOps<wafer::InstrRDMAOp>(*module);
  ASSERT_EQ(wdmas.size(), 1u);
  ASSERT_EQ(rdmas.size(), 1u);
  wafer::InstrWDMAOp originalWDMA = wdmas.front();

  EXPECT_EQ(
      wafer::tensor_program_scheduling::promoteFullBufferHandoffs(*module), 0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  expectUnmodifiedSpillEdge(*module, spill, originalWDMA, rdmas,
                            /*expectCompilerOwnedSpill=*/false);
  EXPECT_EQ(collectOps<mlir::memref::AllocOp>(*module).size(), 2u);
}

TEST_F(FullBufferHandoffTest, RejectsCompleteNonTensorLayoutEdge) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseNonTensorHandoffModule();
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  llvm::SmallVector<wafer::TileRegionOp, 4> regions =
      collectOps<wafer::TileRegionOp>(*module);
  ASSERT_EQ(regions.size(), 2u);
  mlir::Value spill = regions.front().getInputs().front();
  auto spillType = mlir::cast<mlir::MemRefType>(spill.getType());
  wafer::MemoryAttr spillMemory = wafer::getWaferMemoryAttr(spillType);
  ASSERT_TRUE(spillMemory);
  EXPECT_EQ(spillMemory.getLayout(), wafer::MemLayout::Cx);
  std::optional<wafer::WaferPhysicalTensorInfo> physical =
      wafer::computeWaferPhysicalTensorInfo(spillType);
  ASSERT_TRUE(physical);
  ASSERT_EQ(physical->physicalBytes, 256);

  llvm::SmallVector<wafer::InstrWDMAOp, 4> wdmas =
      collectOps<wafer::InstrWDMAOp>(*module);
  llvm::SmallVector<wafer::InstrRDMAOp, 4> rdmas =
      collectOps<wafer::InstrRDMAOp>(*module);
  ASSERT_EQ(wdmas.size(), 1u);
  ASSERT_EQ(rdmas.size(), 1u);
  wafer::InstrWDMAOp originalWDMA = wdmas.front();
  EXPECT_EQ(originalWDMA.getByteCountAttr().getInt(), physical->physicalBytes);
  EXPECT_EQ(originalWDMA.getInnerBytesAttr().getInt(), physical->physicalBytes);
  EXPECT_EQ(rdmas.front().getByteCountAttr().getInt(), physical->physicalBytes);
  EXPECT_EQ(rdmas.front().getInnerBytesAttr().getInt(),
            physical->physicalBytes);

  EXPECT_EQ(
      wafer::tensor_program_scheduling::promoteFullBufferHandoffs(*module), 0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  expectUnmodifiedSpillEdge(*module, spill, originalWDMA, rdmas,
                            /*expectCompilerOwnedSpill=*/true);
  EXPECT_EQ(
      wafer::getWaferMemoryAttr(
          mlir::cast<mlir::MemRefType>(regions.front().getResult(0).getType()))
          .getLayout(),
      wafer::MemLayout::Cx);
}

} // namespace
