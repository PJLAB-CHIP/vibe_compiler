#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

TEST(ReadyOrderTest, RecomputesMinimumCompletionAfterReordering) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @ready_then_normalize() {
    %compute_dest = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %compute_source = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %dma_dest = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %ddr = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.elementwise <neg> %compute_source into %compute_dest
        : memref<4xf16, #wafer.memory<spm, tensor>>
      into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [1]
    wafer.instr.rdma %ddr to %dma_dest
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  EXPECT_NE(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  ASSERT_TRUE(mlir::succeeded(wafer::placeRequiredNCCJoins(*module)));
  ASSERT_TRUE(mlir::succeeded(wafer::placeRequiredNCCJoins(*module)));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  llvm::SmallVector<wafer::SyncNCCJoinOp, 2> joins;
  module->walk([&](wafer::SyncNCCJoinOp join) { joins.push_back(join); });
  ASSERT_EQ(joins.size(), 1u);
  ASSERT_EQ(joins.front().getParticipants().size(), 1u);
  EXPECT_EQ(joins.front().getParticipants().front(), 0);
  EXPECT_TRUE(mlir::isa<mlir::func::ReturnOp>(joins.front()->getNextNode()));
}

TEST(ReadyOrderTest, MovesIndependentDMABeforeComputeAndPreservesFence) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %compute_dest = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %compute_source = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %dma_dest = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
      %compute_source, %compute_source into %compute_dest
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %ddr to %dma_dest
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
      %compute_source, %compute_source into %compute_dest
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  EXPECT_NE(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  llvm::SmallVector<mlir::Operation *, 4> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface, wafer::SyncNCCJoinOp>(
            operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 4u);
  EXPECT_TRUE(mlir::isa<wafer::InstrRDMAOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(ordered[1]));
  EXPECT_TRUE(mlir::isa<wafer::SyncNCCJoinOp>(ordered[2]));
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(ordered[3]));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(ReadyOrderTest, DifferentWorkerJoinDoesNotBlockReadyWorker) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %compute_dest = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %compute_source = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %dma_dest = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.elementwise #wafer.instr_elementwise_kind<neg>
      %compute_source into %compute_dest
      : memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [1]
  wafer.instr.rdma %ddr to %dma_dest
      {byte_count = 8 : i64, inner_bytes = 8 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4xf16, #wafer.memory<ddr, tensor>>
     to memref<4xf16, #wafer.memory<spm, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_NE(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  llvm::SmallVector<mlir::Operation *, 3> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface, wafer::SyncNCCJoinOp>(
            operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 3u);
  EXPECT_TRUE(mlir::isa<wafer::InstrRDMAOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(ordered[1]));
  EXPECT_TRUE(mlir::isa<wafer::SyncNCCJoinOp>(ordered[2]));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(ReadyOrderTest, MovesDTEAcrossUnrelatedJoinAndSetup) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %buffer = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  %alias = memref.cast %buffer
      : memref<4xf16, #wafer.memory<spm, tensor>>
     to memref<4xf16, strided<[1], offset: ?>,
               #wafer.memory<spm, tensor>>
  %token = wafer.instr.dte_send %alias
      {peer = 0 : i64, bytes = 8 : i64,
       message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
      : memref<4xf16, strided<[1], offset: ?>,
               #wafer.memory<spm, tensor>> -> !async.token
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_NE(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  llvm::SmallVector<mlir::Operation *, 2> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface, wafer::SyncNCCJoinOp>(
            operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 2u);
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::SyncNCCJoinOp>(ordered[1]));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(ReadyOrderTest, PlacesIndependentGemmInsideDTEIssueWaitWindow) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @pipeline() {
    %send_buffer = memref.alloc()
        : memref<8xf16, #wafer.memory<spm, tensor>>
    %lhs = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, cx>>
    %rhs = memref.alloc()
        : memref<3x4xf16, #wafer.memory<spm, cx>>
    %result = memref.alloc()
        : memref<2x4xf16, #wafer.memory<spm, cx>>
    %token = wafer.instr.dte_send %send_buffer
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<8xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    wafer.instr.gemm %lhs, %rhs into %result
        {m = 2 : i64, k = 3 : i64, n = 4 : i64}
        : memref<2x3xf16, #wafer.memory<spm, cx>>,
          memref<3x4xf16, #wafer.memory<spm, cx>>
      into memref<2x4xf16, #wafer.memory<spm, cx>>
    return
  }
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  ASSERT_TRUE(mlir::succeeded(wafer::placeRequiredNCCJoins(*module)));
  EXPECT_NE(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  ASSERT_TRUE(mlir::succeeded(wafer::placeRequiredNCCJoins(*module)));

  llvm::SmallVector<mlir::Operation *, 4> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface,
                  wafer::SyncNCCJoinOp>(operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 4u);
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrGemmOp>(ordered[1]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(ordered[2]));
  EXPECT_TRUE(mlir::isa<wafer::SyncNCCJoinOp>(ordered[3]));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(ReadyOrderTest, PreservesSingleSenderReleaseBeforeNextDTEIssue) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @single_sender() {
    %send0 = memref.alloc()
        : memref<8xf16, #wafer.memory<spm, tensor>>
    %send1 = memref.alloc()
        : memref<8xf16, #wafer.memory<spm, tensor>>
    %lhs = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, cx>>
    %rhs = memref.alloc()
        : memref<3x4xf16, #wafer.memory<spm, cx>>
    %result = memref.alloc()
        : memref<2x4xf16, #wafer.memory<spm, cx>>
    %token0 = wafer.instr.dte_send %send0
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<8xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token0 : !async.token
    %token1 = wafer.instr.dte_send %send1
        {peer = 0 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 1, slice = 0>}
        : memref<8xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token1 : !async.token
    wafer.instr.gemm %lhs, %rhs into %result
        {m = 2 : i64, k = 3 : i64, n = 4 : i64}
        : memref<2x3xf16, #wafer.memory<spm, cx>>,
          memref<3x4xf16, #wafer.memory<spm, cx>>
      into memref<2x4xf16, #wafer.memory<spm, cx>>
    return
  }
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(wafer::placeRequiredNCCJoins(*module)));
  EXPECT_NE(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);

  llvm::SmallVector<mlir::Operation *, 8> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface,
                  wafer::SyncNCCJoinOp>(operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 6u);
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrGemmOp>(ordered[1]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(ordered[2]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(ordered[3]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(ordered[4]));
  EXPECT_TRUE(mlir::isa<wafer::SyncNCCJoinOp>(ordered[5]));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(ReadyOrderTest, ParticipatingJoinOrdersLaterIssueOnSameWorker) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %compute_dest = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %compute_source = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %dma_dest = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.elementwise #wafer.instr_elementwise_kind<neg>
      %compute_source into %compute_dest
      : memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  %ddr_alias = memref.cast %ddr
      : memref<4xf16, #wafer.memory<ddr, tensor>>
     to memref<4xf16, strided<[1], offset: ?>,
               #wafer.memory<ddr, tensor>>
  wafer.instr.rdma %ddr_alias to %dma_dest
      {byte_count = 8 : i64, inner_bytes = 8 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4xf16, strided<[1], offset: ?>,
               #wafer.memory<ddr, tensor>>
     to memref<4xf16, #wafer.memory<spm, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  llvm::SmallVector<mlir::Operation *, 3> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface, wafer::SyncNCCJoinOp>(
            operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 3u);
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::SyncNCCJoinOp>(ordered[1]));
  EXPECT_TRUE(mlir::isa<wafer::InstrRDMAOp>(ordered[2]));
}

TEST(ReadyOrderTest, RetainsValueHazards) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %buffer = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %zero = arith.constant 0.000000e+00 : f16
  wafer.instr.fill %buffer, %zero
      : memref<4x8xf16, #wafer.memory<spm, tensor>>, f16
  wafer.instr.wdma %buffer to %ddr
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<ddr, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  EXPECT_EQ(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
}

TEST(ReadyOrderTest, DoesNotMoveInstructionsAcrossArgWritebackBarrier) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %compute_source = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %compute_dest = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %arg_input = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %arg_value = memref.alloc()
      : memref<1xf16, #wafer.memory<spm, tensor>>
  %arg_index = memref.alloc()
      : memref<1xi32, #wafer.memory<spm, tensor>>
  %dma_dest = memref.alloc()
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.elementwise #wafer.instr_elementwise_kind<neg>
      %compute_source into %compute_dest
      : memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax>
      %arg_input into %arg_value, %arg_index {elem_count = 4 : i64}
      : memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<1xf16, #wafer.memory<spm, tensor>>,
         memref<1xi32, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %ddr to %dma_dest
      {byte_count = 8 : i64, inner_bytes = 8 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>,
       worker = #wafer.ncc_worker<worker1>}
      : memref<4xf16, #wafer.memory<ddr, tensor>>
     to memref<4xf16, #wafer.memory<spm, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  llvm::SmallVector<mlir::Operation *, 3> ordered;
  module->walk([&](wafer::WaferInstructionOpInterface op) {
    ordered.push_back(op.getOperation());
  });
  ASSERT_EQ(ordered.size(), 3u);
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrPeripheralOp>(ordered[1]));
  EXPECT_TRUE(mlir::isa<wafer::InstrRDMAOp>(ordered[2]));
}

TEST(ReadyOrderTest, RetainsHazardsThroughMemrefViews) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %buffer = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %alias = memref.cast %buffer
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, strided<[8, 1], offset: ?>,
               #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %zero = arith.constant 0.000000e+00 : f16
  wafer.instr.fill %buffer, %zero
      : memref<4x8xf16, #wafer.memory<spm, tensor>>, f16
  wafer.instr.wdma %alias to %ddr
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, strided<[8, 1], offset: ?>,
               #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<ddr, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
}

TEST(ReadyOrderTest, MovesAcrossIndependentProductionSetupOperations) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %compute_dest = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %compute_source = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
      %compute_source, %compute_source into %compute_dest
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %cast = memref.cast %ddr
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<4x8xf16, strided<[8, 1], offset: ?>,
               #wafer.memory<ddr, tensor>>
  %dma_dest = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %cast to %dma_dest
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, strided<[8, 1], offset: ?>,
               #wafer.memory<ddr, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  EXPECT_NE(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  llvm::SmallVector<mlir::Operation *, 2> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface>(operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 2u);
  EXPECT_TRUE(mlir::isa<wafer::InstrRDMAOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(ordered[1]));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(ReadyOrderTest, RetainsDTEReceiveWaitBeforeDestinationConsumer) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %buffer = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %token = wafer.instr.dte_recv %buffer
      {peer = 0 : i64, bytes = 64 : i64,
       message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>> -> !async.token
  wafer.instr.dte_wait %token : !async.token
  wafer.instr.wdma %buffer to %ddr
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<ddr, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  llvm::SmallVector<mlir::Operation *, 3> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface>(operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 3u);
  EXPECT_TRUE(mlir::isa<wafer::InstrDTERecvOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(ordered[1]));
  EXPECT_TRUE(mlir::isa<wafer::InstrWDMAOp>(ordered[2]));
}

TEST(ReadyOrderTest, RetainsDTESendWaitBeforeSourceOverwrite) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %buffer = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %token = wafer.instr.dte_send %buffer
      {peer = 0 : i64, bytes = 64 : i64,
       message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>> -> !async.token
  wafer.instr.dte_wait %token : !async.token
  wafer.instr.rdma %ddr to %buffer
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  llvm::SmallVector<mlir::Operation *, 3> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface>(operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 3u);
  EXPECT_TRUE(mlir::isa<wafer::InstrDTESendOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrDTEWaitOp>(ordered[1]));
  EXPECT_TRUE(mlir::isa<wafer::InstrRDMAOp>(ordered[2]));
}

TEST(ReadyOrderTest, RetainsDTEWaitAcrossStandardBufferEffect) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  %buffer = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %ddr = memref.alloc()
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %dma_dest = memref.alloc()
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  %zero = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %token = wafer.instr.dte_recv %buffer
      {peer = 0 : i64, bytes = 64 : i64,
       message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>> -> !async.token
  wafer.instr.dte_wait %token : !async.token
  memref.store %zero, %buffer[%c0, %c0]
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %ddr to %dma_dest
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  auto wait = *module->getOps<wafer::InstrDTEWaitOp>().begin();
  auto store = *module->getOps<mlir::memref::StoreOp>().begin();
  EXPECT_TRUE(wait->isBeforeInBlock(store));
}

} // namespace
