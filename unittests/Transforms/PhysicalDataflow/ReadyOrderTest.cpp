#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

TEST(ReadyOrderTest, MovesIndependentDMABeforeComputeAndPreservesFence) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
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
  wafer.instr.local_fence
  wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
      %compute_source, %compute_source into %compute_dest
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>
}
)mlir", mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  EXPECT_NE(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
  llvm::SmallVector<mlir::Operation *, 4> ordered;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface,
                  wafer::SyncLocalFenceOp>(operation))
      ordered.push_back(operation);
  });
  ASSERT_EQ(ordered.size(), 4u);
  EXPECT_TRUE(mlir::isa<wafer::InstrRDMAOp>(ordered[0]));
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(ordered[1]));
  EXPECT_TRUE(mlir::isa<wafer::SyncLocalFenceOp>(ordered[2]));
  EXPECT_TRUE(mlir::isa<wafer::InstrElementwiseOp>(ordered[3]));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(ReadyOrderTest, RetainsValueHazards) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
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
)mlir", mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  EXPECT_EQ(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
}

TEST(ReadyOrderTest, RetainsHazardsThroughMemrefViews) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
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
)mlir", mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::scheduleIndependentInstructionsByReadyOrder(
                module->getOperation()),
            0u);
}

TEST(ReadyOrderTest, MovesAcrossIndependentProductionSetupOperations) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
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
)mlir", mlir::ParserConfig(&context));
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
  wafer::registerAllDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
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
)mlir", mlir::ParserConfig(&context));
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
  wafer::registerAllDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
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
)mlir", mlir::ParserConfig(&context));
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
  wafer::registerAllDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
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
)mlir", mlir::ParserConfig(&context));
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
