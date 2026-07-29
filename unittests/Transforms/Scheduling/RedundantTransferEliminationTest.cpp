//===- RedundantTransferEliminationTest.cpp - Full-buffer copy tests -----===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

#include "Wafer/InitAll.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

#include <memory>

namespace {

template <typename OpTy> unsigned countOps(mlir::ModuleOp root) {
  unsigned count = 0;
  root->walk([&](OpTy) { ++count; });
  return count;
}

class RedundantTransferEliminationTest : public ::testing::Test {
protected:
  RedundantTransferEliminationTest() {
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

  void expectTransferPreserved(llvm::StringRef source) {
    mlir::OwningOpRef<mlir::ModuleOp> module = parse(source);
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    ASSERT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 1u);

    EXPECT_EQ(
        wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
            *module),
        0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 1u);
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(RedundantTransferEliminationTest,
       ElidesExactFullBufferNonSingletonReshape) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<3x2xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<3x2xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %value = memref.load %dest[%c0, %c0]
        : memref<3x2xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  ASSERT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 1u);
  ASSERT_EQ(countOps<mlir::memref::AllocOp>(*module), 2u);

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
  EXPECT_EQ(countOps<mlir::memref::ReinterpretCastOp>(*module), 1u);
}

TEST_F(RedundantTransferEliminationTest, ElidesEqualNonCompactPhysicalMaps) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<2x2xf16, strided<[3, 1]>,
                 #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x2xf16, strided<[3, 1]>,
                 #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 10 : i64, inner_bytes = 10 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x2xf16, strided<[3, 1]>,
                 #wafer.memory<spm, tensor>>
       to memref<2x2xf16, strided<[3, 1]>,
                 #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %value = memref.load %dest[%c0, %c0]
        : memref<2x2xf16, strided<[3, 1]>,
                 #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
}

TEST_F(RedundantTransferEliminationTest,
       ElidesExactCrossEncodingViewWithSourceMemorySpace) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<1x128xf16, #wafer.memory<spm, cx>>
    %dest = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 256 : i64, inner_bytes = 256 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<1x128xf16, #wafer.memory<spm, cx>>
       to memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %value = memref.load %dest[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
  EXPECT_EQ(countOps<mlir::memref::ReinterpretCastOp>(*module), 1u);
  module->walk([&](mlir::memref::ReinterpretCastOp view) {
    auto resultType = mlir::cast<mlir::MemRefType>(view.getType());
    EXPECT_EQ(wafer::getWaferMemoryAttr(resultType).getLayout(),
              wafer::MemLayout::Cx);
  });
}

TEST_F(RedundantTransferEliminationTest,
       ElidesRelaxedFullCastAndRaisesDonatedStorageAlignment) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %source_cast = memref.cast %source
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, strided<[1], offset: ?>,
                 #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, cx>>
    wafer.instr.gather_scatter %source_cast to %dest
        {byte_count = 256 : i64, inner_bytes = 256 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<128xf16, strided<[1], offset: ?>,
                 #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<spm, cx>>
    wafer.instr.local_fence
    %value = memref.load %dest[%c0]
        : memref<128xf16, #wafer.memory<spm, cx>>
    return %value : f16
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
  mlir::memref::AllocOp allocation;
  module->walk(
      [&](mlir::memref::AllocOp candidate) { allocation = candidate; });
  ASSERT_TRUE(allocation);
  ASSERT_TRUE(allocation.getAlignment().has_value());
  EXPECT_EQ(*allocation.getAlignment(), 256u);
}

TEST_F(RedundantTransferEliminationTest,
       PreservesCopyWhenReplacementViolatesConsumerLayoutContract) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() {
    %source = memref.alloc()
        : memref<4x64xf16, #wafer.memory<spm, tensor>>
    %source_cast = memref.cast %source
        : memref<4x64xf16, #wafer.memory<spm, tensor>>
       to memref<4x64xf16, strided<[64, 1], offset: ?>,
                 #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<4x64xf16, #wafer.memory<spm, cx>>
    wafer.instr.gather_scatter %source_cast to %dest
        {byte_count = 512 : i64, inner_bytes = 512 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<4x64xf16, strided<[64, 1], offset: ?>,
                 #wafer.memory<spm, tensor>>
       to memref<4x64xf16, #wafer.memory<spm, cx>>
    wafer.instr.local_fence
    %reduced = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, cx>>
    wafer.instr.reduce #wafer.instr_reduce_kind<sum> %dest into %reduced
        {dim = 0 : i64}
        : memref<4x64xf16, #wafer.memory<spm, cx>>
      into memref<4xf16, #wafer.memory<spm, cx>>
    wafer.instr.local_fence
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 1u);
  module->walk([&](mlir::memref::AllocOp allocation) {
    if (allocation.getType().getShape() == llvm::ArrayRef<int64_t>({4, 64}) &&
        wafer::getWaferMemoryAttr(allocation.getType()).getLayout() ==
            wafer::MemLayout::Tensor)
      EXPECT_FALSE(allocation.getAlignment().has_value());
  });
  EXPECT_EQ(countOps<mlir::memref::ReinterpretCastOp>(*module), 0u);
}

TEST_F(RedundantTransferEliminationTest,
       ElidesFullReshapeBeforeComputeConsumer) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<3x2xf16, #wafer.memory<spm, tensor>>
    %computed = memref.alloc()
        : memref<3x2xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<3x2xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    wafer.instr.elementwise <neg> %dest into %computed
        : memref<3x2xf16, #wafer.memory<spm, tensor>>
      into memref<3x2xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %value = memref.load %computed[%c0, %c0]
        : memref<3x2xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 2u);
  EXPECT_EQ(countOps<mlir::memref::ReinterpretCastOp>(*module), 1u);
}

TEST_F(RedundantTransferEliminationTest,
       ElidesFullCopyWithReadOnlySourceAndDestinationFanout) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() -> (f16, f16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %source_value = memref.load %source[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest_value = memref.load %dest[%c1, %c1]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %source_value, %dest_value : f16, f16
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
}

TEST_F(RedundantTransferEliminationTest,
       ElidesFullCopyByDonatingDeadSourceStorageToMutableDestination) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() -> (f16, f16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    memref.store %zero, %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %written_value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %copied_value = memref.load %dest[%c0, %c1]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %written_value, %copied_value : f16, f16
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
}

TEST_F(RedundantTransferEliminationTest, PreservesPartialCopy) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 6 : i64, inner_bytes = 6 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest, PreservesFullPayloadPermutation) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 2 : i64,
         src_strides = array<i64: 2, 6, 0>,
         src_iterations = array<i64: 3, 2, 1>,
         dst_strides = array<i64: 4, 2, 0>,
         dst_iterations = array<i64: 3, 2, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest, PreservesNonCompactLayoutChange) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<2x2xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x2xf16, strided<[1, 2]>,
                 #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x2xf16, #wafer.memory<spm, tensor>>
       to memref<2x2xf16, strided<[1, 2]>,
                 #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %value = memref.load %dest[%c0, %c0]
        : memref<2x2xf16, strided<[1, 2]>,
                 #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesExplicitDestinationDeallocation) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    memref.dealloc %dest
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            0u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 1u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 2u);
  EXPECT_EQ(countOps<mlir::memref::DeallocOp>(*module), 1u);
}

TEST_F(RedundantTransferEliminationTest, PreservesNonOwnedSourceStorage) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main(
      %source: memref<2x3xf16, #wafer.memory<spm, tensor>>,
      %maybe_alias: memref<2x3xf16, #wafer.memory<spm, tensor>>) -> f16 {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f16
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    memref.store %zero, %maybe_alias[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesUnknownCallResultSourceOrigin) {
  expectTransferPreserved(R"mlir(
module {
  func.func private @identity(
      memref<2x3xf16, #wafer.memory<spm, tensor>>)
      -> memref<2x3xf16, #wafer.memory<spm, tensor>>

  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f16
    %root = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %source = func.call @identity(%root)
        : (memref<2x3xf16, #wafer.memory<spm, tensor>>)
       -> memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    memref.store %zero, %root[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesOffsetSourceViewWithStrongerDestinationAlignment) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %root = memref.alloc() {alignment = 256 : i64}
        : memref<129xf16, #wafer.memory<spm, tensor>>
    %source = memref.reinterpret_cast %root to
        offset: [1], sizes: [128], strides: [1]
        : memref<129xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, strided<[1], offset: ?>,
                 #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {alignment = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, cx>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 256 : i64, inner_bytes = 256 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<128xf16, strided<[1], offset: ?>,
                 #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<spm, cx>>
    wafer.instr.local_fence
    %value = memref.load %dest[%c0]
        : memref<128xf16, #wafer.memory<spm, cx>>
    return %value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesTransferInUnsupportedControlFlow) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main(%condition: i1) -> f16 {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.if %condition {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
    }
    %value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesDestinationWriteSnapshotSemantics) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() -> (f16, f16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    memref.store %zero, %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest_value = memref.load %dest[%c0, %c1]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %source_value = memref.load %source[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %dest_value, %source_value : f16, f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesSourceWriteSnapshotSemantics) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() -> (f16, f16) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    memref.store %zero, %source[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest_value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %source_value = memref.load %source[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %dest_value, %source_value : f16, f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesCopyAcrossOutstandingDTEReadUntilWait) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %token = wafer.instr.dte_send %dest
        {peer = 1 : i64, bytes = 12 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>> -> !async.token
    memref.store %zero, %source[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %token : !async.token
    %source_value = memref.load %source[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %source_value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesDonationAcrossDTEReadIssuedBeforeCopyAndNotYetWaited) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %source
        {peer = 1 : i64, bytes = 12 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    memref.store %zero, %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %token : !async.token
    %value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       ElidesReadOnlySharingAfterExactDTEWait) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() -> (f16, f16) {
    %c0 = arith.constant 0 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        {byte_count = 12 : i64, inner_bytes = 12 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %token = wafer.instr.dte_send %dest
        {peer = 1 : i64, bytes = 12 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    %source_value = memref.load %source[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest_value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %source_value, %dest_value : f16, f16
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
}

TEST_F(RedundantTransferEliminationTest,
       ElidesReadOnlyFullCopyInStaticPositiveLoop) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c4 step %c1 {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      %unused = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
}

TEST_F(RedundantTransferEliminationTest,
       ElidesExactCrossEncodingReshapeInStaticPositiveLoop) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<1x128xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, cx>>
    scf.for %index = %c0 to %c2 step %c1 {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 256 : i64, inner_bytes = 256 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<1x128xf16, #wafer.memory<spm, tensor>>
         to memref<128xf16, #wafer.memory<spm, cx>>
      wafer.instr.local_fence
      %unused = memref.load %dest[%c0]
          : memref<128xf16, #wafer.memory<spm, cx>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
  EXPECT_EQ(countOps<mlir::memref::ReinterpretCastOp>(*module), 1u);
  module->walk([&](mlir::memref::ReinterpretCastOp view) {
    auto resultType = mlir::cast<mlir::MemRefType>(view.getType());
    EXPECT_EQ(wafer::getWaferMemoryAttr(resultType).getLayout(),
              wafer::MemLayout::Tensor);
  });
}

TEST_F(RedundantTransferEliminationTest,
       ElidesLoopCopyWithReadOnlyConsumerAfterExactDTEWait) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      %token = wafer.instr.dte_send %dest
          {peer = 1 : i64, bytes = 12 : i64,
           message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>> -> !async.token
      wafer.instr.dte_wait %token : !async.token
      %unused = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  EXPECT_EQ(wafer::tensor_program_scheduling::elideRedundantFullBufferTransfers(
                *module),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
  EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
}

TEST_F(RedundantTransferEliminationTest, PreservesCopyInDynamicTripLoop) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main(%upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %upper step %c1 {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      %unused = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest, PreservesCopyInZeroTripLoop) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c0 step %c1 {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      %unused = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest, PreservesCopyInNestedLoop) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %outer = %c0 to %c2 step %c1 {
      scf.for %inner = %c0 to %c2 step %c1 {
        wafer.instr.gather_scatter %source to %dest
            {byte_count = 12 : i64, inner_bytes = 12 : i64,
             src_strides = array<i64: 0, 0, 0>,
             src_iterations = array<i64: 1, 1, 1>,
             dst_strides = array<i64: 0, 0, 0>,
             dst_iterations = array<i64: 1, 1, 1>}
            : memref<2x3xf16, #wafer.memory<spm, tensor>>
           to memref<2x3xf16, #wafer.memory<spm, tensor>>
        wafer.instr.local_fence
        %unused = memref.load %dest[%c0, %c0]
            : memref<2x3xf16, #wafer.memory<spm, tensor>>
      }
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest, PreservesConditionalCopyInLoop) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main(%condition: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      scf.if %condition {
        wafer.instr.gather_scatter %source to %dest
            {byte_count = 12 : i64, inner_bytes = 12 : i64,
             src_strides = array<i64: 0, 0, 0>,
             src_iterations = array<i64: 1, 1, 1>,
             dst_strides = array<i64: 0, 0, 0>,
             dst_iterations = array<i64: 1, 1, 1>}
            : memref<2x3xf16, #wafer.memory<spm, tensor>>
           to memref<2x3xf16, #wafer.memory<spm, tensor>>
        wafer.instr.local_fence
        %unused = memref.load %dest[%c0, %c0]
            : memref<2x3xf16, #wafer.memory<spm, tensor>>
      }
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesCopyWithLoopLocalDestinationRoot) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      %dest = memref.alloc()
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      %unused = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest, PreservesLoopCarriedDestinationRoot) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %looped = scf.for %index = %c0 to %c2 step %c1
        iter_args(%iter = %dest)
        -> (memref<2x3xf16, #wafer.memory<spm, tensor>>) {
      wafer.instr.gather_scatter %source to %iter
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      %unused = memref.load %iter[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
      scf.yield %iter
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest, PreservesMutableDestinationInLoop) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      memref.store %zero, %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesSourceOverwriteAfterLoopCopy) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      memref.store %zero, %source[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
      %unused = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesDestinationAccessBeforeLoopCopy) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      %before = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      %after = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest, PreservesDestinationUseAfterLoop) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() -> f16 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
    }
    %value = memref.load %dest[%c0, %c0]
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %value : f16
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesLoopCopyWithDTECompletionAcrossBackedge) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %initial = wafer.instr.dte_send %source
        {peer = 1 : i64, bytes = 12 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>> -> !async.token
    %pending = scf.for %index = %c0 to %c2 step %c1
        iter_args(%previous = %initial) -> (!async.token) {
      wafer.instr.dte_wait %previous : !async.token
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      %next = wafer.instr.dte_send %dest
          {peer = 1 : i64, bytes = 12 : i64,
           message = #wafer.dte_message<communication = 1, phase = collective_permute, round = 0, slice = 0>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>> -> !async.token
      scf.yield %next : !async.token
    }
    wafer.instr.dte_wait %pending : !async.token
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesLoopCopyAcrossOutstandingDTEReadUntilExactWait) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %source = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      %token = wafer.instr.dte_send %source
          {peer = 1 : i64, bytes = 12 : i64,
           message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>> -> !async.token
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      wafer.instr.dte_wait %token : !async.token
      %unused = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest,
       PreservesSelfCopyWhenLoopTripCountIsDynamic) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main(%upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buffer = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %upper step %c1 {
      wafer.instr.gather_scatter %buffer to %buffer
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
    }
    return
  }
}
)mlir");
}

TEST_F(RedundantTransferEliminationTest, PreservesCopyWithLoopLocalSourceRoot) {
  expectTransferPreserved(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dest = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      %source = memref.alloc()
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 12 : i64, inner_bytes = 12 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
         to memref<2x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
      %unused = memref.load %dest[%c0, %c0]
          : memref<2x3xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
}

} // namespace
