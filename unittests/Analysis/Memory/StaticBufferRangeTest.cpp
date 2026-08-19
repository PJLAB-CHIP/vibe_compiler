//===- StaticBufferRangeTest.cpp - Static buffer range tests ------------===//

#include "Wafer/Analysis/Memory/StaticBufferRange.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

class StaticBufferRangeTest : public testing::Test {
protected:
  StaticBufferRangeTest() {
    wafer::registerWaferCoreDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
};

TEST_F(StaticBufferRangeTest, ResolvesNestedSubviewToAllocationBytes) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %root = memref.alloc() :
        memref<32xf16, #wafer.memory<spm, tensor>>
    %outer = memref.subview %root[4] [20] [1]
        : memref<32xf16, #wafer.memory<spm, tensor>>
       to memref<20xf16, strided<[1], offset: 4>,
                 #wafer.memory<spm, tensor>>
    %inner = memref.subview %outer[3] [5] [1]
        : memref<20xf16, strided<[1], offset: 4>,
                 #wafer.memory<spm, tensor>>
       to memref<5xf16, strided<[1], offset: 7>,
                 #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::memref::AllocOp allocation;
  llvm::SmallVector<mlir::memref::SubViewOp, 2> subviews;
  module->walk([&](mlir::memref::AllocOp op) { allocation = op; });
  module->walk([&](mlir::memref::SubViewOp op) { subviews.push_back(op); });
  ASSERT_TRUE(allocation);
  ASSERT_EQ(subviews.size(), 2u);

  std::optional<wafer::analysis::StaticBufferRange> resolved =
      wafer::analysis::resolveStaticBufferRange(subviews.back().getResult());
  ASSERT_TRUE(resolved);
  EXPECT_EQ(resolved->root, allocation.getResult());
  EXPECT_EQ(resolved->bytes,
            (wafer::analysis::StaticByteRange{/*begin=*/14, /*end=*/24}));
}

TEST_F(StaticBufferRangeTest,
       ResolvesTileBoundaryCastAndTwoDimensionalSubview) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %boundary: memref<4x8xf32, #wafer.memory<ddr, tensor>>) {
    %result = wafer.tile.region(
        %boundary : memref<4x8xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4x8xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4x8xf32, #wafer.memory<ddr, tensor>>):
      %cast = memref.cast %arg0
          : memref<4x8xf32, #wafer.memory<ddr, tensor>>
         to memref<4x8xf32, #wafer.memory<ddr, tensor>>
      %middle = memref.subview %cast[1, 0] [2, 8] [1, 1]
          : memref<4x8xf32, #wafer.memory<ddr, tensor>>
         to memref<2x8xf32, strided<[8, 1], offset: 8>,
                   #wafer.memory<ddr, tensor>>
      wafer.tile.yield %arg0
          : memref<4x8xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  mlir::memref::SubViewOp subview;
  module->walk([&](mlir::memref::SubViewOp op) { subview = op; });
  ASSERT_TRUE(subview);

  std::optional<wafer::analysis::StaticBufferRange> resolved =
      wafer::analysis::resolveStaticBufferRange(subview.getResult());
  ASSERT_TRUE(resolved);
  EXPECT_EQ(resolved->root, function.getArgument(0));
  EXPECT_EQ(resolved->bytes,
            (wafer::analysis::StaticByteRange{/*begin=*/32, /*end=*/96}));
}

TEST_F(StaticBufferRangeTest, DistinguishesOverlapAndDisjointLeavesOnOneRoot) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %root = memref.alloc() :
        memref<4x8xf32, #wafer.memory<spm, tensor>>
    %left = memref.subview %root[0, 0] [2, 8] [1, 1]
        : memref<4x8xf32, #wafer.memory<spm, tensor>>
       to memref<2x8xf32, strided<[8, 1], offset: 0>,
                 #wafer.memory<spm, tensor>>
    %overlap = memref.subview %root[1, 0] [2, 8] [1, 1]
        : memref<4x8xf32, #wafer.memory<spm, tensor>>
       to memref<2x8xf32, strided<[8, 1], offset: 8>,
                 #wafer.memory<spm, tensor>>
    %disjoint = memref.subview %root[3, 0] [1, 8] [1, 1]
        : memref<4x8xf32, #wafer.memory<spm, tensor>>
       to memref<1x8xf32, strided<[8, 1], offset: 24>,
                 #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::memref::SubViewOp, 3> subviews;
  module->walk([&](mlir::memref::SubViewOp op) { subviews.push_back(op); });
  ASSERT_EQ(subviews.size(), 3u);
  auto left =
      wafer::analysis::resolveStaticBufferRange(subviews[0].getResult());
  auto overlap =
      wafer::analysis::resolveStaticBufferRange(subviews[1].getResult());
  auto disjoint =
      wafer::analysis::resolveStaticBufferRange(subviews[2].getResult());
  ASSERT_TRUE(left);
  ASSERT_TRUE(overlap);
  ASSERT_TRUE(disjoint);
  EXPECT_EQ(left->root, overlap->root);
  EXPECT_EQ(overlap->root, disjoint->root);
  EXPECT_TRUE(
      wafer::analysis::staticByteRangesOverlap(left->bytes, overlap->bytes));
  EXPECT_TRUE(wafer::analysis::staticByteRangesAreDisjoint(left->bytes,
                                                           disjoint->bytes));
}

TEST_F(StaticBufferRangeTest, RejectsStridedAndDynamicSubviewRanges) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%offset: index) {
    %root = memref.alloc() :
        memref<32xf16, #wafer.memory<spm, tensor>>
    %strided = memref.subview %root[2] [4] [2]
        : memref<32xf16, #wafer.memory<spm, tensor>>
       to memref<4xf16, strided<[2], offset: 2>,
                 #wafer.memory<spm, tensor>>
    %dynamic = memref.subview %root[%offset] [4] [1]
        : memref<32xf16, #wafer.memory<spm, tensor>>
       to memref<4xf16, strided<[1], offset: ?>,
                 #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::memref::SubViewOp, 2> subviews;
  module->walk([&](mlir::memref::SubViewOp op) { subviews.push_back(op); });
  ASSERT_EQ(subviews.size(), 2u);
  EXPECT_FALSE(
      wafer::analysis::resolveStaticBufferRange(subviews[0].getResult()));
  EXPECT_FALSE(
      wafer::analysis::resolveStaticBufferRange(subviews[1].getResult()));
}

TEST_F(StaticBufferRangeTest, RejectsByteOffsetOverflow) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %buffer: memref<4xf16,
          strided<[1], offset: 9223372036854775806>,
          #wafer.memory<spm, tensor>>) {
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  EXPECT_FALSE(wafer::analysis::getStaticByteRange(function.getArgument(0)));
  EXPECT_FALSE(
      wafer::analysis::resolveStaticBufferRange(function.getArgument(0)));
}

} // namespace
