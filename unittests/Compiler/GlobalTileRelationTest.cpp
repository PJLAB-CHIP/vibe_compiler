#include "Wafer/Compiler/GlobalTileRelation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <limits>
#include <memory>

using namespace wafer;
using namespace wafer::compiler;

namespace {

frontend::ProgramRankSlice slice(int64_t rank,
                                 std::vector<int64_t> offsets,
                                 std::vector<int64_t> sizes,
                                 std::vector<int64_t> strides) {
  frontend::ProgramRankSlice value;
  value.logicalRank = rank;
  value.replicaId = 0;
  value.offsets = std::move(offsets);
  value.sizes = std::move(sizes);
  value.strides = std::move(strides);
  return value;
}

std::unique_ptr<mlir::MLIRContext> createSymbolicViewContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

mlir::OwningOpRef<mlir::ModuleOp>
parseSymbolicViews(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(&context));
}

llvm::SmallVector<mlir::memref::SubViewOp, 4>
findSubviews(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::memref::SubViewOp, 4> subviews;
  module.walk(
      [&](mlir::memref::SubViewOp subview) { subviews.push_back(subview); });
  return subviews;
}

TEST(GlobalTileRelationTest, ComposesAndInvertsPartitionedTile) {
  frontend::ProgramRankSlice rankSlice =
      slice(1, {8, 0}, {8, 16}, {1, 1});
  StaticTileRegion local{{2, 4}, {3, 5}, {1, 2}};
  llvm::Expected<StaticTileRegion> global =
      mapRankLocalTileToGlobal(rankSlice, {16, 16}, {8, 16}, local);
  ASSERT_TRUE(static_cast<bool>(global)) << llvm::toString(global.takeError());
  EXPECT_EQ(global->offsets, (std::vector<int64_t>{10, 4}));
  EXPECT_EQ(global->sizes, (std::vector<int64_t>{3, 5}));
  EXPECT_EQ(global->strides, (std::vector<int64_t>{1, 2}));

  llvm::Expected<StaticTileRegion> projected =
      mapGlobalTileToRankLocal(rankSlice, {16, 16}, {8, 16}, *global);
  ASSERT_TRUE(static_cast<bool>(projected))
      << llvm::toString(projected.takeError());
  EXPECT_EQ(*projected, local);
}

TEST(GlobalTileRelationTest, ReplicatedSlicesMapToEquivalentGlobalTile) {
  frontend::ProgramRankSlice first =
      slice(0, {0, 0}, {16, 16}, {1, 1});
  frontend::ProgramRankSlice second =
      slice(1, {0, 0}, {16, 16}, {1, 1});
  StaticTileRegion local{{4, 2}, {4, 8}, {1, 1}};
  auto lhs = mapRankLocalTileToGlobal(first, {16, 16}, {16, 16}, local);
  auto rhs = mapRankLocalTileToGlobal(second, {16, 16}, {16, 16}, local);
  ASSERT_TRUE(static_cast<bool>(lhs)) << llvm::toString(lhs.takeError());
  ASSERT_TRUE(static_cast<bool>(rhs)) << llvm::toString(rhs.takeError());
  EXPECT_EQ(compareStaticTiles(*lhs, *rhs), StaticTileRelation::Equivalent);
}

TEST(GlobalTileRelationTest, RejectsGapOverlapAndNonIntegralProjection) {
  frontend::ProgramRankSlice rankSlice =
      slice(0, {0}, {4}, {2});
  StaticTileRegion outside{{4}, {2}, {1}};
  auto invalidLocal =
      mapRankLocalTileToGlobal(rankSlice, {8}, {4}, outside);
  EXPECT_FALSE(static_cast<bool>(invalidLocal));
  llvm::consumeError(invalidLocal.takeError());

  StaticTileRegion nonIntegral{{1}, {2}, {2}};
  auto invalidGlobal =
      mapGlobalTileToRankLocal(rankSlice, {8}, {4}, nonIntegral);
  EXPECT_FALSE(static_cast<bool>(invalidGlobal));
  llvm::consumeError(invalidGlobal.takeError());

  StaticTileRegion left{{0}, {4}, {1}};
  StaticTileRegion gap{{4}, {4}, {1}};
  StaticTileRegion overlap{{3}, {4}, {1}};
  EXPECT_EQ(compareStaticTiles(left, gap), StaticTileRelation::Disjoint);
  EXPECT_EQ(compareStaticTiles(left, overlap),
            StaticTileRelation::OverlappingOrUnknown);
}

TEST(GlobalTileRelationTest, FailsClosedOnArithmeticOverflow) {
  frontend::ProgramRankSlice rankSlice =
      slice(0, {0}, {2}, {std::numeric_limits<int64_t>::max()});
  StaticTileRegion tile{{1}, {1}, {2}};
  auto result = mapRankLocalTileToGlobal(
      rankSlice, {std::numeric_limits<int64_t>::max()}, {2}, tile);
  EXPECT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
}

TEST(GlobalTileRelationTest,
     EquivalentLoopViewsIgnoreRankAndReplicaIdentifiers) {
  auto context = createSymbolicViewContext();
  auto module = parseSymbolicViews(*context, R"mlir(
module {
  func.func @first(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c4 {
      %view = memref.subview %arg0[%iv] [4] [1]
          : memref<16xf32>
         to memref<4xf32, strided<[1], offset: ?>>
    }
    return
  }
  func.func @second(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c4 {
      %view = memref.subview %arg0[%iv] [4] [1]
          : memref<16xf32>
         to memref<4xf32, strided<[1], offset: ?>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto subviews = findSubviews(*module);
  ASSERT_EQ(subviews.size(), 2u);
  auto lhs = resolveBoundaryTileView(subviews[0].getResult(), {16});
  auto rhs = resolveBoundaryTileView(subviews[1].getResult(), {16});
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  EXPECT_FALSE(lhs->staticLocalTile);
  EXPECT_FALSE(rhs->staticLocalTile);

  auto lhsSlice = slice(3, {0}, {16}, {1});
  lhsSlice.replicaId = 7;
  auto rhsSlice = slice(11, {0}, {16}, {1});
  rhsSlice.replicaId = 13;
  EXPECT_EQ(compareRankBoundaryTileViews(*lhs, lhsSlice, *rhs, rhsSlice, {16},
                                         {16}),
            StaticTileRelation::Equivalent);
  EXPECT_EQ(hashRankBoundaryTileView(*lhs, lhsSlice, {16}, {16}),
            hashRankBoundaryTileView(*rhs, rhsSlice, {16}, {16}));
}

TEST(GlobalTileRelationTest, SameLoopBoundsDoNotHideDifferentExpressions) {
  auto context = createSymbolicViewContext();
  auto module = parseSymbolicViews(*context, R"mlir(
module {
  func.func @identity(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c12 = arith.constant 12 : index
    scf.for %iv = %c0 to %c12 step %c4 {
      %view = memref.subview %arg0[%iv] [4] [1]
          : memref<16xf32>
         to memref<4xf32, strided<[1], offset: ?>>
    }
    return
  }
  func.func @shifted(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c12 = arith.constant 12 : index
    scf.for %iv = %c0 to %c12 step %c4 {
      %shifted = arith.addi %iv, %c1 : index
      %view = memref.subview %arg0[%shifted] [4] [1]
          : memref<16xf32>
         to memref<4xf32, strided<[1], offset: ?>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto subviews = findSubviews(*module);
  ASSERT_EQ(subviews.size(), 2u);
  auto lhs = resolveBoundaryTileView(subviews[0].getResult(), {16});
  auto rhs = resolveBoundaryTileView(subviews[1].getResult(), {16});
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  auto first = slice(0, {0}, {16}, {1});
  auto second = slice(1, {0}, {16}, {1});
  EXPECT_EQ(compareRankBoundaryTileViews(*lhs, first, *rhs, second, {16},
                                         {16}),
            StaticTileRelation::OverlappingOrUnknown);
}

TEST(GlobalTileRelationTest, SameBoundsInDifferentLoopContextsAreNotEqual) {
  auto context = createSymbolicViewContext();
  auto module = parseSymbolicViews(*context, R"mlir(
module {
  func.func @direct(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c4 {
      %view = memref.subview %arg0[%iv] [4] [1]
          : memref<16xf32>
         to memref<4xf32, strided<[1], offset: ?>>
    }
    return
  }
  func.func @nested(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %outer = %c0 to %c1 step %c1 {
      scf.for %iv = %c0 to %c16 step %c4 {
        %view = memref.subview %arg0[%iv] [4] [1]
            : memref<16xf32>
           to memref<4xf32, strided<[1], offset: ?>>
      }
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto subviews = findSubviews(*module);
  ASSERT_EQ(subviews.size(), 2u);
  auto lhs = resolveBoundaryTileView(subviews[0].getResult(), {16});
  auto rhs = resolveBoundaryTileView(subviews[1].getResult(), {16});
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  auto first = slice(0, {0}, {16}, {1});
  auto second = slice(1, {0}, {16}, {1});
  EXPECT_EQ(compareRankBoundaryTileViews(*lhs, first, *rhs, second, {16},
                                         {16}),
            StaticTileRelation::OverlappingOrUnknown);
}

TEST(GlobalTileRelationTest, RejectsUnknownOffsetsAndDynamicStrides) {
  auto context = createSymbolicViewContext();
  auto module = parseSymbolicViews(*context, R"mlir(
module {
  func.func @unknown_offset(%arg0: memref<16xf32>, %offset: index) {
    %view = memref.subview %arg0[%offset] [4] [1]
        : memref<16xf32>
       to memref<4xf32, strided<[1], offset: ?>>
    return
  }
  func.func @dynamic_stride(%arg0: memref<16xf32>, %stride: index) {
    %view = memref.subview %arg0[0] [4] [%stride]
        : memref<16xf32>
       to memref<4xf32, strided<[?]>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto subviews = findSubviews(*module);
  ASSERT_EQ(subviews.size(), 2u);
  EXPECT_FALSE(resolveBoundaryTileView(subviews[0].getResult(), {16}));
  EXPECT_FALSE(resolveBoundaryTileView(subviews[1].getResult(), {16}));
}

TEST(GlobalTileRelationTest,
     PrivateHelperArgumentOrdinalIsNotAProgramBoundaryIdentity) {
  auto context = createSymbolicViewContext();
  auto module = parseSymbolicViews(*context, R"mlir(
module {
  func.func private @helper(%arg0: memref<16xf32>) {
    %view = memref.subview %arg0[0] [4] [1]
        : memref<16xf32>
       to memref<4xf32, strided<[1], offset: 0>>
    return
  }
  func.func @entry(%arg0: memref<16xf32>) {
    func.call @helper(%arg0) : (memref<16xf32>) -> ()
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto subviews = findSubviews(*module);
  ASSERT_EQ(subviews.size(), 1u);
  EXPECT_FALSE(resolveBoundaryTileView(subviews.front().getResult(), {16}));
  EXPECT_FALSE(hashStructuredOperationPath(subviews.front()));
}

TEST(GlobalTileRelationTest, RejectsOutOfBoundsAndOverflowingLoopViews) {
  auto context = createSymbolicViewContext();
  auto module = parseSymbolicViews(*context, R"mlir(
module {
  func.func @out_of_bounds(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c8 {
      %view = memref.subview %arg0[%iv] [9] [1]
          : memref<16xf32>
         to memref<9xf32, strided<[1], offset: ?>>
    }
    return
  }
  func.func @overflow(%arg0: memref<16xf32>) {
    %min = arith.constant -9223372036854775808 : index
    %max = arith.constant 9223372036854775807 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %min to %max step %c1 {
      %view = memref.subview %arg0[%iv] [1] [1]
          : memref<16xf32>
         to memref<1xf32, strided<[1], offset: ?>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto subviews = findSubviews(*module);
  ASSERT_EQ(subviews.size(), 2u);
  EXPECT_FALSE(resolveBoundaryTileView(subviews[0].getResult(), {16}));
  EXPECT_FALSE(resolveBoundaryTileView(subviews[1].getResult(), {16}));
}

TEST(GlobalTileRelationTest, DynamicCompareRejectsMalformedRankDomain) {
  auto context = createSymbolicViewContext();
  auto module = parseSymbolicViews(*context, R"mlir(
module {
  func.func @first(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c4 {
      %view = memref.subview %arg0[%iv] [4] [1]
          : memref<16xf32>
         to memref<4xf32, strided<[1], offset: ?>>
    }
    return
  }
  func.func @second(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c4 {
      %view = memref.subview %arg0[%iv] [4] [1]
          : memref<16xf32>
         to memref<4xf32, strided<[1], offset: ?>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto subviews = findSubviews(*module);
  ASSERT_EQ(subviews.size(), 2u);
  auto lhs = resolveBoundaryTileView(subviews[0].getResult(), {16});
  auto rhs = resolveBoundaryTileView(subviews[1].getResult(), {16});
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  auto valid = slice(0, {0}, {16}, {1});
  auto malformed = slice(1, {0}, {15}, {1});
  EXPECT_EQ(compareRankBoundaryTileViews(*lhs, valid, *rhs, malformed, {16},
                                         {16}),
            StaticTileRelation::OverlappingOrUnknown);

  malformed = slice(1, {0}, {16},
                    {std::numeric_limits<int64_t>::max()});
  EXPECT_EQ(compareRankBoundaryTileViews(
                *lhs, valid, *rhs, malformed,
                {std::numeric_limits<int64_t>::max()}, {16}),
            StaticTileRelation::OverlappingOrUnknown);
}

} // namespace
