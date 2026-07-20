//===- IndexRelationTest.cpp - MLIR-backed index relation tests ----------===//

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

#include <limits>

namespace {

using wafer::analysis::IndexRelation;
using wafer::analysis::IndexRelationLimits;
using wafer::analysis::IndexRelationResult;
using wafer::analysis::IndexRelationStatus;

TEST(IndexRelationTest, RepresentsIdentityPermutationAndBroadcastExactly) {
  IndexRelationResult identity = IndexRelation::identity({2, 3});
  ASSERT_TRUE(identity.isExact());
  EXPECT_TRUE(identity.get()->contains({1, 2}, {1, 2}));
  EXPECT_FALSE(identity.get()->contains({1, 2}, {0, 2}));

  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &context);
  IndexRelationResult permutation = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1, d0}, &context),
      /*destinationShape=*/{2, 3}, /*sourceShape=*/{3, 2});
  ASSERT_TRUE(permutation.isExact());
  EXPECT_TRUE(permutation.get()->contains({1, 2}, {2, 1}));

  IndexRelationResult broadcast = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(2, 0, {d1}, &context),
      /*destinationShape=*/{4, 3}, /*sourceShape=*/{3});
  ASSERT_TRUE(broadcast.isExact());
  EXPECT_TRUE(broadcast.get()->contains({0, 2}, {2}));
  EXPECT_TRUE(broadcast.get()->contains({3, 2}, {2}));
}

TEST(IndexRelationTest, ComposesSliceAndReshapePointwise) {
  IndexRelationResult slice = IndexRelation::staticSlice(
      /*destinationShape=*/{2, 2}, /*sourceShape=*/{4, 4},
      /*offsets=*/{1, 1}, /*strides=*/{1, 1});
  IndexRelationResult reshape =
      IndexRelation::staticReshape(/*destinationShape=*/{4, 4},
                                   /*sourceShape=*/{16});
  ASSERT_TRUE(slice.isExact());
  ASSERT_TRUE(reshape.isExact());

  IndexRelationResult composed = slice.get()->compose(*reshape.get());
  ASSERT_TRUE(composed.isExact());
  EXPECT_TRUE(composed.get()->contains({0, 0}, {5}));
  EXPECT_TRUE(composed.get()->contains({1, 1}, {10}));
  EXPECT_FALSE(composed.get()->contains({1, 1}, {11}));
}

TEST(IndexRelationTest, ResolvesMixedSliceOperandsWithValueBounds) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  mlir::Value one =
      builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  mlir::Value two =
      builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 2);

  IndexRelationResult relation = IndexRelation::slice(
      /*destinationShape=*/{2, 2}, /*sourceShape=*/{5, 5},
      /*offsets=*/{mlir::OpFoldResult(one), mlir::OpFoldResult(one)},
      /*strides=*/{mlir::OpFoldResult(two), mlir::OpFoldResult(two)});
  ASSERT_TRUE(relation.isExact());
  EXPECT_TRUE(relation.get()->contains({1, 1}, {3, 3}));
}

TEST(IndexRelationTest, DistinguishesBoundUnsupportedInvalidAndBudgetFailure) {
  mlir::MLIRContext context;
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineMap identity = mlir::AffineMap::get(1, 0, {d0}, &context);
  IndexRelationResult bound = IndexRelation::fromAffineMap(
      identity, /*destinationShape=*/{mlir::ShapedType::kDynamic},
      /*sourceShape=*/{mlir::ShapedType::kDynamic});
  EXPECT_EQ(bound.status, IndexRelationStatus::SoundBound);
  ASSERT_TRUE(bound.get());
  IndexRelationResult composedBound = bound.get()->compose(*bound.get());
  EXPECT_EQ(composedBound.status, IndexRelationStatus::SoundBound);
  EXPECT_FALSE(composedBound.isExact());

  mlir::AffineExpr symbol = mlir::getAffineSymbolExpr(0, &context);
  IndexRelationResult unsupported = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(1, 1, {d0 + symbol}, &context), {4}, {4});
  EXPECT_EQ(unsupported.status, IndexRelationStatus::Unsupported);
  EXPECT_FALSE(unsupported.get());

  IndexRelationResult invalid = IndexRelation::staticReshape({2, 3}, {5});
  EXPECT_EQ(invalid.status, IndexRelationStatus::Invalid);
  IndexRelationResult overflow =
      IndexRelation::staticReshape({std::numeric_limits<int64_t>::max(), 2},
                                   {std::numeric_limits<int64_t>::max(), 2});
  EXPECT_EQ(overflow.status, IndexRelationStatus::Invalid);

  IndexRelationResult exhausted =
      IndexRelation::fromAffineMap(identity, {4}, {4},
                                   IndexRelationLimits{/*maxVariables=*/1,
                                                       /*maxDisjuncts=*/1});
  EXPECT_EQ(exhausted.status, IndexRelationStatus::ResourceExhausted);
}

} // namespace
