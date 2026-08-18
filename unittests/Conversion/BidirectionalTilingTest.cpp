#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "../../lib/Wafer/Conversion/WaferTensorProgramToTileRegion/Internal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <initializer_list>
#include <string>

namespace {

void registerTilingDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
}

template <typename OpTy> OpTy findSingleOp(mlir::ModuleOp module) {
  OpTy found;
  module.walk([&](OpTy op) {
    EXPECT_FALSE(found);
    found = op;
  });
  return found;
}

llvm::SmallVector<mlir::OpFoldResult, 4>
indexAttrs(mlir::OpBuilder &builder, std::initializer_list<int64_t> values) {
  llvm::SmallVector<mlir::OpFoldResult, 4> result;
  for (int64_t value : values)
    result.push_back(builder.getIndexAttr(value));
  return result;
}

void expectConstantValues(llvm::ArrayRef<mlir::OpFoldResult> actual,
                          std::initializer_list<int64_t> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (auto [value, expectedValue] : llvm::zip(actual, expected)) {
    std::optional<int64_t> constant = mlir::getConstantIntValue(value);
    ASSERT_TRUE(constant);
    EXPECT_EQ(*constant, expectedValue);
  }
}

TEST(CandidateOutputMapTest, AcceptsOnlyZeroConstantsAtUnitExtents) {
  mlir::MLIRContext context;
  mlir::Type f16 = mlir::Float16Type::get(&context);
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr zero = mlir::getAffineConstantExpr(0, &context);
  mlir::AffineExpr one = mlir::getAffineConstantExpr(1, &context);

  EXPECT_TRUE(wafer::tensor_program_to_tile_region::
                  isProjectedPermutationWithUnitConstants(
                      mlir::AffineMap::get(1, 0, {d0, zero}, &context),
                      mlir::RankedTensorType::get({4, 1}, f16)));
  EXPECT_FALSE(wafer::tensor_program_to_tile_region::
                   isProjectedPermutationWithUnitConstants(
                       mlir::AffineMap::get(1, 0, {d0, one}, &context),
                       mlir::RankedTensorType::get({4, 2}, f16)));
  EXPECT_FALSE(wafer::tensor_program_to_tile_region::
                   isProjectedPermutationWithUnitConstants(
                       mlir::AffineMap::get(1, 0, {d0, zero}, &context),
                       mlir::RankedTensorType::get({4, 2}, f16)));
  EXPECT_FALSE(wafer::tensor_program_to_tile_region::
                   isProjectedPermutationWithUnitConstants(
                       mlir::AffineMap::get(1, 0, {d0, d0}, &context),
                       mlir::RankedTensorType::get({4, 4}, f16)));
}

TEST(BidirectionalTilingTest,
     MaterializesGenericConsumerFromBoundaryOperandTile) {
  mlir::DialectRegistry registry;
  registerTilingDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @generic(%input: tensor<4x6xf32>, %init: tensor<4x6xf32>)
      -> tensor<4x6xf32> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<4x6xf32>)
        outs(%init : tensor<4x6xf32>) {
    ^bb0(%value: f32, %old: f32):
      %next = arith.addf %value, %old : f32
      linalg.yield %next : f32
    } -> tensor<4x6xf32>
    return %result : tensor<4x6xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  mlir::linalg::GenericOp consumer =
      findSingleOp<mlir::linalg::GenericOp>(*module);
  ASSERT_TRUE(consumer);

  mlir::OpBuilder builder(consumer);
  auto offsets = indexAttrs(builder, {1, 2});
  auto sizes = indexAttrs(builder, {2, 3});
  std::string failureReason;
  mlir::FailureOr<wafer::OperandTileMaterialization> materialized =
      wafer::materializeConsumerFromOperandTile(consumer, builder,
                                                /*operandNumber=*/0, offsets,
                                                sizes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  expectConstantValues(materialized->iterationDomain.offsets, {1, 2});
  expectConstantValues(materialized->iterationDomain.sizes, {2, 3});
  ASSERT_EQ(materialized->tiledOperations.size(), 1u);
  EXPECT_TRUE(mlir::isa<mlir::linalg::GenericOp>(
      materialized->tiledOperations.front()));
  ASSERT_EQ(materialized->tiledValues.size(), 1u);
  auto tiledType = mlir::cast<mlir::RankedTensorType>(
      materialized->tiledValues.front().getType());
  EXPECT_EQ(tiledType.getShape(), llvm::ArrayRef<int64_t>({2, 3}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(BidirectionalTilingTest,
     MaterializesNamedMatmulConsumerAndFillsUnmappedIterationDimension) {
  mlir::DialectRegistry registry;
  registerTilingDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @matmul(%lhs: tensor<4x8xf32>, %rhs: tensor<8x6xf32>,
                    %init: tensor<4x6xf32>) -> tensor<4x6xf32> {
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x6xf32>)
        outs(%init : tensor<4x6xf32>) -> tensor<4x6xf32>
    return %result : tensor<4x6xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  mlir::linalg::MatmulOp consumer =
      findSingleOp<mlir::linalg::MatmulOp>(*module);
  ASSERT_TRUE(consumer);

  mlir::OpBuilder builder(consumer);
  auto offsets = indexAttrs(builder, {1, 2});
  auto sizes = indexAttrs(builder, {2, 3});
  std::string failureReason;
  mlir::FailureOr<wafer::OperandTileMaterialization> materialized =
      wafer::materializeConsumerFromOperandTile(consumer, builder,
                                                /*operandNumber=*/0, offsets,
                                                sizes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  expectConstantValues(materialized->iterationDomain.offsets, {1, 0, 2});
  expectConstantValues(materialized->iterationDomain.sizes, {2, 6, 3});
  ASSERT_EQ(materialized->tiledOperations.size(), 1u);
  EXPECT_TRUE(
      mlir::isa<mlir::linalg::MatmulOp>(materialized->tiledOperations.front()));
  auto tiledType = mlir::cast<mlir::RankedTensorType>(
      materialized->tiledValues.front().getType());
  EXPECT_EQ(tiledType.getShape(), llvm::ArrayRef<int64_t>({2, 6}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(BidirectionalTilingTest,
     MaterializesAndMergesGenericPartialReductionThroughInterface) {
  mlir::DialectRegistry registry;
  registerTilingDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce(%input: tensor<4x8xf32>, %init: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc  : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  mlir::linalg::GenericOp reduction =
      findSingleOp<mlir::linalg::GenericOp>(*module);
  ASSERT_TRUE(reduction);

  mlir::OpBuilder builder(reduction);
  auto offsets = indexAttrs(builder, {1, 0});
  auto sizes = indexAttrs(builder, {2, 4});
  std::string failureReason;
  mlir::FailureOr<wafer::PartialReductionTileMaterialization> materialized =
      wafer::materializePartialReductionTile(reduction, builder, offsets, sizes,
                                             &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(materialized->reductionDimensions, (llvm::SmallVector<int, 2>{1}));
  ASSERT_EQ(materialized->partialOperations.size(), 1u);
  EXPECT_TRUE(mlir::isa<mlir::linalg::GenericOp>(
      materialized->partialOperations.front()));
  ASSERT_EQ(materialized->mergeOperations.size(), 1u);
  EXPECT_TRUE(mlir::isa<mlir::linalg::GenericOp>(
      materialized->mergeOperations.front()));
  EXPECT_EQ(findSingleOp<mlir::linalg::ReduceOp>(*module),
            mlir::linalg::ReduceOp{});
  ASSERT_EQ(materialized->mergedValues.size(), 1u);
  auto mergedType = mlir::cast<mlir::RankedTensorType>(
      materialized->mergedValues.front().getType());
  EXPECT_EQ(mergedType.getShape(), llvm::ArrayRef<int64_t>({2}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(BidirectionalTilingTest,
     MaterializesAndMergesNamedMatmulPartialReductionThroughInterface) {
  mlir::DialectRegistry registry;
  registerTilingDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @matmul(%lhs: tensor<4x8xi32>, %rhs: tensor<8x6xi32>,
                    %init: tensor<4x6xi32>) -> tensor<4x6xi32> {
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xi32>, tensor<8x6xi32>)
        outs(%init : tensor<4x6xi32>) -> tensor<4x6xi32>
    return %result : tensor<4x6xi32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  mlir::linalg::MatmulOp reduction =
      findSingleOp<mlir::linalg::MatmulOp>(*module);
  ASSERT_TRUE(reduction);

  mlir::OpBuilder builder(reduction);
  auto offsets = indexAttrs(builder, {0, 0, 0});
  auto sizes = indexAttrs(builder, {4, 6, 4});
  std::string failureReason;
  mlir::FailureOr<wafer::PartialReductionTileMaterialization> materialized =
      wafer::materializePartialReductionTile(reduction, builder, offsets, sizes,
                                             &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(materialized->reductionDimensions, (llvm::SmallVector<int, 2>{2}));
  ASSERT_EQ(materialized->partialOperations.size(), 1u);
  EXPECT_TRUE(mlir::isa<mlir::linalg::GenericOp>(
      materialized->partialOperations.front()));
  ASSERT_EQ(materialized->mergeOperations.size(), 1u);
  EXPECT_TRUE(mlir::isa<mlir::linalg::GenericOp>(
      materialized->mergeOperations.front()));
  EXPECT_EQ(findSingleOp<mlir::linalg::ReduceOp>(*module),
            mlir::linalg::ReduceOp{});
  ASSERT_EQ(materialized->mergedValues.size(), 1u);
  EXPECT_EQ(materialized->mergedValues.front().getType(),
            reduction->getResult(0).getType());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(BidirectionalTilingTest,
     RejectsNumericallyUnsupportedPartialReductionBeforeMaterialization) {
  mlir::DialectRegistry registry;
  registerTilingDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce(%input: tensor<4x8xi32>, %init: tensor<4xi32>)
      -> tensor<4xi32> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xi32>)
        outs(%init : tensor<4xi32>) {
    ^bb0(%value: i32, %acc: i32):
      %maximum = arith.maxui %value, %acc : i32
      linalg.yield %maximum : i32
    } -> tensor<4xi32>
    return %result : tensor<4xi32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  mlir::linalg::GenericOp reduction =
      findSingleOp<mlir::linalg::GenericOp>(*module);
  ASSERT_TRUE(reduction);

  mlir::OpBuilder builder(reduction);
  auto offsets = indexAttrs(builder, {0, 0});
  auto sizes = indexAttrs(builder, {4, 4});
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(wafer::materializePartialReductionTile(
      reduction, builder, offsets, sizes, &failureReason)));
  EXPECT_EQ(failureReason,
            "candidate reduction split cannot preserve unsigned min/max "
            "semantics with the current reduce kind");
  EXPECT_EQ(findSingleOp<mlir::linalg::GenericOp>(*module), reduction);
  unsigned fillCount = 0;
  module->walk([&](mlir::linalg::FillOp) { ++fillCount; });
  EXPECT_EQ(fillCount, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

} // namespace
