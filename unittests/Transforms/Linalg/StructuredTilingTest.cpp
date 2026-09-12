#include "Wafer/Transforms/Linalg/StructuredTiling.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Linalg/ContractionAccumulation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <initializer_list>
#include <string>

namespace {

TEST(ContractionAccumulationTest, WideStateSurvivesKLoopsAndOriginalUses) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (auto dtype : {"f16", "bf16"})
    for (int64_t k : {1024, 1025, 1031})
      for (unsigned variant : {0u, 1u, 2u, 3u}) {
        const bool generic = variant != 0;
        const bool transposedRhs = variant == 2;
        const std::string lhs =
            "tensor<2x32x" + std::to_string(k) + "x" + dtype + ">";
        const std::string rhs =
            transposedRhs
                ? "tensor<2x65x" + std::to_string(k) + "x" + dtype + ">"
                : "tensor<2x" + std::to_string(k) + "x65x" + dtype + ">";
        const std::string result = std::string("tensor<2x32x65x") + dtype + ">";
        const std::string operation =
            generic
                ? std::string("linalg.generic {indexing_maps = "
                              "[affine_map<(b,m,n,k)->(b,m,k)>, ") +
                      (transposedRhs ? "affine_map<(b,m,n,k)->(b,n,k)>, "
                                     : "affine_map<(b,m,n,k)->(b,k,n)>, ") +
                      "affine_map<(b,m,n,k)->(b,m,n)>], "
                      "iterator_types = "
                      "[\"parallel\",\"parallel\",\"parallel\",\"reduction\"]}"
                : "linalg.batch_matmul";
        std::string source = "module { func.func @test(%a: " + lhs +
                             ", %b: " + rhs + ", %init: " + result + ") -> (" +
                             result + ", " + result + ") { %r = " + operation +
                             " ins(%a, %b : " + lhs + ", " + rhs +
                             ") outs(%init : " + result + ")";
        const std::string flags = variant == 3 ? " fastmath<contract>" : "";
        if (generic)
          source += std::string(" { ^bb0(%x: ") + dtype + ", %y: " + dtype +
                    ", %z: " + dtype + "): %p = arith.mulf %x, %y" + flags +
                    " : " + dtype + " %s = arith.addf %z, %p" + flags + " : " +
                    dtype + " linalg.yield %s : " + dtype + " }";
        source += " -> " + result + " return %r, %r : " + result + ", " +
                  result + " } }";
        auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
        ASSERT_TRUE(module);
        auto function = *module->getOps<mlir::func::FuncOp>().begin();
        ASSERT_TRUE(
            mlir::succeeded(wafer::promoteContractionAccumulation(function)));
        mlir::linalg::LinalgOp contraction;
        unsigned narrow = 0;
        function.walk([&](mlir::linalg::LinalgOp op) {
          if (op.getNumReductionLoops()) {
            ASSERT_FALSE(contraction);
            contraction = op;
          }
        });
        function.walk([&](mlir::arith::TruncFOp) { ++narrow; });
        ASSERT_TRUE(contraction);
        contraction->walk([&](mlir::arith::MulFOp multiply) {
          EXPECT_EQ(multiply.getFastmath(),
                    variant == 3 ? mlir::arith::FastMathFlags::contract
                                 : mlir::arith::FastMathFlags::none);
        });
        EXPECT_EQ(narrow, 1u);
        EXPECT_EQ(contraction.getDpsInputs()[0], function.getArgument(0));
        EXPECT_EQ(contraction.getDpsInputs()[1], function.getArgument(1));
        EXPECT_TRUE(mlir::cast<mlir::RankedTensorType>(
                        contraction->getResult(0).getType())
                        .getElementType()
                        .isF32());
        auto returned = mlir::cast<mlir::func::ReturnOp>(
            function.getBody().front().getTerminator());
        EXPECT_EQ(returned.getOperand(0), returned.getOperand(1));
        EXPECT_EQ(returned.getOperand(0).getType(),
                  function.getArgument(2).getType());
        // The user-provided init is converted once, outside every K block.
        auto initCast = contraction.getDpsInits()[0]
                            .getDefiningOp<mlir::linalg::GenericOp>();
        ASSERT_TRUE(initCast);
        EXPECT_EQ(initCast.getInputs()[0], function.getArgument(2));
        ASSERT_TRUE(
            mlir::succeeded(wafer::promoteContractionAccumulation(function)));
        mlir::IRRewriter rewriter(&context);
        rewriter.setInsertionPoint(contraction);
        // Spatial partial and its merge use the same widened SSA contract.
        for (int64_t begin : {int64_t{0}, ((k - 1) / 128) * 128}) {
          llvm::SmallVector<mlir::OpFoldResult> offsets, partialSizes;
          for (int64_t offset : {int64_t{0}, int64_t{0}, int64_t{0}, begin})
            offsets.push_back(rewriter.getIndexAttr(offset));
          for (int64_t size : {int64_t{2}, int64_t{32}, int64_t{65},
                               std::min(int64_t{128}, k - begin)})
            partialSizes.push_back(rewriter.getIndexAttr(size));
          std::string detail;
          auto partial = wafer::materializePartialReductionTile(
              contraction, rewriter, offsets, partialSizes, &detail);
          ASSERT_TRUE(mlir::succeeded(partial)) << detail;
          EXPECT_EQ(partial->reductionDimensions,
                    (llvm::SmallVector<int, 2>{3}));
          ASSERT_EQ(partial->mergedValues.size(), 1u);
          EXPECT_EQ(partial->mergedValues[0].getType(),
                    contraction->getResult(0).getType());
          for (auto *operation : partial->partialOperations)
            for (auto value : operation->getResults())
              EXPECT_TRUE(mlir::cast<mlir::RankedTensorType>(value.getType())
                              .getElementType()
                              .isF32());
        }
        rewriter.setInsertionPoint(contraction);
        mlir::scf::SCFTilingOptions options;
        llvm::SmallVector<mlir::OpFoldResult> sizes;
        for (int64_t size : {1, 16, 32, 128})
          sizes.push_back(rewriter.getIndexAttr(size));
        options.setTileSizes(sizes);
        auto tiled = mlir::scf::tileUsingSCF(
            rewriter,
            mlir::cast<mlir::TilingInterface>(contraction.getOperation()),
            options);
        ASSERT_TRUE(mlir::succeeded(tiled));
        ASSERT_EQ(tiled->loops.size(), 4u);
        auto reductionLoop = mlir::cast<mlir::scf::ForOp>(tiled->loops.back());
        EXPECT_EQ(mlir::getConstantIntValue(reductionLoop.getUpperBound()), k);
        EXPECT_EQ(mlir::getConstantIntValue(reductionLoop.getStep()), 128);
        EXPECT_TRUE(mlir::cast<mlir::RankedTensorType>(
                        reductionLoop.getInitArgs()[0].getType())
                        .getElementType()
                        .isF32());
        rewriter.replaceOp(contraction, tiled->replacements);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      }
}

void registerTilingDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect, wafer::WaferDialect>();
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

TEST(StructuredTilingTest, MaterializesGenericConsumerFromBoundaryOperandTile) {
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

TEST(StructuredTilingTest,
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

TEST(StructuredTilingTest,
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

TEST(StructuredTilingTest,
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

TEST(StructuredTilingTest,
     MaterializesUnsignedReductionWithoutANumericPolicyGate) {
  mlir::DialectRegistry registry;
  registerTilingDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce(%input: tensor<2x1025x1031xi32>,
                    %init: tensor<2x1025xi32>) -> tensor<2x1025xi32> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%input : tensor<2x1025x1031xi32>)
        outs(%init : tensor<2x1025xi32>) {
    ^bb0(%value: i32, %acc: i32):
      %maximum = arith.maxui %value, %acc : i32
      linalg.yield %maximum : i32
    } -> tensor<2x1025xi32>
    return %result : tensor<2x1025xi32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  mlir::linalg::GenericOp reduction =
      findSingleOp<mlir::linalg::GenericOp>(*module);
  ASSERT_TRUE(reduction);

  mlir::OpBuilder builder(reduction);
  auto offsets = indexAttrs(builder, {0, 0, 0});
  auto sizes = indexAttrs(builder, {2, 128, 64});
  std::string failureReason;
  auto materialized = wafer::materializePartialReductionTile(
      reduction, builder, offsets, sizes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(materialized->reductionDimensions, (llvm::SmallVector<int, 2>{2}));
  EXPECT_FALSE(materialized->partialOperations.empty());
  EXPECT_FALSE(materialized->mergeOperations.empty());
  unsigned maximumCount = 0;
  module->walk([&](mlir::arith::MaxUIOp) { ++maximumCount; });
  EXPECT_GE(maximumCount, 3u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST(StructuredTilingTest, TilesOnlineAttentionK2AsThreeCarriedStates) {
  mlir::DialectRegistry registry;
  registerTilingDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
#q = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>
#k = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>
#v = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>
#s = affine_map<(b, h, m, k1, k2, n) -> ()>
#acc = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>
#row = affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>
module {
  func.func @online(
      %query: tensor<2x16x1025x128xf16>,
      %key: tensor<2x16x1031x128xf16>,
      %value: tensor<2x16x1031x64xf16>, %scale: f32,
      %accumulator: tensor<2x16x1025x64xf16>,
      %maximum: tensor<2x16x1025xf32>,
      %sum: tensor<2x16x1025xf32>)
      -> (tensor<2x16x1025x64xf16>, tensor<2x16x1025xf32>,
          tensor<2x16x1025xf32>) {
    %next_accumulator, %next_maximum, %next_sum =
        wafer.linalg_ext.online_attention
        ins(%query, %key, %value, %scale : tensor<2x16x1025x128xf16>,
            tensor<2x16x1031x128xf16>, tensor<2x16x1031x64xf16>, f32)
        outs(%accumulator, %maximum, %sum : tensor<2x16x1025x64xf16>,
            tensor<2x16x1025xf32>, tensor<2x16x1025xf32>)
        indexing_maps = [#q, #k, #v, #s, #acc, #row, #row] score {
        ^bb0(%attention_0_dot: f16, %attention_0_scale: f32):
          %attention_0_converted = arith.extf %attention_0_dot : f16 to f32
          %attention_0_scaled = arith.mulf %attention_0_converted, %attention_0_scale : f32
          wafer.linalg_ext.attention.yield %attention_0_scaled : f32
        }
        -> (tensor<2x16x1025x64xf16>, tensor<2x16x1025xf32>,
            tensor<2x16x1025xf32>)
    return %next_accumulator, %next_maximum, %next_sum :
        tensor<2x16x1025x64xf16>, tensor<2x16x1025xf32>,
        tensor<2x16x1025xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  wafer::LinalgExtOnlineAttentionOp attention =
      findSingleOp<wafer::LinalgExtOnlineAttentionOp>(*module);
  ASSERT_TRUE(attention);

  mlir::OpBuilder builder(attention);
  auto illegalK1Tile = attention.getTiledImplementation(
      builder, indexAttrs(builder, {0, 0, 0, 0, 0, 0}),
      indexAttrs(builder, {2, 16, 1025, 64, 1031, 64}));
  EXPECT_TRUE(mlir::failed(illegalK1Tile));
  unsigned onlineCount = 0;
  module->walk([&](wafer::LinalgExtOnlineAttentionOp) { ++onlineCount; });
  EXPECT_EQ(onlineCount, 1u);

  mlir::IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(attention);
  mlir::scf::SCFTilingOptions options;
  options.setTileSizes(indexAttrs(rewriter, {0, 0, 0, 0, 256, 0}));
  auto tiled = mlir::scf::tileUsingSCF(
      rewriter, mlir::cast<mlir::TilingInterface>(attention.getOperation()),
      options);
  ASSERT_TRUE(mlir::succeeded(tiled));
  ASSERT_EQ(tiled->loops.size(), 1u);
  auto loop =
      mlir::dyn_cast<mlir::scf::ForOp>(tiled->loops.front().getOperation());
  ASSERT_TRUE(loop);
  EXPECT_EQ(loop.getNumRegionIterArgs(), 3u);
  rewriter.replaceOp(attention, tiled->replacements);

  wafer::LinalgExtOnlineAttentionOp tiledAttention =
      findSingleOp<wafer::LinalgExtOnlineAttentionOp>(*module);
  ASSERT_TRUE(tiledAttention);
  auto tiledKeyType =
      mlir::cast<mlir::RankedTensorType>(tiledAttention.getKey().getType());
  EXPECT_TRUE(tiledKeyType.isDynamicDim(2));
  EXPECT_EQ(tiledAttention.getNumResults(), 3u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

} // namespace
