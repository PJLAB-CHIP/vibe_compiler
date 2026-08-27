//===- StructuredGraphNormalizationTest.cpp -----------------------------===//

#include "Wafer/Conversion/StableHLOToLinalg/StructuredGraphNormalization.h"
#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

class StructuredGraphNormalizationTest : public ::testing::Test {
protected:
  StructuredGraphNormalizationTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                    mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  template <typename Op> static size_t count(mlir::Operation *root) {
    size_t result = 0;
    root->walk([&](Op) { ++result; });
    return result;
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredGraphNormalizationTest,
       RaggedTransposeIsAbsorbedIntoElementwiseMap) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    module {
      func.func @transpose_elementwise(
          %arg0: tensor<2x1025x64xf32>) -> tensor<2x64x1025xf32> {
        %transpose_init = tensor.empty() : tensor<2x64x1025xf32>
        %transposed = linalg.transpose
            ins(%arg0 : tensor<2x1025x64xf32>)
            outs(%transpose_init : tensor<2x64x1025xf32>)
            permutation = [0, 2, 1]
        %result_init = tensor.empty() : tensor<2x64x1025xf32>
        %result = linalg.generic {
            indexing_maps = [#identity, #identity],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%transposed : tensor<2x64x1025xf32>)
            outs(%result_init : tensor<2x64x1025xf32>) {
          ^bb0(%input: f32, %output: f32):
            %negated = arith.negf %input : f32
            linalg.yield %negated : f32
        } -> tensor<2x64x1025xf32>
        return %result : tensor<2x64x1025xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function =
      module->lookupSymbol<mlir::func::FuncOp>("transpose_elementwise");
  ASSERT_TRUE(function);
  mlir::linalg::TransposeOp transpose;
  mlir::linalg::GenericOp elementwise;
  function.walk(
      [&](mlir::linalg::TransposeOp operation) { transpose = operation; });
  function.walk(
      [&](mlir::linalg::GenericOp operation) { elementwise = operation; });
  ASSERT_TRUE(transpose);
  ASSERT_TRUE(elementwise);

  mlir::linalg::LinalgOp transposeInterface = transpose;
  llvm::SmallVector<mlir::AffineMap, 2> transposeMaps =
      transposeInterface.getIndexingMapsArray();
  ASSERT_EQ(transposeMaps.size(), 2u);
  wafer::analysis::IndexRelationResult relation =
      wafer::analysis::IndexRelation::fromCommonIterationDomain(
          transposeMaps[1], {2, 64, 1025}, transposeMaps[0], {2, 1025, 64},
          transposeInterface.getStaticLoopRanges());
  ASSERT_TRUE(relation.isExact());
  mlir::AffineMap destinationToIteration =
      mlir::inversePermutation(transposeMaps[1]);
  ASSERT_TRUE(destinationToIteration);
  mlir::AffineMap projected = transposeMaps[0].compose(destinationToIteration);
  wafer::analysis::IndexRelationResult projectedRelation =
      wafer::analysis::IndexRelation::fromAffineMap(projected, {2, 64, 1025},
                                                    {2, 1025, 64});
  ASSERT_TRUE(projectedRelation.isExact());
  EXPECT_TRUE(
      projectedRelation.get()->isEquivalentTo(*relation.get()).isProvenTrue());

  wafer::StructuredGraphNormalizationStatistics statistics;
  mlir::FailureOr<wafer::StructuredGraphNormalizationOutcome> outcome =
      wafer::normalizeStructuredTensorGraph(function, {}, &statistics);
  ASSERT_TRUE(mlir::succeeded(outcome));
  EXPECT_EQ(*outcome, wafer::StructuredGraphNormalizationOutcome::Changed);
  EXPECT_EQ(statistics.components, 1u);
  EXPECT_EQ(statistics.changedComponents, 1u);
  EXPECT_EQ(statistics.accessTransformsRemoved, 1u);
  EXPECT_EQ(count<mlir::linalg::TransposeOp>(function), 0u);
  EXPECT_EQ(count<mlir::linalg::GenericOp>(function), 1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string failureReason;
  mlir::FailureOr<wafer::compiler::detail::StructuredDAGAnalysis> dag =
      wafer::compiler::detail::StructuredDAGAnalysis::create(function,
                                                             &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  ASSERT_EQ(dag->getNodes().size(), 1u);
  EXPECT_TRUE(
      mlir::isa<mlir::linalg::GenericOp>(dag->getNodes().front().operation));
}

TEST_F(StructuredGraphNormalizationTest,
       BudgetExhaustionLeavesRaggedComponentByteIdentical) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    module {
      func.func @transpose_elementwise(
          %arg0: tensor<2x1025x64xf32>) -> tensor<2x64x1025xf32> {
        %transpose_init = tensor.empty() : tensor<2x64x1025xf32>
        %transposed = linalg.transpose
            ins(%arg0 : tensor<2x1025x64xf32>)
            outs(%transpose_init : tensor<2x64x1025xf32>)
            permutation = [0, 2, 1]
        %result_init = tensor.empty() : tensor<2x64x1025xf32>
        %result = linalg.generic {
            indexing_maps = [#identity, #identity],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%transposed : tensor<2x64x1025xf32>)
            outs(%result_init : tensor<2x64x1025xf32>) {
          ^bb0(%input: f32, %output: f32):
            %negated = arith.negf %input : f32
            linalg.yield %negated : f32
        } -> tensor<2x64x1025xf32>
        return %result : tensor<2x64x1025xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function =
      module->lookupSymbol<mlir::func::FuncOp>("transpose_elementwise");
  ASSERT_TRUE(function);
  mlir::linalg::GenericOp elementwise;
  function.walk(
      [&](mlir::linalg::GenericOp operation) { elementwise = operation; });
  ASSERT_TRUE(elementwise);
  const std::string before = print(function);

  wafer::StructuredGraphNormalizationStatistics statistics;
  mlir::FailureOr<wafer::StructuredGraphNormalizationOutcome> outcome =
      wafer::normalizeStructuredTensorGraph(
          function,
          {/*maximumRelationQueries=*/64, /*maximumENodes=*/1,
           /*maximumMatches=*/64, /*maximumIterations=*/8},
          &statistics);
  ASSERT_TRUE(mlir::succeeded(outcome));
  EXPECT_EQ(*outcome,
            wafer::StructuredGraphNormalizationOutcome::BudgetExhausted);
  EXPECT_EQ(statistics.budgetExhaustedComponents, 1u);
  EXPECT_EQ(print(function), before);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(StructuredGraphNormalizationTest,
       AlignedBatchContractionFeedsPinnedPartialReductionTiling) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
    module {
      func.func @batch_matmul(
          %lhs: tensor<2x1024x64xf32>,
          %rhs: tensor<2x1024x128xf32>,
          %init: tensor<2x64x128xf32>) -> tensor<2x64x128xf32> {
        %transpose_init = tensor.empty() : tensor<2x64x1024xf32>
        %transposed = linalg.transpose
            ins(%lhs : tensor<2x1024x64xf32>)
            outs(%transpose_init : tensor<2x64x1024xf32>)
            permutation = [0, 2, 1]
        %result = linalg.batch_matmul
            ins(%transposed, %rhs :
                tensor<2x64x1024xf32>, tensor<2x1024x128xf32>)
            outs(%init : tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
        return %result : tensor<2x64x128xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("batch_matmul");
  ASSERT_TRUE(function);
  mlir::linalg::BatchMatmulOp contraction;
  function.walk(
      [&](mlir::linalg::BatchMatmulOp operation) { contraction = operation; });
  ASSERT_TRUE(contraction);

  mlir::FailureOr<wafer::StructuredGraphNormalizationOutcome> outcome =
      wafer::normalizeStructuredTensorGraph(function);
  ASSERT_TRUE(mlir::succeeded(outcome));
  EXPECT_EQ(*outcome, wafer::StructuredGraphNormalizationOutcome::Changed);
  EXPECT_EQ(count<mlir::linalg::TransposeOp>(function), 0u);
  mlir::linalg::BatchMatmulTransposeAOp normalized;
  function.walk([&](mlir::linalg::BatchMatmulTransposeAOp operation) {
    normalized = operation;
  });
  ASSERT_TRUE(normalized);
  EXPECT_EQ(normalized.getNumLoops(), 4u);
  EXPECT_EQ(llvm::count(normalized.getIteratorTypesArray(),
                        mlir::utils::IteratorType::reduction),
            1u);

  mlir::OpBuilder builder(normalized);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets{
      builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0),
      builder.getIndexAttr(0)};
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes{
      builder.getIndexAttr(2), builder.getIndexAttr(64),
      builder.getIndexAttr(128), builder.getIndexAttr(1024)};
  std::string failureReason;
  mlir::FailureOr<wafer::PartialReductionTileMaterialization> tiled =
      wafer::materializePartialReductionTile(normalized.getOperation(), builder,
                                             offsets, sizes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failureReason;
  ASSERT_EQ(tiled->partialOperations.size(), 1u);
  ASSERT_EQ(tiled->mergeOperations.size(), 1u);
  EXPECT_EQ(tiled->reductionDimensions, (llvm::SmallVector<int, 2>{3}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(StructuredGraphNormalizationTest,
       RaggedReductionResultReindexFeedsPinnedReductionTiling) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
    #identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    #result = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    module {
      func.func @reindex_reduction(
          %input: tensor<2x4x64x1031xf32>, %init: tensor<2x64x4xf32>)
          -> tensor<2x64x4xf32> {
        %init_empty = tensor.empty() : tensor<2x4x64xf32>
        %init_transposed = linalg.transpose
            ins(%init : tensor<2x64x4xf32>)
            outs(%init_empty : tensor<2x4x64xf32>) permutation = [0, 2, 1]
        %reduced = linalg.generic {
            indexing_maps = [#identity, #result],
            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%input : tensor<2x4x64x1031xf32>)
            outs(%init_transposed : tensor<2x4x64xf32>) {
          ^bb0(%element: f32, %accumulator: f32):
            %sum = arith.addf %accumulator, %element : f32
            linalg.yield %sum : f32
        } -> tensor<2x4x64xf32>
        %result_empty = tensor.empty() : tensor<2x64x4xf32>
        %result = linalg.transpose
            ins(%reduced : tensor<2x4x64xf32>)
            outs(%result_empty : tensor<2x64x4xf32>) permutation = [0, 2, 1]
        return %result : tensor<2x64x4xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("reindex_reduction");
  ASSERT_TRUE(function);

  wafer::StructuredGraphNormalizationStatistics statistics;
  mlir::FailureOr<wafer::StructuredGraphNormalizationOutcome> outcome =
      wafer::normalizeStructuredTensorGraph(function, {}, &statistics);
  ASSERT_TRUE(mlir::succeeded(outcome));
  EXPECT_EQ(*outcome, wafer::StructuredGraphNormalizationOutcome::Changed);
  EXPECT_GE(statistics.resultReindexApplications, 1u);
  EXPECT_EQ(count<mlir::linalg::TransposeOp>(function), 0u);
  mlir::linalg::GenericOp normalized;
  function.walk(
      [&](mlir::linalg::GenericOp operation) { normalized = operation; });
  ASSERT_TRUE(normalized);
  EXPECT_EQ(normalized.getIteratorTypesArray()[3],
            mlir::utils::IteratorType::reduction);
  EXPECT_EQ(normalized.getIndexingMapsArray().back(),
            mlir::AffineMap::get(4, 0,
                                 {mlir::getAffineDimExpr(0, context.get()),
                                  mlir::getAffineDimExpr(2, context.get()),
                                  mlir::getAffineDimExpr(1, context.get())},
                                 context.get()));

  std::string failureReason;
  mlir::FailureOr<wafer::compiler::detail::StructuredDAGAnalysis> dag =
      wafer::compiler::detail::StructuredDAGAnalysis::create(function,
                                                             &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  ASSERT_EQ(dag->getNodes().size(), 1u);

  mlir::OpBuilder builder(normalized);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets{
      builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0),
      builder.getIndexAttr(0)};
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes{
      builder.getIndexAttr(2), builder.getIndexAttr(4),
      builder.getIndexAttr(64), builder.getIndexAttr(1024)};
  mlir::FailureOr<wafer::PartialReductionTileMaterialization> tiled =
      wafer::materializePartialReductionTile(normalized.getOperation(), builder,
                                             offsets, sizes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failureReason;
  EXPECT_EQ(tiled->reductionDimensions, (llvm::SmallVector<int, 2>{3}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(StructuredGraphNormalizationTest,
       RaggedBatchContractionResultReindexFeedsPinnedReductionTiling) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
    #lhs = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
    #rhs = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
    #out = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
    module {
      func.func @reindex_contraction(
          %lhs: tensor<2x64x1031xf32>, %rhs: tensor<2x1031x128xf32>,
          %init: tensor<2x64x128xf32>) -> tensor<2x64x128xf32> {
        %init_empty = tensor.empty() : tensor<2x128x64xf32>
        %init_transposed = linalg.transpose
            ins(%init : tensor<2x64x128xf32>)
            outs(%init_empty : tensor<2x128x64xf32>) permutation = [0, 2, 1]
        %contracted = linalg.generic {
            indexing_maps = [#lhs, #rhs, #out],
            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%lhs, %rhs : tensor<2x64x1031xf32>, tensor<2x1031x128xf32>)
            outs(%init_transposed : tensor<2x128x64xf32>) {
          ^bb0(%left: f32, %right: f32, %accumulator: f32):
            %product = arith.mulf %left, %right : f32
            %sum = arith.addf %accumulator, %product : f32
            linalg.yield %sum : f32
        } -> tensor<2x128x64xf32>
        %result_empty = tensor.empty() : tensor<2x64x128xf32>
        %result = linalg.transpose
            ins(%contracted : tensor<2x128x64xf32>)
            outs(%result_empty : tensor<2x64x128xf32>) permutation = [0, 2, 1]
        return %result : tensor<2x64x128xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function =
      module->lookupSymbol<mlir::func::FuncOp>("reindex_contraction");
  ASSERT_TRUE(function);

  wafer::StructuredGraphNormalizationStatistics statistics;
  mlir::FailureOr<wafer::StructuredGraphNormalizationOutcome> outcome =
      wafer::normalizeStructuredTensorGraph(function, {}, &statistics);
  ASSERT_TRUE(mlir::succeeded(outcome));
  EXPECT_EQ(*outcome, wafer::StructuredGraphNormalizationOutcome::Changed);
  EXPECT_GE(statistics.resultReindexApplications, 1u);
  EXPECT_EQ(count<mlir::linalg::TransposeOp>(function), 0u);
  EXPECT_EQ(count<mlir::linalg::GenericOp>(function), 0u);
  mlir::linalg::BatchMatmulOp normalized;
  function.walk(
      [&](mlir::linalg::BatchMatmulOp operation) { normalized = operation; });
  ASSERT_TRUE(normalized);

  mlir::OpBuilder builder(normalized);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets{
      builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0),
      builder.getIndexAttr(0)};
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes{
      builder.getIndexAttr(2), builder.getIndexAttr(64),
      builder.getIndexAttr(128), builder.getIndexAttr(1024)};
  std::string failureReason;
  mlir::FailureOr<wafer::PartialReductionTileMaterialization> tiled =
      wafer::materializePartialReductionTile(normalized.getOperation(), builder,
                                             offsets, sizes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(tiled)) << failureReason;
  EXPECT_EQ(tiled->reductionDimensions, (llvm::SmallVector<int, 2>{3}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(StructuredGraphNormalizationTest,
       AlignedFifteenUseFanoutRemovesSharedInverseAccessWithoutComputeCopies) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    module {
      func.func @fanout(
          %arg0: tensor<2x1024x64xf32>) -> tensor<2x1024x64xf32> {
        %first_init = tensor.empty() : tensor<1024x2x64xf32>
        %first = linalg.transpose
            ins(%arg0 : tensor<2x1024x64xf32>)
            outs(%first_init : tensor<1024x2x64xf32>)
            permutation = [1, 0, 2]
        %second_init = tensor.empty() : tensor<2x1024x64xf32>
        %second = linalg.transpose
            ins(%first : tensor<1024x2x64xf32>)
            outs(%second_init : tensor<2x1024x64xf32>)
            permutation = [1, 0, 2]
        %result_init = tensor.empty() : tensor<2x1024x64xf32>
        %result = linalg.generic {
            indexing_maps = [#identity, #identity],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%second : tensor<2x1024x64xf32>)
            outs(%result_init : tensor<2x1024x64xf32>) {
          ^bb0(%input: f32, %output: f32):
            %negated = arith.negf %input : f32
            linalg.yield %negated : f32
        } -> tensor<2x1024x64xf32>
        return %result : tensor<2x1024x64xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("fanout");
  ASSERT_TRUE(function);
  llvm::SmallVector<mlir::linalg::TransposeOp, 2> transposes;
  mlir::linalg::GenericOp consumer;
  function.walk([&](mlir::linalg::TransposeOp operation) {
    transposes.push_back(operation);
  });
  function.walk(
      [&](mlir::linalg::GenericOp operation) { consumer = operation; });
  ASSERT_EQ(transposes.size(), 2u);
  ASSERT_TRUE(consumer);

  auto oldReturn = mlir::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  mlir::OpBuilder builder(oldReturn);
  llvm::SmallVector<mlir::Value, 15> results{consumer.getResult(0)};
  for (unsigned index = 1; index < 15; ++index)
    results.push_back(builder.clone(*consumer)->getResult(0));
  llvm::SmallVector<mlir::Type, 15> resultTypes(
      results.size(), consumer.getResult(0).getType());
  function.setType(
      builder.getFunctionType(function.getArgumentTypes(), resultTypes));
  builder.create<mlir::func::ReturnOp>(oldReturn.getLoc(), results);
  oldReturn.erase();
  ASSERT_EQ(llvm::range_size(transposes.back().getResult().getUses()), 15u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::StructuredGraphNormalizationStatistics statistics;
  mlir::FailureOr<wafer::StructuredGraphNormalizationOutcome> outcome =
      wafer::normalizeStructuredTensorGraph(function, {}, &statistics);
  ASSERT_TRUE(mlir::succeeded(outcome));
  EXPECT_EQ(*outcome, wafer::StructuredGraphNormalizationOutcome::Changed);
  EXPECT_EQ(statistics.accessTransformsRemoved, 2u);
  EXPECT_EQ(count<mlir::linalg::TransposeOp>(function), 0u);
  EXPECT_EQ(count<mlir::linalg::GenericOp>(function), 15u);
  EXPECT_EQ(count<mlir::arith::NegFOp>(function), 15u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

} // namespace
