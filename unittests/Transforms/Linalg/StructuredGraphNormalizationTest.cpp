//===- StructuredGraphNormalizationTest.cpp -----------------------------===//

#include "Wafer/Transforms/Linalg/StructuredGraphNormalization.h"
#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/Analysis/Linalg/StructuredDAGAnalysis.h"
#include "Wafer/Transforms/Linalg/StructuredTiling.h"

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
          %input: tensor<2x4x64x1031xf32>, %init: tensor<2x4x64xf32>)
          -> tensor<2x4x64xf32> {
        %expanded = tensor.expand_shape %input [[0], [1], [2], [3, 4]]
            output_shape [2, 4, 64, 1, 1031] :
            tensor<2x4x64x1031xf32> into tensor<2x4x64x1x1031xf32>
        %reshaped = tensor.collapse_shape %expanded [[0], [1], [2], [3, 4]] :
            tensor<2x4x64x1x1031xf32> into tensor<2x4x64x1031xf32>
        %input_empty = tensor.empty() : tensor<2x64x4x1031xf32>
        %input_transposed = linalg.transpose
            ins(%reshaped : tensor<2x4x64x1031xf32>)
            outs(%input_empty : tensor<2x64x4x1031xf32>)
            permutation = [0, 2, 1, 3]
        %init_empty = tensor.empty() : tensor<2x64x4xf32>
        %init_transposed = linalg.transpose
            ins(%init : tensor<2x4x64xf32>)
            outs(%init_empty : tensor<2x64x4xf32>) permutation = [0, 2, 1]
        %reduced = linalg.generic {
            indexing_maps = [#identity, #result],
            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%input_transposed : tensor<2x64x4x1031xf32>)
            outs(%init_transposed : tensor<2x64x4xf32>) {
          ^bb0(%element: f32, %accumulator: f32):
            %sum = arith.addf %accumulator, %element : f32
            linalg.yield %sum : f32
        } -> tensor<2x64x4xf32>
        %result_empty = tensor.empty() : tensor<2x4x64xf32>
        %result = linalg.transpose
            ins(%reduced : tensor<2x64x4xf32>)
            outs(%result_empty : tensor<2x4x64xf32>) permutation = [0, 2, 1]
        return %result : tensor<2x4x64xf32>
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
  EXPECT_EQ(statistics.budgetExhaustedComponents, 0u);
  EXPECT_GE(statistics.compositionApplications, 1u);
  EXPECT_GE(statistics.computeAbsorptionApplications, 1u);
  EXPECT_GE(statistics.resultReindexApplications, 1u);
  EXPECT_EQ(count<mlir::linalg::TransposeOp>(function), 0u);
  mlir::linalg::GenericOp normalized;
  function.walk(
      [&](mlir::linalg::GenericOp operation) { normalized = operation; });
  ASSERT_TRUE(normalized);
  EXPECT_EQ(normalized.getIteratorTypesArray()[3],
            mlir::utils::IteratorType::reduction);
  ASSERT_EQ(normalized.getIndexingMapsArray().size(), 2u);
  EXPECT_EQ(normalized.getIndexingMapsArray().front(),
            mlir::AffineMap::get(4, 0,
                                 {mlir::getAffineDimExpr(0, context.get()),
                                  mlir::getAffineDimExpr(2, context.get()),
                                  mlir::getAffineDimExpr(1, context.get()),
                                  mlir::getAffineDimExpr(3, context.get())},
                                 context.get()));
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
      builder.getIndexAttr(2), builder.getIndexAttr(64),
      builder.getIndexAttr(4), builder.getIndexAttr(1024)};
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
    #lhs = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
    #rhs = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
    #out = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
    module {
      func.func @reindex_contraction(
          %lhs: tensor<2x64x1031xf32>, %rhs: tensor<2x1031x128xf32>,
          %init: tensor<2x64x128xf32>) -> tensor<2x64x128xf32> {
        %lhs_expanded = tensor.expand_shape %lhs [[0], [1], [2, 3]]
            output_shape [2, 64, 1, 1031] :
            tensor<2x64x1031xf32> into tensor<2x64x1x1031xf32>
        %lhs_reshaped = tensor.collapse_shape %lhs_expanded [[0], [1], [2, 3]] :
            tensor<2x64x1x1031xf32> into tensor<2x64x1031xf32>
        %lhs_empty = tensor.empty() : tensor<2x1031x64xf32>
        %lhs_transposed = linalg.transpose
            ins(%lhs_reshaped : tensor<2x64x1031xf32>)
            outs(%lhs_empty : tensor<2x1031x64xf32>) permutation = [0, 2, 1]
        %init_empty = tensor.empty() : tensor<2x128x64xf32>
        %init_transposed = linalg.transpose
            ins(%init : tensor<2x64x128xf32>)
            outs(%init_empty : tensor<2x128x64xf32>) permutation = [0, 2, 1]
        %contracted = linalg.generic {
            indexing_maps = [#lhs, #rhs, #out],
            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%lhs_transposed, %rhs :
                tensor<2x1031x64xf32>, tensor<2x1031x128xf32>)
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
  EXPECT_EQ(statistics.budgetExhaustedComponents, 0u);
  EXPECT_GE(statistics.compositionApplications, 1u);
  EXPECT_GE(statistics.computeAbsorptionApplications, 1u);
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
       RaggedHeterogeneousFanoutUsesOneSourceWithoutComputeCopies) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
    #identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #reduce_last = affine_map<(d0, d1, d2) -> (d0, d1)>
    module {
      func.func @heterogeneous_fanout(
          %lhs: tensor<2x1031x64xf32>, %rhs: tensor<2x1031x128xf32>,
          %reduction_init: tensor<2x64xf32>,
          %contraction_init: tensor<2x64x128xf32>)
          -> (tensor<2x64x1031xf32>, tensor<2x64xf32>,
              tensor<2x64x128xf32>) {
        %shared_init = tensor.empty() : tensor<2x64x1031xf32>
        %shared = linalg.transpose
            ins(%lhs : tensor<2x1031x64xf32>)
            outs(%shared_init : tensor<2x64x1031xf32>)
            permutation = [0, 2, 1]
        %elementwise_init = tensor.empty() : tensor<2x64x1031xf32>
        %elementwise = linalg.generic {
            indexing_maps = [#identity3, #identity3],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%shared : tensor<2x64x1031xf32>)
            outs(%elementwise_init : tensor<2x64x1031xf32>) {
          ^bb0(%input: f32, %output: f32):
            %negated = arith.negf %input : f32
            linalg.yield %negated : f32
        } -> tensor<2x64x1031xf32>
        %reduced = linalg.generic {
            indexing_maps = [#identity3, #reduce_last],
            iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%shared : tensor<2x64x1031xf32>)
            outs(%reduction_init : tensor<2x64xf32>) {
          ^bb0(%input: f32, %accumulator: f32):
            %sum = arith.addf %accumulator, %input : f32
            linalg.yield %sum : f32
        } -> tensor<2x64xf32>
        %contracted = linalg.batch_matmul
            ins(%shared, %rhs :
                tensor<2x64x1031xf32>, tensor<2x1031x128xf32>)
            outs(%contraction_init : tensor<2x64x128xf32>)
            -> tensor<2x64x128xf32>
        return %elementwise, %reduced, %contracted :
            tensor<2x64x1031xf32>, tensor<2x64xf32>, tensor<2x64x128xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function =
      module->lookupSymbol<mlir::func::FuncOp>("heterogeneous_fanout");
  ASSERT_TRUE(function);

  wafer::StructuredGraphNormalizationStatistics statistics;
  mlir::FailureOr<wafer::StructuredGraphNormalizationOutcome> outcome =
      wafer::normalizeStructuredTensorGraph(function, {}, &statistics);
  ASSERT_TRUE(mlir::succeeded(outcome));
  EXPECT_EQ(*outcome, wafer::StructuredGraphNormalizationOutcome::Changed);
  EXPECT_EQ(statistics.multiUseAccessPropagations, 1u);
  EXPECT_EQ(statistics.accessTransformsRemoved, 1u);
  EXPECT_EQ(count<mlir::linalg::TransposeOp>(function), 0u);
  EXPECT_EQ(count<mlir::linalg::GenericOp>(function), 2u);
  EXPECT_EQ(count<mlir::arith::NegFOp>(function), 1u);

  mlir::linalg::GenericOp elementwise;
  mlir::linalg::GenericOp reduction;
  function.walk([&](mlir::linalg::GenericOp operation) {
    if (llvm::is_contained(operation.getIteratorTypesArray(),
                           mlir::utils::IteratorType::reduction))
      reduction = operation;
    else
      elementwise = operation;
  });
  ASSERT_TRUE(elementwise);
  ASSERT_TRUE(reduction);
  EXPECT_EQ(elementwise.getDpsInputs().front(), function.getArgument(0));
  EXPECT_EQ(reduction.getDpsInputs().front(), function.getArgument(0));
  EXPECT_EQ(reduction.getDpsInits().front(), function.getArgument(2));
  EXPECT_TRUE(llvm::equal(reduction.getIteratorTypesArray(),
                          llvm::ArrayRef<mlir::utils::IteratorType>{
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::reduction}));
  llvm::SmallVector<unsigned, 3> permutation{0, 2, 1};
  mlir::AffineMap swap =
      mlir::AffineMap::getPermutationMap(permutation, context.get());
  EXPECT_EQ(elementwise.getIndexingMapsArray().front(), swap);
  EXPECT_EQ(reduction.getIndexingMapsArray().front(), swap);

  mlir::linalg::BatchMatmulTransposeAOp contraction;
  function.walk([&](mlir::linalg::BatchMatmulTransposeAOp operation) {
    contraction = operation;
  });
  ASSERT_TRUE(contraction);
  EXPECT_EQ(contraction.getDpsInputs()[0], function.getArgument(0));
  EXPECT_EQ(contraction.getDpsInputs()[1], function.getArgument(1));
  EXPECT_EQ(contraction.getDpsInits().front(), function.getArgument(3));

  std::string failureReason;
  mlir::OpBuilder reductionBuilder(reduction);
  llvm::SmallVector<mlir::OpFoldResult, 3> reductionOffsets{
      reductionBuilder.getIndexAttr(0), reductionBuilder.getIndexAttr(0),
      reductionBuilder.getIndexAttr(0)};
  llvm::SmallVector<mlir::OpFoldResult, 3> reductionSizes{
      reductionBuilder.getIndexAttr(2), reductionBuilder.getIndexAttr(64),
      reductionBuilder.getIndexAttr(1024)};
  mlir::FailureOr<wafer::PartialReductionTileMaterialization> tiledReduction =
      wafer::materializePartialReductionTile(reduction.getOperation(),
                                             reductionBuilder, reductionOffsets,
                                             reductionSizes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(tiledReduction)) << failureReason;
  EXPECT_EQ(tiledReduction->reductionDimensions,
            (llvm::SmallVector<int, 2>{2}));

  mlir::OpBuilder contractionBuilder(contraction);
  llvm::SmallVector<mlir::OpFoldResult, 4> contractionOffsets{
      contractionBuilder.getIndexAttr(0), contractionBuilder.getIndexAttr(0),
      contractionBuilder.getIndexAttr(0), contractionBuilder.getIndexAttr(0)};
  llvm::SmallVector<mlir::OpFoldResult, 4> contractionSizes{
      contractionBuilder.getIndexAttr(2), contractionBuilder.getIndexAttr(64),
      contractionBuilder.getIndexAttr(128),
      contractionBuilder.getIndexAttr(1024)};
  mlir::FailureOr<wafer::PartialReductionTileMaterialization> tiledContraction =
      wafer::materializePartialReductionTile(
          contraction.getOperation(), contractionBuilder, contractionOffsets,
          contractionSizes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(tiledContraction)) << failureReason;
  EXPECT_EQ(tiledContraction->reductionDimensions,
            (llvm::SmallVector<int, 2>{3}));
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
  EXPECT_GE(statistics.multiUseAccessPropagations, 1u);
  EXPECT_EQ(statistics.accessTransformsRemoved, 2u);
  EXPECT_EQ(count<mlir::linalg::TransposeOp>(function), 0u);
  EXPECT_EQ(count<mlir::linalg::GenericOp>(function), 15u);
  EXPECT_EQ(count<mlir::arith::NegFOp>(function), 15u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(StructuredGraphNormalizationTest,
       RaggedAlternatingAccessChainsRemainBoundedThroughThirtyTwoComputes) {
  for (unsigned computeCount : {4u, 8u, 16u, 32u}) {
    SCOPED_TRACE("compute-count=" + std::to_string(computeCount));
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    module {
      func.func @deep_chain_)mlir"
           << computeCount << R"mlir((
          %arg0: tensor<2x1025x64xf32>) -> tensor<2x1025x64xf32> {
  )mlir";
    std::string current = "%arg0";
    std::string currentType = "tensor<2x1025x64xf32>";
    for (unsigned index = 0; index < computeCount; ++index) {
      std::string targetType =
          index % 2 == 0 ? "tensor<2x64x1025xf32>" : "tensor<2x1025x64xf32>";
      stream << "        %transpose_empty" << index
             << " = tensor.empty() : " << targetType << "\n";
      stream << "        %transpose" << index << " = linalg.transpose ins("
             << current << " : " << currentType << ") outs(%transpose_empty"
             << index << " : " << targetType << ") permutation = [0, 2, 1]\n";
      stream << "        %compute_empty" << index
             << " = tensor.empty() : " << targetType << "\n";
      stream << "        %compute" << index << R"mlir( = linalg.generic {
            indexing_maps = [#identity, #identity],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%transpose)mlir"
             << index << " : " << targetType << ") outs(%compute_empty" << index
             << " : " << targetType << R"mlir() {
          ^bb0(%input: f32, %output: f32):
            %negated = arith.negf %input : f32
            linalg.yield %negated : f32
        } -> )mlir"
             << targetType << "\n";
      current = "%compute" + std::to_string(index);
      currentType = targetType;
    }
    stream << "        return " << current << " : " << currentType << R"mlir(
      }
    }
  )mlir";
    stream.flush();

    mlir::OwningOpRef<mlir::ModuleOp> module = parse(source);
    ASSERT_TRUE(module) << source;
    auto function = module->lookupSymbol<mlir::func::FuncOp>(
        "deep_chain_" + std::to_string(computeCount));
    ASSERT_TRUE(function);
    ASSERT_EQ(count<mlir::linalg::TransposeOp>(function), computeCount);
    ASSERT_EQ(count<mlir::linalg::GenericOp>(function), computeCount);
    ASSERT_EQ(count<mlir::arith::NegFOp>(function), computeCount);

    wafer::StructuredGraphNormalizationStatistics statistics;
    mlir::FailureOr<wafer::StructuredGraphNormalizationOutcome> outcome =
        wafer::normalizeStructuredTensorGraph(function, {}, &statistics);
    ASSERT_TRUE(mlir::succeeded(outcome));
    EXPECT_EQ(*outcome, wafer::StructuredGraphNormalizationOutcome::Changed);
    EXPECT_EQ(statistics.budgetExhaustedComponents, 0u);
    EXPECT_EQ(count<mlir::linalg::TransposeOp>(function), 0u);
    EXPECT_EQ(count<mlir::linalg::GenericOp>(function), computeCount);
    EXPECT_EQ(count<mlir::arith::NegFOp>(function), computeCount);
    EXPECT_GE(statistics.accessTransformsRemoved, computeCount);
    EXPECT_LT(statistics.eNodes, computeCount * 16u);
    EXPECT_LT(statistics.relationQueries, computeCount * 40u);
    EXPECT_LT(statistics.rewriteMatches, computeCount * 40u);
    if (computeCount == 32) {
      RecordProperty("max_chain_input_operations", statistics.inputOperations);
      RecordProperty("max_chain_output_operations",
                     statistics.outputOperations);
      RecordProperty("max_chain_access_transforms_removed",
                     statistics.accessTransformsRemoved);
      RecordProperty("max_chain_e_nodes", statistics.eNodes);
      RecordProperty("max_chain_relation_queries", statistics.relationQueries);
      RecordProperty("max_chain_rewrite_matches", statistics.rewriteMatches);
    }

    std::string failureReason;
    mlir::FailureOr<wafer::compiler::detail::StructuredDAGAnalysis> dag =
        wafer::compiler::detail::StructuredDAGAnalysis::create(function,
                                                               &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    EXPECT_EQ(dag->getNodes().size(), computeCount);
    const std::string once = print(function);
    mlir::FailureOr<wafer::StructuredGraphNormalizationOutcome> second =
        wafer::normalizeStructuredTensorGraph(function);
    ASSERT_TRUE(mlir::succeeded(second));
    EXPECT_EQ(*second, wafer::StructuredGraphNormalizationOutcome::Unchanged);
    EXPECT_EQ(print(function), once);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

} // namespace
