//===- StructuredImplementationAlternativeTest.cpp ---------------------===//

#include "../../lib/Wafer/Compiler/AttentionImplementationAlternative.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/AttentionSemantics.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <string>

namespace {

using wafer::compiler::detail::AttentionImplementationAlternativeProvider;
using wafer::compiler::detail::buildStructuredAlternativeStructuralEstimates;
using wafer::compiler::detail::StructuredAlternativeDomain;
using wafer::compiler::detail::StructuredAlternativeParameters;
using wafer::compiler::detail::StructuredAlternativeTileShape;
using wafer::compiler::detail::StructuredImplementationAlternativePoints;
using wafer::compiler::detail::StructuredImplementationAlternativeQuery;

class StructuredImplementationAlternativeTest : public ::testing::Test {
protected:
  StructuredImplementationAlternativeTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseDecodeProgram() {
    std::string path = std::string(WAFER_TEST_SOURCE_DIR) +
                       "/test/Transforms/materialize-flash-decoding.mlir";
    return mlir::parseSourceFile<mlir::ModuleOp>(
        path, mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseOfficialHFPrefillProgram() {
    std::string path =
        std::string(WAFER_TEST_SOURCE_DIR) +
        "/unittests/Compiler/Inputs/official-hf-attention-prefill-linalg.mlir";
    return mlir::parseSourceFile<mlir::ModuleOp>(
        path, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

template <typename OpTy> static unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](OpTy) { ++count; });
  return count;
}

static bool hasResultTensorShape(mlir::ModuleOp module,
                                 llvm::ArrayRef<int64_t> shape) {
  bool found = false;
  module.walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults()) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
      if (type && type.hasStaticShape() && llvm::equal(type.getShape(), shape))
        found = true;
    }
  });
  return found;
}

static const wafer::compiler::detail::StructuredImplementationAlternativePoint *
findPoint(StructuredImplementationAlternativePoints &points,
          llvm::ArrayRef<int64_t> outputTileSizes,
          llvm::ArrayRef<int64_t> reductionTileSizes) {
  auto point = llvm::find_if(points, [&](const auto &candidate) {
    const StructuredAlternativeParameters &parameters =
        candidate->getParameters();
    return llvm::equal(parameters.outputTileSizes, outputTileSizes) &&
           llvm::equal(parameters.reductionTileSizes, reductionTileSizes) &&
           parameters.parallelPartitionCount == 1;
  });
  return point == points.end() ? nullptr : point->get();
}

static unsigned countCompilerOwnedDDRAllocations(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](mlir::memref::AllocOp allocation) {
    count += wafer::isWaferDDRMemRefType(allocation.getType()) ? 1u : 0u;
  });
  return count;
}

class CoveredTopLevelMatmulPoint final
    : public wafer::compiler::detail::StructuredImplementationAlternativePoint {
public:
  CoveredTopLevelMatmulPoint()
      : StructuredImplementationAlternativePoint(
            {/*stableKey=*/"test.covered-top-level-matmul",
             /*stableOrdinal=*/0},
            {/*outputShape=*/{2, 4}, /*reductionShape=*/{}},
            {/*outputTileSizes=*/{2, 4}, /*reductionTileSizes=*/{},
             /*parallelPartitionCount=*/1},
            {/*logicalOutputElements=*/8,
             /*logicalReductionElements=*/1,
             /*outputTileCount=*/1,
             /*reductionTileCount=*/1,
             /*parallelPartitionCount=*/1,
             /*structuredWorkUnitUpperBound=*/1}) {}

  mlir::LogicalResult
  materialize(mlir::ModuleOp isolatedStructuredModule,
              std::string *failureReason = nullptr,
              wafer::compiler::detail::
                  StructuredImplementationAlternativeMaterialization *result =
                      nullptr) const final {
    if (result)
      *result = {};
    mlir::func::FuncOp function =
        isolatedStructuredModule.lookupSymbol<mlir::func::FuncOp>("main");
    if (!function || !function.getBody().hasOneBlock()) {
      if (failureReason)
        *failureReason = "covered matmul point requires one main entry block";
      return mlir::failure();
    }
    unsigned covered = 0;
    for (mlir::Operation &operation :
         function.getBody().front().without_terminator()) {
      if (!mlir::isa<mlir::linalg::FillOp, mlir::linalg::MatmulOp>(operation))
        continue;
      ++covered;
      if (result)
        result->coveredTopLevelOperations.push_back(&operation);
    }
    if (covered == 2)
      return mlir::success();
    if (failureReason)
      *failureReason = "covered matmul point did not find its complete graph";
    return mlir::failure();
  }
};

static void appendNonViewResultConsumer(mlir::ModuleOp module) {
  mlir::func::FuncOp function = module.lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(function);
  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(function.getBody().front().back());
  mlir::Value source = returnOp.getOperand(0);
  auto type = mlir::cast<mlir::RankedTensorType>(source.getType());
  mlir::OpBuilder builder(returnOp);
  mlir::Value init =
      builder
          .create<mlir::tensor::EmptyOp>(returnOp.getLoc(), type.getShape(),
                                         type.getElementType())
          .getResult();
  mlir::AffineMap identity = mlir::AffineMap::getMultiDimIdentityMap(
      type.getRank(), module.getContext());
  llvm::SmallVector<mlir::utils::IteratorType, 4> iterators(
      type.getRank(), mlir::utils::IteratorType::parallel);
  auto negate = builder.create<mlir::linalg::GenericOp>(
      returnOp.getLoc(), mlir::TypeRange{type}, mlir::ValueRange{source},
      mlir::ValueRange{init},
      llvm::ArrayRef<mlir::AffineMap>{identity, identity}, iterators,
      [&](mlir::OpBuilder &nested, mlir::Location loc,
          mlir::ValueRange arguments) {
        mlir::Value value =
            nested.create<mlir::arith::NegFOp>(loc, arguments.front());
        nested.create<mlir::linalg::YieldOp>(loc, value);
      });
  returnOp.setOperand(0, negate.getResult(0));
}

TEST(StructuredImplementationAlternativeEstimateTest,
     DerivesOnlyOverflowSafeStructuralCounts) {
  StructuredAlternativeDomain domain;
  domain.outputShape = {2, 3};
  domain.reductionShape = {7};

  StructuredAlternativeParameters online;
  online.outputTileSizes = {1, 3};
  online.reductionTileSizes = {3};
  auto onlineEstimate =
      buildStructuredAlternativeStructuralEstimates(domain, online);
  ASSERT_TRUE(mlir::succeeded(onlineEstimate));
  EXPECT_EQ(onlineEstimate->logicalOutputElements, 6u);
  EXPECT_EQ(onlineEstimate->logicalReductionElements, 7u);
  EXPECT_EQ(onlineEstimate->outputTileCount, 2u);
  EXPECT_EQ(onlineEstimate->reductionTileCount, 3u);
  EXPECT_EQ(onlineEstimate->parallelPartitionCount, 1u);
  EXPECT_EQ(onlineEstimate->structuredWorkUnitUpperBound, 6u);
  EXPECT_FALSE(onlineEstimate->estimatedPeakLiveBytes.has_value());
  EXPECT_FALSE(onlineEstimate->estimatedComputeScalarOps.has_value());

  StructuredAlternativeParameters split = online;
  split.parallelPartitionCount = 2;
  auto splitEstimate =
      buildStructuredAlternativeStructuralEstimates(domain, split);
  ASSERT_TRUE(mlir::succeeded(splitEstimate));
  EXPECT_EQ(splitEstimate->structuredWorkUnitUpperBound, 8u);

  StructuredAlternativeParameters invalid = online;
  invalid.outputTileSizes = {3, 3};
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(buildStructuredAlternativeStructuralEstimates(
      domain, invalid, &failureReason)));
  EXPECT_NE(failureReason.find("within the domain"), std::string::npos);
}

TEST_F(StructuredImplementationAlternativeTest,
       ProviderCoveragePreventsCommonTraversalFromRetilingCompleteGraph) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(%lhs: tensor<2x3xf16>, %rhs: tensor<3x4xf16>)
      -> tensor<2x4xf16> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<2x4xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<2x4xf16>) -> tensor<2x4xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<2x3xf16>, tensor<3x4xf16>)
        outs(%init : tensor<2x4xf16>) -> tensor<2x4xf16>
    return %result : tensor<2x4xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);

  CoveredTopLevelMatmulPoint point;
  std::string failureReason;
  auto tile = wafer::compiler::detail::
      materializeStructuredImplementationAlternativeToTileRegion(
          *source, point, /*logicalRank=*/0, &failureReason);
  ASSERT_TRUE(mlir::succeeded(tile)) << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**tile)));
  EXPECT_EQ(countOps<mlir::scf::ForOp>(**tile), 0u);
  EXPECT_EQ(countOps<wafer::ComputeGemmOp>(**tile), 1u);
}

TEST_F(StructuredImplementationAlternativeTest,
       SemanticNonMatchIsSuccessWithNoPoints) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @pointwise(%input: tensor<4xf16>) -> tensor<4xf16> {
    return %input : tensor<4xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);

  AttentionImplementationAlternativeProvider provider;
  StructuredImplementationAlternativePoints points;
  StructuredImplementationAlternativeQuery query;
  std::string failureReason;
  EXPECT_TRUE(
      mlir::succeeded(provider.query(*module, query, points, &failureReason)))
      << failureReason;
  EXPECT_TRUE(points.empty());
  EXPECT_TRUE(failureReason.empty());
}

TEST_F(StructuredImplementationAlternativeTest,
       PrecomputedScoreInputDoesNotAdvertiseFlashImplementation) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseDecodeProgram();
  ASSERT_TRUE(source);
  mlir::FailureOr<wafer::tensor_program_to_tile_region::AttentionSemantics>
      semantics =
          wafer::tensor_program_to_tile_region::analyzeAttentionSemantics(
              *source);
  ASSERT_TRUE(mlir::succeeded(semantics));
  auto scoreResult = mlir::dyn_cast<mlir::OpResult>(semantics->scores);
  ASSERT_TRUE(scoreResult);

  mlir::func::FuncOp function;
  source->walk([&](mlir::func::FuncOp candidate) { function = candidate; });
  ASSERT_TRUE(function);
  function.insertArgument(/*index=*/0, semantics->scoreType,
                          /*argAttrs=*/{}, function.getLoc());
  scoreResult.replaceAllUsesWith(function.getArgument(0));
  if (scoreResult.getOwner()->use_empty())
    scoreResult.getOwner()->erase();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  AttentionImplementationAlternativeProvider provider;
  StructuredImplementationAlternativePoints points;
  StructuredImplementationAlternativeQuery query;
  std::string failureReason;
  EXPECT_TRUE(
      mlir::succeeded(provider.query(*source, query, points, &failureReason)))
      << failureReason;
  EXPECT_TRUE(points.empty());
  EXPECT_TRUE(failureReason.empty());
}

TEST_F(StructuredImplementationAlternativeTest,
       CanonicalizesPointIdentityAcrossQueryAxisOrder) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseDecodeProgram();
  ASSERT_TRUE(source);

  llvm::SmallVector<StructuredAlternativeTileShape, 2> outputForward{{1, 3},
                                                                     {2, 3}};
  llvm::SmallVector<StructuredAlternativeTileShape, 2> outputReverse{{2, 3},
                                                                     {1, 3}};
  llvm::SmallVector<StructuredAlternativeTileShape, 2> reductionForward{{3},
                                                                        {7}};
  llvm::SmallVector<StructuredAlternativeTileShape, 2> reductionReverse{{7},
                                                                        {3}};
  llvm::SmallVector<int64_t, 2> partitionsForward{2, 3};
  llvm::SmallVector<int64_t, 2> partitionsReverse{3, 2};

  AttentionImplementationAlternativeProvider provider;
  StructuredImplementationAlternativePoints forward;
  StructuredImplementationAlternativePoints reverse;
  ASSERT_TRUE(mlir::succeeded(provider.query(
      *source, {outputForward, reductionForward, partitionsForward}, forward)));
  ASSERT_TRUE(mlir::succeeded(provider.query(
      *source, {outputReverse, reductionReverse, partitionsReverse}, reverse)));
  ASSERT_EQ(forward.size(), 12u);
  ASSERT_EQ(reverse.size(), forward.size());
  for (size_t index = 0; index < forward.size(); ++index) {
    EXPECT_EQ(forward[index]->getIdentity().stableKey,
              reverse[index]->getIdentity().stableKey);
    EXPECT_EQ(forward[index]->getIdentity().stableOrdinal,
              static_cast<uint64_t>(index));
    EXPECT_EQ(reverse[index]->getIdentity().stableOrdinal,
              static_cast<uint64_t>(index));
  }
}

TEST_F(StructuredImplementationAlternativeTest,
       DecodeQueryEnumeratesTypedOnlineAndSplitPointsAndMaterializesClone) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseDecodeProgram();
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  llvm::SmallVector<StructuredAlternativeTileShape, 2> outputTiles;
  outputTiles.push_back({1, 3});
  outputTiles.push_back({1, 3});
  llvm::SmallVector<StructuredAlternativeTileShape, 2> reductionTiles;
  reductionTiles.push_back({3});
  reductionTiles.push_back({3});
  llvm::SmallVector<int64_t, 2> partitions{2, 2};
  StructuredImplementationAlternativeQuery query{outputTiles, reductionTiles,
                                                 partitions};

  AttentionImplementationAlternativeProvider provider;
  EXPECT_EQ(provider.getStableKey(), "wafer.structured-attention");
  StructuredImplementationAlternativePoints points;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(provider.query(*source, query, points, &failureReason)))
      << failureReason;
  ASSERT_EQ(points.size(), 8u);
  for (auto [ordinal, point] : llvm::enumerate(points))
    EXPECT_EQ(point->getIdentity().stableOrdinal,
              static_cast<uint64_t>(ordinal));

  const wafer::compiler::detail::StructuredImplementationAlternativePoint
      *online = nullptr;
  const wafer::compiler::detail::StructuredImplementationAlternativePoint
      *split = nullptr;
  bool hasFullDomainOnline = false;
  bool hasFullDomainSplit = false;
  for (const auto &point : points) {
    EXPECT_EQ(point->getDomain().outputShape,
              (llvm::SmallVector<int64_t, 4>{2, 3}));
    EXPECT_EQ(point->getDomain().reductionShape,
              (llvm::SmallVector<int64_t, 2>{7}));
    const auto &parameters = point->getParameters();
    const bool requestedTiles =
        parameters.outputTileSizes == (llvm::SmallVector<int64_t, 4>{1, 3}) &&
        parameters.reductionTileSizes == (llvm::SmallVector<int64_t, 2>{3});
    const bool fullDomain =
        parameters.outputTileSizes == (llvm::SmallVector<int64_t, 4>{2, 3}) &&
        parameters.reductionTileSizes == (llvm::SmallVector<int64_t, 2>{7});
    if (requestedTiles && parameters.parallelPartitionCount == 1)
      online = point.get();
    if (requestedTiles && parameters.parallelPartitionCount == 2)
      split = point.get();
    hasFullDomainOnline |= fullDomain && parameters.parallelPartitionCount == 1;
    hasFullDomainSplit |= fullDomain && parameters.parallelPartitionCount == 2;
  }
  ASSERT_NE(online, nullptr);
  ASSERT_NE(split, nullptr);
  EXPECT_TRUE(hasFullDomainOnline);
  EXPECT_TRUE(hasFullDomainSplit);
  EXPECT_EQ(online->getStructuralEstimates().structuredWorkUnitUpperBound, 6u);
  EXPECT_EQ(split->getStructuralEstimates().structuredWorkUnitUpperBound, 8u);
  ASSERT_TRUE(
      online->getStructuralEstimates().estimatedPeakLiveBytes.has_value());
  ASSERT_TRUE(
      split->getStructuralEstimates().estimatedPeakLiveBytes.has_value());
  ASSERT_TRUE(
      online->getStructuralEstimates().estimatedComputeScalarOps.has_value());
  ASSERT_TRUE(
      split->getStructuralEstimates().estimatedComputeScalarOps.has_value());
  EXPECT_EQ(*online->getStructuralEstimates().estimatedPeakLiveBytes, 64u);
  EXPECT_EQ(*split->getStructuralEstimates().estimatedPeakLiveBytes, 74u);
  EXPECT_EQ(*online->getStructuralEstimates().estimatedComputeScalarOps, 224u);
  EXPECT_EQ(*split->getStructuralEstimates().estimatedComputeScalarOps, 232u);

  mlir::OwningOpRef<mlir::ModuleOp> onlineClone =
      mlir::cast<mlir::ModuleOp>(source->getOperation()->clone());
  ASSERT_TRUE(
      mlir::succeeded(online->materialize(*onlineClone, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*onlineClone)));
  // This source computes scores from Q and the updated K cache.  A provider
  // point is only useful as FlashAttention when the score producer is fused
  // into the K/V-block recurrence instead of leaving a full score tensor
  // beside an online-softmax rewrite.
  EXPECT_FALSE(hasResultTensorShape(*onlineClone, {2, 7}));

  mlir::OwningOpRef<mlir::ModuleOp> isolated =
      mlir::cast<mlir::ModuleOp>(source->getOperation()->clone());
  ASSERT_TRUE(mlir::succeeded(split->materialize(*isolated, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*isolated)));
  unsigned logarithms = 0;
  isolated->walk([&](mlir::math::LogOp) { ++logarithms; });
  EXPECT_EQ(logarithms, 2u);
  EXPECT_FALSE(hasResultTensorShape(*isolated, {2, 7}));

  // The provider owns no source Operation pointers and querying/materializing
  // the disposable clone must leave the source capability query unchanged.
  StructuredImplementationAlternativePoints repeated;
  ASSERT_TRUE(
      mlir::succeeded(provider.query(*source, query, repeated, &failureReason)))
      << failureReason;
  ASSERT_EQ(repeated.size(), points.size());
  for (size_t index = 0; index < points.size(); ++index) {
    EXPECT_EQ(repeated[index]->getIdentity().stableKey,
              points[index]->getIdentity().stableKey);
    EXPECT_EQ(repeated[index]->getIdentity().stableOrdinal,
              points[index]->getIdentity().stableOrdinal);
  }
}

TEST_F(StructuredImplementationAlternativeTest,
       ScalarSeedsIncludeCompleteValueChannelAttentionPoints) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseDecodeProgram();
  ASSERT_TRUE(source);

  llvm::SmallVector<int64_t, 1> outputSeeds{1};
  llvm::SmallVector<int64_t, 1> reductionSeeds{3};
  StructuredImplementationAlternativeQuery query;
  query.outputTileSizeSeeds = outputSeeds;
  query.reductionTileSizeSeeds = reductionSeeds;
  AttentionImplementationAlternativeProvider provider;
  StructuredImplementationAlternativePoints points;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(provider.query(*source, query, points, &failureReason)))
      << failureReason;
  EXPECT_NE(findPoint(points, {1, 3}, {3}), nullptr);
  EXPECT_NE(findPoint(points, {1, 1}, {3}), nullptr);
}

TEST_F(StructuredImplementationAlternativeTest,
       OfficialHFPrefillFusesThroughSingleTileDimensions) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseOfficialHFPrefillProgram();
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  llvm::SmallVector<StructuredAlternativeTileShape, 1> outputTiles{
      {1, 1, 64, 64}};
  llvm::SmallVector<StructuredAlternativeTileShape, 1> reductionTiles{{64}};
  AttentionImplementationAlternativeProvider provider;
  StructuredImplementationAlternativePoints points;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(provider.query(
      *source, {outputTiles, reductionTiles, {}}, points, &failureReason)))
      << failureReason;
  const auto *point =
      findPoint(points, outputTiles.front(), reductionTiles.front());
  ASSERT_NE(point, nullptr);

  mlir::OwningOpRef<mlir::ModuleOp> materialized =
      mlir::cast<mlir::ModuleOp>(source->getOperation()->clone());
  ASSERT_TRUE(
      mlir::succeeded(point->materialize(*materialized, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*materialized)));

  // The rank-3 QK contraction is the full score producer in the exact HF
  // reshape topology. It must disappear with the dead probability closure.
  EXPECT_FALSE(hasResultTensorShape(*materialized, {1, 1024, 1024}));
  EXPECT_EQ(countCompilerOwnedDDRAllocations(*materialized), 0u);

  unsigned loopCount = 0;
  bool hasSingleTripLoop = false;
  bool hasDynamicBounds = false;
  materialized->walk([&](mlir::scf::ForOp loop) {
    ++loopCount;
    std::optional<int64_t> lower =
        mlir::getConstantIntValue(loop.getLowerBound());
    std::optional<int64_t> upper =
        mlir::getConstantIntValue(loop.getUpperBound());
    std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
    if (!lower || !upper || !step || *step <= 0) {
      hasDynamicBounds = true;
      return;
    }
    int64_t trips = (*upper - *lower + *step - 1) / *step;
    hasSingleTripLoop |= trips == 1;
  });
  EXPECT_FALSE(hasDynamicBounds);
  EXPECT_FALSE(hasSingleTripLoop);
  EXPECT_EQ(loopCount, 2u);
}

TEST_F(StructuredImplementationAlternativeTest,
       TypedComputePriorAccountsForValueChannelRecomputation) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseOfficialHFPrefillProgram();
  ASSERT_TRUE(source);

  llvm::SmallVector<StructuredAlternativeTileShape, 2> outputTiles{
      {1, 1, 64, 64}, {1, 1, 64, 32}};
  llvm::SmallVector<StructuredAlternativeTileShape, 1> reductionTiles{{64}};
  AttentionImplementationAlternativeProvider provider;
  StructuredImplementationAlternativePoints points;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(provider.query(
      *source, {outputTiles, reductionTiles, {}}, points, &failureReason)))
      << failureReason;
  const auto *completeValue = findPoint(points, outputTiles[0], {64});
  const auto *splitValue = findPoint(points, outputTiles[1], {64});
  ASSERT_NE(completeValue, nullptr);
  ASSERT_NE(splitValue, nullptr);
  ASSERT_TRUE(completeValue->getStructuralEstimates()
                  .estimatedComputeScalarOps.has_value());
  ASSERT_TRUE(splitValue->getStructuralEstimates()
                  .estimatedComputeScalarOps.has_value());
  EXPECT_LT(*completeValue->getStructuralEstimates().estimatedComputeScalarOps,
            *splitValue->getStructuralEstimates().estimatedComputeScalarOps);
  EXPECT_EQ(completeValue->getStructuralEstimates().logicalOutputElements,
            splitValue->getStructuralEstimates().logicalOutputElements);
}

TEST_F(StructuredImplementationAlternativeTest,
       ExactOutputViewUsesPublicBoundaryThroughOrdinaryTileLowering) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseOfficialHFPrefillProgram();
  ASSERT_TRUE(source);

  llvm::SmallVector<StructuredAlternativeTileShape, 1> outputTiles{
      {1, 1, 64, 64}};
  llvm::SmallVector<StructuredAlternativeTileShape, 1> reductionTiles{{64}};
  AttentionImplementationAlternativeProvider provider;
  StructuredImplementationAlternativePoints points;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(provider.query(
      *source, {outputTiles, reductionTiles, {}}, points, &failureReason)))
      << failureReason;
  const auto *point =
      findPoint(points, outputTiles.front(), reductionTiles.front());
  ASSERT_NE(point, nullptr);

  auto tile = wafer::compiler::detail::
      materializeStructuredImplementationAlternativeToTileRegion(
          *source, *point, /*logicalRank=*/0, &failureReason);
  ASSERT_TRUE(mlir::succeeded(tile)) << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**tile)));
}

TEST_F(StructuredImplementationAlternativeTest,
       NonViewOutputConsumerKeepsCompilerOwnedDestination) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseOfficialHFPrefillProgram();
  ASSERT_TRUE(source);
  appendNonViewResultConsumer(*source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  llvm::SmallVector<StructuredAlternativeTileShape, 1> outputTiles{
      {1, 1, 64, 64}};
  llvm::SmallVector<StructuredAlternativeTileShape, 1> reductionTiles{{64}};
  AttentionImplementationAlternativeProvider provider;
  StructuredImplementationAlternativePoints points;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(provider.query(
      *source, {outputTiles, reductionTiles, {}}, points, &failureReason)))
      << failureReason;
  const auto *point =
      findPoint(points, outputTiles.front(), reductionTiles.front());
  ASSERT_NE(point, nullptr);

  mlir::OwningOpRef<mlir::ModuleOp> materialized =
      mlir::cast<mlir::ModuleOp>(source->getOperation()->clone());
  ASSERT_TRUE(
      mlir::succeeded(point->materialize(*materialized, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*materialized)));
  EXPECT_EQ(countCompilerOwnedDDRAllocations(*materialized), 1u);

  auto tile = wafer::compiler::detail::
      materializeStructuredImplementationAlternativeToTileRegion(
          *source, *point, /*logicalRank=*/0, &failureReason);
  ASSERT_TRUE(mlir::succeeded(tile)) << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**tile)));
}

} // namespace
