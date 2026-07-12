#include "Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

void registerConversionDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
}

wafer::GroupOp findSingleGroup(mlir::ModuleOp module) {
  wafer::GroupOp found;
  module.walk([&](wafer::GroupOp group) {
    EXPECT_FALSE(found);
    found = group;
  });
  return found;
}

std::string formatStaticSlice(mlir::memref::SubViewOp subview) {
  std::string text;
  llvm::raw_string_ostream os(text);
  for (auto [index, offset] : llvm::enumerate(subview.getStaticOffsets())) {
    if (index)
      os << ',';
    os << offset;
  }
  os << ':';
  for (auto [index, size] : llvm::enumerate(subview.getStaticSizes())) {
    if (index)
      os << ',';
    os << size;
  }
  return text;
}

std::vector<std::string> collectStoreSlices(mlir::ModuleOp module) {
  std::vector<std::string> slices;
  module.walk([&](wafer::StorageStoreOp store) {
    auto subview = store.getDest().getDefiningOp<mlir::memref::SubViewOp>();
    EXPECT_TRUE(subview);
    if (subview)
      slices.push_back(formatStaticSlice(subview));
  });
  std::sort(slices.begin(), slices.end());
  return slices;
}

template <typename OpT> unsigned countOps(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](OpT) { ++count; });
  return count;
}

std::string collectGemmStoreOrder(mlir::ModuleOp module) {
  std::string order;
  module.walk([&](mlir::Operation *op) {
    if (mlir::isa<wafer::ComputeGemmOp>(op))
      order.push_back('G');
    if (mlir::isa<wafer::StorageStoreOp>(op))
      order.push_back('S');
  });
  return order;
}

std::string lowerF32ReductionAndGetFailure(llvm::StringRef combinerBody,
                                           bool splitReduction) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string sourceText = R"mlir(
module {
  func.func @reduce(%input: tensor<2x5xf32>, %out: tensor<2xf32>)
      -> tensor<2xf32> {
    %group = wafer.group ins(%input : tensor<2x5xf32>)
        outs(%out : tensor<2xf32>) {
    ^bb0(%input_arg: tensor<2x5xf32>, %out_arg: tensor<2xf32>):
      %zero = arith.constant 0.0 : f32
      %empty = tensor.empty() : tensor<2xf32>
      %init = linalg.fill ins(%zero : f32)
          outs(%empty : tensor<2xf32>) -> tensor<2xf32>
      %result = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0)>
          ],
          iterator_types = ["parallel", "reduction"]
        } ins(%input_arg : tensor<2x5xf32>)
          outs(%init : tensor<2xf32>) {
      ^bb0(%value: f32, %acc: f32):
)mlir";
  sourceText.append(combinerBody.begin(), combinerBody.end());
  sourceText += R"mlir(
        linalg.yield %combined : f32
      } -> tensor<2xf32>
      wafer.group.yield %result : tensor<2xf32>
    } : tensor<2xf32>
    return %group : tensor<2xf32>
  }
}
)mlir";

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(&context));
  if (!source)
    return "test source parse failed";
  wafer::GroupOp group = findSingleGroup(*source);
  if (!group)
    return "test source has no group";

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  mlir::LogicalResult result =
      splitReduction
          ? wafer::lowerCompleteCandidateGroupToTileRegionModule(
                group, /*candidateTileSizes=*/{1},
                /*candidateReductionTileSizes=*/{3}, lowered, &failureReason,
                /*currentLogicalRank=*/0)
          : wafer::lowerGroupToTileRegionModule(
                group, lowered, &failureReason, /*currentLogicalRank=*/0);
  if (mlir::succeeded(result))
    return "unexpected lowering success";
  return failureReason;
}

TEST(WaferGroupToTileRegionTest, MaterializesCompleteTwoByTwoTailTraversal) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @matmul(%lhs: tensor<5x6xf16>, %rhs: tensor<6x7xf16>,
                    %out: tensor<5x7xf16>) -> tensor<5x7xf16> {
    %group = wafer.group
        ins(%lhs, %rhs : tensor<5x6xf16>, tensor<6x7xf16>)
        outs(%out : tensor<5x7xf16>) {
    ^bb0(%lhs_arg: tensor<5x6xf16>, %rhs_arg: tensor<6x7xf16>,
         %out_arg: tensor<5x7xf16>):
      %zero = arith.constant 0.0 : f16
      %init = linalg.fill ins(%zero : f16)
          outs(%out_arg : tensor<5x7xf16>) -> tensor<5x7xf16>
      %result = linalg.matmul
          ins(%lhs_arg, %rhs_arg : tensor<5x6xf16>, tensor<6x7xf16>)
          outs(%init : tensor<5x7xf16>) -> tensor<5x7xf16>
      wafer.group.yield %result : tensor<5x7xf16>
    } : tensor<5x7xf16>
    return %group : tensor<5x7xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  wafer::GroupOp group = findSingleGroup(*source);
  ASSERT_TRUE(group);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(wafer::lowerCompleteCandidateGroupToTileRegionModule(
          group, /*candidateTileSizes=*/{3, 4},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::ComputeGemmOp>(*lowered), 4u);
  EXPECT_EQ(
      collectStoreSlices(*lowered),
      (std::vector<std::string>{"0,0:3,4", "0,4:3,3", "3,0:2,4", "3,4:2,3"}));
  EXPECT_EQ(collectGemmStoreOrder(*lowered), "GSGSGSGS");

  mlir::PassManager pm(&context);
  wafer::buildLowerGroupsToInstrPipeline(pm, /*logicalRank=*/0);
  ASSERT_TRUE(mlir::succeeded(pm.run(*lowered)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::InstrGemmOp>(*lowered), 4u);
  EXPECT_EQ(countOps<wafer::InstrWDMAOp>(*lowered), 4u);
}

TEST(WaferGroupToTileRegionTest, AppliesReductionSplitToEveryOutputTile) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce(%input: tensor<2x5xf32>, %out: tensor<2xf32>)
      -> tensor<2xf32> {
    %group = wafer.group ins(%input : tensor<2x5xf32>)
        outs(%out : tensor<2xf32>) {
    ^bb0(%input_arg: tensor<2x5xf32>, %out_arg: tensor<2xf32>):
      %zero = arith.constant 0.0 : f32
      %empty = tensor.empty() : tensor<2xf32>
      %init = linalg.fill ins(%zero : f32)
          outs(%empty : tensor<2xf32>) -> tensor<2xf32>
      %result = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0)>
          ],
          iterator_types = ["parallel", "reduction"]
        } ins(%input_arg : tensor<2x5xf32>)
          outs(%init : tensor<2xf32>) {
      ^bb0(%value: f32, %acc: f32):
        %sum = arith.addf %value, %acc : f32
        linalg.yield %sum : f32
      } -> tensor<2xf32>
      wafer.group.yield %result : tensor<2xf32>
    } : tensor<2xf32>
    return %group : tensor<2xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  wafer::GroupOp group = findSingleGroup(*source);
  ASSERT_TRUE(group);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(wafer::lowerCompleteCandidateGroupToTileRegionModule(
          group, /*candidateTileSizes=*/{1},
          /*candidateReductionTileSizes=*/{3}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::ComputeReduceOp>(*lowered), 4u);
  EXPECT_EQ(collectStoreSlices(*lowered),
            (std::vector<std::string>{"0:1", "1:1"}));

  mlir::PassManager pm(&context);
  wafer::buildLowerGroupsToInstrPipeline(pm, /*logicalRank=*/0);
  ASSERT_TRUE(mlir::succeeded(pm.run(*lowered)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::InstrReduceOp>(*lowered), 4u);
  EXPECT_EQ(countOps<wafer::InstrWDMAOp>(*lowered), 2u);
}

TEST(WaferGroupToTileRegionTest,
     ReusesSPMAcrossCompletedElementwiseTraversalTiles) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @add(%lhs: tensor<2x100001xi64>,
                 %rhs: tensor<2x100001xi64>,
                 %out: tensor<2x100001xi64>) -> tensor<2x100001xi64> {
    %group = wafer.group
        ins(%lhs, %rhs : tensor<2x100001xi64>, tensor<2x100001xi64>)
        outs(%out : tensor<2x100001xi64>) {
    ^bb0(%lhs_arg: tensor<2x100001xi64>,
         %rhs_arg: tensor<2x100001xi64>,
         %out_arg: tensor<2x100001xi64>):
      %result = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%lhs_arg, %rhs_arg
              : tensor<2x100001xi64>, tensor<2x100001xi64>)
          outs(%out_arg : tensor<2x100001xi64>) {
        ^bb0(%left: i64, %right: i64, %init: i64):
          %sum = arith.addi %left, %right : i64
          linalg.yield %sum : i64
        } -> tensor<2x100001xi64>
      wafer.group.yield %result : tensor<2x100001xi64>
    } : tensor<2x100001xi64>
    return %group : tensor<2x100001xi64>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  wafer::GroupOp group = findSingleGroup(*source);
  ASSERT_TRUE(group);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(wafer::lowerCompleteCandidateGroupToTileRegionModule(
          group, /*candidateTileSizes=*/{1, 50001},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  EXPECT_EQ(countOps<wafer::SyncLocalFenceOp>(*lowered), 4u);

  mlir::LogicalResult planned = wafer::planSPMMemoryModule(
      *lowered, /*spmBase=*/65536, /*spmLimit=*/3080192,
      /*spmAlignment=*/256);
  if (mlir::failed(planned))
    lowered->print(llvm::errs());
  EXPECT_TRUE(mlir::succeeded(planned));
}

TEST(WaferGroupToTileRegionTest, RejectsUntiledLinalgProducerChains) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @producer_chain(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                            %out: tensor<4xf32>) -> tensor<4xf32> {
    %group = wafer.group
        ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
        outs(%out : tensor<4xf32>) {
    ^bb0(%lhs_arg: tensor<4xf32>, %rhs_arg: tensor<4xf32>,
         %out_arg: tensor<4xf32>):
      %empty = tensor.empty() : tensor<4xf32>
      %producer = linalg.map
          ins(%lhs_arg : tensor<4xf32>)
          outs(%empty : tensor<4xf32>)
          (%value: f32) {
            %doubled = arith.addf %value, %value : f32
            linalg.yield %doubled : f32
          }
      %result = linalg.map
          ins(%producer, %rhs_arg : tensor<4xf32>, tensor<4xf32>)
          outs(%out_arg : tensor<4xf32>)
          (%left: f32, %right: f32) {
            %sum = arith.addf %left, %right : f32
            linalg.yield %sum : f32
          }
      wafer.group.yield %result : tensor<4xf32>
    } : tensor<4xf32>
    return %group : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  wafer::GroupOp group = findSingleGroup(*source);
  ASSERT_TRUE(group);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(wafer::lowerCompleteCandidateGroupToTileRegionModule(
      group, /*candidateTileSizes=*/{2},
      /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
      /*currentLogicalRank=*/0)));
  EXPECT_EQ(failureReason,
            "complete candidate traversal does not support tensor producer "
            "chains");
}

TEST(WaferGroupToTileRegionTest, RejectsUnsignedReductionSplitSemantics) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce_unsigned(%input: tensor<2x5xi32>, %out: tensor<2xi32>)
      -> tensor<2xi32> {
    %group = wafer.group ins(%input : tensor<2x5xi32>)
        outs(%out : tensor<2xi32>) {
    ^bb0(%input_arg: tensor<2x5xi32>, %out_arg: tensor<2xi32>):
      %zero = arith.constant 0 : i32
      %empty = tensor.empty() : tensor<2xi32>
      %init = linalg.fill ins(%zero : i32)
          outs(%empty : tensor<2xi32>) -> tensor<2xi32>
      %result = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0)>
          ],
          iterator_types = ["parallel", "reduction"]
        } ins(%input_arg : tensor<2x5xi32>)
          outs(%init : tensor<2xi32>) {
      ^bb0(%value: i32, %acc: i32):
        %max = arith.maxui %value, %acc : i32
        linalg.yield %max : i32
      } -> tensor<2xi32>
      wafer.group.yield %result : tensor<2xi32>
    } : tensor<2xi32>
    return %group : tensor<2xi32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  wafer::GroupOp group = findSingleGroup(*source);
  ASSERT_TRUE(group);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(wafer::lowerCompleteCandidateGroupToTileRegionModule(
      group, /*candidateTileSizes=*/{1},
      /*candidateReductionTileSizes=*/{3}, lowered, &failureReason,
      /*currentLogicalRank=*/0)));
  EXPECT_EQ(failureReason,
            "candidate reduction split cannot preserve unsigned min/max "
            "semantics with the current reduce kind");
}

TEST(WaferGroupToTileRegionTest, RejectsReductionThatIgnoresAccumulator) {
  EXPECT_EQ(lowerF32ReductionAndGetFailure(
                "        %combined = arith.addf %value, %value : f32\n",
                /*splitReduction=*/false),
            "linalg.generic reduction requires one exact combiner wired to "
            "the reduced value and accumulator");
}

TEST(WaferGroupToTileRegionTest, RejectsMaxNumReductionSplitSemantics) {
  EXPECT_EQ(lowerF32ReductionAndGetFailure(
                "        %combined = arith.maxnumf %value, %acc : f32\n",
                /*splitReduction=*/true),
            "candidate reduction split cannot preserve maxnum/minnum NaN "
            "semantics with the current reduce kind");
}

TEST(WaferGroupToTileRegionTest,
     AcceptsExactEagerMaterializationBudgetWithCartesianProducts) {
  uint64_t outputTileCount = 0;
  EXPECT_EQ(wafer::detail::checkedStaticTileProduct(
                /*ranges=*/{127}, /*tileSizes=*/{2}, outputTileCount),
            wafer::detail::CheckedStaticTileProductStatus::Success);
  EXPECT_EQ(outputTileCount, 64u);

  uint64_t reductionChunkCount = 0;
  EXPECT_EQ(wafer::detail::checkedStaticTileProduct(
                /*ranges=*/{190}, /*tileSizes=*/{3}, reductionChunkCount),
            wafer::detail::CheckedStaticTileProductStatus::Success);
  EXPECT_EQ(reductionChunkCount, 64u);

  uint64_t materializationCount = 0;
  EXPECT_EQ(wafer::detail::checkCompleteCandidateExpansionBudget(
                outputTileCount, {reductionChunkCount}, materializationCount),
            wafer::detail::CompleteCandidateExpansionStatus::WithinBudget);
  EXPECT_EQ(materializationCount,
            wafer::detail::kCompleteCandidateMaterializationBudget);
}

TEST(WaferGroupToTileRegionTest, RejectsOnePastEagerMaterializationBudget) {
  uint64_t outputTileCount = 0;
  EXPECT_EQ(wafer::detail::checkedStaticTileProduct(
                /*ranges=*/{8193}, /*tileSizes=*/{2}, outputTileCount),
            wafer::detail::CheckedStaticTileProductStatus::Success);
  EXPECT_EQ(outputTileCount,
            wafer::detail::kCompleteCandidateMaterializationBudget + 1);

  uint64_t materializationCount = 0;
  EXPECT_EQ(
      wafer::detail::checkCompleteCandidateExpansionBudget(
          outputTileCount, /*reductionChunkCounts=*/{1}, materializationCount),
      wafer::detail::CompleteCandidateExpansionStatus::BudgetExceeded);
  EXPECT_EQ(materializationCount,
            wafer::detail::kCompleteCandidateMaterializationBudget + 1);
}

TEST(WaferGroupToTileRegionTest, RejectsUnrepresentableCartesianProduct) {
  uint64_t tileCount = 0;
  EXPECT_EQ(wafer::detail::checkedStaticTileProduct(
                /*ranges=*/{std::numeric_limits<int64_t>::max(),
                            std::numeric_limits<int64_t>::max()},
                /*tileSizes=*/{1, 1}, tileCount),
            wafer::detail::CheckedStaticTileProductStatus::Overflow);

  uint64_t materializationCount = 0;
  EXPECT_EQ(wafer::detail::checkCompleteCandidateExpansionBudget(
                std::numeric_limits<uint64_t>::max(),
                /*reductionChunkCounts=*/{2}, materializationCount),
            wafer::detail::CompleteCandidateExpansionStatus::CountOverflow);
}

TEST(WaferGroupToTileRegionTest,
     RejectsUnrepresentableTraversalBeforeMaterialization) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @map(
      %input: tensor<4294967296x4294967296xf32>,
      %out: tensor<4294967296x4294967296xf32>)
      -> tensor<4294967296x4294967296xf32> {
    %group = wafer.group
        ins(%input : tensor<4294967296x4294967296xf32>)
        outs(%out : tensor<4294967296x4294967296xf32>) {
    ^bb0(%input_arg: tensor<4294967296x4294967296xf32>,
         %out_arg: tensor<4294967296x4294967296xf32>):
      %result = linalg.map
          ins(%input_arg : tensor<4294967296x4294967296xf32>)
          outs(%out_arg : tensor<4294967296x4294967296xf32>)
          (%value: f32) {
            linalg.yield %value : f32
          }
      wafer.group.yield %result : tensor<4294967296x4294967296xf32>
    } : tensor<4294967296x4294967296xf32>
    return %group : tensor<4294967296x4294967296xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  wafer::GroupOp group = findSingleGroup(*source);
  ASSERT_TRUE(group);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(wafer::lowerCompleteCandidateGroupToTileRegionModule(
      group, /*candidateTileSizes=*/{1, 1},
      /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
      /*currentLogicalRank=*/0)));
  EXPECT_EQ(failureReason,
            "complete candidate traversal output tile count is not "
            "representable");
  EXPECT_FALSE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

TEST(WaferGroupToTileRegionTest,
     EagerMaterializationBudgetFailurePreservesSourceAndOutput) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @map(%input: tensor<8193xf32>, %out: tensor<8193xf32>)
      -> tensor<8193xf32> {
    %group = wafer.group ins(%input : tensor<8193xf32>)
        outs(%out : tensor<8193xf32>) {
    ^bb0(%input_arg: tensor<8193xf32>, %out_arg: tensor<8193xf32>):
      %result = linalg.map
          ins(%input_arg : tensor<8193xf32>)
          outs(%out_arg : tensor<8193xf32>)
          (%value: f32) {
            linalg.yield %value : f32
          }
      wafer.group.yield %result : tensor<8193xf32>
    } : tensor<8193xf32>
    return %group : tensor<8193xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  wafer::GroupOp group = findSingleGroup(*source);
  ASSERT_TRUE(group);

  auto sentinel = mlir::parseSourceString<mlir::ModuleOp>(
      "module { func.func private @sentinel() }", mlir::ParserConfig(&context));
  ASSERT_TRUE(sentinel);
  mlir::OwningOpRef<mlir::ModuleOp> lowered = std::move(sentinel);
  mlir::Operation *sentinelOperation = lowered->getOperation();

  std::string sourceBefore;
  llvm::raw_string_ostream sourceBeforeStream(sourceBefore);
  source->print(sourceBeforeStream);

  std::string failureReason;
  EXPECT_TRUE(mlir::failed(wafer::lowerCompleteCandidateGroupToTileRegionModule(
      group, /*candidateTileSizes=*/{2},
      /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
      /*currentLogicalRank=*/0)));
  EXPECT_EQ(failureReason,
            "complete candidate traversal exceeds the eager materialization "
            "budget; this is an implementation resource limit, not an IR or "
            "target legality restriction");
  EXPECT_EQ(lowered->getOperation(), sentinelOperation);

  std::string sourceAfter;
  llvm::raw_string_ostream sourceAfterStream(sourceAfter);
  source->print(sourceAfterStream);
  EXPECT_EQ(sourceAfter, sourceBefore);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

} // namespace
