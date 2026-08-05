//===- RankTileMaterializationTest.cpp - Complete-rank Tile boundary ----===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

template <typename OpTy>
unsigned countOperations(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](OpTy) { ++count; });
  return count;
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os);
  return text;
}

TEST(RankTileMaterializationTest,
     PreservesCompleteRankAcrossDependentStructuredBoundaries) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<4xf16>) -> tensor<1x4xf16> {
    %producer_out = tensor.empty() : tensor<4xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf16>) outs(%producer_out : tensor<4xf16>) {
    ^bb0(%value: f16, %old: f16):
      linalg.yield %value : f16
    } -> tensor<4xf16>
    %reshaped = tensor.expand_shape %producer [[0, 1]]
        output_shape [1, 4] : tensor<4xf16> into tensor<1x4xf16>
    %consumer_out = tensor.empty() : tensor<1x4xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%reshaped : tensor<1x4xf16>)
        outs(%consumer_out : tensor<1x4xf16>) {
    ^bb0(%value: f16, %old: f16):
      linalg.yield %value : f16
    } -> tensor<1x4xf16>
    return %consumer : tensor<1x4xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);
  const std::string sourceBefore = printModule(*source);

  unsigned materializedRegionCount = 0;
  auto materialized =
      wafer::tensor_program_scheduling::materializeCompleteRankTileProgram(
          *source, /*logicalRank=*/0, &materializedRegionCount);

  ASSERT_TRUE(mlir::succeeded(materialized));
  ASSERT_TRUE(*materialized);
  EXPECT_EQ(materializedRegionCount, 1u);
  EXPECT_EQ(countOperations<wafer::TileRegionOp>((*materialized)->getOperation()),
            1u);
  EXPECT_EQ(countOperations<mlir::func::FuncOp>((*materialized)->getOperation()),
            1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**materialized)));
  EXPECT_EQ(printModule(*source), sourceBefore);

  unsigned terminalOperationCount = 0;
  bool hasPlacement = false;
  (*materialized)->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface,
                  wafer::SyncNCCJoinOp>(operation))
      ++terminalOperationCount;
    hasPlacement |= operation->hasAttr(wafer::kWaferSPMOffsetAttrName) ||
                    operation->hasAttr(wafer::kWaferDDROffsetAttrName);
  });
  EXPECT_EQ(terminalOperationCount, 0u);
  EXPECT_FALSE(hasPlacement);
}

TEST(RankTileMaterializationTest,
     PreservesDiamondFanoutSharedInputAndMultipleRootsInOneRank) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<4xf16>, %shared: tensor<4xf16>)
      -> (tensor<4xf16>, tensor<4xf16>) {
    %a_out = tensor.empty() : tensor<4xf16>
    %a = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf16>) outs(%a_out : tensor<4xf16>) {
    ^bb0(%value: f16, %old: f16):
      %next = arith.negf %value : f16
      linalg.yield %next : f16
    } -> tensor<4xf16>
    %b_out = tensor.empty() : tensor<4xf16>
    %b = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%a : tensor<4xf16>) outs(%b_out : tensor<4xf16>) {
    ^bb0(%value: f16, %old: f16):
      %next = arith.mulf %value, %value : f16
      linalg.yield %next : f16
    } -> tensor<4xf16>
    %c_out = tensor.empty() : tensor<4xf16>
    %c = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%a, %shared : tensor<4xf16>, tensor<4xf16>)
        outs(%c_out : tensor<4xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %old: f16):
      %next = arith.addf %lhs, %rhs : f16
      linalg.yield %next : f16
    } -> tensor<4xf16>
    %d_out = tensor.empty() : tensor<4xf16>
    %d = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%b, %c : tensor<4xf16>, tensor<4xf16>)
        outs(%d_out : tensor<4xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %old: f16):
      %next = arith.addf %lhs, %rhs : f16
      linalg.yield %next : f16
    } -> tensor<4xf16>
    %e_out = tensor.empty() : tensor<4xf16>
    %e = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%c, %shared : tensor<4xf16>, tensor<4xf16>)
        outs(%e_out : tensor<4xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %old: f16):
      %next = arith.mulf %lhs, %rhs : f16
      linalg.yield %next : f16
    } -> tensor<4xf16>
    return %d, %e : tensor<4xf16>, tensor<4xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  auto materialized =
      wafer::tensor_program_scheduling::materializeCompleteRankTileProgram(
          *source, /*logicalRank=*/0);
  ASSERT_TRUE(mlir::succeeded(materialized));
  ASSERT_TRUE(*materialized);
  EXPECT_EQ(countOperations<wafer::TileRegionOp>((*materialized)->getOperation()),
            1u);
  EXPECT_GE(countOperations<mlir::scf::ForOp>((*materialized)->getOperation()),
            5u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**materialized)));

  auto function = *(*materialized)->getOps<mlir::func::FuncOp>().begin();
  auto returnOp = mlir::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  ASSERT_EQ(returnOp.getNumOperands(), 2u);
  for (mlir::Value result : returnOp.getOperands()) {
    auto toTensor = result.getDefiningOp<mlir::bufferization::ToTensorOp>();
    ASSERT_TRUE(toTensor);
    EXPECT_TRUE(wafer::isWaferDDRMemRefType(toTensor.getMemref().getType()));
  }
}

TEST(RankTileMaterializationTest,
     KeepsDifferentResultShapesAsTraversalsInOneResidencyRegion) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<2x4xf16>)
      -> (tensor<2x4xf16>, tensor<2xf16>) {
    %converted_out = tensor.empty() : tensor<2x4xf16>
    %converted = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<2x4xf16>)
        outs(%converted_out : tensor<2x4xf16>) {
    ^bb0(%value: f16, %old: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<2x4xf16>
    %squared_out = tensor.empty() : tensor<2x4xf16>
    %squared = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%converted : tensor<2x4xf16>)
        outs(%squared_out : tensor<2x4xf16>) {
    ^bb0(%value: f16, %old: f16):
      %product = arith.mulf %value, %value : f16
      linalg.yield %product : f16
    } -> tensor<2x4xf16>
    %zero = arith.constant 0.0 : f16
    %reduced_out = tensor.empty() : tensor<2xf16>
    %reduced_init = linalg.fill ins(%zero : f16)
        outs(%reduced_out : tensor<2xf16>) -> tensor<2xf16>
    %reduced = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%converted : tensor<2x4xf16>)
        outs(%reduced_init : tensor<2xf16>) {
    ^bb0(%value: f16, %accumulator: f16):
      %sum = arith.addf %accumulator, %value : f16
      linalg.yield %sum : f16
    } -> tensor<2xf16>
    return %squared, %reduced : tensor<2x4xf16>, tensor<2xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  unsigned materializedRegionCount = 0;
  auto materialized =
      wafer::tensor_program_scheduling::materializeCompleteRankTileProgram(
          *source, /*logicalRank=*/0, &materializedRegionCount);

  ASSERT_TRUE(mlir::succeeded(materialized));
  ASSERT_TRUE(*materialized);
  EXPECT_EQ(materializedRegionCount, 1u);
  EXPECT_EQ(countOperations<wafer::TileRegionOp>((*materialized)->getOperation()),
            1u);
  EXPECT_GE(countOperations<mlir::scf::ForOp>((*materialized)->getOperation()),
            3u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**materialized)));

  auto function = *(*materialized)->getOps<mlir::func::FuncOp>().begin();
  auto returnOp = mlir::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  ASSERT_EQ(returnOp.getNumOperands(), 2u);
  for (mlir::Value result : returnOp.getOperands()) {
    auto toTensor = result.getDefiningOp<mlir::bufferization::ToTensorOp>();
    ASSERT_TRUE(toTensor);
    EXPECT_TRUE(wafer::isWaferDDRMemRefType(toTensor.getMemref().getType()));
  }

  wafer::TileRegionOp region;
  (*materialized)->walk([&](wafer::TileRegionOp current) { region = current; });
  ASSERT_TRUE(region);
  llvm::SmallVector<mlir::scf::ForOp, 2> topLevelTraversals;
  for (mlir::scf::ForOp loop : region.getOps<mlir::scf::ForOp>())
    topLevelTraversals.push_back(loop);
  ASSERT_GE(topLevelTraversals.size(), 2u) << printModule(**materialized);

  // Different result domains become separate traversals in one residency
  // region. Selecting the second traversal as a region cut is legal only
  // because every shaped crossing is an explicit DDR boundary.
  auto partition =
      wafer::tensor_program_scheduling::partitionTileRegionAtDDRBoundary(
          region, topLevelTraversals[1].getOperation());
  ASSERT_TRUE(mlir::succeeded(partition)) << printModule(**materialized);
  EXPECT_EQ(countOperations<wafer::TileRegionOp>((*materialized)->getOperation()),
            2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**materialized)));
}

TEST(RankTileMaterializationTest,
     PropagatesUnitTileThroughRankChangingViewIntoMatmul) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<1x16x4xf16>,
                  %weight: tensor<4x8xf16>) -> tensor<1x16x8xf16> {
    %collapsed = tensor.collapse_shape %input [[0, 1], [2]]
        : tensor<1x16x4xf16> into tensor<16x4xf16>
    %zero = arith.constant 0.0 : f16
    %matmul_out = tensor.empty() : tensor<16x8xf16>
    %matmul_init = linalg.fill ins(%zero : f16)
        outs(%matmul_out : tensor<16x8xf16>) -> tensor<16x8xf16>
    %matmul = linalg.matmul ins(%collapsed, %weight
        : tensor<16x4xf16>, tensor<4x8xf16>)
        outs(%matmul_init : tensor<16x8xf16>) -> tensor<16x8xf16>
    %expanded = tensor.expand_shape %matmul [[0, 1], [2]]
        output_shape [1, 16, 8]
        : tensor<16x8xf16> into tensor<1x16x8xf16>
    %output = tensor.empty() : tensor<1x16x8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%expanded : tensor<1x16x8xf16>)
        outs(%output : tensor<1x16x8xf16>) {
    ^bb0(%value: f16, %old: f16):
      linalg.yield %value : f16
    } -> tensor<1x16x8xf16>
    return %result : tensor<1x16x8xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);
  auto materialized =
      wafer::tensor_program_scheduling::materializeCompleteRankTileProgram(
          *source, /*logicalRank=*/0);
  ASSERT_TRUE(mlir::succeeded(materialized));

  llvm::SmallVector<wafer::ComputeGemmOp, 2> gemms;
  (*materialized)->walk(
      [&](wafer::ComputeGemmOp gemm) { gemms.push_back(gemm); });
  ASSERT_EQ(gemms.size(), 1u) << printModule(**materialized);
  auto lhsType = mlir::cast<mlir::MemRefType>(gemms.front().getLhs().getType());
  auto rhsType = mlir::cast<mlir::MemRefType>(gemms.front().getRhs().getType());
  auto resultType =
      mlir::cast<mlir::MemRefType>(gemms.front().getResult().getType());
  EXPECT_EQ(lhsType.getShape(), llvm::ArrayRef<int64_t>({1, 4}))
      << printModule(**materialized);
  EXPECT_EQ(rhsType.getShape(), llvm::ArrayRef<int64_t>({4, 1}))
      << printModule(**materialized);
  EXPECT_EQ(resultType.getShape(), llvm::ArrayRef<int64_t>({1, 1}))
      << printModule(**materialized);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**materialized)));
}

TEST(RankTileMaterializationTest,
     TilesCanonicalConcatWithExplicitBranchLocalDDRLoads) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%first: tensor<1x2x2x4xf16>,
                  %second: tensor<1x2x2x4xf16>)
      -> tensor<1x2x2x8xf16> {
    %c4 = arith.constant 4 : index
    %out = tensor.empty() : tensor<1x2x2x8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3)
                                     -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } outs(%out : tensor<1x2x2x8xf16>) {
    ^bb0(%old: f16):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %i3 = linalg.index 3 : index
      %in_first = arith.cmpi ult, %i3, %c4 : index
      %value = scf.if %in_first -> f16 {
        %first_value = tensor.extract %first[%i0, %i1, %i2, %i3]
            : tensor<1x2x2x4xf16>
        scf.yield %first_value : f16
      } else {
        %second_i3 = arith.subi %i3, %c4 : index
        %second_value = tensor.extract %second[%i0, %i1, %i2, %second_i3]
            : tensor<1x2x2x4xf16>
        scf.yield %second_value : f16
      }
      linalg.yield %value : f16
    } -> tensor<1x2x2x8xf16>
    return %result : tensor<1x2x2x8xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);
  auto materialized =
      wafer::tensor_program_scheduling::materializeCompleteRankTileProgram(
          *source, /*logicalRank=*/0);
  ASSERT_TRUE(mlir::succeeded(materialized));
  EXPECT_GT(countOperations<mlir::scf::IfOp>((*materialized)->getOperation()),
            0u);
  (*materialized)->walk([&](mlir::memref::AllocOp allocation) {
    auto type = allocation.getType();
    if (!wafer::isWaferSPMMemRefType(type))
      return;
    EXPECT_EQ(type.getNumElements(), 1u) << printModule(**materialized);
  });
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**materialized)));

  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::convertTileRegionToInstrModule(
      **materialized, &failureReason)))
      << failureReason << printModule(**materialized);
  EXPECT_GT(countOperations<wafer::InstrRDMAOp>((*materialized)->getOperation()),
            0u);
  EXPECT_GT(countOperations<wafer::InstrWDMAOp>((*materialized)->getOperation()),
            0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**materialized)));
}

TEST(RankTileMaterializationTest, RejectsNonConcatLinalgIndexPayload) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<4xi32>) -> tensor<4xi32> {
    %out = tensor.empty() : tensor<4xi32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xi32>) outs(%out : tensor<4xi32>) {
    ^bb0(%value: i32, %old: i32):
      %index = linalg.index 0 : index
      %index_i32 = arith.index_cast %index : index to i32
      %sum = arith.addi %value, %index_i32 : i32
      linalg.yield %sum : i32
    } -> tensor<4xi32>
    return %result : tensor<4xi32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);
  mlir::OwningOpRef<mlir::ModuleOp> rankModule;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(wafer::lowerCompleteRankTensorProgramToTileRegionModule(
      *source, rankModule, &failureReason, /*currentLogicalRank=*/0)));
  EXPECT_NE(failureReason.find("unsupported linalg.generic body op linalg.index"),
            std::string::npos)
      << failureReason;
}

TEST(RankTileMaterializationTest,
     MaterializesExplicitDDRSpillAsTwoLegalResidencyRegions) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto materialized = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %result = wafer.tile.region(%boundary
        : memref<4xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%input: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %resident = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %input into %resident
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %resident, %input
          : memref<4xf16, #wafer.memory<spm, tensor>>
         -> memref<4xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %input
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(materialized);

  wafer::TileRegionOp originalRegion;
  (*materialized)->walk([&](wafer::TileRegionOp region) {
    ASSERT_FALSE(originalRegion);
    originalRegion = region;
  });
  ASSERT_TRUE(originalRegion);
  auto load = *originalRegion.getOps<wafer::StorageLoadOp>().begin();
  auto originalStore =
      *originalRegion.getOps<wafer::StorageStoreOp>().begin();
  mlir::Value root = load.getDest();
  ASSERT_TRUE(root.getDefiningOp());
  ASSERT_EQ(originalStore.getSource(), root);

  // The same cut is illegal while the live SPM root crosses it. The action
  // must fail before mutating the actual region.
  auto illegalPartition =
      wafer::tensor_program_scheduling::partitionTileRegionAtDDRBoundary(
          originalRegion, originalStore.getOperation());
  EXPECT_TRUE(mlir::failed(illegalPartition));
  EXPECT_EQ(countOperations<wafer::TileRegionOp>(materialized->getOperation()),
            1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized)));

  auto spill = wafer::tensor_program_scheduling::materializeSelectiveTileSpill(
      originalRegion, root, load.getOperation(), originalStore.getOperation());
  ASSERT_TRUE(mlir::succeeded(spill));
  ASSERT_TRUE(spill->reloadAllocation);
  EXPECT_TRUE(wafer::isWaferDDRMemRefType(spill->ddrBuffer.getType()));
  EXPECT_TRUE(wafer::isWaferSPMMemRefType(spill->reloadedValue.getType()));

  auto partition =
      wafer::tensor_program_scheduling::partitionTileRegionAtDDRBoundary(
          originalRegion, spill->reloadAllocation);
  ASSERT_TRUE(mlir::succeeded(partition));
  EXPECT_EQ(countOperations<wafer::TileRegionOp>(materialized->getOperation()),
            2u);
  for (wafer::TileRegionOp region : {partition->head, partition->tail}) {
    for (mlir::Value input : region.getInputs())
      if (mlir::isa<mlir::ShapedType>(input.getType()))
        EXPECT_TRUE(wafer::isWaferDDRMemRefType(input.getType()));
    for (mlir::Value result : region.getResults())
      if (mlir::isa<mlir::ShapedType>(result.getType()))
        EXPECT_TRUE(wafer::isWaferDDRMemRefType(result.getType()));
  }
  EXPECT_EQ(countOperations<wafer::StorageStoreOp>(
                partition->head.getOperation()),
            1u);
  EXPECT_EQ(countOperations<wafer::StorageLoadOp>(
                partition->tail.getOperation()),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*materialized)));

  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::convertTileRegionToInstrModule(
      *materialized, &failureReason)))
      << failureReason;
  EXPECT_GE(countOperations<wafer::InstrRDMAOp>(materialized->getOperation()),
            2u);
  EXPECT_GE(countOperations<wafer::InstrWDMAOp>(materialized->getOperation()),
            2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized)));
}

TEST(RankTileMaterializationTest,
     SelectiveSpillKeepsUnrelatedRootResidentInTheSameRegion) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto materialized = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %boundary_a: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %boundary_b: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %result = wafer.tile.region(%boundary_a, %boundary_b
        : memref<4xf16, #wafer.memory<ddr, tensor>>,
          memref<4xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%ddr_a: memref<4xf16, #wafer.memory<ddr, tensor>>,
         %ddr_b: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %resident_a = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %ddr_a into %resident_a
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         into memref<4xf16, #wafer.memory<spm, tensor>>
      %resident_b = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %ddr_b into %resident_b
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         into memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %resident_b, %ddr_b
          : memref<4xf16, #wafer.memory<spm, tensor>>
         -> memref<4xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.store %resident_a, %ddr_a
          : memref<4xf16, #wafer.memory<spm, tensor>>
         -> memref<4xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %ddr_a
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(materialized);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*materialized)));

  wafer::TileRegionOp region;
  materialized->walk([&](wafer::TileRegionOp current) { region = current; });
  ASSERT_TRUE(region);

  llvm::SmallVector<wafer::StorageLoadOp, 2> loads;
  llvm::SmallVector<wafer::StorageStoreOp, 2> stores;
  for (wafer::StorageLoadOp load : region.getOps<wafer::StorageLoadOp>())
    loads.push_back(load);
  for (wafer::StorageStoreOp store : region.getOps<wafer::StorageStoreOp>())
    stores.push_back(store);
  ASSERT_EQ(loads.size(), 2u);
  ASSERT_EQ(stores.size(), 2u);
  mlir::Value unrelatedRoot = loads[0].getDest();
  mlir::Value spillRoot = loads[1].getDest();
  ASSERT_EQ(stores[0].getSource(), spillRoot);
  ASSERT_EQ(stores[1].getSource(), unrelatedRoot);

  auto spill = wafer::tensor_program_scheduling::materializeSelectiveTileSpill(
      region, spillRoot, loads[1].getOperation(), stores[0].getOperation());
  ASSERT_TRUE(mlir::succeeded(spill));
  EXPECT_EQ(stores[0].getSource(), spill->reloadedValue);
  EXPECT_EQ(stores[1].getSource(), unrelatedRoot);
  EXPECT_TRUE(wafer::isWaferSPMMemRefType(unrelatedRoot.getType()));
  EXPECT_TRUE(wafer::isWaferDDRMemRefType(spill->ddrBuffer.getType()));
  EXPECT_EQ(countOperations<wafer::TileRegionOp>(materialized->getOperation()),
            1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized)));
}

} // namespace
