#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
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
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
}

mlir::func::FuncOp findSingleTensorProgram(mlir::ModuleOp module) {
  mlir::func::FuncOp found;
  module.walk([&](mlir::func::FuncOp function) {
    EXPECT_FALSE(found);
    found = function;
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
      } ins(%input : tensor<2x5xf32>)
        outs(%init : tensor<2xf32>) {
    ^bb0(%value: f32, %acc: f32):
)mlir";
  sourceText.append(combinerBody.begin(), combinerBody.end());
  sourceText += R"mlir(
      linalg.yield %combined : f32
    } -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
)mlir";

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(&context));
  if (!source)
    return "test source parse failed";
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  if (!function)
    return "test source has no tensor program";

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  mlir::LogicalResult result =
      splitReduction
          ? wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
                function, /*candidateTileSizes=*/{1},
                /*candidateReductionTileSizes=*/{3}, lowered, &failureReason,
                /*currentLogicalRank=*/0)
          : wafer::lowerTensorProgramToTileRegionModule(
                function, lowered, &failureReason, /*currentLogicalRank=*/0);
  if (mlir::succeeded(result))
    return "unexpected lowering success";
  return failureReason;
}

TEST(WaferTensorProgramToTileRegionTest,
     LowersStandaloneTensorProgramFunction) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @candidate(%lhs: tensor<5x6xf16>, %rhs: tensor<6x7xf16>,
                       %out: tensor<5x7xf16>) -> tensor<5x7xf16> {
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<5x7xf16>) -> tensor<5x7xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<5x6xf16>, tensor<6x7xf16>)
        outs(%init : tensor<5x7xf16>) -> tensor<5x7xf16>
    return %result : tensor<5x7xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  auto function = mlir::cast<mlir::func::FuncOp>(source->getBody()->front());

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{3, 4},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::TileRegionOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::ComputeGemmOp>(*lowered), 4u);
}

TEST(WaferTensorProgramToTileRegionTest,
     BatchedMatmulMaterializesImplicitNCxGemmStorage) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @batch_matmul(%lhs: tensor<2x16x128xf16>,
                          %rhs: tensor<2x128x16xf16>,
                          %out: tensor<2x16x16xf16>)
      -> tensor<2x16x16xf16> {
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x16x16xf16>) -> tensor<2x16x16xf16>
    %result = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<2x16x128xf16>, tensor<2x128x16xf16>)
        outs(%init : tensor<2x16x16xf16>) -> tensor<2x16x16xf16>
    return %result : tensor<2x16x16xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToTileRegionModule(
      function, lowered, &failureReason, /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));

  llvm::SmallVector<wafer::ComputeGemmOp, 1> gemms;
  lowered->walk([&](wafer::ComputeGemmOp op) { gemms.push_back(op); });
  ASSERT_EQ(gemms.size(), 1u);
  auto batchCount =
      gemms.front()->getAttrOfType<mlir::IntegerAttr>("batch_count");
  ASSERT_TRUE(batchCount);
  EXPECT_EQ(batchCount.getInt(), 2);
  for (mlir::Type type :
       {gemms.front().getLhs().getType(), gemms.front().getRhs().getType(),
        gemms.front().getResult().getType()}) {
    auto memref = mlir::cast<mlir::MemRefType>(type);
    EXPECT_EQ(wafer::getWaferMemoryAttr(memref).getLayout(),
              wafer::MemLayout::NCx);
  }

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  llvm::SmallVector<wafer::InstrGemmOp, 1> instructions;
  lowered->walk([&](wafer::InstrGemmOp op) { instructions.push_back(op); });
  ASSERT_EQ(instructions.size(), 1u);
  EXPECT_EQ(
      wafer::getWaferMemoryAttr(
          mlir::cast<mlir::MemRefType>(instructions.front().getLhs().getType()))
          .getLayout(),
      wafer::MemLayout::NCx);
}

TEST(WaferTensorProgramToTileRegionTest,
     DirectMappedBoundaryRouteEliminatesStagedCxMaterialization) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @matmul(%lhs: tensor<2x3xf16>, %rhs: tensor<3x2xf16>,
                    %out: tensor<2x2xf16>) -> tensor<2x2xf16> {
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x2xf16>) -> tensor<2x2xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<2x3xf16>, tensor<3x2xf16>)
        outs(%init : tensor<2x2xf16>) -> tensor<2x2xf16>
    return %result : tensor<2x2xf16>
  }
}
)mlir",
                                              mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> direct;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToTileRegionModule(
      function, direct, &failureReason, /*currentLogicalRank=*/0, std::nullopt,
      /*useDirectMappedBoundaryTransfer=*/true)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*direct, &failureReason)))
      << failureReason;
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*direct), 0u);
  unsigned mapped = 0;
  direct->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::InstrRDMAOp, wafer::InstrWDMAOp>(operation))
      mapped +=
          operation->hasAttr("src_offset") || operation->hasAttr("dst_offset");
  });
  EXPECT_GT(mapped, 0u);
}

TEST(WaferTensorProgramToTileRegionTest,
     MaterializesTargetReciprocalAlternativeThroughSourceInterface) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reciprocal(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%value: f32, %init: f32):
      %one = arith.constant 1.0 : f32
      %reciprocal = arith.divf %one, %value : f32
      linalg.yield %reciprocal : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  auto generic = *function.getOps<mlir::linalg::GenericOp>().begin();
  auto interface = mlir::dyn_cast<wafer::WaferTargetImplementationOpInterface>(
      generic.getOperation());
  ASSERT_TRUE(interface);
  llvm::SmallVector<wafer::TargetImplementationCandidate, 2> candidates;
  interface.collectTargetImplementationCandidates(
      wafer::WaferTargetCapabilities{}, candidates);
  ASSERT_EQ(candidates.size(), 2u);
  EXPECT_EQ(candidates[0].kind, wafer::TargetImplementationKind::Generic);
  EXPECT_EQ(candidates[1].kind,
            wafer::TargetImplementationKind::GenericReciprocal);
  wafer::WaferTargetCapabilities noReciprocal;
  noReciprocal.supportsElementwiseReciprocal = false;
  candidates.clear();
  interface.collectTargetImplementationCandidates(noReciprocal, candidates);
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates.front().kind, wafer::TargetImplementationKind::Generic);

  mlir::OwningOpRef<mlir::ModuleOp> baseline;
  mlir::OwningOpRef<mlir::ModuleOp> alternative;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{}, baseline, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{}, alternative, &failureReason,
          /*currentLogicalRank=*/0,
          wafer::TargetImplementationKind::GenericReciprocal)))
      << failureReason;

  llvm::SmallVector<wafer::ComputeElementwiseKind, 4> baselineKinds;
  llvm::SmallVector<wafer::ComputeElementwiseKind, 4> alternativeKinds;
  baseline->walk([&](wafer::ComputeElementwiseOp op) {
    baselineKinds.push_back(op.getKind());
  });
  alternative->walk([&](wafer::ComputeElementwiseOp op) {
    alternativeKinds.push_back(op.getKind());
  });
  EXPECT_TRUE(
      llvm::is_contained(baselineKinds, wafer::ComputeElementwiseKind::Div));
  EXPECT_FALSE(
      llvm::is_contained(baselineKinds, wafer::ComputeElementwiseKind::Recip));
  EXPECT_TRUE(llvm::is_contained(alternativeKinds,
                                 wafer::ComputeElementwiseKind::Recip));
  EXPECT_FALSE(
      llvm::is_contained(alternativeKinds, wafer::ComputeElementwiseKind::Div));
  EXPECT_GT(countOps<wafer::ComputeFillOp>(*baseline), 0u);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*baseline, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*alternative, &failureReason)))
      << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*baseline)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*alternative)));
  EXPECT_GT(countOps<wafer::InstrElementwiseOp>(*baseline), 0u);
  EXPECT_GT(countOps<wafer::InstrElementwiseOp>(*alternative), 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

TEST(WaferTensorProgramToTileRegionTest,
     MaterializesCompleteTwoByTwoTailTraversal) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @matmul(%lhs: tensor<5x6xf16>, %rhs: tensor<6x7xf16>,
                    %out: tensor<5x7xf16>) -> tensor<5x7xf16> {
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<5x7xf16>) -> tensor<5x7xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<5x6xf16>, tensor<6x7xf16>)
        outs(%init : tensor<5x7xf16>) -> tensor<5x7xf16>
    return %result : tensor<5x7xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{3, 4},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::ComputeGemmOp>(*lowered), 4u);
  EXPECT_EQ(
      collectStoreSlices(*lowered),
      (std::vector<std::string>{"-9223372036854775808,-9223372036854775808:3,4",
                                "-9223372036854775808,4:3,3",
                                "3,-9223372036854775808:2,4", "3,4:2,3"}));
  EXPECT_EQ(collectGemmStoreOrder(*lowered), "GSGSGSGS");

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::InstrGemmOp>(*lowered), 4u);
  EXPECT_EQ(countOps<wafer::InstrWDMAOp>(*lowered), 4u);
}

TEST(WaferTensorProgramToTileRegionTest,
     PreservesCandidateTraversalScopeForOneFullStaticTile) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @map(%input: tensor<4x8xf32>, %out: tensor<4x8xf32>)
      -> tensor<4x8xf32> {
    %result = linalg.map
        ins(%input : tensor<4x8xf32>)
        outs(%out : tensor<4x8xf32>)
        (%value: f32) {
          linalg.yield %value : f32
        }
    return %result : tensor<4x8xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{4, 8},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*lowered), 2u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(*lowered), 1u);
}

TEST(WaferTensorProgramToTileRegionTest,
     TilesTerminalAllReduceAndFusesItsMatmulProducer) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  func.func @matmul_all_reduce(
      %lhs: tensor<4x8xf16>, %rhs: tensor<8x6xf16>,
      %out: tensor<4x6xf16>) -> tensor<4x6xf16> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<4x6xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<4x6xf16>) -> tensor<4x6xf16>
    %local = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%init : tensor<4x6xf16>) -> tensor<4x6xf16>
    %result = wafer.linalg_ext.collective.all_reduce
        ins(%local : tensor<4x6xf16>)
        outs(%out : tensor<4x6xf16>) {
    ^bb0(%left: f16, %right: f16):
      %sum = arith.addf %left, %right : f16
      wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 7 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4x6xf16>
    return %result : tensor<4x6xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2, 3},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*lowered), 2u);
  EXPECT_EQ(countOps<wafer::ComputeGemmOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::CommAllReduceOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(*lowered), 1u);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::InstrGemmOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::CommAllReduceOp>(*lowered), 0u);
  EXPECT_GT(countOps<wafer::InstrDTEWaitOp>(*lowered), 0u);
  EXPECT_GT(countOps<wafer::InstrElementwiseOp>(*lowered), 0u);
}

TEST(WaferTensorProgramToTileRegionTest,
     PromotesCollectiveInputBeforeAllReduceCommunication) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  func.func @matmul_promoted_all_reduce(
      %lhs: tensor<4x8xf16>, %rhs: tensor<8x6xf16>,
      %out: tensor<4x6xf32>) -> tensor<4x6xf32> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<4x6xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<4x6xf16>) -> tensor<4x6xf16>
    %local = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%init : tensor<4x6xf16>) -> tensor<4x6xf16>
    %result = wafer.linalg_ext.collective.all_reduce
        ins(%local : tensor<4x6xf16>)
        outs(%out : tensor<4x6xf32>) {
    ^bb0(%left: f32, %right: f32):
      %sum = arith.addf %left, %right : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 9 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4x6xf32>
    return %result : tensor<4x6xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2, 3},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::ComputeGemmOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::ComputeConvertOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::CommAllReduceOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(*lowered), 1u);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::ComputeConvertOp>(*lowered), 0u);
  EXPECT_EQ(countOps<wafer::InstrConvertOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::CommAllReduceOp>(*lowered), 0u);
  EXPECT_GT(countOps<wafer::InstrDTEWaitOp>(*lowered), 0u);
}

TEST(WaferTensorProgramToTileRegionTest,
     LowersF16GenericReductionChunks) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce(%input: tensor<2x5xf16>, %out: tensor<2xf16>)
      -> tensor<2xf16> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<2xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<2xf16>) -> tensor<2xf16>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<2x5xf16>)
        outs(%init : tensor<2xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %sum = arith.addf %value, %acc : f16
      linalg.yield %sum : f16
    } -> tensor<2xf16>
    return %result : tensor<2xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{3}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));

  llvm::SmallVector<wafer::ComputeReduceOp, 2> reductions;
  llvm::SmallVector<wafer::ComputeElementwiseOp, 1> combines;
  lowered->walk([&](wafer::ComputeReduceOp op) { reductions.push_back(op); });
  lowered->walk(
      [&](wafer::ComputeElementwiseOp op) { combines.push_back(op); });
  ASSERT_EQ(reductions.size(), 2u);
  ASSERT_EQ(combines.size(), 1u);
  auto firstInputType =
      mlir::cast<mlir::MemRefType>(reductions[0].getInput().getType());
  auto tailInputType =
      mlir::cast<mlir::MemRefType>(reductions[1].getInput().getType());
  EXPECT_EQ(firstInputType.getDimSize(1), 3);
  EXPECT_EQ(tailInputType.getDimSize(1), 2);
  EXPECT_EQ(combines[0].getKind(), wafer::ComputeElementwiseKind::Add);
  ASSERT_EQ(combines[0].getInputs().size(), 2u);
  auto previousLayout =
      combines[0].getInputs()[0].getDefiningOp<wafer::LayoutMaterializeOp>();
  auto partialLayout =
      combines[0].getInputs()[1].getDefiningOp<wafer::LayoutMaterializeOp>();
  ASSERT_TRUE(previousLayout);
  ASSERT_TRUE(partialLayout);
  EXPECT_EQ(previousLayout.getSource(), reductions[0].getResult());
  EXPECT_EQ(partialLayout.getSource(), reductions[1].getResult());

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::ComputeReduceOp>(*lowered), 0u);
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(*lowered), 0u);
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*lowered), 6u);

  mlir::OwningOpRef<mlir::ModuleOp> singleTile;
  failureReason.clear();
  ASSERT_TRUE(
      mlir::succeeded(wafer::lowerCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileOffsets=*/{0},
          /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{3}, singleTile, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(singleTile);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*singleTile)));
  EXPECT_EQ(countOps<wafer::ComputeReduceOp>(*singleTile), 2u);
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(*singleTile), 1u);
}

TEST(WaferTensorProgramToTileRegionTest,
     LowersBF16MaximumChunks) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce(%input: tensor<2x5xbf16>, %out: tensor<2xbf16>)
      -> tensor<2xbf16> {
    %zero = arith.constant 0.0 : bf16
    %empty = tensor.empty() : tensor<2xbf16>
    %init = linalg.fill ins(%zero : bf16)
        outs(%empty : tensor<2xbf16>) -> tensor<2xbf16>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<2x5xbf16>)
        outs(%init : tensor<2xbf16>) {
    ^bb0(%value: bf16, %acc: bf16):
      %maximum = arith.maximumf %value, %acc : bf16
      linalg.yield %maximum : bf16
    } -> tensor<2xbf16>
    return %result : tensor<2xbf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{3}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  llvm::SmallVector<wafer::ComputeReduceOp, 2> reductions;
  llvm::SmallVector<wafer::ComputeElementwiseOp, 1> combines;
  lowered->walk([&](wafer::ComputeReduceOp op) { reductions.push_back(op); });
  lowered->walk(
      [&](wafer::ComputeElementwiseOp op) { combines.push_back(op); });
  ASSERT_EQ(reductions.size(), 2u);
  ASSERT_EQ(combines.size(), 1u);
}

TEST(WaferTensorProgramToTileRegionTest,
     LowersIntegerMatmulContractingChunksAndTailInSourceOrder) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @matmul(%lhs: tensor<2x5xi32>, %rhs: tensor<5x3xi32>,
                    %out: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %zero = arith.constant 0 : i32
    %init = linalg.fill ins(%zero : i32)
        outs(%out : tensor<2x3xi32>) -> tensor<2x3xi32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<2x5xi32>, tensor<5x3xi32>)
        outs(%init : tensor<2x3xi32>) -> tensor<2x3xi32>
    return %result : tensor<2x3xi32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2, 3},
          /*candidateReductionTileSizes=*/{3}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));

  llvm::SmallVector<wafer::ComputeGemmOp, 2> gemms;
  llvm::SmallVector<wafer::ComputeElementwiseOp, 1> combines;
  lowered->walk([&](wafer::ComputeGemmOp op) { gemms.push_back(op); });
  lowered->walk(
      [&](wafer::ComputeElementwiseOp op) { combines.push_back(op); });
  ASSERT_EQ(gemms.size(), 2u);
  ASSERT_EQ(combines.size(), 1u);
  auto firstLhsType = mlir::cast<mlir::MemRefType>(gemms[0].getLhs().getType());
  auto tailLhsType = mlir::cast<mlir::MemRefType>(gemms[1].getLhs().getType());
  EXPECT_EQ(firstLhsType.getDimSize(1), 3);
  EXPECT_EQ(tailLhsType.getDimSize(1), 2);
  EXPECT_EQ(combines[0].getKind(), wafer::ComputeElementwiseKind::Add);
  ASSERT_EQ(combines[0].getInputs().size(), 2u);
  auto previousLayout =
      combines[0].getInputs()[0].getDefiningOp<wafer::LayoutMaterializeOp>();
  auto partialLayout =
      combines[0].getInputs()[1].getDefiningOp<wafer::LayoutMaterializeOp>();
  ASSERT_TRUE(previousLayout);
  ASSERT_TRUE(partialLayout);
  EXPECT_EQ(previousLayout.getSource(), gemms[0].getResult());
  EXPECT_EQ(partialLayout.getSource(), gemms[1].getResult());

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  llvm::SmallVector<int64_t, 2> contractingSizes;
  lowered->walk(
      [&](wafer::InstrGemmOp op) { contractingSizes.push_back(op.getK()); });
  EXPECT_EQ(contractingSizes, (llvm::SmallVector<int64_t, 2>{3, 2}));
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*lowered), 1u);
}

TEST(WaferTensorProgramToTileRegionTest,
     LowersF16AndBF16MatmulContractingChunks) {
  for (llvm::StringRef typeSpelling : {"f16", "bf16"}) {
    mlir::DialectRegistry registry;
    registerConversionDialects(registry);
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    std::string sourceText = R"mlir(
module {
  func.func @matmul(%lhs: tensor<2x5xTYPE>, %rhs: tensor<5x3xTYPE>,
                    %out: tensor<2x3xTYPE>) -> tensor<2x3xTYPE> {
    %zero = arith.constant 0.0 : TYPE
    %init = linalg.fill ins(%zero : TYPE)
        outs(%out : tensor<2x3xTYPE>) -> tensor<2x3xTYPE>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<2x5xTYPE>, tensor<5x3xTYPE>)
        outs(%init : tensor<2x3xTYPE>) -> tensor<2x3xTYPE>
    return %result : tensor<2x3xTYPE>
  }
}
)mlir";
    for (size_t position = sourceText.find("TYPE");
         position != std::string::npos;
         position = sourceText.find("TYPE", position + typeSpelling.size()))
      sourceText.replace(position, 4, typeSpelling);

    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        sourceText, mlir::ParserConfig(&context));
    ASSERT_TRUE(source) << typeSpelling.str();
    mlir::func::FuncOp function = findSingleTensorProgram(*source);
    ASSERT_TRUE(function);

    mlir::OwningOpRef<mlir::ModuleOp> lowered;
    std::string failureReason;
    ASSERT_TRUE(mlir::succeeded(
        wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
            function, /*candidateTileSizes=*/{2, 3},
            /*candidateReductionTileSizes=*/{3}, lowered, &failureReason,
            /*currentLogicalRank=*/0)))
        << typeSpelling.str() << ": " << failureReason;
    ASSERT_TRUE(lowered);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));

    llvm::SmallVector<wafer::ComputeGemmOp, 2> gemms;
    lowered->walk([&](wafer::ComputeGemmOp op) { gemms.push_back(op); });
    ASSERT_EQ(gemms.size(), 2u);
    auto firstLhsType =
        mlir::cast<mlir::MemRefType>(gemms[0].getLhs().getType());
    auto tailLhsType =
        mlir::cast<mlir::MemRefType>(gemms[1].getLhs().getType());
    EXPECT_EQ(firstLhsType.getDimSize(1), 3);
    EXPECT_EQ(tailLhsType.getDimSize(1), 2);
    EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(*lowered), 1u);

    ASSERT_TRUE(mlir::succeeded(
        wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
        << failureReason;
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
    llvm::SmallVector<wafer::InstrGemmOp, 2> instrGemms;
    lowered->walk(
        [&](wafer::InstrGemmOp op) { instrGemms.push_back(op); });
    ASSERT_EQ(instrGemms.size(), 2u);
    EXPECT_EQ(instrGemms[0].getK(), 3);
    EXPECT_EQ(instrGemms[1].getK(), 2);
    EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*lowered), 1u);
  }
}

TEST(WaferTensorProgramToTileRegionTest,
     AcceptsOrderedReduceAfterTerminalJoinCoalescing) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce(
      %input: memref<1023x1xf32, #wafer.memory<spm, cx>>)
      -> memref<1xf32, #wafer.memory<spm, cx>> {
    %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
        {dimensions = array<i64: 0>, init_value = 0.000000e+00 : f32}
        : (memref<1023x1xf32, #wafer.memory<spm, cx>>)
       -> memref<1xf32, #wafer.memory<spm, cx>>
    return %result : memref<1xf32, #wafer.memory<spm, cx>>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*module, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  uint64_t terminalOperationCount = 0;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface, wafer::SyncNCCJoinOp>(
            operation))
      ++terminalOperationCount;
  });
  // Ordered operations on one worker remain in one hardware dependency
  // stream.  Only the function exit needs to materialize a completion join;
  // accounting must not restore one blocking wait after every operation.
  EXPECT_EQ(countOps<wafer::SyncNCCJoinOp>(*module), 1u);
  EXPECT_EQ(terminalOperationCount, 2049u);
  EXPECT_LT(terminalOperationCount,
            wafer::detail::kStaticTerminalOperationBudget);
  EXPECT_EQ(countOps<wafer::ComputeReduceOp>(*module), 0u);
  EXPECT_EQ(countOps<wafer::InstrReduceOp>(*module), 0u);
}

TEST(WaferTensorProgramToTileRegionTest,
     ReusesSPMAcrossOrderedElementwiseTraversalWithoutPerTileJoin) {
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
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%lhs, %rhs : tensor<2x100001xi64>, tensor<2x100001xi64>)
        outs(%out : tensor<2x100001xi64>) {
      ^bb0(%left: i64, %right: i64, %init: i64):
        %sum = arith.addi %left, %right : i64
        linalg.yield %sum : i64
      } -> tensor<2x100001xi64>
    return %result : tensor<2x100001xi64>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{1, 50001},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  EXPECT_EQ(countOps<wafer::SyncNCCJoinOp>(*lowered), 1u);

  mlir::LogicalResult planned = wafer::planSPMMemoryModule(
      *lowered, /*spmBase=*/65536, /*spmLimit=*/3080192,
      /*spmAlignment=*/256);
  if (mlir::failed(planned))
    lowered->print(llvm::errs());
  EXPECT_TRUE(mlir::succeeded(planned));
}

TEST(WaferTensorProgramToTileRegionTest,
     LowersRankReducedStaticExtractAndOutputInsert) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @rank_reduced_output(%input: tensor<2x4xf32>,
                                 %out: tensor<2x4xf32>)
      -> tensor<2x4xf32> {
    %row = tensor.extract_slice %input[0, 0] [1, 4] [1, 1]
        : tensor<2x4xf32> to tensor<4xf32>
    %result = tensor.insert_slice %row into %out[1, 0] [1, 4] [1, 1]
        : tensor<4xf32> into tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToTileRegionModule(
      function, lowered, &failureReason, /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));

  std::vector<std::string> rankReducedSlices;
  lowered->walk([&](mlir::memref::SubViewOp subview) {
    ASSERT_EQ(subview.getSourceType().getRank(), 2);
    ASSERT_EQ(subview.getType().getRank(), 1);
    rankReducedSlices.push_back(formatStaticSlice(subview));
  });
  std::sort(rankReducedSlices.begin(), rankReducedSlices.end());
  EXPECT_EQ(rankReducedSlices, (std::vector<std::string>{"0,0:1,4"}));
  EXPECT_EQ(countOps<wafer::StorageLoadOp>(*lowered), 2u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::MoveExtractSliceOp>(*lowered), 0u);
  llvm::SmallVector<wafer::MoveInsertSliceOp, 1> inserts;
  lowered->walk(
      [&](wafer::MoveInsertSliceOp insert) { inserts.push_back(insert); });
  ASSERT_EQ(inserts.size(), 1u);
  const llvm::SmallVector<int64_t, 2> expectedOffsets{1, 0};
  const llvm::SmallVector<int64_t, 2> expectedSizes{1, 4};
  const llvm::SmallVector<int64_t, 2> expectedStrides{1, 1};
  EXPECT_TRUE(inserts.front().getOffsets() ==
              llvm::ArrayRef<int64_t>(expectedOffsets));
  EXPECT_TRUE(inserts.front().getSizes() ==
              llvm::ArrayRef<int64_t>(expectedSizes));
  EXPECT_TRUE(inserts.front().getStrides() ==
              llvm::ArrayRef<int64_t>(expectedStrides));
  auto insertedSourceType =
      mlir::cast<mlir::MemRefType>(inserts.front().getSource().getType());
  auto insertedDestType =
      mlir::cast<mlir::MemRefType>(inserts.front().getDest().getType());
  EXPECT_EQ(insertedSourceType.getRank(), 1);
  EXPECT_EQ(insertedDestType.getRank(), 2);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::InstrRDMAOp>(*lowered), 2u);
  EXPECT_EQ(countOps<wafer::InstrWDMAOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*lowered), 2u);
}

TEST(WaferTensorProgramToTileRegionTest,
     FusesLinalgProducerChainsIntoTraversal) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @producer_chain(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                            %out: tensor<4xf32>) -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %producer = linalg.map
        ins(%lhs : tensor<4xf32>)
        outs(%empty : tensor<4xf32>)
        (%value: f32) {
          %doubled = arith.addf %value, %value : f32
          linalg.yield %doubled : f32
        }
    %result = linalg.map
        ins(%producer, %rhs : tensor<4xf32>, tensor<4xf32>)
        outs(%out : tensor<4xf32>)
        (%left: f32, %right: f32) {
          %sum = arith.addf %left, %right : f32
          linalg.yield %sum : f32
        }
    return %result : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::TileRegionOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(*lowered), 2u);
  EXPECT_EQ(countOps<wafer::StorageLoadOp>(*lowered), 2u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(*lowered), 1u);
}

TEST(WaferTensorProgramToTileRegionTest,
     FusesTransposeProducerForDirectSingleTileCandidate) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @transpose_matmul(
      %weight: tensor<8x16xf16>,
      %activation: tensor<2x16xf16>,
      %out: tensor<2x8xf16>) -> tensor<2x8xf16> {
    %transpose_empty = tensor.empty() : tensor<16x8xf16>
    %transpose = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d1, d0)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%weight : tensor<8x16xf16>)
        outs(%transpose_empty : tensor<16x8xf16>) {
      ^bb0(%value: f16, %init: f16):
        linalg.yield %value : f16
      } -> tensor<16x8xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x8xf16>) -> tensor<2x8xf16>
    %result = linalg.matmul
        ins(%activation, %transpose
            : tensor<2x16xf16>, tensor<16x8xf16>)
        outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
    return %result : tensor<2x8xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(wafer::lowerCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileOffsets=*/{0, 0},
          /*candidateTileSizes=*/{2, 4},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));

  unsigned untiledTransposeResultBuffers = 0;
  lowered->walk([&](mlir::memref::AllocOp alloc) {
    mlir::MemRefType type = alloc.getType();
    if (type.getRank() == 2 && type.getDimSize(0) == 16 &&
        type.getDimSize(1) == 8)
      ++untiledTransposeResultBuffers;
  });
  EXPECT_EQ(untiledTransposeResultBuffers, 0u);
}

TEST(WaferTensorProgramToTileRegionTest,
     RejectsUnsignedReductionSplitSemantics) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce_unsigned(%input: tensor<2x5xi32>, %out: tensor<2xi32>)
      -> tensor<2xi32> {
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
      } ins(%input : tensor<2x5xi32>)
        outs(%init : tensor<2xi32>) {
    ^bb0(%value: i32, %acc: i32):
      %max = arith.maxui %value, %acc : i32
      linalg.yield %max : i32
    } -> tensor<2xi32>
    return %result : tensor<2xi32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  EXPECT_TRUE(
      mlir::failed(wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{1},
          /*candidateReductionTileSizes=*/{3}, lowered, &failureReason,
          /*currentLogicalRank=*/0)));
  EXPECT_EQ(failureReason,
            "candidate reduction split cannot preserve unsigned min/max "
            "semantics with the current reduce kind");
}

TEST(WaferTensorProgramToTileRegionTest,
     RejectsReductionThatIgnoresAccumulator) {
  EXPECT_EQ(lowerF32ReductionAndGetFailure(
                "        %combined = arith.addf %value, %value : f32\n",
                /*splitReduction=*/false),
            "linalg.generic reduction requires one exact combiner wired to "
            "the reduced value and accumulator");
}

TEST(WaferTensorProgramToTileRegionTest,
     RejectsSplitReductionThatIgnoresAccumulator) {
  EXPECT_EQ(lowerF32ReductionAndGetFailure(
                "        %combined = arith.addf %value, %value : f32\n",
                /*splitReduction=*/true),
            "candidate reduction split requires one exact combiner wired to "
            "the reduced value and accumulator");
}

TEST(WaferTensorProgramToTileRegionTest, RejectsMaxNumReductionSplitSemantics) {
  EXPECT_EQ(lowerF32ReductionAndGetFailure(
                "        %combined = arith.maxnumf %value, %acc : f32\n",
                /*splitReduction=*/true),
            "candidate reduction split cannot preserve maxnum/minnum NaN "
            "semantics with the current reduce kind");
}

TEST(WaferTensorProgramToTileRegionTest,
     RejectsMultipleReductionAxisCandidateSplit) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @reduce_two_axes(%input: tensor<2x3x5xf32>,
                             %out: tensor<2xf32>) -> tensor<2xf32> {
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32)
        outs(%out : tensor<2xf32>) -> tensor<2xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction", "reduction"]
      } ins(%input : tensor<2x3x5xf32>)
        outs(%init : tensor<2xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  EXPECT_TRUE(
      mlir::failed(wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{2, 3}, lowered, &failureReason,
          /*currentLogicalRank=*/0)));
  EXPECT_EQ(failureReason,
            "candidate reduction split requires exactly one reduction axis");
  EXPECT_FALSE(lowered);
}

TEST(WaferTensorProgramToTileRegionTest,
     ChecksStaticTraversalProductWithoutOverflow) {
  uint64_t outputTileCount = 0;
  EXPECT_EQ(wafer::detail::checkedStaticTileProduct(
                /*ranges=*/{127}, /*tileSizes=*/{2}, outputTileCount),
            wafer::detail::CheckedStaticTileProductStatus::Success);
  EXPECT_EQ(outputTileCount, 64u);

  EXPECT_EQ(wafer::detail::checkedStaticTileProduct(
                /*ranges=*/{std::numeric_limits<int64_t>::max(),
                            std::numeric_limits<int64_t>::max()},
                /*tileSizes=*/{1, 1}, outputTileCount),
            wafer::detail::CheckedStaticTileProductStatus::Overflow);
}

TEST(WaferTensorProgramToTileRegionTest,
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
    %result = linalg.map
        ins(%input : tensor<4294967296x4294967296xf32>)
        outs(%out : tensor<4294967296x4294967296xf32>)
        (%value: f32) {
          linalg.yield %value : f32
        }
    return %result : tensor<4294967296x4294967296xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  EXPECT_TRUE(
      mlir::failed(wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{1, 1},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)));
  EXPECT_EQ(failureReason,
            "complete candidate traversal output tile count is not "
            "representable");
  EXPECT_FALSE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

TEST(WaferTensorProgramToTileRegionTest,
     CompactTraversalAcceptsFormerEagerExpansionBoundary) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @map(%input: tensor<8193xf32>, %out: tensor<8193xf32>)
      -> tensor<8193xf32> {
    %result = linalg.map
        ins(%input : tensor<8193xf32>)
        outs(%out : tensor<8193xf32>)
        (%value: f32) {
          linalg.yield %value : f32
        }
    return %result : tensor<8193xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  auto sentinel = mlir::parseSourceString<mlir::ModuleOp>(
      "module { func.func private @sentinel() }", mlir::ParserConfig(&context));
  ASSERT_TRUE(sentinel);
  mlir::OwningOpRef<mlir::ModuleOp> lowered = std::move(sentinel);

  std::string sourceBefore;
  llvm::raw_string_ostream sourceBeforeStream(sourceBefore);
  source->print(sourceBeforeStream);

  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*lowered), 1u);

  std::string sourceAfter;
  llvm::raw_string_ostream sourceAfterStream(sourceAfter);
  source->print(sourceAfterStream);
  EXPECT_EQ(sourceAfter, sourceBefore);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

TEST(WaferTensorProgramToTileRegionTest,
     OperandDrivenCompleteCloneReachesInstructionAndSPMPlanning) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @pointwise(%lhs: tensor<4x6xf32>, %rhs: tensor<4x6xf32>,
                       %out: tensor<4x6xf32>) -> tensor<4x6xf32> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%lhs, %rhs : tensor<4x6xf32>, tensor<4x6xf32>)
        outs(%out : tensor<4x6xf32>) {
    ^bb0(%left: f32, %right: f32, %old: f32):
      %sum = arith.addf %left, %right : f32
      linalg.yield %sum : f32
    } -> tensor<4x6xf32>
    return %result : tensor<4x6xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2, 3},
          /*candidateReductionTileSizes=*/{}, lowered, &failureReason,
          /*currentLogicalRank=*/0, /*selectedAlternative=*/std::nullopt,
          /*useDirectMappedBoundaryTransfer=*/false,
          wafer::CandidateTileTraversalKind::OperandDriven)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::TileRegionOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::StorageLoadOp>(*lowered), 2u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(*lowered), 1u);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
      *lowered, /*spmBase=*/65536, /*spmLimit=*/3080192,
      /*spmAlignment=*/256)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
}

TEST(WaferTensorProgramToTileRegionTest,
     PartialReductionCompleteCloneReachesInstructionAndSPMPlanning) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @sum(%input: tensor<4x8xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32)
        outs(%out : tensor<4xf32>) -> tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{4}, lowered, &failureReason,
          /*currentLogicalRank=*/0, /*selectedAlternative=*/std::nullopt,
          /*useDirectMappedBoundaryTransfer=*/false,
          wafer::CandidateTileTraversalKind::PartialReduction)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_GT(countOps<wafer::ComputeElementwiseOp>(*lowered), 0u);
  EXPECT_GT(countOps<wafer::ComputeReduceOp>(*lowered), 0u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(*lowered), 1u);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
      *lowered, /*spmBase=*/65536, /*spmLimit=*/3080192,
      /*spmAlignment=*/256)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
}

TEST(WaferTensorProgramToTileRegionTest,
     PartialReductionCompleteCloneFailsNumericLegalityAtomically) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @unsigned_max(%input: tensor<4x8xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %zero = arith.constant 0 : i32
    %init = linalg.fill ins(%zero : i32)
        outs(%out : tensor<4xi32>) -> tensor<4xi32>
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
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  auto sentinel = mlir::parseSourceString<mlir::ModuleOp>(
      "module { func.func private @sentinel() }", mlir::ParserConfig(&context));
  ASSERT_TRUE(sentinel);
  mlir::OwningOpRef<mlir::ModuleOp> lowered = std::move(sentinel);
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{4}, lowered, &failureReason,
          /*currentLogicalRank=*/0, /*selectedAlternative=*/std::nullopt,
          /*useDirectMappedBoundaryTransfer=*/false,
          wafer::CandidateTileTraversalKind::PartialReduction)));
  EXPECT_EQ(failureReason,
            "candidate reduction split cannot preserve unsigned min/max "
            "semantics with the current reduce kind");
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(lowered->lookupSymbol<mlir::func::FuncOp>("sentinel"));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

TEST(WaferTensorProgramToTileRegionTest,
     PartialReductionFeedsExistingTypedAllReduceThroughOneSSAChain) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  func.func @local_sum_all_reduce(%input: tensor<4x8xf32>,
                                  %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %local = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    %reduced = wafer.linalg_ext.collective.all_reduce
        ins(%local : tensor<4xf32>)
        outs(%out : tensor<4xf32>) {
    ^bb0(%left: f32, %right: f32):
      %sum = arith.addf %left, %right : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 73 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xf32>
    return %reduced : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{4}, lowered, &failureReason,
          /*currentLogicalRank=*/0, /*selectedAlternative=*/std::nullopt,
          /*useDirectMappedBoundaryTransfer=*/false,
          wafer::CandidateTileTraversalKind::PartialReduction)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_GT(countOps<wafer::ComputeReduceOp>(*lowered), 0u);
  EXPECT_GT(countOps<wafer::ComputeElementwiseOp>(*lowered), 0u);
  EXPECT_EQ(countOps<wafer::CommAllReduceOp>(*lowered), 1u);

  wafer::CommAllReduceOp allReduce;
  lowered->walk([&](wafer::CommAllReduceOp op) { allReduce = op; });
  ASSERT_TRUE(allReduce);
  ASSERT_TRUE(allReduce.getInput().getDefiningOp());
  EXPECT_TRUE((mlir::isa<wafer::ComputeElementwiseOp,
                         wafer::ComputeReduceOp>(
      allReduce.getInput().getDefiningOp())));

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
      *lowered, /*spmBase=*/65536, /*spmLimit=*/3080192,
      /*spmAlignment=*/256)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_GT(countOps<wafer::InstrDTEWaitOp>(*lowered), 0u);
  EXPECT_GT(countOps<wafer::InstrElementwiseOp>(*lowered), 0u);
}

TEST(WaferTensorProgramToTileRegionTest,
     TypedAllReduceDoesNotBypassLocalPartialNumericLegality) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @local_unsigned_max_all_reduce(%input: tensor<4x8xi32>,
                                           %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<4xi32>
    %init = linalg.fill ins(%zero : i32)
        outs(%empty : tensor<4xi32>) -> tensor<4xi32>
    %local = linalg.generic {
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
    %reduced = wafer.linalg_ext.collective.all_reduce
        ins(%local : tensor<4xi32>)
        outs(%out : tensor<4xi32>) {
    ^bb0(%left: i32, %right: i32):
      %maximum = arith.maxui %left, %right : i32
      wafer.linalg_ext.collective.yield %maximum : i32
    } {channel_id = 74 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xi32>
    return %reduced : tensor<4xi32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  auto sentinel = mlir::parseSourceString<mlir::ModuleOp>(
      "module { func.func private @sentinel() }", mlir::ParserConfig(&context));
  ASSERT_TRUE(sentinel);
  mlir::OwningOpRef<mlir::ModuleOp> lowered = std::move(sentinel);
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{4}, lowered, &failureReason,
          /*currentLogicalRank=*/0, /*selectedAlternative=*/std::nullopt,
          /*useDirectMappedBoundaryTransfer=*/false,
          wafer::CandidateTileTraversalKind::PartialReduction)));
  EXPECT_EQ(failureReason,
            "candidate reduction split cannot preserve unsigned min/max "
            "semantics with the current reduce kind");
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(lowered->lookupSymbol<mlir::func::FuncOp>("sentinel"));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

} // namespace
