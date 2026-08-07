#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Pipelines/Pipelines.h"
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
     BatchedMatmulInitScalarSurvivesExactLeadingDimensionCollapse) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @collapsed_batch_matmul(
      %lhs: tensor<1x2x3x4xf16>, %rhs: tensor<1x2x4x5xf16>,
      %out: tensor<1x2x3x5xf16>) -> tensor<1x2x3x5xf16> {
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<1x2x3x5xf16>) -> tensor<1x2x3x5xf16>
    %lhs3 = tensor.collapse_shape %lhs [[0, 1], [2], [3]]
        : tensor<1x2x3x4xf16> into tensor<2x3x4xf16>
    %rhs3 = tensor.collapse_shape %rhs [[0, 1], [2], [3]]
        : tensor<1x2x4x5xf16> into tensor<2x4x5xf16>
    %init3 = tensor.collapse_shape %init [[0, 1], [2], [3]]
        : tensor<1x2x3x5xf16> into tensor<2x3x5xf16>
    %product = linalg.batch_matmul
        ins(%lhs3, %rhs3 : tensor<2x3x4xf16>, tensor<2x4x5xf16>)
        outs(%init3 : tensor<2x3x5xf16>) -> tensor<2x3x5xf16>
    %result = tensor.expand_shape %product [[0, 1], [2], [3]]
        output_shape [1, 2, 3, 5]
        : tensor<2x3x5xf16> into tensor<1x2x3x5xf16>
    return %result : tensor<1x2x3x5xf16>
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
    %ones = arith.constant dense<1.0> : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%ones, %input : tensor<4xf32>, tensor<4xf32>)
        outs(%out : tensor<4xf32>) {
    ^bb0(%one: f32, %value: f32, %init: f32):
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
     LowersNestedCollectivePermuteThroughTypedCollectiveDispatcher) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
    %condition = arith.constant true
    scf.if %condition {
      %out = tensor.empty() : tensor<4xf32>
      %permuted = wafer.linalg_ext.collective.collective_permute
          ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>)
          {source_target_pairs = array<i64: 0, 1, 1, 0>,
           channel_id = 91 : i64} -> tensor<4xf32>
      scf.yield
    }
    %result_out = tensor.empty() : tensor<4xf32>
    %result = linalg.map ins(%input : tensor<4xf32>)
        outs(%result_out : tensor<4xf32>)
        (%value: f32) {
          linalg.yield %value : f32
        }
    return %result : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(wafer::lowerCompleteRankTensorProgramToTileRegionModule(
          *source, lowered, &failureReason, /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  EXPECT_EQ(countOps<wafer::TileRegionOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::CommPeerSendOp>(*lowered), 1u);
  EXPECT_EQ(countOps<wafer::CommPeerRecvOp>(*lowered), 1u);
}

TEST(WaferTensorProgramToTileRegionTest, LowersF16GenericReductionChunks) {
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
  llvm::SmallVector<wafer::ComputeElementwiseOp, 2> chunkBodies;
  llvm::SmallVector<wafer::ComputeElementwiseOp, 1> merges;
  lowered->walk([&](wafer::ComputeReduceOp op) { reductions.push_back(op); });
  lowered->walk([&](wafer::ComputeElementwiseOp op) {
    auto type = mlir::cast<mlir::MemRefType>(op.getResult().getType());
    if (type.getRank() == 2)
      chunkBodies.push_back(op);
    if (type.getRank() == 1)
      merges.push_back(op);
  });
  ASSERT_EQ(reductions.size(), 2u);
  ASSERT_EQ(chunkBodies.size(), 2u);
  ASSERT_EQ(merges.size(), 1u);
  auto firstInputType =
      mlir::cast<mlir::MemRefType>(reductions[0].getInput().getType());
  auto tailInputType =
      mlir::cast<mlir::MemRefType>(reductions[1].getInput().getType());
  EXPECT_EQ(firstInputType.getDimSize(1), 3);
  EXPECT_EQ(tailInputType.getDimSize(1), 2);
  for (auto [chunkBody, reduction] :
       llvm::zip_equal(chunkBodies, reductions)) {
    EXPECT_EQ(chunkBody.getKind(), wafer::ComputeElementwiseKind::Add);
    auto reductionInput =
        reduction.getInput().getDefiningOp<wafer::LayoutMaterializeOp>();
    ASSERT_TRUE(reductionInput);
    EXPECT_EQ(reductionInput.getSource(), chunkBody.getResult());
  }
  EXPECT_EQ(merges[0].getKind(), wafer::ComputeElementwiseKind::Add);
  ASSERT_EQ(merges[0].getInputs().size(), 2u);
  auto previousLayout =
      merges[0].getInputs()[0].getDefiningOp<wafer::LayoutMaterializeOp>();
  auto partialLayout =
      merges[0].getInputs()[1].getDefiningOp<wafer::LayoutMaterializeOp>();
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
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(*singleTile), 3u);
}

TEST(WaferTensorProgramToTileRegionTest, LowersBF16MaximumChunks) {
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
  llvm::SmallVector<wafer::ComputeElementwiseOp, 2> chunkBodies;
  llvm::SmallVector<wafer::ComputeElementwiseOp, 1> merges;
  lowered->walk([&](wafer::ComputeReduceOp op) { reductions.push_back(op); });
  lowered->walk([&](wafer::ComputeElementwiseOp op) {
    auto type = mlir::cast<mlir::MemRefType>(op.getResult().getType());
    if (type.getRank() == 2)
      chunkBodies.push_back(op);
    if (type.getRank() == 1)
      merges.push_back(op);
  });
  ASSERT_EQ(reductions.size(), 2u);
  ASSERT_EQ(chunkBodies.size(), 2u);
  ASSERT_EQ(merges.size(), 1u);
  llvm::SmallVector<int64_t, 2> chunkExtents;
  for (auto [chunkBody, reduction] :
       llvm::zip_equal(chunkBodies, reductions)) {
    EXPECT_EQ(chunkBody.getKind(), wafer::ComputeElementwiseKind::Max);
    auto reductionInput =
        reduction.getInput().getDefiningOp<wafer::LayoutMaterializeOp>();
    ASSERT_TRUE(reductionInput);
    EXPECT_EQ(reductionInput.getSource(), chunkBody.getResult());
    auto inputType =
        mlir::cast<mlir::MemRefType>(reduction.getInput().getType());
    EXPECT_TRUE(mlir::isa<mlir::BFloat16Type>(inputType.getElementType()));
    chunkExtents.push_back(inputType.getDimSize(1));
  }
  EXPECT_EQ(chunkExtents, (llvm::SmallVector<int64_t, 2>{3, 2}));
  EXPECT_EQ(merges[0].getKind(), wafer::ComputeElementwiseKind::Max);
  ASSERT_EQ(merges[0].getInputs().size(), 2u);
  auto previousLayout =
      merges[0].getInputs()[0].getDefiningOp<wafer::LayoutMaterializeOp>();
  auto partialLayout =
      merges[0].getInputs()[1].getDefiningOp<wafer::LayoutMaterializeOp>();
  ASSERT_TRUE(previousLayout);
  ASSERT_TRUE(partialLayout);
  EXPECT_EQ(previousLayout.getSource(), reductions[0].getResult());
  EXPECT_EQ(partialLayout.getSource(), reductions[1].getResult());
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

  llvm::SmallVector<wafer::ComputeReduceOp, 2> reductions;
  llvm::SmallVector<wafer::ComputeElementwiseOp, 2> products;
  llvm::SmallVector<wafer::ComputeElementwiseOp, 2> chunkBodies;
  llvm::SmallVector<wafer::ComputeElementwiseOp, 1> merges;
  lowered->walk([&](wafer::ComputeReduceOp op) { reductions.push_back(op); });
  lowered->walk([&](wafer::ComputeElementwiseOp op) {
    auto type = mlir::cast<mlir::MemRefType>(op.getResult().getType());
    if (type.getRank() == 3 &&
        op.getKind() == wafer::ComputeElementwiseKind::Mul)
      products.push_back(op);
    if (type.getRank() == 3 &&
        op.getKind() == wafer::ComputeElementwiseKind::Add)
      chunkBodies.push_back(op);
    if (type.getRank() == 2 &&
        op.getKind() == wafer::ComputeElementwiseKind::Add)
      merges.push_back(op);
  });
  ASSERT_EQ(reductions.size(), 2u);
  ASSERT_EQ(products.size(), 2u);
  ASSERT_EQ(chunkBodies.size(), 2u);
  ASSERT_EQ(merges.size(), 1u);

  llvm::SmallVector<int64_t, 2> contractingSizes;
  for (auto [product, chunkBody, reduction] :
       llvm::zip_equal(products, chunkBodies, reductions)) {
    auto productType =
        mlir::cast<mlir::MemRefType>(product.getResult().getType());
    contractingSizes.push_back(productType.getDimSize(2));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(productType.getElementType()));
    ASSERT_EQ(product.getInputs().size(), 2u);
    EXPECT_TRUE(product.getInputs()[0].getDefiningOp<wafer::MoveBroadcastOp>());
    EXPECT_TRUE(product.getInputs()[1].getDefiningOp<wafer::MoveBroadcastOp>());
    EXPECT_TRUE(llvm::is_contained(chunkBody.getInputs(), product.getResult()));
    auto reductionInput =
        reduction.getInput().getDefiningOp<wafer::LayoutMaterializeOp>();
    ASSERT_TRUE(reductionInput);
    EXPECT_EQ(reductionInput.getSource(), chunkBody.getResult());
  }
  EXPECT_EQ(contractingSizes, (llvm::SmallVector<int64_t, 2>{3, 2}));

  ASSERT_EQ(merges[0].getInputs().size(), 2u);
  auto previousLayout =
      merges[0].getInputs()[0].getDefiningOp<wafer::LayoutMaterializeOp>();
  auto partialLayout =
      merges[0].getInputs()[1].getDefiningOp<wafer::LayoutMaterializeOp>();
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

    llvm::SmallVector<wafer::ComputeReduceOp, 2> reductions;
    llvm::SmallVector<wafer::ComputeElementwiseOp, 2> products;
    llvm::SmallVector<wafer::ComputeElementwiseOp, 2> chunkBodies;
    llvm::SmallVector<wafer::ComputeElementwiseOp, 1> merges;
    lowered->walk(
        [&](wafer::ComputeReduceOp op) { reductions.push_back(op); });
    lowered->walk([&](wafer::ComputeElementwiseOp op) {
      auto type = mlir::cast<mlir::MemRefType>(op.getResult().getType());
      if (type.getRank() == 3 &&
          op.getKind() == wafer::ComputeElementwiseKind::Mul)
        products.push_back(op);
      if (type.getRank() == 3 &&
          op.getKind() == wafer::ComputeElementwiseKind::Add)
        chunkBodies.push_back(op);
      if (type.getRank() == 2 &&
          op.getKind() == wafer::ComputeElementwiseKind::Add)
        merges.push_back(op);
    });
    ASSERT_EQ(reductions.size(), 2u) << typeSpelling.str();
    ASSERT_EQ(products.size(), 2u) << typeSpelling.str();
    ASSERT_EQ(chunkBodies.size(), 2u) << typeSpelling.str();
    ASSERT_EQ(merges.size(), 1u) << typeSpelling.str();

    llvm::SmallVector<int64_t, 2> contractingSizes;
    for (auto [product, chunkBody, reduction] :
         llvm::zip_equal(products, chunkBodies, reductions)) {
      auto productType =
          mlir::cast<mlir::MemRefType>(product.getResult().getType());
      contractingSizes.push_back(productType.getDimSize(2));
      if (typeSpelling == "f16")
        EXPECT_TRUE(
            mlir::isa<mlir::Float16Type>(productType.getElementType()));
      else
        EXPECT_TRUE(
            mlir::isa<mlir::BFloat16Type>(productType.getElementType()));
      ASSERT_EQ(product.getInputs().size(), 2u);
      EXPECT_TRUE(
          product.getInputs()[0].getDefiningOp<wafer::MoveBroadcastOp>());
      EXPECT_TRUE(
          product.getInputs()[1].getDefiningOp<wafer::MoveBroadcastOp>());
      EXPECT_TRUE(
          llvm::is_contained(chunkBody.getInputs(), product.getResult()));
      auto reductionInput =
          reduction.getInput().getDefiningOp<wafer::LayoutMaterializeOp>();
      ASSERT_TRUE(reductionInput);
      EXPECT_EQ(reductionInput.getSource(), chunkBody.getResult());
    }
    EXPECT_EQ(contractingSizes, (llvm::SmallVector<int64_t, 2>{3, 2}))
        << typeSpelling.str();

    ASSERT_EQ(merges[0].getInputs().size(), 2u);
    auto previousLayout =
        merges[0].getInputs()[0].getDefiningOp<wafer::LayoutMaterializeOp>();
    auto partialLayout =
        merges[0].getInputs()[1].getDefiningOp<wafer::LayoutMaterializeOp>();
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
     MaterializesSplatConstantsAtTheirConcreteFullOrSlicedDemand) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @scale(%input: tensor<8xf32>, %out: tensor<8xf32>)
      -> tensor<8xf32> {
    %scale = arith.constant dense<1.250000e-01> : tensor<8xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%input, %scale : tensor<8xf32>, tensor<8xf32>)
        outs(%out : tensor<8xf32>) {
    ^bb0(%value: f32, %factor: f32, %init: f32):
      %scaled = arith.mulf %value, %factor : f32
      linalg.yield %scaled : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  ASSERT_TRUE(function);

  mlir::OwningOpRef<mlir::ModuleOp> full;
  mlir::OwningOpRef<mlir::ModuleOp> tiled;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToTileRegionModule(
      function, full, &failureReason, /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{}, tiled, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;

  auto collectFillShapes = [](mlir::ModuleOp module) {
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 2> shapes;
    module.walk([&](wafer::ComputeFillOp fill) {
      auto type = mlir::dyn_cast<mlir::MemRefType>(fill.getDest().getType());
      if (type)
        shapes.emplace_back(type.getShape().begin(), type.getShape().end());
    });
    return shapes;
  };
  auto fullFillShapes = collectFillShapes(*full);
  auto tiledFillShapes = collectFillShapes(*tiled);
  ASSERT_EQ(fullFillShapes.size(), 1u);
  ASSERT_EQ(tiledFillShapes.size(), 1u);
  EXPECT_TRUE(llvm::equal(fullFillShapes.front(), llvm::ArrayRef<int64_t>{8}));
  EXPECT_TRUE(
      llvm::equal(tiledFillShapes.front(), llvm::ArrayRef<int64_t>{2}));
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

  // The extract and insert are both external DDR views.  Preserve their
  // source/destination identity and carry the row through one rank-one SPM
  // value; materializing the full output in SPM would be redundant.
  llvm::SmallVector<wafer::StorageLoadOp, 1> loads;
  llvm::SmallVector<wafer::StorageStoreOp, 1> stores;
  llvm::SmallVector<wafer::TileRegionOp, 1> regions;
  lowered->walk([&](wafer::StorageLoadOp load) { loads.push_back(load); });
  lowered->walk([&](wafer::StorageStoreOp store) { stores.push_back(store); });
  lowered->walk([&](wafer::TileRegionOp region) { regions.push_back(region); });
  ASSERT_EQ(loads.size(), 1u);
  ASSERT_EQ(stores.size(), 1u);
  ASSERT_EQ(regions.size(), 1u);

  auto inputSubview =
      loads.front().getSource().getDefiningOp<mlir::memref::SubViewOp>();
  auto outputSubview =
      stores.front().getDest().getDefiningOp<mlir::memref::SubViewOp>();
  ASSERT_TRUE(inputSubview);
  ASSERT_TRUE(outputSubview);
  EXPECT_EQ(formatStaticSlice(inputSubview), "0,0:1,4");
  EXPECT_EQ(formatStaticSlice(outputSubview), "1,0:1,4");
  EXPECT_EQ(inputSubview.getSourceType().getRank(), 2);
  EXPECT_EQ(inputSubview.getType().getRank(), 1);
  EXPECT_EQ(outputSubview.getSourceType().getRank(), 2);
  EXPECT_EQ(outputSubview.getType().getRank(), 1);

  mlir::Block &body = regions.front().getBody().front();
  ASSERT_EQ(body.getNumArguments(), 2u);
  EXPECT_EQ(inputSubview.getSource(), body.getArgument(0));
  EXPECT_EQ(outputSubview.getSource(), body.getArgument(1));
  EXPECT_EQ(loads.front().getDest(), stores.front().getSource());
  auto rowBufferType =
      mlir::cast<mlir::MemRefType>(loads.front().getDest().getType());
  EXPECT_TRUE(wafer::isWaferSPMMemRefType(rowBufferType));
  EXPECT_EQ(rowBufferType.getShape(), llvm::ArrayRef<int64_t>({4}));
  EXPECT_TRUE(loads.front().getDest().getDefiningOp<mlir::memref::AllocOp>());

  auto yield = mlir::dyn_cast<wafer::TileYieldOp>(body.getTerminator());
  ASSERT_TRUE(yield);
  ASSERT_EQ(yield.getValues().size(), 1u);
  EXPECT_EQ(yield.getValues().front(), body.getArgument(1));
  ASSERT_EQ(regions.front().getNumResults(), 1u);
  ASSERT_TRUE(regions.front().getResult(0).hasOneUse());
  auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(
      *regions.front().getResult(0).getUsers().begin());
  ASSERT_TRUE(toTensor);
  ASSERT_TRUE(toTensor.getResult().hasOneUse());
  EXPECT_TRUE(mlir::isa<mlir::func::ReturnOp>(
      *toTensor.getResult().getUsers().begin()));

  EXPECT_EQ(countOps<wafer::MoveExtractSliceOp>(*lowered), 0u);
  EXPECT_EQ(countOps<wafer::MoveInsertSliceOp>(*lowered), 0u);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  llvm::SmallVector<wafer::InstrRDMAOp, 1> rdmas;
  llvm::SmallVector<wafer::InstrWDMAOp, 1> wdmas;
  lowered->walk([&](wafer::InstrRDMAOp rdma) { rdmas.push_back(rdma); });
  lowered->walk([&](wafer::InstrWDMAOp wdma) { wdmas.push_back(wdma); });
  ASSERT_EQ(rdmas.size(), 1u);
  ASSERT_EQ(wdmas.size(), 1u);
  EXPECT_EQ(rdmas.front().getSource(), inputSubview.getResult());
  EXPECT_EQ(rdmas.front().getDest(), wdmas.front().getSource());
  EXPECT_EQ(wdmas.front().getDest(), outputSubview.getResult());
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*lowered), 0u);
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
     StreamsLargeExternalInsertChainThroughBoundedSPMTiles) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @cache_append(
      %past: tensor<1x32x1023x128xf16>,
      %next: tensor<1x32x1x128xf16>)
      -> (tensor<1x32x1024x128xf16>, tensor<1x32x1x128xf16>) {
    %cache_empty = tensor.empty() : tensor<1x32x1024x128xf16>
    %prefix = tensor.insert_slice %past into %cache_empty[0, 0, 0, 0]
        [1, 32, 1023, 128] [1, 1, 1, 1]
        : tensor<1x32x1023x128xf16> into tensor<1x32x1024x128xf16>
    %updated = tensor.insert_slice %next into %prefix[0, 0, 1023, 0]
        [1, 32, 1, 128] [1, 1, 1, 1]
        : tensor<1x32x1x128xf16> into tensor<1x32x1024x128xf16>
    %tail = tensor.extract_slice %updated[0, 0, 1023, 0]
        [1, 32, 1, 128] [1, 1, 1, 1]
        : tensor<1x32x1024x128xf16> to tensor<1x32x1x128xf16>
    %mapped_empty = tensor.empty() : tensor<1x32x1x128xf16>
    %mapped = linalg.map
        ins(%tail : tensor<1x32x1x128xf16>)
        outs(%mapped_empty : tensor<1x32x1x128xf16>)
        (%value: f16) {
          linalg.yield %value : f16
        }
    return %updated, %mapped
        : tensor<1x32x1024x128xf16>, tensor<1x32x1x128xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(
      mlir::succeeded(wafer::lowerCompleteRankTensorProgramToTileRegionModule(
          *source, lowered, &failureReason, /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*lowered)));

  unsigned fullCacheSPMAllocations = 0;
  uint64_t largestSPMAllocationBytes = 0;
  lowered->walk([&](mlir::memref::AllocOp alloc) {
    mlir::MemRefType type = alloc.getType();
    if (!wafer::isWaferSPMMemRefType(type))
      return;
    uint64_t elements = 1;
    for (int64_t extent : type.getShape())
      elements *= static_cast<uint64_t>(extent);
    uint64_t bytes =
        elements * static_cast<uint64_t>(
                       type.getElementType().getIntOrFloatBitWidth() / 8);
    largestSPMAllocationBytes = std::max(largestSPMAllocationBytes, bytes);
    fullCacheSPMAllocations +=
        type.getShape() == llvm::ArrayRef<int64_t>({1, 32, 1024, 128});
  });
  EXPECT_EQ(fullCacheSPMAllocations, 0u);
  EXPECT_LE(largestSPMAllocationBytes, (3080192u - 65536u) / 4u);
  EXPECT_GT(countOps<wafer::StorageLoadOp>(*lowered), 0u);
  EXPECT_GT(countOps<wafer::StorageStoreOp>(*lowered), 0u);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  mlir::PassManager preparation(&context);
  wafer::buildPrepareScheduledRankCandidatePipeline(preparation);
  ASSERT_TRUE(mlir::succeeded(preparation.run(*lowered)));
  ASSERT_TRUE(mlir::succeeded(wafer::rebuildMinimumNCCJoins(*lowered)));
  EXPECT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
      *lowered, /*spmBase=*/65536, /*spmLimit=*/3080192,
      /*spmAlignment=*/256)));
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
            "ordered reduction chunk chain requires exactly one reduction "
            "axis");
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
          wafer::CandidateTileTraversalKind::ResultDriven)))
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
  EXPECT_TRUE(
      mlir::failed(wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{4}, lowered, &failureReason,
          /*currentLogicalRank=*/0, /*selectedAlternative=*/std::nullopt,
          /*useDirectMappedBoundaryTransfer=*/false,
          wafer::CandidateTileTraversalKind::ResultDriven)));
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
          wafer::CandidateTileTraversalKind::ResultDriven)))
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
  EXPECT_TRUE((mlir::isa<wafer::ComputeElementwiseOp, wafer::ComputeReduceOp>(
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
  EXPECT_TRUE(
      mlir::failed(wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
          function, /*candidateTileSizes=*/{2},
          /*candidateReductionTileSizes=*/{4}, lowered, &failureReason,
          /*currentLogicalRank=*/0, /*selectedAlternative=*/std::nullopt,
          /*useDirectMappedBoundaryTransfer=*/false,
          wafer::CandidateTileTraversalKind::ResultDriven)));
  EXPECT_EQ(failureReason,
            "candidate reduction split cannot preserve unsigned min/max "
            "semantics with the current reduce kind");
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(lowered->lookupSymbol<mlir::func::FuncOp>("sentinel"));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

TEST(WaferTensorProgramToTileRegionTest,
     LowersBlockedPointwiseAndCompatibleConvertWithoutMaterialization) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @blocked_pointwise(
      %lhs: memref<3x65xf16, #wafer.memory<spm, cx>>,
      %rhs: memref<3x65xf16, #wafer.memory<spm, cx>>)
      -> memref<3x65xf16, #wafer.memory<spm, cx>> {
    %sum = wafer.tile.elementwise #wafer.elementwise_kind<add> %lhs, %rhs
        : (memref<3x65xf16, #wafer.memory<spm, cx>>,
           memref<3x65xf16, #wafer.memory<spm, cx>>)
       -> memref<3x65xf16, #wafer.memory<spm, cx>>
    return %sum : memref<3x65xf16, #wafer.memory<spm, cx>>
  }

  func.func @blocked_convert(
      %input: memref<2x2x197xf16, #wafer.memory<spm, ncx>>)
      -> memref<2x2x197xbf16, #wafer.memory<spm, ncx>> {
    %converted = wafer.tile.compute.convert %input
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
       to memref<2x2x197xbf16, #wafer.memory<spm, ncx>>
    return %converted : memref<2x2x197xbf16, #wafer.memory<spm, ncx>>
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
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(*module), 0u);
  EXPECT_EQ(countOps<wafer::ComputeConvertOp>(*module), 0u);
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*module), 1u);
  EXPECT_EQ(countOps<wafer::InstrConvertOp>(*module), 1u);
  EXPECT_EQ(countOps<wafer::LayoutMaterializeOp>(*module), 0u);
  EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
}

TEST(WaferTensorProgramToTileRegionTest,
     RejectsBlockedConvertWithDtypeSpecificTraversalMismatch) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @blocked_convert(
      %input: memref<2x2x197xf16, #wafer.memory<spm, ncx>>)
      -> memref<2x2x197xf32, #wafer.memory<spm, ncx>> {
    %converted = wafer.tile.compute.convert %input
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
       to memref<2x2x197xf32, #wafer.memory<spm, ncx>>
    return %converted : memref<2x2x197xf32, #wafer.memory<spm, ncx>>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  std::string failureReason;
  EXPECT_TRUE(mlir::failed(
      wafer::convertTileRegionToInstrModule(*module, &failureReason)));
  EXPECT_NE(failureReason.find("incompatible physical element traversal"),
            std::string::npos)
      << failureReason;
  EXPECT_EQ(countOps<wafer::ComputeConvertOp>(*module), 1u);
  EXPECT_EQ(countOps<wafer::InstrConvertOp>(*module), 0u);
}

TEST(WaferTensorProgramToTileRegionTest,
     ConnectionIdentityDistinguishesTwoOperandsOfOneConsumer) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @two_edges(%input: tensor<8xf16>) -> tensor<8xf16> {
    %producer_out = tensor.empty() : tensor<8xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<8xf16>) outs(%producer_out : tensor<8xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<8xf16>
    %consumer_out = tensor.empty() : tensor<8xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer, %producer : tensor<8xf16>, tensor<8xf16>)
      outs(%consumer_out : tensor<8xf16>) {
    ^bb0(%left: f16, %right: f16, %unused: f16):
      %sum = arith.addf %left, %right : f16
      linalg.yield %sum : f16
    } -> tensor<8xf16>
    return %consumer : tensor<8xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  std::string failureReason;
  auto topology = wafer::getCompleteRankCandidateConnectionTopology(
      *source, &failureReason);
  ASSERT_TRUE(mlir::succeeded(topology)) << failureReason;
  ASSERT_EQ(topology->domains.size(), 2u);
  EXPECT_EQ(topology->connectionCount, 2u);
  EXPECT_TRUE(topology->requiresGeneralDAGBeam);
  const auto &left = topology->domains[0];
  const auto &right = topology->domains[1];
  EXPECT_EQ(left.producerOperationOrdinal, right.producerOperationOrdinal);
  EXPECT_EQ(left.producerResultNumber, 0u);
  EXPECT_EQ(right.producerResultNumber, 0u);
  EXPECT_EQ(left.consumerOperationOrdinal, right.consumerOperationOrdinal);
  EXPECT_EQ(left.consumerOperandNumber, 0u);
  EXPECT_EQ(right.consumerOperandNumber, 1u);
  EXPECT_EQ(left.producerResultShape,
            (llvm::SmallVector<int64_t, 4>{8}));
  EXPECT_EQ(left.consumerOperandShape,
            (llvm::SmallVector<int64_t, 4>{8}));
  EXPECT_EQ(left.consumerResultShape,
            (llvm::SmallVector<int64_t, 4>{8}));

  wafer::CandidateTraversalConnectionChoice invalidCoupled;
  invalidCoupled.action =
      wafer::CandidateTraversalConnectionAction::CoupledResident;
  invalidCoupled.producerTileSizes = {2};
  invalidCoupled.consumerTileSizes = {4};
  wafer::CandidateTraversalConnectionChoice separated;
  separated.action =
      wafer::CandidateTraversalConnectionAction::SeparatedDDR;
  separated.producerTileSizes = {2};
  separated.consumerTileSizes = {4};
  mlir::OwningOpRef<mlir::ModuleOp> rejected;
  llvm::SmallVector<wafer::CandidateTraversalConnectionChoice, 2>
      invalidChoices{invalidCoupled, separated};
  EXPECT_TRUE(mlir::failed(
      wafer::lowerCompleteRankConnectionChoicesTensorProgramToTileRegionModule(
          *source, invalidChoices, rejected, &failureReason,
          /*currentLogicalRank=*/0)));
  EXPECT_NE(failureReason.find("cannot select a producer tile independently"),
            std::string::npos)
      << failureReason;

  wafer::CandidateTraversalConnectionChoice coupled;
  coupled.action = wafer::CandidateTraversalConnectionAction::CoupledResident;
  coupled.consumerTileSizes = {4};
  llvm::SmallVector<wafer::CandidateTraversalConnectionChoice, 2> choices{
      coupled, separated};
  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  ASSERT_TRUE(mlir::succeeded(
      wafer::lowerCompleteRankConnectionChoicesTensorProgramToTileRegionModule(
          *source, choices, lowered, &failureReason,
          /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  llvm::SmallVector<int64_t, 4> loopSteps;
  lowered->walk([&](mlir::scf::ForOp loop) {
    std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
    if (step)
      loopSteps.push_back(*step);
  });
  EXPECT_TRUE(llvm::is_contained(loopSteps, 4));
  EXPECT_TRUE(llvm::is_contained(loopSteps, 2));
}

TEST(WaferTensorProgramToTileRegionTest,
     LowersGenericNaturalLogThroughTypedInstructionKind) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @natural_log(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %logged = math.log %value : f32
      linalg.yield %logged : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = findSingleTensorProgram(*source);
  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToTileRegionModule(
      function, lowered, &failureReason, /*currentLogicalRank=*/0)))
      << failureReason;
  ASSERT_TRUE(lowered);
  llvm::SmallVector<wafer::ComputeElementwiseOp, 1> tileOps;
  lowered->walk([&](wafer::ComputeElementwiseOp op) { tileOps.push_back(op); });
  ASSERT_EQ(tileOps.size(), 1u);
  EXPECT_EQ(tileOps.front().getKind(), wafer::ComputeElementwiseKind::Ln);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*lowered, &failureReason)))
      << failureReason;
  llvm::SmallVector<wafer::InstrElementwiseOp, 1> instrOps;
  lowered->walk([&](wafer::InstrElementwiseOp op) { instrOps.push_back(op); });
  ASSERT_EQ(instrOps.size(), 1u);
  EXPECT_EQ(instrOps.front().getKind(), wafer::InstrElementwiseKind::Ln);
}

TEST(WaferTensorProgramToTileRegionTest,
     StandaloneFlashAttention2PassMaterializesStateWithoutProbabilityStorage) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @attention(%scores: tensor<2x7xf16>,
                       %values: tensor<7x3xf16>,
                       %projection: tensor<3x4xf16>,
                       %out: tensor<2x4xf16>) -> tensor<2x4xf16> {
    %zero = arith.constant 0.0 : f16
    %lowest = arith.constant -6.550400e+04 : f16
    %max_empty = tensor.empty() : tensor<2xf16>
    %max_init = linalg.fill ins(%lowest : f16)
        outs(%max_empty : tensor<2xf16>) -> tensor<2xf16>
    %row_max = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%scores : tensor<2x7xf16>)
        outs(%max_init : tensor<2xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %next = arith.maximumf %acc, %value : f16
      linalg.yield %next : f16
    } -> tensor<2xf16>
    %max_broadcast_empty = tensor.empty() : tensor<2x7xf16>
    %max_broadcast = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0)>,
          affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%row_max : tensor<2xf16>)
        outs(%max_broadcast_empty : tensor<2x7xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<2x7xf16>
    %shift_empty = tensor.empty() : tensor<2x7xf16>
    %shifted = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%scores, %max_broadcast : tensor<2x7xf16>, tensor<2x7xf16>)
        outs(%shift_empty : tensor<2x7xf16>) {
    ^bb0(%value: f16, %maximum: f16, %unused: f16):
      %next = arith.subf %value, %maximum : f16
      linalg.yield %next : f16
    } -> tensor<2x7xf16>
    %exp_empty = tensor.empty() : tensor<2x7xf16>
    %exponential = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%shifted : tensor<2x7xf16>)
        outs(%exp_empty : tensor<2x7xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %next = math.exp %value : f16
      linalg.yield %next : f16
    } -> tensor<2x7xf16>
    %sum_empty = tensor.empty() : tensor<2xf16>
    %sum_init = linalg.fill ins(%zero : f16)
        outs(%sum_empty : tensor<2xf16>) -> tensor<2xf16>
    %row_sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%exponential : tensor<2x7xf16>)
        outs(%sum_init : tensor<2xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %next = arith.addf %acc, %value : f16
      linalg.yield %next : f16
    } -> tensor<2xf16>
    %sum_broadcast_empty = tensor.empty() : tensor<2x7xf16>
    %sum_broadcast = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0)>,
          affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%row_sum : tensor<2xf16>)
        outs(%sum_broadcast_empty : tensor<2x7xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<2x7xf16>
    %prob_empty = tensor.empty() : tensor<2x7xf16>
    %probability = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%exponential, %sum_broadcast
          : tensor<2x7xf16>, tensor<2x7xf16>)
        outs(%prob_empty : tensor<2x7xf16>) {
    ^bb0(%numerator: f16, %denominator: f16, %unused: f16):
      %next = arith.divf %numerator, %denominator : f16
      linalg.yield %next : f16
    } -> tensor<2x7xf16>
    %attention_empty = tensor.empty() : tensor<2x3xf16>
    %attention_init = linalg.fill ins(%zero : f16)
        outs(%attention_empty : tensor<2x3xf16>) -> tensor<2x3xf16>
    %attention_output = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%probability, %values : tensor<2x7xf16>, tensor<7x3xf16>)
        outs(%attention_init : tensor<2x3xf16>) {
    ^bb0(%prob: f16, %value: f16, %acc: f16):
      %product = arith.mulf %prob, %value : f16
      %next = arith.addf %acc, %product : f16
      linalg.yield %next : f16
    } -> tensor<2x3xf16>
    %out_init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x4xf16>) -> tensor<2x4xf16>
    %output = linalg.matmul
        ins(%attention_output, %projection
            : tensor<2x3xf16>, tensor<3x4xf16>)
        outs(%out_init : tensor<2x4xf16>) -> tensor<2x4xf16>
    return %output : tensor<2x4xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  wafer::MaterializeFlashAttention2PassOptions options;
  options.outputTileSizes = "1,3";
  options.keyValueTileSize = 3;
  mlir::PassManager materialization(&context);
  materialization.addPass(wafer::createMaterializeFlashAttention2Pass(options));
  ASSERT_TRUE(mlir::succeeded(materialization.run(*source)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
  EXPECT_EQ(countOps<mlir::scf::IfOp>(*source), 0u);
  unsigned loopCarriedFlashStates = 0;
  source->walk([&](mlir::scf::ForOp loop) {
    loopCarriedFlashStates += loop.getRegionIterArgs().size() == 3;
  });
  EXPECT_GT(loopCarriedFlashStates, 0u);

  unsigned fullProbabilityDestinations = 0;
  source->walk([&](mlir::tensor::EmptyOp empty) {
    mlir::RankedTensorType type = empty.getType();
    fullProbabilityDestinations +=
        type.getRank() == 2 && type.getDimSize(0) == 2 &&
        type.getDimSize(1) == 7;
  });
  EXPECT_EQ(fullProbabilityDestinations, 0u);
}

TEST(WaferTensorProgramToTileRegionTest,
     StandaloneFlashAttention2PassRejectsSemanticNonMatch) {
  mlir::DialectRegistry registry;
  registerConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @standalone_probability(%probability: tensor<2x7xf16>)
      -> tensor<2x7xf16> {
    return %probability : tensor<2x7xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  wafer::MaterializeFlashAttention2PassOptions options;
  options.outputTileSizes = "1,3";
  options.keyValueTileSize = 3;
  mlir::PassManager materialization(&context);
  materialization.addPass(wafer::createMaterializeFlashAttention2Pass(options));
  EXPECT_TRUE(mlir::failed(materialization.run(*source)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
}

} // namespace
