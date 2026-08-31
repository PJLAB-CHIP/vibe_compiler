//===- StructuredToTileTest.cpp ---------------------------------------===//

#include "Wafer/Transforms/Tile/StructuredToTile.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

template <typename OpTy> unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](OpTy) { ++count; });
  return count;
}

class StructuredToTileTest : public ::testing::Test {
protected:
  StructuredToTileTest() {
    registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        text, mlir::ParserConfig(context.get()));
  }

  static std::string makeSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id3 = affine_map<(b, m, n) -> (b, m, n)>
#row = affine_map<(b, m, n) -> (b, m)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%lhs: tensor<2x)mlir"
           << extent << R"mlir(x64xf16>, %rhs: tensor<2x64x32xf16>) {
      %result, %row_result = wafer.tile.region(
          %lhs, %rhs : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>, tensor<2x64x32xf16>)
          -> (tensor<2x)mlir"
           << extent << R"mlir(x32xf16>, tensor<2x)mlir" << extent
           << R"mlir(xf16>) {
      ^bb0(%local_lhs: tensor<2x)mlir"
           << extent << R"mlir(x64xf16>, %local_rhs: tensor<2x64x32xf16>):
        %zero = arith.constant 0.000000e+00 : f16
        %mm_empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>
        %mm_init = linalg.fill ins(%zero : f16)
            outs(%mm_empty : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>) -> tensor<2x)mlir" << extent
           << R"mlir(x32xf16>
        %mm = linalg.batch_matmul
            ins(%local_lhs, %local_rhs : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>, tensor<2x64x32xf16>)
            outs(%mm_init : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>) -> tensor<2x)mlir" << extent
           << R"mlir(x32xf16>
        %mapped_empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id3, #id3, #id3],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%mm, %mm : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>, tensor<2x)mlir" << extent
           << R"mlir(x32xf16>)
            outs(%mapped_empty : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>) {
          ^bb1(%lhs_value: f16, %rhs_value: f16, %old: f16):
            %difference = arith.subf %lhs_value, %rhs_value : f16
            %value = math.exp %difference : f16
            linalg.yield %value : f16
        } -> tensor<2x)mlir"
           << extent << R"mlir(x32xf16>
        %row_empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(xf16>
        %row_init = linalg.fill ins(%zero : f16)
            outs(%row_empty : tensor<2x)mlir"
           << extent << R"mlir(xf16>) -> tensor<2x)mlir" << extent
           << R"mlir(xf16>
        %reduced = linalg.generic {
            indexing_maps = [#id3, #row],
            iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%mapped : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>)
            outs(%row_init : tensor<2x)mlir"
           << extent << R"mlir(xf16>) {
          ^bb1(%value: f16, %acc: f16):
            %next = arith.addf %value, %acc : f16
            linalg.yield %next : f16
        } -> tensor<2x)mlir"
           << extent << R"mlir(xf16>
        wafer.tile.yield %mapped, %reduced
            : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>, tensor<2x)mlir" << extent
           << R"mlir(xf16>
      }
      return
    }
  }
}
)mlir";
    return text;
  }

  static std::string makeConvSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<1x)mlir"
           << extent + 2
           << R"mlir(x66x16xf16>, %weight: tensor<3x3x16x32xf16>) {
      %result = wafer.tile.region(
          %input, %weight : tensor<1x)mlir"
           << extent + 2 << R"mlir(x66x16xf16>, tensor<3x3x16x32xf16>)
          -> (tensor<1x)mlir"
           << extent << R"mlir(x64x32xf16>) {
      ^bb0(%local_input: tensor<1x)mlir"
           << extent + 2
           << R"mlir(x66x16xf16>, %local_weight: tensor<3x3x16x32xf16>):
        %zero = arith.constant 0.000000e+00 : f16
        %empty = tensor.empty() : tensor<1x)mlir"
           << extent << R"mlir(x64x32xf16>
        %init = linalg.fill ins(%zero : f16)
            outs(%empty : tensor<1x)mlir"
           << extent << R"mlir(x64x32xf16>) -> tensor<1x)mlir" << extent
           << R"mlir(x64x32xf16>
        %conv = linalg.conv_2d_nhwc_hwcf
            ins(%local_input, %local_weight : tensor<1x)mlir"
           << extent + 2 << R"mlir(x66x16xf16>, tensor<3x3x16x32xf16>)
            outs(%init : tensor<1x)mlir"
           << extent << R"mlir(x64x32xf16>) -> tensor<1x)mlir" << extent
           << R"mlir(x64x32xf16>
        wafer.tile.yield %conv : tensor<1x)mlir"
           << extent << R"mlir(x64x32xf16>
      }
      return
    }
  }
}
)mlir";
    return text;
  }

  static std::string makeTwoRegionSource(int64_t extent, bool crossTile) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @source(%input: tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) {
      %first = wafer.tile.region(
          %input : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
           << R"mlir(x64xf16>) {
      ^bb0(%local: tensor<1x)mlir"
           << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
      }
)mlir";
    if (!crossTile) {
      stream << R"mlir(      %second = wafer.tile.region(
          %first : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
             << R"mlir(x64xf16>) {
      ^bb0(%local: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.mulf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
      }
)mlir";
    }
    stream << R"mlir(      return
    }
  }
)mlir";
    if (crossTile) {
      stream << R"mlir(  wafer.tile.module card_id = 0 tile_id = 1 {
    func.func @destination(%placeholder: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
      %second = wafer.tile.region(
          %placeholder : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
             << R"mlir(x64xf16>) {
      ^bb0(%local: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.mulf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
      }
      return
    }
  }
)mlir";
    }
    stream << "}\n";
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredToTileTest,
       LowersContractionExpressionAndReductionAtRealisticScale) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeSource(extent));
    ASSERT_TRUE(module);
    TileRegionOp region;
    module->walk([&](TileRegionOp current) { region = current; });
    ASSERT_TRUE(region);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    relations.structuralOutputs.push_back({1, region.getResult(1)});
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;

    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    EXPECT_EQ(lowered.statistics.contractions, 1u);
    EXPECT_EQ(lowered.statistics.reductions, 1u);
    EXPECT_EQ(lowered.statistics.elementwiseExpressions, 1u);
    EXPECT_GE(lowered.statistics.elementwiseOperations, 2u);
    EXPECT_EQ(countOps<mlir::linalg::LinalgOp>(module->getOperation()), 0u);
    EXPECT_EQ(countOps<ComputeGemmOp>(module->getOperation()), 1u);
    EXPECT_EQ(countOps<ComputeReduceOp>(module->getOperation()), 1u);
    EXPECT_GE(countOps<ComputeElementwiseOp>(module->getOperation()), 2u);
    EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
    EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
        module->getOperation(), relations)));
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.ddrLoads, 2u);
    EXPECT_EQ(movement.statistics.ddrStores, 2u);
    EXPECT_EQ(movement.statistics.outputCopiesRemoved, 2u);
    EXPECT_TRUE(relations.structuralOutputs.empty());
    EXPECT_TRUE(relations.boundaryRelations.empty());
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
    EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
        module->getOperation(), relations)));
  }
}

TEST_F(StructuredToTileTest,
       LowersOrdinaryConvolutionFromCurrentMapsWithoutShapeNames) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeConvSource(extent));
    ASSERT_TRUE(module);
    TileRegionOp region;
    module->walk([&](TileRegionOp current) { region = current; });
    ASSERT_TRUE(region);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    EXPECT_EQ(lowered.statistics.convolutions, 1u);
    EXPECT_EQ(countOps<ComputeConvOp>(module->getOperation()), 1u);
    EXPECT_EQ(countOps<mlir::linalg::LinalgOp>(module->getOperation()), 0u);
    EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.ddrLoads, 2u);
    EXPECT_EQ(movement.statistics.ddrStores, 1u);
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
  }
}

TEST_F(StructuredToTileTest, SameTileRegionsUseOneExplicitDDRStage) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeTwoRegionSource(extent, /*crossTile=*/false));
    ASSERT_TRUE(module);
    llvm::SmallVector<TileRegionOp, 2> regions;
    module->walk([&](TileRegionOp region) { regions.push_back(region); });
    ASSERT_EQ(regions.size(), 2u);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, regions[1].getResult(0)});
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.interRegionDDRStages, 1u);
    EXPECT_EQ(movement.statistics.ddrLoads, 2u);
    EXPECT_EQ(movement.statistics.ddrStores, 2u);
    EXPECT_EQ(movement.statistics.peerSends, 0u);
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
  }
}

TEST_F(StructuredToTileTest, CrossTileRelationBecomesOneMatchedPeerTransfer) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeTwoRegionSource(extent, /*crossTile=*/true));
    ASSERT_TRUE(module);
    llvm::SmallVector<TileRegionOp, 2> regions;
    module->walk([&](TileRegionOp region) { regions.push_back(region); });
    ASSERT_EQ(regions.size(), 2u);
    StructuredMaterializationRelations relations;
    relations.boundaryRelations.push_back(
        {regions[0].getResult(0), regions[1].getBody().getArgument(0)});
    relations.structuralOutputs.push_back({0, regions[1].getResult(0)});
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.ddrLoads, 1u);
    EXPECT_EQ(movement.statistics.ddrStores, 1u);
    EXPECT_EQ(movement.statistics.peerSends, 1u);
    EXPECT_EQ(movement.statistics.peerReceives, 1u);
    EXPECT_EQ(movement.statistics.peerWaits, 2u);
    EXPECT_EQ(countOps<CommPeerSendOp>(module->getOperation()), 1u);
    EXPECT_EQ(countOps<CommPeerRecvOp>(module->getOperation()), 1u);
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
  }
}

} // namespace
