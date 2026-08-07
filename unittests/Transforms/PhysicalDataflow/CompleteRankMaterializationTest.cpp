//===- CompleteRankMaterializationTest.cpp - Baseline utility tests -----===//

#include "Wafer/Transforms/CompleteRankMaterialization.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

static std::unique_ptr<mlir::MLIRContext> createContext() {
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
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

static std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

TEST(CompleteRankMaterializationTest,
     BuildsOneVerifiedUnplacedTileRegionWithoutMutatingSource) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<4xf16>) -> tensor<4xf16> {
    %out = tensor.empty() : tensor<4xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf16>) outs(%out : tensor<4xf16>) {
    ^bb0(%value: f16, %old: f16):
      %next = arith.negf %value : f16
      linalg.yield %next : f16
    } -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  const std::string sourceBefore = printModule(*source);

  auto baseline = wafer::materializeConservativeCompleteRankBaseline(
      *source, /*logicalRank=*/3);

  ASSERT_TRUE(mlir::succeeded(baseline));
  ASSERT_TRUE(*baseline);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**baseline)));
  EXPECT_EQ(printModule(*source), sourceBefore);

  unsigned tileRegions = 0;
  unsigned instructionOps = 0;
  bool hasPhysicalFact = false;
  (*baseline)->walk([&](mlir::Operation *operation) {
    tileRegions += mlir::isa<wafer::TileRegionOp>(operation);
    instructionOps +=
        mlir::isa<wafer::WaferInstructionOpInterface, wafer::SyncNCCJoinOp>(
            operation);
    hasPhysicalFact |= operation->hasAttr(wafer::kWaferSPMOffsetAttrName) ||
                       operation->hasAttr(wafer::kWaferDDROffsetAttrName);
  });
  EXPECT_EQ(tileRegions, 1u);
  EXPECT_EQ(instructionOps, 0u);
  EXPECT_FALSE(hasPhysicalFact);
}

TEST(CompleteRankMaterializationTest,
     OutlinesDenseConstantsAndClearsDerivedFactsWithoutStructuredRoots) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() -> tensor<4xf16> {
    %spm = memref.alloc() {
      wafer.spm.offset = #wafer.spm_offset<65536>
    } : memref<4xf16, #wafer.memory<spm, tensor>>
    memref.dealloc %spm : memref<4xf16, #wafer.memory<spm, tensor>>
    %constant = arith.constant dense<[1.0, 2.0, 3.0, 4.0]>
      : tensor<4xf16>
    return %constant : tensor<4xf16>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  const std::string sourceBefore = printModule(*source);

  auto baseline = wafer::materializeConservativeCompleteRankBaseline(
      *source, /*logicalRank=*/0);

  ASSERT_TRUE(mlir::succeeded(baseline));
  ASSERT_TRUE(*baseline);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**baseline)));
  EXPECT_EQ(printModule(*source), sourceBefore);

  unsigned tileRegions = 0;
  unsigned globals = 0;
  unsigned rankedTensorConstants = 0;
  bool hasPhysicalFact = false;
  (*baseline)->walk([&](mlir::Operation *operation) {
    tileRegions += mlir::isa<wafer::TileRegionOp>(operation);
    globals += mlir::isa<mlir::memref::GlobalOp>(operation);
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(operation))
      rankedTensorConstants +=
          mlir::isa<mlir::RankedTensorType>(constant.getType());
    hasPhysicalFact |= operation->hasAttr(wafer::kWaferSPMOffsetAttrName) ||
                       operation->hasAttr(wafer::kWaferDDROffsetAttrName);
  });
  EXPECT_EQ(tileRegions, 0u);
  EXPECT_EQ(globals, 1u);
  EXPECT_EQ(rankedTensorConstants, 0u);
  EXPECT_FALSE(hasPhysicalFact);
}

TEST(CompleteRankMaterializationTest,
     RankInvarianceDependsOnlyOnTypedCollectives) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto rankInvariant =
      mlir::parseSourceString<mlir::ModuleOp>("module {}", context.get());
  ASSERT_TRUE(rankInvariant);
  EXPECT_TRUE(wafer::isCompleteRankTensorProgramRankInvariant(*rankInvariant));

  auto rankDependent = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<1xf16>) -> tensor<1xf16> {
    %out = tensor.empty() : tensor<1xf16>
    %result = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<1xf16>) outs(%out : tensor<1xf16>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %sum = arith.addf %lhs, %rhs : f16
      wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 1 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<1xf16>
    return %result : tensor<1xf16>
  }
}
)mlir",
                                                               context.get());
  ASSERT_TRUE(rankDependent);
  EXPECT_FALSE(wafer::isCompleteRankTensorProgramRankInvariant(*rankDependent));
}

TEST(CompleteRankMaterializationTest,
     SplitResidencyRanksStaticSPMPhysicalBytesBeforeTileOpCount) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %ddr_a: memref<64xf16, #wafer.memory<ddr, tensor>>,
      %ddr_b: memref<1024xf16, #wafer.memory<ddr, tensor>>,
      %ddr_c: memref<896xf16, #wafer.memory<ddr, tensor>>,
      %zero: f16) {
    %result = wafer.tile.region(%ddr_a, %ddr_b, %ddr_c, %zero
        : memref<64xf16, #wafer.memory<ddr, tensor>>,
          memref<1024xf16, #wafer.memory<ddr, tensor>>,
          memref<896xf16, #wafer.memory<ddr, tensor>>, f16)
        -> (memref<896xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%a: memref<64xf16, #wafer.memory<ddr, tensor>>,
         %b: memref<1024xf16, #wafer.memory<ddr, tensor>>,
         %c: memref<896xf16, #wafer.memory<ddr, tensor>>,
         %value: f16):
      %resident_a = memref.alloc()
          : memref<64xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %a into %resident_a
          : memref<64xf16, #wafer.memory<ddr, tensor>>
         into memref<64xf16, #wafer.memory<spm, tensor>>
      wafer.tile.fill %resident_a, %value
          : memref<64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.fill %resident_a, %value
          : memref<64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.fill %resident_a, %value
          : memref<64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.fill %resident_a, %value
          : memref<64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.fill %resident_a, %value
          : memref<64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.fill %resident_a, %value
          : memref<64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.fill %resident_a, %value
          : memref<64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.fill %resident_a, %value
          : memref<64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.store %resident_a, %a
          : memref<64xf16, #wafer.memory<spm, tensor>>
         -> memref<64xf16, #wafer.memory<ddr, tensor>>

      %resident_b = memref.alloc()
          : memref<1024xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %b into %resident_b
          : memref<1024xf16, #wafer.memory<ddr, tensor>>
         into memref<1024xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %resident_b, %b
          : memref<1024xf16, #wafer.memory<spm, tensor>>
         -> memref<1024xf16, #wafer.memory<ddr, tensor>>

      %resident_c = memref.alloc()
          : memref<896xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %c into %resident_c
          : memref<896xf16, #wafer.memory<ddr, tensor>>
         into memref<896xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %resident_c, %c
          : memref<896xf16, #wafer.memory<spm, tensor>>
         -> memref<896xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %c
          : memref<896xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  std::string failureReason;
  auto split = wafer::materializeCompleteRankTileResidencySibling(
      *source, wafer::CandidateTileResidencyAction::SplitAtExplicitDDRBoundary,
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(split)) << failureReason;

  unsigned sourceRegionCount = 0;
  source->walk([&](wafer::TileRegionOp) { ++sourceRegionCount; });
  EXPECT_EQ(sourceRegionCount, 1u);
  llvm::SmallVector<wafer::TileRegionOp, 2> regions;
  (*split)->walk([&](wafer::TileRegionOp region) {
    if (!region->getParentOfType<wafer::TileRegionOp>())
      regions.push_back(region);
  });
  ASSERT_EQ(regions.size(), 2u);

  auto getAllocationElements = [](wafer::TileRegionOp region) {
    llvm::SmallVector<int64_t, 3> elements;
    for (mlir::memref::AllocOp allocation :
         region.getOps<mlir::memref::AllocOp>())
      if (wafer::isWaferSPMMemRefType(allocation.getType()))
        elements.push_back(allocation.getType().getNumElements());
    return elements;
  };
  EXPECT_EQ(getAllocationElements(regions[0]),
            (llvm::SmallVector<int64_t, 3>{64, 1024}));
  EXPECT_EQ(getAllocationElements(regions[1]),
            (llvm::SmallVector<int64_t, 3>{896}));
  for (wafer::TileRegionOp region : regions) {
    for (mlir::Value input : region.getInputs())
      if (mlir::isa<mlir::ShapedType>(input.getType()))
        EXPECT_TRUE(wafer::isWaferDDRMemRefType(input.getType()));
    for (mlir::Value result : region.getResults())
      if (mlir::isa<mlir::ShapedType>(result.getType()))
        EXPECT_TRUE(wafer::isWaferDDRMemRefType(result.getType()));
  }
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**split)));
}

} // namespace
