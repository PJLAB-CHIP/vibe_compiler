//===- GatherLoweringTest.cpp - Indexed slice tiling and storage --------===//

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "Wafer/Transforms/Tile/GatherLowering.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {
using namespace wafer;
using namespace wafer::compiler::detail;

class GatherLoweringTest : public ::testing::Test {
protected:
  GatherLoweringTest() {
    registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(int64_t length,
                                         llvm::StringRef dtype,
                                         bool region) {
    std::string source;
    llvm::raw_string_ostream os(source);
    auto table = "tensor<2048x64x" + dtype.str() + ">";
    auto indices = "tensor<2x" + std::to_string(length) + "x1xi64>";
    auto output = "tensor<2x" + std::to_string(length) + "x64x" + dtype.str() + ">";
    os << "module {\n";
    if (region)
      os << "wafer.tile.module card_id = 0 tile_id = 0 {\n";
    os << "func.func @main(%table: " << table << ", %ids: " << indices
       << ") -> " << output << " {\n";
    if (region)
      os << "%result = wafer.tile.region(%table, %ids : " << table << ", "
         << indices << ") -> (" << output << ") {\n"
         << "^bb0(%local: " << table << ", %index: " << indices << "):\n";
    os << "%g = tensor.gather " << (region ? "%local[%index]" : "%table[%ids]")
       << " gather_dims([0]) : (" << table << ", " << indices << ") -> "
       << output << "\n";
    if (region)
      os << "wafer.tile.yield %g : " << output << "\n}\n";
    os << "return " << (region ? "%result" : "%g") << " : " << output
       << "\n}\n}";
    if (region)
      os << "\n}";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(GatherLoweringTest, OutputTilesPreserveIndexedAxisAndCoverEveryRow) {
  for (int64_t length : {1024, 1025, 1031})
    for (unsigned tiles : {4, 16})
      for (llvm::StringRef dtype : {"f16", "bf16"}) {
        SCOPED_TRACE(length);
        SCOPED_TRACE(tiles);
        SCOPED_TRACE(dtype.str());
        auto module = parse(length, dtype, false);
        ASSERT_TRUE(module);
        mlir::tensor::GatherOp gather;
        module->walk([&](mlir::tensor::GatherOp op) { gather = op; });
        auto tiling = mlir::cast<mlir::TilingInterface>(gather.getOperation());
        mlir::OpBuilder builder(gather);
        std::vector<unsigned> coverage(length * 64, 0);
        for (unsigned tile = 0; tile < tiles; ++tile) {
          int64_t begin = length * tile / tiles;
          int64_t end = length * (tile + 1) / tiles;
          for (int64_t row = begin; row < end; row += 64)
            for (int64_t column : {0, 48}) {
              int64_t rows = std::min<int64_t>(64, end - row);
              int64_t columns = std::min<int64_t>(48, 64 - column);
              llvm::SmallVector<mlir::OpFoldResult> offsets{
                  builder.getIndexAttr(0), builder.getIndexAttr(row),
                  builder.getIndexAttr(column)};
              llvm::SmallVector<mlir::OpFoldResult> sizes{
                  builder.getIndexAttr(2), builder.getIndexAttr(rows),
                  builder.getIndexAttr(columns)};
              auto result = tiling.generateResultTileValue(builder, 0, offsets, sizes);
              ASSERT_TRUE(mlir::succeeded(result));
              auto tiled = mlir::cast<mlir::tensor::GatherOp>(result->tiledOps.front());
              EXPECT_EQ(tiled.getResultType().getShape(),
                        llvm::ArrayRef<int64_t>({2, rows, columns}));
              EXPECT_EQ(tiled.getSourceType().getShape(),
                        llvm::ArrayRef<int64_t>({2048, columns}));
              EXPECT_EQ(tiled.getIndicesType().getShape(),
                        llvm::ArrayRef<int64_t>({2, rows, 1}));
              EXPECT_FALSE(tiled.getUnique());
              auto tableSlice = tiled.getSource().getDefiningOp<mlir::tensor::ExtractSliceOp>();
              EXPECT_EQ(mlir::getConstantIntValue(tableSlice.getMixedOffsets()[0]), 0);
              EXPECT_EQ(mlir::getConstantIntValue(tableSlice.getMixedOffsets()[1]), column);
              auto idSlice = tiled.getIndices().getDefiningOp<mlir::tensor::ExtractSliceOp>();
              EXPECT_EQ(mlir::getConstantIntValue(idSlice.getMixedOffsets()[1]), row);
              for (int64_t r = row; r < row + rows; ++r)
                for (int64_t c = column; c < column + columns; ++c)
                  ++coverage[r * 64 + c];
            }
        }
        EXPECT_TRUE(llvm::all_of(coverage, [](unsigned n) { return n == 1; }));
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
      }
}

TEST_F(GatherLoweringTest, SelectedRowsBufferizeIntoOneOutputAllocation) {
  for (int64_t length : {1024, 1025, 1031})
    for (llvm::StringRef dtype : {"f16", "bf16"}) {
      SCOPED_TRACE(length);
      SCOPED_TRACE(dtype.str());
      auto module = parse(length, dtype, true);
      ASSERT_TRUE(module);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      TileRegionOp region;
      module->walk([&](TileRegionOp op) { region = op; });
      StructuredMaterializationRelations relations;
      relations.structuralOutputs.push_back({0, region.getResult(0)});
      auto result = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(result.succeeded()) << result.detail;
      unsigned outputs = 0, scalarReads = 0, copies = 0;
      module->walk([&](mlir::memref::AllocOp alloc) {
        EXPECT_FALSE(alloc->getParentOfType<mlir::scf::ForOp>());
        if (alloc.getType().getShape() == llvm::ArrayRef<int64_t>({2, length, 64}))
          ++outputs;
      });
      module->walk([&](mlir::memref::LoadOp) { ++scalarReads; });
      module->walk([&](mlir::memref::CopyOp) { ++copies; });
      EXPECT_EQ(outputs, 1u);
      EXPECT_EQ(scalarReads, 1u);
      EXPECT_GE(copies, 1u);
      EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(*module, relations)));
    }
}

TEST_F(GatherLoweringTest, TemporalBlocksAndTailsKeepStaticGatherOperands) {
  for (int64_t length : {1024, 1025, 1031})
    for (llvm::StringRef dtype : {"f16", "bf16"}) {
      SCOPED_TRACE(length);
      SCOPED_TRACE(dtype.str());
      auto module = parse(length, dtype, true);
      ASSERT_TRUE(module);
      TileRegionOp region;
      module->walk([&](TileRegionOp op) { region = op; });
      auto domain = buildTemporalDomain(region);
      ASSERT_TRUE(domain.succeeded()) << domain.failure->detail;
      auto first = domain.domain->getFirstChoice();
      ASSERT_NE(first.getChoice(), nullptr);
      TemporalChoice choice = *first.getChoice();
      ASSERT_EQ(choice.scopes.size(), 1u);
      choice.scopes.front().iteratorTileSizes = {2, 64, 64};
      auto descriptors = domain.domain->getScopeDescriptors();
      auto order = buildFirstTemporalLoopOrder(
          descriptors.front().iterationExtents,
          choice.scopes.front().iteratorTileSizes,
          descriptors.front().precedence);
      ASSERT_TRUE(mlir::succeeded(order));
      choice.scopes.front().loopOrder = *order;
      ASSERT_TRUE(domain.domain->contains(choice));
      StructuredMaterializationRelations relations;
      relations.structuralOutputs.push_back({0, region.getResult(0)});
      TemporalTilingFailure failure;
      auto tiled = applyTemporalTiling({{*domain.domain, choice}}, relations, &failure);
      ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
      unsigned fullBlocks = 0, tails = 0;
      module->walk([&](mlir::tensor::GatherOp gather) {
        EXPECT_TRUE(gather.getIndicesType().hasStaticShape());
        EXPECT_TRUE(gather.getSourceType().hasStaticShape());
        EXPECT_TRUE(gather.getResultType().hasStaticShape());
        auto shape = gather.getResultType().getShape();
        ASSERT_EQ(shape.size(), 3u);
        EXPECT_EQ(shape[0], 2);
        EXPECT_EQ(shape[2], 64);
        if (shape[1] == 64) {
          ++fullBlocks;
          EXPECT_TRUE(gather->getParentOfType<mlir::scf::ForOp>());
        } else {
          ++tails;
          EXPECT_EQ(shape[1], length % 64);
        }
      });
      EXPECT_EQ(fullBlocks, 1u);
      EXPECT_EQ(tails, length % 64 ? 1u : 0u);
      auto lowered = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
}
} // namespace
