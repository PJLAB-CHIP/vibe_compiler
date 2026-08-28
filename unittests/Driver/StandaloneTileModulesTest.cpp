//===- StandaloneTileModulesTest.cpp - Standalone module tests -----------===//

#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace {

static_assert(!std::is_convertible_v<int64_t, wafer::CardId>,
              "card IDs must not accept implicit integer conversion");
static_assert(!std::is_convertible_v<int64_t, wafer::TileId>,
              "Tile IDs must not accept implicit integer conversion");
static_assert(!std::is_same_v<wafer::CardId, wafer::TileId>,
              "card and Tile identities must remain distinct types");

static std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  wafer::WaferDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

template <typename OpT> static unsigned countOps(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](OpT) { ++count; });
  return count;
}

static llvm::SmallVector<int64_t, 4>
collectIntegerConstants(mlir::ModuleOp module) {
  llvm::SmallVector<int64_t, 4> values;
  module.walk([&](mlir::arith::ConstantOp constant) {
    if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      values.push_back(value.getInt());
  });
  return values;
}

TEST(StandaloneTileModulesTest,
     MovesStableTypedTileBodiesIntoStandaloneModules) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {test.module_attribute = "preserved"} {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical
      {axes = ["card"], shape = array<i64: 1>}
  memref.global "private" @shared
      : memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>
  wafer.tile.module card_id = 0 tile_id = 1 {
    %value = arith.constant 20 : i32
  }
  wafer.tile.module card_id = 0 tile_id = 0 {
    %value = arith.constant 10 : i32
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
  mlir::Operation *constantTen = nullptr;
  mlir::Operation *constantTwenty = nullptr;
  source->walk([&](mlir::arith::ConstantOp constant) {
    auto value = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (value && value.getInt() == 10)
      constantTen = constant.getOperation();
    if (value && value.getInt() == 20)
      constantTwenty = constant.getOperation();
  });
  ASSERT_NE(constantTen, nullptr);
  ASSERT_NE(constantTwenty, nullptr);
  wafer::StructuredMaterializationRelations relations;
  relations.operationEmissions = {{7, constantTen}, {9, constantTwenty}};

  std::string failureReason;
  auto standaloneModules = wafer::createStandaloneTileModules(
      std::move(source), &failureReason, &relations);

  ASSERT_TRUE(mlir::succeeded(standaloneModules)) << failureReason;
  ASSERT_EQ(standaloneModules->size(), 2u);
  EXPECT_EQ((*standaloneModules)[0].cardId.getValue(), 0);
  EXPECT_EQ((*standaloneModules)[0].tileId.getValue(), 0);
  EXPECT_EQ((*standaloneModules)[1].cardId.getValue(), 0);
  EXPECT_EQ((*standaloneModules)[1].tileId.getValue(), 1);
  EXPECT_TRUE((*standaloneModules)[0].cardId == wafer::CardId(0));
  EXPECT_TRUE((*standaloneModules)[0].tileId != wafer::TileId(1));

  for (auto &tile : *standaloneModules) {
    ASSERT_TRUE(tile.module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*tile.module)));
    EXPECT_TRUE(tile.module->getOperation()->hasAttr("test.module_attribute"));
    EXPECT_EQ(countOps<wafer::TargetTopologyOp>(*tile.module), 1u);
    EXPECT_EQ(countOps<wafer::ExecutionMeshOp>(*tile.module), 1u);
    EXPECT_EQ(countOps<mlir::memref::GlobalOp>(*tile.module), 1u);
    EXPECT_EQ(countOps<wafer::TileModuleOp>(*tile.module), 0u);
  }

  llvm::SmallVector<int64_t, 4> tileZeroConstants =
      collectIntegerConstants(*(*standaloneModules)[0].module);
  llvm::SmallVector<int64_t, 4> tileOneConstants =
      collectIntegerConstants(*(*standaloneModules)[1].module);
  ASSERT_EQ(tileZeroConstants.size(), 1u);
  ASSERT_EQ(tileOneConstants.size(), 1u);
  EXPECT_EQ(tileZeroConstants.front(), 10);
  EXPECT_EQ(tileOneConstants.front(), 20);
  EXPECT_TRUE((*standaloneModules)[0].module->getOperation()->isProperAncestor(
      constantTen));
  EXPECT_TRUE((*standaloneModules)[1].module->getOperation()->isProperAncestor(
      constantTwenty));
  ASSERT_EQ((*standaloneModules)[0]
                .materializationRelations.operationEmissions.size(),
            1u);
  ASSERT_EQ((*standaloneModules)[1]
                .materializationRelations.operationEmissions.size(),
            1u);
  EXPECT_EQ((*standaloneModules)[0]
                .materializationRelations.operationEmissions.front()
                .structuredNodeId,
            7u);
  EXPECT_EQ((*standaloneModules)[1]
                .materializationRelations.operationEmissions.front()
                .structuredNodeId,
            9u);
}

TEST(StandaloneTileModulesTest, RequiresAtLeastOneTileModule) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  std::string failureReason;
  auto standaloneModules =
      wafer::createStandaloneTileModules(std::move(source), &failureReason);

  EXPECT_TRUE(mlir::failed(standaloneModules));
  EXPECT_EQ(failureReason,
            "source module contains no top-level wafer.tile.module");
}

TEST(StandaloneTileModulesTest, RejectsDuplicatePhysicalTileIdentity) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}
  wafer.tile.module card_id = 0 tile_id = 0 {}
  wafer.tile.module card_id = 0 tile_id = 0 {}
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  std::string failureReason;
  auto standaloneModules =
      wafer::createStandaloneTileModules(std::move(source), &failureReason);

  EXPECT_TRUE(mlir::failed(standaloneModules));
  EXPECT_EQ(failureReason, "source Tile module set is invalid");
}

TEST(StandaloneTileModulesTest,
     CreatesStandaloneModuleForVerifierValidPartialTileDomain) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}
  memref.global "private" @ragged_shared
      : memref<2x1025x64xf16, #wafer.memory<ddr, tensor>>
  wafer.tile.module card_id = 0 tile_id = 0 {}
  wafer.tile.module card_id = 0 tile_id = 1 {}
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  wafer::TileModuleOp tileToErase;
  for (wafer::TileModuleOp tile : source->getOps<wafer::TileModuleOp>()) {
    if (tile.getTileIdAttr().getInt() == 1) {
      tileToErase = tile;
      break;
    }
  }
  ASSERT_TRUE(tileToErase);
  tileToErase.erase();
  std::string failureReason;
  auto standaloneModules =
      wafer::createStandaloneTileModules(std::move(source), &failureReason);

  ASSERT_TRUE(mlir::succeeded(standaloneModules)) << failureReason;
  ASSERT_EQ(standaloneModules->size(), 1u);
  EXPECT_EQ(standaloneModules->front().tileId, wafer::TileId(0));
  EXPECT_EQ(
      countOps<mlir::memref::GlobalOp>(*standaloneModules->front().module), 1u);
}

TEST(StandaloneTileModulesTest, RejectsExecutableOperationOutsideTileModule) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  %value = arith.constant 1 : i32
  wafer.tile.module card_id = 0 tile_id = 0 {}
}
)mlir",
      mlir::ParserConfig(context.get(), /*verifyAfterParse=*/false));
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  std::string failureReason;
  auto standaloneModules =
      wafer::createStandaloneTileModules(std::move(source), &failureReason);

  EXPECT_TRUE(mlir::failed(standaloneModules));
  EXPECT_EQ(failureReason,
            "source module contains a non-declaration operation outside a "
            "wafer.tile.module");
}

TEST(StandaloneTileModulesTest, RejectsImplicitCrossTileSSA) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}
  %implicitly_shared = arith.constant 1 : i32
  wafer.tile.module card_id = 0 tile_id = 0 {
    %value = arith.addi %implicitly_shared, %implicitly_shared : i32
  }
  wafer.tile.module card_id = 0 tile_id = 1 {
    %value = arith.addi %implicitly_shared, %implicitly_shared : i32
  }
}
)mlir",
      mlir::ParserConfig(context.get(), /*verifyAfterParse=*/false));
  ASSERT_TRUE(source);
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  EXPECT_TRUE(mlir::failed(mlir::verify(*source)));

  std::string failureReason;
  auto standaloneModules =
      wafer::createStandaloneTileModules(std::move(source), &failureReason);

  EXPECT_TRUE(mlir::failed(standaloneModules));
  EXPECT_EQ(failureReason, "source module is not verifier-legal");
}

} // namespace
