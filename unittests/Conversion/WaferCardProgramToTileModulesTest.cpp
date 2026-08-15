//===- WaferCardProgramToTileModulesTest.cpp - Module splitting tests
//----------===//

#include "Wafer/Conversion/WaferCardProgramToTileModules/WaferCardProgramToTileModules.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace {

static_assert(!std::is_convertible_v<int64_t, wafer::PhysicalCardId>,
              "physical card IDs must not accept implicit integer conversion");
static_assert(!std::is_convertible_v<int64_t, wafer::PhysicalTileId>,
              "physical Tile IDs must not accept implicit integer conversion");
static_assert(!std::is_same_v<wafer::PhysicalCardId, wafer::PhysicalTileId>,
              "physical card and Tile identities must remain distinct types");

static std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  wafer::WaferDialect>();
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

TEST(WaferCardProgramToTileModulesTest,
     SplitsStableTypedTileModulesWithoutMutatingSource) {
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
  wafer.card.program card_id = 0 {
    memref.global "private" @shared
        : memref<1xi32, #wafer.memory<ddr, tensor>>
    wafer.tile.program tile_id = 1 {
      %value = arith.constant 20 : i32
    }
    wafer.tile.program tile_id = 0 {
      %value = arith.constant 10 : i32
    }
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
  const std::string sourceBefore = printModule(*source);

  std::string failureReason;
  auto tileModules =
      wafer::splitCardProgramIntoPhysicalTileModules(*source, &failureReason);

  ASSERT_TRUE(mlir::succeeded(tileModules)) << failureReason;
  ASSERT_EQ(tileModules->size(), 2u);
  EXPECT_EQ((*tileModules)[0].cardId.getValue(), 0);
  EXPECT_EQ((*tileModules)[0].tileId.getValue(), 0);
  EXPECT_EQ((*tileModules)[1].cardId.getValue(), 0);
  EXPECT_EQ((*tileModules)[1].tileId.getValue(), 1);
  EXPECT_TRUE((*tileModules)[0].cardId == wafer::PhysicalCardId(0));
  EXPECT_TRUE((*tileModules)[0].tileId != wafer::PhysicalTileId(1));
  EXPECT_EQ(printModule(*source), sourceBefore);

  for (auto &tile : *tileModules) {
    ASSERT_TRUE(tile.module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*tile.module)));
    EXPECT_TRUE(tile.module->getOperation()->hasAttr("test.module_attribute"));
    EXPECT_EQ(countOps<wafer::TargetTopologyOp>(*tile.module), 1u);
    EXPECT_EQ(countOps<wafer::ExecutionMeshOp>(*tile.module), 1u);
    EXPECT_EQ(countOps<mlir::memref::GlobalOp>(*tile.module), 1u);
    EXPECT_EQ(countOps<wafer::CardProgramOp>(*tile.module), 0u);
    EXPECT_EQ(countOps<wafer::TileProgramOp>(*tile.module), 0u);
  }

  llvm::SmallVector<int64_t, 4> tileZeroConstants =
      collectIntegerConstants(*(*tileModules)[0].module);
  llvm::SmallVector<int64_t, 4> tileOneConstants =
      collectIntegerConstants(*(*tileModules)[1].module);
  ASSERT_EQ(tileZeroConstants.size(), 1u);
  ASSERT_EQ(tileOneConstants.size(), 1u);
  EXPECT_EQ(tileZeroConstants.front(), 10);
  EXPECT_EQ(tileOneConstants.front(), 20);
}

TEST(WaferCardProgramToTileModulesTest, RequiresExactlyOneCardProgram) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 2>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  wafer.card.program card_id = 0 {
    wafer.tile.program tile_id = 0 {}
  }
  wafer.card.program card_id = 1 {
    wafer.tile.program tile_id = 0 {}
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  std::string failureReason;
  auto tileModules =
      wafer::splitCardProgramIntoPhysicalTileModules(*source, &failureReason);

  EXPECT_TRUE(mlir::failed(tileModules));
  EXPECT_EQ(failureReason,
            "expected exactly one direct wafer.card.program in source module");
}

TEST(WaferCardProgramToTileModulesTest, RejectsUnverifiedSource) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}
  wafer.card.program card_id = 0 {
    wafer.tile.program tile_id = 0 {}
    wafer.tile.program tile_id = 1 {}
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  wafer::CardProgramOp card = *source->getOps<wafer::CardProgramOp>().begin();
  wafer::TileProgramOp tileToErase;
  for (wafer::TileProgramOp tile :
       card.getBody().front().getOps<wafer::TileProgramOp>()) {
    if (tile.getTileIdAttr().getInt() == 1) {
      tileToErase = tile;
      break;
    }
  }
  ASSERT_TRUE(tileToErase);
  tileToErase.erase();
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  std::string failureReason;
  auto tileModules =
      wafer::splitCardProgramIntoPhysicalTileModules(*source, &failureReason);

  EXPECT_TRUE(mlir::failed(tileModules));
  EXPECT_EQ(failureReason, "source module is not verifier-legal");
}

TEST(WaferCardProgramToTileModulesTest,
     RejectsExecutableOperationOutsideTileProgram) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  wafer.card.program card_id = 0 {
    %value = arith.constant 1 : i32
    wafer.tile.program tile_id = 0 {}
  }
}
)mlir",
      mlir::ParserConfig(context.get(), /*verifyAfterParse=*/false));
  ASSERT_TRUE(source);
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  ASSERT_TRUE(mlir::failed(mlir::verify(*source)));

  std::string failureReason;
  auto tileModules =
      wafer::splitCardProgramIntoPhysicalTileModules(*source, &failureReason);

  EXPECT_TRUE(mlir::failed(tileModules));
  EXPECT_EQ(failureReason, "source module is not verifier-legal");
}

TEST(WaferCardProgramToTileModulesTest, RejectsImplicitCrossTileSSA) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}
  wafer.card.program card_id = 0 {
    %implicitly_shared = arith.constant 1 : i32
    wafer.tile.program tile_id = 0 {
      %value = arith.addi %implicitly_shared, %implicitly_shared : i32
    }
    wafer.tile.program tile_id = 1 {
      %value = arith.addi %implicitly_shared, %implicitly_shared : i32
    }
  }
}
)mlir",
      mlir::ParserConfig(context.get(), /*verifyAfterParse=*/false));
  ASSERT_TRUE(source);
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  EXPECT_TRUE(mlir::failed(mlir::verify(*source)));

  std::string failureReason;
  auto tileModules =
      wafer::splitCardProgramIntoPhysicalTileModules(*source, &failureReason);

  EXPECT_TRUE(mlir::failed(tileModules));
  EXPECT_EQ(failureReason, "source module is not verifier-legal");
}

} // namespace
