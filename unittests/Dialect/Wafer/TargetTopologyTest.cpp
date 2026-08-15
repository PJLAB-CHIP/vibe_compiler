//===- TargetTopologyTest.cpp - Physical topology view tests -----------===//

#include "Wafer/IR/Target/TargetTopology.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

class TargetTopologyTest : public ::testing::Test {
protected:
  TargetTopologyTest() {
    registry.insert<wafer::WaferDialect>();
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  static llvm::SmallVector<int64_t, 16>
  getValues(llvm::ArrayRef<wafer::TileId> ids) {
    llvm::SmallVector<int64_t, 16> values;
    values.reserve(ids.size());
    for (wafer::TileId id : ids)
      values.push_back(id.getValue());
    return values;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(TargetTopologyTest, DerivesSingleCardFourByFourWithoutExecutionMesh) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
}
)mlir");
  ASSERT_TRUE(module);
  EXPECT_TRUE(module->getOps<wafer::ExecutionMeshOp>().empty());

  auto topology = wafer::TargetTopology::create(*module);
  ASSERT_TRUE(mlir::succeeded(topology));
  EXPECT_EQ(topology->getCardCount(), 1);
  EXPECT_EQ(topology->getTilesPerCard(), 16);

  auto cardCoordinate = topology->getCardCoordinate(wafer::CardId(0));
  auto tileCoordinate = topology->getTileCoordinate(wafer::TileId(15));
  ASSERT_TRUE(cardCoordinate);
  ASSERT_TRUE(tileCoordinate);
  EXPECT_EQ(*cardCoordinate, (wafer::CardCoordinate{0, 0}));
  EXPECT_EQ(*tileCoordinate, (wafer::TileCoordinate{3, 3}));
  EXPECT_EQ(topology->getCardId({0, 0})->getValue(), 0);
  EXPECT_EQ(topology->getTileId({3, 3})->getValue(), 15);

  auto available = topology->getAvailableTileIds(wafer::CardId(0));
  ASSERT_TRUE(available);
  ASSERT_EQ(available->size(), 16u);
  EXPECT_EQ(available->front().getValue(), 0);
  EXPECT_EQ(available->back().getValue(), 15);

  auto neighbors = topology->getOnCardNeighbors(wafer::CardId(0),
                                                wafer::TileId(5));
  ASSERT_TRUE(mlir::succeeded(neighbors));
  EXPECT_EQ(getValues(*neighbors),
            (llvm::SmallVector<int64_t, 16>{1, 4, 6, 9}));
  EXPECT_TRUE(topology->areOnCardAdjacent(wafer::CardId(0),
                                          wafer::TileId(5),
                                          wafer::TileId(6)));
  EXPECT_EQ(topology->getOnCardShortestHopDistance(wafer::CardId(0),
                                                   wafer::TileId(0),
                                                   wafer::TileId(15)),
            6u);

  auto path = topology->getCanonicalOnCardPath(wafer::CardId(0),
                                               wafer::TileId(0),
                                               wafer::TileId(15));
  ASSERT_TRUE(mlir::succeeded(path));
  ASSERT_EQ(path->size(), 6u);
  EXPECT_EQ(path->front().source.getValue(), 0);
  EXPECT_EQ(path->front().destination.getValue(), 1);
  EXPECT_EQ(path->back().source.getValue(), 11);
  EXPECT_EQ(path->back().destination.getValue(), 15);
}

TEST_F(TargetTopologyTest, UnavailableTileKeepsStableRowMajorId) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>,
       unavailable_tiles = array<i64: 0, 0, 0, 1>}
}
)mlir");
  ASSERT_TRUE(module);
  auto topology = wafer::TargetTopology::create(*module);
  ASSERT_TRUE(mlir::succeeded(topology));

  auto unavailableCoordinate =
      topology->getTileCoordinate(wafer::TileId(1));
  auto stableCoordinate = topology->getTileCoordinate(wafer::TileId(2));
  ASSERT_TRUE(unavailableCoordinate);
  ASSERT_TRUE(stableCoordinate);
  EXPECT_EQ(*unavailableCoordinate, (wafer::TileCoordinate{0, 1}));
  EXPECT_EQ(*stableCoordinate, (wafer::TileCoordinate{1, 0}));
  EXPECT_FALSE(topology->isTileAvailable(wafer::CardId(0),
                                         wafer::TileId(1)));

  auto available = topology->getAvailableTileIds(wafer::CardId(0));
  ASSERT_TRUE(available);
  EXPECT_EQ(getValues(*available), (llvm::SmallVector<int64_t, 16>{0, 2, 3}));
  EXPECT_EQ(topology->getOnCardShortestHopDistance(wafer::CardId(0),
                                                   wafer::TileId(0),
                                                   wafer::TileId(3)),
            2u);
  EXPECT_TRUE(mlir::failed(topology->getOnCardNeighbors(
      wafer::CardId(0), wafer::TileId(1))));
}

TEST_F(TargetTopologyTest, AvailabilityIsScopedPerCard) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 2>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>,
       unavailable_tiles = array<i64: 0, 0, 0, 1,
                                      0, 1, 1, 0>}
}
)mlir");
  ASSERT_TRUE(module);
  auto topology = wafer::TargetTopology::create(*module);
  ASSERT_TRUE(mlir::succeeded(topology));
  EXPECT_EQ(topology->getCardCount(), 2);
  EXPECT_EQ(topology->getCardCoordinate(wafer::CardId(1)),
            (wafer::CardCoordinate{0, 1}));

  auto cardZero = topology->getAvailableTileIds(wafer::CardId(0));
  auto cardOne = topology->getAvailableTileIds(wafer::CardId(1));
  ASSERT_TRUE(cardZero);
  ASSERT_TRUE(cardOne);
  EXPECT_EQ(getValues(*cardZero), (llvm::SmallVector<int64_t, 16>{0, 2, 3}));
  EXPECT_EQ(getValues(*cardOne), (llvm::SmallVector<int64_t, 16>{0, 1, 3}));
  EXPECT_FALSE(topology->isTileAvailable(wafer::CardId(0),
                                         wafer::TileId(1)));
  EXPECT_TRUE(topology->isTileAvailable(wafer::CardId(1),
                                        wafer::TileId(1)));
}

TEST_F(TargetTopologyTest, DoesNotUseExecutionMeshSelection) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {axes = ["card_partition"], shape = array<i64: 1>}
}
)mlir");
  ASSERT_TRUE(module);
  auto topology = wafer::TargetTopology::create(*module);
  ASSERT_TRUE(mlir::succeeded(topology));
  auto available = topology->getAvailableTileIds(wafer::CardId(0));
  ASSERT_TRUE(available);
  EXPECT_EQ(getValues(*available),
            (llvm::SmallVector<int64_t, 16>{0, 1, 2, 3}));
}

TEST_F(TargetTopologyTest, RejectsDuplicateDirectTopology) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @first
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.target.topology @second
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  EXPECT_TRUE(
      mlir::failed(wafer::TargetTopology::create(*module, &failureReason)));
  EXPECT_EQ(failureReason,
            "expected exactly one direct wafer.target.topology in source "
            "module");
}

TEST_F(TargetTopologyTest, RejectsNestedTopology) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @direct
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  module @nested {
    wafer.target.topology @nested_target
        {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
         tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  EXPECT_TRUE(
      mlir::failed(wafer::TargetTopology::create(*module, &failureReason)));
  EXPECT_EQ(failureReason,
            "wafer.target.topology must be directly nested under the source "
            "module");
}

TEST_F(TargetTopologyTest, RejectsMissingTopology) {
  auto module = parse("module {}");
  ASSERT_TRUE(module);
  std::string failureReason;
  EXPECT_TRUE(
      mlir::failed(wafer::TargetTopology::create(*module, &failureReason)));
  EXPECT_EQ(failureReason,
            "expected exactly one direct wafer.target.topology in source "
            "module");
}

} // namespace
