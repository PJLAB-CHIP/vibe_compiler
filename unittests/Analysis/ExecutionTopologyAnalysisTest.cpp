//===- ExecutionTopologyAnalysisTest.cpp --------------------------------===//

#include "Wafer/Analysis/ExecutionTopologyAnalysis.h"
#include "Wafer/InitAll.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

class ExecutionTopologyAnalysisTest : public ::testing::Test {
protected:
  ExecutionTopologyAnalysisTest() {
    wafer::registerAllDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(ExecutionTopologyAnalysisTest,
       MapsAllAvailableRanksInLinearEndpointOrder) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @topology {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @mesh {
    topology = @topology, axes = ["rank"], shape = array<i64: 4>,
    policy = "all_available", endpoints = array<i64>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto topology = wafer::analysis::ExecutionTopologyAnalysis::create(*module);
  ASSERT_TRUE(mlir::succeeded(topology));
  ASSERT_EQ(topology->getRankCount(), 4);

  auto rank0 = topology->getRankEndpoint(0);
  auto rank3 = topology->getRankEndpoint(3);
  ASSERT_TRUE(rank0);
  ASSERT_TRUE(rank3);
  EXPECT_EQ(rank0->cardY, 0);
  EXPECT_EQ(rank0->cardX, 0);
  EXPECT_EQ(rank0->tileY, 0);
  EXPECT_EQ(rank0->tileX, 0);
  EXPECT_EQ(rank3->tileY, 1);
  EXPECT_EQ(rank3->tileX, 1);
  EXPECT_EQ(topology->getShortestHopDistance(0, 3), 2u);
  // Four undirected edges, counted once in each direction.
  EXPECT_EQ(topology->getDirectedLinkCount(), 8u);
}

TEST_F(ExecutionTopologyAnalysisTest,
       PreservesExplicitRankOrderAndRoutesAroundUnavailableEndpoints) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @topology {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 3, 3>,
    unavailable_tiles = array<i64: 0, 0, 1, 1>
  }
  wafer.execution.mesh @mesh {
    topology = @topology, axes = ["rank"], shape = array<i64: 2>,
    policy = "explicit",
    endpoints = array<i64: 0, 0, 1, 2, 0, 0, 1, 0>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto topology = wafer::analysis::ExecutionTopologyAnalysis::create(*module);
  ASSERT_TRUE(mlir::succeeded(topology));

  auto rank0 = topology->getRankEndpoint(0);
  auto rank1 = topology->getRankEndpoint(1);
  ASSERT_TRUE(rank0);
  ASSERT_TRUE(rank1);
  EXPECT_EQ(rank0->tileX, 2);
  EXPECT_EQ(rank1->tileX, 0);
  EXPECT_EQ(topology->getShortestHopDistance(0, 1), 4u);
  // A full 3x3 mesh has 24 directed links; removing the center removes its
  // eight directed incidences.
  EXPECT_EQ(topology->getDirectedLinkCount(), 16u);

  auto matrix = topology->getShortestHopMatrix({1, 0});
  ASSERT_TRUE(mlir::succeeded(matrix));
  EXPECT_EQ(*matrix, (llvm::SmallVector<uint64_t, 16>{0, 4, 4, 0}));

  auto route = topology->getCanonicalShortestPath(0, 1);
  ASSERT_TRUE(mlir::succeeded(route));
  ASSERT_EQ(route->size(), 4u);
  // Row-major BFS tie-breaking chooses the upper path around the unavailable
  // center rather than depending on incidental neighbor insertion order.
  EXPECT_EQ((*route)[0].source,
            (wafer::analysis::ExecutionEndpoint{0, 0, 1, 2}));
  EXPECT_EQ((*route)[0].destination,
            (wafer::analysis::ExecutionEndpoint{0, 0, 0, 2}));
  EXPECT_EQ((*route)[3].source,
            (wafer::analysis::ExecutionEndpoint{0, 0, 0, 0}));
  EXPECT_EQ((*route)[3].destination,
            (wafer::analysis::ExecutionEndpoint{0, 0, 1, 0}));
}

TEST_F(ExecutionTopologyAnalysisTest, CountsDirectedLinksInFourByFourMesh) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @topology {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @mesh {
    topology = @topology, axes = ["rank"], shape = array<i64: 16>,
    policy = "all_available", endpoints = array<i64>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto topology = wafer::analysis::ExecutionTopologyAnalysis::create(*module);
  ASSERT_TRUE(mlir::succeeded(topology));
  // 4 * 3 horizontal plus 4 * 3 vertical links, each bidirectional.
  EXPECT_EQ(topology->getDirectedLinkCount(), 48u);
}

TEST_F(ExecutionTopologyAnalysisTest, DerivesCardTorusWraparoundFromTypedIR) {
  auto meshModule = parse(R"mlir(
module {
  wafer.target.topology @topology {
    card_grid = array<i64: 1, 4>, card_interconnect = "mesh",
    tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @mesh {
    topology = @topology, axes = ["rank"], shape = array<i64: 2>,
    policy = "explicit",
    endpoints = array<i64: 0, 0, 0, 0, 0, 3, 0, 0>
  }
}
)mlir");
  auto torusModule = parse(R"mlir(
module {
  wafer.target.topology @topology {
    card_grid = array<i64: 1, 4>, card_interconnect = "torus",
    tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @mesh {
    topology = @topology, axes = ["rank"], shape = array<i64: 2>,
    policy = "explicit",
    endpoints = array<i64: 0, 0, 0, 0, 0, 3, 0, 0>
  }
}
)mlir");
  ASSERT_TRUE(meshModule);
  ASSERT_TRUE(torusModule);
  auto mesh = wafer::analysis::ExecutionTopologyAnalysis::create(*meshModule);
  auto torus = wafer::analysis::ExecutionTopologyAnalysis::create(*torusModule);
  ASSERT_TRUE(mlir::succeeded(mesh));
  ASSERT_TRUE(mlir::succeeded(torus));
  EXPECT_EQ(mesh->getShortestHopDistance(0, 1), 3u);
  EXPECT_EQ(torus->getShortestHopDistance(0, 1), 1u);
  auto torusRoute = torus->getCanonicalShortestPath(0, 1);
  ASSERT_TRUE(mlir::succeeded(torusRoute));
  ASSERT_EQ(torusRoute->size(), 1u);
  EXPECT_EQ(torusRoute->front().source,
            (wafer::analysis::ExecutionEndpoint{0, 0, 0, 0}));
  EXPECT_EQ(torusRoute->front().destination,
            (wafer::analysis::ExecutionEndpoint{0, 3, 0, 0}));
  EXPECT_TRUE(mlir::failed(torus->getCanonicalShortestPath(0, 2)));
}

TEST_F(ExecutionTopologyAnalysisTest,
       EquivalenceIncludesUnavailableEndpointFacts) {
  auto availableModule = parse(R"mlir(
module {
  wafer.target.topology @topology {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @mesh {
    topology = @topology, axes = ["rank"], shape = array<i64: 2>,
    policy = "explicit",
    endpoints = array<i64: 0, 0, 0, 0, 0, 0, 0, 1>
  }
}
)mlir");
  auto unavailableModule = parse(R"mlir(
module {
  wafer.target.topology @topology {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 2, 2>,
    unavailable_tiles = array<i64: 0, 0, 1, 1>
  }
  wafer.execution.mesh @mesh {
    topology = @topology, axes = ["rank"], shape = array<i64: 2>,
    policy = "explicit",
    endpoints = array<i64: 0, 0, 0, 0, 0, 0, 0, 1>
  }
}
)mlir");
  ASSERT_TRUE(availableModule);
  ASSERT_TRUE(unavailableModule);
  auto available =
      wafer::analysis::ExecutionTopologyAnalysis::create(*availableModule);
  auto unavailable =
      wafer::analysis::ExecutionTopologyAnalysis::create(*unavailableModule);
  ASSERT_TRUE(mlir::succeeded(available));
  ASSERT_TRUE(mlir::succeeded(unavailable));
  auto availableDistances = available->getShortestHopMatrix({0, 1});
  auto unavailableDistances = unavailable->getShortestHopMatrix({0, 1});
  ASSERT_TRUE(mlir::succeeded(availableDistances));
  ASSERT_TRUE(mlir::succeeded(unavailableDistances));
  EXPECT_EQ(*availableDistances, *unavailableDistances);
  EXPECT_FALSE(available->isEquivalentTo(*unavailable));
}

TEST_F(ExecutionTopologyAnalysisTest, RejectsMissingTypedTopologyFacts) {
  auto module = parse("module {}");
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::failed(
      wafer::analysis::ExecutionTopologyAnalysis::create(*module)));
}

} // namespace
