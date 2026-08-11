//===- WholeDAGPlacementFrontierTest.cpp -------------------------------===//

#include "../../lib/Wafer/Compiler/WholeDAGPlacementFrontier.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>
#include <string>

using namespace wafer;
using namespace wafer::compiler::detail;

namespace {

std::string makeIndependentPointwiseBranches(size_t branchCount) {
  std::string function = "\n  func.func @branches(%input: tensor<1xf16>) -> (";
  for (size_t index = 0; index < branchCount; ++index) {
    if (index != 0)
      function += ", ";
    function += "tensor<1xf16>";
  }
  function += ") {\n";
  for (size_t index = 0; index < branchCount; ++index) {
    const std::string suffix = std::to_string(index);
    function += "    %out" + suffix + " = tensor.empty() : tensor<1xf16>\n";
    function += "    %value" + suffix +
                " = linalg.map ins(%input : tensor<1xf16>) "
                "outs(%out" +
                suffix + " : tensor<1xf16>) (%element: f16) {\n";
    function += "      linalg.yield %element : f16\n    }\n";
  }
  function += "    return ";
  for (size_t index = 0; index < branchCount; ++index) {
    if (index != 0)
      function += ", ";
    function += "%value" + std::to_string(index);
  }
  function += " : ";
  for (size_t index = 0; index < branchCount; ++index) {
    if (index != 0)
      function += ", ";
    function += "tensor<1xf16>";
  }
  function += "\n  }\n";
  return function;
}

std::string makePointwiseChain(size_t nodeCount) {
  std::string function = "\n  func.func @long_chain(%input: tensor<1xf16>) -> "
                         "tensor<1xf16> {\n";
  for (size_t index = 0; index < nodeCount; ++index) {
    const std::string suffix = std::to_string(index);
    const std::string input =
        index == 0 ? "%input" : "%value" + std::to_string(index - 1);
    function += "    %out" + suffix + " = tensor.empty() : tensor<1xf16>\n";
    function += "    %value" + suffix + " = linalg.map ins(" + input +
                " : tensor<1xf16>) outs(%out" + suffix +
                " : tensor<1xf16>) (%element: f16) {\n";
    function += "      linalg.yield %element : f16\n    }\n";
  }
  function += "    return %value" + std::to_string(nodeCount - 1) +
              " : tensor<1xf16>\n  }\n";
  return function;
}

class WholeDAGPlacementFrontierTest : public ::testing::Test {
protected:
  WholeDAGPlacementFrontierTest() {
    registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef function) {
    std::string source = R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir";
    source.append(function.begin(), function.end());
    source.append("\n}\n");
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  CardDAGAnalysis buildDAG(mlir::ModuleOp module) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    std::string failureReason;
    auto dag = CardDAGAnalysis::create(function, &failureReason);
    EXPECT_TRUE(mlir::succeeded(dag)) << failureReason;
    if (mlir::failed(dag))
      std::abort();
    return std::move(*dag);
  }

  llvm::SmallVector<WholeDAGPlacementCandidate, 12>
  derive(mlir::ModuleOp module, WholeDAGPlacementFrontierStatistics &stats) {
    std::string failureReason;
    auto topology = PhysicalTopology::create(module, &failureReason);
    EXPECT_TRUE(mlir::succeeded(topology)) << failureReason;
    if (mlir::failed(topology))
      return {};
    CardDAGAnalysis dag = buildDAG(module);
    auto candidates = deriveWholeDAGPlacementFrontier(
        dag, *topology, PhysicalCardId(0), &stats, &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidates)) << failureReason;
    return mlir::succeeded(candidates)
               ? std::move(*candidates)
               : llvm::SmallVector<WholeDAGPlacementCandidate, 12>{};
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(WholeDAGPlacementFrontierTest,
       ChainExploresBoundedPartialAndDisjointTopologyPlacements) {
  auto module = parse(R"mlir(
  func.func @chain(%input: tensor<32xf16>) -> tensor<32xf16> {
    %out0 = tensor.empty() : tensor<32xf16>
    %out1 = tensor.empty() : tensor<32xf16>
    %producer = linalg.map ins(%input : tensor<32xf16>)
        outs(%out0 : tensor<32xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %consumer = linalg.map ins(%producer : tensor<32xf16>)
        outs(%out1 : tensor<32xf16>) (%value: f16) {
      %next = arith.mulf %value, %value : f16
      linalg.yield %next : f16
    }
    return %consumer : tensor<32xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  ASSERT_FALSE(candidates.empty());
  EXPECT_GT(candidates.size(), 12u);
  EXPECT_GT(statistics.expandedStates, 1u);
  EXPECT_TRUE(llvm::any_of(candidates, [](const auto &candidate) {
    return candidate.partialOverlapEdgeCount != 0;
  }));
  EXPECT_TRUE(llvm::any_of(candidates, [](const auto &candidate) {
    return candidate.disjointEdgeCount != 0;
  }));
  EXPECT_TRUE(llvm::any_of(candidates, [](const auto &candidate) {
    return countPeerFragments(candidate.edgePlan) == 0;
  }));
  EXPECT_TRUE(llvm::any_of(candidates, [](const auto &candidate) {
    return countPeerFragments(candidate.edgePlan) != 0 &&
           candidate.partialOverlapEdgeCount != 0;
  }));
  bool sawScheduledResidency = false;
  for (const auto &candidate : candidates) {
    EXPECT_EQ(candidate.localEdgeCount + candidate.sameGroupRemapEdgeCount +
                  candidate.partialOverlapEdgeCount +
                  candidate.disjointEdgeCount,
              1u);
    EXPECT_GT(candidate.schedule.eventCount, 0u);
    EXPECT_GT(candidate.schedule.makespan, 0u);
    sawScheduledResidency |= candidate.schedule.peakLiveSPMBytes != 0;
    for (const WholeDAGNodePlacement &placement : candidate.nodePlacements) {
      ASSERT_FALSE(placement.iteratorPartitionFactors.empty());
      ASSERT_LT(placement.spatialIteratorDimension,
                placement.iteratorPartitionFactors.size());
      EXPECT_EQ(placement.iteratorPartitionFactors
                    [placement.spatialIteratorDimension],
                placement.tiles.size());
      for (auto [iterator, factor] :
           llvm::enumerate(placement.iteratorPartitionFactors))
        if (iterator != placement.spatialIteratorDimension)
          EXPECT_EQ(factor, 1u);
    }
  }
  EXPECT_TRUE(sawScheduledResidency)
      << "the placement frontier must schedule actual local edge residency";
}

TEST_F(WholeDAGPlacementFrontierTest,
       FullMeshCatalogContainsEveryRectangleAndPrefersCompactTwoByTwo) {
  auto module = parse(R"mlir(
  func.func @single(%input: tensor<4xf16>) -> tensor<4xf16> {
    %out = tensor.empty() : tensor<4xf16>
    %value = linalg.map ins(%input : tensor<4xf16>)
        outs(%out : tensor<4xf16>) (%element: f16) {
      linalg.yield %element : f16
    }
    return %value : tensor<4xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  ASSERT_FALSE(candidates.empty());
  EXPECT_EQ(statistics.placementGroupCount, 100u);
  ASSERT_EQ(candidates.front().nodePlacements.size(), 1u);
  llvm::SmallVector<PhysicalTileId, 4> expected = {
      PhysicalTileId(0), PhysicalTileId(1), PhysicalTileId(4),
      PhysicalTileId(5)};
  EXPECT_EQ(candidates.front().nodePlacements.front().tiles, expected);
}

TEST_F(WholeDAGPlacementFrontierTest,
       SamePhysicalGroupShardRemapIsNotPartialOverlap) {
  auto module = parse(R"mlir(
  func.func @remap(%input: tensor<2x2xf16>)
      -> (tensor<2x2xf16>, tensor<2x2xf16>) {
    %out0 = tensor.empty() : tensor<2x2xf16>
    %out1 = tensor.empty() : tensor<2x2xf16>
    %producer = linalg.map ins(%input : tensor<2x2xf16>)
        outs(%out0 : tensor<2x2xf16>) (%element: f16) {
      linalg.yield %element : f16
    }
    %consumer = linalg.map ins(%producer : tensor<2x2xf16>)
        outs(%out1 : tensor<2x2xf16>) (%element: f16) {
      linalg.yield %element : f16
    }
    return %producer, %consumer : tensor<2x2xf16>, tensor<2x2xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  auto remap = llvm::find_if(candidates, [](const auto &candidate) {
    return candidate.sameGroupRemapEdgeCount != 0;
  });
  ASSERT_NE(remap, candidates.end());
  ASSERT_EQ(remap->nodePlacements.size(), 2u);
  EXPECT_EQ(remap->nodePlacements[0].tiles, remap->nodePlacements[1].tiles);
  EXPECT_NE(remap->nodePlacements[0].shardDimension,
            remap->nodePlacements[1].shardDimension);
  EXPECT_EQ(remap->partialOverlapEdgeCount, 0u);
  EXPECT_EQ(remap->disjointEdgeCount, 0u);
  EXPECT_GT(countPeerFragments(remap->edgePlan), 0u);
}

TEST_F(WholeDAGPlacementFrontierTest,
       IndependentBranchesCanUtilizeAllPhysicalTiles) {
  auto module = parse(makeIndependentPointwiseBranches(16));
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  auto allTiles = llvm::find_if(candidates, [](const auto &candidate) {
    llvm::DenseSet<int64_t> utilized;
    for (const WholeDAGNodePlacement &placement : candidate.nodePlacements)
      for (PhysicalTileId tile : placement.tiles)
        utilized.insert(tile.getValue());
    return utilized.size() == 16;
  });
  ASSERT_NE(allTiles, candidates.end());
  EXPECT_EQ(allTiles->outputPlacements.size(), 16u);
  EXPECT_EQ(allTiles->parallelComponentCount, 16u);
}

TEST_F(WholeDAGPlacementFrontierTest,
       LongDAGUsesDominanceFrontierWithoutStructuralLengthFailure) {
  constexpr size_t nodeCount = 129;
  auto module = parse(makePointwiseChain(nodeCount));
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  ASSERT_FALSE(candidates.empty());
  EXPECT_GT(statistics.expandedStates, 128u);
  EXPECT_EQ(candidates.front().nodePlacements.size(), nodeCount);
}

TEST_F(WholeDAGPlacementFrontierTest,
       MultiOutputFanoutCanPlaceBranchesOnDifferentDestinationGroups) {
  auto module = parse(R"mlir(
  func.func @fanout(%input: tensor<32xf16>)
      -> (tensor<32xf16>, tensor<32xf16>) {
    %out0 = tensor.empty() : tensor<32xf16>
    %out1 = tensor.empty() : tensor<32xf16>
    %out2 = tensor.empty() : tensor<32xf16>
    %source = linalg.map ins(%input : tensor<32xf16>)
        outs(%out0 : tensor<32xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %left = linalg.map ins(%source : tensor<32xf16>)
        outs(%out1 : tensor<32xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %right = linalg.map ins(%source : tensor<32xf16>)
        outs(%out2 : tensor<32xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %left, %right : tensor<32xf16>, tensor<32xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  auto fanout = llvm::find_if(candidates, [](const auto &candidate) {
    if (candidate.outputPlacements.size() != 2 ||
        candidate.outputPlacements[0].tiles ==
            candidate.outputPlacements[1].tiles)
      return false;
    llvm::DenseSet<int64_t> movedEdges;
    for (const SpatialEdgeStrategy &strategy : candidate.edgePlan.strategies)
      for (const SpatialEdgeFragment &fragment : strategy.fragments)
        if (fragment.kind == SpatialEdgeFragmentKind::Peer)
          movedEdges.insert(fragment.communicationId);
    return movedEdges.size() == 2;
  });
  ASSERT_NE(fanout, candidates.end());
}

TEST_F(WholeDAGPlacementFrontierTest,
       FullSpatialShardDoesNotPreemptTemporalResidencySearch) {
  auto module = parse(R"mlir(
  func.func @large_chain(%input: tensor<33554432xf16>)
      -> tensor<33554432xf16> {
    %out0 = tensor.empty() : tensor<33554432xf16>
    %out1 = tensor.empty() : tensor<33554432xf16>
    %producer = linalg.map ins(%input : tensor<33554432xf16>)
        outs(%out0 : tensor<33554432xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %consumer = linalg.map ins(%producer : tensor<33554432xf16>)
        outs(%out1 : tensor<33554432xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %consumer : tensor<33554432xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  ASSERT_FALSE(candidates.empty());
  EXPECT_TRUE(llvm::any_of(candidates, [](const auto &candidate) {
    return llvm::any_of(
        candidate.edgePlan.strategies, [](const SpatialEdgeStrategy &strategy) {
          return strategy.action == SpatialEdgeAction::CoupledFusion;
        });
  }));
}

TEST_F(WholeDAGPlacementFrontierTest,
       DiamondFaninRetainsAnExpressibleTwoOperandBoundary) {
  auto module = parse(R"mlir(
  func.func @diamond(%input: tensor<32xf16>) -> tensor<32xf16> {
    %out0 = tensor.empty() : tensor<32xf16>
    %out1 = tensor.empty() : tensor<32xf16>
    %out2 = tensor.empty() : tensor<32xf16>
    %out3 = tensor.empty() : tensor<32xf16>
    %source = linalg.map ins(%input : tensor<32xf16>)
        outs(%out0 : tensor<32xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %left = linalg.map ins(%source : tensor<32xf16>)
        outs(%out1 : tensor<32xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %right = linalg.map ins(%source : tensor<32xf16>)
        outs(%out2 : tensor<32xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %join = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%left, %right : tensor<32xf16>, tensor<32xf16>)
        outs(%out3 : tensor<32xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %old: f16):
        %next = arith.addf %lhs, %rhs : f16
        linalg.yield %next : f16
    } -> tensor<32xf16>
    return %join : tensor<32xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  EXPECT_TRUE(llvm::any_of(candidates, [](const auto &candidate) {
    llvm::DenseSet<int64_t> edges;
    for (const SpatialEdgeStrategy &strategy : candidate.edgePlan.strategies)
      for (const SpatialEdgeFragment &fragment : strategy.fragments)
        if (fragment.kind == SpatialEdgeFragmentKind::Peer)
          edges.insert(fragment.communicationId);
    return edges.size() == 2;
  }));
}

TEST_F(WholeDAGPlacementFrontierTest,
       ThreeStageChainCanUseThreeDistinctPhysicalTileGroups) {
  auto module = parse(R"mlir(
  func.func @chain(%input: tensor<64xf16>) -> tensor<64xf16> {
    %out0 = tensor.empty() : tensor<64xf16>
    %out1 = tensor.empty() : tensor<64xf16>
    %out2 = tensor.empty() : tensor<64xf16>
    %first = linalg.map ins(%input : tensor<64xf16>)
        outs(%out0 : tensor<64xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %second = linalg.map ins(%first : tensor<64xf16>)
        outs(%out1 : tensor<64xf16>) (%value: f16) {
      %next = arith.mulf %value, %value : f16
      linalg.yield %next : f16
    }
    %third = linalg.map ins(%second : tensor<64xf16>)
        outs(%out2 : tensor<64xf16>) (%value: f16) {
      %next = arith.subf %value, %value : f16
      linalg.yield %next : f16
    }
    return %third : tensor<64xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  auto pipeline = llvm::find_if(candidates, [](const auto &candidate) {
    return candidate.distinctTileGroupCount >= 3 &&
           candidate.nodePlacements.size() == 3;
  });
  ASSERT_NE(pipeline, candidates.end());
  EXPECT_FALSE(pipeline->nodePlacements[0].tiles.empty());
  EXPECT_NE(pipeline->nodePlacements[0].tiles,
            pipeline->nodePlacements[1].tiles);
  EXPECT_NE(pipeline->nodePlacements[1].tiles,
            pipeline->nodePlacements[2].tiles);
  EXPECT_EQ(pipeline->outputPlacements.front().tiles,
            pipeline->nodePlacements[2].tiles);
}

TEST_F(WholeDAGPlacementFrontierTest,
       ReductionDataTransitionFailsClosedWithoutSuppressingLocalAlternatives) {
  auto module = parse(R"mlir(
  func.func @reduce(%input: tensor<32x4xf16>) -> tensor<32xf16> {
    %out0 = tensor.empty() : tensor<32x4xf16>
    %out1 = tensor.empty() : tensor<32xf16>
    %producer = linalg.map ins(%input : tensor<32x4xf16>)
        outs(%out0 : tensor<32x4xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out1 : tensor<32xf16>) -> tensor<32xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%producer : tensor<32x4xf16>) outs(%init : tensor<32xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<32xf16>
    return %sum : tensor<32xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics statistics;
  auto candidates = derive(*module, statistics);
  EXPECT_GT(statistics.rejectedTransitions, 0u);
  for (const auto &candidate : candidates) {
    ASSERT_EQ(candidate.nodePlacements.size(), 3u);
    const WholeDAGNodePlacement &reduction = candidate.nodePlacements[2];
    ASSERT_EQ(reduction.iteratorPartitionFactors.size(), 2u);
    EXPECT_EQ(reduction.spatialIteratorDimension, 0u);
    EXPECT_EQ(reduction.iteratorPartitionFactors[0], reduction.tiles.size());
    EXPECT_EQ(reduction.iteratorPartitionFactors[1], 1u)
        << "reduction iterators remain explicit but unpartitioned";
    for (const SpatialEdgeStrategy &strategy : candidate.edgePlan.strategies)
      for (const SpatialEdgeFragment &fragment : strategy.fragments)
        if (fragment.kind == SpatialEdgeFragmentKind::Peer)
          EXPECT_NE(fragment.communicationId, 0)
              << "the reduction input edge must never be approximated as a "
                 "box";
  }
}

TEST_F(WholeDAGPlacementFrontierTest, CompleteFrontierOrderIsDeterministic) {
  auto module = parse(R"mlir(
  func.func @chain(%input: tensor<32xf16>) -> tensor<32xf16> {
    %out0 = tensor.empty() : tensor<32xf16>
    %out1 = tensor.empty() : tensor<32xf16>
    %first = linalg.map ins(%input : tensor<32xf16>)
        outs(%out0 : tensor<32xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %second = linalg.map ins(%first : tensor<32xf16>)
        outs(%out1 : tensor<32xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %second : tensor<32xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  WholeDAGPlacementFrontierStatistics firstStats;
  auto first = derive(*module, firstStats);
  WholeDAGPlacementFrontierStatistics secondStats;
  auto second = derive(*module, secondStats);
  ASSERT_EQ(first.size(), second.size());
  EXPECT_GT(first.size(), 3u);
  EXPECT_EQ(firstStats.expandedStates, secondStats.expandedStates);
  for (size_t index = 0; index < first.size(); ++index) {
    EXPECT_EQ(first[index].distinctTileGroupCount,
              second[index].distinctTileGroupCount);
    EXPECT_EQ(first[index].partialOverlapEdgeCount,
              second[index].partialOverlapEdgeCount);
    EXPECT_EQ(first[index].sameGroupRemapEdgeCount,
              second[index].sameGroupRemapEdgeCount);
    EXPECT_EQ(first[index].disjointEdgeCount, second[index].disjointEdgeCount);
    EXPECT_EQ(first[index].topologyHopByteWork,
              second[index].topologyHopByteWork);
    EXPECT_EQ(first[index].topologyCompactnessWork,
              second[index].topologyCompactnessWork);
    ASSERT_EQ(first[index].nodePlacements.size(),
              second[index].nodePlacements.size());
    for (auto [lhs, rhs] : llvm::zip_equal(first[index].nodePlacements,
                                           second[index].nodePlacements)) {
      EXPECT_EQ(lhs.node, rhs.node);
      EXPECT_EQ(lhs.shardDimension, rhs.shardDimension);
      EXPECT_EQ(lhs.tiles, rhs.tiles);
    }
  }
}

} // namespace
