//===- StructuredDAGPlacementEnumerationTest.cpp
//-------------------------------===//

#include "../../lib/Wafer/Compiler/StructuredDAGPlacementEnumeration.h"
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

class StructuredDAGPlacementEnumerationTest : public ::testing::Test {
protected:
  StructuredDAGPlacementEnumerationTest() {
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

  StructuredDAGAnalysis buildDAG(mlir::ModuleOp module) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function, &failureReason);
    EXPECT_TRUE(mlir::succeeded(dag)) << failureReason;
    if (mlir::failed(dag))
      std::abort();
    return std::move(*dag);
  }

  static StructuredDAGNodePlacement
  placement(StructuredDAGNodeID node, unsigned dimension,
            std::initializer_list<int64_t> tileValues) {
    StructuredDAGNodePlacement result;
    result.node = node;
    result.spatialPartition =
        StructuredDAGSpatialPartition{dimension, dimension};
    for (int64_t value : tileValues)
      result.tiles.push_back(TileId(value));
    return result;
  }

  llvm::SmallVector<StructuredDAGPlacementCandidate, 12>
  derive(mlir::ModuleOp module,
         StructuredDAGPlacementEnumerationStatistics &stats) {
    std::string failureReason;
    auto topology = TargetTopology::create(module, &failureReason);
    EXPECT_TRUE(mlir::succeeded(topology)) << failureReason;
    if (mlir::failed(topology))
      return {};
    StructuredDAGAnalysis dag = buildDAG(module);
    auto candidates = enumerateStructuredDAGPlacements(
        dag, *topology, CardId(0), &stats, &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidates)) << failureReason;
    return mlir::succeeded(candidates)
               ? std::move(*candidates)
               : llvm::SmallVector<StructuredDAGPlacementCandidate, 12>{};
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredDAGPlacementEnumerationTest,
       MultiResultProducerEntersProductionDomainAndEvaluator) {
  auto module = parse(R"mlir(
  func.func @multi(%input: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
    %preout = tensor.empty() : tensor<8xf16>
    %pre = linalg.map ins(%input : tensor<8xf16>)
        outs(%preout : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %out0 = tensor.empty() : tensor<8xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %producer:2 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%pre : tensor<8xf16>)
        outs(%out0, %out1 : tensor<8xf16>, tensor<8xf16>) {
      ^bb0(%in: f16, %o0: f16, %o1: f16):
        %sum = arith.addf %in, %o0 : f16
        linalg.yield %sum, %sum : f16, f16
    } -> (tensor<8xf16>, tensor<8xf16>)
    %out2 = tensor.empty() : tensor<8xf16>
    %consumer0 = linalg.map ins(%producer#0 : tensor<8xf16>)
        outs(%out2 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %out3 = tensor.empty() : tensor<8xf16>
    %consumer1 = linalg.map ins(%producer#1 : tensor<8xf16>)
        outs(%out3 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %consumer0, %consumer1 : tensor<8xf16>, tensor<8xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto topology = TargetTopology::create(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(topology)) << failureReason;
  StructuredDAGAnalysis dag = buildDAG(*module);
  auto domain = deriveStructuredDAGPlacementSearchDomain(
      dag, *topology, CardId(0), &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  ASSERT_EQ(domain->nodeOptions.size(), 4u);

  auto firstPair = llvm::find_if(domain->nodeOptions.front(),
                                 [](const StructuredDAGNodePlacement &option) {
                                   return option.tiles.size() == 2;
                                 });
  ASSERT_NE(firstPair, domain->nodeOptions.front().end());
  llvm::SmallVector<StructuredDAGNodePlacement, 16> placements;
  for (const auto &options : domain->nodeOptions) {
    auto matching = llvm::find_if(options, [&](const auto &option) {
      return option.tiles == firstPair->tiles &&
             option.spatialPartition &&
             option.spatialPartition->iteratorDimension == 0;
    });
    ASSERT_NE(matching, options.end());
    placements.push_back(*matching);
  }

  StructuredDAGPlacementEvaluator evaluator(dag, *topology, CardId(0));
  StructuredDAGPlacementLegality legality;
  auto candidate = evaluator.evaluate(placements, &failureReason, &legality);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_EQ(legality.status, analysis::ExactDemandStatus::Satisfied);
  EXPECT_TRUE(candidate->edgeCarrierComplete);
  EXPECT_EQ(candidate->nodePlacements.size(), 4u);
}

TEST_F(StructuredDAGPlacementEnumerationTest,
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
  StructuredDAGPlacementEnumerationStatistics statistics;
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
    for (const StructuredDAGNodePlacement &placement :
         candidate.nodePlacements) {
      ASSERT_FALSE(placement.iteratorPartitionFactors.empty());
      ASSERT_TRUE(placement.spatialPartition);
      ASSERT_LT(placement.spatialPartition->iteratorDimension,
                placement.iteratorPartitionFactors.size());
      EXPECT_EQ(
          placement.iteratorPartitionFactors
              [placement.spatialPartition->iteratorDimension],
          placement.tiles.size());
      for (auto [iterator, factor] :
           llvm::enumerate(placement.iteratorPartitionFactors))
        if (iterator != placement.spatialPartition->iteratorDimension)
          EXPECT_EQ(factor, 1u);
    }
  }
  EXPECT_TRUE(sawScheduledResidency)
      << "the placement enumeration must schedule actual local edge residency";
}

TEST_F(StructuredDAGPlacementEnumerationTest,
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
  StructuredDAGPlacementEnumerationStatistics statistics;
  auto candidates = derive(*module, statistics);
  ASSERT_FALSE(candidates.empty());
  EXPECT_EQ(statistics.placementGroupCount, 100u);
  ASSERT_EQ(candidates.front().nodePlacements.size(), 1u);
  llvm::SmallVector<TileId, 4> expected = {TileId(0), TileId(1), TileId(4),
                                           TileId(5)};
  EXPECT_EQ(candidates.front().nodePlacements.front().tiles, expected);
}

TEST_F(StructuredDAGPlacementEnumerationTest,
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
  StructuredDAGPlacementEnumerationStatistics statistics;
  auto candidates = derive(*module, statistics);
  auto remap = llvm::find_if(candidates, [](const auto &candidate) {
    return candidate.sameGroupRemapEdgeCount != 0;
  });
  ASSERT_NE(remap, candidates.end());
  ASSERT_EQ(remap->nodePlacements.size(), 2u);
  EXPECT_EQ(remap->nodePlacements[0].tiles, remap->nodePlacements[1].tiles);
  ASSERT_TRUE(remap->nodePlacements[0].spatialPartition);
  ASSERT_TRUE(remap->nodePlacements[1].spatialPartition);
  EXPECT_NE(remap->nodePlacements[0].spatialPartition->resultDimension,
            remap->nodePlacements[1].spatialPartition->resultDimension);
  EXPECT_EQ(remap->partialOverlapEdgeCount, 0u);
  EXPECT_EQ(remap->disjointEdgeCount, 0u);
  EXPECT_GT(countPeerFragments(remap->edgePlan), 0u);
}

TEST_F(StructuredDAGPlacementEnumerationTest,
       IndependentBranchesCanUtilizeAllTiles) {
  auto module = parse(makeIndependentPointwiseBranches(16));
  ASSERT_TRUE(module);
  StructuredDAGPlacementEnumerationStatistics statistics;
  auto candidates = derive(*module, statistics);
  auto allTiles = llvm::find_if(candidates, [](const auto &candidate) {
    llvm::DenseSet<int64_t> utilized;
    for (const StructuredDAGNodePlacement &placement : candidate.nodePlacements)
      for (TileId tile : placement.tiles)
        utilized.insert(tile.getValue());
    return utilized.size() == 16;
  });
  ASSERT_NE(allTiles, candidates.end());
  EXPECT_EQ(allTiles->outputPlacements.size(), 16u);
  EXPECT_EQ(allTiles->parallelComponentCount, 16u);
}

TEST_F(StructuredDAGPlacementEnumerationTest,
       LongDAGUsesNondominatedStatesWithoutStructuralLengthFailure) {
  constexpr size_t nodeCount = 129;
  auto module = parse(makePointwiseChain(nodeCount));
  ASSERT_TRUE(module);
  StructuredDAGPlacementEnumerationStatistics statistics;
  auto candidates = derive(*module, statistics);
  ASSERT_FALSE(candidates.empty());
  EXPECT_GT(statistics.expandedStates, 128u);
  EXPECT_EQ(candidates.front().nodePlacements.size(), nodeCount);
}

TEST_F(StructuredDAGPlacementEnumerationTest,
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
  StructuredDAGPlacementEnumerationStatistics statistics;
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

TEST_F(StructuredDAGPlacementEnumerationTest,
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
  StructuredDAGPlacementEnumerationStatistics statistics;
  auto candidates = derive(*module, statistics);
  ASSERT_FALSE(candidates.empty());
  EXPECT_TRUE(llvm::any_of(candidates, [](const auto &candidate) {
    return llvm::any_of(
        candidate.edgePlan.strategies, [](const SpatialEdgeStrategy &strategy) {
          return strategy.action == SpatialEdgeAction::LocalShardResidency;
        });
  }));
}

TEST_F(StructuredDAGPlacementEnumerationTest,
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
  StructuredDAGPlacementEnumerationStatistics statistics;
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

TEST_F(StructuredDAGPlacementEnumerationTest,
       ThreeStageChainCanUseThreeDistinctTileGroups) {
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
  StructuredDAGPlacementEnumerationStatistics statistics;
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

TEST_F(StructuredDAGPlacementEnumerationTest,
       ReductionDataTransitionKeepsExactFragmentsAndLocalAlternatives) {
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
  StructuredDAGPlacementEnumerationStatistics statistics;
  auto candidates = derive(*module, statistics);
  EXPECT_GT(statistics.rejectedTransitions, 0u);
  for (const auto &candidate : candidates) {
    ASSERT_EQ(candidate.nodePlacements.size(), 3u);
    const StructuredDAGNodePlacement &reduction = candidate.nodePlacements[2];
    ASSERT_EQ(reduction.iteratorPartitionFactors.size(), 2u);
    ASSERT_TRUE(reduction.spatialPartition);
    EXPECT_EQ(reduction.spatialPartition->iteratorDimension, 0u);
    EXPECT_EQ(reduction.iteratorPartitionFactors[0], reduction.tiles.size());
    EXPECT_EQ(reduction.iteratorPartitionFactors[1], 1u)
        << "reduction iterators remain explicit but unpartitioned";
    for (const SpatialEdgeStrategy &strategy : candidate.edgePlan.strategies)
      for (const SpatialEdgeFragment &fragment : strategy.fragments)
        if (fragment.kind == SpatialEdgeFragmentKind::Peer)
          EXPECT_NE(fragment.communicationId, 0)
              << "the reduction input edge must keep its exact edge identity";
  }
}

TEST_F(StructuredDAGPlacementEnumerationTest,
       CompletePlacementOrderIsDeterministic) {
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
  StructuredDAGPlacementEnumerationStatistics firstStats;
  auto first = derive(*module, firstStats);
  StructuredDAGPlacementEnumerationStatistics secondStats;
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
      EXPECT_EQ(lhs.spatialPartition, rhs.spatialPartition);
      EXPECT_EQ(lhs.tiles, rhs.tiles);
    }
  }
}

TEST_F(StructuredDAGPlacementEnumerationTest,
       IndeterminateDemandAbortsEvaluationWithTypedStatus) {
  std::string dims = "1";
  for (unsigned index = 1; index < 17; ++index)
    dims += "x1";
  auto module = parse(R"mlir(
  func.func @rank17(%input: tensor<)mlir" +
                      dims + R"mlir(xf16>) -> tensor<)mlir" + dims +
                      R"mlir(xf16> {
    %out0 = tensor.empty() : tensor<)mlir" +
                      dims + R"mlir(xf16>
    %producer = linalg.map ins(%input : tensor<)mlir" +
                      dims + R"mlir(xf16>)
        outs(%out0 : tensor<)mlir" +
                      dims + R"mlir(xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %out1 = tensor.empty() : tensor<)mlir" +
                      dims + R"mlir(xf16>
    %consumer = linalg.map ins(%producer : tensor<)mlir" +
                      dims + R"mlir(xf16>)
        outs(%out1 : tensor<)mlir" +
                      dims + R"mlir(xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %consumer : tensor<)mlir" +
                      dims + R"mlir(xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto topology = TargetTopology::create(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(topology)) << failureReason;
  StructuredDAGAnalysis dag = buildDAG(*module);

  StructuredDAGPlacementEvaluator evaluator(dag, *topology, CardId(0));
  StructuredDAGPlacementLegality legality;
  auto result = evaluator.evaluate({placement(0, 0, {0}), placement(1, 0, {0})},
                                   &failureReason, &legality);
  EXPECT_TRUE(mlir::failed(result));
  // The 34-variable relation space exceeds the IndexRelation budget; this is
  // a machinery failure, never a placement rejection.
  EXPECT_EQ(legality.status, analysis::ExactDemandStatus::IndeterminateFailure);
}

TEST_F(StructuredDAGPlacementEnumerationTest,
       UnsupportedDemandAbortsEvaluationWithTypedStatus) {
  auto module = parse(R"mlir(
  func.func @ambiguous(%input: tensor<16xf16>) -> tensor<8xf16> {
    %out0 = tensor.empty() : tensor<16xf16>
    %producer = linalg.map ins(%input : tensor<16xf16>)
        outs(%out0 : tensor<16xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %t = tensor.extract_slice %producer[0] [8] [1]
        : tensor<16xf16> to tensor<8xf16>
    %empty0 = tensor.empty() : tensor<8xf16>
    %stage = tensor.insert_slice %t into %empty0[0] [8] [1]
        : tensor<8xf16> into tensor<8xf16>
    %cat = tensor.insert_slice %t into %stage[0] [8] [1]
        : tensor<8xf16> into tensor<8xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %consumer = linalg.map ins(%cat : tensor<8xf16>)
        outs(%out1 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %consumer : tensor<8xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto topology = TargetTopology::create(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(topology)) << failureReason;
  StructuredDAGAnalysis dag = buildDAG(*module);

  StructuredDAGPlacementEvaluator evaluator(dag, *topology, CardId(0));
  StructuredDAGPlacementLegality legality;
  auto result = evaluator.evaluate({placement(0, 0, {0}), placement(1, 0, {0})},
                                   &failureReason, &legality);
  EXPECT_TRUE(mlir::failed(result));
  // One support operation carries the producer through two of its operands;
  // no placement fixes that semantic ambiguity.
  EXPECT_EQ(legality.status,
            analysis::ExactDemandStatus::UnsupportedSemanticRelation);
}

TEST_F(StructuredDAGPlacementEnumerationTest,
       CarrierFailureDoesNotRejectThePlacementTrial) {
  auto module = parse(R"mlir(
  func.func @strided(%input: tensor<8xf16>) -> tensor<4xf16> {
    %out0 = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%out0 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %out1 = tensor.empty() : tensor<4xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0 * 2)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%producer : tensor<8xf16>)
        outs(%out1 : tensor<4xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<4xf16>
    return %consumer : tensor<4xf16>
  }
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto topology = TargetTopology::create(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(topology)) << failureReason;
  StructuredDAGAnalysis dag = buildDAG(*module);

  StructuredDAGPlacementEvaluator evaluator(dag, *topology, CardId(0));
  StructuredDAGPlacementLegality legality;
  auto result = evaluator.evaluate({placement(0, 0, {0}), placement(1, 0, {0})},
                                   &failureReason, &legality);
  // The strided demand is logically exact; the canonical carrier cannot
  // express it and only marks the candidate carrier-incomplete.
  ASSERT_TRUE(mlir::succeeded(result)) << failureReason;
  EXPECT_EQ(legality.status, analysis::ExactDemandStatus::Satisfied);
  EXPECT_FALSE(result->edgeCarrierComplete);
}

} // namespace
