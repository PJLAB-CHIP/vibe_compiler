//===- SpatialDomainTest.cpp -------------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"

#include "TestSupport/CodeGen/ExecutableTestSupport.h"
#include "TestSupport/Planning/SpatialPlanReference.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPartitionPropagation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

struct BuiltSpatialDomain {
  StructuredDAGAnalysis dag;
  SpatialPlanDomain domain;
};

class SpatialDomainTest : public ::testing::Test {
protected:
  SpatialDomainTest() {
    registerWaferCoreDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  static std::string withTopology(std::string source, int64_t rows,
                                  int64_t columns,
                                  llvm::StringRef unavailable = {}) {
    size_t insertion = source.find("module {");
    EXPECT_NE(insertion, std::string::npos);
    insertion += std::string("module {").size();
    std::string topology;
    llvm::raw_string_ostream stream(topology);
    stream << "\n  wafer.target.topology @target\n"
           << "      {card_grid = array<i64: 1, 1>, card_interconnect = "
              "\"mesh\",\n"
           << "       tile_grid = array<i64: " << rows << ", " << columns
           << ">, unavailable_tiles = array<i64";
    if (!unavailable.empty())
      stream << ": " << unavailable;
    stream << ">}\n";
    source.insert(insertion, topology);
    return source;
  }

  static std::string mapSource(int64_t extent, llvm::StringRef elementType,
                               bool rankFour = false) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "module {\n  func.func @main(%input: tensor<2x";
    if (rankFour)
      stream << "2x";
    stream << extent << "x128x" << elementType << ">) -> tensor<2x";
    if (rankFour)
      stream << "2x";
    stream << extent << "x128x" << elementType << "> {\n"
           << "    %empty = tensor.empty() : tensor<2x";
    if (rankFour)
      stream << "2x";
    stream << extent << "x128x" << elementType << ">\n"
           << "    %result = linalg.map ins(%input : tensor<2x";
    if (rankFour)
      stream << "2x";
    stream << extent << "x128x" << elementType << ">) outs(%empty : tensor<2x";
    if (rankFour)
      stream << "2x";
    stream << extent << "x128x" << elementType << ">) (%value: " << elementType
           << ") {\n"
           << "      linalg.yield %value : " << elementType << "\n"
           << "    }\n    return %result : tensor<2x";
    if (rankFour)
      stream << "2x";
    stream << extent << "x128x" << elementType << ">\n  }\n}\n";
    return source;
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  std::optional<BuiltSpatialDomain> build(mlir::ModuleOp module,
                                          std::string &failureReason) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    auto dag = StructuredDAGAnalysis::create(function, &failureReason);
    if (mlir::failed(dag))
      return std::nullopt;
    auto topology = TargetTopology::create(module, &failureReason);
    if (mlir::failed(topology))
      return std::nullopt;
    SpatialPlanDomainResult domain =
        buildSpatialPlanDomain(*dag, *topology, CardId(0));
    if (!domain.succeeded()) {
      failureReason = domain.failure ? domain.failure->detail
                                     : "spatial domain returned no detail";
      return std::nullopt;
    }
    return BuiltSpatialDomain{std::move(*dag), std::move(*domain.domain)};
  }

  static std::optional<std::set<SpatialPlan>>
  enumerate(const SpatialPlanDomain &domain, size_t limit = 100000) {
    std::set<SpatialPlan> plans;
    SpatialPlan current = domain.getFirstPlan();
    while (true) {
      if (!domain.contains(current) || !plans.insert(current).second ||
          plans.size() > limit)
        return std::nullopt;
      SpatialPlanSuccessor next = domain.getNextPlan(current);
      if (next.kind == SpatialPlanSuccessorKind::End)
        break;
      if (next.kind != SpatialPlanSuccessorKind::Successor || !next.plan)
        return std::nullopt;
      current = std::move(*next.plan);
    }
    return plans;
  }

  static void expectExactShardCoverage(const NodeExecutionPartition &node,
                                       llvm::ArrayRef<int64_t> extents) {
    uint64_t expected = 1;
    for (int64_t extent : extents)
      expected *= static_cast<uint64_t>(extent);
    uint64_t covered = 0;
    for (const ExecutionShard &shard : node.shards) {
      ASSERT_EQ(shard.iterationDomain.size(), extents.size());
      uint64_t volume = 1;
      for (auto [axis, interval] : llvm::enumerate(shard.iterationDomain)) {
        EXPECT_GE(interval.offset, 0);
        EXPECT_GT(interval.size, 0);
        EXPECT_LE(interval.getEnd(), extents[axis]);
        volume *= static_cast<uint64_t>(interval.size);
      }
      covered += volume;
    }
    EXPECT_EQ(covered, expected);
    for (size_t lhs = 0; lhs < node.shards.size(); ++lhs)
      for (size_t rhs = lhs + 1; rhs < node.shards.size(); ++rhs) {
        bool disjoint = false;
        for (size_t axis = 0; axis < extents.size(); ++axis) {
          const IteratorInterval &left = node.shards[lhs].iterationDomain[axis];
          const IteratorInterval &right =
              node.shards[rhs].iterationDomain[axis];
          disjoint |=
              left.getEnd() <= right.offset || right.getEnd() <= left.offset;
        }
        EXPECT_TRUE(disjoint);
      }
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(SpatialDomainTest,
       RealAlignedAndRaggedPlansCloseThroughDemandAndRootWork) {
  struct Case {
    int64_t extent;
    llvm::StringRef type;
    bool rankFour;
  };
  for (const Case &test :
       {Case{1024, "f16", false}, Case{1025, "bf16", true}}) {
    SCOPED_TRACE(test.extent);
    auto module = parse(
        withTopology(mapSource(test.extent, test.type, test.rankFour), 4, 4));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto built = build(*module, failureReason);
    ASSERT_TRUE(built) << failureReason;
    auto proposals = built->domain.getProposals();
    ASSERT_FALSE(proposals.empty());
    ASSERT_TRUE(built->domain.contains(proposals.front()));
    auto assignment = built->domain.close(proposals.front(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(assignment)) << failureReason;
    ASSERT_EQ(assignment->nodes.size(), 1u);
    ASSERT_EQ(assignment->nodes.front().shards.size(), 16u);
    expectExactShardCoverage(
        assignment->nodes.front(),
        built->domain.getProblem().getRoots().front().iteratorExtents);

    SpatialDomainEvaluation evaluation =
        built->domain.evaluate(built->dag, proposals.front());
    ASSERT_TRUE(evaluation.isSatisfied());
    const analysis::ExactDemandProof *proof =
        analysis::getExactDemandProof(*evaluation.demand);
    ASSERT_NE(proof, nullptr);
    auto rootWork = RootRegionWorkAnalysis::create(
        built->dag, *evaluation.assignment, *proof, &failureReason);
    ASSERT_TRUE(mlir::succeeded(rootWork)) << failureReason;
    size_t tilesWithWork = 0;
    for (TileId tile :
         built->domain.getProblem().getStructuralProblem().getAvailableTiles())
      tilesWithWork +=
          std::holds_alternative<analysis::RootRegionWork>(rootWork->query(
              built->domain.getProblem().getRoots().front().root, tile));
    EXPECT_EQ(tilesWithWork, 16u);
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(SpatialDomainTest, DirectionCursorCoversAllBalancedSupportsAndRatios) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (int64_t side : {2, 4}) {
      auto module = parse(withTopology(mapSource(extent, "f16"), side, side));
      std::string detail;
      auto built = build(*module, detail);
      ASSERT_TRUE(built) << detail;
      auto anchor = built->domain.getProposals().front();
      SpatialDirectionCursor cursor;
      std::set<std::vector<int64_t>> actual;
      std::set<std::vector<size_t>> firstSupports;
      const size_t supportCount = side == 2 ? 6 : 7;
      while (true) {
        auto next = built->domain.getNextDirectionPlan(anchor, cursor);
        if (next.kind == SpatialPlanSuccessorKind::End)
          break;
        ASSERT_EQ(next.kind, SpatialPlanSuccessorKind::Successor);
        ASSERT_TRUE(next.plan);
        ASSERT_TRUE(built->domain.contains(*next.plan));
        const auto &node = next.plan->nodes.front();
        std::vector<int64_t> factors;
        std::vector<size_t> support;
        for (const auto &axis : node.axes) {
          EXPECT_EQ(axis.scheme, IteratorPartitionScheme::BalancedParts);
          factors.push_back(axis.parameter);
          if (axis.parameter > 1)
            support.push_back(axis.iterator);
        }
        ASSERT_TRUE(actual.insert(factors).second);
        ASSERT_LT(actual.size(), 1000u);
        if (actual.size() <= supportCount) {
          EXPECT_TRUE(firstSupports.insert(support).second);
        }
        auto assignment = built->domain.close(*next.plan, &detail);
        ASSERT_TRUE(mlir::succeeded(assignment)) << detail;
        expectExactShardCoverage(
            assignment->nodes.front(),
            built->domain.getProblem().getRoots().front().iteratorExtents);
        auto evaluation = built->domain.evaluate(built->dag, *next.plan);
        ASSERT_TRUE(evaluation.isSatisfied());
      }
      std::set<std::vector<int64_t>> expected;
      const int64_t tiles = side * side;
      for (int64_t batch = 1; batch <= 2; ++batch)
        for (int64_t rows = 1; rows <= tiles; ++rows)
          for (int64_t columns = 1; columns <= tiles; ++columns)
            if (batch * rows * columns <= tiles && batch * rows * columns > 1)
              expected.insert({batch, rows, columns});
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(firstSupports.size(), supportCount);
    }
  }
}

TEST_F(SpatialDomainTest,
       TinyRawSuccessorsMatchIndependentSchemeAndEmbeddingOracle) {
  // Extent 4 and three Tiles are intentionally bounded: UniformExtent(3)
  // produces [3, 1], distinct from every BalancedParts interval vector.
  auto module = parse(withTopology(R"mlir(
module {
  func.func @main(%input: tensor<1x4x1xf16>) -> tensor<1x4x1xf16> {
    %empty = tensor.empty() : tensor<1x4x1xf16>
    %result = linalg.map ins(%input : tensor<1x4x1xf16>)
        outs(%empty : tensor<1x4x1xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<1x4x1xf16>
  }
}
)mlir",
                                   1, 4, "0, 0, 0, 1"));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  ASSERT_EQ(built->domain.getProblem().getRoots().size(), 1u);
  const SpatialRootDomainFacts root =
      built->domain.getProblem().getRoots().front();
  wafer::test::ReferenceSpatialRoot reference;
  reference.root = root.root;
  reference.iteratorExtents = {1, 4, 1};
  reference.partitionableIterators = {1, 1, 1};
  reference.reductionIterators = {0, 0, 0};
  reference.resultParallelIterators = {1, 1, 1};
  std::set<SpatialPlan> expected = wafer::test::enumerateReferenceSpatialPlans(
      {reference},
      built->domain.getProblem().getStructuralProblem().getAvailableTiles());
  auto actual = enumerate(built->domain);
  ASSERT_TRUE(actual);
  EXPECT_EQ(*actual, expected);
  EXPECT_EQ(actual->size(), 21u);
  // Project the independent complete oracle onto axes. The axis cursor must
  // reach all of them before enumerating embeddings, with one real witness
  // per tuple; the complete successor above still covers all 21 placements.
  std::set<std::vector<IteratorPartition>> expectedAxes, visitedAxes;
  for (const SpatialPlan &plan : expected) {
    const auto &axes = plan.nodes.front().axes;
    expectedAxes.emplace(axes.begin(), axes.end());
  }
  SpatialPlan axisPlan = built->domain.getFirstPlan();
  while (true) {
    ASSERT_TRUE(expected.count(axisPlan));
    const auto &axes = axisPlan.nodes.front().axes;
    ASSERT_TRUE(visitedAxes.emplace(axes.begin(), axes.end()).second);
    auto next = built->domain.getNextPlan(axisPlan,
                                          SpatialSuccessorDomain::AxisSchemes);
    if (next.kind == SpatialPlanSuccessorKind::End)
      break;
    ASSERT_EQ(next.kind, SpatialPlanSuccessorKind::Successor);
    ASSERT_TRUE(next.plan);
    axisPlan = std::move(*next.plan);
  }
  EXPECT_EQ(visitedAxes, expectedAxes);
  for (const SpatialPlan &proposal : built->domain.getProposals())
    EXPECT_TRUE(expected.count(proposal));

  SpatialPlan invalid = built->domain.getFirstPlan();
  invalid.nodes.front().embedding = {TileId(0), TileId(0)};
  EXPECT_FALSE(built->domain.contains(invalid));
  SpatialPlanSuccessor successor = built->domain.getNextPlan(invalid);
  EXPECT_EQ(successor.kind, SpatialPlanSuccessorKind::Failure);
  ASSERT_TRUE(successor.failure);
  EXPECT_EQ(successor.failure->kind, SpatialDomainFailureKind::BrokenContract);
  invalid = built->domain.getFirstPlan();
  invalid.nodes.front().embedding = {TileId(1)};
  EXPECT_FALSE(built->domain.contains(invalid));

  SpatialDomainProblemResult reordered =
      buildSpatialDomainProblem(built->dag, {TileId(3), TileId(0), TileId(2)});
  ASSERT_TRUE(reordered.succeeded());
  EXPECT_EQ(reordered.problem->getStructuralProblem().getAvailableTiles(),
            (llvm::ArrayRef<TileId>{TileId(0), TileId(2), TileId(3)}));
  SpatialDomainProblemResult duplicate =
      buildSpatialDomainProblem(built->dag, {TileId(0), TileId(0)});
  EXPECT_FALSE(duplicate.succeeded());
  ASSERT_TRUE(duplicate.failure);
  EXPECT_EQ(duplicate.failure->kind, SpatialDomainFailureKind::BrokenContract);
}

TEST_F(SpatialDomainTest,
       TinyReductionSuccessorsMatchIndependentPerGroupMergeOracle) {
  // This bounded oracle is intentionally tiny; 1024/1025 reduction mechanics
  // are covered by the real-scale tests below.
  auto module = parse(withTopology(R"mlir(
#input = affine_map<(b, k, n) -> (b, k, n)>
#output = affine_map<(b, k, n) -> (b, n)>
module {
  func.func @main(%input: tensor<1x4x1xf16>, %init: tensor<1x1xf16>)
      -> tensor<1x1xf16> {
    %result = linalg.generic {
        indexing_maps = [#input, #output],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%input : tensor<1x4x1xf16>) outs(%init : tensor<1x1xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<1x1xf16>
    return %result : tensor<1x1xf16>
  }
}
)mlir",
                                   1, 2));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  const SpatialRootDomainFacts root =
      built->domain.getProblem().getRoots().front();
  wafer::test::ReferenceSpatialRoot reference;
  reference.root = root.root;
  reference.iteratorExtents = {1, 4, 1};
  reference.partitionableIterators = {1, 1, 1};
  reference.reductionIterators = {0, 1, 0};
  reference.resultParallelIterators = {1, 0, 1};
  reference.resultParallelIteratorsByGroup = {{1, 0, 1}};
  reference.reductionResultGroupCount = 1;
  auto expected = wafer::test::enumerateReferenceSpatialPlans(
      {reference},
      built->domain.getProblem().getStructuralProblem().getAvailableTiles());
  auto actual = enumerate(built->domain);
  ASSERT_TRUE(actual);
  EXPECT_EQ(*actual, expected);
  EXPECT_EQ(actual->size(), 10u);
}

TEST_F(SpatialDomainTest,
       CompactTwoByTwoProposalDoesNotRemoveDisconnectedEmbedding) {
  // This is a bounded proposal-order check; realistic close/demand coverage
  // for the same partition/embedding mechanism is provided above.
  auto module = parse(withTopology(R"mlir(
module {
  func.func @main(%input: tensor<1x4x1xf16>) -> tensor<1x4x1xf16> {
    %empty = tensor.empty() : tensor<1x4x1xf16>
    %result = linalg.map ins(%input : tensor<1x4x1xf16>)
        outs(%empty : tensor<1x4x1xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<1x4x1xf16>
  }
}
)mlir",
                                   4, 4));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  auto proposals = built->domain.getProposals();
  ASSERT_FALSE(proposals.empty());
  EXPECT_EQ(proposals.front().nodes.front().embedding,
            (llvm::SmallVector<TileId, 16>{TileId(0), TileId(1), TileId(4),
                                           TileId(5)}));

  SpatialPlan disconnected = proposals.front();
  disconnected.nodes.front().embedding = {TileId(0), TileId(3), TileId(12),
                                          TileId(15)};
  EXPECT_TRUE(built->domain.contains(disconnected));
}

TEST_F(SpatialDomainTest, ScalarRawDomainHasOneEmptyCellOnEveryAvailableTile) {
  auto module = parse(withTopology(R"mlir(
module {
  func.func @main(%value: f16) -> tensor<f16> {
    %empty = tensor.empty() : tensor<f16>
    %result = linalg.fill ins(%value : f16) outs(%empty : tensor<f16>)
        -> tensor<f16>
    return %result : tensor<f16>
  }
}
)mlir",
                                   1, 4));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  auto plans = enumerate(built->domain);
  ASSERT_TRUE(plans);
  EXPECT_EQ(plans->size(), 4u);
  for (const SpatialPlan &plan : *plans) {
    ASSERT_TRUE(plan.nodes.front().axes.empty());
    ASSERT_EQ(plan.nodes.front().embedding.size(), 1u);
    auto assignment = built->domain.close(plan, &failureReason);
    ASSERT_TRUE(mlir::succeeded(assignment)) << failureReason;
    EXPECT_TRUE(
        assignment->nodes.front().shards.front().iterationDomain.empty());
  }
}

TEST_F(SpatialDomainTest,
       OrdinaryReductionHasPerParallelGroupAndEveryMergeTileChoice) {
  auto module = parse(withTopology(R"mlir(
#input = affine_map<(b, k, n) -> (b, k, n)>
#output = affine_map<(b, k, n) -> (b, n)>
module {
  func.func @main(%input: tensor<2x1025x128xf16>,
                  %init: tensor<2x128xf16>) -> tensor<2x128xf16> {
    %result = linalg.generic {
        indexing_maps = [#input, #output],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%init : tensor<2x128xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<2x128xf16>
    return %result : tensor<2x128xf16>
  }
}
)mlir",
                                   4, 4));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  const SpatialRootDomainFacts root =
      built->domain.getProblem().getRoots().front();
  ASSERT_TRUE(root.partitionableReductionIterators.test(1));
  auto proposals = built->domain.getProposals();
  ASSERT_FALSE(proposals.empty());
  ASSERT_EQ(proposals.front().nodes.front().axes.size(), 3u);
  EXPECT_EQ(proposals.front().nodes.front().axes[1].parameter, 1)
      << "the first constructive proposal must not add an optional spatial "
         "reduction";
  EXPECT_EQ(proposals.front().nodes.front().embedding.size(), 16u);

  SpatialPlan plan = built->domain.getFirstPlan();
  NodeSpatialPlan &node = plan.nodes.front();
  node.axes = {{0, IteratorPartitionScheme::BalancedParts, 2},
               {1, IteratorPartitionScheme::BalancedParts, 2},
               {2, IteratorPartitionScheme::BalancedParts, 1}};
  node.embedding = {TileId(0), TileId(1), TileId(4), TileId(5)};
  auto groups = deriveSpatialReductionGroups(root, node.axes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(groups)) << failureReason;
  ASSERT_EQ(groups->size(), 2u);
  node.reductionMerges = {{(*groups)[0], TileId(15)},
                          {(*groups)[1], TileId(0)}};
  ASSERT_TRUE(built->domain.contains(plan));
  SpatialDomainEvaluation evaluation = built->domain.evaluate(built->dag, plan);
  ASSERT_TRUE(evaluation.isSatisfied());
  const analysis::ExactDemandProof *proof =
      analysis::getExactDemandProof(*evaluation.demand);
  ASSERT_NE(proof, nullptr);
  ASSERT_EQ(proof->reductionMerges.size(), 2u);
  EXPECT_EQ(proof->reductionMerges[0].mergeTile, TileId(15));
  EXPECT_EQ(proof->reductionMerges[1].mergeTile, TileId(0));
  for (const auto &merge : proof->reductionMerges)
    EXPECT_EQ(merge.contributions.size(), 2u);

  for (int64_t tile = 0; tile < 16; ++tile) {
    node.reductionMerges[0].tile = TileId(tile);
    EXPECT_TRUE(built->domain.contains(plan));
  }
  node.reductionMerges.pop_back();
  EXPECT_FALSE(built->domain.contains(plan));
}

TEST_F(SpatialDomainTest,
       TinyChainAndIndependentBranchesMatchProgramCartesianOracle) {
  constexpr llvm::StringLiteral chain = R"mlir(
module {
  func.func @main(%input: tensor<1x2x1xf16>) -> tensor<1x2x1xf16> {
    %e0 = tensor.empty() : tensor<1x2x1xf16>
    %first = linalg.map ins(%input : tensor<1x2x1xf16>)
        outs(%e0 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<1x2x1xf16>
    %second = linalg.map ins(%first : tensor<1x2x1xf16>)
        outs(%e1 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    return %second : tensor<1x2x1xf16>
  }
}
)mlir";
  constexpr llvm::StringLiteral branches = R"mlir(
module {
  func.func @main(%lhs: tensor<1x2x1xf16>, %rhs: tensor<1x2x1xf16>)
      -> (tensor<1x2x1xf16>, tensor<1x2x1xf16>) {
    %e0 = tensor.empty() : tensor<1x2x1xf16>
    %left = linalg.map ins(%lhs : tensor<1x2x1xf16>)
        outs(%e0 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<1x2x1xf16>
    %right = linalg.map ins(%rhs : tensor<1x2x1xf16>)
        outs(%e1 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    return %left, %right : tensor<1x2x1xf16>, tensor<1x2x1xf16>
  }
}
)mlir";
  for (llvm::StringRef source : {chain, branches}) {
    auto module = parse(withTopology(source.str(), 1, 2));
    ASSERT_TRUE(module);
    std::string failureReason;
    auto built = build(*module, failureReason);
    ASSERT_TRUE(built) << failureReason;
    llvm::SmallVector<wafer::test::ReferenceSpatialRoot, 2> roots;
    for (const SpatialRootDomainFacts &root :
         built->domain.getProblem().getRoots()) {
      wafer::test::ReferenceSpatialRoot reference;
      reference.root = root.root;
      reference.iteratorExtents = {1, 2, 1};
      reference.partitionableIterators = {1, 1, 1};
      reference.reductionIterators = {0, 0, 0};
      reference.resultParallelIterators = {1, 1, 1};
      roots.push_back(std::move(reference));
    }
    auto expected = wafer::test::enumerateReferenceSpatialPlans(
        roots,
        built->domain.getProblem().getStructuralProblem().getAvailableTiles());
    auto actual = enumerate(built->domain);
    ASSERT_TRUE(actual);
    EXPECT_EQ(*actual, expected);
    EXPECT_EQ(actual->size(), 16u);
  }
}

TEST_F(SpatialDomainTest, TinyDiamondProgramDomainEqualsNodeCartesianOracle) {
  // The tiny shape bounds the exact product to 4^4 points; the real graph
  // class is covered below with 1025-scale close/evaluation.
  auto module = parse(withTopology(R"mlir(
module {
  func.func @main(%input: tensor<1x2x1xf16>) -> tensor<1x2x1xf16> {
    %e0 = tensor.empty() : tensor<1x2x1xf16>
    %source = linalg.map ins(%input : tensor<1x2x1xf16>)
        outs(%e0 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<1x2x1xf16>
    %left = linalg.map ins(%source : tensor<1x2x1xf16>)
        outs(%e1 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<1x2x1xf16>
    %right = linalg.map ins(%source : tensor<1x2x1xf16>)
        outs(%e2 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    %e3 = tensor.empty() : tensor<1x2x1xf16>
    %join = linalg.generic {
        indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                         affine_map<(b, m, n) -> (b, m, n)>,
                         affine_map<(b, m, n) -> (b, m, n)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%left, %right : tensor<1x2x1xf16>, tensor<1x2x1xf16>)
        outs(%e3 : tensor<1x2x1xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %old: f16):
        %sum = arith.addf %lhs, %rhs : f16
        linalg.yield %sum : f16
    } -> tensor<1x2x1xf16>
    return %join : tensor<1x2x1xf16>
  }
}
)mlir",
                                   1, 2));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  llvm::SmallVector<wafer::test::ReferenceSpatialRoot, 4> roots;
  for (const SpatialRootDomainFacts &root :
       built->domain.getProblem().getRoots()) {
    wafer::test::ReferenceSpatialRoot reference;
    reference.root = root.root;
    reference.iteratorExtents = {1, 2, 1};
    reference.partitionableIterators = {1, 1, 1};
    reference.reductionIterators = {0, 0, 0};
    reference.resultParallelIterators = {1, 1, 1};
    roots.push_back(std::move(reference));
  }
  auto expected = wafer::test::enumerateReferenceSpatialPlans(
      roots,
      built->domain.getProblem().getStructuralProblem().getAvailableTiles());
  auto actual = enumerate(built->domain);
  ASSERT_TRUE(actual);
  EXPECT_EQ(*actual, expected);
  EXPECT_EQ(actual->size(), 256u);
}

TEST_F(SpatialDomainTest,
       RealDiamondProposalClosesAndPreservesExactDemandAcrossEdges) {
  std::string source = R"mlir(
module {
  func.func @main(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %e0 = tensor.empty() : tensor<2x1025x128xf16>
    %source = linalg.map ins(%input : tensor<2x1025x128xf16>)
        outs(%e0 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2x1025x128xf16>
    %left = linalg.map ins(%source : tensor<2x1025x128xf16>)
        outs(%e1 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2x1025x128xf16>
    %right = linalg.map ins(%source : tensor<2x1025x128xf16>)
        outs(%e2 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e3 = tensor.empty() : tensor<2x1025x128xf16>
    %join = linalg.generic {
        indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                         affine_map<(b, m, n) -> (b, m, n)>,
                         affine_map<(b, m, n) -> (b, m, n)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%left, %right : tensor<2x1025x128xf16>,
                            tensor<2x1025x128xf16>)
        outs(%e3 : tensor<2x1025x128xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %old: f16):
        %sum = arith.addf %lhs, %rhs : f16
        linalg.yield %sum : f16
    } -> tensor<2x1025x128xf16>
    return %join : tensor<2x1025x128xf16>
  }
}
)mlir";
  auto module = parse(withTopology(std::move(source), 4, 4));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  auto proposals = built->domain.getProposals();
  ASSERT_FALSE(proposals.empty());
  SpatialDomainEvaluation evaluation =
      built->domain.evaluate(built->dag, proposals.front());
  ASSERT_TRUE(evaluation.isSatisfied());
  const analysis::ExactDemandProof *proof =
      analysis::getExactDemandProof(*evaluation.demand);
  ASSERT_NE(proof, nullptr);
  EXPECT_GE(proof->dependencyDemands.size(), built->dag.getEdges().size());
  EXPECT_FALSE(proof->finalOwners.empty());
}

TEST_F(SpatialDomainTest, AttentionModesConstrainSingleAndMultipleK2Domains) {
  auto module = parse(withTopology(R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @main(%q: tensor<2x1025x128xf16>,
                  %k: tensor<2x1031x128xf16>,
                  %v: tensor<2x1031x64xf16>, %scale: f32)
      -> (tensor<2x1025x64xf16>, tensor<2x1025x64xf16>) {
    %e0 = tensor.empty() : tensor<2x1025x64xf16>
    %fa = wafer.linalg_ext.attention
        ins(%q, %k, %v, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%e0 : tensor<2x1025x64xf16>)
        algorithm(<flash_attention>) indexing_maps = [#q, #k, #v, #s, #o] score {
    ^bb0(%attention_0_dot: f16, %attention_0_scale: f32):
      %attention_0_converted = arith.extf %attention_0_dot : f16 to f32
      %attention_0_scaled = arith.mulf %attention_0_converted, %attention_0_scale : f32
      wafer.linalg_ext.attention.yield %attention_0_scaled : f32
    }
        -> tensor<2x1025x64xf16>
    %e1 = tensor.empty() : tensor<2x1025x64xf16>
    %fd = wafer.linalg_ext.attention
        ins(%q, %k, %v, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%e1 : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>) indexing_maps = [#q, #k, #v, #s, #o] score {
    ^bb0(%attention_1_dot: f16, %attention_1_scale: f32):
      %attention_1_converted = arith.extf %attention_1_dot : f16 to f32
      %attention_1_scaled = arith.mulf %attention_1_converted, %attention_1_scale : f32
      wafer.linalg_ext.attention.yield %attention_1_scaled : f32
    }
        -> tensor<2x1025x64xf16>
    return %fa, %fd : tensor<2x1025x64xf16>, tensor<2x1025x64xf16>
  }
}
)mlir",
                                   4, 4));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  ASSERT_EQ(built->domain.getProblem().getRoots().size(), 2u);
  SpatialPlan first = built->domain.getFirstPlan();
  ASSERT_TRUE(built->domain.contains(first));
  for (auto [root, node] :
       llvm::zip_equal(built->domain.getProblem().getRoots(), first.nodes)) {
    ASSERT_TRUE(root.attention);
    EXPECT_EQ(checkAttentionSpatialConstraints(*root.attention,
                                               root.iteratorExtents, node.axes),
              AttentionSpatialConstraintViolation::None);
  }

  SpatialPlan invalidFA = first;
  invalidFA.nodes[0].axes[3] = {3, IteratorPartitionScheme::BalancedParts, 2};
  invalidFA.nodes[0].embedding = {TileId(0), TileId(1)};
  invalidFA.nodes[0].reductionMerges.clear();
  EXPECT_FALSE(built->domain.contains(invalidFA));

  SpatialPlan invalidFD = first;
  invalidFD.nodes[1].axes[3] = {3, IteratorPartitionScheme::BalancedParts, 1};
  invalidFD.nodes[1].embedding = {TileId(0)};
  invalidFD.nodes[1].reductionMerges.clear();
  EXPECT_FALSE(built->domain.contains(invalidFD));

  SpatialDomainEvaluation evaluation =
      built->domain.evaluate(built->dag, first);
  EXPECT_TRUE(evaluation.isSatisfied());
  // A larger output-parallel proposal cannot remove mandatory K2 splitting.
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto *returnOp = function.getBody().front().getTerminator();
  auto fd =
      mlir::cast<LinalgExtAttentionOp>(returnOp->getOperand(1).getDefiningOp());
  mlir::OpBuilder builder(returnOp);
  auto type = mlir::cast<mlir::RankedTensorType>(fd.getResult(0).getType());
  auto empty = builder.create<mlir::tensor::EmptyOp>(
      fd.getLoc(), type.getShape(), type.getElementType());
  auto identity = builder.getMultiDimIdentityMap(type.getRank());
  auto consumer = builder.create<mlir::linalg::GenericOp>(
      fd.getLoc(), mlir::TypeRange{type}, mlir::ValueRange{fd.getResult(0)},
      mlir::ValueRange{empty},
      llvm::ArrayRef<mlir::AffineMap>{identity, identity},
      llvm::SmallVector<mlir::utils::IteratorType>(
          type.getRank(), mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &body, mlir::Location loc, mlir::ValueRange args) {
        body.create<mlir::linalg::YieldOp>(loc, args.front());
      });
  returnOp->setOperand(1, consumer.getResult(0));
  auto updated = build(*module, failureReason);
  ASSERT_TRUE(updated) << failureReason;
  auto seed = updated->domain.getFirstPlan();
  auto position = [&](mlir::Operation *op) {
    auto key = updated->domain.getProblem().getSemanticRoots().find(op)->key;
    return llvm::find_if(seed.nodes,
                         [&](const auto &node) { return node.root == key; }) -
           seed.nodes.begin();
  };
  auto fdIndex = position(fd), consumerIndex = position(consumer);
  seed.nodes[consumerIndex].axes[1].parameter = 16;
  seed.nodes[consumerIndex].embedding.clear();
  for (int64_t tile = 0; tile < 16; ++tile)
    seed.nodes[consumerIndex].embedding.push_back(TileId(tile));
  ASSERT_TRUE(updated->domain.contains(seed));
  for (auto order : {SpatialPropagationOrder::ProducersFirst,
                     SpatialPropagationOrder::ConsumersFirst}) {
    auto coordinated = propagateSpatialPartitions(updated->domain, updated->dag,
                                                  seed, {}, order);
    ASSERT_TRUE(mlir::succeeded(coordinated));
    EXPECT_EQ(coordinated->nodes[consumerIndex].axes[1].parameter, 16);
    EXPECT_GT(coordinated->nodes[fdIndex].axes[3].parameter, 1);
    EXPECT_TRUE(
        updated->domain.evaluate(updated->dag, *coordinated).isSatisfied());
  }
}

TEST_F(SpatialDomainTest,
       RankSixMaskedDecodingPartitionsAtLeastOneKeyValueIterator) {
  auto module = parse(withTopology(R"mlir(
#q = affine_map<(b, m, k1, k2a, k2b, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2a, k2b, n) -> (b, k2a, k2b, k1)>
#v = affine_map<(b, m, k1, k2a, k2b, n) -> (b, k2a, k2b, n)>
#s = affine_map<(b, m, k1, k2a, k2b, n) -> ()>
#mask = affine_map<(b, m, k1, k2a, k2b, n) -> (m, k2a, k2b)>
#o = affine_map<(b, m, k1, k2a, k2b, n) -> (b, m, n)>
module {
  func.func @main(%q: tensor<2x1025x128xf16>,
                  %k: tensor<2x33x31x128xf16>,
                  %v: tensor<2x33x31x64xf16>, %scale: f32,
                  %mask: tensor<1025x33x31xf16>)
      -> tensor<2x1025x64xf16> {
    %empty = tensor.empty() : tensor<2x1025x64xf16>
    %result = wafer.linalg_ext.attention
        ins(%q, %k, %v, %scale, %mask : tensor<2x1025x128xf16>,
            tensor<2x33x31x128xf16>, tensor<2x33x31x64xf16>, f32,
            tensor<1025x33x31xf16>)
        outs(%empty : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>)
        indexing_maps = [#q, #k, #v, #s, #mask, #o] score {
    ^bb0(%attention_2_dot: f16, %attention_2_scale: f32, %attention_2_mask: f16):
      %attention_2_converted = arith.extf %attention_2_dot : f16 to f32
      %attention_2_scaled = arith.mulf %attention_2_converted, %attention_2_scale : f32
      %attention_2_converted_mask = arith.extf %attention_2_mask : f16 to f32
      %attention_2_masked = arith.addf %attention_2_scaled, %attention_2_converted_mask : f32
      wafer.linalg_ext.attention.yield %attention_2_masked : f32
    }
        -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}
)mlir",
                                   4, 4));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  const SpatialRootDomainFacts root =
      built->domain.getProblem().getRoots().front();
  ASSERT_TRUE(root.attention);
  ASSERT_EQ(root.attention->keyValueReductionIterators.size(), 2u);
  SpatialPlan first = built->domain.getFirstPlan();
  ASSERT_TRUE(built->domain.contains(first));
  int64_t keyValueCells = 1;
  for (unsigned iterator : root.attention->keyValueReductionIterators) {
    auto count = getIteratorPartitionIntervalCount(
        root.iteratorExtents[iterator], first.nodes.front().axes[iterator]);
    ASSERT_TRUE(mlir::succeeded(count));
    keyValueCells *= *count;
  }
  EXPECT_GT(keyValueCells, 1);
  EXPECT_TRUE(built->domain.evaluate(built->dag, first).isSatisfied());

  SpatialPlan invalid = first;
  for (unsigned iterator : root.attention->keyValueReductionIterators)
    invalid.nodes.front().axes[iterator] = {
        iterator, IteratorPartitionScheme::BalancedParts, 1};
  invalid.nodes.front().embedding = {TileId(0)};
  invalid.nodes.front().reductionMerges.clear();
  EXPECT_FALSE(built->domain.contains(invalid));
}

TEST_F(SpatialDomainTest,
       MultiResultPartialReductionCreatesOneMergePerResultAndParallelCell) {
  auto module = parse(withTopology(R"mlir(
#input = affine_map<(b, k, n) -> (b, k, n)>
#sum_output = affine_map<(b, k, n) -> (b, n)>
#max_output = affine_map<(b, k, n) -> (n)>
module {
  func.func @main(%input: tensor<2x1024x128xf16>,
                  %sum: tensor<2x128xf16>, %max: tensor<128xf16>)
      -> (tensor<2x128xf16>, tensor<128xf16>) {
    %results:2 = linalg.generic {
        indexing_maps = [#input, #sum_output, #max_output],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%input : tensor<2x1024x128xf16>)
        outs(%sum, %max : tensor<2x128xf16>, tensor<128xf16>) {
      ^bb0(%value: f16, %sum_acc: f16, %max_acc: f16):
        %next_sum = arith.addf %value, %sum_acc : f16
        %next_max = arith.maximumf %value, %max_acc : f16
        linalg.yield %next_sum, %next_max : f16, f16
    } -> (tensor<2x128xf16>, tensor<128xf16>)
    return %results#0, %results#1 : tensor<2x128xf16>, tensor<128xf16>
  }
}
)mlir",
                                   4, 4));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  const SpatialRootDomainFacts root =
      built->domain.getProblem().getRoots().front();
  EXPECT_EQ(root.reductionResultGroupCount, 2u);
  ASSERT_EQ(root.resultParallelIteratorsByGroup.size(), 2u);
  EXPECT_TRUE(root.resultParallelIteratorsByGroup[0].test(0));
  EXPECT_TRUE(root.resultParallelIteratorsByGroup[0].test(2));
  EXPECT_FALSE(root.resultParallelIteratorsByGroup[1].test(0));
  EXPECT_TRUE(root.resultParallelIteratorsByGroup[1].test(2));
  EXPECT_FALSE(root.partitionableParallelIterators.test(0));
  EXPECT_TRUE(root.partitionableParallelIterators.test(2));
  llvm::SmallVector<IteratorPartition, 4> axes = {
      {0, IteratorPartitionScheme::BalancedParts, 1},
      {1, IteratorPartitionScheme::BalancedParts, 2},
      {2, IteratorPartitionScheme::BalancedParts, 1}};
  auto groups = deriveSpatialReductionGroups(root, axes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(groups)) << failureReason;
  ASSERT_EQ(groups->size(), 2u);
  EXPECT_EQ((*groups)[0].resultGroup, 0u);
  EXPECT_EQ((*groups)[1].resultGroup, 1u);
  EXPECT_EQ((*groups)[1].parallelCoordinate,
            (llvm::SmallVector<uint32_t, 4>{0}));
}

TEST_F(SpatialDomainTest,
       TwoRaggedReductionAxesFormOneCompleteContributionFiber) {
  auto module = parse(withTopology(R"mlir(
#input = affine_map<(b, k0, k1, n) -> (b, k0, k1, n)>
#output = affine_map<(b, k0, k1, n) -> (b, n)>
module {
  func.func @main(%input: tensor<2x1025x1031x128xf16>,
                  %init: tensor<2x128xf16>) -> tensor<2x128xf16> {
    %result = linalg.generic {
        indexing_maps = [#input, #output],
        iterator_types = ["parallel", "reduction", "reduction", "parallel"]}
        ins(%input : tensor<2x1025x1031x128xf16>)
        outs(%init : tensor<2x128xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<2x128xf16>
    return %result : tensor<2x128xf16>
  }
}
)mlir",
                                   4, 4));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  const SpatialRootDomainFacts root =
      built->domain.getProblem().getRoots().front();
  ASSERT_TRUE(root.partitionableReductionIterators.test(1));
  ASSERT_TRUE(root.partitionableReductionIterators.test(2));
  llvm::SmallVector<IteratorPartition, 4> axes = {
      {0, IteratorPartitionScheme::BalancedParts, 1},
      {1, IteratorPartitionScheme::BalancedParts, 2},
      {2, IteratorPartitionScheme::BalancedParts, 2},
      {3, IteratorPartitionScheme::BalancedParts, 1}};
  auto groups = deriveSpatialReductionGroups(root, axes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(groups)) << failureReason;
  ASSERT_EQ(groups->size(), 1u);
  SpatialPlan plan = built->domain.getFirstPlan();
  plan.nodes.front().axes = axes;
  plan.nodes.front().embedding = {TileId(0), TileId(1), TileId(4), TileId(5)};
  plan.nodes.front().reductionMerges = {{groups->front(), TileId(15)}};
  EXPECT_TRUE(built->domain.contains(plan));
  auto assignment = built->domain.close(plan, &failureReason);
  ASSERT_TRUE(mlir::succeeded(assignment)) << failureReason;
  EXPECT_EQ(assignment->nodes.front().shards.size(), 4u);
}

TEST_F(SpatialDomainTest,
       IndependentComponentsReceiveDisjointProposalWithoutPruningRawSet) {
  auto module = parse(withTopology(R"mlir(
module {
  func.func @main(%lhs: tensor<2x1024x128xf16>,
                  %rhs: tensor<2x1024x128xf16>)
      -> (tensor<2x1024x128xf16>, tensor<2x1024x128xf16>) {
    %e0 = tensor.empty() : tensor<2x1024x128xf16>
    %left = linalg.map ins(%lhs : tensor<2x1024x128xf16>)
        outs(%e0 : tensor<2x1024x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2x1024x128xf16>
    %right = linalg.map ins(%rhs : tensor<2x1024x128xf16>)
        outs(%e1 : tensor<2x1024x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    return %left, %right : tensor<2x1024x128xf16>,
                           tensor<2x1024x128xf16>
  }
}
)mlir",
                                   4, 4));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto built = build(*module, failureReason);
  ASSERT_TRUE(built) << failureReason;
  ASSERT_EQ(built->domain.getProblem().getComponents().size(), 2u);
  auto proposals = built->domain.getProposals();
  ASSERT_GE(proposals.size(), 2u);
  bool foundDisjoint = false;
  for (const SpatialPlan &proposal : proposals) {
    ASSERT_TRUE(built->domain.contains(proposal));
    foundDisjoint |=
        llvm::none_of(proposal.nodes[0].embedding, [&](TileId tile) {
          return llvm::is_contained(proposal.nodes[1].embedding, tile);
        });
  }
  EXPECT_TRUE(foundDisjoint);

  SpatialPlan overlapping = proposals.front();
  overlapping.nodes[1] = overlapping.nodes[0];
  overlapping.nodes[1].root = built->domain.getProblem().getRoots()[1].root;
  EXPECT_TRUE(built->domain.contains(overlapping))
      << "component-disjoint proposal must not remove overlapping raw siblings";
}

TEST_F(SpatialDomainTest,
       OperandReuseProposalsFollowMapsAndKeepOriginalPartitions) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (int64_t tiles : {4, 16}) {
      for (unsigned shape : {0u, 1u, 2u}) {
        for (bool permuted : {false, true}) {
          SCOPED_TRACE(std::to_string(extent) + ":" + std::to_string(tiles) +
                       ":" + std::to_string(shape) + ":" +
                       std::to_string(permuted));
          const int64_t m = shape == 0 ? 16 : extent;
          const int64_t n = shape == 1 ? 16 : extent;
          // The second batch coordinate also covers rank-four operands;
          // operand permutation must not turn an input-reuse rule into an
          // implicit M/N spelling or dimension-order rule.
          auto type = [&](int64_t first, int64_t second) {
            return "tensor<1x" + std::string(permuted ? "1x" : "") +
                   std::to_string(first) + "x" + std::to_string(second) +
                   "xf16>";
          };
          const std::string lhs = permuted ? type(128, m) : type(m, 128);
          const std::string rhs = permuted ? type(128, n) : type(n, 128);
          const std::string output = type(m, n);
          const std::string batch = permuted ? "b, h, " : "b, ";
          const std::string domain = batch + "m, n, k";
          auto map = [&](llvm::StringRef axes) {
            return "affine_map<(" + domain + ") -> (" + batch + axes.str() +
                   ")>";
          };
          std::string source;
          llvm::raw_string_ostream stream(source);
          stream
              << "module { func.func @main(%lhs: " << lhs << ", %rhs: " << rhs
              << ", %init: " << output << ") -> " << output
              << " {\n%result = linalg.generic {indexing_maps = ["
              << map(permuted ? "k, m" : "m, k") << ", "
              << map(permuted ? "k, n" : "n, k") << ", " << map("m, n")
              << "], iterator_types = [\"parallel\", "
              << (permuted ? "\"parallel\", " : "")
              << "\"parallel\", \"parallel\", \"reduction\"]} ins(%lhs, %rhs : "
              << lhs << ", " << rhs << ") outs(%init : " << output
              << ") { ^bb0(%a: f16, %b: f16, %old: f16):\n"
                 "%mul = arith.mulf %a, %b : f16\n"
                 "%sum = arith.addf %mul, %old : f16\n"
                 "linalg.yield %sum : f16\n} -> "
              << output << "\nreturn %result : " << output << "\n}}";
          auto module = parse(withTopology(stream.str(), 2, tiles / 2));
          ASSERT_TRUE(module);
          const std::string before = print(module->getOperation());
          std::string detail;
          auto built = build(*module, detail);
          ASSERT_TRUE(built) << detail;
          auto proposals = built->domain.getProposals();
          ASSERT_FALSE(proposals.empty());
          const unsigned mAxis = permuted ? 2 : 1;
          const unsigned nAxis = mAxis + 1;
          const auto &first = proposals.front().nodes.front();
          EXPECT_EQ(first.embedding.size(), static_cast<size_t>(tiles));
          const int64_t balanced = tiles == 4 ? 2 : 4;
          EXPECT_EQ(first.axes[mAxis].parameter, shape == 0   ? 1
                                                 : shape == 1 ? tiles
                                                              : balanced);
          EXPECT_EQ(first.axes[nAxis].parameter, shape == 1   ? 1
                                                 : shape == 0 ? tiles
                                                              : balanced);
          EXPECT_TRUE(llvm::any_of(proposals, [&](const SpatialPlan &plan) {
            const auto &node = plan.nodes.front();
            return node.axes[mAxis].parameter == tiles &&
                   node.axes[nAxis].parameter == 1;
          }));
          for (const SpatialPlan &proposal : proposals)
            EXPECT_TRUE(built->domain.contains(proposal));
          auto assignment = built->domain.close(proposals.front(), &detail);
          ASSERT_TRUE(mlir::succeeded(assignment)) << detail;
          expectExactShardCoverage(
              assignment->nodes.front(),
              built->domain.getProblem().getRoots().front().iteratorExtents);
          auto evaluation =
              built->domain.evaluate(built->dag, proposals.front());
          ASSERT_TRUE(evaluation.isSatisfied());
          const auto *proof = analysis::getExactDemandProof(*evaluation.demand);
          ASSERT_NE(proof, nullptr);
          auto works = RootRegionWorkAnalysis::create(
              built->dag, *evaluation.assignment, *proof, &detail);
          ASSERT_TRUE(mlir::succeeded(works)) << detail;
          EXPECT_EQ(print(module->getOperation()), before);
        }
      }
    }
  }
}

TEST_F(SpatialDomainTest, NonProjectedInputKeepsConstructiveProposalOrder) {
  auto module = parse(withTopology(R"mlir(
module {
  func.func @main(%input: tensor<1x1151x128xf16>) -> tensor<1x1024x128xf16> {
    %empty = tensor.empty() : tensor<1x1024x128xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(b, m, n) -> (b, m + n, n)>,
                         affine_map<(b, m, n) -> (b, m, n)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<1x1151x128xf16>)
        outs(%empty : tensor<1x1024x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<1x1024x128xf16>
    return %result : tensor<1x1024x128xf16>
  }
})mlir",
                                   4, 4));
  ASSERT_TRUE(module);
  std::string detail;
  auto built = build(*module, detail);
  ASSERT_TRUE(built) << detail;
  auto proposals = built->domain.getProposals();
  ASSERT_FALSE(proposals.empty());
  const auto &first = proposals.front().nodes.front();
  EXPECT_EQ(first.axes[1].parameter, 16);
  EXPECT_EQ(first.axes[2].parameter, 1);
  EXPECT_TRUE(built->domain.contains(proposals.front()));
}

TEST_F(SpatialDomainTest,
       PartitionsFollowActualViewsAndCoordinateInitializers) {
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t tiles : {4, 16})
      for (int64_t inner : {1, 128})
        for (bool sharedInit : {false, true}) {
          SCOPED_TRACE(std::to_string(extent) + ":" + std::to_string(tiles) +
                       ":" + std::to_string(sharedInit) + ":" +
                       std::to_string(inner));
          const bool rectangular = inner == 1 || extent % tiles == 0;
          std::string source;
          llvm::raw_string_ostream out(source);
          const std::string tensor =
              "tensor<1x16x" + std::to_string(extent * inner) + "xf16>";
          const std::string view = "tensor<1x16x" + std::to_string(extent) +
                                   "x" + std::to_string(inner) + "xf16>";
          const std::string transposed = "tensor<1x" + std::to_string(extent) +
                                         "x16x" + std::to_string(inner) +
                                         "xf16>";
          const std::string rhs =
              "tensor<1x128x" + std::to_string(extent * inner) + "xf16>";
          out << "module { func.func @main(%lhs: tensor<1x16x128xf16>, %rhs: "
              << rhs << ") -> (" << transposed
              << (sharedInit ? ", " + tensor : "") << ") {\n"
              << "%zero = arith.constant 0.0 : f16\n"
              << "%empty = tensor.empty() : " << tensor << "\n"
              << "%init = linalg.fill ins(%zero : f16) outs(%empty : " << tensor
              << ") -> " << tensor << "\n"
              << "%gemm = linalg.batch_matmul ins(%lhs, %rhs : "
                 "tensor<1x16x128xf16>, "
              << rhs << ") outs(%init : " << tensor << ") -> " << tensor << "\n"
              << "%view = tensor.expand_shape %gemm [[0], [1], [2, 3]] "
                 "output_shape [1, 16, "
              << extent << ", " << inner << "] : " << tensor << " into " << view
              << "\n"
              << "%pe = tensor.empty() : " << view << "\n"
              << "%point = linalg.map ins(%view : " << view
              << ") outs(%pe : " << view
              << ") (%x: f16) { %y = arith.addf %x, %x : f16\nlinalg.yield %y "
                 ": "
                 "f16 }\n"
              << "%te = tensor.empty() : " << transposed << "\n"
              << "%trans = linalg.transpose ins(%point : " << view
              << ") outs(%te : " << transposed
              << ") permutation = [0, 2, 1, 3]\n";
          if (sharedInit)
            out << "%other = linalg.map ins(%init : " << tensor
                << ") outs(%empty : " << tensor
                << ") (%x: f16) { linalg.yield %x : f16 }\n";
          out << "return %trans" << (sharedInit ? ", %other" : "") << " : "
              << transposed << (sharedInit ? ", " + tensor : "") << "\n}}";
          auto module = parse(withTopology(out.str(), 2, tiles / 2));
          ASSERT_TRUE(module);
          const auto before = print(module->getOperation());
          std::string detail;
          auto built = build(*module, detail);
          ASSERT_TRUE(built) << detail;
          auto raw = built->domain.getProposals();
          ASSERT_FALSE(raw.empty());
          auto propagated = propagateSpatialPartitions(
              built->domain, built->dag, raw.front(), {});
          ASSERT_TRUE(mlir::succeeded(propagated));
          EXPECT_TRUE(built->domain.contains(*propagated));
          analysis::IndexRelationLimits limited;
          limited.maxVariables = 1;
          auto unknown = propagateSpatialPartitions(built->domain, built->dag,
                                                    raw.front(), limited);
          ASSERT_TRUE(mlir::succeeded(unknown));
          EXPECT_EQ(*unknown, raw.front());
          for (const auto &node : built->dag.getNodes()) {
            auto binding = built->domain.getProblem().getSemanticRoots().find(
                node.operation);
            ASSERT_NE(binding, nullptr);
            auto partition =
                llvm::find_if(propagated->nodes, [&](const auto &p) {
                  return p.root == binding->key;
                });
            ASSERT_NE(partition, propagated->nodes.end());
            if (mlir::isa<mlir::linalg::MapOp>(node.operation) &&
                mlir::cast<mlir::RankedTensorType>(
                    node.operation->getResult(0).getType())
                        .getRank() == 4) {
              EXPECT_EQ(partition->axes[1].parameter, rectangular ? 1 : tiles);
              EXPECT_EQ(partition->axes[2].parameter, rectangular ? tiles : 1);
            }
            if (mlir::isa<mlir::linalg::FillOp>(node.operation)) {
              // When the N split cannot cross a ragged reshape, backward
              // demand now coordinates GEMM and its init along consumer rows.
              const bool rows = sharedInit || !rectangular;
              EXPECT_EQ(partition->axes[1].parameter, rows ? tiles : 1);
              EXPECT_EQ(partition->axes[2].parameter, rows ? 1 : tiles);
            }
          }
          auto evaluation = built->domain.evaluate(built->dag, *propagated);
          ASSERT_TRUE(evaluation.isSatisfied());
          const auto *proof = analysis::getExactDemandProof(*evaluation.demand);
          ASSERT_NE(proof, nullptr);
          for (const auto &dependency : proof->dependencyDemands)
            for (const auto &destination : dependency.perDestination)
              for (const auto &sourceDemand : destination.sources)
                if (!sharedInit) {
                  for (const auto &owner : sourceDemand.eligibleFinalOwners)
                    EXPECT_EQ(owner.tile, destination.destinationTile);
                }
          for (const auto &node : evaluation.assignment->nodes) {
            const auto *facts = built->domain.getProblem().findRoot(node.root);
            ASSERT_NE(facts, nullptr);
            expectExactShardCoverage(node, facts->iteratorExtents);
          }
          auto proposals = built->domain.getGraphCoherentProposals(built->dag);
          ASSERT_TRUE(mlir::succeeded(proposals));
          for (const auto &seed : raw)
            EXPECT_TRUE(llvm::is_contained(*proposals, seed));
          EXPECT_EQ(print(module->getOperation()), before);
        }
}

TEST_F(SpatialDomainTest,
       GraphCoherentProposalKeepsMultiProducerClosureInTheRawDomain) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto parsed =
        wafer::compiler::testing::parseMultiProducerJoinProgram(extent);
    ASSERT_TRUE(parsed.module);
    const std::string before = print(parsed.module->getOperation());
    std::string failureReason;
    auto built = build(*parsed.module, failureReason);
    ASSERT_TRUE(built) << failureReason;
    llvm::SmallVector<SpatialPlan, 4> raw = built->domain.getProposals();
    ASSERT_FALSE(raw.empty());
    ASSERT_TRUE(llvm::any_of(raw, [](const SpatialPlan &plan) {
      return llvm::any_of(plan.nodes, [](const NodeSpatialPlan &node) {
        return node.embedding.size() > 1;
      });
    }));

    auto coherent = built->domain.getGraphCoherentProposals(built->dag);
    ASSERT_TRUE(mlir::succeeded(coherent));
    ASSERT_FALSE(coherent->empty());
    const SpatialPlan &first = coherent->front();
    ASSERT_TRUE(built->domain.contains(first));
    ASSERT_EQ(first.nodes.size(), 3u);
    ASSERT_EQ(first.nodes.front().embedding.size(), 16u);
    for (const NodeSpatialPlan &node : first.nodes) {
      EXPECT_EQ(node.embedding.size(), 16u);
      std::set<int64_t> tileIds;
      for (TileId tile : node.embedding)
        tileIds.insert(tile.getValue());
      EXPECT_EQ(tileIds.size(), node.embedding.size());
    }
    SpatialDomainEvaluation evaluation =
        built->domain.evaluate(built->dag, first);
    EXPECT_TRUE(evaluation.isSatisfied());
    for (const SpatialPlan &rawPlan : raw) {
      EXPECT_TRUE(built->domain.contains(rawPlan));
      EXPECT_TRUE(llvm::is_contained(*coherent, rawPlan));
    }
    EXPECT_EQ(print(parsed.module->getOperation()), before);
  }
}

TEST_F(SpatialDomainTest,
       NonProjectedResultMapStaysInDomainAsOneUnpartitionedCell) {
  // linalg permits a result indexing map that is not a projected permutation;
  // its verifier only requires symbol-free maps with consistent shapes. Such a
  // root cannot prove which result elements a parallel partition owns, so it
  // must stay in the domain as one unpartitioned cell instead of failing the
  // complete spatial domain. The pairs cover 1024 and the non-divisible 1025.
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "module {\n"
           << "  func.func @main(%input: tensor<" << extent
           << "x4x4xf16>, %init: tensor<" << extent << "x7xf16>) -> tensor<"
           << extent << "x7xf16> {\n"
           << "    %result = linalg.generic {\n"
           << "        indexing_maps = [affine_map<(m, n, k) -> (m, n, k)>,\n"
           << "                         affine_map<(m, n, k) -> (m, n + k)>],\n"
           << "        iterator_types = [\"parallel\", \"parallel\", "
              "\"parallel\"]}\n"
           << "        ins(%input : tensor<" << extent << "x4x4xf16>)\n"
           << "        outs(%init : tensor<" << extent << "x7xf16>) {\n"
           << "      ^bb0(%value: f16, %acc: f16):\n"
           << "        %next = arith.addf %value, %acc : f16\n"
           << "        linalg.yield %next : f16\n"
           << "    } -> tensor<" << extent << "x7xf16>\n"
           << "    return %result : tensor<" << extent << "x7xf16>\n"
           << "  }\n}\n";
    auto module = parse(withTopology(stream.str(), 4, 4));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto built = build(*module, failureReason);
    ASSERT_TRUE(built) << failureReason;

    const SpatialRootDomainFacts root =
        built->domain.getProblem().getRoots().front();
    EXPECT_FALSE(root.partitionableParallelIterators.any())
        << "a non-projected result map must not authorize a parallel split";
    EXPECT_FALSE(root.partitionableReductionIterators.any());

    SpatialPlan plan = built->domain.getFirstPlan();
    ASSERT_TRUE(built->domain.contains(plan));
    ASSERT_EQ(plan.nodes.size(), 1u);
    EXPECT_EQ(plan.nodes.front().embedding.size(), 1u);
    for (const IteratorPartition &axis : plan.nodes.front().axes)
      EXPECT_EQ(axis.parameter, 1);

    auto assignment = built->domain.close(plan, &failureReason);
    ASSERT_TRUE(mlir::succeeded(assignment)) << failureReason;
    ASSERT_EQ(assignment->nodes.front().shards.size(), 1u);
    expectExactShardCoverage(assignment->nodes.front(), root.iteratorExtents);

    SpatialDomainEvaluation evaluation =
        built->domain.evaluate(built->dag, plan);
    ASSERT_TRUE(evaluation.isSatisfied());
    EXPECT_NE(analysis::getExactDemandProof(*evaluation.demand), nullptr);
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

size_t spatialNodeIndex(const SpatialPlanDomain &domain,
                        const SpatialPlan &plan, mlir::Operation *operation) {
  const auto *root = domain.getProblem().getSemanticRoots().find(operation);
  assert(root);
  auto node = llvm::find_if(
      plan.nodes, [&](const auto &value) { return value.root == root->key; });
  assert(node != plan.nodes.end());
  return std::distance(plan.nodes.begin(), node);
}

TEST_F(SpatialDomainTest, RelationsCoordinateContractionsInBothDirections) {
  for (bool conv : {false, true})
    for (bool generic : {false, true})
      for (int64_t extent : {1024, 1025, 1031})
        for (int64_t tiles : {4, 16})
          for (bool reduction : {false, true})
            for (auto order : {SpatialPropagationOrder::ProducersFirst,
                               SpatialPropagationOrder::ConsumersFirst}) {
              SCOPED_TRACE(
                  std::to_string(conv) + ":" + std::to_string(generic) + ":" +
                  std::to_string(extent) + ":" + std::to_string(tiles) + ":" +
                  std::to_string(reduction) + ":" + std::to_string(int(order)));
              auto module =
                  parse(wafer::compiler::testing::spatialContractionSource(
                      extent, conv, generic, tiles));
              ASSERT_TRUE(module);
              const auto before = print(module->getOperation());
              std::string detail;
              auto built = build(*module, detail);
              ASSERT_TRUE(built) << detail;
              SpatialPlan seed = built->domain.getFirstPlan();
              ASSERT_EQ(seed.nodes.size(), 3u);
              auto *consumerOp = built->dag.getFunction()
                                     .getBody()
                                     .front()
                                     .getTerminator()
                                     ->getOperand(0)
                                     .getDefiningOp();
              size_t consumerIndex =
                  spatialNodeIndex(built->domain, seed, consumerOp);
              size_t producerIndex =
                  spatialNodeIndex(built->domain, seed,
                                   consumerOp->getOperand(0).getDefiningOp());
              const bool forward =
                  order == SpatialPropagationOrder::ProducersFirst;
              size_t target = forward ? producerIndex : consumerIndex;
              unsigned axis = reduction ? (forward ? 2 : (conv ? 4 : 3)) : 1;
              seed.nodes[target].axes[axis].parameter = tiles;
              seed.nodes[target].embedding.clear();
              for (int64_t tile = 0; tile < tiles; ++tile)
                seed.nodes[target].embedding.push_back(TileId(tile));
              if (!forward && reduction) {
                auto groups = deriveSpatialReductionGroups(
                    built->domain.getProblem().getRoots()[consumerIndex],
                    seed.nodes[consumerIndex].axes);
                ASSERT_TRUE(mlir::succeeded(groups));
                for (const auto &group : *groups)
                  seed.nodes[consumerIndex].reductionMerges.push_back(
                      {group, TileId(tiles - 1)});
              }
              ASSERT_TRUE(built->domain.contains(seed));
              auto coordinated = propagateSpatialPartitions(
                  built->domain, built->dag, seed, {}, order);
              ASSERT_TRUE(mlir::succeeded(coordinated));
              ASSERT_TRUE(built->domain.contains(*coordinated));
              EXPECT_EQ(coordinated->nodes[producerIndex]
                            .axes[reduction ? 2 : 1]
                            .parameter,
                        tiles);
              const auto &consumer = coordinated->nodes[consumerIndex];
              EXPECT_EQ(consumer.axes[reduction ? (conv ? 4 : 3) : 1].parameter,
                        tiles);
              EXPECT_EQ(consumer.embedding.size(), size_t(tiles));
              ASSERT_EQ(consumer.reductionMerges.size(), reduction ? 1u : 0u);
              if (reduction) {
                EXPECT_EQ(consumer.reductionMerges.front().tile,
                          TileId(forward ? 0 : tiles - 1));
              }
              auto evaluated = built->domain.evaluate(built->dag, *coordinated);
              ASSERT_TRUE(evaluated.isSatisfied());
              for (const auto &node : evaluated.assignment->nodes)
                expectExactShardCoverage(node, built->domain.getProblem()
                                                   .findRoot(node.root)
                                                   ->iteratorExtents);
              const auto *proof =
                  analysis::getExactDemandProof(*evaluated.demand);
              ASSERT_NE(proof, nullptr);
              for (const auto &dependency : proof->dependencyDemands) {
                if (dependency.consumer != consumer.root ||
                    dependency.kind != analysis::DemandOperandKind::InitInput)
                  continue;
                ASSERT_EQ(dependency.perDestination.size(),
                          reduction ? 1u : size_t(tiles));
                if (reduction) {
                  EXPECT_TRUE(std::holds_alternative<ReductionGroupId>(
                      dependency.perDestination.front().destination));
                  EXPECT_EQ(dependency.perDestination.front().destinationTile,
                            consumer.reductionMerges.front().tile);
                }
              }
              if (reduction) {
                ASSERT_EQ(proof->reductionMerges.size(), 1u);
                EXPECT_EQ(proof->reductionMerges.front().contributions.size(),
                          size_t(tiles));
              }
              auto work = RootRegionWorkAnalysis::create(
                  built->dag, *evaluated.assignment, *proof, &detail);
              ASSERT_TRUE(mlir::succeeded(work)) << detail;
              EXPECT_EQ(print(module->getOperation()), before);
            }
}

TEST_F(SpatialDomainTest,
       ReductionProducerPreservesIndependentConsumerOutputAxis) {
  for (int64_t extent : {1024, 1025, 1031})
    for (auto order : {SpatialPropagationOrder::ProducersFirst,
                       SpatialPropagationOrder::ConsumersFirst}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(static_cast<unsigned>(order));
      std::string text;
      llvm::raw_string_ostream ir(text);
      ir << R"mlir(module {
        func.func @main(%input: tensor<1x1024x128xf16>, %weight: tensor<1x)mlir"
         << extent << R"mlir(x1024xf16>) -> tensor<1x)mlir" << extent
         << R"mlir(xf16> {
          %zero = arith.constant 0.0 : f16
          %pe = tensor.empty() : tensor<1x1024xf16>
          %pi = linalg.fill ins(%zero : f16) outs(%pe : tensor<1x1024xf16>) -> tensor<1x1024xf16>
          %p = linalg.generic {indexing_maps = [affine_map<(b,k,r)->(b,k,r)>, affine_map<(b,k,r)->(b,k)>], iterator_types = ["parallel", "parallel", "reduction"]}
              ins(%input : tensor<1x1024x128xf16>) outs(%pi : tensor<1x1024xf16>) {
          ^bb0(%x: f16, %acc: f16):
            %sum = arith.addf %x, %acc : f16
            linalg.yield %sum : f16
          } -> tensor<1x1024xf16>
          %ce = tensor.empty() : tensor<1x)mlir"
         << extent << R"mlir(xf16>
          %ci = linalg.fill ins(%zero : f16) outs(%ce : tensor<1x)mlir"
         << extent << R"mlir(xf16>) -> tensor<1x)mlir" << extent << R"mlir(xf16>
          %c = linalg.generic {indexing_maps = [affine_map<(b,n,k)->(b,k)>, affine_map<(b,n,k)->(b,n,k)>, affine_map<(b,n,k)->(b,n)>], iterator_types = ["parallel", "parallel", "reduction"]}
              ins(%p, %weight : tensor<1x1024xf16>, tensor<1x)mlir"
         << extent << R"mlir(x1024xf16>) outs(%ci : tensor<1x)mlir" << extent
         << R"mlir(xf16>) {
          ^bb0(%a: f16, %w: f16, %acc: f16):
            %product = arith.mulf %a, %w : f16
            %sum = arith.addf %product, %acc : f16
            linalg.yield %sum : f16
          } -> tensor<1x)mlir"
         << extent << R"mlir(xf16>
          return %c : tensor<1x)mlir"
         << extent << R"mlir(xf16>
        }
      })mlir";
      auto module = parse(withTopology(text, 4, 4));
      ASSERT_TRUE(module);
      std::string detail;
      auto built = build(*module, detail);
      ASSERT_TRUE(built) << detail;
      auto function = *module->getOps<mlir::func::FuncOp>().begin();
      auto *consumer = function.getBody()
                           .front()
                           .getTerminator()
                           ->getOperand(0)
                           .getDefiningOp();
      auto *producer = consumer->getOperand(0).getDefiningOp();
      auto seed = built->domain.getFirstPlan();
      size_t producerIndex = spatialNodeIndex(built->domain, seed, producer);
      size_t consumerIndex = spatialNodeIndex(built->domain, seed, consumer);
      seed.nodes[producerIndex].axes[1].parameter = 4;
      seed.nodes[producerIndex].embedding = {TileId(0), TileId(4), TileId(8),
                                             TileId(12)};
      seed.nodes[consumerIndex].axes[1].parameter = 16;
      seed.nodes[consumerIndex].embedding.clear();
      for (int64_t tile = 0; tile < 16; ++tile)
        seed.nodes[consumerIndex].embedding.push_back(TileId(tile));
      ASSERT_TRUE(built->domain.contains(seed));
      auto coordinated = propagateSpatialPartitions(built->domain, built->dag,
                                                    seed, {}, order);
      ASSERT_TRUE(mlir::succeeded(coordinated));
      EXPECT_EQ(coordinated->nodes[consumerIndex].axes[1].parameter, 16);
      EXPECT_EQ(coordinated->nodes[consumerIndex].axes[2].parameter, 1);
      EXPECT_TRUE(coordinated->nodes[consumerIndex].reductionMerges.empty());
      auto evaluation = built->domain.evaluate(built->dag, *coordinated);
      ASSERT_TRUE(evaluation.isSatisfied());
      for (auto &node : evaluation.assignment->nodes)
        expectExactShardCoverage(
            node,
            built->domain.getProblem().findRoot(node.root)->iteratorExtents);
      EXPECT_TRUE(built->domain.contains(seed));
    }
}

TEST_F(SpatialDomainTest, RepeatedDemandRetainsIndependentConsumerPartitions) {
  for (int64_t extent : {1024, 1025, 1031}) {
    auto module = parse(wafer::compiler::testing::spatialContractionSource(
        extent, false, false, 16));
    ASSERT_TRUE(module);
    std::string detail;
    auto built = build(*module, detail);
    ASSERT_TRUE(built) << detail;
    auto seed = built->domain.getFirstPlan();
    auto *consumerOp = built->dag.getFunction()
                           .getBody()
                           .front()
                           .getTerminator()
                           ->getOperand(0)
                           .getDefiningOp();
    size_t consumerIndex = spatialNodeIndex(built->domain, seed, consumerOp);
    size_t producerIndex = spatialNodeIndex(
        built->domain, seed, consumerOp->getOperand(0).getDefiningOp());
    auto &gemm = seed.nodes[consumerIndex];
    gemm.axes[1].parameter = 4;
    gemm.axes[2].parameter = 4;
    gemm.embedding.clear();
    for (int64_t tile = 0; tile < 16; ++tile)
      gemm.embedding.push_back(TileId(tile));
    auto coordinated =
        propagateSpatialPartitions(built->domain, built->dag, seed, {},
                                   SpatialPropagationOrder::ConsumersFirst);
    ASSERT_TRUE(mlir::succeeded(coordinated));
    EXPECT_EQ(coordinated->nodes[producerIndex].embedding.size(), 4u);
    EXPECT_EQ(coordinated->nodes[producerIndex].axes[1].parameter, 4);
    EXPECT_EQ(coordinated->nodes[consumerIndex].axes[1].parameter, 4);
    EXPECT_EQ(coordinated->nodes[consumerIndex].axes[2].parameter, 4);
    EXPECT_EQ(coordinated->nodes[consumerIndex].embedding.size(), 16u);
    EXPECT_TRUE(built->domain.evaluate(built->dag, *coordinated).isSatisfied());
  }
}

TEST_F(SpatialDomainTest,
       AffineReverseHasExactButUnrepresentableRaggedPartition) {
  for (int64_t extent : {1024, 1025}) {
    std::string text;
    llvm::raw_string_ostream os(text);
    const std::string type = "tensor<1x" + std::to_string(extent) + "x128xf16>";
    os << "module { func.func @main(%input: " << type << ") -> " << type
       << " {\n"
       << "%e = tensor.empty() : " << type << "\n"
       << "%p = linalg.map ins(%input : " << type << ") outs(%e : " << type
       << ") (%x: f16) { linalg.yield %x : f16 }\n"
       << "%r = linalg.generic {indexing_maps = [affine_map<(b,m,n)->(b,"
       << extent - 1
       << "-m,n)>,affine_map<(b,m,n)->(b,m,n)>], iterator_types = "
          "[\"parallel\",\"parallel\",\"parallel\"]} ins(%p : "
       << type << ") outs(%e : " << type
       << ") { ^bb0(%x: f16,%old: f16): linalg.yield %x : f16 } -> " << type
       << "\nreturn %r : " << type << "\n}}";
    auto module = parse(withTopology(os.str(), 1, 3));
    ASSERT_TRUE(module);
    std::string detail;
    auto built = build(*module, detail);
    ASSERT_TRUE(built) << detail;
    auto seed = built->domain.getFirstPlan();
    for (auto &node : seed.nodes) {
      node.axes[1].parameter = 3;
      node.embedding = {TileId(0), TileId(1), TileId(2)};
    }
    mlir::linalg::GenericOp consumer;
    module->walk([&](mlir::linalg::GenericOp op) { consumer = op; });
    auto relation = analysis::IndexRelation::fromAffineMap(
        consumer.getIndexingMapsArray()[0], {1, extent, 128}, {1, extent, 128});
    ASSERT_TRUE(relation.isExact());
    auto intervals =
        getIteratorPartitionIntervals(extent, seed.nodes[0].axes[1], 3);
    ASSERT_TRUE(mlir::succeeded(intervals));
    std::vector<std::pair<int64_t, int64_t>> actual;
    for (const auto &part : *intervals) {
      auto image = relation.get()->getExactStaticRectangularImage(
          {0, part.offset, 0}, {1, part.size, 128});
      ASSERT_TRUE(image.isExact());
      actual.emplace_back(image.domain->offsets[1], image.domain->sizes[1]);
    }
    std::sort(actual.begin(), actual.end());
    const std::vector<std::pair<int64_t, int64_t>> expected =
        extent == 1025 ? std::vector<std::pair<int64_t, int64_t>>{{0, 341},
                                                                  {341, 342},
                                                                  {683, 342}}
                       : std::vector<std::pair<int64_t, int64_t>>{
                             {0, 341}, {341, 341}, {682, 342}};
    EXPECT_EQ(actual, expected);
    for (auto order : {SpatialPropagationOrder::ProducersFirst,
                       SpatialPropagationOrder::ConsumersFirst}) {
      auto coordinated = propagateSpatialPartitions(built->domain, built->dag,
                                                    seed, {}, order);
      ASSERT_TRUE(mlir::succeeded(coordinated));
      EXPECT_EQ(*coordinated, seed);
    }
  }
}

TEST_F(SpatialDomainTest, ReadingProducerRequiresAgreementAcrossConsumers) {
  for (int64_t extent : {1024, 1025})
    for (bool conflict : {false, true}) {
      auto module = parse(wafer::compiler::testing::spatialContractionSource(
          extent, false, false, 16));
      ASSERT_TRUE(module);
      auto function = *module->getOps<mlir::func::FuncOp>().begin();
      auto *terminator = function.getBody().front().getTerminator();
      auto *first = terminator->getOperand(0).getDefiningOp();
      mlir::OpBuilder builder(terminator);
      auto *second = builder.clone(*first);
      terminator->setOperands({first->getResult(0), second->getResult(0)});
      function.setType(builder.getFunctionType(
          function.getArgumentTypes(),
          {first->getResult(0).getType(), second->getResult(0).getType()}));
      std::string detail;
      auto built = build(*module, detail);
      ASSERT_TRUE(built) << detail;
      auto seed = built->domain.getFirstPlan();
      size_t firstIndex = spatialNodeIndex(built->domain, seed, first);
      size_t secondIndex = spatialNodeIndex(built->domain, seed, second);
      size_t producer = spatialNodeIndex(built->domain, seed,
                                         first->getOperand(0).getDefiningOp());
      seed.nodes[firstIndex].axes[1].parameter = 4;
      seed.nodes[secondIndex].axes[conflict ? 2 : 1].parameter = 4;
      seed.nodes[firstIndex].embedding = {TileId(0), TileId(1), TileId(2),
                                          TileId(3)};
      seed.nodes[secondIndex].embedding = seed.nodes[firstIndex].embedding;
      auto coordinated =
          propagateSpatialPartitions(built->domain, built->dag, seed, {},
                                     SpatialPropagationOrder::ConsumersFirst);
      ASSERT_TRUE(mlir::succeeded(coordinated));
      EXPECT_EQ(coordinated->nodes[producer].axes[1].parameter,
                conflict ? 1 : 4);
      EXPECT_TRUE(
          built->domain.evaluate(built->dag, *coordinated).isSatisfied());
    }
}

TEST_F(SpatialDomainTest,
       OverlappingConvolutionHaloDoesNotBecomeAnOwnedPartition) {
  for (int64_t extent : {1024, 1025}) {
    std::string text;
    llvm::raw_string_ostream os(text);
    const std::string input =
        "tensor<1x" + std::to_string(extent + 2) + "x128xf16>";
    const std::string output =
        "tensor<1x" + std::to_string(extent) + "x128xf16>";
    os << "module { func.func @main(%input: " << input
       << ", %kernel: tensor<3x128x128xf16>, %init: " << output << ") -> "
       << output << " {\n"
       << "%e = tensor.empty() : " << input
       << "\n%p = linalg.map ins(%input : " << input << ") outs(%e : " << input
       << ") (%x: f16) { linalg.yield %x : f16 }\n"
       << "%r = linalg.conv_1d_nwc_wcf {strides = dense<1> : tensor<1xi64>, "
          "dilations = dense<1> : tensor<1xi64>} ins(%p,%kernel : "
       << input << ",tensor<3x128x128xf16>) outs(%init : " << output << ") -> "
       << output << "\nreturn %r : " << output << "\n}}";
    auto module = parse(withTopology(os.str(), 1, 4));
    ASSERT_TRUE(module);
    std::string detail;
    auto built = build(*module, detail);
    ASSERT_TRUE(built) << detail;
    auto *consumer = built->dag.getFunction()
                         .getBody()
                         .front()
                         .getTerminator()
                         ->getOperand(0)
                         .getDefiningOp();
    auto seed = built->domain.getFirstPlan();
    auto index = spatialNodeIndex(built->domain, seed, consumer);
    seed.nodes[index].axes[1].parameter = 4;
    seed.nodes[index].embedding = {TileId(0), TileId(1), TileId(2), TileId(3)};
    auto coordinated =
        propagateSpatialPartitions(built->domain, built->dag, seed, {},
                                   SpatialPropagationOrder::ConsumersFirst);
    ASSERT_TRUE(mlir::succeeded(coordinated));
    EXPECT_EQ(*coordinated, seed);
    EXPECT_TRUE(built->domain.evaluate(built->dag, *coordinated).isSatisfied());
  }
}

TEST_F(SpatialDomainTest,
       BackwardOutputDemandRetainsAnIndependentReductionAxis) {
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t reductionParts : {4, 16}) {
      SCOPED_TRACE(reductionParts);
      auto module = parse(wafer::compiler::testing::spatialContractionSource(
          extent, false, false, 16));
      ASSERT_TRUE(module);
      auto function = *module->getOps<mlir::func::FuncOp>().begin();
      auto *terminator = function.getBody().front().getTerminator();
      auto *gemm = terminator->getOperand(0).getDefiningOp();
      auto *producer = gemm->getOperand(0).getDefiningOp();
      mlir::OpBuilder builder(terminator);
      mlir::IRMapping mapping;
      mapping.map(producer->getOperand(0), gemm->getResult(0));
      auto *pointwise = builder.clone(*producer, mapping);
      terminator->setOperand(0, pointwise->getResult(0));
      std::string detail;
      auto built = build(*module, detail);
      ASSERT_TRUE(built) << detail;
      auto seed = built->domain.getFirstPlan();
      size_t gemmIndex = spatialNodeIndex(built->domain, seed, gemm);
      size_t consumerIndex = spatialNodeIndex(built->domain, seed, pointwise);
      seed.nodes[gemmIndex].axes[3].parameter = reductionParts;
      seed.nodes[gemmIndex].embedding.clear();
      for (int64_t tile = 0; tile < reductionParts; ++tile)
        seed.nodes[gemmIndex].embedding.push_back(TileId(tile));
      auto groups = deriveSpatialReductionGroups(
          built->domain.getProblem().getRoots()[gemmIndex],
          seed.nodes[gemmIndex].axes);
      ASSERT_TRUE(mlir::succeeded(groups));
      ASSERT_EQ(groups->size(), 1u);
      seed.nodes[gemmIndex].reductionMerges = {
          {groups->front(), TileId(reductionParts - 1)}};
      seed.nodes[consumerIndex].axes[1].parameter = 4;
      seed.nodes[consumerIndex].embedding = {TileId(0), TileId(4), TileId(8),
                                             TileId(12)};
      ASSERT_TRUE(built->domain.contains(seed));
      auto coordinated =
          propagateSpatialPartitions(built->domain, built->dag, seed, {},
                                     SpatialPropagationOrder::ConsumersFirst);
      ASSERT_TRUE(mlir::succeeded(coordinated));
      EXPECT_EQ(coordinated->nodes[gemmIndex].axes[1].parameter, 4);
      EXPECT_EQ(coordinated->nodes[gemmIndex].axes[3].parameter,
                reductionParts == 4 ? 4 : 1);
      EXPECT_EQ(coordinated->nodes[gemmIndex].embedding.size(),
                reductionParts == 4 ? 16u : 4u);
      ASSERT_EQ(coordinated->nodes[gemmIndex].reductionMerges.size(),
                reductionParts == 4 ? 4u : 0u);
      auto evaluation = built->domain.evaluate(built->dag, *coordinated);
      ASSERT_TRUE(evaluation.isSatisfied());
      auto *proof = analysis::getExactDemandProof(*evaluation.demand);
      ASSERT_NE(proof, nullptr);
      ASSERT_EQ(proof->reductionMerges.size(), reductionParts == 4 ? 4u : 0u);
      for (const auto &merge : proof->reductionMerges)
        EXPECT_EQ(merge.contributions.size(), 4u);
    }
}

TEST_F(SpatialDomainTest, UnchangedInputDoesNotHideAnotherProducerPartition) {
  for (int64_t extent : {1024, 1025}) {
    auto module = parse(wafer::compiler::testing::spatialContractionSource(
        extent, false, false, 4));
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    auto *gemm = function.getBody()
                     .front()
                     .getTerminator()
                     ->getOperand(0)
                     .getDefiningOp();
    auto *left = gemm->getOperand(0).getDefiningOp();
    auto weight = gemm->getOperand(1);
    auto type = mlir::cast<mlir::RankedTensorType>(weight.getType());
    mlir::OpBuilder builder(gemm);
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        gemm->getLoc(), type.getShape(), type.getElementType());
    mlir::IRMapping mapping;
    mapping.map(left->getOperand(0), weight);
    mapping.map(left->getOperand(1), empty.getResult());
    auto *right = builder.clone(*left, mapping);
    right->getResult(0).setType(type);
    gemm->setOperand(1, right->getResult(0));
    std::string detail;
    auto built = build(*module, detail);
    ASSERT_TRUE(built) << detail;
    auto seed = built->domain.getFirstPlan();
    size_t rightIndex = spatialNodeIndex(built->domain, seed, right);
    size_t gemmIndex = spatialNodeIndex(built->domain, seed, gemm);
    seed.nodes[rightIndex].axes[1].parameter = 4;
    seed.nodes[rightIndex].embedding = {TileId(0), TileId(1), TileId(2),
                                        TileId(3)};
    auto coordinated =
        propagateSpatialPartitions(built->domain, built->dag, seed, {});
    ASSERT_TRUE(mlir::succeeded(coordinated));
    EXPECT_EQ(coordinated->nodes[gemmIndex].axes[3].parameter, 4);
    EXPECT_TRUE(built->domain.evaluate(built->dag, *coordinated).isSatisfied());
  }
}

} // namespace
