//===- SpatialDomainTest.cpp -------------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"
#include "TestSupport/Planning/SpatialPlanReference.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
        algorithm(<flash_attention>) indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    %e1 = tensor.empty() : tensor<2x1025x64xf16>
    %fd = wafer.linalg_ext.attention
        ins(%q, %k, %v, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%e1 : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>) indexing_maps = [#q, #k, #v, #s, #o]
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
        indexing_maps = [#q, #k, #v, #s, #mask, #o]
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

} // namespace
