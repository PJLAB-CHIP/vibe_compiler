//===- RootWorkDomainTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

class RootWorkDomainTest : public ::testing::Test {
protected:
  RootWorkDomainTest() {
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

  static mlir::func::FuncOp function(mlir::ModuleOp module) {
    return *module.getOps<mlir::func::FuncOp>().begin();
  }

  static llvm::SmallVector<TileId, 16> allTiles() {
    llvm::SmallVector<TileId, 16> tiles;
    for (int64_t tile = 0; tile < 16; ++tile)
      tiles.push_back(TileId(tile));
    return tiles;
  }

  static mlir::FailureOr<ExactDemandProof>
  deriveDemand(const StructuredDAGAnalysis &dag,
               const SpatialAssignment &assignment,
               std::string *failureReason) {
    auto session = DemandPlanningSession::create(dag, IndexRelationLimits(),
                                                 failureReason);
    if (mlir::failed(session))
      return mlir::failure();
    ExactDemandOutcome outcome = session->query(assignment);
    session->close();
    const ExactDemandProof *proof = getExactDemandProof(outcome);
    if (!proof) {
      if (failureReason)
        *failureReason = std::visit(
            [](const auto &value) -> std::string {
              using T = std::decay_t<decltype(value)>;
              if constexpr (std::is_same_v<T, ExactDemandProof>)
                return {};
              else
                return value.detail;
            },
            outcome);
      return mlir::failure();
    }
    return *proof;
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(RootWorkDomainTest, AlignedAndRaggedSuccessorMatchesEveryNonemptySite) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "module {\n  func.func @main(%input: tensor<2x" << extent
           << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
           << "    %empty = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %result = linalg.map ins(%input : tensor<2x" << extent
           << "x128xf16>) outs(%empty : tensor<2x" << extent
           << "x128xf16>) (%value: f16) {\n"
           << "      linalg.yield %value : f16\n    }\n"
           << "    return %result : tensor<2x" << extent
           << "x128xf16>\n  }\n}\n";
    auto module = parse(source);
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto coordinate =
        buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
    auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
    ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
    llvm::SmallVector<TileId, 16> reversed = allTiles();
    std::reverse(reversed.begin(), reversed.end());
    auto domain = RootWorkDomain::create(*dag, coordinate->assignment, *proof,
                                         reversed, &failureReason);
    ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;

    std::vector<RootRegionWorkId> actual;
    RootWorkSuccessor successor = domain->getFirstWork();
    while (successor.getKind() == RootWorkSuccessorKind::Work) {
      const RootRegionWork *work = successor.getWork();
      const RootWorkCursor *cursor = successor.getCursor();
      ASSERT_NE(work, nullptr);
      ASSERT_NE(cursor, nullptr);
      actual.push_back(work->id);
      successor = domain->getNextWork(*cursor);
    }
    EXPECT_EQ(successor.getKind(), RootWorkSuccessorKind::End);
    ASSERT_EQ(actual.size(), 16u);
    EXPECT_TRUE(std::is_sorted(actual.begin(), actual.end()));

    auto reference = RootRegionWorkAnalysis::create(
        *dag, coordinate->assignment, *proof, &failureReason);
    ASSERT_TRUE(mlir::succeeded(reference)) << failureReason;
    std::vector<RootRegionWorkId> expected;
    for (const SemanticRootBinding &root : coordinate->semanticRoots.getRoots())
      for (TileId tile : allTiles())
        if (std::holds_alternative<RootRegionWork>(
                reference->query(root.key, tile)))
          expected.push_back({root.key, tile});
    EXPECT_EQ(actual, expected);

    RootWorkCollectionOutcome collection = collectRootWorks(*domain);
    const RootWorkCollection *works = getRootWorkCollection(collection);
    ASSERT_NE(works, nullptr);
    EXPECT_EQ(works->works.size(), actual.size());
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(RootWorkDomainTest,
       SpatialContributionsAndRemoteMergeSitesRemainDistinctWork) {
  auto module = parse(R"mlir(
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
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  SpatialDomainProblemResult problem =
      buildSpatialDomainProblem(*dag, allTiles());
  ASSERT_TRUE(problem.succeeded());
  const SpatialRootDomainFacts &root = problem.problem->getRoots().front();
  SpatialPlan plan;
  NodeSpatialPlan node;
  node.root = root.root;
  node.axes = {{0, IteratorPartitionScheme::BalancedParts, 2},
               {1, IteratorPartitionScheme::BalancedParts, 2},
               {2, IteratorPartitionScheme::BalancedParts, 1}};
  node.embedding = {TileId(0), TileId(1), TileId(4), TileId(5)};
  auto groups = deriveSpatialReductionGroups(root, node.axes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(groups)) << failureReason;
  ASSERT_EQ(groups->size(), 2u);
  node.reductionMerges = {{(*groups)[0], TileId(14)},
                          {(*groups)[1], TileId(15)}};
  plan.nodes.push_back(std::move(node));
  auto assignment = closeSpatialPlanStructure(
      problem.problem->getStructuralProblem(), plan, &failureReason);
  ASSERT_TRUE(mlir::succeeded(assignment)) << failureReason;
  auto proof = deriveDemand(*dag, *assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  auto domain = RootWorkDomain::create(*dag, *assignment, *proof, allTiles(),
                                       &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;

  std::vector<RootRegionWork> works;
  RootWorkSuccessor successor = domain->getFirstWork();
  while (successor.getKind() == RootWorkSuccessorKind::Work) {
    const RootWorkCursor *cursor = successor.getCursor();
    ASSERT_NE(cursor, nullptr);
    works.push_back(*successor.getWork());
    successor = domain->getNextWork(*cursor);
  }
  EXPECT_EQ(successor.getKind(), RootWorkSuccessorKind::End);
  ASSERT_EQ(works.size(), 6u);
  EXPECT_EQ(works[0].id.tile, TileId(0));
  EXPECT_EQ(works[1].id.tile, TileId(1));
  EXPECT_EQ(works[2].id.tile, TileId(4));
  EXPECT_EQ(works[3].id.tile, TileId(5));
  EXPECT_EQ(works[4].id.tile, TileId(14));
  EXPECT_EQ(works[5].id.tile, TileId(15));
  for (size_t index = 0; index < 4; ++index) {
    EXPECT_EQ(works[index].execution.size(), 1u);
    EXPECT_EQ(works[index].contributions.size(), 1u);
  }
  for (size_t index = 4; index < works.size(); ++index) {
    EXPECT_TRUE(works[index].execution.empty());
    EXPECT_EQ(works[index].merges.size(), 1u);
    EXPECT_EQ(works[index].results.size(), 1u);
    EXPECT_EQ(works[index].merges.front().mergeTile, works[index].id.tile);
  }

  EXPECT_EQ(successor.getCursor(), nullptr);
}

TEST_F(RootWorkDomainTest,
       ExactUnionWorkLimitRemainsTypedAfterEarlierRootSiblings) {
  auto module = parse(R"mlir(
module {
  func.func @main(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %empty = tensor.empty() : tensor<2x1025x128xf16>
    %consumer = linalg.map ins(%input : tensor<2x1025x128xf16>)
        outs(%empty : tensor<2x1025x128xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %consumer : tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  auto proof = deriveDemand(*dag, coordinate->assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proof)) << failureReason;
  ASSERT_FALSE(proof->dependencyDemands.empty());
  ASSERT_FALSE(proof->dependencyDemands.front().perDestination.empty());
  auto &sources =
      proof->dependencyDemands.front().perDestination.back().sources;
  ASSERT_EQ(sources.size(), 1u);
  SourceDemand duplicate = sources.front();
  IndexSetResult singleton =
      IndexRelation::staticRectangularDomain({0, 0, 0}, {1, 1, 1});
  ASSERT_TRUE(singleton.isExact());
  StaticRectangularIndexSet box{{0, 0, 0}, {1, 1, 1}};
  duplicate.requiredDomain = ExactIndexSet(std::move(*singleton.set),
                                           ExactIndexSetForm::BoxUnion, {box});
  sources.push_back(std::move(duplicate));
  auto domain = RootWorkDomain::create(*dag, coordinate->assignment, *proof,
                                       allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  IndexRelationLimits limits;
  limits.maxDisjuncts = 1;
  limits.maxRectangularPieces = 1;
  RootWorkSuccessor successor = domain->getFirstWork(limits);
  unsigned precedingWorks = 0;
  while (successor.getKind() == RootWorkSuccessorKind::Work) {
    ++precedingWorks;
    const RootWorkCursor *cursor = successor.getCursor();
    ASSERT_NE(cursor, nullptr);
    successor = domain->getNextWork(*cursor, limits);
  }
  EXPECT_GT(precedingWorks, 0u);
  EXPECT_EQ(successor.getKind(), RootWorkSuccessorKind::Indeterminate);
  ASSERT_NE(successor.getFailure(), nullptr);
  EXPECT_TRUE(std::holds_alternative<RootRegionWorkLimitReached>(
      *successor.getFailure()));
}

TEST(RootWorkOutcomeTest, ClassifiesEveryTypedRootWorkOutcome) {
  EXPECT_EQ(classifyRootWorkOutcome(RootRegionWork{}),
            RootWorkSiteOutcomeKind::Work);
  EXPECT_EQ(classifyRootWorkOutcome(NoRootRegionWork{}),
            RootWorkSiteOutcomeKind::NoWork);
  EXPECT_EQ(classifyRootWorkOutcome(UnsupportedRootRegionWork{}),
            RootWorkSiteOutcomeKind::Unsupported);
  EXPECT_EQ(classifyRootWorkOutcome(RootRegionWorkLimitReached{}),
            RootWorkSiteOutcomeKind::Indeterminate);
  EXPECT_EQ(classifyRootWorkOutcome(BrokenRootRegionWork{}),
            RootWorkSiteOutcomeKind::CompilerBug);
}

} // namespace
