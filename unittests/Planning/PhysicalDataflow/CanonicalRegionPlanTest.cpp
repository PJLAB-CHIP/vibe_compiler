//===- CanonicalRegionPlanTest.cpp ------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

class CanonicalRegionPlanTest : public ::testing::Test {
protected:
  CanonicalRegionPlanTest() {
    wafer::registerWaferCoreDialects(registry);
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

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  static ExactIndexSet box(llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes) {
    IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
    EXPECT_TRUE(set.isExact());
    StaticRectangularIndexSet rectangle{llvm::to_vector(offsets),
                                        llvm::to_vector(sizes)};
    return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion,
                         {rectangle});
  }

  static ExactIndexSet empty(unsigned rank) {
    return ExactIndexSet(
        mlir::presburger::PresburgerSet::getEmpty(
            mlir::presburger::PresburgerSpace::getSetSpace(rank)),
        ExactIndexSetForm::BoxUnion);
  }

  static std::string rootWorkDetail(const RootRegionWorkOutcome &outcome) {
    return std::visit(
        [](const auto &value) -> std::string {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, RootRegionWork> ||
                        std::is_same_v<T, NoRootRegionWork>)
            return {};
          else
            return value.detail;
        },
        outcome);
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

  static mlir::FailureOr<std::vector<RootRegionWork>>
  collectRootWorks(const StructuredDAGAnalysis &dag,
                   const SpatialAssignment &assignment,
                   const ExactDemandProof &proof, llvm::ArrayRef<TileId> tiles,
                   std::string *failureReason) {
    auto analysis =
        RootRegionWorkAnalysis::create(dag, assignment, proof, failureReason);
    if (mlir::failed(analysis))
      return mlir::failure();
    std::vector<RootRegionWork> works;
    for (const StructuredDAGNode &node : dag.getNodes()) {
      const SemanticRootKey *root = analysis->getRoot(node.id);
      if (!root)
        return mlir::failure();
      for (TileId tile : tiles) {
        RootRegionWorkOutcome outcome = analysis->query(*root, tile);
        if (std::holds_alternative<NoRootRegionWork>(outcome))
          continue;
        const RootRegionWork *work = std::get_if<RootRegionWork>(&outcome);
        if (!work) {
          if (failureReason)
            *failureReason = rootWorkDetail(outcome);
          return mlir::failure();
        }
        works.push_back(*work);
      }
    }
    return works;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CanonicalRegionPlanTest,
       AlignedAndRaggedAllTileWorkFormsSingletonExternalBoundaryGroups) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @map(%input: tensor<2x)mlir"
           << extent << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
           << "    %empty = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %result = linalg.generic {\n"
           << "        indexing_maps = [#id, #id],\n"
           << "        iterator_types = [\"parallel\", \"parallel\", "
              "\"parallel\"]}\n"
           << "        ins(%input : tensor<2x" << extent
           << "x128xf16>) outs(%empty : tensor<2x" << extent << "x128xf16>) {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        linalg.yield %value : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    return %result : tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
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
    auto works = collectRootWorks(*dag, coordinate->assignment, *proof,
                                  allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
    ASSERT_EQ(works->size(), 16u);
    CanonicalRegionPlanOutcome outcome = buildCanonicalRegionPlan(*works);
    const RegionPlan *plan = getRegionPlan(outcome);
    ASSERT_NE(plan, nullptr);
    ASSERT_EQ(plan->groups.size(), 16u);
    for (const RegionGroupPlan &group : plan->groups) {
      ASSERT_EQ(group.mandatoryRoots.size(), 1u);
      EXPECT_EQ(group.tile, group.mandatoryRoots.front().tile);
      ASSERT_EQ(group.executions.size(), 1u);
      EXPECT_TRUE(std::holds_alternative<RequiredRootExecution>(
          group.executions.front().id.source));
      ASSERT_EQ(group.externalBindings.size(), 1u);
      EXPECT_EQ(group.externalBindings.front().fragment.source.kind,
                RootBoundaryKind::ProgramInput);
      EXPECT_FALSE(
          group.externalBindings.front().fragment.ownerTile.has_value());
    }
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalRegionPlanTest,
       DiamondFanoutKeepsRootsSingletonAndFragmentsUseExactProducerIdentity) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @fanout(%input: tensor<2x1025x128xf16>)
      -> (tensor<2x1025x128xf16>, tensor<2x1025x128xf16>) {
    %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
    %producer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%producer_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    %left_empty = tensor.empty() : tensor<2x1025x128xf16>
    %left = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer : tensor<2x1025x128xf16>)
        outs(%left_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    %right_empty = tensor.empty() : tensor<2x1025x128xf16>
    %right = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer : tensor<2x1025x128xf16>)
        outs(%right_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    %join_empty = tensor.empty() : tensor<2x1025x128xf16>
    %join = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%left, %right : tensor<2x1025x128xf16>,
                            tensor<2x1025x128xf16>)
        outs(%join_empty : tensor<2x1025x128xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %old: f16):
        %value = arith.addf %lhs, %rhs : f16
        linalg.yield %value : f16
    } -> tensor<2x1025x128xf16>
    return %join, %right : tensor<2x1025x128xf16>,
                           tensor<2x1025x128xf16>
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
  auto works = collectRootWorks(*dag, coordinate->assignment, *proof,
                                allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
  ASSERT_EQ(works->size(), 64u);
  CanonicalRegionPlanOutcome outcome = buildCanonicalRegionPlan(*works);
  const RegionPlan *plan = getRegionPlan(outcome);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->groups.size(), 64u);
  unsigned structuredFragments = 0;
  std::set<DemandFragmentId> fragmentIds;
  for (const RegionGroupPlan &group : plan->groups) {
    EXPECT_EQ(group.mandatoryRoots.size(), 1u);
    for (const ExternalUseBinding &binding : group.externalBindings) {
      EXPECT_TRUE(fragmentIds.insert(binding.fragment).second);
      if (binding.fragment.source.kind != RootBoundaryKind::StructuredResult)
        continue;
      ++structuredFragments;
      EXPECT_TRUE(binding.fragment.ownerTile.has_value());
      EXPECT_TRUE(binding.fragment.ownerShard.has_value() ||
                  binding.fragment.reductionGroup.has_value());
    }
  }
  EXPECT_EQ(structuredFragments, 64u);

  std::reverse(works->begin(), works->end());
  CanonicalRegionPlanOutcome reversedOutcome = buildCanonicalRegionPlan(*works);
  const RegionPlan *reversed = getRegionPlan(reversedOutcome);
  ASSERT_NE(reversed, nullptr);
  ASSERT_EQ(reversed->groups.size(), plan->groups.size());
  for (auto [expected, actual] :
       llvm::zip_equal(plan->groups, reversed->groups)) {
    EXPECT_EQ(actual.tile, expected.tile);
    EXPECT_EQ(actual.mandatoryRoots, expected.mandatoryRoots);
    ASSERT_EQ(actual.executions.size(), expected.executions.size());
    EXPECT_EQ(actual.executions.front().id, expected.executions.front().id);
    ASSERT_EQ(actual.externalBindings.size(), expected.externalBindings.size());
    for (auto [expectedBinding, actualBinding] :
         llvm::zip_equal(expected.externalBindings, actual.externalBindings))
      EXPECT_EQ(actualBinding.fragment, expectedBinding.fragment);
  }
}

TEST_F(CanonicalRegionPlanTest,
       ContributionAndCoupledComponentsDoNotCreateIndependentExecutions) {
  auto module = parse(R"mlir(
module {
  func.func @holder() {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::Operation *holder = &function(*module).getBody().front().front();
  SemanticRootKey root;
  root.anchorIndex = 0;
  RootRegionWork work;
  work.id = {root, TileId(3)};
  work.rootOperation = holder;
  LogicalShardId shard{root, {0, 0, 0}};
  work.execution.push_back({shard, {{0, 1}, {0, 1025}, {0, 128}}});

  ReductionGroupId first{root, 0, {0, 0}};
  ReductionGroupId second{root, 0, {0, 1}};
  ReductionContribution contribution;
  contribution.shard = shard;
  contribution.tile = TileId(3);
  contribution.components.resize(3);
  work.contributions.push_back(
      {first, TileId(3),
       ReductionInitialization::CoupledIdentityPerContribution,
       ReductionAlgebraKind::CoupledReduction, CoupledReductionRule{},
       contribution});
  ReductionMergeRequirement firstMerge;
  firstMerge.group = first;
  firstMerge.mergeTile = TileId(3);
  firstMerge.algebra = ReductionAlgebraKind::CoupledReduction;
  firstMerge.coupledRule = CoupledReductionRule{};
  firstMerge.components.resize(3);
  firstMerge.contributions.push_back(contribution);
  ReductionMergeRequirement secondMerge = firstMerge;
  secondMerge.group = second;
  work.merges = {firstMerge, secondMerge};
  work.results.push_back({0, std::nullopt, first, {}});
  work.results.push_back({0, std::nullopt, second, {}});

  CanonicalRegionPlanOutcome outcome = buildCanonicalRegionPlan({work});
  const RegionPlan *plan = getRegionPlan(outcome);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->groups.size(), 1u);
  ASSERT_EQ(plan->groups.front().executions.size(), 3u);
  EXPECT_EQ(
      llvm::count_if(plan->groups.front().executions,
                     [](const ExecutionInstancePlan &execution) {
                       return std::holds_alternative<RequiredRootExecution>(
                           execution.id.source);
                     }),
      1u);
  EXPECT_EQ(
      llvm::count_if(plan->groups.front().executions,
                     [](const ExecutionInstancePlan &execution) {
                       return std::holds_alternative<RequiredMergeExecution>(
                           execution.id.source);
                     }),
      2u);
}

TEST_F(CanonicalRegionPlanTest,
       MultiResultSourcesOnDifferentTilesRemainDistinctUseFragments) {
  auto module = parse(R"mlir(
module {
  func.func @holder() {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::Operation *holder = &function(*module).getBody().front().front();
  SemanticRootKey consumerRoot;
  consumerRoot.anchorIndex = 0;
  SemanticRootKey producerRoot;
  producerRoot.anchorIndex = 1;
  RootRegionWork work;
  work.id = {consumerRoot, TileId(0)};
  work.rootOperation = holder;
  LogicalShardId consumerShard{consumerRoot, {0}};
  work.execution.push_back({consumerShard, {{0, 1025}}});
  ExactIndexSet complete = box({0}, {1025});
  RootOperandWork operand;
  operand.operand = 0;
  operand.uses.push_back({{0, consumerShard}, complete, complete});
  work.operands.push_back(operand);

  for (uint32_t result = 0; result < 2; ++result) {
    ExactIndexSet piece = result == 0 ? box({0}, {500}) : box({500}, {525});
    LogicalShardId ownerShard{producerRoot, {result}};
    RootBoundaryWork boundary;
    boundary.id = {RootBoundaryKind::StructuredResult, producerRoot, result};
    boundary.requiredDomain = piece;
    RootBoundaryUseWork use;
    use.id = {0, consumerShard};
    use.requiredDomain = piece;
    use.eligibleFinalOwners.push_back(
        {ownerShard, std::nullopt, TileId(result + 1), piece});
    boundary.consumerUses.push_back(std::move(use));
    work.boundaries.push_back(std::move(boundary));
  }

  CanonicalRegionPlanOutcome outcome = buildCanonicalRegionPlan({work});
  const RegionPlan *plan = getRegionPlan(outcome);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->groups.front().externalBindings.size(), 2u);
  EXPECT_EQ(plan->groups.front().externalBindings[0].fragment.source.index, 0u);
  EXPECT_EQ(plan->groups.front().externalBindings[1].fragment.source.index, 1u);
  ASSERT_TRUE(
      plan->groups.front().externalBindings[0].fragment.ownerTile.has_value());
  ASSERT_TRUE(
      plan->groups.front().externalBindings[1].fragment.ownerTile.has_value());
  EXPECT_EQ(*plan->groups.front().externalBindings[0].fragment.ownerTile,
            TileId(1));
  EXPECT_EQ(*plan->groups.front().externalBindings[1].fragment.ownerTile,
            TileId(2));
}

TEST_F(CanonicalRegionPlanTest,
       DuplicateWorkAndMissingStructuredOwnerAreContractFailures) {
  auto module = parse(R"mlir(
module {
  func.func @holder() {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::Operation *holder = &function(*module).getBody().front().front();
  SemanticRootKey root;
  RootRegionWork work;
  work.id = {root, TileId(0)};
  work.rootOperation = holder;
  LogicalShardId shard{root, {0}};
  work.execution.push_back({shard, {{0, 1024}}});

  CanonicalRegionPlanOutcome duplicate = buildCanonicalRegionPlan({work, work});
  const auto *duplicateFailure = std::get_if<BrokenRegionPlan>(&duplicate);
  ASSERT_NE(duplicateFailure, nullptr);
  EXPECT_EQ(duplicateFailure->reason,
            BrokenRegionPlanReason::DuplicateRootWork);

  RootOperandWork operand;
  operand.operand = 0;
  ExactIndexSet emptyDomain = empty(1);
  operand.uses.push_back({{0, shard}, emptyDomain, emptyDomain});
  work.operands.push_back(operand);
  RootBoundaryWork boundary;
  boundary.id.kind = RootBoundaryKind::StructuredResult;
  boundary.id.semantic = root;
  boundary.requiredDomain = emptyDomain;
  boundary.consumerUses.push_back({{0, shard}, emptyDomain, {}});
  work.boundaries.push_back(boundary);
  CanonicalRegionPlanOutcome exactEmpty = buildCanonicalRegionPlan({work});
  const RegionPlan *emptyPlan = getRegionPlan(exactEmpty);
  ASSERT_NE(emptyPlan, nullptr);
  ASSERT_EQ(emptyPlan->groups.front().externalBindings.size(), 1u);
  EXPECT_FALSE(emptyPlan->groups.front()
                   .externalBindings.front()
                   .fragment.ownerTile.has_value());

  ExactIndexSet domain = box({0}, {1024});
  work.operands.front().uses.front().consumerExecutionDomain = domain;
  work.operands.front().uses.front().operandDemand = domain;
  work.boundaries.front().requiredDomain = domain;
  work.boundaries.front().consumerUses.front().requiredDomain = domain;
  CanonicalRegionPlanOutcome missingOwner = buildCanonicalRegionPlan({work});
  const auto *ownerFailure = std::get_if<BrokenRegionPlan>(&missingOwner);
  ASSERT_NE(ownerFailure, nullptr);
  EXPECT_EQ(ownerFailure->reason,
            BrokenRegionPlanReason::MissingBoundaryFragment);
}

} // namespace
