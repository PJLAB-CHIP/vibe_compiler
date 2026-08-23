//===- CanonicalTemporalPlanTest.cpp ----------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalTemporalPlan.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

class CanonicalTemporalPlanTest : public ::testing::Test {
protected:
  CanonicalTemporalPlanTest() {
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
    if (!proof)
      return mlir::failure();
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
        const RootRegionWork *work = getRootRegionWork(outcome);
        if (!work)
          return mlir::failure();
        works.push_back(*work);
      }
    }
    return works;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CanonicalTemporalPlanTest,
       AlignedAndRaggedScopesUseTheirOwnFullLocalExtents) {
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
    CanonicalRegionPlanOutcome regionOutcome = buildCanonicalRegionPlan(*works);
    const RegionPlan *regions = getRegionPlan(regionOutcome);
    ASSERT_NE(regions, nullptr);
    CanonicalTemporalPlanOutcome outcome =
        buildCanonicalTemporalPlan(*regions, *works);
    const TemporalPlan *plan = getTemporalPlan(outcome);
    ASSERT_NE(plan, nullptr);
    ASSERT_EQ(plan->scopes.size(), 16u);

    std::map<ExecutionInstanceId, llvm::SmallVector<int64_t, 4>> expected;
    for (const RootRegionWork &work : *works)
      for (const RootExecutionWork &execution : work.execution) {
        llvm::SmallVector<int64_t, 4> sizes;
        for (const IteratorInterval &interval : execution.iterationDomain)
          sizes.push_back(interval.size);
        expected.try_emplace(ExecutionInstanceId{RequiredRootExecution{
                                 work.id, execution.shard}},
                             std::move(sizes));
      }
    llvm::SmallVector<int64_t, 16> queryAxisSizes;
    for (const TemporalScopePlan &scope : plan->scopes) {
      const ExecutionInstanceId *execution = getRequiredExecution(scope.id);
      ASSERT_NE(execution, nullptr);
      EXPECT_TRUE(isTopLevelScope(scope.id));
      ASSERT_TRUE(expected.count(*execution));
      EXPECT_EQ(scope.iteratorTileSizes, expected[*execution]);
      EXPECT_TRUE(scope.waveLoopOrder.empty());
      queryAxisSizes.push_back(scope.iteratorTileSizes[1]);
    }
    if (extent == 1024)
      EXPECT_EQ(*llvm::min_element(queryAxisSizes),
                *llvm::max_element(queryAxisSizes));
    else
      EXPECT_LT(*llvm::min_element(queryAxisSizes),
                *llvm::max_element(queryAxisSizes));

    std::reverse(works->begin(), works->end());
    CanonicalTemporalPlanOutcome reversed =
        buildCanonicalTemporalPlan(*regions, *works);
    const TemporalPlan *reversedPlan = getTemporalPlan(reversed);
    ASSERT_NE(reversedPlan, nullptr);
    ASSERT_EQ(reversedPlan->scopes.size(), plan->scopes.size());
    for (auto [expectedScope, actualScope] :
         llvm::zip_equal(plan->scopes, reversedPlan->scopes)) {
      EXPECT_EQ(actualScope.id, expectedScope.id);
      EXPECT_EQ(actualScope.iteratorTileSizes, expectedScope.iteratorTileSizes);
      EXPECT_EQ(actualScope.waveLoopOrder, expectedScope.waveLoopOrder);
    }
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalTemporalPlanTest,
       ReductionAndFlashDecodingMergesDoNotCreateTemporalScopes) {
  auto module = parse(R"mlir(
module {
  func.func @holder() {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::Operation *holder = &function(*module).getBody().front().front();
  SemanticRootKey reductionRoot;
  reductionRoot.anchorIndex = 0;
  RootRegionWork reduction;
  reduction.id = {reductionRoot, TileId(0)};
  reduction.rootOperation = holder;
  LogicalShardId reductionShard{reductionRoot, {0, 0, 0, 0}};
  reduction.execution.push_back(
      {reductionShard, {{0, 1}, {0, 17}, {0, 19}, {0, 1025}}});
  ReductionGroupId reductionGroup{reductionRoot, 0, {0, 0}};
  ReductionMergeRequirement reductionMerge;
  reductionMerge.group = reductionGroup;
  reductionMerge.mergeTile = TileId(0);
  reduction.merges.push_back(reductionMerge);
  reduction.results.push_back({0, std::nullopt, reductionGroup, {}});

  SemanticRootKey attentionRoot;
  attentionRoot.anchorIndex = 1;
  RootRegionWork attention;
  attention.id = {attentionRoot, TileId(1)};
  attention.rootOperation = holder;
  LogicalShardId attentionShard{attentionRoot, {0, 0, 0, 0, 0}};
  attention.execution.push_back(
      {attentionShard, {{0, 1}, {0, 513}, {0, 128}, {0, 516}, {0, 64}}});
  ReductionGroupId first{attentionRoot, 0, {0, 0}};
  ReductionGroupId second{attentionRoot, 0, {0, 1}};
  ReductionMergeRequirement attentionMerge;
  attentionMerge.group = first;
  attentionMerge.mergeTile = TileId(1);
  attentionMerge.algebra = ReductionAlgebraKind::CoupledReduction;
  attentionMerge.coupledRule = CoupledReductionRule{};
  attentionMerge.components.resize(3);
  attention.merges.push_back(attentionMerge);
  attentionMerge.group = second;
  attention.merges.push_back(attentionMerge);
  attention.results.push_back({0, std::nullopt, first, {}});
  attention.results.push_back({0, std::nullopt, second, {}});

  std::array<RootRegionWork, 2> works{reduction, attention};
  CanonicalRegionPlanOutcome regionOutcome = buildCanonicalRegionPlan(works);
  const RegionPlan *regions = getRegionPlan(regionOutcome);
  ASSERT_NE(regions, nullptr);
  EXPECT_EQ(llvm::count_if(regions->groups,
                           [](const RegionGroupPlan &group) {
                             return group.executions.size() == 2;
                           }),
            1u);
  EXPECT_EQ(llvm::count_if(regions->groups,
                           [](const RegionGroupPlan &group) {
                             return group.executions.size() == 3;
                           }),
            1u);
  CanonicalTemporalPlanOutcome outcome =
      buildCanonicalTemporalPlan(*regions, works);
  const TemporalPlan *plan = getTemporalPlan(outcome);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->scopes.size(), 2u);
  auto reductionScope =
      llvm::find_if(plan->scopes, [&](const TemporalScopePlan &scope) {
        const ExecutionInstanceId *id = getRequiredExecution(scope.id);
        const auto *execution =
            id ? std::get_if<RequiredRootExecution>(&id->source) : nullptr;
        return execution && execution->work == reduction.id;
      });
  auto attentionScope =
      llvm::find_if(plan->scopes, [&](const TemporalScopePlan &scope) {
        const ExecutionInstanceId *id = getRequiredExecution(scope.id);
        const auto *execution =
            id ? std::get_if<RequiredRootExecution>(&id->source) : nullptr;
        return execution && execution->work == attention.id;
      });
  ASSERT_NE(reductionScope, plan->scopes.end());
  ASSERT_NE(attentionScope, plan->scopes.end());
  EXPECT_EQ(reductionScope->iteratorTileSizes,
            (llvm::SmallVector<int64_t, 4>{1, 17, 19, 1025}));
  EXPECT_EQ(attentionScope->iteratorTileSizes,
            (llvm::SmallVector<int64_t, 4>{1, 513, 128, 516, 64}));
  EXPECT_TRUE(reductionScope->waveLoopOrder.empty());
  EXPECT_TRUE(attentionScope->waveLoopOrder.empty());
}

TEST_F(CanonicalTemporalPlanTest,
       RankZeroHasOneEmptyScopeWhileMergeOnlyWorkHasNone) {
  auto module = parse(R"mlir(
module {
  func.func @holder() {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::Operation *holder = &function(*module).getBody().front().front();
  SemanticRootKey scalarRoot;
  scalarRoot.anchorIndex = 0;
  RootRegionWork scalar;
  scalar.id = {scalarRoot, TileId(0)};
  scalar.rootOperation = holder;
  scalar.execution.push_back({{scalarRoot, {}}, {}});

  SemanticRootKey mergeRoot;
  mergeRoot.anchorIndex = 1;
  RootRegionWork mergeOnly;
  mergeOnly.id = {mergeRoot, TileId(1)};
  mergeOnly.rootOperation = holder;
  ReductionMergeRequirement merge;
  merge.group = {mergeRoot, 0, {0}};
  merge.mergeTile = TileId(1);
  mergeOnly.merges.push_back(merge);
  mergeOnly.results.push_back({0, std::nullopt, merge.group, {}});

  std::array<RootRegionWork, 2> works{scalar, mergeOnly};
  CanonicalRegionPlanOutcome regionOutcome = buildCanonicalRegionPlan(works);
  const RegionPlan *regions = getRegionPlan(regionOutcome);
  ASSERT_NE(regions, nullptr);
  CanonicalTemporalPlanOutcome outcome =
      buildCanonicalTemporalPlan(*regions, works);
  const TemporalPlan *plan = getTemporalPlan(outcome);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->scopes.size(), 1u);
  EXPECT_TRUE(plan->scopes.front().iteratorTileSizes.empty());
  EXPECT_TRUE(plan->scopes.front().waveLoopOrder.empty());
}

TEST_F(CanonicalTemporalPlanTest,
       MissingDuplicateAndInvalidExecutionsAreContractFailures) {
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
  CanonicalRegionPlanOutcome regionOutcome = buildCanonicalRegionPlan({work});
  const RegionPlan *regions = getRegionPlan(regionOutcome);
  ASSERT_NE(regions, nullptr);

  RegionPlan missing = *regions;
  missing.groups.front().executions.clear();
  CanonicalTemporalPlanOutcome missingOutcome =
      buildCanonicalTemporalPlan(missing, {work});
  const auto *missingFailure = std::get_if<BrokenTemporalPlan>(&missingOutcome);
  ASSERT_NE(missingFailure, nullptr);
  EXPECT_EQ(missingFailure->reason, BrokenTemporalPlanReason::MissingExecution);

  RegionPlan duplicate = *regions;
  duplicate.groups.front().executions.push_back(
      duplicate.groups.front().executions.front());
  CanonicalTemporalPlanOutcome duplicateOutcome =
      buildCanonicalTemporalPlan(duplicate, {work});
  const auto *duplicateFailure =
      std::get_if<BrokenTemporalPlan>(&duplicateOutcome);
  ASSERT_NE(duplicateFailure, nullptr);
  EXPECT_EQ(duplicateFailure->reason, BrokenTemporalPlanReason::DuplicateScope);

  RootRegionWork invalid = work;
  invalid.execution.front().iterationDomain.front().size = 0;
  CanonicalRegionPlanOutcome invalidRegion =
      buildCanonicalRegionPlan({invalid});
  const RegionPlan *invalidRegions = getRegionPlan(invalidRegion);
  ASSERT_NE(invalidRegions, nullptr);
  CanonicalTemporalPlanOutcome invalidOutcome =
      buildCanonicalTemporalPlan(*invalidRegions, {invalid});
  const auto *invalidFailure = std::get_if<BrokenTemporalPlan>(&invalidOutcome);
  ASSERT_NE(invalidFailure, nullptr);
  EXPECT_EQ(invalidFailure->reason,
            BrokenTemporalPlanReason::InvalidLocalExtent);
}

} // namespace
