//===- CanonicalSerializedExecutionPlanTest.cpp ----------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalSerializedExecutionPlan.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

class CanonicalSerializedExecutionPlanTest : public ::testing::Test {
protected:
  CanonicalSerializedExecutionPlanTest() {
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

  static std::string mapSource(int64_t extent) {
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
    return source;
  }

  static std::pair<RegionPlan, TemporalPlan>
  makeRootAndMergePlans(int64_t extent, bool rankZero = false) {
    SemanticRootKey root;
    root.anchorIndex = 0;
    RootRegionWorkId firstWork{root, TileId(0)};
    RootRegionWorkId secondWork{root, TileId(1)};
    LogicalShardId firstShard{root, {0}};
    LogicalShardId secondShard{root, {1}};
    ReductionGroupId mergeGroup{root, 0, {0}};

    ExecutionInstanceId firstExecution{
        RequiredRootExecution{firstWork, firstShard}};
    ExecutionInstanceId secondExecution{
        RequiredRootExecution{secondWork, secondShard}};
    ExecutionInstanceId mergeExecution{
        RequiredMergeExecution{secondWork, mergeGroup}};

    RegionGroupPlan firstGroup;
    firstGroup.tile = TileId(0);
    firstGroup.mandatoryRoots.push_back(firstWork);
    firstGroup.executions.push_back({firstExecution});
    RegionGroupPlan secondGroup;
    secondGroup.tile = TileId(1);
    secondGroup.mandatoryRoots.push_back(secondWork);
    if (!rankZero)
      secondGroup.executions.push_back({secondExecution});
    secondGroup.executions.push_back({mergeExecution});

    RegionPlan regions;
    regions.groups = {firstGroup, secondGroup};
    TemporalPlan temporal;
    TemporalScopePlan firstScope;
    firstScope.execution = firstExecution;
    TemporalScopePlan secondScope;
    secondScope.execution = secondExecution;
    if (!rankZero) {
      firstScope.iteratorTileSizes = {1, extent, 128};
      secondScope.iteratorTileSizes = {1, extent, 128};
      temporal.scopes = {firstScope, secondScope};
    } else {
      temporal.scopes = {firstScope};
    }
    return {std::move(regions), std::move(temporal)};
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CanonicalSerializedExecutionPlanTest,
       AlignedAndRaggedAllTileExecutionsAreStable) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(mapSource(extent));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto prefix = wafer::test::buildCanonicalPlanningPrefix(*dag, allTiles(),
                                                            &failureReason);
    ASSERT_TRUE(mlir::succeeded(prefix)) << failureReason;
    CanonicalSerializedExecutionPlanOutcome outcome =
        buildCanonicalSerializedExecutionPlan(prefix->regions,
                                              prefix->temporal);
    const SerializedExecutionPlan *plan = getSerializedExecutionPlan(outcome);
    ASSERT_NE(plan, nullptr);
    ASSERT_EQ(plan->executions.size(), 16u);
    EXPECT_TRUE(llvm::all_of(plan->executions, [](const auto &execution) {
      return std::holds_alternative<RequiredRootExecution>(execution.source);
    }));
    EXPECT_TRUE(
        std::is_sorted(plan->executions.begin(), plan->executions.end()));

    RegionPlan reversedRegions = prefix->regions;
    TemporalPlan reversedTemporal = prefix->temporal;
    std::reverse(reversedRegions.groups.begin(), reversedRegions.groups.end());
    for (RegionGroupPlan &group : reversedRegions.groups)
      std::reverse(group.executions.begin(), group.executions.end());
    std::reverse(reversedTemporal.scopes.begin(),
                 reversedTemporal.scopes.end());
    CanonicalSerializedExecutionPlanOutcome reversed =
        buildCanonicalSerializedExecutionPlan(reversedRegions,
                                              reversedTemporal);
    const SerializedExecutionPlan *reversedPlan =
        getSerializedExecutionPlan(reversed);
    ASSERT_NE(reversedPlan, nullptr);
    EXPECT_EQ(reversedPlan->executions, plan->executions);
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalSerializedExecutionPlanTest,
       MultiRootFanoutKeepsEveryExecutionIdentity) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @fanout(%input: tensor<2x1025x128xf16>)
      -> (tensor<2x1025x128xf16>, tensor<2x1025x128xf16>) {
    %a0 = tensor.empty() : tensor<2x1025x128xf16>
    %a = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%a0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    %b0 = tensor.empty() : tensor<2x1025x128xf16>
    %b = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a : tensor<2x1025x128xf16>)
        outs(%b0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    %c0 = tensor.empty() : tensor<2x1025x128xf16>
    %c = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a : tensor<2x1025x128xf16>)
        outs(%c0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    return %b, %c : tensor<2x1025x128xf16>, tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto prefix = wafer::test::buildCanonicalPlanningPrefix(*dag, allTiles(),
                                                          &failureReason);
  ASSERT_TRUE(mlir::succeeded(prefix)) << failureReason;
  CanonicalSerializedExecutionPlanOutcome outcome =
      buildCanonicalSerializedExecutionPlan(prefix->regions, prefix->temporal);
  const SerializedExecutionPlan *plan = getSerializedExecutionPlan(outcome);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->executions.size(), 48u);
  std::set<SemanticRootKey> roots;
  for (const ExecutionInstanceId &execution : plan->executions) {
    const auto *root = std::get_if<RequiredRootExecution>(&execution.source);
    ASSERT_NE(root, nullptr);
    roots.insert(root->work.root);
  }
  EXPECT_EQ(roots.size(), 3u);
}

TEST_F(CanonicalSerializedExecutionPlanTest,
       OrdinaryAndCoupledMergesDoNotCreateComponentExecutions) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto [regions, temporal] = makeRootAndMergePlans(extent);
    CanonicalSerializedExecutionPlanOutcome outcome =
        buildCanonicalSerializedExecutionPlan(regions, temporal);
    const SerializedExecutionPlan *plan = getSerializedExecutionPlan(outcome);
    ASSERT_NE(plan, nullptr);
    ASSERT_EQ(plan->executions.size(), 3u);
    EXPECT_EQ(
        llvm::count_if(plan->executions,
                       [](const auto &execution) {
                         return std::holds_alternative<RequiredRootExecution>(
                             execution.source);
                       }),
        2u);
    EXPECT_EQ(
        llvm::count_if(plan->executions,
                       [](const auto &execution) {
                         return std::holds_alternative<RequiredMergeExecution>(
                             execution.source);
                       }),
        1u);
  }

  for (const auto &[queryExtent, keyValueExtent] :
       {std::pair<int64_t, int64_t>{1024, 1024}, {1025, 1031}}) {
    SCOPED_TRACE(queryExtent);
    auto module = parse(wafer::test::buildFlashDecodingPlanningFixture(
        queryExtent, keyValueExtent));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto prefix = wafer::test::buildCanonicalPlanningPrefix(*dag, allTiles(),
                                                            &failureReason);
    ASSERT_TRUE(mlir::succeeded(prefix)) << failureReason;
    CanonicalSerializedExecutionPlanOutcome outcome =
        buildCanonicalSerializedExecutionPlan(prefix->regions,
                                              prefix->temporal);
    const SerializedExecutionPlan *plan = getSerializedExecutionPlan(outcome);
    ASSERT_NE(plan, nullptr);
    const size_t mergeCount = llvm::count_if(
        plan->executions, [](const ExecutionInstanceId &execution) {
          return std::holds_alternative<RequiredMergeExecution>(
              execution.source);
        });
    EXPECT_EQ(mergeCount, prefix->demand.reductionMerges.size());
    EXPECT_EQ(plan->executions.size(), 16u + mergeCount);
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalSerializedExecutionPlanTest,
       RankZeroAndMergeOnlyExecutionsRemainExplicit) {
  auto [regions, temporal] = makeRootAndMergePlans(/*extent=*/0,
                                                   /*rankZero=*/true);
  CanonicalSerializedExecutionPlanOutcome outcome =
      buildCanonicalSerializedExecutionPlan(regions, temporal);
  const SerializedExecutionPlan *plan = getSerializedExecutionPlan(outcome);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->executions.size(), 2u);
  EXPECT_TRUE(temporal.scopes.front().iteratorTileSizes.empty());
  EXPECT_EQ(
      llvm::count_if(plan->executions,
                     [](const auto &execution) {
                       return std::holds_alternative<RequiredMergeExecution>(
                           execution.source);
                     }),
      1u);
}

TEST_F(CanonicalSerializedExecutionPlanTest,
       DuplicateMissingAndUnexpectedInputsFailClosed) {
  auto [regions, temporal] = makeRootAndMergePlans(/*extent=*/1024);

  CanonicalSerializedExecutionPlanOutcome emptyOutcome =
      buildCanonicalSerializedExecutionPlan({}, {});
  const auto *emptyFailure =
      std::get_if<BrokenSerializedExecutionPlan>(&emptyOutcome);
  ASSERT_NE(emptyFailure, nullptr);
  EXPECT_EQ(emptyFailure->reason,
            BrokenSerializedExecutionPlanReason::EmptyExecutionSet);

  RegionPlan duplicateExecution = regions;
  duplicateExecution.groups.front().executions.push_back(
      duplicateExecution.groups.front().executions.front());
  CanonicalSerializedExecutionPlanOutcome duplicateExecutionOutcome =
      buildCanonicalSerializedExecutionPlan(duplicateExecution, temporal);
  const auto *duplicateExecutionFailure =
      std::get_if<BrokenSerializedExecutionPlan>(&duplicateExecutionOutcome);
  ASSERT_NE(duplicateExecutionFailure, nullptr);
  EXPECT_EQ(duplicateExecutionFailure->reason,
            BrokenSerializedExecutionPlanReason::DuplicateExecution);

  TemporalPlan missingScope = temporal;
  missingScope.scopes.pop_back();
  CanonicalSerializedExecutionPlanOutcome missingScopeOutcome =
      buildCanonicalSerializedExecutionPlan(regions, missingScope);
  const auto *missingScopeFailure =
      std::get_if<BrokenSerializedExecutionPlan>(&missingScopeOutcome);
  ASSERT_NE(missingScopeFailure, nullptr);
  EXPECT_EQ(missingScopeFailure->reason,
            BrokenSerializedExecutionPlanReason::MissingTemporalScope);

  TemporalPlan duplicateScope = temporal;
  duplicateScope.scopes.push_back(duplicateScope.scopes.front());
  CanonicalSerializedExecutionPlanOutcome duplicateScopeOutcome =
      buildCanonicalSerializedExecutionPlan(regions, duplicateScope);
  const auto *duplicateScopeFailure =
      std::get_if<BrokenSerializedExecutionPlan>(&duplicateScopeOutcome);
  ASSERT_NE(duplicateScopeFailure, nullptr);
  EXPECT_EQ(duplicateScopeFailure->reason,
            BrokenSerializedExecutionPlanReason::DuplicateTemporalScope);

  TemporalPlan unexpectedScope = temporal;
  TemporalScopePlan mergeScope;
  mergeScope.execution = regions.groups.back().executions.back().id;
  unexpectedScope.scopes.push_back(mergeScope);
  CanonicalSerializedExecutionPlanOutcome unexpectedScopeOutcome =
      buildCanonicalSerializedExecutionPlan(regions, unexpectedScope);
  const auto *unexpectedScopeFailure =
      std::get_if<BrokenSerializedExecutionPlan>(&unexpectedScopeOutcome);
  ASSERT_NE(unexpectedScopeFailure, nullptr);
  EXPECT_EQ(unexpectedScopeFailure->reason,
            BrokenSerializedExecutionPlanReason::UnexpectedTemporalScope);

  CanonicalSerializedExecutionPlanOutcome repeated =
      buildCanonicalSerializedExecutionPlan(regions, temporal);
  EXPECT_NE(getSerializedExecutionPlan(repeated), nullptr);
}

} // namespace
