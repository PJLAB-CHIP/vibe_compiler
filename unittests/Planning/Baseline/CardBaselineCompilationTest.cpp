//===- CardBaselineCompilationTest.cpp -------------------------------===//

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"
#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Planning/Baseline/BaselineTemporalPlan.h"
#include "Wafer/Planning/Baseline/CanonicalBaselinePlan.h"
#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace {

wafer::compiler::detail::CompleteCandidatePlan makeMaterializationPlan(
    const wafer::compiler::detail::CanonicalBaselinePlan &plan) {
  return {plan.spatial, plan.demand, plan.rootWorks, plan.regions,
          plan.temporal, plan.preparedAttention};
}
using namespace wafer::compiler::testing;

void expectNoAvoidableNCCDrains(
    const wafer::compiler::detail::CardExecutableLoweringResult &executable) {
  const auto &cost = executable.resourceCost;
  uint64_t requiredCrossingJoins = 0;
  uint64_t requiredCrossingParticipants = 0;
  std::string invalidDrainDetails;
  llvm::raw_string_ostream invalidDrainStream(invalidDrainDetails);
  for (const wafer::compiler::TileExecutable &tile : executable.tiles) {
    tile.getModule().walk([&](wafer::SyncNCCJoinOp join) {
      mlir::Operation *next = join->getNextNode();
      if (mlir::isa_and_nonnull<mlir::func::ReturnOp>(next))
        return;
      ++requiredCrossingJoins;
      requiredCrossingParticipants += join.getParticipants().size();
      bool validCrossing =
          mlir::isa_and_nonnull<wafer::InstrDTESendOp,
                                wafer::InstrDTERecvOp>(next) &&
          !join->getParentOfType<mlir::scf::ForOp>();
      if (!validCrossing) {
        invalidDrainStream << "tile=" << tile.getTileId().getValue()
                           << " join=" << join << " next=";
        if (next)
          invalidDrainStream << *next;
        invalidDrainStream << '\n';
      }
    });
  }
  invalidDrainStream.flush();
  ASSERT_TRUE(cost.aggregateSteadyStateNCCJoinCount.isKnown());
  ASSERT_TRUE(cost.aggregateNonTerminalNCCJoinCount.isKnown());
  ASSERT_TRUE(cost.aggregateSteadyStateNCCParticipantWaitCount.isKnown());
  ASSERT_TRUE(cost.aggregateNonTerminalNCCParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.aggregateSteadyStateNCCJoinCount.value, 0u);
  EXPECT_TRUE(invalidDrainDetails.empty()) << invalidDrainDetails;
  EXPECT_EQ(cost.aggregateNonTerminalNCCJoinCount.value,
            requiredCrossingJoins);
  EXPECT_EQ(cost.aggregateSteadyStateNCCParticipantWaitCount.value, 0u);
  EXPECT_EQ(cost.aggregateNonTerminalNCCParticipantWaitCount.value,
            requiredCrossingParticipants);
}

TEST(CardBaselineCompilationTest,
     MaterializesOneBalancedBaselineThroughExactGates) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  // Exercise the production candidate transaction so the required-join
  // placement and actual SPM feedback use the same CardModule owner.
  auto executable = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, programMetadata(), executionConfig(), diagnostics,
      programData, &baselineStatistics, /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  expectCompleteTileDomain(executable->executable,
                           executable->tileDataflowIRTrace);
  expectNoAvoidableNCCDrains(executable->executable);

  // The baseline API has no search statistics parameter. Its complete work is
  // recorded in the policy-free baseline statistics.
  EXPECT_EQ(baselineStatistics.materializationRejections, 0u);
  EXPECT_EQ(baselineStatistics.exactGates.tileModuleLoweringAttempts, 1u);
  EXPECT_EQ(baselineStatistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(baselineStatistics.baselineCardModuleMaterializations, 1u);
  EXPECT_EQ(baselineStatistics.baselineTileEntryMaterializations, 16u);
  EXPECT_GT(baselineStatistics.baselineMaximumTileMaterializationWorkers, 1u);
  EXPECT_EQ(baselineStatistics.baselineSourcePreparations, 1u);
  EXPECT_EQ(baselineStatistics.baselineMaterializationPreparations, 1u);
  EXPECT_EQ(baselineStatistics.baselineTileIRPrints, 16u);
  EXPECT_EQ(baselineStatistics.exactDemandSatisfiedEdges, 0u);
  EXPECT_EQ(baselineStatistics.spatialCoordinateQueries, 1u);
  EXPECT_EQ(baselineStatistics.actualSPMCapacityRejections, 0u);
  EXPECT_EQ(baselineStatistics.actualTemporalRefinements, 0u);
  EXPECT_EQ(diagnosticsText.find("card-executable-search policy=none"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("card-executable-baseline-controller"),
            std::string::npos)
      << diagnosticsText;
  // The deterministic baseline derives one canonical coordinate directly; no
  // search-state or enumeration statistics is consumed or reported.
  EXPECT_EQ(diagnosticsText.find("search_states="), std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("placement_enumeration="), std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("card-executable-baseline-admission"),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardBaselineCompilationTest,
     DecodeCacheOutputsAvoidWholeTensorSPMInActualCardModule) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
#q = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>
#k = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>
#v = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>
#s = affine_map<(b, h, m, k1, k2, n) -> ()>
#o = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @decode(
      %query: tensor<1x32x1x128xf16>,
      %past_key: tensor<1x32x1023x128xf16>,
      %new_key: tensor<1x32x1x128xf16>,
      %past_value: tensor<1x32x1023x128xf16>,
      %new_value: tensor<1x32x1x128xf16>)
      -> (tensor<1x32x1x128xf16>, tensor<1x32x1024x128xf16>,
          tensor<1x32x1024x128xf16>) {
    %key_row_out = tensor.empty() : tensor<1x32x1x128xf16>
    %key_row = linalg.generic {
        indexing_maps = [affine_map<(b, h, m, d) -> (b, h, m, d)>,
                         affine_map<(b, h, m, d) -> (b, h, m, d)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%new_key : tensor<1x32x1x128xf16>)
        outs(%key_row_out : tensor<1x32x1x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<1x32x1x128xf16>
    %key_empty = tensor.empty() : tensor<1x32x1024x128xf16>
    %key_prefix = tensor.insert_slice %past_key into %key_empty[0, 0, 0, 0]
        [1, 32, 1023, 128] [1, 1, 1, 1]
        : tensor<1x32x1023x128xf16> into tensor<1x32x1024x128xf16>
    %key_cache = tensor.insert_slice %key_row into %key_prefix[0, 0, 1023, 0]
        [1, 32, 1, 128] [1, 1, 1, 1]
        : tensor<1x32x1x128xf16> into tensor<1x32x1024x128xf16>

    %value_row_out = tensor.empty() : tensor<1x32x1x128xf16>
    %value_row = linalg.generic {
        indexing_maps = [affine_map<(b, h, m, d) -> (b, h, m, d)>,
                         affine_map<(b, h, m, d) -> (b, h, m, d)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%new_value : tensor<1x32x1x128xf16>)
        outs(%value_row_out : tensor<1x32x1x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<1x32x1x128xf16>
    %value_empty = tensor.empty() : tensor<1x32x1024x128xf16>
    %value_prefix = tensor.insert_slice %past_value into %value_empty[0, 0, 0, 0]
        [1, 32, 1023, 128] [1, 1, 1, 1]
        : tensor<1x32x1023x128xf16> into tensor<1x32x1024x128xf16>
    %value_cache = tensor.insert_slice %value_row into %value_prefix[0, 0, 1023, 0]
        [1, 32, 1, 128] [1, 1, 1, 1]
        : tensor<1x32x1x128xf16> into tensor<1x32x1024x128xf16>

    %scale = arith.constant 0.08838834764831845 : f32
    %attention_out = tensor.empty() : tensor<1x32x1x128xf16>
    %attention = wafer.linalg_ext.attention
        ins(%query, %key_cache, %value_cache, %scale :
            tensor<1x32x1x128xf16>, tensor<1x32x1024x128xf16>,
            tensor<1x32x1024x128xf16>, f32)
        outs(%attention_out : tensor<1x32x1x128xf16>)
        algorithm(<flash_decoding>)
        indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<1x32x1x128xf16>
    %output_out = tensor.empty() : tensor<1x32x1x128xf16>
    %output = linalg.generic {
        indexing_maps = [affine_map<(b, h, m, d) -> (b, h, m, d)>,
                         affine_map<(b, h, m, d) -> (b, h, m, d)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%attention : tensor<1x32x1x128xf16>)
        outs(%output_out : tensor<1x32x1x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<1x32x1x128xf16>
    return %output, %key_cache, %value_cache
        : tensor<1x32x1x128xf16>, tensor<1x32x1024x128xf16>,
          tensor<1x32x1024x128xf16>
  }
}

)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);

  wafer::frontend::FrontendProgramVerificationResult metadata;
  metadata.numPartitions = 1;
  metadata.programUserInputCount = 5;
  metadata.distributedInputs = {
      boundary(0, {1, 32, 1, 128}), boundary(1, {1, 32, 1023, 128}),
      boundary(2, {1, 32, 1, 128}), boundary(3, {1, 32, 1023, 128}),
      boundary(4, {1, 32, 1, 128})};
  metadata.distributedOutputs = {boundary(0, {1, 32, 1, 128}),
                                 boundary(1, {1, 32, 1024, 128}),
                                 boundary(2, {1, 32, 1024, 128})};

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto program = wafer::compiler::detail::analyzeCardProgram(
      *module, metadata, executionConfig(), diagnostics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(program)) << diagnosticsText;
  std::string failureReason;
  auto plan = wafer::compiler::detail::buildCanonicalBaselinePlan(
      **program, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  wafer::compiler::detail::CandidateMaterializationStatistics statistics;
  auto materializationPlan = makeMaterializationPlan(*plan);
  auto materialized = wafer::compiler::detail::materializeCardCandidate(
      *module, wafer::CardId(0), **program, materializationPlan, &statistics,
      diagnostics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(materialized)) << diagnosticsText;

  unsigned wholeCacheSPMAllocations = 0;
  std::string allocationDetails;
  llvm::raw_string_ostream allocationStream(allocationDetails);
  materialized->module->walk([&](mlir::memref::AllocOp allocation) {
    if (wafer::isWaferSPMMemRefType(allocation.getType()) &&
        allocation.getType().getShape() ==
            llvm::ArrayRef<int64_t>({1, 32, 1024, 128})) {
      ++wholeCacheSPMAllocations;
      if (wholeCacheSPMAllocations <= 2) {
        allocationStream << "allocation=" << allocation << " users=[";
        for (mlir::Operation *user : allocation.getResult().getUsers())
          allocationStream << *user << ',';
        allocationStream << "]\n";
      }
    }
  });
  allocationStream.flush();
  EXPECT_EQ(wholeCacheSPMAllocations, 0u) << allocationDetails;
  unsigned tileRegions = 0;
  materialized->module->walk([&](wafer::TileRegionOp) { ++tileRegions; });
  EXPECT_GT(tileRegions, 0u);
  EXPECT_LT(tileRegions, 256u);
  unsigned attentionOps = 0;
  materialized->module->walk(
      [&](wafer::LinalgExtAttentionOp) { ++attentionOps; });
  EXPECT_EQ(attentionOps, 0u);
  unsigned executableLinalgOps = 0;
  materialized->module->walk(
      [&](mlir::linalg::LinalgOp) { ++executableLinalgOps; });
  EXPECT_EQ(executableLinalgOps, 0u);

  std::optional<wafer::compiler::detail::SemanticRootKey> attentionRoot;
  for (const wafer::analysis::RootRegionWork &work : plan->rootWorks)
    if (mlir::isa_and_nonnull<wafer::LinalgExtAttentionOp>(
            work.rootOperation)) {
      if (attentionRoot)
        EXPECT_TRUE(*attentionRoot == work.id.root);
      attentionRoot = work.id.root;
    }
  ASSERT_TRUE(attentionRoot);
  auto refined = wafer::compiler::detail::refineBaselineTemporalPlan(
      plan->temporal, plan->rootWorks,
      llvm::ArrayRef<wafer::compiler::detail::SemanticRootKey>{&*attentionRoot,
                                                               1},
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(refined)) << failureReason;
  ASSERT_TRUE(*refined);
  ASSERT_TRUE(
      mlir::succeeded(wafer::compiler::detail::recloseCanonicalBaselinePlan(
          *plan, &failureReason)))
      << failureReason;
  wafer::compiler::detail::CandidateMaterializationStatistics refinedStatistics;
  materializationPlan = makeMaterializationPlan(*plan);
  auto refinedMaterialized = wafer::compiler::detail::materializeCardCandidate(
      *module, wafer::CardId(0), **program, materializationPlan,
      &refinedStatistics, diagnostics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(refinedMaterialized)) << diagnosticsText;
  unsigned residentKVAllocations = 0;
  unsigned fullKVAllocations = 0;
  std::string refinedAllocationDetails;
  llvm::raw_string_ostream refinedAllocationStream(refinedAllocationDetails);
  refinedMaterialized->module->walk([&](mlir::memref::AllocOp allocation) {
    if (!wafer::isWaferSPMMemRefType(allocation.getType()))
      return;
    llvm::ArrayRef<int64_t> shape = allocation.getType().getShape();
    residentKVAllocations += llvm::is_contained(shape, int64_t{256});
    if (!llvm::is_contained(shape, int64_t{512}))
      return;
    ++fullKVAllocations;
    if (fullKVAllocations <= 4)
      refinedAllocationStream << allocation << '\n';
  });
  refinedAllocationStream.flush();
  EXPECT_GT(residentKVAllocations, 0u);
  EXPECT_EQ(fullKVAllocations, 0u) << refinedAllocationDetails;

  std::string executableDiagnosticsText;
  llvm::raw_string_ostream executableDiagnostics(executableDiagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  auto executable = wafer::compiler::detail::compileCardBaseline(
      *module, metadata, executionConfig(), executableDiagnostics, programData,
      &baselineStatistics, /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/false);
  executableDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << executableDiagnosticsText;
  expectNoAvoidableNCCDrains(executable->executable);
}

TEST(CardBaselineCompilationTest,
     DirectPrefillOutputScopesPassActualAlignedAndRaggedMemoryPlanning) {
  for (const auto &[queryExtent, keyValueExtent, withMask] :
       {std::tuple<int64_t, int64_t, bool>{1024, 1024, false},
        {1025, 1031, true}}) {
    SCOPED_TRACE(queryExtent);
    std::string source = wafer::test::buildFlashAttentionPlanningFixture(
        queryExtent, keyValueExtent, withMask);
    const size_t scaleArgument = source.find(", %scale: f32");
    ASSERT_NE(scaleArgument, std::string::npos);
    source.erase(scaleArgument, std::string(", %scale: f32").size());
    const size_t outputInit = source.find("    %out = tensor.empty");
    ASSERT_NE(outputInit, std::string::npos);
    source.insert(outputInit,
                  "    %scale = arith.constant 0.08838834764831845 : f32\n");
    const size_t moduleBody = source.find("module {");
    ASSERT_NE(moduleBody, std::string::npos);
    source.insert(moduleBody + std::string("module {").size(), R"mlir(
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
)mlir");

    mlir::DialectRegistry registry;
    wafer::compiler::detail::registerCompilationDialects(registry);
    auto context = std::make_shared<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(module);

    wafer::frontend::FrontendProgramVerificationResult metadata;
    metadata.numPartitions = 1;
    metadata.programUserInputCount = withMask ? 4 : 3;
    metadata.distributedInputs = {boundary(0, {2, queryExtent, 128}),
                                  boundary(1, {2, keyValueExtent, 128}),
                                  boundary(2, {2, keyValueExtent, 64})};
    if (withMask)
      metadata.distributedInputs.push_back(
          boundary(3, {queryExtent, keyValueExtent}));
    metadata.distributedOutputs = {boundary(0, {2, queryExtent, 64})};

    std::string diagnosticsText;
    llvm::raw_string_ostream diagnostics(diagnosticsText);
    wafer::compiler::detail::BaselineStatistics statistics;
    wafer::compiler::ProgramDataHandoff programData;
    auto executable = wafer::compiler::detail::compileCardBaseline(
        *module, metadata, executionConfig(), diagnostics, programData,
        &statistics, /*tilePipelineParallelism=*/0,
        /*captureTileDataflowIRTrace=*/false);
    diagnostics.flush();
    ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
    expectNoAvoidableNCCDrains(executable->executable);
    EXPECT_EQ(statistics.actualTemporalRefinements,
              statistics.actualSPMCapacityRejections);
    EXPECT_EQ(statistics.baselineCardModuleMaterializations,
              statistics.actualTemporalRefinements + 1);
    EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations,
              statistics.baselineCardModuleMaterializations);
    EXPECT_EQ(statistics.baselineTileEntryMaterializations,
              statistics.baselineCardModuleMaterializations * 16);
  }
}

struct RankFourPrefillCase {
  int64_t batchExtent;
  int64_t headExtent;
  int64_t queryExtent;
  int64_t keyValueExtent;
  int64_t queryKeyExtent;
  int64_t valueExtent;
  bool withMask;
};

class CardBaselineRankFourPrefillTest
    : public ::testing::TestWithParam<RankFourPrefillCase> {};

TEST_P(CardBaselineRankFourPrefillTest,
       KeepsBatchAndHeadAxesThroughActualMemoryPlanning) {
  const RankFourPrefillCase testCase = GetParam();
  SCOPED_TRACE(testCase.queryExtent);
  std::string source = wafer::test::buildRank4FlashAttentionPlanningFixture(
      testCase.batchExtent, testCase.headExtent, testCase.queryExtent,
      testCase.keyValueExtent, testCase.queryKeyExtent, testCase.valueExtent,
      testCase.withMask);
  const size_t scaleArgument = source.find(", %scale: f32");
  ASSERT_NE(scaleArgument, std::string::npos);
  source.erase(scaleArgument, std::string(", %scale: f32").size());
  const size_t outputInit = source.find("    %out = tensor.empty");
  ASSERT_NE(outputInit, std::string::npos);
  source.insert(outputInit,
                "    %scale = arith.constant 0.08838834764831845 : f32\n");
  const size_t moduleBody = source.find("module {");
  ASSERT_NE(moduleBody, std::string::npos);
  source.insert(moduleBody + std::string("module {").size(), R"mlir(
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
)mlir");

  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);

  wafer::frontend::FrontendProgramVerificationResult metadata;
  metadata.numPartitions = 1;
  metadata.programUserInputCount = testCase.withMask ? 4 : 3;
  metadata.distributedInputs = {
      boundary(0, {testCase.batchExtent, testCase.headExtent,
                   testCase.queryExtent, testCase.queryKeyExtent}),
      boundary(1, {testCase.batchExtent, testCase.headExtent,
                   testCase.keyValueExtent, testCase.queryKeyExtent}),
      boundary(2, {testCase.batchExtent, testCase.headExtent,
                   testCase.keyValueExtent, testCase.valueExtent})};
  if (testCase.withMask)
    metadata.distributedInputs.push_back(
        boundary(3, {testCase.batchExtent, testCase.headExtent,
                     testCase.queryExtent, testCase.keyValueExtent}));
  metadata.distributedOutputs = {
      boundary(0, {testCase.batchExtent, testCase.headExtent,
                   testCase.queryExtent, testCase.valueExtent})};

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::compileCardBaseline(
      *module, metadata, executionConfig(), diagnostics, programData,
      &statistics, /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/false);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  expectNoAvoidableNCCDrains(executable->executable);
  ASSERT_EQ(executable->executable.tiles.size(), 16u);
  EXPECT_EQ(statistics.actualTemporalRefinements,
            statistics.actualSPMCapacityRejections);
  EXPECT_EQ(statistics.baselineCardModuleMaterializations,
            statistics.actualTemporalRefinements + 1);
  EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations,
            statistics.baselineCardModuleMaterializations);
  EXPECT_EQ(statistics.baselineTileEntryMaterializations,
            statistics.baselineCardModuleMaterializations * 16);
}

INSTANTIATE_TEST_SUITE_P(
    AlignedAndRagged, CardBaselineRankFourPrefillTest,
    ::testing::Values(RankFourPrefillCase{2, 2, 1024, 1024, 128, 64, false},
                      RankFourPrefillCase{1, 2, 1025, 1031, 64, 128, true}));

TEST(CardBaselineCompilationTest,
     ExtractsIndependentStructuredOwnersIntoFinalRegions) {
  ParsedProgram parsed = parseBranchProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, branchMetadata(), executionConfig(), diagnostics,
      programData, &baselineStatistics, /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  ASSERT_EQ(executable->executable.tiles.size(), 16u);
  EXPECT_EQ(baselineStatistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(baselineStatistics.baselineCardModuleMaterializations, 1u);
  EXPECT_EQ(baselineStatistics.baselineTileEntryMaterializations, 16u);
  EXPECT_GT(baselineStatistics.baselineMaximumTileMaterializationWorkers, 1u);
  // Two independent structured roots on one Tile form multiple sequential
  // regions: the shared Tile (Tile 0) carries one region per root.
  ASSERT_EQ(executable->tileDataflowIRTrace.size(), 16u);
  EXPECT_EQ(countOccurrences(executable->tileDataflowIRTrace.front(),
                             "wafer.tile.region"),
            2u)
      << executable->tileDataflowIRTrace.front();
  EXPECT_EQ(diagnosticsText.find("multiple structured compute roots"),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardBaselineCompilationTest,
     CarriesBroadcastDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseBroadcastProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             broadcastProgramMetadata());
}

TEST(CardBaselineCompilationTest,
     DerivesTheUnpartitionedCanonicalCoordinateForScalarOutputs) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<2x1024x1xf16>) -> tensor<f16> {
    %resultOut = tensor.empty() : tensor<f16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%resultOut : tensor<f16>) -> tensor<f16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> ()>],
        iterator_types = ["reduction", "reduction", "reduction"]
      } ins(%input : tensor<2x1024x1xf16>) outs(%init : tensor<f16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<f16>
    return %result : tensor<f16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {2, 1024, 1})};
  program.distributedOutputs = {boundary(0, {})};

  // The root has no parallel result axis. The canonical coordinate is the
  // typed unpartitioned form: one participating Tile, unit partition factors
  // over every iterator, the complete result domain and no fabricated shard
  // axis. This test proves that coordinate through the complete executable
  // gate; the broader movement-demand cases remain in the demand gate suite.
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::compileCardBaseline(
      *module, program, executionConfig(), diagnostics, programData,
      &baselineStatistics, /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/false);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_EQ(baselineStatistics.baselineTileIRPrints, 0u);
  ASSERT_FALSE(executable->executable.tiles.empty());
  // The unpartitioned root occupies exactly one Tile: the complete result
  // domain lives on a single participating Tile. The complete executable
  // still materializes one entry per available Tile for the full-card ABI,
  // but every other entry is the typed no-work form (a bare return), so
  // exactly one entry carries a compute body.
  size_t computeEntries = 0;
  wafer::TileId computeTile(-1);
  for (const wafer::compiler::TileExecutable &tile :
       executable->executable.tiles) {
    mlir::func::FuncOp entry =
        tile.getModule().lookupSymbol<mlir::func::FuncOp>(
            tile.getEntrySymbol());
    ASSERT_TRUE(entry);
    // Every no-work entry is the typed shell contract (entry + return, plus
    // the ABI preparation allocation); all compute, movement and dataflow
    // ops live in Wafer-owned dialects, so their presence is the typed
    // marker of a participating entry.
    bool hasDataflowOp = false;
    entry.walk([&](mlir::Operation *operation) {
      if (operation->getDialect() &&
          operation->getDialect()->getNamespace().starts_with("wafer"))
        hasDataflowOp = true;
    });
    if (hasDataflowOp) {
      ++computeEntries;
      computeTile = tile.getTileId();
    }
  }
  EXPECT_EQ(computeEntries, 1u);
  // The unpartitioned canonical coordinate selects the first available Tile.
  EXPECT_EQ(computeTile.getValue(), 0);
}

TEST(CardBaselineCompilationTest,
     CarriesReductionDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseReductionDemandProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             reductionDemandProgramMetadata());
}

TEST(CardBaselineCompilationTest,
     CarriesWindowDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseWindowDemandProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             windowDemandProgramMetadata());
}

TEST(CardBaselineCompilationTest,
     CarriesStridedDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseStridedDemandProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             stridedDemandProgramMetadata());
}

TEST(CardBaselineCompilationTest,
     CarriesMultiPieceDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseMultiPieceDemandProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             multiPieceDemandProgramMetadata());
}

TEST(CardBaselineCompilationTest,
     ReconstructsMultiProducerInputThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseMultiProducerJoinProgram();
  expectDemandProgramCompletesExecutableGate(
      parsed, multiProducerJoinProgramMetadata());
}

TEST(CardBaselineCompilationTest, ProducesStableCardModuleAndCardExecutableIR) {
  ParsedProgram firstProgram = parseProgram();
  ParsedProgram secondProgram = parseProgram();
  ASSERT_TRUE(firstProgram.module);
  ASSERT_TRUE(secondProgram.module);
  std::string firstDiagnosticsText;
  std::string secondDiagnosticsText;
  llvm::raw_string_ostream firstDiagnostics(firstDiagnosticsText);
  llvm::raw_string_ostream secondDiagnostics(secondDiagnosticsText);
  wafer::compiler::detail::BaselineStatistics firstStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  wafer::compiler::detail::BaselineStatistics secondStatistics;
  auto first = wafer::compiler::detail::compileCardBaseline(
      *firstProgram.module, programMetadata(), executionConfig(),
      firstDiagnostics, programData, &firstStatistics,
      /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/true);
  auto second = wafer::compiler::detail::compileCardBaseline(
      *secondProgram.module, programMetadata(), executionConfig(),
      secondDiagnostics, programData, &secondStatistics,
      /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/true);
  firstDiagnostics.flush();
  secondDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(first)) << firstDiagnosticsText;
  ASSERT_TRUE(mlir::succeeded(second)) << secondDiagnosticsText;
  ASSERT_EQ(first->executable.tiles.size(), second->executable.tiles.size());
  EXPECT_EQ(first->tileDataflowIRTrace, second->tileDataflowIRTrace);
  for (auto [firstTile, secondTile] :
       llvm::zip(first->executable.tiles, second->executable.tiles)) {
    std::string firstExecutableIR;
    std::string secondExecutableIR;
    llvm::raw_string_ostream firstStream(firstExecutableIR);
    llvm::raw_string_ostream secondStream(secondExecutableIR);
    firstTile.getModule().print(firstStream);
    secondTile.getModule().print(secondStream);
    firstStream.flush();
    secondStream.flush();
    EXPECT_EQ(firstExecutableIR, secondExecutableIR);
  }
  EXPECT_EQ(firstStatistics.baselineCardModuleMaterializations,
            secondStatistics.baselineCardModuleMaterializations);
  EXPECT_EQ(firstStatistics.baselineTileEntryMaterializations,
            secondStatistics.baselineTileEntryMaterializations);
  EXPECT_EQ(firstStatistics.baselineMaximumTileMaterializationWorkers,
            secondStatistics.baselineMaximumTileMaterializationWorkers);
  EXPECT_EQ(firstStatistics.baselineMaterializationPreparations,
            secondStatistics.baselineMaterializationPreparations);
  EXPECT_EQ(firstStatistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(secondStatistics.exactGates.cardModuleCompilationInvocations, 1u);
}

TEST(CardBaselineCompilationTest,
     UsesWaveBoundedProducerCarriersAndConsumerDemand) {
  ParsedProgram parsed = parseLargeProducerStageProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, largeProducerStageProgramMetadata(), executionConfig(),
      diagnostics, programData, &baselineStatistics,
      /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_GT(baselineStatistics.actualSPMCapacityRejections, 0u);
  EXPECT_EQ(baselineStatistics.materializationRejections,
            baselineStatistics.actualSPMCapacityRejections);
  EXPECT_EQ(baselineStatistics.actualTemporalRefinements,
            baselineStatistics.actualSPMCapacityRejections);
  const uint64_t candidateCount =
      baselineStatistics.actualTemporalRefinements + 1;
  EXPECT_EQ(baselineStatistics.baselineSourcePreparations, candidateCount);
  EXPECT_EQ(baselineStatistics.baselineMaterializationPreparations,
            candidateCount);
  EXPECT_EQ(baselineStatistics.baselineCardModuleMaterializations,
            candidateCount);
  EXPECT_EQ(baselineStatistics.baselineTileEntryMaterializations,
            candidateCount * 16);
  EXPECT_EQ(baselineStatistics.exactGates.cardModuleCompilationInvocations,
            candidateCount);
  ASSERT_FALSE(executable->tileDataflowIRTrace.empty());
  llvm::StringRef firstTileIR = executable->tileDataflowIRTrace.front();
  EXPECT_GE(countOccurrences(firstTileIR, "wafer.tile.region"), 2u)
      << firstTileIR.str();
  EXPECT_GT(countOccurrences(firstTileIR, "wafer.tile.store"), 1u)
      << firstTileIR.str();
  EXPECT_GT(countOccurrences(firstTileIR, "wafer.tile.load"), 1u)
      << firstTileIR.str();
  size_t peerEndpoints = 0;
  size_t temporalLoops = 0;
  for (llvm::StringRef tileIR : executable->tileDataflowIRTrace) {
    peerEndpoints += countOccurrences(tileIR, "wafer.tile.peer_send");
    peerEndpoints += countOccurrences(tileIR, "wafer.tile.peer_recv");
    temporalLoops += countOccurrences(tileIR, "scf.for");
  }
  EXPECT_LT(peerEndpoints, 256u);
  EXPECT_GT(temporalLoops, 0u);
  EXPECT_NE(diagnosticsText.find(
                "card-executable-compilation outcome=exact-rejection"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("exhausted its temporal domain"),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardBaselineCompilationTest,
     UsesFiniteTemporalTraversalWhenOneWaveExceedsSPM) {
  ParsedProgram baselineProgram = parseLargeTemporalProgram();
  ASSERT_TRUE(baselineProgram.module);
  std::string baselineDiagnosticsText;
  llvm::raw_string_ostream baselineDiagnostics(baselineDiagnosticsText);
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto baseline = wafer::compiler::detail::compileCardBaseline(
      *baselineProgram.module, largeTemporalProgramMetadata(),
      executionConfig(), baselineDiagnostics, programData, &baselineStatistics,
      /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/true);
  baselineDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << baselineDiagnosticsText;
  // The initial full local coordinate is actualized first. Only its typed SPM
  // conflicts advance the deterministic temporal coordinate; the accepted
  // CardModule is retained rather than rebuilt.
  EXPECT_GT(baselineStatistics.actualSPMCapacityRejections, 0u);
  EXPECT_EQ(baselineStatistics.materializationRejections,
            baselineStatistics.actualSPMCapacityRejections);
  EXPECT_EQ(baselineStatistics.actualTemporalRefinements,
            baselineStatistics.actualSPMCapacityRejections);
  const uint64_t candidateCount =
      baselineStatistics.actualTemporalRefinements + 1;
  EXPECT_EQ(baselineStatistics.baselineCardModuleMaterializations,
            candidateCount);
  EXPECT_EQ(baselineStatistics.baselineSourcePreparations, candidateCount);
  EXPECT_EQ(baselineStatistics.baselineMaterializationPreparations,
            candidateCount);
  EXPECT_EQ(baselineStatistics.baselineTileEntryMaterializations,
            candidateCount * 16);
  EXPECT_EQ(baselineStatistics.exactGates.cardModuleCompilationInvocations,
            candidateCount);
  EXPECT_NE(baselineDiagnosticsText.find(
                "card-executable-compilation outcome=exact-rejection"),
            std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(baselineDiagnosticsText.find("card-exact-spm-conflict-certificate"),
            std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(
      baselineDiagnosticsText.find("tile-execution-allocation-feedback-joint"),
      std::string::npos)
      << baselineDiagnosticsText;
  for (llvm::StringRef tileDataflowIR : baseline->tileDataflowIRTrace) {
    EXPECT_NE(tileDataflowIR.find("scf.for"), llvm::StringRef::npos)
        << tileDataflowIR.str();
    const size_t stores = countOccurrences(tileDataflowIR, "wafer.tile.store");
    EXPECT_GT(stores, 1u);
    EXPECT_LE(stores, 81u);
    EXPECT_NE(tileDataflowIR.find("memref.alloc"), llvm::StringRef::npos)
        << tileDataflowIR.str();
  }
}

} // namespace
