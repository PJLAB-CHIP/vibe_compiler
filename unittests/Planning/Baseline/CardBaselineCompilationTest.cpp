//===- CardBaselineCompilationTest.cpp -------------------------------===//

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "gtest/gtest.h"

namespace {
using namespace wafer::compiler::testing;

TEST(CardBaselineCompilationTest,
     MaterializesOneBalancedBaselineThroughExactGates) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  std::vector<std::string> tileDataflowIRTrace;
  auto executable =
      wafer::compiler::detail::compileCardBaseline(
          *parsed.module, programMetadata(), executionConfig(), diagnostics,
          programData, &baselineStatistics, /*tilePipelineParallelism=*/0,
          &tileDataflowIRTrace);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  expectCompleteTileDomain(executable->executable, tileDataflowIRTrace);

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
  EXPECT_EQ(baselineStatistics.spatialLegalizationTransitions, 0u);
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
     ExtractsIndependentStructuredOwnersIntoFinalRegions) {
  ParsedProgram parsed = parseBranchProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  std::vector<std::string> tileDataflowIRTrace;
  auto executable =
      wafer::compiler::detail::compileCardBaseline(
          *parsed.module, branchMetadata(), executionConfig(), diagnostics,
          programData, &baselineStatistics, /*tilePipelineParallelism=*/0,
          &tileDataflowIRTrace);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  ASSERT_EQ(executable->executable.tiles.size(), 16u);
  EXPECT_EQ(baselineStatistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(baselineStatistics.baselineCardModuleMaterializations, 1u);
  EXPECT_EQ(baselineStatistics.baselineTileEntryMaterializations, 16u);
  EXPECT_GT(baselineStatistics.baselineMaximumTileMaterializationWorkers, 1u);
  // Two independent structured roots on one Tile form multiple sequential
  // regions: the shared Tile (Tile 0) carries one region per root.
  ASSERT_EQ(tileDataflowIRTrace.size(), 16u);
  EXPECT_EQ(countOccurrences(tileDataflowIRTrace.front(),
                             "wafer.tile.region"),
            2u)
      << tileDataflowIRTrace.front();
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
  func.func @main(%input: tensor<32x32xf16>) -> tensor<f16> {
    %resultOut = tensor.empty() : tensor<f16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%resultOut : tensor<f16>) -> tensor<f16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> ()>],
        iterator_types = ["reduction", "reduction"]
      } ins(%input : tensor<32x32xf16>) outs(%init : tensor<f16>) {
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
  program.distributedInputs = {boundary(0, {32, 32})};
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
  auto executable =
      wafer::compiler::detail::compileCardBaseline(
          *module, program, executionConfig(), diagnostics, programData,
          &baselineStatistics, /*tilePipelineParallelism=*/0,
          /*tileDataflowIRTrace=*/nullptr);
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
    mlir::func::FuncOp entry = tile.getModule().lookupSymbol<mlir::func::FuncOp>(
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

TEST(CardBaselineCompilationTest,
     ProducesStableCardModuleAndCardExecutableIR) {
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
  std::vector<std::string> firstTileDataflowIRTrace;
  std::vector<std::string> secondTileDataflowIRTrace;
  auto first = wafer::compiler::detail::compileCardBaseline(
      *firstProgram.module, programMetadata(), executionConfig(),
      firstDiagnostics, programData, &firstStatistics,
      /*tilePipelineParallelism=*/0, &firstTileDataflowIRTrace);
  auto second = wafer::compiler::detail::compileCardBaseline(
      *secondProgram.module, programMetadata(), executionConfig(),
      secondDiagnostics, programData, &secondStatistics,
      /*tilePipelineParallelism=*/0, &secondTileDataflowIRTrace);
  firstDiagnostics.flush();
  secondDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(first)) << firstDiagnosticsText;
  ASSERT_TRUE(mlir::succeeded(second)) << secondDiagnosticsText;
  ASSERT_EQ(first->executable.tiles.size(), second->executable.tiles.size());
  EXPECT_EQ(firstTileDataflowIRTrace, secondTileDataflowIRTrace);
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
     DeterministicSpatialCoordinateAdvancesOneMonotoneState) {
  using wafer::compiler::detail::DeterministicSpatialAdvance;
  using wafer::compiler::detail::StructuredDAGNodePlacement;
  using wafer::compiler::detail::StructuredDAGSpatialPartition;
  llvm::SmallVector<StructuredDAGNodePlacement, 3> placements;
  placements.push_back(
      {0, StructuredDAGSpatialPartition{0, 0},
       {wafer::TileId(0), wafer::TileId(1), wafer::TileId(2),
        wafer::TileId(3), wafer::TileId(4)},
       {5, 1}});
  placements.push_back(
      {1, StructuredDAGSpatialPartition{1, 0},
       {wafer::TileId(0), wafer::TileId(1), wafer::TileId(2)},
       {1, 3}});
  placements.push_back({2, std::nullopt, {wafer::TileId(0)}, {1, 1}});

  unsigned transitions = 0;
  while (true) {
    std::string failureReason;
    auto result =
        wafer::compiler::detail::advanceDeterministicSpatialCoordinate(
            placements, &failureReason);
    ASSERT_TRUE(mlir::succeeded(result)) << failureReason;
    if (*result == DeterministicSpatialAdvance::Exhausted)
      break;
    ++transitions;
  }
  EXPECT_EQ(transitions, 4u);
  for (const StructuredDAGNodePlacement &placement : placements) {
    EXPECT_EQ(placement.tiles.size(), 1u);
    if (placement.spatialPartition)
      EXPECT_EQ(placement.iteratorPartitionFactors
                    [placement.spatialPartition->iteratorDimension],
                1u);
    else
      EXPECT_EQ(placement.iteratorPartitionFactors,
                (llvm::SmallVector<uint32_t, 4>{1, 1}));
  }

  placements.front().iteratorPartitionFactors[0] = 2;
  const size_t originalTileCount = placements.front().tiles.size();
  std::string failureReason;
  auto malformed =
      wafer::compiler::detail::advanceDeterministicSpatialCoordinate(
          placements, &failureReason);
  EXPECT_TRUE(mlir::failed(malformed));
  EXPECT_EQ(placements.front().tiles.size(), originalTileCount);
  EXPECT_FALSE(failureReason.empty());
}

TEST(CardBaselineCompilationTest,
     UsesWaveBoundedProducerCarriersAndConsumerDemand) {
  ParsedProgram parsed = parseLargeProducerStageProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  std::vector<std::string> tileDataflowIRTrace;
  auto executable =
      wafer::compiler::detail::compileCardBaseline(
          *parsed.module, largeProducerStageProgramMetadata(), executionConfig(),
          diagnostics, programData, &baselineStatistics,
          /*tilePipelineParallelism=*/0, &tileDataflowIRTrace);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_EQ(baselineStatistics.materializationRejections, 0u);
  EXPECT_EQ(baselineStatistics.baselineSourcePreparations, 1u);
  EXPECT_EQ(baselineStatistics.baselineMaterializationPreparations, 1u);
  EXPECT_EQ(baselineStatistics.baselineCardModuleMaterializations, 1u);
  EXPECT_EQ(baselineStatistics.baselineTileEntryMaterializations, 16u);
  EXPECT_EQ(baselineStatistics.exactGates.cardModuleCompilationInvocations, 1u);
  ASSERT_FALSE(tileDataflowIRTrace.empty());
  llvm::StringRef firstTileIR = tileDataflowIRTrace.front();
  EXPECT_GE(countOccurrences(firstTileIR, "wafer.tile.region"), 2u)
      << firstTileIR.str();
  EXPECT_GT(countOccurrences(firstTileIR, "wafer.tile.store"), 1u)
      << firstTileIR.str();
  EXPECT_GT(countOccurrences(firstTileIR, "wafer.tile.load"), 1u)
      << firstTileIR.str();
  EXPECT_EQ(diagnosticsText.find("card-executable-baseline-temporal-refinement"),
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
  std::vector<std::string> tileDataflowIRTrace;
  auto baseline =
      wafer::compiler::detail::compileCardBaseline(
          *baselineProgram.module, largeTemporalProgramMetadata(),
          executionConfig(), baselineDiagnostics, programData, &baselineStatistics,
          /*tilePipelineParallelism=*/0, &tileDataflowIRTrace);
  baselineDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << baselineDiagnosticsText;
  // Temporal wave shapes are derived before physical materialization. The
  // baseline therefore constructs and lowers one CardModule without a
  // failure-driven retry path.
  EXPECT_EQ(baselineStatistics.materializationRejections, 0u);
  EXPECT_EQ(baselineStatistics.baselineCardModuleMaterializations, 1u);
  EXPECT_EQ(baselineStatistics.baselineSourcePreparations, 1u);
  EXPECT_EQ(baselineStatistics.baselineMaterializationPreparations, 1u);
  EXPECT_EQ(baselineStatistics.baselineTileEntryMaterializations, 16u);
  EXPECT_EQ(baselineStatistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(baselineDiagnosticsText.find(
                "card-executable-compilation outcome=exact-rejection"),
            std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(
      baselineDiagnosticsText.find("card-exact-spm-conflict-certificate"),
      std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(
      baselineDiagnosticsText.find("tile-execution-allocation-feedback-joint"),
      std::string::npos)
      << baselineDiagnosticsText;
  for (llvm::StringRef tileDataflowIR : tileDataflowIRTrace) {
    EXPECT_NE(tileDataflowIR.find("scf.for"), llvm::StringRef::npos)
        << tileDataflowIR.str();
    const size_t stores = countOccurrences(tileDataflowIR, "wafer.tile.store");
    EXPECT_GT(stores, 1u);
    EXPECT_LE(stores, 81u);
    EXPECT_NE(tileDataflowIR.find("memref.dealloc"), llvm::StringRef::npos)
        << tileDataflowIR.str();
  }
}


} // namespace
