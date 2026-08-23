//===- CardExecutableSearchTest.cpp -----------------------------------===//

#include "Wafer/Planning/Search/CardExecutableSearch.h"
#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "gtest/gtest.h"

namespace {
using namespace wafer::compiler::testing;

TEST(CardExecutableSearchTest,
     EmptyEvaluationBudgetFailsWithoutCompilingABaselineFallback) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  wafer::compiler::detail::CardExecutableSearchSummary summary;
  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, programMetadata(), executionConfig(), diagnostics,
      programData, wafer::compiler::detail::SearchWorkBudget::bounded(0),
      &summary);

  diagnostics.flush();
  EXPECT_TRUE(mlir::failed(selected));
  EXPECT_EQ(summary.work.evaluated, 0u);
  EXPECT_EQ(summary.work.accepted, 0u);
  EXPECT_EQ(summary.winnerUpdates, 0u);
  EXPECT_EQ(diagnosticsText.find("card-executable-compilation outcome="),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSearchTest, CompleteCandidateRunsThroughTheSharedExactGate) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  wafer::compiler::detail::CardExecutableSearchSummary summary;
  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, programMetadata(), executionConfig(), diagnostics,
      programData, wafer::compiler::detail::SearchWorkBudget::bounded(1),
      &summary);

  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(selected)) << diagnosticsText;
  ASSERT_EQ(selected->executable.tiles.size(), 16u);
  EXPECT_EQ(summary.work.evaluated, 1u);
  EXPECT_EQ(summary.work.accepted + summary.work.exactRejected +
                summary.work.indeterminate,
            1u);
  EXPECT_EQ(
      summary.coverage,
      wafer::compiler::detail::CardExecutableSearchCoverage::BudgetedFeasible);
  EXPECT_EQ(summary.winnerUpdates, 1u);
}

TEST(CardExecutableSearchTest,
     LargeStageBoundaryReportsActualRejectionWithoutBaselineFallback) {
  ParsedProgram parsed = parseLargeProducerStageProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  wafer::compiler::detail::CardExecutableSearchSummary summary;
  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, largeProducerStageProgramMetadata(), executionConfig(),
      diagnostics, programData,
      wafer::compiler::detail::SearchWorkBudget::bounded(1), &summary);

  diagnostics.flush();
  EXPECT_TRUE(mlir::failed(selected));
  EXPECT_EQ(summary.work.evaluated, 1u);
  EXPECT_EQ(summary.work.accepted, 0u) << summary.lastDetail;
  EXPECT_EQ(summary.work.exactRejected, 1u) << summary.lastDetail;
  EXPECT_EQ(summary.work.indeterminate, 0u) << summary.lastDetail;
  EXPECT_EQ(summary.winnerUpdates, 0u);
}

TEST(CardExecutableSearchTest,
     TransposedWeightReportsActualRejectionWithoutBaselineFallback) {
  ParsedProgram parsed = parseLargeTransposedWeightProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  wafer::compiler::detail::CardExecutableSearchSummary summary;
  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, largeTransposedWeightProgramMetadata(), executionConfig(),
      diagnostics, programData,
      wafer::compiler::detail::SearchWorkBudget::bounded(1), &summary);

  diagnostics.flush();
  EXPECT_TRUE(mlir::failed(selected));
  EXPECT_EQ(summary.work.evaluated, 1u);
  EXPECT_EQ(summary.work.accepted, 0u) << "proposal=" << summary.proposalDetail
                                       << " candidate=" << summary.lastDetail;
  EXPECT_EQ(summary.work.exactRejected, 1u) << summary.lastDetail;
  EXPECT_EQ(summary.work.indeterminate, 0u) << summary.lastDetail;
  EXPECT_EQ(summary.winnerUpdates, 0u);
}

TEST(CardExecutableSearchTest,
     KeepsFailedCoupledSiblingTypedAndRetainsAnAcceptedSearchOwner) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff baselineProgramData;
  auto baseline = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, dependentProgramMetadata(), executionConfig(),
      diagnostics, baselineProgramData, /*baselineStatistics=*/nullptr,
      /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << diagnosticsText;
  ASSERT_TRUE(
      baseline->executable.resourceCost.aggregateDDRReadBytes.isKnown());
  const uint64_t baselineDDRRead =
      baseline->executable.resourceCost.aggregateDDRReadBytes.value;
  ASSERT_EQ(baseline->tileDataflowIRTrace.size(), 16u);
  const std::vector<std::string> baselineTrace = baseline->tileDataflowIRTrace;
  wafer::compiler::ProgramDataHandoff searchProgramData;
  wafer::compiler::detail::CardExecutableSearchSummary summary;
  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, dependentProgramMetadata(), executionConfig(),
      diagnostics, searchProgramData,
      wafer::compiler::detail::SearchWorkBudget::bounded(2), &summary,
      /*captureTileDataflowIRTrace=*/true);

  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(selected)) << diagnosticsText;
  EXPECT_EQ(summary.work.evaluated, 2u);
  EXPECT_EQ(summary.work.accepted, 1u) << "proposal=" << summary.proposalDetail
                                       << " candidate=" << summary.lastDetail;
  EXPECT_EQ(summary.work.accepted + summary.work.exactRejected +
                summary.work.indeterminate,
            2u);
  ASSERT_TRUE(
      selected->executable.resourceCost.aggregateDDRReadBytes.isKnown());
  EXPECT_LE(selected->executable.resourceCost.aggregateDDRReadBytes.value,
            baselineDDRRead);
  EXPECT_GT(summary.winnerUpdates, 0u);
  EXPECT_EQ(selected->tileDataflowIRTrace.size(),
            selected->executable.tiles.size());
  EXPECT_NE(selected->tileDataflowIRTrace, baselineTrace);
}

} // namespace
