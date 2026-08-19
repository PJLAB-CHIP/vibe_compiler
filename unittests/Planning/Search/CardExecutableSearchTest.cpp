//===- CardExecutableSearchTest.cpp -----------------------------------===//

#include "Wafer/Planning/Search/CardExecutableSearch.h"
#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "gtest/gtest.h"

namespace {
using namespace wafer::compiler::testing;

TEST(CardExecutableSearchTest, CompleteCandidateRunsThroughTheSharedExactGate) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto baseline = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, programMetadata(), executionConfig(), diagnostics,
      programData, &statistics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << diagnosticsText;
  ASSERT_EQ(baseline->executable.tiles.size(), 16u);
  wafer::compiler::detail::CardExecutableSearchSummary summary;
  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, *baseline->programAnalysis,
      std::move(baseline->executable), programMetadata(), executionConfig(),
      diagnostics, programData,
      wafer::compiler::detail::SearchWorkBudget::bounded(1, 1),
      wafer::TargetMemoryPolicy{}, &summary);

  ASSERT_TRUE(mlir::succeeded(selected));
  ASSERT_EQ(selected->tiles.size(), 16u);
  EXPECT_EQ(summary.work.evaluated, 1u);
  EXPECT_EQ(summary.work.accepted + summary.work.exactRejected +
                summary.work.indeterminate,
            1u);
  EXPECT_EQ(
      summary.coverage,
      wafer::compiler::detail::CardExecutableSearchCoverage::BudgetedFeasible);
  EXPECT_EQ(statistics.baselineCardModuleMaterializations, 1u);
  EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations, 1u);
}

TEST(CardExecutableSearchTest,
     LargeStageBoundaryEvaluatesOneCapacityBoundedActualCandidate) {
  ParsedProgram parsed = parseLargeProducerStageProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto baseline = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, largeProducerStageProgramMetadata(), executionConfig(),
      diagnostics, programData);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << diagnosticsText;
  wafer::compiler::detail::CardExecutableSearchSummary summary;
  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, *baseline->programAnalysis,
      std::move(baseline->executable), largeProducerStageProgramMetadata(),
      executionConfig(), diagnostics, programData,
      wafer::compiler::detail::SearchWorkBudget::bounded(1, 1),
      wafer::TargetMemoryPolicy{}, &summary);

  ASSERT_TRUE(mlir::succeeded(selected));
  EXPECT_EQ(summary.work.evaluated, 1u);
  EXPECT_EQ(summary.work.accepted, 1u) << summary.lastDetail;
  EXPECT_EQ(summary.work.exactRejected, 0u) << summary.lastDetail;
  EXPECT_EQ(summary.work.indeterminate, 0u) << summary.lastDetail;
}

TEST(CardExecutableSearchTest,
     TransposedWeightEvaluatesOnlyItsSelectedOperandWindows) {
  ParsedProgram parsed = parseLargeTransposedWeightProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto baseline = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, largeTransposedWeightProgramMetadata(), executionConfig(),
      diagnostics, programData);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << diagnosticsText;
  wafer::compiler::detail::CardExecutableSearchSummary summary;
  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, *baseline->programAnalysis,
      std::move(baseline->executable), largeTransposedWeightProgramMetadata(),
      executionConfig(), diagnostics, programData,
      wafer::compiler::detail::SearchWorkBudget::bounded(1, 1),
      wafer::TargetMemoryPolicy{}, &summary);

  ASSERT_TRUE(mlir::succeeded(selected));
  EXPECT_EQ(summary.work.evaluated, 1u);
  EXPECT_EQ(summary.work.accepted, 1u) << "proposal=" << summary.proposalDetail
                                       << " candidate=" << summary.lastDetail;
  EXPECT_EQ(summary.work.exactRejected, 0u) << summary.lastDetail;
  EXPECT_EQ(summary.work.indeterminate, 0u) << summary.lastDetail;
}

TEST(CardExecutableSearchTest,
     CoupledNeighborhoodRepairsMovementAndFindsLowerDDRActualCandidate) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto baseline = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, dependentProgramMetadata(), executionConfig(),
      diagnostics, programData);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << diagnosticsText;
  ASSERT_TRUE(
      baseline->executable.resourceCost.aggregateDDRReadBytes.isKnown());
  const uint64_t baselineDDRRead =
      baseline->executable.resourceCost.aggregateDDRReadBytes.value;
  wafer::compiler::detail::CardExecutableSearchSummary summary;
  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, *baseline->programAnalysis,
      std::move(baseline->executable), dependentProgramMetadata(),
      executionConfig(), diagnostics, programData,
      wafer::compiler::detail::SearchWorkBudget::bounded(2, 2),
      wafer::TargetMemoryPolicy{}, &summary);

  ASSERT_TRUE(mlir::succeeded(selected));
  EXPECT_EQ(summary.work.evaluated, 2u);
  EXPECT_EQ(summary.work.accepted, 2u) << "proposal=" << summary.proposalDetail
                                       << " candidate=" << summary.lastDetail;
  ASSERT_TRUE(selected->resourceCost.aggregateDDRReadBytes.isKnown());
  EXPECT_LT(selected->resourceCost.aggregateDDRReadBytes.value,
            baselineDDRRead);
}

} // namespace
