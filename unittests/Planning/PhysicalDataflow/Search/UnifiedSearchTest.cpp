//===- UnifiedSearchTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/UnifiedSearch.h"

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;
using namespace wafer::compiler::testing;

std::string print(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  return text;
}

struct SearchFixture {
  std::unique_ptr<CardProgramAnalysis> analysis;
  std::optional<PhysicalDataflowPlanningProblem> problem;
  std::unique_ptr<PhysicalDataflowPlanningSession> session;
};

SearchFixture prepare(mlir::ModuleOp module, llvm::raw_ostream &diagnostics,
                      std::string &failureReason) {
  SearchFixture result;
  auto analysis = analyzeCardProgram(module, programMetadata(),
                                     executionConfig(), diagnostics);
  if (mlir::failed(analysis))
    return result;
  result.analysis = std::move(*analysis);
  auto problem = PhysicalDataflowPlanningProblem::create(
      *result.analysis, CardId(0), analysis::IndexRelationLimits(),
      &failureReason);
  if (mlir::failed(problem))
    return result;
  result.problem.emplace(std::move(*problem));
  result.session =
      std::make_unique<PhysicalDataflowPlanningSession>(*result.problem);
  return result;
}

TEST(UnifiedSearchTest,
     CompleteTypedTraversalRetainsOneActualWinnerWithoutRebuilding) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  const std::string before = print(parsed.module->getOperation());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
  ASSERT_TRUE(fixture.session) << failureReason;
  wafer::compiler::ProgramDataHandoff programData;
  UnifiedSearchOptions options;
  options.stopAfterFirstAccepted = true;
  UnifiedSearchResult searched =
      runUnifiedSearch(*parsed.module, *fixture.session, programMetadata(),
                       executionConfig(), diagnostics, programData, options,
                       /*tilePipelineParallelism=*/0,
                       /*captureTileDataflowIRTrace=*/true);
  ASSERT_TRUE(searched.hasWinner()) << searched.failureDetail;
  EXPECT_EQ(searched.control.coverage,
            SearchControllerCoverage::FeasiblePartial);
  EXPECT_EQ(searched.work.candidateActualizations, 1u);
  EXPECT_GE(searched.planning.eventGraphsBuilt, 1u);
  EXPECT_GE(searched.planning.postStructureEventGraphsBuilt, 1u);
  EXPECT_EQ(searched.control.statistics.accepted, 1u);
  EXPECT_GT(searched.work.successorSteps, 8u);
  EXPECT_EQ(print(parsed.module->getOperation()), before);
  std::vector<std::string> traces =
      searched.control.winner->compilation.tileDataflowIRTrace;
  CardExecutableLoweringResult executable =
      searched.control.winner->takeExecutable();
  expectCompleteTileDomain(executable, traces);
}

TEST(UnifiedSearchTest,
     WorkExhaustionBeforeACompleteStateIsIncompleteNotNoSolution) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
  ASSERT_TRUE(fixture.session) << failureReason;
  wafer::compiler::ProgramDataHandoff programData;
  UnifiedSearchOptions options;
  options.planningCredits = 3;
  options.stopAfterFirstAccepted = false;
  UnifiedSearchResult searched =
      runUnifiedSearch(*parsed.module, *fixture.session, programMetadata(),
                       executionConfig(), diagnostics, programData, options);
  EXPECT_FALSE(searched.hasWinner());
  EXPECT_FALSE(searched.frontierExhausted);
  EXPECT_EQ(searched.control.coverage,
            SearchControllerCoverage::IncompleteNoCandidate);
  EXPECT_EQ(searched.work.candidateActualizations, 0u);
  EXPECT_LE(searched.work.successorSteps, 3u);
}

TEST(UnifiedSearchTest,
     RepeatedSessionsChooseTheSameSemanticWinnerAndActualIR) {
  std::vector<CompleteCandidateKey> keys;
  std::vector<std::string> traces;
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    ParsedProgram parsed = parseProgram();
    ASSERT_TRUE(parsed.module);
    std::string diagnosticsText;
    llvm::raw_string_ostream diagnostics(diagnosticsText);
    std::string failureReason;
    SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
    ASSERT_TRUE(fixture.session) << failureReason;
    wafer::compiler::ProgramDataHandoff programData;
    UnifiedSearchOptions options;
    options.stopAfterFirstAccepted = true;
    UnifiedSearchResult searched = runUnifiedSearch(
        *parsed.module, *fixture.session, programMetadata(), executionConfig(),
        diagnostics, programData, options, /*tilePipelineParallelism=*/1,
        /*captureTileDataflowIRTrace=*/true);
    ASSERT_TRUE(searched.control.winner) << searched.failureDetail;
    keys.push_back(searched.control.winner->key);
    std::string joined;
    for (const std::string &trace :
         searched.control.winner->compilation.tileDataflowIRTrace)
      joined += trace;
    traces.push_back(std::move(joined));
  }
  EXPECT_EQ(keys[0], keys[1]);
  EXPECT_EQ(traces[0], traces[1]);
}

} // namespace
