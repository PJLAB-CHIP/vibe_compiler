//===- UnifiedSearchTest.cpp ------------------------------------------===//

#include "Wafer/Driver/PhysicalDataflow/UnifiedSearch.h"
#include "Wafer/Driver/PhysicalDataflow/StructuredProgramAnalysis.h"

#include "Wafer/Planning/PhysicalDataflow/PlanningProfile.h"

#include "TestSupport/CodeGen/ExecutableTestSupport.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <optional>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;
using namespace wafer::compiler::testing;

struct SearchFixture {
  std::unique_ptr<StructuredProgramAnalysis> analysis;
  std::optional<PhysicalDataflowPlanningProblem> problem;
  std::unique_ptr<PhysicalDataflowPlanningSession> session;
};

SearchFixture prepare(mlir::ModuleOp module, llvm::raw_ostream &diagnostics,
                      std::string &failureReason,
                      PlanningProfileSink *profile = nullptr,
                      uint64_t maximumRegionProposals = 16) {
  SearchFixture result;
  auto analysis = analyzeStructuredProgram(module, programMetadata(),
                                           executionConfig(), diagnostics);
  if (mlir::failed(analysis))
    return result;
  result.analysis = std::move(*analysis);
  auto admission = PhysicalDataflowPlanningProblem::create(
      *result.analysis, CardId(0), analysis::IndexRelationLimits());
  auto *problem = std::get_if<PhysicalDataflowPlanningProblem>(&admission);
  if (!problem) {
    failureReason = std::get<SpatialDomainFailure>(admission).detail;
    return result;
  }
  result.problem.emplace(std::move(*problem));
  result.session = std::make_unique<PhysicalDataflowPlanningSession>(
      *result.problem, maximumRegionProposals, profile);
  return result;
}

RuntimeLaunchContract makeLaunch() {
  return llvm::cantFail(RuntimeLaunchContract::createKernel(
      KernelLaunchForm::Grid, KernelEntryABI::TileMajorPointerTable,
      {RuntimeLaunchPhaseRole::Main}));
}

class AcceptingEvaluator final : public StructuralCandidateEvaluator {
public:
  StructuralCandidateEvaluation
  evaluate(const RegionState &state, uint64_t actualizationCredits) override {
    EXPECT_GT(actualizationCredits, 0u);
    observed.push_back(StructuralCandidateKey::create(state));
    InstructionProgramAggregateCost cost;
    cost.aggregateInstructionCount.value = 1;
    ExecutableCompilationResult compilation;
    compilation.status = ExecutableCompilationStatus::Accepted;
    compilation.executable.emplace(std::vector<compiler::TileExecutable>{},
                                   makeLaunch(), std::move(cost));
    ActualCandidateResult result;
    result.status = ActualCandidateStatus::Accepted;
    result.compilation.emplace(std::move(compilation));
    return {std::move(result), 1};
  }

  std::vector<StructuralCandidateKey> observed;
};

class FirstInnerDomainIncompleteEvaluator final
    : public StructuralCandidateEvaluator {
public:
  StructuralCandidateEvaluation
  evaluate(const RegionState &state, uint64_t actualizationCredits) override {
    EXPECT_GT(actualizationCredits, 0u);
    observed.push_back(StructuralCandidateKey::create(state));
    if (observed.size() == 1) {
      ActualCandidateResult result;
      result.status = ActualCandidateStatus::Indeterminate;
      result.detail = "bounded inner current-IR traversal";
      return {std::move(result), 2, false};
    }
    AcceptingEvaluator accepting;
    return accepting.evaluate(state, actualizationCredits);
  }

  std::vector<StructuralCandidateKey> observed;
};

class FirstAcceptingThenUnsupportedEvaluator final
    : public StructuralCandidateEvaluator {
public:
  StructuralCandidateEvaluation
  evaluate(const RegionState &state, uint64_t actualizationCredits) override {
    EXPECT_GT(actualizationCredits, 0u);
    observed.push_back(StructuralCandidateKey::create(state));
    if (observed.size() == 1) {
      AcceptingEvaluator accepting;
      return accepting.evaluate(state, actualizationCredits);
    }
    ActualCandidateResult result;
    result.status = ActualCandidateStatus::Unsupported;
    result.detail = "test structural candidate is unsupported";
    return {std::move(result), 1, true};
  }

  std::vector<StructuralCandidateKey> observed;
};

std::string print(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  stream.flush();
  return text;
}

TEST(UnifiedSearchTest,
     StructuralTraversalUsesCallerOwnedActualizerAndRetainsItsOwner) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  const std::string before = print(parsed.module->getOperation());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
  ASSERT_TRUE(fixture.session) << failureReason;
  AcceptingEvaluator evaluator;
  UnifiedSearchOptions options;
  options.termination = SearchTerminationPolicy::FirstAccepted;
  UnifiedSearchTrace trace;
  UnifiedSearchResult result =
      runUnifiedSearch(*fixture.session, evaluator, options);

  ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
  ASSERT_EQ(evaluator.observed.size(), 1u);
  ASSERT_EQ(trace.candidates.size(), 0u);
  EXPECT_EQ(result.control.winner->key, evaluator.observed.front());
  EXPECT_TRUE(llvm::all_of(result.control.winner->key.getRegionPlan().groups,
                           [](const RegionGroupPlan &group) {
                             return group.mandatoryRoots.size() == 1 &&
                                    group.replicas.empty() &&
                                    group.localBindings.empty();
                           }));
  EXPECT_EQ(result.work.structuralStatesActualized, 1u);
  EXPECT_EQ(result.work.candidateActualizations, 1u);
  EXPECT_EQ(result.control.statistics.accepted, 1u);
  EXPECT_EQ(print(parsed.module->getOperation()), before);
}

TEST(UnifiedSearchTest,
     OneCreditResumeReachesTheSameFirstAcceptedStructuralKey) {
  std::vector<StructuralCandidateKey> winners;
  for (unsigned repetition = 0; repetition < 2; ++repetition) {
    ParsedProgram parsed = parseProgram();
    ASSERT_TRUE(parsed.module);
    std::string diagnosticsText;
    llvm::raw_string_ostream diagnostics(diagnosticsText);
    std::string failureReason;
    SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
    ASSERT_TRUE(fixture.session) << failureReason;
    AcceptingEvaluator evaluator;
    UnifiedSearchOptions options;
    options.termination = SearchTerminationPolicy::FirstAccepted;
    UnifiedSearchTrace trace;
    UnifiedSearchSession search(*fixture.session, evaluator, options, &trace);
    UnifiedSearchResumeStatus status = UnifiedSearchResumeStatus::Paused;
    for (uint64_t steps = 0;
         status == UnifiedSearchResumeStatus::Paused && steps < 10000; ++steps)
      status = search.resume(1).status;
    EXPECT_EQ(status, UnifiedSearchResumeStatus::AcceptedCheckpoint);
    ASSERT_EQ(trace.candidates.size(), 1u);
    UnifiedSearchResult result = search.finish();
    ASSERT_TRUE(result.control.winner);
    winners.push_back(result.control.winner->key);
  }
  EXPECT_EQ(winners[0], winners[1]);
}

TEST(UnifiedSearchTest,
     CreditExhaustionBeforeActualizationIsIncompleteNotNoSolution) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
  ASSERT_TRUE(fixture.session) << failureReason;
  AcceptingEvaluator evaluator;
  UnifiedSearchOptions options;
  options.planningCredits = 2;
  UnifiedSearchResult result =
      runUnifiedSearch(*fixture.session, evaluator, options);
  EXPECT_FALSE(result.hasWinner());
  EXPECT_FALSE(result.frontierExhausted);
  EXPECT_EQ(result.control.coverage,
            SearchControllerCoverage::IncompleteNoCandidate);
  EXPECT_TRUE(evaluator.observed.empty());
}

TEST(UnifiedSearchTest,
     StructuralCandidateBudgetRetainsTheActualIncumbentAsPartialCoverage) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
  ASSERT_TRUE(fixture.session) << failureReason;
  AcceptingEvaluator evaluator;
  UnifiedSearchOptions options;
  options.structuralCandidateCredits = 1;
  UnifiedSearchResult result =
      runUnifiedSearch(*fixture.session, evaluator, options);
  ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
  EXPECT_EQ(evaluator.observed.size(), 1u);
  EXPECT_EQ(result.work.structuralStatesActualized, 1u);
  EXPECT_FALSE(result.frontierExhausted);
  EXPECT_EQ(result.control.coverage, SearchControllerCoverage::FeasiblePartial);
  EXPECT_EQ(result.control.statistics.accepted, 1u);
  EXPECT_EQ(result.control.statistics.exhaustedReservations, 1u);
}

TEST(UnifiedSearchTest,
     ActualizationBudgetRetainsTheActualIncumbentWithoutStartingAnotherLeaf) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
  ASSERT_TRUE(fixture.session) << failureReason;
  AcceptingEvaluator evaluator;
  UnifiedSearchOptions options;
  options.structuralCandidateCredits = 4;
  options.candidateActualizationCredits = 1;
  UnifiedSearchResult result =
      runUnifiedSearch(*fixture.session, evaluator, options);
  ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
  EXPECT_EQ(evaluator.observed.size(), 1u);
  EXPECT_EQ(result.work.structuralStatesActualized, 1u);
  EXPECT_EQ(result.work.candidateActualizations, 1u);
  EXPECT_FALSE(result.frontierExhausted);
  EXPECT_EQ(result.control.coverage, SearchControllerCoverage::FeasiblePartial);
  EXPECT_EQ(result.control.statistics.accepted, 1u);
}

TEST(UnifiedSearchTest,
     InitialProposalsAppendBoundedNeighborsAroundTheActualIncumbent) {
  ParsedProgram parsed = parseThreeStageDependentProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture =
      prepare(*parsed.module, diagnostics, failureReason, nullptr, 2);
  ASSERT_TRUE(fixture.session) << failureReason;
  FirstAcceptingThenUnsupportedEvaluator evaluator;
  UnifiedSearchOptions options;
  options.structuralCandidateCredits = 3;
  options.candidateActualizationCredits = 3;
  options.maximumRegionRefinementCandidates = 1;
  UnifiedSearchResult result =
      runUnifiedSearch(*fixture.session, evaluator, options);
  ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
  ASSERT_EQ(evaluator.observed.size(), 3u);
  EXPECT_EQ(evaluator.observed[0].getRegionPlan().groups.size(), 48u);
  EXPECT_EQ(evaluator.observed[1].getRegionPlan().groups.size(), 47u);
  EXPECT_EQ(evaluator.observed[2].getRegionPlan().groups.size(), 16u);
  EXPECT_EQ(result.work.structuralStatesActualized, 3u);
  EXPECT_EQ(result.work.candidateActualizations, 3u);
  EXPECT_EQ(result.control.statistics.accepted, 1u);
  EXPECT_EQ(result.control.statistics.unsupported, 2u);
}

TEST(UnifiedSearchTest,
     IncompleteInnerDomainDoesNotDiscardTheRemainingStructuralFrontier) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
  ASSERT_TRUE(fixture.session) << failureReason;
  FirstInnerDomainIncompleteEvaluator evaluator;
  UnifiedSearchOptions options;
  options.termination = SearchTerminationPolicy::FirstAccepted;
  UnifiedSearchResult result =
      runUnifiedSearch(*fixture.session, evaluator, options);
  ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
  EXPECT_EQ(evaluator.observed.size(), 2u);
  EXPECT_EQ(result.work.incompleteInnerDomains, 1u);
  EXPECT_EQ(result.control.statistics.indeterminate, 1u);
  EXPECT_EQ(result.control.statistics.accepted, 1u);
  EXPECT_EQ(result.control.coverage, SearchControllerCoverage::FeasiblePartial);
}

TEST(UnifiedSearchTest, OptionalProfileDoesNotChangeTheFirstStructuralChoice) {
  std::vector<StructuralCandidateKey> winners;
  PlanningProfileSink profile;
  for (unsigned invocation = 0; invocation < 2; ++invocation) {
    ParsedProgram parsed = parseProgram();
    ASSERT_TRUE(parsed.module);
    std::string diagnosticsText;
    llvm::raw_string_ostream diagnostics(diagnosticsText);
    std::string failureReason;
    PlanningProfileSink *sink = invocation == 0 ? nullptr : &profile;
    SearchFixture fixture =
        prepare(*parsed.module, diagnostics, failureReason, sink);
    ASSERT_TRUE(fixture.session) << failureReason;
    AcceptingEvaluator evaluator;
    UnifiedSearchOptions options;
    options.termination = SearchTerminationPolicy::FirstAccepted;
    options.profile = sink;
    UnifiedSearchResult result =
        runUnifiedSearch(*fixture.session, evaluator, options);
    ASSERT_TRUE(result.control.winner);
    winners.push_back(result.control.winner->key);
  }
  EXPECT_EQ(winners[0], winners[1]);
  const PlanningProfileStatistics &statistics = profile.getStatistics();
  EXPECT_EQ(statistics.candidateActualizations, 1u);
  EXPECT_EQ(statistics.acceptedCandidates, 1u);
  EXPECT_EQ(statistics.winnerHandoffs, 1u);
  EXPECT_GT(statistics.peakFrontierDepth, 1u);
}

} // namespace
