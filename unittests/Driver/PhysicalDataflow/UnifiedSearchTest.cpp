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

class TestCandidateSession final : public StructuralCandidateSession {
public:
  TestCandidateSession(
      ActualCandidateResult result,
      CandidateContinuation next = CandidateContinuation::Exhausted)
      : result(std::move(result)), next(next) {}
  StructuralCandidateEvaluation advance() override {
    if (advanced)
      return {{}, 0, CandidateContinuation::Exhausted};
    advanced = true;
    return {std::move(result), 1, next};
  }

private:
  ActualCandidateResult result;
  CandidateContinuation next;
  bool advanced = false;
};

ActualCandidateResult acceptedResult(uint64_t count = 1) {
  InstructionProgramAggregateCost cost;
  cost.aggregateInstructionCount.value = count;
  ExecutableCompilationResult compilation;
  compilation.status = ExecutableCompilationStatus::Accepted;
  compilation.executable.emplace(std::vector<compiler::TileExecutable>{},
                                 makeLaunch(), std::move(cost));
  ActualCandidateResult result;
  result.status = ActualCandidateStatus::Accepted;
  result.compilation.emplace(std::move(compilation));
  return result;
}

class AcceptingEvaluator final : public StructuralCandidateEvaluator {
public:
  std::unique_ptr<StructuralCandidateSession>
  start(const RegionState &state) override {
    observed.push_back(StructuralCandidateKey::create(state));
    return std::make_unique<TestCandidateSession>(acceptedResult());
  }
  std::vector<StructuralCandidateKey> observed;
};

class FirstInnerDomainIncompleteEvaluator final
    : public StructuralCandidateEvaluator {
public:
  std::unique_ptr<StructuralCandidateSession>
  start(const RegionState &state) override {
    observed.push_back(StructuralCandidateKey::create(state));
    if (observed.size() == 1) {
      ActualCandidateResult result;
      result.status = ActualCandidateStatus::Indeterminate;
      result.detail = "bounded inner current-IR traversal";
      return std::make_unique<TestCandidateSession>(
          std::move(result), CandidateContinuation::Explore);
    }
    return std::make_unique<TestCandidateSession>(acceptedResult());
  }
  std::vector<StructuralCandidateKey> observed;
};

class FirstAcceptingThenUnsupportedEvaluator final
    : public StructuralCandidateEvaluator {
public:
  std::unique_ptr<StructuralCandidateSession>
  start(const RegionState &state) override {
    observed.push_back(StructuralCandidateKey::create(state));
    if (observed.size() == 1)
      return std::make_unique<TestCandidateSession>(acceptedResult());
    ActualCandidateResult result;
    result.status = ActualCandidateStatus::Unsupported;
    result.detail = "test structural candidate is unsupported";
    return std::make_unique<TestCandidateSession>(std::move(result));
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

TEST(UnifiedSearchTest, RetentionWidthDoesNotLimitTotalVisitedStructures) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
  ASSERT_TRUE(fixture.session) << failureReason;
  AcceptingEvaluator evaluator;
  UnifiedSearchOptions options;
  options.retainedBranches = 1;
  options.candidateActualizationCredits = 3;
  UnifiedSearchResult result =
      runUnifiedSearch(*fixture.session, evaluator, options);
  ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
  EXPECT_EQ(evaluator.observed.size(), 3u);
  EXPECT_EQ(result.work.structuralStatesActualized, 3u);
  EXPECT_FALSE(result.frontierExhausted);
  EXPECT_EQ(result.control.coverage, SearchControllerCoverage::FeasiblePartial);
  EXPECT_EQ(result.control.statistics.accepted, 3u);
  EXPECT_EQ(result.control.statistics.exhaustedReservations, 0u);
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
  options.retainedBranches = 4;
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
  options.retainedBranches = 3;
  options.candidateActualizationCredits = 3;
  options.maximumRegionRefinementCandidates = 1;
  UnifiedSearchResult result =
      runUnifiedSearch(*fixture.session, evaluator, options);
  ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
  ASSERT_EQ(evaluator.observed.size(), 3u);
  EXPECT_EQ(evaluator.observed[0].getRegionPlan().groups.size(), 48u);
  // Spatial families interleave with Region successors. Refinement cannot
  // monopolize the first three actual evaluations.
  EXPECT_FALSE(evaluator.observed[0].getSpatialPlan() ==
               evaluator.observed[1].getSpatialPlan());
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
  EXPECT_EQ(result.work.incompleteInnerDomains, 0u);
  EXPECT_GT(result.work.resumedCandidates, 0u);
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

// A bounded independent scheduling oracle: values stand for already evaluated
// executables. Real IR/offset/ownership gates are tested in
// ExecutableCompilation.
struct AttemptLog {
  std::vector<std::pair<uint64_t, uint64_t>> visits;
  uint64_t live = 0;
  uint64_t peak = 0;
  uint64_t started = 0;
};

class ProgressiveSession final : public StructuralCandidateSession {
public:
  explicit ProgressiveSession(AttemptLog &log) : log(log), id(log.started++) {
    log.peak = std::max(log.peak, ++log.live);
  }
  ~ProgressiveSession() override { --log.live; }
  StructuralCandidateEvaluation advance() override {
    log.visits.emplace_back(id, step);
    if (step++ < 2) {
      ActualCandidateResult result;
      result.status = ActualCandidateStatus::Indeterminate;
      return {std::move(result), 1, CandidateContinuation::Repair};
    }
    const uint64_t counts[] = {90, 50, 75, 40};
    return {acceptedResult(counts[step - 3] + id), 1,
            step == 6 ? CandidateContinuation::Exhausted
                      : CandidateContinuation::Improve};
  }

private:
  AttemptLog &log;
  uint64_t id;
  uint64_t step = 0;
};

class ProgressiveEvaluator final : public StructuralCandidateEvaluator {
public:
  explicit ProgressiveEvaluator(AttemptLog &log) : log(log) {}
  std::unique_ptr<StructuralCandidateSession>
  start(const RegionState &) override {
    return std::make_unique<ProgressiveSession>(log);
  }

private:
  AttemptLog &log;
};

class ExploringEvaluator final : public StructuralCandidateEvaluator {
  class Session final : public StructuralCandidateSession {
  public:
    StructuralCandidateEvaluation advance() override {
      if (!visited) {
        visited = true;
        ActualCandidateResult result;
        result.status = ActualCandidateStatus::Unsupported;
        return {std::move(result), 1, CandidateContinuation::Explore};
      }
      return {acceptedResult(), 1, CandidateContinuation::Exhausted};
    }

  private:
    bool visited = false;
  };

public:
  std::unique_ptr<StructuralCandidateSession>
  start(const RegionState &) override {
    return std::make_unique<Session>();
  }
};

TEST(UnifiedSearchTest, ExploresParametersBeforeAnyFeasibleLeafExists) {
  for (uint64_t width : {1, 3, 8}) {
    auto parsed = parseProgram();
    ASSERT_TRUE(parsed.module);
    std::string detail, output;
    llvm::raw_string_ostream diagnostics(output);
    auto fixture = prepare(*parsed.module, diagnostics, detail);
    ASSERT_TRUE(fixture.session) << detail;
    ExploringEvaluator evaluator;
    UnifiedSearchOptions options;
    options.retainedBranches = width;
    options.candidateActualizationCredits = 14;
    options.costCohort = *SearchCostCohort::create(SearchCostPolicy{});
    auto result = runUnifiedSearch(*fixture.session, evaluator, options);
    ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
    EXPECT_GT(result.work.resumedCandidates, 0u);
    EXPECT_EQ(result.control.statistics.unsupported,
              result.control.statistics.accepted);
    EXPECT_EQ(result.work.candidateActualizations, 14u);
  }
}

TEST(UnifiedSearchTest, ResumptionPreservesPrefixAndBestAcrossBudgets) {
  for (uint64_t width : {1, 3, 8}) {
    std::vector<std::pair<uint64_t, uint64_t>> previous;
    uint64_t previousBest = std::numeric_limits<uint64_t>::max();
    for (uint64_t trials : {14, 42, 126}) {
      SCOPED_TRACE(::testing::Message()
                   << "width=" << width << ", trials=" << trials);
      auto parsed = parseProgram();
      ASSERT_TRUE(parsed.module);
      std::string detail, output;
      llvm::raw_string_ostream diagnostics(output);
      auto fixture = prepare(*parsed.module, diagnostics, detail);
      ASSERT_TRUE(fixture.session) << detail;
      AttemptLog log;
      ProgressiveEvaluator evaluator(log);
      UnifiedSearchOptions options;
      options.retainedBranches = width;
      options.candidateActualizationCredits = trials;
      options.costCohort = *SearchCostCohort::create(SearchCostPolicy{});
      auto result = runUnifiedSearch(*fixture.session, evaluator, options);
      ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
      ASSERT_EQ(log.visits.size(), trials);
      ASSERT_GE(log.visits.size(), previous.size());
      EXPECT_TRUE(
          std::equal(previous.begin(), previous.end(), log.visits.begin()));
      EXPECT_EQ(log.live, 0u);
      EXPECT_LE(log.peak, width);
      EXPECT_EQ(result.work.peakRetainedBranches, log.peak);
      EXPECT_GT(result.work.resumedCandidates, 0u);
      EXPECT_EQ(result.work.candidateActualizations, trials);
      const auto *objective =
          std::get_if<KnownSearchObjective>(&result.control.winner->objective);
      ASSERT_NE(objective, nullptr);
      EXPECT_LE(objective->estimatedDurationPicoseconds, previousBest);
      previousBest = objective->estimatedDurationPicoseconds;
      previous = log.visits;
      for (uint64_t id = 0; id < log.started; ++id) {
        uint64_t expectedStep = 0;
        for (auto [owner, step] : log.visits)
          if (owner == id) {
            EXPECT_EQ(step, expectedStep++);
          }
      }
    }
  }
}

TEST(UnifiedSearchTest,
     StructuralExplorationDoesNotRestartAnActiveRepairChain) {
  auto parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string detail, output;
  llvm::raw_string_ostream diagnostics(output);
  auto fixture = prepare(*parsed.module, diagnostics, detail);
  ASSERT_TRUE(fixture.session) << detail;
  AttemptLog log;
  ProgressiveEvaluator evaluator(log);
  UnifiedSearchOptions options;
  options.retainedBranches = 2;
  options.candidateActualizationCredits = 14;
  options.costCohort = *SearchCostCohort::create(SearchCostPolicy{});
  UnifiedSearchSession session(*fixture.session, evaluator, options);
  while (session.resume(1).status == UnifiedSearchResumeStatus::Paused) {
  }
  auto result = session.finish();
  ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
  ASSERT_GE(log.visits.size(), 5u);
  EXPECT_EQ(log.visits[0], std::make_pair(uint64_t(0), uint64_t(0)));
  EXPECT_EQ(log.visits[1], std::make_pair(uint64_t(0), uint64_t(1)));
  EXPECT_EQ(log.visits[2], std::make_pair(uint64_t(1), uint64_t(0)));
  EXPECT_EQ(log.visits[3], std::make_pair(uint64_t(0), uint64_t(2)));
  EXPECT_EQ(log.visits[4], std::make_pair(uint64_t(0), uint64_t(3)));
}

TEST(UnifiedSearchTest, QueuedRepairSurvivesInterleavedLocalExploration) {
  class Evaluator final : public StructuralCandidateEvaluator {
    class Session final : public StructuralCandidateSession {
    public:
      StructuralCandidateEvaluation advance() override {
        if (++visits < 4) {
          ActualCandidateResult result;
          result.status = ActualCandidateStatus::Indeterminate;
          return {std::move(result), 1, CandidateContinuation::Explore,
                  CandidateRetention::PendingCapacityRepair};
        }
        return {acceptedResult(), 1, CandidateContinuation::Exhausted};
      }

    private:
      unsigned visits = 0;
    };

  public:
    std::unique_ptr<StructuralCandidateSession>
    start(const RegionState &) override {
      return std::make_unique<Session>();
    }
  } evaluator;
  auto parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string detail, output;
  llvm::raw_string_ostream diagnostics(output);
  auto fixture = prepare(*parsed.module, diagnostics, detail);
  ASSERT_TRUE(fixture.session) << detail;
  UnifiedSearchOptions options;
  options.retainedBranches = 1;
  options.candidateActualizationCredits = 4;
  options.costCohort = *SearchCostCohort::create(SearchCostPolicy{});
  UnifiedSearchSession session(*fixture.session, evaluator, options);
  while (session.resume(1).status == UnifiedSearchResumeStatus::Paused) {
  }
  auto result = session.finish();
  ASSERT_TRUE(result.hasWinner()) << result.failureDetail;
  EXPECT_EQ(result.work.structuralStatesActualized, 1u);
  EXPECT_EQ(result.work.resumedCandidates, 3u);
  EXPECT_EQ(result.work.retiredBranches, 0u);
}

} // namespace
