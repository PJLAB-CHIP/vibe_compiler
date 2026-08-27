//===- UnifiedSearchTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/UnifiedSearch.h"

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProfile.h"

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <set>
#include <type_traits>

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
                      std::string &failureReason,
                      const frontend::FrontendProgramVerificationResult
                          &metadata = programMetadata()) {
  SearchFixture result;
  auto analysis =
      analyzeCardProgram(module, metadata, executionConfig(), diagnostics);
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

/// Test-only recursive composer. It deliberately does not call
/// UnifiedSearchSession, ActualResultController, or Q50.F; each axis still
/// uses its real production continuation whose local domain has a direct
/// independent oracle in the owning test suite.
class ReferencePrefixComposer {
public:
  ReferencePrefixComposer(PhysicalDataflowPlanningSession &session,
                          size_t completeLimit)
      : session(session), completeLimit(completeLimit) {}

  bool run() {
    while (!done()) {
      SpatialExpansionResult next = session.resumeSpatial();
      if (next.getKind() == SpatialExpansionKind::ParentExhausted)
        return true;
      if (next.getKind() == SpatialExpansionKind::Unsupported)
        continue;
      if (next.getKind() != SpatialExpansionKind::StateQueued)
        return fail(next.getDetail());
      std::optional<SpatialState> state = session.takeNextSpatialState();
      if (!state)
        return fail("reference spatial frontier lost its state");
      record(*state);
      if (!regions(std::move(*state)))
        return false;
    }
    return true;
  }

  std::vector<UnifiedSearchPrefixKey> prefixes;
  std::vector<ScheduledState> completeStates;
  std::string failureDetail;

private:
  template <typename State> void record(const State &state) {
    if constexpr (std::is_same_v<State, SpatialState> ||
                  std::is_same_v<State, RegionState> ||
                  std::is_same_v<State, TemporalState>)
      prefixes.emplace_back(state);
  }

  bool done() const { return completeStates.size() >= completeLimit; }

  bool fail(llvm::StringRef detail) {
    failureDetail = detail.str();
    return false;
  }

  bool regions(SpatialState state) {
    RegionContinuation continuation =
        session.createRegionContinuation(std::move(state));
    while (!done()) {
      std::string detail;
      auto next = session.resumeRegion(continuation, &detail);
      if (mlir::failed(next))
        return fail(detail);
      if (!*next)
        return true;
      record(**next);
      if (!temporals(std::move(**next)))
        return false;
    }
    return true;
  }

  bool temporals(RegionState state) {
    TemporalContinuation continuation =
        session.createTemporalContinuation(std::move(state));
    while (!done()) {
      TemporalExpansionResult next = session.resumeTemporal(continuation);
      if (next.getKind() == TemporalExpansionKind::ParentExhausted)
        return true;
      if (next.getKind() == TemporalExpansionKind::Unsupported) {
        if (continuation.isExhausted())
          return true;
        continue;
      }
      if (next.getKind() != TemporalExpansionKind::State)
        return fail(next.getDetail());
      std::optional<TemporalState> child = next.takeState();
      if (!child)
        return fail("reference temporal continuation lost its state");
      record(*child);
      if (!representations(std::move(*child)))
        return false;
    }
    return true;
  }

  bool representations(TemporalState state) {
    RepresentationContinuation continuation =
        session.createRepresentationContinuation(std::move(state));
    while (!done()) {
      RepresentationExpansionResult next =
          session.resumeRepresentation(continuation);
      if (next.getKind() == RepresentationExpansionKind::ParentExhausted ||
          next.getKind() == RepresentationExpansionKind::Unsupported)
        return true;
      if (next.getKind() != RepresentationExpansionKind::State)
        return fail(next.getDetail());
      std::optional<RepresentationState> child = next.takeState();
      if (!child)
        return fail("reference representation continuation lost its state");
      record(*child);
      if (!movements(std::move(*child)))
        return false;
    }
    return true;
  }

  bool movements(RepresentationState state) {
    MovementContinuation continuation =
        session.createMovementContinuation(std::move(state));
    while (!done()) {
      MovementExpansionResult next = session.resumeMovement(continuation);
      if (next.getKind() == MovementExpansionKind::ParentExhausted ||
          next.getKind() == MovementExpansionKind::Unsupported)
        return true;
      if (next.getKind() != MovementExpansionKind::State)
        return fail(next.getDetail());
      std::optional<MovementState> child = next.takeState();
      if (!child)
        return fail("reference movement continuation lost its state");
      record(*child);
      if (!initialStorage(std::move(*child)))
        return false;
    }
    return true;
  }

  bool initialStorage(MovementState state) {
    StorageContinuation continuation =
        session.createStorageContinuation(std::move(state));
    while (!done()) {
      StorageExpansionResult next = session.resumeStorage(continuation);
      if (next.getKind() == StorageExpansionKind::ParentExhausted ||
          next.getKind() == StorageExpansionKind::Unsupported)
        return true;
      if (next.getKind() != StorageExpansionKind::State)
        return fail(next.getDetail());
      std::optional<InitialBufferState> child = next.takeState();
      if (!child)
        return fail("reference initial storage continuation lost its state");
      record(*child);
      if (!structures(std::move(*child)))
        return false;
    }
    return true;
  }

  bool structures(InitialBufferState state) {
    ExecutionStructureContinuation continuation =
        session.createExecutionStructureContinuation(std::move(state));
    while (!done()) {
      std::string detail;
      auto next = session.resumeExecutionStructure(continuation, &detail);
      if (mlir::failed(next))
        return fail(detail);
      if (!*next)
        return true;
      record(**next);
      if (!postStructureStorage(std::move(**next)))
        return false;
    }
    return true;
  }

  bool postStructureStorage(ExecutionStructureState state) {
    StructureSpecificStorageContinuation continuation =
        session.createStructureSpecificStorageContinuation(std::move(state));
    while (!done()) {
      std::string detail;
      auto next = session.resumeStructureSpecificStorage(continuation, &detail);
      if (mlir::failed(next))
        return fail(detail);
      if (!*next)
        return true;
      record(**next);
      if (!schedules(std::move(**next)))
        return false;
    }
    return true;
  }

  bool schedules(BufferState state) {
    ScheduleContinuation continuation =
        session.createScheduleContinuation(std::move(state));
    while (!done()) {
      std::string detail;
      auto next = session.resumeSchedule(continuation, &detail);
      if (mlir::failed(next))
        return fail(detail);
      if (!*next)
        return true;
      record(**next);
      completeStates.push_back(std::move(**next));
    }
    return true;
  }

  PhysicalDataflowPlanningSession &session;
  size_t completeLimit;
};

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
  options.termination = SearchTerminationPolicy::FirstAccepted;
  UnifiedSearchResult searched =
      runUnifiedSearch(*parsed.module, *fixture.session, programMetadata(),
                       executionConfig(), diagnostics, programData, options,
                       /*tilePipelineParallelism=*/0,
                       /*captureTileDataflowIRTrace=*/true);
  ASSERT_TRUE(searched.hasWinner()) << searched.failureDetail;
  EXPECT_EQ(searched.control.coverage,
            SearchControllerCoverage::FeasiblePartial);
  EXPECT_EQ(searched.work.candidateActualizations, 1u);
  EXPECT_EQ(searched.work.structuralStatesActualized, 1u);
  EXPECT_EQ(searched.work.scheduledStatesVisited, 0u);
  EXPECT_EQ(searched.planning.representationStatesQueued, 0u);
  EXPECT_EQ(searched.planning.movementStatesQueued, 0u);
  EXPECT_EQ(searched.planning.storageStatesQueued, 0u);
  EXPECT_EQ(searched.planning.executionStructureStatesQueued, 0u);
  EXPECT_EQ(searched.planning.scheduleStatesQueued, 0u);
  EXPECT_EQ(searched.planning.eventGraphsBuilt, 0u);
  EXPECT_EQ(searched.planning.postStructureEventGraphsBuilt, 0u)
      << "current search must not build a future EventGraph";
  EXPECT_EQ(searched.control.statistics.accepted, 1u);
  EXPECT_GE(searched.work.successorSteps, 3u);
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
  options.termination = SearchTerminationPolicy::Exhaustive;
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

TEST(UnifiedSearchTest, OneCreditResumeReachesTheSameFirstAcceptedCheckpoint) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::string failureReason;
  SearchFixture fixture = prepare(*parsed.module, diagnostics, failureReason);
  ASSERT_TRUE(fixture.session) << failureReason;
  wafer::compiler::ProgramDataHandoff programData;
  UnifiedSearchOptions options;
  options.termination = SearchTerminationPolicy::FirstAccepted;
  UnifiedSearchTrace trace;
  UnifiedSearchSession search(*parsed.module, *fixture.session,
                              programMetadata(), executionConfig(), diagnostics,
                              programData, options, 0, false, &trace);
  UnifiedSearchResumeStatus status = UnifiedSearchResumeStatus::Paused;
  for (uint64_t steps = 0;
       status == UnifiedSearchResumeStatus::Paused && steps < 10000; ++steps)
    status = search.resume(1).status;
  EXPECT_EQ(status, UnifiedSearchResumeStatus::AcceptedCheckpoint)
      << diagnostics.str();
  ASSERT_EQ(trace.candidates.size(), 1u);
  UnifiedSearchResult result = search.finish();
  EXPECT_TRUE(result.control.winner);
}

TEST(UnifiedSearchTest,
     RepeatedSessionsChooseTheSameSemanticWinnerAndActualIR) {
  std::vector<StructuralCandidateKey> keys;
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
    options.termination = SearchTerminationPolicy::FirstAccepted;
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

TEST(UnifiedSearchTest,
     OptionalPlanningProfileObservesButDoesNotChangeTheAcceptedWinner) {
  std::vector<StructuralCandidateKey> keys;
  std::vector<std::string> traces;
  PlanningProfileSink profile;
  for (unsigned invocation = 0; invocation < 2; ++invocation) {
    ParsedProgram parsed = parseProgram();
    ASSERT_TRUE(parsed.module);
    std::string diagnosticsText;
    llvm::raw_string_ostream diagnostics(diagnosticsText);
    std::string failureReason;
    auto programAnalysis = analyzeCardProgram(*parsed.module, programMetadata(),
                                              executionConfig(), diagnostics);
    ASSERT_TRUE(mlir::succeeded(programAnalysis)) << diagnostics.str();
    auto problem = PhysicalDataflowPlanningProblem::create(
        **programAnalysis, CardId(0), analysis::IndexRelationLimits(),
        &failureReason);
    ASSERT_TRUE(mlir::succeeded(problem)) << failureReason;
    PlanningProfileSink *sink = invocation == 0 ? nullptr : &profile;
    PhysicalDataflowPlanningSession session(*problem, sink);
    wafer::compiler::ProgramDataHandoff programData;
    UnifiedSearchOptions options;
    options.termination = SearchTerminationPolicy::FirstAccepted;
    options.profile = sink;
    UnifiedSearchResult searched =
        runUnifiedSearch(*parsed.module, session, programMetadata(),
                         executionConfig(), diagnostics, programData, options,
                         /*tilePipelineParallelism=*/0,
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
  const PlanningProfileStatistics &statistics = profile.getStatistics();
  EXPECT_EQ(statistics.candidateActualizations, 1u);
  EXPECT_EQ(statistics.acceptedCandidates, 1u);
  EXPECT_EQ(statistics.winnerHandoffs, 1u);
  EXPECT_TRUE(statistics.timeToFirstAcceptedMilliseconds.has_value());
  EXPECT_GT(statistics.peakFrontierDepth, 1u);
  uint64_t totalHits = 0;
  uint64_t totalMisses = 0;
  for (const PlanningMemoProfile &memo : statistics.memos) {
    totalHits += memo.hits;
    totalMisses += memo.misses;
    EXPECT_EQ(memo.lookups, memo.hits + memo.misses);
  }
  EXPECT_GT(totalHits, 0u);
  EXPECT_GT(totalMisses, 0u);
}

TEST(UnifiedSearchTest,
     ParentByParentPrefixMatchesRecursiveComposerAcrossEveryResumeCut) {
  constexpr size_t kStructuralLimit = 2;
  struct RunSummary {
    std::vector<UnifiedSearchCandidateTrace> candidates;
    UnifiedSearchWork work;
    SearchControllerStatistics control;
    std::optional<StructuralCandidateKey> winner;
    std::string source;
  };
  auto run = [&]() -> RunSummary {
    RunSummary summary;
    ParsedProgram parsed = parseProgram();
    EXPECT_TRUE(parsed.module);
    if (!parsed.module)
      return summary;
    const std::string sourceBefore = print(parsed.module->getOperation());
    std::string diagnosticsText;
    llvm::raw_string_ostream diagnostics(diagnosticsText);
    std::string failureReason;
    SearchFixture fixture =
        prepare(*parsed.module, diagnostics, failureReason);
    EXPECT_TRUE(fixture.session) << failureReason;
    if (!fixture.session)
      return summary;
    wafer::compiler::ProgramDataHandoff data;
    UnifiedSearchOptions options;
    options.termination = SearchTerminationPolicy::Exhaustive;
    UnifiedSearchTrace trace;
    UnifiedSearchSession search(
        *parsed.module, *fixture.session, programMetadata(), executionConfig(),
        diagnostics, data, options, /*tilePipelineParallelism=*/0,
        /*captureTileDataflowIRTrace=*/false, &trace);
    for (uint64_t steps = 0;
         trace.candidates.size() < kStructuralLimit && steps < 10000; ++steps) {
      UnifiedSearchResumeResult resumed = search.resume(1);
      if (resumed.status != UnifiedSearchResumeStatus::Paused)
        break;
    }
    EXPECT_EQ(trace.candidates.size(), kStructuralLimit) << diagnosticsText;
    UnifiedSearchResult result = search.finish();
    summary.candidates = std::move(trace.candidates);
    summary.work = result.work;
    summary.control = result.control.statistics;
    if (result.control.winner)
      summary.winner = result.control.winner->key;
    summary.source = print(parsed.module->getOperation());
    EXPECT_EQ(summary.source, sourceBefore);
    return summary;
  };

  // The explicit one-credit resumable traversal must be deterministic and
  // must stop at structural keys; no future physical/schedule key participates.
  RunSummary first = run();
  ASSERT_EQ(first.candidates.size(), kStructuralLimit);
  std::set<StructuralCandidateKey> uniqueKeys;
  for (const UnifiedSearchCandidateTrace &candidate : first.candidates)
    EXPECT_TRUE(uniqueKeys.insert(candidate.key).second);
  EXPECT_EQ(first.control.reserved, kStructuralLimit);
  EXPECT_EQ(first.work.structuralStatesActualized, kStructuralLimit);
  EXPECT_EQ(first.work.scheduledStatesVisited, 0u);
  EXPECT_EQ(first.work.duplicateCompleteKeys, 0u);

  RunSummary replay = run();
  EXPECT_EQ(replay.candidates, first.candidates);
  EXPECT_EQ(replay.winner, first.winner);
  EXPECT_EQ(replay.source, first.source);
}

} // namespace
