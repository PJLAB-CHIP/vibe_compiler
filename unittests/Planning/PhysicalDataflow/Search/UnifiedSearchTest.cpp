//===- UnifiedSearchTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/UnifiedSearch.h"

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <set>

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
     ParentByParentPrefixMatchesRecursiveComposerAcrossEveryResumeCut) {
  constexpr size_t kCompleteLimit = 2;

  ParsedProgram referenceProgram = parseProgram();
  ASSERT_TRUE(referenceProgram.module);
  std::string referenceDiagnosticsText;
  llvm::raw_string_ostream referenceDiagnostics(referenceDiagnosticsText);
  std::string failureReason;
  SearchFixture referenceFixture =
      prepare(*referenceProgram.module, referenceDiagnostics, failureReason);
  ASSERT_TRUE(referenceFixture.session) << failureReason;
  ReferencePrefixComposer reference(*referenceFixture.session, kCompleteLimit);
  ASSERT_TRUE(reference.run()) << reference.failureDetail;
  ASSERT_EQ(reference.completeStates.size(), kCompleteLimit);

  ParsedProgram production = parseProgram();
  ASSERT_TRUE(production.module);
  const std::string sourceBefore = print(production.module->getOperation());
  std::string productionDiagnosticsText;
  llvm::raw_string_ostream productionDiagnostics(productionDiagnosticsText);
  SearchFixture productionFixture =
      prepare(*production.module, productionDiagnostics, failureReason);
  ASSERT_TRUE(productionFixture.session) << failureReason;
  wafer::compiler::ProgramDataHandoff productionData;
  UnifiedSearchOptions options;
  options.termination = SearchTerminationPolicy::Exhaustive;
  UnifiedSearchTrace trace;
  UnifiedSearchSession productionSearch(
      *production.module, *productionFixture.session, programMetadata(),
      executionConfig(), productionDiagnostics, productionData, options,
      /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/false, &trace);
  UnifiedSearchResumeResult zero = productionSearch.resume(0);
  EXPECT_EQ(zero.status, UnifiedSearchResumeStatus::Paused);
  EXPECT_EQ(zero.consumedCredits, 0u);
  for (uint64_t steps = 0;
       trace.candidates.size() < kCompleteLimit && steps < 10000; ++steps) {
    UnifiedSearchResumeResult resumed = productionSearch.resume(1);
    EXPECT_EQ(resumed.consumedCredits, 1u);
    if (resumed.status == UnifiedSearchResumeStatus::Indeterminate) {
      ASSERT_EQ(trace.candidates.size(), kCompleteLimit)
          << productionDiagnostics.str();
      break;
    }
    ASSERT_EQ(resumed.status, UnifiedSearchResumeStatus::Paused)
        << productionDiagnostics.str();
  }
  ASSERT_EQ(trace.candidates.size(), kCompleteLimit)
      << productionDiagnostics.str();
  ASSERT_EQ(trace.prefixes, reference.prefixes);

  std::vector<UnifiedSearchCandidateTrace> oracle;
  std::optional<CompleteCandidateKey> expectedWinner;
  for (const ScheduledState &state : reference.completeStates) {
    ParsedProgram fresh = parseProgram();
    ASSERT_TRUE(fresh.module);
    std::string actualDiagnosticsText;
    llvm::raw_string_ostream actualDiagnostics(actualDiagnosticsText);
    std::string actualFailure;
    SearchFixture actualFixture =
        prepare(*fresh.module, actualDiagnostics, actualFailure);
    ASSERT_TRUE(actualFixture.session) << actualFailure;
    auto key = CompleteCandidateKey::create(state, &actualFailure);
    ASSERT_TRUE(mlir::succeeded(key)) << actualFailure;
    wafer::compiler::ProgramDataHandoff actualData;
    FullFeasibilityStatistics statistics;
    FullFeasibilityResult actual =
        actualFixture.session->evaluateScheduledState(
            *fresh.module, state, programMetadata(), executionConfig(),
            actualDiagnostics, actualData, &statistics);
    EXPECT_EQ(statistics.evaluations, 1u) << actual.detail << "\n"
                                          << actualDiagnostics.str();
    oracle.push_back({*key, actual.status});
    if (actual.isAccepted() && (!expectedWinner || *key < *expectedWinner))
      expectedWinner = *key;
  }
  EXPECT_EQ(trace.candidates, oracle);

  // Oracle-B actualizations above must not perturb a later fresh traversal.
  ParsedProgram replay = parseProgram();
  ASSERT_TRUE(replay.module);
  std::string replayDiagnosticsText;
  llvm::raw_string_ostream replayDiagnostics(replayDiagnosticsText);
  SearchFixture replayFixture =
      prepare(*replay.module, replayDiagnostics, failureReason);
  ASSERT_TRUE(replayFixture.session) << failureReason;
  wafer::compiler::ProgramDataHandoff replayData;
  UnifiedSearchTrace replayTrace;
  UnifiedSearchSession replaySearch(*replay.module, *replayFixture.session,
                                    programMetadata(), executionConfig(),
                                    replayDiagnostics, replayData, options, 0,
                                    false, &replayTrace);
  for (uint64_t steps = 0;
       replayTrace.candidates.size() < kCompleteLimit && steps < 10000;
       ++steps) {
    UnifiedSearchResumeResult resumed = replaySearch.resume(1);
    if (resumed.status == UnifiedSearchResumeStatus::Indeterminate)
      break;
    ASSERT_EQ(resumed.status, UnifiedSearchResumeStatus::Paused)
        << replayDiagnostics.str();
  }
  EXPECT_EQ(replayTrace.candidates, trace.candidates)
      << replayDiagnostics.str();
  (void)replaySearch.finish();

  std::set<CompleteCandidateKey> uniqueKeys;
  for (const UnifiedSearchCandidateTrace &candidate : trace.candidates)
    EXPECT_TRUE(uniqueKeys.insert(candidate.key).second);

  UnifiedSearchResult result = productionSearch.finish();
  EXPECT_FALSE(result.frontierExhausted);
  EXPECT_EQ(result.planning.fullFeasibilityEvaluations, kCompleteLimit);
  EXPECT_EQ(result.control.statistics.reserved, kCompleteLimit);
  EXPECT_EQ(result.work.duplicateCompleteKeys, 0u);
  EXPECT_EQ(result.work.resumeCalls, result.work.successorSteps +
                                         result.work.scheduledStatesVisited +
                                         1);
  EXPECT_EQ(print(production.module->getOperation()), sourceBefore);
  EXPECT_EQ(static_cast<bool>(result.control.winner),
            expectedWinner.has_value());
  if (expectedWinner) {
    ASSERT_TRUE(result.control.winner);
    EXPECT_EQ(result.control.winner->key, *expectedWinner);
    EXPECT_EQ(trace.winnerHandoffs, 1u);
  }
  EXPECT_EQ(productionSearch.resume(1).status,
            UnifiedSearchResumeStatus::Finished);
}

} // namespace
