//===- UnifiedSearch.cpp - Resumable physical-dataflow traversal ------===//

#include "Wafer/Planning/PhysicalDataflow/Search/UnifiedSearch.h"

#include "llvm/ADT/StringRef.h"

#include <limits>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {
namespace {

struct SpatialFrame {};

struct TemporalFeedbackFrame {
  TemporalState state;
  std::vector<SemanticRootKey> causalRoots;
};

using FrontierFrame =
    std::variant<SpatialFrame, RegionContinuation, TemporalContinuation,
                 RepresentationContinuation, MovementContinuation,
                 StorageContinuation, ExecutionStructureContinuation,
                 StructureSpecificStorageContinuation, ScheduleContinuation,
                 ScheduledState, TemporalFeedbackFrame>;

bool isTerminal(UnifiedSearchResumeStatus status) {
  return status != UnifiedSearchResumeStatus::Paused;
}

} // namespace

struct UnifiedSearchSession::Impl {
  Impl(mlir::ModuleOp tensorProgram,
       PhysicalDataflowPlanningSession &planningSession,
       const frontend::FrontendProgramVerificationResult &program,
       const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
       ProgramDataHandoff &programData, const UnifiedSearchOptions &options,
       unsigned tilePipelineParallelism, bool captureTileDataflowIRTrace,
       UnifiedSearchTrace *trace)
      : tensorProgram(tensorProgram), planningSession(planningSession),
        program(program), executionConfig(executionConfig),
        diagnostics(diagnostics), programData(programData),
        termination(options.termination),
        controller(ActualResultControllerOptions{
            std::numeric_limits<uint64_t>::max(), options.costCohort,
            ExactRejectionCachePolicy::Enabled}),
        tilePipelineParallelism(tilePipelineParallelism),
        captureTileDataflowIRTrace(captureTileDataflowIRTrace), trace(trace) {
    frontier.emplace_back(SpatialFrame{});
  }

  template <typename State> void recordPrefix(const State &state) {
    if (trace)
      trace->prefixes.emplace_back(state);
  }

  void fail(llvm::StringRef detail) {
    status = UnifiedSearchResumeStatus::CompilerBug;
    failureDetail = detail.str();
  }

  void pauseIndeterminate(llvm::StringRef detail) {
    status = UnifiedSearchResumeStatus::Indeterminate;
    failureDetail = detail.str();
  }

  void markUnsupportedAndPop() {
    sawUnsupportedPrefix = true;
    frontier.pop_back();
  }

  void stepSpatial() {
    ++work.successorSteps;
    SpatialExpansionResult expansion = planningSession.resumeSpatial();
    switch (expansion.getKind()) {
    case SpatialExpansionKind::StateQueued: {
      std::optional<SpatialState> state =
          planningSession.takeNextSpatialState();
      if (!state) {
        fail("spatial continuation lost its queued state");
        return;
      }
      recordPrefix(*state);
      frontier.emplace_back(
          planningSession.createRegionContinuation(std::move(*state)));
      return;
    }
    case SpatialExpansionKind::Unsupported:
      sawUnsupportedPrefix = true;
      return;
    case SpatialExpansionKind::Indeterminate:
      pauseIndeterminate(expansion.getDetail());
      return;
    case SpatialExpansionKind::ParentExhausted:
      frontier.pop_back();
      return;
    case SpatialExpansionKind::CompilerBug:
      fail(expansion.getDetail());
      return;
    }
  }

  void stepRegion(RegionContinuation &continuation) {
    ++work.successorSteps;
    std::string detail;
    auto next = planningSession.resumeRegion(continuation, &detail);
    if (mlir::failed(next)) {
      fail(detail.empty() ? "region continuation failed" : detail);
      return;
    }
    if (!*next) {
      frontier.pop_back();
      return;
    }
    recordPrefix(**next);
    frontier.emplace_back(
        planningSession.createTemporalContinuation(std::move(**next)));
  }

  void stepTemporal(TemporalContinuation &continuation) {
    ++work.successorSteps;
    TemporalExpansionResult next = planningSession.resumeTemporal(continuation);
    switch (next.getKind()) {
    case TemporalExpansionKind::State: {
      std::optional<TemporalState> state = next.takeState();
      if (!state) {
        fail("temporal continuation lost its state");
        return;
      }
      recordPrefix(*state);
      frontier.emplace_back(
          planningSession.createRepresentationContinuation(std::move(*state)));
      return;
    }
    case TemporalExpansionKind::Unsupported:
      sawUnsupportedPrefix = true;
      if (continuation.isExhausted())
        frontier.pop_back();
      return;
    case TemporalExpansionKind::Indeterminate:
      pauseIndeterminate(next.getDetail());
      return;
    case TemporalExpansionKind::ParentExhausted:
      frontier.pop_back();
      return;
    case TemporalExpansionKind::CompilerBug:
      fail(next.getDetail());
      return;
    }
  }

  void stepRepresentation(RepresentationContinuation &continuation) {
    ++work.successorSteps;
    RepresentationExpansionResult next =
        planningSession.resumeRepresentation(continuation);
    switch (next.getKind()) {
    case RepresentationExpansionKind::State: {
      std::optional<RepresentationState> state = next.takeState();
      if (!state) {
        fail("representation continuation lost its state");
        return;
      }
      recordPrefix(*state);
      frontier.emplace_back(
          planningSession.createMovementContinuation(std::move(*state)));
      return;
    }
    case RepresentationExpansionKind::Unsupported:
      markUnsupportedAndPop();
      return;
    case RepresentationExpansionKind::ParentExhausted:
      frontier.pop_back();
      return;
    case RepresentationExpansionKind::CompilerBug:
      fail(next.getDetail());
      return;
    }
  }

  void stepMovement(MovementContinuation &continuation) {
    ++work.successorSteps;
    MovementExpansionResult next = planningSession.resumeMovement(continuation);
    switch (next.getKind()) {
    case MovementExpansionKind::State: {
      std::optional<MovementState> state = next.takeState();
      if (!state) {
        fail("movement continuation lost its state");
        return;
      }
      recordPrefix(*state);
      frontier.emplace_back(
          planningSession.createStorageContinuation(std::move(*state)));
      return;
    }
    case MovementExpansionKind::Unsupported:
      markUnsupportedAndPop();
      return;
    case MovementExpansionKind::ParentExhausted:
      frontier.pop_back();
      return;
    case MovementExpansionKind::CompilerBug:
      fail(next.getDetail());
      return;
    }
  }

  void stepStorage(StorageContinuation &continuation) {
    ++work.successorSteps;
    StorageExpansionResult next = planningSession.resumeStorage(continuation);
    switch (next.getKind()) {
    case StorageExpansionKind::State: {
      std::optional<InitialBufferState> state = next.takeState();
      if (!state) {
        fail("initial storage continuation lost its state");
        return;
      }
      recordPrefix(*state);
      frontier.emplace_back(
          planningSession.createExecutionStructureContinuation(
              std::move(*state)));
      return;
    }
    case StorageExpansionKind::Unsupported:
      markUnsupportedAndPop();
      return;
    case StorageExpansionKind::Indeterminate:
      pauseIndeterminate(next.getDetail());
      return;
    case StorageExpansionKind::ParentExhausted:
      frontier.pop_back();
      return;
    case StorageExpansionKind::CompilerBug:
      fail(next.getDetail());
      return;
    }
  }

  void stepExecutionStructure(ExecutionStructureContinuation &continuation) {
    ++work.successorSteps;
    std::string detail;
    auto next = planningSession.resumeExecutionStructure(continuation, &detail);
    if (mlir::failed(next)) {
      fail(detail.empty() ? "execution-structure continuation failed" : detail);
      return;
    }
    if (!*next) {
      frontier.pop_back();
      return;
    }
    recordPrefix(**next);
    frontier.emplace_back(
        planningSession.createStructureSpecificStorageContinuation(
            std::move(**next)));
  }

  void
  stepStructureStorage(StructureSpecificStorageContinuation &continuation) {
    ++work.successorSteps;
    std::string detail;
    auto next =
        planningSession.resumeStructureSpecificStorage(continuation, &detail);
    if (mlir::failed(next)) {
      fail(detail.empty() ? "post-K storage continuation failed" : detail);
      return;
    }
    if (!*next) {
      frontier.pop_back();
      return;
    }
    recordPrefix(**next);
    frontier.emplace_back(
        planningSession.createScheduleContinuation(std::move(**next)));
  }

  void stepSchedule(ScheduleContinuation &continuation) {
    ++work.successorSteps;
    std::string detail;
    auto next = planningSession.resumeSchedule(continuation, &detail);
    if (mlir::failed(next)) {
      fail(detail.empty() ? "schedule continuation failed" : detail);
      return;
    }
    if (!*next) {
      frontier.pop_back();
      return;
    }
    recordPrefix(**next);
    frontier.emplace_back(std::move(**next));
  }

  void stepCandidate(ScheduledState state) {
    ++work.scheduledStatesVisited;
    std::string keyFailure;
    auto key = CompleteCandidateKey::create(state, &keyFailure);
    if (mlir::failed(key)) {
      fail(keyFailure.empty() ? "complete state has no valid semantic key"
                              : keyFailure);
      return;
    }
    CandidateReservation reservation = controller.reserve(*key);
    if (reservation == CandidateReservation::Duplicate) {
      ++work.duplicateCompleteKeys;
      return;
    }
    if (reservation != CandidateReservation::Granted) {
      fail(reservation == CandidateReservation::Exhausted
               ? "actual-result controller exhausted unexpectedly"
               : "actual-result controller rejected a unique complete key");
      return;
    }
    FullFeasibilityStatistics actualStatistics;
    FullFeasibilityResult actual = planningSession.evaluateScheduledState(
        tensorProgram, state, program, executionConfig, diagnostics,
        programData, &actualStatistics, tilePipelineParallelism,
        captureTileDataflowIRTrace);
    work.candidateActualizations += actualStatistics.candidateActualizations;
    const FullFeasibilityStatus actualStatus = actual.status;
    const std::string actualDetail = actual.detail;
    std::vector<SemanticRootKey> causalRoots = actual.causalRoots;
    TemporalState temporal = state.getBufferState()
                                 .getExecutionStructureState()
                                 .getInitialBufferState()
                                 .getMovementState()
                                 .getRepresentationState()
                                 .getTemporalState();
    if (trace)
      trace->candidates.push_back({*key, actualStatus});
    CandidateRecordOutcome recorded =
        controller.record(*key, std::move(actual));
    if (recorded == CandidateRecordOutcome::CompilerBug) {
      fail("actual-result controller rejected a typed actual result");
      return;
    }
    if (recorded == CandidateRecordOutcome::Indeterminate) {
      pauseIndeterminate(actualDetail.empty()
                             ? "complete candidate actualization is unknown"
                             : actualDetail);
      return;
    }
    if (recorded == CandidateRecordOutcome::Accepted &&
        termination == SearchTerminationPolicy::FirstAccepted)
      status = UnifiedSearchResumeStatus::AcceptedCheckpoint;
    if (recorded == CandidateRecordOutcome::ExactRejection &&
        !causalRoots.empty())
      frontier.emplace_back(
          TemporalFeedbackFrame{std::move(temporal), std::move(causalRoots)});
  }

  void stepTemporalFeedback(TemporalFeedbackFrame feedback) {
    ++work.successorSteps;
    std::string detail;
    auto refined = planningSession.refineTemporalStateFromActualFeedback(
        feedback.state, feedback.causalRoots, &detail);
    if (mlir::failed(refined)) {
      fail(detail.empty() ? "actual temporal proposal failed" : detail);
      return;
    }
    if (!*refined)
      return;
    recordPrefix(**refined);
    frontier.emplace_back(
        planningSession.createRepresentationContinuation(std::move(**refined)));
  }

  void step() {
    if (frontier.empty())
      return;
    if (std::holds_alternative<SpatialFrame>(frontier.back())) {
      stepSpatial();
      return;
    }
    if (auto *continuation =
            std::get_if<RegionContinuation>(&frontier.back())) {
      stepRegion(*continuation);
      return;
    }
    if (auto *continuation =
            std::get_if<TemporalContinuation>(&frontier.back())) {
      stepTemporal(*continuation);
      return;
    }
    if (auto *continuation =
            std::get_if<RepresentationContinuation>(&frontier.back())) {
      stepRepresentation(*continuation);
      return;
    }
    if (auto *continuation =
            std::get_if<MovementContinuation>(&frontier.back())) {
      stepMovement(*continuation);
      return;
    }
    if (auto *continuation =
            std::get_if<StorageContinuation>(&frontier.back())) {
      stepStorage(*continuation);
      return;
    }
    if (auto *continuation =
            std::get_if<ExecutionStructureContinuation>(&frontier.back())) {
      stepExecutionStructure(*continuation);
      return;
    }
    if (auto *continuation = std::get_if<StructureSpecificStorageContinuation>(
            &frontier.back())) {
      stepStructureStorage(*continuation);
      return;
    }
    if (auto *continuation =
            std::get_if<ScheduleContinuation>(&frontier.back())) {
      stepSchedule(*continuation);
      return;
    }
    if (std::holds_alternative<TemporalFeedbackFrame>(frontier.back())) {
      TemporalFeedbackFrame feedback =
          std::move(std::get<TemporalFeedbackFrame>(frontier.back()));
      frontier.pop_back();
      stepTemporalFeedback(std::move(feedback));
      return;
    }
    ScheduledState state = std::move(std::get<ScheduledState>(frontier.back()));
    frontier.pop_back();
    stepCandidate(std::move(state));
  }

  UnifiedSearchResumeResult resume(uint64_t credits) {
    if (finalized)
      return {UnifiedSearchResumeStatus::Finished, 0};
    if (isTerminal(status))
      return {status, 0};
    ++work.resumeCalls;
    uint64_t consumed = 0;
    while (consumed < credits) {
      if (frontier.empty()) {
        status = UnifiedSearchResumeStatus::FrontierExhausted;
        return {status, consumed};
      }
      step();
      ++consumed;
      if (isTerminal(status))
        return {status, consumed};
    }
    if (frontier.empty())
      status = UnifiedSearchResumeStatus::FrontierExhausted;
    return {status, consumed};
  }

  UnifiedSearchResult finish() {
    if (finalized) {
      UnifiedSearchResult result;
      result.control.coverage = SearchControllerCoverage::Failed;
      result.failureDetail = "unified search session was finished twice";
      return result;
    }
    if (status == UnifiedSearchResumeStatus::CompilerBug)
      controller.markCompilerBug();
    const bool exactExhaustion =
        status == UnifiedSearchResumeStatus::FrontierExhausted &&
        !sawUnsupportedPrefix;
    UnifiedSearchResult result;
    result.control =
        controller.finish(exactExhaustion ? SearchFrontierStatus::Exhausted
                                          : SearchFrontierStatus::Incomplete);
    result.planning = planningSession.getWork();
    result.work = work;
    result.frontierExhausted =
        status == UnifiedSearchResumeStatus::FrontierExhausted;
    result.failureDetail = std::move(failureDetail);
    if (trace && result.control.winner)
      ++trace->winnerHandoffs;
    frontier.clear();
    finalized = true;
    return result;
  }

  mlir::ModuleOp tensorProgram;
  PhysicalDataflowPlanningSession &planningSession;
  frontend::FrontendProgramVerificationResult program;
  ExecutionConfig executionConfig;
  llvm::raw_ostream &diagnostics;
  ProgramDataHandoff &programData;
  SearchTerminationPolicy termination;
  ActualResultController controller;
  unsigned tilePipelineParallelism;
  bool captureTileDataflowIRTrace;
  UnifiedSearchTrace *trace = nullptr;
  std::vector<FrontierFrame> frontier;
  UnifiedSearchWork work;
  UnifiedSearchResumeStatus status = UnifiedSearchResumeStatus::Paused;
  bool sawUnsupportedPrefix = false;
  bool finalized = false;
  std::string failureDetail;
};

UnifiedSearchSession::UnifiedSearchSession(
    mlir::ModuleOp tensorProgram, PhysicalDataflowPlanningSession &session,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const UnifiedSearchOptions &options,
    unsigned tilePipelineParallelism, bool captureTileDataflowIRTrace,
    UnifiedSearchTrace *trace)
    : impl(std::make_unique<Impl>(tensorProgram, session, program,
                                  executionConfig, diagnostics, programData,
                                  options, tilePipelineParallelism,
                                  captureTileDataflowIRTrace, trace)) {}

UnifiedSearchSession::~UnifiedSearchSession() = default;

UnifiedSearchResumeResult UnifiedSearchSession::resume(uint64_t credits) {
  return impl->resume(credits);
}

UnifiedSearchResult UnifiedSearchSession::finish() { return impl->finish(); }

UnifiedSearchResult runUnifiedSearch(
    mlir::ModuleOp tensorProgram, PhysicalDataflowPlanningSession &session,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const UnifiedSearchOptions &options,
    unsigned tilePipelineParallelism, bool captureTileDataflowIRTrace) {
  UnifiedSearchSession search(tensorProgram, session, program, executionConfig,
                              diagnostics, programData, options,
                              tilePipelineParallelism,
                              captureTileDataflowIRTrace);
  (void)search.resume(options.planningCredits);
  return search.finish();
}

} // namespace wafer::compiler::detail
