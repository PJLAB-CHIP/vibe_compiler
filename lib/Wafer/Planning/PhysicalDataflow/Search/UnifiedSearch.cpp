//===- UnifiedSearch.cpp - Typed physical-dataflow traversal ----------===//

#include "Wafer/Planning/PhysicalDataflow/Search/UnifiedSearch.h"

#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

class UnifiedSearchRunner {
public:
  UnifiedSearchRunner(
      mlir::ModuleOp tensorProgram, PhysicalDataflowPlanningSession &session,
      const frontend::FrontendProgramVerificationResult &program,
      const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
      ProgramDataHandoff &programData, const UnifiedSearchOptions &options,
      unsigned tilePipelineParallelism, bool captureTileDataflowIRTrace)
      : tensorProgram(tensorProgram), session(session), program(program),
        executionConfig(executionConfig), diagnostics(diagnostics),
        programData(programData), options(options),
        remainingCredits(options.planningCredits),
        controller(options.planningCredits, options.costCohort),
        tilePipelineParallelism(tilePipelineParallelism),
        captureTileDataflowIRTrace(captureTileDataflowIRTrace) {}

  UnifiedSearchResult run() {
    while (!stopped) {
      if (!reserveStep())
        break;
      SpatialExpansionResult expansion = session.resumeSpatial();
      ++work.successorSteps;
      switch (expansion.getKind()) {
      case SpatialExpansionKind::StateQueued: {
        std::optional<SpatialState> state = session.takeNextSpatialState();
        if (!state) {
          fail("spatial continuation lost its queued state");
          break;
        }
        expandRegion(std::move(*state));
        break;
      }
      case SpatialExpansionKind::Unsupported:
        break;
      case SpatialExpansionKind::Indeterminate:
        pause(expansion.getDetail());
        break;
      case SpatialExpansionKind::ParentExhausted:
        frontierExhausted = true;
        stopped = true;
        break;
      case SpatialExpansionKind::CompilerBug:
        fail(expansion.getDetail());
        break;
      }
    }
    if (compilerBug)
      controller.markCompilerBug();
    UnifiedSearchResult result;
    result.control = controller.finish(frontierExhausted);
    result.planning = session.getWork();
    result.work = work;
    result.frontierExhausted = frontierExhausted;
    result.failureDetail = std::move(failureDetail);
    return result;
  }

private:
  bool reserveStep() {
    if (remainingCredits == std::numeric_limits<uint64_t>::max())
      return true;
    if (remainingCredits == 0) {
      stopped = true;
      return false;
    }
    --remainingCredits;
    return true;
  }

  void fail(llvm::StringRef detail) {
    compilerBug = true;
    stopped = true;
    failureDetail = detail.str();
  }

  void pause(llvm::StringRef detail) {
    stopped = true;
    failureDetail = detail.str();
  }

  void expandRegion(SpatialState spatial) {
    RegionContinuation continuation =
        session.createRegionContinuation(std::move(spatial));
    while (!stopped) {
      if (!reserveStep())
        return;
      std::string detail;
      auto next = session.resumeRegion(continuation, &detail);
      ++work.successorSteps;
      if (mlir::failed(next)) {
        fail(detail);
        return;
      }
      if (!*next)
        return;
      expandTemporal(std::move(**next));
    }
  }

  void expandTemporal(RegionState region) {
    TemporalContinuation continuation =
        session.createTemporalContinuation(std::move(region));
    while (!stopped) {
      if (!reserveStep())
        return;
      TemporalExpansionResult next = session.resumeTemporal(continuation);
      ++work.successorSteps;
      if (next.getKind() == TemporalExpansionKind::ParentExhausted)
        return;
      if (next.getKind() == TemporalExpansionKind::Unsupported)
        return;
      if (next.getKind() == TemporalExpansionKind::Indeterminate) {
        pause(next.getDetail());
        return;
      }
      if (next.getKind() != TemporalExpansionKind::State) {
        fail(next.getDetail());
        return;
      }
      std::optional<TemporalState> state = next.takeState();
      if (!state) {
        fail("temporal continuation lost its state");
        return;
      }
      expandRepresentation(std::move(*state));
    }
  }

  void expandRepresentation(TemporalState temporal) {
    RepresentationContinuation continuation =
        session.createRepresentationContinuation(std::move(temporal));
    while (!stopped) {
      if (!reserveStep())
        return;
      RepresentationExpansionResult next =
          session.resumeRepresentation(continuation);
      ++work.successorSteps;
      if (next.getKind() == RepresentationExpansionKind::ParentExhausted)
        return;
      if (next.getKind() == RepresentationExpansionKind::Unsupported)
        return;
      if (next.getKind() != RepresentationExpansionKind::State) {
        fail(next.getDetail());
        return;
      }
      std::optional<RepresentationState> state = next.takeState();
      if (!state) {
        fail("representation continuation lost its state");
        return;
      }
      expandMovement(std::move(*state));
    }
  }

  void expandMovement(RepresentationState representations) {
    MovementContinuation continuation =
        session.createMovementContinuation(std::move(representations));
    while (!stopped) {
      if (!reserveStep())
        return;
      MovementExpansionResult next = session.resumeMovement(continuation);
      ++work.successorSteps;
      if (next.getKind() == MovementExpansionKind::ParentExhausted)
        return;
      if (next.getKind() == MovementExpansionKind::Unsupported)
        return;
      if (next.getKind() != MovementExpansionKind::State) {
        fail(next.getDetail());
        return;
      }
      std::optional<MovementState> state = next.takeState();
      if (!state) {
        fail("movement continuation lost its state");
        return;
      }
      expandInitialStorage(std::move(*state));
    }
  }

  void expandInitialStorage(MovementState movement) {
    StorageContinuation continuation =
        session.createStorageContinuation(std::move(movement));
    while (!stopped) {
      if (!reserveStep())
        return;
      StorageExpansionResult next = session.resumeStorage(continuation);
      ++work.successorSteps;
      if (next.getKind() == StorageExpansionKind::ParentExhausted)
        return;
      if (next.getKind() == StorageExpansionKind::Unsupported)
        return;
      if (next.getKind() == StorageExpansionKind::Indeterminate) {
        pause(next.getDetail());
        return;
      }
      if (next.getKind() != StorageExpansionKind::State) {
        fail(next.getDetail());
        return;
      }
      std::optional<InitialBufferState> state = next.takeState();
      if (!state) {
        fail("initial storage continuation lost its state");
        return;
      }
      expandExecutionStructure(std::move(*state));
    }
  }

  void expandExecutionStructure(InitialBufferState buffers) {
    ExecutionStructureContinuation continuation =
        session.createExecutionStructureContinuation(std::move(buffers));
    while (!stopped) {
      if (!reserveStep())
        return;
      std::string detail;
      auto next = session.resumeExecutionStructure(continuation, &detail);
      ++work.successorSteps;
      if (mlir::failed(next)) {
        fail(detail);
        return;
      }
      if (!*next)
        return;
      expandStructureStorage(std::move(**next));
    }
  }

  void expandStructureStorage(ExecutionStructureState structure) {
    StructureSpecificStorageContinuation continuation =
        session.createStructureSpecificStorageContinuation(
            std::move(structure));
    while (!stopped) {
      if (!reserveStep())
        return;
      std::string detail;
      auto next = session.resumeStructureSpecificStorage(continuation, &detail);
      ++work.successorSteps;
      if (mlir::failed(next)) {
        fail(detail);
        return;
      }
      if (!*next)
        return;
      expandSchedule(std::move(**next));
    }
  }

  void expandSchedule(BufferState buffers) {
    ScheduleContinuation continuation =
        session.createScheduleContinuation(std::move(buffers));
    while (!stopped) {
      if (!reserveStep())
        return;
      std::string detail;
      auto next = session.resumeSchedule(continuation, &detail);
      ++work.successorSteps;
      if (mlir::failed(next)) {
        fail(detail);
        return;
      }
      if (!*next)
        return;
      visitScheduled(std::move(**next));
    }
  }

  void visitScheduled(ScheduledState state) {
    ++work.scheduledStatesVisited;
    const ClosedSchedulePlan &plan = state.getSchedulePlan();
    if (controller.isForbidden(plan))
      return;
    if (!reserveStep())
      return;
    CandidateReservation reservation = controller.reserve(plan);
    if (reservation == CandidateReservation::Exhausted) {
      stopped = true;
      return;
    }
    if (reservation != CandidateReservation::Granted) {
      fail("unified search produced a duplicate or invalid complete plan");
      return;
    }
    FullFeasibilityStatistics actualStatistics;
    FullFeasibilityResult actual = session.evaluateScheduledState(
        tensorProgram, state, program, executionConfig, diagnostics,
        programData, &actualStatistics, tilePipelineParallelism,
        captureTileDataflowIRTrace);
    work.candidateActualizations += actualStatistics.candidateActualizations;
    const FullFeasibilityStatus actualStatus = actual.status;
    std::vector<SemanticRootKey> causalRoots = actual.causalRoots;
    TemporalState temporal = state.getBufferState()
                                 .getExecutionStructureState()
                                 .getInitialBufferState()
                                 .getMovementState()
                                 .getRepresentationState()
                                 .getTemporalState();
    CandidateRecordOutcome recorded =
        controller.record(plan, std::move(actual));
    if (recorded == CandidateRecordOutcome::CompilerBug) {
      fail("unified search actual-result controller rejected a typed result");
      return;
    }
    if (recorded == CandidateRecordOutcome::Indeterminate) {
      stopped = true;
      return;
    }
    if (actualStatus == FullFeasibilityStatus::ExactRejection &&
        !causalRoots.empty()) {
      if (!reserveStep())
        return;
      std::string detail;
      auto refined = session.refineTemporalStateFromActualFeedback(
          temporal, causalRoots, &detail);
      ++work.successorSteps;
      if (mlir::failed(refined)) {
        fail(detail);
        return;
      }
      if (!*refined) {
        pause("actual SPM feedback exhausted its temporal domain");
        return;
      }
      expandRepresentation(std::move(**refined));
      if (!stopped)
        pause("actual-feedback proposal completed without an accepted result");
      return;
    }
    if (recorded == CandidateRecordOutcome::Accepted &&
        options.stopAfterFirstAccepted)
      stopped = true;
  }

  mlir::ModuleOp tensorProgram;
  PhysicalDataflowPlanningSession &session;
  const frontend::FrontendProgramVerificationResult &program;
  const ExecutionConfig &executionConfig;
  llvm::raw_ostream &diagnostics;
  ProgramDataHandoff &programData;
  const UnifiedSearchOptions &options;
  uint64_t remainingCredits;
  ActualResultController controller;
  unsigned tilePipelineParallelism;
  bool captureTileDataflowIRTrace;
  UnifiedSearchWork work;
  bool stopped = false;
  bool frontierExhausted = false;
  bool compilerBug = false;
  std::string failureDetail;
};

} // namespace

UnifiedSearchResult runUnifiedSearch(
    mlir::ModuleOp tensorProgram, PhysicalDataflowPlanningSession &session,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const UnifiedSearchOptions &options,
    unsigned tilePipelineParallelism, bool captureTileDataflowIRTrace) {
  return UnifiedSearchRunner(tensorProgram, session, program, executionConfig,
                             diagnostics, programData, options,
                             tilePipelineParallelism,
                             captureTileDataflowIRTrace)
      .run();
}

} // namespace wafer::compiler::detail
