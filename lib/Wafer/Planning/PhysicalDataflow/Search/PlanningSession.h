//===- PlanningSession.h - Spatial planning frontier --------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSESSION_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSESSION_H

#include "Wafer/Planning/PhysicalDataflow/EventGraph.h"
#include "Wafer/Planning/PhysicalDataflow/ExecutionStructureDomain.h"
#include "Wafer/Planning/PhysicalDataflow/FullFeasibility.h"
#include "Wafer/Planning/PhysicalDataflow/MovementDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RepresentationDomain.h"
#include "Wafer/Planning/PhysicalDataflow/ScheduleDomain.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProblem.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningState.h"
#include "Wafer/Planning/PhysicalDataflow/StorageDomain.h"
#include "Wafer/Planning/PhysicalDataflow/StructureSpecificStorageDomain.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

struct PlanningWorkCounts {
  uint64_t spatialSuccessorSteps = 0;
  uint64_t spatialDemandQueries = 0;
  uint64_t spatialStatesQueued = 0;
  uint64_t rootWorkSuccessorSteps = 0;
  uint64_t rootWorksValidated = 0;
  uint64_t regionSuccessorSteps = 0;
  uint64_t regionStatesQueued = 0;
  uint64_t temporalSuccessorSteps = 0;
  uint64_t temporalStatesQueued = 0;
  uint64_t unsupportedTemporalChoices = 0;
  uint64_t indeterminateTemporalChoices = 0;
  uint64_t structuralReadinessQueries = 0;
  uint64_t representationSuccessorSteps = 0;
  uint64_t representationStatesQueued = 0;
  uint64_t unsupportedRepresentationChoices = 0;
  uint64_t movementSuccessorSteps = 0;
  uint64_t movementStatesQueued = 0;
  uint64_t unsupportedMovementChoices = 0;
  uint64_t storageSuccessorSteps = 0;
  uint64_t storageStatesQueued = 0;
  uint64_t unsupportedStorageChoices = 0;
  uint64_t eventGraphQueries = 0;
  uint64_t eventGraphsBuilt = 0;
  uint64_t executionStructureQueries = 0;
  uint64_t executionStructureSuccessorSteps = 0;
  uint64_t executionStructureStatesQueued = 0;
  uint64_t structureSpecificStorageQueries = 0;
  uint64_t structureSpecificStorageSuccessorSteps = 0;
  uint64_t structureSpecificStorageStatesQueued = 0;
  uint64_t scheduleQueries = 0;
  uint64_t scheduleSuccessorSteps = 0;
  uint64_t scheduleStatesQueued = 0;
  uint64_t fullFeasibilityEvaluations = 0;
  uint64_t candidateActualizations = 0;
  uint64_t duplicateSpatialChoices = 0;
  uint64_t unsupportedSpatialChoices = 0;
  uint64_t indeterminateSpatialChoices = 0;
};

enum class SpatialChoiceOutcomeKind : uint8_t {
  Satisfied,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

SpatialChoiceOutcomeKind
classifySpatialChoiceOutcome(const analysis::ExactDemandOutcome &outcome);

enum class SpatialExpansionKind : uint8_t {
  StateQueued,
  Unsupported,
  Indeterminate,
  ParentExhausted,
  CompilerBug,
};

class SpatialExpansionResult {
public:
  SpatialExpansionKind getKind() const { return kind; }
  llvm::StringRef getDetail() const { return detail; }

private:
  SpatialExpansionResult(SpatialExpansionKind kind, std::string detail = {})
      : kind(kind), detail(std::move(detail)) {}

  SpatialExpansionKind kind;
  std::string detail;

  friend class PhysicalDataflowPlanningSession;
};

/// Public-search result at the first missing planning coordinate. This is
/// evidence about a validated partial prefix, not a complete candidate and
/// cannot be materialized or published.
class IncompletePlanningDomain {
public:
  const ScheduledState &getState() const { return state; }
  const EventGraph &getEventGraph() const { return eventGraph; }
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return requiredCoordinate;
  }
  bool hasRemainingSpatialWork() const { return remainingSpatialWork; }
  const PlanningWorkCounts &getWork() const { return work; }

private:
  IncompletePlanningDomain(ScheduledState state, EventGraph eventGraph,
                           RequiredPlanningCoordinate requiredCoordinate,
                           bool remainingSpatialWork, PlanningWorkCounts work)
      : state(std::move(state)), eventGraph(std::move(eventGraph)),
        requiredCoordinate(requiredCoordinate),
        remainingSpatialWork(remainingSpatialWork), work(work) {}

  ScheduledState state;
  EventGraph eventGraph;
  RequiredPlanningCoordinate requiredCoordinate =
      RequiredPlanningCoordinate::FullFeasibility;
  bool remainingSpatialWork = false;
  PlanningWorkCounts work;

  friend class PhysicalDataflowPlanningSession;
};

/// Session-owned continuation for one validated RegionState. The cursor and
/// derived scope descriptors are not part of TemporalState identity.
class TemporalContinuation {
public:
  const RegionState &getParent() const { return parent; }
  bool isExhausted() const { return exhausted; }

private:
  explicit TemporalContinuation(RegionState parent)
      : parent(std::move(parent)) {}

  RegionState parent;
  std::optional<TemporalCursor> cursor;
  bool started = false;
  bool exhausted = false;

  friend class PhysicalDataflowPlanningSession;
};

enum class TemporalExpansionKind : uint8_t {
  State,
  Unsupported,
  Indeterminate,
  ParentExhausted,
  CompilerBug,
};

class TemporalExpansionResult {
public:
  TemporalExpansionKind getKind() const { return kind; }
  const TemporalState *getState() const { return state ? &*state : nullptr; }
  std::optional<TemporalState> takeState() {
    std::optional<TemporalState> result = std::move(state);
    state.reset();
    return result;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  TemporalExpansionResult(TemporalExpansionKind kind,
                          std::optional<TemporalState> state = {},
                          std::string detail = {})
      : kind(kind), state(std::move(state)), detail(std::move(detail)) {}

  TemporalExpansionKind kind;
  std::optional<TemporalState> state;
  std::string detail;

  friend class PhysicalDataflowPlanningSession;
};

class RepresentationContinuation {
public:
  const TemporalState &getParent() const { return parent; }
  bool isExhausted() const { return exhausted; }

private:
  explicit RepresentationContinuation(TemporalState parent)
      : parent(std::move(parent)) {}

  TemporalState parent;
  std::optional<RepresentationCursor> cursor;
  bool readinessChecked = false;
  bool started = false;
  bool exhausted = false;

  friend class PhysicalDataflowPlanningSession;
};

enum class RepresentationExpansionKind : uint8_t {
  State,
  Unsupported,
  ParentExhausted,
  CompilerBug,
};

class RepresentationExpansionResult {
public:
  RepresentationExpansionKind getKind() const { return kind; }
  std::optional<RepresentationState> takeState() {
    std::optional<RepresentationState> result = std::move(state);
    state.reset();
    return result;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  RepresentationExpansionResult(RepresentationExpansionKind kind,
                                std::optional<RepresentationState> state = {},
                                std::string detail = {})
      : kind(kind), state(std::move(state)), detail(std::move(detail)) {}

  RepresentationExpansionKind kind;
  std::optional<RepresentationState> state;
  std::string detail;

  friend class PhysicalDataflowPlanningSession;
};

class MovementContinuation {
public:
  const RepresentationState &getParent() const { return parent; }
  bool isExhausted() const { return exhausted; }

private:
  explicit MovementContinuation(RepresentationState parent)
      : parent(std::move(parent)) {}

  RepresentationState parent;
  std::optional<MovementCursor> cursor;
  bool started = false;
  bool exhausted = false;

  friend class PhysicalDataflowPlanningSession;
};

enum class MovementExpansionKind : uint8_t {
  State,
  Unsupported,
  ParentExhausted,
  CompilerBug,
};

class MovementExpansionResult {
public:
  MovementExpansionKind getKind() const { return kind; }
  std::optional<MovementState> takeState() {
    std::optional<MovementState> result = std::move(state);
    state.reset();
    return result;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  MovementExpansionResult(MovementExpansionKind kind,
                          std::optional<MovementState> state = {},
                          std::string detail = {})
      : kind(kind), state(std::move(state)), detail(std::move(detail)) {}

  MovementExpansionKind kind;
  std::optional<MovementState> state;
  std::string detail;

  friend class PhysicalDataflowPlanningSession;
};

class StorageContinuation {
public:
  const MovementState &getParent() const { return parent; }
  bool isExhausted() const { return exhausted; }

private:
  explicit StorageContinuation(MovementState parent)
      : parent(std::move(parent)) {}

  MovementState parent;
  std::optional<StorageCursor> cursor;
  bool started = false;
  bool exhausted = false;

  friend class PhysicalDataflowPlanningSession;
};

enum class StorageExpansionKind : uint8_t {
  State,
  Unsupported,
  ParentExhausted,
  CompilerBug,
};

class StorageExpansionResult {
public:
  StorageExpansionKind getKind() const { return kind; }
  std::optional<InitialBufferState> takeState() {
    std::optional<InitialBufferState> result = std::move(state);
    state.reset();
    return result;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  StorageExpansionResult(StorageExpansionKind kind,
                         std::optional<InitialBufferState> state = {},
                         std::string detail = {})
      : kind(kind), state(std::move(state)), detail(std::move(detail)) {}

  StorageExpansionKind kind;
  std::optional<InitialBufferState> state;
  std::string detail;

  friend class PhysicalDataflowPlanningSession;
};

class ExecutionStructureContinuation {
public:
  const InitialBufferState &getParent() const { return parent; }
  bool isExhausted() const { return exhausted; }

private:
  explicit ExecutionStructureContinuation(InitialBufferState parent)
      : parent(std::move(parent)) {}

  InitialBufferState parent;
  std::optional<ExecutionStructureCursor> cursor;
  bool started = false;
  bool exhausted = false;

  friend class PhysicalDataflowPlanningSession;
};

class StructureSpecificStorageContinuation {
public:
  const ExecutionStructureState &getParent() const { return parent; }
  bool isExhausted() const { return exhausted; }

private:
  explicit StructureSpecificStorageContinuation(ExecutionStructureState parent)
      : parent(std::move(parent)) {}

  ExecutionStructureState parent;
  std::optional<StructureSpecificStorageCursor> cursor;
  bool started = false;
  bool exhausted = false;

  friend class PhysicalDataflowPlanningSession;
};

class ScheduleContinuation {
public:
  const BufferState &getParent() const { return parent; }
  bool isExhausted() const { return exhausted; }

private:
  explicit ScheduleContinuation(BufferState parent)
      : parent(std::move(parent)) {}

  BufferState parent;
  std::optional<ScheduleCursor> cursor;
  bool started = false;
  bool exhausted = false;

  friend class PhysicalDataflowPlanningSession;
};

/// Session-owned continuation for one validated SpatialState. The cursor is
/// not part of RegionState identity and creates no IR.
class RegionContinuation {
public:
  const SpatialState &getParent() const { return parent; }
  bool isExhausted() const { return exhausted; }

private:
  explicit RegionContinuation(SpatialState parent)
      : parent(std::move(parent)) {}

  SpatialState parent;
  std::optional<RegionCursor> cursor;
  std::vector<RegionPlan> proposals;
  std::set<RegionPlan> emitted;
  size_t nextProposal = 0;
  bool proposalsInitialized = false;
  bool rawStarted = false;
  bool exhausted = false;

  friend class PhysicalDataflowPlanningSession;
};

/// Query-local owner of the spatial continuation, stable dedup set, and
/// deterministic frontier. It borrows one immutable planning problem and
/// never owns or creates candidate IR.
class PhysicalDataflowPlanningSession {
public:
  explicit PhysicalDataflowPlanningSession(
      const PhysicalDataflowPlanningProblem &problem)
      : problem(problem) {}

  PhysicalDataflowPlanningSession(const PhysicalDataflowPlanningSession &) =
      delete;
  PhysicalDataflowPlanningSession &
  operator=(const PhysicalDataflowPlanningSession &) = delete;

  /// Advances through checked proposals followed by the complete canonical
  /// successor. At most one new SpatialState is queued per call.
  SpatialExpansionResult resumeSpatial();

  std::optional<SpatialState> takeNextSpatialState();

  RegionContinuation createRegionContinuation(SpatialState parent) const {
    return RegionContinuation(std::move(parent));
  }
  mlir::FailureOr<std::optional<RegionState>>
  resumeRegion(RegionContinuation &continuation,
               std::string *failureReason = nullptr);

  TemporalContinuation createTemporalContinuation(RegionState parent) const {
    return TemporalContinuation(std::move(parent));
  }
  TemporalExpansionResult resumeTemporal(TemporalContinuation &continuation);

  mlir::FailureOr<std::optional<TemporalState>>
  refineTemporalStateFromActualFeedback(
      const TemporalState &state, llvm::ArrayRef<SemanticRootKey> causalRoots,
      std::string *failureReason = nullptr);

  RepresentationContinuation
  createRepresentationContinuation(TemporalState parent) const {
    return RepresentationContinuation(std::move(parent));
  }
  RepresentationExpansionResult
  resumeRepresentation(RepresentationContinuation &continuation);

  MovementContinuation
  createMovementContinuation(RepresentationState parent) const {
    return MovementContinuation(std::move(parent));
  }
  MovementExpansionResult resumeMovement(MovementContinuation &continuation);

  StorageContinuation createStorageContinuation(MovementState parent) const {
    return StorageContinuation(std::move(parent));
  }
  StorageExpansionResult resumeStorage(StorageContinuation &continuation);

  ExecutionStructureContinuation
  createExecutionStructureContinuation(InitialBufferState parent) const {
    return ExecutionStructureContinuation(std::move(parent));
  }
  mlir::FailureOr<std::optional<ExecutionStructureState>>
  resumeExecutionStructure(ExecutionStructureContinuation &continuation,
                           std::string *failureReason = nullptr);

  StructureSpecificStorageContinuation
  createStructureSpecificStorageContinuation(
      ExecutionStructureState parent) const {
    return StructureSpecificStorageContinuation(std::move(parent));
  }
  mlir::FailureOr<std::optional<BufferState>> resumeStructureSpecificStorage(
      StructureSpecificStorageContinuation &continuation,
      std::string *failureReason = nullptr);

  ScheduleContinuation createScheduleContinuation(BufferState parent) const {
    return ScheduleContinuation(std::move(parent));
  }
  mlir::FailureOr<std::optional<ScheduledState>>
  resumeSchedule(ScheduleContinuation &continuation,
                 std::string *failureReason = nullptr);

  /// Drives only far enough to prove that public search reached the deepest
  /// current prefix. Unsupported, indeterminate, or broken outcomes return
  /// without exhausting the preserved continuation.
  mlir::FailureOr<IncompletePlanningDomain>
  getFirstIncompleteState(std::string *failureReason = nullptr);

  /// Actualizes one complete ScheduledState exactly once. The returned
  /// accepted executable is retained; rejected candidate IR is destroyed
  /// before return. This method never advances or repairs the planning state.
  FullFeasibilityResult evaluateScheduledState(
      mlir::ModuleOp tensorProgram, const ScheduledState &state,
      const frontend::FrontendProgramVerificationResult &program,
      const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
      ProgramDataHandoff &programData,
      FullFeasibilityStatistics *statistics = nullptr,
      unsigned tilePipelineParallelism = 0,
      bool captureTileDataflowIRTrace = false);

  const PlanningWorkCounts &getWork() const { return work; }
  bool hasRemainingSpatialWork() const;
  bool isSpatialExhausted() const {
    return canonicalCursor == CanonicalCursor::Exhausted &&
           !pausedSpatialChoice.has_value();
  }

private:
  enum class CanonicalCursor : uint8_t { NotStarted, LastChoice, Exhausted };

  SpatialExpansionResult evaluateAndQueue(SpatialPlan choice,
                                          bool proposalChoice);
  mlir::FailureOr<RegionDomain *>
  getOrCreateRegionDomain(const SpatialState &spatial,
                          std::string *failureReason = nullptr);
  struct TemporalDomainLookup {
    TemporalDomain *domain = nullptr;
    std::optional<TemporalDomainFailure> failure;
  };

  TemporalDomainLookup getOrCreateTemporalDomain(const RegionState &region);

  struct RepresentationDomainLookup {
    RepresentationDomain *domain = nullptr;
    std::optional<RepresentationDomainFailure> failure;
  };

  RepresentationDomainLookup
  getOrCreateRepresentationDomain(const TemporalState &temporal);

  struct MovementDomainLookup {
    MovementDomain *domain = nullptr;
    std::optional<MovementDomainFailure> failure;
  };

  MovementDomainLookup
  getOrCreateMovementDomain(const RepresentationState &representations);

  struct StorageDomainLookup {
    StorageDomain *domain = nullptr;
    std::optional<StorageDomainFailure> failure;
  };

  StorageDomainLookup getOrCreateStorageDomain(const MovementState &movement);

  struct EventGraphLookup {
    EventGraph *graph = nullptr;
    std::optional<EventGraphFailure> failure;
  };

  EventGraphLookup getOrCreateEventGraph(const InitialBufferState &buffers);

  struct ExecutionStructureDomainLookup {
    ExecutionStructureDomain *domain = nullptr;
    std::optional<ExecutionStructureDomainFailure> failure;
  };

  ExecutionStructureDomainLookup
  getOrCreateExecutionStructureDomain(const InitialBufferState &buffers,
                                      const EventGraph &eventGraph);

  struct StructureSpecificStorageDomainLookup {
    StructureSpecificStorageDomain *domain = nullptr;
    std::optional<StructureSpecificStorageFailure> failure;
  };

  StructureSpecificStorageDomainLookup
  getOrCreateStructureSpecificStorage(const ExecutionStructureState &structure,
                                      const EventGraph &eventGraph);

  struct ScheduleDomainLookup {
    ScheduleDomain *domain = nullptr;
    std::optional<ScheduleDomainFailure> failure;
  };

  ScheduleDomainLookup getOrCreateScheduleDomain(const BufferState &buffers,
                                                 const EventGraph &eventGraph);

  const PhysicalDataflowPlanningProblem &problem;
  CanonicalCursor canonicalCursor = CanonicalCursor::NotStarted;
  std::optional<SpatialPlan> lastCanonicalChoice;
  std::optional<SpatialPlan> pausedSpatialChoice;
  bool pausedChoiceIsProposal = false;
  std::set<SpatialPlan> resolvedProposalChoices;
  std::set<SpatialState> frontier;
  std::map<SpatialPlan, std::vector<analysis::RootRegionWork>> rootWorkCache;
  std::map<SpatialPlan, RegionDomain> regionDomainCache;
  std::map<RegionState, TemporalDomain> temporalDomainCache;
  std::map<TemporalState, RepresentationDomain> representationDomainCache;
  std::map<RepresentationState, MovementDomain> movementDomainCache;
  std::map<MovementState, StorageDomain> storageDomainCache;
  std::map<InitialBufferState, EventGraph> eventGraphCache;
  std::map<InitialBufferState, ExecutionStructureDomain>
      executionStructureDomainCache;
  std::map<ExecutionStructureState, StructureSpecificStorageDomain>
      structureSpecificStorageDomainCache;
  std::map<BufferState, ScheduleDomain> scheduleDomainCache;
  PlanningWorkCounts work;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSESSION_H
