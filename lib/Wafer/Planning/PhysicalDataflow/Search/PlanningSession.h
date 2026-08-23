//===- PlanningSession.h - Spatial planning frontier --------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSESSION_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSESSION_H

#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProblem.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningState.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>

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
  const TemporalState &getState() const { return state; }
  RequiredPlanningCoordinate getRequiredCoordinate() const {
    return state.getRequiredCoordinate();
  }
  bool hasRemainingSpatialWork() const { return remainingSpatialWork; }
  const PlanningWorkCounts &getWork() const { return work; }

private:
  IncompletePlanningDomain(TemporalState state, bool remainingSpatialWork,
                           PlanningWorkCounts work)
      : state(std::move(state)), remainingSpatialWork(remainingSpatialWork),
        work(work) {}

  TemporalState state;
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
  bool started = false;
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

  /// Drives only far enough to prove that public search reached the deepest
  /// current prefix. Unsupported, indeterminate, or broken outcomes return
  /// without exhausting the preserved continuation.
  mlir::FailureOr<IncompletePlanningDomain>
  getFirstIncompleteState(std::string *failureReason = nullptr);

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
  PlanningWorkCounts work;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSESSION_H
