//===- PlanningSession.h - Structural planning frontier -----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSESSION_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSESSION_H

#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningMemo.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningProblem.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningState.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
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

/// Session-local traversal of explicit Spatial, Region and free Temporal
/// choices. It borrows immutable source analysis and never owns candidate IR.
class PhysicalDataflowPlanningSession {
public:
  explicit PhysicalDataflowPlanningSession(
      const PhysicalDataflowPlanningProblem &problem,
      PlanningProfileSink *profile = nullptr)
      : problem(problem),
        spatialProposalCache(PlanningMemoKind::SpatialProposals, profile),
        rootWorkCache(PlanningMemoKind::RootWork, profile),
        regionDomainCache(PlanningMemoKind::RegionDomain, profile),
        temporalDomainCache(PlanningMemoKind::TemporalDomain, profile) {}

  PhysicalDataflowPlanningSession(const PhysicalDataflowPlanningSession &) =
      delete;
  PhysicalDataflowPlanningSession &
  operator=(const PhysicalDataflowPlanningSession &) = delete;

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
      std::string *failureReason = nullptr,
      unsigned proposalRefinementSteps = 1);

  const PhysicalDataflowPlanningProblem &getProblem() const { return problem; }
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
  PlanningMemo<uint8_t, std::vector<SpatialPlan>> spatialProposalCache;
  PlanningMemo<SpatialPlan, std::vector<analysis::RootRegionWork>>
      rootWorkCache;
  PlanningMemo<SpatialPlan, RegionDomain> regionDomainCache;
  PlanningMemo<RegionState, TemporalDomain> temporalDomainCache;
  PlanningWorkCounts work;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGSESSION_H
