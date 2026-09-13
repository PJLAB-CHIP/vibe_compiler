//===- UnifiedSearch.h - Typed physical-dataflow traversal ---*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_UNIFIEDSEARCH_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_UNIFIEDSEARCH_H

#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Driver/PhysicalDataflow/ActualResultController.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningSession.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

enum class SearchTerminationPolicy : uint8_t {
  Exhaustive,
  FirstAccepted,
};

struct UnifiedSearchOptions {
  uint64_t planningCredits = std::numeric_limits<uint64_t>::max();
  uint64_t retainedBranches = 8;
  uint64_t candidateActualizationCredits = std::numeric_limits<uint64_t>::max();
  uint64_t maximumRegionRefinementCandidates = 0;
  SearchTerminationPolicy termination = SearchTerminationPolicy::Exhaustive;
  std::optional<analysis::SearchCostCohort> costCohort;
  ExactRejectionCachePolicy exactRejectionCache =
      ExactRejectionCachePolicy::Enabled;
  PlanningProfileSink *profile = nullptr;
};

struct UnifiedSearchWork {
  uint64_t resumeCalls = 0;
  uint64_t successorSteps = 0;
  uint64_t structuralStatesActualized = 0;
  uint64_t candidateActualizations = 0;
  uint64_t duplicateCompleteKeys = 0;
  uint64_t incompleteInnerDomains = 0;
  uint64_t peakRetainedBranches = 0;
  uint64_t resumedCandidates = 0;
  uint64_t stageYields = 0;
  uint64_t retiredBranches = 0;
  uint64_t localRegionRefinements = 0;
  uint64_t nonIncumbentRegionRefinements = 0;
};

using UnifiedSearchPrefixKey = std::variant<SpatialState, RegionState>;

struct UnifiedSearchCandidateTrace {
  StructuralCandidateKey key;
  ActualCandidateStatus status = ActualCandidateStatus::CompilerBug;

  friend bool operator==(const UnifiedSearchCandidateTrace &lhs,
                         const UnifiedSearchCandidateTrace &rhs) {
    return lhs.key == rhs.key && lhs.status == rhs.status;
  }
};

/// Optional test/report sink. Production compilation passes null and does not
/// retain prefix or candidate ledgers.
struct UnifiedSearchTrace {
  std::vector<UnifiedSearchPrefixKey> prefixes;
  std::vector<UnifiedSearchCandidateTrace> candidates;
  uint64_t winnerHandoffs = 0;
};

enum class CandidateContinuation : uint8_t {
  Exhausted,
  Explore,
  Repair,
  Improve,
};

enum class CandidateRetention : uint8_t {
  Replaceable,
  PendingCapacityRepair,
  UnfinishedActualization,
};

struct StructuralCandidateEvaluation {
  /// Empty with Exhausted closes the domain. Empty with a live continuation
  /// yields between verified stages and retains UnfinishedActualization.
  std::optional<ActualCandidateResult> result;
  uint64_t actualizations = 0;
  CandidateContinuation continuation = CandidateContinuation::Exhausted;
  /// Queued repair survives even when another local work class runs next.
  CandidateRetention retention = CandidateRetention::Replaceable;
};

/// Owns immutable actual IR checkpoints and their current-epoch cursors. Each
/// advance evaluates at most one new leaf. A suspended session never recreates
/// a checkpoint or borrows handles from a discarded candidate.
class StructuralCandidateSession {
public:
  virtual ~StructuralCandidateSession() = default;
  virtual StructuralCandidateEvaluation advance() = 0;
};

/// Starts an owned current-IR session. Construction itself does not actualize
/// a candidate; the first advance is charged just like subsequent leaves.
class StructuralCandidateEvaluator {
public:
  virtual ~StructuralCandidateEvaluator() = default;
  virtual std::unique_ptr<StructuralCandidateSession>
  start(const RegionState &state) = 0;
};

enum class UnifiedSearchResumeStatus : uint8_t {
  Paused,
  FrontierExhausted,
  AcceptedCheckpoint,
  CandidateBudgetExhausted,
  Indeterminate,
  CompilerBug,
  Finished,
};

struct UnifiedSearchResumeResult {
  UnifiedSearchResumeStatus status = UnifiedSearchResumeStatus::CompilerBug;
  uint64_t consumedCredits = 0;
};

struct UnifiedSearchResult {
  SearchControllerResult control;
  PlanningWorkCounts planning;
  UnifiedSearchWork work;
  bool frontierExhausted = false;
  std::string failureDetail;

  bool hasWinner() const { return control.winner.has_value(); }
};

/// Resumable, deterministic parent-by-parent traversal. The session owns only
/// typed continuations, bounded owned candidate sessions and controller state.
class UnifiedSearchSession {
public:
  UnifiedSearchSession(PhysicalDataflowPlanningSession &session,
                       StructuralCandidateEvaluator &evaluator,
                       const UnifiedSearchOptions &options,
                       UnifiedSearchTrace *trace = nullptr);
  ~UnifiedSearchSession();

  UnifiedSearchSession(const UnifiedSearchSession &) = delete;
  UnifiedSearchSession &operator=(const UnifiedSearchSession &) = delete;

  UnifiedSearchResumeResult resume(uint64_t credits);
  UnifiedSearchResult finish();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

UnifiedSearchResult runUnifiedSearch(PhysicalDataflowPlanningSession &session,
                                     StructuralCandidateEvaluator &evaluator,
                                     const UnifiedSearchOptions &options);

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_UNIFIEDSEARCH_H
