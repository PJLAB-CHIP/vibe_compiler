//===- UnifiedSearch.h - Typed physical-dataflow traversal ---*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_UNIFIEDSEARCH_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_UNIFIEDSEARCH_H

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
  uint64_t structuralCandidateCredits = std::numeric_limits<uint64_t>::max();
  SearchTerminationPolicy termination = SearchTerminationPolicy::Exhaustive;
  std::optional<SearchCostCohort> costCohort;
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

struct StructuralCandidateEvaluation {
  ActualCandidateResult result;
  uint64_t actualizations = 0;
  bool domainExhausted = true;
};

/// Caller-owned synchronous current-IR actualizer. Implementations materialize
/// one RegionState, build and immediately consume temporal choices from that
/// actual IR, and return one typed aggregate outcome. This structural search
/// core never owns candidate IR or a complete/shadow materializer. Because the
/// controller key ends at RegionState, ExactRejection is valid only after the
/// evaluator has closed every relevant current-IR inner choice; one temporal
/// candidate rejection cannot be lifted to the structural key.
class StructuralCandidateEvaluator {
public:
  virtual ~StructuralCandidateEvaluator() = default;
  virtual StructuralCandidateEvaluation evaluate(const RegionState &state) = 0;
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
/// typed continuations and controller state; candidate IR lives solely inside
/// one synchronous actual evaluation.
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
