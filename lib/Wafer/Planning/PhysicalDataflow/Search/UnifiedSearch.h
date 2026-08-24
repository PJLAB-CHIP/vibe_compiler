//===- UnifiedSearch.h - Typed physical-dataflow traversal ---*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_UNIFIEDSEARCH_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_UNIFIEDSEARCH_H

#include "Wafer/Planning/PhysicalDataflow/Search/ActualResultController.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningSession.h"

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
  SearchTerminationPolicy termination = SearchTerminationPolicy::Exhaustive;
  std::optional<SearchCostCohort> costCohort;
};

struct UnifiedSearchWork {
  uint64_t resumeCalls = 0;
  uint64_t successorSteps = 0;
  uint64_t scheduledStatesVisited = 0;
  uint64_t candidateActualizations = 0;
  uint64_t duplicateCompleteKeys = 0;
};

using UnifiedSearchPrefixKey =
    std::variant<SpatialState, RegionState, TemporalState, RepresentationState,
                 MovementState, InitialBufferState, ExecutionStructureState,
                 BufferState, ScheduledState>;

struct UnifiedSearchCandidateTrace {
  CompleteCandidateKey key;
  FullFeasibilityStatus status = FullFeasibilityStatus::CompilerBug;

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

enum class UnifiedSearchResumeStatus : uint8_t {
  Paused,
  FrontierExhausted,
  AcceptedCheckpoint,
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
/// one synchronous Q50.F evaluation.
class UnifiedSearchSession {
public:
  UnifiedSearchSession(
      mlir::ModuleOp tensorProgram, PhysicalDataflowPlanningSession &session,
      const frontend::FrontendProgramVerificationResult &program,
      const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
      ProgramDataHandoff &programData, const UnifiedSearchOptions &options,
      unsigned tilePipelineParallelism = 0,
      bool captureTileDataflowIRTrace = false,
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

UnifiedSearchResult runUnifiedSearch(
    mlir::ModuleOp tensorProgram, PhysicalDataflowPlanningSession &session,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const UnifiedSearchOptions &options,
    unsigned tilePipelineParallelism = 0,
    bool captureTileDataflowIRTrace = false);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_UNIFIEDSEARCH_H
