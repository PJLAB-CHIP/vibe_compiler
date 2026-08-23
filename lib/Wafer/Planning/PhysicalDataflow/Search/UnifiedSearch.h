//===- UnifiedSearch.h - Typed physical-dataflow traversal ---*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_UNIFIEDSEARCH_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_UNIFIEDSEARCH_H

#include "Wafer/Planning/PhysicalDataflow/Search/ActualResultController.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningSession.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace wafer::compiler::detail {

struct UnifiedSearchOptions {
  uint64_t planningCredits = std::numeric_limits<uint64_t>::max();
  bool stopAfterFirstAccepted = false;
  std::optional<SearchCostCohort> costCohort;
};

struct UnifiedSearchWork {
  uint64_t successorSteps = 0;
  uint64_t scheduledStatesVisited = 0;
  uint64_t candidateActualizations = 0;
};

struct UnifiedSearchResult {
  SearchControllerResult control;
  PlanningWorkCounts planning;
  UnifiedSearchWork work;
  bool frontierExhausted = false;
  std::string failureDetail;

  bool hasWinner() const { return control.winner.has_value(); }
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
