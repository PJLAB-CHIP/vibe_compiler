//===- SearchCurrentIR.h - Current-IR physical search -------*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_SEARCHCURRENTIR_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_SEARCHCURRENTIR_H

#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Driver/PhysicalDataflow/UnifiedSearch.h"

namespace wafer::compiler::detail {

struct SearchCurrentIROptions {
  uint64_t layoutWorkLimit = UINT64_C(1048576);
  uint64_t maximumTemporalCandidatesPerStructuralState = 16;
  bool stopTemporalAfterFirstAccepted = true;
  uint64_t planningCredits = std::numeric_limits<uint64_t>::max();
  SearchTerminationPolicy termination = SearchTerminationPolicy::FirstAccepted;
  CurrentIRDownstreamOptions downstream;
};

struct SearchCurrentIRStatistics {
  uint64_t structuralMaterializations = 0;
  uint64_t temporalDomainsBuilt = 0;
  uint64_t temporalCandidateActualizations = 0;
  uint64_t temporalApplications = 0;
  uint64_t communicationRegionClosures = 0;
  uint64_t layoutInvocations = 0;
  uint64_t layoutFeasibleFallbacks = 0;
  uint64_t acceptedTemporalCandidates = 0;
  uint64_t exactRejectedTemporalCandidates = 0;
  uint64_t actualCapacityRefinements = 0;
  uint64_t unsupportedTemporalCandidates = 0;
  uint64_t indeterminateTemporalCandidates = 0;
  uint64_t incomparableTemporalObjectives = 0;
  PlanningWorkCounts planning;
  UnifiedSearchWork traversal;
  SearchControllerStatistics controller;
  SearchControllerCoverage coverage = SearchControllerCoverage::Failed;
  CurrentIRDownstreamStatistics downstream;
};

/// Traverses explicit Spatial/Region choices and actualizes each selected
/// structural state from current TensorProgram IR. Temporal alternatives are
/// cloned once from that structural owner with IRMapping, immediately applied,
/// and sent through the shared actual memory/target leaf. The retained winner
/// is the same actual owner returned by the leaf; it is never rebuilt.
ExecutableCompilationResult compileSearchCurrentIR(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const SearchCurrentIROptions &options,
    SearchCurrentIRStatistics *statistics = nullptr,
    ExecutableLoweringStatistics *executableStatistics = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_SEARCHCURRENTIR_H
