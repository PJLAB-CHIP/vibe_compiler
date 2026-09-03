//===- SearchCurrentIR.h - Current-IR physical search -------*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_SEARCHCURRENTIR_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_SEARCHCURRENTIR_H

#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Driver/PhysicalDataflow/UnifiedSearch.h"
#include "Wafer/Support/OptimizationConfig.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace wafer::compiler::detail {

struct SearchCurrentIROptions {
  uint64_t layoutWorkLimit = UINT64_C(1048576);
  uint64_t maximumTemporalCandidatesPerStructuralState = 8;
  SearchLimits limits;
  bool stopTemporalAfterFirstAccepted = true;
  uint64_t planningCredits = std::numeric_limits<uint64_t>::max();
  SearchTerminationPolicy termination = SearchTerminationPolicy::Exhaustive;
  CurrentIRDownstreamOptions downstream;

  uint64_t getRefinementLimit() const {
    return limits.width > 2 ? std::min<uint64_t>(2, limits.width - 2) : 0;
  }
  uint64_t getInitialProposalLimit() const {
    return limits.width - getRefinementLimit();
  }
};

struct SearchCurrentIRStatistics {
  uint64_t structuralMaterializations = 0;
  uint64_t temporalDomainsBuilt = 0;
  uint64_t temporalCandidateActualizations = 0;
  uint64_t temporalApplications = 0;
  uint64_t communicationRegionClosures = 0;
  uint64_t layoutInvocations = 0;
  uint64_t layoutFeasibleFallbacks = 0;
  uint64_t movementCandidateActualizations = 0;
  uint64_t recursiveDoublingCandidates = 0;
  uint64_t recursiveDoublingAccepted = 0;
  uint64_t recursiveDoublingWinners = 0;
  uint64_t incomparableMovementObjectives = 0;
  uint64_t acceptedTemporalCandidates = 0;
  uint64_t exactRejectedTemporalCandidates = 0;
  uint64_t actualCapacityRefinements = 0;
  uint64_t unavailableCapacityRefinements = 0;
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
/// cloned once from that structural owner with IRMapping and immediately
/// applied. A non-native complete AllGather may then clone the post-layout
/// owner once for Ring/recursive-doubling movement; both actual owners consume
/// the shared memory/target leaf and count against actualization credits. The
/// retained winner is the same actual owner returned by the leaf; it is never
/// rebuilt.
ExecutableCompilationResult compileSearchCurrentIR(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const SearchCurrentIROptions &options,
    SearchCurrentIRStatistics *statistics = nullptr,
    ExecutableLoweringStatistics *executableStatistics = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_SEARCHCURRENTIR_H
