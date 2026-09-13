//===- SearchCurrentIR.h - Current-IR physical search -------*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_SEARCHCURRENTIR_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_SEARCHCURRENTIR_H

#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Driver/PhysicalDataflow/UnifiedSearch.h"
#include "Wafer/Support/OptimizationConfig.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>

namespace wafer::compiler::detail {

struct SearchCurrentIROptions {
  uint64_t layoutWorkLimit = UINT64_C(1048576);
  SearchLimits limits;
  uint64_t planningCredits = std::numeric_limits<uint64_t>::max();
  SearchTerminationPolicy termination = SearchTerminationPolicy::Exhaustive;
  // Optional explicit cancellation boundary for callers. Production search
  // uses its requested trial budget, without an implicit wall-time cutoff.
  std::optional<std::chrono::steady_clock::time_point> deadline;
  CurrentIRDownstreamOptions downstream;

  // Proposal generation is independent of retention width and leaf budget.
  uint64_t getRefinementLimit() const { return 2; }
  uint64_t getInitialProposalLimit() const { return 6; }
};

struct SearchCurrentIRStatistics {
  uint64_t structuralMaterializations = 0;
  uint64_t temporalDomainsBuilt = 0;
  uint64_t temporalCandidateActualizations = 0;
  uint64_t temporalApplications = 0;
  uint64_t peakSessionTemporalPrefixes = 0;
  uint64_t peakSessionIRModules = 0;
  uint64_t temporalBackpressureTurns = 0;
  uint64_t communicationRegionClosures = 0;
  uint64_t regionPreservingCandidates = 0;
  uint64_t regionPreservingAccepted = 0;
  uint64_t mergedRegionCandidates = 0;
  uint64_t mergedRegionAccepted = 0;
  uint64_t sharedDDRCandidates = 0;
  uint64_t sharedDDRAccepted = 0;
  uint64_t inputSharingCandidates = 0;
  uint64_t inputSharingAccepted = 0;
  uint64_t layoutInvocations = 0;
  uint64_t layoutFeasibleFallbacks = 0;
  uint64_t movementCandidateActualizations = 0;
  uint64_t recursiveDoublingCandidates = 0;
  uint64_t recursiveDoublingAccepted = 0;
  uint64_t dimensionOrderedAllToAllCandidates = 0;
  uint64_t dimensionOrderedAllToAllAccepted = 0;
  uint64_t distributedRingCandidates = 0;
  uint64_t distributedRingAccepted = 0;
  uint64_t acceptedCandidates = 0;
  uint64_t exactRejectedCandidates = 0;
  uint64_t actualCapacityRefinements = 0;
  uint64_t unavailableCapacityRefinements = 0;
  uint64_t unsupportedCandidates = 0;
  uint64_t indeterminateCandidates = 0;
  PlanningWorkCounts planning;
  UnifiedSearchWork traversal;
  SearchControllerStatistics controller;
  SearchControllerCoverage coverage = SearchControllerCoverage::Failed;
  CurrentIRDownstreamStatistics downstream;
};

/// Traverses explicit Spatial/Region choices and actualizes each selected
/// structural state from current TensorProgram IR. Temporal alternatives are
/// cloned once from that structural owner with IRMapping and immediately
/// applied. Optional communication closure clones that owner while retaining
/// the original Regions. Each Region choice may clone its post-layout owner
/// for causally ordered shared DDR, recursive-doubling AllGather,
/// dimension-ordered AllToAll, distributed Ring reduction, or their bounded
/// combinations. A specialized clone is discarded
/// unless its requested current-IR rewrite actually occurs. Every retained
/// owner consumes the same memory/target leaf and counts against actualization
/// credits; the winner is returned directly and is never rebuilt.
ExecutableCompilationResult compileSearchCurrentIR(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const SearchCurrentIROptions &options,
    SearchCurrentIRStatistics *statistics = nullptr,
    ExecutableLoweringStatistics *executableStatistics = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_SEARCHCURRENTIR_H
