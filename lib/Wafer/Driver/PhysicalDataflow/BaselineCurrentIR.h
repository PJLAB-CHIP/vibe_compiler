//===- BaselineCurrentIR.h - Deterministic current-IR baseline -*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_BASELINECURRENTIR_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_BASELINECURRENTIR_H

#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"

namespace wafer::compiler::detail {

// Internal qualification choice. Ordinary none/search product callers never
// supply it; test drivers select a realization using the same transformations.
struct CommunicationCandidateSelection {
  bool mergeRegions = true;
  bool reusePeerInputs = false;
  bool pipelineLoads = false;
  BoundaryMovementOptions movement;
};

struct BaselineCurrentIROptions {
  uint64_t layoutWorkLimit = UINT64_C(1048576);
  uint32_t maximumCapacityAttempts = 16;
  CurrentIRDownstreamOptions downstream;
  const CommunicationCandidateSelection *qualification = nullptr;
};

struct BaselineCurrentIRStatistics {
  uint64_t attempts = 0;
  uint64_t capacityRefinements = 0;
  uint64_t spatialMaterializations = 0;
  uint64_t temporalApplications = 0;
  uint64_t communicationRegionClosures = 0;
  uint64_t layoutInvocations = 0;
  uint64_t layoutFeasibleFallbacks = 0;
  BoundaryMovementStatistics boundaryMovement;
  CurrentIRDownstreamStatistics downstream;
};

/// Runs the deterministic policy from one normalized structured TensorProgram.
/// Spatial and Region coordinates come from the canonical builders, not from
/// search state. Every attempt owns one actual candidate; only an actual SPM
/// capacity rejection permits a new attempt with smaller explicit temporal
/// tiles.
ExecutableCompilationResult compileBaselineCurrentIR(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, const BaselineCurrentIROptions &options,
    BaselineCurrentIRStatistics *statistics = nullptr,
    ExecutableLoweringStatistics *executableStatistics = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_BASELINECURRENTIR_H
