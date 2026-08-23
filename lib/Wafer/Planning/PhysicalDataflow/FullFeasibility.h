//===- FullFeasibility.h - Complete candidate actual admission -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_FULLFEASIBILITY_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_FULLFEASIBILITY_H

#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProblem.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningState.h"
#include "Wafer/Planning/PhysicalDataflow/StructureSpecificStorageDomain.h"

#include "Wafer/Program/ProgramData.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace wafer::compiler::detail {

enum class FullFeasibilityStatus : uint8_t {
  Accepted,
  ExactRejection,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

struct FullFeasibilityStatistics {
  uint64_t evaluations = 0;
  uint64_t candidateActualizations = 0;
  uint64_t executableGateInvocations = 0;
  CardExecutableLoweringStatistics exactGates;
};

/// Move-only actual result for one complete ScheduledState. Accepted keeps
/// the one Q50.0 executable; rejected results keep only typed witness facts.
struct FullFeasibilityResult {
  FullFeasibilityStatus status = FullFeasibilityStatus::CompilerBug;
  std::optional<CardExecutableCompilationResult> compilation;
  std::vector<SemanticRootKey> causalRoots;
  std::string detail;

  bool isAccepted() const {
    return status == FullFeasibilityStatus::Accepted && compilation &&
           compilation->isAccepted();
  }
  bool isExactRejection() const {
    return status == FullFeasibilityStatus::ExactRejection;
  }
  CardExecutableLoweringResult takeExecutable() {
    return compilation->takeExecutable();
  }
};

FullFeasibilityResult evaluateCompleteCandidate(
    mlir::ModuleOp tensorProgram,
    const PhysicalDataflowPlanningProblem &problem, const ScheduledState &state,
    const EventGraph &eventGraph,
    llvm::ArrayRef<SlotLifetimeRequirement> slotLifetimes,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData,
    FullFeasibilityStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0,
    bool captureTileDataflowIRTrace = false);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_FULLFEASIBILITY_H
