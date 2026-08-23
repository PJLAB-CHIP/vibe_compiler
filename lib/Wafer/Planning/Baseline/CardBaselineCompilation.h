//===- CardBaselineCompilation.h ----------------------------*- C++ -*-===//

#ifndef WAFER_COMPILER_CARDBASELINECOMPILATION_H
#define WAFER_COMPILER_CARDBASELINECOMPILATION_H

#include "Wafer/CodeGen/Executable/CardExecutableLowering.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Driver/Compilation.h"
#include "Wafer/Frontend/Program/Program.h"
#include "Wafer/Target/Core/TargetMemory.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler::detail {

/// Optional baseline compilation statistics. The production driver supplies
/// this sink only for an explicitly requested diagnostic run. Baseline
/// legality, control flow and output never depend on these values.
struct BaselineStatistics {
  uint64_t exactDemandSatisfiedEdges = 0;
  uint64_t spatialCoordinateQueries = 0;
  uint64_t actualSPMCapacityRejections = 0;
  uint64_t actualTemporalRefinements = 0;
  uint64_t baselineSourcePreparations = 0;
  uint64_t baselineMaterializationPreparations = 0;
  uint64_t baselineCardModuleMaterializations = 0;
  uint64_t baselineTileEntryMaterializations = 0;
  uint64_t baselineMaximumTileMaterializationWorkers = 1;
  uint64_t baselineTileIRPrints = 0;
  uint64_t materializationRejections = 0;
  uint64_t indeterminateCompilationFailures = 0;
  uint64_t rotatingSlotAllocationsMaterialized = 0;
  CardExecutableLoweringStatistics exactGates;
};

/// Semantic baseline result. Diagnostic IR snapshots are present only when
/// explicitly requested and move with the executable produced by the same
/// baseline compilation.
struct CardBaselineCompilationResult {
  explicit CardBaselineCompilationResult(
      CardExecutableLoweringResult executable,
      std::vector<std::string> tileDataflowIRTrace)
      : executable(std::move(executable)),
        tileDataflowIRTrace(std::move(tileDataflowIRTrace)) {}

  CardExecutableLoweringResult executable;
  std::vector<std::string> tileDataflowIRTrace;
};

/// Materializes and admits the deterministic functional baseline. The source
/// module is borrowed and unchanged. This boundary owns no candidate family,
/// score, selector or search statistics. Optional diagnostic IR remains owned
/// by the returned baseline result.
mlir::FailureOr<CardBaselineCompilationResult>
compileCardBaseline(mlir::ModuleOp tensorProgram,
                    const frontend::FrontendProgramVerificationResult &program,
                    const ExecutionConfig &executionConfig,
                    llvm::raw_ostream &diagnostics,
                    ProgramDataHandoff &programData,
                    BaselineStatistics *baselineStatistics = nullptr,
                    unsigned tilePipelineParallelism = 0,
                    bool captureTileDataflowIRTrace = false);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDBASELINECOMPILATION_H
