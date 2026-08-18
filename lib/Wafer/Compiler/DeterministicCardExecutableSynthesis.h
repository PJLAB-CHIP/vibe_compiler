//===- DeterministicCardExecutableSynthesis.h ----------------*- C++ -*-===//

#ifndef WAFER_COMPILER_DETERMINISTICCARDEXECUTABLESYNTHESIS_H
#define WAFER_COMPILER_DETERMINISTICCARDEXECUTABLESYNTHESIS_H

#include "CardExecutableLowering.h"
#include "StructuredDAGPlacement.h"

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/Support/TargetPolicy.h"

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

/// Policy-free baseline work ledger. None of these values is persisted in IR
/// or package files, and the search statistics type is not in this API.
struct DeterministicBaselineLedger {
  uint64_t exactDemandSatisfiedEdges = 0;
  uint64_t spatialCoordinateQueries = 0;
  uint64_t spatialLegalizationTransitions = 0;
  analysis::ExactDemandStatus demandAbortStatus =
      analysis::ExactDemandStatus::Satisfied;
  std::string demandAbortDetail;
  uint64_t baselineSourcePreparations = 0;
  uint64_t baselineMaterializationPreparations = 0;
  uint64_t baselineCardModuleMaterializations = 0;
  uint64_t baselineRootShardMaterializations = 0;
  uint64_t baselineTileEntryMaterializations = 0;
  uint64_t baselineRegionSPMCapacityChecks = 0;
  uint64_t baselineRegionSPMCapacityOverflowProofs = 0;
  uint64_t baselineRegionSPMChecksRequiringFunctionScope = 0;
  uint64_t baselineFunctionScopedSPMCapacityChecks = 0;
  uint64_t baselineFunctionSPMCapacityOverflowProofs = 0;
  uint64_t baselineRegionSPMCapacityAnalysisFailures = 0;
  uint64_t baselineMaximumRegionSPMQueryWorkers = 0;
  uint64_t baselineMaximumFunctionSPMQueryWorkers = 0;
  uint64_t baselineTileIRPrints = 0;
  uint64_t materializationRejections = 0;
  uint64_t indeterminateCompilationFailures = 0;
  uint64_t rotatingSlotAllocationsMaterialized = 0;
  CardExecutableLoweringStatistics exactGates;
};

/// Semantic baseline result. Diagnostic IR inspection is written only to an
/// explicit caller-owned sink and is not carried by this result.
struct DeterministicCardExecutableSynthesisResult {
  explicit DeterministicCardExecutableSynthesisResult(
      CardExecutableLoweringResult executable)
      : executable(std::move(executable)) {}

  CardExecutableLoweringResult executable;
};

enum class DeterministicSpatialAdvance : uint8_t {
  Advanced,
  Exhausted,
};

mlir::FailureOr<DeterministicSpatialAdvance>
advanceDeterministicSpatialCoordinate(
    llvm::SmallVectorImpl<StructuredDAGNodePlacement> &placements,
    std::string *failureReason = nullptr);

llvm::SmallVector<int64_t, 4>
deriveCapacityTemporalShape(llvm::ArrayRef<int64_t> maximumShardShape,
                            uint64_t elementBytes, uint64_t tensorMultiplicity,
                            const TargetMemoryPolicy &memory,
                            unsigned additionalWaveRefinements);

/// Materializes and admits the deterministic functional baseline. The source
/// module is borrowed and unchanged. This boundary owns no candidate family,
/// score, selector, search statistics or printed-IR result field.
mlir::FailureOr<DeterministicCardExecutableSynthesisResult>
synthesizeDeterministicCardExecutable(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    llvm::raw_ostream &diagnostics, ProgramDataHandoff &programData,
    DeterministicBaselineLedger *baselineLedger = nullptr,
    unsigned tilePipelineParallelism = 0,
    std::vector<std::string> *tileDataflowIRTrace = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_DETERMINISTICCARDEXECUTABLESYNTHESIS_H
