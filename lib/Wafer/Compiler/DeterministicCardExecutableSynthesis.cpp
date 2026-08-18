//===- DeterministicCardExecutableSynthesis.cpp -----------------------===//

#include "DeterministicCardExecutableSynthesis.h"
#include "DeterministicCardExecutableSynthesisInternal.h"

namespace wafer::compiler::detail {

mlir::FailureOr<DeterministicSpatialAdvance>
advanceDeterministicSpatialCoordinate(
    llvm::SmallVectorImpl<StructuredDAGNodePlacement> &placements,
    std::string *failureReason) {
  return advanceDeterministicSpatialCoordinateImpl(placements, failureReason);
}

llvm::SmallVector<int64_t, 4>
deriveCapacityTemporalShape(llvm::ArrayRef<int64_t> maximumShardShape,
                            uint64_t elementBytes, uint64_t tensorMultiplicity,
                            const TargetMemoryPolicy &memory,
                            unsigned additionalWaveRefinements) {
  return deriveCapacityTemporalShapeImplForBaseline(
      maximumShardShape, elementBytes, tensorMultiplicity, memory,
      additionalWaveRefinements);
}

mlir::FailureOr<DeterministicCardExecutableSynthesisResult>
synthesizeDeterministicCardExecutable(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    llvm::raw_ostream &diagnostics, ProgramDataHandoff &programData,
    DeterministicBaselineLedger *baselineLedger,
    unsigned tilePipelineParallelism,
    std::vector<std::string> *tileDataflowIRTrace) {
  return synthesizeDeterministicCardExecutableImpl(
      tensorProgram, program, executionConfig, diagnostics, programData,
      baselineLedger, tilePipelineParallelism, tileDataflowIRTrace);
}

} // namespace wafer::compiler::detail
