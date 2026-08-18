//===- DeterministicCardExecutableSynthesisInternal.h --------*- C++ -*-===//

#ifndef WAFER_COMPILER_DETERMINISTICCARDEXECUTABLESYNTHESISINTERNAL_H
#define WAFER_COMPILER_DETERMINISTICCARDEXECUTABLESYNTHESISINTERNAL_H

#include "DeterministicCardExecutableSynthesis.h"

namespace wafer::compiler::detail {

mlir::FailureOr<DeterministicSpatialAdvance>
advanceDeterministicSpatialCoordinateImpl(
    llvm::SmallVectorImpl<StructuredDAGNodePlacement> &placements,
    std::string *failureReason);

llvm::SmallVector<int64_t, 4> deriveCapacityTemporalShapeImplForBaseline(
    llvm::ArrayRef<int64_t> maximumShardShape, uint64_t elementBytes,
    uint64_t tensorMultiplicity, const TargetMemoryPolicy &memory,
    unsigned additionalWaveRefinements);

mlir::FailureOr<DeterministicCardExecutableSynthesisResult>
synthesizeDeterministicCardExecutableImpl(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData,
    DeterministicBaselineLedger *baselineLedger,
    unsigned tilePipelineParallelism,
    std::vector<std::string> *tileDataflowIRTrace);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_DETERMINISTICCARDEXECUTABLESYNTHESISINTERNAL_H
