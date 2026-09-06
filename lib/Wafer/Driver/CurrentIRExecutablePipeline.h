//===- CurrentIRExecutablePipeline.h - Current IR downstream -*- C++ -*-===//

#ifndef WAFER_DRIVER_CURRENTIREXECUTABLEPIPELINE_H
#define WAFER_DRIVER_CURRENTIREXECUTABLEPIPELINE_H

#include "Wafer/Driver/ExecutableCompilation.h"
#include "Wafer/Transforms/Tile/ExecutionStructure.h"

#include <cstdint>
#include <vector>

namespace wafer::compiler::detail {

struct CurrentIRDownstreamOptions {
  std::vector<TilePipelineChoice> executionPipelines;
  ExecutionStructureLimits executionLimits;
  unsigned tilePipelineParallelism = 0;
  bool captureTileDataflowIR = false;
  /// Stop after completion-closed canonical Instr and actual resource
  /// analysis. Search uses this leaf to rank candidates before target LLVM/ABI
  /// lowering; baseline and finalization leave it false.
  bool stopBeforeTarget = false;
};

struct CurrentIRDownstreamStatistics {
  uint64_t preTargetResourceAnalyses = 0;
  uint64_t targetFinalizations = 0;
  uint64_t materializedExecutionPipelines = 0;
  uint64_t tileRegionsLowered = 0;
  uint64_t instructionOperations = 0;
  uint64_t rdmaOperations = 0;
  uint64_t wdmaOperations = 0;
  uint64_t gatherScatterOperations = 0;
  uint64_t dteSendOperations = 0;
  uint64_t dteBroadcastOperations = 0;
  uint64_t dteScatterOperations = 0;
  uint64_t dteUnicastSendsCoalesced = 0;
  uint64_t dteReceiveOperations = 0;
  uint64_t dteWaitOperations = 0;
  uint64_t nccJoinOperations = 0;
};

/// Consumes one physical, movement-closed current-IR candidate and executes
/// the shared downstream stage sequence: execution structure, standalone Tile
/// fanout, Tile-to-Instr, exact transfer cleanup, fresh completion and the
/// unique actual memory/target leaf. It never constructs a structural choice,
/// repairs a failed candidate or rebuilds an accepted owner.
ExecutableCompilationResult compileCurrentIRCandidateToExecutable(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    StructuredMaterializationRelations relations, CardId expectedCardId,
    llvm::ArrayRef<TileId> expectedTileIds,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData,
    const CurrentIRDownstreamOptions &options = {},
    CurrentIRDownstreamStatistics *downstreamStatistics = nullptr,
    ExecutableLoweringStatistics *executableStatistics = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_CURRENTIREXECUTABLEPIPELINE_H
