//===- CurrentIRExecutablePipeline.h - Current IR downstream -*- C++ -*-===//

#ifndef WAFER_DRIVER_CURRENTIREXECUTABLEPIPELINE_H
#define WAFER_DRIVER_CURRENTIREXECUTABLEPIPELINE_H

#include "Wafer/Driver/ExecutableCompilation.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/LoadPipelining.h"

#include <cstdint>
#include <vector>

namespace wafer::compiler::detail {

enum class CommunicationProposalPolicy { Fixed, DependencyOrdered };

struct CurrentIRDownstreamOptions {
  LayoutMaterializationPlacement movementPlacement =
      LayoutMaterializationPlacement::FirstUse;
  bool distanceOneLoadPipeline = false;
  CommunicationProposalPolicy communication =
      CommunicationProposalPolicy::Fixed;
  unsigned tilePipelineParallelism = 0;
  bool captureTileDataflowIR = false;
  TileSPMCapacityObserver capacityObserver = nullptr;
};

struct CurrentIRDownstreamStatistics {
  uint64_t pipelinedLoops = 0;
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

/// Applies the common storage/movement transformations before querying or
/// binding any pipeline choice. Rebuilds actual owners and verifies the result.
mlir::LogicalResult
optimizeCurrentIRStorage(mlir::ModuleOp module,
                         StructuredMaterializationRelations &relations);

/// Consumes the optimized physical current IR produced by
/// optimizeCurrentIRStorage and executes loop pipelining, standalone Tile
/// fanout, Tile-to-Instr, exact transfer cleanup,
/// fresh completion and the unique actual memory/target leaf. It never
/// constructs a structural choice, repairs a failed candidate or rebuilds an
/// accepted owner.
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
