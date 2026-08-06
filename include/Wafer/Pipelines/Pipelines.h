//===- Pipelines.h - Wafer named pipeline registration ---------*- C++ -*-===//

#ifndef WAFER_PIPELINES_PIPELINES_H
#define WAFER_PIPELINES_PIPELINES_H

#include "Wafer/Target/TargetIdentity.h"

#include <cstdint>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace wafer {

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm);
/// Finalizes one independently scheduled rank candidate.  The input already
/// contains explicit tiled task/instruction dataflow; this pipeline closes
/// function-boundary bufferization and recomputes rank-wide SPM/DDR offsets.
/// It does not perform candidate selection or cross-rank acceptance.
void buildFinalizeScheduledTensorProgramPipeline(mlir::OpPassManager &pm);

/// Finalizes canonical Instr after terminal Tile-to-Instr conversion through
/// function-boundary bufferization and SPM placement. DDR placement remains
/// absent so the all-rank coordinator can recompute it on a disposable
/// whole-variant clone.
void buildFinalizeScheduledRankCandidatePipeline(mlir::OpPassManager &pm);

/// Canonicalizes and bufferizes one canonical, unplaced Instr sibling without
/// assigning SPM or DDR offsets. Terminal scheduling uses this boundary so
/// completion can be erased and rebuilt from the final buffer/effect IR before
/// lifetime and SPM planning observe it.
void buildPrepareScheduledRankCandidatePipeline(mlir::OpPassManager &pm);

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm);
void buildPlanSPMMemoryPipeline(mlir::OpPassManager &pm);
void buildPlanDDRMemoryPipeline(mlir::OpPassManager &pm);

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm);
#endif

void registerWaferPipelines();

} // namespace wafer

#endif // WAFER_PIPELINES_PIPELINES_H
