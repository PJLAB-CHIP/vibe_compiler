//===- Pipelines.h - Wafer named pipeline registration ---------*- C++ -*-===//

#ifndef WAFER_PIPELINES_PIPELINES_H
#define WAFER_PIPELINES_PIPELINES_H

#include "Wafer/Target/TargetProfile.h"

#include <cstdint>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace wafer {

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm);
/// Schedules a verified per-rank structured tensor program and commits the
/// accepted, memory-planned instruction program.  The input contract is
/// Linalg/Tensor/SCF SSA dataflow; callers do not form or serialize scheduling
/// boundaries.
void buildScheduleTensorProgramToSelectedInstrPipeline(mlir::OpPassManager &pm,
                                                       int64_t logicalRank);

/// Finalizes one independently scheduled rank candidate.  The input already
/// contains explicit tiled task/instruction dataflow; this pipeline closes
/// function-boundary bufferization and recomputes rank-wide SPM/DDR offsets.
/// It does not perform candidate selection or cross-rank acceptance.
void buildFinalizeScheduledTensorProgramPipeline(mlir::OpPassManager &pm);

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm);
void buildPlanSPMMemoryPipeline(mlir::OpPassManager &pm);
void buildPlanDDRMemoryPipeline(mlir::OpPassManager &pm);

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm);
#endif

void registerWaferPipelines();

} // namespace wafer

#endif // WAFER_PIPELINES_PIPELINES_H
