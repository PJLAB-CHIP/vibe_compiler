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

/// Canonicalizes and bufferizes one canonical, unplaced Instr sibling without
/// assigning SPM or DDR offsets. Executable finalization uses this boundary so
/// completion can be erased and rebuilt from the final buffer/effect IR before
/// lifetime and SPM planning observe it.
void buildPreparePhysicalTileCandidatePipeline(mlir::OpPassManager &pm);

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm);
void buildPlanSPMMemoryPipeline(mlir::OpPassManager &pm);
void buildPlanDDRMemoryPipeline(mlir::OpPassManager &pm);

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm);
#endif

void registerWaferPipelines();

} // namespace wafer

#endif // WAFER_PIPELINES_PIPELINES_H
