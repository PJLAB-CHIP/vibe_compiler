//===- Pipelines.h - Wafer named pipeline registration ---------*- C++ -*-===//

#ifndef WAFER_PIPELINES_PIPELINES_H
#define WAFER_PIPELINES_PIPELINES_H

#include <cstdint>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace wafer {

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm);
void buildFormLogicalGroupsPipeline(mlir::OpPassManager &pm);
void buildLowerGroupsToTileRegionPipeline(mlir::OpPassManager &pm);
void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm);
void buildLowerGroupsToInstrPipeline(mlir::OpPassManager &pm);
void buildPlanSPMMemoryPipeline(mlir::OpPassManager &pm);
void buildPlanDDRMemoryPipeline(mlir::OpPassManager &pm);
void buildLowerGroupsToMemoryPlannedInstrPipeline(mlir::OpPassManager &pm);
void buildLowerGroupsToDDRMemoryPlannedInstrPipeline(mlir::OpPassManager &pm);
void buildLowerGroupsToSelectedInstrPipeline(mlir::OpPassManager &pm);

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm,
                                               int64_t defaultTileCount = 16);
#endif

void registerWaferPipelines();

} // namespace wafer

#endif // WAFER_PIPELINES_PIPELINES_H
