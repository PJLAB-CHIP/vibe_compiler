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
void buildPlaceSPMBuffersPipeline(mlir::OpPassManager &pm);
void buildLowerGroupsToPlacedInstrPipeline(mlir::OpPassManager &pm);

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm,
                                               int64_t defaultTileCount = 16);
#endif

void registerWaferPipelines();

} // namespace wafer

#endif // WAFER_PIPELINES_PIPELINES_H
