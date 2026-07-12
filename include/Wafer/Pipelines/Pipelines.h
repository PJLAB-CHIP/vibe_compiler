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
void buildLowerGroupsToTileRegionPipeline(mlir::OpPassManager &pm,
                                          int64_t logicalRank);
void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm);
void buildLowerGroupsToInstrPipeline(mlir::OpPassManager &pm,
                                     int64_t logicalRank);
void buildPlanSPMMemoryPipeline(mlir::OpPassManager &pm);
void buildPlanDDRMemoryPipeline(mlir::OpPassManager &pm);
void buildLowerGroupsToMemoryPlannedInstrPipeline(mlir::OpPassManager &pm,
                                                  int64_t logicalRank);
void buildLowerGroupsToDDRMemoryPlannedInstrPipeline(mlir::OpPassManager &pm,
                                                     int64_t logicalRank);
void buildLowerGroupsToTargetLLVMPipeline(mlir::OpPassManager &pm,
                                          int64_t logicalRank);
void buildLowerGroupsToSelectedInstrPipeline(mlir::OpPassManager &pm,
                                             int64_t logicalRank);

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm);
#endif

void registerWaferPipelines();

} // namespace wafer

#endif // WAFER_PIPELINES_PIPELINES_H
