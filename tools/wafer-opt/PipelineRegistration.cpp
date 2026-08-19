//===- PipelineRegistration.cpp - wafer-opt named pipelines ------------===//

#include "PipelineRegistration.h"

#include "Wafer/Conversion/StableHLOToLinalg/Pipelines.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/Pipelines.h"
#include "Wafer/Transforms/MemoryPlanningPipelines.h"
#include "Wafer/Transforms/SpmdPipelines.h"

#include "mlir/Pass/PassRegistry.h"

void registerWaferOptPipelines() {
  mlir::PassPipelineRegistration<>(
      "wafer-normalize-imported-stablehlo",
      "Normalize imported StableHLO without crossing its legality boundary",
      [](mlir::OpPassManager &pm) {
        wafer::buildNormalizeImportedStablehloPipeline(pm);
      });
  mlir::PassPipelineRegistration<>(
      "wafer-legalize-stablehlo-to-structured-tensor",
      "Legalize supported StableHLO to structured tensor IR",
      [](mlir::OpPassManager &pm) {
        wafer::buildLegalizeStablehloToStructuredTensorPipeline(pm);
      });
  mlir::PassPipelineRegistration<>(
      "wafer-simplify-structured-tensor",
      "Apply bounded Wafer simplification to structured tensor IR",
      [](mlir::OpPassManager &pm) {
        wafer::buildSimplifyStructuredTensorPipeline(pm);
      });
  mlir::PassPipelineRegistration<>(
      "wafer-lower-stablehlo-to-linalg",
      "Lower StableHLO tensor IR to structured Linalg/Tensor IR",
      [](mlir::OpPassManager &pm) {
        wafer::buildStablehloToLinalgPipeline(pm);
      });
  mlir::PassPipelineRegistration<>(
      "wafer-lower-tile-region-to-instr",
      "Lower executable TileRegion ops to Instr IR and place required joins",
      [](mlir::OpPassManager &pm) {
        wafer::buildLowerTileRegionToInstrPipeline(pm);
      });
  mlir::PassPipelineRegistration<>(
      "wafer-bufferize-instr-functions",
      "Canonicalize and function-boundary bufferize Instr functions",
      [](mlir::OpPassManager &pm) {
        wafer::buildBufferizeInstrFunctionsPipeline(pm);
      });
  mlir::PassPipelineRegistration<>(
      "wafer-prepare-instr-for-memory-planning",
      "Prepare Instr function boundaries and joins for memory planning",
      [](mlir::OpPassManager &pm) {
        wafer::buildPrepareInstrForMemoryPlanningPipeline(pm);
      });
#ifdef WAFER_ENABLE_SHARDY
  mlir::PassPipelineRegistration<>(
      "wafer-propagate-stablehlo-sharding",
      "Apply Wafer sharding seeds and run Shardy propagation",
      [](mlir::OpPassManager &pm) {
        wafer::buildStablehloShardingPropagationPipeline(pm);
      });
#endif
}
