//===- PipelineRegistration.cpp - wafer-opt named pipelines ------------===//

#include "PipelineRegistration.h"

#include "Wafer/Conversion/StableHLOToLinalg/Pipelines.h"
#include "Wafer/Conversion/TileToInstr/Pipelines.h"
#include "Wafer/Transforms/Instr/MemoryPlanningPipelines.h"
#include "Wafer/Transforms/StableHLO/FrontendVerification.h"
#include "Wafer/Transforms/StableHLO/SpmdPipelines.h"

#include "mlir/Pass/PassRegistry.h"

void registerWaferOptPipelines() {
  mlir::PassPipelineRegistration<>(
      "wafer-frontend-verification",
      "Verify imported frontend IR without producing a program directory",
      [](mlir::OpPassManager &pm) {
        wafer::buildFrontendVerificationPipeline(pm);
      });
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
      "wafer-normalize-attention",
      "Form self-contained attention semantics in structured tensor IR",
      [](mlir::OpPassManager &pm) {
        wafer::buildNormalizeAttentionPipeline(pm);
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
#ifdef WAFER_ENABLE_SHARDY
  mlir::PassPipelineRegistration<>(
      "wafer-propagate-stablehlo-sharding",
      "Apply Wafer sharding seeds and run Shardy propagation",
      [](mlir::OpPassManager &pm) {
        wafer::buildStablehloShardingPropagationPipeline(pm);
      });
#endif
}
