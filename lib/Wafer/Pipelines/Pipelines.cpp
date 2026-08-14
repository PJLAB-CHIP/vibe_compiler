//===- Pipelines.cpp - Wafer named pipeline registration -----------------===//

#include "Wafer/Pipelines/Pipelines.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#endif

namespace wafer {

void buildNormalizeImportedStablehloPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createNormalizeStablehloCollectivesPass());
  pm.addPass(createFoldDefaultStablehloExecutionIdsPass());
  pm.addPass(createFoldConstantIntegerTensorCastsPass());
  pm.addPass(createLowerStaticStablehloConcatenatePass());
}

void buildLegalizeStablehloToStructuredTensorPipeline(
    mlir::OpPassManager &pm) {
  pm.addPass(createLegalizeStablehloToLinalgPass());
}

void buildSimplifyStructuredTensorPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createFoldStaticTensorOpsPass());
}

void addInstrFunctionBoundaryBufferizationPass(mlir::OpPassManager &pm) {
  pm.addPass(createBufferizeInstrFunctionBoundariesPass());
}

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm) {
  buildNormalizeImportedStablehloPipeline(pm);
  buildLegalizeStablehloToStructuredTensorPipeline(pm);
  buildSimplifyStructuredTensorPipeline(pm);
  pm.addPass(mlir::createCanonicalizerPass());
}

void buildBufferizeInstrFunctionsPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  addInstrFunctionBoundaryBufferizationPass(pm);
  pm.addPass(mlir::createCanonicalizerPass());
}

void buildPrepareInstrForMemoryPlanningPipeline(mlir::OpPassManager &pm) {
  buildBufferizeInstrFunctionsPipeline(pm);
  addRecomputeRequiredNCCJoinPlacementPass(
      pm.nest<mlir::func::FuncOp>());
}

void addMaterializeTargetTopologyPass(
    mlir::OpPassManager &pm,
    const MaterializeTargetTopologyPassOptions &options) {
  pm.addPass(createMaterializeTargetTopologyPass(options));
}

void addMaterializeExecutionMeshPass(
    mlir::OpPassManager &pm,
    const MaterializeExecutionMeshPassOptions &options) {
  pm.addPass(createMaterializeExecutionMeshPass(options));
}

void addTileRegionToInstrConversionPass(mlir::OpPassManager &pm) {
  pm.addPass(createConvertTileRegionToInstrPass());
}

void addRequiredNCCJoinPlacementPass(mlir::OpPassManager &pm) {
  pm.addPass(createPlaceRequiredNCCJoinsPass());
}

void addRecomputeRequiredNCCJoinPlacementPass(mlir::OpPassManager &pm) {
  pm.addPass(createRebuildRequiredNCCJoinsPass());
}

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm) {
  mlir::OpPassManager &functionPM = pm.nest<mlir::func::FuncOp>();
  addTileRegionToInstrConversionPass(functionPM.nest<TileRegionOp>());
  addRequiredNCCJoinPlacementPass(functionPM);
}

void addAssignSPMOffsetsPass(mlir::OpPassManager &pm,
                             const PlanSPMMemoryPassOptions &options,
                             SPMMemoryPlanningFailure *failure) {
  pm.addPass(createPlanSPMMemoryPassWithFailure(options, failure));
}

void addAssignDDROffsetsPass(mlir::OpPassManager &pm,
                             const PlanDDRMemoryPassOptions &options) {
  pm.addPass(createPlanDDRMemoryPass(options));
}

void addLowerAffineControlAndIndexingPass(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createLowerAffinePass());
}

void addLowerInstrToTargetLLVMPass(
    mlir::OpPassManager &pm, const TargetConversionRequest &request) {
  pm.addPass(createLowerInstrToTargetLLVMPass(request));
}

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createApplyDefaultSpmdShardingPass());
  mlir::sdy::addPropagationPipeline(pm);
}
#endif

void registerWaferPipelines() {
  static bool registered = [] {
    mlir::PassPipelineRegistration<>(
        "wafer-normalize-imported-stablehlo",
        "Normalize imported StableHLO without crossing its legality boundary",
        [](mlir::OpPassManager &pm) {
          buildNormalizeImportedStablehloPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-legalize-stablehlo-to-structured-tensor",
        "Legalize supported StableHLO to structured tensor IR",
        [](mlir::OpPassManager &pm) {
          buildLegalizeStablehloToStructuredTensorPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-simplify-structured-tensor",
        "Apply bounded Wafer simplification to structured tensor IR",
        [](mlir::OpPassManager &pm) {
          buildSimplifyStructuredTensorPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-stablehlo-to-linalg",
        "Lower StableHLO tensor IR to structured Linalg/Tensor IR",
        [](mlir::OpPassManager &pm) { buildStablehloToLinalgPipeline(pm); });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-tile-region-to-instr",
        "Lower executable wafer.tile.region ops to wafer.instr IR and "
        "place function-required NCC joins",
        [](mlir::OpPassManager &pm) {
          buildLowerTileRegionToInstrPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-bufferize-instr-functions",
        "Canonicalize and function-boundary bufferize Instr functions without "
        "assigning memory offsets",
        [](mlir::OpPassManager &pm) {
          buildBufferizeInstrFunctionsPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-prepare-instr-for-memory-planning",
        "Bufferize Instr function boundaries and recompute function-required "
        "NCC joins before memory planning",
        [](mlir::OpPassManager &pm) {
          buildPrepareInstrForMemoryPlanningPipeline(pm);
        });
#ifdef WAFER_ENABLE_SHARDY
    mlir::PassPipelineRegistration<>(
        "wafer-propagate-stablehlo-sharding",
        "Apply Wafer default StableHLO/SDY sharding seeds when needed and run "
        "Shardy propagation",
        [](mlir::OpPassManager &pm) {
          buildStablehloShardingPropagationPipeline(pm);
        });
#endif
    return true;
  }();
  (void)registered;
}

} // namespace wafer
