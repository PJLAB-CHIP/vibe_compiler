//===- Pipelines.cpp - StableHLO to structured tensor pipelines --------===//

#include "Wafer/Conversion/StableHLOToLinalg/Pipelines.h"

#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace wafer {

void buildNormalizeImportedStablehloPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createNormalizeStablehloCollectivesPass());
  pm.addPass(createFoldDefaultStablehloExecutionIdsPass());
  pm.addPass(createFoldConstantIntegerTensorCastsPass());
  pm.addPass(createLowerStaticStablehloConcatenatePass());
}

void buildLegalizeStablehloToStructuredTensorPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createLegalizeStablehloToLinalgPass());
}

void buildSimplifyStructuredTensorPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createFoldStaticTensorOpsPass());
}

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm) {
  buildNormalizeImportedStablehloPipeline(pm);
  buildLegalizeStablehloToStructuredTensorPipeline(pm);
  buildSimplifyStructuredTensorPipeline(pm);
  pm.addPass(mlir::createCanonicalizerPass());
}

} // namespace wafer
