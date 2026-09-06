//===- Pipelines.cpp - StableHLO to structured tensor pipelines --------===//

#include "Wafer/Conversion/StableHLOToLinalg/Pipelines.h"

#include "Wafer/Conversion/Passes.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

void buildNormalizeAttentionPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(createFormAttentionOpsPass());
}

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm) {
  // The program contract permits private pure helpers behind one public
  // entry. Inline them before StableHLO legalization so downstream Tensor IR
  // and physical-dataflow stages retain their single-function boundary.
  pm.addPass(mlir::createInlinerPass());
  buildNormalizeImportedStablehloPipeline(pm);
  buildLegalizeStablehloToStructuredTensorPipeline(pm);
  buildSimplifyStructuredTensorPipeline(pm);
  pm.addPass(mlir::createCanonicalizerPass());
  buildNormalizeAttentionPipeline(pm);
  pm.addNestedPass<mlir::func::FuncOp>(
      createNormalizeStructuredTensorGraphPass());
}

} // namespace wafer
