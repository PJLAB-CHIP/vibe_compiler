//===- Pipelines.cpp - Instr memory preparation pipelines --------------===//

#include "Wafer/Transforms/MemoryPlanningPipelines.h"

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace wafer {

void buildBufferizeInstrFunctionsPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createBufferizeInstrFunctionBoundariesPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void buildPrepareInstrForMemoryPlanningPipeline(mlir::OpPassManager &pm) {
  buildBufferizeInstrFunctionsPipeline(pm);
  pm.nest<mlir::func::FuncOp>().addPass(createRebuildRequiredNCCJoinsPass());
}

} // namespace wafer
