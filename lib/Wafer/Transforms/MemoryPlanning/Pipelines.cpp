//===- Pipelines.cpp - Instr memory preparation pipelines --------------===//

#include "Wafer/Transforms/MemoryPlanningPipelines.h"

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace wafer {

void buildBufferizeInstrFunctionsPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createBufferizeInstrFunctionBoundariesPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());
}

} // namespace wafer
