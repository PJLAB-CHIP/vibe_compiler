//===- Pipelines.cpp - Tile current-IR pipelines -----------------------===//

#include "Wafer/Transforms/Tile/Pipelines.h"

#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace wafer {

void buildResolveLayoutsAndBufferizePipeline(mlir::OpPassManager &pm) {
  pm.addPass(createResolveCurrentLayoutsAndBufferizePass());
}

} // namespace wafer
