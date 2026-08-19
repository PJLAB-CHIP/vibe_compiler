//===- Pipelines.cpp - StableHLO sharding propagation ------------------===//

#include "Wafer/Transforms/SpmdPipelines.h"

#ifdef WAFER_ENABLE_SHARDY

#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"

namespace wafer {

void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createApplyDefaultSpmdShardingPass());
  mlir::sdy::addPropagationPipeline(pm);
}

} // namespace wafer

#endif
