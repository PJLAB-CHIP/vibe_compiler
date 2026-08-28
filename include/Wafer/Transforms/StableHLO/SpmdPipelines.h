//===- SpmdPipelines.h - StableHLO sharding propagation ------*- C++ -*-===//
#pragma once

namespace mlir {
class OpPassManager;
}

namespace wafer {

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm);
#endif

} // namespace wafer
