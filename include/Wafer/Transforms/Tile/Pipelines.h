//===- Pipelines.h - Tile current-IR pipelines -------------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_PIPELINES_H
#define WAFER_TRANSFORMS_TILE_PIPELINES_H

namespace mlir {
class OpPassManager;
}

namespace wafer {

void buildResolveLayoutsAndBufferizePipeline(mlir::OpPassManager &pm);

} // namespace wafer

#endif // WAFER_TRANSFORMS_TILE_PIPELINES_H
