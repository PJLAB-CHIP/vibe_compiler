//===- TiledOutputStores.h - Forward collected tiles to DDR -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_TILEDOUTPUTSTORES_H
#define WAFER_TRANSFORMS_TILE_TILEDOUTPUTSTORES_H

namespace mlir {
class ModuleOp;
}

namespace wafer::compiler::detail {
struct BoundaryMovementStatistics;

/// Replace a write-only SPM collection buffer and its terminal DDR stores with
/// stores of the already materialized tiles into those same DDR destinations.
void materializeTiledOutputStores(mlir::ModuleOp module,
                                  BoundaryMovementStatistics &statistics);
} // namespace wafer::compiler::detail

#endif
