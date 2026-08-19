//===- Pipelines.h - TileRegion to Instr pipeline ------------*- C++ -*-===//
#pragma once

namespace mlir {
class OpPassManager;
}

namespace wafer {

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm);

} // namespace wafer
