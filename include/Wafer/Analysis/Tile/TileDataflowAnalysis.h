//===- TileDataflowAnalysis.h - Tile dataflow queries ----------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_TILE_TILEDATAFLOWANALYSIS_H
#define WAFER_ANALYSIS_TILE_TILEDATAFLOWANALYSIS_H

namespace mlir {
class Operation;
} // namespace mlir

namespace wafer::analysis {

/// Returns true when `root` contains a typed Tile dataflow operation consumed
/// by Tile-to-Instr conversion. TileRegionOp and TileYieldOp are structural
/// boundaries and therefore do not themselves require instruction lowering.
bool containsTileDataflowOperations(mlir::Operation *root);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_TILE_TILEDATAFLOWANALYSIS_H
