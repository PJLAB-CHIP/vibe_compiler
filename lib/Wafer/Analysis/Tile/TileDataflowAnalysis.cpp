//===- TileDataflowAnalysis.cpp - Tile dataflow queries -----------------===//

#include "Wafer/Analysis/Tile/TileDataflowAnalysis.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Operation.h"

namespace wafer::analysis {

bool containsTileDataflowOperations(mlir::Operation *root) {
  if (!root)
    return false;
  bool found = false;
  root->walk([&](mlir::Operation *operation) {
    if (!mlir::isa<WaferTileDataflowOpInterface>(operation))
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

} // namespace wafer::analysis
