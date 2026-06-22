//===- TopologyOps.cpp - Wafer target topology verifier implementation ----===//

#include "Wafer/IR/WaferDialect.h"

using namespace wafer;

namespace {

static bool isSupportedCardInterconnect(llvm::StringRef interconnect) {
  return interconnect == "mesh" || interconnect == "torus";
}

static mlir::LogicalResult verifyGrid(mlir::Operation *op,
                                      llvm::StringRef attrName,
                                      llvm::ArrayRef<int64_t> grid) {
  if (grid.size() != 2)
    return op->emitOpError()
           << attrName << " must contain row and column counts";
  if (grid[0] <= 0 || grid[1] <= 0)
    return op->emitOpError() << attrName << " entries must be positive";
  return mlir::success();
}

} // namespace

mlir::LogicalResult TargetTopologyOp::verify() {
  llvm::ArrayRef<int64_t> cardGrid = getCardGridAttr().asArrayRef();
  if (mlir::failed(verifyGrid(getOperation(), "card_grid", cardGrid)))
    return mlir::failure();

  llvm::StringRef cardInterconnect = getCardInterconnectAttr().getValue();
  if (!isSupportedCardInterconnect(cardInterconnect))
    return emitOpError("card_interconnect must be mesh or torus");

  llvm::ArrayRef<int64_t> tileGrid = getTileGridAttr().asArrayRef();
  if (mlir::failed(verifyGrid(getOperation(), "tile_grid", tileGrid)))
    return mlir::failure();

  llvm::ArrayRef<int64_t> unavailableTiles =
      getUnavailableTilesAttr().asArrayRef();
  if (unavailableTiles.size() % 4 != 0)
    return emitOpError(
        "unavailable_tiles must contain card_y/card_x/tile_y/tile_x tuples");

  for (size_t index = 0; index < unavailableTiles.size(); index += 4) {
    int64_t cardY = unavailableTiles[index];
    int64_t cardX = unavailableTiles[index + 1];
    int64_t tileY = unavailableTiles[index + 2];
    int64_t tileX = unavailableTiles[index + 3];
    if (cardY < 0 || cardY >= cardGrid[0])
      return emitOpError("unavailable_tiles card_y coordinate ")
             << cardY << " is outside card_grid";
    if (cardX < 0 || cardX >= cardGrid[1])
      return emitOpError("unavailable_tiles card_x coordinate ")
             << cardX << " is outside card_grid";
    if (tileY < 0 || tileY >= tileGrid[0])
      return emitOpError("unavailable_tiles tile_y coordinate ")
             << tileY << " is outside tile_grid";
    if (tileX < 0 || tileX >= tileGrid[1])
      return emitOpError("unavailable_tiles tile_x coordinate ")
             << tileX << " is outside tile_grid";

    for (size_t priorIndex = 0; priorIndex < index; priorIndex += 4) {
      if (unavailableTiles[priorIndex] == cardY &&
          unavailableTiles[priorIndex + 1] == cardX &&
          unavailableTiles[priorIndex + 2] == tileY &&
          unavailableTiles[priorIndex + 3] == tileX)
        return emitOpError(
            "unavailable_tiles contains duplicate tile coordinate");
    }
  }

  return mlir::success();
}
