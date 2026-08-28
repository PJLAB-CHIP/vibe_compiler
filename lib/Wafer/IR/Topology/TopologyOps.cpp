//===- TopologyOps.cpp - Wafer target topology verifier implementation ----===//

#include "Wafer/IR/Topology/TargetExecutionFacts.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>

using namespace wafer;

void wafer::cloneTargetExecutionFacts(mlir::ModuleOp sourceModule,
                                      mlir::ModuleOp destinationModule) {
  mlir::OpBuilder builder(destinationModule.getBodyRegion());
  for (TargetTopologyOp topology : sourceModule.getOps<TargetTopologyOp>())
    builder.clone(*topology.getOperation());
  for (ExecutionMeshOp mesh : sourceModule.getOps<ExecutionMeshOp>())
    builder.clone(*mesh.getOperation());
}

namespace {

static bool isSupportedCardInterconnect(llvm::StringRef interconnect) {
  return interconnect == "mesh" || interconnect == "torus";
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
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

static mlir::FailureOr<int64_t>
computePartitionCount(mlir::Operation *op, llvm::ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    op->emitOpError("shape must not be empty");
    return mlir::failure();
  }
  int64_t partitionCount = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      op->emitOpError("shape entries must be positive");
      return mlir::failure();
    }
    if (!checkedMul(partitionCount, dim, partitionCount)) {
      op->emitOpError("partition count is too large to verify");
      return mlir::failure();
    }
  }
  return partitionCount;
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

mlir::LogicalResult ExecutionMeshOp::verify() {
  if (!mlir::isa<mlir::ModuleOp>(getOperation()->getParentOp()))
    return emitOpError("must be a direct module member");

  for (mlir::NamedAttribute attr : getOperation()->getAttrs()) {
    llvm::StringRef name = attr.getName().getValue();
    if (name != getSymNameAttrName() && name != getAxesAttrName() &&
        name != getShapeAttrName())
      return emitOpError("does not accept attribute '")
             << name << "'; it contains only logical axes and shape";
  }

  mlir::ArrayAttr axes = getAxesAttr();
  llvm::ArrayRef<int64_t> shape = getShapeAttr().asArrayRef();
  if (axes.empty())
    return emitOpError("axes must not be empty");
  if (axes.size() != shape.size())
    return emitOpError("axes count must match shape rank");

  llvm::SmallVector<llvm::StringRef, 4> seenAxes;
  for (mlir::Attribute axisAttr : axes) {
    auto axis = mlir::cast<mlir::StringAttr>(axisAttr).getValue();
    if (axis.empty())
      return emitOpError("axis name must not be empty");
    if (llvm::is_contained(seenAxes, axis))
      return emitOpError("axis name must be unique");
    seenAxes.push_back(axis);
  }

  if (mlir::failed(computePartitionCount(getOperation(), shape)))
    return mlir::failure();

  return mlir::success();
}
