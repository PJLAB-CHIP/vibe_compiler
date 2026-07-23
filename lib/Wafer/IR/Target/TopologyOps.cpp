//===- TopologyOps.cpp - Wafer target topology verifier implementation ----===//

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/IR/Target/TopologyUtils.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace wafer;

void wafer::cloneTargetExecutionFacts(mlir::ModuleOp sourceModule,
                                      mlir::ModuleOp destinationModule) {
  mlir::OpBuilder builder(destinationModule.getBodyRegion());
  for (TargetTopologyOp topology :
       sourceModule.getOps<TargetTopologyOp>())
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

static bool isUnavailableEndpoint(llvm::ArrayRef<int64_t> unavailableTiles,
                                  int64_t cardY, int64_t cardX, int64_t tileY,
                                  int64_t tileX) {
  for (size_t index = 0; index < unavailableTiles.size(); index += 4) {
    if (unavailableTiles[index] == cardY &&
        unavailableTiles[index + 1] == cardX &&
        unavailableTiles[index + 2] == tileY &&
        unavailableTiles[index + 3] == tileX)
      return true;
  }
  return false;
}

static mlir::FailureOr<int64_t>
computeEndpointCount(mlir::Operation *op, llvm::ArrayRef<int64_t> cardGrid,
                     llvm::ArrayRef<int64_t> tileGrid) {
  int64_t cardCount = 0;
  int64_t rowCount = 0;
  int64_t endpointCount = 0;
  if (!checkedMul(cardGrid[0], cardGrid[1], cardCount) ||
      !checkedMul(cardCount, tileGrid[0], rowCount) ||
      !checkedMul(rowCount, tileGrid[1], endpointCount)) {
    op->emitOpError("endpoint count is too large to verify");
    return mlir::failure();
  }
  return endpointCount;
}

static mlir::FailureOr<int64_t>
computeRankCount(mlir::Operation *op, llvm::ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    op->emitOpError("shape must not be empty");
    return mlir::failure();
  }
  int64_t rankCount = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      op->emitOpError("shape entries must be positive");
      return mlir::failure();
    }
    if (!checkedMul(rankCount, dim, rankCount)) {
      op->emitOpError("rank count is too large to verify");
      return mlir::failure();
    }
  }
  return rankCount;
}

static int64_t linearizeEndpoint(llvm::ArrayRef<int64_t> cardGrid,
                                 llvm::ArrayRef<int64_t> tileGrid,
                                 int64_t cardY, int64_t cardX, int64_t tileY,
                                 int64_t tileX) {
  int64_t cardIndex = cardY * cardGrid[1] + cardX;
  int64_t tileIndex = tileY * tileGrid[1] + tileX;
  return cardIndex * tileGrid[0] * tileGrid[1] + tileIndex;
}

static void delinearizeEndpoint(int64_t index, llvm::ArrayRef<int64_t> cardGrid,
                                llvm::ArrayRef<int64_t> tileGrid,
                                int64_t &cardY, int64_t &cardX, int64_t &tileY,
                                int64_t &tileX) {
  int64_t tileCount = tileGrid[0] * tileGrid[1];
  int64_t cardIndex = index / tileCount;
  int64_t tileIndex = index % tileCount;
  cardY = cardIndex / cardGrid[1];
  cardX = cardIndex % cardGrid[1];
  tileY = tileIndex / tileGrid[1];
  tileX = tileIndex % tileGrid[1];
}

static mlir::FailureOr<std::vector<char>>
buildAvailableEndpointMask(mlir::Operation *op,
                           llvm::ArrayRef<int64_t> cardGrid,
                           llvm::ArrayRef<int64_t> tileGrid,
                           llvm::ArrayRef<int64_t> unavailableTiles) {
  mlir::FailureOr<int64_t> endpointCount =
      computeEndpointCount(op, cardGrid, tileGrid);
  if (mlir::failed(endpointCount))
    return mlir::failure();
  if (static_cast<uint64_t>(*endpointCount) >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    op->emitOpError("endpoint count is too large to verify");
    return mlir::failure();
  }

  std::vector<char> available(static_cast<size_t>(*endpointCount), true);
  for (size_t index = 0; index < unavailableTiles.size(); index += 4) {
    if (unavailableTiles[index] < 0 || unavailableTiles[index] >= cardGrid[0] ||
        unavailableTiles[index + 1] < 0 ||
        unavailableTiles[index + 1] >= cardGrid[1] ||
        unavailableTiles[index + 2] < 0 ||
        unavailableTiles[index + 2] >= tileGrid[0] ||
        unavailableTiles[index + 3] < 0 ||
        unavailableTiles[index + 3] >= tileGrid[1]) {
      op->emitOpError(
          "referenced target topology contains unavailable endpoint outside "
          "grid");
      return mlir::failure();
    }
    int64_t endpointIndex =
        linearizeEndpoint(cardGrid, tileGrid, unavailableTiles[index],
                          unavailableTiles[index + 1],
                          unavailableTiles[index + 2],
                          unavailableTiles[index + 3]);
    available[static_cast<size_t>(endpointIndex)] = false;
  }
  return available;
}

static void appendIfAvailable(llvm::SmallVectorImpl<int64_t> &stack,
                              const std::vector<char> &available,
                              llvm::ArrayRef<int64_t> cardGrid,
                              llvm::ArrayRef<int64_t> tileGrid,
                              int64_t currentIndex, int64_t cardY,
                              int64_t cardX, int64_t tileY, int64_t tileX) {
  int64_t endpointIndex =
      linearizeEndpoint(cardGrid, tileGrid, cardY, cardX, tileY, tileX);
  if (endpointIndex != currentIndex &&
      available[static_cast<size_t>(endpointIndex)])
    stack.push_back(endpointIndex);
}

static void appendAvailableNeighbors(llvm::SmallVectorImpl<int64_t> &stack,
                                     const std::vector<char> &available,
                                     llvm::ArrayRef<int64_t> cardGrid,
                                     llvm::ArrayRef<int64_t> tileGrid,
                                     llvm::StringRef cardInterconnect,
                                     int64_t endpointIndex) {
  int64_t cardY = 0;
  int64_t cardX = 0;
  int64_t tileY = 0;
  int64_t tileX = 0;
  delinearizeEndpoint(endpointIndex, cardGrid, tileGrid, cardY, cardX, tileY,
                      tileX);

  if (tileY > 0)
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY, cardX, tileY - 1, tileX);
  if (tileY + 1 < tileGrid[0])
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY, cardX, tileY + 1, tileX);
  if (tileX > 0)
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY, cardX, tileY, tileX - 1);
  if (tileX + 1 < tileGrid[1])
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY, cardX, tileY, tileX + 1);

  if (cardY > 0)
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY - 1, cardX, tileY, tileX);
  if (cardY + 1 < cardGrid[0])
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY + 1, cardX, tileY, tileX);
  if (cardX > 0)
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY, cardX - 1, tileY, tileX);
  if (cardX + 1 < cardGrid[1])
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY, cardX + 1, tileY, tileX);

  if (cardInterconnect != "torus")
    return;

  if (cardGrid[0] > 1) {
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      (cardY + cardGrid[0] - 1) % cardGrid[0], cardX, tileY,
                      tileX);
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      (cardY + 1) % cardGrid[0], cardX, tileY, tileX);
  }
  if (cardGrid[1] > 1) {
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY, (cardX + cardGrid[1] - 1) % cardGrid[1], tileY,
                      tileX);
    appendIfAvailable(stack, available, cardGrid, tileGrid, endpointIndex,
                      cardY, (cardX + 1) % cardGrid[1], tileY, tileX);
  }
}

static mlir::FailureOr<std::vector<char>>
computeReachableEndpoints(mlir::Operation *op, TargetTopologyOp topologyOp,
                          int64_t startEndpoint) {
  llvm::ArrayRef<int64_t> cardGrid = topologyOp.getCardGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> tileGrid = topologyOp.getTileGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> unavailableTiles =
      topologyOp.getUnavailableTilesAttr().asArrayRef();

  mlir::FailureOr<std::vector<char>> available =
      buildAvailableEndpointMask(op, cardGrid, tileGrid, unavailableTiles);
  if (mlir::failed(available))
    return mlir::failure();

  std::vector<char> visited(available->size(), false);
  llvm::SmallVector<int64_t, 16> stack;
  stack.push_back(startEndpoint);

  llvm::StringRef cardInterconnect =
      topologyOp.getCardInterconnectAttr().getValue();
  while (!stack.empty()) {
    int64_t endpointIndex = stack.pop_back_val();
    size_t vectorIndex = static_cast<size_t>(endpointIndex);
    if (visited[vectorIndex] || !(*available)[vectorIndex])
      continue;

    visited[vectorIndex] = true;
    appendAvailableNeighbors(stack, *available, cardGrid, tileGrid,
                             cardInterconnect, endpointIndex);
  }

  return visited;
}

static mlir::LogicalResult verifyAllAvailableConnected(
    mlir::Operation *op, TargetTopologyOp topologyOp, int64_t availableCount) {
  llvm::ArrayRef<int64_t> cardGrid = topologyOp.getCardGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> tileGrid = topologyOp.getTileGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> unavailableTiles =
      topologyOp.getUnavailableTilesAttr().asArrayRef();

  mlir::FailureOr<std::vector<char>> available =
      buildAvailableEndpointMask(op, cardGrid, tileGrid, unavailableTiles);
  if (mlir::failed(available))
    return mlir::failure();

  int64_t startEndpoint = -1;
  for (size_t index = 0; index < available->size(); ++index) {
    if ((*available)[index]) {
      startEndpoint = static_cast<int64_t>(index);
      break;
    }
  }
  if (startEndpoint < 0)
    return op->emitOpError(
        "all_available execution mesh requires at least one available endpoint");

  mlir::FailureOr<std::vector<char>> visited =
      computeReachableEndpoints(op, topologyOp, startEndpoint);
  if (mlir::failed(visited))
    return mlir::failure();

  int64_t visitedAvailable = 0;
  for (size_t index = 0; index < available->size(); ++index) {
    if ((*available)[index] && (*visited)[index])
      ++visitedAvailable;
  }
  if (visitedAvailable != availableCount)
    return op->emitOpError(
        "all_available execution mesh available endpoints must be connected");

  return mlir::success();
}

static mlir::LogicalResult verifyExplicitEndpointsConnected(
    mlir::Operation *op, TargetTopologyOp topologyOp,
    llvm::ArrayRef<int64_t> endpoints) {
  if (endpoints.empty())
    return mlir::success();

  llvm::ArrayRef<int64_t> cardGrid = topologyOp.getCardGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> tileGrid = topologyOp.getTileGridAttr().asArrayRef();
  int64_t startEndpoint =
      linearizeEndpoint(cardGrid, tileGrid, endpoints[0], endpoints[1],
                        endpoints[2], endpoints[3]);

  mlir::FailureOr<std::vector<char>> visited =
      computeReachableEndpoints(op, topologyOp, startEndpoint);
  if (mlir::failed(visited))
    return mlir::failure();

  for (size_t index = 0; index < endpoints.size(); index += 4) {
    int64_t endpointIndex =
        linearizeEndpoint(cardGrid, tileGrid, endpoints[index],
                          endpoints[index + 1], endpoints[index + 2],
                          endpoints[index + 3]);
    if (!(*visited)[static_cast<size_t>(endpointIndex)])
      return op->emitOpError(
          "explicit endpoints must belong to one connected available component");
  }

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

mlir::LogicalResult ExecutionMeshOp::verify() {
  mlir::ModuleOp moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return emitOpError("must be nested under a module");

  TargetTopologyOp topologyOp =
      moduleOp.lookupSymbol<TargetTopologyOp>(getTopologyAttr().getValue());
  if (!topologyOp)
    return emitOpError("references unknown target topology symbol @")
           << getTopologyAttr().getValue();

  llvm::ArrayRef<int64_t> cardGrid = topologyOp.getCardGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> tileGrid = topologyOp.getTileGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> unavailableTiles =
      topologyOp.getUnavailableTilesAttr().asArrayRef();
  if (cardGrid.size() != 2 || tileGrid.size() != 2 ||
      unavailableTiles.size() % 4 != 0 ||
      !isSupportedCardInterconnect(
          topologyOp.getCardInterconnectAttr().getValue()))
    return emitOpError("references invalid target topology symbol @")
           << topologyOp.getSymName();

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

  mlir::FailureOr<int64_t> rankCount =
      computeRankCount(getOperation(), shape);
  if (mlir::failed(rankCount))
    return mlir::failure();

  llvm::StringRef policy = getPolicyAttr().getValue();
  llvm::ArrayRef<int64_t> endpoints = getEndpointsAttr().asArrayRef();

  if (policy == "all_available") {
    if (!endpoints.empty())
      return emitOpError(
          "all_available execution mesh must not carry explicit endpoints");

    mlir::FailureOr<int64_t> endpointCount =
        computeEndpointCount(getOperation(), cardGrid, tileGrid);
    if (mlir::failed(endpointCount))
      return mlir::failure();
    int64_t availableCount =
        *endpointCount - static_cast<int64_t>(unavailableTiles.size() / 4);
    if (*rankCount != availableCount)
      return emitOpError(
          "all_available rank count must equal available endpoint count");
    return verifyAllAvailableConnected(getOperation(), topologyOp,
                                       availableCount);
  }

  if (policy != "explicit")
    return emitOpError("policy must be all_available or explicit");

  if (endpoints.size() % 4 != 0)
    return emitOpError(
        "endpoints must contain card_y/card_x/tile_y/tile_x tuples");
  int64_t endpointCount = static_cast<int64_t>(endpoints.size() / 4);
  if (endpointCount != *rankCount)
    return emitOpError(
        "explicit endpoint count must equal execution mesh rank count");

  for (size_t index = 0; index < endpoints.size(); index += 4) {
    int64_t cardY = endpoints[index];
    int64_t cardX = endpoints[index + 1];
    int64_t tileY = endpoints[index + 2];
    int64_t tileX = endpoints[index + 3];
    if (cardY < 0 || cardY >= cardGrid[0])
      return emitOpError("explicit endpoint card_y coordinate ")
             << cardY << " is outside card_grid";
    if (cardX < 0 || cardX >= cardGrid[1])
      return emitOpError("explicit endpoint card_x coordinate ")
             << cardX << " is outside card_grid";
    if (tileY < 0 || tileY >= tileGrid[0])
      return emitOpError("explicit endpoint tile_y coordinate ")
             << tileY << " is outside tile_grid";
    if (tileX < 0 || tileX >= tileGrid[1])
      return emitOpError("explicit endpoint tile_x coordinate ")
             << tileX << " is outside tile_grid";
    if (isUnavailableEndpoint(unavailableTiles, cardY, cardX, tileY, tileX))
      return emitOpError(
          "explicit endpoint references unavailable tile coordinate");

    for (size_t priorIndex = 0; priorIndex < index; priorIndex += 4) {
      if (endpoints[priorIndex] == cardY && endpoints[priorIndex + 1] == cardX &&
          endpoints[priorIndex + 2] == tileY &&
          endpoints[priorIndex + 3] == tileX)
        return emitOpError("explicit endpoints contain duplicate coordinate");
    }
  }

  if (mlir::failed(
          verifyExplicitEndpointsConnected(getOperation(), topologyOp,
                                           endpoints)))
    return mlir::failure();

  return mlir::success();
}
