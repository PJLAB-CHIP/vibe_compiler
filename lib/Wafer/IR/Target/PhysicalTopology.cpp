//===- PhysicalTopology.cpp - Verifier-grade physical topology view ------===//

#include "Wafer/IR/Target/PhysicalTopology.h"

#include "Wafer/IR/WaferDialect.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace wafer {
namespace {

static mlir::FailureOr<PhysicalTopology>
failTopology(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

static bool checkedMultiplyPositive(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool hasNestedTopology(mlir::ModuleOp module) {
  bool nested = false;
  module.walk([&](TargetTopologyOp topology) {
    if (topology->getParentOp() == module.getOperation())
      return mlir::WalkResult::advance();
    nested = true;
    return mlir::WalkResult::interrupt();
  });
  return nested;
}

} // namespace

mlir::FailureOr<PhysicalTopology>
PhysicalTopology::create(mlir::ModuleOp module, std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (!module)
    return failTopology(failureReason, "source module is null");
  if (hasNestedTopology(module))
    return failTopology(
        failureReason,
        "wafer.target.topology must be directly nested under the source "
        "module");

  llvm::SmallVector<TargetTopologyOp, 2> topologies(
      module.getOps<TargetTopologyOp>());
  if (topologies.size() != 1)
    return failTopology(
        failureReason,
        "expected exactly one direct wafer.target.topology in source module");
  TargetTopologyOp topology = topologies.front();

  llvm::ArrayRef<int64_t> cardGridAttr =
      topology.getCardGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> tileGridAttr =
      topology.getTileGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> unavailable =
      topology.getUnavailableTilesAttr().asArrayRef();
  llvm::StringRef interconnect = topology.getCardInterconnectAttr().getValue();
  if (cardGridAttr.size() != 2 || tileGridAttr.size() != 2 ||
      cardGridAttr[0] <= 0 || cardGridAttr[1] <= 0 || tileGridAttr[0] <= 0 ||
      tileGridAttr[1] <= 0 || unavailable.size() % 4 != 0 ||
      (interconnect != "mesh" && interconnect != "torus"))
    return failTopology(failureReason, "wafer.target.topology is malformed");

  PhysicalTopology result;
  result.cardGrid = {cardGridAttr[0], cardGridAttr[1]};
  result.tileGrid = {tileGridAttr[0], tileGridAttr[1]};
  int64_t endpointCount = 0;
  if (!checkedMultiplyPositive(cardGridAttr[0], cardGridAttr[1],
                               result.cardCount) ||
      !checkedMultiplyPositive(tileGridAttr[0], tileGridAttr[1],
                               result.tilesPerCard) ||
      !checkedMultiplyPositive(result.cardCount, result.tilesPerCard,
                               endpointCount) ||
      static_cast<uint64_t>(endpointCount) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return failTopology(failureReason,
                        "wafer.target.topology is too large to represent");

  result.endpointAvailability.assign(static_cast<size_t>(endpointCount), 1);
  for (size_t index = 0; index < unavailable.size(); index += 4) {
    PhysicalCardCoordinate cardCoordinate{unavailable[index],
                                          unavailable[index + 1]};
    PhysicalTileCoordinate tileCoordinate{unavailable[index + 2],
                                          unavailable[index + 3]};
    std::optional<PhysicalCardId> cardId = result.getCardId(cardCoordinate);
    std::optional<PhysicalTileId> tileId = result.getTileId(tileCoordinate);
    if (!cardId || !tileId)
      return failTopology(
          failureReason,
          "wafer.target.topology has an unavailable Tile outside its grids");
    size_t endpoint = result.getEndpointIndex(*result.getCardIndex(*cardId),
                                              *result.getTileIndex(*tileId));
    if (!result.endpointAvailability[endpoint])
      return failTopology(
          failureReason,
          "wafer.target.topology has a duplicate unavailable Tile");
    result.endpointAvailability[endpoint] = 0;
  }

  result.availableTilesByCard.resize(static_cast<size_t>(result.cardCount));
  for (int64_t card = 0; card < result.cardCount; ++card) {
    auto &available = result.availableTilesByCard[static_cast<size_t>(card)];
    for (int64_t tile = 0; tile < result.tilesPerCard; ++tile) {
      if (result.isTileAvailable(PhysicalCardId(card), PhysicalTileId(tile)))
        available.push_back(PhysicalTileId(tile));
    }
  }
  return result;
}

std::optional<size_t>
PhysicalTopology::getCardIndex(PhysicalCardId cardId) const {
  int64_t value = cardId.getValue();
  if (value < 0 || value >= cardCount)
    return std::nullopt;
  return static_cast<size_t>(value);
}

std::optional<size_t>
PhysicalTopology::getTileIndex(PhysicalTileId tileId) const {
  int64_t value = tileId.getValue();
  if (value < 0 || value >= tilesPerCard)
    return std::nullopt;
  return static_cast<size_t>(value);
}

size_t PhysicalTopology::getEndpointIndex(size_t cardIndex,
                                          size_t tileIndex) const {
  return cardIndex * static_cast<size_t>(tilesPerCard) + tileIndex;
}

std::optional<PhysicalCardCoordinate>
PhysicalTopology::getCardCoordinate(PhysicalCardId cardId) const {
  std::optional<size_t> index = getCardIndex(cardId);
  if (!index)
    return std::nullopt;
  return PhysicalCardCoordinate{
      static_cast<int64_t>(*index / static_cast<size_t>(cardGrid[1])),
      static_cast<int64_t>(*index % static_cast<size_t>(cardGrid[1]))};
}

std::optional<PhysicalCardId>
PhysicalTopology::getCardId(PhysicalCardCoordinate coordinate) const {
  if (coordinate.y < 0 || coordinate.y >= cardGrid[0] || coordinate.x < 0 ||
      coordinate.x >= cardGrid[1])
    return std::nullopt;
  return PhysicalCardId(coordinate.y * cardGrid[1] + coordinate.x);
}

std::optional<PhysicalTileCoordinate>
PhysicalTopology::getTileCoordinate(PhysicalTileId tileId) const {
  std::optional<size_t> index = getTileIndex(tileId);
  if (!index)
    return std::nullopt;
  return PhysicalTileCoordinate{
      static_cast<int64_t>(*index / static_cast<size_t>(tileGrid[1])),
      static_cast<int64_t>(*index % static_cast<size_t>(tileGrid[1]))};
}

std::optional<PhysicalTileId>
PhysicalTopology::getTileId(PhysicalTileCoordinate coordinate) const {
  if (coordinate.y < 0 || coordinate.y >= tileGrid[0] || coordinate.x < 0 ||
      coordinate.x >= tileGrid[1])
    return std::nullopt;
  return PhysicalTileId(coordinate.y * tileGrid[1] + coordinate.x);
}

std::optional<llvm::ArrayRef<PhysicalTileId>>
PhysicalTopology::getAvailableTileIds(PhysicalCardId cardId) const {
  std::optional<size_t> card = getCardIndex(cardId);
  if (!card)
    return std::nullopt;
  return llvm::ArrayRef<PhysicalTileId>(availableTilesByCard[*card]);
}

bool PhysicalTopology::isTileAvailable(PhysicalCardId cardId,
                                       PhysicalTileId tileId) const {
  std::optional<size_t> card = getCardIndex(cardId);
  std::optional<size_t> tile = getTileIndex(tileId);
  if (!card || !tile)
    return false;
  return endpointAvailability[getEndpointIndex(*card, *tile)] != 0;
}

mlir::FailureOr<llvm::SmallVector<PhysicalTileId, 4>>
PhysicalTopology::getOnCardNeighbors(PhysicalCardId cardId,
                                     PhysicalTileId tileId) const {
  if (!isTileAvailable(cardId, tileId))
    return mlir::failure();
  std::optional<PhysicalTileCoordinate> coordinate = getTileCoordinate(tileId);
  if (!coordinate)
    return mlir::failure();

  llvm::SmallVector<PhysicalTileId, 4> neighbors;
  auto append = [&](int64_t y, int64_t x) {
    std::optional<PhysicalTileId> neighbor =
        getTileId(PhysicalTileCoordinate{y, x});
    if (neighbor && isTileAvailable(cardId, *neighbor))
      neighbors.push_back(*neighbor);
  };
  append(coordinate->y - 1, coordinate->x);
  append(coordinate->y + 1, coordinate->x);
  append(coordinate->y, coordinate->x - 1);
  append(coordinate->y, coordinate->x + 1);
  llvm::sort(neighbors, [](PhysicalTileId lhs, PhysicalTileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  return neighbors;
}

bool PhysicalTopology::areOnCardAdjacent(PhysicalCardId cardId,
                                         PhysicalTileId source,
                                         PhysicalTileId destination) const {
  mlir::FailureOr<llvm::SmallVector<PhysicalTileId, 4>> neighbors =
      getOnCardNeighbors(cardId, source);
  return mlir::succeeded(neighbors) &&
         llvm::is_contained(*neighbors, destination);
}

std::optional<uint64_t> PhysicalTopology::getOnCardShortestHopDistance(
    PhysicalCardId cardId, PhysicalTileId source,
    PhysicalTileId destination) const {
  std::optional<size_t> sourceIndex = getTileIndex(source);
  std::optional<size_t> destinationIndex = getTileIndex(destination);
  if (!sourceIndex || !destinationIndex || !isTileAvailable(cardId, source) ||
      !isTileAvailable(cardId, destination))
    return std::nullopt;

  const uint64_t unvisited = std::numeric_limits<uint64_t>::max();
  std::vector<uint64_t> distances(static_cast<size_t>(tilesPerCard), unvisited);
  llvm::SmallVector<PhysicalTileId, 16> queue;
  distances[*sourceIndex] = 0;
  queue.push_back(source);
  for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
    PhysicalTileId current = queue[cursor];
    size_t currentIndex = *getTileIndex(current);
    if (current == destination)
      return distances[currentIndex];
    auto neighbors = getOnCardNeighbors(cardId, current);
    if (mlir::failed(neighbors))
      return std::nullopt;
    for (PhysicalTileId neighbor : *neighbors) {
      size_t neighborIndex = *getTileIndex(neighbor);
      if (distances[neighborIndex] != unvisited)
        continue;
      distances[neighborIndex] = distances[currentIndex] + 1;
      queue.push_back(neighbor);
    }
  }
  return std::nullopt;
}

mlir::FailureOr<llvm::SmallVector<PhysicalTileDirectedLink, 8>>
PhysicalTopology::getCanonicalOnCardPath(PhysicalCardId cardId,
                                         PhysicalTileId source,
                                         PhysicalTileId destination) const {
  std::optional<size_t> sourceIndex = getTileIndex(source);
  std::optional<size_t> destinationIndex = getTileIndex(destination);
  if (!sourceIndex || !destinationIndex || !isTileAvailable(cardId, source) ||
      !isTileAvailable(cardId, destination))
    return mlir::failure();

  llvm::SmallVector<PhysicalTileDirectedLink, 8> path;
  if (source == destination)
    return path;

  const int64_t unvisited = -1;
  std::vector<int64_t> predecessor(static_cast<size_t>(tilesPerCard),
                                   unvisited);
  llvm::SmallVector<PhysicalTileId, 16> queue;
  predecessor[*sourceIndex] = source.getValue();
  queue.push_back(source);
  for (size_t cursor = 0;
       cursor < queue.size() && predecessor[*destinationIndex] == unvisited;
       ++cursor) {
    PhysicalTileId current = queue[cursor];
    auto neighbors = getOnCardNeighbors(cardId, current);
    if (mlir::failed(neighbors))
      return mlir::failure();
    for (PhysicalTileId neighbor : *neighbors) {
      size_t neighborIndex = *getTileIndex(neighbor);
      if (predecessor[neighborIndex] != unvisited)
        continue;
      predecessor[neighborIndex] = current.getValue();
      queue.push_back(neighbor);
    }
  }
  if (predecessor[*destinationIndex] == unvisited)
    return mlir::failure();

  llvm::SmallVector<PhysicalTileId, 8> reversePath;
  for (PhysicalTileId current = destination; current != source;
       current = PhysicalTileId(predecessor[*getTileIndex(current)]))
    reversePath.push_back(current);
  reversePath.push_back(source);
  std::reverse(reversePath.begin(), reversePath.end());
  path.reserve(reversePath.size() - 1);
  for (size_t index = 1; index < reversePath.size(); ++index)
    path.push_back(
        PhysicalTileDirectedLink{reversePath[index - 1], reversePath[index]});
  return path;
}

} // namespace wafer
