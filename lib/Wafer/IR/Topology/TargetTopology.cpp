//===- TargetTopology.cpp - Verifier-grade physical topology view ------===//

#include "Wafer/IR/Topology/TargetTopology.h"

#include "Wafer/IR/WaferDialect.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace wafer {
namespace {

static mlir::FailureOr<TargetTopology> failTopology(std::string *failureReason,
                                                    llvm::StringRef message) {
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

mlir::FailureOr<TargetTopology>
TargetTopology::create(mlir::ModuleOp module, std::string *failureReason) {
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

  TargetTopology result;
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
    CardCoordinate cardCoordinate{unavailable[index], unavailable[index + 1]};
    TileCoordinate tileCoordinate{unavailable[index + 2],
                                  unavailable[index + 3]};
    std::optional<CardId> cardId = result.getCardId(cardCoordinate);
    std::optional<TileId> tileId = result.getTileId(tileCoordinate);
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
      if (result.isTileAvailable(CardId(card), TileId(tile)))
        available.push_back(TileId(tile));
    }
  }
  return result;
}

std::optional<size_t> TargetTopology::getCardIndex(CardId cardId) const {
  int64_t value = cardId.getValue();
  if (value < 0 || value >= cardCount)
    return std::nullopt;
  return static_cast<size_t>(value);
}

std::optional<size_t> TargetTopology::getTileIndex(TileId tileId) const {
  int64_t value = tileId.getValue();
  if (value < 0 || value >= tilesPerCard)
    return std::nullopt;
  return static_cast<size_t>(value);
}

size_t TargetTopology::getEndpointIndex(size_t cardIndex,
                                        size_t tileIndex) const {
  return cardIndex * static_cast<size_t>(tilesPerCard) + tileIndex;
}

std::optional<CardCoordinate>
TargetTopology::getCardCoordinate(CardId cardId) const {
  std::optional<size_t> index = getCardIndex(cardId);
  if (!index)
    return std::nullopt;
  return CardCoordinate{
      static_cast<int64_t>(*index / static_cast<size_t>(cardGrid[1])),
      static_cast<int64_t>(*index % static_cast<size_t>(cardGrid[1]))};
}

std::optional<CardId>
TargetTopology::getCardId(CardCoordinate coordinate) const {
  if (coordinate.y < 0 || coordinate.y >= cardGrid[0] || coordinate.x < 0 ||
      coordinate.x >= cardGrid[1])
    return std::nullopt;
  return CardId(coordinate.y * cardGrid[1] + coordinate.x);
}

std::optional<TileCoordinate>
TargetTopology::getTileCoordinate(TileId tileId) const {
  std::optional<size_t> index = getTileIndex(tileId);
  if (!index)
    return std::nullopt;
  return TileCoordinate{
      static_cast<int64_t>(*index / static_cast<size_t>(tileGrid[1])),
      static_cast<int64_t>(*index % static_cast<size_t>(tileGrid[1]))};
}

std::optional<TileId>
TargetTopology::getTileId(TileCoordinate coordinate) const {
  if (coordinate.y < 0 || coordinate.y >= tileGrid[0] || coordinate.x < 0 ||
      coordinate.x >= tileGrid[1])
    return std::nullopt;
  return TileId(coordinate.y * tileGrid[1] + coordinate.x);
}

std::optional<llvm::ArrayRef<TileId>>
TargetTopology::getAvailableTileIds(CardId cardId) const {
  std::optional<size_t> card = getCardIndex(cardId);
  if (!card)
    return std::nullopt;
  return llvm::ArrayRef<TileId>(availableTilesByCard[*card]);
}

bool TargetTopology::isTileAvailable(CardId cardId, TileId tileId) const {
  std::optional<size_t> card = getCardIndex(cardId);
  std::optional<size_t> tile = getTileIndex(tileId);
  if (!card || !tile)
    return false;
  return endpointAvailability[getEndpointIndex(*card, *tile)] != 0;
}

mlir::FailureOr<llvm::SmallVector<TileId, 4>>
TargetTopology::getOnCardNeighbors(CardId cardId, TileId tileId) const {
  if (!isTileAvailable(cardId, tileId))
    return mlir::failure();
  std::optional<TileCoordinate> coordinate = getTileCoordinate(tileId);
  if (!coordinate)
    return mlir::failure();

  llvm::SmallVector<TileId, 4> neighbors;
  auto append = [&](int64_t y, int64_t x) {
    std::optional<TileId> neighbor = getTileId(TileCoordinate{y, x});
    if (neighbor && isTileAvailable(cardId, *neighbor))
      neighbors.push_back(*neighbor);
  };
  append(coordinate->y - 1, coordinate->x);
  append(coordinate->y + 1, coordinate->x);
  append(coordinate->y, coordinate->x - 1);
  append(coordinate->y, coordinate->x + 1);
  llvm::sort(neighbors, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  return neighbors;
}

bool TargetTopology::areOnCardAdjacent(CardId cardId, TileId source,
                                       TileId destination) const {
  mlir::FailureOr<llvm::SmallVector<TileId, 4>> neighbors =
      getOnCardNeighbors(cardId, source);
  return mlir::succeeded(neighbors) &&
         llvm::is_contained(*neighbors, destination);
}

std::optional<uint64_t>
TargetTopology::getOnCardShortestHopDistance(CardId cardId, TileId source,
                                             TileId destination) const {
  std::optional<size_t> sourceIndex = getTileIndex(source);
  std::optional<size_t> destinationIndex = getTileIndex(destination);
  if (!sourceIndex || !destinationIndex || !isTileAvailable(cardId, source) ||
      !isTileAvailable(cardId, destination))
    return std::nullopt;

  const uint64_t unvisited = std::numeric_limits<uint64_t>::max();
  std::vector<uint64_t> distances(static_cast<size_t>(tilesPerCard), unvisited);
  llvm::SmallVector<TileId, 16> queue;
  distances[*sourceIndex] = 0;
  queue.push_back(source);
  for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
    TileId current = queue[cursor];
    size_t currentIndex = *getTileIndex(current);
    if (current == destination)
      return distances[currentIndex];
    auto neighbors = getOnCardNeighbors(cardId, current);
    if (mlir::failed(neighbors))
      return std::nullopt;
    for (TileId neighbor : *neighbors) {
      size_t neighborIndex = *getTileIndex(neighbor);
      if (distances[neighborIndex] != unvisited)
        continue;
      distances[neighborIndex] = distances[currentIndex] + 1;
      queue.push_back(neighbor);
    }
  }
  return std::nullopt;
}

mlir::FailureOr<llvm::SmallVector<TileLink, 8>>
TargetTopology::getCanonicalOnCardPath(CardId cardId, TileId source,
                                       TileId destination) const {
  std::optional<size_t> sourceIndex = getTileIndex(source);
  std::optional<size_t> destinationIndex = getTileIndex(destination);
  if (!sourceIndex || !destinationIndex || !isTileAvailable(cardId, source) ||
      !isTileAvailable(cardId, destination))
    return mlir::failure();

  llvm::SmallVector<TileLink, 8> path;
  if (source == destination)
    return path;

  const int64_t unvisited = -1;
  std::vector<int64_t> predecessor(static_cast<size_t>(tilesPerCard),
                                   unvisited);
  llvm::SmallVector<TileId, 16> queue;
  predecessor[*sourceIndex] = source.getValue();
  queue.push_back(source);
  for (size_t cursor = 0;
       cursor < queue.size() && predecessor[*destinationIndex] == unvisited;
       ++cursor) {
    TileId current = queue[cursor];
    auto neighbors = getOnCardNeighbors(cardId, current);
    if (mlir::failed(neighbors))
      return mlir::failure();
    for (TileId neighbor : *neighbors) {
      size_t neighborIndex = *getTileIndex(neighbor);
      if (predecessor[neighborIndex] != unvisited)
        continue;
      predecessor[neighborIndex] = current.getValue();
      queue.push_back(neighbor);
    }
  }
  if (predecessor[*destinationIndex] == unvisited)
    return mlir::failure();

  llvm::SmallVector<TileId, 8> reversePath;
  for (TileId current = destination; current != source;
       current = TileId(predecessor[*getTileIndex(current)]))
    reversePath.push_back(current);
  reversePath.push_back(source);
  std::reverse(reversePath.begin(), reversePath.end());
  path.reserve(reversePath.size() - 1);
  for (size_t index = 1; index < reversePath.size(); ++index)
    path.push_back(TileLink{reversePath[index - 1], reversePath[index]});
  return path;
}

} // namespace wafer
