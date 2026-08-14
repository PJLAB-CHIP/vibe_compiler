//===- ExecutionTopologyAnalysis.cpp - Execution topology facts --------===//

#include "ExecutionTopologyAnalysis.h"

#include "Wafer/IR/WaferDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace wafer::analysis {
namespace {

static bool checkedMultiplySize(size_t lhs, size_t rhs, size_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool checkedMultiplyPositive(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs)
    return false;
  result = lhs * rhs;
  return true;
}

static std::optional<size_t>
linearizeEndpoint(llvm::ArrayRef<int64_t> cardGrid,
                  llvm::ArrayRef<int64_t> tileGrid,
                  const ExecutionEndpoint &endpoint) {
  if (cardGrid.size() != 2 || tileGrid.size() != 2 || endpoint.cardY < 0 ||
      endpoint.cardY >= cardGrid[0] || endpoint.cardX < 0 ||
      endpoint.cardX >= cardGrid[1] || endpoint.tileY < 0 ||
      endpoint.tileY >= tileGrid[0] || endpoint.tileX < 0 ||
      endpoint.tileX >= tileGrid[1])
    return std::nullopt;

  int64_t cardIndex = endpoint.cardY * cardGrid[1] + endpoint.cardX;
  int64_t tileIndex = endpoint.tileY * tileGrid[1] + endpoint.tileX;
  int64_t tileCount = tileGrid[0] * tileGrid[1];
  int64_t linear = cardIndex * tileCount + tileIndex;
  if (linear < 0 ||
      static_cast<uint64_t>(linear) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return std::nullopt;
  return static_cast<size_t>(linear);
}

static ExecutionEndpoint delinearizeEndpoint(size_t index,
                                             llvm::ArrayRef<int64_t> cardGrid,
                                             llvm::ArrayRef<int64_t> tileGrid) {
  const size_t cardColumns = static_cast<size_t>(cardGrid[1]);
  const size_t tileRows = static_cast<size_t>(tileGrid[0]);
  const size_t tileColumns = static_cast<size_t>(tileGrid[1]);
  const size_t tilesPerCard = tileRows * tileColumns;
  const size_t cardIndex = index / tilesPerCard;
  const size_t tileIndex = index % tilesPerCard;
  return {static_cast<int64_t>(cardIndex / cardColumns),
          static_cast<int64_t>(cardIndex % cardColumns),
          static_cast<int64_t>(tileIndex / tileColumns),
          static_cast<int64_t>(tileIndex % tileColumns)};
}

static void appendIfAvailable(llvm::SmallVectorImpl<size_t> &neighbors,
                              llvm::ArrayRef<unsigned char> available,
                              llvm::ArrayRef<int64_t> cardGrid,
                              llvm::ArrayRef<int64_t> tileGrid,
                              const ExecutionEndpoint &endpoint) {
  std::optional<size_t> index = linearizeEndpoint(cardGrid, tileGrid, endpoint);
  if (index && available[*index])
    neighbors.push_back(*index);
}

static void appendAvailableNeighbors(llvm::SmallVectorImpl<size_t> &neighbors,
                                     llvm::ArrayRef<unsigned char> available,
                                     llvm::ArrayRef<int64_t> cardGrid,
                                     llvm::ArrayRef<int64_t> tileGrid,
                                     llvm::StringRef cardInterconnect,
                                     size_t endpointIndex) {
  ExecutionEndpoint endpoint =
      delinearizeEndpoint(endpointIndex, cardGrid, tileGrid);
  if (endpoint.tileY > 0)
    appendIfAvailable(
        neighbors, available, cardGrid, tileGrid,
        {endpoint.cardY, endpoint.cardX, endpoint.tileY - 1, endpoint.tileX});
  if (endpoint.tileY + 1 < tileGrid[0])
    appendIfAvailable(
        neighbors, available, cardGrid, tileGrid,
        {endpoint.cardY, endpoint.cardX, endpoint.tileY + 1, endpoint.tileX});
  if (endpoint.tileX > 0)
    appendIfAvailable(
        neighbors, available, cardGrid, tileGrid,
        {endpoint.cardY, endpoint.cardX, endpoint.tileY, endpoint.tileX - 1});
  if (endpoint.tileX + 1 < tileGrid[1])
    appendIfAvailable(
        neighbors, available, cardGrid, tileGrid,
        {endpoint.cardY, endpoint.cardX, endpoint.tileY, endpoint.tileX + 1});

  if (endpoint.cardY > 0)
    appendIfAvailable(
        neighbors, available, cardGrid, tileGrid,
        {endpoint.cardY - 1, endpoint.cardX, endpoint.tileY, endpoint.tileX});
  if (endpoint.cardY + 1 < cardGrid[0])
    appendIfAvailable(
        neighbors, available, cardGrid, tileGrid,
        {endpoint.cardY + 1, endpoint.cardX, endpoint.tileY, endpoint.tileX});
  if (endpoint.cardX > 0)
    appendIfAvailable(
        neighbors, available, cardGrid, tileGrid,
        {endpoint.cardY, endpoint.cardX - 1, endpoint.tileY, endpoint.tileX});
  if (endpoint.cardX + 1 < cardGrid[1])
    appendIfAvailable(
        neighbors, available, cardGrid, tileGrid,
        {endpoint.cardY, endpoint.cardX + 1, endpoint.tileY, endpoint.tileX});

  if (cardInterconnect != "torus")
    return;
  if (cardGrid[0] > 1) {
    appendIfAvailable(neighbors, available, cardGrid, tileGrid,
                      {(endpoint.cardY + cardGrid[0] - 1) % cardGrid[0],
                       endpoint.cardX, endpoint.tileY, endpoint.tileX});
    appendIfAvailable(neighbors, available, cardGrid, tileGrid,
                      {(endpoint.cardY + 1) % cardGrid[0], endpoint.cardX,
                       endpoint.tileY, endpoint.tileX});
  }
  if (cardGrid[1] > 1) {
    appendIfAvailable(neighbors, available, cardGrid, tileGrid,
                      {endpoint.cardY,
                       (endpoint.cardX + cardGrid[1] - 1) % cardGrid[1],
                       endpoint.tileY, endpoint.tileX});
    appendIfAvailable(neighbors, available, cardGrid, tileGrid,
                      {endpoint.cardY, (endpoint.cardX + 1) % cardGrid[1],
                       endpoint.tileY, endpoint.tileX});
  }
}

static bool hasNestedTopologyFacts(mlir::ModuleOp module) {
  bool nested = false;
  module.walk([&](mlir::Operation *operation) {
    if (!mlir::isa<TargetTopologyOp, ExecutionMeshOp>(operation) ||
        operation->getParentOp() == module.getOperation())
      return mlir::WalkResult::advance();
    nested = true;
    return mlir::WalkResult::interrupt();
  });
  return nested;
}

} // namespace

mlir::FailureOr<ExecutionTopologyAnalysis>
ExecutionTopologyAnalysis::create(mlir::ModuleOp module) {
  if (!module || hasNestedTopologyFacts(module))
    return mlir::failure();

  llvm::SmallVector<TargetTopologyOp, 2> topologies;
  llvm::SmallVector<ExecutionMeshOp, 2> meshes;
  for (TargetTopologyOp topology : module.getOps<TargetTopologyOp>())
    topologies.push_back(topology);
  for (ExecutionMeshOp mesh : module.getOps<ExecutionMeshOp>())
    meshes.push_back(mesh);
  if (topologies.size() != 1 || meshes.size() != 1)
    return mlir::failure();

  TargetTopologyOp topology = topologies.front();
  ExecutionMeshOp mesh = meshes.front();
  if (mesh.getTopologyAttr().getValue() != topology.getSymName())
    return mlir::failure();

  llvm::ArrayRef<int64_t> cardGridAttr =
      topology.getCardGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> tileGridAttr =
      topology.getTileGridAttr().asArrayRef();
  llvm::StringRef interconnect = topology.getCardInterconnectAttr().getValue();
  llvm::ArrayRef<int64_t> unavailable =
      topology.getUnavailableTilesAttr().asArrayRef();
  if (cardGridAttr.size() != 2 || tileGridAttr.size() != 2 ||
      cardGridAttr[0] <= 0 || cardGridAttr[1] <= 0 || tileGridAttr[0] <= 0 ||
      tileGridAttr[1] <= 0 ||
      (interconnect != "mesh" && interconnect != "torus") ||
      unavailable.size() % 4 != 0)
    return mlir::failure();

  int64_t cardCount = 0;
  int64_t tileCount = 0;
  int64_t endpointCount64 = 0;
  if (!checkedMultiplyPositive(cardGridAttr[0], cardGridAttr[1], cardCount) ||
      !checkedMultiplyPositive(tileGridAttr[0], tileGridAttr[1], tileCount) ||
      !checkedMultiplyPositive(cardCount, tileCount, endpointCount64) ||
      static_cast<uint64_t>(endpointCount64) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return mlir::failure();
  const size_t endpointCount = static_cast<size_t>(endpointCount64);

  ExecutionTopologyAnalysis result;
  result.cardGrid = {cardGridAttr[0], cardGridAttr[1]};
  result.tileGrid = {tileGridAttr[0], tileGridAttr[1]};
  result.cardInterconnect.assign(interconnect.begin(), interconnect.end());

  std::vector<unsigned char> available(endpointCount, 1);
  for (size_t tuple = 0; tuple < unavailable.size(); tuple += 4) {
    ExecutionEndpoint endpoint{unavailable[tuple], unavailable[tuple + 1],
                               unavailable[tuple + 2], unavailable[tuple + 3]};
    std::optional<size_t> index =
        linearizeEndpoint(result.cardGrid, result.tileGrid, endpoint);
    if (!index || !available[*index])
      return mlir::failure();
    available[*index] = 0;
  }
  result.endpointAvailability.assign(available.begin(), available.end());
  llvm::SmallVector<size_t, 8> linkNeighbors;
  for (size_t endpoint = 0; endpoint < endpointCount; ++endpoint) {
    if (!available[endpoint])
      continue;
    linkNeighbors.clear();
    appendAvailableNeighbors(linkNeighbors, available, result.cardGrid,
                             result.tileGrid, result.getCardInterconnect(),
                             endpoint);
    llvm::sort(linkNeighbors);
    linkNeighbors.erase(std::unique(linkNeighbors.begin(), linkNeighbors.end()),
                        linkNeighbors.end());
    if (linkNeighbors.size() >
        std::numeric_limits<uint64_t>::max() - result.directedLinkCount)
      return mlir::failure();
    result.directedLinkCount += static_cast<uint64_t>(linkNeighbors.size());
  }

  int64_t rankCount64 = 1;
  llvm::ArrayRef<int64_t> shape = mesh.getShapeAttr().asArrayRef();
  if (shape.empty())
    return mlir::failure();
  for (int64_t dimension : shape)
    if (!checkedMultiplyPositive(rankCount64, dimension, rankCount64))
      return mlir::failure();
  if (static_cast<uint64_t>(rankCount64) >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return mlir::failure();
  const size_t rankCount = static_cast<size_t>(rankCount64);

  llvm::StringRef policy = mesh.getPolicyAttr().getValue();
  llvm::ArrayRef<int64_t> endpoints = mesh.getEndpointsAttr().asArrayRef();
  if (policy == "all_available") {
    if (!endpoints.empty())
      return mlir::failure();
    for (size_t index = 0; index < available.size(); ++index)
      if (available[index])
        result.rankEndpoints.push_back(
            delinearizeEndpoint(index, result.cardGrid, result.tileGrid));
    if (result.rankEndpoints.size() != rankCount)
      return mlir::failure();
  } else if (policy == "explicit") {
    size_t expectedEndpointValues = 0;
    if (!checkedMultiplySize(rankCount, 4, expectedEndpointValues) ||
        endpoints.size() != expectedEndpointValues)
      return mlir::failure();
    llvm::SmallVector<size_t, 16> selected;
    selected.reserve(rankCount);
    for (size_t tuple = 0; tuple < endpoints.size(); tuple += 4) {
      ExecutionEndpoint endpoint{endpoints[tuple], endpoints[tuple + 1],
                                 endpoints[tuple + 2], endpoints[tuple + 3]};
      std::optional<size_t> index =
          linearizeEndpoint(result.cardGrid, result.tileGrid, endpoint);
      if (!index || !available[*index] || llvm::is_contained(selected, *index))
        return mlir::failure();
      selected.push_back(*index);
      result.rankEndpoints.push_back(endpoint);
    }
  } else {
    return mlir::failure();
  }

  size_t matrixSize = 0;
  if (!checkedMultiplySize(rankCount, rankCount, matrixSize))
    return mlir::failure();
  result.shortestHopDistances.assign(matrixSize,
                                     std::numeric_limits<uint64_t>::max());

  std::vector<uint64_t> endpointDistances(endpointCount,
                                          std::numeric_limits<uint64_t>::max());
  llvm::SmallVector<size_t, 64> queue;
  llvm::SmallVector<size_t, 8> neighbors;
  for (size_t sourceRank = 0; sourceRank < rankCount; ++sourceRank) {
    std::fill(endpointDistances.begin(), endpointDistances.end(),
              std::numeric_limits<uint64_t>::max());
    std::optional<size_t> sourceEndpoint = linearizeEndpoint(
        result.cardGrid, result.tileGrid, result.rankEndpoints[sourceRank]);
    if (!sourceEndpoint)
      return mlir::failure();
    endpointDistances[*sourceEndpoint] = 0;
    queue.clear();
    queue.push_back(*sourceEndpoint);
    for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
      const size_t current = queue[cursor];
      const uint64_t currentDistance = endpointDistances[current];
      if (currentDistance == std::numeric_limits<uint64_t>::max())
        return mlir::failure();
      neighbors.clear();
      appendAvailableNeighbors(neighbors, available, result.cardGrid,
                               result.tileGrid, result.getCardInterconnect(),
                               current);
      for (size_t neighbor : neighbors) {
        if (endpointDistances[neighbor] != std::numeric_limits<uint64_t>::max())
          continue;
        if (currentDistance == std::numeric_limits<uint64_t>::max() - 1)
          return mlir::failure();
        endpointDistances[neighbor] = currentDistance + 1;
        queue.push_back(neighbor);
      }
    }

    for (size_t destinationRank = 0; destinationRank < rankCount;
         ++destinationRank) {
      std::optional<size_t> destinationEndpoint =
          linearizeEndpoint(result.cardGrid, result.tileGrid,
                            result.rankEndpoints[destinationRank]);
      if (!destinationEndpoint || endpointDistances[*destinationEndpoint] ==
                                      std::numeric_limits<uint64_t>::max())
        return mlir::failure();
      result.shortestHopDistances[sourceRank * rankCount + destinationRank] =
          endpointDistances[*destinationEndpoint];
    }
  }
  return result;
}

std::optional<ExecutionEndpoint>
ExecutionTopologyAnalysis::getRankEndpoint(int64_t logicalRank) const {
  if (logicalRank < 0 ||
      static_cast<uint64_t>(logicalRank) >= rankEndpoints.size())
    return std::nullopt;
  return rankEndpoints[static_cast<size_t>(logicalRank)];
}

std::optional<uint64_t> ExecutionTopologyAnalysis::getShortestHopDistance(
    int64_t sourceRank, int64_t destinationRank) const {
  if (sourceRank < 0 || destinationRank < 0 ||
      static_cast<uint64_t>(sourceRank) >= rankEndpoints.size() ||
      static_cast<uint64_t>(destinationRank) >= rankEndpoints.size())
    return std::nullopt;
  const size_t rankCount = rankEndpoints.size();
  uint64_t distance =
      shortestHopDistances[static_cast<size_t>(sourceRank) * rankCount +
                           static_cast<size_t>(destinationRank)];
  if (distance == std::numeric_limits<uint64_t>::max())
    return std::nullopt;
  return distance;
}

mlir::FailureOr<llvm::SmallVector<ExecutionDirectedLink, 8>>
ExecutionTopologyAnalysis::getCanonicalShortestPath(
    int64_t sourceRank, int64_t destinationRank) const {
  if (sourceRank < 0 || destinationRank < 0 ||
      static_cast<uint64_t>(sourceRank) >= rankEndpoints.size() ||
      static_cast<uint64_t>(destinationRank) >= rankEndpoints.size())
    return mlir::failure();

  std::optional<size_t> sourceEndpoint = linearizeEndpoint(
      cardGrid, tileGrid, rankEndpoints[static_cast<size_t>(sourceRank)]);
  std::optional<size_t> destinationEndpoint = linearizeEndpoint(
      cardGrid, tileGrid, rankEndpoints[static_cast<size_t>(destinationRank)]);
  if (!sourceEndpoint || !destinationEndpoint ||
      !endpointAvailability[*sourceEndpoint] ||
      !endpointAvailability[*destinationEndpoint])
    return mlir::failure();

  llvm::SmallVector<ExecutionDirectedLink, 8> path;
  if (*sourceEndpoint == *destinationEndpoint)
    return path;

  const size_t endpointCount = endpointAvailability.size();
  const size_t unvisited = std::numeric_limits<size_t>::max();
  std::vector<size_t> predecessor(endpointCount, unvisited);
  llvm::SmallVector<size_t, 64> queue;
  llvm::SmallVector<size_t, 8> neighbors;
  predecessor[*sourceEndpoint] = *sourceEndpoint;
  queue.push_back(*sourceEndpoint);
  for (size_t cursor = 0;
       cursor < queue.size() && predecessor[*destinationEndpoint] == unvisited;
       ++cursor) {
    const size_t current = queue[cursor];
    neighbors.clear();
    appendAvailableNeighbors(neighbors, endpointAvailability, cardGrid,
                             tileGrid, getCardInterconnect(), current);
    llvm::sort(neighbors);
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
    for (size_t neighbor : neighbors) {
      if (predecessor[neighbor] != unvisited)
        continue;
      predecessor[neighbor] = current;
      queue.push_back(neighbor);
    }
  }
  if (predecessor[*destinationEndpoint] == unvisited)
    return mlir::failure();

  llvm::SmallVector<size_t, 8> reverseEndpoints;
  for (size_t current = *destinationEndpoint; current != *sourceEndpoint;
       current = predecessor[current])
    reverseEndpoints.push_back(current);
  reverseEndpoints.push_back(*sourceEndpoint);
  std::reverse(reverseEndpoints.begin(), reverseEndpoints.end());
  path.reserve(reverseEndpoints.size() - 1);
  for (size_t index = 1; index < reverseEndpoints.size(); ++index)
    path.push_back(
        {delinearizeEndpoint(reverseEndpoints[index - 1], cardGrid, tileGrid),
         delinearizeEndpoint(reverseEndpoints[index], cardGrid, tileGrid)});
  return path;
}

mlir::FailureOr<llvm::SmallVector<uint64_t, 16>>
ExecutionTopologyAnalysis::getShortestHopMatrix(
    llvm::ArrayRef<int64_t> logicalRanks) const {
  size_t matrixSize = 0;
  if (!checkedMultiplySize(logicalRanks.size(), logicalRanks.size(),
                           matrixSize))
    return mlir::failure();
  llvm::SmallVector<uint64_t, 16> matrix;
  matrix.reserve(matrixSize);
  for (int64_t source : logicalRanks)
    for (int64_t destination : logicalRanks) {
      std::optional<uint64_t> distance =
          getShortestHopDistance(source, destination);
      if (!distance)
        return mlir::failure();
      matrix.push_back(*distance);
    }
  return matrix;
}

bool ExecutionTopologyAnalysis::isEquivalentTo(
    const ExecutionTopologyAnalysis &other) const {
  return cardGrid == other.cardGrid && tileGrid == other.tileGrid &&
         cardInterconnect == other.cardInterconnect &&
         endpointAvailability == other.endpointAvailability &&
         rankEndpoints == other.rankEndpoints &&
         shortestHopDistances == other.shortestHopDistances &&
         directedLinkCount == other.directedLinkCount;
}

} // namespace wafer::analysis
