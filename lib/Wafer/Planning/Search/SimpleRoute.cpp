//===- SimpleRoute.cpp - Lazy simple physical routes ------------------===//

#include "Wafer/Planning/Search/SimpleRoute.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>

namespace wafer::compiler::detail::simple_route {

bool isValidSimpleRoute(const TargetTopology &topology, CardId cardId,
                        TileId source, TileId destination,
                        llvm::ArrayRef<TileLink> route) {
  if (source == destination)
    return route.empty();
  if (route.empty() || route.front().source != source ||
      route.back().destination != destination)
    return false;
  llvm::DenseSet<int64_t> visited;
  visited.insert(source.getValue());
  TileId current = source;
  for (const TileLink &link : route) {
    if (link.source != current ||
        !topology.areOnCardAdjacent(cardId, link.source, link.destination) ||
        !visited.insert(link.destination.getValue()).second)
      return false;
    current = link.destination;
  }
  return current == destination;
}

std::optional<llvm::SmallVector<TileLink, 8>>
getFirstSimpleRoute(const TargetTopology &topology, CardId cardId,
                    TileId source, TileId destination) {
  std::optional<llvm::SmallVector<TileLink, 8>> result;
  llvm::SmallVector<TileId, 16> path{source};
  llvm::DenseSet<int64_t> visited{source.getValue()};
  std::function<void(TileId)> visit = [&](TileId current) {
    if (result)
      return;
    if (current == destination) {
      llvm::SmallVector<TileLink, 8> links;
      for (size_t index = 1; index < path.size(); ++index)
        links.push_back(TileLink{path[index - 1], path[index]});
      result = std::move(links);
      return;
    }
    auto neighbors = topology.getOnCardNeighbors(cardId, current);
    if (mlir::failed(neighbors))
      return;
    for (TileId neighbor : *neighbors) {
      if (!visited.insert(neighbor.getValue()).second)
        continue;
      path.push_back(neighbor);
      visit(neighbor);
      path.pop_back();
      visited.erase(neighbor.getValue());
    }
  };
  visit(source);
  return result;
}

RouteSuccessor getNextSimpleRoute(const TargetTopology &topology, CardId cardId,
                                  TileId source, TileId destination,
                                  llvm::ArrayRef<TileLink> currentRoute) {
  RouteSuccessor result;
  llvm::SmallVector<TileId, 16> path{source};
  llvm::DenseSet<int64_t> visited{source.getValue()};
  std::function<void(TileId)> visit = [&](TileId current) {
    if (result.next)
      return;
    if (current == destination) {
      llvm::SmallVector<TileLink, 8> links;
      for (size_t index = 1; index < path.size(); ++index)
        links.push_back(TileLink{path[index - 1], path[index]});
      if (result.currentFound) {
        result.next = std::move(links);
      } else if (llvm::equal(links, currentRoute)) {
        result.currentFound = true;
      }
      return;
    }
    auto neighbors = topology.getOnCardNeighbors(cardId, current);
    if (mlir::failed(neighbors))
      return;
    for (TileId neighbor : *neighbors) {
      if (!visited.insert(neighbor.getValue()).second)
        continue;
      path.push_back(neighbor);
      visit(neighbor);
      path.pop_back();
      visited.erase(neighbor.getValue());
      if (result.next)
        return;
    }
  };
  visit(source);
  return result;
}

} // namespace wafer::compiler::detail::simple_route
