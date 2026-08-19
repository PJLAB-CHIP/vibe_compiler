//===- BufferingApply.cpp - Lower selected buffer scopes --------------===//

#include "Wafer/Planning/Search/BufferingApply.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {

mlir::FailureOr<std::vector<llvm::SmallVector<SelectedBufferingScope, 4>>>
buildSelectedBufferingScopes(
    const CardProgramAnalysis &program, const CardBufferingDomain &domain,
    const CardBufferingAssignment &assignment,
    const CardDataMovementDomain &movementDomain,
    const CardDataMovementAssignment &movementAssignment,
    llvm::ArrayRef<TileId> expectedTileIds, std::string *failureReason) {
  auto fail = [&](llvm::StringRef message)
      -> mlir::FailureOr<
          std::vector<llvm::SmallVector<SelectedBufferingScope, 4>>> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  if (!domain.contains(assignment) ||
      !movementDomain.contains(movementAssignment) || expectedTileIds.empty())
    return fail("selected buffering scopes received a stale assignment or "
                "empty Tile domain");

  llvm::DenseSet<int64_t> seenTiles;
  for (TileId tile : expectedTileIds)
    if (!seenTiles.insert(tile.getValue()).second)
      return fail("selected buffering Tile domain contains a duplicate");
  std::vector<llvm::SmallVector<SelectedBufferingScope, 4>> result(
      expectedTileIds.size());
  for (const BufferingChoice &choice : assignment.groups) {
    if (choice.slotCount == 1)
      continue;
    auto tile = llvm::find(expectedTileIds, choice.tile);
    if (tile == expectedTileIds.end())
      return fail("selected buffering scope names an unavailable Tile");
    SelectedBufferingScope scope;
    for (StructuredDAGEdgeID edgeId : choice.pipelinedEdges) {
      const StructuredDAGEdge *edge = program.dag.getEdge(edgeId);
      if (!edge || !llvm::is_contained(choice.groupNodes, edge->producer) ||
          !llvm::is_contained(choice.groupNodes, edge->consumer))
        return fail("selected buffering edge is outside its coupled group");
      auto movement = llvm::find_if(
          movementAssignment.edges, [&](const DataMovementChoice &candidate) {
            return candidate.edge == edgeId &&
                   candidate.destinationTile == choice.tile;
          });
      if (movement == movementAssignment.edges.end() ||
          movement->kind != DataMovementKind::Retained)
        return fail("selected buffering edge is not retained in its exact "
                    "coupled group");
      SelectedBufferRequest request;
      request.producerNode = edge->producer;
      request.consumerNode = edge->consumer;
      request.bufferCount = choice.slotCount;
      request.requireLocalDataflow = true;
      scope.requests.push_back(std::move(request));
    }
    if (scope.requests.empty())
      return fail("multi-slot buffering scope has no exact logical edge");
    result[static_cast<size_t>(tile - expectedTileIds.begin())].push_back(
        std::move(scope));
  }
  return result;
}

} // namespace wafer::compiler::detail
