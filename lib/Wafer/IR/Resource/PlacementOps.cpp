//===- PlacementOps.cpp - Wafer Placement verifier implementation
//----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult PlacementMapOp::verify() {
  int64_t logicalRankCount = getLogicalRankCountAttr().getInt();
  if (logicalRankCount <= 0)
    return emitOpError("logical rank count must be positive");

  int64_t cardYCount = getCardYCountAttr().getInt();
  int64_t cardXCount = getCardXCountAttr().getInt();
  int64_t tileYCount = getTileYCountAttr().getInt();
  int64_t tileXCount = getTileXCountAttr().getInt();
  if (cardYCount <= 0 || cardXCount <= 0 || tileYCount <= 0 || tileXCount <= 0)
    return emitOpError("physical topology dimensions must be positive");

  int64_t expectedCoordEntries = 0;
  if (!checkedMul(logicalRankCount, 4, expectedCoordEntries))
    return emitOpError("logical rank count is too large to verify");

  llvm::ArrayRef<int64_t> coords = getPhysicalTileCoordsAttr().asArrayRef();
  if (static_cast<int64_t>(coords.size()) != expectedCoordEntries)
    return emitOpError("physical tile mapping must contain one 4D coordinate "
                       "per logical rank");

  int64_t cardCount = 0;
  int64_t rowCount = 0;
  int64_t totalTileCount = 0;
  if (!checkedMul(cardYCount, cardXCount, cardCount) ||
      !checkedMul(cardCount, tileYCount, rowCount) ||
      !checkedMul(rowCount, tileXCount, totalTileCount))
    return emitOpError("physical topology tile count is too large to verify");

  llvm::DenseSet<int64_t> badTileIds;
  for (int64_t badTileId : getBadTileIdsAttr().asArrayRef()) {
    if (badTileId < 0 || badTileId >= totalTileCount)
      return emitOpError("bad tile id must be within physical topology");
    badTileIds.insert(badTileId);
  }

  llvm::DenseSet<int64_t> usedTileIds;
  for (int64_t rank = 0; rank < logicalRankCount; ++rank) {
    int64_t base = rank * 4;
    int64_t cardY = coords[base];
    int64_t cardX = coords[base + 1];
    int64_t tileY = coords[base + 2];
    int64_t tileX = coords[base + 3];

    if (cardY < 0 || cardY >= cardYCount || cardX < 0 || cardX >= cardXCount ||
        tileY < 0 || tileY >= tileYCount || tileX < 0 || tileX >= tileXCount)
      return emitOpError("physical coordinate for logical rank ")
             << rank << " is outside target topology";

    std::optional<int64_t> tileId = getPhysicalTileId(
        cardY, cardX, tileY, tileX, cardXCount, tileYCount, tileXCount);
    if (!tileId)
      return emitOpError("physical tile id is too large to verify");

    if (badTileIds.contains(*tileId))
      return emitOpError("maps logical rank ")
             << rank << " to bad tile id " << *tileId;
    if (!usedTileIds.insert(*tileId).second)
      return emitOpError("maps multiple logical ranks to physical tile id ")
             << *tileId;
  }

  return mlir::success();
}
