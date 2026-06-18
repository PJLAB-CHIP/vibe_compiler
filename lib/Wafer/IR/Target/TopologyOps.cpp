//===- TopologyOps.cpp - Wafer target topology verifier implementation ----===//

#include "Wafer/IR/WaferDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;

namespace {

static bool isSupportedEncoding(llvm::StringRef encoding) {
  return encoding == "row_major_4d" || encoding == "card_major_4d" ||
         encoding == "runtime_capability" || encoding == "imported";
}

static mlir::LogicalResult
recordAvailability(mlir::Operation *op, llvm::StringRef setName,
                   llvm::ArrayRef<int64_t> ids,
                   const llvm::DenseSet<int64_t> &knownTileIds,
                   llvm::DenseMap<int64_t, llvm::StringRef> &seenAvailability) {
  llvm::DenseSet<int64_t> seenInSet;
  for (int64_t id : ids) {
    if (!knownTileIds.contains(id))
      return op->emitOpError()
             << setName << " tile id " << id
             << " does not reference a known encoded tile id";
    if (!seenInSet.insert(id).second)
      return op->emitOpError()
             << setName << " contains duplicate encoded tile id " << id;
    auto inserted = seenAvailability.try_emplace(id, setName);
    if (!inserted.second)
      return op->emitOpError()
             << "encoded tile id " << id
             << " appears in multiple availability sets";
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult TargetTopologyOp::verify() {
  if (!isSupportedEncoding(getIdEncodingAttr().getValue()))
    return emitOpError("unsupported tile id encoding provenance");

  mlir::ArrayAttr axes = getAxesAttr();
  if (axes.empty())
    return emitOpError("axes must not be empty");

  llvm::DenseSet<llvm::StringRef> seenAxes;
  for (mlir::Attribute axisAttr : axes) {
    auto axis = mlir::cast<mlir::StringAttr>(axisAttr).getValue();
    if (axis.empty())
      return emitOpError("axis name must not be empty");
    if (!seenAxes.insert(axis).second)
      return emitOpError("axis name must be unique");
  }

  llvm::ArrayRef<int64_t> tileIds = getTileIdsAttr().asArrayRef();
  if (tileIds.empty())
    return emitOpError("tile_ids must not be empty");

  llvm::ArrayRef<int64_t> tileCoords = getTileCoordsAttr().asArrayRef();
  int64_t axisCount = static_cast<int64_t>(axes.size());
  if (static_cast<int64_t>(tileCoords.size()) !=
      static_cast<int64_t>(tileIds.size()) * axisCount)
    return emitOpError(
        "tile_coords length must equal tile_ids length times axis count");

  llvm::DenseSet<int64_t> knownTileIds;
  for (int64_t id : tileIds) {
    if (id < 0)
      return emitOpError("encoded tile ids must be non-negative");
    if (!knownTileIds.insert(id).second)
      return emitOpError("maps multiple tile coordinates to encoded tile id ")
             << id;
  }

  for (auto [index, coord] : llvm::enumerate(tileCoords)) {
    if (coord < 0)
      return emitOpError("tile coordinate entry ")
             << static_cast<int64_t>(index) << " must be non-negative";
  }

  llvm::DenseMap<int64_t, llvm::StringRef> seenAvailability;
  if (mlir::failed(recordAvailability(
          getOperation(), "available", getAvailableTileIdsAttr().asArrayRef(),
          knownTileIds, seenAvailability)))
    return mlir::failure();
  if (mlir::failed(recordAvailability(getOperation(), "bad",
                                      getBadTileIdsAttr().asArrayRef(),
                                      knownTileIds, seenAvailability)))
    return mlir::failure();
  if (mlir::failed(recordAvailability(
          getOperation(), "pg-disabled",
          getPgDisabledTileIdsAttr().asArrayRef(), knownTileIds,
          seenAvailability)))
    return mlir::failure();

  if (seenAvailability.size() != knownTileIds.size())
    return emitOpError("every encoded tile id must appear in exactly one "
                       "availability set");

  llvm::ArrayRef<int64_t> links = getLinksAttr().asArrayRef();
  if (links.size() % 2 != 0)
    return emitOpError("links must contain source/destination tile id pairs");
  for (int64_t endpoint : links) {
    if (!knownTileIds.contains(endpoint))
      return emitOpError("link endpoint ")
             << endpoint << " does not reference a known encoded tile id";
  }

  return mlir::success();
}
