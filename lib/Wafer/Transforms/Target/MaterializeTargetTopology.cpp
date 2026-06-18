//===- MaterializeTargetTopology.cpp - Materialize Wafer topology facts ---===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <memory>

namespace wafer {
#define GEN_PASS_DEF_MATERIALIZETARGETTOPOLOGYPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
parseOptionalI64List(llvm::StringRef text, llvm::StringRef optionName,
                     mlir::Operation *anchor) {
  llvm::SmallVector<int64_t, 8> values;
  text = text.trim();
  if (text.empty())
    return values;

  if (text.starts_with(",") || text.ends_with(",")) {
    anchor->emitError() << "topology_failure: invalid empty entry in "
                        << optionName;
    return mlir::failure();
  }

  llvm::StringRef rest = text;
  while (!rest.empty()) {
    auto split = rest.split(',');
    llvm::StringRef part = split.first.trim();
    if (part.empty()) {
      anchor->emitError() << "topology_failure: invalid empty entry in "
                          << optionName;
      return mlir::failure();
    }
    int64_t value = 0;
    if (part.getAsInteger(10, value)) {
      anchor->emitError() << "topology_failure: invalid integer in "
                          << optionName << ": " << part;
      return mlir::failure();
    }
    values.push_back(value);
    rest = split.second;
  }
  return values;
}

static mlir::LogicalResult
validateTopologyShape(mlir::ModuleOp moduleOp, int64_t cardYCount,
                      int64_t cardXCount, int64_t tileYCount,
                      int64_t tileXCount, int64_t &totalTileCount) {
  if (cardYCount <= 0 || cardXCount <= 0 || tileYCount <= 0 ||
      tileXCount <= 0)
    return moduleOp.emitError()
           << "topology_failure: target topology dimensions must be positive";

  int64_t cardCount = 0;
  int64_t rowCount = 0;
  if (!checkedMul(cardYCount, cardXCount, cardCount) ||
      !checkedMul(cardCount, tileYCount, rowCount) ||
      !checkedMul(rowCount, tileXCount, totalTileCount))
    return moduleOp.emitError()
           << "topology_failure: target topology tile count is too large";

  return mlir::success();
}

static mlir::LogicalResult
validateExcludedTiles(mlir::ModuleOp moduleOp, llvm::StringRef optionName,
                      llvm::ArrayRef<int64_t> tileIds,
                      const llvm::DenseSet<int64_t> &knownTileIds,
                      llvm::DenseSet<int64_t> &excludedTiles) {
  for (int64_t tileId : tileIds) {
    if (!knownTileIds.contains(tileId))
      return moduleOp.emitError()
             << "topology_failure: " << optionName
             << " entry must reference a known encoded tile id";
    if (!excludedTiles.insert(tileId).second)
      return moduleOp.emitError()
             << "topology_failure: duplicate tile id in " << optionName;
  }
  return mlir::success();
}

static mlir::LogicalResult materializeTargetTopology(
    mlir::ModuleOp moduleOp, llvm::StringRef topologyName,
    llvm::StringRef idEncoding, int64_t cardYCount, int64_t cardXCount,
    int64_t tileYCount, int64_t tileXCount, llvm::StringRef tileIdRemapOption,
    llvm::StringRef badTileIdsOption,
    llvm::StringRef pgDisabledTileIdsOption) {
  bool hasTopology = false;
  moduleOp.walk([&](TargetTopologyOp) { hasTopology = true; });
  if (hasTopology)
    return moduleOp.emitError()
           << "topology_failure: module already contains "
              "wafer.target.topology";

  if (topologyName.empty())
    return moduleOp.emitError()
           << "topology_failure: topology symbol name must not be empty";

  int64_t totalTileCount = 0;
  if (mlir::failed(validateTopologyShape(moduleOp, cardYCount, cardXCount,
                                         tileYCount, tileXCount,
                                         totalTileCount)))
    return mlir::failure();

  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> tileIdRemap =
      parseOptionalI64List(tileIdRemapOption, "tile-id-remap", moduleOp);
  if (mlir::failed(tileIdRemap))
    return mlir::failure();
  if (!tileIdRemap->empty() &&
      static_cast<int64_t>(tileIdRemap->size()) != totalTileCount)
    return moduleOp.emitError()
           << "topology_failure: tile-id-remap length must equal target "
              "topology tile count";
  if (tileIdRemap->empty() && idEncoding != "row_major_4d")
    return moduleOp.emitError()
           << "topology_failure: non-default id encoding requires "
              "tile-id-remap";

  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> badTileIds =
      parseOptionalI64List(badTileIdsOption, "bad-tile-ids", moduleOp);
  if (mlir::failed(badTileIds))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> pgDisabledTileIds =
      parseOptionalI64List(pgDisabledTileIdsOption, "pg-disabled-tile-ids",
                           moduleOp);
  if (mlir::failed(pgDisabledTileIds))
    return mlir::failure();

  llvm::SmallVector<int64_t, 64> tileCoords;
  llvm::SmallVector<int64_t, 16> tileIds;
  llvm::SmallVector<int64_t, 16> availableTileIds;
  llvm::SmallVector<int64_t, 64> links;
  tileCoords.reserve(static_cast<size_t>(totalTileCount) * 4);
  tileIds.reserve(totalTileCount);
  availableTileIds.reserve(totalTileCount);

  llvm::DenseSet<int64_t> knownTileIds;
  if (!tileIdRemap->empty()) {
    for (int64_t tileId : *tileIdRemap) {
      if (tileId < 0)
        return moduleOp.emitError()
               << "topology_failure: tile-id-remap entries must be "
                  "non-negative";
      if (!knownTileIds.insert(tileId).second)
        return moduleOp.emitError()
               << "topology_failure: tile-id-remap entries must be unique";
      tileIds.push_back(tileId);
    }
  } else {
    for (int64_t tileId = 0; tileId < totalTileCount; ++tileId) {
      knownTileIds.insert(tileId);
      tileIds.push_back(tileId);
    }
  }

  llvm::DenseSet<int64_t> badTiles;
  if (mlir::failed(validateExcludedTiles(moduleOp, "bad-tile-ids",
                                         *badTileIds, knownTileIds, badTiles)))
    return mlir::failure();

  llvm::DenseSet<int64_t> pgDisabledTiles;
  if (mlir::failed(validateExcludedTiles(moduleOp, "pg-disabled-tile-ids",
                                         *pgDisabledTileIds, knownTileIds,
                                         pgDisabledTiles)))
    return mlir::failure();

  for (int64_t tileId : *badTileIds) {
    if (pgDisabledTiles.contains(tileId))
      return moduleOp.emitError()
             << "topology_failure: tile id " << tileId
             << " cannot be both bad and PG-disabled";
  }

  int64_t ordinal = 0;
  for (int64_t cardY = 0; cardY < cardYCount; ++cardY) {
    for (int64_t cardX = 0; cardX < cardXCount; ++cardX) {
      for (int64_t tileY = 0; tileY < tileYCount; ++tileY) {
        for (int64_t tileX = 0; tileX < tileXCount; ++tileX) {
          int64_t tileId = tileIds[ordinal];

          tileCoords.append({cardY, cardX, tileY, tileX});
          if (!badTiles.contains(tileId) && !pgDisabledTiles.contains(tileId))
            availableTileIds.push_back(tileId);

          if (tileX + 1 < tileXCount)
            links.append({tileId, tileIds[ordinal + 1]});
          if (tileY + 1 < tileYCount)
            links.append({tileId, tileIds[ordinal + tileXCount]});
          ++ordinal;
        }
      }
    }
  }

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  builder.create<TargetTopologyOp>(
      moduleOp.getLoc(), builder.getStringAttr(topologyName),
      builder.getStrArrayAttr({"card_y", "card_x", "tile_y", "tile_x"}),
      builder.getStringAttr(idEncoding),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), tileCoords),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), tileIds),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), availableTileIds),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), *badTileIds),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), *pgDisabledTileIds),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), links));

  return mlir::success();
}

struct MaterializeTargetTopologyPass
    : public impl::MaterializeTargetTopologyPassBase<
          MaterializeTargetTopologyPass> {
  using impl::MaterializeTargetTopologyPassBase<
      MaterializeTargetTopologyPass>::MaterializeTargetTopologyPassBase;

  void runOnOperation() final {
    if (mlir::failed(materializeTargetTopology(
            getOperation(), topologyName, idEncoding, cardYCount, cardXCount,
            tileYCount, tileXCount, tileIdRemap, badTileIds,
            pgDisabledTileIds))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace wafer
