//===- PlanPlacement.cpp - Plan Wafer physical placement ------------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace wafer {
#define GEN_PASS_DEF_PLANPLACEMENTPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

struct PhysicalCoord {
  int64_t cardY = 0;
  int64_t cardX = 0;
  int64_t tileY = 0;
  int64_t tileX = 0;
};

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static std::optional<int64_t>
getPhysicalTileId(int64_t cardY, int64_t cardX, int64_t tileY, int64_t tileX,
                  int64_t cardXCount, int64_t tileYCount,
                  int64_t tileXCount) {
  int64_t cardBase = 0;
  if (!checkedMul(cardY, cardXCount, cardBase))
    return std::nullopt;
  int64_t cardIndex = 0;
  if (!checkedAdd(cardBase, cardX, cardIndex))
    return std::nullopt;

  int64_t tileBase = 0;
  if (!checkedMul(cardIndex, tileYCount, tileBase))
    return std::nullopt;
  int64_t tileRow = 0;
  if (!checkedAdd(tileBase, tileY, tileRow))
    return std::nullopt;

  int64_t tileIdBase = 0;
  if (!checkedMul(tileRow, tileXCount, tileIdBase))
    return std::nullopt;
  int64_t tileId = 0;
  if (!checkedAdd(tileIdBase, tileX, tileId))
    return std::nullopt;
  return tileId;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
parseOptionalI64List(llvm::StringRef text, llvm::StringRef optionName,
                     mlir::Operation *anchor) {
  llvm::SmallVector<int64_t, 8> values;
  text = text.trim();
  if (text.empty())
    return values;

  if (text.starts_with(",") || text.ends_with(",")) {
    anchor->emitError() << "invalid empty entry in " << optionName;
    return mlir::failure();
  }

  llvm::StringRef rest = text;
  while (!rest.empty()) {
    auto split = rest.split(',');
    llvm::StringRef part = split.first.trim();
    if (part.empty()) {
      anchor->emitError() << "invalid empty entry in " << optionName;
      return mlir::failure();
    }
    int64_t value = 0;
    if (part.getAsInteger(10, value)) {
      anchor->emitError() << "invalid integer in " << optionName << ": "
                          << part;
      return mlir::failure();
    }
    values.push_back(value);
    rest = split.second;
  }
  return values;
}

static mlir::LogicalResult
validatePositiveTopology(mlir::ModuleOp moduleOp, int64_t logicalRankCount,
                         int64_t cardYCount, int64_t cardXCount,
                         int64_t tileYCount, int64_t tileXCount,
                         int64_t &totalTileCount) {
  if (logicalRankCount <= 0)
    return moduleOp.emitError()
           << "placement_failure: logical rank count must be positive";
  if (cardYCount <= 0 || cardXCount <= 0 || tileYCount <= 0 ||
      tileXCount <= 0)
    return moduleOp.emitError()
           << "placement_failure: physical topology dimensions must be "
              "positive";

  int64_t cardCount = 0;
  int64_t rowCount = 0;
  if (!checkedMul(cardYCount, cardXCount, cardCount) ||
      !checkedMul(cardCount, tileYCount, rowCount) ||
      !checkedMul(rowCount, tileXCount, totalTileCount))
    return moduleOp.emitError()
           << "placement_failure: physical topology tile count is too large";

  return mlir::success();
}

static mlir::FailureOr<llvm::SmallVector<PhysicalCoord, 8>>
computeDeterministicPlacement(mlir::ModuleOp moduleOp, int64_t logicalRankCount,
                              int64_t cardYCount, int64_t cardXCount,
                              int64_t tileYCount, int64_t tileXCount,
                              llvm::ArrayRef<int64_t> badTileIds) {
  int64_t totalTileCount = 0;
  if (mlir::failed(validatePositiveTopology(moduleOp, logicalRankCount,
                                            cardYCount, cardXCount, tileYCount,
                                            tileXCount, totalTileCount)))
    return mlir::failure();

  llvm::DenseSet<int64_t> badTiles;
  for (int64_t badTileId : badTileIds) {
    if (badTileId < 0 || badTileId >= totalTileCount) {
      moduleOp.emitError()
          << "placement_failure: bad tile id must be within physical topology";
      return mlir::failure();
    }
    badTiles.insert(badTileId);
  }

  llvm::SmallVector<PhysicalCoord, 8> goodTiles;
  for (int64_t cardY = 0; cardY < cardYCount; ++cardY) {
    for (int64_t cardX = 0; cardX < cardXCount; ++cardX) {
      for (int64_t tileY = 0; tileY < tileYCount; ++tileY) {
        for (int64_t tileX = 0; tileX < tileXCount; ++tileX) {
          std::optional<int64_t> tileId =
              getPhysicalTileId(cardY, cardX, tileY, tileX, cardXCount,
                                tileYCount, tileXCount);
          if (!tileId) {
            moduleOp.emitError()
                << "placement_failure: physical tile id is too large";
            return mlir::failure();
          }
          if (badTiles.contains(*tileId))
            continue;
          goodTiles.push_back(PhysicalCoord{cardY, cardX, tileY, tileX});
        }
      }
    }
  }

  if (static_cast<int64_t>(goodTiles.size()) < logicalRankCount) {
    moduleOp.emitError()
        << "placement_failure: logical rank count " << logicalRankCount
        << " exceeds available good tile count " << goodTiles.size();
    return mlir::failure();
  }

  goodTiles.truncate(logicalRankCount);
  return goodTiles;
}

static mlir::LogicalResult
planPlacementModule(mlir::ModuleOp moduleOp, int64_t logicalRankCount,
                    int64_t cardYCount, int64_t cardXCount,
                    int64_t tileYCount, int64_t tileXCount,
                    llvm::StringRef badTileIdsOption) {
  bool hasPlacement = false;
  moduleOp.walk([&](PlacementMapOp) { hasPlacement = true; });
  if (hasPlacement)
    return moduleOp.emitError()
           << "placement_failure: module already contains wafer.placement.map";

  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> parsedBadTileIds =
      parseOptionalI64List(badTileIdsOption, "bad-tile-ids", moduleOp);
  if (mlir::failed(parsedBadTileIds))
    return mlir::failure();

  mlir::FailureOr<llvm::SmallVector<PhysicalCoord, 8>> placement =
      computeDeterministicPlacement(moduleOp, logicalRankCount, cardYCount,
                                    cardXCount, tileYCount, tileXCount,
                                    *parsedBadTileIds);
  if (mlir::failed(placement))
    return mlir::failure();

  llvm::SmallVector<int64_t, 16> coords;
  llvm::SmallVector<int64_t, 8> blockIds;
  coords.reserve(static_cast<size_t>(logicalRankCount) * 4);
  blockIds.reserve(logicalRankCount);
  for (auto [rank, coord] : llvm::enumerate(*placement)) {
    blockIds.push_back(static_cast<int64_t>(rank));
    coords.push_back(coord.cardY);
    coords.push_back(coord.cardX);
    coords.push_back(coord.tileY);
    coords.push_back(coord.tileX);
  }

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  builder.create<PlacementMapOp>(
      moduleOp.getLoc(), builder.getI64IntegerAttr(logicalRankCount),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), blockIds),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), coords),
      builder.getI64IntegerAttr(cardYCount),
      builder.getI64IntegerAttr(cardXCount),
      builder.getI64IntegerAttr(tileYCount),
      builder.getI64IntegerAttr(tileXCount),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), *parsedBadTileIds));

  return mlir::success();
}

struct PlanPlacementPass
    : public impl::PlanPlacementPassBase<PlanPlacementPass> {
  using impl::PlanPlacementPassBase<
      PlanPlacementPass>::PlanPlacementPassBase;

  PlanPlacementPass(int64_t logicalRankCount, int64_t cardYCount,
                    int64_t cardXCount, int64_t tileYCount,
                    int64_t tileXCount, llvm::StringRef badTileIds) {
    this->logicalRankCount = logicalRankCount;
    this->cardYCount = cardYCount;
    this->cardXCount = cardXCount;
    this->tileYCount = tileYCount;
    this->tileXCount = tileXCount;
    this->badTileIds = badTileIds.str();
  }

  void runOnOperation() final {
    if (mlir::failed(planPlacementModule(
            getOperation(), logicalRankCount, cardYCount, cardXCount,
            tileYCount, tileXCount, badTileIds))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
createPlanPlacementPass(int64_t logicalRankCount, int64_t cardYCount,
                        int64_t cardXCount, int64_t tileYCount,
                        int64_t tileXCount, llvm::StringRef badTileIds) {
  return std::make_unique<PlanPlacementPass>(
      logicalRankCount, cardYCount, cardXCount, tileYCount, tileXCount,
      badTileIds);
}

} // namespace wafer
