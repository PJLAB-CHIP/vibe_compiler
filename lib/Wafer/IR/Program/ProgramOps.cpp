//===- ProgramOps.cpp - Wafer whole-card program verification ------------===//

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"

#include <cstdint>
#include <limits>

using namespace wafer;

namespace {

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static mlir::FailureOr<TargetTopologyOp>
getCurrentTopology(mlir::Operation *owner, mlir::ModuleOp module) {
  TargetTopologyOp current;
  for (TargetTopologyOp topology : module.getOps<TargetTopologyOp>()) {
    if (current) {
      owner->emitOpError(
          "requires exactly one direct wafer.target.topology in its module");
      return mlir::failure();
    }
    current = topology;
  }
  if (!current) {
    owner->emitOpError(
        "requires one direct wafer.target.topology in its module");
    return mlir::failure();
  }
  return current;
}

struct TopologyShape {
  mlir::DenseI64ArrayAttr unavailableTiles;
  int64_t cardColumns = 0;
  int64_t tileColumns = 0;
  int64_t cardCount = 0;
  int64_t tileCount = 0;
};

static mlir::FailureOr<TopologyShape> getTopologyShape(mlir::Operation *owner,
                                                       mlir::ModuleOp module) {
  mlir::FailureOr<TargetTopologyOp> topology =
      getCurrentTopology(owner, module);
  if (mlir::failed(topology))
    return mlir::failure();

  llvm::ArrayRef<int64_t> cardGrid = topology->getCardGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> tileGrid = topology->getTileGridAttr().asArrayRef();
  llvm::ArrayRef<int64_t> unavailable =
      topology->getUnavailableTilesAttr().asArrayRef();
  if (cardGrid.size() != 2 || tileGrid.size() != 2 || cardGrid[0] <= 0 ||
      cardGrid[1] <= 0 || tileGrid[0] <= 0 || tileGrid[1] <= 0 ||
      unavailable.size() % 4 != 0) {
    owner->emitOpError("references malformed wafer.target.topology @")
        << topology->getSymName();
    return mlir::failure();
  }

  TopologyShape shape;
  shape.unavailableTiles = topology->getUnavailableTilesAttr();
  shape.cardColumns = cardGrid[1];
  shape.tileColumns = tileGrid[1];
  if (!checkedMul(cardGrid[0], cardGrid[1], shape.cardCount) ||
      !checkedMul(tileGrid[0], tileGrid[1], shape.tileCount)) {
    owner->emitOpError("target topology is too large to verify");
    return mlir::failure();
  }
  return shape;
}

static bool isUnavailable(const TopologyShape &shape, int64_t cardId,
                          int64_t tileId) {
  int64_t cardY = cardId / shape.cardColumns;
  int64_t cardX = cardId % shape.cardColumns;
  int64_t tileY = tileId / shape.tileColumns;
  int64_t tileX = tileId % shape.tileColumns;
  llvm::ArrayRef<int64_t> unavailable = shape.unavailableTiles.asArrayRef();
  for (size_t index = 0; index < unavailable.size(); index += 4) {
    if (unavailable[index] == cardY && unavailable[index + 1] == cardX &&
        unavailable[index + 2] == tileY && unavailable[index + 3] == tileX)
      return true;
  }
  return false;
}

static mlir::LogicalResult verifyCardId(mlir::Operation *owner,
                                        const TopologyShape &shape,
                                        int64_t cardId) {
  if (cardId < 0 || cardId >= shape.cardCount)
    return owner->emitOpError("card_id ")
           << cardId << " is outside target card grid [0, " << shape.cardCount
           << ")";
  return mlir::success();
}

static mlir::LogicalResult verifyTileId(mlir::Operation *owner,
                                        const TopologyShape &shape,
                                        int64_t cardId, int64_t tileId) {
  if (tileId < 0 || tileId >= shape.tileCount)
    return owner->emitOpError("tile_id ")
           << tileId << " is outside target Tile grid [0, " << shape.tileCount
           << ")";
  if (isUnavailable(shape, cardId, tileId))
    return owner->emitOpError("tile_id ")
           << tileId << " is unavailable for card_id " << cardId;
  return mlir::success();
}

template <typename ProgramOp>
static mlir::LogicalResult
verifyOnlyIdentifierAttribute(ProgramOp op, mlir::StringAttr identifierName) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (attr.getName() != identifierName)
      return op.emitOpError("does not accept semantic attribute '")
             << attr.getName() << "'";
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult CardProgramOp::verify() {
  auto module = mlir::dyn_cast<mlir::ModuleOp>(getOperation()->getParentOp());
  if (!module)
    return emitOpError("must be directly nested under a builtin.module");
  if (!getBody().hasOneBlock())
    return emitOpError("must contain exactly one body block");

  mlir::FailureOr<TopologyShape> shape =
      getTopologyShape(getOperation(), module);
  if (mlir::failed(shape))
    return mlir::failure();

  int64_t cardId = getCardId();
  if (mlir::failed(verifyCardId(getOperation(), *shape, cardId)))
    return mlir::failure();

  for (CardProgramOp sibling : module.getOps<CardProgramOp>()) {
    if (sibling != *this && sibling.getCardId() == cardId)
      return emitOpError("card_id must be unique in its module; duplicate ")
             << cardId;
  }

  llvm::DenseSet<int64_t> seenTileIds;
  for (TileProgramOp tile : getBody().front().getOps<TileProgramOp>()) {
    int64_t tileId = tile.getTileId();
    if (mlir::failed(verifyTileId(getOperation(), *shape, cardId, tileId)))
      return mlir::failure();
    if (!seenTileIds.insert(tileId).second)
      return emitOpError("tile_id must be unique in a card program; duplicate ")
             << tileId;
  }

  for (int64_t tileId = 0; tileId < shape->tileCount; ++tileId) {
    if (isUnavailable(*shape, cardId, tileId))
      continue;
    if (!seenTileIds.contains(tileId))
      return emitOpError("is missing available tile_id ") << tileId;
  }

  if (mlir::failed(verifyOnlyIdentifierAttribute(*this, getCardIdAttrName())))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult TileProgramOp::verify() {
  CardProgramOp card =
      mlir::dyn_cast<CardProgramOp>(getOperation()->getParentOp());
  if (!card)
    return emitOpError("must be directly nested under wafer.card.program");
  if (!getBody().hasOneBlock())
    return emitOpError("must contain exactly one body block");

  auto module = mlir::dyn_cast<mlir::ModuleOp>(card->getParentOp());
  if (!module)
    return emitOpError(
        "requires its wafer.card.program to be directly nested under a "
        "builtin.module");

  mlir::FailureOr<TopologyShape> shape =
      getTopologyShape(getOperation(), module);
  if (mlir::failed(shape))
    return mlir::failure();
  if (mlir::failed(verifyCardId(getOperation(), *shape, card.getCardId())) ||
      mlir::failed(
          verifyTileId(getOperation(), *shape, card.getCardId(), getTileId())))
    return mlir::failure();

  if (mlir::failed(verifyOnlyIdentifierAttribute(*this, getTileIdAttrName())))
    return mlir::failure();
  return mlir::success();
}
