//===- ProgramOps.cpp - Wafer whole-card program verification ------------===//

#include "Wafer/IR/Target/PhysicalTopology.h"
#include "Wafer/IR/WaferDialect.h"

#include "../Common/WaferIRVerification.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <string>

using namespace wafer;

namespace {

static mlir::FailureOr<PhysicalTopology>
getPhysicalTopology(mlir::Operation *owner, mlir::ModuleOp module) {
  std::string failureReason;
  mlir::FailureOr<PhysicalTopology> topology =
      PhysicalTopology::create(module, &failureReason);
  if (mlir::failed(topology)) {
    owner->emitOpError() << failureReason;
    return mlir::failure();
  }
  return topology;
}

static mlir::LogicalResult verifyCardId(mlir::Operation *owner,
                                        const PhysicalTopology &topology,
                                        PhysicalCardId cardId) {
  if (!topology.getCardCoordinate(cardId))
    return owner->emitOpError("card_id ")
           << cardId.getValue() << " is outside target card grid [0, "
           << topology.getCardCount() << ")";
  return mlir::success();
}

static mlir::LogicalResult verifyTileId(mlir::Operation *owner,
                                        const PhysicalTopology &topology,
                                        PhysicalCardId cardId,
                                        PhysicalTileId tileId) {
  if (!topology.getTileCoordinate(tileId))
    return owner->emitOpError("tile_id ")
           << tileId.getValue() << " is outside target Tile grid [0, "
           << topology.getTilesPerCard() << ")";
  if (!topology.isTileAvailable(cardId, tileId))
    return owner->emitOpError("tile_id ")
           << tileId.getValue() << " is unavailable for card_id "
           << cardId.getValue();
  return mlir::success();
}

static std::optional<int64_t> getPeerPhysicalTileId(mlir::Operation *op) {
  if (auto peerSend = mlir::dyn_cast<CommPeerSendOp>(op))
    return peerSend.getPeerAttr().getInt();
  if (auto peerRecv = mlir::dyn_cast<CommPeerRecvOp>(op))
    return peerRecv.getPeerAttr().getInt();
  if (auto dteSend = mlir::dyn_cast<InstrDTESendOp>(op))
    return dteSend.getPeerAttr().getInt();
  if (auto dteRecv = mlir::dyn_cast<InstrDTERecvOp>(op))
    return dteRecv.getPeerAttr().getInt();
  return std::nullopt;
}

static mlir::LogicalResult
verifyProgramPeerDomain(TileProgramOp tileProgram,
                        const PhysicalTopology &topology,
                        PhysicalCardId cardId) {
  mlir::WalkResult walkResult =
      tileProgram.walk([&](mlir::Operation *operation) {
        std::optional<int64_t> peer = getPeerPhysicalTileId(operation);
        if (!peer)
          return mlir::WalkResult::advance();
        if (*peer < 0 ||
            !topology.isTileAvailable(cardId, PhysicalTileId(*peer))) {
          operation->emitOpError()
              << "peer tile_id " << *peer
              << " is outside the available physical Tile domain for card_id "
              << cardId.getValue();
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
  return walkResult.wasInterrupted() ? mlir::failure() : mlir::success();
}

template <typename ProgramOp>
static mlir::LogicalResult
verifyOnlyIdentifierAttribute(ProgramOp op, mlir::StringAttr identifierName) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (attr.getName() != identifierName &&
        !wafer::detail::isExternalDiscardableAttribute(op, attr))
      return op.emitOpError("does not accept semantic attribute '")
             << attr.getName().getValue() << "'";
  }
  return mlir::success();
}

static bool isCardSharedDeclaration(mlir::Operation &operation) {
  if (!mlir::isa<mlir::SymbolOpInterface>(operation))
    return false;
  return llvm::all_of(operation.getRegions(),
                      [](mlir::Region &region) { return region.empty(); });
}

} // namespace

mlir::LogicalResult CardProgramOp::verify() {
  auto module = mlir::dyn_cast<mlir::ModuleOp>(getOperation()->getParentOp());
  if (!module)
    return emitOpError("must be directly nested under a builtin.module");
  if (!getBody().hasOneBlock())
    return emitOpError("must contain exactly one body block");

  mlir::FailureOr<PhysicalTopology> topology =
      getPhysicalTopology(getOperation(), module);
  if (mlir::failed(topology))
    return mlir::failure();

  PhysicalCardId cardId(getCardIdAttr().getInt());
  if (mlir::failed(verifyCardId(getOperation(), *topology, cardId)))
    return mlir::failure();

  for (CardProgramOp sibling : module.getOps<CardProgramOp>()) {
    if (sibling != *this &&
        sibling.getCardIdAttr().getInt() == cardId.getValue())
      return emitOpError("card_id must be unique in its module; duplicate ")
             << cardId.getValue();
  }

  llvm::DenseSet<int64_t> seenTileIds;
  for (mlir::Operation &operation : getBody().front()) {
    auto tile = mlir::dyn_cast<TileProgramOp>(operation);
    if (!tile) {
      if (!isCardSharedDeclaration(operation))
        return emitOpError("body may contain only card-shared declarations and "
                           "wafer.tile.program operations");
      continue;
    }
    PhysicalTileId tileId(tile.getTileIdAttr().getInt());
    if (mlir::failed(verifyTileId(getOperation(), *topology, cardId, tileId)))
      return mlir::failure();
    if (mlir::failed(verifyProgramPeerDomain(tile, *topology, cardId)))
      return mlir::failure();
    if (!seenTileIds.insert(tileId.getValue()).second)
      return emitOpError("tile_id must be unique in a card program; duplicate ")
             << tileId.getValue();
  }

  std::optional<llvm::ArrayRef<PhysicalTileId>> available =
      topology->getAvailableTileIds(cardId);
  if (!available)
    return emitOpError("cannot derive available Tiles for card_id ")
           << cardId.getValue();
  for (PhysicalTileId tileId : *available) {
    if (!seenTileIds.contains(tileId.getValue()))
      return emitOpError("is missing available tile_id ") << tileId.getValue();
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

  if (!mlir::isa<mlir::ModuleOp>(card->getParentOp()))
    return emitOpError(
        "requires its wafer.card.program to be directly nested under a "
        "builtin.module");

  // The parent CardProgram verifier owns topology construction and checks the
  // complete child domain once. Keep this leaf verifier local so verifying a
  // card does not rebuild identical module-wide topology for every Tile.
  if (getTileIdAttr().getInt() < 0)
    return emitOpError("tile_id must be non-negative");

  if (mlir::failed(verifyOnlyIdentifierAttribute(*this, getTileIdAttrName())))
    return mlir::failure();
  return mlir::success();
}
