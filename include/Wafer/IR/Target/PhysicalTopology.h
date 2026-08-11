//===- PhysicalTopology.h - Verifier-grade physical topology view -*- C++
//-*-===//

#ifndef WAFER_IR_TARGET_PHYSICALTOPOLOGY_H
#define WAFER_IR_TARGET_PHYSICALTOPOLOGY_H

#include "Wafer/Target/PhysicalIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace wafer {

struct PhysicalCardCoordinate {
  int64_t y = 0;
  int64_t x = 0;

  friend bool operator==(PhysicalCardCoordinate lhs,
                         PhysicalCardCoordinate rhs) {
    return lhs.y == rhs.y && lhs.x == rhs.x;
  }
};

struct PhysicalTileCoordinate {
  int64_t y = 0;
  int64_t x = 0;

  friend bool operator==(PhysicalTileCoordinate lhs,
                         PhysicalTileCoordinate rhs) {
    return lhs.y == rhs.y && lhs.x == rhs.x;
  }
};

/// One directed, on-card physical Tile adjacency link.
struct PhysicalTileDirectedLink {
  PhysicalTileId source;
  PhysicalTileId destination;

  friend bool operator==(const PhysicalTileDirectedLink &lhs,
                         const PhysicalTileDirectedLink &rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
};

/// Immutable physical topology facts derived solely from the unique direct
/// wafer.target.topology in a module. wafer.execution.mesh is deliberately
/// outside this query: card partitioning and physical Tile topology are
/// separate domains.
///
/// Card and Tile IDs are stable row-major IDs in their full grids.
/// Unavailable Tiles remain addressable as IDs/coordinates but are omitted
/// from availability, adjacency and path queries; they are never renumbered.
class PhysicalTopology {
public:
  static mlir::FailureOr<PhysicalTopology>
  create(mlir::ModuleOp module, std::string *failureReason = nullptr);

  int64_t getCardCount() const { return cardCount; }
  int64_t getTilesPerCard() const { return tilesPerCard; }
  llvm::ArrayRef<int64_t> getCardGrid() const { return cardGrid; }
  llvm::ArrayRef<int64_t> getTileGrid() const { return tileGrid; }

  std::optional<PhysicalCardCoordinate>
  getCardCoordinate(PhysicalCardId cardId) const;
  std::optional<PhysicalCardId>
  getCardId(PhysicalCardCoordinate coordinate) const;

  std::optional<PhysicalTileCoordinate>
  getTileCoordinate(PhysicalTileId tileId) const;
  std::optional<PhysicalTileId>
  getTileId(PhysicalTileCoordinate coordinate) const;

  std::optional<llvm::ArrayRef<PhysicalTileId>>
  getAvailableTileIds(PhysicalCardId cardId) const;
  bool isTileAvailable(PhysicalCardId cardId, PhysicalTileId tileId) const;

  /// Returns available Manhattan-neighbor Tiles on the same card, ordered by
  /// stable physical tile_id. Invalid or unavailable endpoints fail.
  mlir::FailureOr<llvm::SmallVector<PhysicalTileId, 4>>
  getOnCardNeighbors(PhysicalCardId cardId, PhysicalTileId tileId) const;

  bool areOnCardAdjacent(PhysicalCardId cardId, PhysicalTileId source,
                         PhysicalTileId destination) const;

  /// Returns an unweighted shortest-hop distance through available on-card
  /// Tile links. Invalid, unavailable or disconnected queries return nullopt.
  std::optional<uint64_t>
  getOnCardShortestHopDistance(PhysicalCardId cardId, PhysicalTileId source,
                               PhysicalTileId destination) const;

  /// Returns a deterministic shortest on-card path. Equal-length alternatives
  /// use ascending physical tile_id during BFS. The route is a modeling fact,
  /// not a target routing or congestion claim.
  mlir::FailureOr<llvm::SmallVector<PhysicalTileDirectedLink, 8>>
  getCanonicalOnCardPath(PhysicalCardId cardId, PhysicalTileId source,
                         PhysicalTileId destination) const;

private:
  std::optional<size_t> getCardIndex(PhysicalCardId cardId) const;
  std::optional<size_t> getTileIndex(PhysicalTileId tileId) const;
  size_t getEndpointIndex(size_t cardIndex, size_t tileIndex) const;

  std::array<int64_t, 2> cardGrid{};
  std::array<int64_t, 2> tileGrid{};
  int64_t cardCount = 0;
  int64_t tilesPerCard = 0;
  llvm::SmallVector<unsigned char, 64> endpointAvailability;
  llvm::SmallVector<llvm::SmallVector<PhysicalTileId, 16>, 4>
      availableTilesByCard;
};

} // namespace wafer

#endif // WAFER_IR_TARGET_PHYSICALTOPOLOGY_H
