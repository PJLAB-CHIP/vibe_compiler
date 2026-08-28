//===- TargetTopology.h - Verifier-grade target topology view -*- C++
//-*-===//

#ifndef WAFER_IR_TOPOLOGY_TARGETTOPOLOGY_H
#define WAFER_IR_TOPOLOGY_TARGETTOPOLOGY_H

#include "Wafer/Target/Core/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace wafer {

struct CardCoordinate {
  int64_t y = 0;
  int64_t x = 0;

  friend bool operator==(CardCoordinate lhs, CardCoordinate rhs) {
    return lhs.y == rhs.y && lhs.x == rhs.x;
  }
};

struct TileCoordinate {
  int64_t y = 0;
  int64_t x = 0;

  friend bool operator==(TileCoordinate lhs, TileCoordinate rhs) {
    return lhs.y == rhs.y && lhs.x == rhs.x;
  }
};

/// One directed, on-card Tile adjacency link.
struct TileLink {
  TileId source;
  TileId destination;

  friend bool operator==(const TileLink &lhs, const TileLink &rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
};

/// Immutable target-topology facts derived solely from the unique direct
/// wafer.target.topology in a module. wafer.execution.mesh is deliberately
/// outside this query: card partitioning and Tile topology are
/// separate domains.
///
/// Card and Tile IDs are stable row-major IDs in their full grids.
/// Unavailable Tiles remain addressable as IDs/coordinates but are omitted
/// from availability, adjacency and path queries; they are never renumbered.
class TargetTopology {
public:
  static mlir::FailureOr<TargetTopology>
  create(mlir::ModuleOp module, std::string *failureReason = nullptr);

  int64_t getCardCount() const { return cardCount; }
  int64_t getTilesPerCard() const { return tilesPerCard; }
  llvm::ArrayRef<int64_t> getCardGrid() const { return cardGrid; }
  llvm::ArrayRef<int64_t> getTileGrid() const { return tileGrid; }

  std::optional<CardCoordinate> getCardCoordinate(CardId cardId) const;
  std::optional<CardId> getCardId(CardCoordinate coordinate) const;

  std::optional<TileCoordinate> getTileCoordinate(TileId tileId) const;
  std::optional<TileId> getTileId(TileCoordinate coordinate) const;

  std::optional<llvm::ArrayRef<TileId>>
  getAvailableTileIds(CardId cardId) const;
  bool isTileAvailable(CardId cardId, TileId tileId) const;

  /// Returns available Manhattan-neighbor Tiles on the same card, ordered by
  /// stable tile_id. Invalid or unavailable endpoints fail.
  mlir::FailureOr<llvm::SmallVector<TileId, 4>>
  getOnCardNeighbors(CardId cardId, TileId tileId) const;

  bool areOnCardAdjacent(CardId cardId, TileId source,
                         TileId destination) const;

  /// Returns an unweighted shortest-hop distance through available on-card
  /// Tile links. Invalid, unavailable or disconnected queries return nullopt.
  std::optional<uint64_t>
  getOnCardShortestHopDistance(CardId cardId, TileId source,
                               TileId destination) const;

  /// Returns a deterministic shortest on-card path. Equal-length alternatives
  /// use ascending tile_id during BFS. The route is a modeling fact,
  /// not a target routing or congestion claim.
  mlir::FailureOr<llvm::SmallVector<TileLink, 8>>
  getCanonicalOnCardPath(CardId cardId, TileId source,
                         TileId destination) const;

private:
  std::optional<size_t> getCardIndex(CardId cardId) const;
  std::optional<size_t> getTileIndex(TileId tileId) const;
  size_t getEndpointIndex(size_t cardIndex, size_t tileIndex) const;

  std::array<int64_t, 2> cardGrid{};
  std::array<int64_t, 2> tileGrid{};
  int64_t cardCount = 0;
  int64_t tilesPerCard = 0;
  llvm::SmallVector<unsigned char, 64> endpointAvailability;
  llvm::SmallVector<llvm::SmallVector<TileId, 16>, 4> availableTilesByCard;
};

} // namespace wafer

#endif // WAFER_IR_TOPOLOGY_TARGETTOPOLOGY_H
