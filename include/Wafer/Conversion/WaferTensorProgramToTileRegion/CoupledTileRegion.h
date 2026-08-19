//===- CoupledTileRegion.h - Selected structured groups -*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_COUPLEDTILEREGION_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_COUPLEDTILEREGION_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/SingleRootTileRegion.h"

namespace wafer {

struct StructuredNodeTemporalTile {
  uint32_t structuredNodeId = 0;
  llvm::SmallVector<int64_t, 4> iteratorTileSizes;
  llvm::SmallVector<uint32_t, 4> waveLoopOrder;
};

/// One already-selected group of node shards that must share one TileRegion.
/// Every shard names the same physical Tile and a distinct structured node.
/// Singleton groups are the ordinary single-root representation.
struct StructuredNodeShardGroup {
  llvm::SmallVector<StructuredNodeIterationShard, 4> shards;
  /// Empty before temporal selection. A complete selected group carries
  /// exactly one entry for every shard/node.
  llvm::SmallVector<StructuredNodeTemporalTile, 4> temporalTiles;
};

/// Materializes a complete explicit node-shard partition. Multi-node groups
/// use one consumer-driven traversal and keep internal producer values in the
/// common region; values crossing groups remain typed DDR function boundaries.
/// This API performs no grouping, enumeration, or winner selection.
mlir::LogicalResult lowerStructuredNodeGroupsToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<TileId> availableTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<StructuredNodeShardGroup> groups,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    StructuredMaterializationRelations *materializationRelations = nullptr,
    std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_COUPLEDTILEREGION_H
