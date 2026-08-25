//===- CoupledTileRegion.h - Selected structured groups -*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_COUPLEDTILEREGION_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_COUPLEDTILEREGION_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/SingleRootTileRegion.h"
#include "Wafer/IR/WaferDialect.h"

namespace wafer {

struct StructuredNodeTemporalTile {
  uint32_t structuredNodeId = 0;
  llvm::SmallVector<int64_t, 4> iteratorTileSizes;
  llvm::SmallVector<uint32_t, 4> waveLoopOrder;
};

/// Candidate-local node form of a parent-dependent nested temporal class.
/// One class may carry several entries when several exact consumer uses share
/// the same producer execution and plan.
struct StructuredNodeNestedTemporalTile {
  uint32_t producerNodeId = 0;
  uint32_t parentNodeId = 0;
  unsigned producerResult = 0;
  unsigned parentOperand = 0;
  llvm::SmallVector<int64_t, 4> requestedResultExtents;
  llvm::SmallVector<int64_t, 4> producerIterationExtents;
  llvm::SmallVector<int64_t, 4> iteratorTileSizes;
  llvm::SmallVector<uint32_t, 4> waveLoopOrder;
};

/// Exact selected SSA edge inside one candidate-local group. Replica
/// producers use this relation to rewire only the consumer occurrence in this
/// group; the shared immutable source graph is never globally rewired.
struct StructuredNodeLocalUse {
  uint32_t producerNodeId = 0;
  uint32_t consumerNodeId = 0;
  unsigned producerResult = 0;
  unsigned consumerOperand = 0;
};

/// One selected primary physical version for every shaped operand/result of a
/// structured node in this Tile shard. Scalar entries are std::nullopt. The
/// conversion may create target-required derived versions explicitly, but
/// subsequent consumers observe only the selected primary result version.
struct StructuredNodePhysicalRepresentation {
  uint32_t structuredNodeId = 0;
  /// Selected decompositions may have target-internal operands whose natural
  /// layout is fixed by the direct typed compute builder rather than by a
  /// RegionValueVersionId. When true, operand entries remain null and the
  /// emitter preserves those natural operand buffers while still applying the
  /// selected result versions below.
  bool preserveNaturalOperands = false;
  llvm::SmallVector<std::optional<MemLayout>, 4> operandLayouts;
  llvm::SmallVector<uint8_t, 4> sharedOperands;
  llvm::SmallVector<std::optional<MemLayout>, 2> resultLayouts;
};

struct StructuredNodeComputeImplementation {
  uint32_t structuredNodeId = 0;
  StructuredComputeImplementation implementation =
      StructuredComputeImplementation::Natural;
};

/// One already-selected group of node shards that must share one TileRegion.
/// Every shard names the same physical Tile and a distinct structured node.
/// Singleton groups are the ordinary single-root representation.
struct StructuredNodeShardGroup {
  llvm::SmallVector<StructuredNodeIterationShard, 4> shards;
  /// Empty before temporal selection. A complete selected group carries
  /// exactly one entry for every shard/node.
  llvm::SmallVector<StructuredNodeTemporalTile, 4> temporalTiles;
  /// Parent-dependent plans for consumer-nested executions. Absolute wave
  /// offsets remain actual SSA facts and are not duplicated here.
  llvm::SmallVector<StructuredNodeNestedTemporalTile, 4> nestedTemporalTiles;
  llvm::SmallVector<StructuredNodeLocalUse, 4> localUses;
  /// Empty before physical-representation selection. A complete selected
  /// group carries exactly one entry for every shard/node.
  llvm::SmallVector<StructuredNodePhysicalRepresentation, 4> representations;
  /// Empty uses the conservative natural lowering. A complete selected group
  /// carries exactly one implementation entry for every shard/node.
  llvm::SmallVector<StructuredNodeComputeImplementation, 4> implementations;
  /// Pure structured producers selected for consumer-local recomputation.
  /// They enter the closure but are not scheduled-node/emission identities.
  llvm::SmallVector<uint32_t, 2> recomputedProducerNodes;
  /// Exact iterator work for recomputed producers on this group's Tile.
  /// Entries are keyed by `structuredNodeId`; they do not add mandatory
  /// source executions or observable results.
  llvm::SmallVector<StructuredNodeIterationShard, 2> recomputedProducerShards;
  /// Selected scheduled nodes whose result tiles must be materialized once
  /// before consumer traversal. Consumer slices reuse the exact cached tile;
  /// scheduled nodes omitted here remain consumer-nested. This is a
  /// query-local construction directive, not persisted IR metadata.
  llvm::SmallVector<uint32_t, 2> independentlyMaterializedNodes;
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
