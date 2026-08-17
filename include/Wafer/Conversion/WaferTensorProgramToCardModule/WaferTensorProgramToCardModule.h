//===- WaferTensorProgramToCardModule.h - Card MPMD lowering -*- C++
//-*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOCARDMODULE_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOCARDMODULE_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"
#include "Wafer/Target/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace wafer {

/// Query-local spatial placement of one observable result.  The result domain
/// is divided in `activeTileIds` order along `shardDimension`.
struct OutputTileMapping {
  unsigned outputIndex = 0;
  unsigned shardDimension = 0;
  llvm::SmallVector<TileId, 16> activeTileIds;
  /// Static per-dimension temporal tile selected for this output.  Card
  /// materialization intersects it with each balanced spatial shard, so a
  /// smaller boundary shard becomes an ordinary finite tail class.  This is
  /// query-local input and is expressed by actual loops in the resulting IR.
  llvm::SmallVector<int64_t, 4> temporalTileSizes;
};

/// Complete physical placement consumed atomically by CardModule
/// materialization.  Every function result appears exactly once.  Different
/// results may use different dimensions and disjoint Tile sets; Tiles not
/// present in any output mapping receive a verifier-legal no-work entry.  This
/// object is never persisted in IR.
struct TileMapping {
  /// Explicit controller-owned lowering contract.  This query-local enum is
  /// consumed directly by CardModule materialization; it is not serialized
  /// into selected IR and is never recovered from edge actions or names.
  SpatialDataflowMaterializationMode materializationMode =
      SpatialDataflowMaterializationMode::JointDataflow;
  llvm::SmallVector<OutputTileMapping, 4> outputs;
  /// Per-structured-operation iterator tiles selected jointly with placement.
  /// Every scheduled source operation appears exactly once.  Reduction
  /// iterator sizes become typed accumulator traversal in the materialized
  /// Tile modules; parallel iterator sizes bound compact output traversal,
  /// while exact operand demand remains derived from consumer/result indexing
  /// relations.
  llvm::SmallVector<StructuredOpTemporalTile, 16> operationTemporalTiles;
  /// One exact action for every selected current-SSA producer/consumer edge.
  /// Demand, locality and remote fragments intentionally share this single
  /// carrier so CardModule materialization cannot observe a partial plan.
  llvm::SmallVector<SpatialEdgeStrategy, 16> edgeStrategies;
};

/// Materializes one explicit spatial mapping for a card-local structured
/// tensor program.
///
/// The input must contain one verified single-block functional tensor program,
/// one
/// physical target topology and one single-partition logical execution mesh.
/// Every static output domain is divided into balanced, nonempty contiguous
/// shards over the Tiles selected for that output. Each active Tile
/// owns the ordinary TilingInterface-driven TileRegion realization of only
/// its selected output roots and actual fused producer closure. Each selected
/// output is traversed by its own structured temporal loop nest and finite
/// tail classes; different outputs need not share a temporal tile shape.
/// Remaining output domains on that Tile take a typed no-store path; Tiles
/// with no selected output own a defined no-work entry. Every Tile entry has
/// the same full-card input/result ABI. Source arguments are preserved exactly;
/// private scheduling destinations are introduced and consumed inside this
/// conversion, leaving compiler-owned result roots for Tile memory
/// planning and exact target output binding.
///
/// The result is one owning module containing module-scope target topology and
/// logical mesh, one wafer.card.module, card-shared declarations exactly once,
/// and all-and-only available wafer.tile.module bodies. No search decision or
/// mapping side data is persisted. The source and `cardModule` output are
/// unchanged on failure.
mlir::LogicalResult lowerTensorProgramToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    std::string *failureReason = nullptr,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    StructuredMaterializationRelations *materializationRelations = nullptr,
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>> observableOutputRootNodes =
        {});

/// Materializes exactly one Tile entry function for scoped feasibility
/// probing. The result module contains the module-scope target topology,
/// logical mesh and the single detached Tile entry; it deliberately builds no
/// CardModule/TileModule shell, no sibling Tile modules and no no-work
/// wrappers. A capacity rejection of this Tile also rejects the complete
/// mapping, while a passing module is discarded and followed by
/// `lowerTensorProgramToCardModule`.
mlir::LogicalResult lowerTensorProgramToTileModule(
    mlir::ModuleOp sourceModule, CardId cardId, TileId tileId,
    const TileMapping &mapping, mlir::OwningOpRef<mlir::ModuleOp> &tileModule,
    std::string *failureReason = nullptr,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    StructuredMaterializationRelations *materializationRelations = nullptr);

/// Materializes only the shard of one structured root on one participating
/// Tile for scoped feasibility probing: the result module carries the module
/// topology facts and a single detached Tile entry whose pull closure covers
/// exactly `targetRoot`'s output shards and its non-root support. Sibling
/// roots, other Tiles, CardModule/TileModule shells and no-work wrappers are
/// never created. `observableOutputRootNodes` is the DAG's per-function-result
/// nearest-root list used to select the shards owned by `targetRoot`.
mlir::LogicalResult lowerTensorProgramToTileRootShard(
    mlir::ModuleOp sourceModule, CardId cardId, TileId tileId,
    const TileMapping &mapping, uint32_t targetRoot,
    mlir::OwningOpRef<mlir::ModuleOp> &tileModule,
    std::string *failureReason = nullptr,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>> observableOutputRootNodes =
        {},
    StructuredMaterializationRelations *materializationRelations = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOCARDMODULE_H
