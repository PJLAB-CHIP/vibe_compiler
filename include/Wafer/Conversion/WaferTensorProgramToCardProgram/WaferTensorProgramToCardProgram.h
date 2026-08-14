//===- WaferTensorProgramToCardProgram.h - Card MPMD lowering -*- C++
//-*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOCARDPROGRAM_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOCARDPROGRAM_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"
#include "Wafer/Target/PhysicalIds.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace wafer {

/// Query-local spatial placement of one observable result.  The result domain
/// is divided in `activeTileIds` order along `shardDimension`.
struct CardOutputSpatialMapping {
  unsigned outputIndex = 0;
  unsigned shardDimension = 0;
  llvm::SmallVector<PhysicalTileId, 16> activeTileIds;
  /// Static per-dimension temporal tile selected for this output.  Card
  /// materialization intersects it with each balanced spatial shard, so a
  /// smaller boundary shard becomes an ordinary finite tail class.  This is
  /// query-local input and is expressed by actual loops in the resulting IR.
  llvm::SmallVector<int64_t, 4> temporalTileSizes;
};

/// Complete physical placement consumed atomically by CardProgram
/// materialization.  Every function result appears exactly once.  Different
/// results may use different dimensions and disjoint Tile sets; Tiles not
/// present in any output mapping receive a verifier-legal no-work entry.  This
/// object is never persisted in IR.
struct CardSpatialMapping {
  /// Explicit controller-owned lowering contract.  This query-local enum is
  /// consumed directly by CardProgram materialization; it is not serialized
  /// into selected IR and is never recovered from edge actions or names.
  SpatialDataflowMaterializationMode materializationMode =
      SpatialDataflowMaterializationMode::JointDataflow;
  llvm::SmallVector<CardOutputSpatialMapping, 4> outputs;
  /// Per-structured-operation iterator tiles selected jointly with placement.
  /// Every scheduled source operation appears exactly once.  Reduction
  /// iterator sizes become typed accumulator traversal in the materialized
  /// Tile programs; parallel iterator sizes bound compact output traversal,
  /// while exact operand demand remains derived from consumer/result indexing
  /// relations.
  llvm::SmallVector<StructuredOpTemporalTile, 16> operationTemporalTiles;
  /// One exact action for every selected current-SSA producer/consumer edge.
  /// Demand, locality and remote fragments intentionally share this single
  /// carrier so CardProgram materialization cannot observe a partial plan.
  llvm::SmallVector<SpatialEdgeStrategy, 16> edgeStrategies;
};

/// Materializes one explicit spatial mapping for a card-local structured
/// tensor program.
///
/// The input must contain one verified single-block functional tensor program,
/// one
/// physical target topology and one single-partition logical execution mesh.
/// Every static output domain is divided into balanced, nonempty contiguous
/// shards over the physical Tiles selected for that output. Each active Tile
/// owns the ordinary TilingInterface-driven TileRegion realization of only
/// its selected output roots and actual fused producer closure. Each selected
/// output is traversed by its own structured temporal loop nest and finite
/// tail classes; different outputs need not share a temporal tile shape.
/// Remaining output domains on that Tile take a typed no-store path; Tiles
/// with no selected output own a defined no-work entry. Every Tile entry has
/// the same full-card input/result ABI. Source arguments are preserved exactly;
/// private scheduling destinations are introduced and consumed inside this
/// conversion, leaving compiler-owned result roots for physical Tile
/// finalization and exact target output binding.
///
/// The result is one owning module containing module-scope target topology and
/// logical mesh, one wafer.card.program, card-shared declarations exactly once,
/// and all-and-only available wafer.tile.program bodies. No search decision or
/// mapping side data is persisted. The source and `cardModule` output are
/// unchanged on failure.
mlir::LogicalResult lowerTensorProgramToCardProgram(
    mlir::ModuleOp sourceModule, PhysicalCardId cardId,
    const CardSpatialMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    std::string *failureReason = nullptr,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    StructuredMaterializationRelations *materializationRelations = nullptr);

/// Materializes the exact selected body of one physical Tile while emitting
/// verifier-legal no-work scaffolding for the other available Tiles.  This is
/// a query-local negative preflight only: rejection by the selected Tile is a
/// sufficient rejection of the whole candidate, while a passing result must
/// be discarded and followed by `lowerTensorProgramToCardProgram` for the
/// complete artifact.  No probe result may be selected or packaged.
mlir::LogicalResult lowerTensorProgramToCardProgramFailureProbe(
    mlir::ModuleOp sourceModule, PhysicalCardId cardId,
    PhysicalTileId probeTileId, const CardSpatialMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &probeCardModule,
    std::string *failureReason = nullptr,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    StructuredMaterializationRelations *materializationRelations = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOCARDPROGRAM_H
