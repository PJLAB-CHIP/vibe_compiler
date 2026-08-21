//===- SingleRootTileRegion.h - Exact structured node regions -*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_SINGLEROOTTILEREGION_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_SINGLEROOTTILEREGION_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/Analysis/PhysicalDataflow/SpatialAssignment.h"
#include "Wafer/IR/Target/TargetTopology.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer {

/// One already-selected, nonempty rectangular shard of a structured node's
/// iteration domain. The rectangle is expressed in TilingInterface iterator
/// order. It contains no temporal, fusion, representation, movement, buffer,
/// schedule, or cost choice.
struct StructuredNodeIterationShard {
  uint32_t structuredNodeId = 0;
  TileId tile{0};
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  /// Per-output groups for which this shard is a partial contribution. Empty
  /// means the shard directly produces complete results. Each group carries
  /// its own merge Tile; there is no node-wide merge endpoint.
  llvm::SmallVector<compiler::detail::ReductionGroupPlacement, 2>
      reductionGroups;
};

/// Materializes selected structured-node shards as actual single-root
/// TileRegion functions under a complete Card/Tile ownership hierarchy.
///
/// Each request becomes exactly one private function in its selected
/// `wafer.tile.module`. Function shaped arguments/results are explicit DDR
/// boundary obligations: this stage deliberately does not select the later
/// physical movement that satisfies a cross-node boundary. Pure tensor
/// transforms in the requested root's backward closure are materialized in
/// the same region, while every other structured node is a hard boundary.
/// Empty Tiles are retained as verifier-legal empty Tile modules.
///
/// The source and output are unchanged on failure. The returned CardModule is
/// an intermediate actual-IR owner for subsequent region/fusion, temporal,
/// representation, movement, buffering and scheduling stages; it is not a
/// complete executable candidate by itself.
mlir::LogicalResult lowerStructuredNodeShardsToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<TileId> availableTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<StructuredNodeIterationShard> shards,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    StructuredMaterializationRelations *materializationRelations = nullptr,
    std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_SINGLEROOTTILEREGION_H
