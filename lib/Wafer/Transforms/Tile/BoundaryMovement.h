//===- BoundaryMovement.h - Close physical Tile boundaries -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_BOUNDARYMOVEMENT_H
#define WAFER_TRANSFORMS_TILE_BOUNDARYMOVEMENT_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

enum class BoundaryMovementFailureKind : uint8_t {
  None,
  Unsupported,
  BrokenContract,
  CompilerFailure,
};

struct BoundaryMovementStatistics {
  uint64_t ddrLoads = 0;
  uint64_t ddrStores = 0;
  uint64_t peerSends = 0;
  uint64_t peerReceives = 0;
  uint64_t peerWaits = 0;
  uint64_t interRegionDDRStages = 0;
  uint64_t outputCopiesRemoved = 0;
  uint64_t tensorBridgesRemoved = 0;
};

struct BoundaryMovementResult {
  BoundaryMovementFailureKind failure = BoundaryMovementFailureKind::None;
  BoundaryMovementStatistics statistics;
  std::string detail;

  bool succeeded() const {
    return failure == BoundaryMovementFailureKind::None;
  }
};

/// Converts logical TileRegion tensor boundaries to actual physical movement.
/// External and same-Tile inter-region values use explicit DDR load/store;
/// cross-Tile endpoint relations become matching peer send/recv tokens and
/// waits. The supplied current endpoint relations are consumed exactly once.
/// This transformation does not choose layout, execution structure, worker,
/// NCC completion or memory offsets.
BoundaryMovementResult
materializeTileBoundaryMovement(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations);

/// Verifies that TileRegion shaped boundaries are Wafer DDR memrefs, no
/// tensor/memref bridge remains, and no SPM root crosses a TileRegion.
mlir::LogicalResult verifyPhysicalTileDataflow(mlir::ModuleOp module);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_BOUNDARYMOVEMENT_H
