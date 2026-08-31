//===- StructuredToTile.h - Lower current structured compute -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_STRUCTUREDTOTILE_H
#define WAFER_TRANSFORMS_TILE_STRUCTUREDTOTILE_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

enum class StructuredToTileFailureKind : uint8_t {
  None,
  Unsupported,
  BrokenContract,
  CompilerFailure,
};

struct StructuredToTileStatistics {
  uint64_t fills = 0;
  uint64_t contractions = 0;
  uint64_t convolutions = 0;
  uint64_t reductions = 0;
  uint64_t elementwiseExpressions = 0;
  uint64_t elementwiseOperations = 0;
  uint64_t converts = 0;
  uint64_t passthroughMovements = 0;
};

struct StructuredToTileResult {
  StructuredToTileFailureKind failure = StructuredToTileFailureKind::None;
  StructuredToTileStatistics statistics;
  std::string detail;

  bool succeeded() const {
    return failure == StructuredToTileFailureKind::None;
  }
};

/// Deterministically lowers all executable Linalg operations nested in
/// TileRegions to existing typed wafer.tile compute and movement operations.
/// The transformation reads only the current memref operands, indexing maps,
/// scalar regions and DPS destinations. It does not select placement, fusion,
/// layout, movement route, execution structure, worker or completion.
StructuredToTileResult
lowerStructuredComputeToTile(mlir::ModuleOp module,
                             StructuredMaterializationRelations &relations);

/// Verifies the output seam of lowerStructuredComputeToTile. No executable
/// Linalg operation may remain in a TileRegion; every shaped Tile dataflow
/// operand/result must be an SPM Wafer memref.
mlir::LogicalResult verifyStructuredComputeLowered(mlir::ModuleOp module);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_STRUCTUREDTOTILE_H
