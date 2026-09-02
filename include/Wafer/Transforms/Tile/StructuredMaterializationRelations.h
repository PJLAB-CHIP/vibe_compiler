//===- StructuredMaterializationRelations.h - Current IR ownership -*- C++
//-*-===//

#ifndef WAFER_TRANSFORMS_TILE_STRUCTUREDMATERIALIZATIONRELATIONS_H
#define WAFER_TRANSFORMS_TILE_STRUCTUREDMATERIALIZATIONRELATIONS_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer {

inline constexpr char kWaferCrossTileBoundaryInputAttrName[] =
    "wafer.cross_tile_boundary_input";

enum class MaterializedBufferRole : uint8_t {
  Operand,
  Result,
  Scratch,
  Movement,
};

/// One actual current-IR operation/buffer ownership edge.  Both endpoints are
/// live handles in the same candidate epoch.  This relation never refers back
/// to a source structured node and is discarded after the actual memory gate.
struct MaterializedBufferRelation {
  mlir::Operation *owner = nullptr;
  mlir::Value buffer;
  MaterializedBufferRole role = MaterializedBufferRole::Operand;
};

/// One observable TensorProgram result piece that already exists as a current
/// structural tensor endpoint. Layout/bufferization retargets this relation to
/// the corresponding actual destination before it is converted into the
/// buffer-level output relation above.
struct StructuredOutputRelation {
  unsigned outputIndex = 0;
  mlir::Value endpoint;
};

/// One selected external Region binding whose two current tensor endpoints
/// already exist in the candidate IR. Same-Tile endpoints remain directly
/// connected by SSA; cross-Tile endpoints cannot be joined by SSA because
/// TileModule is IsolatedFromAbove. Movement consumes this exact relation after
/// layout/bufferization. No route, buffer, storage, event or completion fact is
/// represented here.
struct StructuredBoundaryRelation {
  mlir::Value sourceEndpoint;
  mlir::Value destinationEndpoint;
};

/// Actual operation/buffer ownership recorded during one candidate rewrite.
/// Every pointer/value is valid only for the owning current IR epoch.
struct StructuredMaterializationRelations {
  llvm::SmallVector<MaterializedBufferRelation, 32> buffers;
  llvm::SmallVector<StructuredOutputRelation, 8> structuralOutputs;
  llvm::SmallVector<StructuredBoundaryRelation, 8> boundaryRelations;
};

} // namespace wafer

#endif // WAFER_TRANSFORMS_TILE_STRUCTUREDMATERIALIZATIONRELATIONS_H
