//===- SpatialRegionMaterialization.h - Selected structural IR -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_SPATIALREGIONMATERIALIZATION_H
#define WAFER_TRANSFORMS_LINALG_SPATIALREGIONMATERIALIZATION_H

#include "Wafer/Analysis/Linalg/StructuredDAGAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/RootRegionWork.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <string>

namespace wafer {

enum class SpatialRegionMaterializationFailureKind : uint8_t {
  None,
  Unsupported,
  BrokenContract,
  CompilerFailure,
};

struct SpatialRegionMaterializationFailure {
  SpatialRegionMaterializationFailureKind kind =
      SpatialRegionMaterializationFailureKind::None;
  std::string detail;
};

/// Move-only actual result of one selected Spatial/Region transformation.
/// The source TensorProgram remains unchanged. Every relation refers only to
/// values owned by `module`. At this boundary only observable outputs and
/// cross-Tile endpoint pairs are populated; plan-ID keyed attribution is empty.
struct SpatialRegionMaterializationResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
};

/// Materializes one closed Spatial/Region choice into all-and-only TileModules
/// and non-nested structural TileRegions. Graph attention is converted to
/// actual per-Tile online state and selected spatial merge/finalize, while
/// QK/PV decomposition remains downstream. This operation does not enumerate
/// choices or generate temporal loops, fusion, layout, buffers, movement,
/// Instr or completion.
mlir::FailureOr<SpatialRegionMaterializationResult> materializeSpatialRegions(
    mlir::ModuleOp source, CardId cardId, llvm::ArrayRef<TileId> availableTiles,
    llvm::ArrayRef<compiler::detail::StructuredOperationNodeMapping>
        operationNodes,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const compiler::detail::RegionPlan &regionPlan,
    SpatialRegionMaterializationFailure *failure = nullptr);

} // namespace wafer

#endif // WAFER_TRANSFORMS_LINALG_SPATIALREGIONMATERIALIZATION_H
