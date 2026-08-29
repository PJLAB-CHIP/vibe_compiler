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
#include "llvm/ADT/SmallVector.h"

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

/// Actual TileRegion selected for one or more current Region executions. This
/// relation is consumed by compact tiling/selected-attention lowering and is
/// invalidated if the TileRegion is replaced; it does not describe future
/// operations inside the region.
struct SpatialRegionExecutionRelation {
  compiler::detail::RegionExecutionId execution;
  TileId tile{0};
  TileRegionOp region;
};

/// Move-only actual result of one selected Spatial/Region transformation.
/// The source TensorProgram remains unchanged. Every relation refers only to
/// values or operations owned by `module` and must be retargeted with the same
/// rewrite transaction that changes them.
struct SpatialRegionMaterializationResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
  llvm::SmallVector<SpatialRegionExecutionRelation, 16> regionExecutions;
};

/// Materializes one closed Spatial/Region choice into all-and-only TileModules
/// and non-nested structural TileRegions. This operation does not enumerate
/// choices and does not generate temporal loops, fusion, attention
/// decomposition, layout, buffers, movement, Instr or completion.
mlir::FailureOr<SpatialRegionMaterializationResult> materializeSpatialRegions(
    mlir::ModuleOp source, CardId cardId, llvm::ArrayRef<TileId> availableTiles,
    llvm::ArrayRef<compiler::detail::StructuredOperationNodeMapping>
        operationNodes,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const compiler::detail::RegionPlan &regionPlan,
    SpatialRegionMaterializationFailure *failure = nullptr);

} // namespace wafer

#endif // WAFER_TRANSFORMS_LINALG_SPATIALREGIONMATERIALIZATION_H
