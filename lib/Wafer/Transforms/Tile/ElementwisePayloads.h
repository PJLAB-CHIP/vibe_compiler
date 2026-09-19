//===- ElementwisePayloads.h - Expose target expression SSA -------*- C++
//-*-===//

#ifndef WAFER_TRANSFORMS_TILE_ELEMENTWISEPAYLOADS_H
#define WAFER_TRANSFORMS_TILE_ELEMENTWISEPAYLOADS_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail {
/// Materialize independently executed pointwise payload steps before layout
/// selection. Projected values retain their compact dependency domain.
mlir::LogicalResult
materializeElementwisePayloads(mlir::ModuleOp module,
                               StructuredMaterializationRelations &relations);
/// Reject a layout query that bypassed target payload materialization.
bool hasUnmaterializedElementwisePayloads(mlir::ModuleOp module);
} // namespace wafer::compiler::detail

#endif
