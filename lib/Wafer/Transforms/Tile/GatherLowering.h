//===- GatherLowering.h - Selected tensor gather materialization -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_GATHERLOWERING_H
#define WAFER_TRANSFORMS_TILE_GATHERLOWERING_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail {
mlir::LogicalResult lowerTensorGathers(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations);
} // namespace wafer::compiler::detail

#endif
