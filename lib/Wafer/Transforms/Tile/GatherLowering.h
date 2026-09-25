//===- GatherLowering.h - Selected tensor gather materialization -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_GATHERLOWERING_H
#define WAFER_TRANSFORMS_TILE_GATHERLOWERING_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail {
mlir::LogicalResult
lowerTensorGathers(mlir::ModuleOp module,
                   StructuredMaterializationRelations &relations,
                   mlir::RewriterBase::Listener *externalListener = nullptr);
} // namespace wafer::compiler::detail

#endif
