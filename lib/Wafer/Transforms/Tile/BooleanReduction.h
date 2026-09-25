//===- BooleanReduction.h - Exact boolean reduction representation -------===//

#ifndef WAFER_TRANSFORMS_TILE_BOOLEANREDUCTION_H
#define WAFER_TRANSFORMS_TILE_BOOLEANREDUCTION_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail {
mlir::LogicalResult lowerBooleanReductions(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    mlir::RewriterBase::Listener *externalListener = nullptr);
} // namespace wafer::compiler::detail

#endif
