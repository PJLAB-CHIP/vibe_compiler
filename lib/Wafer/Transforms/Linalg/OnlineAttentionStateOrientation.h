//===- OnlineAttentionStateOrientation.h - Private state reindexing ------===//
#ifndef WAFER_TRANSFORMS_LINALG_ONLINEATTENTIONSTATEORIENTATION_H
#define WAFER_TRANSFORMS_LINALG_ONLINEATTENTIONSTATEORIENTATION_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "mlir/IR/PatternMatch.h"

namespace wafer::compiler::detail {
mlir::LogicalResult orientOnlineAttentionAccumulators(
    mlir::ModuleOp module, const StructuredMaterializationRelations &relations,
    mlir::IRRewriter &rewriter);
} // namespace wafer::compiler::detail
#endif
