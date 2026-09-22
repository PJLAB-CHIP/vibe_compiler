//===- MovementFusion.cpp - Tile MovementFusion -------------------===//

#include "Wafer/Transforms/Tile/MovementFusion.h"
#include "Wafer/IR/WaferDialect.h"

namespace wafer::compiler::detail {
void fuseTransposeLayoutMovements(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations) {
  llvm::SmallVector<LayoutMaterializeOp> layouts;
  module.walk([&](LayoutMaterializeOp layout) { layouts.push_back(layout); });
  for (auto layout : layouts) {
    auto transpose = layout.getSource().getDefiningOp<MoveTransposeOp>();
    if (!transpose || !transpose.getResult().hasOneUse() ||
        transpose->getBlock() != layout->getBlock() ||
        pipelineOperations.contains(transpose) ||
        pipelineOperations.contains(layout))
      continue;
    rewriter.setInsertionPoint(transpose);
    auto fused = rewriter.create<MoveTransposeOp>(
        transpose.getLoc(), layout.getResult().getType(), transpose.getSource(),
        transpose.getPermutationAttr());
    rewriter.replaceOp(layout, fused.getResult());
    rewriter.eraseOp(transpose);
  }
}

} // namespace wafer::compiler::detail
