//===- MovementFusion.cpp - Tile MovementFusion -------------------===//

#include "Wafer/Transforms/Tile/MovementFusion.h"
#include "Wafer/Analysis/Tile/MovementEndpoint.h"
#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace wafer::compiler::detail {
namespace {
bool hasMatchingBlockedOrder(mlir::Value ddr, mlir::MemRefType spm) {
  auto type = mlir::cast<mlir::MemRefType>(ddr.getType());
  auto memory = getWaferMemoryAttr(type);
  auto local = getWaferMemoryAttr(spm);
  auto blocked = [](MemLayout layout) {
    return layout == MemLayout::Cx || layout == MemLayout::NCx;
  };
  if (!memory || !local || memory.getSpace() != MemorySpace::DDR ||
      local.getSpace() != MemorySpace::SPM || !blocked(memory.getLayout()) ||
      !blocked(local.getLayout()) || !spm.getLayout().isIdentity() ||
      !spm.hasStaticShape() || type.getShape() != spm.getShape() ||
      type.getElementType() != spm.getElementType())
    return false;
  // NCx separates N from C blocks; Cx includes N in the spatial volume.
  // They have the same order for a single batch, including retained C tails.
  if (memory.getLayout() != local.getLayout() &&
      (spm.getRank() < 3 || spm.getDimSize(0) != 1))
    return false;
  auto endpoint = analysis::resolveMovementEndpoint(ddr);
  auto identity = analysis::IndexRelation::identity(spm.getShape());
  auto physical = computeWaferPhysicalTensorInfo(spm);
  if (mlir::failed(endpoint) || !identity.isExact() || !physical ||
      physical->bitPackedElement)
    return false;
  // This is a same-layer movement composition. The actual resulting load or
  // store still passes the unique descriptor/target gate in this candidate.
  return mlir::succeeded(analysis::TransferRealizability::proveMappedTransfer(
      endpoint->getType(), spm, spm.getShape(), endpoint->viewToBase,
      *identity.get()));
}
} // namespace

void fuseDMALayoutMovements(mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<LayoutMaterializeOp> layouts;
  module.walk([&](LayoutMaterializeOp layout) { layouts.push_back(layout); });
  for (auto layout : layouts) {
    auto allocation = layout.getSource().getDefiningOp<mlir::memref::AllocOp>();
    if (allocation && llvm::hasNItems(allocation.getResult().getUses(), 2)) {
      StorageLoadOp load;
      for (auto *user : allocation.getResult().getUsers())
        if (auto candidate = mlir::dyn_cast<StorageLoadOp>(user))
          if (candidate.getDest() == allocation.getResult())
            load = candidate;
      auto resultType =
          mlir::cast<mlir::MemRefType>(layout.getResult().getType());
      if (load && load->getBlock() == layout->getBlock() &&
          load->isBeforeInBlock(layout) &&
          hasMatchingBlockedOrder(load.getSource(), resultType)) {
        rewriter.setInsertionPoint(load);
        auto dest =
            rewriter.create<mlir::memref::AllocOp>(layout.getLoc(), resultType);
        rewriter.create<StorageLoadOp>(load.getLoc(), load.getSource(), dest);
        rewriter.replaceOp(layout, dest.getResult());
        rewriter.eraseOp(load);
        rewriter.eraseOp(allocation);
        continue;
      }
    }
    if (!layout.getResult().hasOneUse())
      continue;
    auto store =
        mlir::dyn_cast<StorageStoreOp>(*layout.getResult().getUsers().begin());
    if (!store || store.getSource() != layout.getResult() ||
        store->getBlock() != layout->getBlock() ||
        !layout->isBeforeInBlock(store) ||
        !hasMatchingBlockedOrder(
            store.getDest(),
            mlir::cast<mlir::MemRefType>(layout.getSource().getType())))
      continue;
    // Delaying the source read to the original store must preserve its
    // snapshot. Pure address arithmetic may intervene; effects may not.
    bool unchanged = true;
    for (auto *op = layout->getNextNode(); op != store; op = op->getNextNode())
      unchanged &= mlir::isMemoryEffectFree(op);
    if (!unchanged)
      continue;
    rewriter.modifyOpInPlace(
        store, [&] { store.getSourceMutable().assign(layout.getSource()); });
    rewriter.eraseOp(layout);
  }
}

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
