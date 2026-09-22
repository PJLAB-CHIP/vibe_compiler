//===- PhysicalBoundaryViews.cpp - Preserve blocked DDR view roots --------===//

#include "BoundaryMovement.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {

void materializeBlockedDDRBoundaryViews(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  module.walk([&](TileRegionOp region) {
    const unsigned inputCount = region.getInputs().size();
    for (unsigned index = 0; index < inputCount; ++index) {
      mlir::Value input = region.getInputs()[index];
      auto type = mlir::dyn_cast<mlir::MemRefType>(input.getType());
      auto memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
      if (!memory || memory.getSpace() != MemorySpace::DDR ||
          (memory.getLayout() != MemLayout::Cx &&
           memory.getLayout() != MemLayout::NCx))
        continue;

      llvm::SmallVector<mlir::memref::SubViewOp> views;
      mlir::Value root = input;
      while (auto view = root.getDefiningOp<mlir::memref::SubViewOp>()) {
        views.push_back(view);
        root = view.getSource();
      }
      if (views.empty())
        continue;

      // A subview type alone does not retain a blocked root's global shape.
      // Bind the actual root and reproduce its existing SSA view chain inside
      // the isolated region. No data movement or allocation is introduced.
      auto argument = region.getBody().front().getArgument(index);
      llvm::SmallVector<mlir::OpOperand *> uses;
      for (mlir::OpOperand &use : argument.getUses())
        uses.push_back(&use);
      rewriter.modifyOpInPlace(region, [&] {
        region.getInputsMutable()[index].set(root);
        argument.setType(root.getType());
      });
      mlir::IRMapping mapping;
      mapping.map(root, argument);
      rewriter.setInsertionPointToStart(&region.getBody().front());
      for (auto view : llvm::reverse(views)) {
        for (mlir::Value operand : view->getOperands()) {
          if (mapping.contains(operand))
            continue;
          auto existing = llvm::find(region.getInputs(), operand);
          mlir::BlockArgument local;
          if (existing != region.getInputs().end()) {
            local = region.getBody().front().getArgument(
                std::distance(region.getInputs().begin(), existing));
          } else {
            rewriter.modifyOpInPlace(region, [&] {
              region.getInputsMutable().append(operand);
              local = region.getBody().front().addArgument(operand.getType(),
                                                           operand.getLoc());
            });
          }
          mapping.map(operand, local);
        }
        rewriter.clone(*view, mapping);
      }
      mlir::Value localView = mapping.lookup(input);
      for (mlir::OpOperand *use : uses)
        rewriter.modifyOpInPlace(use->getOwner(), [&] { use->set(localView); });
      for (auto view : views)
        if (view->use_empty())
          rewriter.eraseOp(view);
    }
  });
}

} // namespace wafer::compiler::detail
