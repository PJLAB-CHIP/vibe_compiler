//===- CompactLayoutAssignment.cpp - Layout materialization cleanup -------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer {
namespace {

static bool isInverseMaterializationPair(wafer::LayoutMaterializeOp outer) {
  auto inner = outer.getSource().getDefiningOp<wafer::LayoutMaterializeOp>();
  if (!inner || !inner->hasOneUse())
    return false;
  return inner.getSource().getType() == outer.getResult().getType();
}

static bool cleanupInversePairs(mlir::ModuleOp module) {
  bool changed = false;

  while (true) {
    llvm::SmallVector<wafer::LayoutMaterializeOp> inversePairs;
    module.walk([&](wafer::LayoutMaterializeOp materialize) {
      if (isInverseMaterializationPair(materialize))
        inversePairs.push_back(materialize);
    });

    if (inversePairs.empty())
      break;

    changed = true;
    for (wafer::LayoutMaterializeOp outer : inversePairs) {
      auto inner =
          outer.getSource().getDefiningOp<wafer::LayoutMaterializeOp>();
      if (!inner || !inner->hasOneUse())
        continue;
      mlir::Value original = inner.getSource();
      outer.getResult().replaceAllUsesWith(original);
      outer.erase();
      inner.erase();
    }
  }

  return changed;
}

struct CompactLayoutAssignmentPass
    : public mlir::PassWrapper<CompactLayoutAssignmentPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompactLayoutAssignmentPass)

  llvm::StringRef getArgument() const final {
    return "wafer-compact-layout-assignment";
  }

  llvm::StringRef getDescription() const final {
    return "remove redundant Wafer layout materialization pairs";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<wafer::WaferDialect>();
  }

  void runOnOperation() final { (void)cleanupInversePairs(getOperation()); }
};

} // namespace

std::unique_ptr<mlir::Pass> createCompactLayoutAssignmentPass() {
  return std::make_unique<CompactLayoutAssignmentPass>();
}

} // namespace wafer
