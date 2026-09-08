//===- PromoteStablehloLogistic.cpp - Logistic opmath precision
//------------===//

#include "Wafer/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {

#define GEN_PASS_DEF_PROMOTESTABLEHLOLOGISTICPASS
#include "Wafer/Conversion/WaferConversionPasses.h.inc"

namespace {

struct PromoteStablehloLogisticPass final
    : impl::PromoteStablehloLogisticPassBase<PromoteStablehloLogisticPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<mlir::stablehlo::LogisticOp> candidates;
    getOperation().walk([&](mlir::stablehlo::LogisticOp op) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
      if (type &&
          (type.getElementType().isF16() || type.getElementType().isBF16()))
        candidates.push_back(op);
    });
    mlir::IRRewriter rewriter(&getContext());
    for (mlir::stablehlo::LogisticOp op : candidates) {
      auto type = mlir::cast<mlir::RankedTensorType>(op.getType());
      auto computeType = type.clone(rewriter.getF32Type());
      rewriter.setInsertionPoint(op);
      auto input = rewriter.create<mlir::stablehlo::ConvertOp>(
          op.getLoc(), computeType, op.getOperand());
      auto result = rewriter.create<mlir::stablehlo::LogisticOp>(
          op.getLoc(), computeType, input.getResult());
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(
          op, type, result.getResult());
    }
#endif
  }
};

} // namespace
} // namespace wafer
