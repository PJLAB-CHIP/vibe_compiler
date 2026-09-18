//===- TensorInitialization.cpp - Uniform tensor DPS conversion --------===//

#include "TensorInitialization.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallVector.h"

mlir::FailureOr<wafer::TensorInitializationStatistics>
wafer::lowerUniformTensorInitializers(mlir::RewriterBase &rewriter,
                                      mlir::Operation *root) {
  TensorInitializationStatistics statistics;
  llvm::SmallVector<mlir::tensor::PadOp> pads;
  root->walk([&](mlir::tensor::PadOp pad) {
    if (pad.getConstantPaddingValue())
      pads.push_back(pad);
  });
  for (auto pad : pads) {
    rewriter.setInsertionPoint(pad);
    if (mlir::failed(
            mlir::linalg::rewriteInDestinationPassingStyle(rewriter, pad)))
      return mlir::failure();
    ++statistics.pads;
  }

  llvm::SmallVector<mlir::tensor::GenerateOp> generates;
  root->walk([&](mlir::tensor::GenerateOp op) { generates.push_back(op); });
  for (auto op : generates) {
    if (!mlir::isMemoryEffectFree(op))
      continue;
    auto yield =
        mlir::cast<mlir::tensor::YieldOp>(op.getBody().front().getTerminator());
    mlir::Value value = yield.getValue();
    mlir::Attribute constant;
    bool isConstant = mlir::matchPattern(value, mlir::m_Constant(&constant));
    if (value.getParentRegion() == &op.getRegion() && !isConstant)
      continue;
    rewriter.setInsertionPoint(op);
    if (isConstant && value.getParentRegion() == &op.getRegion()) {
      auto *definition = value.getDefiningOp();
      auto *materialized = definition->getDialect()->materializeConstant(
          rewriter, constant, value.getType(), value.getLoc());
      if (!materialized)
        return mlir::failure();
      value = materialized->getResult(0);
    }
    auto type = op.getType();
    auto empty = rewriter.create<mlir::tensor::EmptyOp>(op.getLoc(), type,
                                                        op.getDynamicExtents());
    auto fill = rewriter.create<mlir::linalg::FillOp>(op.getLoc(), value,
                                                      empty.getResult());
    rewriter.replaceOp(op, fill.getResults());
    ++statistics.generates;
  }
  return statistics;
}
