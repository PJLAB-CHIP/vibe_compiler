//===- ScalarInitialization.h - DPS scalar-init analysis ------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_STRUCTURED_SCALARINITIALIZATION_H
#define WAFER_ANALYSIS_STRUCTURED_SCALARINITIALIZATION_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::analysis {
namespace detail {

inline bool onlyFeedsScalarInitializedReduction(
    mlir::Value value, llvm::DenseSet<mlir::Value> &visited) {
  if (value.use_empty() || !visited.insert(value).second)
    return false;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(owner)) {
      const bool hasReduction = llvm::any_of(
          linalg.getIteratorTypesArray(), mlir::linalg::isReductionIterator);
      if (linalg.isDpsInit(&use) && hasReduction)
        continue;
      return false;
    }
    if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(owner)) {
      if (expand.getSrc() == value &&
          onlyFeedsScalarInitializedReduction(expand.getResult(), visited))
        continue;
      return false;
    }
    if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(owner)) {
      if (collapse.getSrc() == value &&
          onlyFeedsScalarInitializedReduction(collapse.getResult(), visited))
        continue;
      return false;
    }
    auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(owner);
    if (!extract || extract.getSource() != value ||
        !onlyFeedsScalarInitializedReduction(extract.getResult(), visited))
      return false;
  }
  return true;
}

} // namespace detail

/// Returns true only when every use of `value`, through exact tensor
/// reshape/slice forwarding, is a DPS init of a structured reduction. Such a
/// value carries the scalar initializer into the consumer traversal and does
/// not require a standalone full-tensor fill execution.
inline bool onlyFeedsScalarInitializedReduction(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  return detail::onlyFeedsScalarInitializedReduction(value, visited);
}

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_STRUCTURED_SCALARINITIALIZATION_H
