//===- StructuredOptimizationInternal.h - Normalization helpers -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_STRUCTUREDOPTIMIZATION_INTERNAL_H
#define WAFER_TRANSFORMS_STRUCTUREDOPTIMIZATION_INTERNAL_H

#include "Wafer/Transforms/StructuredOptimization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"

namespace wafer::structured_optimization::detail {

bool initializeWorkSummary(mlir::ModuleOp module,
                           TensorNormalizationWorkSummary &work);
bool initializeWorkSummary(
    llvm::ArrayRef<mlir::func::FuncOp> schedulableFunctions,
    TensorNormalizationWorkSummary &work);
bool addWorkCounter(uint64_t &target, uint64_t value);
bool multiplyWorkCounter(uint64_t lhs, uint64_t rhs, uint64_t &result);
bool consumeFuel(TensorNormalizationWorkSummary &work, uint64_t amount);
bool mergeWork(TensorNormalizationWorkSummary &destination,
               const TensorNormalizationWorkSummary &source);
std::string getCanonicalOperationPath(mlir::Operation *operation,
                                      mlir::ModuleOp root);
TensorNormalizationOutcome makeFailure(
    TensorNormalizationStatus status, TensorNormalizationFamily family,
    llvm::StringRef reason, mlir::Operation *operation, mlir::ModuleOp root,
    TensorNormalizationWorkSummary work);

/// Establishes the traversal/view subset needed by candidate-local physical
/// planning while tensor bufferization adapters are still present. Full
/// allocation-root, alias, and completion verification remains owned by the
/// post-bufferization selected-payload entry point.
SelectedPayloadNormalizationOutcome
normalizeSelectedPayloadStructureForPlanning(mlir::ModuleOp module);

} // namespace wafer::structured_optimization::detail

#endif // WAFER_TRANSFORMS_STRUCTUREDOPTIMIZATION_INTERNAL_H
