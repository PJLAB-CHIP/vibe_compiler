//===- TensorPreparation.h - Expose final tensor subsets --------*- C++ -*-===//
#ifndef WAFER_TRANSFORMS_TILE_TENSORPREPARATION_H
#define WAFER_TRANSFORMS_TILE_TENSORPREPARATION_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "mlir/IR/PatternMatch.h"
#include <string>

namespace wafer::compiler::detail {
enum class TensorPreparationStatus {
  Success,
  Unsupported,
  BrokenContract,
  CompilerFailure
};
struct TensorPreparationResult {
  TensorPreparationStatus status = TensorPreparationStatus::CompilerFailure;
  std::string detail;
  bool succeeded() const { return status == TensorPreparationStatus::Success; }
};

/// Expose subset-producing tensor computations before final read discovery.
/// The caller owns selection and subsequent layout/bufferization. Current
/// SSA replacements are forwarded so no choice is recovered by name or order.
TensorPreparationResult prepareCurrentTensorInput(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    mlir::RewriterBase::Listener *externalListener = nullptr);

/// Retain coordinate-bearing singleton scopes until read choices are consumed.
void preserveTensorIterationScopes(mlir::RewritePatternSet &patterns);
} // namespace wafer::compiler::detail
#endif
