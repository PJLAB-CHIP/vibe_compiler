//===- TensorAssemblyMaterialization.h - Selected tensor demand ----------===//

#ifndef WAFER_TRANSFORMS_LINALG_TENSORASSEMBLYMATERIALIZATION_H
#define WAFER_TRANSFORMS_LINALG_TENSORASSEMBLYMATERIALIZATION_H

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

std::optional<int64_t> resolveStaticIndex(mlir::OpFoldResult value);

struct MaterializedTensorAssemblyRead {
  mlir::Value value;
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> sourceReads;
};

enum class TensorSubsetMaterializationStatus {
  Exact,
  Unsupported,
  ResourceExhausted,
  BrokenContract,
  CompilerFailure,
};

struct TensorSubsetMaterializationResult {
  TensorSubsetMaterializationStatus status =
      TensorSubsetMaterializationStatus::Unsupported;
  std::optional<MaterializedTensorAssemblyRead> materialized;
  uint64_t work = 0;
  uint64_t blockTemplates = 0;
  std::string detail;
};

// Query and preflight the actual selected read, then emit compact Tensor/SCF.
// The caller owns the candidate transaction, source binding, read replacement
// and any selected producer fusion. No outer compute loop is changed here.
// A caller-proved invariant placement may name an enclosing nonempty for-loop;
// the generator rechecks dominance and every crossed loop before emission.
TensorSubsetMaterializationResult
materializeTensorSubsetRead(mlir::RewriterBase &rewriter,
                            mlir::tensor::ExtractSliceOp read,
                            const analysis::IndexRelationLimits &limits =
                                analysis::IndexRelationLimits(),
                            mlir::Operation *insertionPoint = nullptr);

analysis::TensorSubsetDemandResult
queryTensorSubsetSlice(mlir::tensor::ExtractSliceOp read,
                       analysis::IndexRelationWork &work);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_LINALG_TENSORASSEMBLYMATERIALIZATION_H
