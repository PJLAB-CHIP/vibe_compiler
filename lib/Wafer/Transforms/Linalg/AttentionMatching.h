//===- AttentionMatching.h - Structured attention graph proof -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_ATTENTIONMATCHING_H
#define WAFER_TRANSFORMS_LINALG_ATTENTIONMATCHING_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"

namespace wafer::attention_normalization {

/// Complete immutable proof used by one normalization transaction. Operation
/// and value handles are valid only until the function is first mutated.
struct AttentionMatch {
  mlir::Operation *root = nullptr;
  mlir::Value query;
  mlir::Value key;
  mlir::Value value;
  mlir::Value scale;
  mlir::Value mask;
  mlir::RankedTensorType outputType;
  AttentionAlgorithm algorithm = AttentionAlgorithm::FlashAttention;
  llvm::SmallVector<mlir::AffineMap, 6> indexingMaps;
};

llvm::SmallVector<AttentionMatch, 4>
collectAttentionMatches(mlir::func::FuncOp function);

} // namespace wafer::attention_normalization

#endif // WAFER_TRANSFORMS_LINALG_ATTENTIONMATCHING_H
