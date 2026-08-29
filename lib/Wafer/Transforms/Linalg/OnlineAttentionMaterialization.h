//===- OnlineAttentionMaterialization.h - Actual online state -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_ONLINEATTENTIONMATERIALIZATION_H
#define WAFER_TRANSFORMS_LINALG_ONLINEATTENTIONMATERIALIZATION_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

struct OnlineAttentionState {
  mlir::Value accumulator;
  mlir::Value maximum;
  mlir::Value sum;
};

mlir::FailureOr<OnlineAttentionState> materializeOnlineAttentionTile(
    LinalgExtAttentionOp source, mlir::Value query, mlir::Value key,
    mlir::Value value, mlir::Value scale, mlir::Value mask,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, mlir::OpBuilder &builder);

mlir::FailureOr<mlir::Value>
materializeOnlineAttentionFinalize(LinalgExtAttentionOp source,
                                   const OnlineAttentionState &state,
                                   mlir::OpBuilder &builder);

mlir::FailureOr<OnlineAttentionState> materializeOnlineAttentionStateMerge(
    LinalgExtAttentionOp source, const OnlineAttentionState &left,
    const OnlineAttentionState &right, mlir::OpBuilder &builder);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_LINALG_ONLINEATTENTIONMATERIALIZATION_H
