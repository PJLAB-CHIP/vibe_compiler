//===- AttentionGemmUtils.h - Attention contraction helpers -----*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_ATTENTIONGEMMUTILS_H
#define WAFER_TRANSFORMS_ATTENTIONGEMMUTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace wafer {

struct AttentionGemmDims {
  int64_t lhsMDim = 2;
  int64_t lhsContractingDim = 3;
  int64_t rhsContractingDim = 3;
  int64_t rhsNDim = 2;
  int64_t resultMDim = 2;
  int64_t resultNDim = 3;
};

std::optional<AttentionGemmDims>
matchAttentionGemm(mlir::linalg::GenericOp generic);

void setAttentionGemmAttrs(mlir::Operation *op, mlir::OpBuilder &builder,
                           mlir::RankedTensorType resultType,
                           AttentionGemmDims dims);

} // namespace wafer

#endif // WAFER_TRANSFORMS_ATTENTIONGEMMUTILS_H
