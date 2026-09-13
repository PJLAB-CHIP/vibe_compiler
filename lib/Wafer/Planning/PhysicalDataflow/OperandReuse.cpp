//===- OperandReuse.cpp - Current operand access invariance ---------------===//

#include "OperandReuse.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/MathExtras.h"

#include <utility>

namespace wafer::compiler::detail {

std::optional<llvm::SmallVector<OperandProjection, 4>>
getReadOperandProjections(mlir::Operation *operation) {
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!linalg)
    return std::nullopt;
  llvm::SmallVector<OperandProjection, 4> result;
  for (mlir::OpOperand &operand : linalg->getOpOperands()) {
    if (!linalg.payloadUsesValueFromOperand(&operand))
      continue;
    mlir::AffineMap map = linalg.getMatchingIndexingMap(&operand);
    if (!map || map.getNumSymbols())
      return std::nullopt;
    mlir::Type element = operand.get().getType();
    uint64_t elements = 1;
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(element)) {
      if (!shaped.hasStaticShape())
        return std::nullopt;
      for (int64_t extent : shaped.getShape()) {
        if (extent <= 0)
          return std::nullopt;
        elements = llvm::SaturatingMultiply(elements, uint64_t(extent));
      }
      element = shaped.getElementType();
    }
    if (!element.isIntOrFloat())
      return std::nullopt;
    OperandProjection projection;
    projection.bytes = llvm::SaturatingMultiply(
        elements, uint64_t((element.getIntOrFloatBitWidth() + 7) / 8));
    projection.iterators.resize(map.getNumDims(), false);
    // Absence is a cheap exact invariance proof, including affine strides and
    // broadcasts. Presence makes no claim about overlap between windows.
    for (mlir::AffineExpr expression : map.getResults())
      for (unsigned iterator = 0; iterator < map.getNumDims(); ++iterator)
        if (expression.isFunctionOfDim(iterator))
          projection.iterators.set(iterator);
    result.push_back(std::move(projection));
  }
  return result;
}

} // namespace wafer::compiler::detail
