//===- TensorAssemblyMaterialization.cpp - Selected tensor demand --------===//

#include "TensorAssemblyMaterialization.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace wafer::compiler::detail {

mlir::FailureOr<mlir::Value> materializeTensorAssemblyTile(
    mlir::OpBuilder &builder, mlir::Location location,
    mlir::RankedTensorType resultType,
    const analysis::StaticRectangularIndexSet &requested,
    llvm::ArrayRef<TensorAssemblyTilePiece> pieces) {
  const size_t rank = resultType.getRank();
  if (pieces.empty() || !resultType.hasStaticShape() ||
      requested.offsets.size() != rank || requested.sizes.size() != rank ||
      resultType.getShape() != llvm::ArrayRef<int64_t>(requested.sizes))
    return mlir::failure();
  int64_t requestedElements = 1;
  for (int64_t size : requested.sizes)
    if (size <= 0 ||
        llvm::MulOverflow(requestedElements, size, requestedElements))
      return mlir::failure();
  int64_t coveredElements = 0;
  for (const auto &piece : pieces) {
    if (!piece.value)
      return mlir::failure();
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(piece.value.getType());
    if (!type || !type.hasStaticShape() ||
        type.getElementType() != resultType.getElementType() ||
        type.getEncoding() != resultType.getEncoding() ||
        piece.resultWindow.offsets.size() != rank ||
        piece.resultWindow.sizes.size() != rank ||
        type.getShape() != llvm::ArrayRef<int64_t>(piece.resultWindow.sizes))
      return mlir::failure();
    int64_t pieceElements = 1;
    for (auto [origin, extent, offset, size] : llvm::zip_equal(
             requested.offsets, requested.sizes, piece.resultWindow.offsets,
             piece.resultWindow.sizes)) {
      int64_t end = 0;
      if (size <= 0 || origin < 0 || offset < origin || extent <= 0 ||
          llvm::AddOverflow(origin, extent, end) || offset > end ||
          size > end - offset ||
          llvm::MulOverflow(pieceElements, size, pieceElements))
        return mlir::failure();
    }
    if (llvm::AddOverflow(coveredElements, pieceElements, coveredElements))
      return mlir::failure();
  }
  if (coveredElements != requestedElements)
    return mlir::failure();
  for (size_t left = 0; left < pieces.size(); ++left)
    for (size_t right = left + 1; right < pieces.size(); ++right) {
      bool disjoint = false;
      for (size_t axis = 0; axis < rank; ++axis) {
        int64_t leftEnd = pieces[left].resultWindow.offsets[axis] +
                          pieces[left].resultWindow.sizes[axis];
        int64_t rightEnd = pieces[right].resultWindow.offsets[axis] +
                           pieces[right].resultWindow.sizes[axis];
        disjoint |= leftEnd <= pieces[right].resultWindow.offsets[axis] ||
                    rightEnd <= pieces[left].resultWindow.offsets[axis];
      }
      if (!disjoint)
        return mlir::failure();
    }
  if (pieces.size() == 1 &&
      pieces.front().resultWindow.offsets == requested.offsets &&
      pieces.front().resultWindow.sizes == requested.sizes)
    return pieces.front().value;

  mlir::Value assembled = builder.create<mlir::tensor::EmptyOp>(
      location, resultType.getShape(), resultType.getElementType(),
      resultType.getEncoding());
  for (const auto &piece : pieces) {
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets, sizes, strides;
    for (auto [offset, origin, size] :
         llvm::zip_equal(piece.resultWindow.offsets, requested.offsets,
                         piece.resultWindow.sizes)) {
      offsets.push_back(builder.getIndexAttr(offset - origin));
      sizes.push_back(builder.getIndexAttr(size));
      strides.push_back(builder.getIndexAttr(1));
    }
    assembled = builder.create<mlir::tensor::InsertSliceOp>(
        location, piece.value, assembled, offsets, sizes, strides);
  }
  return assembled;
}

} // namespace wafer::compiler::detail
