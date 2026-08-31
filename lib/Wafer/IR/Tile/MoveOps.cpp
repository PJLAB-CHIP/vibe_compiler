//===- MoveOps.cpp - Wafer movement verifier implementation ------------===//

#include "Wafer/IR/WaferDialect.h"

#include "WaferIRVerification.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

namespace {

static mlir::LogicalResult
getSPMBufferTensor(mlir::Operation *op, mlir::Type type, llvm::StringRef role,
                   mlir::RankedTensorType &tensorType) {
  std::optional<mlir::RankedTensorType> logicalTensor =
      getLogicalTensorType(type);
  if (!logicalTensor)
    return op->emitOpError() << role << " must be a Wafer buffer";
  if (!hasWaferMemorySpace(type, MemorySpace::SPM))
    return op->emitOpError() << role << " must use SPM memory space";
  tensorType = *logicalTensor;
  return mlir::success();
}

static mlir::LogicalResult verifySameElementLayoutAndSpace(
    mlir::Operation *op, mlir::Type lhsType, mlir::RankedTensorType lhsTensor,
    mlir::Type rhsType, mlir::RankedTensorType rhsTensor,
    llvm::StringRef messagePrefix) {
  if (lhsTensor.getElementType() != rhsTensor.getElementType())
    return op->emitOpError() << messagePrefix << " element types must match";
  if (getWaferLayout(lhsType) != getWaferLayout(rhsType))
    return op->emitOpError() << messagePrefix << " layout must match";
  if (getWaferMemorySpace(lhsType) != getWaferMemorySpace(rhsType))
    return op->emitOpError() << messagePrefix << " memory space must match";
  return mlir::success();
}

static mlir::LogicalResult verifySliceAttrs(mlir::Operation *op,
                                            mlir::RankedTensorType baseTensor,
                                            mlir::RankedTensorType sliceTensor,
                                            llvm::ArrayRef<int64_t> offsets,
                                            llvm::ArrayRef<int64_t> sizes,
                                            llvm::ArrayRef<int64_t> strides) {
  int64_t rank = baseTensor.getRank();
  if (static_cast<int64_t>(offsets.size()) != rank ||
      static_cast<int64_t>(sizes.size()) != rank ||
      static_cast<int64_t>(strides.size()) != rank)
    return op->emitOpError(
        "slice offsets/sizes/strides must match base tensor rank");
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (offsets[dim] < 0)
      return op->emitOpError("slice offsets must be non-negative");
    if (sizes[dim] <= 0)
      return op->emitOpError("slice sizes must be positive");
    if (strides[dim] <= 0)
      return op->emitOpError("slice strides must be positive");

    int64_t baseDim = baseTensor.getDimSize(dim);
    if (baseDim == mlir::ShapedType::kDynamic)
      continue;
    int64_t lastElementOffset = offsets[dim] + (sizes[dim] - 1) * strides[dim];
    if (lastElementOffset >= baseDim)
      return op->emitOpError("slice range must fit within base tensor shape");
  }

  auto fullSliceTensor =
      mlir::RankedTensorType::get(sizes, baseTensor.getElementType());
  if (mlir::isRankReducedType(fullSliceTensor, sliceTensor) !=
      mlir::SliceVerificationResult::Success)
    return op->emitOpError(
        "slice tensor type must match sizes with optional rank reduction");
  return mlir::success();
}

static mlir::LogicalResult verifyPermutation(mlir::Operation *op,
                                             llvm::ArrayRef<int64_t> values,
                                             int64_t rank) {
  if (static_cast<int64_t>(values.size()) != rank)
    return op->emitOpError("transpose permutation must match tensor rank");
  llvm::DenseSet<int64_t> seen;
  for (int64_t value : values) {
    if (value < 0 || value >= rank)
      return op->emitOpError(
          "transpose permutation entries must be within tensor rank");
    if (!seen.insert(value).second)
      return op->emitOpError("transpose permutation entries must be unique");
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult MoveExtractSliceOp::verify() {
  mlir::RankedTensorType sourceTensor;
  mlir::RankedTensorType resultTensor;
  if (mlir::failed(getSPMBufferTensor(getOperation(), getSource().getType(),
                                      "extract_slice source", sourceTensor)) ||
      mlir::failed(getSPMBufferTensor(getOperation(), getResult().getType(),
                                      "extract_slice result", resultTensor)))
    return mlir::failure();
  if (mlir::failed(verifySameElementLayoutAndSpace(
          getOperation(), getSource().getType(), sourceTensor,
          getResult().getType(), resultTensor, "extract_slice")))
    return mlir::failure();
  return verifySliceAttrs(
      getOperation(), sourceTensor, resultTensor, getOffsetsAttr().asArrayRef(),
      getSizesAttr().asArrayRef(), getStridesAttr().asArrayRef());
}

mlir::LogicalResult MoveInsertSliceOp::verify() {
  if (DDRResourceAttr resource = getDdrResourceAttr())
    if (resource.getResourceId() < 0)
      return emitOpError("card DDR movement resource must be non-negative");
  mlir::RankedTensorType sourceTensor;
  mlir::RankedTensorType destTensor;
  if (mlir::failed(getSPMBufferTensor(getOperation(), getSource().getType(),
                                      "insert_slice source", sourceTensor)) ||
      mlir::failed(getSPMBufferTensor(getOperation(), getDest().getType(),
                                      "insert_slice dest", destTensor)))
    return mlir::failure();
  if (mlir::failed(verifySameElementLayoutAndSpace(
          getOperation(), getSource().getType(), sourceTensor,
          getDest().getType(), destTensor, "insert_slice")))
    return mlir::failure();
  return verifySliceAttrs(
      getOperation(), destTensor, sourceTensor, getOffsetsAttr().asArrayRef(),
      getSizesAttr().asArrayRef(), getStridesAttr().asArrayRef());
}

mlir::LogicalResult MoveCopyOp::verify() {
  if (DDRResourceAttr resource = getDdrResourceAttr())
    if (resource.getResourceId() < 0)
      return emitOpError("card DDR movement resource must be non-negative");
  mlir::RankedTensorType sourceTensor;
  mlir::RankedTensorType resultTensor;
  if (mlir::failed(getSPMBufferTensor(getOperation(), getSource().getType(),
                                      "copy source", sourceTensor)) ||
      mlir::failed(getSPMBufferTensor(getOperation(), getResult().getType(),
                                      "copy result", resultTensor)))
    return mlir::failure();
  if (mlir::failed(verifySameElementLayoutAndSpace(
          getOperation(), getSource().getType(), sourceTensor,
          getResult().getType(), resultTensor, "copy")))
    return mlir::failure();
  if (sourceTensor.getShape() != resultTensor.getShape())
    return emitOpError("copy source and result shapes must match");
  return mlir::success();
}

mlir::LogicalResult MoveCopyIntoOp::verify() {
  mlir::RankedTensorType sourceTensor;
  mlir::RankedTensorType destTensor;
  if (mlir::failed(getSPMBufferTensor(getOperation(), getSource().getType(),
                                      "copy_into source", sourceTensor)) ||
      mlir::failed(getSPMBufferTensor(getOperation(), getDest().getType(),
                                      "copy_into dest", destTensor)))
    return mlir::failure();
  if (sourceTensor != destTensor)
    return emitOpError("copy_into source and dest tensor types must match");
  return mlir::success();
}

mlir::LogicalResult MoveReshapeOp::verify() {
  mlir::RankedTensorType sourceTensor;
  mlir::RankedTensorType resultTensor;
  if (mlir::failed(getSPMBufferTensor(getOperation(), getSource().getType(),
                                      "reshape_copy source", sourceTensor)) ||
      mlir::failed(getSPMBufferTensor(getOperation(), getResult().getType(),
                                      "reshape_copy result", resultTensor)))
    return mlir::failure();
  if (mlir::failed(verifySameElementLayoutAndSpace(
          getOperation(), getSource().getType(), sourceTensor,
          getResult().getType(), resultTensor, "reshape_copy")))
    return mlir::failure();
  if (!sourceTensor.hasStaticShape() || !resultTensor.hasStaticShape())
    return emitOpError("reshape_copy requires static tensor shapes");
  if (sourceTensor.getNumElements() != resultTensor.getNumElements())
    return emitOpError("reshape_copy must preserve static element count");
  return mlir::success();
}

mlir::LogicalResult MoveTransposeOp::verify() {
  mlir::RankedTensorType sourceTensor;
  mlir::RankedTensorType resultTensor;
  if (mlir::failed(getSPMBufferTensor(getOperation(), getSource().getType(),
                                      "transpose source", sourceTensor)) ||
      mlir::failed(getSPMBufferTensor(getOperation(), getResult().getType(),
                                      "transpose result", resultTensor)))
    return mlir::failure();
  if (mlir::failed(verifySameElementLayoutAndSpace(
          getOperation(), getSource().getType(), sourceTensor,
          getResult().getType(), resultTensor, "transpose")))
    return mlir::failure();

  llvm::ArrayRef<int64_t> permutation = getPermutationAttr().asArrayRef();
  if (resultTensor.getRank() != sourceTensor.getRank())
    return emitOpError("transpose source and result ranks must match");
  if (mlir::failed(verifyPermutation(getOperation(), permutation,
                                     sourceTensor.getRank())))
    return mlir::failure();

  for (auto [resultDim, sourceDim] : llvm::enumerate(permutation)) {
    if (hasStaticMismatch(resultTensor.getDimSize(resultDim),
                          sourceTensor.getDimSize(sourceDim)))
      return emitOpError("transpose result shape must match permutation");
  }
  return mlir::success();
}

mlir::LogicalResult MoveBroadcastOp::verify() {
  mlir::RankedTensorType sourceTensor;
  mlir::RankedTensorType resultTensor;
  if (mlir::failed(getSPMBufferTensor(getOperation(), getSource().getType(),
                                      "broadcast source", sourceTensor)) ||
      mlir::failed(getSPMBufferTensor(getOperation(), getResult().getType(),
                                      "broadcast result", resultTensor)))
    return mlir::failure();
  if (mlir::failed(verifySameElementLayoutAndSpace(
          getOperation(), getSource().getType(), sourceTensor,
          getResult().getType(), resultTensor, "broadcast")))
    return mlir::failure();

  llvm::ArrayRef<int64_t> dimensions = getDimensionsAttr().asArrayRef();
  if (static_cast<int64_t>(dimensions.size()) != sourceTensor.getRank())
    return emitOpError("broadcast dimensions must match source tensor rank");
  llvm::DenseSet<int64_t> seen;
  for (auto [sourceDim, resultDim] : llvm::enumerate(dimensions)) {
    if (resultDim < 0 || resultDim >= resultTensor.getRank())
      return emitOpError(
          "broadcast dimensions must be within result tensor rank");
    if (!seen.insert(resultDim).second)
      return emitOpError("broadcast dimensions must be unique");
    if (hasStaticMismatch(sourceTensor.getDimSize(sourceDim),
                          resultTensor.getDimSize(resultDim)))
      return emitOpError(
          "broadcast source shape must match mapped result dims");
  }
  return mlir::success();
}
