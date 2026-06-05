//===- MoveOps.cpp - Wafer movement verifier implementation ------------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

namespace {

static mlir::LogicalResult getSPMStorage(mlir::Operation *op, mlir::Type type,
                                         llvm::StringRef role,
                                         StorageType &storageType) {
  storageType = mlir::dyn_cast<StorageType>(type);
  if (!storageType)
    return op->emitOpError() << role << " must be a storage";
  if (!hasStorageMemorySpace(storageType, MemorySpace::SPM))
    return op->emitOpError() << role << " must use SPM memory space";
  return mlir::success();
}

static mlir::LogicalResult
verifySameElementLayoutAndSpace(mlir::Operation *op, StorageType lhs,
                                StorageType rhs,
                                llvm::StringRef messagePrefix) {
  mlir::RankedTensorType lhsTensor = getStorageTensorType(lhs);
  mlir::RankedTensorType rhsTensor = getStorageTensorType(rhs);
  if (lhsTensor.getElementType() != rhsTensor.getElementType())
    return op->emitOpError() << messagePrefix << " element types must match";
  if (getStorageLayout(lhs).getValue() != getStorageLayout(rhs).getValue())
    return op->emitOpError() << messagePrefix << " mem_layout must match";
  if (getStorageMemorySpace(lhs).getValue() !=
      getStorageMemorySpace(rhs).getValue())
    return op->emitOpError() << messagePrefix << " memory_space must match";
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

static void
appendMoveResourceEffects(mlir::Type readType, mlir::Type writeType,
                          llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(readType));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(writeType));
  appendResourceEffect(effects, WaferResourceKind::Movement,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(writeType));
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
  StorageType sourceType;
  StorageType resultType;
  if (mlir::failed(getSPMStorage(getOperation(), getSource().getType(),
                                 "extract_slice source", sourceType)) ||
      mlir::failed(getSPMStorage(getOperation(), getResult().getType(),
                                 "extract_slice result", resultType)))
    return mlir::failure();
  if (mlir::failed(verifySameElementLayoutAndSpace(
          getOperation(), sourceType, resultType, "extract_slice")))
    return mlir::failure();
  return verifySliceAttrs(
      getOperation(), getStorageTensorType(sourceType),
      getStorageTensorType(resultType), getOffsetsAttr().asArrayRef(),
      getSizesAttr().asArrayRef(), getStridesAttr().asArrayRef());
}

void MoveExtractSliceOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto sourceType = mlir::dyn_cast<StorageType>(getSource().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            sourceType);
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult MoveExtractSliceOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 2> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void MoveExtractSliceOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendMoveResourceEffects(getSource().getType(), getResult().getType(),
                            effects);
}

mlir::LogicalResult MoveExtractSliceOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult MoveInsertSliceOp::verify() {
  StorageType sourceType;
  StorageType destType;
  StorageType resultType;
  if (mlir::failed(getSPMStorage(getOperation(), getSource().getType(),
                                 "insert_slice source", sourceType)) ||
      mlir::failed(getSPMStorage(getOperation(), getDest().getType(),
                                 "insert_slice dest", destType)) ||
      mlir::failed(getSPMStorage(getOperation(), getResult().getType(),
                                 "insert_slice result", resultType)))
    return mlir::failure();
  if (destType != resultType)
    return emitOpError("insert_slice result type must match dest type");
  if (mlir::failed(verifySameElementLayoutAndSpace(getOperation(), sourceType,
                                                   destType, "insert_slice")))
    return mlir::failure();
  return verifySliceAttrs(
      getOperation(), getStorageTensorType(destType),
      getStorageTensorType(sourceType), getOffsetsAttr().asArrayRef(),
      getSizesAttr().asArrayRef(), getStridesAttr().asArrayRef());
}

void MoveInsertSliceOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto sourceType = mlir::dyn_cast<StorageType>(getSource().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            sourceType);
  if (auto destType = mlir::dyn_cast<StorageType>(getDest().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 1, destType);
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult MoveInsertSliceOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 3> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void MoveInsertSliceOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
  appendResourceEffect(effects, WaferResourceKind::Movement,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
}

mlir::LogicalResult MoveInsertSliceOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult MoveCopyOp::verify() {
  StorageType sourceType;
  StorageType resultType;
  if (mlir::failed(getSPMStorage(getOperation(), getSource().getType(),
                                 "copy source", sourceType)) ||
      mlir::failed(getSPMStorage(getOperation(), getResult().getType(),
                                 "copy result", resultType)))
    return mlir::failure();
  if (sourceType != resultType)
    return emitOpError("copy source and result types must match");
  return mlir::success();
}

void MoveCopyOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto sourceType = mlir::dyn_cast<StorageType>(getSource().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            sourceType);
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult MoveCopyOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 2> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void MoveCopyOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendMoveResourceEffects(getSource().getType(), getResult().getType(),
                            effects);
}

mlir::LogicalResult MoveCopyOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult MoveTransposeOp::verify() {
  StorageType sourceType;
  StorageType resultType;
  if (mlir::failed(getSPMStorage(getOperation(), getSource().getType(),
                                 "transpose source", sourceType)) ||
      mlir::failed(getSPMStorage(getOperation(), getResult().getType(),
                                 "transpose result", resultType)))
    return mlir::failure();
  if (mlir::failed(verifySameElementLayoutAndSpace(getOperation(), sourceType,
                                                   resultType, "transpose")))
    return mlir::failure();

  mlir::RankedTensorType sourceTensor = getStorageTensorType(sourceType);
  mlir::RankedTensorType resultTensor = getStorageTensorType(resultType);
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

void MoveTransposeOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto sourceType = mlir::dyn_cast<StorageType>(getSource().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            sourceType);
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult MoveTransposeOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 2> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void MoveTransposeOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendMoveResourceEffects(getSource().getType(), getResult().getType(),
                            effects);
}

mlir::LogicalResult MoveTransposeOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult MoveBroadcastOp::verify() {
  StorageType sourceType;
  StorageType resultType;
  if (mlir::failed(getSPMStorage(getOperation(), getSource().getType(),
                                 "broadcast source", sourceType)) ||
      mlir::failed(getSPMStorage(getOperation(), getResult().getType(),
                                 "broadcast result", resultType)))
    return mlir::failure();
  if (mlir::failed(verifySameElementLayoutAndSpace(getOperation(), sourceType,
                                                   resultType, "broadcast")))
    return mlir::failure();

  mlir::RankedTensorType sourceTensor = getStorageTensorType(sourceType);
  mlir::RankedTensorType resultTensor = getStorageTensorType(resultType);
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

void MoveBroadcastOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto sourceType = mlir::dyn_cast<StorageType>(getSource().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            sourceType);
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult MoveBroadcastOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 2> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void MoveBroadcastOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendMoveResourceEffects(getSource().getType(), getResult().getType(),
                            effects);
}

mlir::LogicalResult MoveBroadcastOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
