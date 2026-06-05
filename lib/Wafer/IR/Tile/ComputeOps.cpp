//===- ComputeOps.cpp - Wafer Compute verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult ComputeFillOp::verify() {
  auto destType = mlir::dyn_cast<StorageType>(getDest().getType());
  if (!destType)
    return emitOpError("expects storage destination");
  if (!hasStorageMemorySpace(destType, MemorySpace::SPM))
    return emitOpError("fill destination must use SPM memory space");
  if (!hasStorageLayout(destType, MemLayout::Tensor))
    return emitOpError("fill destination must use tensor mem_layout");

  mlir::RankedTensorType destTensor = getStorageTensorType(destType);
  if (getValue().getType() != destTensor.getElementType())
    return emitOpError(
        "fill value type must match destination tensor element type");
  return mlir::success();
}

void ComputeFillOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto destType = mlir::dyn_cast<StorageType>(getDest().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0, destType);
}

mlir::LogicalResult ComputeFillOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 1> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void ComputeFillOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendResourceEffect(effects, WaferResourceKind::Compute,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult ComputeFillOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 2> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult ComputeGemmOp::verify() {
  auto lhsType = mlir::dyn_cast<StorageType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<StorageType>(getRhs().getType());
  auto resultType = mlir::dyn_cast<StorageType>(getResult().getType());
  if (!lhsType || !rhsType || !resultType)
    return emitOpError("expects storage operands and result");

  for (StorageType type : {lhsType, rhsType, resultType}) {
    if (!hasStorageMemorySpace(type, MemorySpace::SPM))
      return emitOpError("gemm storage values must use SPM memory space");
    if (!hasStorageLayout(type, MemLayout::Cx))
      return emitOpError("gemm storage values must use cx mem_layout");
  }

  mlir::RankedTensorType lhsTensor = getStorageTensorType(lhsType);
  mlir::RankedTensorType rhsTensor = getStorageTensorType(rhsType);
  mlir::RankedTensorType resultTensor = getStorageTensorType(resultType);

  if (lhsTensor.getElementType() != rhsTensor.getElementType() ||
      lhsTensor.getElementType() != resultTensor.getElementType())
    return emitOpError("gemm operand and result element types must match");

  if (lhsTensor.getRank() != 2 || rhsTensor.getRank() != 2 ||
      resultTensor.getRank() != 2) {
    BatchedGemmDimAttrs attrs;
    return verifyBatchedGemmTileContract(getOperation(), lhsTensor, rhsTensor,
                                         resultTensor, attrs);
  }

  if (hasAnyBatchedGemmAttrs(getOperation()))
    return emitOpError("gemm rank-2 form must not carry batched GEMM attrs");

  if (hasStaticMismatch(lhsTensor.getDimSize(1), rhsTensor.getDimSize(0)))
    return emitOpError("gemm lhs K dimension must match rhs K dimension");
  if (hasStaticMismatch(lhsTensor.getDimSize(0), resultTensor.getDimSize(0)) ||
      hasStaticMismatch(rhsTensor.getDimSize(1), resultTensor.getDimSize(1)))
    return emitOpError("gemm result shape must be lhs M by rhs N");

  return mlir::success();
}

void ComputeGemmOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto lhsType = mlir::dyn_cast<StorageType>(getLhs().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0, lhsType);
  if (auto rhsType = mlir::dyn_cast<StorageType>(getRhs().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 1, rhsType);
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult ComputeGemmOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void ComputeGemmOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getLhs().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getRhs().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
  appendResourceEffect(effects, WaferResourceKind::Compute,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
}

mlir::LogicalResult ComputeGemmOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult ComputeElementwiseOp::verify() {
  return verifyElementwiseTileContract(getOperation(), getKindAttr().getValue(),
                                       getInputs(), getResult().getType());
}

void ComputeElementwiseOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  for (auto [index, value] : llvm::enumerate(getInputs())) {
    if (auto inputType = mlir::dyn_cast<StorageType>(value.getType()))
      appendLayoutRequirement(requirements, WaferValueRole::Operand, index,
                              inputType);
  }
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult ComputeElementwiseOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void ComputeElementwiseOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  for (auto [index, value] : llvm::enumerate(getInputs())) {
    appendResourceEffect(effects, WaferResourceKind::SPM,
                         WaferResourceAccess::Read, WaferValueRole::Operand,
                         index, getCompactByteSizeOrUnknown(value.getType()));
  }
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
  appendResourceEffect(effects, WaferResourceKind::Compute,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
}

mlir::LogicalResult ComputeElementwiseOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult ComputeReduceOp::verify() {
  return verifyReduceTileContract(getOperation(), getInput(),
                                  getResult().getType());
}

void ComputeReduceOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto inputType = mlir::dyn_cast<StorageType>(getInput().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            inputType);
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult ComputeReduceOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void ComputeReduceOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getInput().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
  appendResourceEffect(effects, WaferResourceKind::Compute,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
}

mlir::LogicalResult ComputeReduceOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
