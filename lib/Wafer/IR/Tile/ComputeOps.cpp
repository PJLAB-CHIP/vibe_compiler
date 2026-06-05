//===- ComputeOps.cpp - Wafer Compute verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult ComputeFillOp::verify() {
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (!destTensor)
    return emitOpError("expects Wafer buffer destination");
  if (!hasWaferMemorySpace(getDest().getType(), MemorySpace::SPM))
    return emitOpError("fill destination must use SPM memory space");
  if (!hasWaferLayout(getDest().getType(), MemLayout::Tensor))
    return emitOpError("fill destination must use tensor layout");

  if (getValue().getType() != destTensor->getElementType())
    return emitOpError(
        "fill value type must match destination tensor element type");
  return mlir::success();
}

void ComputeFillOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          getDest().getType());
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
  std::optional<mlir::RankedTensorType> lhsTensor =
      getLogicalTensorType(getLhs().getType());
  std::optional<mlir::RankedTensorType> rhsTensor =
      getLogicalTensorType(getRhs().getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(getResult().getType());
  if (!lhsTensor || !rhsTensor || !resultTensor)
    return emitOpError("expects Wafer buffer operands and result");

  for (mlir::Type type :
       {getLhs().getType(), getRhs().getType(), getResult().getType()}) {
    if (!hasWaferMemorySpace(type, MemorySpace::SPM))
      return emitOpError("gemm storage values must use SPM memory space");
    if (!hasWaferLayout(type, MemLayout::Cx))
      return emitOpError("gemm storage values must use cx layout");
  }

  if (lhsTensor->getElementType() != rhsTensor->getElementType() ||
      lhsTensor->getElementType() != resultTensor->getElementType())
    return emitOpError("gemm operand and result element types must match");

  if (lhsTensor->getRank() != 2 || rhsTensor->getRank() != 2 ||
      resultTensor->getRank() != 2) {
    BatchedGemmDimAttrs attrs;
    return verifyBatchedGemmTileContract(getOperation(), *lhsTensor,
                                         *rhsTensor, *resultTensor, attrs);
  }

  if (hasAnyBatchedGemmAttrs(getOperation()))
    return emitOpError("gemm rank-2 form must not carry batched GEMM attrs");

  if (hasStaticMismatch(lhsTensor->getDimSize(1), rhsTensor->getDimSize(0)))
    return emitOpError("gemm lhs K dimension must match rhs K dimension");
  if (hasStaticMismatch(lhsTensor->getDimSize(0),
                        resultTensor->getDimSize(0)) ||
      hasStaticMismatch(rhsTensor->getDimSize(1),
                        resultTensor->getDimSize(1)))
    return emitOpError("gemm result shape must be lhs M by rhs N");

  return mlir::success();
}

void ComputeGemmOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          getLhs().getType());
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 1,
                          getRhs().getType());
  appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                          getResult().getType());
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
    appendLayoutRequirement(requirements, WaferValueRole::Operand, index,
                            value.getType());
  }
  appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                          getResult().getType());
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
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          getInput().getType());
  appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                          getResult().getType());
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
