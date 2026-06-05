//===- SPMOps.cpp - Wafer SPM verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult StorageAllocOp::verify() {
  auto resultType = mlir::dyn_cast<StorageType>(getResult().getType());
  if (!resultType)
    return emitOpError("expects storage result");
  if (!hasStorageMemorySpace(resultType, MemorySpace::SPM))
    return emitOpError("result must use SPM memory space");
  return mlir::success();
}

void StorageAllocOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult StorageAllocOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 1> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void StorageAllocOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
}

mlir::LogicalResult StorageAllocOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 1> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult StorageLoadOp::verify() {
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(getSource().getType());
  auto resultType = mlir::dyn_cast<StorageType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitOpError("expects ranked tensor source and storage result");

  if (resultType.getTensorType() != sourceType)
    return emitOpError(
        "tile.load result tensor type must match source tensor type");
  if (!hasStorageMemorySpace(resultType, MemorySpace::SPM))
    return emitOpError("tile.load result must use SPM memory space");
  if (!hasStorageLayout(resultType, MemLayout::Tensor))
    return emitOpError("tile.load result must use tensor mem_layout");

  return mlir::success();
}

void StorageLoadOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult StorageLoadOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void StorageLoadOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::DDR,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
  appendResourceEffect(effects, WaferResourceKind::Movement,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
}

mlir::LogicalResult StorageLoadOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult StorageStoreOp::verify() {
  auto sourceType = mlir::dyn_cast<StorageType>(getSource().getType());
  auto destType = mlir::dyn_cast<mlir::RankedTensorType>(getDest().getType());
  if (!sourceType || !destType)
    return emitOpError("expects storage source and ranked tensor dest");

  if (sourceType.getTensorType() != destType)
    return emitOpError(
        "tile.store source tensor type must match dest tensor type");
  if (!hasStorageMemorySpace(sourceType, MemorySpace::SPM))
    return emitOpError("tile.store source must use SPM memory space");
  if (!hasStorageLayout(sourceType, MemLayout::Tensor))
    return emitOpError("tile.store source must use tensor mem_layout for "
                       "external writeback");

  return mlir::success();
}

void StorageStoreOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto sourceType = mlir::dyn_cast<StorageType>(getSource().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            sourceType);
}

mlir::LogicalResult StorageStoreOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void StorageStoreOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::DDR,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendResourceEffect(effects, WaferResourceKind::Movement,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
}

mlir::LogicalResult StorageStoreOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
