//===- SPMOps.cpp - Wafer SPM verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult StorageLoadOp::verify() {
  std::optional<mlir::RankedTensorType> sourceType =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(getResult().getType());
  if (!sourceType || !resultTensor)
    return emitOpError("expects DDR memref source and SPM memref result");

  if (*resultTensor != sourceType)
    return emitOpError(
        "tile.load result tensor type must match source tensor type");
  if (!hasWaferMemorySpace(getSource().getType(), MemorySpace::DDR))
    return emitOpError("tile.load source must use DDR memory space");
  if (!hasWaferLayout(getSource().getType(), MemLayout::Tensor))
    return emitOpError("tile.load source must use tensor layout");
  if (!hasWaferMemorySpace(getResult().getType(), MemorySpace::SPM))
    return emitOpError("tile.load result must use SPM memory space");
  if (!hasWaferLayout(getResult().getType(), MemLayout::Tensor))
    return emitOpError("tile.load result must use tensor layout");

  return mlir::success();
}

void StorageLoadOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          getSource().getType());
  appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                          getResult().getType());
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
  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> destType =
      getLogicalTensorType(getDest().getType());
  if (!sourceTensor || !destType)
    return emitOpError("expects SPM memref source and DDR memref dest");

  if (*sourceTensor != destType)
    return emitOpError(
        "tile.store source tensor type must match dest tensor type");
  if (!hasWaferMemorySpace(getSource().getType(), MemorySpace::SPM))
    return emitOpError("tile.store source must use SPM memory space");
  if (!hasWaferLayout(getSource().getType(), MemLayout::Tensor))
    return emitOpError("tile.store source must use tensor layout for "
                       "external writeback");
  if (!hasWaferMemorySpace(getDest().getType(), MemorySpace::DDR))
    return emitOpError("tile.store dest must use DDR memory space");
  if (!hasWaferLayout(getDest().getType(), MemLayout::Tensor))
    return emitOpError("tile.store dest must use tensor layout");

  return mlir::success();
}

void StorageStoreOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          getSource().getType());
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 1,
                          getDest().getType());
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
