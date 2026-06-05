//===- LayoutOps.cpp - Wafer Layout verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult LayoutMaterializeOp::verify() {
  auto sourceType = mlir::dyn_cast<StorageType>(getSource().getType());
  auto resultType = mlir::dyn_cast<StorageType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitOpError("expects storage source and result types");

  if (sourceType.getTensorType() != resultType.getTensorType())
    return emitOpError("layout materialize must preserve logical tensor type");

  if (getStorageMemorySpace(sourceType).getValue() !=
      getStorageMemorySpace(resultType).getValue())
    return emitOpError("layout materialize must preserve memory space");

  if (getStorageLayout(sourceType).getValue() ==
      getStorageLayout(resultType).getValue())
    return emitOpError("layout materialize must change mem_layout");

  return mlir::success();
}

void LayoutMaterializeOp::collectWaferMaterializationLayouts(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto sourceType = mlir::dyn_cast<StorageType>(getSource().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            sourceType);
  if (auto resultType = mlir::dyn_cast<StorageType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult
LayoutMaterializeOp::verifyWaferLayoutMaterializationContract() {
  llvm::SmallVector<WaferLayoutRequirement, 4> requirements;
  collectWaferMaterializationLayouts(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

void LayoutMaterializeOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Result, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
  appendResourceEffect(effects, WaferResourceKind::Movement,
                       WaferResourceAccess::Issue, WaferValueRole::None, 0,
                       getCompactByteSizeOrUnknown(getResult().getType()));
}

mlir::LogicalResult LayoutMaterializeOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
