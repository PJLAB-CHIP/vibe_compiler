//===- LayoutOps.cpp - Wafer Layout verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult LayoutMaterializeOp::verify() {
  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(getResult().getType());
  if (!sourceTensor || !resultTensor)
    return emitOpError("expects Wafer buffer source and result types");

  if (*sourceTensor != *resultTensor)
    return emitOpError("layout materialize must preserve logical tensor type");

  if (getWaferMemorySpace(getSource().getType()) !=
      getWaferMemorySpace(getResult().getType()))
    return emitOpError("layout materialize must preserve memory space");

  if (getWaferLayout(getSource().getType()) ==
      getWaferLayout(getResult().getType()))
    return emitOpError("layout materialize must change layout");

  return mlir::success();
}

void LayoutMaterializeOp::collectWaferMaterializationLayouts(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          getSource().getType());
  appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                          getResult().getType());
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
