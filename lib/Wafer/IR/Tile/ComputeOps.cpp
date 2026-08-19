//===- ComputeOps.cpp - Wafer Compute verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "WaferIRVerification.h"

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
  FillDomain domain = getFillDomain().value_or(FillDomain::LogicalValid);
  if (domain == FillDomain::LogicalValid &&
      !hasWaferLayout(getDest().getType(), MemLayout::Tensor))
    return emitOpError("logical_valid fill destination must use tensor layout");
  if (domain == FillDomain::PhysicalFootprint) {
    auto type = mlir::cast<mlir::MemRefType>(getDest().getType());
    std::optional<WaferPhysicalTensorInfo> info =
        computeWaferPhysicalTensorInfo(type);
    if (!info || info->physicalBytes <= 0)
      return emitOpError(
          "physical_footprint fill requires a static positive physical "
          "destination footprint");
  }

  if (getValue().getType() != destTensor->getElementType())
    return emitOpError(
        "fill value type must match destination tensor element type");
  return mlir::success();
}

mlir::LogicalResult ComputeConvertOp::verify() {
  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(getResult().getType());
  if (!sourceTensor || !resultTensor)
    return emitOpError("expects Wafer buffer source and result");
  std::optional<MemLayout> sourceLayout = getWaferLayout(getSource().getType());
  std::optional<MemLayout> resultLayout = getWaferLayout(getResult().getType());
  for (mlir::Type type : {getSource().getType(), getResult().getType()}) {
    if (!hasWaferMemorySpace(type, MemorySpace::SPM))
      return emitOpError("convert storage values must use SPM memory space");
  }
  if (!sourceLayout || !resultLayout || sourceLayout != resultLayout)
    return emitOpError(
        "convert source and result must use the same Wafer layout family");
  if (sourceTensor->getShape() != resultTensor->getShape())
    return emitOpError("convert source and result shapes must match");
  mlir::Type sourceElement = sourceTensor->getElementType();
  mlir::Type resultElement = resultTensor->getElementType();
  if (sourceElement == resultElement)
    return emitOpError("convert source and result element types must differ");
  auto isSupportedFloat = [](mlir::Type type) {
    return mlir::isa<mlir::Float16Type, mlir::BFloat16Type, mlir::Float32Type>(
        type);
  };
  if (!isSupportedFloat(sourceElement) || !isSupportedFloat(resultElement))
    return emitOpError("convert currently requires f16, bf16 or f32 element "
                       "types");
  return mlir::success();
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

  auto verifyStorage =
      [&](mlir::Type type,
          mlir::RankedTensorType tensor) -> mlir::LogicalResult {
    if (!hasWaferMemorySpace(type, MemorySpace::SPM))
      return emitOpError("gemm storage values must use SPM memory space");
    const MemLayout expectedLayout =
        tensor.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
    if (!hasWaferLayout(type, expectedLayout))
      return emitOpError(tensor.getRank() > 2
                             ? "batched gemm storage values must use ncx layout"
                             : "rank-2 gemm storage values must use cx layout");
    return mlir::success();
  };
  if (mlir::failed(verifyStorage(getLhs().getType(), *lhsTensor)) ||
      mlir::failed(verifyStorage(getRhs().getType(), *rhsTensor)) ||
      mlir::failed(verifyStorage(getResult().getType(), *resultTensor)))
    return mlir::failure();

  if (lhsTensor->getElementType() != rhsTensor->getElementType() ||
      lhsTensor->getElementType() != resultTensor->getElementType())
    return emitOpError("gemm operand and result element types must match");

  if (static_cast<bool>(getLhsOrientationAttr()) !=
      static_cast<bool>(getRhsOrientationAttr()))
    return emitOpError(
        "lhs_orientation and rhs_orientation must either both be present for "
        "oriented GEMM or both be absent for normal/normal GEMM");
  GemmOrientation lhsOrientation =
      getLhsOrientation().value_or(GemmOrientation::Normal);
  GemmOrientation rhsOrientation =
      getRhsOrientation().value_or(GemmOrientation::Normal);

  if (lhsTensor->getRank() != 2 || rhsTensor->getRank() != 2 ||
      resultTensor->getRank() != 2) {
    BatchedGemmDimAttrs attrs;
    return verifyBatchedGemmTileContract(getOperation(), *lhsTensor, *rhsTensor,
                                         *resultTensor, lhsOrientation,
                                         rhsOrientation, attrs);
  }

  if (hasAnyBatchedGemmAttrs(getOperation()))
    return emitOpError("gemm rank-2 form must not carry batched GEMM attrs");

  int64_t lhsMDim = lhsOrientation == GemmOrientation::Normal ? 0 : 1;
  int64_t lhsKDim = lhsOrientation == GemmOrientation::Normal ? 1 : 0;
  int64_t rhsKDim = rhsOrientation == GemmOrientation::Normal ? 0 : 1;
  int64_t rhsNDim = rhsOrientation == GemmOrientation::Normal ? 1 : 0;
  if (hasStaticMismatch(lhsTensor->getDimSize(lhsKDim),
                        rhsTensor->getDimSize(rhsKDim)))
    return emitOpError("gemm lhs K dimension must match rhs K dimension");
  if (hasStaticMismatch(lhsTensor->getDimSize(lhsMDim),
                        resultTensor->getDimSize(0)) ||
      hasStaticMismatch(rhsTensor->getDimSize(rhsNDim),
                        resultTensor->getDimSize(1)))
    return emitOpError("gemm result shape must be lhs M by rhs N");

  return mlir::success();
}

mlir::LogicalResult ComputeConvOp::verify() {
  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(getInput().getType());
  std::optional<mlir::RankedTensorType> weightTensor =
      getLogicalTensorType(getWeight().getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(getResult().getType());
  if (!inputTensor || !weightTensor || !resultTensor)
    return emitOpError("expects Wafer buffer operands and result");
  for (mlir::Type type :
       {getInput().getType(), getWeight().getType(), getResult().getType()}) {
    if (!hasWaferMemorySpace(type, MemorySpace::SPM))
      return emitOpError("convolution storage values must use SPM memory "
                         "space");
    if (!hasWaferLayout(type, MemLayout::NCx))
      return emitOpError("convolution storage values must use ncx layout");
  }
  if (inputTensor->getElementType() != weightTensor->getElementType() ||
      inputTensor->getElementType() != resultTensor->getElementType())
    return emitOpError(
        "convolution operand and result element types must match");
  return verifyCanonicalConv2DGeometry(
      getOperation(), *inputTensor, *weightTensor, *resultTensor,
      getPadsAttr().asArrayRef(), getUnpadsAttr().asArrayRef(),
      getStridesAttr().asArrayRef(), getDilationsAttr().asArrayRef());
}

mlir::LogicalResult ComputeElementwiseOp::verify() {
  return verifyElementwiseTileContract(getOperation(), getKindAttr().getValue(),
                                       getInputs(), getResult().getType(),
                                       getIndexingMapsAttr());
}

mlir::LogicalResult ComputeElementwiseIntoOp::verify() {
  return verifyElementwiseTileContract(getOperation(), getKindAttr().getValue(),
                                       getInputs(), getDest().getType());
}

mlir::LogicalResult ComputeReduceOp::verify() {
  return verifyReduceTileContract(getOperation(), getInput(),
                                  getResult().getType());
}
