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

template <typename GemmOp>
static mlir::LogicalResult verifyGemm(GemmOp op, mlir::Value output) {
  if (auto partial = op.getPsum()) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(partial.getType());
    auto outputType = mlir::dyn_cast<mlir::MemRefType>(output.getType());
    if (!type || !outputType || !type.getElementType().isF32() ||
        type.getShape() != outputType.getShape() ||
        type.getLayout() != outputType.getLayout() ||
        type.getMemorySpace() != outputType.getMemorySpace())
      return op.emitOpError(
          "psum must be F32 with destination shape and layout");
  }
  std::optional<mlir::RankedTensorType> lhsTensor =
      getLogicalTensorType(op.getLhs().getType());
  std::optional<mlir::RankedTensorType> rhsTensor =
      getLogicalTensorType(op.getRhs().getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(output.getType());
  if (!lhsTensor || !rhsTensor || !resultTensor)
    return op.emitOpError("expects Wafer buffer operands and result");

  auto verifyStorage =
      [&](mlir::Type type,
          mlir::RankedTensorType tensor) -> mlir::LogicalResult {
    if (!hasWaferMemorySpace(type, MemorySpace::SPM))
      return op.emitOpError("gemm storage values must use SPM memory space");
    const MemLayout expectedLayout =
        tensor.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
    if (!hasWaferLayout(type, expectedLayout))
      return op.emitOpError(
          tensor.getRank() > 2
              ? "batched gemm storage values must use ncx layout"
              : "rank-2 gemm storage values must use cx layout");
    return mlir::success();
  };
  if (mlir::failed(verifyStorage(op.getLhs().getType(), *lhsTensor)) ||
      mlir::failed(verifyStorage(op.getRhs().getType(), *rhsTensor)) ||
      mlir::failed(verifyStorage(output.getType(), *resultTensor)))
    return mlir::failure();

  auto inputType = lhsTensor->getElementType();
  auto outputType = resultTensor->getElementType();
  if (inputType != rhsTensor->getElementType() ||
      (inputType != outputType &&
       !((inputType.isF16() || inputType.isBF16()) && outputType.isF32())))
    return op.emitOpError("gemm requires equal input types and the same output "
                          "type or f16/bf16 inputs with f32 output");

  if (static_cast<bool>(op.getLhsOrientationAttr()) !=
      static_cast<bool>(op.getRhsOrientationAttr()))
    return op.emitOpError(
        "lhs_orientation and rhs_orientation must either both be present for "
        "oriented GEMM or both be absent for normal/normal GEMM");
  GemmOrientation lhsOrientation =
      op.getLhsOrientation().value_or(GemmOrientation::Normal);
  GemmOrientation rhsOrientation =
      op.getRhsOrientation().value_or(GemmOrientation::Normal);

  if (lhsTensor->getRank() != 2 || rhsTensor->getRank() != 2 ||
      resultTensor->getRank() != 2) {
    BatchedGemmDimAttrs attrs;
    return verifyBatchedGemmTileContract(op.getOperation(), *lhsTensor,
                                         *rhsTensor, *resultTensor,
                                         lhsOrientation, rhsOrientation, attrs);
  }

  if (hasAnyBatchedGemmAttrs(op.getOperation()))
    return op.emitOpError("gemm rank-2 form must not carry batched GEMM attrs");

  int64_t lhsMDim = lhsOrientation == GemmOrientation::Normal ? 0 : 1;
  int64_t lhsKDim = lhsOrientation == GemmOrientation::Normal ? 1 : 0;
  int64_t rhsKDim = rhsOrientation == GemmOrientation::Normal ? 0 : 1;
  int64_t rhsNDim = rhsOrientation == GemmOrientation::Normal ? 1 : 0;
  if (hasStaticMismatch(lhsTensor->getDimSize(lhsKDim),
                        rhsTensor->getDimSize(rhsKDim)))
    return op.emitOpError("gemm lhs K dimension must match rhs K dimension");
  if (hasStaticMismatch(lhsTensor->getDimSize(lhsMDim),
                        resultTensor->getDimSize(0)) ||
      hasStaticMismatch(rhsTensor->getDimSize(rhsNDim),
                        resultTensor->getDimSize(1)))
    return op.emitOpError("gemm result shape must be lhs M by rhs N");

  return mlir::success();
}

mlir::LogicalResult ComputeGemmOp::verify() {
  return verifyGemm(*this, getResult());
}

mlir::LogicalResult ComputeGemmIntoOp::verify() {
  if (getDest() == getLhs() || getDest() == getRhs() || getDest() == getPsum())
    return emitOpError("destination must not be a GEMM input or psum");
  return verifyGemm(*this, getDest());
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
  }
  if (!hasWaferLayout(getInput().getType(), MemLayout::NCx) ||
      !hasWaferLayout(getResult().getType(), MemLayout::NCx) ||
      !hasWaferLayout(getWeight().getType(), MemLayout::Cx))
    return emitOpError(
        "convolution input/result must use ncx and weight must use cx layout");
  if (inputTensor->getElementType() != weightTensor->getElementType() ||
      (inputTensor->getElementType() != resultTensor->getElementType() &&
       !((inputTensor->getElementType().isF16() ||
          inputTensor->getElementType().isBF16()) &&
         resultTensor->getElementType().isF32())))
    return emitOpError("convolution inputs must match and result must match or "
                       "widen f16/bf16 to f32");
  return verifyCanonicalConv2DGeometry(
      getOperation(), *inputTensor, *weightTensor, *resultTensor,
      getPadsAttr().asArrayRef(), getUnpadsAttr().asArrayRef(),
      getStridesAttr().asArrayRef(), getDilationsAttr().asArrayRef());
}

mlir::LogicalResult ComputePoolOp::verify() {
  auto input = getLogicalTensorType(getInput().getType());
  auto output = getLogicalTensorType(getResult().getType());
  if (!input || !output || input->getRank() != 4 || output->getRank() != 4 ||
      !input->hasStaticShape() || !output->hasStaticShape())
    return emitOpError("requires static rank-4 NHWC buffers");
  for (mlir::Type type : {getInput().getType(), getResult().getType()})
    if (!hasWaferMemorySpace(type, MemorySpace::SPM) ||
        !hasWaferLayout(type, MemLayout::NCx))
      return emitOpError("requires NCx SPM input and result");
  if (input->getElementType() != output->getElementType() ||
      !mlir::isa<mlir::FloatType>(input->getElementType()))
    return emitOpError("requires matching floating-point element types");
  if (llvm::any_of(input->getShape(), [](int64_t size) { return size <= 0; }) ||
      llvm::any_of(output->getShape(),
                   [](int64_t size) { return size <= 0; }) ||
      getKernel().size() != 2 || getStrides().size() != 2 ||
      getDilations().size() != 2)
    return emitOpError(
        "requires positive extents and two kernel/stride/dilation dimensions");
  if (input->getDimSize(0) != output->getDimSize(0) ||
      input->getDimSize(3) != output->getDimSize(3))
    return emitOpError("must preserve batch and channel extents");
  for (unsigned axis = 0; axis != 2; ++axis) {
    int64_t kernel = getKernel()[axis], stride = getStrides()[axis];
    int64_t dilation = getDilations()[axis];
    if (kernel <= 0 || stride <= 0 || dilation <= 0)
      return emitOpError("kernel, stride and dilation must be positive");
    __int128 effective = static_cast<__int128>(kernel - 1) * dilation + 1;
    int64_t source = input->getDimSize(axis + 1);
    if (effective > source ||
        (source - static_cast<int64_t>(effective)) / stride + 1 !=
            output->getDimSize(axis + 1))
      return emitOpError(
          "result extent disagrees with its input window geometry");
  }
  return mlir::success();
}

mlir::LogicalResult ComputeElementwiseOp::verify() {
  return verifyElementwiseTileContract(getOperation(), getKindAttr().getValue(),
                                       getInputs(), getResult().getType(),
                                       getIndexingMapsAttr());
}

mlir::LogicalResult ComputeElementwiseIntoOp::verify() {
  return verifyElementwiseTileContract(getOperation(), getKindAttr().getValue(),
                                       getInputs(), getDest().getType(),
                                       getIndexingMapsAttr());
}

mlir::LogicalResult ComputeReduceOp::verify() {
  return verifyReduceTileContract(getOperation(), getInput(),
                                  getResult().getType());
}
