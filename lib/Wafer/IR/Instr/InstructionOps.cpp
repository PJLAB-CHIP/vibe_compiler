//===- InstructionOps.cpp - Wafer instruction verifier implementation ----===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

namespace {

static mlir::LogicalResult verifyWaferMemRef(mlir::Operation *op,
                                             mlir::Type type,
                                             MemorySpace memorySpace,
                                             llvm::StringRef role) {
  if (!getLogicalTensorType(type))
    return op->emitOpError() << role << " must be a Wafer memref";
  if (!hasWaferMemorySpace(type, memorySpace)) {
    return op->emitOpError()
           << role << " must be a "
           << (memorySpace == MemorySpace::SPM ? "SPM" : "DDR")
           << " Wafer memref";
  }
  return mlir::success();
}

static mlir::LogicalResult verifySPMMemRef(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef role) {
  return verifyWaferMemRef(op, type, MemorySpace::SPM, role);
}

static mlir::LogicalResult verifyDDRMemRef(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef role) {
  return verifyWaferMemRef(op, type, MemorySpace::DDR, role);
}

static mlir::LogicalResult verifyPositiveI64Attr(mlir::Operation *op,
                                                 mlir::IntegerAttr attr,
                                                 llvm::StringRef name) {
  if (attr.getInt() <= 0)
    return op->emitOpError() << name << " must be positive";
  return mlir::success();
}

static mlir::LogicalResult verifyNonNegativeOptionalI64Attr(
    mlir::Operation *op, mlir::IntegerAttr attr, llvm::StringRef name) {
  if (attr && attr.getInt() < 0)
    return op->emitOpError() << name << " must be non-negative";
  return mlir::success();
}

static mlir::LogicalResult verifyDescriptorArray(mlir::Operation *op,
                                                 mlir::DenseI64ArrayAttr attr,
                                                 llvm::StringRef name,
                                                 bool positive) {
  if (attr.size() != 3)
    return op->emitOpError() << name << " must contain exactly 3 entries";
  for (int64_t value : attr.asArrayRef()) {
    if (positive && value <= 0)
      return op->emitOpError() << name << " entries must be positive";
    if (!positive && value < 0)
      return op->emitOpError() << name << " entries must be non-negative";
  }
  return mlir::success();
}

static int64_t getOptionalI64AttrValue(mlir::IntegerAttr attr) {
  return attr ? attr.getInt() : 0;
}

static mlir::FailureOr<int64_t> getMovementDescriptorEnd(
    mlir::Operation *op, int64_t offset, int64_t innerBytes,
    mlir::DenseI64ArrayAttr strides, mlir::DenseI64ArrayAttr iterations,
    llvm::StringRef role) {
  int64_t end = 0;
  if (!checkedAdd(offset, innerBytes, end))
    return op->emitOpError()
           << role << " descriptor byte range overflows int64";

  if (!strides || !iterations)
    return end;

  for (auto [stride, iteration] :
       llvm::zip(strides.asArrayRef(), iterations.asArrayRef())) {
    int64_t span = 0;
    if (!checkedMul(stride, iteration - 1, span) ||
        !checkedAdd(end, span, end))
      return op->emitOpError()
             << role << " descriptor byte range overflows int64";
  }
  return end;
}

static mlir::LogicalResult verifyDescriptorWithinPhysicalRange(
    mlir::Operation *op, mlir::Type type, mlir::IntegerAttr offsetAttr,
    mlir::IntegerAttr innerBytesAttr, mlir::DenseI64ArrayAttr strides,
    mlir::DenseI64ArrayAttr iterations, llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return mlir::success();

  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes < 0)
    return mlir::success();

  mlir::FailureOr<int64_t> end = getMovementDescriptorEnd(
      op, getOptionalI64AttrValue(offsetAttr), innerBytesAttr.getInt(), strides,
      iterations, role);
  if (mlir::failed(end))
    return mlir::failure();
  if (*end > info->physicalBytes)
    return op->emitOpError()
           << role << " descriptor byte range exceeds physical byte size";
  return mlir::success();
}

static mlir::LogicalResult verifyMovementDescriptor(
    mlir::Operation *op, mlir::IntegerAttr byteCount,
    mlir::IntegerAttr innerBytes, mlir::DenseI64ArrayAttr srcStrides,
    mlir::DenseI64ArrayAttr srcIterations, mlir::DenseI64ArrayAttr dstStrides,
    mlir::DenseI64ArrayAttr dstIterations) {
  if (mlir::failed(verifyPositiveI64Attr(op, byteCount, "byte_count")) ||
      mlir::failed(verifyPositiveI64Attr(op, innerBytes, "inner_bytes")))
    return mlir::failure();
  if (innerBytes.getInt() > byteCount.getInt())
    return op->emitOpError("inner_bytes must not exceed byte_count");
  if (srcStrides &&
      mlir::failed(verifyDescriptorArray(op, srcStrides, "src_strides",
                                         /*positive=*/false)))
    return mlir::failure();
  if (srcIterations &&
      mlir::failed(verifyDescriptorArray(op, srcIterations, "src_iterations",
                                         /*positive=*/true)))
    return mlir::failure();
  if (dstStrides &&
      mlir::failed(verifyDescriptorArray(op, dstStrides, "dst_strides",
                                         /*positive=*/false)))
    return mlir::failure();
  if (dstIterations &&
      mlir::failed(verifyDescriptorArray(op, dstIterations, "dst_iterations",
                                         /*positive=*/true)))
    return mlir::failure();
  return mlir::success();
}

static bool isAlignedGemmLayout(MemLayout layout) {
  return layout == MemLayout::Cx || layout == MemLayout::NCx;
}

static mlir::LogicalResult verifyAlignedSPMGemmMemRef(mlir::Operation *op,
                                                      mlir::Type type,
                                                      llvm::StringRef role) {
  if (mlir::failed(verifySPMMemRef(op, type, role)))
    return mlir::failure();
  std::optional<MemLayout> layout = getWaferLayout(type);
  if (!layout || !isAlignedGemmLayout(*layout))
    return op->emitOpError("lhs, rhs and dest must use aligned SPM layouts");
  return mlir::success();
}

static mlir::LogicalResult verifySameElementType(mlir::Operation *op,
                                                 mlir::RankedTensorType lhs,
                                                 mlir::RankedTensorType rhs,
                                                 llvm::StringRef message) {
  if (lhs.getElementType() != rhs.getElementType())
    return op->emitOpError() << message;
  return mlir::success();
}

static mlir::LogicalResult verifyInstructionReduceContract(mlir::Operation *op,
                                                           mlir::Value input,
                                                           mlir::Value dest,
                                                           mlir::Value init) {
  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(input.getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(dest.getType());
  if (!inputTensor)
    return op->emitOpError("expects Wafer buffer input");
  if (!destTensor)
    return op->emitOpError("expects Wafer buffer dest");

  if (!hasWaferMemorySpace(input.getType(), MemorySpace::SPM) ||
      !hasWaferMemorySpace(dest.getType(), MemorySpace::SPM))
    return op->emitOpError("reduce operands/dest must use SPM memory space");

  MemLayout expectedInputLayout =
      inputTensor->getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  MemLayout expectedDestLayout =
      destTensor->getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  if (!hasWaferLayout(input.getType(), expectedInputLayout) ||
      !hasWaferLayout(dest.getType(), expectedDestLayout)) {
    if (inputTensor->getRank() > 2 || destTensor->getRank() > 2)
      return op->emitOpError(
          "reduce rank > 2 operands/dest must use ncx layout");
    return op->emitOpError("reduce rank <= 2 operands/dest must use cx layout");
  }

  if (inputTensor->getElementType() != destTensor->getElementType())
    return op->emitOpError(
        "reduce input element type must match dest element type");

  auto dimensions = op->getAttrOfType<mlir::DenseI64ArrayAttr>("dimensions");
  if (!dimensions)
    return op->emitOpError("requires reduce dimensions attr");
  llvm::ArrayRef<int64_t> dims = dimensions.asArrayRef();
  if (dims.empty())
    return op->emitOpError("reduce dimensions must be non-empty");
  if (static_cast<int64_t>(dims.size()) > inputTensor->getRank())
    return op->emitOpError("reduce dimensions cannot exceed input tensor rank");

  llvm::DenseSet<int64_t> reducedDims;
  for (int64_t dim : dims) {
    if (dim < 0 || dim >= inputTensor->getRank())
      return op->emitOpError(
          "reduce dimensions must be within input tensor rank");
    if (!reducedDims.insert(dim).second)
      return op->emitOpError("reduce dimensions must be unique");
  }

  if (destTensor->getRank() !=
      inputTensor->getRank() - static_cast<int64_t>(dims.size()))
    return op->emitOpError(
        "reduce dest rank must match input rank minus reduce dimensions");

  int64_t destDim = 0;
  for (int64_t inputDim = 0; inputDim < inputTensor->getRank(); ++inputDim) {
    if (reducedDims.contains(inputDim))
      continue;
    if (hasStaticMismatch(inputTensor->getDimSize(inputDim),
                          destTensor->getDimSize(destDim)))
      return op->emitOpError(
          "reduce dest shape must match non-reduced input dimensions");
    ++destDim;
  }

  if (init && init.getType() != inputTensor->getElementType())
    return op->emitOpError(
        "reduce init operand type must match input element type");

  mlir::Attribute initValue = op->getAttr("init_value");
  if (initValue) {
    auto typedInit = mlir::dyn_cast<mlir::TypedAttr>(initValue);
    if (!typedInit || typedInit.getType() != inputTensor->getElementType())
      return op->emitOpError(
          "reduce init_value type must match input element type");
  }

  return mlir::success();
}

static mlir::Type getMemRefElementType(mlir::Type type) {
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type))
    return memrefType.getElementType();
  return {};
}

static void appendInstructionIssueEffect(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects,
    WaferResourceKind resource, int64_t bytes) {
  appendResourceEffect(effects, resource, WaferResourceAccess::Issue,
                       WaferValueRole::None, 0, bytes);
}

static mlir::LogicalResult verifyNoEmptyVariadicInputs(mlir::Operation *op,
                                                       mlir::ValueRange inputs,
                                                       llvm::StringRef name) {
  if (inputs.empty())
    return op->emitOpError() << name << " requires at least one input";
  return mlir::success();
}

} // namespace

mlir::LogicalResult InstrRDMAOp::verify() {
  if (mlir::failed(
          verifyDDRMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  return verifyMovementDescriptor(getOperation(), getByteCountAttr(),
                                  getInnerBytesAttr(), getSrcStridesAttr(),
                                  getSrcIterationsAttr(), {}, {});
}

InstrFamily InstrRDMAOp::getInstructionFamily() { return InstrFamily::RDMA; }

mlir::LogicalResult InstrRDMAOp::verifyInstructionContract() {
  return verify();
}

void InstrRDMAOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::DDR,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getByteCountAttr().getInt());
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getByteCountAttr().getInt());
  appendInstructionIssueEffect(effects, WaferResourceKind::Movement,
                               getByteCountAttr().getInt());
}

mlir::LogicalResult InstrRDMAOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrWDMAOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifyDDRMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  return verifyMovementDescriptor(getOperation(), getByteCountAttr(),
                                  getInnerBytesAttr(), {}, {},
                                  getDstStridesAttr(), getDstIterationsAttr());
}

InstrFamily InstrWDMAOp::getInstructionFamily() { return InstrFamily::WDMA; }

mlir::LogicalResult InstrWDMAOp::verifyInstructionContract() {
  return verify();
}

void InstrWDMAOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getByteCountAttr().getInt());
  appendResourceEffect(effects, WaferResourceKind::DDR,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getByteCountAttr().getInt());
  appendInstructionIssueEffect(effects, WaferResourceKind::Movement,
                               getByteCountAttr().getInt());
}

mlir::LogicalResult InstrWDMAOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrGatherScatterOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  if (mlir::failed(verifyNonNegativeOptionalI64Attr(
          getOperation(), getSrcOffsetAttr(), "src_offset")) ||
      mlir::failed(verifyNonNegativeOptionalI64Attr(
          getOperation(), getDstOffsetAttr(), "dst_offset")) ||
      mlir::failed(verifyMovementDescriptor(
          getOperation(), getByteCountAttr(), getInnerBytesAttr(),
          getSrcStridesAttr(), getSrcIterationsAttr(), getDstStridesAttr(),
          getDstIterationsAttr())) ||
      mlir::failed(verifyDescriptorWithinPhysicalRange(
          getOperation(), getSource().getType(), getSrcOffsetAttr(),
          getInnerBytesAttr(), getSrcStridesAttr(), getSrcIterationsAttr(),
          "source")) ||
      mlir::failed(verifyDescriptorWithinPhysicalRange(
          getOperation(), getDest().getType(), getDstOffsetAttr(),
          getInnerBytesAttr(), getDstStridesAttr(), getDstIterationsAttr(),
          "dest")))
    return mlir::failure();
  return mlir::success();
}

InstrFamily InstrGatherScatterOp::getInstructionFamily() {
  return InstrFamily::TDMA;
}

mlir::LogicalResult InstrGatherScatterOp::verifyInstructionContract() {
  return verify();
}

void InstrGatherScatterOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getByteCountAttr().getInt());
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getByteCountAttr().getInt());
  appendInstructionIssueEffect(effects, WaferResourceKind::Movement,
                               getByteCountAttr().getInt());
}

mlir::LogicalResult InstrGatherScatterOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrFillOp::verify() {
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  if (getValue().getType() != destTensor->getElementType())
    return emitOpError("fill value type must match dest element type");
  return mlir::success();
}

InstrFamily InstrFillOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrFillOp::verifyInstructionContract() {
  return verify();
}

void InstrFillOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrFillOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 2> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrElementwiseOp::verify() {
  if (mlir::failed(verifyNoEmptyVariadicInputs(getOperation(), getInputs(),
                                               "elementwise")))
    return mlir::failure();
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    if (mlir::failed(verifySPMMemRef(getOperation(), input.getType(), "input")))
      return mlir::failure();
  }
  return verifyElementwiseTileContract(getOperation(), getKindAttr().getValue(),
                                       getInputs(), getDest().getType());
}

InstrFamily InstrElementwiseOp::getInstructionFamily() {
  return InstrFamily::CT;
}

mlir::LogicalResult InstrElementwiseOp::verifyInstructionContract() {
  return verify();
}

void InstrElementwiseOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    appendResourceEffect(effects, WaferResourceKind::SPM,
                         WaferResourceAccess::Read, WaferValueRole::Operand,
                         index, getCompactByteSizeOrUnknown(input.getType()));
  }
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand,
                       getInputs().size(),
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrElementwiseOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrReduceOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getInput().getType(), "input")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  return verifyInstructionReduceContract(getOperation(), getInput(), getDest(),
                                         getInit());
}

InstrFamily InstrReduceOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrReduceOp::verifyInstructionContract() {
  return verify();
}

void InstrReduceOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getInput().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrReduceOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrConvertOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  mlir::Type srcElement = getMemRefElementType(getSource().getType());
  mlir::Type dstElement = getMemRefElementType(getDest().getType());
  if (getSrcDtype() != srcElement)
    return emitOpError("src_dtype must match source element type");
  if (getDstDtype() != dstElement)
    return emitOpError("dst_dtype must match dest element type");
  return mlir::success();
}

InstrFamily InstrConvertOp::getInstructionFamily() {
  return InstrFamily::CT;
}

mlir::LogicalResult InstrConvertOp::verifyInstructionContract() {
  return verify();
}

void InstrConvertOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrConvertOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrGemmOp::verify() {
  if (mlir::failed(verifyAlignedSPMGemmMemRef(getOperation(),
                                              getLhs().getType(), "lhs")) ||
      mlir::failed(verifyAlignedSPMGemmMemRef(getOperation(),
                                              getRhs().getType(), "rhs")) ||
      mlir::failed(verifyAlignedSPMGemmMemRef(getOperation(),
                                              getDest().getType(), "dest")))
    return mlir::failure();

  std::optional<mlir::RankedTensorType> lhsTensor =
      getLogicalTensorType(getLhs().getType());
  std::optional<mlir::RankedTensorType> rhsTensor =
      getLogicalTensorType(getRhs().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (mlir::failed(
          verifySameElementType(getOperation(), *lhsTensor, *rhsTensor,
                                "gemm lhs and rhs element types must match")) ||
      mlir::failed(verifySameElementType(
          getOperation(), *lhsTensor, *destTensor,
          "gemm operand and dest element types must match")))
    return mlir::failure();

  int64_t m = getMAttr().getInt();
  int64_t k = getKAttr().getInt();
  int64_t n = getNAttr().getInt();
  if (m <= 0 || k <= 0 || n <= 0)
    return emitOpError("m, k and n must be positive");

  if (lhsTensor->getRank() == 2 && rhsTensor->getRank() == 2 &&
      destTensor->getRank() == 2) {
    if (hasAnyBatchedGemmAttrs(getOperation()))
      return emitOpError("rank-2 gemm must not carry batched GEMM attrs");
    if (hasStaticMismatch(lhsTensor->getDimSize(0), m) ||
        hasStaticMismatch(lhsTensor->getDimSize(1), k) ||
        hasStaticMismatch(rhsTensor->getDimSize(0), k) ||
        hasStaticMismatch(rhsTensor->getDimSize(1), n) ||
        hasStaticMismatch(destTensor->getDimSize(0), m) ||
        hasStaticMismatch(destTensor->getDimSize(1), n))
      return emitOpError("m/k/n attrs must match GEMM operand shapes");
    return mlir::success();
  }

  BatchedGemmDimAttrs attrs;
  if (mlir::failed(verifyBatchedGemmTileContract(
          getOperation(), *lhsTensor, *rhsTensor, *destTensor, attrs)))
    return mlir::failure();
  if (hasStaticMismatch(attrs.lhsMDim >= 0
                            ? lhsTensor->getDimSize(attrs.lhsMDim)
                            : mlir::ShapedType::kDynamic,
                        m) ||
      hasStaticMismatch(attrs.lhsContractingDim >= 0
                            ? lhsTensor->getDimSize(attrs.lhsContractingDim)
                            : mlir::ShapedType::kDynamic,
                        k) ||
      hasStaticMismatch(attrs.rhsNDim >= 0
                            ? rhsTensor->getDimSize(attrs.rhsNDim)
                            : mlir::ShapedType::kDynamic,
                        n))
    return emitOpError("m/k/n attrs must match batched GEMM dimensions");
  return mlir::success();
}

InstrFamily InstrGemmOp::getInstructionFamily() { return InstrFamily::NE; }

mlir::LogicalResult InstrGemmOp::verifyInstructionContract() {
  return verify();
}

void InstrGemmOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getLhs().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getRhs().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 2,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrGemmOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
