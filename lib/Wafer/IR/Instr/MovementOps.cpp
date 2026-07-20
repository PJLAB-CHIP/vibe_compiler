//===- MovementOps.cpp - Wafer movement instruction operations -----------===//

#include "Wafer/IR/WaferDialect.h"

#include "InstructionVerifierUtils.h"
#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>
#include <utility>

using namespace wafer;
using namespace wafer::detail;
using namespace wafer::instr_detail;

namespace {

static mlir::LogicalResult
verifyNonNegativeOptionalI64Attr(mlir::Operation *op, mlir::IntegerAttr attr,
                                 llvm::StringRef name) {
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

static mlir::LogicalResult
verifyPermutationI64Array(mlir::Operation *op, mlir::DenseI64ArrayAttr attr,
                          llvm::StringRef name) {
  if (mlir::failed(verifyI64Array(op, attr, name, 4, /*positive=*/false)))
    return mlir::failure();
  bool seen[4] = {false, false, false, false};
  for (int64_t value : attr.asArrayRef()) {
    if (value >= 4)
      return op->emitOpError()
             << name << " entries must be in the range [0, 3]";
    if (seen[value])
      return op->emitOpError() << name << " entries must form a permutation";
    seen[value] = true;
  }
  return mlir::success();
}

static mlir::LogicalResult verifyAxesI64Array(mlir::Operation *op,
                                              mlir::DenseI64ArrayAttr attr,
                                              llvm::StringRef name,
                                              int64_t expectedSize) {
  if (expectedSize > 0 && attr.size() != expectedSize)
    return op->emitOpError()
           << name << " must contain exactly " << expectedSize << " entries";
  if (expectedSize < 0 && attr.empty())
    return op->emitOpError() << name << " must contain at least one entry";

  bool seen[4] = {false, false, false, false};
  for (int64_t value : attr.asArrayRef()) {
    if (value < 0 || value >= 4)
      return op->emitOpError()
             << name << " entries must be in the range [0, 3]";
    if (seen[value])
      return op->emitOpError() << name << " entries must be unique";
    seen[value] = true;
  }
  return mlir::success();
}

static mlir::LogicalResult verifyAxesWithinRank(mlir::Operation *op,
                                                mlir::DenseI64ArrayAttr attr,
                                                llvm::StringRef name,
                                                int64_t sourceRank,
                                                int64_t destRank) {
  for (int64_t value : attr.asArrayRef()) {
    if (value >= sourceRank || value >= destRank)
      return op->emitOpError()
             << name << " entries must be within source/dest tensor rank";
  }
  return mlir::success();
}

static int64_t getOptionalI64AttrValue(mlir::IntegerAttr attr) {
  return attr ? attr.getInt() : 0;
}

static mlir::FailureOr<int64_t>
getMovementDescriptorEnd(mlir::Operation *op, int64_t offset,
                         int64_t innerBytes, mlir::DenseI64ArrayAttr strides,
                         mlir::DenseI64ArrayAttr iterations,
                         llvm::StringRef role) {
  int64_t end = 0;
  if (!checkedAdd(offset, innerBytes, end))
    return op->emitOpError() << "target_range_overflow: " << role
                             << " descriptor byte range overflows int64";

  if (!strides || !iterations)
    return end;

  for (auto [stride, iteration] :
       llvm::zip(strides.asArrayRef(), iterations.asArrayRef())) {
    int64_t span = 0;
    if (!checkedMul(stride, iteration - 1, span) || !checkedAdd(end, span, end))
      return op->emitOpError() << "target_range_overflow: " << role
                               << " descriptor byte range overflows int64";
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
    return op->emitOpError() << "target_geometry_mismatch: " << role
                             << " physical byte size must be statically known";

  mlir::FailureOr<int64_t> end = getMovementDescriptorEnd(
      op, getOptionalI64AttrValue(offsetAttr), innerBytesAttr.getInt(), strides,
      iterations, role);
  if (mlir::failed(end))
    return mlir::failure();
  if (*end > info->physicalBytes)
    return op->emitOpError()
           << "target_geometry_mismatch: " << role
           << " descriptor byte range exceeds physical byte size";
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
  if (static_cast<bool>(srcStrides) != static_cast<bool>(srcIterations))
    return op->emitOpError(
        "src_strides and src_iterations must either both be present or both "
        "be absent");
  if (static_cast<bool>(dstStrides) != static_cast<bool>(dstIterations))
    return op->emitOpError(
        "dst_strides and dst_iterations must either both be present or both "
        "be absent");
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

  auto verifyPayload = [&](mlir::DenseI64ArrayAttr iterations,
                           llvm::StringRef role) -> mlir::LogicalResult {
    if (!iterations)
      return mlir::success();
    int64_t payloadBytes = innerBytes.getInt();
    for (int64_t iteration : iterations.asArrayRef()) {
      if (!checkedMul(payloadBytes, iteration, payloadBytes))
        return op->emitOpError()
               << "target_range_overflow: " << role
               << " descriptor payload byte count overflows int64";
    }
    if (payloadBytes != byteCount.getInt())
      return op->emitOpError() << "target_geometry_mismatch: " << role
                               << " descriptor payload must equal byte_count";
    return mlir::success();
  };

  if (mlir::failed(verifyPayload(srcIterations, "source")) ||
      mlir::failed(verifyPayload(dstIterations, "destination")))
    return mlir::failure();
  return mlir::success();
}

static mlir::LogicalResult
verifyBytesWithinPhysicalRange(mlir::Operation *op, mlir::Type type,
                               mlir::IntegerAttr offsetAttr, int64_t bytes,
                               llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return mlir::success();
  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes < 0)
    return op->emitOpError() << "target_geometry_mismatch: " << role
                             << " physical byte size must be statically known";
  int64_t end = 0;
  if (!checkedAdd(getOptionalI64AttrValue(offsetAttr), bytes, end))
    return op->emitOpError() << "target_range_overflow: " << role
                             << " byte range overflows int64";
  if (end > info->physicalBytes)
    return op->emitOpError() << "target_geometry_mismatch: " << role
                             << " byte count exceeds physical byte size";
  return mlir::success();
}

static mlir::LogicalResult
verifyPadShapeRelation(mlir::Operation *op, mlir::DenseI64ArrayAttr sourceShape,
                       mlir::DenseI64ArrayAttr destShape,
                       mlir::DenseI64ArrayAttr pads) {
  llvm::ArrayRef<int64_t> source = sourceShape.asArrayRef();
  llvm::ArrayRef<int64_t> dest = destShape.asArrayRef();
  llvm::ArrayRef<int64_t> pad = pads.asArrayRef();
  int64_t expectedH = 0;
  int64_t expectedW = 0;
  if (!checkedAdd(source[1], pad[0], expectedH) ||
      !checkedAdd(expectedH, pad[1], expectedH) ||
      !checkedAdd(source[2], pad[2], expectedW) ||
      !checkedAdd(expectedW, pad[3], expectedW))
    return op->emitOpError(
        "target_range_overflow: pad destination shape overflows int64");
  if (dest[0] != source[0] || dest[1] != expectedH || dest[2] != expectedW ||
      dest[3] != source[3])
    return op->emitOpError(
        "target_geometry_mismatch: pad destination shape does not match "
        "source and pads");
  return mlir::success();
}

static mlir::LogicalResult verifyImg2ColShapeRelation(
    mlir::Operation *op, mlir::DenseI64ArrayAttr sourceShape,
    mlir::DenseI64ArrayAttr destShape, mlir::DenseI64ArrayAttr pads,
    mlir::DenseI64ArrayAttr kernelStrides) {
  llvm::ArrayRef<int64_t> source = sourceShape.asArrayRef();
  llvm::ArrayRef<int64_t> dest = destShape.asArrayRef();
  llvm::ArrayRef<int64_t> pad = pads.asArrayRef();
  llvm::ArrayRef<int64_t> kernelStride = kernelStrides.asArrayRef();
  mlir::FailureOr<int64_t> expectedH = computeWindowedOutputDim(
      op, source[1], kernelStride[0], kernelStride[2], /*dilation=*/1, pad[0],
      pad[1], /*unpadBefore=*/0, /*unpadAfter=*/0, "img2col height");
  mlir::FailureOr<int64_t> expectedW = computeWindowedOutputDim(
      op, source[2], kernelStride[1], kernelStride[3], /*dilation=*/1, pad[2],
      pad[3], /*unpadBefore=*/0, /*unpadAfter=*/0, "img2col width");
  int64_t expectedC = 0;
  int64_t kernelElements = 0;
  if (mlir::failed(expectedH) || mlir::failed(expectedW))
    return mlir::failure();
  if (!checkedMul(kernelStride[0], kernelStride[1], kernelElements) ||
      !checkedMul(source[3], kernelElements, expectedC))
    return op->emitOpError(
        "target_range_overflow: img2col destination channels overflow int64");
  if (dest[0] != source[0] || dest[1] != *expectedH || dest[2] != *expectedW ||
      dest[3] != expectedC)
    return op->emitOpError(
        "target_geometry_mismatch: img2col destination shape does not match "
        "source/kernel/stride/pad");
  return mlir::success();
}

static mlir::LogicalResult verifyMovementDescriptorTargetWidths(
    mlir::Operation *op, mlir::IntegerAttr byteCount,
    mlir::IntegerAttr innerBytes, mlir::DenseI64ArrayAttr strides,
    mlir::DenseI64ArrayAttr iterations) {
  if (mlir::failed(verifyUInt32Value(op, byteCount.getInt(), "byte_count")) ||
      mlir::failed(verifyUInt32Value(op, innerBytes.getInt(), "inner_bytes")) ||
      mlir::failed(verifyUInt32Array(op, strides, "descriptor stride")) ||
      mlir::failed(verifyUInt32Array(op, iterations, "descriptor iteration")))
    return mlir::failure();
  return mlir::success();
}

static std::optional<int64_t> getTargetElementStorageBytes(mlir::Type type) {
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    switch (integerType.getWidth()) {
    case 1:
    case 8:
      return 1;
    case 16:
      return 2;
    case 32:
      return 4;
    case 64:
      return 8;
    default:
      return std::nullopt;
    }
  }
  if (mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(type))
    return 2;
  if (mlir::isa<mlir::Float32Type>(type))
    return 4;
  return std::nullopt;
}

static mlir::LogicalResult
verifyMovementElementContract(mlir::Operation *op, mlir::Type sourceType,
                              mlir::Type destType, mlir::IntegerAttr innerBytes,
                              bool targetFormatCarriesElementType) {
  std::optional<mlir::RankedTensorType> source =
      getLogicalTensorType(sourceType);
  std::optional<mlir::RankedTensorType> dest = getLogicalTensorType(destType);
  if (!source || !dest)
    return op->emitOpError(
        "target_geometry_mismatch: movement operands must be ranked Wafer "
        "buffers");
  if (source->getElementType() != dest->getElementType())
    return op->emitOpError(
        "target_geometry_mismatch: movement source and destination element "
        "types must match");
  if (!targetFormatCarriesElementType)
    return mlir::success();

  if (source->getElementType().isInteger(1)) {
    constexpr int64_t maxBoolInnerBytes =
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max() / 8U);
    if (innerBytes.getInt() > maxBoolInnerBytes)
      return op->emitOpError(
          "target_abi_narrowing: bitpacked BOOL inner_bytes cannot be "
          "converted to a uint32_t logical element count");
    return mlir::success();
  }

  std::optional<int64_t> elementBytes =
      getTargetElementStorageBytes(source->getElementType());
  if (!elementBytes)
    return op->emitOpError(
        "target_abi_narrowing: movement element type is not encodable by the "
        "target data-format ABI");
  if (innerBytes.getInt() % *elementBytes != 0)
    return op->emitOpError(
        "target_geometry_mismatch: inner_bytes must be divisible by the "
        "target element byte width");
  return mlir::success();
}

} // namespace

mlir::LogicalResult InstrRDMAOp::verify() {
  if ((*this)->hasAttr("dst_strides") || (*this)->hasAttr("dst_iterations"))
    return emitOpError(
        "RDMA destination is sequential and must not carry destination "
        "stride fields");
  if (mlir::failed(
          verifyDDRMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  if (static_cast<bool>(getSrcOffsetAttr()) !=
      static_cast<bool>(getDstOffsetAttr()))
    return emitOpError(
        "src_offset and dst_offset must either both be present for mapped "
        "DMA or both be absent for compact DMA");
  if (mlir::failed(verifyNonNegativeOptionalI64Attr(
          getOperation(), getSrcOffsetAttr(), "src_offset")) ||
      mlir::failed(verifyNonNegativeOptionalI64Attr(
          getOperation(), getDstOffsetAttr(), "dst_offset")) ||
      mlir::failed(verifyMovementDescriptor(
          getOperation(), getByteCountAttr(), getInnerBytesAttr(),
          getSrcStridesAttr(), getSrcIterationsAttr(), {}, {})) ||
      mlir::failed(verifyMovementElementContract(
          getOperation(), getSource().getType(), getDest().getType(),
          getInnerBytesAttr(), /*targetFormatCarriesElementType=*/true)) ||
      mlir::failed(verifyMovementDescriptorTargetWidths(
          getOperation(), getByteCountAttr(), getInnerBytesAttr(),
          getSrcStridesAttr(), getSrcIterationsAttr())) ||
      mlir::failed(verifyDescriptorWithinPhysicalRange(
          getOperation(), getSource().getType(), getSrcOffsetAttr(),
          getInnerBytesAttr(), getSrcStridesAttr(), getSrcIterationsAttr(),
          "source")) ||
      mlir::failed(verifyBytesWithinPhysicalRange(
          getOperation(), getDest().getType(), getDstOffsetAttr(),
          getByteCountAttr().getInt(), "destination")))
    return mlir::failure();
  return mlir::success();
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
  if ((*this)->hasAttr("src_strides") || (*this)->hasAttr("src_iterations"))
    return emitOpError(
        "WDMA source is sequential and must not carry source stride fields");
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifyDDRMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  if (static_cast<bool>(getSrcOffsetAttr()) !=
      static_cast<bool>(getDstOffsetAttr()))
    return emitOpError(
        "src_offset and dst_offset must either both be present for mapped "
        "DMA or both be absent for compact DMA");
  if (mlir::failed(verifyNonNegativeOptionalI64Attr(
          getOperation(), getSrcOffsetAttr(), "src_offset")) ||
      mlir::failed(verifyNonNegativeOptionalI64Attr(
          getOperation(), getDstOffsetAttr(), "dst_offset")) ||
      mlir::failed(verifyMovementDescriptor(
          getOperation(), getByteCountAttr(), getInnerBytesAttr(), {}, {},
          getDstStridesAttr(), getDstIterationsAttr())) ||
      mlir::failed(verifyMovementElementContract(
          getOperation(), getSource().getType(), getDest().getType(),
          getInnerBytesAttr(), /*targetFormatCarriesElementType=*/true)) ||
      mlir::failed(verifyMovementDescriptorTargetWidths(
          getOperation(), getByteCountAttr(), getInnerBytesAttr(),
          getDstStridesAttr(), getDstIterationsAttr())) ||
      mlir::failed(verifyBytesWithinPhysicalRange(
          getOperation(), getSource().getType(), getSrcOffsetAttr(),
          getByteCountAttr().getInt(), "source")) ||
      mlir::failed(verifyDescriptorWithinPhysicalRange(
          getOperation(), getDest().getType(), getDstOffsetAttr(),
          getInnerBytesAttr(), getDstStridesAttr(), getDstIterationsAttr(),
          "destination")))
    return mlir::failure();
  return mlir::success();
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
      mlir::failed(verifyMovementElementContract(
          getOperation(), getSource().getType(), getDest().getType(),
          getInnerBytesAttr(), /*targetFormatCarriesElementType=*/false)) ||
      mlir::failed(verifyMovementDescriptorTargetWidths(
          getOperation(), getByteCountAttr(), getInnerBytesAttr(),
          getSrcStridesAttr(), getSrcIterationsAttr())) ||
      mlir::failed(verifyMovementDescriptorTargetWidths(
          getOperation(), getByteCountAttr(), getInnerBytesAttr(),
          getDstStridesAttr(), getDstIterationsAttr())) ||
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

mlir::LogicalResult InstrTDMADataMoveOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")) ||
      mlir::failed(verifyI64Array(getOperation(), getSourceShapeAttr(),
                                  "source_shape", 4, /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getDestShapeAttr(),
                                  "dest_shape", 4, /*positive=*/true)))
    return mlir::failure();

  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (mlir::failed(
          verifyShapeAttrMatchesBuffer(getOperation(), getSource().getType(),
                                       getSourceShapeAttr(), "source_shape")) ||
      mlir::failed(
          verifyShapeAttrMatchesBuffer(getOperation(), getDest().getType(),
                                       getDestShapeAttr(), "dest_shape")))
    return mlir::failure();
  if (getPermutationAttr() &&
      mlir::failed(verifyPermutationI64Array(
          getOperation(), getPermutationAttr(), "permutation")))
    return mlir::failure();
  if (getAxesAttr() &&
      mlir::failed(verifyAxesI64Array(getOperation(), getAxesAttr(), "axes",
                                      /*expectedSize=*/-1)))
    return mlir::failure();
  if (getAxesAttr() && mlir::failed(verifyAxesWithinRank(
                           getOperation(), getAxesAttr(), "axes",
                           sourceTensor->getRank(), destTensor->getRank())))
    return mlir::failure();
  if (getPadsAttr() &&
      mlir::failed(verifyI64Array(getOperation(), getPadsAttr(), "pads", 4,
                                  /*positive=*/false)))
    return mlir::failure();
  if (getKernelStridesAttr() &&
      mlir::failed(verifyI64Array(getOperation(), getKernelStridesAttr(),
                                  "kernel_strides", 4,
                                  /*positive=*/true)))
    return mlir::failure();
  if (mlir::failed(verifyUInt16Array(getOperation(), getPadsAttr(), "pads")) ||
      mlir::failed(verifyUInt16Array(getOperation(), getKernelStridesAttr(),
                                     "kernel_strides")))
    return mlir::failure();

  switch (getKindAttr().getValue()) {
  case InstrDataMoveKind::Mirror:
    if (sourceTensor->getRank() != destTensor->getRank())
      return emitOpError("mirror data_move requires rank-compatible operands");
    if (!getAxesAttr())
      return emitOpError("mirror data_move requires axes attr");
    if (getPermutationAttr())
      return emitOpError("mirror data_move must not have permutation attr");
    if (getPadsAttr())
      return emitOpError("mirror data_move must not have pads attr");
    if (getKernelStridesAttr())
      return emitOpError("mirror data_move must not have kernel_strides attr");
    break;
  case InstrDataMoveKind::Rotate90:
  case InstrDataMoveKind::Rotate180:
  case InstrDataMoveKind::Rotate270:
    if (sourceTensor->getRank() != destTensor->getRank())
      return emitOpError("rotate data_move requires rank-compatible operands");
    if (!getAxesAttr())
      return emitOpError("rotate data_move requires axes attr");
    if (mlir::failed(verifyAxesI64Array(getOperation(), getAxesAttr(), "axes",
                                        /*expectedSize=*/2)))
      return mlir::failure();
    if (getPermutationAttr())
      return emitOpError("rotate data_move must not have permutation attr");
    if (getPadsAttr())
      return emitOpError("rotate data_move must not have pads attr");
    if (getKernelStridesAttr())
      return emitOpError("rotate data_move must not have kernel_strides attr");
    break;
  case InstrDataMoveKind::Nchw2Nhwc:
  case InstrDataMoveKind::Nhwc2Nchw:
  case InstrDataMoveKind::TensorNom:
    if (getPermutationAttr())
      return emitOpError(
          "non-parameterized data_move must not have permutation attr");
    if (getPadsAttr())
      return emitOpError("non-parameterized data_move must not have pads attr");
    if (getKernelStridesAttr())
      return emitOpError(
          "non-parameterized data_move must not have kernel_strides attr");
    if (getAxesAttr())
      return emitOpError("non-parameterized data_move must not have axes attr");
    break;
  case InstrDataMoveKind::Transpose:
    if (!getPermutationAttr())
      return emitOpError("transpose data_move requires permutation attr");
    if (getAxesAttr())
      return emitOpError("transpose data_move must not have axes attr");
    if (getPadsAttr())
      return emitOpError("transpose data_move must not have pads attr");
    if (getKernelStridesAttr())
      return emitOpError(
          "transpose data_move must not have kernel_strides attr");
    break;
  case InstrDataMoveKind::Pad:
    if (!getPadsAttr())
      return emitOpError("pad data_move requires pads attr");
    if (getPermutationAttr())
      return emitOpError("pad data_move must not have permutation attr");
    if (getKernelStridesAttr())
      return emitOpError("pad data_move must not have kernel_strides attr");
    if (getAxesAttr())
      return emitOpError("pad data_move must not have axes attr");
    if (mlir::failed(verifyPadShapeRelation(getOperation(),
                                            getSourceShapeAttr(),
                                            getDestShapeAttr(), getPadsAttr())))
      return mlir::failure();
    break;
  case InstrDataMoveKind::Img2Col:
    if (!getPadsAttr())
      return emitOpError("img2col data_move requires pads attr");
    if (!getKernelStridesAttr())
      return emitOpError("img2col data_move requires kernel_strides attr");
    if (getPermutationAttr())
      return emitOpError("img2col data_move must not have permutation attr");
    if (getAxesAttr())
      return emitOpError("img2col data_move must not have axes attr");
    if (mlir::failed(verifyImg2ColShapeRelation(
            getOperation(), getSourceShapeAttr(), getDestShapeAttr(),
            getPadsAttr(), getKernelStridesAttr())))
      return mlir::failure();
    break;
  }

  return verifySameElementType(getOperation(), *sourceTensor, *destTensor,
                               "tdma_data_move source and dest element types "
                               "must match");
}

InstrFamily InstrTDMADataMoveOp::getInstructionFamily() {
  return InstrFamily::TDMA;
}

mlir::LogicalResult InstrTDMADataMoveOp::verifyInstructionContract() {
  return verify();
}

void InstrTDMADataMoveOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Movement,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrTDMADataMoveOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
