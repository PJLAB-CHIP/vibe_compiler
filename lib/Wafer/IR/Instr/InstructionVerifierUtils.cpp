//===- InstructionVerifierUtils.cpp - Shared instruction verification -----===//

#include "InstructionVerifierUtils.h"

#include "OpVerifierUtils.h"

#include <limits>
#include <optional>

using namespace wafer;
using namespace wafer::detail;

namespace wafer::instr_detail {

namespace {

mlir::LogicalResult verifyWaferMemRef(mlir::Operation *op, mlir::Type type,
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

} // namespace

mlir::LogicalResult verifySPMMemRef(mlir::Operation *op, mlir::Type type,
                                    llvm::StringRef role) {
  return verifyWaferMemRef(op, type, MemorySpace::SPM, role);
}

mlir::LogicalResult verifyDDRMemRef(mlir::Operation *op, mlir::Type type,
                                    llvm::StringRef role) {
  return verifyWaferMemRef(op, type, MemorySpace::DDR, role);
}

mlir::LogicalResult verifyPositiveI64Attr(mlir::Operation *op,
                                          mlir::IntegerAttr attr,
                                          llvm::StringRef name) {
  if (attr.getInt() <= 0)
    return op->emitOpError() << name << " must be positive";
  return mlir::success();
}

mlir::LogicalResult verifyI64Array(mlir::Operation *op,
                                   mlir::DenseI64ArrayAttr attr,
                                   llvm::StringRef name, int64_t expectedSize,
                                   bool positive) {
  if (attr.size() != expectedSize)
    return op->emitOpError()
           << name << " must contain exactly " << expectedSize << " entries";
  for (int64_t value : attr.asArrayRef()) {
    if (positive && value <= 0)
      return op->emitOpError() << name << " entries must be positive";
    if (!positive && value < 0)
      return op->emitOpError() << name << " entries must be non-negative";
  }
  return mlir::success();
}

mlir::LogicalResult verifyUInt32Value(mlir::Operation *op, int64_t value,
                                      llvm::StringRef name) {
  if (value < 0 ||
      static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max())
    return op->emitOpError()
           << "target_abi_narrowing: " << name << " must fit uint32_t";
  return mlir::success();
}

mlir::LogicalResult verifyUInt16Value(mlir::Operation *op, int64_t value,
                                      llvm::StringRef name) {
  if (value < 0 ||
      static_cast<uint64_t>(value) > std::numeric_limits<uint16_t>::max())
    return op->emitOpError()
           << "target_abi_narrowing: " << name << " must fit uint16_t";
  return mlir::success();
}

mlir::LogicalResult verifyUInt32Array(mlir::Operation *op,
                                      mlir::DenseI64ArrayAttr values,
                                      llvm::StringRef name) {
  if (!values)
    return mlir::success();
  for (int64_t value : values.asArrayRef())
    if (mlir::failed(verifyUInt32Value(op, value, name)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult verifyUInt16Array(mlir::Operation *op,
                                      mlir::DenseI64ArrayAttr values,
                                      llvm::StringRef name) {
  if (!values)
    return mlir::success();
  for (int64_t value : values.asArrayRef()) {
    if (value < 0 ||
        static_cast<uint64_t>(value) > std::numeric_limits<uint16_t>::max())
      return op->emitOpError() << "target_abi_narrowing: " << name
                               << " entries must fit uint16_t";
  }
  return mlir::success();
}

mlir::LogicalResult verifyShapeAttrMatchesBuffer(mlir::Operation *op,
                                                 mlir::Type type,
                                                 mlir::DenseI64ArrayAttr shape,
                                                 llvm::StringRef name) {
  std::optional<mlir::RankedTensorType> tensor = getLogicalTensorType(type);
  if (!tensor || !tensor->hasStaticShape() || tensor->getRank() > 4)
    return op->emitOpError()
           << "target_geometry_mismatch: " << name
           << " requires a static Wafer buffer of rank at most 4";
  if (!shape || shape.size() != 4)
    return op->emitOpError() << "target_geometry_mismatch: " << name
                             << " must contain exactly 4 entries";
  int64_t leadingOnes = 4 - tensor->getRank();
  for (int64_t index = 0; index < 4; ++index) {
    int64_t expected =
        index < leadingOnes ? 1 : tensor->getDimSize(index - leadingOnes);
    if (shape.asArrayRef()[index] != expected)
      return op->emitOpError() << "target_geometry_mismatch: " << name
                               << " must match the logical buffer shape";
  }
  return verifyUInt16Array(op, shape, name);
}

mlir::FailureOr<int64_t>
computeWindowedOutputDim(mlir::Operation *op, int64_t input, int64_t kernel,
                         int64_t stride, int64_t dilation, int64_t padBefore,
                         int64_t padAfter, int64_t unpadBefore,
                         int64_t unpadAfter, llvm::StringRef role) {
  int64_t padded = 0;
  int64_t effectiveKernel = 0;
  int64_t dilatedSpan = 0;
  if (!checkedAdd(input, padBefore, padded) ||
      !checkedAdd(padded, padAfter, padded) ||
      !checkedMul(kernel - 1, dilation, dilatedSpan) ||
      !checkedAdd(dilatedSpan, 1, effectiveKernel))
    return op->emitOpError()
           << "target_range_overflow: " << role << " geometry overflows int64";
  if (stride <= 0 || padded < effectiveKernel)
    return op->emitOpError() << "target_geometry_mismatch: " << role
                             << " kernel exceeds the padded input";

  int64_t output = (padded - effectiveKernel) / stride + 1;
  int64_t totalUnpad = 0;
  if (!checkedAdd(unpadBefore, unpadAfter, totalUnpad) || output <= totalUnpad)
    return op->emitOpError() << "target_geometry_mismatch: " << role
                             << " unpadding removes the complete output";
  return output - totalUnpad;
}

mlir::LogicalResult verifySameElementType(mlir::Operation *op,
                                          mlir::RankedTensorType lhs,
                                          mlir::RankedTensorType rhs,
                                          llvm::StringRef message) {
  if (lhs.getElementType() != rhs.getElementType())
    return op->emitOpError() << message;
  return mlir::success();
}

mlir::LogicalResult verifyOptionalUInt32Attr(mlir::Operation *op,
                                             mlir::IntegerAttr attr,
                                             llvm::StringRef name) {
  if (!attr)
    return mlir::success();
  int64_t value = attr.getInt();
  if (value < 0 || value > std::numeric_limits<uint32_t>::max())
    return op->emitOpError()
           << "target_abi_narrowing: " << name << " must fit uint32_t";
  return mlir::success();
}

mlir::LogicalResult verifyOptionalRoundingMode(mlir::Operation *op,
                                               mlir::IntegerAttr attr,
                                               llvm::StringRef name) {
  if (!attr)
    return mlir::success();
  int64_t value = attr.getInt();
  if (value < 0 || value > 4)
    return op->emitOpError() << name << " must be a RND_MODE value in [0, 4]";
  return mlir::success();
}

void appendInstructionIssueEffect(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects,
    WaferResourceKind resource, int64_t bytes) {
  appendResourceEffect(effects, resource, WaferResourceAccess::Issue,
                       WaferValueRole::None, 0, bytes);
}

} // namespace wafer::instr_detail
