//===- WaferDialect.cpp - Wafer dialect implementation -------------------===//

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <limits>
#include <optional>

using namespace wafer;

#include "Wafer/IR/WaferEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Wafer/IR/WaferAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Wafer/IR/WaferTypes.cpp.inc"

#include "Wafer/IR/WaferOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Wafer/IR/WaferOps.cpp.inc"

namespace {

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static std::optional<int64_t> getElementBitWidth(mlir::Type elementType) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
    return floatType.getWidth();
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType))
    return integerType.getWidth();
  if (mlir::isa<mlir::IndexType>(elementType))
    return 64;
  if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(elementType)) {
    std::optional<int64_t> elementBits =
        getElementBitWidth(complexType.getElementType());
    if (!elementBits)
      return std::nullopt;
    int64_t complexBits = 0;
    if (!checkedMul(*elementBits, 2, complexBits))
      return std::nullopt;
    return complexBits;
  }
  return std::nullopt;
}

static int64_t ceilDivToBytes(int64_t bits) {
  return bits / 8 + (bits % 8 == 0 ? 0 : 1);
}

static std::optional<int64_t> getElementStorageBytes(mlir::Type elementType) {
  std::optional<int64_t> elementBits = getElementBitWidth(elementType);
  if (!elementBits || *elementBits <= 0)
    return std::nullopt;
  return ceilDivToBytes(*elementBits);
}

static std::optional<int64_t> alignTo(int64_t value, int64_t alignment) {
  if (value < 0 || alignment <= 0)
    return std::nullopt;
  int64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  int64_t delta = alignment - remainder;
  int64_t result = 0;
  if (!checkedAdd(value, delta, result))
    return std::nullopt;
  return result;
}

static std::optional<int64_t>
getStaticElementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t elements = 1;
  for (int64_t dim : shape) {
    if (dim == mlir::ShapedType::kDynamic)
      return std::nullopt;
    int64_t next = 0;
    if (!checkedMul(elements, dim, next))
      return std::nullopt;
    elements = next;
  }
  return elements;
}

static std::optional<int64_t> getStaticProduct(llvm::ArrayRef<int64_t> shape) {
  int64_t product = 1;
  for (int64_t dim : shape) {
    if (dim == mlir::ShapedType::kDynamic)
      return std::nullopt;
    int64_t next = 0;
    if (!checkedMul(product, dim, next))
      return std::nullopt;
    product = next;
  }
  return product;
}

static bool usesInt8ChannelBlock(mlir::Type elementType) {
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType);
  return integerType && integerType.getWidth() == 8;
}

static int64_t alignCxTail(int64_t tail) {
  if (tail <= 4)
    return 4;
  if (tail <= 8)
    return 8;
  if (tail <= 16)
    return 16;
  if (tail <= 32)
    return 32;
  return 64;
}

static bool computeCxC0(int64_t c, int64_t cBlock, int64_t retainThreshold,
                        int64_t &cxBlocks, int64_t &c0, int64_t &alignedC) {
  if (c < 0 || cBlock <= 0 || retainThreshold <= 0)
    return false;
  int64_t q = c / cBlock;
  int64_t r = c % cBlock;
  if (r == 0) {
    cxBlocks = q;
    c0 = 0;
    return checkedMul(q, cBlock, alignedC);
  }
  if (r <= retainThreshold) {
    cxBlocks = q;
    c0 = alignCxTail(r);
    int64_t full = 0;
    if (!checkedMul(q, cBlock, full))
      return false;
    return checkedAdd(full, c0, alignedC);
  }
  cxBlocks = q + 1;
  c0 = 0;
  return checkedMul(cxBlocks, cBlock, alignedC);
}

static std::optional<int64_t>
getStaticPhysicalElementCount(llvm::ArrayRef<int64_t> shape,
                              mlir::Type elementType, MemLayout layout,
                              WaferPhysicalTensorInfo &info) {
  if (shape.empty())
    return getStaticElementCount(shape);
  if (layout != MemLayout::Cx && layout != MemLayout::NCx)
    return getStaticElementCount(shape);

  int64_t c = shape.back();
  if (c == mlir::ShapedType::kDynamic)
    return std::nullopt;

  int64_t cBlock = usesInt8ChannelBlock(elementType) ? 128 : 64;
  int64_t retainThreshold = usesInt8ChannelBlock(elementType) ? 64 : 32;
  int64_t cxBlocks = 0;
  int64_t c0 = 0;
  int64_t alignedC = 0;
  if (!computeCxC0(c, cBlock, retainThreshold, cxBlocks, c0, alignedC))
    return std::nullopt;

  std::optional<int64_t> elementBytes = getElementStorageBytes(elementType);
  if (!elementBytes || *elementBytes <= 0 || 256 % *elementBytes != 0)
    return std::nullopt;
  int64_t bankAlignElements = 256 / *elementBytes;

  info.cBlock = cBlock;
  info.cxBlocks = cxBlocks;
  info.c0 = c0;
  info.alignedC = alignedC;
  info.tailC = c0;
  info.elementBytes = *elementBytes;
  info.bankAlignElements = bankAlignElements;

  int64_t physicalElements = 0;
  if (layout == MemLayout::NCx) {
    int64_t n = 1;
    llvm::ArrayRef<int64_t> hwShape;
    if (shape.size() > 1) {
      n = shape.front();
      hwShape = shape.drop_front().drop_back();
    }
    if (n == mlir::ShapedType::kDynamic)
      return std::nullopt;
    std::optional<int64_t> hwElements = getStaticProduct(hwShape);
    if (!hwElements)
      return std::nullopt;
    int64_t unalignedBatch = 0;
    if (!checkedMul(*hwElements, alignedC, unalignedBatch))
      return std::nullopt;
    std::optional<int64_t> batchElements =
        alignTo(unalignedBatch, bankAlignElements);
    if (!batchElements)
      return std::nullopt;
    if (!checkedMul(n, *batchElements, physicalElements))
      return std::nullopt;
    int64_t outerElements = 0;
    if (!checkedMul(n, *hwElements, outerElements))
      return std::nullopt;
    info.outerElements = outerElements;
    info.hwElements = *hwElements;
    info.batchElements = *batchElements;
  } else {
    std::optional<int64_t> outerElements = getStaticProduct(shape.drop_back());
    if (!outerElements)
      return std::nullopt;
    int64_t unalignedBatch = 0;
    if (!checkedMul(*outerElements, alignedC, unalignedBatch))
      return std::nullopt;
    std::optional<int64_t> batchElements =
        alignTo(unalignedBatch, bankAlignElements);
    if (!batchElements)
      return std::nullopt;
    physicalElements = *batchElements;
    info.outerElements = *outerElements;
    info.hwElements = *outerElements;
    info.batchElements = *batchElements;
  }
  return physicalElements;
}

static std::optional<int64_t> getStaticByteSize(mlir::Type elementType,
                                                int64_t elementCount) {
  std::optional<int64_t> elementBits = getElementBitWidth(elementType);
  if (!elementBits || *elementBits <= 0)
    return std::nullopt;
  int64_t totalBits = 0;
  if (!checkedMul(elementCount, *elementBits, totalBits))
    return std::nullopt;
  return ceilDivToBytes(totalBits);
}

static std::optional<int64_t>
getStaticStridedElementSpan(mlir::MemRefType type) {
  if (!type.hasStaticShape())
    return std::nullopt;

  llvm::SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (mlir::failed(mlir::getStridesAndOffset(type, strides, offset)))
    return std::nullopt;
  if (static_cast<int64_t>(strides.size()) != type.getRank())
    return std::nullopt;

  if (type.getRank() == 0)
    return 1;

  int64_t span = 1;
  for (auto [dim, stride] : llvm::zip(type.getShape(), strides)) {
    if (dim == 0)
      return 0;
    if (dim < 0 || stride == mlir::ShapedType::kDynamic || stride < 0)
      return std::nullopt;
    int64_t dimSpan = 0;
    if (!checkedMul(dim - 1, stride, dimSpan) ||
        !checkedAdd(span, dimSpan, span))
      return std::nullopt;
  }
  return span;
}

static std::optional<int64_t> linearizeIndex(llvm::ArrayRef<int64_t> shape,
                                             llvm::ArrayRef<int64_t> indices) {
  if (shape.size() != indices.size())
    return std::nullopt;
  int64_t linear = 0;
  for (auto [dim, index] : llvm::zip_equal(shape, indices)) {
    if (dim == mlir::ShapedType::kDynamic || dim < 0 || index < 0 ||
        index >= dim)
      return std::nullopt;
    int64_t scaled = 0;
    if (!checkedMul(linear, dim, scaled))
      return std::nullopt;
    if (!checkedAdd(scaled, index, linear))
      return std::nullopt;
  }
  return linear;
}

} // namespace

MemoryAttr wafer::getWaferMemoryAttr(mlir::MemRefType type) {
  return mlir::dyn_cast_or_null<MemoryAttr>(type.getMemorySpace());
}

bool wafer::isWaferMemRefType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  return memrefType && static_cast<bool>(getWaferMemoryAttr(memrefType));
}

bool wafer::isWaferSPMMemRefType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return false;
  MemoryAttr memory = getWaferMemoryAttr(memrefType);
  return memory && memory.getSpace() == MemorySpace::SPM;
}

bool wafer::isWaferDDRMemRefType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return false;
  MemoryAttr memory = getWaferMemoryAttr(memrefType);
  return memory && memory.getSpace() == MemorySpace::DDR;
}

std::optional<WaferPhysicalTensorInfo>
wafer::computeWaferPhysicalTensorInfo(mlir::MemRefType type) {
  MemoryAttr memory = getWaferMemoryAttr(type);
  if (!memory)
    return std::nullopt;

  WaferPhysicalTensorInfo info;
  info.logicalTensorType =
      mlir::RankedTensorType::get(type.getShape(), type.getElementType());
  info.memorySpace = memory.getSpace();
  info.layout = memory.getLayout();
  if (auto integerType =
          mlir::dyn_cast<mlir::IntegerType>(type.getElementType()))
    info.bitPackedElement = integerType.getWidth() == 1;
  if (std::optional<int64_t> elementBytes =
          getElementStorageBytes(type.getElementType()))
    info.elementBytes = *elementBytes;

  std::optional<int64_t> compactElements =
      getStaticElementCount(type.getShape());
  if (compactElements) {
    if (std::optional<int64_t> bytes =
            getStaticByteSize(type.getElementType(), *compactElements))
      info.compactBytes = *bytes;
  }

  std::optional<int64_t> physicalElements;
  if (info.layout == MemLayout::Tensor || info.layout == MemLayout::NTensor)
    physicalElements = getStaticStridedElementSpan(type);
  else
    physicalElements = getStaticPhysicalElementCount(
        type.getShape(), type.getElementType(), info.layout, info);
  if (physicalElements) {
    info.physicalElements = *physicalElements;
    if (std::optional<int64_t> bytes =
            getStaticByteSize(type.getElementType(), *physicalElements))
      info.physicalBytes = *bytes;
  }

  return info;
}

std::optional<int64_t> wafer::computeWaferPhysicalElementByteOffset(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> logicalIndices) {
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info || info->elementBytes <= 0 || info->bitPackedElement)
    return std::nullopt;
  if (type.getRank() != static_cast<int64_t>(logicalIndices.size()))
    return std::nullopt;
  for (auto [dim, index] : llvm::zip_equal(type.getShape(), logicalIndices)) {
    if (dim == mlir::ShapedType::kDynamic || dim < 0 || index < 0 ||
        index >= dim)
      return std::nullopt;
  }

  if (info->layout != MemLayout::Cx && info->layout != MemLayout::NCx) {
    llvm::SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (mlir::failed(mlir::getStridesAndOffset(type, strides, offset)) ||
        strides.size() != logicalIndices.size())
      return std::nullopt;

    int64_t linear = 0;
    for (auto [index, stride] : llvm::zip_equal(logicalIndices, strides)) {
      if (stride == mlir::ShapedType::kDynamic || stride < 0)
        return std::nullopt;
      int64_t scaled = 0;
      if (!checkedMul(index, stride, scaled) ||
          !checkedAdd(linear, scaled, linear))
        return std::nullopt;
    }

    int64_t byteOffset = 0;
    if (!checkedMul(linear, info->elementBytes, byteOffset))
      return std::nullopt;
    return byteOffset;
  }

  if (type.getRank() == 0)
    return std::nullopt;
  int64_t logicalC = logicalIndices.back();
  int64_t fullC = 0;
  if (!checkedMul(info->cxBlocks, info->cBlock, fullC))
    return std::nullopt;
  int64_t physicalElementOffset = 0;
  if (info->layout == MemLayout::NCx) {
    int64_t n = type.getRank() > 1 ? logicalIndices.front() : 0;
    llvm::ArrayRef<int64_t> hwShape;
    llvm::ArrayRef<int64_t> hwIndices;
    if (type.getRank() > 1) {
      hwShape = type.getShape().drop_front().drop_back();
      hwIndices = logicalIndices.drop_front().drop_back();
    }
    std::optional<int64_t> hwIndex = linearizeIndex(hwShape, hwIndices);
    if (!hwIndex)
      return std::nullopt;
    int64_t batchBase = 0;
    if (!checkedMul(n, info->batchElements, batchBase))
      return std::nullopt;
    if (logicalC < fullC) {
      int64_t cb = logicalC / info->cBlock;
      int64_t lane = logicalC % info->cBlock;
      int64_t blockOffset = 0;
      int64_t blockStride = 0;
      if (!checkedMul(info->hwElements, info->cBlock, blockStride) ||
          !checkedMul(cb, blockStride, blockOffset))
        return std::nullopt;
      int64_t hwOffset = 0;
      if (!checkedMul(*hwIndex, info->cBlock, hwOffset))
        return std::nullopt;
      if (!checkedAdd(batchBase, blockOffset, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, hwOffset, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, lane, physicalElementOffset))
        return std::nullopt;
    } else {
      if (info->c0 <= 0)
        return std::nullopt;
      int64_t tailLane = logicalC - fullC;
      int64_t fullBlockElements = 0;
      int64_t fullBlockStride = 0;
      if (!checkedMul(info->hwElements, info->cBlock, fullBlockStride) ||
          !checkedMul(info->cxBlocks, fullBlockStride, fullBlockElements))
        return std::nullopt;
      int64_t hwOffset = 0;
      if (!checkedMul(*hwIndex, info->c0, hwOffset))
        return std::nullopt;
      if (!checkedAdd(batchBase, fullBlockElements, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, hwOffset, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, tailLane, physicalElementOffset))
        return std::nullopt;
    }
  } else {
    llvm::ArrayRef<int64_t> outerShape = type.getShape().drop_back();
    llvm::ArrayRef<int64_t> outerIndices = logicalIndices.drop_back();
    std::optional<int64_t> outerIndex =
        linearizeIndex(outerShape, outerIndices);
    if (!outerIndex)
      return std::nullopt;
    if (logicalC < fullC) {
      int64_t cb = logicalC / info->cBlock;
      int64_t lane = logicalC % info->cBlock;
      int64_t blockOffset = 0;
      int64_t blockStride = 0;
      if (!checkedMul(info->outerElements, info->cBlock, blockStride) ||
          !checkedMul(cb, blockStride, blockOffset))
        return std::nullopt;
      int64_t outerOffset = 0;
      if (!checkedMul(*outerIndex, info->cBlock, outerOffset))
        return std::nullopt;
      if (!checkedAdd(blockOffset, outerOffset, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, lane, physicalElementOffset))
        return std::nullopt;
    } else {
      if (info->c0 <= 0)
        return std::nullopt;
      int64_t tailLane = logicalC - fullC;
      int64_t fullBlockElements = 0;
      int64_t fullBlockStride = 0;
      if (!checkedMul(info->outerElements, info->cBlock, fullBlockStride) ||
          !checkedMul(info->cxBlocks, fullBlockStride, fullBlockElements))
        return std::nullopt;
      int64_t outerOffset = 0;
      if (!checkedMul(*outerIndex, info->c0, outerOffset))
        return std::nullopt;
      if (!checkedAdd(fullBlockElements, outerOffset, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, tailLane, physicalElementOffset))
        return std::nullopt;
    }
  }

  int64_t byteOffset = 0;
  if (!checkedMul(physicalElementOffset, info->elementBytes, byteOffset))
    return std::nullopt;
  return byteOffset;
}

mlir::LogicalResult
DTEMessageAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                       int64_t communicationId, DTEProtocolPhase phase,
                       int64_t round, int64_t payloadSlice) {
  (void)phase;
  if (communicationId < 0)
    return emitError() << "dte_message communication id must be non-negative";
  if (round < 0)
    return emitError() << "dte_message protocol round must be non-negative";
  if (payloadSlice < 0)
    return emitError() << "dte_message payload slice must be non-negative";
  return mlir::success();
}

mlir::LogicalResult DirectDTEBindingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    DTEAllocationProfile allocationProfile, int64_t receiverFsmId,
    int64_t remoteReceiverOffset, DTECompletionProfile completionProfile) {
  (void)allocationProfile;
  (void)completionProfile;
  if (receiverFsmId < 0 || receiverFsmId > 3)
    return emitError()
           << "direct_dte_binding receiver FSM id must be within [0, 3]";
  if (remoteReceiverOffset < 0)
    return emitError()
           << "direct_dte_binding remote receiver offset must be non-negative";
  return mlir::success();
}

void WaferDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Wafer/IR/WaferAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Wafer/IR/WaferTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Wafer/IR/WaferOps.cpp.inc"
      >();
}
