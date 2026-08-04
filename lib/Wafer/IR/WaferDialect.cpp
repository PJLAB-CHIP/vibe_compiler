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
#include <numeric>
#include <optional>

using namespace wafer;

#include "Wafer/IR/WaferEnums.cpp.inc"
#include "Wafer/IR/WaferPhysicalEncodingInterfaces.cpp.inc"

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

  std::optional<int64_t> elementBits = getElementBitWidth(elementType);
  std::optional<int64_t> elementBytes = getElementStorageBytes(elementType);
  if (!elementBits || *elementBits <= 0 ||
      kWaferSPMBankLineBytes > std::numeric_limits<int64_t>::max() / int64_t{8})
    return std::nullopt;
  const int64_t bankLineBits = kWaferSPMBankLineBytes * int64_t{8};
  if (bankLineBits % *elementBits != 0)
    return std::nullopt;
  int64_t bankAlignElements = bankLineBits / *elementBits;

  info.cBlock = cBlock;
  info.cxBlocks = cxBlocks;
  info.c0 = c0;
  info.alignedC = alignedC;
  info.tailC = c0;
  info.elementBytes = elementBytes.value_or(0);
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

mlir::FailureOr<int64_t>
MemoryAttr::getPhysicalFootprintBytes(mlir::MemRefType type) const {
  if (getWaferMemoryAttr(type) != *this)
    return mlir::failure();
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info || info->physicalBytes < 0)
    return mlir::failure();
  return info->physicalBytes;
}

mlir::FailureOr<int64_t>
MemoryAttr::getMinimumAlignmentBytes(mlir::MemRefType type) const {
  if (getWaferMemoryAttr(type) != *this)
    return mlir::failure();
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info || info->physicalBytes < 0)
    return mlir::failure();
  if (getLayout() == MemLayout::Cx || getLayout() == MemLayout::NCx)
    return kWaferSPMBankLineBytes;
  if (info->bitPackedElement)
    return 1;
  std::optional<int64_t> elementBytes =
      getElementStorageBytes(type.getElementType());
  if (!elementBytes)
    return mlir::failure();
  return *elementBytes;
}

mlir::FailureOr<int64_t>
MemoryAttr::getValidElementCount(mlir::MemRefType type) const {
  if (getWaferMemoryAttr(type) != *this)
    return mlir::failure();
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  std::optional<int64_t> elements = getStaticElementCount(type.getShape());
  if (!info || info->physicalBytes < 0 || !elements)
    return mlir::failure();
  return *elements;
}

mlir::FailureOr<int64_t>
MemoryAttr::getPaddingElementCount(mlir::MemRefType type) const {
  if (getWaferMemoryAttr(type) != *this)
    return mlir::failure();
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  std::optional<int64_t> valid = getStaticElementCount(type.getShape());
  if (!info || !valid || info->physicalElements < *valid)
    return mlir::failure();
  return info->physicalElements - *valid;
}

mlir::FailureOr<int64_t>
MemoryAttr::getPhysicalElementBitWidth(mlir::MemRefType type) const {
  if (getWaferMemoryAttr(type) != *this)
    return mlir::failure();
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  std::optional<int64_t> bitWidth = getElementBitWidth(type.getElementType());
  if (!info || info->physicalBytes < 0 || !bitWidth || *bitWidth <= 0)
    return mlir::failure();
  return *bitWidth;
}

mlir::FailureOr<llvm::SmallVector<WaferPhysicalLayoutPiece, 2>>
MemoryAttr::getPhysicalLayoutPieces(mlir::MemRefType type) const {
  if (getWaferMemoryAttr(type) != *this || !type.hasStaticShape())
    return mlir::failure();
  std::optional<WaferPhysicalTensorInfo> infoStorage =
      computeWaferPhysicalTensorInfo(type);
  std::optional<int64_t> elementBits =
      getElementBitWidth(type.getElementType());
  if (!infoStorage || infoStorage->physicalBytes < 0 || !elementBits ||
      *elementBits <= 0)
    return mlir::failure();
  const WaferPhysicalTensorInfo &info = *infoStorage;
  llvm::ArrayRef<int64_t> shape = type.getShape();
  mlir::MLIRContext *context = type.getContext();

  llvm::SmallVector<WaferPhysicalLayoutPiece, 2> pieces;
  auto addPiece = [&](llvm::ArrayRef<int64_t> lower,
                      llvm::ArrayRef<int64_t> upper,
                      llvm::ArrayRef<int64_t> periods,
                      mlir::AffineExpr bitOffset) -> mlir::LogicalResult {
    if (lower.size() != shape.size() || upper.size() != shape.size() ||
        periods.size() != shape.size())
      return mlir::failure();
    for (auto [dim, bounds] : llvm::enumerate(llvm::zip(lower, upper))) {
      auto [low, high] = bounds;
      if (low < 0 || high < low || high > shape[dim])
        return mlir::failure();
      if (low == high)
        return mlir::success();
    }
    if (llvm::any_of(periods, [](int64_t period) { return period < 0; }))
      return mlir::failure();
    pieces.push_back(
        {llvm::SmallVector<int64_t, 4>(lower),
         llvm::SmallVector<int64_t, 4>(upper),
         llvm::SmallVector<int64_t, 4>(periods),
         mlir::AffineMap::get(shape.size(), 0, bitOffset, context)});
    return mlir::success();
  };

  llvm::SmallVector<int64_t, 4> lower(shape.size(), 0);
  llvm::SmallVector<int64_t, 4> upper(shape);
  llvm::SmallVector<int64_t, 4> noPeriods(shape.size(), 0);
  if (getLayout() == MemLayout::Tensor || getLayout() == MemLayout::NTensor) {
    llvm::SmallVector<int64_t, 4> elementStrides;
    int64_t ignoredViewOffset = 0;
    if (mlir::failed(mlir::getStridesAndOffset(type, elementStrides,
                                               ignoredViewOffset)) ||
        elementStrides.size() != shape.size())
      return mlir::failure();
    int64_t strideUnitBits = info.bitPackedElement ? *elementBits : 0;
    if (!info.bitPackedElement &&
        (!checkedMul(info.elementBytes, int64_t{8}, strideUnitBits) ||
         strideUnitBits <= 0))
      return mlir::failure();
    mlir::AffineExpr bitOffset = mlir::getAffineConstantExpr(0, context);
    for (auto [dim, stride] : llvm::enumerate(elementStrides)) {
      int64_t strideBits = 0;
      if (stride == mlir::ShapedType::kDynamic || stride < 0 ||
          !checkedMul(stride, strideUnitBits, strideBits))
        return mlir::failure();
      bitOffset = bitOffset + mlir::getAffineDimExpr(dim, context) * strideBits;
    }
    if (mlir::failed(addPiece(lower, upper, noPeriods, bitOffset)))
      return mlir::failure();
    return pieces;
  }

  if ((getLayout() != MemLayout::Cx && getLayout() != MemLayout::NCx) ||
      shape.empty() || info.cBlock <= 0 || info.cxBlocks < 0 || info.c0 < 0)
    return mlir::failure();

  int64_t storageBits = 0;
  int64_t fullC = 0;
  int64_t blockedOuterElements =
      getLayout() == MemLayout::NCx ? info.hwElements : info.outerElements;
  int64_t blockStrideElements = 0;
  int64_t fullBlockElements = 0;
  if (!checkedMul(info.cxBlocks, info.cBlock, fullC) ||
      !checkedMul(blockedOuterElements, info.cBlock, blockStrideElements) ||
      !checkedMul(info.cxBlocks, blockStrideElements, fullBlockElements) ||
      blockedOuterElements <= 0)
    return mlir::failure();
  storageBits = *elementBits;

  const unsigned channelDim = shape.size() - 1;
  mlir::AffineExpr channel = mlir::getAffineDimExpr(channelDim, context);
  mlir::AffineExpr outer = mlir::getAffineConstantExpr(0, context);
  const unsigned firstOuterDim = getLayout() == MemLayout::NCx ? 1 : 0;
  for (unsigned dim = firstOuterDim; dim < channelDim; ++dim)
    outer = outer * shape[dim] + mlir::getAffineDimExpr(dim, context);

  mlir::AffineExpr batchBase = mlir::getAffineConstantExpr(0, context);
  if (getLayout() == MemLayout::NCx && shape.size() > 1)
    batchBase = mlir::getAffineDimExpr(0, context) * info.batchElements;

  const int64_t logicalC = shape.back();
  const int64_t fullUpper = std::min(logicalC, fullC);
  if (fullUpper > 0) {
    llvm::SmallVector<int64_t, 4> fullUpperBounds(upper);
    llvm::SmallVector<int64_t, 4> fullPeriods(noPeriods);
    fullUpperBounds[channelDim] = fullUpper;
    fullPeriods[channelDim] = info.cBlock;
    mlir::AffineExpr physicalElements =
        batchBase + channel.floorDiv(info.cBlock) * blockStrideElements +
        outer * info.cBlock + channel % info.cBlock;
    if (mlir::failed(addPiece(lower, fullUpperBounds, fullPeriods,
                              physicalElements * storageBits)))
      return mlir::failure();
  }

  if (fullC < logicalC) {
    if (info.c0 <= 0)
      return mlir::failure();
    llvm::SmallVector<int64_t, 4> tailLowerBounds(lower);
    tailLowerBounds[channelDim] = fullC;
    mlir::AffineExpr physicalElements =
        batchBase + fullBlockElements + outer * info.c0 + channel - fullC;
    if (mlir::failed(addPiece(tailLowerBounds, upper, noPeriods,
                              physicalElements * storageBits)))
      return mlir::failure();
  }
  return pieces;
}

mlir::FailureOr<WaferPhysicalElementSpan> MemoryAttr::getPhysicalElementSpan(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> logicalIndices) const {
  if (getWaferMemoryAttr(type) != *this)
    return mlir::failure();
  std::optional<int64_t> bitOffset =
      computeWaferPhysicalElementBitOffset(type, logicalIndices);
  std::optional<int64_t> bitLength = getElementBitWidth(type.getElementType());
  if (!bitOffset || !bitLength || *bitLength <= 0)
    return mlir::failure();
  return WaferPhysicalElementSpan{*bitOffset, *bitLength};
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

  // Cx/NCx already own a block-major physical map in the memory encoding.
  // A second non-identity MemRef layout would describe a different view whose
  // composition is not representable by the current type contract. Reject it
  // instead of silently ignoring its strides/offset. Bitpacked blocked
  // buffers use the same block/tail ordering with bit-addressed lanes.
  if ((info.layout == MemLayout::Cx || info.layout == MemLayout::NCx) &&
      (type.getRank() == 0 || !type.getLayout().isIdentity()))
    return std::nullopt;

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

mlir::FailureOr<int64_t> wafer::computeWaferRequiredAlignmentBytes(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> additionalRequirements) {
  auto encoding = mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
      type.getMemorySpace());
  if (!encoding)
    return mlir::failure();
  mlir::FailureOr<int64_t> natural = encoding.getMinimumAlignmentBytes(type);
  if (mlir::failed(natural) || *natural <= 0)
    return mlir::failure();

  int64_t combined = *natural;
  for (int64_t requirement : additionalRequirements) {
    if (requirement <= 0)
      return mlir::failure();
    int64_t scaled = combined / std::gcd(combined, requirement);
    if (scaled > std::numeric_limits<int64_t>::max() / requirement)
      return mlir::failure();
    combined = scaled * requirement;
  }
  return combined;
}

std::optional<int64_t> wafer::computeWaferPhysicalElementByteOffset(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> logicalIndices) {
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info)
    return std::nullopt;
  return computeWaferPhysicalElementByteOffset(type, *info, logicalIndices);
}

std::optional<WaferStaticPhysicalOffsetCalculator>
WaferStaticPhysicalOffsetCalculator::create(mlir::MemRefType type) {
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info || info->physicalBytes <= 0 || info->elementBytes <= 0 ||
      info->bitPackedElement)
    return std::nullopt;

  llvm::SmallVector<int64_t, 4> shape(type.getShape());
  if (llvm::any_of(shape, [](int64_t dim) {
        return dim == mlir::ShapedType::kDynamic || dim <= 0;
      }))
    return std::nullopt;

  auto multiply = [](int64_t lhs, int64_t rhs) -> std::optional<int64_t> {
    int64_t result = 0;
    if (!checkedMul(lhs, rhs, result))
      return std::nullopt;
    return result;
  };

  llvm::SmallVector<int64_t, 4> byteStrides;
  llvm::SmallVector<int64_t, 4> linearStrides;
  int64_t fullC = 0;
  int64_t blockStrideBytes = 0;
  int64_t fullBlockBytes = 0;
  if (info->layout != MemLayout::Cx && info->layout != MemLayout::NCx) {
    llvm::SmallVector<int64_t, 4> elementStrides;
    int64_t offset = 0;
    if (mlir::failed(mlir::getStridesAndOffset(type, elementStrides, offset)) ||
        elementStrides.size() != static_cast<size_t>(type.getRank()))
      return std::nullopt;
    byteStrides.reserve(elementStrides.size());
    for (int64_t stride : elementStrides) {
      if (stride == mlir::ShapedType::kDynamic || stride < 0)
        return std::nullopt;
      std::optional<int64_t> byteStride = multiply(stride, info->elementBytes);
      if (!byteStride)
        return std::nullopt;
      byteStrides.push_back(*byteStride);
    }
  } else {
    if (shape.empty() || info->cBlock <= 0 || info->cxBlocks < 0 ||
        info->outerElements <= 0 || info->c0 < 0)
      return std::nullopt;

    linearStrides.resize(shape.size() - 1, 1);
    int64_t linearStride = 1;
    for (int64_t dim = static_cast<int64_t>(linearStrides.size()) - 1; dim >= 0;
         --dim) {
      linearStrides[dim] = linearStride;
      std::optional<int64_t> next = multiply(linearStride, shape[dim]);
      if (!next)
        return std::nullopt;
      linearStride = *next;
    }

    std::optional<int64_t> computedFullC =
        multiply(info->cxBlocks, info->cBlock);
    int64_t blockOuterElements =
        info->layout == MemLayout::NCx ? info->hwElements : info->outerElements;
    std::optional<int64_t> blockStrideElements =
        multiply(blockOuterElements, info->cBlock);
    if (!computedFullC || !blockStrideElements)
      return std::nullopt;
    std::optional<int64_t> computedBlockStrideBytes =
        multiply(*blockStrideElements, info->elementBytes);
    std::optional<int64_t> fullBlockElements =
        multiply(info->cxBlocks, *blockStrideElements);
    if (!computedBlockStrideBytes || !fullBlockElements)
      return std::nullopt;
    std::optional<int64_t> computedFullBlockBytes =
        multiply(*fullBlockElements, info->elementBytes);
    if (!computedFullBlockBytes)
      return std::nullopt;
    fullC = *computedFullC;
    blockStrideBytes = *computedBlockStrideBytes;
    fullBlockBytes = *computedFullBlockBytes;
  }
  return WaferStaticPhysicalOffsetCalculator(
      std::move(*info), std::move(shape), std::move(byteStrides),
      std::move(linearStrides), fullC, blockStrideBytes, fullBlockBytes);
}

std::optional<int64_t> WaferStaticPhysicalOffsetCalculator::getByteOffset(
    llvm::ArrayRef<int64_t> logicalIndices) const {
  if (shape.size() != logicalIndices.size())
    return std::nullopt;
  for (auto [dim, index] : llvm::zip_equal(shape, logicalIndices))
    if (index < 0 || index >= dim)
      return std::nullopt;
  int64_t byteOffset = getByteOffsetForValidIndices(logicalIndices);
  if (byteOffset < 0 || byteOffset > info.physicalBytes - info.elementBytes)
    return std::nullopt;
  return byteOffset;
}

namespace {

/// Return the blocked-layout lane ordinal independently of the element storage
/// width. Byte-addressable and bitpacked consumers scale this one encoding-
/// owned ordering into bytes or bits respectively.
static std::optional<int64_t>
computeBlockedPhysicalElementOffset(mlir::MemRefType type,
                                    const WaferPhysicalTensorInfo &info,
                                    llvm::ArrayRef<int64_t> logicalIndices) {
  if (info.layout != MemLayout::Cx && info.layout != MemLayout::NCx)
    return std::nullopt;
  if (type.getRank() == 0 ||
      type.getRank() != static_cast<int64_t>(logicalIndices.size()))
    return std::nullopt;
  for (auto [dim, index] : llvm::zip_equal(type.getShape(), logicalIndices)) {
    if (dim == mlir::ShapedType::kDynamic || dim < 0 || index < 0 ||
        index >= dim)
      return std::nullopt;
  }

  int64_t logicalC = logicalIndices.back();
  int64_t fullC = 0;
  if (!checkedMul(info.cxBlocks, info.cBlock, fullC))
    return std::nullopt;
  int64_t physicalElementOffset = 0;
  if (info.layout == MemLayout::NCx) {
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
    if (!checkedMul(n, info.batchElements, batchBase))
      return std::nullopt;
    if (logicalC < fullC) {
      int64_t cb = logicalC / info.cBlock;
      int64_t lane = logicalC % info.cBlock;
      int64_t blockOffset = 0;
      int64_t blockStride = 0;
      if (!checkedMul(info.hwElements, info.cBlock, blockStride) ||
          !checkedMul(cb, blockStride, blockOffset))
        return std::nullopt;
      int64_t hwOffset = 0;
      if (!checkedMul(*hwIndex, info.cBlock, hwOffset))
        return std::nullopt;
      if (!checkedAdd(batchBase, blockOffset, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, hwOffset, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, lane, physicalElementOffset))
        return std::nullopt;
    } else {
      if (info.c0 <= 0)
        return std::nullopt;
      int64_t tailLane = logicalC - fullC;
      int64_t fullBlockElements = 0;
      int64_t fullBlockStride = 0;
      if (!checkedMul(info.hwElements, info.cBlock, fullBlockStride) ||
          !checkedMul(info.cxBlocks, fullBlockStride, fullBlockElements))
        return std::nullopt;
      int64_t hwOffset = 0;
      if (!checkedMul(*hwIndex, info.c0, hwOffset))
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
      int64_t cb = logicalC / info.cBlock;
      int64_t lane = logicalC % info.cBlock;
      int64_t blockOffset = 0;
      int64_t blockStride = 0;
      if (!checkedMul(info.outerElements, info.cBlock, blockStride) ||
          !checkedMul(cb, blockStride, blockOffset))
        return std::nullopt;
      int64_t outerOffset = 0;
      if (!checkedMul(*outerIndex, info.cBlock, outerOffset))
        return std::nullopt;
      if (!checkedAdd(blockOffset, outerOffset, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, lane, physicalElementOffset))
        return std::nullopt;
    } else {
      if (info.c0 <= 0)
        return std::nullopt;
      int64_t tailLane = logicalC - fullC;
      int64_t fullBlockElements = 0;
      int64_t fullBlockStride = 0;
      if (!checkedMul(info.outerElements, info.cBlock, fullBlockStride) ||
          !checkedMul(info.cxBlocks, fullBlockStride, fullBlockElements))
        return std::nullopt;
      int64_t outerOffset = 0;
      if (!checkedMul(*outerIndex, info.c0, outerOffset))
        return std::nullopt;
      if (!checkedAdd(fullBlockElements, outerOffset, physicalElementOffset) ||
          !checkedAdd(physicalElementOffset, tailLane, physicalElementOffset))
        return std::nullopt;
    }
  }
  if (physicalElementOffset < 0 ||
      physicalElementOffset >= info.physicalElements)
    return std::nullopt;
  return physicalElementOffset;
}

} // namespace

std::optional<int64_t> wafer::computeWaferPhysicalElementByteOffset(
    mlir::MemRefType type, const WaferPhysicalTensorInfo &info,
    llvm::ArrayRef<int64_t> logicalIndices) {
  if (info.elementBytes <= 0 || info.bitPackedElement)
    return std::nullopt;
  if (type.getRank() != static_cast<int64_t>(logicalIndices.size()))
    return std::nullopt;
  for (auto [dim, index] : llvm::zip_equal(type.getShape(), logicalIndices)) {
    if (dim == mlir::ShapedType::kDynamic || dim < 0 || index < 0 ||
        index >= dim)
      return std::nullopt;
  }

  if (info.layout != MemLayout::Cx && info.layout != MemLayout::NCx) {
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
    if (!checkedMul(linear, info.elementBytes, byteOffset))
      return std::nullopt;
    return byteOffset;
  }

  std::optional<int64_t> physicalElementOffset =
      computeBlockedPhysicalElementOffset(type, info, logicalIndices);
  int64_t byteOffset = 0;
  if (!physicalElementOffset ||
      !checkedMul(*physicalElementOffset, info.elementBytes, byteOffset))
    return std::nullopt;
  return byteOffset;
}

std::optional<int64_t> wafer::computeWaferPhysicalElementBitOffset(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> logicalIndices) {
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info)
    return std::nullopt;

  if (!info->bitPackedElement) {
    std::optional<int64_t> byteOffset =
        computeWaferPhysicalElementByteOffset(type, logicalIndices);
    int64_t bitOffset = 0;
    if (!byteOffset || !checkedMul(*byteOffset, 8, bitOffset))
      return std::nullopt;
    return bitOffset;
  }

  if (info->layout == MemLayout::Cx || info->layout == MemLayout::NCx)
    return computeBlockedPhysicalElementOffset(type, *info, logicalIndices);
  if (type.getRank() != static_cast<int64_t>(logicalIndices.size()))
    return std::nullopt;
  for (auto [dim, index] : llvm::zip_equal(type.getShape(), logicalIndices)) {
    if (dim == mlir::ShapedType::kDynamic || dim < 0 || index < 0 ||
        index >= dim)
      return std::nullopt;
  }

  llvm::SmallVector<int64_t> strides;
  int64_t ignoredViewOffset = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(type, strides, ignoredViewOffset)) ||
      strides.size() != logicalIndices.size())
    return std::nullopt;

  int64_t bitOffset = 0;
  for (auto [index, stride] : llvm::zip_equal(logicalIndices, strides)) {
    if (stride == mlir::ShapedType::kDynamic || stride < 0)
      return std::nullopt;
    int64_t scaled = 0;
    if (!checkedMul(index, stride, scaled) ||
        !checkedAdd(bitOffset, scaled, bitOffset))
      return std::nullopt;
  }

  int64_t capacityBits = 0;
  if (info->physicalBytes < 0 ||
      !checkedMul(info->physicalBytes, 8, capacityBits) ||
      bitOffset >= capacityBits)
    return std::nullopt;
  return bitOffset;
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
    DTERemoteAddressMode remoteAddressMode, int64_t remoteReceiverAddress,
    mlir::DenseI64ArrayAttr routeBindings,
    DTECompletionProfile completionProfile) {
  (void)allocationProfile;
  (void)completionProfile;
  if (receiverFsmId < 0 || receiverFsmId > 3)
    return emitError()
           << "direct_dte_binding receiver FSM id must be within [0, 3]";
  if (remoteAddressMode == DTERemoteAddressMode::Absolute &&
      remoteReceiverAddress < 0)
    return emitError()
           << "direct_dte_binding absolute remote receiver address must be "
              "non-negative";
  llvm::ArrayRef<int64_t> table = routeBindings.asArrayRef();
  if (remoteAddressMode != DTERemoteAddressMode::SelectorTable) {
    if (!table.empty())
      return emitError()
             << "direct_dte_binding route table requires selector_table mode";
    return mlir::success();
  }
  if (table.empty() || table.size() % 3 != 0)
    return emitError()
           << "direct_dte_binding route table must contain non-empty "
              "(selector, remote, receiver_fsm) triples";
  for (size_t index = 0; index < table.size(); index += 3) {
    if (table[index] < 0 || table[index + 1] < 0)
      return emitError()
             << "direct_dte_binding selector and remote address must be "
                "non-negative";
    if (table[index + 2] < 0 || table[index + 2] > 3)
      return emitError() << "direct_dte_binding route receiver FSM id must be "
                            "within [0, 3]";
    if (table[index] != static_cast<int64_t>(index / 3))
      return emitError()
             << "direct_dte_binding route selectors must be contiguous "
                "from zero";
  }
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
