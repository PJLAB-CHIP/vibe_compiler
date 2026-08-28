//===- WaferDialect.cpp - Wafer dialect implementation -------------------===//

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/DirectDTE.h"

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

mlir::LogicalResult
wafer::verifyNoSchemaFreeSemanticAttributes(mlir::Operation *operation) {
  std::optional<mlir::RegisteredOperationName> registered =
      operation->getRegisteredInfo();
  for (mlir::NamedAttribute attribute : operation->getAttrs()) {
    if (registered && llvm::is_contained(registered->getAttributeNames(),
                                         attribute.getName()))
      continue;
    llvm::StringRef name = attribute.getName().getValue();
    auto [dialectNamespace, suffix] = name.split('.');
    if (!dialectNamespace.empty() && !suffix.empty() &&
        dialectNamespace != WaferDialect::getDialectNamespace())
      continue;
    return operation->emitOpError(
               "does not accept schema-free semantic attribute '")
           << name << "'";
  }
  return mlir::success();
}

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
  int64_t blockedOuterElements = info.blockOuterElements;
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

  mlir::AffineExpr outerSliceBase = mlir::getAffineConstantExpr(0, context);
  if (getLayout() == MemLayout::NCx && shape.size() > 1)
    outerSliceBase =
        mlir::getAffineDimExpr(0, context) * info.outerSliceStrideElements;

  const int64_t logicalC = shape.back();
  const int64_t fullUpper = std::min(logicalC, fullC);
  if (fullUpper > 0) {
    llvm::SmallVector<int64_t, 4> fullUpperBounds(upper);
    llvm::SmallVector<int64_t, 4> fullPeriods(noPeriods);
    fullUpperBounds[channelDim] = fullUpper;
    fullPeriods[channelDim] = info.cBlock;
    mlir::AffineExpr physicalElements =
        outerSliceBase + channel.floorDiv(info.cBlock) * blockStrideElements +
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
        outerSliceBase + fullBlockElements + outer * info.c0 + channel - fullC;
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
  auto integerType = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
  info.bitPackedElement = integerType && integerType.getWidth() == 1;
  if (std::optional<int64_t> elementBytes =
          getElementStorageBytes(type.getElementType()))
    info.elementBytes = *elementBytes;

  // Cx/NCx already own a block-major physical map in the memory encoding.
  // A second non-identity MemRef layout would describe a different view whose
  // composition is not representable by the current type contract. Reject it
  // instead of silently ignoring its strides/offset. Bitpacked blocked
  // buffers use the same block/tail ordering with bit-addressed lanes.
  if ((info.layout == MemLayout::Cx || info.layout == MemLayout::NCx) &&
      (type.getRank() == 0 || !type.getLayout().isIdentity()))
    return std::nullopt;

  std::optional<int64_t> elementBits =
      getElementBitWidth(type.getElementType());
  if (!elementBits)
    return std::nullopt;
  // The pure target calculator intentionally accepts only concrete geometry.
  // The MLIR type adapter must still distinguish a legal dynamic buffer from
  // an invalid encoding; retain the known type facts and leave sizes unknown.
  if (!type.hasStaticShape())
    return info;
  std::optional<PhysicalTensorLayout> physicalLayout;
  switch (info.layout) {
  case MemLayout::Tensor:
    physicalLayout = PhysicalTensorLayout::Tensor;
    break;
  case MemLayout::NTensor:
    physicalLayout = PhysicalTensorLayout::NTensor;
    break;
  case MemLayout::Cx:
    physicalLayout = PhysicalTensorLayout::Cx;
    break;
  case MemLayout::NCx:
    physicalLayout = PhysicalTensorLayout::NCx;
    break;
  }
  if (!physicalLayout)
    return std::nullopt;
  std::optional<PhysicalTensorGeometry> geometry =
      computePhysicalTensorGeometry(type.getShape(), *elementBits,
                                    integerType && integerType.getWidth() == 8,
                                    *physicalLayout,
                                    getStaticStridedElementSpan(type));
  if (!geometry)
    return std::nullopt;
  info.compactBytes = geometry->compactBytes;
  info.physicalBytes = geometry->physicalBytes;
  info.physicalElements = geometry->physicalElements;
  info.elementBytes = geometry->elementBytes;
  info.cBlock = geometry->cBlock;
  info.cxBlocks = geometry->cxBlocks;
  info.c0 = geometry->c0;
  info.alignedC = geometry->alignedC;
  info.tailC = geometry->tailC;
  info.outerElements = geometry->outerElements;
  info.blockOuterElements = geometry->blockOuterElements;
  info.outerSliceStrideElements = geometry->outerSliceStrideElements;
  info.bankAlignElements = geometry->bankAlignElements;
  info.bitPackedElement = geometry->bitPackedElement;

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

namespace {

static PhysicalTensorGeometry
projectPhysicalTensorGeometry(const WaferPhysicalTensorInfo &info) {
  PhysicalTensorGeometry geometry;
  switch (info.layout) {
  case MemLayout::Tensor:
    geometry.layout = PhysicalTensorLayout::Tensor;
    break;
  case MemLayout::NTensor:
    geometry.layout = PhysicalTensorLayout::NTensor;
    break;
  case MemLayout::Cx:
    geometry.layout = PhysicalTensorLayout::Cx;
    break;
  case MemLayout::NCx:
    geometry.layout = PhysicalTensorLayout::NCx;
    break;
  }
  geometry.compactBytes = info.compactBytes;
  geometry.physicalBytes = info.physicalBytes;
  geometry.physicalElements = info.physicalElements;
  geometry.elementBytes = info.elementBytes;
  geometry.cBlock = info.cBlock;
  geometry.cxBlocks = info.cxBlocks;
  geometry.c0 = info.c0;
  geometry.alignedC = info.alignedC;
  geometry.tailC = info.tailC;
  geometry.outerElements = info.outerElements;
  geometry.blockOuterElements = info.blockOuterElements;
  geometry.outerSliceStrideElements = info.outerSliceStrideElements;
  geometry.bankAlignElements = info.bankAlignElements;
  geometry.bitPackedElement = info.bitPackedElement;
  return geometry;
}

static std::optional<StaticPhysicalTensorOffsetCalculator>
createSharedOffsetCalculator(mlir::MemRefType type,
                             const WaferPhysicalTensorInfo &info) {
  llvm::SmallVector<int64_t, 4> elementStrides;
  if (info.layout != MemLayout::Cx && info.layout != MemLayout::NCx) {
    int64_t ignoredOffset = 0;
    if (mlir::failed(
            mlir::getStridesAndOffset(type, elementStrides, ignoredOffset)) ||
        elementStrides.size() != static_cast<size_t>(type.getRank()))
      return std::nullopt;
  }
  return StaticPhysicalTensorOffsetCalculator::create(
      projectPhysicalTensorGeometry(info), type.getShape(), elementStrides);
}

} // namespace

std::optional<WaferStaticPhysicalOffsetCalculator>
WaferStaticPhysicalOffsetCalculator::create(mlir::MemRefType type) {
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info || info->physicalBytes <= 0 || info->elementBytes <= 0 ||
      info->bitPackedElement)
    return std::nullopt;

  std::optional<StaticPhysicalTensorOffsetCalculator> calculator =
      createSharedOffsetCalculator(type, *info);
  if (!calculator)
    return std::nullopt;
  return WaferStaticPhysicalOffsetCalculator(std::move(*info),
                                             std::move(*calculator));
}

std::optional<int64_t> WaferStaticPhysicalOffsetCalculator::getByteOffset(
    llvm::ArrayRef<int64_t> logicalIndices) const {
  std::optional<int64_t> bitOffset = calculator.getBitOffset(logicalIndices);
  if (!bitOffset || *bitOffset % 8 != 0)
    return std::nullopt;
  int64_t byteOffset = *bitOffset / 8;
  if (byteOffset < 0 || byteOffset > info.physicalBytes - info.elementBytes)
    return std::nullopt;
  return byteOffset;
}

std::optional<int64_t> wafer::computeWaferPhysicalElementByteOffset(
    mlir::MemRefType type, const WaferPhysicalTensorInfo &info,
    llvm::ArrayRef<int64_t> logicalIndices) {
  if (info.elementBytes <= 0 || info.bitPackedElement)
    return std::nullopt;
  std::optional<StaticPhysicalTensorOffsetCalculator> calculator =
      createSharedOffsetCalculator(type, info);
  if (!calculator)
    return std::nullopt;
  std::optional<int64_t> bitOffset = calculator->getBitOffset(logicalIndices);
  if (!bitOffset || *bitOffset < 0 || *bitOffset % 8 != 0)
    return std::nullopt;
  return *bitOffset / 8;
}

std::optional<int64_t> wafer::computeWaferPhysicalElementBitOffset(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> logicalIndices) {
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info)
    return std::nullopt;

  std::optional<StaticPhysicalTensorOffsetCalculator> calculator =
      createSharedOffsetCalculator(type, *info);
  if (!calculator)
    return std::nullopt;
  return calculator->getBitOffset(logicalIndices);
}

mlir::LogicalResult
DTEMessageAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                       int64_t communicationId, int64_t round,
                       int64_t payloadSlice) {
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
  constexpr int64_t maximumReceiverFSM =
      TargetDirectDTEResourceLimits::receiverFSMsPerTile - 1;
  if (receiverFsmId < 0 || receiverFsmId > maximumReceiverFSM)
    return emitError()
           << "direct_dte_binding receiver FSM id must be within [0, "
           << maximumReceiverFSM << "]";
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
    if (table[index + 2] < 0 || table[index + 2] > maximumReceiverFSM)
      return emitError() << "direct_dte_binding route receiver FSM id must be "
                            "within [0, "
                         << maximumReceiverFSM << "]";
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
