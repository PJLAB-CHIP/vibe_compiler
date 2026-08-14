//===- PhysicalLayout.cpp - Pure target physical tensor layout ----------===//

#include "Wafer/Target/PhysicalLayout.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace wafer {
namespace {

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 ||
      (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs))
    return false;
  result = lhs * rhs;
  return true;
}

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static std::optional<int64_t> product(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    int64_t next = 0;
    if (!checkedMul(result, value, next))
      return std::nullopt;
    result = next;
  }
  return result;
}

static std::optional<int64_t> alignTo(int64_t value, int64_t alignment) {
  if (value < 0 || alignment <= 0)
    return std::nullopt;
  const int64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  int64_t result = 0;
  return checkedAdd(value, alignment - remainder, result)
             ? std::optional<int64_t>(result)
             : std::nullopt;
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

static bool computeCxC0(int64_t channels, int64_t block,
                        int64_t retainThreshold, int64_t &blocks, int64_t &tail,
                        int64_t &alignedChannels) {
  if (channels < 0 || block <= 0 || retainThreshold <= 0)
    return false;
  const int64_t quotient = channels / block;
  const int64_t remainder = channels % block;
  if (remainder == 0) {
    blocks = quotient;
    tail = 0;
    return checkedMul(quotient, block, alignedChannels);
  }
  if (remainder <= retainThreshold) {
    blocks = quotient;
    tail = alignCxTail(remainder);
    int64_t full = 0;
    return checkedMul(quotient, block, full) &&
           checkedAdd(full, tail, alignedChannels);
  }
  blocks = quotient + 1;
  tail = 0;
  return checkedMul(blocks, block, alignedChannels);
}

static std::optional<int64_t> byteSize(int64_t elements,
                                       int64_t elementBitWidth) {
  int64_t bits = 0;
  if (elementBitWidth <= 0 || !checkedMul(elements, elementBitWidth, bits))
    return std::nullopt;
  return bits / 8 + (bits % 8 == 0 ? 0 : 1);
}

} // namespace

llvm::StringRef stringifyPhysicalTensorLayout(PhysicalTensorLayout layout) {
  switch (layout) {
  case PhysicalTensorLayout::Tensor:
    return "tensor";
  case PhysicalTensorLayout::NTensor:
    return "ntensor";
  case PhysicalTensorLayout::Cx:
    return "cx";
  case PhysicalTensorLayout::NCx:
    return "ncx";
  }
  llvm_unreachable("unknown physical tensor layout");
}

std::optional<PhysicalTensorGeometry> computePhysicalTensorGeometry(
    llvm::ArrayRef<int64_t> shape, int64_t elementBitWidth,
    bool usesInt8ChannelBlock, PhysicalTensorLayout layout,
    std::optional<int64_t> physicalElementSpan) {
  if (elementBitWidth <= 0 ||
      llvm::any_of(shape, [](int64_t dimension) { return dimension < 0; }))
    return std::nullopt;
  std::optional<int64_t> compactElements = product(shape);
  if (!compactElements)
    return std::nullopt;

  PhysicalTensorGeometry geometry;
  geometry.layout = layout;
  geometry.elementBytes = (elementBitWidth + 7) / 8;
  geometry.bitPackedElement = elementBitWidth == 1;
  geometry.compactBytes =
      byteSize(*compactElements, elementBitWidth).value_or(-1);

  if (layout == PhysicalTensorLayout::Tensor ||
      layout == PhysicalTensorLayout::NTensor) {
    if (!physicalElementSpan || *physicalElementSpan < 0)
      return std::nullopt;
    geometry.physicalElements = *physicalElementSpan;
    geometry.physicalBytes =
        byteSize(*physicalElementSpan, elementBitWidth).value_or(-1);
    return geometry.physicalBytes < 0
               ? std::nullopt
               : std::optional<PhysicalTensorGeometry>(geometry);
  }

  if (shape.empty())
    return std::nullopt;
  const int64_t channels = shape.back();
  const int64_t block = usesInt8ChannelBlock ? 128 : 64;
  const int64_t retainThreshold = usesInt8ChannelBlock ? 64 : 32;
  if (!computeCxC0(channels, block, retainThreshold, geometry.cxBlocks,
                   geometry.c0, geometry.alignedC))
    return std::nullopt;
  constexpr int64_t bankLineBits = 256 * 8;
  if (bankLineBits % elementBitWidth != 0)
    return std::nullopt;
  geometry.cBlock = block;
  geometry.tailC = geometry.c0;
  geometry.bankAlignElements = bankLineBits / elementBitWidth;

  int64_t physicalElements = 0;
  if (layout == PhysicalTensorLayout::NCx) {
    const int64_t batches = shape.size() > 1 ? shape.front() : 1;
    llvm::ArrayRef<int64_t> hwShape = shape.size() > 1
                                          ? shape.drop_front().drop_back()
                                          : llvm::ArrayRef<int64_t>{};
    std::optional<int64_t> hwElements = product(hwShape);
    int64_t unalignedBatch = 0;
    if (!hwElements ||
        !checkedMul(*hwElements, geometry.alignedC, unalignedBatch))
      return std::nullopt;
    std::optional<int64_t> batchElements =
        alignTo(unalignedBatch, geometry.bankAlignElements);
    if (!batchElements ||
        !checkedMul(batches, *batchElements, physicalElements) ||
        !checkedMul(batches, *hwElements, geometry.outerElements))
      return std::nullopt;
    geometry.hwElements = *hwElements;
    geometry.batchElements = *batchElements;
  } else {
    std::optional<int64_t> outerElements = product(shape.drop_back());
    int64_t unalignedBatch = 0;
    if (!outerElements ||
        !checkedMul(*outerElements, geometry.alignedC, unalignedBatch))
      return std::nullopt;
    std::optional<int64_t> batchElements =
        alignTo(unalignedBatch, geometry.bankAlignElements);
    if (!batchElements)
      return std::nullopt;
    physicalElements = *batchElements;
    geometry.outerElements = *outerElements;
    geometry.hwElements = *outerElements;
    geometry.batchElements = *batchElements;
  }
  geometry.physicalElements = physicalElements;
  geometry.physicalBytes =
      byteSize(physicalElements, elementBitWidth).value_or(-1);
  return geometry.physicalBytes < 0
             ? std::nullopt
             : std::optional<PhysicalTensorGeometry>(geometry);
}

std::optional<StaticPhysicalTensorOffsetCalculator>
StaticPhysicalTensorOffsetCalculator::create(
    PhysicalTensorGeometry geometry, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> elementStrides) {
  if (geometry.physicalBytes < 0 || geometry.physicalElements < 0 ||
      geometry.elementBytes <= 0 ||
      llvm::any_of(shape, [](int64_t dimension) { return dimension <= 0; }))
    return std::nullopt;
  const int64_t elementBitWidth =
      geometry.bitPackedElement ? 1 : geometry.elementBytes * 8;
  llvm::SmallVector<int64_t, 4> bitStrides;
  llvm::SmallVector<int64_t, 4> linearStrides;
  int64_t fullC = 0;
  int64_t blockStrideElements = 0;
  int64_t fullBlockElements = 0;
  if (geometry.layout == PhysicalTensorLayout::Tensor ||
      geometry.layout == PhysicalTensorLayout::NTensor) {
    if (elementStrides.size() != shape.size())
      return std::nullopt;
    for (int64_t stride : elementStrides) {
      int64_t bits = 0;
      if (!checkedMul(stride, elementBitWidth, bits))
        return std::nullopt;
      bitStrides.push_back(bits);
    }
  } else {
    linearStrides.resize(shape.size() - 1, 1);
    int64_t linearStride = 1;
    for (int64_t dimension = static_cast<int64_t>(linearStrides.size()) - 1;
         dimension >= 0; --dimension) {
      linearStrides[dimension] = linearStride;
      if (!checkedMul(linearStride, shape[dimension], linearStride))
        return std::nullopt;
    }
    const int64_t blockOuterElements =
        geometry.layout == PhysicalTensorLayout::NCx ? geometry.hwElements
                                                     : geometry.outerElements;
    if (!checkedMul(geometry.cxBlocks, geometry.cBlock, fullC) ||
        !checkedMul(blockOuterElements, geometry.cBlock, blockStrideElements) ||
        !checkedMul(geometry.cxBlocks, blockStrideElements, fullBlockElements))
      return std::nullopt;
  }
  return StaticPhysicalTensorOffsetCalculator(
      std::move(geometry), llvm::SmallVector<int64_t, 4>(shape),
      std::move(bitStrides), std::move(linearStrides), fullC,
      blockStrideElements, fullBlockElements, elementBitWidth);
}

std::optional<int64_t> StaticPhysicalTensorOffsetCalculator::getBitOffset(
    llvm::ArrayRef<int64_t> logicalIndices) const {
  if (shape.size() != logicalIndices.size())
    return std::nullopt;
  for (auto [dimension, index] : llvm::zip_equal(shape, logicalIndices))
    if (index < 0 || index >= dimension)
      return std::nullopt;

  int64_t physicalElements = 0;
  if (geometry.layout == PhysicalTensorLayout::Tensor ||
      geometry.layout == PhysicalTensorLayout::NTensor) {
    int64_t bitOffset = 0;
    for (auto [index, stride] : llvm::zip_equal(logicalIndices, bitStrides)) {
      int64_t contribution = 0;
      if (!checkedMul(index, stride, contribution) ||
          !checkedAdd(bitOffset, contribution, bitOffset))
        return std::nullopt;
    }
    return bitOffset;
  }

  const int64_t logicalC = logicalIndices.back();
  int64_t outerIndex = 0;
  const int64_t firstOuterDimension =
      geometry.layout == PhysicalTensorLayout::NCx ? 1 : 0;
  for (int64_t dimension = firstOuterDimension;
       dimension < static_cast<int64_t>(logicalIndices.size()) - 1;
       ++dimension) {
    int64_t contribution = 0;
    if (!checkedMul(logicalIndices[dimension], linearStrides[dimension],
                    contribution) ||
        !checkedAdd(outerIndex, contribution, outerIndex))
      return std::nullopt;
  }
  if (geometry.layout == PhysicalTensorLayout::NCx &&
      logicalIndices.size() > 1 &&
      !checkedMul(logicalIndices.front(), geometry.batchElements,
                  physicalElements))
    return std::nullopt;
  if (logicalC < fullC) {
    int64_t blockBase = 0;
    int64_t outerBase = 0;
    if (!checkedMul(logicalC / geometry.cBlock, blockStrideElements,
                    blockBase) ||
        !checkedMul(outerIndex, geometry.cBlock, outerBase) ||
        !checkedAdd(outerBase, logicalC % geometry.cBlock, outerBase) ||
        !checkedAdd(physicalElements, blockBase, physicalElements) ||
        !checkedAdd(physicalElements, outerBase, physicalElements))
      return std::nullopt;
  } else {
    int64_t tailOffset = 0;
    if (!checkedMul(outerIndex, geometry.c0, tailOffset) ||
        !checkedAdd(tailOffset, logicalC - fullC, tailOffset) ||
        !checkedAdd(physicalElements, fullBlockElements, physicalElements) ||
        !checkedAdd(physicalElements, tailOffset, physicalElements))
      return std::nullopt;
  }
  int64_t bitOffset = 0;
  return checkedMul(physicalElements, elementBitWidth, bitOffset)
             ? std::optional<int64_t>(bitOffset)
             : std::nullopt;
}

} // namespace wafer
