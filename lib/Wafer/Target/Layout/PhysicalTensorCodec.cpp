//===- PhysicalTensorCodec.cpp - Wafer physical tensor codec ------------===//

#include "Wafer/Target/Layout/PhysicalTensorCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <optional>

namespace wafer {
namespace {

constexpr LogicalScalarCodecPolicy kTargetPhysicalCodecPolicy{
    LogicalByteOrder::LittleEndian,
    LogicalBitOrder::LeastSignificantBitFirstWithinByte,
    NonCanonicalEncodingPolicy::Reject};

llvm::Error codecError(PhysicalTensorCodecErrorCode code,
                       const llvm::Twine &detail) {
  return llvm::make_error<PhysicalTensorCodecError>(code, detail.str());
}

struct OwnedPhysicalLayout {
  PhysicalTensorGeometry info;
  StaticPhysicalTensorOffsetCalculator offsets;
};

llvm::Expected<OwnedPhysicalLayout>
makePhysicalLayout(const PhysicalTensorDescriptor &key) {
  const LogicalFormatDescriptor *format =
      findLogicalFormatDescriptor(key.getFormat());
  if (!format)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "tensor has an unknown format or layout");
  std::vector<int64_t> shape;
  shape.reserve(key.getShape().size());
  for (uint64_t dimension : key.getShape()) {
    if (dimension > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return codecError(
          PhysicalTensorCodecErrorCode::InvalidLayout,
          "tensor dimension exceeds the physical layout helper domain");
    shape.push_back(static_cast<int64_t>(dimension));
  }
  if (key.getElementCount() >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "tensor element count exceeds layout helper domain");
  std::optional<PhysicalTensorGeometry> info = computePhysicalTensorGeometry(
      shape, format->storageBits,
      key.getFormat() == LogicalFormat::I8 ||
          key.getFormat() == LogicalFormat::U8,
      key.getLayout(), static_cast<int64_t>(key.getElementCount()));
  if (!info)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "shared layout helper rejected the static tensor");
  llvm::SmallVector<int64_t, 4> elementStrides(shape.size(), 1);
  int64_t stride = 1;
  for (int64_t dimension = static_cast<int64_t>(shape.size()) - 1;
       dimension >= 0; --dimension) {
    elementStrides[dimension] = stride;
    if (shape[dimension] != 0 &&
        stride > std::numeric_limits<int64_t>::max() / shape[dimension])
      return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                        "tensor row-major stride overflows int64");
    stride *= shape[dimension];
  }
  std::optional<StaticPhysicalTensorOffsetCalculator> offsets =
      StaticPhysicalTensorOffsetCalculator::create(*info, shape,
                                                   elementStrides);
  if (!offsets)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "static offset calculator rejected the tensor");
  return OwnedPhysicalLayout{std::move(*info), std::move(*offsets)};
}

template <typename Callback>
llvm::Error forEachCoordinate(llvm::ArrayRef<uint64_t> shape,
                              Callback callback) {
  if (llvm::is_contained(shape, UINT64_C(0)))
    return llvm::Error::success();
  std::vector<int64_t> coordinate(shape.size(), 0);
  if (shape.empty())
    return callback(llvm::ArrayRef<int64_t>(coordinate));
  while (true) {
    if (llvm::Error error = callback(llvm::ArrayRef<int64_t>(coordinate)))
      return error;
    int64_t dimension = static_cast<int64_t>(shape.size()) - 1;
    for (; dimension >= 0; --dimension) {
      ++coordinate[static_cast<size_t>(dimension)];
      if (coordinate[static_cast<size_t>(dimension)] <
          static_cast<int64_t>(shape[static_cast<size_t>(dimension)]))
        break;
      coordinate[static_cast<size_t>(dimension)] = 0;
    }
    if (dimension < 0)
      return llvm::Error::success();
  }
}

template <typename Callback>
llvm::Error forEachPhysicalBitOffset(const PhysicalTensorDescriptor &key,
                                     const OwnedPhysicalLayout &layout,
                                     Callback callback) {
  return forEachCoordinate(
      key.getShape(), [&](llvm::ArrayRef<int64_t> coordinate) -> llvm::Error {
        std::optional<int64_t> bitOffset =
            layout.offsets.getBitOffset(coordinate);
        if (!bitOffset || *bitOffset < 0)
          return codecError(
              PhysicalTensorCodecErrorCode::InvalidLayout,
              "shared layout helper could not map a logical coordinate");
        return callback(static_cast<uint64_t>(*bitOffset));
      });
}

} // namespace

llvm::StringRef
stringifyPhysicalTensorCodecErrorCode(PhysicalTensorCodecErrorCode code) {
  switch (code) {
  case PhysicalTensorCodecErrorCode::InvalidLayout:
    return "invalid-layout";
  case PhysicalTensorCodecErrorCode::InvalidStorageSize:
    return "invalid-storage-size";
  case PhysicalTensorCodecErrorCode::InvalidLogicalValueCount:
    return "invalid-logical-value-count";
  case PhysicalTensorCodecErrorCode::InvalidLogicalEncoding:
    return "invalid-logical-encoding";
  }
  llvm_unreachable("unknown physical tensor codec error code");
}

char PhysicalTensorCodecError::ID;

void PhysicalTensorCodecError::log(llvm::raw_ostream &stream) const {
  stream << "physical tensor codec "
         << stringifyPhysicalTensorCodecErrorCode(code) << ": " << detail;
}

std::error_code PhysicalTensorCodecError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<uint64_t>
getPhysicalTensorStorageBytes(const PhysicalTensorDescriptor &key) {
  llvm::Expected<OwnedPhysicalLayout> layout = makePhysicalLayout(key);
  if (!layout)
    return layout.takeError();
  return static_cast<uint64_t>(layout->info.physicalBytes);
}

llvm::Expected<std::vector<RawLogicalValue>>
unpackPhysicalTensorLogicalValues(const PhysicalTensorDescriptor &key,
                                  llvm::ArrayRef<uint8_t> storage) {
  llvm::Expected<OwnedPhysicalLayout> layout = makePhysicalLayout(key);
  if (!layout)
    return layout.takeError();
  if (storage.size() != static_cast<uint64_t>(layout->info.physicalBytes))
    return codecError(PhysicalTensorCodecErrorCode::InvalidStorageSize,
                      llvm::Twine("tensor requires ") +
                          llvm::Twine(layout->info.physicalBytes) +
                          " physical bytes, got " +
                          llvm::Twine(storage.size()));
  if (key.getElementCount() > std::numeric_limits<size_t>::max())
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "tensor element count exceeds host size_t");
  const LogicalScalarCodecPolicy policy = kTargetPhysicalCodecPolicy;
  std::vector<RawLogicalValue> result;
  result.reserve(static_cast<size_t>(key.getElementCount()));
  if (llvm::Error error = forEachPhysicalBitOffset(
          key, *layout, [&](uint64_t offset) -> llvm::Error {
            llvm::Expected<RawLogicalValue> value =
                readRawLogicalValue(key.getFormat(), storage, offset, policy);
            if (!value)
              return codecError(
                  PhysicalTensorCodecErrorCode::InvalidLogicalEncoding,
                  llvm::toString(value.takeError()));
            result.push_back(*value);
            return llvm::Error::success();
          }))
    return std::move(error);
  return result;
}

llvm::Expected<std::vector<uint8_t>>
packPhysicalTensorLogicalValues(const PhysicalTensorDescriptor &key,
                                llvm::ArrayRef<RawLogicalValue> values,
                                llvm::ArrayRef<uint8_t> storageTemplate) {
  llvm::Expected<OwnedPhysicalLayout> layout = makePhysicalLayout(key);
  if (!layout)
    return layout.takeError();
  if (storageTemplate.size() !=
      static_cast<uint64_t>(layout->info.physicalBytes))
    return codecError(PhysicalTensorCodecErrorCode::InvalidStorageSize,
                      llvm::Twine("tensor requires ") +
                          llvm::Twine(layout->info.physicalBytes) +
                          " physical template bytes, got " +
                          llvm::Twine(storageTemplate.size()));
  if (values.size() != key.getElementCount())
    return codecError(PhysicalTensorCodecErrorCode::InvalidLogicalValueCount,
                      "logical value count does not match the tensor key");
  const LogicalScalarCodecPolicy policy = kTargetPhysicalCodecPolicy;
  std::vector<uint8_t> result(storageTemplate.begin(), storageTemplate.end());
  size_t valueIndex = 0;
  if (llvm::Error error = forEachPhysicalBitOffset(
          key, *layout, [&](uint64_t offset) -> llvm::Error {
            RawLogicalValue value = values[valueIndex++];
            if (value.format != key.getFormat())
              return codecError(
                  PhysicalTensorCodecErrorCode::InvalidLogicalEncoding,
                  "logical value format does not match the tensor key");
            if (llvm::Error writeError =
                    writeRawLogicalValue(value, result, offset, policy))
              return codecError(
                  PhysicalTensorCodecErrorCode::InvalidLogicalEncoding,
                  llvm::toString(std::move(writeError)));
            return llvm::Error::success();
          }))
    return std::move(error);
  return result;
}

llvm::Expected<std::vector<uint8_t>>
packPhysicalTensorLogicalValues(const PhysicalTensorDescriptor &key,
                                llvm::ArrayRef<RawLogicalValue> values,
                                uint8_t paddingFill) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return bytes.takeError();
  if (*bytes > std::numeric_limits<size_t>::max())
    return codecError(PhysicalTensorCodecErrorCode::InvalidStorageSize,
                      "physical tensor byte count exceeds host size_t");
  return packPhysicalTensorLogicalValues(
      key, values,
      std::vector<uint8_t>(static_cast<size_t>(*bytes), paddingFill));
}

struct PhysicalTensorWindowPlan::Impl {
  OwnedPhysicalLayout layout;
  uint64_t storageBytes = 0;
  uint64_t elementCount = 0;
  uint64_t physicalElementCount = 0;
  uint64_t elementBitWidth = 0;
  std::vector<uint64_t> shape;
  uint64_t nextPhysicalElement = 0;
  uint64_t plannedLogicalElements = 0;

  /// Inverts the physical geometry for one physical element. Padding has no
  /// logical owner and returns nullopt. This is the same block-major geometry
  /// used by StaticPhysicalTensorOffsetCalculator::getBitOffset.
  std::optional<uint64_t>
  logicalIndexForPhysicalElement(uint64_t physicalElement) const {
    if (layout.info.layout == PhysicalTensorLayout::Tensor ||
        layout.info.layout == PhysicalTensorLayout::NTensor)
      return physicalElement < elementCount
                 ? std::optional<uint64_t>(physicalElement)
                 : std::nullopt;

    const uint64_t channels = shape.back();
    const uint64_t block = static_cast<uint64_t>(layout.info.cBlock);
    const uint64_t blocks = static_cast<uint64_t>(layout.info.cxBlocks);
    const uint64_t tail = static_cast<uint64_t>(layout.info.c0);
    const uint64_t blockOuter =
        static_cast<uint64_t>(layout.info.blockOuterElements);
    const uint64_t blockStride = blockOuter * block;
    const uint64_t fullBlockElements = blocks * blockStride;

    uint64_t outerSlice = 0;
    uint64_t slicePhysical = physicalElement;
    if (layout.info.layout == PhysicalTensorLayout::NCx) {
      const uint64_t outerSliceStrideElements =
          static_cast<uint64_t>(layout.info.outerSliceStrideElements);
      outerSlice = physicalElement / outerSliceStrideElements;
      slicePhysical = physicalElement % outerSliceStrideElements;
      const uint64_t outerSliceCount = shape.size() > 1 ? shape.front() : 1;
      if (outerSlice >= outerSliceCount)
        return std::nullopt;
    }

    uint64_t outer = 0;
    uint64_t channel = 0;
    if (slicePhysical < fullBlockElements) {
      const uint64_t blockIndex = slicePhysical / blockStride;
      const uint64_t withinBlock = slicePhysical % blockStride;
      outer = withinBlock / block;
      channel = blockIndex * block + withinBlock % block;
    } else {
      const uint64_t tailPhysical = slicePhysical - fullBlockElements;
      if (tail == 0 || tailPhysical >= blockOuter * tail)
        return std::nullopt;
      outer = tailPhysical / tail;
      channel = blocks * block + tailPhysical % tail;
    }
    if (outer >= blockOuter || channel >= channels)
      return std::nullopt;
    const uint64_t globalOuter = outerSlice * blockOuter + outer;
    const uint64_t logicalIndex = globalOuter * channels + channel;
    return logicalIndex < elementCount ? std::optional<uint64_t>(logicalIndex)
                                       : std::nullopt;
  }
};

PhysicalTensorWindowPlan::PhysicalTensorWindowPlan() = default;
PhysicalTensorWindowPlan::PhysicalTensorWindowPlan(
    PhysicalTensorWindowPlan &&) noexcept = default;
PhysicalTensorWindowPlan &PhysicalTensorWindowPlan::operator=(
    PhysicalTensorWindowPlan &&) noexcept = default;
PhysicalTensorWindowPlan::~PhysicalTensorWindowPlan() = default;

llvm::Expected<PhysicalTensorWindowPlan>
PhysicalTensorWindowPlan::create(const PhysicalTensorDescriptor &key) {
  llvm::Expected<OwnedPhysicalLayout> layout = makePhysicalLayout(key);
  if (!layout)
    return layout.takeError();
  const LogicalFormatDescriptor *format =
      findLogicalFormatDescriptor(key.getFormat());
  if (!format)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "tensor has an unknown format or layout");
  if (format->storageBits <= 0 ||
      (format->storageBits != 1 && format->storageBits % 8 != 0))
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "window planning requires bitpacked or byte-sized "
                      "logical storage");
  if (layout->info.physicalElements < 0 ||
      static_cast<uint64_t>(layout->info.physicalElements) <
          key.getElementCount())
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "physical geometry cannot cover all logical elements");
  const uint64_t storageBytes =
      static_cast<uint64_t>(layout->info.physicalBytes);
  const uint64_t physicalElementCount =
      static_cast<uint64_t>(layout->info.physicalElements);
  PhysicalTensorWindowPlan plan;
  plan.impl = std::make_unique<Impl>(Impl{
      std::move(*layout), storageBytes, key.getElementCount(),
      physicalElementCount, static_cast<uint64_t>(format->storageBits),
      std::vector<uint64_t>(key.getShape().begin(), key.getShape().end())});
  return plan;
}

uint64_t PhysicalTensorWindowPlan::getStorageBytes() const {
  return impl ? impl->storageBytes : 0;
}
uint64_t PhysicalTensorWindowPlan::getElementCount() const {
  return impl ? impl->elementCount : 0;
}
uint64_t PhysicalTensorWindowPlan::getPlannedValueCount() const {
  return impl ? impl->plannedLogicalElements : 0;
}
bool PhysicalTensorWindowPlan::done() const {
  return impl && impl->nextPhysicalElement == impl->physicalElementCount;
}

llvm::Expected<PhysicalTensorWindowPlan::WriteWindow>
PhysicalTensorWindowPlan::takeNext(uint64_t maxBytes, uint64_t maxLogicalValues,
                                   uint8_t paddingFill) {
  if (!impl)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "physical window plan was moved from");
  Impl &state = *impl;
  if (state.nextPhysicalElement == state.physicalElementCount)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLogicalValueCount,
                      "physical window plan is exhausted");
  if (maxBytes == 0 || maxLogicalValues == 0)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLogicalValueCount,
                      "physical window budgets must be positive");

  const uint64_t remainingPhysical =
      state.physicalElementCount - state.nextPhysicalElement;
  uint64_t byteLimitedElements = 0;
  if (state.elementBitWidth == 1) {
    byteLimitedElements = maxBytes > std::numeric_limits<uint64_t>::max() / 8
                              ? std::numeric_limits<uint64_t>::max()
                              : maxBytes * 8;
  } else {
    byteLimitedElements = maxBytes / (state.elementBitWidth / 8);
  }
  uint64_t physicalCount =
      std::min({remainingPhysical, byteLimitedElements, maxLogicalValues});
  if (state.elementBitWidth == 1 && physicalCount < remainingPhysical)
    physicalCount -= physicalCount % 8;
  if (physicalCount == 0)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLogicalValueCount,
                      "physical window budgets cannot hold one aligned "
                      "element span");

  const uint64_t startBit = state.nextPhysicalElement * state.elementBitWidth;
  const uint64_t spanBits = physicalCount * state.elementBitWidth;
  const uint64_t spanBytes = (spanBits + 7) / 8;
  if (spanBytes > std::numeric_limits<size_t>::max())
    return codecError(PhysicalTensorCodecErrorCode::InvalidStorageSize,
                      "window span exceeds host size_t");
  if (startBit % 8 != 0)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "window start is not byte-aligned");

  WriteWindow window;
  window.physicalOffset = startBit / 8;
  window.bytes.assign(static_cast<size_t>(spanBytes), paddingFill);
  window.elements.reserve(static_cast<size_t>(
      std::min(physicalCount,
               static_cast<uint64_t>(std::numeric_limits<size_t>::max()))));
  for (uint64_t index = 0; index < physicalCount; ++index) {
    const uint64_t physicalElement = state.nextPhysicalElement + index;
    std::optional<uint64_t> logicalIndex =
        state.logicalIndexForPhysicalElement(physicalElement);
    if (!logicalIndex)
      continue;
    window.elements.push_back(
        ElementWrite{*logicalIndex, index * state.elementBitWidth});
  }
  const uint64_t nextPlanned =
      state.plannedLogicalElements + window.elements.size();
  const uint64_t nextPhysical = state.nextPhysicalElement + physicalCount;
  if (nextPlanned > state.elementCount ||
      (nextPhysical == state.physicalElementCount &&
       nextPlanned != state.elementCount))
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "inverse physical geometry does not cover each logical "
                      "element exactly once");
  state.plannedLogicalElements = nextPlanned;
  state.nextPhysicalElement = nextPhysical;
  return window;
}

} // namespace wafer
