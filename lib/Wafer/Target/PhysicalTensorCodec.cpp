//===- PhysicalTensorCodec.cpp - Wafer physical tensor codec ------------===//

#include "Wafer/Target/PhysicalTensorCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>
#include <optional>

namespace wafer {
namespace {

llvm::Error codecError(PhysicalTensorCodecErrorCode code,
                       const llvm::Twine &detail) {
  return llvm::make_error<PhysicalTensorCodecError>(code, detail.str());
}

struct OwnedPhysicalLayout {
  PhysicalTensorGeometry info;
  StaticPhysicalTensorOffsetCalculator offsets;
};

llvm::Expected<OwnedPhysicalLayout>
makePhysicalLayout(const NumericTensorKey &key) {
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
llvm::Error forEachPhysicalBitOffset(const NumericTensorKey &key,
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
getPhysicalTensorStorageBytes(const NumericTensorKey &key) {
  llvm::Expected<OwnedPhysicalLayout> layout = makePhysicalLayout(key);
  if (!layout)
    return layout.takeError();
  return static_cast<uint64_t>(layout->info.physicalBytes);
}

llvm::Expected<std::vector<RawLogicalValue>>
unpackPhysicalTensorLogicalValues(const NumericTensorKey &key,
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
  const LogicalScalarCodecPolicy policy =
      getModelProfileRecord(ModelProfileId::formalDeterministic())
          .numericDecodePolicy;
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
packPhysicalTensorLogicalValues(const NumericTensorKey &key,
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
  const LogicalScalarCodecPolicy policy =
      getModelProfileRecord(ModelProfileId::formalDeterministic())
          .numericEncodePolicy;
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
packPhysicalTensorLogicalValues(const NumericTensorKey &key,
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


struct PhysicalTensorWindowPacker::Impl {
  NumericTensorKey key;
  OwnedPhysicalLayout layout;
  uint64_t storageBytes;
  uint64_t elementCount;
  int64_t elementBitWidth;
  int64_t blockedWindowBoundary;
  llvm::SmallVector<int64_t, 4> shape;
  uint64_t nextValueIndex = 0;

  /// Maps a row-major logical index to its coordinate.
  llvm::SmallVector<int64_t, 4> coordinateForValue(uint64_t index) const {
    llvm::SmallVector<int64_t, 4> coordinate(shape.size(), 0);
    uint64_t remaining = index;
    for (size_t reverse = shape.size(); reverse > 0; --reverse) {
      size_t dim = reverse - 1;
      coordinate[dim] =
          static_cast<int64_t>(remaining % static_cast<uint64_t>(shape[dim]));
      remaining /= static_cast<uint64_t>(shape[dim]);
    }
    return coordinate;
  }

  llvm::Expected<uint64_t> bitOffsetForValue(uint64_t index) const {
    std::optional<int64_t> offset =
        layout.offsets.getBitOffset(coordinateForValue(index));
    if (!offset || *offset < 0)
      return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                        "shared layout helper could not map a window start");
    return static_cast<uint64_t>(*offset);
  }
};

PhysicalTensorWindowPacker::PhysicalTensorWindowPacker() = default;
PhysicalTensorWindowPacker::PhysicalTensorWindowPacker(
    PhysicalTensorWindowPacker &&) noexcept = default;
PhysicalTensorWindowPacker &
PhysicalTensorWindowPacker::operator=(PhysicalTensorWindowPacker &&) noexcept =
    default;
PhysicalTensorWindowPacker::~PhysicalTensorWindowPacker() = default;

llvm::Expected<PhysicalTensorWindowPacker>
PhysicalTensorWindowPacker::create(const NumericTensorKey &key) {
  llvm::Expected<OwnedPhysicalLayout> layout = makePhysicalLayout(key);
  if (!layout)
    return layout.takeError();
  const LogicalFormatDescriptor *format =
      findLogicalFormatDescriptor(key.getFormat());
  if (!format)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "tensor has an unknown format or layout");
  if (format->storageBits <= 0 || format->storageBits == 1)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "bit-packed window packing is unsupported");
  int64_t blockedWindowBoundary = 0;
  if (key.getLayout() == PhysicalTensorLayout::Cx ||
      key.getLayout() == PhysicalTensorLayout::NCx) {
    if (key.getShape().empty())
      return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                        "blocked layout requires a channel dimension");
    blockedWindowBoundary =
        static_cast<int64_t>(key.getShape().back());
  }
  llvm::SmallVector<int64_t, 4> shape;
  for (uint64_t dimension : key.getShape())
    shape.push_back(static_cast<int64_t>(dimension));
  PhysicalTensorWindowPacker packer;
  packer.impl = std::make_unique<Impl>(
      Impl{key, std::move(*layout),
           static_cast<uint64_t>(layout->info.physicalBytes),
           key.getElementCount(), format->storageBits,
           blockedWindowBoundary, std::move(shape)});
  return packer;
}

uint64_t PhysicalTensorWindowPacker::getStorageBytes() const {
  return impl ? impl->storageBytes : 0;
}
uint64_t PhysicalTensorWindowPacker::getElementCount() const {
  return impl ? impl->elementCount : 0;
}
uint64_t PhysicalTensorWindowPacker::getPackedValueCount() const {
  return impl ? impl->nextValueIndex : 0;
}

llvm::Expected<PhysicalTensorWindowPacker::WriteWindow>
PhysicalTensorWindowPacker::packNext(llvm::ArrayRef<RawLogicalValue> values,
                                     uint64_t budgetBytes,
                                     uint8_t paddingFill) {
  if (!impl)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "window packer was moved from");
  Impl &state = *impl;
  if (state.nextValueIndex == state.elementCount)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLogicalValueCount,
                      "window packer is exhausted");
  if (values.empty())
    return codecError(PhysicalTensorCodecErrorCode::InvalidLogicalValueCount,
                      "window packer requires at least one value");
  const uint64_t remainingValues =
      state.elementCount - state.nextValueIndex;
  uint64_t valueCount = std::min<uint64_t>(
      values.size(), remainingValues);
  // Linear layouts accept any window; blocked layouts may only end at a
  // channel-row boundary so the packed span stays contiguous.
  if (state.blockedWindowBoundary > 0) {
    const uint64_t boundary = static_cast<uint64_t>(state.blockedWindowBoundary);
    const uint64_t nextBoundary = llvm::alignTo(state.nextValueIndex, boundary);
    if (nextBoundary < state.nextValueIndex + valueCount)
      valueCount = nextBoundary - state.nextValueIndex;
    if (valueCount == 0)
      valueCount = std::min<uint64_t>(boundary, remainingValues);
  }
  if (valueCount == 0)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLogicalValueCount,
                      "window packer produced an empty window");
  // Cap by the byte budget; the single-row fallback above may exceed it.
  const uint64_t bitsPerWindow = static_cast<uint64_t>(state.elementBitWidth) *
                                 valueCount;
  if (bitsPerWindow > 0 &&
      (bitsPerWindow + 7) / 8 > budgetBytes &&
      state.blockedWindowBoundary == 0) {
    uint64_t budgetValues = budgetBytes * 8 /
                            static_cast<uint64_t>(state.elementBitWidth);
    valueCount = std::max<uint64_t>(1, budgetValues);
  }
  uint64_t startBit = 0;
  llvm::Expected<uint64_t> start =
      state.bitOffsetForValue(state.nextValueIndex);
  if (!start)
    return start.takeError();
  startBit = *start;
  llvm::Expected<uint64_t> last =
      state.bitOffsetForValue(state.nextValueIndex + valueCount - 1);
  if (!last)
    return last.takeError();
  const uint64_t spanBits =
      *last + static_cast<uint64_t>(state.elementBitWidth) - startBit;
  const uint64_t spanBytes = (spanBits + 7) / 8;
  if (spanBytes > std::numeric_limits<size_t>::max())
    return codecError(PhysicalTensorCodecErrorCode::InvalidStorageSize,
                      "window span exceeds host size_t");
  if (startBit % 8 != 0)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "window start is not byte-aligned");

  const LogicalScalarCodecPolicy policy =
      getModelProfileRecord(ModelProfileId::formalDeterministic())
          .numericEncodePolicy;
  std::vector<uint8_t> span(static_cast<size_t>(spanBytes), paddingFill);
  for (uint64_t windowIndex = 0; windowIndex < valueCount; ++windowIndex) {
    RawLogicalValue value = values[static_cast<size_t>(windowIndex)];
    if (value.format != state.key.getFormat())
      return codecError(PhysicalTensorCodecErrorCode::InvalidLogicalEncoding,
                        "logical value format does not match the tensor key");
    llvm::Expected<uint64_t> offset = state.bitOffsetForValue(
        state.nextValueIndex + windowIndex);
    if (!offset)
      return offset.takeError();
    if (llvm::Error writeError = writeRawLogicalValue(
            value, span, *offset - startBit, policy))
      return codecError(PhysicalTensorCodecErrorCode::InvalidLogicalEncoding,
                        llvm::toString(std::move(writeError)));
  }
  state.nextValueIndex += valueCount;
  WriteWindow window;
  window.physicalOffset = startBit / 8;
  window.valueCount = valueCount;
  window.bytes = std::move(span);
  return window;
}

} // namespace wafer
