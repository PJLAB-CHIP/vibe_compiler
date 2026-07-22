//===- Tx81ModelABI.cpp - Qualified TX81 model launch wire ABI ----------===//

#include "Wafer/Runtime/Tx81ModelABI.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>

namespace wafer::runtime {
namespace {

struct BootParamHeadLayout {
  uint32_t maxLen;
  uint32_t localMemoryLen;
  uint32_t inputCount;
  uint32_t outputCount;
  uint32_t parameterCount;
  uint32_t reserved;
  uint64_t cacheMemoryLen;
  uint64_t cacheMemoryAddress;
  uint32_t dataLen;
  uint32_t calculationCount;
  uint64_t dataAddress;
};

struct BootParamDyninfoLayout {
  uint64_t address;
  uint64_t size;
  uint32_t dtype;
  uint32_t dimensionCount;
  uint64_t shape[6];
};

struct GraphInfoLayout {
  char moduleName[128];
  char moduleSymbol[128];
  uint32_t moduleSize[16];
  uint64_t moduleAddress[16];
};

struct DynModsLayout {
  uint16_t moduleCount;
  uint8_t padding[6];
  GraphInfoLayout graph;
};

struct GraphTLVLayout {
  uint32_t type;
  uint32_t length;
  uint64_t dynModsAddress;
};

static_assert(std::is_standard_layout_v<BootParamHeadLayout>);
static_assert(sizeof(BootParamHeadLayout) == 56);
static_assert(offsetof(BootParamHeadLayout, cacheMemoryLen) == 24);
static_assert(offsetof(BootParamHeadLayout, cacheMemoryAddress) == 32);
static_assert(offsetof(BootParamHeadLayout, dataLen) == 40);
static_assert(offsetof(BootParamHeadLayout, calculationCount) == 44);
static_assert(offsetof(BootParamHeadLayout, dataAddress) == 48);
static_assert(sizeof(BootParamDyninfoLayout) == 72);
static_assert(offsetof(BootParamDyninfoLayout, dtype) == 16);
static_assert(offsetof(BootParamDyninfoLayout, shape) == 24);
static_assert(sizeof(GraphInfoLayout) == 448);
static_assert(offsetof(GraphInfoLayout, moduleSize) == 256);
static_assert(offsetof(GraphInfoLayout, moduleAddress) == 320);
static_assert(sizeof(DynModsLayout) == 456);
static_assert(offsetof(DynModsLayout, graph) == 8);
static_assert(sizeof(GraphTLVLayout) == 16);

constexpr uint32_t kF32DataFormat = 5;
constexpr uint32_t kDynlibRunType = 7;
constexpr uint32_t kDynlibRunPayloadLength = sizeof(uint64_t);
constexpr uint32_t kNoKcoreCalculationCount = 0xffffffffU;
constexpr uint32_t kQualifiedLocalMemoryBytes = 0x00200000U;

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

template <typename T>
void writeLittleEndian(std::vector<uint8_t> &bytes, size_t offset, T value) {
  static_assert(std::is_unsigned_v<T>);
  for (size_t index = 0; index < sizeof(T); ++index)
    bytes[offset + index] =
        static_cast<uint8_t>(value >> static_cast<unsigned>(index * 8));
}

uint8_t classOrder(Tx81ModelTensorClass tensorClass) {
  switch (tensorClass) {
  case Tx81ModelTensorClass::Input:
    return 0;
  case Tx81ModelTensorClass::Output:
    return 1;
  case Tx81ModelTensorClass::Parameter:
    return 2;
  }
  llvm_unreachable("unknown TX81 model tensor class");
}

} // namespace

llvm::Expected<Tx81ModelBootParamImage> buildTx81ModelBootParam(
    llvm::ArrayRef<Tx81ModelTensorDescriptor> tensors,
    uint64_t dynamicTLVDeviceAddress) {
  if (tensors.empty())
    return invalid("TX81 model BootParam requires at least one tensor");
  if (dynamicTLVDeviceAddress == 0 ||
      dynamicTLVDeviceAddress % alignof(uint64_t) != 0)
    return invalid(
        "TX81 model BootParam TLV device address is zero or misaligned");

  Tx81ModelBootParamImage image;
  image.canonicalTensors.assign(tensors.begin(), tensors.end());
  llvm::sort(image.canonicalTensors, [](const auto &lhs, const auto &rhs) {
    return std::tuple(classOrder(lhs.tensorClass), lhs.logicalRank,
                      lhs.slotOrdinal) <
           std::tuple(classOrder(rhs.tensorClass), rhs.logicalRank,
                      rhs.slotOrdinal);
  });

  uint64_t inputCount = 0;
  uint64_t outputCount = 0;
  uint64_t parameterCount = 0;
  std::set<std::tuple<uint8_t, int64_t, uint64_t>> identities;
  for (const Tx81ModelTensorDescriptor &tensor : image.canonicalTensors) {
    if (tensor.logicalRank < 0 || tensor.logicalRank >= 16)
      return invalid("TX81 model tensor logical rank is outside 0..15");
    if (!identities
             .emplace(classOrder(tensor.tensorClass), tensor.logicalRank,
                      tensor.slotOrdinal)
             .second)
      return invalid("TX81 model tensor identity is duplicated");
    if (tensor.deviceAddress == 0 ||
        tensor.deviceAddress % alignof(float) != 0 || tensor.bytes == 0)
      return invalid(
          "TX81 model tensor has a zero/misaligned address or byte count");
    if (tensor.dtype != "f32")
      return invalid("TX81 model BootParam v1 accepts only f32 tensors");
    if (tensor.shape.empty() || tensor.shape.size() > 6)
      return invalid("TX81 model tensor rank must be between one and six");
    uint64_t elementCount = 1;
    for (int64_t dimension : tensor.shape) {
      if (dimension <= 0 ||
          static_cast<uint64_t>(dimension) >
              std::numeric_limits<uint64_t>::max() / elementCount)
        return invalid("TX81 model tensor shape is invalid or overflows");
      elementCount *= static_cast<uint64_t>(dimension);
    }
    if (elementCount > std::numeric_limits<uint64_t>::max() / sizeof(float) ||
        elementCount * sizeof(float) != tensor.bytes)
      return invalid("TX81 model f32 tensor shape does not match byte count");

    switch (tensor.tensorClass) {
    case Tx81ModelTensorClass::Input:
      ++inputCount;
      break;
    case Tx81ModelTensorClass::Output:
      ++outputCount;
      break;
    case Tx81ModelTensorClass::Parameter:
      ++parameterCount;
      break;
    }
  }
  if (inputCount == 0 || outputCount == 0)
    return invalid("TX81 model BootParam requires input and output tensors");
  // The only exact MaxLen constructor currently qualified is the legacy
  // parameter-free form, which reserves one trailing 72-byte record.
  if (parameterCount != 0)
    return invalid("TX81 model BootParam v1 has no qualified parameter layout");
  if (inputCount > std::numeric_limits<uint32_t>::max() ||
      outputCount > std::numeric_limits<uint32_t>::max())
    return invalid("TX81 model tensor count exceeds uint32");

  constexpr uint64_t headBytes = sizeof(BootParamHeadLayout);
  constexpr uint64_t dyninfoBytes = sizeof(BootParamDyninfoLayout);
  uint64_t recordCount = inputCount + outputCount + 1;
  if (recordCount >
      (std::numeric_limits<uint32_t>::max() - headBytes) / dyninfoBytes)
    return invalid("TX81 model BootParam byte count exceeds uint32");
  uint32_t totalBytes =
      static_cast<uint32_t>(headBytes + recordCount * dyninfoBytes);
  image.bytes.assign(totalBytes, 0);

  writeLittleEndian<uint32_t>(image.bytes, 0, totalBytes);
  writeLittleEndian<uint32_t>(image.bytes, 4, kQualifiedLocalMemoryBytes);
  writeLittleEndian<uint32_t>(image.bytes, 8,
                              static_cast<uint32_t>(inputCount));
  writeLittleEndian<uint32_t>(image.bytes, 12,
                              static_cast<uint32_t>(outputCount));
  writeLittleEndian<uint32_t>(image.bytes, 16, 0);
  writeLittleEndian<uint32_t>(image.bytes, 20, 0);
  writeLittleEndian<uint64_t>(image.bytes, 24, 0);
  writeLittleEndian<uint64_t>(image.bytes, 32, 0);
  writeLittleEndian<uint32_t>(image.bytes, 40, sizeof(GraphTLVLayout));
  writeLittleEndian<uint32_t>(image.bytes, 44,
                              kNoKcoreCalculationCount);
  writeLittleEndian<uint64_t>(image.bytes, 48, dynamicTLVDeviceAddress);

  for (auto [index, tensor] : llvm::enumerate(image.canonicalTensors)) {
    size_t offset = headBytes + index * dyninfoBytes;
    writeLittleEndian<uint64_t>(image.bytes, offset, tensor.deviceAddress);
    writeLittleEndian<uint64_t>(image.bytes, offset + 8, tensor.bytes);
    writeLittleEndian<uint32_t>(image.bytes, offset + 16, kF32DataFormat);
    writeLittleEndian<uint32_t>(
        image.bytes, offset + 20,
        static_cast<uint32_t>(tensor.shape.size()));
    for (auto [dimension, extent] : llvm::enumerate(tensor.shape))
      writeLittleEndian<uint64_t>(image.bytes, offset + 24 + dimension * 8,
                                  static_cast<uint64_t>(extent));
  }
  return image;
}

llvm::Expected<std::vector<uint8_t>>
buildTx81DynlibRunModules(llvm::StringRef moduleName) {
  if (moduleName.empty() || moduleName.size() >= 128 || moduleName.contains('\0'))
    return invalid("TX81 type-7 module name has invalid length or NUL");
  std::vector<uint8_t> bytes(sizeof(DynModsLayout), 0);
  writeLittleEndian<uint16_t>(bytes, 0, 1);
  std::copy(moduleName.bytes_begin(), moduleName.bytes_end(), bytes.begin() + 8);
  return bytes;
}

llvm::Expected<std::vector<uint8_t>>
buildTx81DynlibRunTLV(uint64_t dynModsDeviceAddress) {
  if (dynModsDeviceAddress == 0 ||
      dynModsDeviceAddress % alignof(uint64_t) != 0)
    return invalid("TX81 type-7 DynMods device address is zero or misaligned");
  std::vector<uint8_t> bytes(sizeof(GraphTLVLayout), 0);
  writeLittleEndian<uint32_t>(bytes, 0, kDynlibRunType);
  writeLittleEndian<uint32_t>(bytes, 4, kDynlibRunPayloadLength);
  writeLittleEndian<uint64_t>(bytes, 8, dynModsDeviceAddress);
  return bytes;
}

} // namespace wafer::runtime
