//===- PhysicalTensorCodec.cpp - Wafer physical tensor codec ------------===//

#include "Wafer/Target/PhysicalTensorCodec.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>
#include <memory>
#include <optional>

namespace wafer {
namespace {

llvm::Error codecError(PhysicalTensorCodecErrorCode code,
                       const llvm::Twine &detail) {
  return llvm::make_error<PhysicalTensorCodecError>(code, detail.str());
}

std::optional<mlir::Type> getElementType(mlir::MLIRContext &context,
                                         LogicalFormat format) {
  switch (format) {
  case LogicalFormat::I8:
  case LogicalFormat::U8:
    return mlir::IntegerType::get(&context, 8);
  case LogicalFormat::I16:
  case LogicalFormat::U16:
    return mlir::IntegerType::get(&context, 16);
  case LogicalFormat::I32:
  case LogicalFormat::U32:
    return mlir::IntegerType::get(&context, 32);
  case LogicalFormat::I64:
  case LogicalFormat::U64:
    return mlir::IntegerType::get(&context, 64);
  case LogicalFormat::F16:
    return mlir::Float16Type::get(&context);
  case LogicalFormat::BF16:
    return mlir::BFloat16Type::get(&context);
  case LogicalFormat::F32:
  case LogicalFormat::TF32:
    return mlir::Float32Type::get(&context);
  case LogicalFormat::Bool:
    return mlir::IntegerType::get(&context, 1);
  }
  return std::nullopt;
}

struct OwnedPhysicalLayout {
  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::MemRefType type;
  WaferPhysicalTensorInfo info;
  std::optional<WaferStaticPhysicalOffsetCalculator> byteOffsets;
};

llvm::Expected<OwnedPhysicalLayout>
makePhysicalLayout(const NumericTensorKey &key) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadDialect<WaferDialect>();
  std::optional<mlir::Type> elementType =
      getElementType(*context, key.getFormat());
  if (!elementType)
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
  MemoryAttr memory =
      MemoryAttr::get(context.get(), MemorySpace::SPM, key.getLayout());
  mlir::MemRefType type = mlir::MemRefType::get(
      shape, *elementType, mlir::MemRefLayoutAttrInterface{}, memory);
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info || info->physicalBytes < 0)
    return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                      "shared layout helper rejected the static tensor");
  std::optional<WaferStaticPhysicalOffsetCalculator> byteOffsets;
  if (!info->bitPackedElement) {
    byteOffsets = WaferStaticPhysicalOffsetCalculator::create(type);
    if (!byteOffsets)
      return codecError(PhysicalTensorCodecErrorCode::InvalidLayout,
                        "static byte-offset calculator rejected the tensor");
  }
  return OwnedPhysicalLayout{std::move(registry), std::move(context), type,
                             std::move(*info), std::move(byteOffsets)};
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
  if (!layout.info.bitPackedElement &&
      (layout.info.layout == MemLayout::Tensor ||
       layout.info.layout == MemLayout::NTensor) &&
      layout.info.compactBytes == layout.info.physicalBytes) {
    const uint64_t elementBits =
        static_cast<uint64_t>(layout.info.elementBytes) * UINT64_C(8);
    for (uint64_t index = 0; index < key.getElementCount(); ++index)
      if (llvm::Error error = callback(index * elementBits))
        return error;
    return llvm::Error::success();
  }
  return forEachCoordinate(
      key.getShape(), [&](llvm::ArrayRef<int64_t> coordinate) -> llvm::Error {
        int64_t bitOffset = -1;
        if (layout.info.bitPackedElement) {
          std::optional<int64_t> mapped =
              computeWaferPhysicalElementBitOffset(layout.type, coordinate);
          if (mapped)
            bitOffset = *mapped;
        } else if (layout.byteOffsets) {
          int64_t byteOffset =
              layout.byteOffsets->getByteOffsetForValidIndices(coordinate);
          if (byteOffset >= 0 &&
              byteOffset <=
                  layout.info.physicalBytes - layout.info.elementBytes &&
              byteOffset <= std::numeric_limits<int64_t>::max() / 8)
            bitOffset = byteOffset * 8;
        }
        if (bitOffset < 0)
          return codecError(
              PhysicalTensorCodecErrorCode::InvalidLayout,
              "shared layout helper could not map a logical coordinate");
        return callback(static_cast<uint64_t>(bitOffset));
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
      getModelProfileRecord(ModelProfileId::formalDeterministicV1())
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
      getModelProfileRecord(ModelProfileId::formalDeterministicV1())
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

} // namespace wafer
