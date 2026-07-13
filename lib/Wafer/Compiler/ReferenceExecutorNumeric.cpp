//===- ReferenceExecutorNumeric.cpp - Reference numeric semantics --------===//

#include "ReferenceExecutorInternal.h"

#include "llvm/ADT/APSInt.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace wafer::compiler::reference_detail {

std::optional<int64_t> getDTypeByteWidth(llvm::StringRef dtype) {
  if (dtype == "i1")
    return std::nullopt;
  if (dtype == "i8" || dtype == "ui8")
    return 1;
  if (dtype == "i16" || dtype == "ui16" || dtype == "f16" || dtype == "bf16")
    return 2;
  if (dtype == "i32" || dtype == "ui32" || dtype == "f32" || dtype == "tf32")
    return 4;
  if (dtype == "i64" || dtype == "ui64" || dtype == "f64")
    return 8;
  return std::nullopt;
}

std::optional<int64_t> getCompactByteCount(llvm::StringRef dtype,
                                           llvm::ArrayRef<int64_t> shape) {
  std::optional<int64_t> elementBytes = getDTypeByteWidth(dtype);
  if (!elementBytes)
    return std::nullopt;
  int64_t bytes = *elementBytes;
  for (int64_t dim : shape) {
    if (dim < 0 ||
        (dim != 0 && bytes > std::numeric_limits<int64_t>::max() / dim))
      return std::nullopt;
    bytes *= dim;
  }
  return bytes;
}

std::string elementDType(mlir::Type type) {
  if (mlir::isa<mlir::FloatTF32Type>(type))
    return "tf32";
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return (integer.isUnsigned() ? "ui" : "i") +
           std::to_string(integer.getWidth());
  return {};
}

std::optional<NumericFormat> getNumericFormat(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (!integer.isSignless())
      return std::nullopt;
    if (integer.getWidth() == 8)
      return NumericFormat::Int8;
    if (integer.getWidth() == 16)
      return NumericFormat::Int16;
    if (integer.getWidth() == 32)
      return NumericFormat::Int32;
    return std::nullopt;
  }
  if (type.isBF16())
    return NumericFormat::BFloat16;
  if (type.isF16())
    return NumericFormat::Float16;
  if (mlir::isa<mlir::FloatTF32Type>(type))
    return NumericFormat::TF32;
  if (type.isF32())
    return NumericFormat::Float32;
  return std::nullopt;
}

bool matchesNumericFormat(mlir::Type type, NumericFormat format) {
  return getNumericFormat(type) == format;
}

unsigned getNumericBitWidth(NumericFormat format) {
  switch (format) {
  case NumericFormat::Int8:
    return 8;
  case NumericFormat::Int16:
  case NumericFormat::BFloat16:
  case NumericFormat::Float16:
    return 16;
  case NumericFormat::Int32:
  case NumericFormat::Float32:
  case NumericFormat::TF32:
    return 32;
  }
  llvm_unreachable("unknown reference numeric format");
}

const llvm::fltSemantics *getFloatSemantics(NumericFormat format) {
  switch (format) {
  case NumericFormat::BFloat16:
    return &llvm::APFloat::BFloat();
  case NumericFormat::Float16:
    return &llvm::APFloat::IEEEhalf();
  case NumericFormat::Float32:
    return &llvm::APFloat::IEEEsingle();
  case NumericFormat::TF32:
    return &llvm::APFloat::FloatTF32();
  case NumericFormat::Int8:
  case NumericFormat::Int16:
  case NumericFormat::Int32:
    return nullptr;
  }
  llvm_unreachable("unknown reference numeric format");
}

llvm::Expected<int64_t> getAbsoluteOffset(const BufferView &buffer,
                                          llvm::ArrayRef<int64_t> indices,
                                          int64_t byteWidth) {
  std::optional<int64_t> relative =
      wafer::computeWaferPhysicalElementByteOffset(buffer.type, indices);
  if (!relative)
    return invalid("cannot map logical index to accepted physical layout");
  if (*relative < 0 || byteWidth < 0 ||
      *relative > buffer.physicalBytes - byteWidth)
    return invalid("logical access exceeds accepted buffer extent");
  int64_t absolute = buffer.base + buffer.viewOffset + *relative;
  if (absolute < 0 ||
      absolute > static_cast<int64_t>(buffer.storage->bytes.size()) - byteWidth)
    return invalid("logical access exceeds reference storage arena");
  return absolute;
}

llvm::Expected<NumericValue> readNumeric(const BufferView &buffer,
                                         llvm::ArrayRef<int64_t> indices,
                                         NumericFormat format) {
  if (!matchesNumericFormat(buffer.type.getElementType(), format))
    return invalid("projected convert source format disagrees with buffer");
  unsigned bitWidth = getNumericBitWidth(format);
  int64_t byteWidth = bitWidth / 8;
  auto offset = getAbsoluteOffset(buffer, indices, byteWidth);
  if (!offset)
    return offset.takeError();
  llvm::APInt bits(bitWidth, 0);
  for (int64_t byte = 0; byte < byteWidth; ++byte)
    bits |= llvm::APInt(bitWidth, buffer.storage->bytes[*offset + byte])
            << (byte * 8);
  if (const llvm::fltSemantics *semantics = getFloatSemantics(format)) {
    if (format == NumericFormat::TF32) {
      uint64_t storage = bits.getZExtValue();
      uint64_t compact = ((storage >> 31) << 18) |
                         (((storage >> 23) & 0xff) << 10) |
                         ((storage >> 13) & 0x3ff);
      bits = llvm::APInt(/*numBits=*/19, compact);
    }
    return NumericValue{std::nullopt, llvm::APFloat(*semantics, bits)};
  }
  return NumericValue{std::move(bits), std::nullopt};
}

llvm::Expected<llvm::APInt>
convertNumeric(const NumericValue &source, NumericFormat destFormat,
               llvm::APFloat::roundingMode roundingMode) {
  const llvm::fltSemantics *destSemantics = getFloatSemantics(destFormat);
  if (source.integer && destSemantics) {
    llvm::APFloat result = llvm::APFloat::getZero(*destSemantics);
    result.convertFromAPInt(*source.integer, /*IsSigned=*/true, roundingMode);
    return result.bitcastToAPInt();
  }
  if (source.floating && destSemantics) {
    llvm::APFloat result = *source.floating;
    bool losesInfo = false;
    result.convert(*destSemantics, roundingMode, &losesInfo);
    return result.bitcastToAPInt();
  }
  if (source.floating && !destSemantics) {
    llvm::APSInt result(getNumericBitWidth(destFormat), /*isUnsigned=*/false);
    bool isExact = false;
    llvm::APFloat::opStatus status =
        source.floating->convertToInteger(result, roundingMode, &isExact);
    if ((status & llvm::APFloat::opInvalidOp) != 0)
      return invalid(
          "floating-to-integer convert input is NaN, Inf, or out of range");
    return llvm::APInt(result);
  }
  return invalid("projected convert has unsupported integer-to-integer pair");
}

static llvm::Expected<double> numericAsDouble(const NumericValue &source) {
  if (source.integer) {
    if (source.integer->getBitWidth() > 64)
      return invalid("stochastic integer source is wider than 64 bits");
    return static_cast<double>(source.integer->getSExtValue());
  }
  if (source.floating) {
    llvm::APFloat value = *source.floating;
    bool losesInfo = false;
    value.convert(llvm::APFloat::IEEEdouble(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return value.convertToDouble();
  }
  return invalid("stochastic convert source has no numeric value");
}

static double randomUnitInterval(uint64_t randomBits) {
  return static_cast<double>(randomBits >> 11) * 0x1.0p-53;
}

llvm::Expected<llvm::APInt> convertNumericStochastic(const NumericValue &source,
                                                     NumericFormat destFormat,
                                                     uint64_t randomBits) {
  auto sourceValue = numericAsDouble(source);
  if (!sourceValue)
    return sourceValue.takeError();

  const llvm::fltSemantics *destSemantics = getFloatSemantics(destFormat);
  if (!destSemantics) {
    if (!source.floating)
      return invalid(
          "projected stochastic convert has unsupported integer pair");
    if (!std::isfinite(*sourceValue))
      return invalid(
          "floating-to-integer convert input is NaN, Inf, or out of range");
    unsigned width = getNumericBitWidth(destFormat);
    double minimum = -std::ldexp(1.0, width - 1);
    double maximum = std::ldexp(1.0, width - 1) - 1.0;
    if (*sourceValue < minimum || *sourceValue > maximum)
      return invalid(
          "floating-to-integer convert input is NaN, Inf, or out of range");
    double lower = std::floor(*sourceValue);
    double upper = std::ceil(*sourceValue);
    double chosen = lower;
    if (upper != lower &&
        randomUnitInterval(randomBits) < (*sourceValue - lower))
      chosen = upper;
    int64_t integer = static_cast<int64_t>(chosen);
    return llvm::APInt(width, static_cast<uint64_t>(integer),
                       /*isSigned=*/true);
  }

  if (source.floating && !std::isfinite(*sourceValue))
    return convertNumeric(source, destFormat,
                          llvm::APFloat::rmNearestTiesToEven);

  auto makeBound = [&](llvm::APFloat::roundingMode roundingMode) {
    if (source.integer) {
      llvm::APFloat result = llvm::APFloat::getZero(*destSemantics);
      result.convertFromAPInt(*source.integer, /*IsSigned=*/true, roundingMode);
      return result;
    }
    llvm::APFloat result = *source.floating;
    bool losesInfo = false;
    result.convert(*destSemantics, roundingMode, &losesInfo);
    return result;
  };
  llvm::APFloat lower = makeBound(llvm::APFloat::rmTowardNegative);
  llvm::APFloat upper = makeBound(llvm::APFloat::rmTowardPositive);
  if (lower.bitcastToAPInt() == upper.bitcastToAPInt())
    return lower.bitcastToAPInt();
  if (!lower.isFinite() || !upper.isFinite())
    return invalid("stochastic convert input exceeds finite destination range");

  auto lowerValue = numericAsDouble(
      NumericValue{std::nullopt, std::optional<llvm::APFloat>(lower)});
  auto upperValue = numericAsDouble(
      NumericValue{std::nullopt, std::optional<llvm::APFloat>(upper)});
  if (!lowerValue)
    return lowerValue.takeError();
  if (!upperValue)
    return upperValue.takeError();
  double interval = *upperValue - *lowerValue;
  if (!(interval > 0.0) || *sourceValue < *lowerValue ||
      *sourceValue > *upperValue)
    return invalid("stochastic convert could not bracket source value");
  double upperProbability = (*sourceValue - *lowerValue) / interval;
  return (randomUnitInterval(randomBits) < upperProbability ? upper : lower)
      .bitcastToAPInt();
}

llvm::Error writeNumericBits(const BufferView &buffer,
                             llvm::ArrayRef<int64_t> indices,
                             NumericFormat format, const llvm::APInt &bits) {
  if (!matchesNumericFormat(buffer.type.getElementType(), format))
    return invalid(
        "projected convert destination format disagrees with buffer");
  llvm::APInt storageBits = bits;
  if (format == NumericFormat::TF32) {
    if (bits.getBitWidth() != 19)
      return invalid(
          "projected TF32 convert produced a non-TF32 semantic width");
    uint64_t compact = bits.getZExtValue();
    uint64_t storage = ((compact >> 18) << 31) |
                       (((compact >> 10) & 0xff) << 23) |
                       ((compact & 0x3ff) << 13);
    storageBits = llvm::APInt(/*numBits=*/32, storage);
  } else if (bits.getBitWidth() != getNumericBitWidth(format)) {
    return invalid(
        "projected convert destination format disagrees with buffer");
  }
  int64_t byteWidth = storageBits.getBitWidth() / 8;
  auto offset = getAbsoluteOffset(buffer, indices, byteWidth);
  if (!offset)
    return offset.takeError();
  for (int64_t byte = 0; byte < byteWidth; ++byte)
    buffer.storage->bytes[*offset + byte] =
        static_cast<uint8_t>(storageBits.extractBitsAsZExtValue(8, byte * 8));
  return llvm::Error::success();
}

llvm::Expected<float> readF32(const BufferView &buffer,
                              llvm::ArrayRef<int64_t> indices) {
  if (!buffer.type.getElementType().isF32())
    return invalid("reference floating compute currently requires f32");
  auto offset = getAbsoluteOffset(buffer, indices, sizeof(float));
  if (!offset)
    return offset.takeError();
  float value;
  std::memcpy(&value, buffer.storage->bytes.data() + *offset, sizeof(value));
  return value;
}

llvm::Error writeF32(const BufferView &buffer, llvm::ArrayRef<int64_t> indices,
                     float value) {
  if (!buffer.type.getElementType().isF32())
    return invalid("reference floating compute currently requires f32");
  auto offset = getAbsoluteOffset(buffer, indices, sizeof(float));
  if (!offset)
    return offset.takeError();
  std::memcpy(buffer.storage->bytes.data() + *offset, &value, sizeof(value));
  return llvm::Error::success();
}

static llvm::Expected<int64_t>
getAbsoluteBitOffset(const BufferView &buffer,
                     llvm::ArrayRef<int64_t> indices) {
  auto info = wafer::computeWaferPhysicalTensorInfo(buffer.type);
  if (!buffer.type.getElementType().isInteger(1) || !info ||
      !info->bitPackedElement ||
      (info->layout != wafer::MemLayout::Tensor &&
       info->layout != wafer::MemLayout::NTensor))
    return invalid("reference i1 access requires bitpacked tensor layout");
  if (indices.size() != static_cast<size_t>(buffer.type.getRank()))
    return invalid("reference i1 index rank mismatch");
  llvm::SmallVector<int64_t> strides;
  int64_t elementOffset = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(buffer.type, strides, elementOffset)) ||
      elementOffset == mlir::ShapedType::kDynamic || elementOffset < 0 ||
      strides.size() != indices.size())
    return invalid("reference i1 buffer has no static logical strides");
  for (auto [dimension, index, stride] :
       llvm::zip_equal(buffer.type.getShape(), indices, strides)) {
    if (dimension < 0 || index < 0 || index >= dimension || stride < 0 ||
        stride == mlir::ShapedType::kDynamic)
      return invalid("reference i1 logical index is invalid");
    if (index != 0 && stride > std::numeric_limits<int64_t>::max() / index)
      return invalid("reference i1 logical offset overflows");
    int64_t delta = index * stride;
    if (elementOffset > std::numeric_limits<int64_t>::max() - delta)
      return invalid("reference i1 logical offset overflows");
    elementOffset += delta;
  }
  if (buffer.base < 0 || buffer.viewOffset < 0 ||
      buffer.base > std::numeric_limits<int64_t>::max() - buffer.viewOffset)
    return invalid("reference i1 buffer address overflows");
  int64_t byteBase = buffer.base + buffer.viewOffset;
  if (byteBase > std::numeric_limits<int64_t>::max() / 8 ||
      byteBase * 8 > std::numeric_limits<int64_t>::max() - elementOffset)
    return invalid("reference i1 bit address overflows");
  int64_t bitOffset = byteBase * 8 + elementOffset;
  if (bitOffset < 0 ||
      static_cast<uint64_t>(bitOffset / 8) >= buffer.storage->bytes.size())
    return invalid("reference i1 access exceeds storage");
  return bitOffset;
}

llvm::Expected<bool> readI1(const BufferView &buffer,
                            llvm::ArrayRef<int64_t> indices) {
  auto bitOffset = getAbsoluteBitOffset(buffer, indices);
  if (!bitOffset)
    return bitOffset.takeError();
  uint8_t byte = buffer.storage->bytes[*bitOffset / 8];
  return ((byte >> (*bitOffset % 8)) & 1u) != 0;
}

llvm::Error writeI1(const BufferView &buffer, llvm::ArrayRef<int64_t> indices,
                    bool value) {
  auto bitOffset = getAbsoluteBitOffset(buffer, indices);
  if (!bitOffset)
    return bitOffset.takeError();
  uint8_t &byte = buffer.storage->bytes[*bitOffset / 8];
  uint8_t mask = static_cast<uint8_t>(1u << (*bitOffset % 8));
  byte = value ? static_cast<uint8_t>(byte | mask)
               : static_cast<uint8_t>(byte & ~mask);
  return llvm::Error::success();
}

} // namespace wafer::compiler::reference_detail
