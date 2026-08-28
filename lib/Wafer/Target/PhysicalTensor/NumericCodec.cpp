//===- NumericCodec.cpp - Target-independent logical scalar codec --------===//

#include "Wafer/Target/PhysicalTensor/NumericCodec.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/MathExtras.h"

#include <cstddef>
#include <limits>

namespace wafer {
namespace {

llvm::Error invalidEncoding(LogicalFormat format, uint64_t bits,
                            llvm::StringRef reason) {
  return llvm::createStringError(
      llvm::errc::invalid_argument, "invalid %s raw encoding 0x%llx: %s",
      stringifyLogicalFormat(format).str().c_str(),
      static_cast<unsigned long long>(bits), reason.str().c_str());
}

llvm::Expected<uint64_t> checkedEndBit(uint64_t bitOffset,
                                       uint64_t scalarBits) {
  if (bitOffset > std::numeric_limits<uint64_t>::max() - scalarBits)
    return llvm::createStringError(llvm::errc::result_out_of_range,
                                   "logical scalar bit range overflows");
  return bitOffset + scalarBits;
}

llvm::Error checkStorageRange(uint64_t endBit, size_t storageBytes) {
  if (storageBytes > std::numeric_limits<uint64_t>::max() / 8)
    return llvm::Error::success();
  if (endBit > static_cast<uint64_t>(storageBytes) * 8)
    return llvm::createStringError(
        llvm::errc::result_out_of_range,
        "logical scalar bit range [0, %llu) exceeds %zu-byte storage",
        static_cast<unsigned long long>(endBit), storageBytes);
  return llvm::Error::success();
}

} // namespace

llvm::Expected<RawLogicalValue>
makeRawLogicalValue(LogicalFormat format, uint64_t bits,
                    NonCanonicalEncodingPolicy policy) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(format);
  if (!descriptor)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown logical format value");

  const uint64_t storageMask =
      descriptor->storageBits == 64
          ? std::numeric_limits<uint64_t>::max()
          : llvm::maskTrailingOnes<uint64_t>(descriptor->storageBits);
  if ((bits & ~storageMask) != 0)
    return invalidEncoding(format, bits, "bits exceed the storage width");

  const uint64_t nonCanonicalBits = bits & ~descriptor->canonicalMask;
  if (nonCanonicalBits != 0) {
    if (policy == NonCanonicalEncodingPolicy::Reject)
      return invalidEncoding(format, bits,
                             "noncanonical semantic padding bits are nonzero");
    bits &= descriptor->canonicalMask;
  }
  return RawLogicalValue{format, bits};
}

llvm::Expected<RawLogicalValue>
readRawLogicalValue(LogicalFormat format, llvm::ArrayRef<uint8_t> storage,
                    uint64_t bitOffset, LogicalScalarCodecPolicy policy) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(format);
  if (!descriptor)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown logical format value");

  llvm::Expected<uint64_t> endBit =
      checkedEndBit(bitOffset, descriptor->storageBits);
  if (!endBit)
    return endBit.takeError();
  if (llvm::Error error = checkStorageRange(*endBit, storage.size()))
    return std::move(error);

  if (descriptor->bitpacked) {
    const uint8_t byte = storage[bitOffset / 8];
    const uint64_t position =
        policy.bitOrder == LogicalBitOrder::LeastSignificantBitFirstWithinByte
            ? bitOffset % 8
            : 7 - (bitOffset % 8);
    const uint64_t bits = (byte >> position) & UINT8_C(1);
    return makeRawLogicalValue(format, bits, policy.nonCanonicalEncoding);
  }
  if ((bitOffset % 8) != 0)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "%s scalar requires a byte-aligned bit offset",
        descriptor->canonicalSpelling.str().c_str());

  const size_t byteOffset = static_cast<size_t>(bitOffset / 8);
  const size_t byteCount = descriptor->storageBits / 8;
  uint64_t bits = 0;
  for (size_t index = 0; index < byteCount; ++index) {
    const size_t significance =
        policy.byteOrder == LogicalByteOrder::LittleEndian
            ? index
            : byteCount - index - 1;
    bits |= static_cast<uint64_t>(storage[byteOffset + index])
            << (8 * significance);
  }
  return makeRawLogicalValue(format, bits, policy.nonCanonicalEncoding);
}

llvm::Error writeRawLogicalValue(RawLogicalValue value,
                                 llvm::MutableArrayRef<uint8_t> storage,
                                 uint64_t bitOffset,
                                 LogicalScalarCodecPolicy policy) {
  llvm::Expected<RawLogicalValue> canonical = makeRawLogicalValue(
      value.format, value.bits, policy.nonCanonicalEncoding);
  if (!canonical)
    return canonical.takeError();
  const LogicalFormatDescriptor &descriptor =
      *findLogicalFormatDescriptor(canonical->format);

  llvm::Expected<uint64_t> endBit =
      checkedEndBit(bitOffset, descriptor.storageBits);
  if (!endBit)
    return endBit.takeError();
  if (llvm::Error error = checkStorageRange(*endBit, storage.size()))
    return error;

  if (descriptor.bitpacked) {
    const uint64_t position =
        policy.bitOrder == LogicalBitOrder::LeastSignificantBitFirstWithinByte
            ? bitOffset % 8
            : 7 - (bitOffset % 8);
    const uint8_t mask = static_cast<uint8_t>(UINT8_C(1) << position);
    uint8_t &byte = storage[bitOffset / 8];
    byte = canonical->bits == 0 ? static_cast<uint8_t>(byte & ~mask)
                                : static_cast<uint8_t>(byte | mask);
    return llvm::Error::success();
  }
  if ((bitOffset % 8) != 0)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "%s scalar requires a byte-aligned bit offset",
        descriptor.canonicalSpelling.str().c_str());

  const size_t byteOffset = static_cast<size_t>(bitOffset / 8);
  const size_t byteCount = descriptor.storageBits / 8;
  for (size_t index = 0; index < byteCount; ++index) {
    const size_t significance =
        policy.byteOrder == LogicalByteOrder::LittleEndian
            ? index
            : byteCount - index - 1;
    storage[byteOffset + index] =
        static_cast<uint8_t>(canonical->bits >> (8 * significance));
  }
  return llvm::Error::success();
}

llvm::Expected<LogicalValueClassification>
classifyRawLogicalValue(RawLogicalValue value,
                        NonCanonicalEncodingPolicy policy) {
  llvm::Expected<RawLogicalValue> canonical =
      makeRawLogicalValue(value.format, value.bits, policy);
  if (!canonical)
    return canonical.takeError();
  const LogicalFormatDescriptor &descriptor =
      *findLogicalFormatDescriptor(canonical->format);

  switch (descriptor.category) {
  case LogicalFormatCategory::SignedInteger:
    return LogicalValueClassification{
        LogicalValueClass::SignedInteger,
        ((canonical->bits >> (descriptor.storageBits - 1)) & 1) != 0};
  case LogicalFormatCategory::UnsignedInteger:
    return LogicalValueClassification{LogicalValueClass::UnsignedInteger,
                                      false};
  case LogicalFormatCategory::Boolean:
    return LogicalValueClassification{LogicalValueClass::Boolean, false};
  case LogicalFormatCategory::BinaryFloatingPoint:
    break;
  }

  const uint8_t fractionBits = descriptor.precisionBits - 1;
  const uint8_t semanticShift =
      descriptor.storageBits - descriptor.semanticBits;
  const uint64_t fractionMask = llvm::maskTrailingOnes<uint64_t>(fractionBits)
                                << semanticShift;
  const uint8_t exponentShift = fractionBits + semanticShift;
  const uint64_t exponentValue =
      (canonical->bits >> exponentShift) &
      llvm::maskTrailingOnes<uint64_t>(descriptor.exponentBits);
  const uint64_t maxExponent =
      llvm::maskTrailingOnes<uint64_t>(descriptor.exponentBits);
  const uint64_t fraction = canonical->bits & fractionMask;
  const bool negative =
      ((canonical->bits >> (descriptor.storageBits - 1)) & 1) != 0;

  if (exponentValue == 0)
    return LogicalValueClassification{
        fraction == 0 ? LogicalValueClass::Zero : LogicalValueClass::Subnormal,
        negative};
  if (exponentValue != maxExponent)
    return LogicalValueClassification{LogicalValueClass::Normal, negative};
  if (fraction == 0)
    return LogicalValueClassification{LogicalValueClass::Infinity, negative};

  const uint64_t quietBit = UINT64_C(1) << (semanticShift + fractionBits - 1);
  return LogicalValueClassification{(fraction & quietBit) != 0
                                        ? LogicalValueClass::QuietNaN
                                        : LogicalValueClass::SignalingNaN,
                                    negative};
}

} // namespace wafer
