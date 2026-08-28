//===- TargetFloatArithmetic.cpp - Shared target float primitives -------===//

#include "Wafer/Target/PhysicalTensor/TargetFloatArithmetic.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

namespace wafer::target_numeric_detail {

const llvm::fltSemantics *getFloatSemantics(LogicalFormat format) {
  switch (format) {
  case LogicalFormat::F16:
    return &llvm::APFloat::IEEEhalf();
  case LogicalFormat::BF16:
    return &llvm::APFloat::BFloat();
  case LogicalFormat::F32:
    return &llvm::APFloat::IEEEsingle();
  case LogicalFormat::TF32:
    return &llvm::APFloat::FloatTF32();
  case LogicalFormat::I8:
  case LogicalFormat::I16:
  case LogicalFormat::I32:
  case LogicalFormat::Bool:
  case LogicalFormat::U8:
  case LogicalFormat::U16:
  case LogicalFormat::U32:
  case LogicalFormat::I64:
  case LogicalFormat::U64:
    return nullptr;
  }
  return nullptr;
}

llvm::APFloat decodeFloat(RawLogicalValue value) {
  const LogicalFormatDescriptor &descriptor =
      *findLogicalFormatDescriptor(value.format);
  const llvm::fltSemantics &semantics = *getFloatSemantics(value.format);
  if (value.format == LogicalFormat::TF32) {
    // LLVM FloatTF32 uses the compact 19-bit sign/exponent/fraction encoding;
    // RawLogicalValue keeps those bits in storage [31:13].
    return llvm::APFloat(semantics,
                         llvm::APInt(/*numBits=*/19, value.bits >> 13));
  }
  return llvm::APFloat(semantics,
                       llvm::APInt(descriptor.storageBits, value.bits));
}

std::optional<uint64_t> encodeFloat(const llvm::APFloat &value,
                                    LogicalFormat format) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(format);
  if (!descriptor)
    return std::nullopt;
  llvm::APInt bits = value.bitcastToAPInt();
  if (format == LogicalFormat::TF32) {
    if (bits.getBitWidth() != 19)
      return std::nullopt;
    return bits.getZExtValue() << 13;
  }
  if (bits.getBitWidth() != descriptor->storageBits)
    return std::nullopt;
  return bits.getZExtValue();
}

uint64_t
canonicalPositiveQuietNaNBits(const LogicalFormatDescriptor &descriptor) {
  const uint8_t fractionBits = descriptor.precisionBits - 1;
  const uint8_t semanticShift =
      descriptor.storageBits - descriptor.semanticBits;
  const uint64_t exponentMask =
      (UINT64_C(1) << descriptor.exponentBits) - UINT64_C(1);
  const uint8_t exponentShift = fractionBits + semanticShift;
  const uint64_t quietBit = UINT64_C(1) << (semanticShift + fractionBits - 1);
  return (exponentMask << exponentShift) | quietBit;
}

bool isNaNClass(LogicalValueClass valueClass) {
  return valueClass == LogicalValueClass::QuietNaN ||
         valueClass == LogicalValueClass::SignalingNaN;
}

bool isTinyAfterUnboundedPrecisionRounding(
    RawLogicalValue source, const LogicalFormatDescriptor &sourceDescriptor,
    const LogicalFormatDescriptor &destinationDescriptor,
    TargetRoundingMode mode) {
  const unsigned sourceFractionBits = sourceDescriptor.precisionBits - 1;
  const unsigned sourceSemanticShift =
      sourceDescriptor.storageBits - sourceDescriptor.semanticBits;
  const uint64_t sourceSemanticBits = source.bits >> sourceSemanticShift;
  const uint64_t sourceFractionMask =
      (UINT64_C(1) << sourceFractionBits) - UINT64_C(1);
  const uint64_t sourceFraction = sourceSemanticBits & sourceFractionMask;
  const uint64_t sourceExponentMask =
      (UINT64_C(1) << sourceDescriptor.exponentBits) - UINT64_C(1);
  const uint64_t sourceExponentField =
      (sourceSemanticBits >> sourceFractionBits) & sourceExponentMask;
  const bool negative =
      (sourceSemanticBits >> (sourceDescriptor.semanticBits - 1)) != 0;

  uint64_t significand = sourceFraction;
  const int sourceBias = (1 << (sourceDescriptor.exponentBits - 1)) - 1;
  int exponent = 1 - sourceBias - static_cast<int>(sourceFractionBits);
  if (sourceExponentField != 0) {
    significand |= UINT64_C(1) << sourceFractionBits;
    exponent = static_cast<int>(sourceExponentField) - sourceBias -
               static_cast<int>(sourceFractionBits);
  }
  if (significand == 0)
    return false;

  const unsigned destinationPrecision = destinationDescriptor.precisionBits;
  const unsigned significantBits = llvm::Log2_64(significand) + 1;
  const int shift = static_cast<int>(significantBits) -
                    static_cast<int>(destinationPrecision);
  uint64_t roundedSignificand = significand;
  int roundedExponent = exponent;
  if (shift > 0) {
    roundedSignificand >>= shift;
    roundedExponent += shift;
    const uint64_t lostMask = (UINT64_C(1) << shift) - UINT64_C(1);
    const uint64_t lost = significand & lostMask;
    bool increment = false;
    switch (mode) {
    case TargetRoundingMode::NearestEven: {
      const uint64_t halfway = UINT64_C(1) << (shift - 1);
      increment =
          lost > halfway || (lost == halfway && (roundedSignificand & 1));
      break;
    }
    case TargetRoundingMode::TowardZero:
      break;
    case TargetRoundingMode::TowardPositive:
      increment = !negative && lost != 0;
      break;
    case TargetRoundingMode::TowardNegative:
      increment = negative && lost != 0;
      break;
    case TargetRoundingMode::Stochastic:
      llvm_unreachable("stochastic rounding reached deterministic path");
    }
    if (increment) {
      ++roundedSignificand;
      if (llvm::Log2_64(roundedSignificand) >= destinationPrecision) {
        roundedSignificand >>= 1;
        ++roundedExponent;
      }
    }
  } else if (shift < 0) {
    roundedSignificand <<= -shift;
    roundedExponent += shift;
  }

  const int roundedTopExponent =
      roundedExponent + static_cast<int>(llvm::Log2_64(roundedSignificand));
  const int destinationBias =
      (1 << (destinationDescriptor.exponentBits - 1)) - 1;
  const int destinationMinimumNormalExponent = 1 - destinationBias;
  return roundedTopExponent < destinationMinimumNormalExponent;
}

} // namespace wafer::target_numeric_detail
