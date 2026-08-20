//===- TargetScalarConversion.cpp - Deterministic scalar conversion -----===//

#include "Wafer/Target/Numeric/TargetScalarConversion.h"

#include "Wafer/Target/Numeric/TargetFloatArithmetic.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

namespace wafer {
namespace {

using namespace target_numeric_detail;

llvm::Error conversionError(TargetScalarConversionErrorCode code,
                            const llvm::Twine &detail) {
  return llvm::make_error<TargetScalarConversionError>(code, detail.str());
}

bool isIntegerCategory(LogicalFormatCategory category) {
  return category == LogicalFormatCategory::SignedInteger ||
         category == LogicalFormatCategory::UnsignedInteger;
}

std::optional<llvm::APFloat::roundingMode>
getAPFloatRoundingMode(TargetRoundingMode mode) {
  switch (mode) {
  case TargetRoundingMode::NearestEven:
    return llvm::APFloat::rmNearestTiesToEven;
  case TargetRoundingMode::TowardZero:
    return llvm::APFloat::rmTowardZero;
  case TargetRoundingMode::TowardPositive:
    return llvm::APFloat::rmTowardPositive;
  case TargetRoundingMode::TowardNegative:
    return llvm::APFloat::rmTowardNegative;
  case TargetRoundingMode::Stochastic:
    return std::nullopt;
  }
  return std::nullopt;
}

TargetScalarConversionFlags flagsFromStatus(llvm::APFloat::opStatus status) {
  const unsigned bits = static_cast<unsigned>(status);
  return {
      (bits & static_cast<unsigned>(llvm::APFloat::opInvalidOp)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opDivByZero)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opOverflow)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opUnderflow)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opInexact)) != 0,
  };
}

uint64_t largestFiniteBits(const LogicalFormatDescriptor &descriptor) {
  const uint8_t fractionBits = descriptor.precisionBits - 1;
  const uint8_t semanticShift =
      descriptor.storageBits - descriptor.semanticBits;
  const uint64_t fraction = ((UINT64_C(1) << fractionBits) - UINT64_C(1))
                            << semanticShift;
  const uint64_t exponent =
      ((UINT64_C(1) << descriptor.exponentBits) - UINT64_C(2))
      << (fractionBits + semanticShift);
  return exponent | fraction;
}

bool directedOverflowReturnsFinite(TargetRoundingMode mode, bool negative) {
  switch (mode) {
  case TargetRoundingMode::NearestEven:
    return false;
  case TargetRoundingMode::TowardZero:
    return true;
  case TargetRoundingMode::TowardPositive:
    return negative;
  case TargetRoundingMode::TowardNegative:
    return !negative;
  case TargetRoundingMode::Stochastic:
    return false;
  }
  return false;
}

bool integerMagnitudeExceedsMaxFinite(
    const llvm::APInt &integer, bool isSigned,
    const LogicalFormatDescriptor &destinationDescriptor) {
  llvm::APInt magnitude = integer;
  if (isSigned && integer.isNegative())
    magnitude = -integer;

  const unsigned maximumExponent =
      (1U << (destinationDescriptor.exponentBits - 1)) - 1U;
  if (maximumExponent >= magnitude.getBitWidth())
    return false;
  const unsigned fractionBits = destinationDescriptor.precisionBits - 1;
  if (maximumExponent < fractionBits ||
      destinationDescriptor.precisionBits > magnitude.getBitWidth())
    llvm_unreachable("unsupported floating range for exact integer compare");
  llvm::APInt maximum = llvm::APInt::getLowBitsSet(
      magnitude.getBitWidth(), destinationDescriptor.precisionBits);
  maximum <<= maximumExponent - fractionBits;
  return magnitude.ugt(maximum);
}

bool floatingMagnitudeExceedsMaxFinite(
    const llvm::APFloat &source,
    const LogicalFormatDescriptor &destinationDescriptor) {
  if (!source.isFinite())
    return false;

  llvm::APFloat sourceWide = source;
  sourceWide.clearSign();
  bool sourceLosesInfo = false;
  llvm::APFloat::opStatus sourceStatus =
      sourceWide.convert(llvm::APFloat::IEEEquad(),
                         llvm::APFloat::rmNearestTiesToEven, &sourceLosesInfo);

  llvm::APFloat maximum = decodeFloat(
      {destinationDescriptor.format, largestFiniteBits(destinationDescriptor)});
  bool maximumLosesInfo = false;
  llvm::APFloat::opStatus maximumStatus =
      maximum.convert(llvm::APFloat::IEEEquad(),
                      llvm::APFloat::rmNearestTiesToEven, &maximumLosesInfo);
  if (sourceStatus != llvm::APFloat::opOK || sourceLosesInfo ||
      maximumStatus != llvm::APFloat::opOK || maximumLosesInfo)
    llvm_unreachable("IEEEquad failed to exactly widen a supported format");
  return sourceWide.compare(maximum) == llvm::APFloat::cmpGreaterThan;
}

void completeDirectedOverflowFlags(TargetScalarConversionFlags &flags,
                                   TargetRoundingMode mode, bool negative,
                                   bool magnitudeExceedsMaxFinite) {
  if (magnitudeExceedsMaxFinite &&
      directedOverflowReturnsFinite(mode, negative)) {
    flags.overflow = true;
    flags.inexact = true;
  }
}

} // namespace

llvm::StringRef
stringifyTargetScalarConversionErrorCode(TargetScalarConversionErrorCode code) {
  switch (code) {
  case TargetScalarConversionErrorCode::UnsupportedOperation:
    return "unsupported-operation";
  case TargetScalarConversionErrorCode::InvalidSourceEncoding:
    return "invalid-source-encoding";
  case TargetScalarConversionErrorCode::SourceFormatMismatch:
    return "source-format-mismatch";
  case TargetScalarConversionErrorCode::UnsupportedEndpointKinds:
    return "unsupported-endpoint-kinds";
  case TargetScalarConversionErrorCode::FloatToIntegerNonFinite:
    return "float-to-integer-non-finite";
  case TargetScalarConversionErrorCode::FloatToIntegerOutOfRange:
    return "float-to-integer-out-of-range";
  case TargetScalarConversionErrorCode::UnexpectedAPFloatStatus:
    return "unexpected-apfloat-status";
  case TargetScalarConversionErrorCode::InvalidResultEncoding:
    return "invalid-result-encoding";
  }
  llvm_unreachable("unknown target scalar conversion error code");
}

char TargetScalarConversionError::ID;

void TargetScalarConversionError::log(llvm::raw_ostream &stream) const {
  stream << "target scalar conversion "
         << stringifyTargetScalarConversionErrorCode(code) << ": " << detail;
}

std::error_code TargetScalarConversionError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<TargetScalarConversionResult>
convertTargetScalar(TargetConvertOperation operation,
                    std::optional<TargetConvertParameter> parameter,
                    RawLogicalValue source) {
  const TargetConvertRoute *routeRecord =
      findTargetConvertRoute(operation.getOpcode());
  if (!routeRecord)
    return conversionError(
        TargetScalarConversionErrorCode::UnsupportedOperation,
        "target convert operation lost its validated route");
  const TargetConvertRoute &route = *routeRecord;
  TargetRoundingMode targetRounding = TargetRoundingMode::NearestEven;
  if (route.parameterKind == TargetConvertParameterKind::RoundingMode) {
    if (!parameter || !parameter->getRoundingMode())
      return conversionError(
          TargetScalarConversionErrorCode::UnsupportedOperation,
          "rounding conversion has no typed rounding mode");
    targetRounding = *parameter->getRoundingMode();
  } else if (route.parameterKind != TargetConvertParameterKind::None ||
             parameter) {
    return conversionError(
        TargetScalarConversionErrorCode::UnsupportedOperation,
        "parameterless scalar conversion has inconsistent parameters");
  }

  llvm::Expected<RawLogicalValue> canonicalSource = makeRawLogicalValue(
      source.format, source.bits, NonCanonicalEncodingPolicy::Reject);
  if (!canonicalSource)
    return conversionError(
        TargetScalarConversionErrorCode::InvalidSourceEncoding,
        llvm::toString(canonicalSource.takeError()));
  if (canonicalSource->format != route.source)
    return conversionError(
        TargetScalarConversionErrorCode::SourceFormatMismatch,
        llvm::Twine("expected ") + stringifyLogicalFormat(route.source) +
            " source for route '" + route.canonicalSpelling + "', got " +
            stringifyLogicalFormat(canonicalSource->format));

  const LogicalFormatDescriptor *sourceDescriptor =
      findLogicalFormatDescriptor(route.source);
  const LogicalFormatDescriptor *destinationDescriptor =
      findLogicalFormatDescriptor(route.destination);
  if (!sourceDescriptor || !destinationDescriptor)
    return conversionError(
        TargetScalarConversionErrorCode::UnsupportedEndpointKinds,
        "the target convert route has an unknown endpoint format");

  std::optional<llvm::APFloat::roundingMode> roundingMode =
      getAPFloatRoundingMode(targetRounding);
  if (!roundingMode)
    return conversionError(
        TargetScalarConversionErrorCode::UnsupportedOperation,
        "scalar conversion accepts only deterministic rounding modes");

  auto finish = [&](uint64_t bits, TargetScalarConversionFlags flags)
      -> llvm::Expected<TargetScalarConversionResult> {
    llvm::Expected<RawLogicalValue> result = makeRawLogicalValue(
        route.destination, bits, NonCanonicalEncodingPolicy::Reject);
    if (!result)
      return conversionError(
          TargetScalarConversionErrorCode::InvalidResultEncoding,
          llvm::toString(result.takeError()));
    return TargetScalarConversionResult{*result, flags};
  };

  const bool sourceIsFloat =
      sourceDescriptor->category == LogicalFormatCategory::BinaryFloatingPoint;
  const bool destinationIsFloat = destinationDescriptor->category ==
                                  LogicalFormatCategory::BinaryFloatingPoint;
  const bool sourceIsInteger = isIntegerCategory(sourceDescriptor->category);
  const bool destinationIsInteger =
      isIntegerCategory(destinationDescriptor->category);

  if (sourceIsInteger && destinationIsFloat) {
    llvm::APInt integer(sourceDescriptor->storageBits, canonicalSource->bits);
    const bool sourceIsSigned =
        sourceDescriptor->category == LogicalFormatCategory::SignedInteger;
    const bool sourceIsNegative = sourceIsSigned && integer.isNegative();
    const bool exceedsMaximum = integerMagnitudeExceedsMaxFinite(
        integer, sourceIsSigned, *destinationDescriptor);
    llvm::APFloat result = llvm::APFloat::getZero(
        *getFloatSemantics(destinationDescriptor->format));
    llvm::APFloat::opStatus status =
        result.convertFromAPInt(integer, sourceIsSigned, *roundingMode);
    TargetScalarConversionFlags flags = flagsFromStatus(status);
    completeDirectedOverflowFlags(flags, targetRounding, sourceIsNegative,
                                  exceedsMaximum);
    if (flags.invalid || flags.divByZero)
      return conversionError(
          TargetScalarConversionErrorCode::UnexpectedAPFloatStatus,
          "integer-to-floating conversion raised invalid or divide-by-zero");
    std::optional<uint64_t> bits =
        encodeFloat(result, destinationDescriptor->format);
    if (!bits)
      return conversionError(
          TargetScalarConversionErrorCode::InvalidResultEncoding,
          "APFloat produced an unexpected destination width");
    return finish(*bits, flags);
  }

  if (sourceIsFloat && destinationIsFloat) {
    llvm::Expected<LogicalValueClassification> classification =
        classifyRawLogicalValue(*canonicalSource,
                                NonCanonicalEncodingPolicy::Reject);
    if (!classification)
      return conversionError(
          TargetScalarConversionErrorCode::InvalidSourceEncoding,
          llvm::toString(classification.takeError()));
    if (isNaNClass(classification->valueClass)) {
      TargetScalarConversionFlags flags;
      flags.invalid =
          classification->valueClass == LogicalValueClass::SignalingNaN;
      return finish(canonicalPositiveQuietNaNBits(*destinationDescriptor),
                    flags);
    }

    llvm::APFloat sourceFloating = decodeFloat(*canonicalSource);
    llvm::APFloat result = sourceFloating;
    bool ignoredLosesInfo = false;
    llvm::APFloat::opStatus status =
        result.convert(*getFloatSemantics(destinationDescriptor->format),
                       *roundingMode, &ignoredLosesInfo);
    TargetScalarConversionFlags flags = flagsFromStatus(status);
    flags.underflow =
        flags.inexact && isTinyAfterUnboundedPrecisionRounding(
                             *canonicalSource, *sourceDescriptor,
                             *destinationDescriptor, targetRounding);
    completeDirectedOverflowFlags(flags, targetRounding,
                                  classification->negative,
                                  floatingMagnitudeExceedsMaxFinite(
                                      sourceFloating, *destinationDescriptor));
    if (flags.invalid || flags.divByZero)
      return conversionError(
          TargetScalarConversionErrorCode::UnexpectedAPFloatStatus,
          "finite/infinite floating conversion raised invalid or "
          "divide-by-zero");
    std::optional<uint64_t> bits =
        encodeFloat(result, destinationDescriptor->format);
    if (!bits)
      return conversionError(
          TargetScalarConversionErrorCode::InvalidResultEncoding,
          "APFloat produced an unexpected destination width");
    return finish(*bits, flags);
  }

  if (sourceIsFloat && destinationIsInteger) {
    llvm::Expected<LogicalValueClassification> classification =
        classifyRawLogicalValue(*canonicalSource,
                                NonCanonicalEncodingPolicy::Reject);
    if (!classification)
      return conversionError(
          TargetScalarConversionErrorCode::InvalidSourceEncoding,
          llvm::toString(classification.takeError()));
    if (classification->valueClass == LogicalValueClass::Infinity ||
        isNaNClass(classification->valueClass))
      return conversionError(
          TargetScalarConversionErrorCode::FloatToIntegerNonFinite,
          "NaN and infinity do not produce a floating-to-integer result");

    llvm::APFloat floating = decodeFloat(*canonicalSource);
    llvm::APSInt result(destinationDescriptor->storageBits,
                        destinationDescriptor->category ==
                            LogicalFormatCategory::UnsignedInteger);
    bool ignoredIsExact = false;
    llvm::APFloat::opStatus status =
        floating.convertToInteger(result, *roundingMode, &ignoredIsExact);
    TargetScalarConversionFlags flags = flagsFromStatus(status);
    if (flags.invalid)
      return conversionError(
          TargetScalarConversionErrorCode::FloatToIntegerOutOfRange,
          "finite input is outside the destination integer range after "
          "rounding");
    if (flags.divByZero || flags.overflow || flags.underflow)
      return conversionError(
          TargetScalarConversionErrorCode::UnexpectedAPFloatStatus,
          "floating-to-integer conversion raised a non-integer status");
    return finish(result.getZExtValue(), flags);
  }

  return conversionError(
      TargetScalarConversionErrorCode::UnsupportedEndpointKinds,
      "the target conversion is not integer-to-floating, "
      "floating-to-floating, or floating-to-integer");
}

} // namespace wafer
