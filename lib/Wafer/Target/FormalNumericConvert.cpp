//===- FormalNumericConvert.cpp - Formal scalar conversion ----------===//

#include "FormalNumericInternal.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer {

using namespace formal_detail;

namespace {

bool isIntegerCategory(LogicalFormatCategory category) {
  return category == LogicalFormatCategory::SignedInteger ||
         category == LogicalFormatCategory::UnsignedInteger;
}

std::optional<llvm::APFloat::roundingMode>
getAPFloatRoundingMode(NumericRoundingMode mode) {
  switch (mode) {
  case NumericRoundingMode::NearestEven:
    return llvm::APFloat::rmNearestTiesToEven;
  case NumericRoundingMode::TowardZero:
    return llvm::APFloat::rmTowardZero;
  case NumericRoundingMode::TowardPositive:
    return llvm::APFloat::rmTowardPositive;
  case NumericRoundingMode::TowardNegative:
    return llvm::APFloat::rmTowardNegative;
  case NumericRoundingMode::Stochastic:
    return std::nullopt;
  }
  return std::nullopt;
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

bool directedOverflowReturnsFinite(NumericRoundingMode mode, bool negative) {
  switch (mode) {
  case NumericRoundingMode::NearestEven:
    return false;
  case NumericRoundingMode::TowardZero:
    return true;
  case NumericRoundingMode::TowardPositive:
    return negative;
  case NumericRoundingMode::TowardNegative:
    return !negative;
  case NumericRoundingMode::Stochastic:
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

void completeDirectedOverflowFlags(FormalNumericExceptionFlags &flags,
                                   NumericRoundingMode mode, bool negative,
                                   bool magnitudeExceedsMaxFinite) {
  if (magnitudeExceedsMaxFinite &&
      directedOverflowReturnsFinite(mode, negative)) {
    // LLVM APFloat's directed max-finite path returns opInexact without
    // opOverflow. IEEE overflow still occurred; supplement from an exact
    // source-vs-destination range proof rather than from the result bit
    // pattern.
    flags.overflow = true;
    flags.inexact = true;
  }
}

} // namespace

llvm::Expected<FormalNumericResult>
evaluateFormalConvert(const ResolvedNumericCommand &command,
                      RawLogicalValue source) {
  if (command.getFamily() != NumericCommandFamily::CTConvert ||
      !command.isSupported() || !command.getSemantics() ||
      command.getFormalKernelKind() != FormalKernelKind::Convert ||
      command.getComparatorKind() != NumericComparatorKind::RawExact ||
      command.getFormalBackendKind() !=
          FormalNumericBackendKind::LLVMAPFloatAPInt)
    return formalError(
        FormalNumericErrorCode::UnsupportedResolvedCommand,
        "the resolved command has no complete formal convert identity");

  const NumericCommandKey &key = command.getCommandKey();
  const NumericCTConvertCommand *convert = key.getCTConvert();
  if (!convert)
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "the resolved command is not a CT convert command");
  const NumericSemanticsProfile &semantics = *command.getSemantics();
  const TargetConvertRoute *routeRecord =
      findTargetConvertRoute(convert->opcode);
  if (!routeRecord || convert->source.getFormat() != routeRecord->source ||
      convert->destination.getFormat() != routeRecord->destination)
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "the resolved convert key lost its validated route");
  const TargetConvertRoute &route = *routeRecord;
  const NumericCTConvertSemanticsIdentity &routePolicy =
      *semantics.getCTConvertIdentity();
  if (semantics.getModelProfile() != command.getPattern().getModelProfile() ||
      routePolicy.getFamily() != key.getFamily() ||
      routePolicy.getCTConvertOpcode() != convert->opcode)
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "resolved semantics and exact key disagree");
  if (route.parameterKind == TargetConvertParameterKind::RoundingMode) {
    if (!convert->parameter ||
        convert->parameter->getRoundingMode() != semantics.getRoundingMode())
      return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                         "resolved rounding policy disagrees with exact key");
  } else if (route.parameterKind != TargetConvertParameterKind::None ||
             convert->parameter ||
             semantics.getRoundingMode() != NumericRoundingMode::NearestEven) {
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "resolved parameterless policy is inconsistent");
  }

  llvm::Expected<RawLogicalValue> canonicalSource = makeRawLogicalValue(
      source.format, source.bits, NonCanonicalEncodingPolicy::Reject);
  if (!canonicalSource)
    return formalError(FormalNumericErrorCode::InvalidSourceEncoding,
                       llvm::toString(canonicalSource.takeError()));
  if (canonicalSource->format != route.source)
    return formalError(
        FormalNumericErrorCode::SourceFormatMismatch,
        llvm::Twine("expected ") + stringifyLogicalFormat(route.source) +
            " source for route '" + route.canonicalSpelling + "', got " +
            stringifyLogicalFormat(canonicalSource->format));

  const LogicalFormatDescriptor *sourceDescriptor =
      findLogicalFormatDescriptor(route.source);
  const LogicalFormatDescriptor *destinationDescriptor =
      findLogicalFormatDescriptor(route.destination);
  if (!sourceDescriptor || !destinationDescriptor)
    return formalError(FormalNumericErrorCode::UnsupportedEndpointKinds,
                       "the CT convert route has an unknown endpoint format");

  std::optional<llvm::APFloat::roundingMode> roundingMode =
      getAPFloatRoundingMode(semantics.getRoundingMode());
  if (!roundingMode)
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "the formal executor accepts only the four "
                       "deterministic rounding modes");

  auto finish = [&](uint64_t bits, FormalNumericExceptionFlags flags)
      -> llvm::Expected<FormalNumericResult> {
    llvm::Expected<RawLogicalValue> result = makeRawLogicalValue(
        route.destination, bits, NonCanonicalEncodingPolicy::Reject);
    if (!result)
      return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                         llvm::toString(result.takeError()));
    return FormalNumericResult{*result, flags};
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
    FormalNumericExceptionFlags flags = flagsFromStatus(status);
    completeDirectedOverflowFlags(flags, semantics.getRoundingMode(),
                                  sourceIsNegative, exceedsMaximum);
    if (flags.invalid || flags.divByZero) {
      return formalError(
          FormalNumericErrorCode::UnexpectedAPFloatStatus,
          "integer-to-floating conversion raised invalid or divide-by-zero");
    }
    std::optional<uint64_t> bits =
        encodeFloat(result, destinationDescriptor->format);
    if (!bits) {
      return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                         "APFloat produced an unexpected destination width");
    }
    return finish(*bits, flags);
  }

  if (sourceIsFloat && destinationIsFloat) {
    llvm::Expected<LogicalValueClassification> classification =
        classifyRawLogicalValue(*canonicalSource,
                                NonCanonicalEncodingPolicy::Reject);
    if (!classification)
      return formalError(FormalNumericErrorCode::InvalidSourceEncoding,
                         llvm::toString(classification.takeError()));

    if (isNaNClass(classification->valueClass)) {
      FormalNumericExceptionFlags flags;
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
    // The profile maps APFloat's explicit opStatus; losesInfo is not a second
    // model flag channel.
    FormalNumericExceptionFlags flags = flagsFromStatus(status);
    flags.underflow = flags.inexact &&
                      isTinyAfterUnboundedPrecisionRounding(
                          *canonicalSource, *sourceDescriptor,
                          *destinationDescriptor, semantics.getRoundingMode());
    completeDirectedOverflowFlags(flags, semantics.getRoundingMode(),
                                  classification->negative,
                                  floatingMagnitudeExceedsMaxFinite(
                                      sourceFloating, *destinationDescriptor));
    if (flags.invalid || flags.divByZero) {
      return formalError(
          FormalNumericErrorCode::UnexpectedAPFloatStatus,
          "finite/infinite floating conversion raised invalid or "
          "divide-by-zero");
    }
    std::optional<uint64_t> bits =
        encodeFloat(result, destinationDescriptor->format);
    if (!bits) {
      return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                         "APFloat produced an unexpected destination width");
    }
    return finish(*bits, flags);
  }

  if (sourceIsFloat && destinationIsInteger) {
    llvm::Expected<LogicalValueClassification> classification =
        classifyRawLogicalValue(*canonicalSource,
                                NonCanonicalEncodingPolicy::Reject);
    if (!classification)
      return formalError(FormalNumericErrorCode::InvalidSourceEncoding,
                         llvm::toString(classification.takeError()));
    if (classification->valueClass == LogicalValueClass::Infinity ||
        isNaNClass(classification->valueClass)) {
      return formalError(
          FormalNumericErrorCode::FloatToIntegerNonFinite,
          "NaN and infinity do not commit a floating-to-integer result");
    }

    llvm::APFloat floating = decodeFloat(*canonicalSource);
    llvm::APSInt result(destinationDescriptor->storageBits,
                        destinationDescriptor->category ==
                            LogicalFormatCategory::UnsignedInteger);
    bool ignoredIsExact = false;
    llvm::APFloat::opStatus status =
        floating.convertToInteger(result, *roundingMode, &ignoredIsExact);
    // In particular, -0 reports opOK even though APFloat's IsExact says the
    // integer result cannot retain its sign. Only opStatus maps to model flags.
    FormalNumericExceptionFlags flags = flagsFromStatus(status);
    if (flags.invalid) {
      return formalError(
          FormalNumericErrorCode::FloatToIntegerOutOfRange,
          "finite input is outside the destination integer range after "
          "rounding");
    }
    if (flags.divByZero || flags.overflow || flags.underflow) {
      return formalError(
          FormalNumericErrorCode::UnexpectedAPFloatStatus,
          "floating-to-integer conversion raised a non-integer status");
    }
    return finish(result.getZExtValue(), flags);
  }

  return formalError(FormalNumericErrorCode::UnsupportedEndpointKinds,
                     "the supported semantics row is not integer-to-floating, "
                     "floating-to-floating, or floating-to-integer");
}

llvm::Expected<FormalNumericResult>
executeFormalConvert(FormalNumericExecutionContext &context,
                     const ResolvedNumericCommand &command,
                     RawLogicalValue source) {
  llvm::Expected<FormalNumericResult> result =
      evaluateFormalConvert(command, source);
  if (!result)
    return result.takeError();
  context.recordCommittedFlags(result->flags);
  return std::move(*result);
}

} // namespace wafer
