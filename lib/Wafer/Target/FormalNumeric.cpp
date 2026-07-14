//===- FormalNumeric.cpp - Deterministic formal numeric execution -------===//

#include "Wafer/Target/FormalNumeric.h"

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
namespace {

llvm::Error formalError(FormalNumericErrorCode code,
                        const llvm::Twine &detail) {
  return llvm::make_error<FormalNumericError>(code, detail.str());
}

bool isIntegerCategory(LogicalFormatCategory category) {
  return category == LogicalFormatCategory::SignedInteger ||
         category == LogicalFormatCategory::UnsignedInteger;
}

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

FormalNumericExceptionFlags flagsFromStatus(llvm::APFloat::opStatus status) {
  const unsigned bits = static_cast<unsigned>(status);
  return {
      (bits & static_cast<unsigned>(llvm::APFloat::opInvalidOp)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opDivByZero)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opOverflow)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opUnderflow)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opInexact)) != 0,
  };
}

llvm::APFloat decodeFloat(RawLogicalValue value) {
  const LogicalFormatDescriptor &descriptor =
      *findLogicalFormatDescriptor(value.format);
  const llvm::fltSemantics &semantics = *getFloatSemantics(value.format);
  if (value.format == LogicalFormat::TF32) {
    // LLVM FloatTF32 uses the compact 19-bit sign/exponent/fraction encoding;
    // target-independent RawLogicalValue keeps those bits in storage [31:13].
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

/// IEEE tininess-after is determined after rounding to the destination
/// precision as though the exponent range were unbounded.  This is not the
/// same as classifying the final finite-range encoding: a borderline tiny
/// intermediate can round again to the minimum normal encoding and still
/// raise underflow.  APFloat reports most narrowing underflows, but omits some
/// of these threshold cases, so derive this one profile bit from the exact raw
/// source instead of from a host floating-point approximation.
bool isTinyAfterUnboundedPrecisionRounding(
    RawLogicalValue source, const LogicalFormatDescriptor &sourceDescriptor,
    const LogicalFormatDescriptor &destinationDescriptor,
    NumericRoundingMode mode) {
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
    case NumericRoundingMode::NearestEven: {
      const uint64_t halfway = UINT64_C(1) << (shift - 1);
      increment =
          lost > halfway || (lost == halfway && (roundedSignificand & 1));
      break;
    }
    case NumericRoundingMode::TowardZero:
      break;
    case NumericRoundingMode::TowardPositive:
      increment = !negative && lost != 0;
      break;
    case NumericRoundingMode::TowardNegative:
      increment = negative && lost != 0;
      break;
    case NumericRoundingMode::Stochastic:
      llvm_unreachable("stochastic rounding reached deterministic formal path");
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

void mergeFlags(FormalNumericExceptionFlags &destination,
                FormalNumericExceptionFlags source) {
  destination.invalid |= source.invalid;
  destination.divByZero |= source.divByZero;
  destination.overflow |= source.overflow;
  destination.underflow |= source.underflow;
  destination.inexact |= source.inexact;
}

llvm::Expected<FormalNumericResult>
finishRawResult(LogicalFormat format, uint64_t bits,
                FormalNumericExceptionFlags flags) {
  llvm::Expected<RawLogicalValue> result =
      makeRawLogicalValue(format, bits, NonCanonicalEncodingPolicy::Reject);
  if (!result)
    return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                       llvm::toString(result.takeError()));
  return FormalNumericResult{*result, flags};
}

llvm::Expected<RawLogicalValue> validateOperand(RawLogicalValue operand,
                                                LogicalFormat expectedFormat,
                                                const llvm::Twine &role) {
  if (operand.format != expectedFormat)
    return formalError(FormalNumericErrorCode::OperandFormatMismatch,
                       role + " expected " +
                           stringifyLogicalFormat(expectedFormat) +
                           ", got logical format code " +
                           llvm::Twine(static_cast<unsigned>(operand.format)));
  llvm::Expected<RawLogicalValue> canonical = makeRawLogicalValue(
      operand.format, operand.bits, NonCanonicalEncodingPolicy::Reject);
  if (!canonical)
    return formalError(FormalNumericErrorCode::InvalidOperandEncoding,
                       role + ": " + llvm::toString(canonical.takeError()));
  return std::move(*canonical);
}

llvm::Expected<LogicalValueClassification>
classifyOperand(RawLogicalValue operand, const llvm::Twine &role) {
  llvm::Expected<LogicalValueClassification> classification =
      classifyRawLogicalValue(operand, NonCanonicalEncodingPolicy::Reject);
  if (!classification)
    return formalError(FormalNumericErrorCode::InvalidOperandEncoding,
                       role + ": " +
                           llvm::toString(classification.takeError()));
  return std::move(*classification);
}

struct ExactDyadic {
  bool negative;
  llvm::APInt magnitude;
  int exponent;
};

std::optional<ExactDyadic> decodeFiniteDyadic(RawLogicalValue value) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(value.format);
  if (!descriptor ||
      descriptor->category != LogicalFormatCategory::BinaryFloatingPoint)
    return std::nullopt;
  const unsigned fractionBits = descriptor->precisionBits - 1;
  const unsigned semanticShift =
      descriptor->storageBits - descriptor->semanticBits;
  const uint64_t semanticBits = value.bits >> semanticShift;
  const uint64_t fractionMask = (UINT64_C(1) << fractionBits) - UINT64_C(1);
  const uint64_t fraction = semanticBits & fractionMask;
  const uint64_t exponentMask =
      (UINT64_C(1) << descriptor->exponentBits) - UINT64_C(1);
  const uint64_t exponentField = (semanticBits >> fractionBits) & exponentMask;
  if (exponentField == exponentMask)
    return std::nullopt;

  uint64_t significand = fraction;
  const int bias = (1 << (descriptor->exponentBits - 1)) - 1;
  int exponent = 1 - bias - static_cast<int>(fractionBits);
  if (exponentField != 0) {
    significand |= UINT64_C(1) << fractionBits;
    exponent =
        static_cast<int>(exponentField) - bias - static_cast<int>(fractionBits);
  }
  const bool negative = (semanticBits >> (descriptor->semanticBits - 1)) != 0;
  return ExactDyadic{
      negative,
      llvm::APInt(std::max<unsigned>(1, descriptor->precisionBits),
                  significand),
      exponent};
}

ExactDyadic multiplyDyadics(const ExactDyadic &lhs, const ExactDyadic &rhs) {
  const unsigned lhsBits = std::max(1U, lhs.magnitude.getActiveBits());
  const unsigned rhsBits = std::max(1U, rhs.magnitude.getActiveBits());
  const unsigned width = lhsBits + rhsBits;
  llvm::APInt lhsMagnitude = lhs.magnitude.zextOrTrunc(width);
  llvm::APInt rhsMagnitude = rhs.magnitude.zextOrTrunc(width);
  return {lhs.negative != rhs.negative, lhsMagnitude * rhsMagnitude,
          lhs.exponent + rhs.exponent};
}

ExactDyadic addDyadics(const ExactDyadic &lhs, const ExactDyadic &rhs) {
  if (lhs.magnitude.isZero())
    return rhs;
  if (rhs.magnitude.isZero())
    return lhs;
  const int commonExponent = std::min(lhs.exponent, rhs.exponent);
  const unsigned lhsShift =
      static_cast<unsigned>(lhs.exponent - commonExponent);
  const unsigned rhsShift =
      static_cast<unsigned>(rhs.exponent - commonExponent);
  const unsigned lhsRequired = lhs.magnitude.getActiveBits() + lhsShift;
  const unsigned rhsRequired = rhs.magnitude.getActiveBits() + rhsShift;
  const unsigned width = std::max(lhsRequired, rhsRequired) + 1;
  llvm::APInt lhsMagnitude = lhs.magnitude.zextOrTrunc(width).shl(lhsShift);
  llvm::APInt rhsMagnitude = rhs.magnitude.zextOrTrunc(width).shl(rhsShift);
  if (lhs.negative == rhs.negative)
    return {lhs.negative, lhsMagnitude + rhsMagnitude, commonExponent};
  if (lhsMagnitude.uge(rhsMagnitude))
    return {lhs.negative, lhsMagnitude - rhsMagnitude, commonExponent};
  return {rhs.negative, rhsMagnitude - lhsMagnitude, commonExponent};
}

/// Exact dyadic proof of IEEE tininess after an unbounded-exponent RNE step.
/// This deliberately does not inspect the APFloat result class or change its
/// bits. Finite exponent-range encoding may still be inexact even when this
/// precision-only step is exact, so the caller combines this predicate with
/// APFloat's independently reported inexact bit.
bool isTinyAfterRNE(const ExactDyadic &exact,
                    const LogicalFormatDescriptor &destinationDescriptor) {
  if (exact.magnitude.isZero())
    return false;
  const unsigned precision = destinationDescriptor.precisionBits;
  const unsigned activeBits = exact.magnitude.getActiveBits();
  llvm::APInt rounded = exact.magnitude;
  int roundedExponent = exact.exponent;
  if (activeBits > precision) {
    const unsigned shift = activeBits - precision;
    rounded = exact.magnitude.lshr(shift);
    roundedExponent += static_cast<int>(shift);
    const bool halfwayBit = exact.magnitude[shift - 1];
    const bool lowerBits =
        shift > 1 && !exact.magnitude.trunc(shift - 1).isZero();
    if (halfwayBit && (lowerBits || rounded[0])) {
      ++rounded;
      if (rounded.getActiveBits() > precision) {
        rounded = rounded.lshr(1);
        ++roundedExponent;
      }
    }
  }
  const int roundedTopExponent =
      roundedExponent + static_cast<int>(rounded.getActiveBits()) - 1;
  const int bias = (1 << (destinationDescriptor.exponentBits - 1)) - 1;
  return roundedTopExponent < 1 - bias;
}

bool isLLVMElementwiseOperation(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Abs:
  case NumericElementwiseOperation::Recip:
  case NumericElementwiseOperation::Square:
  case NumericElementwiseOperation::Neg:
  case NumericElementwiseOperation::Max:
  case NumericElementwiseOperation::Min:
  case NumericElementwiseOperation::Add:
  case NumericElementwiseOperation::Sub:
  case NumericElementwiseOperation::Mul:
  case NumericElementwiseOperation::Div:
  case NumericElementwiseOperation::Eq:
  case NumericElementwiseOperation::Ne:
  case NumericElementwiseOperation::Ge:
  case NumericElementwiseOperation::Gt:
  case NumericElementwiseOperation::Le:
  case NumericElementwiseOperation::Lt:
  case NumericElementwiseOperation::LogicNot:
  case NumericElementwiseOperation::LogicAnd:
  case NumericElementwiseOperation::LogicOr:
  case NumericElementwiseOperation::LogicXor:
  case NumericElementwiseOperation::Relu:
    return true;
  case NumericElementwiseOperation::Sqrt:
  case NumericElementwiseOperation::Rsqrt:
  case NumericElementwiseOperation::Log2:
  case NumericElementwiseOperation::Ln:
  case NumericElementwiseOperation::Pow2:
  case NumericElementwiseOperation::Exp:
  case NumericElementwiseOperation::ExpLp:
  case NumericElementwiseOperation::Sin:
  case NumericElementwiseOperation::Cos:
  case NumericElementwiseOperation::Tanh:
  case NumericElementwiseOperation::Sigmoid:
  case NumericElementwiseOperation::SatRelu:
  case NumericElementwiseOperation::LeakyRelu:
  case NumericElementwiseOperation::Softplus:
    return false;
  }
  return false;
}

llvm::Error validateResolvedExecution(const ResolvedNumericCommand &command,
                                      NumericCommandFamily family,
                                      FormalKernelKind kernel,
                                      FormalNumericBackendKind backend) {
  if (command.getFamily() != family || !command.isSupported() ||
      !command.getSemantics() || command.getFormalKernelKind() != kernel ||
      command.getComparatorKind() != NumericComparatorKind::RawExact ||
      command.getFormalBackendKind() != backend)
    return formalError(
        FormalNumericErrorCode::UnsupportedResolvedCommand,
        "the resolved command has no complete matching formal identity");
  return llvm::Error::success();
}

llvm::Error
validateElementwiseResolvedCommand(const ResolvedNumericCommand &command) {
  if (llvm::Error error = validateResolvedExecution(
          command, NumericCommandFamily::CTElementwise,
          FormalKernelKind::Elementwise,
          FormalNumericBackendKind::LLVMAPFloatAPInt))
    return error;
  const NumericCTElementwiseCommand *elementwise =
      command.getCommandKey().getCTElementwise();
  const NumericSemanticsProfile &semantics = *command.getSemantics();
  const NumericCTElementwiseSemanticsIdentity *identity =
      semantics.getCTElementwiseIdentity();
  if (!elementwise || !identity ||
      semantics.getModelProfile() != command.getPattern().getModelProfile() ||
      identity->getTargetProfile() !=
          command.getCommandKey().getTargetProfile() ||
      identity->getOperation() != elementwise->operation ||
      !isLLVMElementwiseOperation(elementwise->operation) ||
      elementwise->inputs.empty() ||
      identity->getInputFormat() != elementwise->inputs.front().getFormat() ||
      identity->getDestinationFormat() !=
          elementwise->destination.getFormat() ||
      elementwise->inputs.size() !=
          getNumericElementwiseArity(elementwise->operation))
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "resolved elementwise semantics and exact key disagree");
  for (const NumericTensorKey &input : elementwise->inputs)
    if (input.getFormat() != identity->getInputFormat())
      return formalError(
          FormalNumericErrorCode::UnsupportedResolvedCommand,
          "resolved elementwise key has heterogeneous input formats");

  const bool logic = isNumericElementwiseLogic(elementwise->operation);
  const bool relation = isNumericElementwiseRelation(elementwise->operation);
  if ((!logic && identity->getInputFormat() != LogicalFormat::F16 &&
       identity->getInputFormat() != LogicalFormat::BF16 &&
       identity->getInputFormat() != LogicalFormat::F32) ||
      (logic && identity->getInputFormat() != LogicalFormat::Bool) ||
      identity->getDestinationFormat() !=
          (relation ? LogicalFormat::Bool : identity->getInputFormat()))
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "elementwise format is outside the LLVM model subset");

  FloatingSignedZeroPolicy signedZeroPolicy =
      logic      ? FloatingSignedZeroPolicy::NotApplicable
      : relation ? FloatingSignedZeroPolicy::PredicateOnly
                 : FloatingSignedZeroPolicy::IEEE754OperationDefined;
  if (elementwise->operation == NumericElementwiseOperation::Max)
    signedZeroPolicy =
        FloatingSignedZeroPolicy::MaximumPositiveUnlessBothNegative;
  else if (elementwise->operation == NumericElementwiseOperation::Min)
    signedZeroPolicy =
        FloatingSignedZeroPolicy::MinimumNegativeUnlessBothPositive;
  const std::optional<NumericRoundingMode> roundingMode =
      logic || relation ? std::nullopt
                        : std::optional<NumericRoundingMode>(
                              NumericRoundingMode::NearestEven);
  if (semantics.getRoundingModePolicy() != roundingMode ||
      semantics.getRoundingPointPolicy() !=
          (logic || relation ? NumericRoundingPointPolicy::NotApplicable
                             : NumericRoundingPointPolicy::ElementwiseResult) ||
      semantics.getFloatToIntegerPolicy() !=
          FloatToIntegerPolicy::NotApplicable ||
      semantics.getFloatingNaNPolicy() !=
          (logic ? FloatingNaNPolicy::NotApplicable
           : relation
               ? FloatingNaNPolicy::OrderedRelationFalseExceptNotEqualTrue
               : FloatingNaNPolicy::CanonicalPositiveQuietNaN) ||
      semantics.getFloatingSignedZeroPolicy() != signedZeroPolicy ||
      semantics.getFloatingSubnormalPolicy() !=
          (logic ? FloatingSubnormalPolicy::NotApplicable
                 : FloatingSubnormalPolicy::Gradual) ||
      semantics.getFloatingTininessPolicy() !=
          (logic || relation ? FloatingTininessPolicy::NotApplicable
                             : FloatingTininessPolicy::AfterRounding) ||
      semantics.getExceptionFlagPolicy() !=
          (logic ? NumericExceptionFlagPolicy::NotApplicable
                 : NumericExceptionFlagPolicy::ModelOnly) ||
      semantics.getFloatingNaNSignalingPolicy() !=
          (logic ? FloatingNaNSignalingPolicy::NotApplicable
                 : FloatingNaNSignalingPolicy::
                       SignalingRaisesInvalidQuietDoesNot) ||
      semantics.getFloatingDenormalModePolicy() !=
          (logic ? FloatingDenormalModePolicy::NotApplicable
                 : FloatingDenormalModePolicy::GradualNoDAZNoFTZ) ||
      semantics.getFloatingOverflowPolicy() !=
          (logic || relation
               ? FloatingOverflowPolicy::NotApplicable
               : FloatingOverflowPolicy::IEEE754AccordingToRoundingMode) ||
      semantics.getSaturationPolicy() !=
          (logic || relation ? NumericSaturationPolicy::NotApplicable
                             : NumericSaturationPolicy::Disabled) ||
      semantics.getTranscendentalEvaluationPolicy() !=
          NumericTranscendentalEvaluationPolicy::NotApplicable ||
      semantics.getGemmAccumulatorPolicy() !=
          NumericGemmAccumulatorPolicy::NotApplicable ||
      semantics.getGemmAccumulatorInitializationPolicy() !=
          NumericGemmAccumulatorInitializationPolicy::NotApplicable ||
      semantics.getGemmReductionOrderPolicy() !=
          NumericGemmReductionOrderPolicy::NotApplicable)
    return formalError(
        FormalNumericErrorCode::UnsupportedResolvedCommand,
        "resolved elementwise semantics has an incompatible numeric policy");
  return llvm::Error::success();
}

llvm::Error validateGemmResolvedCommand(const ResolvedNumericCommand &command) {
  if (llvm::Error error = validateResolvedExecution(
          command, NumericCommandFamily::NEGemm, FormalKernelKind::Gemm,
          FormalNumericBackendKind::LLVMAPFloatAPInt))
    return error;
  const NumericNEGemmCommand *gemm = command.getCommandKey().getNEGemm();
  const NumericSemanticsProfile &semantics = *command.getSemantics();
  const NumericNEGemmSemanticsIdentity *identity =
      semantics.getNEGemmIdentity();
  if (!gemm || !identity ||
      semantics.getModelProfile() != command.getPattern().getModelProfile() ||
      identity->getTargetProfile() !=
          command.getCommandKey().getTargetProfile() ||
      identity->getFormat() != gemm->lhs.getFormat() ||
      gemm->rhs.getFormat() != identity->getFormat() ||
      gemm->destination.getFormat() != identity->getFormat() ||
      (identity->getFormat() != LogicalFormat::F16 &&
       identity->getFormat() != LogicalFormat::BF16 &&
       identity->getFormat() != LogicalFormat::F32) ||
      semantics.getRoundingModePolicy() != NumericRoundingMode::NearestEven ||
      semantics.getRoundingPointPolicy() !=
          NumericRoundingPointPolicy::GemmFusedMultiplyAddAndDestination ||
      semantics.getFloatToIntegerPolicy() !=
          FloatToIntegerPolicy::NotApplicable ||
      semantics.getFloatingNaNPolicy() !=
          FloatingNaNPolicy::CanonicalPositiveQuietNaN ||
      semantics.getFloatingSignedZeroPolicy() !=
          FloatingSignedZeroPolicy::GemmPositiveZeroAccumulatorThenIEEE754 ||
      semantics.getFloatingSubnormalPolicy() !=
          FloatingSubnormalPolicy::Gradual ||
      semantics.getFloatingTininessPolicy() !=
          FloatingTininessPolicy::AfterRounding ||
      semantics.getExceptionFlagPolicy() !=
          NumericExceptionFlagPolicy::ModelOnly ||
      semantics.getFloatingNaNSignalingPolicy() !=
          FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot ||
      semantics.getFloatingDenormalModePolicy() !=
          FloatingDenormalModePolicy::GradualNoDAZNoFTZ ||
      semantics.getFloatingOverflowPolicy() !=
          FloatingOverflowPolicy::IEEE754AccordingToRoundingMode ||
      semantics.getSaturationPolicy() != NumericSaturationPolicy::Disabled ||
      semantics.getTranscendentalEvaluationPolicy() !=
          NumericTranscendentalEvaluationPolicy::NotApplicable ||
      semantics.getGemmAccumulatorPolicy() !=
          NumericGemmAccumulatorPolicy::F32FusedMultiplyAdd ||
      semantics.getGemmAccumulatorInitializationPolicy() !=
          NumericGemmAccumulatorInitializationPolicy::PositiveZero ||
      semantics.getGemmReductionOrderPolicy() !=
          NumericGemmReductionOrderPolicy::IncreasingK)
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "resolved GEMM semantics and exact key disagree");
  return llvm::Error::success();
}

} // namespace

llvm::StringRef stringifyFormalNumericErrorCode(FormalNumericErrorCode code) {
  switch (code) {
  case FormalNumericErrorCode::UnsupportedResolvedCommand:
    return "unsupported-resolved-command";
  case FormalNumericErrorCode::OperandCountMismatch:
    return "operand-count-mismatch";
  case FormalNumericErrorCode::OperandFormatMismatch:
    return "operand-format-mismatch";
  case FormalNumericErrorCode::InvalidOperandEncoding:
    return "invalid-operand-encoding";
  case FormalNumericErrorCode::SourceFormatMismatch:
    return "source-format-mismatch";
  case FormalNumericErrorCode::InvalidSourceEncoding:
    return "invalid-source-encoding";
  case FormalNumericErrorCode::UnsupportedEndpointKinds:
    return "unsupported-endpoint-kinds";
  case FormalNumericErrorCode::FloatToIntegerNonFinite:
    return "float-to-integer-non-finite";
  case FormalNumericErrorCode::FloatToIntegerOutOfRange:
    return "float-to-integer-out-of-range";
  case FormalNumericErrorCode::UnexpectedAPFloatStatus:
    return "unexpected-apfloat-status";
  case FormalNumericErrorCode::InvalidResultEncoding:
    return "invalid-result-encoding";
  case FormalNumericErrorCode::InvalidComparatorOperandEncoding:
    return "invalid-comparator-operand-encoding";
  }
  llvm_unreachable("formal numeric error code is not registered");
}

char FormalNumericError::ID;

void FormalNumericError::log(llvm::raw_ostream &stream) const {
  stream << "formal numeric " << stringifyFormalNumericErrorCode(code) << ": "
         << detail;
}

std::error_code FormalNumericError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

void FormalNumericExecutionContext::recordCommittedFlags(
    FormalNumericExceptionFlags flags) {
  aggregateFlags.invalid |= flags.invalid;
  aggregateFlags.divByZero |= flags.divByZero;
  aggregateFlags.overflow |= flags.overflow;
  aggregateFlags.underflow |= flags.underflow;
  aggregateFlags.inexact |= flags.inexact;
}

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
      findTargetConvertRoute(key.getTargetProfile(), convert->opcode);
  if (!routeRecord || convert->source.getFormat() != routeRecord->source ||
      convert->destination.getFormat() != routeRecord->destination)
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "the resolved convert key lost its validated route");
  const TargetConvertRoute &route = *routeRecord;
  const NumericRoutePolicyIdentity &routePolicy =
      semantics.getRoutePolicyIdentity();
  if (semantics.getModelProfile() != command.getPattern().getModelProfile() ||
      routePolicy.getTargetProfile() != key.getTargetProfile() ||
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

llvm::Expected<FormalNumericResult>
evaluateFormalElementwiseLLVM(const ResolvedNumericCommand &command,
                              llvm::ArrayRef<RawLogicalValue> inputs) {
  if (llvm::Error error = validateElementwiseResolvedCommand(command))
    return std::move(error);
  const NumericCTElementwiseCommand &elementwise =
      *command.getCommandKey().getCTElementwise();
  const NumericCTElementwiseSemanticsIdentity &identity =
      *command.getSemantics()->getCTElementwiseIdentity();
  if (inputs.size() != elementwise.inputs.size())
    return formalError(FormalNumericErrorCode::OperandCountMismatch,
                       llvm::Twine("elementwise operation expects ") +
                           llvm::Twine(elementwise.inputs.size()) +
                           " scalar operands, got " +
                           llvm::Twine(inputs.size()));

  std::vector<RawLogicalValue> canonicalInputs;
  canonicalInputs.reserve(inputs.size());
  for (auto [index, input] : llvm::enumerate(inputs)) {
    llvm::Expected<RawLogicalValue> canonical = validateOperand(
        input, identity.getInputFormat(),
        llvm::Twine("elementwise operand ") + llvm::Twine(index));
    if (!canonical)
      return canonical.takeError();
    canonicalInputs.push_back(*canonical);
  }

  const NumericElementwiseOperation operation = elementwise.operation;
  if (isNumericElementwiseLogic(operation)) {
    const bool lhs = canonicalInputs[0].bits != 0;
    bool result = false;
    switch (operation) {
    case NumericElementwiseOperation::LogicNot:
      result = !lhs;
      break;
    case NumericElementwiseOperation::LogicAnd:
      result = lhs && canonicalInputs[1].bits != 0;
      break;
    case NumericElementwiseOperation::LogicOr:
      result = lhs || canonicalInputs[1].bits != 0;
      break;
    case NumericElementwiseOperation::LogicXor:
      result = lhs != (canonicalInputs[1].bits != 0);
      break;
    default:
      llvm_unreachable("non-logic operation passed the logic validator");
    }
    return finishRawResult(LogicalFormat::Bool, result, {});
  }

  std::vector<LogicalValueClassification> classifications;
  std::vector<llvm::APFloat> floatingInputs;
  classifications.reserve(canonicalInputs.size());
  floatingInputs.reserve(canonicalInputs.size());
  FormalNumericExceptionFlags flags;
  bool hasNaN = false;
  for (auto [index, input] : llvm::enumerate(canonicalInputs)) {
    llvm::Expected<LogicalValueClassification> classification = classifyOperand(
        input, llvm::Twine("elementwise operand ") + llvm::Twine(index));
    if (!classification)
      return classification.takeError();
    hasNaN |= isNaNClass(classification->valueClass);
    flags.invalid |=
        classification->valueClass == LogicalValueClass::SignalingNaN;
    classifications.push_back(*classification);
    floatingInputs.push_back(decodeFloat(input));
  }
  const LogicalFormat resultFormat = identity.getDestinationFormat();
  const LogicalFormatDescriptor &resultDescriptor =
      *findLogicalFormatDescriptor(resultFormat);
  auto finishNaN = [&]() -> llvm::Expected<FormalNumericResult> {
    return finishRawResult(
        resultFormat, canonicalPositiveQuietNaNBits(resultDescriptor), flags);
  };

  if (isNumericElementwiseRelation(operation)) {
    bool result = false;
    if (hasNaN) {
      result = operation == NumericElementwiseOperation::Ne;
    } else {
      const llvm::APFloat::cmpResult comparison =
          floatingInputs[0].compare(floatingInputs[1]);
      switch (operation) {
      case NumericElementwiseOperation::Eq:
        result = comparison == llvm::APFloat::cmpEqual;
        break;
      case NumericElementwiseOperation::Ne:
        result = comparison != llvm::APFloat::cmpEqual;
        break;
      case NumericElementwiseOperation::Ge:
        result = comparison == llvm::APFloat::cmpGreaterThan ||
                 comparison == llvm::APFloat::cmpEqual;
        break;
      case NumericElementwiseOperation::Gt:
        result = comparison == llvm::APFloat::cmpGreaterThan;
        break;
      case NumericElementwiseOperation::Le:
        result = comparison == llvm::APFloat::cmpLessThan ||
                 comparison == llvm::APFloat::cmpEqual;
        break;
      case NumericElementwiseOperation::Lt:
        result = comparison == llvm::APFloat::cmpLessThan;
        break;
      default:
        llvm_unreachable("non-relation operation passed relation validator");
      }
    }
    return finishRawResult(LogicalFormat::Bool, result, flags);
  }

  if ((operation == NumericElementwiseOperation::Abs ||
       operation == NumericElementwiseOperation::Neg ||
       operation == NumericElementwiseOperation::Max ||
       operation == NumericElementwiseOperation::Min ||
       operation == NumericElementwiseOperation::Relu) &&
      hasNaN)
    return finishNaN();

  llvm::APFloat result = floatingInputs[0];
  llvm::APFloat::opStatus status = llvm::APFloat::opOK;
  switch (operation) {
  case NumericElementwiseOperation::Abs:
    result.clearSign();
    break;
  case NumericElementwiseOperation::Neg:
    result.changeSign();
    break;
  case NumericElementwiseOperation::Recip:
    result =
        llvm::APFloat::getOne(*getFloatSemantics(identity.getInputFormat()));
    status =
        result.divide(floatingInputs[0], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Square:
    status =
        result.multiply(floatingInputs[0], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Max:
  case NumericElementwiseOperation::Min: {
    if (floatingInputs[0].isZero() && floatingInputs[1].isZero()) {
      const bool negative =
          operation == NumericElementwiseOperation::Max
              ? classifications[0].negative && classifications[1].negative
              : classifications[0].negative || classifications[1].negative;
      result = llvm::APFloat::getZero(
          *getFloatSemantics(identity.getInputFormat()), negative);
      break;
    }
    const llvm::APFloat::cmpResult comparison =
        floatingInputs[0].compare(floatingInputs[1]);
    const bool chooseRhs = operation == NumericElementwiseOperation::Max
                               ? comparison == llvm::APFloat::cmpLessThan
                               : comparison == llvm::APFloat::cmpGreaterThan;
    if (chooseRhs)
      result = floatingInputs[1];
    break;
  }
  case NumericElementwiseOperation::Add:
    status = result.add(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Sub:
    status =
        result.subtract(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Mul:
    status =
        result.multiply(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Div:
    status =
        result.divide(floatingInputs[1], llvm::APFloat::rmNearestTiesToEven);
    break;
  case NumericElementwiseOperation::Relu:
    if (classifications[0].negative && !floatingInputs[0].isZero())
      result =
          llvm::APFloat::getZero(*getFloatSemantics(identity.getInputFormat()));
    break;
  case NumericElementwiseOperation::Eq:
  case NumericElementwiseOperation::Ne:
  case NumericElementwiseOperation::Ge:
  case NumericElementwiseOperation::Gt:
  case NumericElementwiseOperation::Le:
  case NumericElementwiseOperation::Lt:
  case NumericElementwiseOperation::LogicNot:
  case NumericElementwiseOperation::LogicAnd:
  case NumericElementwiseOperation::LogicOr:
  case NumericElementwiseOperation::LogicXor:
  case NumericElementwiseOperation::Sqrt:
  case NumericElementwiseOperation::Rsqrt:
  case NumericElementwiseOperation::Log2:
  case NumericElementwiseOperation::Ln:
  case NumericElementwiseOperation::Pow2:
  case NumericElementwiseOperation::Exp:
  case NumericElementwiseOperation::ExpLp:
  case NumericElementwiseOperation::Sin:
  case NumericElementwiseOperation::Cos:
  case NumericElementwiseOperation::Tanh:
  case NumericElementwiseOperation::Sigmoid:
  case NumericElementwiseOperation::SatRelu:
  case NumericElementwiseOperation::LeakyRelu:
  case NumericElementwiseOperation::Softplus:
    llvm_unreachable("operation escaped the LLVM elementwise validator");
  }
  mergeFlags(flags, flagsFromStatus(status));

  if ((operation == NumericElementwiseOperation::Mul ||
       operation == NumericElementwiseOperation::Square) &&
      flags.inexact) {
    std::optional<ExactDyadic> lhs = decodeFiniteDyadic(canonicalInputs[0]);
    const RawLogicalValue rhsValue =
        operation == NumericElementwiseOperation::Square ? canonicalInputs[0]
                                                         : canonicalInputs[1];
    std::optional<ExactDyadic> rhs = decodeFiniteDyadic(rhsValue);
    if (lhs && rhs)
      flags.underflow |=
          isTinyAfterRNE(multiplyDyadics(*lhs, *rhs), resultDescriptor);
  }

  if (result.isNaN())
    return finishNaN();
  std::optional<uint64_t> bits = encodeFloat(result, resultFormat);
  if (!bits)
    return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                       "APFloat elementwise result has an unexpected width");
  return finishRawResult(resultFormat, *bits, flags);
}

llvm::Expected<FormalNumericResult>
evaluateFormalGemmFusedMultiplyAdd(const ResolvedNumericCommand &command,
                                   RawLogicalValue lhs, RawLogicalValue rhs,
                                   RawLogicalValue accumulator) {
  if (llvm::Error error = validateGemmResolvedCommand(command))
    return std::move(error);
  const LogicalFormat operandFormat =
      command.getSemantics()->getNEGemmIdentity()->getFormat();
  llvm::Expected<RawLogicalValue> canonicalLhs =
      validateOperand(lhs, operandFormat, "GEMM lhs");
  if (!canonicalLhs)
    return canonicalLhs.takeError();
  llvm::Expected<RawLogicalValue> canonicalRhs =
      validateOperand(rhs, operandFormat, "GEMM rhs");
  if (!canonicalRhs)
    return canonicalRhs.takeError();
  llvm::Expected<RawLogicalValue> canonicalAccumulator =
      validateOperand(accumulator, LogicalFormat::F32, "GEMM accumulator");
  if (!canonicalAccumulator)
    return canonicalAccumulator.takeError();

  FormalNumericExceptionFlags flags;
  for (auto [role, value] :
       {std::pair<llvm::StringRef, RawLogicalValue>("GEMM lhs", *canonicalLhs),
        std::pair<llvm::StringRef, RawLogicalValue>("GEMM rhs", *canonicalRhs),
        std::pair<llvm::StringRef, RawLogicalValue>("GEMM accumulator",
                                                    *canonicalAccumulator)}) {
    llvm::Expected<LogicalValueClassification> classification =
        classifyOperand(value, role);
    if (!classification)
      return classification.takeError();
    flags.invalid |=
        classification->valueClass == LogicalValueClass::SignalingNaN;
  }

  auto widenToF32 = [&](RawLogicalValue value,
                        llvm::StringRef role) -> llvm::Expected<llvm::APFloat> {
    llvm::APFloat widened = decodeFloat(value);
    if (value.format == LogicalFormat::F32)
      return widened;
    bool losesInfo = false;
    const llvm::APFloat::opStatus status =
        widened.convert(llvm::APFloat::IEEEsingle(),
                        llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    FormalNumericExceptionFlags conversionFlags = flagsFromStatus(status);
    mergeFlags(flags, conversionFlags);
    if ((losesInfo && !widened.isNaN()) || conversionFlags.divByZero ||
        conversionFlags.overflow || conversionFlags.underflow ||
        conversionFlags.inexact)
      return formalError(FormalNumericErrorCode::UnexpectedAPFloatStatus,
                         role + " did not widen exactly to F32");
    return widened;
  };
  llvm::Expected<llvm::APFloat> lhsF32 = widenToF32(*canonicalLhs, "GEMM lhs");
  if (!lhsF32)
    return lhsF32.takeError();
  llvm::Expected<llvm::APFloat> rhsF32 = widenToF32(*canonicalRhs, "GEMM rhs");
  if (!rhsF32)
    return rhsF32.takeError();
  llvm::APFloat result = std::move(*lhsF32);
  const llvm::APFloat accumulatorF32 = decodeFloat(*canonicalAccumulator);
  const llvm::APFloat::opStatus status = result.fusedMultiplyAdd(
      *rhsF32, accumulatorF32, llvm::APFloat::rmNearestTiesToEven);
  mergeFlags(flags, flagsFromStatus(status));

  if (flags.inexact) {
    const std::optional<ExactDyadic> lhsExact =
        decodeFiniteDyadic(*canonicalLhs);
    const std::optional<ExactDyadic> rhsExact =
        decodeFiniteDyadic(*canonicalRhs);
    const std::optional<ExactDyadic> accumulatorExact =
        decodeFiniteDyadic(*canonicalAccumulator);
    if (lhsExact && rhsExact && accumulatorExact) {
      const ExactDyadic exact =
          addDyadics(multiplyDyadics(*lhsExact, *rhsExact), *accumulatorExact);
      flags.underflow |= isTinyAfterRNE(
          exact, *findLogicalFormatDescriptor(LogicalFormat::F32));
    }
  }

  if (result.isNaN())
    return finishRawResult(
        LogicalFormat::F32,
        canonicalPositiveQuietNaNBits(
            *findLogicalFormatDescriptor(LogicalFormat::F32)),
        flags);
  std::optional<uint64_t> bits = encodeFloat(result, LogicalFormat::F32);
  if (!bits)
    return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                       "APFloat GEMM accumulator has an unexpected width");
  return finishRawResult(LogicalFormat::F32, *bits, flags);
}

llvm::Expected<FormalNumericResult>
evaluateFormalGemmFinalize(const ResolvedNumericCommand &command,
                           RawLogicalValue accumulator) {
  if (llvm::Error error = validateGemmResolvedCommand(command))
    return std::move(error);
  llvm::Expected<RawLogicalValue> canonicalAccumulator =
      validateOperand(accumulator, LogicalFormat::F32, "GEMM accumulator");
  if (!canonicalAccumulator)
    return canonicalAccumulator.takeError();
  llvm::Expected<LogicalValueClassification> classification =
      classifyOperand(*canonicalAccumulator, "GEMM accumulator");
  if (!classification)
    return classification.takeError();

  const LogicalFormat destinationFormat =
      command.getSemantics()->getNEGemmIdentity()->getFormat();
  const LogicalFormatDescriptor &destinationDescriptor =
      *findLogicalFormatDescriptor(destinationFormat);
  FormalNumericExceptionFlags flags;
  if (isNaNClass(classification->valueClass)) {
    flags.invalid =
        classification->valueClass == LogicalValueClass::SignalingNaN;
    return finishRawResult(destinationFormat,
                           canonicalPositiveQuietNaNBits(destinationDescriptor),
                           flags);
  }

  llvm::APFloat result = decodeFloat(*canonicalAccumulator);
  bool ignoredLosesInfo = false;
  const llvm::APFloat::opStatus status =
      result.convert(*getFloatSemantics(destinationFormat),
                     llvm::APFloat::rmNearestTiesToEven, &ignoredLosesInfo);
  flags = flagsFromStatus(status);
  flags.underflow =
      flags.inexact &&
      isTinyAfterUnboundedPrecisionRounding(
          *canonicalAccumulator,
          *findLogicalFormatDescriptor(LogicalFormat::F32),
          destinationDescriptor, NumericRoundingMode::NearestEven);
  std::optional<uint64_t> bits = encodeFloat(result, destinationFormat);
  if (!bits)
    return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                       "APFloat GEMM destination has an unexpected width");
  return finishRawResult(destinationFormat, *bits, flags);
}

llvm::Expected<bool>
compareFormalNumericResultsExact(const FormalNumericResult &lhs,
                                 const FormalNumericResult &rhs) {
  llvm::Expected<RawLogicalValue> canonicalLhs = makeRawLogicalValue(
      lhs.value.format, lhs.value.bits, NonCanonicalEncodingPolicy::Reject);
  if (!canonicalLhs)
    return formalError(FormalNumericErrorCode::InvalidComparatorOperandEncoding,
                       llvm::Twine("left operand: ") +
                           llvm::toString(canonicalLhs.takeError()));
  llvm::Expected<RawLogicalValue> canonicalRhs = makeRawLogicalValue(
      rhs.value.format, rhs.value.bits, NonCanonicalEncodingPolicy::Reject);
  if (!canonicalRhs)
    return formalError(FormalNumericErrorCode::InvalidComparatorOperandEncoding,
                       llvm::Twine("right operand: ") +
                           llvm::toString(canonicalRhs.takeError()));
  return canonicalLhs->format == canonicalRhs->format &&
         canonicalLhs->bits == canonicalRhs->bits && lhs.flags == rhs.flags;
}

} // namespace wafer
