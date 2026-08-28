//===- FormalNumericSupport.cpp - Exact numeric support -------------===//

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

namespace wafer::formal_detail {

llvm::Error formalError(FormalNumericErrorCode code,
                        const llvm::Twine &detail) {
  return llvm::make_error<FormalNumericError>(code, detail.str());
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

} // namespace wafer::formal_detail

namespace wafer {

using namespace formal_detail;

llvm::StringRef stringifyFormalNumericErrorCode(FormalNumericErrorCode code) {
  switch (code) {
  case FormalNumericErrorCode::UnsupportedOperation:
    return "unsupported-operation";
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

void FormalNumericExecutionContext::mergeExceptionFlags(
    FormalNumericExceptionFlags flags) {
  aggregateFlags.invalid |= flags.invalid;
  aggregateFlags.divByZero |= flags.divByZero;
  aggregateFlags.overflow |= flags.overflow;
  aggregateFlags.underflow |= flags.underflow;
  aggregateFlags.inexact |= flags.inexact;
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
