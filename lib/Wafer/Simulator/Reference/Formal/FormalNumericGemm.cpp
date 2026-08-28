//===- FormalNumericGemm.cpp - Formal GEMM execution ----------------===//

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

llvm::Expected<FormalNumericResult>
evaluateFormalGemmFusedMultiplyAdd(const FormalGemmOperation &operation,
                                   RawLogicalValue lhs, RawLogicalValue rhs,
                                   RawLogicalValue accumulator) {
  if (llvm::Error error = validateFormalGemmOperation(operation))
    return std::move(error);
  const LogicalFormat operandFormat = operation.lhs.getFormat();
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
evaluateFormalGemmFinalize(const FormalGemmOperation &operation,
                           RawLogicalValue accumulator) {
  if (llvm::Error error = validateFormalGemmOperation(operation))
    return std::move(error);
  llvm::Expected<RawLogicalValue> canonicalAccumulator =
      validateOperand(accumulator, LogicalFormat::F32, "GEMM accumulator");
  if (!canonicalAccumulator)
    return canonicalAccumulator.takeError();
  llvm::Expected<LogicalValueClassification> classification =
      classifyOperand(*canonicalAccumulator, "GEMM accumulator");
  if (!classification)
    return classification.takeError();

  const LogicalFormat destinationFormat = operation.destination.getFormat();
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
  flags.underflow = flags.inexact &&
                    isTinyAfterUnboundedPrecisionRounding(
                        *canonicalAccumulator,
                        *findLogicalFormatDescriptor(LogicalFormat::F32),
                        destinationDescriptor, TargetRoundingMode::NearestEven);
  std::optional<uint64_t> bits = encodeFloat(result, destinationFormat);
  if (!bits)
    return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                       "APFloat GEMM destination has an unexpected width");
  return finishRawResult(destinationFormat, *bits, flags);
}

} // namespace wafer
