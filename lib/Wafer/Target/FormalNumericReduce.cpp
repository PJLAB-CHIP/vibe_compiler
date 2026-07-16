//===- FormalNumericReduce.cpp - Formal native reduction step --------===//

#include "FormalNumericInternal.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Error.h"

namespace wafer {

using namespace formal_detail;

llvm::Expected<FormalNumericResult>
evaluateFormalReduceStep(const ResolvedNumericCommand &command,
                         RawLogicalValue accumulator, RawLogicalValue input) {
  if (llvm::Error error = validateReduceResolvedCommand(command))
    return std::move(error);
  llvm::Expected<RawLogicalValue> canonicalAccumulator = validateOperand(
      accumulator, LogicalFormat::F32, "native reduction accumulator");
  llvm::Expected<RawLogicalValue> canonicalInput =
      validateOperand(input, LogicalFormat::F32, "native reduction input");
  if (!canonicalAccumulator)
    return canonicalAccumulator.takeError();
  if (!canonicalInput)
    return canonicalInput.takeError();

  llvm::Expected<LogicalValueClassification> accumulatorClass =
      classifyOperand(*canonicalAccumulator, "native reduction accumulator");
  llvm::Expected<LogicalValueClassification> inputClass =
      classifyOperand(*canonicalInput, "native reduction input");
  if (!accumulatorClass)
    return accumulatorClass.takeError();
  if (!inputClass)
    return inputClass.takeError();

  FormalNumericExceptionFlags flags;
  flags.invalid =
      accumulatorClass->valueClass == LogicalValueClass::SignalingNaN ||
      inputClass->valueClass == LogicalValueClass::SignalingNaN;
  const LogicalFormatDescriptor &descriptor =
      *findLogicalFormatDescriptor(LogicalFormat::F32);
  if (isNaNClass(accumulatorClass->valueClass) ||
      isNaNClass(inputClass->valueClass))
    return finishRawResult(LogicalFormat::F32,
                           canonicalPositiveQuietNaNBits(descriptor), flags);

  llvm::APFloat result = decodeFloat(*canonicalAccumulator);
  llvm::APFloat::opStatus status = result.add(
      decodeFloat(*canonicalInput), llvm::APFloat::rmNearestTiesToEven);
  mergeFlags(flags, flagsFromStatus(status));
  if (result.isNaN())
    return finishRawResult(LogicalFormat::F32,
                           canonicalPositiveQuietNaNBits(descriptor), flags);
  std::optional<uint64_t> bits = encodeFloat(result, LogicalFormat::F32);
  if (!bits)
    return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                       "APFloat native reduction result has an unexpected "
                       "width");
  return finishRawResult(LogicalFormat::F32, *bits, flags);
}

} // namespace wafer
