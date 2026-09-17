//===- FormalNumericReduce.cpp - Formal native reduction step --------===//

#include "FormalNumericInternal.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Error.h"

namespace wafer {

using namespace formal_detail;

llvm::Expected<FormalNumericResult>
evaluateFormalReduceStep(const FormalReduceOperation &operation,
                         RawLogicalValue accumulator, RawLogicalValue input) {
  if (llvm::Error error = validateFormalReduceOperation(operation))
    return std::move(error);
  const LogicalFormat format = operation.input.getFormat();
  llvm::Expected<RawLogicalValue> canonicalAccumulator =
      validateOperand(accumulator, format, "native reduction accumulator");
  llvm::Expected<RawLogicalValue> canonicalInput =
      validateOperand(input, format, "native reduction input");
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
      *findLogicalFormatDescriptor(format);
  if (isNaNClass(accumulatorClass->valueClass) ||
      isNaNClass(inputClass->valueClass))
    return finishRawResult(format, canonicalPositiveQuietNaNBits(descriptor),
                           flags);

  llvm::APFloat result = decodeFloat(*canonicalAccumulator);
  llvm::APFloat inputValue = decodeFloat(*canonicalInput);
  if (operation.operation == TargetReduceOperation::Sum) {
    llvm::APFloat::opStatus status =
        result.add(inputValue, llvm::APFloat::rmNearestTiesToEven);
    mergeFlags(flags, flagsFromStatus(status));
  } else if (operation.operation == TargetReduceOperation::Max) {
    result = llvm::maximum(result, inputValue);
  } else {
    result = llvm::minimum(result, inputValue);
  }
  if (result.isNaN())
    return finishRawResult(format, canonicalPositiveQuietNaNBits(descriptor),
                           flags);
  std::optional<uint64_t> bits = encodeFloat(result, format);
  if (!bits)
    return formalError(FormalNumericErrorCode::InvalidResultEncoding,
                       "APFloat native reduction result has an unexpected "
                       "width");
  return finishRawResult(format, *bits, flags);
}

} // namespace wafer
