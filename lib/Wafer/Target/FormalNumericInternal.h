//===- FormalNumericInternal.h - Formal numeric collaboration -*- C++ -*-===//

#ifndef WAFER_TARGET_FORMALNUMERICINTERNAL_H
#define WAFER_TARGET_FORMALNUMERICINTERNAL_H

#include "Wafer/Target/FormalNumeric.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>

namespace wafer::formal_detail {

llvm::Error formalError(FormalNumericErrorCode code, const llvm::Twine &detail);

const llvm::fltSemantics *getFloatSemantics(LogicalFormat format);
FormalNumericExceptionFlags flagsFromStatus(llvm::APFloat::opStatus status);
llvm::APFloat decodeFloat(RawLogicalValue value);
std::optional<uint64_t> encodeFloat(const llvm::APFloat &value,
                                    LogicalFormat format);
uint64_t
canonicalPositiveQuietNaNBits(const LogicalFormatDescriptor &descriptor);
bool isNaNClass(LogicalValueClass valueClass);
bool isTinyAfterUnboundedPrecisionRounding(
    RawLogicalValue source, const LogicalFormatDescriptor &sourceDescriptor,
    const LogicalFormatDescriptor &destinationDescriptor,
    NumericRoundingMode mode);
void mergeFlags(FormalNumericExceptionFlags &destination,
                FormalNumericExceptionFlags source);
llvm::Expected<FormalNumericResult>
finishRawResult(LogicalFormat format, uint64_t bits,
                FormalNumericExceptionFlags flags);
llvm::Expected<RawLogicalValue> validateOperand(RawLogicalValue operand,
                                                LogicalFormat expectedFormat,
                                                const llvm::Twine &role);
llvm::Expected<LogicalValueClassification>
classifyOperand(RawLogicalValue operand, const llvm::Twine &role);

struct ExactDyadic {
  bool negative;
  llvm::APInt magnitude;
  int exponent;
};

std::optional<ExactDyadic> decodeFiniteDyadic(RawLogicalValue value);
ExactDyadic multiplyDyadics(const ExactDyadic &lhs, const ExactDyadic &rhs);
ExactDyadic addDyadics(const ExactDyadic &lhs, const ExactDyadic &rhs);
bool isTinyAfterRNE(const ExactDyadic &exact,
                    const LogicalFormatDescriptor &destinationDescriptor);

llvm::Error
validateElementwiseResolvedCommand(const ResolvedNumericCommand &command);
llvm::Error validateGemmResolvedCommand(const ResolvedNumericCommand &command);

} // namespace wafer::formal_detail

#endif // WAFER_TARGET_FORMALNUMERICINTERNAL_H
