//===- FormalNumericInternal.h - Formal numeric collaboration -*- C++ -*-===//

#ifndef WAFER_TARGET_FORMALNUMERICINTERNAL_H
#define WAFER_TARGET_FORMALNUMERICINTERNAL_H

#include "Wafer/Simulator/Reference/FormalNumeric.h"
#include "Wafer/Target/PhysicalTensor/TargetFloatArithmetic.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>

namespace wafer::formal_detail {

llvm::Error formalError(FormalNumericErrorCode code, const llvm::Twine &detail);

FormalNumericExceptionFlags flagsFromStatus(llvm::APFloat::opStatus status);
using target_numeric_detail::canonicalPositiveQuietNaNBits;
using target_numeric_detail::decodeFloat;
using target_numeric_detail::encodeFloat;
using target_numeric_detail::getFloatSemantics;
using target_numeric_detail::isNaNClass;
using target_numeric_detail::isTinyAfterUnboundedPrecisionRounding;
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
validateFormalElementwiseOperation(const FormalElementwiseOperation &operation);
llvm::Error validateFormalGemmOperation(const FormalGemmOperation &operation);
llvm::Error
validateFormalReduceOperation(const FormalReduceOperation &operation);

} // namespace wafer::formal_detail

#endif // WAFER_TARGET_FORMALNUMERICINTERNAL_H
