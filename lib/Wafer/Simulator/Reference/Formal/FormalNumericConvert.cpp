//===- FormalNumericConvert.cpp - Formal scalar conversion ------------===//

#include "FormalNumericInternal.h"

#include "Wafer/Target/PhysicalTensor/TargetScalarConversion.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>

namespace wafer {
namespace {

FormalNumericErrorCode
getFormalErrorCode(TargetScalarConversionErrorCode code) {
  switch (code) {
  case TargetScalarConversionErrorCode::UnsupportedOperation:
    return FormalNumericErrorCode::UnsupportedOperation;
  case TargetScalarConversionErrorCode::InvalidSourceEncoding:
    return FormalNumericErrorCode::InvalidSourceEncoding;
  case TargetScalarConversionErrorCode::SourceFormatMismatch:
    return FormalNumericErrorCode::SourceFormatMismatch;
  case TargetScalarConversionErrorCode::UnsupportedEndpointKinds:
    return FormalNumericErrorCode::UnsupportedEndpointKinds;
  case TargetScalarConversionErrorCode::FloatToIntegerNonFinite:
    return FormalNumericErrorCode::FloatToIntegerNonFinite;
  case TargetScalarConversionErrorCode::FloatToIntegerOutOfRange:
    return FormalNumericErrorCode::FloatToIntegerOutOfRange;
  case TargetScalarConversionErrorCode::UnexpectedAPFloatStatus:
    return FormalNumericErrorCode::UnexpectedAPFloatStatus;
  case TargetScalarConversionErrorCode::InvalidResultEncoding:
    return FormalNumericErrorCode::InvalidResultEncoding;
  }
  llvm_unreachable("unknown target scalar conversion error code");
}

FormalNumericExceptionFlags getFormalFlags(TargetScalarConversionFlags flags) {
  return {flags.invalid, flags.divByZero, flags.overflow, flags.underflow,
          flags.inexact};
}

} // namespace

llvm::Expected<FormalNumericResult>
evaluateFormalConvert(const FormalConvertOperation &operation,
                      RawLogicalValue source) {
  llvm::Expected<TargetConvertOperation> targetOperation =
      TargetConvertOperation::create(operation.opcode);
  if (!targetOperation)
    return formal_detail::formalError(
        FormalNumericErrorCode::UnsupportedOperation,
        llvm::toString(targetOperation.takeError()));
  llvm::Expected<TargetScalarConversionResult> converted =
      convertTargetScalar(*targetOperation, operation.parameter, source);
  if (!converted) {
    std::optional<FormalNumericErrorCode> code;
    std::string detail;
    llvm::Error remaining = llvm::handleErrors(
        converted.takeError(), [&](const TargetScalarConversionError &error) {
          code = getFormalErrorCode(error.getCode());
          detail = error.getDetail().str();
        });
    if (remaining)
      return formal_detail::formalError(
          FormalNumericErrorCode::UnexpectedAPFloatStatus,
          llvm::toString(std::move(remaining)));
    if (!code)
      return formal_detail::formalError(
          FormalNumericErrorCode::UnexpectedAPFloatStatus,
          "target scalar conversion returned an unclassified error");
    return formal_detail::formalError(*code, detail);
  }
  return FormalNumericResult{converted->value,
                             getFormalFlags(converted->flags)};
}

llvm::Expected<FormalNumericResult>
executeFormalConvert(FormalNumericExecutionContext &context,
                     const FormalConvertOperation &operation,
                     RawLogicalValue source) {
  llvm::Expected<FormalNumericResult> result =
      evaluateFormalConvert(operation, source);
  if (!result)
    return result.takeError();
  context.mergeExceptionFlags(result->flags);
  return std::move(*result);
}

} // namespace wafer
