//===- TargetScalarConversion.h - Deterministic scalar conversion -*- C++
//-*-===//

#ifndef WAFER_TARGET_TARGETSCALARCONVERSION_H
#define WAFER_TARGET_TARGETSCALARCONVERSION_H

#include "Wafer/Target/TargetOperation.h"
#include "Wafer/Target/PhysicalTensor/NumericCodec.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace wafer {

struct TargetScalarConversionFlags {
  bool invalid = false;
  bool divByZero = false;
  bool overflow = false;
  bool underflow = false;
  bool inexact = false;
};

struct TargetScalarConversionResult {
  RawLogicalValue value;
  TargetScalarConversionFlags flags;
};

enum class TargetScalarConversionErrorCode : uint8_t {
  UnsupportedOperation,
  InvalidSourceEncoding,
  SourceFormatMismatch,
  UnsupportedEndpointKinds,
  FloatToIntegerNonFinite,
  FloatToIntegerOutOfRange,
  UnexpectedAPFloatStatus,
  InvalidResultEncoding,
};

llvm::StringRef
stringifyTargetScalarConversionErrorCode(TargetScalarConversionErrorCode code);

class TargetScalarConversionError final
    : public llvm::ErrorInfo<TargetScalarConversionError> {
public:
  static char ID;

  TargetScalarConversionError(TargetScalarConversionErrorCode code,
                              std::string detail)
      : code(code), detail(std::move(detail)) {}

  TargetScalarConversionErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  TargetScalarConversionErrorCode code;
  std::string detail;
};

/// Effect-free conversion for one already selected target convert operation.
/// The caller supplies every required rounding/zero-point parameter. No
/// process-global policy, model profile, or package default participates.
llvm::Expected<TargetScalarConversionResult>
convertTargetScalar(TargetConvertOperation operation,
                    std::optional<TargetConvertParameter> parameter,
                    RawLogicalValue source);

} // namespace wafer

#endif // WAFER_TARGET_TARGETSCALARCONVERSION_H
