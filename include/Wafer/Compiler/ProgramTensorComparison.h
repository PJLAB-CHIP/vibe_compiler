//===- ProgramTensorComparison.h - Expected output comparison -*- C++ -*-===//

#ifndef WAFER_COMPILER_PROGRAMTENSORCOMPARISON_H
#define WAFER_COMPILER_PROGRAMTENSORCOMPARISON_H

#include "Wafer/Compiler/ProgramInvocation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace wafer::compiler {

/// Stable failure classes for source-expected ProgramTensor comparison.
enum class ProgramTensorComparisonErrorCode : uint8_t {
  InvalidTolerance,
  DTypeMismatch,
  ShapeMismatch,
  ByteSizeMismatch,
  UnsupportedFloatingDType,
  ActualNonFinite,
  ExpectedNonFinite,
  NumericMismatch,
  RawMismatch,
};

llvm::StringRef stringifyProgramTensorComparisonErrorCode(
    ProgramTensorComparisonErrorCode code);

class ProgramTensorComparisonError final
    : public llvm::ErrorInfo<ProgramTensorComparisonError> {
public:
  static char ID;

  ProgramTensorComparisonError(ProgramTensorComparisonErrorCode code,
                               std::string detail)
      : code(code), detail(std::move(detail)) {}

  ProgramTensorComparisonErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ProgramTensorComparisonErrorCode code;
  std::string detail;
};

/// Compares two compact row-major program tensors at the source/model output
/// boundary. F16, BF16, and F32 are decoded from their little-endian IEEE
/// storage and compared elementwise using
/// `abs(actual - expected) <= atol + rtol * abs(expected)`. Any non-finite
/// floating value is rejected. Integer and other non-floating storage is
/// compared byte-for-byte. Floating formats without a published tolerance
/// policy fail closed.
llvm::Error compareProgramTensorExpectedOutput(const ProgramTensor &actual,
                                               const ProgramTensor &expected,
                                               double atol, double rtol);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_PROGRAMTENSORCOMPARISON_H
