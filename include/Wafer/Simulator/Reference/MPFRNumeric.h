//===- MPFRNumeric.h - Managed transcendental formal backend ----*- C++ -*-===//

#ifndef WAFER_TARGET_MPFRNUMERIC_H
#define WAFER_TARGET_MPFRNUMERIC_H

#include "Wafer/Simulator/Reference/FormalNumeric.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>

namespace wafer {

/// Correctly rounded operations provided by the managed MPFR/GMP trusted
/// semantic TCB. They operate on one explicit logical format and never consult
/// host floating-point state.
enum class MPFRFormalOperation : uint8_t {
  Sqrt,
  Rsqrt,
  Log2,
  Ln,
  Pow2,
  Exp,
  Sin,
  Cos,
  Tanh,
  Sigmoid,
  Softplus,
};

struct MPFRFormalRequest {
  MPFRFormalOperation operation;
  TargetRoundingMode roundingMode;
  LogicalFormat resultFormat;
  RawLogicalValue input;
};

enum class MPFRNumericErrorCode : uint8_t {
  UnsupportedRoundingMode,
  UnsupportedFormat,
  FormatMismatch,
  InvalidInputEncoding,
  EnvironmentConfiguration,
  AdaptiveEvaluation,
  ResultEncoding,
};

llvm::StringRef stringifyMPFRNumericErrorCode(MPFRNumericErrorCode code);

class MPFRNumericError final : public llvm::ErrorInfo<MPFRNumericError> {
public:
  static char ID;

  MPFRNumericError(MPFRNumericErrorCode code, std::string detail)
      : code(code), detail(std::move(detail)) {}

  MPFRNumericErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  MPFRNumericErrorCode code;
  std::string detail;
};

/// Execute one same-format F16/BF16/F32/TF32 component operation. Supported
/// elementwise capability rows use F16/BF16/F32 and nearest-even; TF32 and
/// directed rounding for direct operations remain component-only coverage.
/// Direct MPFR operations are correctly rounded in the requested mode.
/// Sigmoid and softplus accept nearest-even and use adaptive directed intervals
/// until both bounds prove identical destination bits and identical
/// underflow/overflow classification. Their non-special finite results are
/// non-dyadic and therefore inexact; intermediates are never rounded to the
/// destination format.
///
/// The MPFR environment scope saves/restores emin, emax, default precision,
/// default rounding and all flags. Every value still uses explicit precision
/// and every operation passes an explicit rounding mode. Exponent-range
/// transitions temporarily widen both endpoints so caller ranges disjoint from
/// the logical format are restored exactly. IEEE-like gradual underflow for a
/// direct operation passes its original ternary result directly to
/// mpfr_subnormalize and reports tininess after rounding.
llvm::Expected<FormalNumericResult>
executeMPFRFormal(const MPFRFormalRequest &request);

} // namespace wafer

#endif // WAFER_TARGET_MPFRNUMERIC_H
