//===- SoftFloatOracle.h - Independent IEEE differential -------*- C++ -*-===//

#ifndef WAFER_TARGET_SOFTFLOATORACLE_H
#define WAFER_TARGET_SOFTFLOATORACLE_H

#include "Wafer/Target/Numeric/Formal/FormalNumeric.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>

namespace wafer {

/// Operations independently evaluated by the managed Berkeley SoftFloat 3e
/// build. This adapter is a conformance oracle for the APFloat production
/// path; it is not a production numeric backend.
enum class SoftFloatOracleOperation : uint8_t {
  Convert,
  Add,
  Subtract,
  Multiply,
  Divide,
  SquareRoot,
  FusedMultiplyAdd,
};

/// SoftFloat exposes tininess detection as ambient library state. Keep the
/// choice explicit even though the first model profile always selects after
/// rounding.
enum class SoftFloatOracleTininess : uint8_t {
  BeforeRounding,
  AfterRounding,
};

/// Fully typed oracle request validated at execution. Only F16 and F32 are
/// accepted. Operand arity and format equality are determined by `operation`;
/// invalid requests fail before any SoftFloat ambient state is modified.
struct SoftFloatOracleRequest {
  SoftFloatOracleOperation operation;
  NumericRoundingMode roundingMode;
  SoftFloatOracleTininess tininess;
  LogicalFormat resultFormat;
  RawLogicalValue lhs;
  std::optional<RawLogicalValue> rhs;
  std::optional<RawLogicalValue> addend;
};

enum class SoftFloatOracleErrorCode : uint8_t {
  UnsupportedRoundingMode,
  UnsupportedFormat,
  InvalidOperandEncoding,
  InvalidOperandArity,
  OperandFormatMismatch,
  InvalidResultEncoding,
};

llvm::StringRef
stringifySoftFloatOracleErrorCode(SoftFloatOracleErrorCode code);

class SoftFloatOracleError final
    : public llvm::ErrorInfo<SoftFloatOracleError> {
public:
  static char ID;

  SoftFloatOracleError(SoftFloatOracleErrorCode code, std::string detail)
      : code(code), detail(std::move(detail)) {}

  SoftFloatOracleErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SoftFloatOracleErrorCode code;
  std::string detail;
};

/// Execute one request under an invocation-local SoftFloat environment. The
/// adapter saves and restores rounding, tininess, extF80 precision and all
/// sticky exception flags on every exit path. The returned flags are only
/// those raised by this operation, including SoftFloat's IEEE
/// unbounded-exponent interpretation of after-rounding tininess.
llvm::Expected<FormalNumericResult>
executeSoftFloatOracle(const SoftFloatOracleRequest &request);

} // namespace wafer

#endif // WAFER_TARGET_SOFTFLOATORACLE_H
