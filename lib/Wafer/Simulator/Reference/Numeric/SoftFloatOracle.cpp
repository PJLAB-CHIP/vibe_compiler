//===- SoftFloatOracle.cpp - Independent IEEE differential ---------------===//

#include "Wafer/Simulator/Reference/SoftFloatOracle.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>

extern "C" {
#include <softfloat.h>
}

namespace wafer {
namespace {

llvm::Error oracleError(SoftFloatOracleErrorCode code,
                        const llvm::Twine &detail) {
  return llvm::make_error<SoftFloatOracleError>(code, detail.str());
}

std::optional<uint_fast8_t> toSoftFloatRoundingMode(TargetRoundingMode mode) {
  switch (mode) {
  case TargetRoundingMode::NearestEven:
    return softfloat_round_near_even;
  case TargetRoundingMode::TowardZero:
    return softfloat_round_minMag;
  case TargetRoundingMode::TowardPositive:
    return softfloat_round_max;
  case TargetRoundingMode::TowardNegative:
    return softfloat_round_min;
  case TargetRoundingMode::Stochastic:
    return std::nullopt;
  }
  return std::nullopt;
}

uint_fast8_t toSoftFloatTininess(SoftFloatOracleTininess tininess) {
  switch (tininess) {
  case SoftFloatOracleTininess::BeforeRounding:
    return softfloat_tininess_beforeRounding;
  case SoftFloatOracleTininess::AfterRounding:
    return softfloat_tininess_afterRounding;
  }
  llvm_unreachable("SoftFloat oracle tininess mode is not registered");
}

bool isSupportedFormat(LogicalFormat format) {
  return format == LogicalFormat::F16 || format == LogicalFormat::F32;
}

llvm::Expected<RawLogicalValue> validateOperand(RawLogicalValue value,
                                                const llvm::Twine &label) {
  llvm::Expected<RawLogicalValue> canonical = makeRawLogicalValue(
      value.format, value.bits, NonCanonicalEncodingPolicy::Reject);
  if (!canonical)
    return oracleError(SoftFloatOracleErrorCode::InvalidOperandEncoding,
                       label + ": " + llvm::toString(canonical.takeError()));
  if (!isSupportedFormat(canonical->format))
    return oracleError(SoftFloatOracleErrorCode::UnsupportedFormat,
                       label + " must use F16 or F32");
  return *canonical;
}

struct ValidatedRequest {
  SoftFloatOracleOperation operation;
  uint_fast8_t roundingMode;
  uint_fast8_t tininess;
  LogicalFormat resultFormat;
  RawLogicalValue lhs;
  std::optional<RawLogicalValue> rhs;
  std::optional<RawLogicalValue> addend;
};

llvm::Expected<ValidatedRequest>
validateRequest(const SoftFloatOracleRequest &request) {
  std::optional<uint_fast8_t> roundingMode =
      toSoftFloatRoundingMode(request.roundingMode);
  if (!roundingMode)
    return oracleError(SoftFloatOracleErrorCode::UnsupportedRoundingMode,
                       "stochastic rounding has no deterministic SoftFloat "
                       "oracle policy");
  if (!isSupportedFormat(request.resultFormat))
    return oracleError(SoftFloatOracleErrorCode::UnsupportedFormat,
                       "result format must be F16 or F32");

  llvm::Expected<RawLogicalValue> lhs = validateOperand(request.lhs, "lhs");
  if (!lhs)
    return lhs.takeError();

  std::optional<RawLogicalValue> rhs;
  if (request.rhs) {
    llvm::Expected<RawLogicalValue> validated =
        validateOperand(*request.rhs, "rhs");
    if (!validated)
      return validated.takeError();
    rhs = *validated;
  }
  std::optional<RawLogicalValue> addend;
  if (request.addend) {
    llvm::Expected<RawLogicalValue> validated =
        validateOperand(*request.addend, "addend");
    if (!validated)
      return validated.takeError();
    addend = *validated;
  }

  auto arityError = [&]() -> llvm::Error {
    return oracleError(SoftFloatOracleErrorCode::InvalidOperandArity,
                       "operand presence does not match the selected "
                       "SoftFloat operation");
  };
  switch (request.operation) {
  case SoftFloatOracleOperation::Convert:
  case SoftFloatOracleOperation::SquareRoot:
    if (rhs || addend)
      return arityError();
    break;
  case SoftFloatOracleOperation::Add:
  case SoftFloatOracleOperation::Subtract:
  case SoftFloatOracleOperation::Multiply:
  case SoftFloatOracleOperation::Divide:
    if (!rhs || addend)
      return arityError();
    break;
  case SoftFloatOracleOperation::FusedMultiplyAdd:
    if (!rhs || !addend)
      return arityError();
    break;
  }

  if (request.operation != SoftFloatOracleOperation::Convert &&
      lhs->format != request.resultFormat)
    return oracleError(SoftFloatOracleErrorCode::OperandFormatMismatch,
                       "lhs format must equal the arithmetic result format");
  if (rhs && rhs->format != request.resultFormat)
    return oracleError(SoftFloatOracleErrorCode::OperandFormatMismatch,
                       "rhs format must equal the arithmetic result format");
  if (addend && addend->format != request.resultFormat)
    return oracleError(SoftFloatOracleErrorCode::OperandFormatMismatch,
                       "addend format must equal the arithmetic result format");

  return ValidatedRequest{request.operation,
                          *roundingMode,
                          toSoftFloatTininess(request.tininess),
                          request.resultFormat,
                          *lhs,
                          rhs,
                          addend};
}

class SoftFloatEnvironmentScope {
public:
  SoftFloatEnvironmentScope(uint_fast8_t roundingMode, uint_fast8_t tininess)
      : savedRoundingMode(softfloat_roundingMode),
        savedTininess(softfloat_detectTininess),
        savedExtF80Precision(extF80_roundingPrecision),
        savedExceptionFlags(softfloat_exceptionFlags) {
    softfloat_roundingMode = roundingMode;
    softfloat_detectTininess = tininess;
    extF80_roundingPrecision = 80;
    softfloat_exceptionFlags = 0;
  }

  ~SoftFloatEnvironmentScope() {
    softfloat_roundingMode = savedRoundingMode;
    softfloat_detectTininess = savedTininess;
    extF80_roundingPrecision = savedExtF80Precision;
    softfloat_exceptionFlags = savedExceptionFlags;
  }

  SoftFloatEnvironmentScope(const SoftFloatEnvironmentScope &) = delete;
  SoftFloatEnvironmentScope &
  operator=(const SoftFloatEnvironmentScope &) = delete;

  uint_fast8_t getRaisedFlags() const { return softfloat_exceptionFlags; }

private:
  uint_fast8_t savedRoundingMode;
  uint_fast8_t savedTininess;
  uint_fast8_t savedExtF80Precision;
  uint_fast8_t savedExceptionFlags;
};

FormalNumericExceptionFlags mapFlags(uint_fast8_t flags) {
  return {
      (flags & softfloat_flag_invalid) != 0,
      (flags & softfloat_flag_infinite) != 0,
      (flags & softfloat_flag_overflow) != 0,
      (flags & softfloat_flag_underflow) != 0,
      (flags & softfloat_flag_inexact) != 0,
  };
}

uint16_t bits16(RawLogicalValue value) {
  return static_cast<uint16_t>(value.bits);
}

uint32_t bits32(RawLogicalValue value) {
  return static_cast<uint32_t>(value.bits);
}

uint64_t executeF16(const ValidatedRequest &request) {
  const float16_t lhs{bits16(request.lhs)};
  switch (request.operation) {
  case SoftFloatOracleOperation::Convert:
    if (request.lhs.format == LogicalFormat::F16)
      return lhs.v;
    return f32_to_f16(float32_t{bits32(request.lhs)}).v;
  case SoftFloatOracleOperation::Add:
    return f16_add(lhs, float16_t{bits16(*request.rhs)}).v;
  case SoftFloatOracleOperation::Subtract:
    return f16_sub(lhs, float16_t{bits16(*request.rhs)}).v;
  case SoftFloatOracleOperation::Multiply:
    return f16_mul(lhs, float16_t{bits16(*request.rhs)}).v;
  case SoftFloatOracleOperation::Divide:
    return f16_div(lhs, float16_t{bits16(*request.rhs)}).v;
  case SoftFloatOracleOperation::SquareRoot:
    return f16_sqrt(lhs).v;
  case SoftFloatOracleOperation::FusedMultiplyAdd:
    return f16_mulAdd(lhs, float16_t{bits16(*request.rhs)},
                      float16_t{bits16(*request.addend)})
        .v;
  }
  llvm_unreachable("SoftFloat F16 oracle operation is not registered");
}

uint64_t executeF32(const ValidatedRequest &request) {
  const float32_t lhs{bits32(request.lhs)};
  switch (request.operation) {
  case SoftFloatOracleOperation::Convert:
    if (request.lhs.format == LogicalFormat::F32)
      return lhs.v;
    return f16_to_f32(float16_t{bits16(request.lhs)}).v;
  case SoftFloatOracleOperation::Add:
    return f32_add(lhs, float32_t{bits32(*request.rhs)}).v;
  case SoftFloatOracleOperation::Subtract:
    return f32_sub(lhs, float32_t{bits32(*request.rhs)}).v;
  case SoftFloatOracleOperation::Multiply:
    return f32_mul(lhs, float32_t{bits32(*request.rhs)}).v;
  case SoftFloatOracleOperation::Divide:
    return f32_div(lhs, float32_t{bits32(*request.rhs)}).v;
  case SoftFloatOracleOperation::SquareRoot:
    return f32_sqrt(lhs).v;
  case SoftFloatOracleOperation::FusedMultiplyAdd:
    return f32_mulAdd(lhs, float32_t{bits32(*request.rhs)},
                      float32_t{bits32(*request.addend)})
        .v;
  }
  llvm_unreachable("SoftFloat F32 oracle operation is not registered");
}

} // namespace

llvm::StringRef
stringifySoftFloatOracleErrorCode(SoftFloatOracleErrorCode code) {
  switch (code) {
  case SoftFloatOracleErrorCode::UnsupportedRoundingMode:
    return "unsupported-rounding-mode";
  case SoftFloatOracleErrorCode::UnsupportedFormat:
    return "unsupported-format";
  case SoftFloatOracleErrorCode::InvalidOperandEncoding:
    return "invalid-operand-encoding";
  case SoftFloatOracleErrorCode::InvalidOperandArity:
    return "invalid-operand-arity";
  case SoftFloatOracleErrorCode::OperandFormatMismatch:
    return "operand-format-mismatch";
  case SoftFloatOracleErrorCode::InvalidResultEncoding:
    return "invalid-result-encoding";
  }
  llvm_unreachable("SoftFloat oracle error code is not registered");
}

char SoftFloatOracleError::ID;

void SoftFloatOracleError::log(llvm::raw_ostream &stream) const {
  stream << "SoftFloat oracle " << stringifySoftFloatOracleErrorCode(code)
         << ": " << detail;
}

std::error_code SoftFloatOracleError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<FormalNumericResult>
executeSoftFloatOracle(const SoftFloatOracleRequest &request) {
  llvm::Expected<ValidatedRequest> validated = validateRequest(request);
  if (!validated)
    return validated.takeError();

  uint64_t bits = 0;
  FormalNumericExceptionFlags flags;
  {
    SoftFloatEnvironmentScope scope(validated->roundingMode,
                                    validated->tininess);
    bits = validated->resultFormat == LogicalFormat::F16
               ? executeF16(*validated)
               : executeF32(*validated);
    flags = mapFlags(scope.getRaisedFlags());
  }

  llvm::Expected<RawLogicalValue> value = makeRawLogicalValue(
      validated->resultFormat, bits, NonCanonicalEncodingPolicy::Reject);
  if (!value)
    return oracleError(SoftFloatOracleErrorCode::InvalidResultEncoding,
                       llvm::toString(value.takeError()));
  return FormalNumericResult{*value, flags};
}

} // namespace wafer
