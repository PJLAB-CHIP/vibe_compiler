//===- MPFRNumeric.cpp - Managed transcendental formal backend ------------===//

#include "Wafer/Simulator/Reference/MPFRNumeric.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include <mpfr.h>

namespace wafer {
namespace {

llvm::Error numericError(MPFRNumericErrorCode code, const llvm::Twine &detail) {
  return llvm::make_error<MPFRNumericError>(code, detail.str());
}

bool isSupportedFormat(LogicalFormat format) {
  return format == LogicalFormat::F16 || format == LogicalFormat::BF16 ||
         format == LogicalFormat::F32 || format == LogicalFormat::TF32;
}

bool isAdaptiveComposite(MPFRFormalOperation operation) {
  return operation == MPFRFormalOperation::Sigmoid ||
         operation == MPFRFormalOperation::Softplus;
}

std::optional<mpfr_rnd_t> toMPFRRoundingMode(TargetRoundingMode mode) {
  switch (mode) {
  case TargetRoundingMode::NearestEven:
    return MPFR_RNDN;
  case TargetRoundingMode::TowardZero:
    return MPFR_RNDZ;
  case TargetRoundingMode::TowardPositive:
    return MPFR_RNDU;
  case TargetRoundingMode::TowardNegative:
    return MPFR_RNDD;
  case TargetRoundingMode::Stochastic:
    return std::nullopt;
  }
  return std::nullopt;
}

struct BinaryFormatParameters {
  const LogicalFormatDescriptor &descriptor;
  unsigned fractionBits;
  unsigned semanticShift;
  int64_t bias;
  mpfr_exp_t emin;
  mpfr_exp_t emax;
};

BinaryFormatParameters
getParameters(const LogicalFormatDescriptor &descriptor) {
  const unsigned fractionBits = descriptor.precisionBits - 1;
  const unsigned semanticShift =
      descriptor.storageBits - descriptor.semanticBits;
  const int64_t bias =
      (INT64_C(1) << (descriptor.exponentBits - 1)) - INT64_C(1);
  // MPFR values are m*2^e with 1/2 <= |m| < 1. IEEE-like subnormals
  // therefore use emin = (smallest-subnormal unbiased exponent) + 1.
  const mpfr_exp_t emin =
      static_cast<mpfr_exp_t>(INT64_C(3) - bias - descriptor.precisionBits);
  const mpfr_exp_t emax = static_cast<mpfr_exp_t>(bias + INT64_C(1));
  return {descriptor, fractionBits, semanticShift, bias, emin, emax};
}

class MPFREnvironmentScope {
public:
  static llvm::Expected<MPFREnvironmentScope>
  create(const BinaryFormatParameters &parameters, mpfr_rnd_t roundingMode) {
    MPFREnvironmentScope scope;
    if (llvm::Error error = scope.setTargetRange(parameters))
      return std::move(error);
    mpfr_set_default_prec(parameters.descriptor.precisionBits);
    mpfr_set_default_rounding_mode(roundingMode);
    mpfr_clear_flags();
    return std::move(scope);
  }

  MPFREnvironmentScope(MPFREnvironmentScope &&other) noexcept
      : savedEmin(other.savedEmin), savedEmax(other.savedEmax),
        savedPrecision(other.savedPrecision),
        savedRounding(other.savedRounding), savedFlags(other.savedFlags),
        changedEmin(other.changedEmin), changedEmax(other.changedEmax),
        active(other.active) {
    other.active = false;
  }

  MPFREnvironmentScope(const MPFREnvironmentScope &) = delete;
  MPFREnvironmentScope &operator=(const MPFREnvironmentScope &) = delete;
  MPFREnvironmentScope &operator=(MPFREnvironmentScope &&) = delete;

  ~MPFREnvironmentScope() {
    if (!active)
      return;
    // The saved values came from this MPFR build and are necessarily valid.
    if (changedEmin || changedEmax) {
      int status = mpfr_set_emin(mpfr_get_emin_min());
      assert(status == 0);
      (void)status;
      status = mpfr_set_emax(mpfr_get_emax_max());
      assert(status == 0);
      (void)status;
      status = mpfr_set_emin(savedEmin);
      assert(status == 0);
      (void)status;
      status = mpfr_set_emax(savedEmax);
      assert(status == 0);
      (void)status;
    }
    mpfr_set_default_prec(savedPrecision);
    mpfr_set_default_rounding_mode(savedRounding);
    mpfr_flags_restore(savedFlags, MPFR_FLAGS_ALL);
  }

  mpfr_flags_t getRaisedFlags() const { return mpfr_flags_save(); }

  llvm::Error setTargetRange(const BinaryFormatParameters &parameters) {
    return setExponentRange(parameters.emin, parameters.emax, "logical-format");
  }

  llvm::Error setWideRange() {
    return setExponentRange(mpfr_get_emin_min(), mpfr_get_emax_max(),
                            "adaptive-wide");
  }

private:
  MPFREnvironmentScope()
      : savedEmin(mpfr_get_emin()), savedEmax(mpfr_get_emax()),
        savedPrecision(mpfr_get_default_prec()),
        savedRounding(mpfr_get_default_rounding_mode()),
        savedFlags(mpfr_flags_save()) {}

  llvm::Error setExponentRange(mpfr_exp_t emin, mpfr_exp_t emax,
                               const llvm::Twine &label) {
    // A caller may have installed a valid range wholly above or below the
    // requested range. Widen first so neither endpoint update is rejected
    // merely because the other endpoint still belongs to the previous range.
    if (mpfr_set_emin(mpfr_get_emin_min()) != 0)
      return numericError(MPFRNumericErrorCode::EnvironmentConfiguration,
                          label + " range rejected the temporary minimum emin");
    changedEmin = true;
    if (mpfr_set_emax(mpfr_get_emax_max()) != 0)
      return numericError(MPFRNumericErrorCode::EnvironmentConfiguration,
                          label + " range rejected the temporary maximum emax");
    changedEmax = true;
    if (mpfr_set_emin(emin) != 0)
      return numericError(MPFRNumericErrorCode::EnvironmentConfiguration,
                          label + " range rejected emin");
    if (mpfr_set_emax(emax) != 0)
      return numericError(MPFRNumericErrorCode::EnvironmentConfiguration,
                          label + " range rejected emax");
    return llvm::Error::success();
  }

  mpfr_exp_t savedEmin;
  mpfr_exp_t savedEmax;
  mpfr_prec_t savedPrecision;
  mpfr_rnd_t savedRounding;
  mpfr_flags_t savedFlags;
  bool changedEmin = false;
  bool changedEmax = false;
  bool active = true;
};

class MPFRValue {
public:
  explicit MPFRValue(mpfr_prec_t precision) { mpfr_init2(value, precision); }
  ~MPFRValue() { mpfr_clear(value); }

  MPFRValue(const MPFRValue &) = delete;
  MPFRValue &operator=(const MPFRValue &) = delete;

  mpfr_ptr get() { return value; }
  mpfr_srcptr get() const { return value; }

private:
  mpfr_t value;
};

class GMPInteger {
public:
  GMPInteger() { mpz_init(value); }
  ~GMPInteger() { mpz_clear(value); }

  GMPInteger(const GMPInteger &) = delete;
  GMPInteger &operator=(const GMPInteger &) = delete;

  mpz_ptr get() { return value; }
  mpz_srcptr get() const { return value; }

private:
  mpz_t value;
};

uint64_t
canonicalPositiveQuietNaNBits(const BinaryFormatParameters &parameters) {
  const uint64_t exponent =
      (UINT64_C(1) << parameters.descriptor.exponentBits) - UINT64_C(1);
  const uint64_t quiet =
      UINT64_C(1) << (parameters.semanticShift + parameters.fractionBits - 1);
  return (exponent << (parameters.semanticShift + parameters.fractionBits)) |
         quiet;
}

uint64_t infinityBits(const BinaryFormatParameters &parameters, bool negative) {
  const uint64_t exponent =
      (UINT64_C(1) << parameters.descriptor.exponentBits) - UINT64_C(1);
  const uint64_t sign =
      negative ? UINT64_C(1) << (parameters.descriptor.storageBits - 1)
               : UINT64_C(0);
  return sign |
         (exponent << (parameters.semanticShift + parameters.fractionBits));
}

uint64_t positivePowerOfTwoBits(const BinaryFormatParameters &parameters,
                                int64_t unbiasedExponent) {
  const uint64_t exponent =
      static_cast<uint64_t>(unbiasedExponent + parameters.bias);
  return exponent << (parameters.semanticShift + parameters.fractionBits);
}

void decodeFinite(const BinaryFormatParameters &parameters,
                  RawLogicalValue input, mpfr_ptr result) {
  const uint64_t sign = input.bits >> (parameters.descriptor.storageBits - 1);
  const uint64_t fractionMask =
      (UINT64_C(1) << parameters.fractionBits) - UINT64_C(1);
  const uint64_t fraction =
      (input.bits >> parameters.semanticShift) & fractionMask;
  const uint64_t exponentMask =
      (UINT64_C(1) << parameters.descriptor.exponentBits) - UINT64_C(1);
  const uint64_t exponent =
      (input.bits >> (parameters.semanticShift + parameters.fractionBits)) &
      exponentMask;

  if (exponent == 0 && fraction == 0) {
    mpfr_set_zero(result, sign ? -1 : 1);
    return;
  }

  uint64_t significand = fraction;
  int64_t binaryExponent =
      INT64_C(2) - parameters.bias - parameters.descriptor.precisionBits;
  if (exponent != 0) {
    significand |= UINT64_C(1) << parameters.fractionBits;
    binaryExponent = static_cast<int64_t>(exponent) - parameters.bias -
                     parameters.fractionBits;
  }
  const int status =
      mpfr_set_uj_2exp(result, static_cast<uintmax_t>(significand),
                       static_cast<mpfr_exp_t>(binaryExponent), MPFR_RNDN);
  assert(status == 0 && "raw logical input must be exactly representable");
  if (sign)
    mpfr_neg(result, result, MPFR_RNDN);
}

void shiftIntegerToExponent(mpz_ptr integer, mpfr_exp_t sourceExponent,
                            int64_t destinationExponent) {
  const int64_t shift =
      static_cast<int64_t>(sourceExponent) - destinationExponent;
  if (shift >= 0)
    mpz_mul_2exp(integer, integer, static_cast<mp_bitcnt_t>(shift));
  else
    mpz_tdiv_q_2exp(integer, integer, static_cast<mp_bitcnt_t>(-shift));
}

llvm::Expected<uint64_t> encodeResult(const BinaryFormatParameters &parameters,
                                      mpfr_srcptr result) {
  if (mpfr_nan_p(result))
    return canonicalPositiveQuietNaNBits(parameters);
  if (mpfr_inf_p(result))
    return infinityBits(parameters, mpfr_signbit(result) != 0);

  const bool negative = mpfr_signbit(result) != 0;
  const uint64_t sign =
      negative ? UINT64_C(1) << (parameters.descriptor.storageBits - 1)
               : UINT64_C(0);
  if (mpfr_zero_p(result))
    return sign;

  GMPInteger integer;
  const mpfr_exp_t integerExponent = mpfr_get_z_2exp(integer.get(), result);
  mpz_abs(integer.get(), integer.get());
  const mpfr_exp_t resultExponent = mpfr_get_exp(result);
  const int64_t minimumNormalExponent = INT64_C(2) - parameters.bias;

  uint64_t exponentField = 0;
  uint64_t fractionField = 0;
  if (static_cast<int64_t>(resultExponent) >= minimumNormalExponent) {
    const int64_t significandExponent = static_cast<int64_t>(resultExponent) -
                                        parameters.descriptor.precisionBits;
    shiftIntegerToExponent(integer.get(), integerExponent, significandExponent);
    if (!mpz_fits_ulong_p(integer.get()))
      return numericError(MPFRNumericErrorCode::ResultEncoding,
                          "normal result significand exceeds its format");
    const uint64_t significand = mpz_get_ui(integer.get());
    const uint64_t implicit = UINT64_C(1) << parameters.fractionBits;
    if (significand < implicit || significand >= (implicit << 1))
      return numericError(MPFRNumericErrorCode::ResultEncoding,
                          "normal result has a malformed significand");
    exponentField = static_cast<uint64_t>(static_cast<int64_t>(resultExponent) -
                                          INT64_C(1) + parameters.bias);
    fractionField = significand - implicit;
  } else {
    const int64_t subnormalUnitExponent =
        INT64_C(2) - parameters.bias - parameters.descriptor.precisionBits;
    shiftIntegerToExponent(integer.get(), integerExponent,
                           subnormalUnitExponent);
    if (!mpz_fits_ulong_p(integer.get()))
      return numericError(MPFRNumericErrorCode::ResultEncoding,
                          "subnormal result fraction exceeds its format");
    fractionField = mpz_get_ui(integer.get());
    if (fractionField == 0 ||
        fractionField >= (UINT64_C(1) << parameters.fractionBits))
      return numericError(MPFRNumericErrorCode::ResultEncoding,
                          "subnormal result has a malformed fraction");
  }

  const uint64_t bits =
      sign |
      (exponentField << (parameters.semanticShift + parameters.fractionBits)) |
      (fractionField << parameters.semanticShift);
  llvm::Expected<RawLogicalValue> canonical = makeRawLogicalValue(
      parameters.descriptor.format, bits, NonCanonicalEncodingPolicy::Reject);
  if (!canonical)
    return numericError(MPFRNumericErrorCode::ResultEncoding,
                        llvm::toString(canonical.takeError()));
  return canonical->bits;
}

FormalNumericExceptionFlags mapFlags(mpfr_flags_t flags, bool invalid) {
  const bool inexact = (flags & MPFR_FLAGS_INEXACT) != 0;
  // MPFR may mark an exact subnormal range transition as underflow. The model
  // follows IEEE tininess-after: underflow is raised only when the rounded
  // result is both tiny after rounding and inexact.
  return {
      invalid,
      (flags & MPFR_FLAGS_DIVBY0) != 0,
      (flags & MPFR_FLAGS_OVERFLOW) != 0,
      (flags & MPFR_FLAGS_UNDERFLOW) != 0 && inexact,
      inexact,
  };
}

struct RoundedIntervalEndpoint {
  uint64_t bits;
  FormalNumericExceptionFlags flags;
};

llvm::Expected<RoundedIntervalEndpoint>
roundIntervalEndpoint(const BinaryFormatParameters &parameters,
                      mpfr_srcptr endpoint, mpfr_rnd_t roundingMode) {
  MPFRValue rounded(parameters.descriptor.precisionBits);
  mpfr_clear_flags();
  int ternary = mpfr_set(rounded.get(), endpoint, roundingMode);
  ternary = mpfr_check_range(rounded.get(), ternary, roundingMode);
  ternary = mpfr_subnormalize(rounded.get(), ternary, roundingMode);
  (void)ternary;
  const FormalNumericExceptionFlags flags =
      mapFlags(mpfr_flags_save(), /*invalid=*/false);
  llvm::Expected<uint64_t> bits = encodeResult(parameters, rounded.get());
  if (!bits)
    return bits.takeError();
  return RoundedIntervalEndpoint{*bits, flags};
}

void computeSigmoidInterval(mpfr_srcptr source, mpfr_prec_t precision,
                            mpfr_ptr lower, mpfr_ptr upper) {
  MPFRValue argument(precision);
  MPFRValue exponentialLower(precision);
  MPFRValue exponentialUpper(precision);
  MPFRValue denominatorLower(precision);
  MPFRValue denominatorUpper(precision);

  if (mpfr_sgn(source) >= 0) {
    // 1 / (1 + exp(-x)); reciprocal reverses the denominator interval.
    mpfr_neg(argument.get(), source, MPFR_RNDN);
    mpfr_exp(exponentialLower.get(), argument.get(), MPFR_RNDD);
    mpfr_exp(exponentialUpper.get(), argument.get(), MPFR_RNDU);
    mpfr_add_ui(denominatorLower.get(), exponentialLower.get(), 1, MPFR_RNDD);
    mpfr_add_ui(denominatorUpper.get(), exponentialUpper.get(), 1, MPFR_RNDU);
    mpfr_ui_div(lower, 1, denominatorUpper.get(), MPFR_RNDD);
    mpfr_ui_div(upper, 1, denominatorLower.get(), MPFR_RNDU);
    return;
  }

  // exp(x) / (1 + exp(x)); this form avoids an overflowing exp(-x). For the
  // lower bound round the matching denominator upward, and conversely for the
  // upper bound.
  mpfr_exp(exponentialLower.get(), source, MPFR_RNDD);
  mpfr_exp(exponentialUpper.get(), source, MPFR_RNDU);
  mpfr_add_ui(denominatorUpper.get(), exponentialLower.get(), 1, MPFR_RNDU);
  mpfr_add_ui(denominatorLower.get(), exponentialUpper.get(), 1, MPFR_RNDD);
  mpfr_div(lower, exponentialLower.get(), denominatorUpper.get(), MPFR_RNDD);
  mpfr_div(upper, exponentialUpper.get(), denominatorLower.get(), MPFR_RNDU);
}

void computeSoftplusInterval(mpfr_srcptr source, mpfr_prec_t precision,
                             mpfr_ptr lower, mpfr_ptr upper) {
  MPFRValue argument(precision);
  MPFRValue exponentialLower(precision);
  MPFRValue exponentialUpper(precision);
  MPFRValue correctionLower(precision);
  MPFRValue correctionUpper(precision);

  const bool positive = mpfr_sgn(source) > 0;
  if (positive)
    mpfr_neg(argument.get(), source, MPFR_RNDN);
  else
    mpfr_set(argument.get(), source, MPFR_RNDN);
  mpfr_exp(exponentialLower.get(), argument.get(), MPFR_RNDD);
  mpfr_exp(exponentialUpper.get(), argument.get(), MPFR_RNDU);
  mpfr_log1p(correctionLower.get(), exponentialLower.get(), MPFR_RNDD);
  mpfr_log1p(correctionUpper.get(), exponentialUpper.get(), MPFR_RNDU);

  if (positive) {
    // x + log1p(exp(-x)) avoids an overflowing exp(x).
    mpfr_add(lower, source, correctionLower.get(), MPFR_RNDD);
    mpfr_add(upper, source, correctionUpper.get(), MPFR_RNDU);
    return;
  }
  mpfr_set(lower, correctionLower.get(), MPFR_RNDD);
  mpfr_set(upper, correctionUpper.get(), MPFR_RNDU);
}

llvm::Expected<FormalNumericResult> evaluateAdaptiveComposite(
    const MPFRFormalRequest &request, const BinaryFormatParameters &parameters,
    RawLogicalValue input, mpfr_srcptr source,
    MPFREnvironmentScope &environment, mpfr_rnd_t roundingMode) {
  assert(roundingMode == MPFR_RNDN &&
         "adaptive composite evaluation requires final nearest-even");

  // The supported formats have precision <= 24. For x >= 32, both
  // 1-sigmoid(x) and softplus(x)-x are less than exp(-x) < 2^-32, safely below
  // half an ulp of their rounded result. For x <= -1024, both positive
  // results are below exp(x) < 2^-1024, safely below half the F32 minimum
  // subnormal (and therefore below every supported format's threshold). These
  // proofs also avoid relying on MPFR's finite exponent ceiling for enormous
  // finite F16/BF16/F32 inputs.
  if (mpfr_cmp_si(source, 32) >= 0) {
    const uint64_t bits = request.operation == MPFRFormalOperation::Sigmoid
                              ? positivePowerOfTwoBits(parameters, 0)
                              : input.bits;
    return FormalNumericResult{{request.resultFormat, bits},
                               {/*invalid=*/false,
                                /*divByZero=*/false,
                                /*overflow=*/false,
                                /*underflow=*/false,
                                /*inexact=*/true}};
  }
  if (mpfr_cmp_si(source, -1024) <= 0)
    return FormalNumericResult{{request.resultFormat, UINT64_C(0)},
                               {/*invalid=*/false,
                                /*divByZero=*/false,
                                /*overflow=*/false,
                                /*underflow=*/true,
                                /*inexact=*/true}};

  mpfr_prec_t workingPrecision = parameters.descriptor.precisionBits;
  for (;;) {
    if (llvm::Error error = environment.setWideRange())
      return std::move(error);
    MPFRValue lower(workingPrecision);
    MPFRValue upper(workingPrecision);
    if (request.operation == MPFRFormalOperation::Sigmoid)
      computeSigmoidInterval(source, workingPrecision, lower.get(),
                             upper.get());
    else
      computeSoftplusInterval(source, workingPrecision, lower.get(),
                              upper.get());

    if (llvm::Error error = environment.setTargetRange(parameters))
      return std::move(error);
    llvm::Expected<RoundedIntervalEndpoint> roundedLower =
        roundIntervalEndpoint(parameters, lower.get(), roundingMode);
    if (!roundedLower)
      return roundedLower.takeError();
    llvm::Expected<RoundedIntervalEndpoint> roundedUpper =
        roundIntervalEndpoint(parameters, upper.get(), roundingMode);
    if (!roundedUpper)
      return roundedUpper.takeError();

    if (roundedLower->bits == roundedUpper->bits &&
        roundedLower->flags.overflow == roundedUpper->flags.overflow &&
        roundedLower->flags.underflow == roundedUpper->flags.underflow) {
      if (roundedLower->flags.invalid || roundedLower->flags.divByZero ||
          roundedUpper->flags.invalid || roundedUpper->flags.divByZero)
        return numericError(MPFRNumericErrorCode::AdaptiveEvaluation,
                            "adaptive composite produced an impossible domain "
                            "or pole flag");
      return FormalNumericResult{{request.resultFormat, roundedLower->bits},
                                 {/*invalid=*/false,
                                  /*divByZero=*/false,
                                  /*overflow=*/roundedLower->flags.overflow,
                                  /*underflow=*/roundedLower->flags.underflow,
                                  /*inexact=*/true}};
    }

    // Directed MPFR enclosures converge to the exact mathematical value. All
    // non-special finite binary inputs produce a non-dyadic sigmoid/softplus
    // result, so the exact value cannot remain on a destination midpoint or
    // flag threshold. Doubling is adaptive, not a fixed guard-bit policy.
    if (workingPrecision > MPFR_PREC_MAX / 2)
      return numericError(MPFRNumericErrorCode::AdaptiveEvaluation,
                          "adaptive interval exhausted MPFR precision");
    workingPrecision *= 2;
  }
}

} // namespace

llvm::StringRef stringifyMPFRNumericErrorCode(MPFRNumericErrorCode code) {
  switch (code) {
  case MPFRNumericErrorCode::UnsupportedRoundingMode:
    return "unsupported-rounding-mode";
  case MPFRNumericErrorCode::UnsupportedFormat:
    return "unsupported-format";
  case MPFRNumericErrorCode::FormatMismatch:
    return "format-mismatch";
  case MPFRNumericErrorCode::InvalidInputEncoding:
    return "invalid-input-encoding";
  case MPFRNumericErrorCode::EnvironmentConfiguration:
    return "environment-configuration";
  case MPFRNumericErrorCode::AdaptiveEvaluation:
    return "adaptive-evaluation";
  case MPFRNumericErrorCode::ResultEncoding:
    return "result-encoding";
  }
  llvm_unreachable("MPFR numeric error code is not registered");
}

char MPFRNumericError::ID;

void MPFRNumericError::log(llvm::raw_ostream &stream) const {
  stream << "MPFR numeric " << stringifyMPFRNumericErrorCode(code) << ": "
         << detail;
}

std::error_code MPFRNumericError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<FormalNumericResult>
executeMPFRFormal(const MPFRFormalRequest &request) {
  if (!isSupportedFormat(request.resultFormat))
    return numericError(MPFRNumericErrorCode::UnsupportedFormat,
                        "result format must be F16, BF16, F32 or TF32");
  if (request.input.format != request.resultFormat)
    return numericError(MPFRNumericErrorCode::FormatMismatch,
                        "the initial MPFR component profile requires "
                        "same-format input and result");
  llvm::Expected<RawLogicalValue> input =
      makeRawLogicalValue(request.input.format, request.input.bits,
                          NonCanonicalEncodingPolicy::Reject);
  if (!input)
    return numericError(MPFRNumericErrorCode::InvalidInputEncoding,
                        llvm::toString(input.takeError()));
  std::optional<mpfr_rnd_t> roundingMode =
      toMPFRRoundingMode(request.roundingMode);
  if (!roundingMode)
    return numericError(MPFRNumericErrorCode::UnsupportedRoundingMode,
                        "stochastic rounding has no deterministic MPFR "
                        "formal policy");
  if (isAdaptiveComposite(request.operation) && *roundingMode != MPFR_RNDN)
    return numericError(
        MPFRNumericErrorCode::UnsupportedRoundingMode,
        "sigmoid and softplus belong to the adaptive final-nearest-even "
        "component policy");

  const LogicalFormatDescriptor &descriptor =
      *findLogicalFormatDescriptor(request.resultFormat);
  const BinaryFormatParameters parameters = getParameters(descriptor);
  llvm::Expected<LogicalValueClassification> classification =
      classifyRawLogicalValue(*input, NonCanonicalEncodingPolicy::Reject);
  if (!classification)
    return numericError(MPFRNumericErrorCode::InvalidInputEncoding,
                        llvm::toString(classification.takeError()));

  if (classification->valueClass == LogicalValueClass::QuietNaN ||
      classification->valueClass == LogicalValueClass::SignalingNaN) {
    const bool invalid =
        classification->valueClass == LogicalValueClass::SignalingNaN;
    return FormalNumericResult{
        {request.resultFormat, canonicalPositiveQuietNaNBits(parameters)},
        {/*invalid=*/invalid}};
  }

  if (isAdaptiveComposite(request.operation) &&
      classification->valueClass == LogicalValueClass::Infinity) {
    uint64_t bits = UINT64_C(0);
    if (!classification->negative)
      bits = request.operation == MPFRFormalOperation::Sigmoid
                 ? positivePowerOfTwoBits(parameters, 0)
                 : infinityBits(parameters, /*negative=*/false);
    return FormalNumericResult{{request.resultFormat, bits}, {}};
  }
  if (request.operation == MPFRFormalOperation::Sigmoid &&
      classification->valueClass == LogicalValueClass::Zero)
    return FormalNumericResult{
        {request.resultFormat, positivePowerOfTwoBits(parameters, -1)}, {}};

  // IEEE sqrt(-0) preserves the sign. The mathematical reciprocal of that
  // result is -infinity; MPFR's direct rec_sqrt deliberately returns +infinity
  // for -0, so make this edge policy target-owned and explicit here.
  if (request.operation == MPFRFormalOperation::Rsqrt &&
      classification->valueClass == LogicalValueClass::Zero &&
      classification->negative) {
    return FormalNumericResult{
        {request.resultFormat, infinityBits(parameters, /*negative=*/true)},
        {/*invalid=*/false, /*divByZero=*/true}};
  }

  llvm::Expected<MPFREnvironmentScope> environment =
      MPFREnvironmentScope::create(parameters, *roundingMode);
  if (!environment)
    return environment.takeError();

  MPFRValue source(descriptor.precisionBits);
  MPFRValue result(descriptor.precisionBits);
  if (classification->valueClass == LogicalValueClass::Infinity)
    mpfr_set_inf(source.get(), classification->negative ? -1 : 1);
  else
    decodeFinite(parameters, *input, source.get());

  if (isAdaptiveComposite(request.operation))
    return evaluateAdaptiveComposite(request, parameters, *input, source.get(),
                                     *environment, *roundingMode);

  int ternary = 0;
  switch (request.operation) {
  case MPFRFormalOperation::Sqrt:
    ternary = mpfr_sqrt(result.get(), source.get(), *roundingMode);
    break;
  case MPFRFormalOperation::Rsqrt:
    ternary = mpfr_rec_sqrt(result.get(), source.get(), *roundingMode);
    break;
  case MPFRFormalOperation::Log2:
    ternary = mpfr_log2(result.get(), source.get(), *roundingMode);
    break;
  case MPFRFormalOperation::Ln:
    ternary = mpfr_log(result.get(), source.get(), *roundingMode);
    break;
  case MPFRFormalOperation::Pow2:
    ternary = mpfr_exp2(result.get(), source.get(), *roundingMode);
    break;
  case MPFRFormalOperation::Exp:
    ternary = mpfr_exp(result.get(), source.get(), *roundingMode);
    break;
  case MPFRFormalOperation::Sin:
    ternary = mpfr_sin(result.get(), source.get(), *roundingMode);
    break;
  case MPFRFormalOperation::Cos:
    ternary = mpfr_cos(result.get(), source.get(), *roundingMode);
    break;
  case MPFRFormalOperation::Tanh:
    ternary = mpfr_tanh(result.get(), source.get(), *roundingMode);
    break;
  case MPFRFormalOperation::Sigmoid:
  case MPFRFormalOperation::Softplus:
    llvm_unreachable("adaptive composite reached the direct MPFR dispatcher");
  }
  ternary = mpfr_subnormalize(result.get(), ternary, *roundingMode);
  (void)ternary;

  const bool invalid =
      mpfr_nan_p(result.get()) &&
      classification->valueClass != LogicalValueClass::QuietNaN;
  const mpfr_flags_t flags = environment->getRaisedFlags();
  llvm::Expected<uint64_t> bits = encodeResult(parameters, result.get());
  if (!bits)
    return bits.takeError();
  return FormalNumericResult{{request.resultFormat, *bits},
                             mapFlags(flags, invalid)};
}

} // namespace wafer
