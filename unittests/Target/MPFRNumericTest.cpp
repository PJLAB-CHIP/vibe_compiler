//===- MPFRNumericTest.cpp - Managed transcendental backend tests ---------===//

#include "Wafer/Target/MPFRNumeric.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <atomic>
#include <cstdint>
#include <iterator>
#include <thread>

#include <mpfr.h>

namespace {

using namespace wafer;

struct FormatCase {
  LogicalFormat format;
  uint64_t positiveZero;
  uint64_t negativeZero;
  uint64_t half;
  uint64_t one;
  uint64_t four;
  uint64_t quietNaN;
  uint64_t signalingNaN;
  uint64_t minimumSubnormal;
  uint64_t ln2;
};

constexpr FormatCase kFormatCases[] = {
    {LogicalFormat::F16, UINT64_C(0), UINT64_C(0x8000), UINT64_C(0x3800),
     UINT64_C(0x3c00), UINT64_C(0x4400), UINT64_C(0x7e00), UINT64_C(0x7c01),
     UINT64_C(1), UINT64_C(0x398c)},
    {LogicalFormat::BF16, UINT64_C(0), UINT64_C(0x8000), UINT64_C(0x3f00),
     UINT64_C(0x3f80), UINT64_C(0x4080), UINT64_C(0x7fc0), UINT64_C(0x7f81),
     UINT64_C(1), UINT64_C(0x3f31)},
    {LogicalFormat::F32, UINT64_C(0), UINT64_C(0x80000000),
     UINT64_C(0x3f000000), UINT64_C(0x3f800000), UINT64_C(0x40800000),
     UINT64_C(0x7fc00000), UINT64_C(0x7f800001), UINT64_C(1),
     UINT64_C(0x3f317218)},
    {LogicalFormat::TF32, UINT64_C(0), UINT64_C(0x80000000),
     UINT64_C(0x3f000000), UINT64_C(0x3f800000), UINT64_C(0x40800000),
     UINT64_C(0x7fc00000), UINT64_C(0x7f802000), UINT64_C(0x2000),
     UINT64_C(0x3f318000)},
};

constexpr MPFRFormalOperation kAllOperations[] = {
    MPFRFormalOperation::Sqrt,     MPFRFormalOperation::Rsqrt,
    MPFRFormalOperation::Log2,     MPFRFormalOperation::Ln,
    MPFRFormalOperation::Pow2,     MPFRFormalOperation::Exp,
    MPFRFormalOperation::Sin,      MPFRFormalOperation::Cos,
    MPFRFormalOperation::Tanh,     MPFRFormalOperation::Sigmoid,
    MPFRFormalOperation::Softplus,
};

uint64_t positiveInfinity(const FormatCase &format) {
  return format.format == LogicalFormat::F16
             ? UINT64_C(0x7c00)
             : (format.format == LogicalFormat::BF16 ? UINT64_C(0x7f80)
                                                     : UINT64_C(0x7f800000));
}

MPFRFormalRequest
request(MPFRFormalOperation operation, const FormatCase &format, uint64_t bits,
        NumericRoundingMode rounding = NumericRoundingMode::NearestEven) {
  return {operation, rounding, format.format, {format.format, bits}};
}

template <typename T>
void expectNumericError(llvm::Expected<T> result,
                        MPFRNumericErrorCode expectedCode) {
  ASSERT_FALSE(static_cast<bool>(result));
  bool matched = false;
  llvm::handleAllErrors(result.takeError(), [&](const MPFRNumericError &error) {
    matched = true;
    EXPECT_EQ(error.getCode(), expectedCode);
  });
  EXPECT_TRUE(matched);
}

class CallerMPFREnvironment {
public:
  CallerMPFREnvironment()
      : emin(mpfr_get_emin()), emax(mpfr_get_emax()),
        precision(mpfr_get_default_prec()),
        rounding(mpfr_get_default_rounding_mode()), flags(mpfr_flags_save()) {}

  ~CallerMPFREnvironment() {
    EXPECT_EQ(mpfr_set_emin(mpfr_get_emin_min()), 0);
    EXPECT_EQ(mpfr_set_emax(mpfr_get_emax_max()), 0);
    EXPECT_EQ(mpfr_set_emin(emin), 0);
    EXPECT_EQ(mpfr_set_emax(emax), 0);
    mpfr_set_default_prec(precision);
    mpfr_set_default_rounding_mode(rounding);
    mpfr_flags_restore(flags, MPFR_FLAGS_ALL);
  }

private:
  mpfr_exp_t emin;
  mpfr_exp_t emax;
  mpfr_prec_t precision;
  mpfr_rnd_t rounding;
  mpfr_flags_t flags;
};

TEST(MPFRNumericTest, ManagedHeaderRuntimeAndTLSIdentityAgree) {
  EXPECT_STREQ(mpfr_get_version(), MPFR_VERSION_STRING);
  EXPECT_NE(mpfr_buildopt_tls_p(), 0);
}

TEST(MPFRNumericTest, ExactKnownPointsCoverEveryComponentFormat) {
  for (const FormatCase &format : kFormatCases) {
    llvm::Expected<FormalNumericResult> squareRoot = executeMPFRFormal(
        request(MPFRFormalOperation::Sqrt, format, format.four));
    ASSERT_TRUE(static_cast<bool>(squareRoot))
        << llvm::toString(squareRoot.takeError());
    EXPECT_EQ(squareRoot->value.format, format.format);
    const uint64_t expectedTwo =
        format.format == LogicalFormat::F16
            ? UINT64_C(0x4000)
            : (format.format == LogicalFormat::BF16 ? UINT64_C(0x4000)
                                                    : UINT64_C(0x40000000));
    EXPECT_EQ(squareRoot->value.bits, expectedTwo);
    EXPECT_FALSE(squareRoot->flags.any());

    llvm::Expected<FormalNumericResult> reciprocal = executeMPFRFormal(
        request(MPFRFormalOperation::Rsqrt, format, format.four));
    ASSERT_TRUE(static_cast<bool>(reciprocal))
        << llvm::toString(reciprocal.takeError());
    EXPECT_EQ(reciprocal->value.bits, format.half);
    EXPECT_FALSE(reciprocal->flags.any());

    llvm::Expected<FormalNumericResult> exponential = executeMPFRFormal(
        request(MPFRFormalOperation::Exp, format, format.positiveZero));
    ASSERT_TRUE(static_cast<bool>(exponential))
        << llvm::toString(exponential.takeError());
    EXPECT_EQ(exponential->value.bits, format.one);
    EXPECT_FALSE(exponential->flags.any());

    llvm::Expected<FormalNumericResult> negativeTanh = executeMPFRFormal(
        request(MPFRFormalOperation::Tanh, format, format.negativeZero));
    ASSERT_TRUE(static_cast<bool>(negativeTanh))
        << llvm::toString(negativeTanh.takeError());
    EXPECT_EQ(negativeTanh->value.bits, format.negativeZero);
    EXPECT_FALSE(negativeTanh->flags.any());

    llvm::Expected<FormalNumericResult> log2 = executeMPFRFormal(
        request(MPFRFormalOperation::Log2, format, format.one));
    ASSERT_TRUE(static_cast<bool>(log2)) << llvm::toString(log2.takeError());
    EXPECT_EQ(log2->value.bits, format.positiveZero);
    EXPECT_FALSE(log2->flags.any());

    llvm::Expected<FormalNumericResult> naturalLog =
        executeMPFRFormal(request(MPFRFormalOperation::Ln, format, format.one));
    ASSERT_TRUE(static_cast<bool>(naturalLog))
        << llvm::toString(naturalLog.takeError());
    EXPECT_EQ(naturalLog->value.bits, format.positiveZero);
    EXPECT_FALSE(naturalLog->flags.any());

    llvm::Expected<FormalNumericResult> pow2 = executeMPFRFormal(
        request(MPFRFormalOperation::Pow2, format, format.positiveZero));
    ASSERT_TRUE(static_cast<bool>(pow2)) << llvm::toString(pow2.takeError());
    EXPECT_EQ(pow2->value.bits, format.one);
    EXPECT_FALSE(pow2->flags.any());

    llvm::Expected<FormalNumericResult> sine = executeMPFRFormal(
        request(MPFRFormalOperation::Sin, format, format.negativeZero));
    ASSERT_TRUE(static_cast<bool>(sine)) << llvm::toString(sine.takeError());
    EXPECT_EQ(sine->value.bits, format.negativeZero);
    EXPECT_FALSE(sine->flags.any());

    llvm::Expected<FormalNumericResult> cosine = executeMPFRFormal(
        request(MPFRFormalOperation::Cos, format, format.negativeZero));
    ASSERT_TRUE(static_cast<bool>(cosine))
        << llvm::toString(cosine.takeError());
    EXPECT_EQ(cosine->value.bits, format.one);
    EXPECT_FALSE(cosine->flags.any());

    for (uint64_t zero : {format.positiveZero, format.negativeZero}) {
      llvm::Expected<FormalNumericResult> sigmoid = executeMPFRFormal(
          request(MPFRFormalOperation::Sigmoid, format, zero));
      ASSERT_TRUE(static_cast<bool>(sigmoid))
          << llvm::toString(sigmoid.takeError());
      EXPECT_EQ(sigmoid->value.bits, format.half);
      EXPECT_FALSE(sigmoid->flags.any());
    }

    llvm::Expected<FormalNumericResult> softplus = executeMPFRFormal(
        request(MPFRFormalOperation::Softplus, format, format.positiveZero));
    ASSERT_TRUE(static_cast<bool>(softplus))
        << llvm::toString(softplus.takeError());
    EXPECT_EQ(softplus->value.bits, format.ln2);
    EXPECT_TRUE(softplus->flags.inexact);
    EXPECT_FALSE(softplus->flags.invalid);
    EXPECT_FALSE(softplus->flags.divByZero);
    EXPECT_FALSE(softplus->flags.overflow);
    EXPECT_FALSE(softplus->flags.underflow);
  }
}

TEST(MPFRNumericTest, SubnormalEncodingUsesFormatOwnedSemanticShift) {
  for (const FormatCase &format : kFormatCases) {
    llvm::Expected<FormalNumericResult> result = executeMPFRFormal(
        request(MPFRFormalOperation::Tanh, format, format.minimumSubnormal));
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    EXPECT_EQ(result->value.bits, format.minimumSubnormal);
    EXPECT_TRUE(result->flags.inexact);
    EXPECT_TRUE(result->flags.underflow);
  }
}

TEST(MPFRNumericTest, NaNAndNegativeDomainPolicyIsTargetOwned) {
  for (const FormatCase &format : kFormatCases) {
    llvm::Expected<FormalNumericResult> quiet = executeMPFRFormal(
        request(MPFRFormalOperation::Exp, format, format.quietNaN));
    ASSERT_TRUE(static_cast<bool>(quiet)) << llvm::toString(quiet.takeError());
    EXPECT_EQ(quiet->value.bits, format.quietNaN);
    EXPECT_FALSE(quiet->flags.invalid);

    llvm::Expected<FormalNumericResult> signaling = executeMPFRFormal(
        request(MPFRFormalOperation::Exp, format, format.signalingNaN));
    ASSERT_TRUE(static_cast<bool>(signaling))
        << llvm::toString(signaling.takeError());
    EXPECT_EQ(signaling->value.bits, format.quietNaN);
    EXPECT_TRUE(signaling->flags.invalid);

    const uint64_t negativeOne = format.one | format.negativeZero;
    llvm::Expected<FormalNumericResult> negativeSqrt = executeMPFRFormal(
        request(MPFRFormalOperation::Sqrt, format, negativeOne));
    ASSERT_TRUE(static_cast<bool>(negativeSqrt))
        << llvm::toString(negativeSqrt.takeError());
    EXPECT_EQ(negativeSqrt->value.bits, format.quietNaN);
    EXPECT_TRUE(negativeSqrt->flags.invalid);

    llvm::Expected<FormalNumericResult> negativeZeroRSqrt = executeMPFRFormal(
        request(MPFRFormalOperation::Rsqrt, format, format.negativeZero));
    ASSERT_TRUE(static_cast<bool>(negativeZeroRSqrt))
        << llvm::toString(negativeZeroRSqrt.takeError());
    const uint64_t expectedNegativeInfinity =
        format.format == LogicalFormat::F16
            ? UINT64_C(0xfc00)
            : (format.format == LogicalFormat::BF16 ? UINT64_C(0xff80)
                                                    : UINT64_C(0xff800000));
    EXPECT_EQ(negativeZeroRSqrt->value.bits, expectedNegativeInfinity);
    EXPECT_TRUE(negativeZeroRSqrt->flags.divByZero);
  }
}

TEST(MPFRNumericTest, NaNPayloadSignAndSignedZeroAreExplicit) {
  constexpr uint64_t quietNaNs[] = {UINT64_C(0xfe55), UINT64_C(0xffe5),
                                    UINT64_C(0xffc12345), UINT64_C(0xffc12000)};
  constexpr uint64_t signalingNaNs[] = {UINT64_C(0xfd55), UINT64_C(0xffa5),
                                        UINT64_C(0xff812345),
                                        UINT64_C(0xff812000)};

  for (size_t index = 0; index < std::size(kFormatCases); ++index) {
    const FormatCase &format = kFormatCases[index];
    for (MPFRFormalOperation operation : kAllOperations) {
      llvm::Expected<FormalNumericResult> quiet =
          executeMPFRFormal(request(operation, format, quietNaNs[index]));
      ASSERT_TRUE(static_cast<bool>(quiet))
          << llvm::toString(quiet.takeError());
      EXPECT_EQ(quiet->value.bits, format.quietNaN);
      EXPECT_FALSE(quiet->flags.invalid);

      llvm::Expected<FormalNumericResult> signaling =
          executeMPFRFormal(request(operation, format, signalingNaNs[index]));
      ASSERT_TRUE(static_cast<bool>(signaling))
          << llvm::toString(signaling.takeError());
      EXPECT_EQ(signaling->value.bits, format.quietNaN);
      EXPECT_TRUE(signaling->flags.invalid);
    }

    llvm::Expected<FormalNumericResult> negativeZero = executeMPFRFormal(
        request(MPFRFormalOperation::Sqrt, format, format.negativeZero));
    ASSERT_TRUE(static_cast<bool>(negativeZero))
        << llvm::toString(negativeZero.takeError());
    EXPECT_EQ(negativeZero->value.bits, format.negativeZero);
    EXPECT_FALSE(negativeZero->flags.any());
  }
}

TEST(MPFRNumericTest, AdaptiveCompositeHardMidpointsRoundUniquely) {
  struct SigmoidCase {
    const FormatCase &format;
    uint64_t input;
    uint64_t expected;
  };
  const SigmoidCase sigmoidCases[] = {
      {kFormatCases[0], UINT64_C(0x9000), UINT64_C(0x3800)},
      {kFormatCases[1], UINT64_C(0xbb80), UINT64_C(0x3f00)},
      {kFormatCases[2], UINT64_C(0xb3800000), UINT64_C(0x3f000000)},
      {kFormatCases[3], UINT64_C(0xba000000), UINT64_C(0x3f000000)},
  };
  // For precision p, x=-2^-p makes x/4 land on the midpoint immediately
  // below 0.5. The much smaller positive cubic term in
  // sigmoid(x)=0.5+x/4-x^3/48+... selects 0.5. A target-precision or fixed
  // guard-bit composite cannot justify this decision; the directed interval
  // must refine until it lies wholly on that side of the midpoint.
  for (const SigmoidCase &testCase : sigmoidCases) {
    llvm::Expected<FormalNumericResult> result = executeMPFRFormal(
        request(MPFRFormalOperation::Sigmoid, testCase.format, testCase.input));
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    EXPECT_EQ(result->value.bits, testCase.expected);
    EXPECT_TRUE(result->flags.inexact);
    EXPECT_FALSE(result->flags.invalid);
    EXPECT_FALSE(result->flags.divByZero);
    EXPECT_FALSE(result->flags.overflow);
    EXPECT_FALSE(result->flags.underflow);
  }

  // This F16 softplus value is only about 2.86e-6 ulp below a destination
  // midpoint; the correctly rounded result is the lower neighbor.
  llvm::Expected<FormalNumericResult> softplus = executeMPFRFormal(request(
      MPFRFormalOperation::Softplus, kFormatCases[0], UINT64_C(0x8430)));
  ASSERT_TRUE(static_cast<bool>(softplus))
      << llvm::toString(softplus.takeError());
  EXPECT_EQ(softplus->value.bits, UINT64_C(0x398b));
  EXPECT_TRUE(softplus->flags.inexact);
}

TEST(MPFRNumericTest, DirectDomainsPolesAndInfinityPoliciesAreExplicit) {
  const FormatCase &format = kFormatCases[2];
  const uint64_t positiveInf = positiveInfinity(format);
  const uint64_t negativeInf = positiveInf | format.negativeZero;
  const uint64_t negativeOne = format.one | format.negativeZero;

  for (MPFRFormalOperation operation :
       {MPFRFormalOperation::Sqrt, MPFRFormalOperation::Rsqrt,
        MPFRFormalOperation::Log2, MPFRFormalOperation::Ln}) {
    llvm::Expected<FormalNumericResult> domain =
        executeMPFRFormal(request(operation, format, negativeOne));
    ASSERT_TRUE(static_cast<bool>(domain))
        << llvm::toString(domain.takeError());
    EXPECT_EQ(domain->value.bits, format.quietNaN);
    EXPECT_TRUE(domain->flags.invalid);
    EXPECT_FALSE(domain->flags.divByZero);
  }

  for (MPFRFormalOperation operation :
       {MPFRFormalOperation::Log2, MPFRFormalOperation::Ln}) {
    for (uint64_t zero : {format.positiveZero, format.negativeZero}) {
      llvm::Expected<FormalNumericResult> pole =
          executeMPFRFormal(request(operation, format, zero));
      ASSERT_TRUE(static_cast<bool>(pole)) << llvm::toString(pole.takeError());
      EXPECT_EQ(pole->value.bits, negativeInf);
      EXPECT_TRUE(pole->flags.divByZero);
      EXPECT_FALSE(pole->flags.invalid);
    }
  }

  for (MPFRFormalOperation operation :
       {MPFRFormalOperation::Sin, MPFRFormalOperation::Cos}) {
    for (uint64_t infinity : {positiveInf, negativeInf}) {
      llvm::Expected<FormalNumericResult> domain =
          executeMPFRFormal(request(operation, format, infinity));
      ASSERT_TRUE(static_cast<bool>(domain))
          << llvm::toString(domain.takeError());
      EXPECT_EQ(domain->value.bits, format.quietNaN);
      EXPECT_TRUE(domain->flags.invalid);
    }
  }

  struct InfinityCase {
    MPFRFormalOperation operation;
    uint64_t positiveExpected;
    uint64_t negativeExpected;
  };
  const InfinityCase infinityCases[] = {
      {MPFRFormalOperation::Sqrt, positiveInf, format.quietNaN},
      {MPFRFormalOperation::Rsqrt, format.positiveZero, format.quietNaN},
      {MPFRFormalOperation::Log2, positiveInf, format.quietNaN},
      {MPFRFormalOperation::Ln, positiveInf, format.quietNaN},
      {MPFRFormalOperation::Pow2, positiveInf, format.positiveZero},
      {MPFRFormalOperation::Exp, positiveInf, format.positiveZero},
      {MPFRFormalOperation::Tanh, format.one, negativeOne},
      {MPFRFormalOperation::Sigmoid, format.one, format.positiveZero},
      {MPFRFormalOperation::Softplus, positiveInf, format.positiveZero},
  };
  for (const InfinityCase &testCase : infinityCases) {
    llvm::Expected<FormalNumericResult> positive =
        executeMPFRFormal(request(testCase.operation, format, positiveInf));
    ASSERT_TRUE(static_cast<bool>(positive))
        << llvm::toString(positive.takeError());
    EXPECT_EQ(positive->value.bits, testCase.positiveExpected);
    EXPECT_FALSE(positive->flags.any());

    llvm::Expected<FormalNumericResult> negative =
        executeMPFRFormal(request(testCase.operation, format, negativeInf));
    ASSERT_TRUE(static_cast<bool>(negative))
        << llvm::toString(negative.takeError());
    EXPECT_EQ(negative->value.bits, testCase.negativeExpected);
    if (testCase.negativeExpected == format.quietNaN)
      EXPECT_TRUE(negative->flags.invalid);
    else
      EXPECT_FALSE(negative->flags.any());
  }

  llvm::Expected<FormalNumericResult> positiveZeroRSqrt = executeMPFRFormal(
      request(MPFRFormalOperation::Rsqrt, format, format.positiveZero));
  ASSERT_TRUE(static_cast<bool>(positiveZeroRSqrt))
      << llvm::toString(positiveZeroRSqrt.takeError());
  EXPECT_EQ(positiveZeroRSqrt->value.bits, positiveInf);
  EXPECT_TRUE(positiveZeroRSqrt->flags.divByZero);
}

TEST(MPFRNumericTest, F32UnderflowAndDirectedRoundingAreObservable) {
  const FormatCase &format = kFormatCases[2];
  llvm::Expected<FormalNumericResult> underflow = executeMPFRFormal(request(
      MPFRFormalOperation::Exp, format, UINT64_C(0xc2d00000)) /* -104 */);
  ASSERT_TRUE(static_cast<bool>(underflow))
      << llvm::toString(underflow.takeError());
  EXPECT_EQ(underflow->value.bits, UINT64_C(0));
  EXPECT_TRUE(underflow->flags.underflow);
  EXPECT_TRUE(underflow->flags.inexact);

  llvm::Expected<FormalNumericResult> downward =
      executeMPFRFormal(request(MPFRFormalOperation::Exp, format, format.one,
                                NumericRoundingMode::TowardNegative));
  llvm::Expected<FormalNumericResult> upward =
      executeMPFRFormal(request(MPFRFormalOperation::Exp, format, format.one,
                                NumericRoundingMode::TowardPositive));
  ASSERT_TRUE(static_cast<bool>(downward))
      << llvm::toString(downward.takeError());
  ASSERT_TRUE(static_cast<bool>(upward)) << llvm::toString(upward.takeError());
  EXPECT_LT(downward->value.bits, upward->value.bits);
  EXPECT_TRUE(downward->flags.inexact);
  EXPECT_TRUE(upward->flags.inexact);
}

TEST(MPFRNumericTest, GradualTininessAndAdaptiveExtremesAreFinalResultOwned) {
  const FormatCase &format = kFormatCases[2];

  llvm::Expected<FormalNumericResult> exactMinimumSubnormal =
      executeMPFRFormal(request(MPFRFormalOperation::Pow2, format,
                                UINT64_C(0xc3150000)) /* -149 */);
  ASSERT_TRUE(static_cast<bool>(exactMinimumSubnormal))
      << llvm::toString(exactMinimumSubnormal.takeError());
  EXPECT_EQ(exactMinimumSubnormal->value.bits, UINT64_C(1));
  EXPECT_FALSE(exactMinimumSubnormal->flags.any());

  llvm::Expected<FormalNumericResult> halfMinimumSubnormal =
      executeMPFRFormal(request(MPFRFormalOperation::Pow2, format,
                                UINT64_C(0xc3160000)) /* -150 */);
  ASSERT_TRUE(static_cast<bool>(halfMinimumSubnormal))
      << llvm::toString(halfMinimumSubnormal.takeError());
  EXPECT_EQ(halfMinimumSubnormal->value.bits, UINT64_C(0));
  EXPECT_TRUE(halfMinimumSubnormal->flags.underflow);
  EXPECT_TRUE(halfMinimumSubnormal->flags.inexact);

  for (MPFRFormalOperation operation :
       {MPFRFormalOperation::Sigmoid, MPFRFormalOperation::Softplus}) {
    llvm::Expected<FormalNumericResult> subnormal = executeMPFRFormal(
        request(operation, format, UINT64_C(0xc2ce0000)) /* -103 */);
    ASSERT_TRUE(static_cast<bool>(subnormal))
        << llvm::toString(subnormal.takeError());
    EXPECT_EQ(subnormal->value.bits, UINT64_C(1));
    EXPECT_TRUE(subnormal->flags.underflow);
    EXPECT_TRUE(subnormal->flags.inexact);

    llvm::Expected<FormalNumericResult> zero = executeMPFRFormal(
        request(operation, format, UINT64_C(0xc2d00000)) /* -104 */);
    ASSERT_TRUE(static_cast<bool>(zero)) << llvm::toString(zero.takeError());
    EXPECT_EQ(zero->value.bits, UINT64_C(0));
    EXPECT_TRUE(zero->flags.underflow);
    EXPECT_TRUE(zero->flags.inexact);

    llvm::Expected<FormalNumericResult> saturatedNegative = executeMPFRFormal(
        request(operation, format, UINT64_C(0xc4800000)) /* -1024 */);
    ASSERT_TRUE(static_cast<bool>(saturatedNegative))
        << llvm::toString(saturatedNegative.takeError());
    EXPECT_EQ(saturatedNegative->value.bits, UINT64_C(0));
    EXPECT_TRUE(saturatedNegative->flags.underflow);
    EXPECT_TRUE(saturatedNegative->flags.inexact);
  }

  llvm::Expected<FormalNumericResult> saturatedSigmoid =
      executeMPFRFormal(request(MPFRFormalOperation::Sigmoid, format,
                                UINT64_C(0x42000000)) /* 32 */);
  ASSERT_TRUE(static_cast<bool>(saturatedSigmoid))
      << llvm::toString(saturatedSigmoid.takeError());
  EXPECT_EQ(saturatedSigmoid->value.bits, format.one);
  EXPECT_TRUE(saturatedSigmoid->flags.inexact);
  EXPECT_FALSE(saturatedSigmoid->flags.overflow);
  EXPECT_FALSE(saturatedSigmoid->flags.underflow);

  llvm::Expected<FormalNumericResult> saturatedSoftplus =
      executeMPFRFormal(request(MPFRFormalOperation::Softplus, format,
                                UINT64_C(0x42000000)) /* 32 */);
  ASSERT_TRUE(static_cast<bool>(saturatedSoftplus))
      << llvm::toString(saturatedSoftplus.takeError());
  EXPECT_EQ(saturatedSoftplus->value.bits, UINT64_C(0x42000000));
  EXPECT_TRUE(saturatedSoftplus->flags.inexact);
  EXPECT_FALSE(saturatedSoftplus->flags.overflow);

  llvm::Expected<FormalNumericResult> maximumSoftplus = executeMPFRFormal(
      request(MPFRFormalOperation::Softplus, format, UINT64_C(0x7f7fffff)));
  ASSERT_TRUE(static_cast<bool>(maximumSoftplus))
      << llvm::toString(maximumSoftplus.takeError());
  EXPECT_EQ(maximumSoftplus->value.bits, UINT64_C(0x7f7fffff));
  EXPECT_TRUE(maximumSoftplus->flags.inexact);
  EXPECT_FALSE(maximumSoftplus->flags.overflow);
}

TEST(MPFRNumericTest, DirectSymmetryAndCompositeMonotonicPairsHold) {
  for (const FormatCase &format : kFormatCases) {
    const uint64_t negativeHalf = format.half | format.negativeZero;
    for (MPFRFormalOperation operation :
         {MPFRFormalOperation::Sin, MPFRFormalOperation::Tanh}) {
      llvm::Expected<FormalNumericResult> positive =
          executeMPFRFormal(request(operation, format, format.half));
      llvm::Expected<FormalNumericResult> negative =
          executeMPFRFormal(request(operation, format, negativeHalf));
      ASSERT_TRUE(static_cast<bool>(positive))
          << llvm::toString(positive.takeError());
      ASSERT_TRUE(static_cast<bool>(negative))
          << llvm::toString(negative.takeError());
      EXPECT_EQ(negative->value.bits,
                positive->value.bits | format.negativeZero);
      EXPECT_EQ(negative->flags, positive->flags);
      EXPECT_TRUE(positive->flags.inexact);
    }

    llvm::Expected<FormalNumericResult> positiveCosine = executeMPFRFormal(
        request(MPFRFormalOperation::Cos, format, format.half));
    llvm::Expected<FormalNumericResult> negativeCosine = executeMPFRFormal(
        request(MPFRFormalOperation::Cos, format, negativeHalf));
    ASSERT_TRUE(static_cast<bool>(positiveCosine))
        << llvm::toString(positiveCosine.takeError());
    ASSERT_TRUE(static_cast<bool>(negativeCosine))
        << llvm::toString(negativeCosine.takeError());
    EXPECT_EQ(negativeCosine->value.bits, positiveCosine->value.bits);
    EXPECT_EQ(negativeCosine->flags, positiveCosine->flags);
    EXPECT_TRUE(positiveCosine->flags.inexact);
  }

  const FormatCase &format = kFormatCases[2];
  const uint64_t negativeOne = format.one | format.negativeZero;
  llvm::Expected<FormalNumericResult> sigmoidNegative = executeMPFRFormal(
      request(MPFRFormalOperation::Sigmoid, format, negativeOne));
  llvm::Expected<FormalNumericResult> sigmoidPositive = executeMPFRFormal(
      request(MPFRFormalOperation::Sigmoid, format, format.one));
  ASSERT_TRUE(static_cast<bool>(sigmoidNegative))
      << llvm::toString(sigmoidNegative.takeError());
  ASSERT_TRUE(static_cast<bool>(sigmoidPositive))
      << llvm::toString(sigmoidPositive.takeError());
  EXPECT_EQ(sigmoidNegative->value.bits, UINT64_C(0x3e89b2b1));
  EXPECT_EQ(sigmoidPositive->value.bits, UINT64_C(0x3f3b26a8));
  EXPECT_LT(sigmoidNegative->value.bits, format.half);
  EXPECT_GT(sigmoidPositive->value.bits, format.half);
  EXPECT_TRUE(sigmoidNegative->flags.inexact);
  EXPECT_TRUE(sigmoidPositive->flags.inexact);

  llvm::Expected<FormalNumericResult> softplusNegative = executeMPFRFormal(
      request(MPFRFormalOperation::Softplus, format, negativeOne));
  llvm::Expected<FormalNumericResult> softplusPositive = executeMPFRFormal(
      request(MPFRFormalOperation::Softplus, format, format.one));
  ASSERT_TRUE(static_cast<bool>(softplusNegative))
      << llvm::toString(softplusNegative.takeError());
  ASSERT_TRUE(static_cast<bool>(softplusPositive))
      << llvm::toString(softplusPositive.takeError());
  EXPECT_EQ(softplusNegative->value.bits, UINT64_C(0x3ea063d6));
  EXPECT_EQ(softplusPositive->value.bits, UINT64_C(0x3fa818f5));
  EXPECT_LT(softplusNegative->value.bits, format.ln2);
  EXPECT_GT(softplusPositive->value.bits, format.ln2);
  EXPECT_TRUE(softplusNegative->flags.inexact);
  EXPECT_TRUE(softplusPositive->flags.inexact);
}

TEST(MPFRNumericTest, F32OverflowFlagsAreExplicit) {
  const FormatCase &format = kFormatCases[2];
  llvm::Expected<FormalNumericResult> overflow = executeMPFRFormal(request(
      MPFRFormalOperation::Exp, format, UINT64_C(0x42c80000)) /* 100 */);
  ASSERT_TRUE(static_cast<bool>(overflow))
      << llvm::toString(overflow.takeError());
  EXPECT_EQ(overflow->value.bits, UINT64_C(0x7f800000));
  EXPECT_TRUE(overflow->flags.overflow);
  EXPECT_TRUE(overflow->flags.inexact);
  EXPECT_FALSE(overflow->flags.invalid);
  EXPECT_FALSE(overflow->flags.divByZero);
}

TEST(MPFRNumericTest, RestoresCompleteCallerEnvironment) {
  CallerMPFREnvironment restoreAtExit;
  ASSERT_EQ(mpfr_set_emin(-777), 0);
  ASSERT_EQ(mpfr_set_emax(888), 0);
  mpfr_set_default_prec(73);
  mpfr_set_default_rounding_mode(MPFR_RNDA);
  mpfr_clear_flags();
  mpfr_flags_set(MPFR_FLAGS_NAN | MPFR_FLAGS_OVERFLOW);

  llvm::Expected<FormalNumericResult> result = executeMPFRFormal(
      request(MPFRFormalOperation::Exp, kFormatCases[2], kFormatCases[2].one));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  llvm::Expected<FormalNumericResult> adaptive = executeMPFRFormal(request(
      MPFRFormalOperation::Softplus, kFormatCases[2], kFormatCases[2].one));
  ASSERT_TRUE(static_cast<bool>(adaptive))
      << llvm::toString(adaptive.takeError());
  EXPECT_EQ(mpfr_get_emin(), -777);
  EXPECT_EQ(mpfr_get_emax(), 888);
  EXPECT_EQ(mpfr_get_default_prec(), 73);
  EXPECT_EQ(mpfr_get_default_rounding_mode(), MPFR_RNDA);
  EXPECT_EQ(mpfr_flags_save(), MPFR_FLAGS_NAN | MPFR_FLAGS_OVERFLOW);
}

TEST(MPFRNumericTest, NestedCallerEnvironmentsRestoreInLIFOOrder) {
  CallerMPFREnvironment restoreAtExit;
  ASSERT_EQ(mpfr_set_emin(-700), 0);
  ASSERT_EQ(mpfr_set_emax(800), 0);
  mpfr_set_default_prec(61);
  mpfr_set_default_rounding_mode(MPFR_RNDD);
  mpfr_clear_flags();
  mpfr_flags_set(MPFR_FLAGS_UNDERFLOW);

  {
    CallerMPFREnvironment restoreOuter;
    ASSERT_EQ(mpfr_set_emin(mpfr_get_emin_min()), 0);
    ASSERT_EQ(mpfr_set_emax(mpfr_get_emax_max()), 0);
    ASSERT_EQ(mpfr_set_emin(-500), 0);
    ASSERT_EQ(mpfr_set_emax(600), 0);
    mpfr_set_default_prec(79);
    mpfr_set_default_rounding_mode(MPFR_RNDU);
    mpfr_clear_flags();
    mpfr_flags_set(MPFR_FLAGS_DIVBY0);

    llvm::Expected<FormalNumericResult> result = executeMPFRFormal(request(
        MPFRFormalOperation::Exp, kFormatCases[2], kFormatCases[2].one));
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    EXPECT_EQ(mpfr_get_emin(), -500);
    EXPECT_EQ(mpfr_get_emax(), 600);
    EXPECT_EQ(mpfr_get_default_prec(), 79);
    EXPECT_EQ(mpfr_get_default_rounding_mode(), MPFR_RNDU);
    EXPECT_EQ(mpfr_flags_save(), MPFR_FLAGS_DIVBY0);
  }

  EXPECT_EQ(mpfr_get_emin(), -700);
  EXPECT_EQ(mpfr_get_emax(), 800);
  EXPECT_EQ(mpfr_get_default_prec(), 61);
  EXPECT_EQ(mpfr_get_default_rounding_mode(), MPFR_RNDD);
  EXPECT_EQ(mpfr_flags_save(), MPFR_FLAGS_UNDERFLOW);
}

TEST(MPFRNumericTest, HandlesAndRestoresDisjointCallerExponentRanges) {
  CallerMPFREnvironment restoreAtExit;

  constexpr struct {
    mpfr_exp_t emin;
    mpfr_exp_t emax;
    mpfr_prec_t precision;
    mpfr_rnd_t rounding;
    mpfr_flags_t flags;
  } cases[] = {
      {-300, -200, 67, MPFR_RNDD, MPFR_FLAGS_UNDERFLOW},
      {200, 300, 83, MPFR_RNDU, MPFR_FLAGS_OVERFLOW},
  };

  for (const auto &testCase : cases) {
    ASSERT_EQ(mpfr_set_emin(mpfr_get_emin_min()), 0);
    ASSERT_EQ(mpfr_set_emax(mpfr_get_emax_max()), 0);
    ASSERT_EQ(mpfr_set_emin(testCase.emin), 0);
    ASSERT_EQ(mpfr_set_emax(testCase.emax), 0);
    mpfr_set_default_prec(testCase.precision);
    mpfr_set_default_rounding_mode(testCase.rounding);
    mpfr_clear_flags();
    mpfr_flags_set(testCase.flags);

    llvm::Expected<FormalNumericResult> result = executeMPFRFormal(request(
        MPFRFormalOperation::Softplus, kFormatCases[2], kFormatCases[2].one));
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    EXPECT_EQ(result->value.bits, UINT64_C(0x3fa818f5));
    EXPECT_EQ(mpfr_get_emin(), testCase.emin);
    EXPECT_EQ(mpfr_get_emax(), testCase.emax);
    EXPECT_EQ(mpfr_get_default_prec(), testCase.precision);
    EXPECT_EQ(mpfr_get_default_rounding_mode(), testCase.rounding);
    EXPECT_EQ(mpfr_flags_save(), testCase.flags);
  }
}

TEST(MPFRNumericTest, EarlySpecialValuePolicyPreservesCallerEnvironment) {
  CallerMPFREnvironment restoreAtExit;
  ASSERT_EQ(mpfr_set_emin(-777), 0);
  ASSERT_EQ(mpfr_set_emax(888), 0);
  mpfr_set_default_prec(73);
  mpfr_set_default_rounding_mode(MPFR_RNDA);
  mpfr_clear_flags();
  mpfr_flags_set(MPFR_FLAGS_NAN | MPFR_FLAGS_OVERFLOW);

  llvm::Expected<FormalNumericResult> quiet = executeMPFRFormal(request(
      MPFRFormalOperation::Exp, kFormatCases[2], kFormatCases[2].quietNaN));
  ASSERT_TRUE(static_cast<bool>(quiet)) << llvm::toString(quiet.takeError());
  EXPECT_EQ(quiet->value.bits, kFormatCases[2].quietNaN);

  llvm::Expected<FormalNumericResult> negativeZero =
      executeMPFRFormal(request(MPFRFormalOperation::Rsqrt, kFormatCases[2],
                                kFormatCases[2].negativeZero));
  ASSERT_TRUE(static_cast<bool>(negativeZero))
      << llvm::toString(negativeZero.takeError());
  EXPECT_EQ(negativeZero->value.bits, UINT64_C(0xff800000));

  llvm::Expected<FormalNumericResult> positiveInfinitySigmoid =
      executeMPFRFormal(request(MPFRFormalOperation::Sigmoid, kFormatCases[2],
                                UINT64_C(0x7f800000)));
  ASSERT_TRUE(static_cast<bool>(positiveInfinitySigmoid))
      << llvm::toString(positiveInfinitySigmoid.takeError());
  EXPECT_EQ(positiveInfinitySigmoid->value.bits, kFormatCases[2].one);

  EXPECT_EQ(mpfr_get_emin(), -777);
  EXPECT_EQ(mpfr_get_emax(), 888);
  EXPECT_EQ(mpfr_get_default_prec(), 73);
  EXPECT_EQ(mpfr_get_default_rounding_mode(), MPFR_RNDA);
  EXPECT_EQ(mpfr_flags_save(), MPFR_FLAGS_NAN | MPFR_FLAGS_OVERFLOW);
}

TEST(MPFRNumericTest, ManagedTLSIsIndependentAcrossOSThreads) {
  std::atomic<unsigned> ready{0};
  std::atomic<bool> firstPassed{false};
  std::atomic<bool> secondPassed{false};

  auto worker = [&](mpfr_exp_t emin, mpfr_exp_t emax, mpfr_prec_t precision,
                    mpfr_rnd_t rounding, mpfr_flags_t flags,
                    MPFRFormalOperation operation, uint64_t input,
                    std::atomic<bool> &passed) {
    const bool configured =
        mpfr_set_emin(emin) == 0 && mpfr_set_emax(emax) == 0;
    mpfr_set_default_prec(precision);
    mpfr_set_default_rounding_mode(rounding);
    mpfr_clear_flags();
    mpfr_flags_set(flags);
    ready.fetch_add(1, std::memory_order_release);
    while (ready.load(std::memory_order_acquire) != 2)
      std::this_thread::yield();

    llvm::Expected<FormalNumericResult> result =
        executeMPFRFormal(request(operation, kFormatCases[2], input));
    const bool restored = configured && static_cast<bool>(result) &&
                          mpfr_get_emin() == emin && mpfr_get_emax() == emax &&
                          mpfr_get_default_prec() == precision &&
                          mpfr_get_default_rounding_mode() == rounding &&
                          mpfr_flags_save() == flags;
    if (!result)
      llvm::consumeError(result.takeError());
    passed.store(restored, std::memory_order_release);
  };

  std::thread first(worker, -700, 800, 61, MPFR_RNDD, MPFR_FLAGS_NAN,
                    MPFRFormalOperation::Softplus, kFormatCases[2].one,
                    std::ref(firstPassed));
  std::thread second(worker, -600, 700, 79, MPFR_RNDU, MPFR_FLAGS_DIVBY0,
                     MPFRFormalOperation::Sigmoid, kFormatCases[2].one,
                     std::ref(secondPassed));
  first.join();
  second.join();

  EXPECT_TRUE(firstPassed.load(std::memory_order_acquire));
  EXPECT_TRUE(secondPassed.load(std::memory_order_acquire));
}

TEST(MPFRNumericTest, RejectsStochasticAndCrossFormatBeforeExecution) {
  MPFRFormalRequest stochastic =
      request(MPFRFormalOperation::Exp, kFormatCases[2], kFormatCases[2].one,
              NumericRoundingMode::Stochastic);
  expectNumericError(executeMPFRFormal(stochastic),
                     MPFRNumericErrorCode::UnsupportedRoundingMode);

  MPFRFormalRequest crossFormat = stochastic;
  crossFormat.roundingMode = NumericRoundingMode::NearestEven;
  crossFormat.resultFormat = LogicalFormat::BF16;
  expectNumericError(executeMPFRFormal(crossFormat),
                     MPFRNumericErrorCode::FormatMismatch);

  MPFRFormalRequest directedComposite =
      request(MPFRFormalOperation::Sigmoid, kFormatCases[2],
              kFormatCases[2].one, NumericRoundingMode::TowardPositive);
  expectNumericError(executeMPFRFormal(directedComposite),
                     MPFRNumericErrorCode::UnsupportedRoundingMode);
}

} // namespace
