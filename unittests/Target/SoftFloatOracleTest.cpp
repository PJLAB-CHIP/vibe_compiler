//===- SoftFloatOracleTest.cpp - Independent IEEE oracle tests ------------===//

#include "Wafer/Target/SoftFloatOracle.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

extern "C" {
#include <softfloat.h>
}

namespace {

using namespace wafer;

SoftFloatOracleRequest
unaryRequest(SoftFloatOracleOperation operation, LogicalFormat resultFormat,
             RawLogicalValue lhs,
             NumericRoundingMode rounding = NumericRoundingMode::NearestEven) {
  return {operation,    rounding, SoftFloatOracleTininess::AfterRounding,
          resultFormat, lhs,      std::nullopt,
          std::nullopt};
}

SoftFloatOracleRequest binaryRequest(
    SoftFloatOracleOperation operation, LogicalFormat format, uint64_t lhs,
    uint64_t rhs,
    NumericRoundingMode rounding = NumericRoundingMode::NearestEven,
    SoftFloatOracleTininess tininess = SoftFloatOracleTininess::AfterRounding) {
  return {operation,   rounding,      tininess,
          format,      {format, lhs}, RawLogicalValue{format, rhs},
          std::nullopt};
}

template <typename T>
void expectOracleError(llvm::Expected<T> result,
                       SoftFloatOracleErrorCode expectedCode) {
  ASSERT_FALSE(static_cast<bool>(result));
  bool matched = false;
  llvm::handleAllErrors(result.takeError(), [&](const SoftFloatOracleError &e) {
    matched = true;
    EXPECT_EQ(e.getCode(), expectedCode);
  });
  EXPECT_TRUE(matched);
}

TEST(SoftFloatOracleTest, F32ToF16TieHonorsDirectedRounding) {
  // 1 + 2^-11 is exactly halfway between the first two F16 values at 1.
  const RawLogicalValue tie{LogicalFormat::F32, UINT64_C(0x3f801000)};

  llvm::Expected<FormalNumericResult> nearest = executeSoftFloatOracle(
      unaryRequest(SoftFloatOracleOperation::Convert, LogicalFormat::F16, tie));
  ASSERT_TRUE(static_cast<bool>(nearest))
      << llvm::toString(nearest.takeError());
  EXPECT_EQ(nearest->value.format, LogicalFormat::F16);
  EXPECT_EQ(nearest->value.bits, UINT64_C(0x3c00));
  EXPECT_TRUE(nearest->flags.inexact);
  EXPECT_FALSE(nearest->flags.overflow);

  llvm::Expected<FormalNumericResult> upward = executeSoftFloatOracle(
      unaryRequest(SoftFloatOracleOperation::Convert, LogicalFormat::F16, tie,
                   NumericRoundingMode::TowardPositive));
  ASSERT_TRUE(static_cast<bool>(upward)) << llvm::toString(upward.takeError());
  EXPECT_EQ(upward->value.format, LogicalFormat::F16);
  EXPECT_EQ(upward->value.bits, UINT64_C(0x3c01));
  EXPECT_TRUE(upward->flags.inexact);
}

TEST(SoftFloatOracleTest, F32DivisionMapsSoftFloatInfiniteFlag) {
  SoftFloatOracleRequest request{
      SoftFloatOracleOperation::Divide,
      NumericRoundingMode::NearestEven,
      SoftFloatOracleTininess::AfterRounding,
      LogicalFormat::F32,
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      RawLogicalValue{LogicalFormat::F32, UINT64_C(0)},
      std::nullopt,
  };
  llvm::Expected<FormalNumericResult> result = executeSoftFloatOracle(request);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->value.format, LogicalFormat::F32);
  EXPECT_EQ(result->value.bits, UINT64_C(0x7f800000));
  EXPECT_EQ(result->flags, (FormalNumericExceptionFlags{/*invalid=*/false,
                                                        /*divByZero=*/true,
                                                        /*overflow=*/false,
                                                        /*underflow=*/false,
                                                        /*inexact=*/false}));
}

TEST(SoftFloatOracleTest, DefaultNaNSpecializationIsObserved) {
  llvm::Expected<FormalNumericResult> result = executeSoftFloatOracle(
      unaryRequest(SoftFloatOracleOperation::SquareRoot, LogicalFormat::F32,
                   {LogicalFormat::F32, UINT64_C(0xbf800000)}));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->value.format, LogicalFormat::F32);
  EXPECT_EQ(result->value.bits, UINT64_C(0x7fc00000));
  EXPECT_TRUE(result->flags.invalid);
}

TEST(SoftFloatOracleTest, ExplicitNaNSignedZeroAndArithmeticFlags) {
  llvm::Expected<FormalNumericResult> quiet = executeSoftFloatOracle(
      binaryRequest(SoftFloatOracleOperation::Add, LogicalFormat::F32,
                    UINT64_C(0x7fc12345), UINT64_C(0x3f800000)));
  ASSERT_TRUE(static_cast<bool>(quiet)) << llvm::toString(quiet.takeError());
  EXPECT_EQ(quiet->value.bits, UINT64_C(0x7fc00000));
  EXPECT_FALSE(quiet->flags.invalid);

  llvm::Expected<FormalNumericResult> signaling = executeSoftFloatOracle(
      binaryRequest(SoftFloatOracleOperation::Add, LogicalFormat::F32,
                    UINT64_C(0x7f812345), UINT64_C(0x3f800000)));
  ASSERT_TRUE(static_cast<bool>(signaling))
      << llvm::toString(signaling.takeError());
  EXPECT_EQ(signaling->value.bits, UINT64_C(0x7fc00000));
  EXPECT_TRUE(signaling->flags.invalid);

  llvm::Expected<FormalNumericResult> positiveZero = executeSoftFloatOracle(
      binaryRequest(SoftFloatOracleOperation::Add, LogicalFormat::F32,
                    UINT64_C(0x3f800000), UINT64_C(0xbf800000)));
  ASSERT_TRUE(static_cast<bool>(positiveZero))
      << llvm::toString(positiveZero.takeError());
  EXPECT_EQ(positiveZero->value.bits, UINT64_C(0));
  EXPECT_FALSE(positiveZero->flags.any());

  llvm::Expected<FormalNumericResult> negativeZero = executeSoftFloatOracle(
      binaryRequest(SoftFloatOracleOperation::Add, LogicalFormat::F32,
                    UINT64_C(0x3f800000), UINT64_C(0xbf800000),
                    NumericRoundingMode::TowardNegative));
  ASSERT_TRUE(static_cast<bool>(negativeZero))
      << llvm::toString(negativeZero.takeError());
  EXPECT_EQ(negativeZero->value.bits, UINT64_C(0x80000000));
  EXPECT_FALSE(negativeZero->flags.any());

  llvm::Expected<FormalNumericResult> signedZero = executeSoftFloatOracle(
      binaryRequest(SoftFloatOracleOperation::Multiply, LogicalFormat::F32,
                    UINT64_C(0x80000000), UINT64_C(0x40000000)));
  ASSERT_TRUE(static_cast<bool>(signedZero))
      << llvm::toString(signedZero.takeError());
  EXPECT_EQ(signedZero->value.bits, UINT64_C(0x80000000));
  EXPECT_FALSE(signedZero->flags.any());

  llvm::Expected<FormalNumericResult> overflow = executeSoftFloatOracle(
      binaryRequest(SoftFloatOracleOperation::Multiply, LogicalFormat::F32,
                    UINT64_C(0x7f7fffff), UINT64_C(0x40000000)));
  ASSERT_TRUE(static_cast<bool>(overflow))
      << llvm::toString(overflow.takeError());
  EXPECT_EQ(overflow->value.bits, UINT64_C(0x7f800000));
  EXPECT_TRUE(overflow->flags.overflow);
  EXPECT_TRUE(overflow->flags.inexact);
}

TEST(SoftFloatOracleTest, AfterRoundingTininessUsesUnboundedExponentPrecision) {
  // This inexact source is in SoftFloat's threshold interval: its raw
  // after-rounding decision remains tiny at destination precision with an
  // unbounded exponent even though finite-range encoding commits the minimum
  // normal result.
  const RawLogicalValue roundsToMinimumNormal{LogicalFormat::F32,
                                              UINT64_C(0xb87fec98)};
  llvm::Expected<FormalNumericResult> minimumNormal = executeSoftFloatOracle(
      unaryRequest(SoftFloatOracleOperation::Convert, LogicalFormat::F16,
                   roundsToMinimumNormal));
  ASSERT_TRUE(static_cast<bool>(minimumNormal))
      << llvm::toString(minimumNormal.takeError());
  EXPECT_EQ(minimumNormal->value.bits, UINT64_C(0x8400));
  EXPECT_TRUE(minimumNormal->flags.inexact);
  EXPECT_TRUE(minimumNormal->flags.underflow);

  SoftFloatOracleRequest before =
      unaryRequest(SoftFloatOracleOperation::Convert, LogicalFormat::F16,
                   roundsToMinimumNormal);
  before.tininess = SoftFloatOracleTininess::BeforeRounding;
  llvm::Expected<FormalNumericResult> rawBefore =
      executeSoftFloatOracle(before);
  ASSERT_TRUE(static_cast<bool>(rawBefore))
      << llvm::toString(rawBefore.takeError());
  EXPECT_EQ(rawBefore->value.bits, UINT64_C(0x8400));
  EXPECT_TRUE(rawBefore->flags.inexact);
  EXPECT_TRUE(rawBefore->flags.underflow);

  llvm::Expected<FormalNumericResult> subnormal = executeSoftFloatOracle(
      unaryRequest(SoftFloatOracleOperation::Convert, LogicalFormat::F16,
                   {LogicalFormat::F32, UINT64_C(0x387fd935)}));
  ASSERT_TRUE(static_cast<bool>(subnormal))
      << llvm::toString(subnormal.takeError());
  EXPECT_EQ(subnormal->value.bits, UINT64_C(0x03ff));
  EXPECT_TRUE(subnormal->flags.inexact);
  EXPECT_TRUE(subnormal->flags.underflow);

  llvm::Expected<FormalNumericResult> exactSubnormal = executeSoftFloatOracle(
      binaryRequest(SoftFloatOracleOperation::Multiply, LogicalFormat::F32,
                    UINT64_C(0x00800000), UINT64_C(0x3f000000)));
  ASSERT_TRUE(static_cast<bool>(exactSubnormal))
      << llvm::toString(exactSubnormal.takeError());
  EXPECT_EQ(exactSubnormal->value.bits, UINT64_C(0x00400000));
  EXPECT_FALSE(exactSubnormal->flags.inexact);
  EXPECT_FALSE(exactSubnormal->flags.underflow);
}

TEST(SoftFloatOracleTest, RestoresCompleteCallerEnvironment) {
  const uint_fast8_t savedRounding = softfloat_roundingMode;
  const uint_fast8_t savedTininess = softfloat_detectTininess;
  const uint_fast8_t savedPrecision = extF80_roundingPrecision;
  const uint_fast8_t savedFlags = softfloat_exceptionFlags;

  softfloat_roundingMode = softfloat_round_near_maxMag;
  softfloat_detectTininess = softfloat_tininess_beforeRounding;
  extF80_roundingPrecision = 32;
  softfloat_exceptionFlags = softfloat_flag_invalid | softfloat_flag_inexact;

  llvm::Expected<FormalNumericResult> result = executeSoftFloatOracle(
      unaryRequest(SoftFloatOracleOperation::SquareRoot, LogicalFormat::F16,
                   {LogicalFormat::F16, UINT64_C(0x4000)}));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(softfloat_roundingMode, softfloat_round_near_maxMag);
  EXPECT_EQ(softfloat_detectTininess, softfloat_tininess_beforeRounding);
  EXPECT_EQ(extF80_roundingPrecision, 32);
  EXPECT_EQ(softfloat_exceptionFlags,
            softfloat_flag_invalid | softfloat_flag_inexact);

  softfloat_roundingMode = savedRounding;
  softfloat_detectTininess = savedTininess;
  extF80_roundingPrecision = savedPrecision;
  softfloat_exceptionFlags = savedFlags;
}

TEST(SoftFloatOracleTest, InvalidPreflightDoesNotModifyCallerEnvironment) {
  const uint_fast8_t savedRounding = softfloat_roundingMode;
  const uint_fast8_t savedTininess = softfloat_detectTininess;
  const uint_fast8_t savedPrecision = extF80_roundingPrecision;
  const uint_fast8_t savedFlags = softfloat_exceptionFlags;

  SoftFloatOracleRequest request =
      unaryRequest(SoftFloatOracleOperation::Convert, LogicalFormat::BF16,
                   {LogicalFormat::F32, UINT64_C(0x3f800000)});
  expectOracleError(executeSoftFloatOracle(request),
                    SoftFloatOracleErrorCode::UnsupportedFormat);
  EXPECT_EQ(softfloat_roundingMode, savedRounding);
  EXPECT_EQ(softfloat_detectTininess, savedTininess);
  EXPECT_EQ(extF80_roundingPrecision, savedPrecision);
  EXPECT_EQ(softfloat_exceptionFlags, savedFlags);
}

TEST(SoftFloatOracleTest, ManagedTLSIsIndependentAcrossOSThreads) {
  std::atomic<unsigned> ready{0};
  std::atomic<bool> firstPassed{false};
  std::atomic<bool> secondPassed{false};

  auto worker = [&](uint_fast8_t rounding, uint_fast8_t tininess,
                    uint_fast8_t precision, uint_fast8_t flags,
                    std::atomic<bool> &passed) {
    softfloat_roundingMode = rounding;
    softfloat_detectTininess = tininess;
    extF80_roundingPrecision = precision;
    softfloat_exceptionFlags = flags;
    ready.fetch_add(1, std::memory_order_release);
    while (ready.load(std::memory_order_acquire) != 2)
      std::this_thread::yield();

    llvm::Expected<FormalNumericResult> result = executeSoftFloatOracle(
        unaryRequest(SoftFloatOracleOperation::SquareRoot, LogicalFormat::F32,
                     {LogicalFormat::F32, UINT64_C(0x40000000)}));
    const bool environmentRestored = static_cast<bool>(result) &&
                                     softfloat_roundingMode == rounding &&
                                     softfloat_detectTininess == tininess &&
                                     extF80_roundingPrecision == precision &&
                                     softfloat_exceptionFlags == flags;
    if (!result)
      llvm::consumeError(result.takeError());
    passed.store(environmentRestored, std::memory_order_release);
  };

  std::thread first(worker, softfloat_round_min,
                    softfloat_tininess_beforeRounding, 32,
                    softfloat_flag_invalid, std::ref(firstPassed));
  std::thread second(worker, softfloat_round_max,
                     softfloat_tininess_afterRounding, 64,
                     softfloat_flag_overflow, std::ref(secondPassed));
  first.join();
  second.join();

  EXPECT_TRUE(firstPassed.load(std::memory_order_acquire));
  EXPECT_TRUE(secondPassed.load(std::memory_order_acquire));
}

} // namespace
