//===- NumericRawRounderTest.cpp - Independent reduced-float oracle -------===//

#include "Wafer/Target/FormalNumeric.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <string>
#include <utility>

namespace {

using namespace wafer;

constexpr ModelProfileId kModelProfile = ModelProfileId::formalDeterministic();

struct IndependentRoundResult {
  uint64_t bits;
  FormalNumericExceptionFlags flags;
};

IndependentRoundResult roundF32ToReduced(uint32_t source,
                                         LogicalFormat destination,
                                         NumericRoundingMode mode) {
  const unsigned discardedBits = destination == LogicalFormat::BF16 ? 16U : 13U;
  const uint32_t destinationMask = ~((UINT32_C(1) << discardedBits) - 1U);
  const uint32_t discardedMask = ~destinationMask;
  const bool negative = (source >> 31) != 0;
  const uint32_t exponent = (source >> 23) & UINT32_C(0xff);
  const uint32_t fraction = source & UINT32_C(0x7fffff);

  FormalNumericExceptionFlags flags;
  if (exponent == UINT32_C(0xff) && fraction != 0) {
    flags.invalid = (fraction & UINT32_C(0x400000)) == 0;
    return {destination == LogicalFormat::BF16 ? UINT64_C(0x7fc0)
                                               : UINT64_C(0x7fc00000),
            flags};
  }
  if (exponent == UINT32_C(0xff))
    return {destination == LogicalFormat::BF16 ? source >> 16
                                               : source & destinationMask,
            flags};

  uint32_t rounded = source & destinationMask;
  const uint32_t discarded = source & discardedMask;
  flags.inexact = discarded != 0;
  bool increment = false;
  if (discarded != 0) {
    switch (mode) {
    case NumericRoundingMode::NearestEven: {
      const uint32_t halfway = UINT32_C(1) << (discardedBits - 1);
      increment = discarded > halfway ||
                  (discarded == halfway &&
                   ((rounded >> discardedBits) & UINT32_C(1)) != 0);
      break;
    }
    case NumericRoundingMode::TowardZero:
      break;
    case NumericRoundingMode::TowardPositive:
      increment = !negative;
      break;
    case NumericRoundingMode::TowardNegative:
      increment = negative;
      break;
    case NumericRoundingMode::Stochastic:
      llvm_unreachable("independent oracle accepts deterministic modes only");
    }
  }
  if (increment)
    rounded += UINT32_C(1) << discardedBits;

  const unsigned destinationPrecision =
      destination == LogicalFormat::BF16 ? 8U : 11U;
  const uint32_t magnitude = source & UINT32_C(0x7fffffff);
  const uint32_t largestFiniteDestination = destination == LogicalFormat::BF16
                                                ? UINT32_C(0x7f7f0000)
                                                : UINT32_C(0x7f7fe000);
  // Directed overflow may return max-finite rather than infinity, so compare
  // the exact source magnitude with the destination's largest finite value.
  flags.overflow = flags.inexact && exponent != UINT32_C(0xff) &&
                   magnitude > largestFiniteDestination;

  // Tininess-after first rounds to destination precision with an unbounded
  // exponent. BF16/TF32 keep the F32 exponent range, which gives a compact
  // independent threshold proof in F32 subnormal integer units.
  bool tinyAfterPrecisionRounding = false;
  if (exponent == 0 && fraction != 0) {
    const uint32_t minimumNormal = UINT32_C(0x00800000);
    const uint32_t greatestTinyAtDestinationPrecision =
        minimumNormal - (UINT32_C(1) << (23U - destinationPrecision));
    switch (mode) {
    case NumericRoundingMode::NearestEven: {
      const uint32_t midpointToMinimumNormal =
          minimumNormal - (UINT32_C(1) << (22U - destinationPrecision));
      tinyAfterPrecisionRounding = fraction < midpointToMinimumNormal;
      break;
    }
    case NumericRoundingMode::TowardZero:
      tinyAfterPrecisionRounding = true;
      break;
    case NumericRoundingMode::TowardPositive:
      tinyAfterPrecisionRounding =
          negative || fraction <= greatestTinyAtDestinationPrecision;
      break;
    case NumericRoundingMode::TowardNegative:
      tinyAfterPrecisionRounding =
          !negative || fraction <= greatestTinyAtDestinationPrecision;
      break;
    case NumericRoundingMode::Stochastic:
      llvm_unreachable("independent oracle accepts deterministic modes only");
    }
  }
  flags.underflow = flags.inexact && tinyAfterPrecisionRounding;
  return {destination == LogicalFormat::BF16 ? rounded >> 16 : rounded, flags};
}

ResolvedNumericCommand resolve(uint16_t opcode, LogicalFormat sourceFormat,
                               LogicalFormat destinationFormat,
                               NumericRoundingMode mode) {
  NumericTensorKey source = llvm::cantFail(NumericTensorKey::create(
      sourceFormat, PhysicalTensorLayout::Tensor, {1}));
  NumericTensorKey destination = llvm::cantFail(NumericTensorKey::create(
      destinationFormat, PhysicalTensorLayout::Tensor, {1}));
  llvm::Expected<NumericCommandKey> key = NumericCommandKey::createCTConvert(
      opcode, std::move(source), std::move(destination),
      NumericConvertParameter::roundingMode(mode));
  NumericCommandKey exact = llvm::cantFail(std::move(key));
  llvm::Expected<ResolvedNumericCommand> command =
      resolveNumericCommand(kModelProfile, std::move(exact));
  ResolvedNumericCommand resolved = llvm::cantFail(std::move(command));
  EXPECT_TRUE(resolved.isSupported());
  return resolved;
}

ResolvedNumericCommand resolveMultiply(LogicalFormat format) {
  NumericTensorKey input = llvm::cantFail(
      NumericTensorKey::create(format, PhysicalTensorLayout::Tensor, {1}));
  llvm::Expected<NumericCommandKey> key =
      NumericCommandKey::createCTElementwise(NumericElementwiseOperation::Mul,
                                             {input, input}, input);
  NumericCommandKey exact = llvm::cantFail(std::move(key));
  ResolvedNumericCommand resolved =
      llvm::cantFail(resolveNumericCommand(kModelProfile, std::move(exact)));
  EXPECT_TRUE(resolved.isSupported());
  return resolved;
}

ResolvedNumericCommand resolveGemm(LogicalFormat format) {
  NumericTensorKey lhs = llvm::cantFail(
      NumericTensorKey::create(format, PhysicalTensorLayout::Cx, {1, 1}));
  NumericTensorKey rhs = llvm::cantFail(
      NumericTensorKey::create(format, PhysicalTensorLayout::NCx, {1, 1}));
  NumericTensorKey destination = llvm::cantFail(
      NumericTensorKey::create(format, PhysicalTensorLayout::Cx, {1, 1}));
  NumericGemmAxes axes = llvm::cantFail(getCanonicalNumericGemmAxes(2));
  NumericCommandKey key = llvm::cantFail(NumericCommandKey::createNEGemm(
      std::move(lhs), std::move(rhs), std::move(destination), 1, 1, 1, 1,
      std::move(axes)));
  ResolvedNumericCommand resolved =
      llvm::cantFail(resolveNumericCommand(kModelProfile, std::move(key)));
  EXPECT_TRUE(resolved.isSupported());
  return resolved;
}

void compareOne(const ResolvedNumericCommand &command, NumericRoundingMode mode,
                LogicalFormat destination, uint32_t source) {
  const IndependentRoundResult expected =
      roundF32ToReduced(source, destination, mode);
  FormalNumericExecutionContext context;
  llvm::Expected<FormalNumericResult> actual =
      executeFormalConvert(context, command, {LogicalFormat::F32, source});
  ASSERT_TRUE(static_cast<bool>(actual))
      << (actual ? std::string() : llvm::toString(actual.takeError()));
  EXPECT_EQ(actual->value.format, destination)
      << "source=0x" << std::hex << source;
  EXPECT_EQ(actual->value.bits, expected.bits)
      << "source=0x" << std::hex << source;
  EXPECT_EQ(actual->flags, expected.flags) << "source=0x" << std::hex << source;
}

TEST(NumericRawRounderTest,
     BF16AndTF32MatchIndependentIntegerOracleForFourModes) {
  constexpr std::array<NumericRoundingMode, 4> modes = {
      NumericRoundingMode::NearestEven, NumericRoundingMode::TowardZero,
      NumericRoundingMode::TowardPositive, NumericRoundingMode::TowardNegative};
  constexpr std::array<uint32_t, 24> boundary = {
      UINT32_C(0x00000000), UINT32_C(0x80000000), UINT32_C(0x00000001),
      UINT32_C(0x80000001), UINT32_C(0x007fffff), UINT32_C(0x807fffff),
      UINT32_C(0x00800000), UINT32_C(0x80800000), UINT32_C(0x3f7fffff),
      UINT32_C(0xbf7fffff), UINT32_C(0x3f800000), UINT32_C(0xbf800000),
      UINT32_C(0x3f801000), UINT32_C(0xbf801000), UINT32_C(0x7f7fffff),
      UINT32_C(0xff7fffff), UINT32_C(0x7f800000), UINT32_C(0xff800000),
      UINT32_C(0x7fc00000), UINT32_C(0xffc00000), UINT32_C(0x7f800001),
      UINT32_C(0xff800001), UINT32_C(0x7fffffff), UINT32_C(0xffffffff)};
  struct Destination {
    LogicalFormat format;
    uint16_t opcode;
  };
  constexpr Destination destinations[] = {
      {LogicalFormat::BF16, /*fp32_bf16=*/167},
      {LogicalFormat::TF32, /*fp32_tf32=*/168},
  };

  for (const Destination &destination : destinations) {
    for (NumericRoundingMode mode : modes) {
      const ResolvedNumericCommand command = resolve(
          destination.opcode, LogicalFormat::F32, destination.format, mode);
      for (uint32_t source : boundary)
        compareOne(command, mode, destination.format, source);

      uint32_t state = UINT32_C(0x9e3779b9);
      for (unsigned index = 0; index < 65536; ++index) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        compareOne(command, mode, destination.format, state);
      }
    }
  }
}

TEST(NumericRawRounderTest,
     ExactDyadicProofCompletesMulAndFMAUnderflowThresholds) {
  const ResolvedNumericCommand f16Multiply =
      resolveMultiply(LogicalFormat::F16);
  llvm::Expected<FormalNumericResult> f16Threshold =
      evaluateFormalElementwiseLLVM(f16Multiply,
                                    {{LogicalFormat::F16, UINT64_C(0x0400)},
                                     {LogicalFormat::F16, UINT64_C(0x3bff)}});
  ASSERT_TRUE(static_cast<bool>(f16Threshold))
      << (f16Threshold ? std::string()
                       : llvm::toString(f16Threshold.takeError()));
  // min-normal * largest-below-one is exactly min-normal - 2^-25.
  // Precision-only RNE leaves that tiny value unchanged, then finite-range
  // subnormal encoding rounds to min-normal. The final class is normal but
  // tininess-after plus inexact must still raise underflow.
  EXPECT_EQ(f16Threshold->value.bits, UINT64_C(0x0400));
  EXPECT_TRUE(f16Threshold->flags.inexact);
  EXPECT_TRUE(f16Threshold->flags.underflow);

  llvm::Expected<FormalNumericResult> f16ExactSubnormal =
      evaluateFormalElementwiseLLVM(f16Multiply,
                                    {{LogicalFormat::F16, UINT64_C(0x0400)},
                                     {LogicalFormat::F16, UINT64_C(0x3800)}});
  ASSERT_TRUE(static_cast<bool>(f16ExactSubnormal));
  EXPECT_EQ(f16ExactSubnormal->value.bits, UINT64_C(0x0200));
  EXPECT_FALSE(f16ExactSubnormal->flags.inexact);
  EXPECT_FALSE(f16ExactSubnormal->flags.underflow);

  const ResolvedNumericCommand f32Gemm = resolveGemm(LogicalFormat::F32);
  llvm::Expected<FormalNumericResult> f32Threshold =
      evaluateFormalGemmFusedMultiplyAdd(
          f32Gemm, {LogicalFormat::F32, UINT64_C(0x00800000)},
          {LogicalFormat::F32, UINT64_C(0x3f7fffff)},
          {LogicalFormat::F32, UINT64_C(0x00000000)});
  ASSERT_TRUE(static_cast<bool>(f32Threshold))
      << (f32Threshold ? std::string()
                       : llvm::toString(f32Threshold.takeError()));
  EXPECT_EQ(f32Threshold->value.bits, UINT64_C(0x00800000));
  EXPECT_TRUE(f32Threshold->flags.inexact);
  EXPECT_TRUE(f32Threshold->flags.underflow);

  llvm::Expected<FormalNumericResult> f32ExactNormal =
      evaluateFormalGemmFusedMultiplyAdd(
          f32Gemm, {LogicalFormat::F32, UINT64_C(0x00800000)},
          {LogicalFormat::F32, UINT64_C(0x3f800000)},
          {LogicalFormat::F32, UINT64_C(0x00000000)});
  ASSERT_TRUE(static_cast<bool>(f32ExactNormal));
  EXPECT_EQ(f32ExactNormal->value.bits, UINT64_C(0x00800000));
  EXPECT_FALSE(f32ExactNormal->flags.inexact);
  EXPECT_FALSE(f32ExactNormal->flags.underflow);
}

} // namespace
