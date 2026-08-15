//===- FormalNumericTest.cpp - Formal numeric execution tests ------------===//

#include "Wafer/Target/FormalNumeric.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using wafer::FormalNumericErrorCode;
using wafer::FormalNumericExceptionFlags;
using wafer::FormalNumericExecutionContext;
using wafer::FormalNumericResult;
using wafer::LogicalFormat;
using wafer::LogicalFormatCategory;
using wafer::NumericCommandKey;
using wafer::NumericConvertParameter;
using wafer::NumericElementwiseOperation;
using wafer::NumericRoundingMode;
using wafer::NumericSemanticsProfile;
using wafer::NumericTensorKey;
using wafer::PhysicalTensorLayout;
using wafer::RawLogicalValue;
using wafer::ResolvedNumericCommand;
using wafer::TargetConvertParameterKind;

constexpr wafer::ModelProfileId kModelProfile =
    wafer::ModelProfileId::formalDeterministic();

static_assert(std::is_default_constructible_v<FormalNumericExecutionContext>);
static_assert(!std::is_copy_constructible_v<FormalNumericExecutionContext>);
static_assert(!std::is_copy_assignable_v<FormalNumericExecutionContext>);
static_assert(std::is_move_constructible_v<FormalNumericExecutionContext>);
static_assert(std::is_move_assignable_v<FormalNumericExecutionContext>);

std::optional<ResolvedNumericCommand>
getResolution(uint16_t opcode,
              NumericRoundingMode mode = NumericRoundingMode::NearestEven) {
  const wafer::TargetConvertRoute *route =
      wafer::findTargetConvertRoute(opcode);
  if (!route) {
    ADD_FAILURE() << "missing test CT convert route " << opcode;
    return std::nullopt;
  }

  std::optional<NumericConvertParameter> parameter;
  if (route->parameterKind == TargetConvertParameterKind::RoundingMode)
    parameter = NumericConvertParameter::roundingMode(mode);
  else if (route->parameterKind == TargetConvertParameterKind::ZeroPoint) {
    ADD_FAILURE() << "test requested unsupported zero-point route " << opcode;
    return std::nullopt;
  }

  llvm::Expected<NumericTensorKey> source = NumericTensorKey::create(
      route->source, PhysicalTensorLayout::Tensor, {1});
  llvm::Expected<NumericTensorKey> destination = NumericTensorKey::create(
      route->destination, PhysicalTensorLayout::Tensor, {1});
  if (!source || !destination) {
    ADD_FAILURE() << (source ? llvm::toString(destination.takeError())
                             : llvm::toString(source.takeError()));
    return std::nullopt;
  }
  llvm::Expected<NumericCommandKey> key = NumericCommandKey::createCTConvert(
      opcode, std::move(*source), std::move(*destination), parameter);
  if (!key) {
    ADD_FAILURE() << llvm::toString(key.takeError());
    return std::nullopt;
  }
  llvm::Expected<ResolvedNumericCommand> resolution =
      wafer::resolveNumericCommand(kModelProfile, std::move(*key));
  if (!resolution) {
    ADD_FAILURE() << llvm::toString(resolution.takeError());
    return std::nullopt;
  }
  if (!resolution->isSupported() || !resolution->getSemantics()) {
    ADD_FAILURE() << "test requested a non-supported numeric resolution";
    return std::nullopt;
  }
  return std::move(*resolution);
}

NumericTensorKey makeTensor(LogicalFormat format, PhysicalTensorLayout layout,
                            std::vector<uint64_t> shape) {
  return llvm::cantFail(
      NumericTensorKey::create(format, layout, std::move(shape)));
}

std::optional<ResolvedNumericCommand>
getElementwiseResolution(NumericElementwiseOperation operation,
                         LogicalFormat inputFormat) {
  const LogicalFormat destinationFormat =
      wafer::isNumericElementwiseRelation(operation) ? LogicalFormat::Bool
                                                     : inputFormat;
  const NumericTensorKey input =
      makeTensor(inputFormat, PhysicalTensorLayout::Tensor, {1});
  std::vector<NumericTensorKey> inputs(
      wafer::getNumericElementwiseArity(operation), input);
  llvm::Expected<NumericCommandKey> key =
      NumericCommandKey::createCTElementwise(
          operation, std::move(inputs),
          makeTensor(destinationFormat, PhysicalTensorLayout::Tensor, {1}));
  if (!key) {
    ADD_FAILURE() << llvm::toString(key.takeError());
    return std::nullopt;
  }
  llvm::Expected<ResolvedNumericCommand> resolution =
      wafer::resolveNumericCommand(kModelProfile, std::move(*key));
  if (!resolution) {
    ADD_FAILURE() << llvm::toString(resolution.takeError());
    return std::nullopt;
  }
  return std::move(*resolution);
}

std::optional<ResolvedNumericCommand> getGemmResolution(LogicalFormat format) {
  llvm::Expected<wafer::NumericGemmAxes> axes =
      wafer::getCanonicalNumericGemmAxes(2);
  if (!axes) {
    ADD_FAILURE() << llvm::toString(axes.takeError());
    return std::nullopt;
  }
  llvm::Expected<NumericCommandKey> key = NumericCommandKey::createNEGemm(
      makeTensor(format, PhysicalTensorLayout::Cx, {1, 1}),
      makeTensor(format, PhysicalTensorLayout::NCx, {1, 1}),
      makeTensor(format, PhysicalTensorLayout::Cx, {1, 1}), 1, 1, 1, 1,
      std::move(*axes));
  if (!key) {
    ADD_FAILURE() << llvm::toString(key.takeError());
    return std::nullopt;
  }
  llvm::Expected<ResolvedNumericCommand> resolution =
      wafer::resolveNumericCommand(kModelProfile, std::move(*key));
  if (!resolution) {
    ADD_FAILURE() << llvm::toString(resolution.takeError());
    return std::nullopt;
  }
  return std::move(*resolution);
}

std::optional<FormalNumericResult>
expectExecute(FormalNumericExecutionContext &context,
              const ResolvedNumericCommand &command, RawLogicalValue source) {
  llvm::Expected<FormalNumericResult> result =
      wafer::executeFormalConvert(context, command, source);
  if (!result) {
    ADD_FAILURE() << llvm::toString(result.takeError());
    return std::nullopt;
  }
  return std::move(*result);
}

std::optional<FormalNumericResult>
expectElementwise(const ResolvedNumericCommand &command,
                  std::vector<RawLogicalValue> inputs) {
  llvm::Expected<FormalNumericResult> result =
      wafer::evaluateFormalElementwiseLLVM(command, inputs);
  if (!result) {
    ADD_FAILURE() << llvm::toString(result.takeError());
    return std::nullopt;
  }
  return std::move(*result);
}

std::optional<FormalNumericResult>
expectGemmFMA(const ResolvedNumericCommand &command, RawLogicalValue lhs,
              RawLogicalValue rhs, RawLogicalValue accumulator) {
  llvm::Expected<FormalNumericResult> result =
      wafer::evaluateFormalGemmFusedMultiplyAdd(command, lhs, rhs, accumulator);
  if (!result) {
    ADD_FAILURE() << llvm::toString(result.takeError());
    return std::nullopt;
  }
  return std::move(*result);
}

template <typename T>
void expectFormalError(llvm::Expected<T> result,
                       FormalNumericErrorCode expectedCode) {
  if (result) {
    ADD_FAILURE() << "expected a typed formal numeric error, got a result";
    return;
  }

  const std::string message = llvm::toString(result.takeError());
  const std::string token =
      (llvm::Twine("formal numeric ") +
       wafer::stringifyFormalNumericErrorCode(expectedCode) + ":")
          .str();
  EXPECT_NE(message.find(token), std::string::npos) << message;
}

std::optional<FormalNumericResult> executeOpcode(uint16_t opcode,
                                                 NumericRoundingMode mode,
                                                 RawLogicalValue source) {
  std::optional<ResolvedNumericCommand> command = getResolution(opcode, mode);
  if (!command)
    return std::nullopt;
  FormalNumericExecutionContext context;
  return expectExecute(context, *command, source);
}

uint64_t largestFiniteBits(const wafer::LogicalFormatDescriptor &descriptor) {
  const uint8_t fractionBits = descriptor.precisionBits - 1;
  const uint8_t semanticShift =
      descriptor.storageBits - descriptor.semanticBits;
  const uint64_t fraction = ((UINT64_C(1) << fractionBits) - UINT64_C(1))
                            << semanticShift;
  const uint64_t largestFiniteExponent =
      ((UINT64_C(1) << descriptor.exponentBits) - UINT64_C(2))
      << (fractionBits + semanticShift);
  return largestFiniteExponent | fraction;
}

TEST(FormalNumericTest, ExecutesZeroAndBoundarySmokeForAll101Rows) {
  llvm::ArrayRef<NumericSemanticsProfile> profiles =
      wafer::getRegisteredNumericCTConvertSemanticsProfiles();
  ASSERT_EQ(profiles.size(), 101u);

  size_t executedRows = 0;
  for (const NumericSemanticsProfile &semantics : profiles) {
    const wafer::TargetConvertRoute &route =
        semantics.getCTConvertKey()->getCTConvertRoute();
    SCOPED_TRACE(
        route.canonicalSpelling.str() + "/" +
        wafer::stringifyNumericRoundingMode(semantics.getRoundingMode()).str());

    std::optional<ResolvedNumericCommand> command =
        getResolution(route.opcode, semantics.getRoundingMode());
    ASSERT_TRUE(command.has_value());
    ASSERT_EQ(command->getSemantics(), &semantics);

    FormalNumericExecutionContext zeroContext;
    std::optional<FormalNumericResult> zero =
        expectExecute(zeroContext, *command, {route.source, 0});
    ASSERT_TRUE(zero.has_value());
    EXPECT_EQ(zero->value.format, route.destination);
    EXPECT_EQ(zero->value.bits, 0u);
    EXPECT_FALSE(zero->flags.any());
    EXPECT_FALSE(zeroContext.getAggregateFlags().any());

    const wafer::LogicalFormatDescriptor *sourceDescriptor =
        wafer::findLogicalFormatDescriptor(route.source);
    const wafer::LogicalFormatDescriptor *destinationDescriptor =
        wafer::findLogicalFormatDescriptor(route.destination);
    ASSERT_NE(sourceDescriptor, nullptr);
    ASSERT_NE(destinationDescriptor, nullptr);

    std::vector<uint64_t> boundaryInputs;
    if (sourceDescriptor->category ==
        LogicalFormatCategory::BinaryFloatingPoint) {
      boundaryInputs.push_back(largestFiniteBits(*sourceDescriptor));
    } else {
      ASSERT_EQ(sourceDescriptor->category,
                LogicalFormatCategory::SignedInteger);
      boundaryInputs.push_back(
          (UINT64_C(1) << (sourceDescriptor->storageBits - 1)) - 1);
      boundaryInputs.push_back(UINT64_C(1)
                               << (sourceDescriptor->storageBits - 1));
    }

    for (uint64_t bits : boundaryInputs) {
      FormalNumericExecutionContext boundaryContext;
      llvm::Expected<FormalNumericResult> boundary =
          wafer::executeFormalConvert(boundaryContext, *command,
                                      {route.source, bits});
      if (!boundary) {
        expectFormalError(std::move(boundary),
                          FormalNumericErrorCode::FloatToIntegerOutOfRange);
        EXPECT_NE(destinationDescriptor->category,
                  LogicalFormatCategory::BinaryFloatingPoint);
        EXPECT_FALSE(boundaryContext.getAggregateFlags().any());
        continue;
      }

      EXPECT_EQ(boundary->value.format, route.destination);
      llvm::Expected<RawLogicalValue> canonical = wafer::makeRawLogicalValue(
          boundary->value.format, boundary->value.bits,
          wafer::NonCanonicalEncodingPolicy::Reject);
      ASSERT_TRUE(static_cast<bool>(canonical))
          << (canonical ? std::string()
                        : llvm::toString(canonical.takeError()));
      EXPECT_EQ(boundaryContext.getAggregateFlags(), boundary->flags);
    }
    ++executedRows;
  }
  EXPECT_EQ(executedRows, 101u);
}

TEST(FormalNumericTest, FourRoundingModesDistinguishFloatToIntegerTies) {
  struct Case {
    NumericRoundingMode mode;
    uint64_t positiveExpected;
    uint64_t negativeExpected;
  };
  constexpr Case cases[] = {
      {NumericRoundingMode::NearestEven, 0x02, 0xfe},
      {NumericRoundingMode::TowardZero, 0x01, 0xff},
      {NumericRoundingMode::TowardPositive, 0x02, 0xff},
      {NumericRoundingMode::TowardNegative, 0x01, 0xfe},
  };

  for (const Case &testCase : cases) {
    SCOPED_TRACE(wafer::stringifyNumericRoundingMode(testCase.mode).str());
    std::optional<FormalNumericResult> positive = executeOpcode(
        /*fp16_int8=*/157, testCase.mode, {LogicalFormat::F16, 0x3e00});
    std::optional<FormalNumericResult> negative = executeOpcode(
        /*fp16_int8=*/157, testCase.mode, {LogicalFormat::F16, 0xbe00});
    ASSERT_TRUE(positive.has_value());
    ASSERT_TRUE(negative.has_value());
    EXPECT_EQ(positive->value.bits, testCase.positiveExpected);
    EXPECT_EQ(negative->value.bits, testCase.negativeExpected);
    EXPECT_TRUE(positive->flags.inexact);
    EXPECT_TRUE(negative->flags.inexact);
  }
}

TEST(FormalNumericTest, FourRoundingModesDistinguishIntegerToFloatTies) {
  struct Case {
    NumericRoundingMode mode;
    uint64_t positiveExpected;
    uint64_t negativeExpected;
  };
  constexpr Case cases[] = {
      {NumericRoundingMode::NearestEven, 0x6800, 0xe800},
      {NumericRoundingMode::TowardZero, 0x6800, 0xe800},
      {NumericRoundingMode::TowardPositive, 0x6801, 0xe800},
      {NumericRoundingMode::TowardNegative, 0x6800, 0xe801},
  };

  for (const Case &testCase : cases) {
    SCOPED_TRACE(wafer::stringifyNumericRoundingMode(testCase.mode).str());
    std::optional<FormalNumericResult> positive = executeOpcode(
        /*int32_fp16=*/147, testCase.mode, {LogicalFormat::I32, 0x00000801});
    std::optional<FormalNumericResult> negative = executeOpcode(
        /*int32_fp16=*/147, testCase.mode, {LogicalFormat::I32, 0xfffff7ff});
    ASSERT_TRUE(positive.has_value());
    ASSERT_TRUE(negative.has_value());
    EXPECT_EQ(positive->value.bits, testCase.positiveExpected);
    EXPECT_EQ(negative->value.bits, testCase.negativeExpected);
    EXPECT_TRUE(positive->flags.inexact);
    EXPECT_TRUE(negative->flags.inexact);
  }
}

TEST(FormalNumericTest, BF16AndCompactTF32UseCanonicalRawEncodings) {
  std::optional<FormalNumericResult> bf16ToTF32 = executeOpcode(
      /*bf16_tf32=*/156, NumericRoundingMode::NearestEven,
      {LogicalFormat::BF16, 0x3fc0});
  ASSERT_TRUE(bf16ToTF32.has_value());
  EXPECT_EQ(bf16ToTF32->value.format, LogicalFormat::TF32);
  EXPECT_EQ(bf16ToTF32->value.bits, UINT64_C(0x3fc00000));
  EXPECT_FALSE(bf16ToTF32->flags.any());

  std::optional<FormalNumericResult> tf32ToBF16 = executeOpcode(
      /*tf32_bf16=*/173, NumericRoundingMode::NearestEven,
      {LogicalFormat::TF32, UINT64_C(0x3fc00000)});
  ASSERT_TRUE(tf32ToBF16.has_value());
  EXPECT_EQ(tf32ToBF16->value.format, LogicalFormat::BF16);
  EXPECT_EQ(tf32ToBF16->value.bits, UINT64_C(0x3fc0));
  EXPECT_FALSE(tf32ToBF16->flags.any());

  struct Case {
    NumericRoundingMode mode;
    uint64_t positiveExpected;
    uint64_t negativeExpected;
  };
  constexpr Case cases[] = {
      {NumericRoundingMode::NearestEven, 0x3f800000, 0xbf800000},
      {NumericRoundingMode::TowardZero, 0x3f800000, 0xbf800000},
      {NumericRoundingMode::TowardPositive, 0x3f802000, 0xbf800000},
      {NumericRoundingMode::TowardNegative, 0x3f800000, 0xbf802000},
  };
  for (const Case &testCase : cases) {
    std::optional<FormalNumericResult> positive = executeOpcode(
        /*fp32_tf32=*/168, testCase.mode,
        {LogicalFormat::F32, UINT64_C(0x3f801000)});
    std::optional<FormalNumericResult> negative = executeOpcode(
        /*fp32_tf32=*/168, testCase.mode,
        {LogicalFormat::F32, UINT64_C(0xbf801000)});
    ASSERT_TRUE(positive.has_value());
    ASSERT_TRUE(negative.has_value());
    EXPECT_EQ(positive->value.bits, testCase.positiveExpected);
    EXPECT_EQ(negative->value.bits, testCase.negativeExpected);
    EXPECT_EQ(positive->value.bits & UINT64_C(0x1fff), 0u);
    EXPECT_EQ(negative->value.bits & UINT64_C(0x1fff), 0u);
    EXPECT_TRUE(positive->flags.inexact);
    EXPECT_TRUE(negative->flags.inexact);
  }
}

TEST(FormalNumericTest, PreservesSignedZeroExceptAtIntegerDestination) {
  std::optional<FormalNumericResult> f32ToF16 = executeOpcode(
      /*fp32_fp16=*/166, NumericRoundingMode::TowardPositive,
      {LogicalFormat::F32, UINT64_C(0x80000000)});
  ASSERT_TRUE(f32ToF16.has_value());
  EXPECT_EQ(f32ToF16->value.bits, UINT64_C(0x8000));
  EXPECT_FALSE(f32ToF16->flags.any());

  std::optional<FormalNumericResult> bf16ToTF32 = executeOpcode(
      /*bf16_tf32=*/156, NumericRoundingMode::NearestEven,
      {LogicalFormat::BF16, UINT64_C(0x8000)});
  ASSERT_TRUE(bf16ToTF32.has_value());
  EXPECT_EQ(bf16ToTF32->value.bits, UINT64_C(0x80000000));
  EXPECT_FALSE(bf16ToTF32->flags.any());

  std::optional<FormalNumericResult> tf32ToI32 = executeOpcode(
      /*tf32_int32=*/171, NumericRoundingMode::TowardNegative,
      {LogicalFormat::TF32, UINT64_C(0x80000000)});
  ASSERT_TRUE(tf32ToI32.has_value());
  EXPECT_EQ(tf32ToI32->value.bits, 0u);
  EXPECT_FALSE(tf32ToI32->flags.any());
}

TEST(FormalNumericTest, CanonicalizesNaNsAndFlagsOnlySignalingNaN) {
  std::optional<ResolvedNumericCommand> f32ToF16 =
      getResolution(/*fp32_fp16=*/166, NumericRoundingMode::NearestEven);
  ASSERT_TRUE(f32ToF16.has_value());

  FormalNumericExecutionContext quietContext;
  std::optional<FormalNumericResult> quiet = expectExecute(
      quietContext, *f32ToF16, {LogicalFormat::F32, UINT64_C(0xffc12345)});
  ASSERT_TRUE(quiet.has_value());
  EXPECT_EQ(quiet->value.bits, UINT64_C(0x7e00));
  EXPECT_FALSE(quiet->flags.any());
  EXPECT_FALSE(quietContext.getAggregateFlags().any());

  FormalNumericExecutionContext signalingContext;
  std::optional<FormalNumericResult> signaling = expectExecute(
      signalingContext, *f32ToF16, {LogicalFormat::F32, UINT64_C(0xff812345)});
  ASSERT_TRUE(signaling.has_value());
  EXPECT_EQ(signaling->value.bits, UINT64_C(0x7e00));
  EXPECT_TRUE(signaling->flags.invalid);
  EXPECT_TRUE(signalingContext.getAggregateFlags().invalid);

  std::optional<ResolvedNumericCommand> tf32ToF32 =
      getResolution(/*tf32_fp32=*/174);
  ASSERT_TRUE(tf32ToF32.has_value());
  FormalNumericExecutionContext tf32Context;
  std::optional<FormalNumericResult> tf32Signaling = expectExecute(
      tf32Context, *tf32ToF32, {LogicalFormat::TF32, UINT64_C(0xff802000)});
  ASSERT_TRUE(tf32Signaling.has_value());
  EXPECT_EQ(tf32Signaling->value.bits, UINT64_C(0x7fc00000));
  EXPECT_TRUE(tf32Signaling->flags.invalid);

  FormalNumericExecutionContext tf32QuietContext;
  std::optional<FormalNumericResult> tf32Quiet =
      expectExecute(tf32QuietContext, *tf32ToF32,
                    {LogicalFormat::TF32, UINT64_C(0xffc02000)});
  ASSERT_TRUE(tf32Quiet.has_value());
  EXPECT_EQ(tf32Quiet->value.bits, UINT64_C(0x7fc00000));
  EXPECT_FALSE(tf32Quiet->flags.invalid);
}

TEST(FormalNumericTest, UsesGradualUnderflowAndTininessAfterRounding) {
  std::optional<FormalNumericResult> exactSubnormal = executeOpcode(
      /*fp32_fp16=*/166, NumericRoundingMode::NearestEven,
      {LogicalFormat::F32, UINT64_C(0x33800000)});
  ASSERT_TRUE(exactSubnormal.has_value());
  EXPECT_EQ(exactSubnormal->value.bits, UINT64_C(0x0001));
  EXPECT_FALSE(exactSubnormal->flags.underflow);
  EXPECT_FALSE(exactSubnormal->flags.inexact);

  std::optional<FormalNumericResult> tinyTie = executeOpcode(
      /*fp32_fp16=*/166, NumericRoundingMode::NearestEven,
      {LogicalFormat::F32, UINT64_C(0x33000000)});
  ASSERT_TRUE(tinyTie.has_value());
  EXPECT_EQ(tinyTie->value.bits, UINT64_C(0x0000));
  EXPECT_TRUE(tinyTie->flags.underflow);
  EXPECT_TRUE(tinyTie->flags.inexact);

  // Exactly halfway between the largest encoded half subnormal and the
  // smallest half normal. The final finite-range encoding is normal, but the
  // result rounded first to 11-bit precision with an unbounded exponent is
  // still tiny. IEEE tininess-after therefore raises underflow.
  std::optional<FormalNumericResult> borderlineTinyRoundsNormal = executeOpcode(
      /*fp32_fp16=*/166, NumericRoundingMode::NearestEven,
      {LogicalFormat::F32, UINT64_C(0x387fe000)});
  ASSERT_TRUE(borderlineTinyRoundsNormal.has_value());
  EXPECT_EQ(borderlineTinyRoundsNormal->value.bits, UINT64_C(0x0400));
  EXPECT_TRUE(borderlineTinyRoundsNormal->flags.underflow);
  EXPECT_TRUE(borderlineTinyRoundsNormal->flags.inexact);

  // At the midpoint between that unbounded-precision tiny value and minimum
  // normal, ties-to-even selects minimum normal before exponent-range
  // encoding. The final bits are the same, but tininess-after is now false.
  std::optional<FormalNumericResult> nonTinyRoundsNormal = executeOpcode(
      /*fp32_fp16=*/166, NumericRoundingMode::NearestEven,
      {LogicalFormat::F32, UINT64_C(0x387ff000)});
  ASSERT_TRUE(nonTinyRoundsNormal.has_value());
  EXPECT_EQ(nonTinyRoundsNormal->value.bits, UINT64_C(0x0400));
  EXPECT_FALSE(nonTinyRoundsNormal->flags.underflow);
  EXPECT_TRUE(nonTinyRoundsNormal->flags.inexact);
}

TEST(FormalNumericTest, ExactOverflowRangeFlagsBothSignsForEveryRoundingMode) {
  struct Case {
    NumericRoundingMode mode;
    uint64_t positiveExpected;
    uint64_t negativeExpected;
  };
  constexpr Case cases[] = {
      {NumericRoundingMode::NearestEven, 0x7c00, 0xfc00},
      {NumericRoundingMode::TowardZero, 0x7bff, 0xfbff},
      {NumericRoundingMode::TowardPositive, 0x7c00, 0xfbff},
      {NumericRoundingMode::TowardNegative, 0x7bff, 0xfc00},
  };
  struct SourceCase {
    uint16_t opcode;
    RawLogicalValue positive;
    RawLogicalValue negative;
  };
  constexpr SourceCase sources[] = {
      {/*int32_fp16=*/147,
       {LogicalFormat::I32, UINT64_C(0x7fffffff)},
       {LogicalFormat::I32, UINT64_C(0x80000000)}},
      {/*fp32_fp16=*/166,
       {LogicalFormat::F32, UINT64_C(0x7f7fffff)},
       {LogicalFormat::F32, UINT64_C(0xff7fffff)}},
  };

  for (const SourceCase &source : sources) {
    for (const Case &testCase : cases) {
      SCOPED_TRACE(llvm::Twine(source.opcode).str() + "/" +
                   wafer::stringifyNumericRoundingMode(testCase.mode).str());
      std::optional<FormalNumericResult> positive =
          executeOpcode(source.opcode, testCase.mode, source.positive);
      std::optional<FormalNumericResult> negative =
          executeOpcode(source.opcode, testCase.mode, source.negative);
      ASSERT_TRUE(positive.has_value());
      ASSERT_TRUE(negative.has_value());
      EXPECT_EQ(positive->value.bits, testCase.positiveExpected);
      EXPECT_EQ(negative->value.bits, testCase.negativeExpected);
      EXPECT_TRUE(positive->flags.overflow);
      EXPECT_TRUE(positive->flags.inexact);
      EXPECT_TRUE(negative->flags.overflow);
      EXPECT_TRUE(negative->flags.inexact);
    }
  }
}

TEST(FormalNumericTest, HonorsSignedIntegerWidthsWithoutHostCasts) {
  std::optional<FormalNumericResult> minusOne = executeOpcode(
      /*int16_fp32=*/145, NumericRoundingMode::TowardPositive,
      {LogicalFormat::I16, UINT64_C(0xffff)});
  ASSERT_TRUE(minusOne.has_value());
  EXPECT_EQ(minusOne->value.bits, UINT64_C(0xbf800000));
  EXPECT_FALSE(minusOne->flags.any());

  std::optional<FormalNumericResult> int32Minimum = executeOpcode(
      /*int32_fp32=*/149, NumericRoundingMode::NearestEven,
      {LogicalFormat::I32, UINT64_C(0x80000000)});
  ASSERT_TRUE(int32Minimum.has_value());
  EXPECT_EQ(int32Minimum->value.bits, UINT64_C(0xcf000000));
  EXPECT_FALSE(int32Minimum->flags.any());

  std::optional<FormalNumericResult> fp16MaximumToI32 = executeOpcode(
      /*fp16_int32=*/159, NumericRoundingMode::NearestEven,
      {LogicalFormat::F16, UINT64_C(0x7bff)});
  ASSERT_TRUE(fp16MaximumToI32.has_value());
  EXPECT_EQ(fp16MaximumToI32->value.bits, UINT64_C(0x0000ffe0));
  EXPECT_FALSE(fp16MaximumToI32->flags.any());
}

TEST(FormalNumericTest, FloatToIntegerRejectsWithoutChangingContext) {
  std::optional<ResolvedNumericCommand> f32ToI8 =
      getResolution(/*fp32_int8=*/163, NumericRoundingMode::NearestEven);
  ASSERT_TRUE(f32ToI8.has_value());

  for (uint64_t bits : {UINT64_C(0x7f800000), UINT64_C(0xff800000),
                        UINT64_C(0x7fc01234), UINT64_C(0x7f801234)}) {
    FormalNumericExecutionContext context;
    const FormalNumericExceptionFlags before = context.getAggregateFlags();
    expectFormalError(wafer::executeFormalConvert(context, *f32ToI8,
                                                  {LogicalFormat::F32, bits}),
                      FormalNumericErrorCode::FloatToIntegerNonFinite);
    EXPECT_EQ(context.getAggregateFlags(), before);
  }

  FormalNumericExecutionContext overflowContext;
  const FormalNumericExceptionFlags beforeOverflow =
      overflowContext.getAggregateFlags();
  expectFormalError(
      wafer::executeFormalConvert(overflowContext, *f32ToI8,
                                  {LogicalFormat::F32, UINT64_C(0x43000000)}),
      FormalNumericErrorCode::FloatToIntegerOutOfRange);
  EXPECT_EQ(overflowContext.getAggregateFlags(), beforeOverflow);

  // 127.5 is valid under toward-zero but is rejected after nearest-even.
  FormalNumericExecutionContext nearestContext;
  expectFormalError(
      wafer::executeFormalConvert(nearestContext, *f32ToI8,
                                  {LogicalFormat::F32, UINT64_C(0x42ff0000)}),
      FormalNumericErrorCode::FloatToIntegerOutOfRange);
  EXPECT_FALSE(nearestContext.getAggregateFlags().any());

  std::optional<ResolvedNumericCommand> towardZero =
      getResolution(/*fp32_int8=*/163, NumericRoundingMode::TowardZero);
  ASSERT_TRUE(towardZero.has_value());
  FormalNumericExecutionContext towardZeroContext;
  std::optional<FormalNumericResult> accepted =
      expectExecute(towardZeroContext, *towardZero,
                    {LogicalFormat::F32, UINT64_C(0x42ff0000)});
  ASSERT_TRUE(accepted.has_value());
  EXPECT_EQ(accepted->value.bits, UINT64_C(0x7f));
  EXPECT_TRUE(accepted->flags.inexact);
  EXPECT_FALSE(accepted->flags.invalid);
}

TEST(FormalNumericTest, ContextsAreInvocationLocalAndAggregateFlags) {
  std::optional<ResolvedNumericCommand> int32ToF16 =
      getResolution(/*int32_fp16=*/147, NumericRoundingMode::NearestEven);
  std::optional<ResolvedNumericCommand> f32ToI8 =
      getResolution(/*fp32_int8=*/163, NumericRoundingMode::NearestEven);
  ASSERT_TRUE(int32ToF16.has_value());
  ASSERT_TRUE(f32ToI8.has_value());

  FormalNumericExecutionContext first;
  std::optional<FormalNumericResult> overflow = expectExecute(
      first, *int32ToF16, {LogicalFormat::I32, UINT64_C(0x7fffffff)});
  ASSERT_TRUE(overflow.has_value());
  EXPECT_TRUE(first.getAggregateFlags().overflow);
  EXPECT_TRUE(first.getAggregateFlags().inexact);
  EXPECT_FALSE(first.getAggregateFlags().invalid);
  EXPECT_FALSE(first.getAggregateFlags().divByZero);

  const FormalNumericExceptionFlags flagsBeforeFailure =
      first.getAggregateFlags();
  expectFormalError(
      wafer::executeFormalConvert(first, *int32ToF16, {LogicalFormat::I16, 0}),
      FormalNumericErrorCode::SourceFormatMismatch);
  EXPECT_EQ(first.getAggregateFlags(), flagsBeforeFailure);

  FormalNumericExecutionContext second;
  std::optional<FormalNumericResult> exact =
      expectExecute(second, *int32ToF16, {LogicalFormat::I32, 0});
  ASSERT_TRUE(exact.has_value());
  EXPECT_FALSE(second.getAggregateFlags().any());

  expectFormalError(
      wafer::executeFormalConvert(first, *f32ToI8,
                                  {LogicalFormat::F32, UINT64_C(0x7f800000)}),
      FormalNumericErrorCode::FloatToIntegerNonFinite);
  EXPECT_EQ(first.getAggregateFlags(), flagsBeforeFailure);
  EXPECT_FALSE(second.getAggregateFlags().any());

  first.clearAggregateFlags();
  EXPECT_FALSE(first.getAggregateFlags().any());
}

TEST(FormalNumericTest, RejectsUnsupportedResolutionWithoutChangingContext) {
  llvm::Expected<NumericTensorKey> source = NumericTensorKey::create(
      LogicalFormat::I32, PhysicalTensorLayout::Tensor, {1});
  llvm::Expected<NumericTensorKey> destination = NumericTensorKey::create(
      LogicalFormat::F16, PhysicalTensorLayout::Tensor, {1});
  ASSERT_TRUE(static_cast<bool>(source));
  ASSERT_TRUE(static_cast<bool>(destination));
  llvm::Expected<NumericCommandKey> key = NumericCommandKey::createCTConvert(
      /*int32_fp16=*/147, std::move(*source), std::move(*destination),
      NumericConvertParameter::roundingMode(NumericRoundingMode::Stochastic));
  ASSERT_TRUE(static_cast<bool>(key))
      << (key ? std::string() : llvm::toString(key.takeError()));
  llvm::Expected<ResolvedNumericCommand> command =
      wafer::resolveNumericCommand(kModelProfile, std::move(*key));
  ASSERT_TRUE(static_cast<bool>(command))
      << (command ? std::string() : llvm::toString(command.takeError()));
  ASSERT_FALSE(command->isSupported());
  ASSERT_EQ(command->getSemantics(), nullptr);

  FormalNumericExecutionContext context;
  const FormalNumericExceptionFlags before = context.getAggregateFlags();
  expectFormalError(
      wafer::executeFormalConvert(context, *command, {LogicalFormat::I32, 0}),
      FormalNumericErrorCode::UnsupportedResolvedCommand);
  EXPECT_EQ(context.getAggregateFlags(), before);
}

TEST(FormalNumericTest, RejectsSourceMismatchAndNoncanonicalEncoding) {
  std::optional<ResolvedNumericCommand> f32ToF16 =
      getResolution(/*fp32_fp16=*/166, NumericRoundingMode::NearestEven);
  ASSERT_TRUE(f32ToF16.has_value());

  FormalNumericExecutionContext mismatchContext;
  expectFormalError(wafer::executeFormalConvert(mismatchContext, *f32ToF16,
                                                {LogicalFormat::F16, 0x3c00}),
                    FormalNumericErrorCode::SourceFormatMismatch);
  EXPECT_FALSE(mismatchContext.getAggregateFlags().any());

  std::optional<ResolvedNumericCommand> tf32ToF32 =
      getResolution(/*tf32_fp32=*/174);
  ASSERT_TRUE(tf32ToF32.has_value());
  FormalNumericExecutionContext noncanonicalContext;
  expectFormalError(
      wafer::executeFormalConvert(noncanonicalContext, *tf32ToF32,
                                  {LogicalFormat::TF32, UINT64_C(0x3f800001)}),
      FormalNumericErrorCode::InvalidSourceEncoding);
  EXPECT_FALSE(noncanonicalContext.getAggregateFlags().any());

  FormalNumericExecutionContext tooWideContext;
  expectFormalError(
      wafer::executeFormalConvert(tooWideContext, *f32ToF16,
                                  {LogicalFormat::F32, UINT64_C(0x100000000)}),
      FormalNumericErrorCode::InvalidSourceEncoding);
  EXPECT_FALSE(tooWideContext.getAggregateFlags().any());
}

TEST(FormalNumericTest,
     EffectFreeConvertUpdatesFlagsOnlyAfterSuccessfulWrapper) {
  std::optional<ResolvedNumericCommand> command =
      getResolution(/*fp32_fp16=*/166, NumericRoundingMode::NearestEven);
  ASSERT_TRUE(command.has_value());

  FormalNumericExecutionContext context;
  FormalNumericExceptionFlags prior;
  prior.overflow = true;
  context.mergeExceptionFlags(prior);
  llvm::Expected<FormalNumericResult> evaluated = wafer::evaluateFormalConvert(
      *command, {LogicalFormat::F32, UINT64_C(0x7f812345)});
  ASSERT_TRUE(static_cast<bool>(evaluated))
      << (evaluated ? std::string() : llvm::toString(evaluated.takeError()));
  EXPECT_EQ(evaluated->value.bits, UINT64_C(0x7e00));
  EXPECT_TRUE(evaluated->flags.invalid);
  EXPECT_EQ(context.getAggregateFlags(), prior);

  std::optional<FormalNumericResult> result = expectExecute(
      context, *command, {LogicalFormat::F32, UINT64_C(0x7f812345)});
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(context.getAggregateFlags().overflow);
  EXPECT_TRUE(context.getAggregateFlags().invalid);
  const FormalNumericExceptionFlags beforeFailure = context.getAggregateFlags();
  expectFormalError(
      wafer::executeFormalConvert(context, *command,
                                  {LogicalFormat::F16, UINT64_C(0x3c00)}),
      FormalNumericErrorCode::SourceFormatMismatch);
  EXPECT_EQ(context.getAggregateFlags(), beforeFailure);
}

TEST(FormalNumericTest,
     ElementwiseLLVMImplementsSignedZeroNaNAndBasicArithmeticPolicies) {
  std::optional<ResolvedNumericCommand> abs = getElementwiseResolution(
      NumericElementwiseOperation::Abs, LogicalFormat::F16);
  std::optional<ResolvedNumericCommand> neg = getElementwiseResolution(
      NumericElementwiseOperation::Neg, LogicalFormat::F16);
  std::optional<ResolvedNumericCommand> maximum = getElementwiseResolution(
      NumericElementwiseOperation::Max, LogicalFormat::F16);
  std::optional<ResolvedNumericCommand> minimum = getElementwiseResolution(
      NumericElementwiseOperation::Min, LogicalFormat::F16);
  std::optional<ResolvedNumericCommand> relu = getElementwiseResolution(
      NumericElementwiseOperation::Relu, LogicalFormat::F16);
  ASSERT_TRUE(abs && neg && maximum && minimum && relu);

  std::optional<FormalNumericResult> result =
      expectElementwise(*abs, {{LogicalFormat::F16, UINT64_C(0x8000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, 0u);
  EXPECT_FALSE(result->flags.any());
  result = expectElementwise(*neg, {{LogicalFormat::F16, 0}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, UINT64_C(0x8000));

  result =
      expectElementwise(*maximum, {{LogicalFormat::F16, UINT64_C(0x8000)},
                                   {LogicalFormat::F16, UINT64_C(0x0000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, UINT64_C(0x0000));
  result =
      expectElementwise(*maximum, {{LogicalFormat::F16, UINT64_C(0x8000)},
                                   {LogicalFormat::F16, UINT64_C(0x8000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, UINT64_C(0x8000));
  result =
      expectElementwise(*minimum, {{LogicalFormat::F16, UINT64_C(0x0000)},
                                   {LogicalFormat::F16, UINT64_C(0x8000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, UINT64_C(0x8000));
  result = expectElementwise(*relu, {{LogicalFormat::F16, UINT64_C(0x8000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, UINT64_C(0x8000));
  result = expectElementwise(*relu, {{LogicalFormat::F16, UINT64_C(0xbc00)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, 0u);

  std::optional<ResolvedNumericCommand> add = getElementwiseResolution(
      NumericElementwiseOperation::Add, LogicalFormat::F16);
  std::optional<ResolvedNumericCommand> multiply = getElementwiseResolution(
      NumericElementwiseOperation::Mul, LogicalFormat::F16);
  std::optional<ResolvedNumericCommand> divide = getElementwiseResolution(
      NumericElementwiseOperation::Div, LogicalFormat::F16);
  ASSERT_TRUE(add && multiply && divide);
  result = expectElementwise(*add, {{LogicalFormat::F16, UINT64_C(0xfe01)},
                                    {LogicalFormat::F16, UINT64_C(0x3c00)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, UINT64_C(0x7e00));
  EXPECT_FALSE(result->flags.invalid);
  result = expectElementwise(*add, {{LogicalFormat::F16, UINT64_C(0x7c01)},
                                    {LogicalFormat::F16, UINT64_C(0x3c00)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, UINT64_C(0x7e00));
  EXPECT_TRUE(result->flags.invalid);
  result =
      expectElementwise(*multiply, {{LogicalFormat::F16, UINT64_C(0x0000)},
                                    {LogicalFormat::F16, UINT64_C(0x7c00)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, UINT64_C(0x7e00));
  EXPECT_TRUE(result->flags.invalid);
  result = expectElementwise(*divide, {{LogicalFormat::F16, UINT64_C(0x3c00)},
                                       {LogicalFormat::F16, UINT64_C(0x0000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, UINT64_C(0x7c00));
  EXPECT_TRUE(result->flags.divByZero);
}

TEST(FormalNumericTest, ElementwiseLLVMImplementsRelationsAndBooleanLogic) {
  std::optional<ResolvedNumericCommand> equal = getElementwiseResolution(
      NumericElementwiseOperation::Eq, LogicalFormat::F32);
  std::optional<ResolvedNumericCommand> notEqual = getElementwiseResolution(
      NumericElementwiseOperation::Ne, LogicalFormat::F32);
  std::optional<ResolvedNumericCommand> less = getElementwiseResolution(
      NumericElementwiseOperation::Lt, LogicalFormat::F32);
  ASSERT_TRUE(equal && notEqual && less);
  std::optional<FormalNumericResult> result =
      expectElementwise(*equal, {{LogicalFormat::F32, UINT64_C(0x00000000)},
                                 {LogicalFormat::F32, UINT64_C(0x80000000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.format, LogicalFormat::Bool);
  EXPECT_EQ(result->value.bits, 1u);
  result =
      expectElementwise(*equal, {{LogicalFormat::F32, UINT64_C(0x7fc00001)},
                                 {LogicalFormat::F32, UINT64_C(0x3f800000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, 0u);
  EXPECT_FALSE(result->flags.invalid);
  result = expectElementwise(*notEqual,
                             {{LogicalFormat::F32, UINT64_C(0x7fc00001)},
                              {LogicalFormat::F32, UINT64_C(0x3f800000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, 1u);
  result =
      expectElementwise(*less, {{LogicalFormat::F32, UINT64_C(0x7f800001)},
                                {LogicalFormat::F32, UINT64_C(0x3f800000)}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, 0u);
  EXPECT_TRUE(result->flags.invalid);

  std::optional<ResolvedNumericCommand> logicNot = getElementwiseResolution(
      NumericElementwiseOperation::LogicNot, LogicalFormat::Bool);
  std::optional<ResolvedNumericCommand> logicAnd = getElementwiseResolution(
      NumericElementwiseOperation::LogicAnd, LogicalFormat::Bool);
  std::optional<ResolvedNumericCommand> logicOr = getElementwiseResolution(
      NumericElementwiseOperation::LogicOr, LogicalFormat::Bool);
  std::optional<ResolvedNumericCommand> logicXor = getElementwiseResolution(
      NumericElementwiseOperation::LogicXor, LogicalFormat::Bool);
  ASSERT_TRUE(logicNot && logicAnd && logicOr && logicXor);
  result = expectElementwise(*logicNot, {{LogicalFormat::Bool, 1}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, 0u);
  result = expectElementwise(
      *logicAnd, {{LogicalFormat::Bool, 1}, {LogicalFormat::Bool, 0}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, 0u);
  result = expectElementwise(
      *logicOr, {{LogicalFormat::Bool, 1}, {LogicalFormat::Bool, 0}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, 1u);
  result = expectElementwise(
      *logicXor, {{LogicalFormat::Bool, 1}, {LogicalFormat::Bool, 1}});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value.bits, 0u);
}

TEST(FormalNumericTest, ElementwiseLLVMRejectsWrongBackendAndOperands) {
  std::optional<ResolvedNumericCommand> exponential = getElementwiseResolution(
      NumericElementwiseOperation::Exp, LogicalFormat::F32);
  ASSERT_TRUE(exponential.has_value());
  ASSERT_TRUE(exponential->isSupported());
  expectFormalError(
      wafer::evaluateFormalElementwiseLLVM(
          *exponential, {{LogicalFormat::F32, UINT64_C(0x3f800000)}}),
      FormalNumericErrorCode::UnsupportedResolvedCommand);

  std::optional<ResolvedNumericCommand> add = getElementwiseResolution(
      NumericElementwiseOperation::Add, LogicalFormat::F16);
  ASSERT_TRUE(add.has_value());
  expectFormalError(wafer::evaluateFormalElementwiseLLVM(
                        *add, {{LogicalFormat::F16, UINT64_C(0x3c00)}}),
                    FormalNumericErrorCode::OperandCountMismatch);
  expectFormalError(wafer::evaluateFormalElementwiseLLVM(
                        *add, {{LogicalFormat::BF16, UINT64_C(0x3f80)},
                               {LogicalFormat::F16, UINT64_C(0x3c00)}}),
                    FormalNumericErrorCode::OperandFormatMismatch);
  expectFormalError(wafer::evaluateFormalElementwiseLLVM(
                        *add, {{static_cast<LogicalFormat>(255), 0},
                               {LogicalFormat::F16, UINT64_C(0x3c00)}}),
                    FormalNumericErrorCode::OperandFormatMismatch);
  expectFormalError(wafer::evaluateFormalElementwiseLLVM(
                        *add, {{LogicalFormat::F16, UINT64_C(0x10000)},
                               {LogicalFormat::F16, UINT64_C(0x3c00)}}),
                    FormalNumericErrorCode::InvalidOperandEncoding);
}

TEST(FormalNumericTest, ElementwiseBackendRegistryHasCompleteScalarCoverage) {
  size_t llvmRows = 0;
  size_t mpfrRows = 0;
  for (const NumericSemanticsProfile &semantics :
       wafer::getRegisteredNumericCTElementwiseSemanticsProfiles()) {
    const wafer::NumericCTElementwiseSemanticsKey *identity =
        semantics.getCTElementwiseKey();
    ASSERT_NE(identity, nullptr);
    std::optional<ResolvedNumericCommand> command = getElementwiseResolution(
        identity->getOperation(), identity->getInputFormat());
    ASSERT_TRUE(command.has_value());
    ASSERT_EQ(command->getSemantics(), &semantics);
    if (command->getFormalBackendKind() ==
        wafer::FormalNumericBackendKind::MPFR) {
      ++mpfrRows;
      expectFormalError(wafer::evaluateFormalElementwiseLLVM(
                            *command, std::vector<RawLogicalValue>(
                                          wafer::getNumericElementwiseArity(
                                              identity->getOperation()),
                                          {identity->getInputFormat(), 0})),
                        FormalNumericErrorCode::UnsupportedResolvedCommand);
      continue;
    }
    ++llvmRows;
    std::vector<RawLogicalValue> inputs(
        wafer::getNumericElementwiseArity(identity->getOperation()),
        {identity->getInputFormat(), 0});
    llvm::Expected<FormalNumericResult> result =
        wafer::evaluateFormalElementwiseLLVM(*command, inputs);
    ASSERT_TRUE(static_cast<bool>(result))
        << (result ? std::string() : llvm::toString(result.takeError()));
    EXPECT_EQ(result->value.format, identity->getDestinationFormat());
  }
  EXPECT_EQ(llvmRows, 55u);
  EXPECT_EQ(mpfrRows, 33u);
}

TEST(FormalNumericTest, GemmScalarPrimitivesUseF32FusedMACAndFinalRNE) {
  std::optional<ResolvedNumericCommand> f32 =
      getGemmResolution(LogicalFormat::F32);
  ASSERT_TRUE(f32.has_value());
  std::optional<FormalNumericResult> fused =
      expectGemmFMA(*f32, {LogicalFormat::F32, UINT64_C(0x3f800001)},
                    {LogicalFormat::F32, UINT64_C(0x3f800001)},
                    {LogicalFormat::F32, UINT64_C(0xbf800002)});
  ASSERT_TRUE(fused.has_value());
  EXPECT_EQ(fused->value.format, LogicalFormat::F32);
  EXPECT_EQ(fused->value.bits, UINT64_C(0x28800000));
  EXPECT_FALSE(fused->flags.any());
  llvm::Expected<FormalNumericResult> finalF32 =
      wafer::evaluateFormalGemmFinalize(*f32, fused->value);
  ASSERT_TRUE(static_cast<bool>(finalF32))
      << (finalF32 ? std::string() : llvm::toString(finalF32.takeError()));
  EXPECT_EQ(finalF32->value.bits, fused->value.bits);
  EXPECT_FALSE(finalF32->flags.any());

  std::optional<ResolvedNumericCommand> f16 =
      getGemmResolution(LogicalFormat::F16);
  ASSERT_TRUE(f16.has_value());
  fused = expectGemmFMA(*f16, {LogicalFormat::F16, UINT64_C(0x3e00)},
                        {LogicalFormat::F16, UINT64_C(0x4000)},
                        {LogicalFormat::F32, UINT64_C(0xbf800000)});
  ASSERT_TRUE(fused.has_value());
  EXPECT_EQ(fused->value.bits, UINT64_C(0x40000000));
  llvm::Expected<FormalNumericResult> finalF16 =
      wafer::evaluateFormalGemmFinalize(
          *f16, {LogicalFormat::F32, UINT64_C(0x3f801000)});
  ASSERT_TRUE(static_cast<bool>(finalF16))
      << (finalF16 ? std::string() : llvm::toString(finalF16.takeError()));
  EXPECT_EQ(finalF16->value.format, LogicalFormat::F16);
  EXPECT_EQ(finalF16->value.bits, UINT64_C(0x3c00));
  EXPECT_TRUE(finalF16->flags.inexact);

  fused = expectGemmFMA(*f16, {LogicalFormat::F16, UINT64_C(0x7c01)},
                        {LogicalFormat::F16, UINT64_C(0x3c00)},
                        {LogicalFormat::F32, UINT64_C(0x00000000)});
  ASSERT_TRUE(fused.has_value());
  EXPECT_EQ(fused->value.bits, UINT64_C(0x7fc00000));
  EXPECT_TRUE(fused->flags.invalid);

  fused = expectGemmFMA(*f32, {LogicalFormat::F32, UINT64_C(0x7f800001)},
                        {LogicalFormat::F32, UINT64_C(0x3f800000)},
                        {LogicalFormat::F32, UINT64_C(0x00000000)});
  ASSERT_TRUE(fused.has_value());
  EXPECT_EQ(fused->value.bits, UINT64_C(0x7fc00000));
  EXPECT_TRUE(fused->flags.invalid);
  expectFormalError(wafer::evaluateFormalGemmFusedMultiplyAdd(
                        *f32, {LogicalFormat::F16, UINT64_C(0x3c00)},
                        {LogicalFormat::F32, UINT64_C(0x3f800000)},
                        {LogicalFormat::F32, UINT64_C(0x00000000)}),
                    FormalNumericErrorCode::OperandFormatMismatch);
  expectFormalError(wafer::evaluateFormalGemmFinalize(
                        *f32, {LogicalFormat::F16, UINT64_C(0x0000)}),
                    FormalNumericErrorCode::OperandFormatMismatch);
}

TEST(FormalNumericTest, ExactComparatorValidatesOperandsAndIncludesFlags) {
  std::optional<ResolvedNumericCommand> f32ToF16 =
      getResolution(/*fp32_fp16=*/166, NumericRoundingMode::NearestEven);
  ASSERT_TRUE(f32ToF16.has_value());
  FormalNumericExecutionContext context;
  std::optional<FormalNumericResult> positive =
      expectExecute(context, *f32ToF16, {LogicalFormat::F32, 0});
  ASSERT_TRUE(positive.has_value());

  FormalNumericResult same = *positive;
  llvm::Expected<bool> equal =
      wafer::compareFormalNumericResultsExact(*positive, same);
  ASSERT_TRUE(static_cast<bool>(equal))
      << (equal ? std::string() : llvm::toString(equal.takeError()));
  EXPECT_TRUE(*equal);

  FormalNumericResult differentBits = same;
  differentBits.value.bits = UINT64_C(0x8000);
  equal = wafer::compareFormalNumericResultsExact(*positive, differentBits);
  ASSERT_TRUE(static_cast<bool>(equal))
      << (equal ? std::string() : llvm::toString(equal.takeError()));
  EXPECT_FALSE(*equal);

  FormalNumericResult differentFormat = same;
  differentFormat.value.format = LogicalFormat::BF16;
  equal = wafer::compareFormalNumericResultsExact(*positive, differentFormat);
  ASSERT_TRUE(static_cast<bool>(equal))
      << (equal ? std::string() : llvm::toString(equal.takeError()));
  EXPECT_FALSE(*equal);

  FormalNumericResult differentFlags = same;
  differentFlags.flags.inexact = true;
  equal = wafer::compareFormalNumericResultsExact(*positive, differentFlags);
  ASSERT_TRUE(static_cast<bool>(equal))
      << (equal ? std::string() : llvm::toString(equal.takeError()));
  EXPECT_FALSE(*equal);

  FormalNumericResult noncanonical = {
      {LogicalFormat::TF32, UINT64_C(0x3f800001)}, {}};
  expectFormalError(wafer::compareFormalNumericResultsExact(noncanonical, same),
                    FormalNumericErrorCode::InvalidComparatorOperandEncoding);
  expectFormalError(wafer::compareFormalNumericResultsExact(same, noncanonical),
                    FormalNumericErrorCode::InvalidComparatorOperandEncoding);
  expectFormalError(
      wafer::compareFormalNumericResultsExact(noncanonical, noncanonical),
      FormalNumericErrorCode::InvalidComparatorOperandEncoding);
}

} // namespace
