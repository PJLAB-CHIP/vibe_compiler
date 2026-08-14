//===- NumericOracleDifferentialTest.cpp - Independent IEEE differential --===//

#include "Wafer/Target/FormalNumeric.h"
#include "Wafer/Target/SoftFloatOracle.h"

#include "gtest/gtest.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;

constexpr ModelProfileId kModelProfile =
    ModelProfileId::formalDeterministicV1();

ResolvedNumericCommand resolveConvert(uint16_t opcode,
                                      NumericRoundingMode rounding) {
  const TargetConvertRoute *route = findTargetConvertRoute(opcode);
  EXPECT_NE(route, nullptr);
  std::optional<NumericConvertParameter> parameter;
  if (route && route->parameterKind == TargetConvertParameterKind::RoundingMode)
    parameter = NumericConvertParameter::roundingMode(rounding);
  if (!route)
    llvm::report_fatal_error("test failed to find a convert route");
  NumericTensorKey source = llvm::cantFail(NumericTensorKey::create(
      route->source, PhysicalTensorLayout::Tensor, {1}));
  NumericTensorKey destination = llvm::cantFail(NumericTensorKey::create(
      route->destination, PhysicalTensorLayout::Tensor, {1}));
  llvm::Expected<NumericCommandKey> key = NumericCommandKey::createCTConvert(
      opcode, std::move(source), std::move(destination), parameter);
  EXPECT_TRUE(static_cast<bool>(key))
      << (key ? std::string() : llvm::toString(key.takeError()));
  if (!key)
    llvm::report_fatal_error("test failed to construct a convert key");
  llvm::Expected<ResolvedNumericCommand> resolved =
      resolveNumericCommand(kModelProfile, std::move(*key));
  EXPECT_TRUE(static_cast<bool>(resolved))
      << (resolved ? std::string() : llvm::toString(resolved.takeError()));
  if (!resolved)
    llvm::report_fatal_error("test failed to resolve a convert key");
  EXPECT_TRUE(resolved->isSupported());
  return std::move(*resolved);
}

void compareOne(const ResolvedNumericCommand &command,
                NumericRoundingMode rounding, RawLogicalValue source,
                LogicalFormat destination) {
  FormalNumericExecutionContext context;
  llvm::Expected<FormalNumericResult> production =
      executeFormalConvert(context, command, source);
  ASSERT_TRUE(static_cast<bool>(production))
      << (production ? std::string() : llvm::toString(production.takeError()));

  llvm::Expected<FormalNumericResult> oracle =
      executeSoftFloatOracle({SoftFloatOracleOperation::Convert, rounding,
                              SoftFloatOracleTininess::AfterRounding,
                              destination, source, std::nullopt, std::nullopt});
  ASSERT_TRUE(static_cast<bool>(oracle))
      << (oracle ? std::string() : llvm::toString(oracle.takeError()));

  EXPECT_EQ(production->value.format, oracle->value.format)
      << "source=0x" << std::hex << source.bits << " rounding=" << std::dec
      << static_cast<unsigned>(rounding);
  EXPECT_EQ(production->value.bits, oracle->value.bits)
      << "source=0x" << std::hex << source.bits << " rounding=" << std::dec
      << static_cast<unsigned>(rounding);
  EXPECT_EQ(production->flags, oracle->flags)
      << "source=0x" << std::hex << source.bits << " rounding=" << std::dec
      << static_cast<unsigned>(rounding);
}

const llvm::fltSemantics &getAPFloatSemantics(LogicalFormat format) {
  switch (format) {
  case LogicalFormat::F16:
    return llvm::APFloat::IEEEhalf();
  case LogicalFormat::F32:
    return llvm::APFloat::IEEEsingle();
  case LogicalFormat::I8:
  case LogicalFormat::I16:
  case LogicalFormat::I32:
  case LogicalFormat::Bool:
  case LogicalFormat::U8:
  case LogicalFormat::U16:
  case LogicalFormat::U32:
  case LogicalFormat::I64:
  case LogicalFormat::U64:
  case LogicalFormat::BF16:
  case LogicalFormat::TF32:
    llvm_unreachable("arithmetic differential only supports F16 and F32");
  }
  llvm_unreachable("unknown logical format");
}

unsigned getAPFloatStorageBits(LogicalFormat format) {
  return format == LogicalFormat::F16 ? 16 : 32;
}

llvm::APFloat decodeAPFloat(LogicalFormat format, uint64_t bits) {
  return llvm::APFloat(getAPFloatSemantics(format),
                       llvm::APInt(getAPFloatStorageBits(format), bits));
}

uint64_t encodeAPFloat(const llvm::APFloat &value) {
  return value.bitcastToAPInt().getZExtValue();
}

llvm::APFloat::roundingMode
getAPFloatRoundingMode(NumericRoundingMode rounding) {
  switch (rounding) {
  case NumericRoundingMode::NearestEven:
    return llvm::APFloat::rmNearestTiesToEven;
  case NumericRoundingMode::TowardZero:
    return llvm::APFloat::rmTowardZero;
  case NumericRoundingMode::TowardPositive:
    return llvm::APFloat::rmTowardPositive;
  case NumericRoundingMode::TowardNegative:
    return llvm::APFloat::rmTowardNegative;
  case NumericRoundingMode::Stochastic:
    llvm_unreachable("stochastic mode reached deterministic differential");
  }
  llvm_unreachable("unknown numeric rounding mode");
}

FormalNumericExceptionFlags
flagsFromAPFloatStatus(llvm::APFloat::opStatus status) {
  const unsigned bits = static_cast<unsigned>(status);
  return {
      (bits & static_cast<unsigned>(llvm::APFloat::opInvalidOp)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opDivByZero)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opOverflow)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opUnderflow)) != 0,
      (bits & static_cast<unsigned>(llvm::APFloat::opInexact)) != 0,
  };
}

uint64_t canonicalQuietNaN(LogicalFormat format) {
  return format == LogicalFormat::F16 ? UINT64_C(0x7e00) : UINT64_C(0x7fc00000);
}

struct APFloatEvaluation {
  llvm::APFloat value;
  llvm::APFloat::opStatus status;
};

APFloatEvaluation squareRootWithAPFloat(llvm::APFloat input,
                                        llvm::APFloat::roundingMode rounding) {
  if (input.isNaN())
    return {llvm::APFloat::getNaN(input.getSemantics()),
            input.isSignaling() ? llvm::APFloat::opInvalidOp
                                : llvm::APFloat::opOK};
  if (input.isNegative() && !input.isZero())
    return {llvm::APFloat::getNaN(input.getSemantics()),
            llvm::APFloat::opInvalidOp};
  if (input.isZero() || input.isInfinity())
    return {std::move(input), llvm::APFloat::opOK};

  // The pinned APFloat API has no sqrt primitive. Use only APFloat operations
  // in IEEEquad for a deterministic Newton refinement, then explicitly round
  // back to the requested semantics. F16/F32 inputs widen exactly, and 16
  // iterations exceed the precision needed to decide either destination.
  llvm::APFloat radicand = input;
  bool losesInfo = false;
  llvm::APFloat::opStatus status =
      radicand.convert(llvm::APFloat::IEEEquad(),
                       llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  if (status != llvm::APFloat::opOK || losesInfo)
    llvm::report_fatal_error("failed to exactly widen sqrt differential input");

  const int exponent = ilogb(radicand);
  int initialExponent = exponent / 2;
  if (exponent > 0 && (exponent & 1))
    ++initialExponent;
  llvm::APFloat estimate =
      llvm::scalbn(llvm::APFloat::getOne(llvm::APFloat::IEEEquad()),
                   initialExponent, llvm::APFloat::rmNearestTiesToEven);
  const llvm::APFloat half =
      llvm::scalbn(llvm::APFloat::getOne(llvm::APFloat::IEEEquad()), -1,
                   llvm::APFloat::rmNearestTiesToEven);
  for (unsigned iteration = 0; iteration < 16; ++iteration) {
    llvm::APFloat quotient = radicand;
    (void)quotient.divide(estimate, llvm::APFloat::rmNearestTiesToEven);
    llvm::APFloat next = estimate;
    (void)next.add(quotient, llvm::APFloat::rmNearestTiesToEven);
    (void)next.multiply(half, llvm::APFloat::rmNearestTiesToEven);
    if (next.bitcastToAPInt() == estimate.bitcastToAPInt())
      break;
    estimate = std::move(next);
  }

  llvm::APFloat rounded = estimate;
  (void)rounded.convert(input.getSemantics(), rounding, &losesInfo);

  // Determine IEEE inexactness without consulting a host floating type. A
  // widened F16/F32 candidate squares exactly in IEEEquad.
  llvm::APFloat candidate = rounded;
  bool candidateLosesInfo = false;
  status = candidate.convert(llvm::APFloat::IEEEquad(),
                             llvm::APFloat::rmNearestTiesToEven,
                             &candidateLosesInfo);
  if (status != llvm::APFloat::opOK || candidateLosesInfo)
    llvm::report_fatal_error("failed to exactly widen sqrt candidate");
  llvm::APFloat squared = candidate;
  status = squared.multiply(candidate, llvm::APFloat::rmNearestTiesToEven);
  if (status != llvm::APFloat::opOK)
    llvm::report_fatal_error("failed to exactly square sqrt candidate");
  const bool exact = squared.compare(radicand) == llvm::APFloat::cmpEqual;
  return {std::move(rounded),
          exact ? llvm::APFloat::opOK : llvm::APFloat::opInexact};
}

FormalNumericResult
executeAPFloatArithmetic(SoftFloatOracleOperation operation,
                         NumericRoundingMode rounding, LogicalFormat format,
                         uint64_t lhsBits,
                         std::optional<uint64_t> rhsBits = std::nullopt,
                         std::optional<uint64_t> addendBits = std::nullopt) {
  llvm::APFloat result = decodeAPFloat(format, lhsBits);
  const llvm::APFloat::roundingMode apRounding =
      getAPFloatRoundingMode(rounding);
  llvm::APFloat::opStatus status = llvm::APFloat::opOK;
  switch (operation) {
  case SoftFloatOracleOperation::Add:
    status = result.add(decodeAPFloat(format, *rhsBits), apRounding);
    break;
  case SoftFloatOracleOperation::Subtract:
    status = result.subtract(decodeAPFloat(format, *rhsBits), apRounding);
    break;
  case SoftFloatOracleOperation::Multiply:
    status = result.multiply(decodeAPFloat(format, *rhsBits), apRounding);
    break;
  case SoftFloatOracleOperation::Divide:
    status = result.divide(decodeAPFloat(format, *rhsBits), apRounding);
    break;
  case SoftFloatOracleOperation::SquareRoot: {
    APFloatEvaluation evaluated =
        squareRootWithAPFloat(std::move(result), apRounding);
    result = std::move(evaluated.value);
    status = evaluated.status;
    break;
  }
  case SoftFloatOracleOperation::FusedMultiplyAdd:
    status =
        result.fusedMultiplyAdd(decodeAPFloat(format, *rhsBits),
                                decodeAPFloat(format, *addendBits), apRounding);
    break;
  case SoftFloatOracleOperation::Convert:
    llvm_unreachable("convert uses the production formal dispatcher");
  }

  const uint64_t bits =
      result.isNaN() ? canonicalQuietNaN(format) : encodeAPFloat(result);
  return {{format, bits}, flagsFromAPFloatStatus(status)};
}

struct APFloatStatusGapCounts {
  uint64_t missingOverflow = 0;
  uint64_t missingUnderflow = 0;
};

void compareArithmeticOne(SoftFloatOracleOperation operation,
                          LogicalFormat format, NumericRoundingMode rounding,
                          uint64_t lhs, std::optional<uint64_t> rhs,
                          std::optional<uint64_t> addend,
                          APFloatStatusGapCounts &statusGaps) {
  const FormalNumericResult production =
      executeAPFloatArithmetic(operation, rounding, format, lhs, rhs, addend);
  llvm::Expected<FormalNumericResult> oracle = executeSoftFloatOracle(
      {operation,
       rounding,
       SoftFloatOracleTininess::AfterRounding,
       format,
       {format, lhs},
       rhs ? std::optional<RawLogicalValue>(RawLogicalValue{format, *rhs})
           : std::nullopt,
       addend ? std::optional<RawLogicalValue>(RawLogicalValue{format, *addend})
              : std::nullopt});
  ASSERT_TRUE(static_cast<bool>(oracle))
      << (oracle ? std::string() : llvm::toString(oracle.takeError()));

  const auto diagnostic = [&]() {
    return std::string(" operation=") +
           std::to_string(static_cast<unsigned>(operation)) +
           " format=" + stringifyLogicalFormat(format).str() +
           " rounding=" + std::to_string(static_cast<unsigned>(rounding));
  };
  EXPECT_EQ(production.value.bits, oracle->value.bits)
      << "lhs=0x" << std::hex << lhs << " rhs=0x" << rhs.value_or(0)
      << " addend=0x" << addend.value_or(0) << diagnostic();
  EXPECT_EQ(production.flags.invalid, oracle->flags.invalid)
      << "lhs=0x" << std::hex << lhs << " rhs=0x" << rhs.value_or(0)
      << " addend=0x" << addend.value_or(0) << diagnostic();
  EXPECT_EQ(production.flags.divByZero, oracle->flags.divByZero)
      << "lhs=0x" << std::hex << lhs << " rhs=0x" << rhs.value_or(0)
      << " addend=0x" << addend.value_or(0) << diagnostic();
  EXPECT_EQ(production.flags.inexact, oracle->flags.inexact)
      << "lhs=0x" << std::hex << lhs << " rhs=0x" << rhs.value_or(0)
      << " addend=0x" << addend.value_or(0) << diagnostic();

  // Pinned LLVM APFloat is the production arithmetic engine but its opStatus
  // omits some directed overflow and IEEE tininess-after threshold flags.
  // Keep those omissions visible and one-directional instead of silently
  // treating APFloat's status bits as the model policy.
  if (production.flags.overflow != oracle->flags.overflow) {
    EXPECT_FALSE(production.flags.overflow) << diagnostic();
    EXPECT_TRUE(oracle->flags.overflow) << diagnostic();
    EXPECT_TRUE(oracle->flags.inexact) << diagnostic();
    ++statusGaps.missingOverflow;
  }
  if (production.flags.underflow != oracle->flags.underflow) {
    EXPECT_FALSE(production.flags.underflow) << diagnostic();
    EXPECT_TRUE(oracle->flags.underflow) << diagnostic();
    EXPECT_TRUE(oracle->flags.inexact) << diagnostic();
    ++statusGaps.missingUnderflow;
  }
}

std::vector<uint64_t> arithmeticBoundaryValues(LogicalFormat format) {
  if (format == LogicalFormat::F16)
    return {
        UINT64_C(0x0000), UINT64_C(0x8000), UINT64_C(0x0001), UINT64_C(0x8001),
        UINT64_C(0x0002), UINT64_C(0x8002), UINT64_C(0x03ff), UINT64_C(0x83ff),
        UINT64_C(0x0400), UINT64_C(0x8400), UINT64_C(0x0401), UINT64_C(0x8401),
        UINT64_C(0x1400), UINT64_C(0x9400), UINT64_C(0x3555), UINT64_C(0xb555),
        UINT64_C(0x3800), UINT64_C(0xb800), UINT64_C(0x3bff), UINT64_C(0xbbff),
        UINT64_C(0x3c00), UINT64_C(0xbc00), UINT64_C(0x3c01), UINT64_C(0xbc01),
        UINT64_C(0x4000), UINT64_C(0xc000), UINT64_C(0x7bff), UINT64_C(0xfbff),
        UINT64_C(0x7c00), UINT64_C(0xfc00), UINT64_C(0x7e00), UINT64_C(0xfe00),
        UINT64_C(0x7c01), UINT64_C(0xfc01)};
  return {UINT64_C(0x00000000), UINT64_C(0x80000000), UINT64_C(0x00000001),
          UINT64_C(0x80000001), UINT64_C(0x00000002), UINT64_C(0x80000002),
          UINT64_C(0x007fffff), UINT64_C(0x807fffff), UINT64_C(0x00800000),
          UINT64_C(0x80800000), UINT64_C(0x00800001), UINT64_C(0x80800001),
          UINT64_C(0x30800000), UINT64_C(0xb0800000), UINT64_C(0x3eaaaaab),
          UINT64_C(0xbeaaaaab), UINT64_C(0x3f000000), UINT64_C(0xbf000000),
          UINT64_C(0x3f7fffff), UINT64_C(0xbf7fffff), UINT64_C(0x3f800000),
          UINT64_C(0xbf800000), UINT64_C(0x3f800001), UINT64_C(0xbf800001),
          UINT64_C(0x40000000), UINT64_C(0xc0000000), UINT64_C(0x7f7fffff),
          UINT64_C(0xff7fffff), UINT64_C(0x7f800000), UINT64_C(0xff800000),
          UINT64_C(0x7fc00000), UINT64_C(0xffc00000), UINT64_C(0x7f800001),
          UINT64_C(0xff800001)};
}

TEST(NumericOracleDifferentialTest, ExhaustiveF16ToF32MatchesSoftFloat) {
  const ResolvedNumericCommand command = resolveConvert(
      /*fp16_fp32=*/161, NumericRoundingMode::NearestEven);
  for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits)
    compareOne(command, NumericRoundingMode::NearestEven,
               {LogicalFormat::F16, bits}, LogicalFormat::F32);
}

TEST(NumericOracleDifferentialTest,
     StratifiedF32ToF16AllDeterministicModesMatchSoftFloat) {
  constexpr std::array<NumericRoundingMode, 4> modes = {
      NumericRoundingMode::NearestEven, NumericRoundingMode::TowardZero,
      NumericRoundingMode::TowardPositive, NumericRoundingMode::TowardNegative};
  constexpr std::array<uint32_t, 24> boundary = {
      UINT32_C(0x00000000), UINT32_C(0x80000000), UINT32_C(0x00000001),
      UINT32_C(0x80000001), UINT32_C(0x007fffff), UINT32_C(0x807fffff),
      UINT32_C(0x00800000), UINT32_C(0x80800000), UINT32_C(0x33000000),
      UINT32_C(0xb3000000), UINT32_C(0x33800000), UINT32_C(0xb3800000),
      UINT32_C(0x387fc000), UINT32_C(0xb87fc000), UINT32_C(0x3f801000),
      UINT32_C(0xbf801000), UINT32_C(0x477fe000), UINT32_C(0xc77fe000),
      UINT32_C(0x7f7fffff), UINT32_C(0xff7fffff), UINT32_C(0x7f800000),
      UINT32_C(0xff800000), UINT32_C(0x7fc00000), UINT32_C(0x7f800001)};

  for (NumericRoundingMode mode : modes) {
    const ResolvedNumericCommand command =
        resolveConvert(/*fp32_fp16=*/166, mode);
    for (uint32_t bits : boundary)
      compareOne(command, mode, {LogicalFormat::F32, bits}, LogicalFormat::F16);

    uint32_t state = UINT32_C(0x6d2b79f5);
    for (unsigned index = 0; index < 65536; ++index) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      compareOne(command, mode, {LogicalFormat::F32, state},
                 LogicalFormat::F16);
    }
  }
}

TEST(NumericOracleDifferentialTest,
     BoundaryF16F32ArithmeticAllModesMatchesSoftFloat) {
  constexpr std::array<LogicalFormat, 2> formats = {LogicalFormat::F16,
                                                    LogicalFormat::F32};
  constexpr std::array<NumericRoundingMode, 4> modes = {
      NumericRoundingMode::NearestEven, NumericRoundingMode::TowardZero,
      NumericRoundingMode::TowardPositive, NumericRoundingMode::TowardNegative};
  constexpr std::array<SoftFloatOracleOperation, 4> binaryOperations = {
      SoftFloatOracleOperation::Add, SoftFloatOracleOperation::Subtract,
      SoftFloatOracleOperation::Multiply, SoftFloatOracleOperation::Divide};
  APFloatStatusGapCounts statusGaps;

  for (LogicalFormat format : formats) {
    const std::vector<uint64_t> values = arithmeticBoundaryValues(format);
    for (NumericRoundingMode mode : modes) {
      for (SoftFloatOracleOperation operation : binaryOperations) {
        for (uint64_t lhs : values) {
          for (uint64_t rhs : values)
            compareArithmeticOne(operation, format, mode, lhs, rhs,
                                 std::nullopt, statusGaps);
        }
      }
      for (uint64_t value : values)
        compareArithmeticOne(SoftFloatOracleOperation::SquareRoot, format, mode,
                             value, std::nullopt, std::nullopt, statusGaps);

      // FMA uses fixed, non-cartesian addend strata so the test distinguishes
      // one-round fused behavior without a cubic value-space explosion.
      for (size_t lhsIndex = 0; lhsIndex < values.size(); ++lhsIndex) {
        for (unsigned stratum = 0; stratum < 8; ++stratum) {
          const uint64_t rhs =
              values[(lhsIndex * 7 + stratum * 5 + 1) % values.size()];
          const uint64_t addend =
              values[(lhsIndex * 11 + stratum * 13 + 3) % values.size()];
          compareArithmeticOne(SoftFloatOracleOperation::FusedMultiplyAdd,
                               format, mode, values[lhsIndex], rhs, addend,
                               statusGaps);
        }
      }
    }
  }
  EXPECT_GT(statusGaps.missingOverflow, UINT64_C(0));
  EXPECT_GT(statusGaps.missingUnderflow, UINT64_C(0));
}

TEST(NumericOracleDifferentialTest,
     FusedMultiplyAddDistinguishesSingleRounding) {
  struct Case {
    LogicalFormat format;
    uint64_t lhs;
    uint64_t rhs;
    uint64_t addend;
    uint64_t expected;
  };
  constexpr Case cases[] = {
      {LogicalFormat::F16, UINT64_C(0xd707), UINT64_C(0x92fc), UINT64_C(0x85bb),
       UINT64_C(0x2e21)},
      {LogicalFormat::F32, UINT64_C(0xfc7aeeef), UINT64_C(0x954c99bc),
       UINT64_C(0xc9184aa1), UINT64_C(0x52488cf1)},
  };
  APFloatStatusGapCounts statusGaps;
  for (const Case &testCase : cases) {
    const FormalNumericResult production = executeAPFloatArithmetic(
        SoftFloatOracleOperation::FusedMultiplyAdd,
        NumericRoundingMode::NearestEven, testCase.format, testCase.lhs,
        testCase.rhs, testCase.addend);
    EXPECT_EQ(production.value.bits, testCase.expected);
    compareArithmeticOne(SoftFloatOracleOperation::FusedMultiplyAdd,
                         testCase.format, NumericRoundingMode::NearestEven,
                         testCase.lhs, testCase.rhs, testCase.addend,
                         statusGaps);
  }
}

TEST(NumericOracleDifferentialTest,
     FixedStratifiedF16F32ArithmeticMatchesSoftFloat) {
  constexpr std::array<LogicalFormat, 2> formats = {LogicalFormat::F16,
                                                    LogicalFormat::F32};
  constexpr std::array<NumericRoundingMode, 4> modes = {
      NumericRoundingMode::NearestEven, NumericRoundingMode::TowardZero,
      NumericRoundingMode::TowardPositive, NumericRoundingMode::TowardNegative};
  constexpr std::array<SoftFloatOracleOperation, 4> binaryOperations = {
      SoftFloatOracleOperation::Add, SoftFloatOracleOperation::Subtract,
      SoftFloatOracleOperation::Multiply, SoftFloatOracleOperation::Divide};
  APFloatStatusGapCounts statusGaps;

  uint64_t state = UINT64_C(0xd1b54a32d192ed03);
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };

  for (LogicalFormat format : formats) {
    const uint64_t mask =
        format == LogicalFormat::F16 ? UINT64_C(0xffff) : UINT64_C(0xffffffff);
    for (unsigned sample = 0; sample < 512; ++sample) {
      const uint64_t lhs = next() & mask;
      const uint64_t rhs = next() & mask;
      const uint64_t addend = next() & mask;
      for (NumericRoundingMode mode : modes) {
        for (SoftFloatOracleOperation operation : binaryOperations)
          compareArithmeticOne(operation, format, mode, lhs, rhs, std::nullopt,
                               statusGaps);
        compareArithmeticOne(SoftFloatOracleOperation::SquareRoot, format, mode,
                             lhs, std::nullopt, std::nullopt, statusGaps);
        compareArithmeticOne(SoftFloatOracleOperation::FusedMultiplyAdd, format,
                             mode, lhs, rhs, addend, statusGaps);
      }
    }
  }
}

} // namespace
