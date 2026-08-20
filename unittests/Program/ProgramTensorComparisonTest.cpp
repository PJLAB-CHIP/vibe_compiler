//===- ProgramTensorComparisonTest.cpp - Output comparison tests ---------===//

#include "Wafer/Program/ProgramTensorComparison.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
#include <vector>

namespace {

using wafer::compiler::ProgramTensor;
using wafer::compiler::ProgramTensorComparisonError;
using wafer::compiler::ProgramTensorComparisonErrorCode;

ProgramTensor makeTensor(llvm::StringRef dtype,
                         std::initializer_list<int64_t> shape,
                         std::initializer_list<uint8_t> bytes) {
  auto tensor = ProgramTensor::create(
      llvm::cantFail(wafer::parseProgramElementType(dtype)), shape, bytes);
  EXPECT_TRUE(static_cast<bool>(tensor));
  return std::move(*tensor);
}

std::optional<ProgramTensorComparisonErrorCode> takeCode(llvm::Error error) {
  if (!error)
    return std::nullopt;
  std::optional<ProgramTensorComparisonErrorCode> code;
  llvm::handleAllErrors(std::move(error),
                        [&](const ProgramTensorComparisonError &comparison) {
                          code = comparison.getCode();
                        });
  return code;
}

TEST(ProgramTensorComparisonTest, F16UsesAbsoluteAndRelativeTolerance) {
  ProgramTensor expected = makeTensor("f16", {1}, {0x00, 0x3c}); // 1.0
  ProgramTensor actual = makeTensor("f16", {1}, {0x01, 0x3c});   // 1.0009765625

  EXPECT_FALSE(wafer::compiler::compareProgramTensorExpectedOutput(
      actual, expected, 0.00048828125, 0.00048828125));
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                actual, expected, 0.0004, 0.0005)),
            ProgramTensorComparisonErrorCode::NumericMismatch);
}

TEST(ProgramTensorComparisonTest, BF16UsesRelativeTolerance) {
  ProgramTensor expected = makeTensor("bf16", {1}, {0x00, 0x3f}); // 0.5
  ProgramTensor actual = makeTensor("bf16", {1}, {0x01, 0x3f});   // 0.50390625

  EXPECT_FALSE(wafer::compiler::compareProgramTensorExpectedOutput(
      actual, expected, 0.0, 0.008));
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                actual, expected, 0.0, 0.007)),
            ProgramTensorComparisonErrorCode::NumericMismatch);
}

TEST(ProgramTensorComparisonTest, F32DecodesLittleEndianStorage) {
  ProgramTensor expected =
      makeTensor("f32", {1}, {0x00, 0x00, 0x80, 0x3f}); // 1.0
  ProgramTensor actual = makeTensor("f32", {1}, {0x01, 0x00, 0x80, 0x3f});

  EXPECT_FALSE(wafer::compiler::compareProgramTensorExpectedOutput(
      actual, expected, 1.2e-7, 0.0));
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                actual, expected, 1.0e-7, 0.0)),
            ProgramTensorComparisonErrorCode::NumericMismatch);
}

TEST(ProgramTensorComparisonTest, NumericMismatchReportsWholeTensorSummary) {
  ProgramTensor expected =
      makeTensor("f16", {3}, {0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c});
  ProgramTensor actual =
      makeTensor("f16", {3}, {0x00, 0x3c, 0x00, 0x40, 0x00, 0x42});
  std::string diagnostic =
      llvm::toString(wafer::compiler::compareProgramTensorExpectedOutput(
          actual, expected, 0.0, 0.0));
  EXPECT_NE(diagnostic.find("mismatches=2/3"), std::string::npos);
  EXPECT_NE(diagnostic.find("first={element=1"), std::string::npos);
  EXPECT_NE(diagnostic.find("max_abs={element=2"), std::string::npos);
}

TEST(ProgramTensorComparisonTest, SignedZeroIsNumericallyEqual) {
  ProgramTensor positiveZero = makeTensor("f16", {1}, {0x00, 0x00});
  ProgramTensor negativeZero = makeTensor("f16", {1}, {0x00, 0x80});
  EXPECT_FALSE(wafer::compiler::compareProgramTensorExpectedOutput(
      negativeZero, positiveZero, 0.0, 0.0));
}

TEST(ProgramTensorComparisonTest, SubnormalIsDecodedAsFinite) {
  ProgramTensor positiveZero = makeTensor("f16", {1}, {0x00, 0x00});
  ProgramTensor minimumSubnormal = makeTensor("f16", {1}, {0x01, 0x00});
  EXPECT_FALSE(wafer::compiler::compareProgramTensorExpectedOutput(
      minimumSubnormal, positiveZero, 0x1p-24, 0.0));
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                minimumSubnormal, positiveZero, 0x1p-25, 0.0)),
            ProgramTensorComparisonErrorCode::NumericMismatch);
}

TEST(ProgramTensorComparisonTest, RejectsActualAndExpectedNonFiniteValues) {
  ProgramTensor finite = makeTensor("f16", {1}, {0x00, 0x3c});
  ProgramTensor infinity = makeTensor("f16", {1}, {0x00, 0x7c});
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                infinity, finite, 100.0, 100.0)),
            ProgramTensorComparisonErrorCode::ActualNonFinite);

  ProgramTensor nan = makeTensor("bf16", {1}, {0xc1, 0x7f});
  ProgramTensor bf16Finite = makeTensor("bf16", {1}, {0x80, 0x3f});
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                bf16Finite, nan, 100.0, 100.0)),
            ProgramTensorComparisonErrorCode::ExpectedNonFinite);
}

TEST(ProgramTensorComparisonTest, NonFloatingStorageRemainsRawExact) {
  ProgramTensor expected = makeTensor("i16", {2}, {0x00, 0x01, 0x00, 0x02});
  ProgramTensor same = makeTensor("i16", {2}, {0x00, 0x01, 0x00, 0x02});
  ProgramTensor different = makeTensor("i16", {2}, {0x00, 0x01, 0x01, 0x02});

  EXPECT_FALSE(wafer::compiler::compareProgramTensorExpectedOutput(
      same, expected, 100.0, 100.0));
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                different, expected, 100.0, 100.0)),
            ProgramTensorComparisonErrorCode::RawMismatch);
}

TEST(ProgramTensorComparisonTest, ReportsDTypeAndShapeMismatchSeparately) {
  ProgramTensor f16 = makeTensor("f16", {1}, {0x00, 0x3c});
  ProgramTensor i16 = makeTensor("i16", {1}, {0x00, 0x3c});
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                f16, i16, 0.0, 0.0)),
            ProgramTensorComparisonErrorCode::DTypeMismatch);

  ProgramTensor flat = makeTensor("f16", {2}, {0x00, 0x3c, 0x00, 0x40});
  ProgramTensor matrix = makeTensor("f16", {1, 2}, {0x00, 0x3c, 0x00, 0x40});
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                flat, matrix, 0.0, 0.0)),
            ProgramTensorComparisonErrorCode::ShapeMismatch);
}

TEST(ProgramTensorComparisonTest, FloatingDTypeWithoutPolicyFailsClosed) {
  ProgramTensor expected =
      makeTensor("f64", {1}, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f});
  ProgramTensor actual =
      makeTensor("f64", {1}, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f});
  EXPECT_EQ(takeCode(wafer::compiler::compareProgramTensorExpectedOutput(
                actual, expected, 0.0, 0.0)),
            ProgramTensorComparisonErrorCode::UnsupportedFloatingDType);
}

TEST(ProgramTensorComparisonTest, DTypeClassificationCoversAdmittedSurface) {
  struct DTypeCase {
    wafer::ProgramElementType dtype;
    int64_t elementBytes;
    bool floating;
  };
  constexpr std::array<DTypeCase, 13> cases{{
      {wafer::ProgramElementType::I8, 1, false},
      {wafer::ProgramElementType::U8, 1, false},
      {wafer::ProgramElementType::I16, 2, false},
      {wafer::ProgramElementType::U16, 2, false},
      {wafer::ProgramElementType::F16, 2, true},
      {wafer::ProgramElementType::BF16, 2, true},
      {wafer::ProgramElementType::I32, 4, false},
      {wafer::ProgramElementType::U32, 4, false},
      {wafer::ProgramElementType::F32, 4, true},
      {wafer::ProgramElementType::TF32, 4, true},
      {wafer::ProgramElementType::I64, 8, false},
      {wafer::ProgramElementType::U64, 8, false},
      {wafer::ProgramElementType::F64, 8, true},
  }};
  for (const DTypeCase &testCase : cases) {
    SCOPED_TRACE(wafer::stringifyProgramElementType(testCase.dtype).str());
    EXPECT_EQ(
        wafer::compiler::computeProgramTensorByteCount(testCase.dtype, {2}),
        testCase.elementBytes * 2);
    EXPECT_EQ(wafer::compiler::isFloatingProgramTensorDType(testCase.dtype),
              testCase.floating);
  }
  auto unknown = wafer::parseProgramElementType("unknown");
  EXPECT_FALSE(static_cast<bool>(unknown));
  llvm::consumeError(unknown.takeError());
}

TEST(ProgramTensorComparisonTest,
     StatisticsUseNearestRankAndSignAwareF16UlpDistance) {
  constexpr uint16_t one = UINT16_C(0x3c00);
  std::vector<uint8_t> expectedBytes(2000);
  std::vector<uint8_t> actualBytes(2000);
  for (size_t index = 0; index < 1000; ++index) {
    expectedBytes[index * 2] = static_cast<uint8_t>(one);
    expectedBytes[index * 2 + 1] = static_cast<uint8_t>(one >> 8);
    uint16_t actual = one;
    if (index >= 989)
      actual += index == 999 ? 2 : 1;
    actualBytes[index * 2] = static_cast<uint8_t>(actual);
    actualBytes[index * 2 + 1] = static_cast<uint8_t>(actual >> 8);
  }
  ProgramTensor expected = llvm::cantFail(ProgramTensor::create(
      wafer::ProgramElementType::F16, {1000}, expectedBytes));
  ProgramTensor actual = llvm::cantFail(ProgramTensor::create(
      wafer::ProgramElementType::F16, {1000}, actualBytes));
  auto statistics = wafer::compiler::computeProgramTensorComparisonStatistics(
      actual, expected);
  ASSERT_TRUE(static_cast<bool>(statistics));
  EXPECT_EQ(statistics->elementCount, 1000u);
  EXPECT_EQ(statistics->exactElementCount, 989u);
  EXPECT_DOUBLE_EQ(statistics->exactFraction, 0.989);
  EXPECT_DOUBLE_EQ(statistics->meanAbsoluteError, 12.0 / 1024.0 / 1000.0);
  EXPECT_DOUBLE_EQ(statistics->p99AbsoluteError, 1.0 / 1024.0);
  EXPECT_DOUBLE_EQ(statistics->p999AbsoluteError, 1.0 / 1024.0);
  EXPECT_DOUBLE_EQ(statistics->maximumAbsoluteError, 2.0 / 1024.0);
  EXPECT_DOUBLE_EQ(statistics->meanUlpDistance, 12.0 / 1000.0);
  EXPECT_EQ(statistics->p99UlpDistance, 1u);
  EXPECT_EQ(statistics->p999UlpDistance, 1u);
  EXPECT_EQ(statistics->maximumUlpDistance, 2u);

  ProgramTensor positiveZero = makeTensor("f16", {1}, {0x00, 0x00});
  ProgramTensor negativeZero = makeTensor("f16", {1}, {0x00, 0x80});
  auto zeroStatistics =
      wafer::compiler::computeProgramTensorComparisonStatistics(negativeZero,
                                                                positiveZero);
  ASSERT_TRUE(static_cast<bool>(zeroStatistics));
  EXPECT_EQ(zeroStatistics->exactElementCount, 1u);
  EXPECT_EQ(zeroStatistics->maximumUlpDistance, 0u);
}

TEST(ProgramTensorComparisonTest, StatisticsCoverBF16AndF32UlpDomains) {
  ProgramTensor bf16Expected = makeTensor("bf16", {1}, {0x00, 0x3f});
  ProgramTensor bf16Actual = makeTensor("bf16", {1}, {0x01, 0x3f});
  auto bf16 = wafer::compiler::computeProgramTensorComparisonStatistics(
      bf16Actual, bf16Expected);
  ASSERT_TRUE(static_cast<bool>(bf16));
  EXPECT_DOUBLE_EQ(bf16->maximumAbsoluteError, 1.0 / 256.0);
  EXPECT_EQ(bf16->maximumUlpDistance, 1u);

  ProgramTensor f32Expected = makeTensor("f32", {1}, {0x00, 0x00, 0x80, 0x3f});
  ProgramTensor f32Actual = makeTensor("f32", {1}, {0x01, 0x00, 0x80, 0x3f});
  auto f32 = wafer::compiler::computeProgramTensorComparisonStatistics(
      f32Actual, f32Expected);
  ASSERT_TRUE(static_cast<bool>(f32));
  EXPECT_DOUBLE_EQ(f32->maximumAbsoluteError, 0x1p-23);
  EXPECT_EQ(f32->maximumUlpDistance, 1u);
}

TEST(ProgramTensorComparisonTest, StatisticsRejectUnsupportedAndNonFinite) {
  ProgramTensor integer = makeTensor("i16", {1}, {0x00, 0x00});
  EXPECT_EQ(takeCode(wafer::compiler::computeProgramTensorComparisonStatistics(
                         integer, integer)
                         .takeError()),
            ProgramTensorComparisonErrorCode::UnsupportedStatisticsDType);

  ProgramTensor finite = makeTensor("f16", {1}, {0x00, 0x3c});
  ProgramTensor infinity = makeTensor("f16", {1}, {0x00, 0x7c});
  EXPECT_EQ(takeCode(wafer::compiler::computeProgramTensorComparisonStatistics(
                         infinity, finite)
                         .takeError()),
            ProgramTensorComparisonErrorCode::ActualNonFinite);
}

} // namespace
