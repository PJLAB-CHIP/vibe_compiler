//===- ProgramTensorComparisonTest.cpp - Output comparison tests ---------===//

#include "Wafer/Compiler/ProgramTensorComparison.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>

namespace {

using wafer::compiler::ProgramTensor;
using wafer::compiler::ProgramTensorComparisonError;
using wafer::compiler::ProgramTensorComparisonErrorCode;

ProgramTensor makeTensor(llvm::StringRef dtype,
                         std::initializer_list<int64_t> shape,
                         std::initializer_list<uint8_t> bytes) {
  auto tensor = ProgramTensor::create(dtype, shape, bytes);
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
    llvm::StringLiteral dtype;
    int64_t elementBytes;
    bool floating;
  };
  constexpr std::array<DTypeCase, 13> cases{{
      {"i8", 1, false},
      {"ui8", 1, false},
      {"i16", 2, false},
      {"ui16", 2, false},
      {"f16", 2, true},
      {"bf16", 2, true},
      {"i32", 4, false},
      {"ui32", 4, false},
      {"f32", 4, true},
      {"tf32", 4, true},
      {"i64", 8, false},
      {"ui64", 8, false},
      {"f64", 8, true},
  }};
  for (const DTypeCase &testCase : cases) {
    SCOPED_TRACE(testCase.dtype.str());
    EXPECT_EQ(
        wafer::compiler::computeProgramTensorByteCount(testCase.dtype, {2}),
        testCase.elementBytes * 2);
    EXPECT_EQ(wafer::compiler::isFloatingProgramTensorDType(testCase.dtype),
              testCase.floating);
  }
  EXPECT_FALSE(wafer::compiler::computeProgramTensorByteCount("unknown", {1}));
  EXPECT_FALSE(wafer::compiler::isFloatingProgramTensorDType("unknown"));
}

} // namespace
