//===- ProgramTensorComparison.cpp - Expected output comparison ----------===//

#include "Wafer/Simulator/Invocation/ProgramTensorComparison.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler {
namespace {

// Admitted formats are at most F32, so their products and squared differences
// fit in F64 even at the extrema. Accumulate in extended precision and avoid
// epsilon denominators: zero vectors have an explicit comparison contract.
struct TensorSimilarity {
  long double dot = 0.0;
  long double actualSquared = 0.0;
  long double expectedSquared = 0.0;
  long double errorSquared = 0.0;

  void add(double actual, double expected) {
    const long double a = actual, b = expected;
    dot += a * b;
    actualSquared += a * a;
    expectedSquared += b * b;
    errorSquared += (a - b) * (a - b);
  }

  double cosine() const {
    if (actualSquared == 0.0 || expectedSquared == 0.0)
      return actualSquared == expectedSquared ? 1.0 : 0.0;
    return static_cast<double>(
        std::clamp(dot / std::sqrt(actualSquared) / std::sqrt(expectedSquared),
                   -1.0L, 1.0L));
  }

  double relativeL2() const {
    if (expectedSquared == 0.0)
      return errorSquared == 0.0 ? 0.0
                                 : std::numeric_limits<double>::infinity();
    return static_cast<double>(std::sqrt(errorSquared / expectedSquared));
  }
};

llvm::Error comparisonError(ProgramTensorComparisonErrorCode code,
                            const std::string &detail) {
  return llvm::make_error<ProgramTensorComparisonError>(code, detail);
}

std::string formatShape(llvm::ArrayRef<int64_t> shape) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << '[';
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0)
      stream << ',';
    stream << shape[index];
  }
  stream << ']';
  return result;
}

uint16_t readLittleEndian16(const uint8_t *bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t readLittleEndian32(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

struct DecodedFloat {
  bool finite = false;
  double value = 0.0;
};

DecodedFloat decodeBinaryFloat(uint32_t bits, unsigned exponentBits,
                               unsigned fractionBits, int exponentBias) {
  const uint32_t fractionMask = (uint32_t{1} << fractionBits) - uint32_t{1};
  const uint32_t exponentMask = (uint32_t{1} << exponentBits) - uint32_t{1};
  const uint32_t fraction = bits & fractionMask;
  const uint32_t exponent = (bits >> fractionBits) & exponentMask;
  const bool negative =
      ((bits >> (fractionBits + exponentBits)) & uint32_t{1}) != 0;
  if (exponent == exponentMask)
    return {};

  double value = 0.0;
  if (exponent == 0) {
    value = std::ldexp(static_cast<double>(fraction),
                       1 - exponentBias - static_cast<int>(fractionBits));
  } else {
    const uint32_t significand = (uint32_t{1} << fractionBits) | fraction;
    value = std::ldexp(static_cast<double>(significand),
                       static_cast<int>(exponent) - exponentBias -
                           static_cast<int>(fractionBits));
  }
  if (negative)
    value = -value;
  return {true, value};
}

DecodedFloat decodeFloatAt(ProgramElementType dtype,
                           llvm::ArrayRef<uint8_t> bytes, size_t offset) {
  if (dtype == ProgramElementType::F16)
    return decodeBinaryFloat(readLittleEndian16(bytes.data() + offset), 5, 10,
                             15);
  if (dtype == ProgramElementType::BF16)
    return decodeBinaryFloat(readLittleEndian16(bytes.data() + offset), 8, 7,
                             127);
  if (dtype == ProgramElementType::F32)
    return decodeBinaryFloat(readLittleEndian32(bytes.data() + offset), 8, 23,
                             127);
  llvm_unreachable("unsupported floating dtype reached tolerant comparator");
}

size_t getElementByteWidth(ProgramElementType dtype) {
  if (dtype == ProgramElementType::F16 || dtype == ProgramElementType::BF16)
    return 2;
  if (dtype == ProgramElementType::F32)
    return 4;
  llvm_unreachable("unsupported floating dtype reached tolerant comparator");
}

bool isSupportedToleranceDType(ProgramElementType dtype) {
  return dtype == ProgramElementType::F16 ||
         dtype == ProgramElementType::BF16 || dtype == ProgramElementType::F32;
}

uint32_t readFloatBitsAt(ProgramElementType dtype,
                         llvm::ArrayRef<uint8_t> bytes, size_t offset) {
  if (dtype == ProgramElementType::F16 || dtype == ProgramElementType::BF16)
    return readLittleEndian16(bytes.data() + offset);
  if (dtype == ProgramElementType::F32)
    return readLittleEndian32(bytes.data() + offset);
  llvm_unreachable("unsupported floating dtype reached raw bit decoder");
}

uint64_t getUlpOrderKey(uint32_t bits, unsigned width) {
  const uint64_t signMask = uint64_t{1} << (width - 1);
  const uint64_t magnitude = static_cast<uint64_t>(bits) & (signMask - 1);
  return (static_cast<uint64_t>(bits) & signMask) != 0 ? signMask - magnitude
                                                       : signMask + magnitude;
}

uint64_t getUlpDistance(ProgramElementType dtype, uint32_t actualBits,
                        uint32_t expectedBits, double actual, double expected) {
  if (actual == expected)
    return 0;
  const unsigned width = dtype == ProgramElementType::F32 ? 32 : 16;
  const uint64_t actualKey = getUlpOrderKey(actualBits, width);
  const uint64_t expectedKey = getUlpOrderKey(expectedBits, width);
  return actualKey >= expectedKey ? actualKey - expectedKey
                                  : expectedKey - actualKey;
}

template <typename T>
T getNearestRankQuantile(const std::vector<T> &sorted, double quantile) {
  assert(!sorted.empty());
  size_t rank = static_cast<size_t>(
      std::ceil(quantile * static_cast<double>(sorted.size())));
  rank = std::max<size_t>(1, std::min(rank, sorted.size()));
  return sorted[rank - 1];
}

std::string numericDetail(size_t elementIndex, double actual, double expected,
                          double absoluteError, double tolerance) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << "element=" << elementIndex << " actual=" << actual
         << " expected=" << expected << " abs_error=" << absoluteError
         << " tolerance=" << tolerance;
  return result;
}

struct NumericMismatchSample {
  size_t elementIndex = 0;
  double actual = 0.0;
  double expected = 0.0;
  double absoluteError = 0.0;
  double tolerance = 0.0;
};

std::string
numericMismatchSummary(size_t mismatchCount, size_t elementCount,
                       const NumericMismatchSample &firstMismatch,
                       const NumericMismatchSample &maximumAbsoluteError,
                       const NumericMismatchSample &maximumToleranceRatio) {
  const double ratio = maximumToleranceRatio.tolerance == 0.0
                           ? std::numeric_limits<double>::infinity()
                           : maximumToleranceRatio.absoluteError /
                                 maximumToleranceRatio.tolerance;
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << "mismatches=" << mismatchCount << '/' << elementCount << " first={"
         << numericDetail(firstMismatch.elementIndex, firstMismatch.actual,
                          firstMismatch.expected, firstMismatch.absoluteError,
                          firstMismatch.tolerance)
         << "} max_abs={"
         << numericDetail(maximumAbsoluteError.elementIndex,
                          maximumAbsoluteError.actual,
                          maximumAbsoluteError.expected,
                          maximumAbsoluteError.absoluteError,
                          maximumAbsoluteError.tolerance)
         << "} max_tolerance_ratio=" << ratio
         << " at_element=" << maximumToleranceRatio.elementIndex;
  return result;
}

} // namespace

llvm::StringRef stringifyProgramTensorComparisonErrorCode(
    ProgramTensorComparisonErrorCode code) {
  switch (code) {
  case ProgramTensorComparisonErrorCode::InvalidTolerance:
    return "invalid-tolerance";
  case ProgramTensorComparisonErrorCode::DTypeMismatch:
    return "dtype-mismatch";
  case ProgramTensorComparisonErrorCode::ShapeMismatch:
    return "shape-mismatch";
  case ProgramTensorComparisonErrorCode::ByteSizeMismatch:
    return "byte-size-mismatch";
  case ProgramTensorComparisonErrorCode::UnsupportedFloatingDType:
    return "unsupported-floating-dtype";
  case ProgramTensorComparisonErrorCode::UnsupportedStatisticsDType:
    return "unsupported-statistics-dtype";
  case ProgramTensorComparisonErrorCode::ActualNonFinite:
    return "actual-nonfinite";
  case ProgramTensorComparisonErrorCode::ExpectedNonFinite:
    return "expected-nonfinite";
  case ProgramTensorComparisonErrorCode::NumericMismatch:
    return "numeric-mismatch";
  case ProgramTensorComparisonErrorCode::RawMismatch:
    return "raw-mismatch";
  }
  llvm_unreachable("program tensor comparison error code is not registered");
}

char ProgramTensorComparisonError::ID;

void ProgramTensorComparisonError::log(llvm::raw_ostream &stream) const {
  stream << "program tensor comparison "
         << stringifyProgramTensorComparisonErrorCode(code) << ": " << detail;
}

std::error_code ProgramTensorComparisonError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Error compareProgramTensorExpectedOutput(
    const ProgramTensor &actual, const ProgramTensor &expected,
    const ProgramTensorComparisonPolicy &policy) {
  if (llvm::Error error = validateProgramTensorComparisonPolicy(policy))
    return error;
  const double atol = policy.atol, rtol = policy.rtol;
  if (actual.getDType() != expected.getDType())
    return comparisonError(
        ProgramTensorComparisonErrorCode::DTypeMismatch,
        "actual=" + stringifyProgramElementType(actual.getDType()).str() +
            " expected=" +
            stringifyProgramElementType(expected.getDType()).str());
  if (actual.getShape() != expected.getShape())
    return comparisonError(ProgramTensorComparisonErrorCode::ShapeMismatch,
                           "actual=" + formatShape(actual.getShape()) +
                               " expected=" + formatShape(expected.getShape()));
  if (actual.getBytes().size() != expected.getBytes().size()) {
    std::string detail;
    llvm::raw_string_ostream stream(detail);
    stream << "actual=" << actual.getBytes().size()
           << " expected=" << expected.getBytes().size();
    return comparisonError(ProgramTensorComparisonErrorCode::ByteSizeMismatch,
                           detail);
  }

  const ProgramElementType dtype = actual.getDType();
  if (isFloatingProgramTensorDType(dtype) && !isSupportedToleranceDType(dtype))
    return comparisonError(
        ProgramTensorComparisonErrorCode::UnsupportedFloatingDType,
        "dtype=" + stringifyProgramElementType(dtype).str() +
            " has no expected-output policy");

  if (!isSupportedToleranceDType(dtype)) {
    const llvm::ArrayRef<uint8_t> actualBytes = actual.getBytes();
    const llvm::ArrayRef<uint8_t> expectedBytes = expected.getBytes();
    for (size_t offset = 0; offset < actualBytes.size(); ++offset) {
      if (actualBytes[offset] == expectedBytes[offset])
        continue;
      std::string detail;
      llvm::raw_string_ostream stream(detail);
      stream << "byte=" << offset
             << " actual=" << static_cast<unsigned>(actualBytes[offset])
             << " expected=" << static_cast<unsigned>(expectedBytes[offset]);
      return comparisonError(ProgramTensorComparisonErrorCode::RawMismatch,
                             detail);
    }
    return llvm::Error::success();
  }

  const size_t elementBytes = getElementByteWidth(dtype);
  const llvm::ArrayRef<uint8_t> actualBytes = actual.getBytes();
  const llvm::ArrayRef<uint8_t> expectedBytes = expected.getBytes();
  const size_t elementCount = actualBytes.size() / elementBytes;
  if (policy.similarity && elementCount == 0)
    return comparisonError(
        ProgramTensorComparisonErrorCode::UnsupportedStatisticsDType,
        "similarity requires a non-empty tensor");
  TensorSimilarity similarity;
  size_t mismatchCount = 0;
  std::optional<NumericMismatchSample> firstMismatch;
  std::optional<NumericMismatchSample> maximumAbsoluteError;
  std::optional<NumericMismatchSample> maximumToleranceRatio;
  for (size_t offset = 0; offset < actualBytes.size(); offset += elementBytes) {
    const size_t elementIndex = offset / elementBytes;
    const DecodedFloat actualValue = decodeFloatAt(dtype, actualBytes, offset);
    const DecodedFloat expectedValue =
        decodeFloatAt(dtype, expectedBytes, offset);
    if (!actualValue.finite) {
      std::string detail;
      llvm::raw_string_ostream(detail) << "element=" << elementIndex;
      return comparisonError(ProgramTensorComparisonErrorCode::ActualNonFinite,
                             detail);
    }
    if (!expectedValue.finite) {
      std::string detail;
      llvm::raw_string_ostream(detail) << "element=" << elementIndex;
      return comparisonError(
          ProgramTensorComparisonErrorCode::ExpectedNonFinite, detail);
    }

    const double absoluteError =
        std::abs(actualValue.value - expectedValue.value);
    if (policy.similarity)
      similarity.add(actualValue.value, expectedValue.value);
    const double tolerance = atol + rtol * std::abs(expectedValue.value);
    if (!std::isfinite(tolerance))
      return comparisonError(ProgramTensorComparisonErrorCode::InvalidTolerance,
                             "computed tolerance is not finite at element=" +
                                 std::to_string(elementIndex));
    if (absoluteError <= tolerance)
      continue;
    const NumericMismatchSample sample{elementIndex, actualValue.value,
                                       expectedValue.value, absoluteError,
                                       tolerance};
    ++mismatchCount;
    if (!firstMismatch)
      firstMismatch = sample;
    if (!maximumAbsoluteError ||
        sample.absoluteError > maximumAbsoluteError->absoluteError)
      maximumAbsoluteError = sample;
    const double ratio = tolerance == 0.0
                             ? std::numeric_limits<double>::infinity()
                             : absoluteError / tolerance;
    const double currentMaximumRatio =
        !maximumToleranceRatio ? -1.0
                               : (maximumToleranceRatio->tolerance == 0.0
                                      ? std::numeric_limits<double>::infinity()
                                      : maximumToleranceRatio->absoluteError /
                                            maximumToleranceRatio->tolerance);
    if (!maximumToleranceRatio || ratio > currentMaximumRatio)
      maximumToleranceRatio = sample;
  }
  if (policy.similarity) {
    if (similarity.cosine() >= policy.similarity->minimumCosine &&
        similarity.relativeL2() <= policy.similarity->maximumRelativeL2)
      return llvm::Error::success();
    std::string detail;
    llvm::raw_string_ostream(detail)
        << "cosine=" << similarity.cosine()
        << " minimum_cosine=" << policy.similarity->minimumCosine
        << " relative_l2=" << similarity.relativeL2()
        << " maximum_relative_l2=" << policy.similarity->maximumRelativeL2
        << " elementwise_mismatches=" << mismatchCount << '/' << elementCount;
    return comparisonError(ProgramTensorComparisonErrorCode::NumericMismatch,
                           detail);
  }
  if (mismatchCount != 0)
    return comparisonError(
        ProgramTensorComparisonErrorCode::NumericMismatch,
        numericMismatchSummary(mismatchCount, elementCount, *firstMismatch,
                               *maximumAbsoluteError, *maximumToleranceRatio));
  return llvm::Error::success();
}

llvm::Expected<ProgramTensorComparisonStatistics>
computeProgramTensorComparisonStatistics(
    const ProgramTensor &actual, const ProgramTensor &expected,
    const ProgramTensorComparisonPolicy &policy) {
  if (llvm::Error error = validateProgramTensorComparisonPolicy(policy))
    return std::move(error);
  if (actual.getDType() != expected.getDType())
    return comparisonError(
        ProgramTensorComparisonErrorCode::DTypeMismatch,
        "actual=" + stringifyProgramElementType(actual.getDType()).str() +
            " expected=" +
            stringifyProgramElementType(expected.getDType()).str());
  if (actual.getShape() != expected.getShape())
    return comparisonError(ProgramTensorComparisonErrorCode::ShapeMismatch,
                           "actual=" + formatShape(actual.getShape()) +
                               " expected=" + formatShape(expected.getShape()));
  if (actual.getBytes().size() != expected.getBytes().size()) {
    std::string detail;
    llvm::raw_string_ostream stream(detail);
    stream << "actual=" << actual.getBytes().size()
           << " expected=" << expected.getBytes().size();
    return comparisonError(ProgramTensorComparisonErrorCode::ByteSizeMismatch,
                           detail);
  }

  const ProgramElementType dtype = actual.getDType();
  if (!isSupportedToleranceDType(dtype))
    return comparisonError(
        ProgramTensorComparisonErrorCode::UnsupportedStatisticsDType,
        "dtype=" + stringifyProgramElementType(dtype).str() +
            " has no source/model numeric statistics policy");

  const size_t elementBytes = getElementByteWidth(dtype);
  const llvm::ArrayRef<uint8_t> actualBytes = actual.getBytes();
  const llvm::ArrayRef<uint8_t> expectedBytes = expected.getBytes();
  const size_t elementCount = actualBytes.size() / elementBytes;
  if (elementCount == 0)
    return comparisonError(
        ProgramTensorComparisonErrorCode::UnsupportedStatisticsDType,
        "numeric statistics require a non-empty tensor");

  ProgramTensorComparisonStatistics result;
  result.elementCount = elementCount;
  long double absoluteErrorSum = 0.0;
  long double ulpDistanceSum = 0.0;
  TensorSimilarity similarity;
  std::vector<double> absoluteErrors;
  std::vector<uint64_t> ulpDistances;
  absoluteErrors.reserve(elementCount);
  ulpDistances.reserve(elementCount);
  for (size_t offset = 0; offset < actualBytes.size(); offset += elementBytes) {
    const size_t elementIndex = offset / elementBytes;
    const DecodedFloat actualValue = decodeFloatAt(dtype, actualBytes, offset);
    const DecodedFloat expectedValue =
        decodeFloatAt(dtype, expectedBytes, offset);
    if (!actualValue.finite)
      return comparisonError(ProgramTensorComparisonErrorCode::ActualNonFinite,
                             "element=" + std::to_string(elementIndex));
    if (!expectedValue.finite)
      return comparisonError(
          ProgramTensorComparisonErrorCode::ExpectedNonFinite,
          "element=" + std::to_string(elementIndex));

    const double absoluteError =
        std::abs(actualValue.value - expectedValue.value);
    const double tolerance =
        policy.atol + policy.rtol * std::abs(expectedValue.value);
    if (!std::isfinite(tolerance))
      return comparisonError(ProgramTensorComparisonErrorCode::InvalidTolerance,
                             "computed tolerance is not finite at element=" +
                                 std::to_string(elementIndex));
    result.elementwiseMismatchCount += absoluteError > tolerance;
    similarity.add(actualValue.value, expectedValue.value);
    const uint64_t ulpDistance =
        getUlpDistance(dtype, readFloatBitsAt(dtype, actualBytes, offset),
                       readFloatBitsAt(dtype, expectedBytes, offset),
                       actualValue.value, expectedValue.value);
    if (absoluteError == 0.0)
      ++result.exactElementCount;
    absoluteErrorSum += absoluteError;
    ulpDistanceSum += ulpDistance;
    absoluteErrors.push_back(absoluteError);
    ulpDistances.push_back(ulpDistance);
  }

  std::sort(absoluteErrors.begin(), absoluteErrors.end());
  std::sort(ulpDistances.begin(), ulpDistances.end());
  result.exactFraction = static_cast<double>(result.exactElementCount) /
                         static_cast<double>(elementCount);
  result.meanAbsoluteError =
      static_cast<double>(absoluteErrorSum / elementCount);
  result.p99AbsoluteError = getNearestRankQuantile(absoluteErrors, 0.99);
  result.p999AbsoluteError = getNearestRankQuantile(absoluteErrors, 0.999);
  result.maximumAbsoluteError = absoluteErrors.back();
  result.cosineSimilarity = similarity.cosine();
  result.relativeL2Error = similarity.relativeL2();
  result.meanUlpDistance = static_cast<double>(ulpDistanceSum / elementCount);
  result.p99UlpDistance = getNearestRankQuantile(ulpDistances, 0.99);
  result.p999UlpDistance = getNearestRankQuantile(ulpDistances, 0.999);
  result.maximumUlpDistance = ulpDistances.back();
  return result;
}

llvm::Error validateProgramTensorComparisonPolicy(
    const ProgramTensorComparisonPolicy &policy) {
  if (!std::isfinite(policy.atol) || !std::isfinite(policy.rtol) ||
      policy.atol < 0.0 || policy.rtol < 0.0)
    return comparisonError(
        ProgramTensorComparisonErrorCode::InvalidTolerance,
        "absolute and relative tolerances must be finite and nonnegative");
  if (policy.similarity &&
      (!std::isfinite(policy.similarity->minimumCosine) ||
       policy.similarity->minimumCosine < 0.0 ||
       policy.similarity->minimumCosine > 1.0 ||
       !std::isfinite(policy.similarity->maximumRelativeL2) ||
       policy.similarity->maximumRelativeL2 < 0.0))
    return comparisonError(ProgramTensorComparisonErrorCode::InvalidTolerance,
                           "similarity requires finite minimum cosine in [0,1] "
                           "and nonnegative relative L2");
  return llvm::Error::success();
}

} // namespace wafer::compiler
