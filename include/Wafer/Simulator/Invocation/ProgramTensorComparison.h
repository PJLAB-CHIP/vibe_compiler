//===- ProgramTensorComparison.h - Expected output comparison -*- C++ -*-===//

#ifndef WAFER_SIMULATOR_INVOCATION_PROGRAMTENSORCOMPARISON_H
#define WAFER_SIMULATOR_INVOCATION_PROGRAMTENSORCOMPARISON_H

#include "Wafer/Simulator/Invocation/ProgramInvocation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace wafer::compiler {

/// Stable failure classes for source-expected ProgramTensor comparison.
enum class ProgramTensorComparisonErrorCode : uint8_t {
  InvalidTolerance,
  DTypeMismatch,
  ShapeMismatch,
  ByteSizeMismatch,
  UnsupportedFloatingDType,
  UnsupportedStatisticsDType,
  ActualNonFinite,
  ExpectedNonFinite,
  NumericMismatch,
  RawMismatch,
};

/// Both conditions must hold for the complete flattened floating output.
struct ProgramTensorSimilarityTolerance {
  double minimumCosine = 1.0;
  double maximumRelativeL2 = 0.0;
};

struct ProgramTensorComparisonPolicy {
  double atol = 0.0;
  double rtol = 0.0;
  // Without this explicit choice, comparison remains elementwise. When set,
  // atol/rtol describe diagnostics only. Integer outputs always remain exact.
  std::optional<ProgramTensorSimilarityTolerance> similarity;
};

llvm::Error validateProgramTensorComparisonPolicy(
    const ProgramTensorComparisonPolicy &policy);

/// Complete finite floating-point error distribution for one source/model
/// tensor pair. Quantiles use the nearest-rank definition over all elements,
/// including exact matches. ULP distances use the destination dtype's
/// sign-aware monotonic encoding; numerically equal signed zeros have distance
/// zero, matching the source/model comparator policy.
struct ProgramTensorComparisonStatistics {
  size_t elementCount = 0;
  size_t exactElementCount = 0;
  double exactFraction = 0.0;
  double meanAbsoluteError = 0.0;
  double p99AbsoluteError = 0.0;
  double p999AbsoluteError = 0.0;
  double maximumAbsoluteError = 0.0;
  size_t elementwiseMismatchCount = 0;
  double cosineSimilarity = 1.0;
  double relativeL2Error = 0.0;
  double meanUlpDistance = 0.0;
  uint64_t p99UlpDistance = 0;
  uint64_t p999UlpDistance = 0;
  uint64_t maximumUlpDistance = 0;
};

llvm::StringRef stringifyProgramTensorComparisonErrorCode(
    ProgramTensorComparisonErrorCode code);

class ProgramTensorComparisonError final
    : public llvm::ErrorInfo<ProgramTensorComparisonError> {
public:
  static char ID;

  ProgramTensorComparisonError(ProgramTensorComparisonErrorCode code,
                               std::string detail)
      : code(code), detail(std::move(detail)) {}

  ProgramTensorComparisonErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ProgramTensorComparisonErrorCode code;
  std::string detail;
};

/// Compares two compact row-major program tensors at the source/model output
/// boundary. F16, BF16, and F32 are decoded from their little-endian IEEE
/// storage and compared using the explicitly supplied policy. Any non-finite
/// floating value is rejected. Integer and other non-floating storage is
/// compared byte-for-byte. Floating formats without a supported tolerance
/// policy fail closed.
llvm::Error
compareProgramTensorExpectedOutput(const ProgramTensor &actual,
                                   const ProgramTensor &expected,
                                   const ProgramTensorComparisonPolicy &policy);

/// Computes a read-only error distribution for finite F16, BF16, or F32
/// tensors after applying the same metadata and decoding policy as the
/// source/model comparator. The policy supplies diagnostic elementwise
/// tolerances only; this analysis does not select or apply an acceptance gate.
llvm::Expected<ProgramTensorComparisonStatistics>
computeProgramTensorComparisonStatistics(
    const ProgramTensor &actual, const ProgramTensor &expected,
    const ProgramTensorComparisonPolicy &policy = {});

} // namespace wafer::compiler

#endif // WAFER_SIMULATOR_INVOCATION_PROGRAMTENSORCOMPARISON_H
