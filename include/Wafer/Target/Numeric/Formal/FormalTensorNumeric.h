//===- FormalTensorNumeric.h - Atomic formal tensor execution -*- C++ -*-===//

#ifndef WAFER_TARGET_FORMALTENSORNUMERIC_H
#define WAFER_TARGET_FORMALTENSORNUMERIC_H

#include "Wafer/Target/Numeric/Formal/FormalNumeric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer {

/// Caller-owned upper bounds for one formal tensor command. There is no
/// implicit default: integration layers must choose both limits before any
/// scalar evaluation or output allocation can occur.
class FormalNumericWorkBudget {
public:
  FormalNumericWorkBudget() = delete;

  static constexpr FormalNumericWorkBudget
  create(uint64_t maximumScalarEvaluations, uint64_t maximumFusedMultiplyAdds) {
    return FormalNumericWorkBudget(maximumScalarEvaluations,
                                   maximumFusedMultiplyAdds);
  }

  uint64_t getMaximumScalarEvaluations() const {
    return maximumScalarEvaluations;
  }
  uint64_t getMaximumFusedMultiplyAdds() const {
    return maximumFusedMultiplyAdds;
  }

private:
  constexpr FormalNumericWorkBudget(uint64_t maximumScalarEvaluations,
                                    uint64_t maximumFusedMultiplyAdds)
      : maximumScalarEvaluations(maximumScalarEvaluations),
        maximumFusedMultiplyAdds(maximumFusedMultiplyAdds) {}

  uint64_t maximumScalarEvaluations;
  uint64_t maximumFusedMultiplyAdds;
};

/// Atomic result of one logical-dense tensor command. Physical Cx/NCx/BOOL
/// mapping remains a discharged layout-materialization precondition and is not
/// copied into this result. Flags are the command-local OR of all returned
/// scalar steps.
struct FormalTensorNumericResult {
  std::vector<RawLogicalValue> values;
  FormalNumericExceptionFlags flags;
};

enum class FormalTensorNumericErrorCode : uint8_t {
  UnsupportedResolvedCommand,
  InputArityMismatch,
  InputElementCountMismatch,
  InputFormatMismatch,
  InvalidInputEncoding,
  ScalarWorkBudgetExceeded,
  MultiplyAccumulateWorkBudgetExceeded,
  WorkCountOverflow,
  ResultInvariantViolation,
};

llvm::StringRef
stringifyFormalTensorNumericErrorCode(FormalTensorNumericErrorCode code);

class FormalTensorNumericError final
    : public llvm::ErrorInfo<FormalTensorNumericError> {
public:
  static char ID;

  FormalTensorNumericError(FormalTensorNumericErrorCode code,
                           std::string message)
      : code(code), detail(std::move(message)) {}

  FormalTensorNumericErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  FormalTensorNumericErrorCode code;
  std::string detail;
};

/// Executes one already-resolved supported convert, elementwise, or GEMM
/// command over logical-dense raw values. Complete command/profile, input size,
/// input encoding, and checked work-budget validation happens before output
/// allocation. Native reduction and every static unsupported capability fail
/// closed. On every error the caller context is unchanged and no result is
/// returned; flags are recorded only after the complete tensor succeeds.
llvm::Expected<FormalTensorNumericResult> executeFormalTensorNumeric(
    FormalNumericExecutionContext &context,
    const ResolvedNumericCommand &command,
    llvm::ArrayRef<llvm::ArrayRef<RawLogicalValue>> inputs,
    FormalNumericWorkBudget budget);

/// Raw-exact tensor comparator for the first deterministic model profile.
/// Length, format/bits, and command-local aggregate flags all participate.
llvm::Expected<bool>
compareFormalTensorNumericResultsExact(const FormalTensorNumericResult &lhs,
                                       const FormalTensorNumericResult &rhs);

} // namespace wafer

#endif // WAFER_TARGET_FORMALTENSORNUMERIC_H
