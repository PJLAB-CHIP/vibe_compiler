//===- FormalNumeric.h - Deterministic formal numeric execution -*- C++ -*-===//

#ifndef WAFER_TARGET_FORMALNUMERIC_H
#define WAFER_TARGET_FORMALNUMERIC_H

#include "Wafer/Target/Numeric/NumericSemantics.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace wafer {

/// Model-only IEEE exception observations for one formal operation. These are
/// deliberately independent of host floating-point state and do not claim that
/// the target ABI exposes an exception register.
struct FormalNumericExceptionFlags {
  bool invalid = false;
  bool divByZero = false;
  bool overflow = false;
  bool underflow = false;
  bool inexact = false;

  constexpr bool any() const {
    return invalid || divByZero || overflow || underflow || inexact;
  }

  friend constexpr bool operator==(FormalNumericExceptionFlags lhs,
                                   FormalNumericExceptionFlags rhs) {
    return lhs.invalid == rhs.invalid && lhs.divByZero == rhs.divByZero &&
           lhs.overflow == rhs.overflow && lhs.underflow == rhs.underflow &&
           lhs.inexact == rhs.inexact;
  }
  friend constexpr bool operator!=(FormalNumericExceptionFlags lhs,
                                   FormalNumericExceptionFlags rhs) {
    return !(lhs == rhs);
  }
};

/// A completed formal result. `value` is always a canonical RawLogicalValue;
/// `flags` contains only the flags raised by this operation.
struct FormalNumericResult {
  RawLogicalValue value;
  FormalNumericExceptionFlags flags;
};

/// Stable typed reasons for a formal operation that cannot produce a result.
enum class FormalNumericErrorCode : uint8_t {
  UnsupportedResolvedCommand,
  OperandCountMismatch,
  OperandFormatMismatch,
  InvalidOperandEncoding,
  SourceFormatMismatch,
  InvalidSourceEncoding,
  UnsupportedEndpointKinds,
  FloatToIntegerNonFinite,
  FloatToIntegerOutOfRange,
  UnexpectedAPFloatStatus,
  InvalidResultEncoding,
  InvalidComparatorOperandEncoding,
};

llvm::StringRef stringifyFormalNumericErrorCode(FormalNumericErrorCode code);

class FormalNumericError final : public llvm::ErrorInfo<FormalNumericError> {
public:
  static char ID;

  FormalNumericError(FormalNumericErrorCode code, std::string message)
      : code(code), detail(std::move(message)) {}

  FormalNumericErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  FormalNumericErrorCode code;
  std::string detail;
};

/// Invocation-owned execution state. There is intentionally no process-global
/// exception state: separate callers receive independent contexts, while a
/// caller may use one context to aggregate the flags of its own operation
/// sequence.
class FormalNumericExecutionContext {
public:
  FormalNumericExecutionContext() = default;
  FormalNumericExecutionContext(const FormalNumericExecutionContext &) = delete;
  FormalNumericExecutionContext &
  operator=(const FormalNumericExecutionContext &) = delete;
  FormalNumericExecutionContext(FormalNumericExecutionContext &&) = default;
  FormalNumericExecutionContext &
  operator=(FormalNumericExecutionContext &&) = default;

  const FormalNumericExceptionFlags &getAggregateFlags() const {
    return aggregateFlags;
  }
  void clearAggregateFlags() { aggregateFlags = {}; }

  /// Merges flags after the caller has completed one whole atomic operation or
  /// tensor successfully. Scalar evaluators below are effect-free; a dispatcher
  /// must not call this on a partial or failing path.
  void mergeExceptionFlags(FormalNumericExceptionFlags flags);

private:
  FormalNumericExceptionFlags aggregateFlags;
};

/// Effect-free core for one supported CT-convert scalar. Every validation or
/// arithmetic failure returns an Error and has no external state to roll back.
llvm::Expected<FormalNumericResult>
evaluateFormalConvert(const ResolvedNumericCommand &command,
                      RawLogicalValue source);

/// Executes one supported resolved CT-convert command. Inputs and outputs are
/// target-independent raw logical encodings; LLVM APFloat/APInt are the
/// production formal arithmetic implementation. Every error, including the
/// profile-defined float-to-integer reject-no-write outcomes, leaves `context`
/// unchanged and returns no result.
llvm::Expected<FormalNumericResult>
executeFormalConvert(FormalNumericExecutionContext &context,
                     const ResolvedNumericCommand &command,
                     RawLogicalValue source);

/// Effect-free scalar evaluator for the LLVM APFloat/APInt subset of the
/// supported CT-elementwise registry. MPFR-classified operations are rejected
/// before arithmetic. `inputs` follows the exact command's typed arity.
llvm::Expected<FormalNumericResult>
evaluateFormalElementwiseLLVM(const ResolvedNumericCommand &command,
                              llvm::ArrayRef<RawLogicalValue> inputs);

/// Effect-free single fused MAC for one supported NE GEMM row. LHS/RHS use
/// the command's F16/BF16/F32 format, while `accumulator` and the returned
/// value are canonical F32. The operation is one F32 fusedMultiplyAdd at RNE.
llvm::Expected<FormalNumericResult>
evaluateFormalGemmFusedMultiplyAdd(const ResolvedNumericCommand &command,
                                   RawLogicalValue lhs, RawLogicalValue rhs,
                                   RawLogicalValue accumulator);

/// Effect-free final RNE write of a canonical F32 GEMM accumulator to the
/// command's original F16/BF16/F32 destination format.
llvm::Expected<FormalNumericResult>
evaluateFormalGemmFinalize(const ResolvedNumericCommand &command,
                           RawLogicalValue accumulator);

/// Effect-free single addition for the supported native F32 sum reduction.
/// Both operands and the returned accumulator are canonical F32. Each step is
/// rounded to nearest-even; tensor traversal order belongs to the tensor
/// dispatcher and is part of the resolved reduction policy.
llvm::Expected<FormalNumericResult>
evaluateFormalReduceStep(const ResolvedNumericCommand &command,
                         RawLogicalValue accumulator, RawLogicalValue input);

/// Raw-exact comparator selected by the first model profile. Both the logical
/// format/bits and the per-operation model flags participate in equality. Both
/// operands are validated as canonical before equality is evaluated.
llvm::Expected<bool>
compareFormalNumericResultsExact(const FormalNumericResult &lhs,
                                 const FormalNumericResult &rhs);

} // namespace wafer

#endif // WAFER_TARGET_FORMALNUMERIC_H
