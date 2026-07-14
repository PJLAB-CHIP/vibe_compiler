//===- TargetModelKernel.h - Plain target transaction kernels -*- C++ -*-===//

#ifndef WAFER_MODEL_TARGETMODELKERNEL_H
#define WAFER_MODEL_TARGETMODELKERNEL_H

#include "Wafer/Model/TargetModelMemory.h"
#include "Wafer/Target/FormalTensorNumeric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer::model {

enum class TargetModelKernelErrorCode : uint8_t {
  InvalidTransactionField,
  WorkBudgetExceeded,
  UnsupportedTransaction,
  MemoryReadFailure,
  NumericResolutionFailure,
  PhysicalCodecFailure,
};

llvm::StringRef
stringifyTargetModelKernelErrorCode(TargetModelKernelErrorCode code);

class TargetModelKernelError final
    : public llvm::ErrorInfo<TargetModelKernelError> {
public:
  static char ID;

  TargetModelKernelError(TargetModelKernelErrorCode code, std::string detail)
      : code(code), detail(std::move(detail)) {}

  TargetModelKernelErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  TargetModelKernelErrorCode code;
  std::string detail;
};

/// Explicit per-command bounds selected by the model integration layer. There
/// is no implicit work or allocation budget.
class TargetModelKernelBudget {
public:
  TargetModelKernelBudget() = delete;

  static constexpr TargetModelKernelBudget
  create(FormalNumericWorkBudget numeric, uint64_t maximumMovementBytes,
         uint64_t maximumMovementSegments) {
    return TargetModelKernelBudget(numeric, maximumMovementBytes,
                                   maximumMovementSegments);
  }

  FormalNumericWorkBudget getNumericBudget() const { return numeric; }
  uint64_t getMaximumMovementBytes() const { return maximumMovementBytes; }
  uint64_t getMaximumMovementSegments() const {
    return maximumMovementSegments;
  }

private:
  constexpr TargetModelKernelBudget(FormalNumericWorkBudget numeric,
                                    uint64_t maximumMovementBytes,
                                    uint64_t maximumMovementSegments)
      : numeric(numeric), maximumMovementBytes(maximumMovementBytes),
        maximumMovementSegments(maximumMovementSegments) {}

  FormalNumericWorkBudget numeric;
  uint64_t maximumMovementBytes;
  uint64_t maximumMovementSegments;
};

enum class TargetModelControlAction : uint8_t {
  None,
  LocalFence,
  DirectDTEBegin,
  DirectDTESend,
  DirectDTEReceive,
  DirectDTEWait,
  DirectDTEFinish,
};

/// Complete effect of one plain command. Numeric flags are still local here;
/// commitTargetModelCommandEffect publishes bytes first, then records flags.
struct TargetModelCommandEffect {
  std::vector<TargetModelByteWrite> pendingWrites;
  FormalNumericExceptionFlags numericFlags;
  TargetModelControlAction controlAction = TargetModelControlAction::None;
};

/// Exhaustive field/optional-field validation for every typed target payload.
/// A valid but not yet modeled family is distinguished later as unsupported.
llvm::Error validateTargetModelTransactionFields(
    const compiler::TargetTransaction &transaction);

/// Effect-free with respect to invocation state: reads private snapshots and
/// returns a complete pending effect. It never writes memory or records flags.
llvm::Expected<TargetModelCommandEffect>
executeTargetModelCommand(const compiler::TargetTransaction &transaction,
                          const InvocationMemoryRegistry &memory,
                          TargetModelKernelBudget budget);

/// Atomic publication point for one completed command effect.
llvm::Error
commitTargetModelCommandEffect(InvocationMemoryRegistry &memory,
                               FormalNumericExecutionContext &context,
                               TargetModelCommandEffect effect);

} // namespace wafer::model

#endif // WAFER_MODEL_TARGETMODELKERNEL_H
