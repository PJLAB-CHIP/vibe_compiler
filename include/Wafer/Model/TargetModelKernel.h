//===- TargetModelKernel.h - Plain target transaction kernels -*- C++ -*-===//

#ifndef WAFER_MODEL_TARGETMODELKERNEL_H
#define WAFER_MODEL_TARGETMODELKERNEL_H

#include "Wafer/Model/TargetModelMemory.h"
#include "Wafer/Target/FormalTensorNumeric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
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
  BulkBackendUnavailable,
  BulkBackendFailure,
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

enum class TargetModelNumericBackend : uint8_t { None, Formal, Bulk };

struct TargetModelBulkTensor {
  NumericTensorKey key;
  std::vector<uint8_t> storage;
};

struct TargetModelBulkRequest {
  ResolvedNumericCommand command;
  std::vector<TargetModelBulkTensor> inputs;
  TargetModelBulkTensor destinationTemplate;
};

struct TargetModelBulkDispatchEvidence {
  uint64_t matmulInvocations = 0;
  uint64_t reorderInvocations = 0;
  uint64_t formalFusedMultiplyAdds = 0;
  std::string admissionRecordDigest;
  std::string implementation;
};

struct TargetModelBulkResult {
  TargetModelBulkTensor destination;
  FormalNumericExceptionFlags flags;
  TargetModelBulkDispatchEvidence evidence;
};

/// Feature-independent dispatch seam. Implementations may only return a
/// result for an exact admitted command/payload/environment row. A successful
/// null optional means that no configured admission matched.
class TargetModelBulkBackend {
public:
  virtual ~TargetModelBulkBackend() = default;

  virtual llvm::Expected<std::optional<TargetModelBulkResult>>
  tryExecute(const TargetModelBulkRequest &request) const = 0;
};

enum class TargetModelGemmDispatchPolicy : uint8_t {
  FormalOnly,
  PreferAdmitted,
};

class TargetModelExecutionPolicy {
public:
  static constexpr TargetModelExecutionPolicy formalOnly() {
    return TargetModelExecutionPolicy(TargetModelGemmDispatchPolicy::FormalOnly,
                                      nullptr);
  }

  static TargetModelExecutionPolicy
  preferAdmitted(const TargetModelBulkBackend &backend) {
    return TargetModelExecutionPolicy(
        TargetModelGemmDispatchPolicy::PreferAdmitted, &backend);
  }

  TargetModelGemmDispatchPolicy getGemmDispatchPolicy() const {
    return gemmDispatchPolicy;
  }
  const TargetModelBulkBackend *getBulkBackend() const { return bulkBackend; }

private:
  constexpr TargetModelExecutionPolicy(
      TargetModelGemmDispatchPolicy gemmDispatchPolicy,
      const TargetModelBulkBackend *bulkBackend)
      : gemmDispatchPolicy(gemmDispatchPolicy), bulkBackend(bulkBackend) {}

  TargetModelGemmDispatchPolicy gemmDispatchPolicy;
  const TargetModelBulkBackend *bulkBackend;
};

/// Complete effect of one plain command. Numeric flags are still local here;
/// commitTargetModelCommandEffect publishes bytes first, then records flags.
struct TargetModelCommandEffect {
  std::vector<TargetModelByteWrite> pendingWrites;
  FormalNumericExceptionFlags numericFlags;
  TargetModelControlAction controlAction = TargetModelControlAction::None;
  TargetModelNumericBackend numericBackend = TargetModelNumericBackend::None;
  TargetModelBulkDispatchEvidence bulkEvidence;
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
                          TargetModelKernelBudget budget,
                          TargetModelExecutionPolicy policy =
                              TargetModelExecutionPolicy::formalOnly());

/// Atomic publication point for one completed command effect.
llvm::Error
commitTargetModelCommandEffect(InvocationMemoryRegistry &memory,
                               FormalNumericExecutionContext &context,
                               TargetModelCommandEffect effect);

} // namespace wafer::model

#endif // WAFER_MODEL_TARGETMODELKERNEL_H
