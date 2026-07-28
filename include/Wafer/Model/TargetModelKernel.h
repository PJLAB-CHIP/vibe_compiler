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
  ManagedReferenceBackendUnavailable,
  ManagedReferenceBackendFailure,
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
  NCCJoin,
  DirectDTEBegin,
  DirectDTESend,
  DirectDTEReceive,
  DirectDTEWait,
  DirectDTEFinish,
};

enum class TargetModelNumericBackend : uint8_t {
  None,
  Formal,
  ManagedReference,
  Bulk,
};

struct TargetModelNumericTensor {
  NumericTensorKey key;
  std::vector<uint8_t> storage;
};

struct TargetModelNumericRequest {
  ResolvedNumericCommand command;
  std::vector<TargetModelNumericTensor> inputs;
  TargetModelNumericTensor destinationTemplate;
};

enum class TargetModelBulkProvenanceKind : uint8_t {
  None,
  ExactQualificationRecord,
  ManagedReferenceEnvironment,
};

struct TargetModelBulkDispatchEvidence {
  uint64_t matmulInvocations = 0;
  uint64_t reorderInvocations = 0;
  uint64_t formalFusedMultiplyAdds = 0;
  TargetModelBulkProvenanceKind provenanceKind =
      TargetModelBulkProvenanceKind::None;
  std::string provenanceDigest;
  std::string implementation;
};

struct TargetModelBulkResult {
  TargetModelNumericTensor destination;
  FormalNumericExceptionFlags flags;
  TargetModelBulkDispatchEvidence evidence;
};

/// Feature-independent dispatch seam. Implementations return a result only
/// when their explicit admission policy accepts the command, payload and
/// environment. Policies may be exact-record or a named managed-reference
/// domain; a successful null optional means no configured admission matched.
class TargetModelBulkBackend {
public:
  virtual ~TargetModelBulkBackend() = default;

  virtual llvm::Expected<std::optional<TargetModelBulkResult>>
  tryExecute(const TargetModelNumericRequest &request) const = 0;
};

struct TargetModelManagedReferenceEvidence {
  uint64_t scalarEvaluations = 0;
  std::string environmentDigest;
  std::string implementation;
};

struct TargetModelManagedReferenceResult {
  TargetModelNumericTensor destination;
  FormalNumericExceptionFlags flags;
  TargetModelManagedReferenceEvidence evidence;
};

/// Explicit model-reference tensor execution seam. It is available only to
/// the managed-reference policy and must either return one completely checked
/// result or fail; there is no scalar-formal fallback.
class TargetModelManagedReferenceBackend {
public:
  virtual ~TargetModelManagedReferenceBackend() = default;

  virtual llvm::Expected<TargetModelManagedReferenceResult>
  execute(const TargetModelNumericRequest &request,
          FormalNumericWorkBudget scalarBudget) const = 0;
};

enum class TargetModelGemmDispatchPolicy : uint8_t {
  FormalOnly,
  PreferAdmitted,
};

enum class TargetModelTensorDispatchPolicy : uint8_t {
  FormalOnly,
  ManagedReference,
};

class TargetModelExecutionPolicy {
public:
  static constexpr TargetModelExecutionPolicy formalOnly() {
    return TargetModelExecutionPolicy(
        TargetModelGemmDispatchPolicy::FormalOnly,
        TargetModelTensorDispatchPolicy::FormalOnly, nullptr, nullptr);
  }

  static TargetModelExecutionPolicy
  preferAdmitted(const TargetModelBulkBackend &backend) {
    return TargetModelExecutionPolicy(
        TargetModelGemmDispatchPolicy::PreferAdmitted,
        TargetModelTensorDispatchPolicy::FormalOnly, &backend, nullptr);
  }

  static TargetModelExecutionPolicy
  managedReference(const TargetModelBulkBackend &bulkBackend,
                   const TargetModelManagedReferenceBackend &tensorBackend) {
    return TargetModelExecutionPolicy(
        TargetModelGemmDispatchPolicy::PreferAdmitted,
        TargetModelTensorDispatchPolicy::ManagedReference, &bulkBackend,
        &tensorBackend);
  }

  TargetModelGemmDispatchPolicy getGemmDispatchPolicy() const {
    return gemmDispatchPolicy;
  }
  TargetModelTensorDispatchPolicy getTensorDispatchPolicy() const {
    return tensorDispatchPolicy;
  }
  const TargetModelBulkBackend *getBulkBackend() const { return bulkBackend; }
  const TargetModelManagedReferenceBackend *getManagedReferenceBackend() const {
    return managedReferenceBackend;
  }

private:
  constexpr TargetModelExecutionPolicy(
      TargetModelGemmDispatchPolicy gemmDispatchPolicy,
      TargetModelTensorDispatchPolicy tensorDispatchPolicy,
      const TargetModelBulkBackend *bulkBackend,
      const TargetModelManagedReferenceBackend *managedReferenceBackend)
      : gemmDispatchPolicy(gemmDispatchPolicy),
        tensorDispatchPolicy(tensorDispatchPolicy), bulkBackend(bulkBackend),
        managedReferenceBackend(managedReferenceBackend) {}

  TargetModelGemmDispatchPolicy gemmDispatchPolicy;
  TargetModelTensorDispatchPolicy tensorDispatchPolicy;
  const TargetModelBulkBackend *bulkBackend;
  const TargetModelManagedReferenceBackend *managedReferenceBackend;
};

/// Complete effect of one plain command. Numeric flags are still local here;
/// commitTargetModelCommandEffect publishes bytes first, then records flags.
struct TargetModelCommandEffect {
  std::vector<TargetModelByteWrite> pendingWrites;
  FormalNumericExceptionFlags numericFlags;
  TargetModelControlAction controlAction = TargetModelControlAction::None;
  TargetModelNumericBackend numericBackend = TargetModelNumericBackend::None;
  TargetModelBulkDispatchEvidence bulkEvidence;
  TargetModelManagedReferenceEvidence managedReferenceEvidence;
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
