//===- TargetModelKernel.h - Plain target command kernels -*- C++ -*-===//

#ifndef WAFER_SIMULATOR_KERNEL_TARGETMODELKERNEL_H
#define WAFER_SIMULATOR_KERNEL_TARGETMODELKERNEL_H

#include "Wafer/Simulator/Memory/TargetModelMemory.h"
#include "Wafer/Simulator/Reference/FormalTensorNumeric.h"

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
  InvalidCommandField,
  WorkBudgetExceeded,
  UnsupportedCommand,
  MemoryReadFailure,
  FormalNumericFailure,
  PhysicalCodecFailure,
  ManagedReferenceBackendUnavailable,
  ManagedReferenceBackendFailure,
  OneDNNBackendUnavailable,
  OneDNNBackendFailure,
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
  DirectDTESendPrepare,
  DirectDTESendIssue,
  DirectDTEReceive,
  DirectDTEWait,
  DirectDTEFinish,
};

enum class TargetModelNumericBackend : uint8_t {
  None,
  Formal,
  ManagedReference,
  OneDNN,
};

struct TargetModelNumericTensor {
  PhysicalTensorDescriptor key;
  std::vector<uint8_t> storage;
};

struct TargetModelNumericOperands {
  std::vector<TargetModelNumericTensor> inputs;
  TargetModelNumericTensor destinationTemplate;
};

struct TargetModelConvertRequest {
  FormalConvertOperation operation;
  TargetModelNumericOperands tensors;
};

struct TargetModelElementwiseRequest {
  FormalElementwiseOperation operation;
  TargetModelNumericOperands tensors;
};

struct TargetModelReduceRequest {
  FormalReduceOperation operation;
  TargetModelNumericOperands tensors;
};

struct TargetModelGemmRequest {
  FormalGemmOperation operation;
  TargetModelNumericOperands tensors;
};

enum class TargetModelOneDNNEvidenceKind : uint8_t {
  None,
  ExactQualificationRecord,
  ManagedReferenceEnvironment,
};

struct TargetModelOneDNNDispatchEvidence {
  uint64_t matmulInvocations = 0;
  uint64_t reorderInvocations = 0;
  uint64_t formalFusedMultiplyAdds = 0;
  TargetModelOneDNNEvidenceKind evidenceKind =
      TargetModelOneDNNEvidenceKind::None;
  std::string evidenceDigest;
  std::string implementation;
};

struct TargetModelOneDNNResult {
  TargetModelNumericTensor destination;
  FormalNumericExceptionFlags flags;
  TargetModelOneDNNDispatchEvidence evidence;
};

/// Feature-independent onednn backend interface. Implementations return a
/// result only when the command, payload and environment match either an exact
/// qualification record or the managed-reference execution domain. A
/// successful null optional means neither execution mode matched.
class TargetModelOneDNNBackend {
public:
  virtual ~TargetModelOneDNNBackend() = default;

  virtual llvm::Expected<std::optional<TargetModelOneDNNResult>>
  tryExecute(const TargetModelGemmRequest &request) const = 0;
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
  execute(const TargetModelConvertRequest &request,
          FormalNumericWorkBudget scalarBudget) const = 0;
  virtual llvm::Expected<TargetModelManagedReferenceResult>
  execute(const TargetModelElementwiseRequest &request,
          FormalNumericWorkBudget scalarBudget) const = 0;
  virtual llvm::Expected<TargetModelManagedReferenceResult>
  execute(const TargetModelReduceRequest &request,
          FormalNumericWorkBudget scalarBudget) const = 0;
};

enum class TargetModelGemmDispatchPolicy : uint8_t {
  FormalOnly,
  OneDNNThenFormal,
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
  onednnThenFormal(const TargetModelOneDNNBackend &backend) {
    return TargetModelExecutionPolicy(
        TargetModelGemmDispatchPolicy::OneDNNThenFormal,
        TargetModelTensorDispatchPolicy::FormalOnly, &backend, nullptr);
  }

  static TargetModelExecutionPolicy
  managedReference(const TargetModelOneDNNBackend &onednnBackend,
                   const TargetModelManagedReferenceBackend &tensorBackend) {
    return TargetModelExecutionPolicy(
        TargetModelGemmDispatchPolicy::OneDNNThenFormal,
        TargetModelTensorDispatchPolicy::ManagedReference, &onednnBackend,
        &tensorBackend);
  }

  TargetModelGemmDispatchPolicy getGemmDispatchPolicy() const {
    return gemmDispatchPolicy;
  }
  TargetModelTensorDispatchPolicy getTensorDispatchPolicy() const {
    return tensorDispatchPolicy;
  }
  const TargetModelOneDNNBackend *getOneDNNBackend() const {
    return onednnBackend;
  }
  const TargetModelManagedReferenceBackend *getManagedReferenceBackend() const {
    return managedReferenceBackend;
  }

private:
  constexpr TargetModelExecutionPolicy(
      TargetModelGemmDispatchPolicy gemmDispatchPolicy,
      TargetModelTensorDispatchPolicy tensorDispatchPolicy,
      const TargetModelOneDNNBackend *onednnBackend,
      const TargetModelManagedReferenceBackend *managedReferenceBackend)
      : gemmDispatchPolicy(gemmDispatchPolicy),
        tensorDispatchPolicy(tensorDispatchPolicy),
        onednnBackend(onednnBackend),
        managedReferenceBackend(managedReferenceBackend) {}

  TargetModelGemmDispatchPolicy gemmDispatchPolicy;
  TargetModelTensorDispatchPolicy tensorDispatchPolicy;
  const TargetModelOneDNNBackend *onednnBackend;
  const TargetModelManagedReferenceBackend *managedReferenceBackend;
};

/// Complete effect of one plain command. Numeric flags remain local until the
/// pending writes can be applied atomically.
struct TargetModelCommandEffect {
  std::vector<TargetModelByteWrite> pendingWrites;
  FormalNumericExceptionFlags numericFlags;
  TargetModelControlAction controlAction = TargetModelControlAction::None;
  TargetModelNumericBackend numericBackend = TargetModelNumericBackend::None;
  TargetModelOneDNNDispatchEvidence onednnEvidence;
  TargetModelManagedReferenceEvidence managedReferenceEvidence;
  std::vector<TargetModelByteRead> pendingReads;
};

/// Exhaustive field/optional-field validation for every typed target payload.
/// A valid but not yet modeled family is distinguished later as unsupported.
llvm::Error
validateTargetModelCommandFields(const compiler::TargetCommand &command);

/// Effect-free with respect to invocation state: reads private snapshots and
/// returns a complete pending effect. It never writes memory or records flags.
llvm::Expected<TargetModelCommandEffect>
executeTargetModelCommand(const compiler::TargetCommand &command,
                          const InvocationMemoryRegistry &memory,
                          TargetModelKernelBudget budget,
                          TargetModelExecutionPolicy policy =
                              TargetModelExecutionPolicy::formalOnly());

/// Applies one complete command effect atomically and then records its numeric
/// exception flags.
llvm::Error
applyTargetModelCommandEffect(InvocationMemoryRegistry &memory,
                              FormalNumericExecutionContext &context,
                              TargetModelCommandEffect effect);

} // namespace wafer::model

#endif // WAFER_SIMULATOR_KERNEL_TARGETMODELKERNEL_H
