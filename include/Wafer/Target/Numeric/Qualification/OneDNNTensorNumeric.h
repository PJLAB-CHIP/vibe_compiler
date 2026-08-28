//===- OneDNNTensorNumeric.h - Qualified onednn tensor execution -*- C++
//-*-===//

#ifndef WAFER_TARGET_ONEDNNTENSORNUMERIC_H
#define WAFER_TARGET_ONEDNNTENSORNUMERIC_H

#include "Wafer/Target/Numeric/Formal/FormalTensorNumeric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer {

/// Caller-owned allocation limits for one onednn operation. Every byte category
/// must be selected explicitly before target storage is decoded or allocated.
class OneDNNNumericWorkBudget {
public:
  OneDNNNumericWorkBudget() = delete;

  static constexpr OneDNNNumericWorkBudget
  create(uint64_t maximumTotalBytes, uint64_t maximumScratchpadBytes,
         uint64_t maximumReorderBytes) {
    return OneDNNNumericWorkBudget(maximumTotalBytes, maximumScratchpadBytes,
                                   maximumReorderBytes);
  }

  uint64_t getMaximumTotalBytes() const { return maximumTotalBytes; }
  uint64_t getMaximumScratchpadBytes() const { return maximumScratchpadBytes; }
  uint64_t getMaximumReorderBytes() const { return maximumReorderBytes; }

private:
  constexpr OneDNNNumericWorkBudget(uint64_t maximumTotalBytes,
                                    uint64_t maximumScratchpadBytes,
                                    uint64_t maximumReorderBytes)
      : maximumTotalBytes(maximumTotalBytes),
        maximumScratchpadBytes(maximumScratchpadBytes),
        maximumReorderBytes(maximumReorderBytes) {}

  uint64_t maximumTotalBytes;
  uint64_t maximumScratchpadBytes;
  uint64_t maximumReorderBytes;
};

/// One immutable target-physical tensor snapshot. The key owns format, layout
/// and shape; storage includes padding and Cx/NCx tails.
class OneDNNTensorStorage {
public:
  OneDNNTensorStorage() = delete;

  static llvm::Expected<OneDNNTensorStorage>
  create(PhysicalTensorDescriptor key, std::vector<uint8_t> storage);

  const PhysicalTensorDescriptor &getKey() const { return key; }
  llvm::ArrayRef<uint8_t> getStorage() const { return storage; }

private:
  OneDNNTensorStorage(PhysicalTensorDescriptor key,
                      std::vector<uint8_t> storage)
      : key(std::move(key)), storage(std::move(storage)) {}

  PhysicalTensorDescriptor key;
  std::vector<uint8_t> storage;
};

/// Version and dependency records for the managed backend linked into this
/// process.
class OneDNNBackendDescriptor {
public:
  OneDNNBackendDescriptor() = delete;

  llvm::StringRef getName() const { return name; }
  llvm::StringRef getVersion() const { return version; }
  llvm::StringRef getCommit() const { return commit; }
  llvm::StringRef getDependencyRecordDigest() const {
    return dependencyRecordDigest;
  }
  llvm::StringRef getLibraryDigest() const { return libraryDigest; }
  llvm::StringRef getDigest() const { return digest; }

private:
  friend llvm::Expected<class OneDNNExecutionEnvironment>
  createManagedOneDNNExecutionEnvironment();

  OneDNNBackendDescriptor(std::string name, std::string version,
                          std::string commit,
                          std::string dependencyRecordDigest,
                          std::string libraryDigest, std::string digest)
      : name(std::move(name)), version(std::move(version)),
        commit(std::move(commit)),
        dependencyRecordDigest(std::move(dependencyRecordDigest)),
        libraryDigest(std::move(libraryDigest)), digest(std::move(digest)) {}

  std::string name;
  std::string version;
  std::string commit;
  std::string dependencyRecordDigest;
  std::string libraryDigest;
  std::string digest;
};

/// Exact process/worker environment used by the initial sequential backend
/// profile. Its digest binds backend, host CPU/features, effective ISA,
/// floating environment, MXCSR and threading runtime.
class OneDNNExecutionEnvironment {
public:
  OneDNNExecutionEnvironment() = delete;

  const OneDNNBackendDescriptor &getBackend() const { return backend; }
  llvm::StringRef getHostCPUName() const { return hostCPUName; }
  llvm::StringRef getHostFeaturesDigest() const { return hostFeaturesDigest; }
  llvm::StringRef getHostPlatformDigest() const { return hostPlatformDigest; }
  llvm::StringRef getEffectiveISA() const { return effectiveISA; }
  int getFloatingRoundingMode() const { return floatingRoundingMode; }
  uint32_t getMXCSR() const { return mxcsr; }
  llvm::StringRef getThreadRuntime() const { return threadRuntime; }
  llvm::StringRef getDigest() const { return digest; }

private:
  friend llvm::Expected<OneDNNExecutionEnvironment>
  createManagedOneDNNExecutionEnvironment();

  OneDNNExecutionEnvironment(OneDNNBackendDescriptor backend,
                             std::string hostCPUName,
                             std::string hostFeaturesDigest,
                             std::string hostPlatformDigest,
                             std::string effectiveISA, int floatingRoundingMode,
                             uint32_t mxcsr, std::string threadRuntime,
                             std::string digest)
      : backend(std::move(backend)), hostCPUName(std::move(hostCPUName)),
        hostFeaturesDigest(std::move(hostFeaturesDigest)),
        hostPlatformDigest(std::move(hostPlatformDigest)),
        effectiveISA(std::move(effectiveISA)),
        floatingRoundingMode(floatingRoundingMode), mxcsr(mxcsr),
        threadRuntime(std::move(threadRuntime)), digest(std::move(digest)) {}

  OneDNNBackendDescriptor backend;
  std::string hostCPUName;
  std::string hostFeaturesDigest;
  std::string hostPlatformDigest;
  std::string effectiveISA;
  int floatingRoundingMode;
  uint32_t mxcsr;
  std::string threadRuntime;
  std::string digest;
};

enum class OneDNNQualificationKind : uint8_t { BitExact, ProfileBounded };

llvm::StringRef stringifyOneDNNQualificationKind(OneDNNQualificationKind kind);

/// Digest of the target-owned raw-layout -> f32 MatMul -> formal destination
/// conversion contract.
llvm::StringRef getOneDNNAdapterContractDigest();

/// Immutable exact-match capability issued only by a read-back validated final
/// qualification record. Callers cannot manufacture this token directly.
class QualifiedOneDNNExecution {
public:
  QualifiedOneDNNExecution() = delete;

  llvm::StringRef getRecordDigest() const { return recordDigest; }
  llvm::StringRef getAdapterDigest() const { return adapterDigest; }
  llvm::StringRef getProblemDigest() const { return problemDigest; }
  llvm::StringRef getInputPayloadDigest() const { return inputPayloadDigest; }
  llvm::StringRef getDestinationTemplateDigest() const {
    return destinationTemplateDigest;
  }
  llvm::StringRef getEnvironmentDigest() const { return environmentDigest; }
  llvm::StringRef getExpectedBackendOutputDigest() const {
    return expectedBackendOutputDigest;
  }
  llvm::StringRef getExpectedImplementation() const {
    return expectedImplementation;
  }
  llvm::StringRef getExpectedResolvedDescriptorDigest() const {
    return expectedResolvedDescriptorDigest;
  }
  OneDNNQualificationKind getKind() const { return kind; }
  FormalNumericExceptionFlags getFormalFlags() const { return formalFlags; }

private:
  friend class VerifiedOneDNNQualificationRecord;

  QualifiedOneDNNExecution(std::string recordDigest, std::string adapterDigest,
                           std::string problemDigest,
                           std::string inputPayloadDigest,
                           std::string destinationTemplateDigest,
                           std::string environmentDigest,
                           std::string expectedBackendOutputDigest,
                           std::string expectedImplementation,
                           std::string expectedResolvedDescriptorDigest,
                           OneDNNQualificationKind kind,
                           FormalNumericExceptionFlags formalFlags)
      : recordDigest(std::move(recordDigest)),
        adapterDigest(std::move(adapterDigest)),
        problemDigest(std::move(problemDigest)),
        inputPayloadDigest(std::move(inputPayloadDigest)),
        destinationTemplateDigest(std::move(destinationTemplateDigest)),
        environmentDigest(std::move(environmentDigest)),
        expectedBackendOutputDigest(std::move(expectedBackendOutputDigest)),
        expectedImplementation(std::move(expectedImplementation)),
        expectedResolvedDescriptorDigest(
            std::move(expectedResolvedDescriptorDigest)),
        kind(kind), formalFlags(formalFlags) {}

  std::string recordDigest;
  std::string adapterDigest;
  std::string problemDigest;
  std::string inputPayloadDigest;
  std::string destinationTemplateDigest;
  std::string environmentDigest;
  std::string expectedBackendOutputDigest;
  std::string expectedImplementation;
  std::string expectedResolvedDescriptorDigest;
  OneDNNQualificationKind kind;
  FormalNumericExceptionFlags formalFlags;
};

struct OneDNNDispatchEvidence {
  uint64_t matmulInvocations = 0;
  uint64_t reorderInvocations = 0;
  uint64_t formalFusedMultiplyAdds = 0;
  uint64_t allocatedBytes = 0;
  uint64_t scratchpadBytes = 0;
  std::string implementation;
  std::string resolvedDescriptorDigest;
};

struct OneDNNTensorNumericResult {
  OneDNNTensorStorage destination;
  FormalNumericExceptionFlags flags;
  OneDNNDispatchEvidence evidence;
};

enum class OneDNNTensorNumericErrorCode : uint8_t {
  UnsupportedOperation,
  UnsupportedFormat,
  InvalidPhysicalLayout,
  InvalidPhysicalStorage,
  InvalidInputEncoding,
  InputArityMismatch,
  QualificationMismatch,
  EnvironmentMismatch,
  WorkCountOverflow,
  TotalByteBudgetExceeded,
  ScratchpadBudgetExceeded,
  ReorderBudgetExceeded,
  BackendConfigurationFailure,
  BackendDescriptorFailure,
  BackendExecutionFailure,
  BackendOutputMismatch,
};

llvm::StringRef
stringifyOneDNNTensorNumericErrorCode(OneDNNTensorNumericErrorCode code);

class OneDNNTensorNumericError final
    : public llvm::ErrorInfo<OneDNNTensorNumericError> {
public:
  static char ID;

  OneDNNTensorNumericError(OneDNNTensorNumericErrorCode code,
                           std::string detail)
      : code(code), detail(std::move(detail)) {}

  OneDNNTensorNumericErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  OneDNNTensorNumericErrorCode code;
  std::string detail;
};

/// Configures process-global oneDNN policy before creating an engine and
/// returns the read-back execution environment. A conflicting pre-initialized
/// oneDNN process fails rather than inheriting ambient state.
llvm::Expected<OneDNNExecutionEnvironment>
createManagedOneDNNExecutionEnvironment();

/// Computes the exact physical footprint using the shared Wafer layout helper.
llvm::Expected<uint64_t>
getOneDNNTensorPhysicalBytes(const PhysicalTensorDescriptor &key);

/// Target-owned codec/layout adapters. Packing starts from the supplied fill
/// byte so padding and tail preservation can be tested explicitly.
llvm::Expected<std::vector<RawLogicalValue>>
unpackOneDNNTensorLogicalValues(const OneDNNTensorStorage &tensor);
llvm::Expected<OneDNNTensorStorage>
packOneDNNTensorLogicalValues(const PhysicalTensorDescriptor &key,
                              llvm::ArrayRef<RawLogicalValue> values,
                              uint8_t paddingFill);

/// The payload digest covers ordered physical input bytes and their typed keys.
/// Destination templates are digested separately by the caller.
std::string
computeOneDNNTensorPayloadDigest(llvm::ArrayRef<OneDNNTensorStorage> tensors);
std::string computeOneDNNTensorStorageDigest(const OneDNNTensorStorage &tensor);
llvm::Expected<std::string>
computeOneDNNGemmProblemDigest(const FormalGemmOperation &operation);

/// Executes exactly one qualified MatMul. The operation, qualification record,
/// environment, physical layout, encoding and byte budgets are checked before
/// backend
/// execution. The caller's inputs remain immutable; the destination is returned
/// only after its digest matches the validated record.
llvm::Expected<OneDNNTensorNumericResult> executeQualifiedOneDNNTensorNumeric(
    const OneDNNExecutionEnvironment &environment,
    const QualifiedOneDNNExecution &execution,
    const FormalGemmOperation &operation,
    llvm::ArrayRef<OneDNNTensorStorage> inputs,
    const OneDNNTensorStorage &destinationTemplate,
    OneDNNNumericWorkBudget budget);

/// Executes one deterministic managed-reference MatMul for a structurally
/// supported f16/bf16/f32 operation whose physical inputs contain only finite
/// values. Unlike record-qualified execution, this scalable path is a
/// model-reference acceleration policy: it is not a claim of raw-exact
/// target arithmetic or hardware correlation. The caller must validate the
/// final model output against its external oracle with an explicit tolerance.
llvm::Expected<OneDNNTensorNumericResult>
executeManagedReferenceOneDNNTensorNumeric(
    const OneDNNExecutionEnvironment &environment,
    const FormalGemmOperation &operation,
    llvm::ArrayRef<OneDNNTensorStorage> inputs,
    const OneDNNTensorStorage &destinationTemplate,
    OneDNNNumericWorkBudget budget);

} // namespace wafer

#endif // WAFER_TARGET_ONEDNNTENSORNUMERIC_H
