//===- BulkTensorNumeric.h - Qualified bulk tensor execution -*- C++ -*-===//

#ifndef WAFER_TARGET_BULKTENSORNUMERIC_H
#define WAFER_TARGET_BULKTENSORNUMERIC_H

#include "Wafer/Target/FormalTensorNumeric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer {

/// Caller-owned allocation limits for one bulk command. Every byte category
/// must be selected explicitly before target storage is decoded or allocated.
class BulkNumericWorkBudget {
public:
  BulkNumericWorkBudget() = delete;

  static constexpr BulkNumericWorkBudget create(uint64_t maximumTotalBytes,
                                                uint64_t maximumScratchpadBytes,
                                                uint64_t maximumReorderBytes) {
    return BulkNumericWorkBudget(maximumTotalBytes, maximumScratchpadBytes,
                                 maximumReorderBytes);
  }

  uint64_t getMaximumTotalBytes() const { return maximumTotalBytes; }
  uint64_t getMaximumScratchpadBytes() const { return maximumScratchpadBytes; }
  uint64_t getMaximumReorderBytes() const { return maximumReorderBytes; }

private:
  constexpr BulkNumericWorkBudget(uint64_t maximumTotalBytes,
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
class BulkTensorStorage {
public:
  BulkTensorStorage() = delete;

  static llvm::Expected<BulkTensorStorage> create(NumericTensorKey key,
                                                  std::vector<uint8_t> storage);

  const NumericTensorKey &getKey() const { return key; }
  llvm::ArrayRef<uint8_t> getStorage() const { return storage; }

private:
  BulkTensorStorage(NumericTensorKey key, std::vector<uint8_t> storage)
      : key(std::move(key)), storage(std::move(storage)) {}

  NumericTensorKey key;
  std::vector<uint8_t> storage;
};

/// Version and dependency records for the managed backend linked into this
/// process.
class BulkBackendDescriptor {
public:
  BulkBackendDescriptor() = delete;

  llvm::StringRef getName() const { return name; }
  llvm::StringRef getVersion() const { return version; }
  llvm::StringRef getCommit() const { return commit; }
  llvm::StringRef getDependencyRecordDigest() const {
    return dependencyRecordDigest;
  }
  llvm::StringRef getLibraryDigest() const { return libraryDigest; }
  llvm::StringRef getDigest() const { return digest; }

private:
  friend llvm::Expected<class BulkExecutionEnvironment>
  createManagedBulkExecutionEnvironment();

  BulkBackendDescriptor(std::string name, std::string version,
                        std::string commit, std::string dependencyRecordDigest,
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
class BulkExecutionEnvironment {
public:
  BulkExecutionEnvironment() = delete;

  const BulkBackendDescriptor &getBackend() const { return backend; }
  llvm::StringRef getHostCPUName() const { return hostCPUName; }
  llvm::StringRef getHostFeaturesDigest() const { return hostFeaturesDigest; }
  llvm::StringRef getHostPlatformDigest() const { return hostPlatformDigest; }
  llvm::StringRef getEffectiveISA() const { return effectiveISA; }
  int getFloatingRoundingMode() const { return floatingRoundingMode; }
  uint32_t getMXCSR() const { return mxcsr; }
  llvm::StringRef getThreadRuntime() const { return threadRuntime; }
  llvm::StringRef getDigest() const { return digest; }

private:
  friend llvm::Expected<BulkExecutionEnvironment>
  createManagedBulkExecutionEnvironment();

  BulkExecutionEnvironment(BulkBackendDescriptor backend,
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

  BulkBackendDescriptor backend;
  std::string hostCPUName;
  std::string hostFeaturesDigest;
  std::string hostPlatformDigest;
  std::string effectiveISA;
  int floatingRoundingMode;
  uint32_t mxcsr;
  std::string threadRuntime;
  std::string digest;
};

enum class BulkQualificationKind : uint8_t { BitExact, ProfileBounded };

llvm::StringRef stringifyBulkQualificationKind(BulkQualificationKind kind);

/// Digest of the target-owned raw-layout -> f32 MatMul -> formal destination
/// conversion contract.
llvm::StringRef getBulkAdapterContractDigest();

/// Immutable exact-match capability issued only by a read-back validated final
/// qualification record. Callers cannot manufacture this token directly.
class QualifiedBulkExecution {
public:
  QualifiedBulkExecution() = delete;

  llvm::StringRef getRecordDigest() const { return recordDigest; }
  llvm::StringRef getAdapterDigest() const { return adapterDigest; }
  llvm::StringRef getSemanticProfileDigest() const {
    return semanticProfileDigest;
  }
  llvm::StringRef getResolutionDigest() const { return resolutionDigest; }
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
  BulkQualificationKind getKind() const { return kind; }
  FormalNumericExceptionFlags getFormalFlags() const { return formalFlags; }

private:
  friend class VerifiedBulkQualificationRecord;

  QualifiedBulkExecution(
      std::string recordDigest, std::string adapterDigest,
      std::string semanticProfileDigest, std::string resolutionDigest,
      std::string inputPayloadDigest, std::string destinationTemplateDigest,
      std::string environmentDigest, std::string expectedBackendOutputDigest,
      std::string expectedImplementation,
      std::string expectedResolvedDescriptorDigest, BulkQualificationKind kind,
      FormalNumericExceptionFlags formalFlags)
      : recordDigest(std::move(recordDigest)),
        adapterDigest(std::move(adapterDigest)),
        semanticProfileDigest(std::move(semanticProfileDigest)),
        resolutionDigest(std::move(resolutionDigest)),
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
  std::string semanticProfileDigest;
  std::string resolutionDigest;
  std::string inputPayloadDigest;
  std::string destinationTemplateDigest;
  std::string environmentDigest;
  std::string expectedBackendOutputDigest;
  std::string expectedImplementation;
  std::string expectedResolvedDescriptorDigest;
  BulkQualificationKind kind;
  FormalNumericExceptionFlags formalFlags;
};

struct BulkDispatchEvidence {
  uint64_t matmulInvocations = 0;
  uint64_t reorderInvocations = 0;
  uint64_t formalFusedMultiplyAdds = 0;
  uint64_t allocatedBytes = 0;
  uint64_t scratchpadBytes = 0;
  std::string implementation;
  std::string resolvedDescriptorDigest;
};

struct BulkTensorNumericResult {
  BulkTensorStorage destination;
  FormalNumericExceptionFlags flags;
  BulkDispatchEvidence evidence;
};

enum class BulkTensorNumericErrorCode : uint8_t {
  UnsupportedResolvedCommand,
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
stringifyBulkTensorNumericErrorCode(BulkTensorNumericErrorCode code);

class BulkTensorNumericError final
    : public llvm::ErrorInfo<BulkTensorNumericError> {
public:
  static char ID;

  BulkTensorNumericError(BulkTensorNumericErrorCode code, std::string detail)
      : code(code), detail(std::move(detail)) {}

  BulkTensorNumericErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  BulkTensorNumericErrorCode code;
  std::string detail;
};

/// Configures process-global oneDNN policy before creating an engine and
/// returns the read-back execution environment. A conflicting pre-initialized
/// oneDNN process fails rather than inheriting ambient state.
llvm::Expected<BulkExecutionEnvironment>
createManagedBulkExecutionEnvironment();

/// Computes the exact physical footprint using the shared Wafer layout helper.
llvm::Expected<uint64_t>
getBulkTensorPhysicalBytes(const NumericTensorKey &key);

/// Target-owned codec/layout adapters. Packing starts from the supplied fill
/// byte so padding and tail preservation can be tested explicitly.
llvm::Expected<std::vector<RawLogicalValue>>
unpackBulkTensorLogicalValues(const BulkTensorStorage &tensor);
llvm::Expected<BulkTensorStorage>
packBulkTensorLogicalValues(const NumericTensorKey &key,
                            llvm::ArrayRef<RawLogicalValue> values,
                            uint8_t paddingFill);

/// The payload digest covers ordered physical input bytes and their typed keys.
/// Destination templates are digested separately by the caller.
std::string
computeBulkTensorPayloadDigest(llvm::ArrayRef<BulkTensorStorage> tensors);
std::string computeBulkTensorStorageDigest(const BulkTensorStorage &tensor);

/// Executes exactly one qualified MatMul. The command, qualification record,
/// environment, physical layout, encoding and byte budgets are checked before
/// backend
/// execution. The caller's inputs remain immutable; the destination is returned
/// only after its digest matches the validated record.
llvm::Expected<BulkTensorNumericResult>
executeQualifiedBulkTensorNumeric(const BulkExecutionEnvironment &environment,
                                  const QualifiedBulkExecution &execution,
                                  const ResolvedNumericCommand &command,
                                  llvm::ArrayRef<BulkTensorStorage> inputs,
                                  const BulkTensorStorage &destinationTemplate,
                                  BulkNumericWorkBudget budget);

/// Executes one deterministic managed-reference MatMul for a structurally
/// supported f16/bf16/f32 command whose physical inputs contain only finite
/// values. Unlike record-qualified execution, this scalable path is a
/// model-reference acceleration policy: it is not a claim of raw-exact
/// target arithmetic or hardware correlation. The caller must validate the
/// final model output against its external oracle with an explicit tolerance.
llvm::Expected<BulkTensorNumericResult>
executeManagedReferenceBulkTensorNumeric(
    const BulkExecutionEnvironment &environment,
    const ResolvedNumericCommand &command,
    llvm::ArrayRef<BulkTensorStorage> inputs,
    const BulkTensorStorage &destinationTemplate, BulkNumericWorkBudget budget);

} // namespace wafer

#endif // WAFER_TARGET_BULKTENSORNUMERIC_H
