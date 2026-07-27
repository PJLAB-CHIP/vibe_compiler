//===- PackageManifest.h - Typed Wafer package format ----------*- C++ -*-===//

#ifndef WAFER_RUNTIME_PACKAGEMANIFEST_H
#define WAFER_RUNTIME_PACKAGEMANIFEST_H

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "Wafer/Target/RuntimeLaunchContract.h"
#include "Wafer/Target/TargetProfile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::runtime {

inline constexpr uint32_t kPackageManifestSchemaVersion = 6;
inline constexpr llvm::StringLiteral kPackageManifestFileName = "manifest.json";
inline constexpr llvm::StringLiteral kDirectDTEStatusABIV1 =
    WAFER_TX81_DIRECT_DTE_STATUS_ABI_V1;
inline constexpr llvm::StringLiteral kDirectDTEStatusABIV2 =
    WAFER_TX81_DIRECT_DTE_STATUS_ABI_V2;
inline constexpr llvm::StringLiteral kDirectDTEStatusABI =
    WAFER_TX81_DIRECT_DTE_STATUS_ABI_V2;
enum class DirectDTEStatusValue : uint32_t {
  Pending = WAFER_TX81_DIRECT_DTE_STATUS_PENDING,
  Success = WAFER_TX81_DIRECT_DTE_STATUS_SUCCESS,
  TransportError = WAFER_TX81_DIRECT_DTE_STATUS_TRANSPORT_ERROR,
};
inline constexpr uint32_t kDirectDTEStatusPoison =
    WAFER_TX81_DIRECT_DTE_STATUS_POISON;
inline constexpr uint64_t kDirectDTEStatusValueOffset =
    WAFER_TX81_DIRECT_DTE_STATUS_V2_VALUE_OFFSET;
inline constexpr uint64_t kDirectDTEStatusValueBytes =
    WAFER_TX81_DIRECT_DTE_STATUS_V2_VALUE_BYTES;
inline constexpr uint64_t kDirectDTEStatusStorageBytes =
    WAFER_TX81_DIRECT_DTE_STATUS_V2_STORAGE_BYTES;
inline constexpr uint64_t kDirectDTEStatusStorageAlignment =
    WAFER_TX81_DIRECT_DTE_STATUS_V2_STORAGE_ALIGNMENT;
static_assert(kDirectDTEStatusValueBytes == sizeof(uint32_t));
static_assert(kDirectDTEStatusValueOffset + kDirectDTEStatusValueBytes <=
              kDirectDTEStatusStorageBytes);

template <typename Tag> class StrongId {
public:
  StrongId() = default;
  explicit StrongId(uint64_t value) : value(value) {}

  uint64_t getValue() const { return value; }
  bool isValid() const { return value != std::numeric_limits<uint64_t>::max(); }

  friend bool operator==(StrongId lhs, StrongId rhs) {
    return lhs.value == rhs.value;
  }
  friend bool operator!=(StrongId lhs, StrongId rhs) { return !(lhs == rhs); }
  friend bool operator<(StrongId lhs, StrongId rhs) {
    return lhs.value < rhs.value;
  }

private:
  uint64_t value = std::numeric_limits<uint64_t>::max();
};

struct ProgramIdTag;
struct ResourceIdTag;
struct ModuleIdTag;
struct EntryIdTag;
struct CompletionIdTag;
using ProgramId = StrongId<ProgramIdTag>;
using ResourceId = StrongId<ResourceIdTag>;
using ModuleId = StrongId<ModuleIdTag>;
using EntryId = StrongId<EntryIdTag>;
using CompletionId = StrongId<CompletionIdTag>;

enum class PackageResourceRole {
  UserInput,
  Parameter,
  Constant,
  Output,
  Workspace,
  TransportStatus,
};

enum class PackageAccessMode { ReadOnly, WriteOnly, ReadWrite };

struct PackageTensorType {
  std::string dtype;
  std::vector<int64_t> shape;
};

struct PackageResourceRecord {
  ResourceId id;
  int64_t logicalRank = -1;
  PackageResourceRole role = PackageResourceRole::UserInput;
  int64_t roleIndex = -1;
  std::string name;
  PackageTensorType type;
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  PackageAccessMode access = PackageAccessMode::ReadOnly;
  bool hostVisible = false;
};

struct PackageABISlotBinding {
  uint64_t ordinal = std::numeric_limits<uint64_t>::max();
  ResourceId resource;
  PackageAccessMode access = PackageAccessMode::ReadOnly;
};

enum class PackageModuleExportRole { Prepare, Main };

struct PackageModuleExportRecord {
  PackageModuleExportRole role = PackageModuleExportRole::Main;
  std::string symbol;
};

struct PackageModuleRecord {
  ModuleId id;
  std::string relativePath;
  std::string digest;
  std::string format;
  std::vector<PackageModuleExportRecord> exports;
};

struct NoTransportRequirements {};

struct DirectDTETransportRequirements {
  ResourceId statusResource;
  std::string statusABI = kDirectDTEStatusABI.str();
  bool hostWatchdogRequired = true;
};

using TransportRequirements =
    std::variant<NoTransportRequirements, DirectDTETransportRequirements>;

struct PackageEntrypointRecord {
  EntryId id;
  int64_t logicalRank = -1;
  ModuleId module;
  std::vector<PackageABISlotBinding> slots;
  CompletionId terminalCompletion;
  TransportRequirements transport;
};

struct PackageCompletionRecord {
  CompletionId id;
  int64_t logicalRank = -1;
  std::string kind;
};

struct PackageManifest {
  PackageManifest(TargetProfileId targetProfile,
                  TargetIdentityId targetIdentity,
                  KernelRuntimeABIId runtimeABI, RuntimeLaunchContract launch,
                  llvm::StringRef moduleFormat)
      : targetProfile(targetProfile), targetIdentity(targetIdentity),
        runtimeABI(runtimeABI), launch(std::move(launch)),
        moduleFormat(moduleFormat.str()) {}

  uint32_t schemaVersion = kPackageManifestSchemaVersion;
  ProgramId program;
  TargetProfileId targetProfile;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId runtimeABI;
  RuntimeLaunchContract launch;
  std::string moduleFormat;
  int64_t rankCount = 0;
  std::vector<PackageResourceRecord> resources;
  std::vector<PackageModuleRecord> modules;
  std::vector<PackageEntrypointRecord> entries;
  std::vector<PackageCompletionRecord> completions;
};

struct PackageParseLimits {
  uint64_t maxJSONBytes = 4 * 1024 * 1024;
  uint64_t maxRecords = 65536;
  uint64_t maxStringBytes = 4096;
  uint64_t maxShapeRank = 16;
  uint64_t maxJSONNesting = 32;
};

class VerifiedPackageManifest {
public:
  VerifiedPackageManifest(VerifiedPackageManifest &&) = default;
  VerifiedPackageManifest &operator=(VerifiedPackageManifest &&) = default;
  VerifiedPackageManifest(const VerifiedPackageManifest &) = delete;
  VerifiedPackageManifest &operator=(const VerifiedPackageManifest &) = delete;

  const PackageManifest &getManifest() const { return manifest; }

private:
  friend llvm::Expected<VerifiedPackageManifest>
  verifyPackageManifest(PackageManifest, llvm::StringRef,
                        const PackageParseLimits &);

  explicit VerifiedPackageManifest(PackageManifest manifest)
      : manifest(std::move(manifest)) {}

  PackageManifest manifest;
};

llvm::StringRef stringifyPackageResourceRole(PackageResourceRole role);
llvm::StringRef stringifyPackageAccessMode(PackageAccessMode access);
llvm::StringRef stringifyPackageModuleExportRole(PackageModuleExportRole role);

llvm::Expected<VerifiedPackageManifest>
verifyPackageManifest(PackageManifest manifest, llvm::StringRef packageRoot,
                      const PackageParseLimits &limits = {});

std::string
serializeCanonicalPackageJson(const VerifiedPackageManifest &manifest);

llvm::Expected<VerifiedPackageManifest>
parseCanonicalPackageJson(llvm::StringRef json, llvm::StringRef packageRoot,
                          const PackageParseLimits &limits = {});

llvm::Expected<VerifiedPackageManifest>
loadVerifiedPackageManifest(llvm::StringRef packageRoot,
                            const PackageParseLimits &limits = {});

struct RuntimeInvocationBinding {
  ResourceId resource;
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  PackageAccessMode access = PackageAccessMode::ReadOnly;
  bool hostVisible = false;
};

struct RuntimeEnvironment {
  RuntimeEnvironment(
      TargetProfileId targetProfile, TargetIdentityId targetIdentity,
      KernelRuntimeABIId runtimeABI, llvm::StringRef moduleFormat,
      uint64_t maxResourceBytes = std::numeric_limits<uint64_t>::max())
      : targetProfile(targetProfile), targetIdentity(targetIdentity),
        runtimeABI(runtimeABI), moduleFormat(moduleFormat.str()),
        maxResourceBytes(maxResourceBytes) {}

  TargetProfileId targetProfile;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId runtimeABI;
  std::string moduleFormat;
  uint64_t maxResourceBytes = std::numeric_limits<uint64_t>::max();
  std::vector<KernelLaunchForm> supportedKernelLaunchForms;
  std::vector<KernelEntryABI> supportedKernelEntryABIs;
  std::vector<ModelEntryABI> supportedModelEntryABIs;
  bool supportsDirectDTE = false;
  std::string directDTEStatusABI;
  bool supportsHostWatchdog = false;
};

struct PlannedRuntimeResource {
  ResourceId resource;
  PackageResourceRole role = PackageResourceRole::UserInput;
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  PackageAccessMode access = PackageAccessMode::ReadOnly;
  bool externallyBound = false;
};

struct PlannedRuntimeLaunchPhase {
  RuntimeLaunchPhaseRole role = RuntimeLaunchPhaseRole::Main;
  std::string symbol;
};

struct RuntimeSessionPlan {
  EntryId entry;
  int64_t logicalRank = -1;
  ModuleId module;
  std::string modulePath;
  std::vector<PlannedRuntimeLaunchPhase> phases;
  CompletionId terminalCompletion;
  std::vector<PlannedRuntimeResource> resources;
  std::vector<ResourceId> launchOrder;
  bool executesBoard = false;
  TransportRequirements transport;
};

struct RuntimeInvocationPlan {
  int64_t rankCount = 0;
  /// One record for every package rank, in canonical logical-rank order.
  std::vector<RuntimeSessionPlan> ranks;
};

llvm::Expected<RuntimeSessionPlan> preflightNoCardRuntimeSession(
    const VerifiedPackageManifest &package, EntryId entry,
    llvm::ArrayRef<RuntimeInvocationBinding> invocationBindings,
    const RuntimeEnvironment &environment);

llvm::Expected<RuntimeInvocationPlan> preflightNoCardRuntimeInvocation(
    const VerifiedPackageManifest &package,
    llvm::ArrayRef<RuntimeInvocationBinding> invocationBindings,
    const RuntimeEnvironment &environment);

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_PACKAGEMANIFEST_H
