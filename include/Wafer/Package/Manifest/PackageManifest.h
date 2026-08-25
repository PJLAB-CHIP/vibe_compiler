//===- PackageManifest.h - Typed Wafer package format ----------*- C++ -*-===//

#ifndef WAFER_PACKAGE_PACKAGEMANIFEST_H
#define WAFER_PACKAGE_PACKAGEMANIFEST_H

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "Wafer/Program/ProgramElementType.h"
#include "Wafer/Target/Core/RuntimeLaunchContract.h"
#include "Wafer/Target/Core/TargetFormat.h"
#include "Wafer/Target/Core/TargetIdentity.h"
#include "Wafer/Target/Core/TopologyIds.h"
#include "Wafer/Target/Layout/PhysicalLayout.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::runtime {

inline constexpr llvm::StringLiteral kPackageManifestFileName = "manifest.json";
inline constexpr llvm::StringLiteral kPackageProgramDataRelativePath =
    "data/program-data.bin";
inline constexpr llvm::StringLiteral kDirectDTEStatusABI =
    WAFER_TX81_DIRECT_DTE_STATUS_ABI;
enum class DirectDTEStatusValue : uint32_t {
  Pending = WAFER_TX81_DIRECT_DTE_STATUS_PENDING,
  Success = WAFER_TX81_DIRECT_DTE_STATUS_SUCCESS,
  TransportError = WAFER_TX81_DIRECT_DTE_STATUS_TRANSPORT_ERROR,
};
inline constexpr uint32_t kDirectDTEStatusPoison =
    WAFER_TX81_DIRECT_DTE_STATUS_POISON;
inline constexpr uint64_t kDirectDTEStatusValueOffset =
    WAFER_TX81_DIRECT_DTE_STATUS_VALUE_OFFSET;
inline constexpr uint64_t kDirectDTEStatusValueBytes =
    WAFER_TX81_DIRECT_DTE_STATUS_VALUE_BYTES;
inline constexpr uint64_t kDirectDTEStatusStorageBytes =
    WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_BYTES;
inline constexpr uint64_t kDirectDTEStatusStorageAlignment =
    WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_ALIGNMENT;
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
struct ProgramTensorIdTag;
struct TargetTensorIdTag;
struct PortIdTag;
struct ModuleIdTag;
struct EntryIdTag;
struct LaunchSlotIdTag;
using ProgramId = StrongId<ProgramIdTag>;
using ProgramTensorId = StrongId<ProgramTensorIdTag>;
using TargetTensorId = StrongId<TargetTensorIdTag>;
using PortId = StrongId<PortIdTag>;
using ModuleId = StrongId<ModuleIdTag>;
using EntryId = StrongId<EntryIdTag>;
using LaunchSlotId = StrongId<LaunchSlotIdTag>;

/// Logical role of one package-owned program tensor. Parameters come from the
/// program argument domain, constants from the captured-constant domain.
enum class ProgramTensorRole { Parameter, Constant };

enum class PackageAccessMode { None, ReadOnly, WriteOnly, ReadWrite };

/// Closed physical memory layout family of the current target. Values match
/// the compiler IR MemLayout enumeration; the manifest keeps its own closed
/// copy so the runtime schema never consumes MLIR IR.
enum class PackageMemLayout : uint32_t {
  Tensor = 0,
  NTensor = 1,
  Cx = 2,
  NCx = 3,
};

/// Logical identity of one package-owned parameter/constant. Identity is the
/// program tensor's role/index and its verified partition slice; it is never
/// recovered from name, path, shape, or digest. The record does not carry a
/// source path.
struct ProgramTensorRecord {
  ProgramTensorId id;
  ProgramTensorRole role = ProgramTensorRole::Parameter;
  /// User-visible index within the program-boundary role domain.
  int64_t roleIndex = -1;
  /// Logical (source) dtype.
  ProgramElementType dtype = ProgramElementType::F32;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  /// Partition/slice identity over the global tensor.
  std::vector<int64_t> sliceOffsets;
  std::vector<int64_t> sliceSizes;

  friend bool operator==(const ProgramTensorRecord &lhs,
                         const ProgramTensorRecord &rhs) {
    return lhs.id == rhs.id && lhs.role == rhs.role &&
           lhs.roleIndex == rhs.roleIndex && lhs.dtype == rhs.dtype &&
           lhs.globalShape == rhs.globalShape &&
           lhs.localShape == rhs.localShape &&
           lhs.sliceOffsets == rhs.sliceOffsets &&
           lhs.sliceSizes == rhs.sliceSizes;
  }
};

/// One compiler-selected target representation of a program tensor. Sharing
/// is expressed exclusively by multiple Tile entry arguments referencing the
/// same TargetTensor; equal fields or ranges never merge automatically.
struct TargetTensorRecord {
  TargetTensorId id;
  ProgramTensorId programTensor;
  /// Selected target dtype.
  LogicalFormat dtype = LogicalFormat::F32;
  PackageMemLayout layout = PackageMemLayout::Tensor;
  /// Logical target shape.
  std::vector<int64_t> shape;
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  /// Byte offset of this target tensor's target-ready bytes inside
  /// program-data.bin. It is a file offset, never a device address or an
  /// allocation identity.
  uint64_t fileOffset = 0;

  friend bool operator==(const TargetTensorRecord &lhs,
                         const TargetTensorRecord &rhs) {
    return lhs.id == rhs.id && lhs.programTensor == rhs.programTensor &&
           lhs.dtype == rhs.dtype && lhs.layout == rhs.layout &&
           lhs.shape == rhs.shape && lhs.bytes == rhs.bytes &&
           lhs.alignment == rhs.alignment && lhs.fileOffset == rhs.fileOffset;
  }
};

/// The single program-data file record. The digest covers target bytes and
/// canonical zero padding of the whole file.
struct ProgramDataRecord {
  std::string relativePath = kPackageProgramDataRelativePath.str();
  uint64_t totalBytes = 0;
  uint64_t baseAlignment = 1;
  std::string digest;
};

/// One caller-visible program-boundary port (input or output). External
/// ports have no TargetTensor identity and no package bytes; the logical
/// descriptor lets the invocation adapter encode caller values, and the
/// target descriptor plus physical bytes/alignment is the selected runtime
/// representation.
struct ExternalPortRecord {
  PortId id;
  /// User-visible index within the port's program-boundary domain.
  int64_t roleIndex = -1;
  /// Logical (source) descriptor.
  ProgramElementType logicalDtype = ProgramElementType::F32;
  std::vector<int64_t> logicalShape;
  /// Selected target descriptor.
  LogicalFormat dtype = LogicalFormat::F32;
  PackageMemLayout layout = PackageMemLayout::Tensor;
  std::vector<int64_t> shape;
  uint64_t bytes = 0;
  uint64_t alignment = 0;

  friend bool operator==(const ExternalPortRecord &lhs,
                         const ExternalPortRecord &rhs) {
    return lhs.id == rhs.id && lhs.roleIndex == rhs.roleIndex &&
           lhs.logicalDtype == rhs.logicalDtype &&
           lhs.logicalShape == rhs.logicalShape && lhs.dtype == rhs.dtype &&
           lhs.layout == rhs.layout && lhs.shape == rhs.shape &&
           lhs.bytes == rhs.bytes && lhs.alignment == rhs.alignment;
  }
};

/// Closed union of what one ordered Tile entry argument references.
struct ExternalInputArgument {
  PortId port;
  friend bool operator==(const ExternalInputArgument &lhs,
                         const ExternalInputArgument &rhs) {
    return lhs.port == rhs.port;
  }
  friend bool operator!=(const ExternalInputArgument &lhs,
                         const ExternalInputArgument &rhs) {
    return !(lhs == rhs);
  }
};
struct TargetTensorArgument {
  TargetTensorId tensor;
  friend bool operator==(const TargetTensorArgument &lhs,
                         const TargetTensorArgument &rhs) {
    return lhs.tensor == rhs.tensor;
  }
  friend bool operator!=(const TargetTensorArgument &lhs,
                         const TargetTensorArgument &rhs) {
    return !(lhs == rhs);
  }
};
struct ExternalOutputArgument {
  PortId port;
  friend bool operator==(const ExternalOutputArgument &lhs,
                         const ExternalOutputArgument &rhs) {
    return lhs.port == rhs.port;
  }
  friend bool operator!=(const ExternalOutputArgument &lhs,
                         const ExternalOutputArgument &rhs) {
    return !(lhs == rhs);
  }
};
/// Entry-local compiler-managed default DDR arena.
struct WorkspaceArgument {
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  friend bool operator==(const WorkspaceArgument &lhs,
                         const WorkspaceArgument &rhs) {
    return lhs.bytes == rhs.bytes && lhs.alignment == rhs.alignment;
  }
  friend bool operator!=(const WorkspaceArgument &lhs,
                         const WorkspaceArgument &rhs) {
    return !(lhs == rhs);
  }
};
/// Card-shared compiler-managed DDR carrier.
struct CardWorkspaceArgument {
  uint64_t resource = std::numeric_limits<uint64_t>::max();
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  friend bool operator==(const CardWorkspaceArgument &lhs,
                         const CardWorkspaceArgument &rhs) {
    return lhs.resource == rhs.resource && lhs.bytes == rhs.bytes &&
           lhs.alignment == rhs.alignment;
  }
  friend bool operator!=(const CardWorkspaceArgument &lhs,
                         const CardWorkspaceArgument &rhs) {
    return !(lhs == rhs);
  }
};
/// Entry-local profiler capture record.
struct ProfileRecordArgument {
  std::string recordABI;
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  friend bool operator==(const ProfileRecordArgument &lhs,
                         const ProfileRecordArgument &rhs) {
    return lhs.recordABI == rhs.recordABI && lhs.bytes == rhs.bytes &&
           lhs.alignment == rhs.alignment;
  }
  friend bool operator!=(const ProfileRecordArgument &lhs,
                         const ProfileRecordArgument &rhs) {
    return !(lhs == rhs);
  }
};
/// Entry-local Direct-DTE status.
struct TransportStatusArgument {
  std::string statusABI;
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  friend bool operator==(const TransportStatusArgument &lhs,
                         const TransportStatusArgument &rhs) {
    return lhs.statusABI == rhs.statusABI && lhs.bytes == rhs.bytes &&
           lhs.alignment == rhs.alignment;
  }
  friend bool operator!=(const TransportStatusArgument &lhs,
                         const TransportStatusArgument &rhs) {
    return !(lhs == rhs);
  }
};

using TileEntryArgumentReference =
    std::variant<ExternalInputArgument, TargetTensorArgument,
                 ExternalOutputArgument, WorkspaceArgument,
                 CardWorkspaceArgument, ProfileRecordArgument,
                 TransportStatusArgument>;

/// One ordered argument of one Tile target entry. Ordinals are dense and
/// zero-based. The argument carries a closed reference and an access mode; it
/// never owns bytes, a file range, a device address, or pointer-row storage.
struct TileEntryArgumentRecord {
  uint64_t ordinal = std::numeric_limits<uint64_t>::max();
  TileEntryArgumentReference reference;
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
  std::string statusABI = kDirectDTEStatusABI.str();
  bool hostWatchdogRequired = true;
};

using TransportRequirements =
    std::variant<NoTransportRequirements, DirectDTETransportRequirements>;

enum class PackageEntryCompletionKind { ReturnAfterLocalDrain };

struct PackageEntrypointRecord {
  EntryId id;
  CardId cardId{0};
  TileId tileId{0};
  LaunchSlotId launchSlot;
  ModuleId module;
  std::vector<TileEntryArgumentRecord> arguments;
  PackageEntryCompletionKind completion =
      PackageEntryCompletionKind::ReturnAfterLocalDrain;
  TransportRequirements transport;
};

struct PackageManifest {
  PackageManifest(TargetIdentityId targetIdentity,
                  KernelRuntimeABIId runtimeABI, RuntimeLaunchContract launch,
                  llvm::StringRef moduleFormat)
      : targetIdentity(targetIdentity), runtimeABI(runtimeABI),
        launch(std::move(launch)), moduleFormat(moduleFormat.str()) {}

  ProgramId program;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId runtimeABI;
  RuntimeLaunchContract launch;
  std::string moduleFormat;
  int64_t cardCount = 0;
  int64_t tileCount = 0;
  ProgramDataRecord programData;
  std::vector<ProgramTensorRecord> programTensors;
  std::vector<TargetTensorRecord> targetTensors;
  std::vector<ExternalPortRecord> inputs;
  std::vector<ExternalPortRecord> outputs;
  std::vector<PackageModuleRecord> modules;
  std::vector<PackageEntrypointRecord> entries;
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

namespace detail {

/// Pre-publication package resource binding. It owns the exact canonical
/// manifest, module and program-data buffers whose bytes were checked against
/// `manifest`, but deliberately carries no committed root identity.
struct BoundExecutablePackage {
  VerifiedPackageManifest manifest;
  std::unique_ptr<llvm::MemoryBuffer> manifestBuffer;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> moduleBuffers;
  std::unique_ptr<llvm::MemoryBuffer> programDataBuffer;
};

struct ExecutablePackageFactory;

llvm::Expected<BoundExecutablePackage>
bindExecutablePackage(llvm::StringRef packageRoot,
                      const PackageParseLimits &limits = {});

/// Opens one regular file by descriptor, derives its size from that descriptor,
/// reads an owned snapshot and closes the descriptor on every exit. The
/// returned bytes are independent of later path replacement and same-inode
/// mutation.
llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
openPackageMember(llvm::StringRef path, llvm::StringRef description);

} // namespace detail

/// The single move-only owner used by compiler results and runtime loaders for
/// a committed executable package. It owns the canonical root identity, the
/// verified manifest, and the exact all-and-only member buffers. Deleting or
/// replacing member paths cannot invalidate the owned content.
class ExecutablePackage {
public:
  ExecutablePackage(ExecutablePackage &&) = default;
  ExecutablePackage &operator=(ExecutablePackage &&) = default;
  ExecutablePackage(const ExecutablePackage &) = delete;
  ExecutablePackage &operator=(const ExecutablePackage &) = delete;

  llvm::StringRef getRootDirectory() const { return rootDirectory; }
  const PackageManifest &getManifest() const {
    return verifiedManifest.getManifest();
  }
  const VerifiedPackageManifest &getVerifiedManifest() const {
    return verifiedManifest;
  }
  const llvm::MemoryBuffer &getManifestBuffer() const {
    return *manifestBuffer;
  }
  const std::vector<std::unique_ptr<llvm::MemoryBuffer>> &
  getModuleBuffers() const {
    return moduleBuffers;
  }
  const llvm::MemoryBuffer &getProgramDataBuffer() const {
    return *programDataBuffer;
  }

private:
  friend struct detail::ExecutablePackageFactory;

  ExecutablePackage(
      std::string rootDirectory, VerifiedPackageManifest verifiedManifest,
      std::unique_ptr<llvm::MemoryBuffer> manifestBuffer,
      std::vector<std::unique_ptr<llvm::MemoryBuffer>> moduleBuffers,
      std::unique_ptr<llvm::MemoryBuffer> programDataBuffer)
      : rootDirectory(std::move(rootDirectory)),
        verifiedManifest(std::move(verifiedManifest)),
        manifestBuffer(std::move(manifestBuffer)),
        moduleBuffers(std::move(moduleBuffers)),
        programDataBuffer(std::move(programDataBuffer)) {}

  std::string rootDirectory;
  VerifiedPackageManifest verifiedManifest;
  std::unique_ptr<llvm::MemoryBuffer> manifestBuffer;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> moduleBuffers;
  std::unique_ptr<llvm::MemoryBuffer> programDataBuffer;
};

namespace detail {

struct ExecutablePackageFactory {
  static ExecutablePackage make(std::string committedRoot,
                                BoundExecutablePackage package) {
    return ExecutablePackage(
        std::move(committedRoot), std::move(package.manifest),
        std::move(package.manifestBuffer), std::move(package.moduleBuffers),
        std::move(package.programDataBuffer));
  }
};

} // namespace detail

llvm::StringRef stringifyProgramTensorRole(ProgramTensorRole role);
llvm::StringRef stringifyPackageAccessMode(PackageAccessMode access);
llvm::StringRef stringifyPackageMemLayout(PackageMemLayout layout);
PhysicalTensorLayout getPhysicalTensorLayout(PackageMemLayout layout);
llvm::StringRef stringifyPackageModuleExportRole(PackageModuleExportRole role);
llvm::StringRef
stringifyPackageEntryCompletionKind(PackageEntryCompletionKind kind);

llvm::Expected<VerifiedPackageManifest>
verifyPackageManifest(PackageManifest manifest, llvm::StringRef packageRoot,
                      const PackageParseLimits &limits = {});

std::string
serializeCanonicalPackageJson(const VerifiedPackageManifest &manifest);

llvm::Expected<VerifiedPackageManifest>
parseCanonicalPackageJson(llvm::StringRef json, llvm::StringRef packageRoot,
                          const PackageParseLimits &limits = {});

llvm::Expected<ExecutablePackage>
loadExecutablePackage(llvm::StringRef packageRoot,
                      const PackageParseLimits &limits = {});

/// One caller-side binding for an external input port. Output ports are
/// prepared and read back by the runtime and never caller-bound.
struct RuntimeInvocationBinding {
  PortId port;
  uint64_t bytes = 0;
  uint64_t alignment = 0;
};

struct RuntimeEnvironment {
  RuntimeEnvironment(
      TargetIdentityId targetIdentity, KernelRuntimeABIId runtimeABI,
      llvm::StringRef moduleFormat,
      uint64_t maxResourceBytes = std::numeric_limits<uint64_t>::max())
      : targetIdentity(targetIdentity), runtimeABI(runtimeABI),
        moduleFormat(moduleFormat.str()), maxResourceBytes(maxResourceBytes) {}

  TargetIdentityId targetIdentity;
  KernelRuntimeABIId runtimeABI;
  std::string moduleFormat;
  uint64_t maxResourceBytes = std::numeric_limits<uint64_t>::max();
  std::vector<KernelLaunchForm> supportedKernelLaunchForms;
  std::vector<KernelEntryABI> supportedKernelEntryABIs;
  bool supportsDirectDTE = false;
  std::string directDTEStatusABI;
  bool supportsHostWatchdog = false;
};

struct PlannedRuntimeLaunchPhase {
  RuntimeLaunchPhaseRole role = RuntimeLaunchPhaseRole::Main;
  std::string symbol;
};

/// One checked child range inside one device allocation. It is never freed
/// on its own; the owning allocation controls its lifetime.
struct RuntimePlannedRange {
  uint64_t offset = 0;
  uint64_t bytes = 0;
};

/// Where one Tile entry argument's device address points.
enum class RuntimeArgumentAddressBase { ProgramData, Invocation };

struct RuntimeArgumentAddress {
  RuntimeArgumentAddressBase base = RuntimeArgumentAddressBase::Invocation;
  uint64_t offset = 0;
};

struct RuntimeEntryLocalRanges {
  std::optional<RuntimePlannedRange> workspace;
  std::optional<RuntimePlannedRange> profileRecord;
  std::optional<RuntimePlannedRange> transportStatus;
};

struct RuntimeSessionPlan {
  EntryId entry;
  CardId cardId{0};
  TileId tileId{0};
  LaunchSlotId launchSlot;
  ModuleId module;
  std::string modulePath;
  std::vector<PlannedRuntimeLaunchPhase> phases;
  PackageEntryCompletionKind completion =
      PackageEntryCompletionKind::ReturnAfterLocalDrain;
  /// Parallel to the entry's ordered arguments: one resolved address per
  /// argument. TargetTensor addresses are program-data base + file offset;
  /// every other address is invocation base + planned offset.
  std::vector<RuntimeArgumentAddress> argumentAddresses;
  TransportRequirements transport;
};

struct RuntimeInvocationPlan {
  int64_t cardCount = 0;
  int64_t tileCount = 0;
  /// The non-empty program-data allocation requirement. When the package has
  /// no TargetTensor this stays zero and the runtime issues no provider call.
  bool programDataRequired = false;
  uint64_t programDataBytes = 0;
  uint64_t programDataAlignment = 1;
  /// TargetTensor child ranges inside the program-data allocation, indexed by
  /// TargetTensor id; each offset equals the manifest file offset.
  std::vector<RuntimePlannedRange> targetTensorRanges;
  /// The single invocation allocation requirement and its child ranges.
  uint64_t invocationBytes = 0;
  uint64_t invocationAlignment = 1;
  /// Input/output child ranges, indexed by PortId.
  std::vector<RuntimePlannedRange> inputRanges;
  std::vector<RuntimePlannedRange> outputRanges;
  /// Card-shared DDR child ranges indexed by CardWorkspace resource id.
  std::vector<RuntimePlannedRange> cardWorkspaceRanges;
  /// Per-Tile entry-local ranges, indexed by launch slot.
  std::vector<RuntimeEntryLocalRanges> tileRanges;
  /// Per-Tile device pointer rows for the TileRowPointerTable entry ABI;
  /// empty for the TileMajorPointerTable ABI whose row travels in the kernel
  /// command packet.
  std::vector<RuntimePlannedRange> pointerRows;
  /// One record for every package Tile, in canonical launch-slot order.
  std::vector<RuntimeSessionPlan> tiles;
};

/// Builds the complete side-effect-free invocation plan: optional program
/// data allocation, one invocation allocation with deterministic non-overlap
/// child ranges, and every Tile argument resolved to a checked address. The
/// plan never serializes back into the package and never calls a provider.
llvm::Expected<RuntimeInvocationPlan> planRuntimeInvocation(
    const VerifiedPackageManifest &package,
    llvm::ArrayRef<RuntimeInvocationBinding> invocationBindings,
    const RuntimeEnvironment &environment);

/// Typed lookup helpers over the canonical manifest tables. They are stable
/// package-model accessors shared by the writer, verifier, planner and
/// profile consumers; identity always comes from typed ids, never names.
const PackageModuleRecord *
findModule(llvm::ArrayRef<PackageModuleRecord> modules, ModuleId id);

const PackageModuleExportRecord *
findModuleExport(const PackageModuleRecord &module,
                 PackageModuleExportRole role);

const ProgramTensorRecord *
findProgramTensor(llvm::ArrayRef<ProgramTensorRecord> tensors,
                  ProgramTensorId id);

const TargetTensorRecord *
findTargetTensor(llvm::ArrayRef<TargetTensorRecord> tensors, TargetTensorId id);

const ExternalPortRecord *findPort(llvm::ArrayRef<ExternalPortRecord> ports,
                                   PortId id);
} // namespace wafer::runtime

#endif // WAFER_PACKAGE_PACKAGEMANIFEST_H
