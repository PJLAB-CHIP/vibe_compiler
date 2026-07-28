//===- BoardRuntime.h - Verified package board execution ------*- C++ -*-===//

#ifndef WAFER_RUNTIME_BOARDRUNTIME_H
#define WAFER_RUNTIME_BOARDRUNTIME_H

#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wafer::runtime {

inline constexpr uint64_t kDefaultBoardCompletionTimeoutMilliseconds = 60000;
inline constexpr uint64_t kMaximumBoardCompletionTimeoutMilliseconds =
    60ULL * 60 * 1000;

enum class BoardRuntimeStage {
  Preflight,
  DeviceSelection,
  ResourceAllocation,
  HostToDevice,
  ModuleLoad,
  EntryResolve,
  Launch,
  Completion,
  DeviceToHost,
  Cleanup,
};

llvm::StringRef stringifyBoardRuntimeStage(BoardRuntimeStage stage);

/// Provider-owned state of the selected device context after a runtime call.
/// Once poisoned, the executor must not issue another provider call. Recovery
/// is an explicit operation outside this invocation; it is never implicit
/// cleanup, reset, or retry.
enum class BoardRuntimeContextState {
  Usable,
  Poisoned,
};

llvm::StringRef
stringifyBoardRuntimeContextState(BoardRuntimeContextState state);

class BoardRuntimeError final : public llvm::ErrorInfo<BoardRuntimeError> {
public:
  static char ID;

  BoardRuntimeError(
      BoardRuntimeStage stage, int64_t logicalRank, EntryId entry,
      std::string detail,
      BoardRuntimeContextState contextState = BoardRuntimeContextState::Usable)
      : stage(stage), logicalRank(logicalRank), entry(entry),
        detail(std::move(detail)), contextState(contextState) {}

  BoardRuntimeStage getStage() const { return stage; }
  int64_t getLogicalRank() const { return logicalRank; }
  EntryId getEntry() const { return entry; }
  llvm::StringRef getDetail() const { return detail; }
  BoardRuntimeContextState getContextState() const { return contextState; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  BoardRuntimeStage stage;
  int64_t logicalRank;
  EntryId entry;
  std::string detail;
  BoardRuntimeContextState contextState;
};

struct BoardDeviceInfo {
  uint32_t deviceId = 0;
  uint32_t runtimeVersion = 0;
  uint64_t freeMemoryBytes = 0;
  uint64_t totalMemoryBytes = 0;
  uint32_t tileCount = 0;
  std::string name;
  std::string pciBusId;
  std::string runtimeLibraryDigest;
  struct Tile {
    uint16_t logicalIndex = 0;
    bool available = false;
    uint32_t physicalX = 0;
    uint32_t physicalY = 0;
  };
  std::vector<Tile> tiles;
};

/// Invocation-external qualification facts that must match the live public
/// runtime inventory before the executor allocates device memory or loads a
/// module. Driver and firmware qualification remains an explicit host-level
/// gate because the public TX runtime does not expose those identities.
struct BoardDeviceQualification {
  uint32_t runtimeVersion = 0;
  uint32_t tileCount = 0;
  std::string name;
  std::string pciBusId;
  std::string runtimeLibraryDigest;
};

struct BoardDeviceMemory {
  uintptr_t value = 0;
};

struct BoardModuleHandle {
  uintptr_t value = 0;
};

struct BoardFunctionHandle {
  uintptr_t value = 0;
};

struct BoardGraphHandle {
  uintptr_t value = 0;
};

/// Controls how a provider observes terminal submission state. Ordinary
/// invocations retain the provider's low-overhead polling cadence. Profiler
/// campaigns request a higher-resolution cadence so host launch-to-completion
/// samples can be qualified by the resolution that was actually observed.
enum class BoardCompletionObservationPolicy {
  Normal,
  ProfileHighResolution,
};

struct BoardCompletionObservation {
  /// Maximum steady-clock gap between consecutive completion observations for
  /// any still-live stream. This is measured by the provider; it is never a
  /// nominal polling constant.
  uint64_t maximumPollGapNanoseconds = 0;
};

using BoardCompletionDeadline = std::chrono::steady_clock::time_point;

/// One immutable, already-digest-verified tile module snapshot. Graph loading
/// must synchronously consume the bytes; it may not retain the ArrayRef.
struct BoardGraphModuleSnapshot {
  uint16_t logicalTile = 0;
  ModuleId module;
  llvm::StringRef digest;
  llvm::ArrayRef<uint8_t> bytes;
};

/// One model launch tensor projected from a verified entry slot. The provider
/// receives typed semantics and a device allocation, never a caller-built raw
/// BootParam buffer.
struct BoardModelTensorLaunch {
  int64_t logicalRank = -1;
  uint64_t slotOrdinal = 0;
  PackageResourceRole role = PackageResourceRole::UserInput;
  BoardDeviceMemory memory;
  uint64_t bytes = 0;
  std::string dtype;
  std::vector<int64_t> shape;
};

/// One canonical logical-rank launch owned by an all-rank provider
/// submission. The provider may implement the common submission using
/// multiple command queues, but callers cannot observe or assemble those
/// queues themselves.
struct BoardRankLaunch {
  int64_t logicalRank = -1;
  EntryId entry;
  BoardFunctionHandle function;
  std::vector<uint64_t> arguments;
};

/// Low-level board calls used by the owner-backed executor. Implementations
/// must not infer resource roles or ABI slots; those are supplied by the
/// verified package plan.
class BoardRuntimeDriver {
public:
  virtual ~BoardRuntimeDriver() = default;

  /// Returns sticky provider state. A poisoned context can only transition by
  /// an explicit recovery action outside board invocation execution.
  virtual BoardRuntimeContextState getContextState() const = 0;

  /// Marks this invocation-local provider context sticky poisoned without
  /// issuing a runtime or device call. The executor uses this after a
  /// terminal command produced an untrustworthy transport outcome. No later
  /// provider call is permitted.
  virtual void quarantine() = 0;

  /// Returns cached provider-owned semantic capability. This accessor must not
  /// issue a runtime or device call.
  virtual const RuntimeEnvironment &getProviderEnvironment() const = 0;

  virtual llvm::Expected<uint32_t> getDeviceCount() = 0;
  virtual llvm::Error selectDevice(uint32_t deviceId) = 0;
  virtual llvm::Expected<BoardDeviceInfo> getDeviceInfo(uint32_t deviceId) = 0;
  virtual llvm::Expected<BoardDeviceMemory> allocate(uint64_t bytes,
                                                     uint64_t alignment) = 0;
  virtual llvm::Error free(BoardDeviceMemory memory) = 0;
  virtual llvm::Error copyHostToDevice(BoardDeviceMemory destination,
                                       llvm::ArrayRef<uint8_t> source) = 0;
  virtual llvm::Error
  copyDeviceToHost(llvm::MutableArrayRef<uint8_t> destination,
                   BoardDeviceMemory source) = 0;
  virtual llvm::Expected<BoardModuleHandle>
  loadModule(llvm::ArrayRef<uint8_t> moduleBytes) = 0;
  virtual llvm::Error unloadModule(BoardModuleHandle module) = 0;
  virtual llvm::Expected<BoardFunctionHandle>
  resolveEntry(BoardModuleHandle module, llvm::StringRef symbol) = 0;

  /// Stages and synchronously loads the complete tile0..tile15 graph module
  /// set. Success owns one graph until unloadGraph; failure must state its
  /// sticky context state through getContextState().
  virtual llvm::Expected<BoardGraphHandle>
  loadGraph(llvm::ArrayRef<BoardGraphModuleSnapshot> modules,
            llvm::StringRef symbol) = 0;
  virtual llvm::Error unloadGraph(BoardGraphHandle graph) = 0;

  /// Submits exactly one typed kernel phase for the complete logical-rank
  /// domain. The first phase establishes provider-owned stream and argument
  /// storage; a later phase may only reuse that state after the previous phase
  /// reached terminal. A failure after an unknown or non-empty accepted subset
  /// must poison the context.
  virtual llvm::Error
  submitKernelPhase(KernelLaunchForm form, RuntimeLaunchPhaseRole phaseRole,
                    llvm::ArrayRef<BoardRankLaunch> launches) = 0;

  /// One txLaunchModel submission owned by a previously loaded graph. The TX
  /// provider alone materializes the qualified BootParam/type-7 wire bytes.
  virtual llvm::Error
  submitModel(BoardGraphHandle graph,
              llvm::ArrayRef<BoardModelTensorLaunch> tensors) = 0;

  /// Waits for the current submitted phase to become terminal. Every phase of
  /// one invocation receives the same absolute host deadline. Timeout or an
  /// untrustworthy terminal state must poison the context.
  virtual llvm::Expected<BoardCompletionObservation>
  waitCurrentSubmission(BoardCompletionDeadline deadline,
                        BoardCompletionObservationPolicy observationPolicy) = 0;

  /// Releases provider-owned submission state after every rank is known
  /// terminal. It must never be called after poison.
  virtual llvm::Error releaseSubmission() = 0;
};

/// One move-owned host buffer bound by typed ResourceId. Read-only and
/// read-write resources carry their initial bytes; write-only resources carry
/// an exact-size destination buffer initialized by the caller.
struct BoardRuntimeBinding {
  ResourceId resource;
  std::vector<uint8_t> bytes;
};

struct BoardRuntimeRequest {
  uint32_t deviceId = 0;
  EntryId entry;
  uint64_t completionTimeoutMilliseconds =
      kDefaultBoardCompletionTimeoutMilliseconds;
  BoardDeviceQualification qualification;
  std::vector<BoardRuntimeBinding> bindings;
};

struct BoardRuntimeInvocationRequest {
  uint32_t deviceId = 0;
  uint64_t completionTimeoutMilliseconds =
      kDefaultBoardCompletionTimeoutMilliseconds;
  BoardCompletionObservationPolicy completionObservationPolicy =
      BoardCompletionObservationPolicy::Normal;
  BoardDeviceQualification qualification;
  std::vector<BoardRuntimeBinding> bindings;
  /// Compiler-owned profiler records are the only internal workspace that a
  /// board invocation may initialize and read back. They are populated by
  /// wafer-run after exact companion verification; they are never exposed as
  /// user ResourceId bindings.
  std::vector<BoardRuntimeBinding> profilerBindings;
};

struct BoardRuntimeOutput {
  ResourceId resource;
  std::vector<uint8_t> bytes;
};

struct BoardRuntimeResult {
  BoardDeviceInfo device;
  EntryId entry;
  int64_t logicalRank = -1;
  ModuleId module;
  CompletionId terminalCompletion;
  std::vector<BoardRuntimeStage> completedStages;
  std::vector<BoardRuntimeOutput> outputs;
};

struct BoardRuntimeRankResult {
  EntryId entry;
  int64_t logicalRank = -1;
  ModuleId module;
  CompletionId terminalCompletion;
};

struct BoardRuntimeInvocationResult {
  BoardDeviceInfo device;
  std::vector<BoardRuntimeRankResult> ranks;
  std::vector<BoardRuntimeStage> completedStages;
  /// Host steady-clock interval from immediately before provider submission
  /// through successful all-rank completion. This is a campaign-level latency
  /// observation, not a tile clock and not per-instruction hardware time.
  uint64_t launchToCompletionNanoseconds = 0;
  /// Provider-measured maximum gap between completion observations. Profiler
  /// analysis uses this to reject latency samples whose terminal observation
  /// cadence is too coarse for the claimed comparison.
  uint64_t completionObservationResolutionNanoseconds = 0;
  std::vector<BoardRuntimeOutput> outputs;
  std::vector<BoardRuntimeOutput> profilerOutputs;
};

/// A capability proving that one concrete driver instance selected and
/// qualified one device inventory. The constructor is private so callers
/// cannot manufacture a session by copying qualification text. Moving the
/// capability invalidates the source; destruction never performs recovery,
/// reset, power, or another provider call. The concrete driver must outlive
/// the capability, and one owner must serialize invocations through it.
class QualifiedBoardRuntimeSession final {
public:
  QualifiedBoardRuntimeSession(const QualifiedBoardRuntimeSession &) = delete;
  QualifiedBoardRuntimeSession &
  operator=(const QualifiedBoardRuntimeSession &) = delete;
  QualifiedBoardRuntimeSession(QualifiedBoardRuntimeSession &&other) noexcept;
  QualifiedBoardRuntimeSession &
  operator=(QualifiedBoardRuntimeSession &&other) noexcept;
  ~QualifiedBoardRuntimeSession() = default;

  bool isUsable() const { return usable; }

private:
  QualifiedBoardRuntimeSession(BoardRuntimeDriver &driver, uint32_t deviceId,
                               uint32_t qualifiedLogicalRankCount,
                               BoardDeviceQualification qualification,
                               BoardDeviceInfo device);

  BoardRuntimeDriver *driver = nullptr;
  uint32_t deviceId = 0;
  uint32_t qualifiedLogicalRankCount = 0;
  BoardDeviceQualification qualification;
  BoardDeviceInfo device;
  bool usable = false;

  friend llvm::Expected<QualifiedBoardRuntimeSession>
  qualifyBoardRuntimeSession(uint32_t, uint32_t,
                             const BoardDeviceQualification &,
                             BoardRuntimeDriver &);
  friend llvm::Expected<BoardRuntimeInvocationResult>
  executeBoardInvocationInSession(const VerifiedPackageManifest &,
                                  llvm::StringRef,
                                  BoardRuntimeInvocationRequest,
                                  QualifiedBoardRuntimeSession &);
  friend llvm::Expected<
      std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
  executeBoardInvocationAndStartSession(const VerifiedPackageManifest &,
                                        llvm::StringRef,
                                        BoardRuntimeInvocationRequest,
                                        BoardRuntimeDriver &);
};

/// Performs device count/selection/inventory queries exactly once and returns
/// a driver-bound capability. `requiredLogicalRankCount` qualifies the dense
/// logical domain 0..N-1; profiler campaigns request all 16 tiles.
llvm::Expected<QualifiedBoardRuntimeSession>
qualifyBoardRuntimeSession(uint32_t deviceId, uint32_t requiredLogicalRankCount,
                           const BoardDeviceQualification &qualification,
                           BoardRuntimeDriver &driver);

/// Executes one complete invocation using a previously qualified capability.
/// The request must name the same device and qualification, and the package
/// rank domain must exactly match the qualified domain. Device
/// count/selection/info are not repeated. A poisoned capability can never be
/// used again.
llvm::Expected<BoardRuntimeInvocationResult>
executeBoardInvocationInSession(const VerifiedPackageManifest &package,
                                llvm::StringRef packageRoot,
                                BoardRuntimeInvocationRequest request,
                                QualifiedBoardRuntimeSession &session);

/// Executes the first invocation through the ordinary one-shot path with the
/// normal completion observer, then returns a capability for later
/// invocations on that already-qualified device. Qualification is performed
/// exactly once. Failure returns no session capability.
llvm::Expected<
    std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
executeBoardInvocationAndStartSession(const VerifiedPackageManifest &package,
                                      llvm::StringRef packageRoot,
                                      BoardRuntimeInvocationRequest request,
                                      BoardRuntimeDriver &driver);

/// Executes the complete verified logical-rank domain as one owner-backed
/// provider session. The manifest selects exactly one kernel or model
/// submission path. Entry transport requirements, including Direct DTE, are
/// verified independently and never select another runtime entry point.
llvm::Expected<BoardRuntimeInvocationResult> executeBoardInvocation(
    const VerifiedPackageManifest &package, llvm::StringRef packageRoot,
    BoardRuntimeInvocationRequest request, BoardRuntimeDriver &driver);

/// Compatibility entry point for a rank-count=1 package. It delegates to the
/// same owner-backed invocation implementation; it cannot select one rank out
/// of a multi-rank package.
llvm::Expected<BoardRuntimeResult>
executeBoardEntry(const VerifiedPackageManifest &package,
                  llvm::StringRef packageRoot, BoardRuntimeRequest request,
                  BoardRuntimeDriver &driver);

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_BOARDRUNTIME_H
