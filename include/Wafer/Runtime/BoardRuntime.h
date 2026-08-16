//===- BoardRuntime.h - Verified package board execution ------*- C++ -*-===//

#ifndef WAFER_RUNTIME_BOARDRUNTIME_H
#define WAFER_RUNTIME_BOARDRUNTIME_H

#include "Wafer/Package/PackageManifest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::runtime {

inline constexpr uint64_t kDefaultBoardCompletionTimeoutMilliseconds = 60000;
inline constexpr uint64_t kMaximumBoardCompletionTimeoutMilliseconds =
    60ULL * 60 * 1000;

enum class BoardRuntimeStage {
  Validation,
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
      BoardRuntimeStage stage, CardId cardId, TileId tileId,
      LaunchSlotId launchSlot, EntryId entry, std::string detail,
      BoardRuntimeContextState contextState = BoardRuntimeContextState::Usable)
      : stage(stage), cardId(cardId), tileId(tileId), launchSlot(launchSlot),
        entry(entry), detail(std::move(detail)), contextState(contextState) {}

  BoardRuntimeStage getStage() const { return stage; }
  CardId getCardId() const { return cardId; }
  TileId getTileId() const { return tileId; }
  LaunchSlotId getLaunchSlot() const { return launchSlot; }
  EntryId getEntry() const { return entry; }
  llvm::StringRef getDetail() const { return detail; }
  BoardRuntimeContextState getContextState() const { return contextState; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  BoardRuntimeStage stage;
  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlot;
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
    TileId tileId{0};
    LaunchSlotId launchSlot;
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

/// Controls how a provider observes terminal submission state. Ordinary
/// invocations retain the provider's low-overhead polling cadence. Profiler
/// profile collection requests a higher-resolution cadence so host
/// launch-to-completion samples can be qualified by the resolution that was
/// actually observed.
enum class BoardCompletionObservationPolicy {
  Normal,
  ProfileHighResolution,
};

/// Controls whether the provider brackets one submitted phase with
/// same-stream device events. This is intentionally separate from host
/// completion observation: stream events measure device execution while the
/// host clock measures submission and terminal-observation latency.
enum class BoardDeviceTimingPolicy {
  Disabled,
  StreamEvents,
};

struct BoardCompletionObservation {
  /// Maximum steady-clock gap between consecutive completion observations for
  /// any still-live stream. This is measured by the provider; it is never a
  /// nominal polling constant.
  uint64_t maximumPollGapNanoseconds = 0;
  /// Same-stream device-event duration for exactly the submitted phase.
  /// Providers must leave this empty when device timing was disabled.
  std::optional<uint64_t> deviceExecutionNanoseconds;
};

using BoardCompletionDeadline = std::chrono::steady_clock::time_point;

/// One canonical Tile launch owned by a card provider
/// submission. The provider may implement the common submission using
/// multiple command queues, but callers cannot observe or assemble those
/// queues themselves.
struct BoardTileLaunch {
  CardId cardId{0};
  TileId tileId{0};
  LaunchSlotId launchSlot;
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

  /// Submits exactly one typed kernel phase for the complete Tile
  /// domain. The first phase establishes provider-owned stream and argument
  /// storage; a later phase may only reuse that state after the previous phase
  /// reached terminal. A failure after an unknown or non-empty accepted subset
  /// must poison the context.
  virtual llvm::Error
  submitKernelPhase(KernelLaunchForm form, RuntimeLaunchPhaseRole phaseRole,
                    llvm::ArrayRef<BoardTileLaunch> launches,
                    BoardDeviceTimingPolicy timingPolicy) = 0;

  /// Waits for the current submitted phase to become terminal. Every phase of
  /// one invocation receives the same absolute host deadline. Timeout or an
  /// untrustworthy terminal state must poison the context.
  virtual llvm::Expected<BoardCompletionObservation>
  waitCurrentSubmission(BoardCompletionDeadline deadline,
                        BoardCompletionObservationPolicy observationPolicy) = 0;

  /// Releases provider-owned submission state after every Tile is known
  /// terminal. It must never be called after poison.
  virtual llvm::Error releaseSubmission() = 0;
};

/// One move-owned host buffer bound to an external input port. The byte count
/// must equal the port's selected target descriptor exactly. Output ports are
/// prepared and read back by the runtime and never caller-bound.
struct BoardRuntimeBinding {
  PortId port;
  std::vector<uint8_t> bytes;
};

struct BoardRuntimeInvocationRequest {
  uint32_t deviceId = 0;
  uint64_t completionTimeoutMilliseconds =
      kDefaultBoardCompletionTimeoutMilliseconds;
  BoardCompletionObservationPolicy completionObservationPolicy =
      BoardCompletionObservationPolicy::Normal;
  BoardDeviceTimingPolicy deviceTimingPolicy =
      BoardDeviceTimingPolicy::Disabled;
  BoardDeviceQualification qualification;
  std::vector<BoardRuntimeBinding> bindings;
  /// Compiler-owned profiler records are the only internal workspace that a
  /// board invocation may initialize and read back. They are populated by
  /// wafer-run after exact instrumentation verification; one exact record
  /// image per launch slot, and every Tile entry must carry a profile-record
  /// argument whose bytes match its image exactly. They are never exposed as
  /// user input bindings.
  std::optional<std::vector<std::vector<uint8_t>>> profilerRecordBytes;
};

struct BoardRuntimeOutput {
  PortId port;
  std::vector<uint8_t> bytes;
};

/// One per-Tile profiler record readback. Records are entry-local compiler
/// workspace, never external ports.
struct BoardRuntimeProfilerOutput {
  LaunchSlotId launchSlot;
  std::vector<uint8_t> bytes;
};

struct BoardRuntimeTileResult {
  EntryId entry;
  CardId cardId{0};
  TileId tileId{0};
  LaunchSlotId launchSlot;
  ModuleId module;
  PackageEntryCompletionKind completion =
      PackageEntryCompletionKind::ReturnAfterLocalDrain;
};

struct BoardRuntimeInvocationResult {
  BoardDeviceInfo device;
  std::vector<BoardRuntimeTileResult> tiles;
  std::vector<BoardRuntimeStage> completedStages;
  /// Host steady-clock interval from immediately before provider submission
  /// through successful card completion. This is a collection-level latency
  /// observation, not a tile clock and not per-instruction hardware time.
  uint64_t launchToCompletionNanoseconds = 0;
  /// Host steady-clock time spent strictly inside provider submission calls,
  /// summed across every launch phase.
  uint64_t hostSubmitNanoseconds = 0;
  /// Sum of provider-observed same-stream device-event durations across every
  /// launch phase. Present exactly when StreamEvents timing was requested.
  std::optional<uint64_t> deviceExecutionNanoseconds;
  /// Provider-measured maximum gap between completion observations. Profiler
  /// analysis uses this to reject latency samples whose terminal observation
  /// cadence is too coarse for the claimed comparison.
  uint64_t completionObservationResolutionNanoseconds = 0;
  std::vector<BoardRuntimeOutput> outputs;
  std::vector<BoardRuntimeProfilerOutput> profilerOutputs;
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
                               uint32_t qualifiedTileCount,
                               BoardDeviceQualification qualification,
                               BoardDeviceInfo device);

  BoardRuntimeDriver *driver = nullptr;
  uint32_t deviceId = 0;
  uint32_t qualifiedTileCount = 0;
  BoardDeviceQualification qualification;
  BoardDeviceInfo device;
  bool usable = false;

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

/// Executes one complete invocation using a previously qualified capability.
/// The request must name the same device and qualification, and the package
/// Tile domain must exactly match the qualified domain. Device
/// count/selection/info are not repeated. A poisoned capability can never be
/// used again.
llvm::Expected<BoardRuntimeInvocationResult>
executeBoardInvocationInSession(const VerifiedPackageManifest &package,
                                llvm::StringRef packageRoot,
                                BoardRuntimeInvocationRequest request,
                                QualifiedBoardRuntimeSession &session);

/// Executes the first complete card invocation through the ordinary
/// one-shot path using the request's completion-observation policy, then
/// returns a capability for later complete invocations on that already-
/// qualified device. There is no public empty-session or arbitrary Tile-count
/// qualification path. Failure returns no session capability.
llvm::Expected<
    std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
executeBoardInvocationAndStartSession(const VerifiedPackageManifest &package,
                                      llvm::StringRef packageRoot,
                                      BoardRuntimeInvocationRequest request,
                                      BoardRuntimeDriver &driver);

/// Executes the complete verified Tile domain as one owner-backed
/// provider session using the kernel submission path. Entry transport
/// requirements, including Direct DTE, are verified independently and never
/// select another runtime entry point.
llvm::Expected<BoardRuntimeInvocationResult> executeBoardInvocation(
    const VerifiedPackageManifest &package, llvm::StringRef packageRoot,
    BoardRuntimeInvocationRequest request, BoardRuntimeDriver &driver);

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_BOARDRUNTIME_H
