//===- BoardRuntime.h - Verified package board execution ------*- C++ -*-===//

#ifndef WAFER_RUNTIME_BOARDRUNTIME_H
#define WAFER_RUNTIME_BOARDRUNTIME_H

#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
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

  /// Establishes one provider-owned submission for the complete logical-rank
  /// domain. A failure after an unknown or non-empty accepted subset must
  /// poison the context. A usable failure guarantees that no submission state
  /// remains live.
  virtual llvm::Error submitAll(llvm::ArrayRef<BoardRankLaunch> launches) = 0;

  /// Waits for every submitted rank with a host deadline. Timeout or an
  /// untrustworthy terminal state must poison the context.
  virtual llvm::Error waitAll(uint64_t timeoutMilliseconds) = 0;

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
  BoardDeviceQualification qualification;
  std::vector<BoardRuntimeBinding> bindings;
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
  std::vector<BoardRuntimeOutput> outputs;
};

/// Executes the complete verified logical-rank domain as one owner-backed
/// provider session. The current TX provider accepts only transport:none;
/// Direct DTE remains a side-effect-free rejection until a proven placement
/// and per-rank argument ABI exists.
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
