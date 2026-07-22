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

/// Low-level board calls used by the owner-backed executor. Implementations
/// must not infer resource roles or ABI slots; those are supplied by the
/// verified package plan.
class BoardRuntimeDriver {
public:
  virtual ~BoardRuntimeDriver() = default;

  /// Returns sticky provider state. A poisoned context can only transition by
  /// an explicit recovery action outside executeBoardEntry().
  virtual BoardRuntimeContextState getContextState() const = 0;

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
  virtual llvm::Error launch(BoardFunctionHandle function,
                             llvm::ArrayRef<uint64_t> arguments) = 0;
  virtual llvm::Error synchronize() = 0;
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

/// Executes exactly one verified rank entry. Direct-DTE entries are rejected
/// until the all-rank provider session can submit and progress the complete
/// rank domain as one execution unit.
llvm::Expected<BoardRuntimeResult>
executeBoardEntry(const VerifiedPackageManifest &package,
                  llvm::StringRef packageRoot, BoardRuntimeRequest request,
                  BoardRuntimeDriver &driver);

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_BOARDRUNTIME_H
