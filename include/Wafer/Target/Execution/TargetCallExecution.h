//===- TargetCallExecution.h - Host target-call execution ------*- C++ -*-===//

#ifndef WAFER_TARGET_EXECUTION_TARGETCALLEXECUTION_H
#define WAFER_TARGET_EXECUTION_TARGETCALLEXECUTION_H

#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Target/Core/TargetCall.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace wafer::compiler {

struct TargetNCCIssueDomain {
  TargetCallTSMEngine engine;
  TargetNCCWorker worker;
  TargetNCCCompletionBehavior completionBehavior;
};

/// One dynamic call effect. Physical identity and launch slot are explicitly
/// bound by the JIT bridge. The ordinal is assigned monotonically inside that
/// Tile context; none of these fields is recovered from a symbol spelling or
/// OS thread.
struct TargetCommand {
  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  uint64_t issueOrdinal;
  target::TargetCommandPayload payload;
  std::optional<TargetNCCIssueDomain> nccIssueDomain = std::nullopt;
};

struct TargetCallTileDescriptor {
  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  std::vector<TileEntryArgument> tileEntryArguments;
  std::vector<uint64_t> slotValues;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId kernelRuntimeABI;
};

struct TargetCallInvocationDescriptor {
  TargetIdentityId targetIdentity;
  std::vector<TargetCallTileDescriptor> tiles;
};

struct TargetCallTileArguments {
  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  std::vector<uint64_t> slots;
};

/// Synchronous invocation-local consumer. A successful issue result is used
/// only by target calls whose exact ABI returns an opaque Direct-DTE event.
/// `completeInvocation` validates the full invocation and makes its result
/// available atomically. The sink must outlive a running executable.
class TargetCommandSink {
public:
  virtual ~TargetCommandSink() = default;
  virtual llvm::Error
  begin(const TargetCallInvocationDescriptor &invocation) = 0;
  virtual llvm::Expected<uint64_t> issue(const TargetCommand &command) = 0;
  virtual llvm::Error completeTile(CardId cardId,
                                   TileId tileId,
                                   LaunchSlotId launchSlotId) = 0;
  virtual llvm::Error completeInvocation() = 0;
  virtual void abort(llvm::StringRef diagnostic) = 0;
};

struct TargetCallExecutionResult {
  int64_t completedTileCount;
  uint64_t issuedCommandCount;
};

/// Owner of one card host materialization. All slots and JIT entries are
/// closed before construction succeeds. A downstream scheduler calls begin,
/// runs each Tile entry from its own process, and finishes only after every
/// Tile completes. executeTile may suspend inside a synchronous sink issue;
/// this is how a SystemC SC_THREAD preserves the JIT stack across wait().
/// Destroying or move-assigning a running executable aborts its sink, so the
/// sink must remain alive until finish or an explicit abort.
class TargetCallExecutable {
public:
  ~TargetCallExecutable();
  TargetCallExecutable(TargetCallExecutable &&);
  TargetCallExecutable &operator=(TargetCallExecutable &&);
  TargetCallExecutable(const TargetCallExecutable &) = delete;
  TargetCallExecutable &operator=(const TargetCallExecutable &) = delete;

  const TargetCallInvocationDescriptor &getInvocationDescriptor() const;
  llvm::Error begin(TargetCommandSink &sink);
  llvm::Error executeTile(LaunchSlotId launchSlotId);
  llvm::Expected<TargetCallExecutionResult> finish();
  void abort(llvm::StringRef diagnostic);

private:
  struct Impl;
  explicit TargetCallExecutable(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;

  friend llvm::Expected<TargetCallExecutable>
  createTargetCallExecutable(const TargetLLVMModules &,
                             llvm::ArrayRef<TargetCallTileArguments>);
};

/// Verifies and materializes the complete Tile domain and owns a copy
/// of every fixed ABI slot before returning. No sink effect occurs during
/// preparation.
/// This path does not compile the repository CRT or construct vendor packets.
llvm::Expected<TargetCallExecutable>
createTargetCallExecutable(const TargetLLVMModules &targetLLVMModules,
                           llvm::ArrayRef<TargetCallTileArguments> arguments);

/// Convenience orchestration for command sinks that never suspend on a
/// cross-Tile dependency. SystemC consumers use createTargetCallExecutable and
/// invoke executeTile from one SC_THREAD per Tile instead.
llvm::Expected<TargetCallExecutionResult>
executeTargetCalls(const TargetLLVMModules &targetLLVMModules,
                          llvm::ArrayRef<TargetCallTileArguments> arguments,
                          TargetCommandSink &sink);

} // namespace wafer::compiler

#endif // WAFER_TARGET_EXECUTION_TARGETCALLEXECUTION_H
