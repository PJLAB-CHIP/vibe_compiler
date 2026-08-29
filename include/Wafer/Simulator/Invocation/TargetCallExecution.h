//===- TargetCallExecution.h - Host target-call execution ------*- C++ -*-===//

#ifndef WAFER_SIMULATOR_INVOCATION_TARGETCALLEXECUTION_H
#define WAFER_SIMULATOR_INVOCATION_TARGETCALLEXECUTION_H

#include "Wafer/Simulator/TargetCall.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace wafer::compiler {

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

#endif // WAFER_SIMULATOR_INVOCATION_TARGETCALLEXECUTION_H
