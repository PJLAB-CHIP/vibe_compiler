//===- TargetCall.h - Simulator target-call contract ----------*- C++ -*-===//

#ifndef WAFER_SIMULATOR_TARGETCALL_H
#define WAFER_SIMULATOR_TARGETCALL_H

#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Target/TargetCall.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
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
  virtual llvm::Error completeTile(CardId cardId, TileId tileId,
                                   LaunchSlotId launchSlotId) = 0;
  virtual llvm::Error completeInvocation() = 0;
  virtual void abort(llvm::StringRef diagnostic) = 0;
};

} // namespace wafer::compiler

#endif // WAFER_SIMULATOR_TARGETCALL_H
