//===- TargetCallFrontend.h - Host target-call execution -------*- C++ -*-===//

#ifndef WAFER_COMPILER_TARGETCALLFRONTEND_H
#define WAFER_COMPILER_TARGETCALLFRONTEND_H

#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Target/TargetCall.h"

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
  NCCWorker worker;
  LocalInstructionCompletion completionBehavior;
};

/// One dynamic call effect. Rank is explicitly bound by the JIT bridge. The
/// ordinal is assigned monotonically inside that rank context; neither field
/// is used to recover call semantics from a symbol spelling or OS thread.
struct TargetTransaction {
  int64_t logicalRank;
  uint64_t issueOrdinal;
  TargetTransactionPayload payload;
  std::optional<TargetNCCIssueDomain> nccIssueDomain = std::nullopt;
};

struct TargetCallRankDescriptor {
  int64_t logicalRank;
  std::vector<KernelABISlot> kernelABISlots;
  std::vector<uint64_t> slotValues;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId kernelRuntimeABI;
};

struct TargetCallInvocationDescriptor {
  TargetProfileId targetProfile;
  std::vector<TargetCallRankDescriptor> ranks;
};

struct TargetCallRankArguments {
  int64_t logicalRank;
  std::vector<uint64_t> slots;
};

/// Synchronous invocation-local consumer. A successful issue result is used
/// only by target calls whose exact ABI returns an opaque Direct-DTE event.
/// All effects remain private until prepareCommit succeeds; commit is the
/// infallible publication point. The sink must outlive a running executable.
class TargetTransactionSink {
public:
  virtual ~TargetTransactionSink() = default;
  virtual llvm::Error
  begin(const TargetCallInvocationDescriptor &invocation) = 0;
  virtual llvm::Expected<uint64_t>
  issue(const TargetTransaction &transaction) = 0;
  virtual llvm::Error terminal(int64_t logicalRank) = 0;
  virtual llvm::Error prepareCommit() = 0;
  virtual void commit() = 0;
  virtual void abort(llvm::StringRef diagnostic) = 0;
};

struct TargetCallExecutionResult {
  int64_t completedRankCount;
  uint64_t issuedTransactionCount;
};

/// Owner of one all-rank host materialization. All slots and JIT entries are
/// closed before construction succeeds. A downstream scheduler calls begin,
/// runs each rank entry from its own process, and commits only after every
/// rank is terminal. executeRank may suspend inside a synchronous sink issue;
/// this is how a SystemC SC_THREAD preserves the JIT stack across wait().
/// Destroying or move-assigning a running executable aborts its sink, so the
/// sink must remain alive until commit or an explicit abort.
class TargetCallExecutable {
public:
  ~TargetCallExecutable();
  TargetCallExecutable(TargetCallExecutable &&);
  TargetCallExecutable &operator=(TargetCallExecutable &&);
  TargetCallExecutable(const TargetCallExecutable &) = delete;
  TargetCallExecutable &operator=(const TargetCallExecutable &) = delete;

  const TargetCallInvocationDescriptor &getInvocationDescriptor() const;
  llvm::Error begin(TargetTransactionSink &sink);
  llvm::Error executeRank(int64_t logicalRank);
  llvm::Expected<TargetCallExecutionResult> commit();
  void abort(llvm::StringRef diagnostic);

private:
  struct Impl;
  explicit TargetCallExecutable(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;

  friend llvm::Expected<TargetCallExecutable>
  prepareTargetCallFrontend(const TargetLLVMModuleBundle &,
                            llvm::ArrayRef<TargetCallRankArguments>);
};

/// Atomically preflights/materializes all ranks and owns a copy of every
/// fixed ABI slot before returning. No sink effect occurs during preparation.
/// This path does not compile the repository CRT or construct vendor packets.
llvm::Expected<TargetCallExecutable>
prepareTargetCallFrontend(const TargetLLVMModuleBundle &bundle,
                          llvm::ArrayRef<TargetCallRankArguments> arguments);

/// Convenience orchestration for transaction sinks that never suspend on a
/// cross-rank dependency. SystemC consumers use prepareTargetCallFrontend and
/// invoke executeRank from one SC_THREAD per rank instead.
llvm::Expected<TargetCallExecutionResult>
executeTargetCallFrontend(const TargetLLVMModuleBundle &bundle,
                          llvm::ArrayRef<TargetCallRankArguments> arguments,
                          TargetTransactionSink &sink);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_TARGETCALLFRONTEND_H
