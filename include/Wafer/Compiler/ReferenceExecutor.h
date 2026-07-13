//===- ReferenceExecutor.h - Accepted-rank semantic execution -*- C++ -*-===//

#ifndef WAFER_COMPILER_REFERENCEEXECUTOR_H
#define WAFER_COMPILER_REFERENCEEXECUTOR_H

#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

/// Owner-backed compact row-major tensor at a typed program boundary.
class ReferenceTensor {
public:
  static llvm::Expected<ReferenceTensor> create(llvm::StringRef dtype,
                                                llvm::ArrayRef<int64_t> shape,
                                                llvm::ArrayRef<uint8_t> bytes);

  llvm::StringRef getDType() const { return dtype; }
  llvm::ArrayRef<int64_t> getShape() const { return shape; }
  llvm::ArrayRef<uint8_t> getBytes() const { return bytes; }

private:
  ReferenceTensor(std::string dtype, std::vector<int64_t> shape,
                  std::vector<uint8_t> bytes)
      : dtype(std::move(dtype)), shape(std::move(shape)),
        bytes(std::move(bytes)) {}

  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

/// Exact non-output program resource supplied to one accepted rank.
struct ReferenceInputBinding {
  ProgramResourceRole role;
  int64_t index;
  ReferenceTensor tensor;
};

struct ReferenceOutputBinding {
  int64_t index;
  ReferenceTensor tensor;
};

/// Complete invocation inputs for one logical rank. Multi-rank execution
/// requires an all-and-only canonical logical-rank domain.
struct ReferenceRankInvocation {
  int64_t logicalRank;
  std::vector<ReferenceInputBinding> inputs;
};

struct ReferenceGlobalOutputBinding {
  int64_t index;
  std::string name;
  ReferenceTensor tensor;
};

/// Invocation policy that is not encoded by accepted instruction IR. An
/// explicit seed is required only when the projected program uses stochastic
/// rounding; deterministic rounding programs do not consume it.
struct ReferenceExecutionOptions {
  std::optional<uint64_t> stochasticSeed;
};

/// Complete logical outputs produced after the accepted rank returns.
class ReferenceExecutionResult {
public:
  int64_t getLogicalRank() const { return logicalRank; }
  const std::vector<ReferenceOutputBinding> &getOutputs() const {
    return outputs;
  }

private:
  friend struct ReferenceExecutionResultBuilder;

  ReferenceExecutionResult(int64_t logicalRank,
                           std::vector<ReferenceOutputBinding> outputs)
      : logicalRank(logicalRank), outputs(std::move(outputs)) {}

  int64_t logicalRank;
  std::vector<ReferenceOutputBinding> outputs;
};

/// Per-rank results plus typed global tensors reconstructed from the accepted
/// output slices. Transport scheduling remains an execution detail and is not
/// serialized into this result.
class ReferenceMultiRankExecutionResult {
public:
  const std::vector<ReferenceExecutionResult> &getRankResults() const {
    return rankResults;
  }
  const std::vector<ReferenceGlobalOutputBinding> &getGlobalOutputs() const {
    return globalOutputs;
  }

private:
  friend struct ReferenceMultiRankExecutionResultBuilder;

  ReferenceMultiRankExecutionResult(
      std::vector<ReferenceExecutionResult> rankResults,
      std::vector<ReferenceGlobalOutputBinding> globalOutputs)
      : rankResults(std::move(rankResults)),
        globalOutputs(std::move(globalOutputs)) {}

  std::vector<ReferenceExecutionResult> rankResults;
  std::vector<ReferenceGlobalOutputBinding> globalOutputs;
};

/// Invocation-local immutable projection of one accepted rank. The program
/// owns the MLIR context needed by copied immutable types, but does not retain
/// operations, values, planner state, or a serializable instruction stream.
class ReferenceProgram {
public:
  struct Impl;

  ReferenceProgram(ReferenceProgram &&) noexcept;
  ReferenceProgram &operator=(ReferenceProgram &&) noexcept;
  ReferenceProgram(const ReferenceProgram &) = delete;
  ReferenceProgram &operator=(const ReferenceProgram &) = delete;
  ~ReferenceProgram();

  int64_t getLogicalRank() const;
  size_t getProjectedOperationCount() const;

private:
  friend struct ReferenceProgramBuilder;
  friend llvm::Expected<ReferenceExecutionResult>
  executeReferenceProgram(const ReferenceProgram &program,
                          llvm::ArrayRef<ReferenceInputBinding> inputs,
                          ReferenceExecutionOptions options);
  friend llvm::Expected<ReferenceMultiRankExecutionResult>
  executeReferenceBundle(const ExecutableBundle &bundle,
                         llvm::ArrayRef<ReferenceRankInvocation> invocations,
                         ReferenceExecutionOptions options);

  explicit ReferenceProgram(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

/// Projects and validates every executable operation/control edge before any
/// invocation tensor is imported or execution storage is allocated.
llvm::Expected<ReferenceProgram>
prepareReferenceRank(const ExecutableBundle &bundle, int64_t logicalRank);

/// Executes only the immutable projection. Mutating or destroying the source
/// bundle after preparation cannot change this program's command semantics.
llvm::Expected<ReferenceExecutionResult>
executeReferenceProgram(const ReferenceProgram &program,
                        llvm::ArrayRef<ReferenceInputBinding> inputs,
                        ReferenceExecutionOptions options = {});

/// Executes the already-selected instruction and accepted memory facts of one
/// rank. This convenience entry first performs complete projection/capability
/// preflight, then executes that immutable program.
llvm::Expected<ReferenceExecutionResult>
executeReferenceRank(const ExecutableBundle &bundle, int64_t logicalRank,
                     llvm::ArrayRef<ReferenceInputBinding> inputs,
                     ReferenceExecutionOptions options = {});

/// Projects every accepted Direct DTE rank before importing any invocation
/// tensor, then executes them with a deterministic logical-rank/event order.
/// The scheduler never uses threads, wall-clock timeout, or operation
/// visitation order as a message identity.
llvm::Expected<ReferenceMultiRankExecutionResult>
executeReferenceBundle(const ExecutableBundle &bundle,
                       llvm::ArrayRef<ReferenceRankInvocation> invocations,
                       ReferenceExecutionOptions options = {});

} // namespace wafer::compiler

#endif // WAFER_COMPILER_REFERENCEEXECUTOR_H
