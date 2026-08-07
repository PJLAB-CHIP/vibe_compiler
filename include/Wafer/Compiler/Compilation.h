//===- Compilation.h - Typed Wafer compiler request ------------*- C++ -*-===//

#ifndef WAFER_COMPILER_COMPILATION_H
#define WAFER_COMPILER_COMPILATION_H

#include "Wafer/Frontend/Program.h"
#include "Wafer/Support/OptimizationConfig.h"
#include "Wafer/Target/RuntimeLaunchContract.h"
#include "Wafer/Target/TargetIdentity.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

class TargetToolchain;

/// Invocation-local diagnostic policy. Detailed timing never changes source
/// semantics, candidate admission, selection, or published artifacts.
enum class CompilationTimingMode { Disabled, Detailed };

/// Validated execution facts for the current single-card compiler boundary.
/// There is deliberately no default configuration: callers must choose the
/// one-rank or complete 16-rank domain explicitly.
class ExecutionConfig {
public:
  static llvm::Expected<ExecutionConfig>
  createForSingleCard(int64_t executionRankCount,
                      RuntimeLaunchKind runtimeLaunchKind);

  int64_t getRankCount() const { return executionRankCount; }
  TargetIdentityId getTargetIdentityId() const {
    return TargetIdentityId::waferTx81SingleCard();
  }
  RuntimeLaunchKind getRuntimeLaunchKind() const { return runtimeLaunchKind; }

  friend bool operator==(const ExecutionConfig &lhs,
                         const ExecutionConfig &rhs) {
    return lhs.executionRankCount == rhs.executionRankCount &&
           lhs.runtimeLaunchKind == rhs.runtimeLaunchKind;
  }
  friend bool operator!=(const ExecutionConfig &lhs,
                         const ExecutionConfig &rhs) {
    return !(lhs == rhs);
  }

private:
  ExecutionConfig(int64_t executionRankCount,
                  RuntimeLaunchKind runtimeLaunchKind)
      : executionRankCount(executionRankCount),
        runtimeLaunchKind(runtimeLaunchKind) {}

  int64_t executionRankCount;
  RuntimeLaunchKind runtimeLaunchKind;
};

/// Move-only semantic input to the compiler driver. Output locations,
/// toolchain helper paths, pass names and per-rank loop indices are
/// orchestration details and intentionally do not belong to this value.
class CompilationRequest {
public:
  static llvm::Expected<CompilationRequest>
  create(llvm::StringRef sourceProgramDirectory,
         ExecutionConfig executionConfig);

  CompilationRequest(CompilationRequest &&) = default;
  CompilationRequest &operator=(CompilationRequest &&) = default;
  CompilationRequest(const CompilationRequest &) = delete;
  CompilationRequest &operator=(const CompilationRequest &) = delete;

  llvm::StringRef getSourceProgramDirectory() const {
    return sourceProgramDirectory;
  }
  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }

private:
  CompilationRequest(llvm::StringRef sourceProgramDirectory,
                     ExecutionConfig executionConfig)
      : sourceProgramDirectory(sourceProgramDirectory.str()),
        executionConfig(executionConfig) {}

  std::string sourceProgramDirectory;
  ExecutionConfig executionConfig;
};

/// Typed orchestration intent for one production compilation transaction.
/// Profiling is a product request, not source-program semantics, and therefore
/// remains outside CompilationRequest and the compiler IR.
class CompilationOptions {
public:
  static CompilationOptions
  standard(OptimizationConfig optimizations = OptimizationConfig::production(),
           CompilationTimingMode timing = CompilationTimingMode::Disabled) {
    return CompilationOptions(/*profileCompanion=*/false, optimizations,
                              timing);
  }

  /// Requests a final-artifact profile companion. The ordinary package is
  /// compiled exactly once; the companion contains profile-only captures for
  /// that same accepted 16-rank artifact.
  static llvm::Expected<CompilationOptions>
  profile(const ExecutionConfig &executionConfig,
          OptimizationConfig optimizations = OptimizationConfig::production(),
          CompilationTimingMode timing = CompilationTimingMode::Disabled);

  bool shouldProduceProfileCompanion() const { return profileCompanion; }
  OptimizationConfig getOptimizationConfig() const { return optimizations; }
  bool shouldReportDetailedTiming() const {
    return timing == CompilationTimingMode::Detailed;
  }

private:
  explicit CompilationOptions(bool profileCompanion,
                              OptimizationConfig optimizations,
                              CompilationTimingMode timing)
      : profileCompanion(profileCompanion), optimizations(optimizations),
        timing(timing) {}

  bool profileCompanion;
  OptimizationConfig optimizations;
  CompilationTimingMode timing;
};

enum class ProgramResourceRole { UserInput, Parameter, Constant, Output };
enum class TerminalCompletionKind { EntryReturnAfterLocalDrain };
enum class TransportContract { None, DirectDTE };
enum class DDRAllocationContract { DefaultArenaRelativeOffsets };

/// A verified program-boundary resource projected to one logical rank. The
/// slice is copied from the frontend verifier's typed result; rank identity is
/// never inferred from its payload locator.
struct RankProgramBinding {
  ProgramResourceRole role;
  /// Accepted entry argument/result index.
  int64_t index;
  /// User-visible index within the resource role's program-boundary domain.
  int64_t programIndex;
  std::string name;
  std::string dtype;
  frontend::ProgramDistributionKind distribution;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  frontend::ProgramRankSlice slice;
};

/// One independently lowered and accepted static-rank program. This type is
/// move-only so an accepted module cannot be accidentally duplicated without
/// rerunning its rank-specific compiler gates.
class RankExecutable {
public:
  RankExecutable(RankExecutable &&) = default;
  RankExecutable &operator=(RankExecutable &&) = default;
  RankExecutable(const RankExecutable &) = delete;
  RankExecutable &operator=(const RankExecutable &) = delete;

  int64_t getLogicalRank() const { return logicalRank; }
  llvm::StringRef getEntrySymbol() const { return entrySymbol; }
  mlir::ModuleOp getModule() const { return *module; }
  llvm::StringRef getSelectedTileIR() const { return selectedTileIR; }
  const std::vector<RankProgramBinding> &getProgramBindings() const {
    return programBindings;
  }
  TerminalCompletionKind getTerminalCompletionKind() const {
    return terminalCompletionKind;
  }
  TransportContract getTransportContract() const { return transportContract; }
  DDRAllocationContract getDDRAllocationContract() const {
    return ddrAllocationContract;
  }

private:
  friend struct ExecutableBundleBuilder;

  RankExecutable(int64_t logicalRank, mlir::OwningOpRef<mlir::ModuleOp> module,
                 llvm::StringRef entrySymbol,
                 std::vector<RankProgramBinding> programBindings,
                 TransportContract transportContract,
                 llvm::StringRef selectedTileIR = {})
      : logicalRank(logicalRank), module(std::move(module)),
        entrySymbol(entrySymbol.str()),
        programBindings(std::move(programBindings)),
        terminalCompletionKind(
            TerminalCompletionKind::EntryReturnAfterLocalDrain),
        transportContract(transportContract),
        ddrAllocationContract(
            DDRAllocationContract::DefaultArenaRelativeOffsets),
        selectedTileIR(selectedTileIR.str()) {}

  int64_t logicalRank;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string entrySymbol;
  std::vector<RankProgramBinding> programBindings;
  TerminalCompletionKind terminalCompletionKind;
  TransportContract transportContract;
  DDRAllocationContract ddrAllocationContract;
  /// Same-invocation snapshot printed at the selected complete-rank Tile
  /// decision boundary before executable finalization. It is inspection
  /// evidence, not a package member or a semantic side channel.
  std::string selectedTileIR;
};

/// Atomic owner of the complete static logical-rank domain. The context is
/// owned alongside all modules and is destroyed only after the rank programs.
class ExecutableBundle {
public:
  ExecutableBundle(ExecutableBundle &&) = default;
  ExecutableBundle &operator=(ExecutableBundle &&) = default;
  ExecutableBundle(const ExecutableBundle &) = delete;
  ExecutableBundle &operator=(const ExecutableBundle &) = delete;

  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const std::vector<RankExecutable> &getRankExecutables() const {
    return rankExecutables;
  }
  const RuntimeLaunchContract &getRuntimeLaunchContract() const {
    return runtimeLaunchContract;
  }

private:
  friend struct ExecutableBundleBuilder;

  ExecutableBundle(ExecutionConfig executionConfig,
                   RuntimeLaunchContract runtimeLaunchContract,
                   std::shared_ptr<mlir::MLIRContext> context,
                   std::vector<RankExecutable> rankExecutables)
      : executionConfig(executionConfig),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        context(std::move(context)),
        rankExecutables(std::move(rankExecutables)) {}

  ExecutionConfig executionConfig;
  RuntimeLaunchContract runtimeLaunchContract;
  std::shared_ptr<mlir::MLIRContext> context;
  std::vector<RankExecutable> rankExecutables;
};

/// Reopens and verifies a structured tensor-program artifact, schedules every
/// configured logical rank in an isolated clone, and returns the bundle only
/// after the all-and-only rank domain has passed executable-admission legality.
llvm::Expected<ExecutableBundle>
compileTensorProgramToExecutableBundle(llvm::StringRef tensorProgramDirectory,
                                       ExecutionConfig executionConfig,
                                       llvm::raw_ostream &diagnostics);

/// Runs the production transaction through executable, target-artifact and
/// typed package bundles. The final root becomes visible only after canonical
/// manifest readback verifies every source tensor-program and target module
/// member.
/// The returned bundle is the same owner-backed accepted rank domain consumed
/// by target-artifact and package assembly; downstream gates must not rebuild
/// it from the published package.
mlir::FailureOr<ExecutableBundle> compileProgram(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics);

/// Runs the production transaction with an explicit typed product request.
/// When profiling is requested, the ordinary output remains the production
/// final production package and a verified sibling `<output>.profile`
/// companion is published only after the ordinary artifact and its
/// profile-only captures have completed their compiler-owned gates.
mlir::FailureOr<ExecutableBundle>
compileProgram(CompilationRequest request,
               llvm::StringRef outputProgramDirectory,
               llvm::StringRef xlaSpmdPartitionerHelper,
               const TargetToolchain &targetToolchain,
               CompilationOptions options, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_COMPILATION_H
