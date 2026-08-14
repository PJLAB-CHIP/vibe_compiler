//===- Compilation.h - Typed Wafer compiler request ------------*- C++ -*-===//

#ifndef WAFER_COMPILER_COMPILATION_H
#define WAFER_COMPILER_COMPILATION_H

#include "Wafer/Frontend/Program.h"
#include "Wafer/Support/OptimizationConfig.h"
#include "Wafer/Target/PhysicalIds.h"
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
/// Card partitioning belongs to the source/SPMD domain. Physical Tiles belong
/// to the target execution domain; the current target always exposes all 16.
/// There is deliberately no default configuration.
class ExecutionConfig {
public:
  static constexpr int64_t kSingleCardPhysicalTileCount = 16;

  static llvm::Expected<ExecutionConfig>
  createForSingleCard(int64_t numPartitions,
                      RuntimeLaunchKind runtimeLaunchKind);

  int64_t getNumPartitions() const { return numPartitions; }
  int64_t getPhysicalTileCount() const { return physicalTileCount; }
  TargetIdentityId getTargetIdentityId() const {
    return TargetIdentityId::waferTx81SingleCard();
  }
  RuntimeLaunchKind getRuntimeLaunchKind() const { return runtimeLaunchKind; }

  friend bool operator==(const ExecutionConfig &lhs,
                         const ExecutionConfig &rhs) {
    return lhs.numPartitions == rhs.numPartitions &&
           lhs.physicalTileCount == rhs.physicalTileCount &&
           lhs.runtimeLaunchKind == rhs.runtimeLaunchKind;
  }
  friend bool operator!=(const ExecutionConfig &lhs,
                         const ExecutionConfig &rhs) {
    return !(lhs == rhs);
  }

private:
  ExecutionConfig(int64_t numPartitions, int64_t physicalTileCount,
                  RuntimeLaunchKind runtimeLaunchKind)
      : numPartitions(numPartitions), physicalTileCount(physicalTileCount),
        runtimeLaunchKind(runtimeLaunchKind) {}

  int64_t numPartitions;
  int64_t physicalTileCount;
  RuntimeLaunchKind runtimeLaunchKind;
};

/// Move-only semantic input to the compiler driver. Output locations,
/// toolchain helper paths, pass names and per-Tile loop indices are
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
  standard(OptimizationConfig optimizations = OptimizationConfig::search(),
           CompilationTimingMode timing = CompilationTimingMode::Disabled) {
    return CompilationOptions(/*profileCompanion=*/false, optimizations,
                              timing);
  }

  /// Requests a production-artifact profile companion. The ordinary package is
  /// compiled exactly once; the companion contains profile-only captures for
  /// that same accepted complete-card physical Tile artifact.
  static llvm::Expected<CompilationOptions>
  profile(const ExecutionConfig &executionConfig,
          OptimizationConfig optimizations = OptimizationConfig::search(),
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
enum class EntryLocalCompletionKind { ReturnAfterLocalDrain };
enum class TransportContract { None, DirectDTE };
enum class DDRAllocationContract { DefaultArenaRelativeOffsets };

/// A verified program-boundary resource projected to one card partition. The
/// slice is copied from the frontend verifier's typed result; partition
/// identity is never inferred from its payload locator.
struct ProgramResourceBinding {
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
  frontend::ProgramPartitionSlice slice;
};

/// Optional same-invocation compiler inspection output. It is deliberately
/// separate from executable, package and runtime artifacts and must never be
/// used to recover compilation semantics.
struct PhysicalTileIRTrace {
  PhysicalCardId physicalCardId{0};
  PhysicalTileId physicalTileId{0};
  LaunchSlotId launchSlotId{0};
  std::string tileDataflowIR;
};

struct CompilationIRTrace {
  std::vector<PhysicalTileIRTrace> physicalTiles;
};

/// One independently lowered and accepted physical Tile program. Its producer
/// assigns identity from the verified physical topology; downstream consumers
/// never recover it from a symbol or module name. This type is move-only so an
/// accepted module cannot be accidentally duplicated without rerunning its
/// Tile-specific compiler gates.
class PhysicalTileExecutable {
public:
  PhysicalTileExecutable(PhysicalTileExecutable &&) = default;
  PhysicalTileExecutable &operator=(PhysicalTileExecutable &&) = default;
  PhysicalTileExecutable(const PhysicalTileExecutable &) = delete;
  PhysicalTileExecutable &operator=(const PhysicalTileExecutable &) = delete;

  PhysicalCardId getPhysicalCardId() const { return physicalCardId; }
  PhysicalTileId getPhysicalTileId() const { return physicalTileId; }
  LaunchSlotId getLaunchSlotId() const { return launchSlotId; }
  llvm::StringRef getEntrySymbol() const { return entrySymbol; }
  mlir::ModuleOp getModule() const { return *module; }
  const std::vector<ProgramResourceBinding> &getProgramBindings() const {
    return programBindings;
  }
  EntryLocalCompletionKind getEntryLocalCompletionKind() const {
    return entryLocalCompletionKind;
  }
  TransportContract getTransportContract() const { return transportContract; }
  DDRAllocationContract getDDRAllocationContract() const {
    return ddrAllocationContract;
  }

private:
  friend struct ExecutableBundleBuilder;

  PhysicalTileExecutable(PhysicalCardId physicalCardId,
                         PhysicalTileId physicalTileId,
                         LaunchSlotId launchSlotId,
                         mlir::OwningOpRef<mlir::ModuleOp> module,
                         llvm::StringRef entrySymbol,
                         std::vector<ProgramResourceBinding> programBindings,
                         TransportContract transportContract)
      : physicalCardId(physicalCardId), physicalTileId(physicalTileId),
        launchSlotId(launchSlotId), module(std::move(module)),
        entrySymbol(entrySymbol.str()),
        programBindings(std::move(programBindings)),
        entryLocalCompletionKind(
            EntryLocalCompletionKind::ReturnAfterLocalDrain),
        transportContract(transportContract),
        ddrAllocationContract(
            DDRAllocationContract::DefaultArenaRelativeOffsets) {}

  PhysicalCardId physicalCardId;
  PhysicalTileId physicalTileId;
  LaunchSlotId launchSlotId;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string entrySymbol;
  std::vector<ProgramResourceBinding> programBindings;
  EntryLocalCompletionKind entryLocalCompletionKind;
  TransportContract transportContract;
  DDRAllocationContract ddrAllocationContract;
};

/// Atomic owner of the all-and-only available physical Tile domain for one
/// card. The context is owned alongside all modules and is destroyed only
/// after the Tile programs.
class ExecutableBundle {
public:
  ExecutableBundle(ExecutableBundle &&) = default;
  ExecutableBundle &operator=(ExecutableBundle &&) = default;
  ExecutableBundle(const ExecutableBundle &) = delete;
  ExecutableBundle &operator=(const ExecutableBundle &) = delete;

  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const std::vector<PhysicalTileExecutable> &
  getPhysicalTileExecutables() const {
    return physicalTileExecutables;
  }
  const RuntimeLaunchContract &getRuntimeLaunchContract() const {
    return runtimeLaunchContract;
  }

private:
  friend struct ExecutableBundleBuilder;

  ExecutableBundle(ExecutionConfig executionConfig,
                   RuntimeLaunchContract runtimeLaunchContract,
                   std::shared_ptr<mlir::MLIRContext> context,
                   std::vector<PhysicalTileExecutable> physicalTileExecutables)
      : executionConfig(executionConfig),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        context(std::move(context)),
        physicalTileExecutables(std::move(physicalTileExecutables)) {}

  ExecutionConfig executionConfig;
  RuntimeLaunchContract runtimeLaunchContract;
  std::shared_ptr<mlir::MLIRContext> context;
  std::vector<PhysicalTileExecutable> physicalTileExecutables;
};

/// Runs the production transaction with an explicit typed product request.
/// It traverses executable, target-artifact and typed package boundaries; the
/// final root becomes visible only after canonical manifest readback verifies
/// every source tensor-program and target module member. The returned bundle
/// is the same owner-backed accepted physical-Tile domain consumed by target
/// artifact and package assembly; downstream gates must not rebuild it from
/// the published package.
/// When profiling is requested, the ordinary output remains the production
/// package, and a verified sibling `<output>.profile` companion is published
/// only after the ordinary artifact and its profile-only captures have
/// completed their compiler-owned gates.
mlir::FailureOr<ExecutableBundle>
compileProgram(CompilationRequest request,
               llvm::StringRef outputProgramDirectory,
               llvm::StringRef xlaSpmdPartitionerHelper,
               const TargetToolchain &targetToolchain,
               CompilationOptions options, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_COMPILATION_H
