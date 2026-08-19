//===- Compilation.h - Typed Wafer compiler request ------------*- C++ -*-===//

#ifndef WAFER_DRIVER_COMPILATION_H
#define WAFER_DRIVER_COMPILATION_H

#include "Wafer/Frontend/Program/Program.h"
#include "Wafer/Support/OptimizationConfig.h"
#include "Wafer/Target/Core/RuntimeLaunchContract.h"
#include "Wafer/Target/Core/TargetIdentity.h"
#include "Wafer/Target/Core/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

class TargetToolchain;
class ProgramDataHandoff;

enum class ProgramResourceRole { UserInput, Parameter, Constant, Output };

/// Stable identity of one logical program tensor inside a compilation.
/// Parameters use the function argument index, captured constants use the
/// exporter position, user inputs and outputs use the program boundary
/// index. The role disambiguates equal numbers across disjoint boundary
/// domains.
struct ProgramTensorId {
  ProgramResourceRole role = ProgramResourceRole::UserInput;
  int64_t roleIndex = -1;

  friend bool operator==(const ProgramTensorId &lhs,
                         const ProgramTensorId &rhs) {
    return lhs.role == rhs.role && lhs.roleIndex == rhs.roleIndex;
  }
  friend bool operator!=(const ProgramTensorId &lhs,
                         const ProgramTensorId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const ProgramTensorId &lhs,
                        const ProgramTensorId &rhs) {
    return std::make_pair(static_cast<uint8_t>(lhs.role), lhs.roleIndex) <
           std::make_pair(static_cast<uint8_t>(rhs.role), rhs.roleIndex);
  }
};

/// Invocation-local diagnostic policy. Detailed timing never changes source
/// semantics, candidate verification, selection, or written outputs.
enum class CompilationTimingMode { Disabled, Detailed };

/// Validated execution facts for the current single-card compiler boundary.
/// Card partitioning belongs to the source/SPMD domain. Tiles belong
/// to the target execution domain; the current target always exposes all 16.
/// There is deliberately no default configuration.
class ExecutionConfig {
public:
  static constexpr int64_t kSingleCardTileCount = 16;

  static llvm::Expected<ExecutionConfig>
  createForSingleCard(int64_t numPartitions);

  int64_t getNumPartitions() const { return numPartitions; }
  int64_t getTileCount() const { return tileCount; }
  TargetIdentityId getTargetIdentityId() const {
    return TargetIdentityId::waferTx81SingleCard();
  }

  friend bool operator==(const ExecutionConfig &lhs,
                         const ExecutionConfig &rhs) {
    return lhs.numPartitions == rhs.numPartitions &&
           lhs.tileCount == rhs.tileCount;
  }
  friend bool operator!=(const ExecutionConfig &lhs,
                         const ExecutionConfig &rhs) {
    return !(lhs == rhs);
  }

private:
  ExecutionConfig(int64_t numPartitions, int64_t tileCount)
      : numPartitions(numPartitions), tileCount(tileCount) {}

  int64_t numPartitions;
  int64_t tileCount;
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
    return CompilationOptions(/*profileInstrumentation=*/false, optimizations,
                              timing);
  }

  /// Requests a profile instrumentation for the production package. The
  /// ordinary package is compiled exactly once; the instrumentation contains
  /// profile-only captures for that same accepted Tile executable set.
  static llvm::Expected<CompilationOptions>
  profile(const ExecutionConfig &executionConfig,
          OptimizationConfig optimizations = OptimizationConfig::search(),
          CompilationTimingMode timing = CompilationTimingMode::Disabled);

  bool shouldProduceProfileInstrumentation() const {
    return profileInstrumentation;
  }
  OptimizationConfig getOptimizationConfig() const { return optimizations; }
  bool shouldReportDetailedTiming() const {
    return timing == CompilationTimingMode::Detailed;
  }

private:
  explicit CompilationOptions(bool profileInstrumentation,
                              OptimizationConfig optimizations,
                              CompilationTimingMode timing)
      : profileInstrumentation(profileInstrumentation),
        optimizations(optimizations), timing(timing) {}

  bool profileInstrumentation;
  OptimizationConfig optimizations;
  CompilationTimingMode timing;
};

enum class EntryLocalCompletionKind { ReturnAfterLocalDrain };
enum class TransportContract { None, DirectDTE };
enum class DDRAllocationContract { DefaultArenaRelativeOffsets };

/// A verified program-boundary resource projected to one card partition. The
/// slice is copied from the frontend verifier's typed result; partition
/// identity is never inferred from its payload locator. The stable program
/// tensor identity resolves parameter/constant payload access through the
/// CardExecutable's ProgramDataHandoff; the binding never carries payload
/// bytes and the slice payload path is provenance only.
struct ProgramResourceBinding {
  ProgramResourceRole role;
  /// Stable logical program tensor identity within this compilation.
  ProgramTensorId programTensorId;
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
/// separate from executable IR, package files, and runtime state and must never
/// be used to recover compilation semantics.
struct TileIRTrace {
  CardId cardId{0};
  TileId tileId{0};
  LaunchSlotId launchSlotId{0};
  std::string tileDataflowIR;
};

struct CompilationIRTrace {
  std::vector<TileIRTrace> tiles;
};

/// One independently lowered and accepted Tile executable. Its producer assigns
/// identity from the verified target topology; downstream consumers
/// never recover it from a symbol or module name. This type is move-only so an
/// accepted module cannot be accidentally duplicated without rerunning its
/// Tile-specific compiler gates.
class TileExecutable {
public:
  TileExecutable(TileExecutable &&) = default;
  TileExecutable &operator=(TileExecutable &&) = default;
  TileExecutable(const TileExecutable &) = delete;
  TileExecutable &operator=(const TileExecutable &) = delete;

  CardId getCardId() const { return cardId; }
  TileId getTileId() const { return tileId; }
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
  friend struct CardExecutableBuilder;

  TileExecutable(CardId cardId, TileId tileId, LaunchSlotId launchSlotId,
                 mlir::OwningOpRef<mlir::ModuleOp> module,
                 llvm::StringRef entrySymbol,
                 std::vector<ProgramResourceBinding> programBindings,
                 TransportContract transportContract)
      : cardId(cardId), tileId(tileId), launchSlotId(launchSlotId),
        module(std::move(module)), entrySymbol(entrySymbol.str()),
        programBindings(std::move(programBindings)),
        entryLocalCompletionKind(
            EntryLocalCompletionKind::ReturnAfterLocalDrain),
        transportContract(transportContract),
        ddrAllocationContract(
            DDRAllocationContract::DefaultArenaRelativeOffsets) {}

  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string entrySymbol;
  std::vector<ProgramResourceBinding> programBindings;
  EntryLocalCompletionKind entryLocalCompletionKind;
  TransportContract transportContract;
  DDRAllocationContract ddrAllocationContract;
};

/// Owns the complete executable for one target card. The context is owned
/// alongside all Tile modules and is destroyed only after them. The program
/// data handoff owns all-and-only payload sources this compilation consumed;
/// it lives exactly as long as the executable so target consumers read
/// parameter/constant bytes from owned content instead of reopening paths.
class CardExecutable {
public:
  CardExecutable(CardExecutable &&);
  CardExecutable &operator=(CardExecutable &&);
  CardExecutable(const CardExecutable &) = delete;
  CardExecutable &operator=(const CardExecutable &) = delete;
  ~CardExecutable();

  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const std::vector<TileExecutable> &getTileExecutables() const {
    return tiles;
  }
  const RuntimeLaunchContract &getRuntimeLaunchContract() const {
    return runtimeLaunchContract;
  }
  const ProgramDataHandoff &getProgramDataHandoff() const;

private:
  friend struct CardExecutableBuilder;

  CardExecutable(ExecutionConfig executionConfig,
                 RuntimeLaunchContract runtimeLaunchContract,
                 std::shared_ptr<mlir::MLIRContext> context,
                 std::vector<TileExecutable> tiles,
                 std::unique_ptr<ProgramDataHandoff> programData);

  ExecutionConfig executionConfig;
  RuntimeLaunchContract runtimeLaunchContract;
  std::shared_ptr<mlir::MLIRContext> context;
  std::vector<TileExecutable> tiles;
  std::unique_ptr<ProgramDataHandoff> programData;
};

/// Host-boundary classification of one compilation transaction failure. The
/// stage is set by the transaction at each stable pipeline phase; the caller's
/// diagnostics stream carries the detailed diagnostics and is never parsed to
/// recover control flow.
enum class CompilationStage {
  SourceVerification,
  SpmdPartitioning,
  TensorProgramPreparation,
  ExecutableCompilation,
  TargetCodeGeneration,
  PackageAssembly,
  PackageCommit,
};

llvm::StringRef stringifyCompilationStage(CompilationStage stage);

/// Typed failure for the compiler library entry. The stage classifies where
/// the transaction stopped; detailed diagnostics remain on the caller-owned
/// stream.
class CompilationFailure final : public llvm::ErrorInfo<CompilationFailure> {
public:
  static char ID;

  explicit CompilationFailure(CompilationStage stage) : stage(stage) {}

  CompilationStage getStage() const { return stage; }

  void log(llvm::raw_ostream &stream) const override {
    stream << "compilation failed at the "
           << stringifyCompilationStage(stage) << " stage";
  }

  std::error_code convertToErrorCode() const override {
    return llvm::errc::operation_not_permitted;
  }

private:
  CompilationStage stage;
};

} // namespace wafer::compiler

#endif // WAFER_DRIVER_COMPILATION_H
