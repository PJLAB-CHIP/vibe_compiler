//===- ExecutableLowering.h - Executable lowering ---------*- C++
//-*-===//

#ifndef WAFER_COMPILER_EXECUTABLELOWERING_H
#define WAFER_COMPILER_EXECUTABLELOWERING_H

#include "Wafer/Analysis/Instr/ScheduleCostAnalysis.h"
#include "Wafer/Driver/Compilation.h"
#include "Wafer/Frontend/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

enum class ExecutableLoweringFailureKind : uint8_t {
  None,
  Contract,
  TileDomain,
  MissingTileModule,
  TileVerification,
  DDRPlanning,
  IndexLowering,
  DirectDTETransport,
  ProgramResources,
  TileExecutableVerification,
  CallClosure,
  ProgramResourceBindings,
  RuntimeLaunchContract,
};

/// Typed failure produced while lowering Tile modules into one device
/// executable. `kind`
/// drives compiler control flow; `detail` and the stable label are diagnostic
/// only and must never be parsed to recover the rejection category.
struct ExecutableLoweringFailure {
  ExecutableLoweringFailureKind kind = ExecutableLoweringFailureKind::None;
  std::string detail;

  explicit operator bool() const {
    return kind != ExecutableLoweringFailureKind::None;
  }

  bool isProvenExactRejection() const;
  llvm::StringRef getDiagnosticLabel() const;
};

struct ExecutableLoweringResult {
  ExecutableLoweringResult(
      std::vector<TileExecutable> tiles,
      RuntimeLaunchContract runtimeLaunchContract,
      analysis::InstructionProgramAggregateCost resourceCost)
      : tiles(std::move(tiles)),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        resourceCost(std::move(resourceCost)) {}

  std::vector<TileExecutable> tiles;
  RuntimeLaunchContract runtimeLaunchContract;
  analysis::InstructionProgramAggregateCost resourceCost;
};

/// Invocation-local device-executable lowering instrumentation. It is never
/// stored in IR, Tile executables, or package files.
struct ExecutableLoweringStatistics {
  uint64_t executableCompilationInvocations = 0;
  uint64_t actualMemoryTargetGateInvocations = 0;
  uint64_t tileModuleLoweringAttempts = 0;
  uint64_t tileModuleLoweringSuccesses = 0;
  uint64_t deviceExecutablesProduced = 0;
  uint64_t maximumTilePipelineWorkers = 1;
  uint64_t currentIRLayoutOptimizationInvocations = 0;
  uint64_t currentIRLayoutPBQPWork = 0;
  uint64_t currentIRLayoutMaterializationsBefore = 0;
  uint64_t currentIRLayoutMaterializationsAfter = 0;
  uint64_t currentIRLayoutMaterializationsErased = 0;
  uint64_t currentIRLayoutMaterializationsReused = 0;
  uint64_t currentIRLayoutHardOnlyInvocations = 0;
  uint64_t redundantFullBufferTransfersEliminated = 0;
};

/// Takes exactly one memory-planned Instr module per available Tile.
/// The input modules receive DDR placement, index lowering,
/// exact Direct-DTE binding, shared-resource validation, Tile
/// executable construction, and runtime launch-contract formation.
/// `failure` is reset on entry and remains empty on success; callers branch on
/// its typed kind, never its diagnostic label or detail. Diagnostic IR traces
/// are deliberately outside this lowering boundary.
mlir::FailureOr<ExecutableLoweringResult> lowerTileModulesToExecutable(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> tileModules,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ExecutableLoweringFailure &failure, ProgramDataHandoff &programData,
    ExecutableLoweringStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

/// Prove from one accepted Tile's current IR that every DDR movement
/// is attached to a function input or returned output root.
bool hasBoundaryOnlyDDRMovementEvidence(const TileExecutable &tile);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_EXECUTABLELOWERING_H
