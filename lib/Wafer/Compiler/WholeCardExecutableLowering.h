//===- WholeCardExecutableLowering.h - Whole-card executable lowering -*- C++
//-*-===//

#ifndef WAFER_COMPILER_WHOLECARDEXECUTABLELOWERING_H
#define WAFER_COMPILER_WHOLECARDEXECUTABLELOWERING_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
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

enum class WholeCardExecutableLoweringFailureKind : uint8_t {
  None,
  Contract,
  PhysicalTileDomain,
  MissingPhysicalTileModule,
  PhysicalTileVerification,
  DDRPlanning,
  IndexLowering,
  DirectDTETransport,
  WholeCardResources,
  PhysicalTileExecutableVerification,
  CallClosure,
  ProgramResourceBindings,
  RuntimeLaunchContract,
  TargetABIPreparation,
  TargetABILowering,
};

/// Typed failure produced while lowering physical-Tile modules into one
/// whole-card executable. `kind`
/// drives compiler control flow; `detail` and the stable label are diagnostic
/// only and must never be parsed to recover the rejection category.
struct WholeCardExecutableLoweringFailure {
  WholeCardExecutableLoweringFailureKind kind =
      WholeCardExecutableLoweringFailureKind::None;
  std::string detail;

  explicit operator bool() const {
    return kind != WholeCardExecutableLoweringFailureKind::None;
  }

  bool isProvenExactRejection() const;
  llvm::StringRef getDiagnosticLabel() const;
};

struct WholeCardExecutable {
  WholeCardExecutable(std::vector<PhysicalTileExecutable> tiles,
                      RuntimeLaunchContract runtimeLaunchContract,
                      analysis::WholeCardInstructionProgramCost resourceCost)
      : tiles(std::move(tiles)),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        resourceCost(std::move(resourceCost)) {}

  std::vector<PhysicalTileExecutable> tiles;
  RuntimeLaunchContract runtimeLaunchContract;
  analysis::WholeCardInstructionProgramCost resourceCost;
};

/// Invocation-local whole-card synthesis instrumentation. It is never stored
/// in IR, physical Tile executables, or package files.
struct WholeCardSynthesisStatistics {
  uint64_t cardProgramCompilationInvocations = 0;
  uint64_t physicalTileModuleLoweringAttempts = 0;
  uint64_t physicalTileModuleLoweringSuccesses = 0;
  uint64_t targetLoweringVerificationInvocations = 0;
  uint64_t targetTileLoweringVerificationInvocations = 0;
  uint64_t wholeCardExecutablesProduced = 0;
  uint64_t maximumTilePipelineWorkers = 1;
};

/// Takes exactly one memory-planned Instr module per available physical Tile.
/// The input modules receive DDR placement, index lowering,
/// exact Direct-DTE binding, whole-card resource validation, physical Tile
/// executable construction, and target ABI/LLVM lowering verification.
/// `failure` is reset on entry and remains empty on success; callers branch on
/// its typed kind, never its diagnostic label or detail. Diagnostic IR traces
/// are deliberately outside this lowering boundary.
mlir::FailureOr<WholeCardExecutable> lowerPhysicalTileModulesToExecutable(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> physicalTileModules,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeCardExecutableLoweringFailure &failure,
    WholeCardSynthesisStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

/// Prove from one accepted physical Tile's current IR that every DDR movement
/// is attached to a function input or returned output root.
bool hasBoundaryOnlyDDRMovementEvidence(const PhysicalTileExecutable &tile);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLECARDEXECUTABLELOWERING_H
