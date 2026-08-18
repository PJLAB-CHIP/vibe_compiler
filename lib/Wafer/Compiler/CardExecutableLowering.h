//===- CardExecutableLowering.h - Card executable lowering -*- C++
//-*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLELOWERING_H
#define WAFER_COMPILER_CARDEXECUTABLELOWERING_H

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

enum class CardExecutableLoweringFailureKind : uint8_t {
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

/// Typed failure produced while lowering Tile modules into one card
/// executable. `kind`
/// drives compiler control flow; `detail` and the stable label are diagnostic
/// only and must never be parsed to recover the rejection category.
struct CardExecutableLoweringFailure {
  CardExecutableLoweringFailureKind kind =
      CardExecutableLoweringFailureKind::None;
  std::string detail;

  explicit operator bool() const {
    return kind != CardExecutableLoweringFailureKind::None;
  }

  bool isProvenExactRejection() const;
  llvm::StringRef getDiagnosticLabel() const;
};

struct CardExecutableLoweringResult {
  CardExecutableLoweringResult(std::vector<TileExecutable> tiles,
                      RuntimeLaunchContract runtimeLaunchContract,
                      analysis::CardInstructionProgramCost resourceCost)
      : tiles(std::move(tiles)),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        resourceCost(std::move(resourceCost)) {}

  std::vector<TileExecutable> tiles;
  RuntimeLaunchContract runtimeLaunchContract;
  analysis::CardInstructionProgramCost resourceCost;
};

/// Invocation-local card-executable lowering instrumentation. It is never
/// stored in IR, Tile executables, or package files.
struct CardExecutableLoweringStatistics {
  uint64_t cardModuleCompilationInvocations = 0;
  uint64_t tileModuleLoweringAttempts = 0;
  uint64_t tileModuleLoweringSuccesses = 0;
  uint64_t cardExecutablesProduced = 0;
  uint64_t maximumTilePipelineWorkers = 1;
};

/// Takes exactly one memory-planned Instr module per available Tile.
/// The input modules receive DDR placement, index lowering,
/// exact Direct-DTE binding, card resource validation, Tile
/// executable construction, and runtime launch-contract formation.
/// `failure` is reset on entry and remains empty on success; callers branch on
/// its typed kind, never its diagnostic label or detail. Diagnostic IR traces
/// are deliberately outside this lowering boundary.
mlir::FailureOr<CardExecutableLoweringResult>
lowerTileModulesToCardExecutable(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> tileModules,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    CardExecutableLoweringFailure &failure, ProgramDataHandoff &programData,
    CardExecutableLoweringStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

/// Prove from one accepted Tile's current IR that every DDR movement
/// is attached to a function input or returned output root.
bool hasBoundaryOnlyDDRMovementEvidence(const TileExecutable &tile);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLELOWERING_H
