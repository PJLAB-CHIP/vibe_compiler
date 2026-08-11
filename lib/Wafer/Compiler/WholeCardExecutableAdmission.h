//===- WholeCardExecutableAdmission.h - Whole-card exact gate -*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLECARDEXECUTABLEADMISSION_H
#define WAFER_COMPILER_WHOLECARDEXECUTABLEADMISSION_H

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Frontend/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

struct AcceptedWholeCardExecutable {
  AcceptedWholeCardExecutable(
      std::vector<PhysicalTileExecutable> tiles,
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
/// in IR, an executable bundle, or a package artifact.
struct WholeCardSynthesisStatistics {
  uint64_t preTargetAttempts = 0;
  uint64_t preTargetAccepted = 0;
  uint64_t targetGateInvocations = 0;
  uint64_t targetTileGateInvocations = 0;
  uint64_t admittedExecutableCount = 0;
  uint64_t maximumTilePipelineWorkers = 1;
};

/// Consumes exactly one owned, already finalized Instr module per available
/// physical Tile. The disposable set receives DDR placement, index lowering,
/// exact Direct-DTE binding, whole-card resource validation, physical Tile
/// executable projection, and target ABI/LLVM preflight atomically.
/// `selectedTileIR` remains query-local evidence owned by finalization and must
/// match the physical Tile domain.
mlir::FailureOr<AcceptedWholeCardExecutable> admitWholeCardExecutable(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> physicalTileModules,
    llvm::ArrayRef<std::shared_ptr<const std::string>> selectedTileIR,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    std::string *failureGate = nullptr,
    WholeCardSynthesisStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

/// Prove from one admitted physical Tile's current IR that every DDR movement
/// is attached to a function input or returned output root.
bool hasBoundaryOnlyDDRMovementEvidence(const PhysicalTileExecutable &tile);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLECARDEXECUTABLEADMISSION_H
