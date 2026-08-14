//===- WholeCardExecutableAdmission.h - Whole-card exact gate -*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLECARDEXECUTABLEADMISSION_H
#define WAFER_COMPILER_WHOLECARDEXECUTABLEADMISSION_H

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

enum class WholeCardAdmissionFailureKind : uint8_t {
  None,
  Contract,
  PhysicalTileDomain,
  PhysicalTileMaterialization,
  PhysicalTileVerification,
  DDRPlanning,
  IndexLowering,
  DirectDTETransport,
  WholeCardResources,
  AcceptedPhysicalTileVerification,
  AcceptedCallClosure,
  PartitionResourceProjection,
  RuntimeLaunchContract,
  TargetABIPreparation,
  TargetABILowering,
};

/// Typed rejection produced by the whole-card admission boundary.  `kind`
/// drives compiler control flow; `detail` and the stable label are diagnostic
/// only and must never be parsed to recover the rejection category.
struct WholeCardAdmissionFailure {
  WholeCardAdmissionFailureKind kind = WholeCardAdmissionFailureKind::None;
  std::string detail;

  explicit operator bool() const {
    return kind != WholeCardAdmissionFailureKind::None;
  }

  bool isProvenExactRejection() const;
  llvm::StringRef getDiagnosticLabel() const;
};

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
  uint64_t cardProgramCompilationInvocations = 0;
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
/// `failure` is reset on entry and remains empty on success; callers branch on
/// its typed kind, never its diagnostic label or detail. Diagnostic IR traces
/// are deliberately outside this semantic admission boundary.
mlir::FailureOr<AcceptedWholeCardExecutable> admitWholeCardExecutable(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> physicalTileModules,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeCardAdmissionFailure &failure,
    WholeCardSynthesisStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

/// Prove from one admitted physical Tile's current IR that every DDR movement
/// is attached to a function input or returned output root.
bool hasBoundaryOnlyDDRMovementEvidence(const PhysicalTileExecutable &tile);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLECARDEXECUTABLEADMISSION_H
