//===- CoordinatedTerminalEvaluation.h - Exact all-rank gate -*- C++ -*-===//

#ifndef WAFER_COMPILER_COORDINATEDTERMINALEVALUATION_H
#define WAFER_COMPILER_COORDINATEDTERMINALEVALUATION_H

#include "CoordinatedDataflowSearch.h"
#include "WholeVariantCoordinator.h"

#include "Wafer/Frontend/Program.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

enum class CoordinatedTerminalFailureKind : uint8_t {
  None,
  RankDomain,
  RankFinalization,
  SPMAllocation,
  WholeVariantExactGate,
  WorkLedger,
};

/// Typed feedback returned to the still-live structured coordinator. It names
/// the failed semantic gate and optional logical rank, but carries no partial
/// placement and cannot mutate or repair the Tile parent.
struct CoordinatedTerminalFailure {
  CoordinatedTerminalFailureKind kind = CoordinatedTerminalFailureKind::None;
  int64_t logicalRank = -1;
  std::string gate;
};

struct FullyGatedCoordinatedVariant {
  FullyGatedCoordinatedVariant(int64_t stableSemanticOrdinal,
                               bool reservedBaseline,
                               AcceptedWholeVariant variant,
                               uint32_t terminalActionOrdinal = 0)
      : stableSemanticOrdinal(stableSemanticOrdinal),
        terminalActionOrdinal(terminalActionOrdinal),
        reservedBaseline(reservedBaseline), variant(std::move(variant)) {}

  FullyGatedCoordinatedVariant(FullyGatedCoordinatedVariant &&) noexcept =
      default;
  FullyGatedCoordinatedVariant &
  operator=(FullyGatedCoordinatedVariant &&) noexcept = default;
  FullyGatedCoordinatedVariant(const FullyGatedCoordinatedVariant &) = delete;
  FullyGatedCoordinatedVariant &
  operator=(const FullyGatedCoordinatedVariant &) = delete;

  int64_t stableSemanticOrdinal = 0;
  uint32_t terminalActionOrdinal = 0;
  bool reservedBaseline = false;
  AcceptedWholeVariant variant;
};

/// One finite all-rank scheduling action materialized from canonical unplaced
/// Instr parents. The action owns exactly one sibling for every logical rank;
/// no rank-local frontier or cross-rank product is exposed.
struct CoordinatedTerminalScheduleAction {
  uint32_t stableOrdinal = 0;
  RankArtifactKind artifactKind = RankArtifactKind::Spill;
  RankBufferingKind bufferingKind = RankBufferingKind::Single;
  uint32_t bufferingPlanOrdinal = 0;
  RankWorkerPlacementKind workerPlacementKind =
      RankWorkerPlacementKind::Unplaced;
  uint32_t workerPlacementPlanOrdinal = 0;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> rankModules;
};

/// Derives the bounded complete-rank ready-order, fixed-slot, and worker
/// siblings from canonical unplaced Instr parents. A requested action is
/// published only when the same typed action succeeds for all ranks.
mlir::FailureOr<std::vector<CoordinatedTerminalScheduleAction>>
deriveCoordinatedTerminalScheduleActions(
    llvm::ArrayRef<mlir::ModuleOp> canonicalInstrParents,
    const OptimizationConfig &optimizations,
    uint64_t *materializationWork = nullptr);

/// Evaluates one complete all-rank Tile candidate on disposable clones. Every
/// rank is lowered and finalized before whole-variant DDR, transport,
/// resources, and ABI gates run. Success consumes the candidate's coordinator
/// reservation atomically; failure also closes the attempted reservation with
/// actual work while leaving the original Tile parent available for a newly
/// reserved coordinator-owned repair sibling.
mlir::FailureOr<std::vector<FullyGatedCoordinatedVariant>>
evaluateCoordinatedTileVariant(
    const CoordinatedTileVariant &tileVariant,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    const OptimizationConfig &optimizations, CoordinatedWorkLedger &ledger,
    llvm::raw_ostream &diagnostics, CoordinatedTerminalFailure &failure,
    WholeVariantSelectionStatistics *statistics = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_COORDINATEDTERMINALEVALUATION_H
