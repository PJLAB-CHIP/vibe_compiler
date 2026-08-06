//===- ScheduledRankFinalization.h - Final rank candidates ------*- C++ -*-===//

#ifndef WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H
#define WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H

#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

enum class RankCompletionPolicy : uint8_t {
  /// Preserve the completion topology already constructed by the rank-local
  /// scheduling mechanics. This is used only by the pre-coordinated
  /// qualification seam until that seam is retired.
  PreserveConstructedTopology,
  /// Erase compiler-derived joins after function-boundary bufferization and
  /// rebuild the minimum completion topology solely from current typed IR.
  RebuildFromCurrentEffects,
};

enum class RankFinalizationFailureKind : uint8_t {
  None,
  Contract,
  WholeVariantFacts,
  TileToInstr,
  Completion,
  Verification,
  FunctionBoundaryBufferization,
  SPMAllocation,
  ExactCost,
};

/// Typed, invocation-local rejection evidence for one finalized rank request.
/// It never carries a partially placed module or becomes an artifact.
struct RankFinalizationFailure {
  RankFinalizationFailureKind kind = RankFinalizationFailureKind::None;
  int64_t stableOrdinal = -1;
};

/// A rank candidate after complete-entry Tile-to-Instr conversion,
/// function-boundary bufferization, fresh completion, SPM replanning, and exact
/// cost closure. The reserved baseline must survive; failures of other
/// alternatives only prune the finalized frontier.
struct FinalizedRankCandidate {
  FinalizedRankCandidate(
      mlir::OwningOpRef<mlir::ModuleOp> module, int64_t stableOrdinal,
      wafer::RankArtifactKind artifactKind, bool reservedBaseline,
      wafer::RankBufferingKind bufferingKind, uint32_t bufferingPlanOrdinal,
      wafer::RankWorkerPlacementKind workerPlacementKind,
      uint32_t workerPlacementPlanOrdinal, uint32_t frontierOrderOrdinal)
      : module(std::move(module)), stableOrdinal(stableOrdinal),
        artifactKind(artifactKind), reservedBaseline(reservedBaseline),
        bufferingKind(bufferingKind),
        bufferingPlanOrdinal(bufferingPlanOrdinal),
        workerPlacementKind(workerPlacementKind),
        workerPlacementPlanOrdinal(workerPlacementPlanOrdinal),
        frontierOrderOrdinal(frontierOrderOrdinal) {}

  FinalizedRankCandidate(FinalizedRankCandidate &&) = default;
  FinalizedRankCandidate &operator=(FinalizedRankCandidate &&) = default;
  FinalizedRankCandidate(const FinalizedRankCandidate &) = delete;
  FinalizedRankCandidate &operator=(const FinalizedRankCandidate &) = delete;

  mlir::OwningOpRef<mlir::ModuleOp> module;
  int64_t stableOrdinal;
  wafer::RankArtifactKind artifactKind;
  bool reservedBaseline;
  wafer::RankBufferingKind bufferingKind;
  uint32_t bufferingPlanOrdinal;
  wafer::RankWorkerPlacementKind workerPlacementKind;
  uint32_t workerPlacementPlanOrdinal;
  /// Canonical pre-finalization request order. This is used only to merge
  /// independently finalized request shards before whole-variant planning.
  uint32_t frontierOrderOrdinal;
  std::shared_ptr<const std::string> selectedTileIR;
};

mlir::FailureOr<std::vector<FinalizedRankCandidate>>
finalizeScheduledRankCandidateFrontier(
    std::vector<wafer::ScheduledRankCandidate> frontier,
    bool requireReservedBaseline = true,
    RankCompletionPolicy completionPolicy =
        RankCompletionPolicy::PreserveConstructedTopology,
    RankFinalizationFailure *failure = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H
