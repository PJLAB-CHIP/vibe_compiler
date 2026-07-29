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

/// A rank candidate after function-boundary bufferization, SPM replanning, and
/// exact cost closure. The reserved baseline must survive; failures of other
/// alternatives only prune the finalized frontier.
struct FinalizedRankCandidate {
  FinalizedRankCandidate(mlir::OwningOpRef<mlir::ModuleOp> module,
                         int64_t stableOrdinal,
                         wafer::RankArtifactKind artifactKind,
                         bool reservedBaseline,
                         wafer::RankBufferingKind bufferingKind,
                         uint32_t bufferingPlanOrdinal,
                         wafer::RankWorkerPlacementKind workerPlacementKind,
                         uint32_t workerPlacementPlanOrdinal)
      : module(std::move(module)), stableOrdinal(stableOrdinal),
        artifactKind(artifactKind), reservedBaseline(reservedBaseline),
        bufferingKind(bufferingKind),
        bufferingPlanOrdinal(bufferingPlanOrdinal),
        workerPlacementKind(workerPlacementKind),
        workerPlacementPlanOrdinal(workerPlacementPlanOrdinal) {}

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
};

mlir::FailureOr<std::vector<FinalizedRankCandidate>>
finalizeScheduledRankCandidateFrontier(
    std::vector<wafer::ScheduledRankCandidate> frontier,
    TargetProfileId targetProfile);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H
