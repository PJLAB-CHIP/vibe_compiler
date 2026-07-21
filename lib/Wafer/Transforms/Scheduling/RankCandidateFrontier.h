//===- RankCandidateFrontier.h - Private rank frontier --------*- C++ -*-===//

#ifndef WAFER_LIB_TRANSFORMS_SCHEDULING_RANKCANDIDATEFRONTIER_H
#define WAFER_LIB_TRANSFORMS_SCHEDULING_RANKCANDIDATEFRONTIER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace wafer {

/// Compiler-private physical derivation of one rank artifact. The value is a
/// cross-rank correspondence fact, not a persisted IR property or a cost.
enum class RankArtifactKind : uint8_t {
  Spill,
  SpillReady,
  Resident,
  ResidentReady,
};

/// One fully materialized and rank-planned scheduling alternative. The stable
/// ordinal identifies one semantic generation and the artifact kind identifies
/// its physical derivation across logical ranks. Neither is persisted in
/// accepted IR or used as a resource preference.
struct ScheduledRankCandidate {
  ScheduledRankCandidate(mlir::OwningOpRef<mlir::ModuleOp> module,
                         int64_t stableOrdinal,
                         RankArtifactKind artifactKind,
                         bool reservedBaseline = false)
      : module(std::move(module)), stableOrdinal(stableOrdinal),
        artifactKind(artifactKind), reservedBaseline(reservedBaseline) {}

  ScheduledRankCandidate(ScheduledRankCandidate &&) = default;
  ScheduledRankCandidate &operator=(ScheduledRankCandidate &&) = default;
  ScheduledRankCandidate(const ScheduledRankCandidate &) = delete;
  ScheduledRankCandidate &operator=(const ScheduledRankCandidate &) = delete;

  mlir::OwningOpRef<mlir::ModuleOp> module;
  int64_t stableOrdinal;
  RankArtifactKind artifactKind;
  /// Compiler-private allowance marker. Exactly one finalized candidate per
  /// rank carries it; it is never persisted in accepted IR or artifacts.
  bool reservedBaseline;
};

struct TensorProgramSchedulingConfig {
  int64_t logicalRank = -1;
  int64_t candidateParallelism = 1;
};

/// Builds a bounded frontier of complete rank alternatives in independent
/// clones. Every returned module has passed task commit, whole-rank SPM
/// planning, verification, and exact cost closure. The source module is
/// never modified. DDR placement, cross-rank transport, card resources, and
/// target ABI are deliberately deferred to the executable-bundle coordinator.
mlir::FailureOr<std::vector<ScheduledRankCandidate>>
buildScheduledRankCandidateFrontier(
    mlir::ModuleOp sourceModule, const TensorProgramSchedulingConfig &config);

} // namespace wafer

#endif // WAFER_LIB_TRANSFORMS_SCHEDULING_RANKCANDIDATEFRONTIER_H
