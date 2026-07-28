//===- RankCandidateFrontier.h - Private rank frontier --------*- C++ -*-===//

#ifndef WAFER_LIB_TRANSFORMS_SCHEDULING_RANKCANDIDATEFRONTIER_H
#define WAFER_LIB_TRANSFORMS_SCHEDULING_RANKCANDIDATEFRONTIER_H

#include "Wafer/Target/TargetProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
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

/// Compiler-private buffering realization. Buffering is orthogonal to storage
/// residency and ready-order: a fixed-slot software pipeline must correspond
/// with the same buffering realization and canonical plan ordinal on every
/// logical rank instead of masquerading as an existing artifact kind.
enum class RankBufferingKind : uint8_t {
  Single,
  StaticFixedSlot,
};

/// One fully materialized and rank-planned scheduling alternative. The stable
/// ordinal identifies one semantic generation; artifact kind and buffering
/// kind/plan ordinal identify independent physical derivation dimensions
/// across logical ranks. None is persisted in accepted IR or used as a
/// resource preference.
struct ScheduledRankCandidate {
  ScheduledRankCandidate(
      mlir::OwningOpRef<mlir::ModuleOp> module, int64_t stableOrdinal,
      RankArtifactKind artifactKind, bool reservedBaseline = false,
      RankBufferingKind bufferingKind = RankBufferingKind::Single,
      uint32_t bufferingPlanOrdinal = 0)
      : module(std::move(module)), stableOrdinal(stableOrdinal),
        artifactKind(artifactKind), reservedBaseline(reservedBaseline),
        bufferingKind(bufferingKind),
        bufferingPlanOrdinal(bufferingPlanOrdinal) {}

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
  RankBufferingKind bufferingKind;
  /// Deterministic loop/plan identity within one semantic generation.
  uint32_t bufferingPlanOrdinal;
};

struct TensorProgramSchedulingConfig {
  int64_t logicalRank = -1;
  int64_t candidateParallelism = 1;
  /// Exact versioned target/ABI contract selected by the production
  /// ExecutionConfig. This is static compile input, never a live-card or
  /// profiler result. There is deliberately no scheduler-local default: an
  /// omitted contract rejects the frontier before any candidate analysis.
  std::optional<TargetProfileId> targetProfile;
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
