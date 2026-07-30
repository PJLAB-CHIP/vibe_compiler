//===- RankCandidateFrontier.h - Private rank frontier --------*- C++ -*-===//

#ifndef WAFER_LIB_TRANSFORMS_SCHEDULING_RANKCANDIDATEFRONTIER_H
#define WAFER_LIB_TRANSFORMS_SCHEDULING_RANKCANDIDATEFRONTIER_H

#include "Wafer/Support/OptimizationConfig.h"
#include "Wafer/Target/TargetProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace wafer {

/// Bounded admission policy for the compiler-private rank frontier. The
/// conservative fallback is outside these bands. Fixed-slot and worker
/// placement candidates have independent allowances so a saturated general
/// search cannot erase an orthogonal physical realization solely because it
/// is generated later.
inline constexpr unsigned kGeneralRankFrontierAdmissionLimit = 256;
inline constexpr unsigned kFixedSlotRankFrontierAdmissionLimit = 8;
inline constexpr unsigned kWorkerPlacementRankFrontierAdmissionLimit = 8;
inline constexpr unsigned kMaximumScheduledRankFrontierSize =
    1 + kGeneralRankFrontierAdmissionLimit +
    kFixedSlotRankFrontierAdmissionLimit +
    kWorkerPlacementRankFrontierAdmissionLimit;

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

/// Compiler-private NCC worker realization. Worker placement is orthogonal to
/// storage, ready order, and buffering so a fixed-slot or later NoC-resident
/// actual clone can independently derive the same canonical component plan.
enum class RankWorkerPlacementKind : uint8_t {
  Unplaced,
  DisjointComponents,
};

/// Invocation-local admission accounting for orthogonal physical
/// realizations. A worker-placed fixed-slot candidate belongs to the worker
/// band because that is the later independent derivation that would otherwise
/// be truncated; its full buffering identity remains on the candidate tuple.
class RankFrontierAdmissionState {
public:
  bool tryAdmit(RankBufferingKind bufferingKind,
                RankWorkerPlacementKind workerPlacementKind) {
    unsigned *count = &generalCount;
    unsigned limit = kGeneralRankFrontierAdmissionLimit;
    if (workerPlacementKind != RankWorkerPlacementKind::Unplaced) {
      count = &workerPlacementCount;
      limit = kWorkerPlacementRankFrontierAdmissionLimit;
    } else if (bufferingKind == RankBufferingKind::StaticFixedSlot) {
      count = &fixedSlotCount;
      limit = kFixedSlotRankFrontierAdmissionLimit;
    }
    if (*count >= limit)
      return false;
    ++*count;
    return true;
  }

  bool allBandsFull() const {
    return generalCount >= kGeneralRankFrontierAdmissionLimit &&
           fixedSlotCount >= kFixedSlotRankFrontierAdmissionLimit &&
           workerPlacementCount >= kWorkerPlacementRankFrontierAdmissionLimit;
  }

  unsigned getGeneralCount() const { return generalCount; }
  unsigned getFixedSlotCount() const { return fixedSlotCount; }
  unsigned getWorkerPlacementCount() const { return workerPlacementCount; }

private:
  unsigned generalCount = 0;
  unsigned fixedSlotCount = 0;
  unsigned workerPlacementCount = 0;
};

/// One fully materialized and rank-planned scheduling alternative. The stable
/// ordinal identifies one semantic generation; artifact kind and buffering
/// kind/plan ordinal plus worker-placement kind/plan ordinal identify
/// independent physical derivation dimensions across logical ranks. None is
/// persisted in accepted IR or used as a resource preference.
struct ScheduledRankCandidate {
  ScheduledRankCandidate(
      mlir::OwningOpRef<mlir::ModuleOp> module, int64_t stableOrdinal,
      RankArtifactKind artifactKind, bool reservedBaseline = false,
      RankBufferingKind bufferingKind = RankBufferingKind::Single,
      uint32_t bufferingPlanOrdinal = 0,
      RankWorkerPlacementKind workerPlacementKind =
          RankWorkerPlacementKind::Unplaced,
      uint32_t workerPlacementPlanOrdinal = 0,
      uint32_t frontierOrderOrdinal = 0)
      : module(std::move(module)), stableOrdinal(stableOrdinal),
        artifactKind(artifactKind), reservedBaseline(reservedBaseline),
        bufferingKind(bufferingKind),
        bufferingPlanOrdinal(bufferingPlanOrdinal),
        workerPlacementKind(workerPlacementKind),
        workerPlacementPlanOrdinal(workerPlacementPlanOrdinal),
        frontierOrderOrdinal(frontierOrderOrdinal) {}

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
  RankWorkerPlacementKind workerPlacementKind;
  /// Deterministic dependency-component assignment identity.
  uint32_t workerPlacementPlanOrdinal;
  /// Invocation-local canonical request order. This is used only to merge
  /// independently evaluated search shards before final rank admission; it is
  /// never persisted or used as semantic correspondence identity.
  uint32_t frontierOrderOrdinal;
};

struct TensorProgramSchedulingConfig {
  int64_t logicalRank = -1;
  int64_t candidateParallelism = 1;
  /// Optional bounded partition of the deterministic rank request sequence.
  /// All requests with the same source/recipe identity remain in one shard so
  /// partition-signature suppression is unchanged. Shards must be merged by
  /// `frontierOrderOrdinal` and readmitted before they form a rank frontier.
  uint32_t requestShardIndex = 0;
  uint32_t requestShardCount = 1;
  /// Exact versioned target/ABI contract selected by the production
  /// ExecutionConfig. This is static compile input, never a live-card or
  /// profiler result. There is deliberately no scheduler-local default: an
  /// omitted contract rejects the frontier before any candidate analysis.
  std::optional<TargetProfileId> targetProfile;
  /// Optional semantic candidate producers enabled for this invocation.
  /// Mandatory legality and lowering stages are not represented here.
  OptimizationConfig optimizations = OptimizationConfig::production();
};

/// Returns true exactly when rank-local scheduling cannot observe
/// `logicalRank` from the current typed tensor program. The only
/// rank-dependent lowering operations are the typed collective family; all
/// other scheduling inputs are common to every rank in the same verified
/// tensor-program artifact.
bool isTensorProgramSchedulingRankInvariant(mlir::ModuleOp sourceModule);

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
