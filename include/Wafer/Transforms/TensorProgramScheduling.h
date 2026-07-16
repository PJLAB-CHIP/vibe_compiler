//===- TensorProgramScheduling.h - Rank scheduling frontier ---*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TENSORPROGRAMSCHEDULING_H
#define WAFER_TRANSFORMS_TENSORPROGRAMSCHEDULING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace wafer {

/// One fully materialized and rank-planned scheduling alternative.  The
/// discovery order is a deterministic compiler-private tie-breaker; neither it
/// nor the estimated cost is persisted in accepted IR.
struct ScheduledRankCandidate {
  ScheduledRankCandidate(mlir::OwningOpRef<mlir::ModuleOp> module,
                         int64_t estimatedTimePs, int64_t discoveryOrder)
      : module(std::move(module)), estimatedTimePs(estimatedTimePs),
        discoveryOrder(discoveryOrder) {}

  ScheduledRankCandidate(ScheduledRankCandidate &&) = default;
  ScheduledRankCandidate &operator=(ScheduledRankCandidate &&) = default;
  ScheduledRankCandidate(const ScheduledRankCandidate &) = delete;
  ScheduledRankCandidate &operator=(const ScheduledRankCandidate &) = delete;

  mlir::OwningOpRef<mlir::ModuleOp> module;
  int64_t estimatedTimePs;
  int64_t discoveryOrder;
};

struct TensorProgramSchedulingConfig {
  int64_t logicalRank = -1;
  int64_t candidateParallelism = 1;
};

/// Builds a bounded frontier of complete rank alternatives in independent
/// clones.  Every returned module has passed task commit, whole-rank SPM/DDR
/// planning, verification, and ranking-cost closure.  The source module is
/// never modified.  Cross-rank transport, card resources, and target ABI are
/// deliberately deferred to the executable-bundle coordinator.
mlir::FailureOr<std::vector<ScheduledRankCandidate>>
buildScheduledRankCandidateFrontier(
    mlir::ModuleOp sourceModule, const TensorProgramSchedulingConfig &config);

/// Recomputes the scalar scheduling cost from a complete instruction-level
/// rank artifact.  Callers use this after transformations such as
/// function-boundary bufferization and physical-memory replanning that may
/// change movement or issue counts.  Unknown or overflowed required metrics
/// fail closed instead of preserving a stale pre-transformation estimate.
mlir::FailureOr<int64_t>
estimateScheduledRankProgramTimePs(mlir::ModuleOp module);

} // namespace wafer

#endif // WAFER_TRANSFORMS_TENSORPROGRAMSCHEDULING_H
