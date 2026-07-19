//===- ScheduledRankFinalization.h - Final rank candidates ------*- C++ -*-===//

#ifndef WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H
#define WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H

#include "Wafer/Transforms/TensorProgramScheduling.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

/// A rank candidate after function-boundary bufferization, physical-memory
/// replanning, and cost recomputation.  Failed alternatives are absent from
/// the finalized frontier; the frontier itself fails only when none survive.
struct FinalizedRankCandidate {
  FinalizedRankCandidate(mlir::OwningOpRef<mlir::ModuleOp> module,
                         int64_t estimatedTimePs, int64_t discoveryOrder)
      : module(std::move(module)), estimatedTimePs(estimatedTimePs),
        discoveryOrder(discoveryOrder) {}

  FinalizedRankCandidate(FinalizedRankCandidate &&) = default;
  FinalizedRankCandidate &operator=(FinalizedRankCandidate &&) = default;
  FinalizedRankCandidate(const FinalizedRankCandidate &) = delete;
  FinalizedRankCandidate &operator=(const FinalizedRankCandidate &) = delete;

  mlir::OwningOpRef<mlir::ModuleOp> module;
  int64_t estimatedTimePs;
  int64_t discoveryOrder;
};

mlir::FailureOr<std::vector<FinalizedRankCandidate>>
finalizeScheduledRankCandidateFrontier(
    std::vector<wafer::ScheduledRankCandidate> frontier);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H
