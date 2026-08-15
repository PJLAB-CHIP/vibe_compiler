//===- WholeDAGEdgeStrategyPlan.h - Exact edge actions --------*- C++ -*-===//

#pragma once

#include "WholeDAGEdgeDemandPlan.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <string>

namespace wafer::compiler::detail {

/// Current canonical fixed-representation lowering of a demand plan.  This is
/// a compatibility carrier consumed by existing CardProgram materialization;
/// logical demand remains owned by `WholeDAGEdgeDemandPlan`.
struct WholeDAGEdgeStrategyPlan {
  // A strategy contains several inline domains/fragments and is intentionally
  // heap-backed here. Placement search keeps many plans alive; an inline array
  // would turn the search queue into a multi-megabyte C++ stack frame.
  llvm::SmallVector<SpatialEdgeStrategy, 0> strategies;
  uint64_t totalPeerBytes = 0;
};

/// Lowers a layout-independent demand plan through the current canonical
/// Tensor-coordinate representation.  Non-rectangular demand remains legal at
/// the logical stage and fails only here while the target lacks a matching
/// descriptor.
mlir::FailureOr<WholeDAGEdgeStrategyPlan>
lowerWholeDAGEdgeDemandPlanToCanonicalStrategies(
    const CardDAGAnalysis &dag, const WholeDAGEdgeDemandPlan &demandPlan,
    std::string *failureReason = nullptr);

/// Query-local compatibility planner for current CardProgram consumers.  It
/// derives a logical demand plan first and then invokes the explicit canonical
/// lowering above; it never reconstructs the relation from layout or action.
class WholeDAGEdgeStrategyPlanner {
public:
  explicit WholeDAGEdgeStrategyPlanner(const CardDAGAnalysis &dag);
  ~WholeDAGEdgeStrategyPlanner();
  WholeDAGEdgeStrategyPlanner(WholeDAGEdgeStrategyPlanner &&) noexcept;
  WholeDAGEdgeStrategyPlanner &
  operator=(WholeDAGEdgeStrategyPlanner &&) noexcept;
  WholeDAGEdgeStrategyPlanner(const WholeDAGEdgeStrategyPlanner &) = delete;
  WholeDAGEdgeStrategyPlanner &
  operator=(const WholeDAGEdgeStrategyPlanner &) = delete;

  mlir::FailureOr<WholeDAGEdgeStrategyPlan>
  derive(CardDAGEdgeID edge, const WholeDAGNodePlacement &producerPlacement,
         const WholeDAGNodePlacement &consumerPlacement,
         std::string *failureReason = nullptr);

  /// Assemble one complete placement while reusing each edge's exact
  /// placement-independent relation proof across all queried placements.
  mlir::FailureOr<WholeDAGEdgeStrategyPlan>
  derive(llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
         std::string *failureReason = nullptr);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/// Counts actual remote fragments in the sole edge-strategy carrier.
uint64_t countPeerFragments(const WholeDAGEdgeStrategyPlan &plan);

/// Compatibility composition for one closed producer/consumer edge when a
/// current CardProgram consumer requires the fixed fragment representation.
/// Placement exploration uses `WholeDAGEdgeDemandPlanner` instead so target
/// representation limits cannot make a logical placement illegal.
mlir::FailureOr<WholeDAGEdgeStrategyPlan>
deriveWholeDAGEdgeStrategyPlan(const CardDAGAnalysis &dag, CardDAGEdgeID edge,
                               const WholeDAGNodePlacement &producerPlacement,
                               const WholeDAGNodePlacement &consumerPlacement,
                               std::string *failureReason = nullptr);

/// Compatibility composition of exact logical demand derivation and the
/// current canonical fixed-representation lowering. Exact one-to-many
/// relations such as reductions remain legal; a target representation failure
/// here does not make the underlying logical demand illegal.
mlir::FailureOr<WholeDAGEdgeStrategyPlan> deriveWholeDAGEdgeStrategyPlan(
    const CardDAGAnalysis &dag,
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
