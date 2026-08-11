//===- WholeDAGEdgeStrategyPlan.h - Exact edge actions --------*- C++ -*-===//

#pragma once

#include "WholeDAGCandidateSchedule.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <string>

namespace wafer::compiler::detail {

/// Query-local exact action set for one whole-DAG placement.  This owns the
/// same `SpatialEdgeStrategy` objects consumed by CardProgram materialization;
/// there is deliberately no parallel demand/local/transfer schema to join or
/// repair later.  Each current-SSA data-input edge has one strategy per
/// destination Tile; DPS init-operand edges carry no strategy because init
/// state stays governed by the consumer's typed lowering.
struct WholeDAGEdgeStrategyPlan {
  // A strategy contains several inline domains/fragments and is intentionally
  // heap-backed here. Placement frontiers embed many plans; an inline array
  // would turn the bounded beam into a multi-megabyte C++ stack frame.
  llvm::SmallVector<SpatialEdgeStrategy, 0> strategies;
  uint64_t totalPeerBytes = 0;
};

/// Query-local exact edge planner.  Placement exploration reuses one instance
/// so the placement-independent SSA/index-relation proof for each DAG edge is
/// derived once while every concrete Tile-group fragment plan remains exact.
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
  derive(CardDAGEdgeID edge,
         const WholeDAGNodePlacement &producerPlacement,
         const WholeDAGNodePlacement &consumerPlacement,
         std::string *failureReason = nullptr);

  /// Checks the exact concrete fragment coverage and target payload bounds
  /// without constructing a retained strategy plan. Placement exploration
  /// uses this for transitions; complete candidates call `derive` once when
  /// they need the materialization carrier.
  mlir::LogicalResult verify(
      CardDAGEdgeID edge,
      const WholeDAGNodePlacement &producerPlacement,
      const WholeDAGNodePlacement &consumerPlacement,
      std::string *failureReason = nullptr);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/// Counts actual remote fragments in the sole edge-strategy carrier.
uint64_t countPeerFragments(const WholeDAGEdgeStrategyPlan &plan);

/// Verifies the placement-independent semantic precondition for a non-local
/// transition of one current DAG edge.  The result depends only on current SSA
/// and structured indexing relations, never on physical Tile identity, and is
/// therefore safe for query-local memoization before fragment construction.
mlir::LogicalResult verifyWholeDAGEdgeNonLocalRelation(
    const CardDAGAnalysis &dag, CardDAGEdgeID edge,
    std::string *failureReason = nullptr);

/// Derives the exact movement for one closed producer/consumer transition.
/// Placement search uses this query as soon as the consumer is placed, so an
/// unsupported non-local relation cannot occupy a bounded partial-state beam.
/// The result has the same canonical fragment contract as the whole-DAG plan.
mlir::FailureOr<WholeDAGEdgeStrategyPlan>
deriveWholeDAGEdgeStrategyPlan(const CardDAGAnalysis &dag, CardDAGEdgeID edge,
                               const WholeDAGNodePlacement &producerPlacement,
                               const WholeDAGNodePlacement &consumerPlacement,
                               std::string *failureReason = nullptr);

/// Derives exact dependent-edge movement from the current SSA edge, static
/// structured indexing maps, and per-node physical placement.  Unsupported,
/// non-rectangular, dynamic, reduction, or non-byte-addressable
/// relations fail closed before an actual clone is admitted.
mlir::FailureOr<WholeDAGEdgeStrategyPlan> deriveWholeDAGEdgeStrategyPlan(
    const CardDAGAnalysis &dag,
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
