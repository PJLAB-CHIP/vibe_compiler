//===- WholeDAGEdgeDemandPlan.h - Exact logical edge demand -*- C++ -*-===//

#pragma once

#include "WholeDAGCandidateSchedule.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <string>

namespace wafer::compiler::detail {

/// Layout-independent ownership of one producer logical shard. The domain is
/// expressed in producer tensor coordinates; it carries no byte address,
/// layout, transport, residency, or communication decision.
struct WholeDAGEdgeProducerShardOwnership {
  PhysicalTileId tile{0};
  mlir::presburger::PresburgerSet logicalDomain;
};

/// Exact logical demand for one destination Tile of one current-SSA data-input
/// edge. `consumerDomain` and `producerDemand` are exact logical index sets.
/// The latter is the IndexRelation image of the former and may be rectangular,
/// strided, or multi-piece. Producer ownership is retained separately so later
/// representation and movement stages can intersect the exact set without
/// reconstructing structured indexing semantics.
struct WholeDAGEdgeDemand {
  CardDAGEdgeID edge = 0;
  PhysicalTileId destinationTile{0};
  mlir::presburger::PresburgerSet consumerDomain;
  mlir::presburger::PresburgerSet producerDemand;
  llvm::SmallVector<WholeDAGEdgeProducerShardOwnership, 4>
      producerShardOwnership;
};

/// Query-local, layout-independent logical demand for one placement.
/// DPS init edges and support-chain edges are absent because their typed
/// consumer/support lowering contracts own those dependencies.
struct WholeDAGEdgeDemandPlan {
  llvm::SmallVector<WholeDAGEdgeDemand, 0> demands;
};

/// Query-local logical demand planner. Placement exploration reuses one
/// instance so the placement-independent SSA/index-relation proof for each DAG
/// edge is derived once while every consumer domain is imaged exactly.
class WholeDAGEdgeDemandPlanner {
public:
  explicit WholeDAGEdgeDemandPlanner(const CardDAGAnalysis &dag);
  ~WholeDAGEdgeDemandPlanner();
  WholeDAGEdgeDemandPlanner(WholeDAGEdgeDemandPlanner &&) noexcept;
  WholeDAGEdgeDemandPlanner &operator=(WholeDAGEdgeDemandPlanner &&) noexcept;
  WholeDAGEdgeDemandPlanner(const WholeDAGEdgeDemandPlanner &) = delete;
  WholeDAGEdgeDemandPlanner &
  operator=(const WholeDAGEdgeDemandPlanner &) = delete;

  mlir::FailureOr<WholeDAGEdgeDemandPlan>
  derive(CardDAGEdgeID edge, const WholeDAGNodePlacement &producerPlacement,
         const WholeDAGNodePlacement &consumerPlacement,
         std::string *failureReason = nullptr);

  mlir::FailureOr<WholeDAGEdgeDemandPlan>
  derive(llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
         std::string *failureReason = nullptr);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

mlir::FailureOr<WholeDAGEdgeDemandPlan>
deriveWholeDAGEdgeDemandPlan(const CardDAGAnalysis &dag, CardDAGEdgeID edge,
                             const WholeDAGNodePlacement &producerPlacement,
                             const WholeDAGNodePlacement &consumerPlacement,
                             std::string *failureReason = nullptr);

mlir::FailureOr<WholeDAGEdgeDemandPlan> deriveWholeDAGEdgeDemandPlan(
    const CardDAGAnalysis &dag,
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
