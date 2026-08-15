//===- StructuredDAGEdgeDemandPlan.h - Exact logical edge demand -*- C++ -*-===//

#pragma once

#include "StructuredDAGCandidateSchedule.h"

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
struct StructuredDAGEdgeProducerShardOwnership {
  TileId tile{0};
  mlir::presburger::PresburgerSet logicalDomain;
};

/// Exact logical demand for one destination Tile of one current-SSA data-input
/// edge. `consumerDomain` and `producerDemand` are exact logical index sets.
/// The latter is the IndexRelation image of the former and may be rectangular,
/// strided, or multi-piece. Producer ownership is retained separately so later
/// representation and movement stages can intersect the exact set without
/// reconstructing structured indexing semantics.
struct StructuredDAGEdgeDemand {
  StructuredDAGEdgeID edge = 0;
  TileId destinationTile{0};
  mlir::presburger::PresburgerSet consumerDomain;
  mlir::presburger::PresburgerSet producerDemand;
  llvm::SmallVector<StructuredDAGEdgeProducerShardOwnership, 4>
      producerShardOwnership;
};

/// Query-local, layout-independent logical demand for one placement.
/// DPS init edges and support-chain edges are absent because their typed
/// consumer/support lowering contracts own those dependencies.
struct StructuredDAGEdgeDemandPlan {
  llvm::SmallVector<StructuredDAGEdgeDemand, 0> demands;
};

/// Query-local logical demand planner. Placement exploration reuses one
/// instance so the placement-independent SSA/index-relation proof for each DAG
/// edge is derived once while every consumer domain is imaged exactly.
class StructuredDAGEdgeDemandPlanner {
public:
  explicit StructuredDAGEdgeDemandPlanner(const StructuredDAGAnalysis &dag);
  ~StructuredDAGEdgeDemandPlanner();
  StructuredDAGEdgeDemandPlanner(StructuredDAGEdgeDemandPlanner &&) noexcept;
  StructuredDAGEdgeDemandPlanner &operator=(StructuredDAGEdgeDemandPlanner &&) noexcept;
  StructuredDAGEdgeDemandPlanner(const StructuredDAGEdgeDemandPlanner &) = delete;
  StructuredDAGEdgeDemandPlanner &
  operator=(const StructuredDAGEdgeDemandPlanner &) = delete;

  mlir::FailureOr<StructuredDAGEdgeDemandPlan>
  derive(StructuredDAGEdgeID edge, const StructuredDAGNodePlacement &producerPlacement,
         const StructuredDAGNodePlacement &consumerPlacement,
         std::string *failureReason = nullptr);

  mlir::FailureOr<StructuredDAGEdgeDemandPlan>
  derive(llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
         std::string *failureReason = nullptr);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

mlir::FailureOr<StructuredDAGEdgeDemandPlan>
deriveStructuredDAGEdgeDemandPlan(const StructuredDAGAnalysis &dag, StructuredDAGEdgeID edge,
                             const StructuredDAGNodePlacement &producerPlacement,
                             const StructuredDAGNodePlacement &consumerPlacement,
                             std::string *failureReason = nullptr);

mlir::FailureOr<StructuredDAGEdgeDemandPlan> deriveStructuredDAGEdgeDemandPlan(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
