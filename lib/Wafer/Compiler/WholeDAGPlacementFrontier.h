//===- WholeDAGPlacementFrontier.h - Joint node placement -------*- C++ -*-===//

#pragma once

#include "WholeDAGEdgeStrategyPlan.h"

#include "Wafer/IR/Target/PhysicalTopology.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace wafer::compiler::detail {

/// Physical placement of one observable function result.  Every structured
/// root of the result has the same node placement; the query-local record can
/// therefore be consumed directly by CardSpatialMapping without recovering a
/// role from operation names or result shapes.
struct WholeDAGObservablePlacement {
  uint32_t outputIndex = 0;
  unsigned shardDimension = 0;
  llvm::SmallVector<PhysicalTileId, 16> tiles;
};

/// One structurally executable node-placement state.  Both schedule and peer
/// plan have already been derived from the current DAG.  This remains
/// query-local and is destroyed once the selected Card/Tile IR is materialized.
struct WholeDAGPlacementCandidate {
  llvm::SmallVector<WholeDAGNodePlacement, 16> nodePlacements;
  llvm::SmallVector<WholeDAGObservablePlacement, 4> outputPlacements;
  WholeDAGCandidateSchedule schedule;
  WholeDAGEdgeStrategyPlan edgePlan;
  uint64_t topologyHopByteWork = 0;
  uint64_t topologyCompactnessWork = 0;
  uint32_t distinctTileGroupCount = 0;
  uint32_t parallelComponentCount = 1;
  uint32_t localEdgeCount = 0;
  uint32_t sameGroupRemapEdgeCount = 0;
  uint32_t partialOverlapEdgeCount = 0;
  uint32_t disjointEdgeCount = 0;
};

/// Query instrumentation only. `expandedStates` counts retained partial
/// node-placement states expanded before any actual IR clone; rejected
/// transitions never reach CardSpatialMapping.
struct WholeDAGPlacementFrontierStatistics {
  uint64_t placementGroupCount = 0;
  uint64_t expandedStates = 0;
  uint64_t rejectedTransitions = 0;
  uint64_t resourceRenamingEquivalentStates = 0;
};

/// Factorized spatial coordinate domain for the common whole-card search.
/// Each node owns every legal static iterator-axis/connected-rectangle option;
/// the Cartesian product is deliberately not materialized here.  The common
/// search changes one coordinate of a complete candidate at a time and sends
/// the resulting placement through the same exact edge/resource evaluation.
struct WholeDAGPlacementSearchDomain {
  llvm::SmallVector<llvm::SmallVector<WholeDAGNodePlacement, 32>, 16>
      nodeOptions;
  uint64_t placementGroupCount = 0;
};

mlir::FailureOr<WholeDAGPlacementSearchDomain>
deriveWholeDAGPlacementSearchDomain(
    const CardDAGAnalysis &dag, const PhysicalTopology &topology,
    PhysicalCardId cardId, std::string *failureReason = nullptr);

/// Query-local evaluator reused by factorized search. It memoizes only the
/// placement-independent exact SSA/index relation of each edge; every concrete
/// fragment, route, residency and resource calendar is rebuilt for the
/// requested complete coordinate assignment.
class WholeDAGPlacementEvaluator {
public:
  WholeDAGPlacementEvaluator(const CardDAGAnalysis &dag,
                             const PhysicalTopology &topology,
                             PhysicalCardId cardId);
  ~WholeDAGPlacementEvaluator();
  WholeDAGPlacementEvaluator(WholeDAGPlacementEvaluator &&) noexcept;
  WholeDAGPlacementEvaluator &
  operator=(WholeDAGPlacementEvaluator &&) noexcept;
  WholeDAGPlacementEvaluator(const WholeDAGPlacementEvaluator &) = delete;
  WholeDAGPlacementEvaluator &
  operator=(const WholeDAGPlacementEvaluator &) = delete;

  mlir::FailureOr<WholeDAGPlacementCandidate>
  evaluate(llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
           std::string *failureReason = nullptr);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/// Closes one complete coordinate assignment through exact observable
/// placement, edge fragments, routes, local residency and the common resource
/// calendar.  It is the sole bridge from the factorized spatial domain to a
/// materializable candidate.
mlir::FailureOr<WholeDAGPlacementCandidate>
evaluateWholeDAGPlacement(
    const CardDAGAnalysis &dag, const PhysicalTopology &topology,
    PhysicalCardId cardId,
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
    std::string *failureReason = nullptr);

/// Explores the finite per-node static shard dimensions and all connected
/// rectangular Tile groups exposed by the verified topology. Each node is
/// assigned independently, so a dependent DAG may form any number of
/// operator-pipeline stages rather than one source/destination cut.
/// Exact-local, same-group redistribution, genuinely partially overlapping
/// and disjoint successor placements are ordinary states in the same domain.
/// Every incoming edge is checked by the exact edge-strategy planner when its
/// consumer transition closes, so unsupported reduction or non-rectangular
/// relations reject only that transition. Query work may be reduced only by
/// exact state equivalence or a component-wise dominance proof; there is no
/// candidate-count, topology-group, node-option, or beam-width policy.
mlir::FailureOr<llvm::SmallVector<WholeDAGPlacementCandidate, 12>>
deriveWholeDAGPlacementFrontier(
    const CardDAGAnalysis &dag, const PhysicalTopology &topology,
    PhysicalCardId cardId,
    WholeDAGPlacementFrontierStatistics *statistics = nullptr,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
