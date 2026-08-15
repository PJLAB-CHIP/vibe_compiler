//===- StructuredDAGPlacementEnumeration.h - Joint node placement -------*- C++
//-*-===//

#pragma once

#include "StructuredDAGEdgeStrategyPlan.h"

#include "Wafer/IR/Target/TargetTopology.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace wafer::compiler::detail {

/// Physical placement of one observable function result.  Every structured
/// root of the result has the same node placement; the query-local record can
/// therefore be consumed directly by TileMapping without recovering a
/// role from operation names or result shapes.
struct StructuredDAGObservablePlacement {
  uint32_t outputIndex = 0;
  unsigned shardDimension = 0;
  llvm::SmallVector<TileId, 16> tiles;
};

/// One structurally executable node-placement state.  Both schedule and peer
/// plan have already been derived from the current DAG.  This remains
/// query-local and is destroyed once the selected Card/Tile IR is materialized.
struct StructuredDAGPlacementCandidate {
  llvm::SmallVector<StructuredDAGNodePlacement, 16> nodePlacements;
  llvm::SmallVector<StructuredDAGObservablePlacement, 4> outputPlacements;
  StructuredDAGCandidateSchedule schedule;
  StructuredDAGEdgeStrategyPlan edgePlan;
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
/// transitions never reach TileMapping.
struct StructuredDAGPlacementEnumerationStatistics {
  uint64_t placementGroupCount = 0;
  uint64_t expandedStates = 0;
  uint64_t rejectedTransitions = 0;
  uint64_t resourceRenamingEquivalentStates = 0;
};

/// Factorized spatial coordinate domain for the common card search.
/// Each node owns every legal static iterator-axis/connected-rectangle option;
/// the Cartesian product is deliberately not materialized here.  The common
/// search changes one coordinate of a complete candidate at a time and sends
/// the resulting placement through the same exact edge/resource evaluation.
struct StructuredDAGPlacementSearchDomain {
  llvm::SmallVector<llvm::SmallVector<StructuredDAGNodePlacement, 32>, 16>
      nodeOptions;
  uint64_t placementGroupCount = 0;
};

mlir::FailureOr<StructuredDAGPlacementSearchDomain>
deriveStructuredDAGPlacementSearchDomain(const StructuredDAGAnalysis &dag,
                                    const TargetTopology &topology,
                                    CardId cardId,
                                    std::string *failureReason = nullptr);

/// Query-local evaluator reused by factorized search. It memoizes only the
/// placement-independent exact SSA/index relation of each edge; every concrete
/// fragment, route, residency and resource calendar is rebuilt for the
/// requested complete coordinate assignment.
class StructuredDAGPlacementEvaluator {
public:
  StructuredDAGPlacementEvaluator(const StructuredDAGAnalysis &dag,
                             const TargetTopology &topology,
                             CardId cardId);
  ~StructuredDAGPlacementEvaluator();
  StructuredDAGPlacementEvaluator(StructuredDAGPlacementEvaluator &&) noexcept;
  StructuredDAGPlacementEvaluator &operator=(StructuredDAGPlacementEvaluator &&) noexcept;
  StructuredDAGPlacementEvaluator(const StructuredDAGPlacementEvaluator &) = delete;
  StructuredDAGPlacementEvaluator &
  operator=(const StructuredDAGPlacementEvaluator &) = delete;

  mlir::FailureOr<StructuredDAGPlacementCandidate>
  evaluate(llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
           std::string *failureReason = nullptr);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/// Closes one complete coordinate assignment through exact observable
/// placement, edge fragments, routes, local residency and the common resource
/// calendar.  It is the sole bridge from the factorized spatial domain to a
/// materializable candidate.
mlir::FailureOr<StructuredDAGPlacementCandidate> evaluateStructuredDAGPlacement(
    const StructuredDAGAnalysis &dag, const TargetTopology &topology,
    CardId cardId, llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
    std::string *failureReason = nullptr);

/// Explores the finite per-node static shard dimensions and all connected
/// rectangular Tile groups exposed by the verified topology. Each node is
/// assigned independently, so a dependent DAG may form any number of
/// operator-pipeline stages rather than one source/destination cut.
/// Exact-local, same-group redistribution, genuinely partially overlapping
/// and disjoint successor placements are ordinary states in the same domain.
/// Every incoming edge is checked by the layout-independent exact demand
/// planner when its consumer transition closes. A legal non-rectangular demand
/// remains in the placement domain; current physical representation limits are
/// applied only when a complete candidate is lowered. Query work may be
/// reduced only by exact state equivalence or a component-wise dominance
/// proof; there is no candidate-count, topology-group, node-option, or
/// beam-width policy.
mlir::FailureOr<llvm::SmallVector<StructuredDAGPlacementCandidate, 12>>
enumerateStructuredDAGPlacements(
    const StructuredDAGAnalysis &dag, const TargetTopology &topology,
    CardId cardId,
    StructuredDAGPlacementEnumerationStatistics *statistics = nullptr,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
