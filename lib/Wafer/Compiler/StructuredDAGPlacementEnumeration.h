//===- StructuredDAGPlacementEnumeration.h - Joint node placement -------*- C++
//-*-===//

#pragma once

#include "StructuredDAGEdgeStrategyPlan.h"

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/IR/Target/TargetTopology.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace wafer::compiler::detail {
/// One static spatial (parallel-iterator) axis of a structured node: the
/// iterator dimension, its mapped result dimension, its static extent and
/// the unit partition factors the canonical coordinate starts from.
struct StaticSpatialAxis {
  unsigned iteratorDimension = 0;
  unsigned resultDimension = 0;
  uint64_t extent = 0;
  llvm::SmallVector<uint32_t, 4> basePartitionFactors;
};

/// The typed spatial-axis facts of one node, sorted by descending result
/// extent. Empty when the node has no parallel result axis; an unpartitioned
/// canonical coordinate (one Tile, unit factors, full result domain) is the
/// caller's degradation for that case, never a fabricated shard axis.
std::optional<llvm::SmallVector<StaticSpatialAxis, 4>>
getNodeSpatialAxes(const StructuredDAGNode &node);



/// Physical placement of one observable function result.  Every structured
/// root of the result has the same node placement; the query-local record can
/// therefore be consumed directly by TileMapping without recovering a
/// role from operation names or result shapes.
struct StructuredDAGObservablePlacement {
  uint32_t outputIndex = 0;
  /// Missing for an unpartitioned output, which is wholly owned by its one
  /// participating Tile. No sentinel or fabricated axis is permitted.
  std::optional<unsigned> shardDimension;
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
  /// False when the canonical carrier could not express every edge of this
  /// candidate. The placement remains logically legal; actual materialization
  /// rejects the physical assignment at its own gate.
  bool edgeCarrierComplete = true;
  uint64_t topologyHopByteWork = 0;
  uint64_t topologyCompactnessWork = 0;
  uint32_t distinctTileGroupCount = 0;
  uint32_t parallelComponentCount = 1;
  uint32_t localEdgeCount = 0;
  uint32_t sameGroupRemapEdgeCount = 0;
  uint32_t partialOverlapEdgeCount = 0;
  uint32_t disjointEdgeCount = 0;
};

/// Typed legality of one evaluated placement trial. Only
/// ProvenLogicalInfeasible deletes the trial; UnsupportedSemanticRelation and
/// IndeterminateFailure stop the owning legalization path.
struct StructuredDAGPlacementLegality {
  analysis::ExactDemandStatus status = analysis::ExactDemandStatus::Satisfied;
  std::string detail;
};

/// Query instrumentation only. `expandedStates` counts retained partial
/// node-placement states expanded before any actual IR clone; rejected
/// transitions never reach TileMapping. `edgeCarrierIncompleteTransitions`
/// counts complete trials whose canonical carrier could not express every
/// edge; those are physical-assignment gaps, not placement rejections.
struct StructuredDAGPlacementEnumerationStatistics {
  uint64_t placementGroupCount = 0;
  uint64_t expandedStates = 0;
  uint64_t rejectedTransitions = 0;
  uint64_t resourceRenamingEquivalentStates = 0;
  uint64_t edgeCarrierIncompleteTransitions = 0;
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

/// Policy-free per-node placement-option derivation from structured
/// iterator semantics and the verified topology. Every node owns every legal
/// static iterator-axis/connected-rectangle option. The deterministic
/// baseline consumes the same option domain as the search; this entry
/// carries no search state, ordering or proposal mechanics.
mlir::FailureOr<
    llvm::SmallVector<llvm::SmallVector<StructuredDAGNodePlacement, 32>, 16>>
deriveStructuredDAGNodePlacementOptions(const StructuredDAGAnalysis &dag,
                                        const TargetTopology &topology,
                                        CardId cardId,
                                        std::string *failureReason = nullptr);

mlir::FailureOr<StructuredDAGPlacementSearchDomain>
deriveStructuredDAGPlacementSearchDomain(const StructuredDAGAnalysis &dag,
                                    const TargetTopology &topology,
                                    CardId cardId,
                                    std::string *failureReason = nullptr);

/// Policy-free closure of one complete node-placement assignment: the exact
/// demand gate over every edge and the canonical edge strategy carrier, plus
/// the observable output placements. It builds no schedule, movement,
/// residency, resource calendar, or candidate metrics. `legality` receives
/// the typed verdict on failure; carrier incompleteness never fails the
/// closure and never changes the logical verdict.
struct StructuredDAGPlacementClosure {
  llvm::SmallVector<StructuredDAGNodePlacement, 16> nodePlacements;
  llvm::SmallVector<StructuredDAGObservablePlacement, 4> outputPlacements;
  StructuredDAGEdgeStrategyPlan edgePlan;
  bool edgeCarrierComplete = true;
};

mlir::FailureOr<StructuredDAGPlacementClosure>
buildStructuredDAGPlacementClosure(
    const StructuredDAGAnalysis &dag, const TargetTopology &topology,
    CardId cardId, llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
    analysis::IREpoch epoch, std::string *failureReason = nullptr,
    StructuredDAGPlacementLegality *legality = nullptr);

/// Query-local evaluator reused by factorized search. It memoizes only the
/// placement-independent exact SSA/index relation of each edge; every concrete
/// fragment, route, residency and resource calendar is rebuilt for the
/// requested complete coordinate assignment.
class StructuredDAGPlacementEvaluator {
public:
  StructuredDAGPlacementEvaluator(
      const StructuredDAGAnalysis &dag, const TargetTopology &topology,
      CardId cardId,
      analysis::IREpoch epoch = analysis::IREpoch::mint());
  ~StructuredDAGPlacementEvaluator();
  StructuredDAGPlacementEvaluator(StructuredDAGPlacementEvaluator &&) noexcept;
  StructuredDAGPlacementEvaluator &operator=(StructuredDAGPlacementEvaluator &&) noexcept;
  StructuredDAGPlacementEvaluator(const StructuredDAGPlacementEvaluator &) = delete;
  StructuredDAGPlacementEvaluator &
  operator=(const StructuredDAGPlacementEvaluator &) = delete;

  /// On failure `legality` receives the typed verdict:
  /// ProvenLogicalInfeasible only for a proven partition/relation/ownership
  /// contradiction of the requested trial; UnsupportedSemanticRelation and
  /// IndeterminateFailure stop the owning legalization path. The canonical
  /// carrier failure never fails evaluation.
  mlir::FailureOr<StructuredDAGPlacementCandidate>
  evaluate(llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
           std::string *failureReason = nullptr,
           StructuredDAGPlacementLegality *legality = nullptr);

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
    std::string *failureReason = nullptr,
    StructuredDAGPlacementLegality *legality = nullptr);

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
    std::string *failureReason = nullptr,
    StructuredDAGPlacementLegality *legality = nullptr);

} // namespace wafer::compiler::detail
