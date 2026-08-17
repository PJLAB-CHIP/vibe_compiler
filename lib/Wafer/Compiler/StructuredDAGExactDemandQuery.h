//===- StructuredDAGExactDemandQuery.h - Typed logical demand -*- C++ -*-===//

#pragma once

#include "StructuredDAGSchedule.h"

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Target/TopologyIds.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <string>

namespace mlir {
class RankedTensorType;
} // namespace mlir

namespace wafer::compiler::detail {

struct StructuredDAGNodePlacement;

/// Typed, policy-free legality of one closed logical shard trial. The verdict
/// is the four-state `analysis::ExactDemandResult`; `FailureOr + string`
/// never appears on this boundary. Only ProvenLogicalInfeasible may delete a
/// placement trial; UnsupportedSemanticRelation and IndeterminateFailure stop
/// the owning legalization path as typed failures. The query reads the
/// current IR and the trial only — never a candidate schedule — and applies
/// no mutation. A query constructed for one IR epoch rejects trials from any
/// other epoch.
class StructuredDAGExactDemandQuery {
public:
  StructuredDAGExactDemandQuery(const StructuredDAGAnalysis &dag,
                                analysis::IREpoch epoch);
  ~StructuredDAGExactDemandQuery();
  StructuredDAGExactDemandQuery(StructuredDAGExactDemandQuery &&) noexcept;
  StructuredDAGExactDemandQuery &
  operator=(StructuredDAGExactDemandQuery &&) noexcept;
  StructuredDAGExactDemandQuery(const StructuredDAGExactDemandQuery &) = delete;
  StructuredDAGExactDemandQuery &
  operator=(const StructuredDAGExactDemandQuery &) = delete;

  analysis::IREpoch getEpoch() const;

  /// Exact demand of one dependency edge for one closed trial. Results are
  /// deterministic in the DAG and the typed trial content: they never depend
  /// on pointer values, hash order, or Tile enumeration order.
  analysis::ExactDemandResult query(StructuredDAGEdgeID edge,
                                    const analysis::LogicalShardTrial &trial);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/// Balanced single-axis result-space ownership of one producer result of
/// the current placement domain. The balanced rectangle is a property of the
/// current placement domain, not a recovery performed by the query; the full
/// spatial domain replaces it with Q50.B. A result whose rank cannot express
/// the shard axis is owned by every Tile of the group as an explicit
/// replica. An empty group or a non-static type reports failure.
mlir::FailureOr<llvm::SmallVector<analysis::LogicalTileBinding, 16>>
buildBalancedOwnership(mlir::RankedTensorType type, unsigned shardDimension,
                       llvm::ArrayRef<TileId> tiles, uint32_t resultIndex,
                       std::string *failureReason = nullptr);

/// Two-node closed trial for one dependency edge: the exact-demand query only
/// observes the edge endpoints. Ownership domains come from the same balanced
/// placement domain as the full trial builder.
mlir::FailureOr<analysis::LogicalShardTrial> buildEdgeShardTrial(
    const StructuredDAGAnalysis &dag,
    const StructuredDAGNodePlacement &producerPlacement,
    const StructuredDAGNodePlacement &consumerPlacement,
    analysis::IREpoch epoch, std::string *failureReason = nullptr);

/// Production adapter: materializes one closed trial from the current
/// placement domain (balanced single-axis result shards over
/// `shardDimension` and the Tile group, complete iteration domains from
/// static loop ranges, unique-partition ownership). The balanced rectangle is
/// a property of the current placement domain, not a recovery performed by
/// the query; the full spatial domain replaces it with Q50.B. Malformed
/// placements (missing/duplicate node, empty group, Tile outside its group,
/// shard dimension outside the result, dynamic or multi-result node) report
/// failure with a reason and never form a trial.
mlir::FailureOr<analysis::LogicalShardTrial> buildLogicalShardTrial(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
    analysis::IREpoch epoch, std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
