//===- ExactDemand.h - Policy-free logical placement demand -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_PHYSICALDATAFLOW_EXACTDEMAND_H
#define WAFER_ANALYSIS_PHYSICALDATAFLOW_EXACTDEMAND_H

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Target/TopologyIds.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class Operation;
} // namespace mlir

namespace wafer::analysis {

/// Generation token for one immutable IR snapshot. Query-local analysis
/// caches are keyed by it, and a query constructed for one epoch rejects
/// trials from any other epoch as an indeterminate compiler failure. IR
/// mutation owners call advance() around rewrites so derived analysis can
/// never silently cross a mutation boundary; queries and analyses never
/// mutate IR and must not advance.
class IREpoch {
public:
  IREpoch() = default;

  static IREpoch current();
  static void advance();

  uint64_t getGeneration() const { return generation; }
  bool isValid() const { return generation != 0; }

  friend bool operator==(IREpoch lhs, IREpoch rhs) {
    return lhs.generation == rhs.generation;
  }
  friend bool operator!=(IREpoch lhs, IREpoch rhs) {
    return !(lhs == rhs);
  }

private:
  explicit constexpr IREpoch(uint64_t generation) : generation(generation) {}
  uint64_t generation = 0;
};

/// Ownership role of one logical shard owner.
enum class TileRole : uint8_t {
  /// Unique partition: the owners tile the node's complete logical domain.
  UniquePartition,
  /// Explicit replication: several owners legitimately own the same logical
  /// elements. The query keeps every eligible owner and never picks a source;
  /// coverage is satisfied when the demand is covered by the owner union.
  ExplicitReplication,
  /// Partial-reduction contribution: every owner is a required contribution
  /// to the merged value, never an interchangeable replica. All intersections
  /// are preserved and handed to the selected merge semantics.
  PartialReductionContribution,
};

/// Per-Tile ownership of one logical node shard. `ownedDomain` is exact and
/// expressed in the node result space; it carries no layout, encoding,
/// bytes, movement, or transport decision. A missing domain is a malformed
/// assignment for the consuming query.
struct LogicalTileBinding {
  TileId tile{0};
  std::optional<mlir::presburger::PresburgerSet> ownedDomain;
  TileRole role = TileRole::UniquePartition;
};

/// Closed logical trial of one structured DAG node: the complete consumer
/// execution domain (iteration space) and every owner's exact result-space
/// domain. `node` is the DAG node identity of the consuming query.
struct LogicalNodeTrial {
  uint32_t node = 0;
  std::optional<mlir::presburger::PresburgerSet> completeIterationDomain;
  llvm::SmallVector<LogicalTileBinding, 16> bindings;
};

/// One closed logical shard trial covering every DAG node of the consuming
/// query. The query never recovers these facts from result axes, shard
/// dimensions, participant counts, or balanced rectangles; the trial is the
/// only placement input it observes.
struct LogicalShardTrial {
  IREpoch epoch;
  llvm::SmallVector<LogicalNodeTrial, 16> nodes;
};

/// Structural role of one dependency edge.
enum class DemandEdgeKind : uint8_t {
  /// Ordinary DPS data input.
  DataInput,
  /// DPS init operand. An explicit structured init producer is an
  /// independent DAG root whose init/update dependency carries an exact
  /// demand; "the consumer typed lowering handles init" never removes it.
  InitInput,
};

/// Typed dependency descriptor: one producer result feeding one consumer
/// operand. `supportChain` holds pure support operations (view/reshape/
/// slice/pad/insert_slice/concat-piece/transpose/cast) between the producer
/// result and the consumer operand, in producer-to-consumer order. Each
/// data-carrying predecessor of a multi-operand support operation forms its
/// own dependency; a unary chain is never assumed.
struct DemandDependency {
  uint32_t producerNode = 0;
  uint32_t producerResult = 0;
  uint32_t consumerNode = 0;
  uint32_t consumerOperand = 0;
  DemandEdgeKind kind = DemandEdgeKind::DataInput;
  llvm::SmallVector<mlir::Operation *, 4> supportChain;
};

/// Verdict of one exact-demand query. Only ProvenLogicalInfeasible may delete
/// a placement trial. UnsupportedSemanticRelation and IndeterminateFailure
/// stop the owning legalization path as typed failures instead of being
/// rewritten into placement legality or a no-good cache.
enum class ExactDemandStatus : uint8_t {
  /// Exact relation image and complete ownership coverage exist; the
  /// uncovered witness is empty.
  Satisfied,
  /// A well-formed trial has a provable partition/relation/ownership
  /// contradiction; `uncoveredWitness` is the direct non-empty witness.
  ProvenLogicalInfeasible,
  /// Verified source semantics are outside the typed relation contract. This
  /// applies to the semantics, not to one physical placement.
  UnsupportedSemanticRelation,
  /// Presburger budget exhaustion, internal error, stale IR epoch, missing
  /// protocol, or unclassified failure.
  IndeterminateFailure,
};

/// Exact intersection of producer demand with one owner's domain.
struct ExactOwnershipIntersection {
  TileId tile{0};
  std::optional<mlir::presburger::PresburgerSet> set;
};

/// Query-local verdict for one dependency. It contains no layout, encoding,
/// bytes, dense fragment, action, route, buffer, or schedule facts; those
/// belong to later representation and movement stages. `detail` is for
/// diagnostics only and is never a legality input. `mergeObligation` is set
/// when the covered demand spans partial-reduction contributions that must
/// all be consumed by the selected merge semantics.
struct ExactDemandResult {
  ExactDemandStatus status = ExactDemandStatus::IndeterminateFailure;
  uint32_t edge = 0;
  /// Typed dependency role recovered from the DPS contract (data input or
  /// init operand). Never recovered from operand order or op names.
  DemandEdgeKind dependencyKind = DemandEdgeKind::DataInput;
  std::optional<mlir::presburger::PresburgerSet> consumerIterationDomain;
  std::optional<mlir::presburger::PresburgerSet> producerDemand;
  llvm::SmallVector<ExactOwnershipIntersection, 8> ownershipIntersections;
  /// Present and non-empty only for ProvenLogicalInfeasible; absent
  /// otherwise.
  std::optional<mlir::presburger::PresburgerSet> uncoveredWitness;
  TileRole role = TileRole::UniquePartition;
  bool mergeObligation = false;
  std::string detail;
};

/// Only proven conclusions may be cached as placement legality.
inline bool isCacheableLegalityConclusion(ExactDemandStatus status) {
  return status == ExactDemandStatus::Satisfied ||
         status == ExactDemandStatus::ProvenLogicalInfeasible;
}

/// Translate a non-Exact relation failure into the demand verdict. Exact must
/// not be passed here: an exact relation is not a failure and the caller
/// continues the query instead. SoundBound cannot form an exact proof, and
/// Invalid/ResourceExhausted are machinery failures; all three stop the
/// owning legalization path as IndeterminateFailure. Only the semantic
/// Unsupported maps to UnsupportedSemanticRelation.
ExactDemandStatus mapIndexRelationStatus(IndexRelationStatus status);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_PHYSICALDATAFLOW_EXACTDEMAND_H
