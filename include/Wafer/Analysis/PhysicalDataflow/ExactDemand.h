//===- ExactDemand.h - Policy-free logical placement demand -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_PHYSICALDATAFLOW_EXACTDEMAND_H
#define WAFER_ANALYSIS_PHYSICALDATAFLOW_EXACTDEMAND_H

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Target/TopologyIds.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace wafer::analysis {

/// Identity token for one immutable IR borrow, minted by the borrow owner
/// (the compilation entry) and shared by every query and trial of that
/// borrow. The token participates in no cache key and no legality decision;
/// a query rejects trials minted for another borrow, but IR-mutation
/// invalidation is the query's own structural snapshot of the borrowed
/// function, never this token. No process-global mutable state may enter
/// compiler semantics.
class IREpoch {
public:
  IREpoch() = default;

  /// Mint one fresh token. Uniqueness is the only guarantee; the token is
  /// never an invalidation mechanism.
  static IREpoch mint();

  bool isValid() const { return token != nullptr; }

  friend bool operator==(const IREpoch &lhs, const IREpoch &rhs) {
    return lhs.token == rhs.token;
  }
  friend bool operator!=(const IREpoch &lhs, const IREpoch &rhs) {
    return !(lhs == rhs);
  }

private:
  struct Token {};
  explicit IREpoch(std::shared_ptr<const Token> token)
      : token(std::move(token)) {}
  std::shared_ptr<const Token> token;
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

/// Per-Tile ownership of one producer result shard. `resultIndex` binds the
/// ownership to a concrete producer result, so a multi-result node carries
/// independent domains and roles per result. `ownedDomain` is exact and
/// expressed in the result space of `resultIndex`; it carries no layout,
/// encoding, bytes, movement, or transport decision. A missing domain is a
/// malformed assignment for the consuming query.
struct LogicalTileBinding {
  TileId tile{0};
  uint32_t resultIndex = 0;
  std::optional<mlir::presburger::PresburgerSet> ownedDomain;
  TileRole role = TileRole::UniquePartition;
};

/// Per-Tile execution domain of one consumer shard, expressed in the node's
/// iteration space. Every shard must be exact and all-and-only tile the
/// complete iteration domain.
struct LogicalExecutionShard {
  TileId tile{0};
  std::optional<mlir::presburger::PresburgerSet> executionDomain;
};

/// Closed logical trial of one structured DAG node: the complete consumer
/// execution domain (iteration space), per-Tile consumer execution shards,
/// and, per producer result, every owner's exact result-space domain.
/// `node` is the DAG node identity of the consuming query. `executionShards`
/// is optional for whole-edge-only proofs; the production adapter populates
/// it and the canonical carrier requires it. Adapter construction failures,
/// including Presburger budget exhaustion, remain typed machinery failures
/// and never masquerade as the optional whole-edge-only form.
struct LogicalNodeTrial {
  uint32_t node = 0;
  std::optional<mlir::presburger::PresburgerSet> completeIterationDomain;
  llvm::SmallVector<LogicalExecutionShard, 16> executionShards;
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
/// operand. `producerToConsumerChain` holds pure tensor transforms (view/reshape/
/// slice/pad/insert_slice/concat-piece/transpose/cast) between the producer
/// result and the consumer operand, in producer-to-consumer order. Each
/// data-carrying predecessor of a multi-operand tensor transform forms its
/// own dependency; a unary chain is never assumed.
struct DemandDependency {
  uint32_t producerNode = 0;
  uint32_t producerResult = 0;
  uint32_t consumerNode = 0;
  uint32_t consumerOperand = 0;
  DemandEdgeKind kind = DemandEdgeKind::DataInput;
  llvm::SmallVector<mlir::Operation *, 4> producerToConsumerChain;
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

/// Per-destination facts of one consumer shard: the shard's exact execution
/// domain, the exact producer demand imaged from it, the per-owner
/// intersections, and the per-shard uncovered witness.
struct ExactDestinationDemand {
  TileId destinationTile{0};
  std::optional<mlir::presburger::PresburgerSet> consumerExecutionDomain;
  std::optional<mlir::presburger::PresburgerSet> producerDemand;
  llvm::SmallVector<ExactOwnershipIntersection, 8> ownershipIntersections;
  /// Present and non-empty only when this shard's demand has an uncovered
  /// portion; absent otherwise.
  std::optional<mlir::presburger::PresburgerSet> uncoveredWitness;
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
  /// Typed dependency identity recovered from the DPS contract (producer
  /// result and consumer operand). Never recovered from operand order or op
  /// names.
  uint32_t producerResult = 0;
  uint32_t consumerOperand = 0;
  /// Typed dependency role recovered from the DPS contract (data input or
  /// init operand). Never recovered from operand order or op names.
  DemandEdgeKind dependencyKind = DemandEdgeKind::DataInput;
  std::optional<mlir::presburger::PresburgerSet> consumerIterationDomain;
  std::optional<mlir::presburger::PresburgerSet> producerDemand;
  /// Exact per-owner intersections, ordered by Tile id: the outcome never
  /// depends on the caller's binding enumeration order.
  llvm::SmallVector<ExactOwnershipIntersection, 8> ownershipIntersections;
  /// Per-destination facts for every consumer execution shard of the trial,
  /// ordered by Tile id. Empty only when the trial carries no execution
  /// shards (whole-edge-only proofs).
  llvm::SmallVector<ExactDestinationDemand, 16> perDestination;
  /// Present and non-empty only for ProvenLogicalInfeasible; absent
  /// otherwise. For overlapping unique-partition owners the witness is the
  /// union of every pairwise overlap, again independent of caller order.
  std::optional<mlir::presburger::PresburgerSet> uncoveredWitness;
  TileRole role = TileRole::UniquePartition;
  bool mergeObligation = false;
  std::string detail;
};

/// Reconstruction action admitted by the exact tensor-operand query.  The
/// enum is deliberately closed: adding a tensor transform requires an exact
/// transfer relation and a matching reconstruction law test in the same
/// change.  Materialization consumes this value instead of rediscovering
/// semantics from operation names or a second matcher.
enum class TensorTransformKind : uint8_t {
  ExpandShape,
  CollapseShape,
  ExtractSlice,
  InsertSlice,
  Pad,
  Cast,
};

/// Exact read of one tensor operand of a tensor transform for one semantic
/// destination Tile.  `demand` is expressed in that operand's index space.
struct TensorTransformInputDemand {
  uint32_t operand = 0;
  std::optional<mlir::presburger::PresburgerSet> demand;
};

/// One local induction step of consumer input reconstruction. `outputDemand` is in
/// the selected result's index space and `operandDemands` are the exact reads
/// sufficient to reconstruct that result on the requested domain. Operation
/// pointers are non-owning references into the immutable query epoch.
struct TensorTransform {
  mlir::Operation *operation = nullptr;
  uint32_t result = 0;
  TensorTransformKind kind = TensorTransformKind::Cast;
  std::optional<mlir::presburger::PresburgerSet> outputDemand;
  llvm::SmallVector<TensorTransformInputDemand, 2> operandDemands;
};

/// Structured-producer stop boundary reached by exact backward propagation.
/// Empty `requiredDomain` is a first-class proof that this producer contributes
/// nothing to this destination. It never becomes a physical edge action.
struct ProducerValueRequirement {
  uint32_t edge = 0;
  uint32_t producerNode = 0;
  uint32_t producerResult = 0;
  mlir::Operation *producer = nullptr;
  std::optional<mlir::presburger::PresburgerSet> requiredDomain;
  llvm::SmallVector<ExactOwnershipIntersection, 8> ownershipIntersections;
};

/// Exact reconstruction recipe for one consumer operand on one destination
/// Tile. Steps are in source block order, hence every preceding tensor value is
/// available before its consumer. The object is query-local and carries no
/// physical representation or route decision.
struct ConsumerInputReconstruction {
  TileId destinationTile{0};
  std::optional<mlir::presburger::PresburgerSet> consumerExecutionDomain;
  std::optional<mlir::presburger::PresburgerSet> operandDemand;
  std::vector<TensorTransform> steps;
  std::vector<ProducerValueRequirement> boundaries;
};

/// Grouped exact-demand verdict for one structured consumer operand. Unlike
/// the per-edge query, this result retains the complete multi-producer tensor operand
/// reconstruction and therefore makes exact-empty branches observable to the
/// single apply without fabricating empty carriers.
struct ConsumerInputDemand {
  ExactDemandStatus status = ExactDemandStatus::IndeterminateFailure;
  uint32_t consumerNode = 0;
  uint32_t consumerOperand = 0;
  mlir::Operation *consumer = nullptr;
  DemandEdgeKind dependencyKind = DemandEdgeKind::DataInput;
  std::vector<ConsumerInputReconstruction> perDestination;
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
