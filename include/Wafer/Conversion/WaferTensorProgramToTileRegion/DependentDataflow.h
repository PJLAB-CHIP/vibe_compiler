//===- DependentDataflow.h - Selected edge-action lowering -*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_DEPENDENTDATAFLOW_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_DEPENDENTDATAFLOW_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/PhysicalIds.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace wafer {

/// Query-local lowering contract selected by the owning controller.  Search
/// candidates use JointDataflow.  The deterministic no-fusion baseline uses
/// IndependentDDRStages so every structured op is separated by an explicit
/// compiler-owned DDR/TileRegion boundary.  Lowering must never infer this
/// choice from a coincidental combination of edge actions.
enum class SpatialDataflowMaterializationMode : uint8_t {
  JointDataflow,
  IndependentDDRStages,
};

/// One query-local physical treatment of a structured SSA dependency. A
/// dependency may be direct or may cross a statically provable unary pure
/// support chain such as expand/collapse shape. This is the only carrier
/// between whole-DAG selection and actual CardProgram materialization: it is
/// never persisted, serialized, or recovered from an operation name/ordinal.
enum class SpatialEdgeAction : uint8_t {
  /// Consumer-driven traversal recursively materializes the producer in the
  /// same TileRegion and keeps the exact producer value resident.
  CoupledFusion,
  /// Producer and consumer use independent traversals in one TileRegion.  An
  /// explicit tile-local staging copy separates their SPM versions. This is
  /// local edge residency, not an op-fusion claim.
  LocalShardResidency,
  /// Exact local and remote fragments are assembled at the destination.  Only
  /// remote fragments become peer send/receive/wait operations.
  PeerFragments,
  /// The selected producer domain is stored to compiler-owned DDR and loaded
  /// again before the consumer while the surrounding TileRegion stays intact.
  SpillReload,
  /// A pure producer is cloned at the consumer placement and traversed there.
  Recompute,
  /// The selected producer domain is stored to compiler-owned DDR; the store
  /// and reload belong to two consecutive TileRegions so all SPM roots are
  /// released at the boundary.
  RegionCut,
  /// Reserved for an exact typed layout conversion.  Materialization rejects
  /// this action unless a concrete source/destination layout request is
  /// available; selection must not invent one from tensor shape or names.
  LocalPhysicalConversion,
};

enum class SpatialEdgeFragmentKind : uint8_t {
  Resident,
  Peer,
};

/// One rectangular fragment of a PeerFragments strategy.  `sourceTile` is the
/// Tile that computes the exact producer domain.  Resident fragments require
/// sourceTile == destinationTile and carry no message fields.  Peer fragments
/// require distinct endpoints and a unique message identity.
struct SpatialEdgeFragment {
  SpatialEdgeFragmentKind kind = SpatialEdgeFragmentKind::Resident;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  PhysicalTileId sourceTile{0};
  uint64_t bytes = 0;
  int64_t communicationId = 0;
  int64_t payloadSlice = 0;
};

/// Exact current-SSA edge strategy selected jointly with spatial placement,
/// temporal tiling and residency. Consumer and producer domains are both
/// explicit because a TilingInterface indexing relation may permute or
/// broadcast dimensions. A same-placement provider may leave both consumer
/// vectors empty only when the live TilingInterface can derive one static
/// consumer result domain from the exact producer demand; lowering resolves
/// and validates that relation before changing IR. For PeerFragments,
/// `fragments` must cover the producer demand all-and-only; every other action
/// has no fragments.
struct SpatialEdgeStrategy {
  mlir::Operation *producer = nullptr;
  unsigned producerResult = 0;
  mlir::Operation *consumer = nullptr;
  unsigned consumerOperand = 0;
  llvm::SmallVector<int64_t, 4> consumerOffsets;
  llvm::SmallVector<int64_t, 4> consumerSizes;
  llvm::SmallVector<int64_t, 4> producerOffsets;
  llvm::SmallVector<int64_t, 4> producerSizes;
  PhysicalTileId sourceTile{0};
  PhysicalTileId destinationTile{0};
  SpatialEdgeAction action = SpatialEdgeAction::LocalShardResidency;
  /// Explicit physical-representation assignment selected by the common
  /// whole-DAG transition.  A false `hasLayoutAssignment` means that current
  /// interfaces expose no provable choice and the incomplete term is omitted;
  /// it never means that Tensor layout was guessed.
  bool hasLayoutAssignment = false;
  MemLayout producerLayout = MemLayout::Tensor;
  MemLayout consumerLayout = MemLayout::Tensor;
  /// Selected rotating-buffer multiplicity for this edge.  The materializer
  /// must either produce the requested slot family or reject the candidate;
  /// it must not silently choose another count.
  uint8_t bufferCount = 1;
  llvm::SmallVector<SpatialEdgeFragment, 4> fragments;
};

/// Returns the topologically ordered pure support path connecting one selected
/// structured producer result to one selected structured consumer operand. A
/// direct dependency returns an empty path. Support operations may join other
/// structured results; those are separate explicit edge strategies and must
/// all be rebound before materialization. The selected producer itself must
/// reach the operand through exactly one path.
mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>>
deriveUnaryPureSupportChain(mlir::Operation *producer, unsigned producerResult,
                            mlir::Operation *consumer, unsigned consumerOperand,
                            std::string *failureReason = nullptr);

/// Maps one exact producer result tile through the current direct/support SSA
/// relation and the consumer's TilingInterface to one exact consumer result
/// tile. This is a legality query over current IR; it does not select or
/// mutate a placement.
mlir::LogicalResult deriveSpatialEdgeConsumerResultDomain(
    const SpatialEdgeStrategy &strategy,
    llvm::SmallVectorImpl<int64_t> &consumerOffsets,
    llvm::SmallVectorImpl<int64_t> &consumerSizes,
    std::string *failureReason = nullptr);

/// Returns whether an edge strategy has actual work on `tile`.  Destination
/// actions are incident on their destination; remote PeerFragments are also
/// incident on the fragment source that emits the send.
bool isSpatialEdgeStrategyIncidentOnTile(const SpatialEdgeStrategy &strategy,
                                         PhysicalTileId tile);

/// Lowers one physical Tile's selected output shards and every selected edge
/// action incident on that Tile.  Each action is reflected by actual IR:
/// fused SSA, explicit local staging, peer fragment assembly, compiler-owned
/// DDR store/reload, cloned pure compute, or a real TileRegion boundary.
/// Unsupported semantics reject this actual candidate atomically; the API
/// never repairs placement or substitutes a different action.
mlir::LogicalResult lowerSpatialEdgeStrategiesToTileRegionModule(
    mlir::ModuleOp sourceModule, unsigned functionalArgumentCount,
    llvm::ArrayRef<SpatialOutputShard> outputShards, PhysicalTileId currentTile,
    SpatialDataflowMaterializationMode materializationMode,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalPartition,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles = {},
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    StructuredMaterializationRelations *materializationRelations = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_DEPENDENTDATAFLOW_H
