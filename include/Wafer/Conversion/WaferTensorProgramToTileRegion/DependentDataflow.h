//===- DependentDataflow.h - Selected edge-action lowering -*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_DEPENDENTDATAFLOW_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_DEPENDENTDATAFLOW_H

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/Core/TopologyIds.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer {

/// Query-local lowering contract selected by the owning controller. Search
/// candidates use JointDataflow. The deterministic no-fusion baseline uses
/// IndependentDDRStages so every structured data dependency is separated by
/// an explicit compiler-owned DDR/TileRegion boundary. Consumer-owned DPS
/// initialization, such as a scalar fill directly absorbed by a reduction,
/// stays in that consumer root and does not invent a data carrier. Lowering
/// must never infer this choice from a coincidental combination of actions.
enum class SpatialDataflowMaterializationMode : uint8_t {
  JointDataflow,
  IndependentDDRStages,
};

/// One query-local physical treatment of a structured SSA dependency. A
/// dependency may be direct or may cross a statically provable unary pure
/// producer-to-consumer tensor chain such as expand/collapse shape. This is the
/// only carrier between structured-DAG selection and actual CardModule
/// materialization: it is never persisted, serialized, or recovered from an
/// operation name/ordinal.
enum class SpatialEdgeAction : uint8_t {
  /// Low-level consumer-driven recursive producer tiling used while building
  /// an already-selected region. Search grouping is represented by actual
  /// node groups/TileRegions, never by this edge action. The producer value
  /// remains inside that region through ordinary SSA.
  RecursiveProducerTiling,
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
  TileId sourceTile{0};
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
/// has no fragments. A finite non-rectangular required producer region uses the
/// rectangle-union form: `producerOffsets`/`producerSizes` are its carrier
/// bounds and `fragmentsDefineProducerDemand` states that the disjoint
/// fragment union, rather than those bounds, is the exact demand.
struct SpatialEdgeStrategy {
  mlir::Operation *producer = nullptr;
  unsigned producerResult = 0;
  mlir::Operation *consumer = nullptr;
  unsigned consumerOperand = 0;
  llvm::SmallVector<int64_t, 4> consumerOffsets;
  llvm::SmallVector<int64_t, 4> consumerSizes;
  llvm::SmallVector<int64_t, 4> producerOffsets;
  llvm::SmallVector<int64_t, 4> producerSizes;
  TileId sourceTile{0};
  TileId destinationTile{0};
  SpatialEdgeAction action = SpatialEdgeAction::LocalShardResidency;
  bool fragmentsDefineProducerDemand = false;
  llvm::SmallVector<SpatialEdgeFragment, 4> fragments;
};

/// Immutable, query-local facts derived once for one selected edge in its
/// scheduling TensorProgram.  The array is positionally paired with the
/// caller's edge-strategy array and never enters IR or candidate identity.
struct SpatialEdgeMaterializationFacts {
  bool requiresConsumerInputReconstruction = false;
  uint64_t consumerScheduleOrdinal = 0;
};

/// Derive selected-edge materialization facts exclusively from the grouped
/// exact-demand proof and stable source-block order. This query does not walk
/// producer chains or recover logical demand from a physical strategy.
mlir::FailureOr<llvm::SmallVector<SpatialEdgeMaterializationFacts, 16>>
deriveSpatialEdgeMaterializationFacts(
    mlir::Block &sourceBody, llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<analysis::DependencyDemand> operandDemands,
    std::string *failureReason = nullptr);

/// Returns whether an edge strategy has actual work on `tile`.  Destination
/// actions are incident on their destination; remote PeerFragments are also
/// incident on the fragment source that emits the send.
bool isSpatialEdgeStrategyIncidentOnTile(const SpatialEdgeStrategy &strategy,
                                         TileId tile);

/// Lowers one Tile's selected output shards and every selected edge
/// action incident on that Tile.  Each action is reflected by actual IR:
/// fused SSA, explicit local staging, peer fragment assembly, compiler-owned
/// DDR store/reload, cloned pure compute, or a real TileRegion boundary.
/// Unsupported semantics reject this actual candidate atomically; the API
/// never repairs placement or substitutes a different action.
mlir::LogicalResult lowerSpatialEdgeStrategiesToTileRegionModule(
    mlir::ModuleOp sourceModule, unsigned functionalArgumentCount,
    llvm::ArrayRef<SpatialOutputShard> outputShards, TileId currentTile,
    SpatialDataflowMaterializationMode materializationMode,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalPartition,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles = {},
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    StructuredMaterializationRelations *materializationRelations = nullptr,
    llvm::ArrayRef<SpatialEdgeMaterializationFacts> edgeFacts = {},
    llvm::ArrayRef<analysis::DependencyDemand> operandDemands = {},
    bool requireOneStructuredRootPerRegion = false);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_DEPENDENTDATAFLOW_H
