//===- BoundaryMovement.h - Close physical Tile boundaries -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_BOUNDARYMOVEMENT_H
#define WAFER_TRANSFORMS_TILE_BOUNDARYMOVEMENT_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

enum class BoundaryMovementFailureKind : uint8_t {
  None,
  Unsupported,
  BrokenContract,
  CompilerFailure,
};

enum class CompleteAllGatherAlgorithm : uint8_t {
  Ring,
  RecursiveDoubling,
};

enum class CompleteAllToAllAlgorithm : uint8_t {
  Direct,
  DimensionOrdered,
};

enum class DistributedReductionAlgorithm : uint8_t {
  Centralized,
  Ring,
};

enum class BoundaryMovementTransport : uint8_t {
  Peer,
  SharedDDR,
};

/// Selects the current communication component containing this exact SSA
/// edge. The edge must belong to the unchanged input (or its IRMapping clone).
struct BoundaryComponentTransport {
  StructuredBoundaryRelation anchor;
  BoundaryMovementTransport transport = BoundaryMovementTransport::Peer;
};

struct BoundaryMovementOptions {
  CompleteAllGatherAlgorithm allGather = CompleteAllGatherAlgorithm::Ring;
  CompleteAllToAllAlgorithm allToAll = CompleteAllToAllAlgorithm::Direct;
  DistributedReductionAlgorithm reduction =
      DistributedReductionAlgorithm::Centralized;
  BoundaryMovementTransport transport = BoundaryMovementTransport::Peer;
  llvm::SmallVector<BoundaryComponentTransport, 4> components;
};

struct BoundaryComponentQuery {
  BoundaryMovementFailureKind failure = BoundaryMovementFailureKind::None;
  llvm::SmallVector<StructuredBoundaryRelation, 4> anchors;
  std::string detail;
  bool succeeded() const {
    return failure == BoundaryMovementFailureKind::None;
  }
};

/// One actual boundary edge per independently selectable current component.
/// Uses the same preflight/grouping as materialization, without creating IR.
BoundaryComponentQuery queryBoundaryMovementComponents(
    mlir::ModuleOp module, const StructuredMaterializationRelations &relations);

struct BoundaryMovementStatistics {
  uint64_t ddrLoads = 0;
  uint64_t ddrStores = 0;
  uint64_t streamedOutputCarriers = 0;
  uint64_t peerSends = 0;
  uint64_t peerReceives = 0;
  uint64_t peerRelaySends = 0;
  uint64_t topologyFanoutGroups = 0;
  uint64_t topologyFanoutRounds = 0;
  uint64_t nativeBroadcastGroups = 0;
  uint64_t nativeBroadcastRounds = 0;
  uint64_t nativeScatterGroups = 0;
  uint64_t nativeScatterRounds = 0;
  uint64_t ringComponents = 0;
  uint64_t ringRounds = 0;
  uint64_t recursiveDoublingComponents = 0;
  uint64_t recursiveDoublingRounds = 0;
  uint64_t recursiveDoublingSeedCopies = 0;
  uint64_t recursiveDoublingSeedDonations = 0;
  uint64_t sparseRoundComponents = 0;
  uint64_t sparseRounds = 0;
  uint64_t dimensionOrderedAllToAllComponents = 0;
  uint64_t dimensionOrderedAllToAllPackCopies = 0;
  uint64_t ringReduceScatterComponents = 0;
  uint64_t ringAllReduceComponents = 0;
  uint64_t distributedReductionCombines = 0;
  uint64_t distributedReductionResultCopies = 0;
  uint64_t noCutDDRComponents = 0;
  uint64_t crossTileDDRStages = 0;
  uint64_t interRegionDDRStages = 0;
  uint64_t outputCopiesRemoved = 0;
  uint64_t tensorBridgesRemoved = 0;
};

struct BoundaryMovementResult {
  BoundaryMovementFailureKind failure = BoundaryMovementFailureKind::None;
  BoundaryMovementStatistics statistics;
  std::string detail;

  bool succeeded() const {
    return failure == BoundaryMovementFailureKind::None;
  }
};

enum class RecursiveDoublingAvailabilityKind : uint8_t {
  Available,
  Unavailable,
  BrokenContract,
};

struct RecursiveDoublingAvailability {
  RecursiveDoublingAvailabilityKind kind =
      RecursiveDoublingAvailabilityKind::Unavailable;
  std::string detail;

  bool isAvailable() const {
    return kind == RecursiveDoublingAvailabilityKind::Available;
  }
};

struct DistributedMovementAvailability {
  bool sharedDDR = false;
  bool dimensionOrderedAllToAll = false;
  bool distributedReduction = false;
  bool brokenContract = false;
  std::string detail;
};

/// Recomputes whether current complete AllGather endpoints admit the
/// recursive-doubling realization. The result is query-local and does not
/// modify IR or retain allocation/message facts.
RecursiveDoublingAvailability analyzeRecursiveDoublingAvailability(
    mlir::ModuleOp module, const StructuredMaterializationRelations &relations);

/// Cheap current-endpoint prefilter for specialized distributed movement.
/// Positive answers are confirmed by actual materialization; negative answers
/// avoid creating an unchanged candidate clone.
DistributedMovementAvailability analyzeDistributedMovementAvailability(
    mlir::ModuleOp module, const StructuredMaterializationRelations &relations);

/// Converts logical TileRegion tensor boundaries to actual physical movement.
/// External and same-Tile inter-region values use explicit DDR load/store;
/// cross-Tile endpoint relations become a topology-aware relay tree, a
/// cut-closed Ring/recursive-doubling/matched peer round, or an explicit
/// causal DDR boundary. Recursive doubling creates its aggregate allocation
/// and slots in this transaction; it never assumes separate buffers are
/// contiguous.
/// Every selected realization is immediately materialized; no route, round or
/// action plan survives this call. Instr completion later places waits from
/// the actual tokens, aliases, effects and target resources. The supplied
/// current endpoint relations are consumed exactly once. This transformation
/// does not choose layout, execution structure, worker, completion placement
/// or memory offsets.
BoundaryMovementResult
materializeTileBoundaryMovement(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations,
                                BoundaryMovementOptions options = {});

/// Verifies that TileRegion shaped boundaries are Wafer DDR memrefs, no
/// tensor/memref bridge remains, and no SPM root crosses a TileRegion.
mlir::LogicalResult verifyPhysicalTileDataflow(mlir::ModuleOp module);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_BOUNDARYMOVEMENT_H
