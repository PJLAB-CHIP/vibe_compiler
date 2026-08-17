//===- StructuredDAGCandidateSchedule.h - Query-local candidate execution -*- C++
//-*-===//

#pragma once

#include "Wafer/IR/Target/TargetTopology.h"
#include "StructuredDAGSchedule.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <string>

namespace wafer::compiler::detail {

/// Physical placement selected for one structured DAG node.  The placement is
/// deliberately query-local: selected execution is represented by the
/// resulting Card/Tile IR, never by serializing this object.
struct StructuredDAGNodePlacement {
  StructuredDAGNodeID node = 0;
  /// Result axis to which the selected parallel iterator maps.  The
  /// CardModule output-shard carrier consumes this axis; keeping both fields
  /// prevents a result dimension from standing in for iterator semantics.
  unsigned shardDimension = 0;
  llvm::SmallVector<TileId, 16> tiles;
  /// Structured iterator selected for physical partitioning.  This is not
  /// inferred from the result rank: reduction and projected-away iterators
  /// remain explicit in `iteratorPartitionFactors` with factor one until an
  /// implementation exposes a legal cross-Tile reduction transition.
  unsigned spatialIteratorDimension = 0;
  llvm::SmallVector<uint32_t, 4> iteratorPartitionFactors;
  /// True for the unpartitioned canonical coordinate of a root without any
  /// parallel result axis: exactly one participating Tile, unit partition
  /// factors over every iterator and the complete result domain. The shard
  /// fields above are unused in that state; consumers must treat the root
  /// as fully owned by its single Tile instead of sharding any axis.
  bool unpartitioned = false;
};

/// One producer-to-consumer edge domain that remains on a Tile.
/// The exact edge/domain derivation owns `footprintBytes`; the scheduler owns
/// only the finite wave lifetime from producer completion to matching consumer
/// completion.
struct StructuredDAGLocalResidency {
  StructuredDAGEdgeID edge = 0;
  TileId tile{0};
  uint64_t footprintBytes = 0;
};

/// One finite query-local data movement that gates an edge wave. `duration`
/// is expressed in the same deterministic scheduler work units as compute;
/// final selection still uses accepted Instr-derived cohort cost.
struct StructuredDAGPeerMovement {
  StructuredDAGEdgeID edge = 0;
  TileId sourceTile{0};
  TileId destinationTile{0};
  ScheduleTime duration = 0;
  uint8_t bufferCount = 1;
  llvm::SmallVector<TileLink, 8> route;
};

enum class StructuredDAGLocalMovementResource : uint8_t {
  TileSPM,
  CardDDR,
};

/// One edge movement on a non-NoC resource.  `tile` owns an explicit SPM
/// movement; CardDDR movements use the card-shared calendar and retain the
/// producer/destination Tile only for diagnostics.  Buffer count is a joint
/// search decision: one buffer delays producer slot reuse, while two or more
/// buffers may overlap movement with the next producer wave.
struct StructuredDAGLocalMovement {
  StructuredDAGEdgeID edge = 0;
  StructuredDAGLocalMovementResource resource =
      StructuredDAGLocalMovementResource::TileSPM;
  TileId tile{0};
  ScheduleTime duration = 0;
  uint8_t bufferCount = 1;
};

enum class StructuredDAGReservationResource : uint8_t {
  Compute,
  TileSPM,
  NoCLink,
  CardDDR,
};

/// Auditable query-local reservation produced by the common event transition.
/// It is discarded after actual IR materialization and never becomes a shadow
/// schedule.  NoC reservations name both endpoints; Tile-local resources use
/// identical source/destination values.
struct StructuredDAGResourceReservation {
  StructuredDAGReservationResource resource = StructuredDAGReservationResource::Compute;
  StructuredDAGEdgeID edge = std::numeric_limits<StructuredDAGEdgeID>::max();
  TileId sourceTile{0};
  TileId destinationTile{0};
  ScheduleTime startTime = 0;
  ScheduleTime finishTime = 0;
};

/// One actual use of StructuredDAGScheduleState while validating and ordering a
/// candidate.  Keeping the finite symbolic-wave trace makes it possible to
/// test that search-policy scheduling really exercised ready/running/completed
/// state without storing a shadow schedule in IR.
struct StructuredDAGCandidateSchedule {
  llvm::SmallVector<StructuredDAGNodePlacement, 16> nodePlacements;
  llvm::SmallVector<RunningOpWave, 32> dispatchedWaves;
  uint64_t eventCount = 0;
  uint64_t movementEventCount = 0;
  uint64_t peerMovementWork = 0;
  uint64_t spmMovementWork = 0;
  uint64_t ddrMovementWork = 0;
  ScheduleTime makespan = 0;
  uint64_t peakLiveSPMBytes = 0;
  llvm::SmallVector<StructuredDAGResourceReservation, 64> resourceReservations;
};

/// Runs the finite prologue/steady/tail classes of the current DAG through the
/// event-driven scheduler using the supplied per-node physical placement.
/// Independent ready waves are dispatched together whenever their Tile sets
/// are disjoint.  Overlapping ready work waits for the next completion event.
///
/// Duration is a deterministic interface/type-derived unit estimate.  It is
/// used only to order scheduler events; final candidate selection continues to
/// use the exact Instr-derived theoretical cost cohort.
mlir::FailureOr<StructuredDAGCandidateSchedule> scheduleStructuredDAGCandidate(
    const StructuredDAGAnalysis &dag, llvm::ArrayRef<TileId> availableTiles,
    llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
    llvm::ArrayRef<StructuredDAGLocalResidency> localResidencies,
    std::string *failureReason = nullptr,
    llvm::ArrayRef<StructuredDAGPeerMovement> peerMovements = {},
    llvm::ArrayRef<StructuredDAGLocalMovement> localMovements = {},
    bool enforceSPMCapacity = true);

} // namespace wafer::compiler::detail
