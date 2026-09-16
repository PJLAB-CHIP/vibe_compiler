//===- CollectiveAlgorithms.h - Generic collective algorithms -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_COLLECTIVEALGORITHMS_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_COLLECTIVEALGORITHMS_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace wafer::compiler::detail {

struct DimensionOrderedAllToAllStep {
  uint64_t source = 0;
  uint64_t relay = 0;
  uint64_t destination = 0;
  uint32_t dimension = 0;
  uint32_t round = 0;

  friend bool operator==(const DimensionOrderedAllToAllStep &lhs,
                         const DimensionOrderedAllToAllStep &rhs) {
    return lhs.source == rhs.source && lhs.relay == rhs.relay &&
           lhs.destination == rhs.destination &&
           lhs.dimension == rhs.dimension && lhs.round == rhs.round;
  }
};

/// Constructs a deterministic dimension-ordered personalized exchange over
/// a row-major logical mesh. Participant IDs are opaque. A two-dimensional
/// path uses the source-row/destination-column relay; a one-dimensional mesh
/// degenerates to direct pairwise steps. No target topology, card ID, packet
/// limit, or transport capability is consulted here.
mlir::FailureOr<llvm::SmallVector<DimensionOrderedAllToAllStep, 64>>
buildDimensionOrderedAllToAll(llvm::ArrayRef<uint64_t> rowMajorParticipants,
                              uint64_t rows, uint64_t columns);

/// Builds a deterministic minimum-cost Hamiltonian ring over participant IDs.
/// Topology ownership stays with the caller through the distance oracle.
mlir::FailureOr<llvm::SmallVector<uint64_t, 16>>
buildMinimumHopRing(
    llvm::ArrayRef<uint64_t> participants, uint64_t maximumParticipants,
    llvm::function_ref<std::optional<uint64_t>(uint64_t, uint64_t)>
        distanceOracle);

struct BroadcastTreeEdge {
  uint64_t source, destination;
};

/// Prim tree over opaque participants and symmetric distances. Edges are
/// emitted in construction order, so a node receives before forwarding.
mlir::FailureOr<llvm::SmallVector<BroadcastTreeEdge, 16>>
buildMinimumHopBroadcastTree(
    llvm::ArrayRef<uint64_t> participants,
    llvm::function_ref<std::optional<uint64_t>(uint64_t, uint64_t)>
        distanceOracle);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_COLLECTIVEALGORITHMS_H
