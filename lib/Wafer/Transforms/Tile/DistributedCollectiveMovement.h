//===- DistributedCollectiveMovement.h - Actual Tile collectives -*- C++
//-*-===//

#ifndef WAFER_TRANSFORMS_TILE_DISTRIBUTEDCOLLECTIVEMOVEMENT_H
#define WAFER_TRANSFORMS_TILE_DISTRIBUTEDCOLLECTIVEMOVEMENT_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace wafer {
class TargetTopology;
}

namespace wafer::compiler::detail {

/// Returns the deterministic minimum-shortest-hop Hamiltonian cycle for a
/// bounded current physical Tile participant set.
mlir::FailureOr<llvm::SmallVector<uint64_t, 16>>
buildMinimumHopTileRing(const TargetTopology &topology,
                        llvm::ArrayRef<uint64_t> participants);

enum class DistributedCollectiveMovementFailureKind : uint8_t {
  None,
  BrokenContract,
  CompilerFailure,
};

struct DistributedCollectiveMovementResult {
  DistributedCollectiveMovementFailureKind failure =
      DistributedCollectiveMovementFailureKind::None;
  uint64_t components = 0;
  uint64_t removedPeerSends = 0;
  uint64_t removedPeerReceives = 0;
  uint64_t createdPeerSends = 0;
  uint64_t createdPeerReceives = 0;
  uint64_t packCopies = 0;
  uint64_t resultCopies = 0;
  uint64_t reduceScatterComponents = 0;
  uint64_t allReduceComponents = 0;
  uint64_t reductionCombines = 0;
  std::string detail;

  bool succeeded() const {
    return failure == DistributedCollectiveMovementFailureKind::None;
  }
};

/// Replaces complete personalized exchanges already represented by matched
/// current Tile peer operations with row/column aggregate movement. Every
/// aggregate, subview, copy, send and receive is created in this call. A
/// component that does not satisfy the exact rectangular-mesh contract is
/// left unchanged.
DistributedCollectiveMovementResult
materializeDimensionOrderedAllToAll(mlir::ModuleOp module);

/// Replaces a complete personalized contribution exchange followed by one
/// closed associative combine tree per destination with an actual Ring
/// ReduceScatter. Near-miss dataflow is left unchanged.
DistributedCollectiveMovementResult
materializeRingReduceScatter(mlir::ModuleOp module);

/// Replaces a closed full-buffer contribution merge and complete result
/// fanout with Ring ReduceScatter followed by Ring AllGather.
DistributedCollectiveMovementResult
materializeRingAllReduce(mlir::ModuleOp module);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_DISTRIBUTEDCOLLECTIVEMOVEMENT_H
