//===- CollectiveTopologyAnalysis.h - Collective topology facts -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_COLLECTIVETOPOLOGYANALYSIS_H
#define WAFER_ANALYSIS_COLLECTIVETOPOLOGYANALYSIS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
class Operation;
} // namespace mlir

namespace wafer::analysis {

/// A cyclic order expressed as indices into the collective rank_group.
/// Rotation is fixed at group index zero only to make materialization
/// deterministic; it does not change the cycle.
struct CollectiveRingOrder {
  llvm::SmallVector<int64_t, 16> groupIndices;
};

/// A rooted ordered binary tree expressed entirely in indices into the
/// collective rank_group. Every node has at most one lower-index child and
/// one higher-index child, so an in-order traversal is exactly rank_group
/// order. Children are stored left-before-right. These values are
/// compiler-private rewrite parameters and are never serialized into accepted
/// IR.
struct CollectiveTree {
  int64_t rootGroupIndex = -1;
  llvm::SmallVector<int64_t, 16> parentGroupIndices;
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 16> childGroupIndices;
  llvm::SmallVector<int64_t, 16> depths;
};

/// Finds the exact minimum-shortest-hop Hamiltonian cycle for the currently
/// supported bounded collective group. Failure means this extra parameter
/// cannot be materialized; callers must reject only that candidate clone.
mlir::FailureOr<CollectiveRingOrder>
buildMinimumHopCollectiveRingOrder(mlir::Operation *anchor,
                                   llvm::ArrayRef<int64_t> rankGroup);

/// Builds the exact minimum-total-shortest-hop ordered binary tree for the
/// currently supported bounded collective group. The ordering constraint
/// preserves the source collective's rank_group reduction order. Equal-hop
/// choices use rooted path distance and logical-rank tie breaks to remain
/// deterministic. Failure means callers must reject only that candidate clone.
mlir::FailureOr<CollectiveTree>
buildMinimumHopCollectiveTree(mlir::Operation *anchor,
                              llvm::ArrayRef<int64_t> rankGroup);

} // namespace wafer::analysis

#endif
