//===- SpatialPlan.h - Compact spatial planning schema --------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SPATIALPLAN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SPATIALPLAN_H

#include "Wafer/Planning/PhysicalDataflow/SpatialAssignment.h"
#include "Wafer/Target/TopologyIds.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace wafer::compiler::detail {

enum class IteratorPartitionScheme : uint8_t {
  BalancedParts,
  UniformExtent,
};

struct IteratorPartition {
  uint32_t iterator = 0;
  IteratorPartitionScheme scheme = IteratorPartitionScheme::BalancedParts;
  int64_t parameter = 1;

  friend bool operator==(const IteratorPartition &lhs,
                         const IteratorPartition &rhs) {
    return lhs.iterator == rhs.iterator && lhs.scheme == rhs.scheme &&
           lhs.parameter == rhs.parameter;
  }
  friend bool operator<(const IteratorPartition &lhs,
                        const IteratorPartition &rhs);
};

mlir::FailureOr<int64_t>
getIteratorPartitionIntervalCount(int64_t extent,
                                  const IteratorPartition &partition,
                                  std::string *failureReason = nullptr);

/// Materializes one partition's exact ordered nonempty intervals. The result
/// is a query value derived only from the typed scheme and extent; callers do
/// not retain it in SpatialPlan identity.
mlir::FailureOr<llvm::SmallVector<IteratorInterval, 4>>
getIteratorPartitionIntervals(int64_t extent,
                              const IteratorPartition &partition,
                              size_t maximumIntervals,
                              std::string *failureReason = nullptr);

struct MergePlacement {
  ReductionGroupId group;
  TileId tile{0};

  friend bool operator==(const MergePlacement &lhs, const MergePlacement &rhs) {
    return lhs.group == rhs.group && lhs.tile == rhs.tile;
  }
  friend bool operator<(const MergePlacement &lhs, const MergePlacement &rhs);
};

struct NodeSpatialPlan {
  SemanticRootKey root;
  llvm::SmallVector<IteratorPartition, 4> axes;
  llvm::SmallVector<TileId, 16> embedding;
  llvm::SmallVector<MergePlacement, 4> reductionMerges;

  friend bool operator==(const NodeSpatialPlan &lhs,
                         const NodeSpatialPlan &rhs) {
    return lhs.root == rhs.root && lhs.axes == rhs.axes &&
           lhs.embedding == rhs.embedding &&
           lhs.reductionMerges == rhs.reductionMerges;
  }
  friend bool operator<(const NodeSpatialPlan &lhs, const NodeSpatialPlan &rhs);
};

struct SpatialPlan {
  llvm::SmallVector<NodeSpatialPlan, 16> nodes;

  friend bool operator==(const SpatialPlan &lhs, const SpatialPlan &rhs) {
    return lhs.nodes == rhs.nodes;
  }
  friend bool operator<(const SpatialPlan &lhs, const SpatialPlan &rhs);
};

/// Static iterator-space facts supplied by normalized structured semantics.
/// The schema deliberately contains no iterator role, attention mode, demand,
/// layout, movement, or resource decision.
struct NodeIterationSpace {
  SemanticRootKey root;
  llvm::SmallVector<int64_t, 4> iteratorExtents;
};

/// Validated immutable inputs needed to close and structurally verify a
/// SpatialPlan. Construction canonicalizes node and Tile order and rejects
/// duplicate identities.
class SpatialPlanningProblem {
public:
  static mlir::FailureOr<SpatialPlanningProblem>
  create(llvm::ArrayRef<NodeIterationSpace> nodes,
         llvm::ArrayRef<TileId> availableTiles,
         std::string *failureReason = nullptr);

  llvm::ArrayRef<NodeIterationSpace> getNodes() const { return nodes; }
  llvm::ArrayRef<TileId> getAvailableTiles() const { return availableTiles; }

private:
  SpatialPlanningProblem(llvm::SmallVector<NodeIterationSpace, 16> nodes,
                         llvm::SmallVector<TileId, 16> availableTiles)
      : nodes(std::move(nodes)), availableTiles(std::move(availableTiles)) {}

  llvm::SmallVector<NodeIterationSpace, 16> nodes;
  llvm::SmallVector<TileId, 16> availableTiles;
};

mlir::LogicalResult
validateSpatialPlanStructure(const SpatialPlanningProblem &problem,
                             const SpatialPlan &plan,
                             std::string *failureReason = nullptr);

mlir::FailureOr<SpatialAssignment>
closeSpatialPlanStructure(const SpatialPlanningProblem &problem,
                          const SpatialPlan &plan,
                          std::string *failureReason = nullptr);

mlir::LogicalResult
validateSpatialAssignmentStructure(const SpatialPlanningProblem &problem,
                                   const SpatialAssignment &assignment,
                                   std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SPATIALPLAN_H
