//===- SpatialAssignment.h - Closed logical execution assignment -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_PHYSICALDATAFLOW_SPATIALASSIGNMENT_H
#define WAFER_ANALYSIS_PHYSICALDATAFLOW_SPATIALASSIGNMENT_H

#include "Wafer/Analysis/PhysicalDataflow/SemanticRoot.h"
#include "Wafer/Target/Core/TopologyIds.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>

namespace wafer::compiler::detail {

struct ReductionGroupId {
  SemanticRootKey root;
  uint32_t resultGroup = 0;
  llvm::SmallVector<uint32_t, 4> parallelCoordinate;

  friend bool operator==(const ReductionGroupId &lhs,
                         const ReductionGroupId &rhs) {
    return lhs.root == rhs.root && lhs.resultGroup == rhs.resultGroup &&
           lhs.parallelCoordinate == rhs.parallelCoordinate;
  }
  friend bool operator!=(const ReductionGroupId &lhs,
                         const ReductionGroupId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const ReductionGroupId &lhs,
                        const ReductionGroupId &rhs) {
    if (lhs.root != rhs.root)
      return lhs.root < rhs.root;
    if (lhs.resultGroup != rhs.resultGroup)
      return lhs.resultGroup < rhs.resultGroup;
    return std::lexicographical_compare(
        lhs.parallelCoordinate.begin(), lhs.parallelCoordinate.end(),
        rhs.parallelCoordinate.begin(), rhs.parallelCoordinate.end());
  }
};

struct IteratorInterval {
  int64_t offset = 0;
  int64_t size = 0;

  int64_t getEnd() const { return offset + size; }

  friend bool operator==(const IteratorInterval &lhs,
                         const IteratorInterval &rhs) {
    return lhs.offset == rhs.offset && lhs.size == rhs.size;
  }
};

struct LogicalShardId {
  SemanticRootKey root;
  llvm::SmallVector<uint32_t, 4> coordinate;

  friend bool operator==(const LogicalShardId &lhs, const LogicalShardId &rhs) {
    return lhs.root == rhs.root && lhs.coordinate == rhs.coordinate;
  }
  friend bool operator!=(const LogicalShardId &lhs, const LogicalShardId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const LogicalShardId &lhs, const LogicalShardId &rhs) {
    if (lhs.root != rhs.root)
      return lhs.root < rhs.root;
    return std::lexicographical_compare(lhs.coordinate.begin(),
                                        lhs.coordinate.end(),
                                        rhs.coordinate.begin(),
                                        rhs.coordinate.end());
  }
};

struct ExecutionShard {
  LogicalShardId shard;
  TileId tile{0};
  llvm::SmallVector<IteratorInterval, 4> iterationDomain;
};

struct ReductionGroupPlacement {
  ReductionGroupId group;
  TileId mergeTile{0};
};

struct NodeExecutionPartition {
  SemanticRootKey root;
  llvm::SmallVector<ExecutionShard, 16> shards;
  llvm::SmallVector<ReductionGroupPlacement, 4> reductionGroups;
};

/// Query-local exact Cartesian execution assignment. It contains no final
/// result ownership, operand demand, physical representation, movement,
/// buffer, schedule, or cost facts.
struct SpatialAssignment {
  llvm::SmallVector<NodeExecutionPartition, 16> nodes;
};

} // namespace wafer::compiler::detail

#endif // WAFER_ANALYSIS_PHYSICALDATAFLOW_SPATIALASSIGNMENT_H
