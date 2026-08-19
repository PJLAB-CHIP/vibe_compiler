//===- StructuredDAGPlacement.h - Query-local node placement ----*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"

#include "Wafer/IR/Target/TargetTopology.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <tuple>

namespace wafer::compiler::detail {

/// Physical partition selected for one structured DAG node. This is a narrow
/// coordinate fact shared by deterministic legalization and search; it owns no
/// option domain, candidate identity, score or transition history.
struct StructuredDAGSpatialPartition {
  unsigned iteratorDimension = 0;
  unsigned resultDimension = 0;

  friend bool operator==(const StructuredDAGSpatialPartition &lhs,
                         const StructuredDAGSpatialPartition &rhs) {
    return lhs.iteratorDimension == rhs.iteratorDimension &&
           lhs.resultDimension == rhs.resultDimension;
  }
  friend bool operator!=(const StructuredDAGSpatialPartition &lhs,
                         const StructuredDAGSpatialPartition &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const StructuredDAGSpatialPartition &lhs,
                        const StructuredDAGSpatialPartition &rhs) {
    return std::tie(lhs.iteratorDimension, lhs.resultDimension) <
           std::tie(rhs.iteratorDimension, rhs.resultDimension);
  }
};

/// Query-local physical placement of one structured DAG node. Selected
/// execution is represented by resulting Card/Tile IR, never by serializing
/// this object.
struct StructuredDAGNodePlacement {
  StructuredDAGNodeID node = 0;
  /// Present only for a partitioned coordinate. Keeping iterator and result
  /// dimensions together prevents an unpartitioned reduction root from
  /// fabricating axis zero merely to satisfy a carrier field.
  std::optional<StructuredDAGSpatialPartition> spatialPartition;
  llvm::SmallVector<TileId, 16> tiles;
  /// Reduction and projected-away iterators remain explicit with factor one
  /// until an implementation exposes a legal cross-Tile reduction transition.
  llvm::SmallVector<uint32_t, 4> iteratorPartitionFactors;
  /// A missing spatialPartition is the unpartitioned canonical coordinate:
  /// exactly one participating Tile, unit factors and the complete result.
};

} // namespace wafer::compiler::detail
