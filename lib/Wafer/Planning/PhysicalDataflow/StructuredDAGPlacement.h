//===- StructuredDAGPlacement.h - Query-local node placement ----*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"

#include "Wafer/IR/Target/TargetTopology.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer::compiler::detail {

/// Query-local physical placement of one structured DAG node. Selected
/// execution is represented by resulting Card/Tile IR, never by serializing
/// this object.
struct StructuredDAGNodePlacement {
  StructuredDAGNodeID node = 0;
  /// Complete structured-iterator factor vector. The factor product equals
  /// `tiles.size()`; all-one is an unpartitioned single-Tile coordinate.
  llvm::SmallVector<uint32_t, 4> iteratorPartitionFactors;
  /// Canonical row-major logical partition coordinate to physical Tile. Every
  /// entry is distinct; Tile order is semantic because it is the embedding.
  llvm::SmallVector<TileId, 16> tiles;
};

} // namespace wafer::compiler::detail
