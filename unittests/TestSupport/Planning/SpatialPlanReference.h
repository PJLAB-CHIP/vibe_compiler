//===- SpatialPlanReference.h - Independent tiny spatial oracle -*- C++ -*-===//

#ifndef WAFER_TESTSUPPORT_PLANNING_SPATIALPLANREFERENCE_H
#define WAFER_TESTSUPPORT_PLANNING_SPATIALPLANREFERENCE_H

#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <set>

namespace wafer::test {

enum class ReferenceKeyValueRequirement : uint8_t {
  None,
  SingleCell,
  MultipleCells,
};

/// Explicit tiny-semantics input for the nested-loop oracle. This type does
/// not include or call the production spatial-domain implementation.
struct ReferenceSpatialRoot {
  compiler::detail::SemanticRootKey root;
  llvm::SmallVector<int64_t, 4> iteratorExtents;
  llvm::SmallVector<uint8_t, 4> partitionableIterators;
  llvm::SmallVector<uint8_t, 4> reductionIterators;
  llvm::SmallVector<uint8_t, 4> resultParallelIterators;
  llvm::SmallVector<llvm::SmallVector<uint8_t, 4>, 4>
      resultParallelIteratorsByGroup;
  uint32_t reductionResultGroupCount = 0;
  llvm::SmallVector<uint8_t, 2> keyValueReductionIterators;
  ReferenceKeyValueRequirement keyValueRequirement =
      ReferenceKeyValueRequirement::None;
};

/// Exhaustively enumerates the finite reference set. Intended only for 2--4
/// Tiles and bounded extents in unit tests.
std::set<compiler::detail::SpatialPlan>
enumerateReferenceSpatialPlans(llvm::ArrayRef<ReferenceSpatialRoot> roots,
                               llvm::ArrayRef<TileId> availableTiles);

} // namespace wafer::test

#endif // WAFER_TESTSUPPORT_PLANNING_SPATIALPLANREFERENCE_H
