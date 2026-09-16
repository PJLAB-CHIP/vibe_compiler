//===- AccessReuse.h - Materialize selected access reuse --------*- C++ -*-===//
#ifndef WAFER_TRANSFORMS_TILE_ACCESSREUSE_H
#define WAFER_TRANSFORMS_TILE_ACCESSREUSE_H

#include "BoundaryMovement.h"
#include "Wafer/Planning/PhysicalDataflow/AccessReuse.h"

namespace wafer::compiler::detail {

enum class AccessReuseFailureKind : uint8_t {
  None,
  Unsupported,
  Indeterminate,
  BrokenContract,
  CompilerFailure
};

struct AccessReuseResult {
  AccessReuseFailureKind failure = AccessReuseFailureKind::None;
  std::string detail;
  BoundaryMovementStatistics movement;
  uint64_t residentWindows = 0;
  uint64_t slidingWindows = 0;
  uint64_t twoLevelWindows = 0;
  uint64_t replacedLoads = 0;
  bool succeeded() const { return failure == AccessReuseFailureKind::None; }
};

AccessReuseResult
materializeAccessReuse(mlir::ModuleOp module,
                       StructuredMaterializationRelations &relations,
                       const AccessReuseChoice &choice);

} // namespace wafer::compiler::detail
#endif
