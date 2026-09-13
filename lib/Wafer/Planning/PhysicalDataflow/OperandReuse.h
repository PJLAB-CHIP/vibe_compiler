//===- OperandReuse.h - Logical read projections for proposal ordering ----===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_OPERANDREUSE_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_OPERANDREUSE_H

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {
class Operation;
}

namespace wafer::compiler::detail {

/// A source-level read projection. An unset iterator proves access invariance,
/// not storage residency or an actual DDR load. No IR handles are retained.
struct OperandProjection {
  llvm::SmallBitVector iterators;
  uint64_t bytes = 0;
};

/// Includes only operands whose elements are read by the current payload.
/// Unknown shapes/accesses yield no ranking evidence, never a rejection.
std::optional<llvm::SmallVector<OperandProjection, 4>>
getReadOperandProjections(mlir::Operation *operation);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_OPERANDREUSE_H
