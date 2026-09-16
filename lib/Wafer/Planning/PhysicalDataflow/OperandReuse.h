//===- OperandReuse.h - Logical read projections for proposal ordering ----===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_OPERANDREUSE_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_OPERANDREUSE_H

#include "llvm/ADT/ArrayRef.h"
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
  std::optional<uint64_t> programArgument;
  llvm::SmallVector<int64_t, 4> offsets, sizes;
};

/// Includes only DPS inputs read by the current payload, not mutable inits.
/// When supplied, iteration extents describe the selected logical scope; byte
/// weights use its exact projected access image, not the unsliced operand.
/// Unknown shapes/accesses yield no ranking evidence, never a rejection.
std::optional<llvm::SmallVector<OperandProjection, 4>>
getReadOperandProjections(mlir::Operation *operation,
                          llvm::ArrayRef<int64_t> iterationExtents = {});

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_OPERANDREUSE_H
