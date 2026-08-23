//===- TemporalTileShape.h ----------------------------------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_TEMPORALTILESHAPE_H
#define WAFER_COMPILER_PLANNING_TEMPORALTILESHAPE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class Operation;
}

namespace wafer::compiler::detail {

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
deriveLocalIteratorExtents(mlir::Operation *operation,
                           llvm::ArrayRef<uint32_t> partitionFactors,
                           std::string *failureReason = nullptr);

/// Projects a concrete iterator tile through the structured operation's
/// indexing map. These functions compute exact logical tensor shapes only;
/// they do not predict allocations, layout materialization or SPM legality.
std::optional<llvm::SmallVector<int64_t, 4>>
getStructuredResultTileShape(mlir::Operation *operation,
                             unsigned resultNumber,
                             llvm::ArrayRef<int64_t> iteratorTileShape);

std::optional<llvm::SmallVector<int64_t, 4>>
getStructuredOperandTileShape(mlir::Operation *operation,
                              unsigned operandNumber,
                              llvm::ArrayRef<int64_t> iteratorTileShape);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_TEMPORALTILESHAPE_H
