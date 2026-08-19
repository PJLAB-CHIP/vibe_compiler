//===- StructuredOperationTileFootprint.h ------------------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_STRUCTUREDOPERATIONTILEFOOTPRINT_H
#define WAFER_COMPILER_PLANNING_STRUCTUREDOPERATIONTILEFOOTPRINT_H

#include "Wafer/Target/Core/TargetMemory.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace wafer::compiler::detail {

std::optional<uint64_t> estimateStructuredOperationTileResidencyBytes(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> iteratorTileShape,
    const TargetMemoryPolicy &memory);

/// Conservative physical SPM upper bound for the current structured-op
/// lowering. Unlike the logical residency estimate above, this includes the
/// Tensor staging and Cx/NCx materializations for every static prologue,
/// steady and tail wave class that the current temporal lowering template may
/// keep live together. Missing support means that baseline legalization cannot
/// prove the wave set fits; callers must fail closed rather than treating it
/// as a zero or non-binding estimate.
std::optional<uint64_t> getStructuredOperationLoweringSPMUpperBoundBytes(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> localIteratorShape,
    llvm::ArrayRef<int64_t> iteratorTileShape,
    const TargetMemoryPolicy &memory);

std::optional<unsigned> selectStructuredOperationTileRefinementAxis(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> fullIteratorShape,
    llvm::ArrayRef<int64_t> currentIteratorTileShape,
    const TargetMemoryPolicy &memory);

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
deriveStructuredOperationTemporalTileShape(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> localIteratorShape,
    const TargetMemoryPolicy &memory);

std::optional<llvm::SmallVector<int64_t, 4>>
getStructuredResultTileShape(mlir::Operation *operation, unsigned resultNumber,
                             llvm::ArrayRef<int64_t> iteratorTileShape);

std::optional<llvm::SmallVector<int64_t, 4>>
getStructuredOperandTileShape(mlir::Operation *operation,
                              unsigned operandNumber,
                              llvm::ArrayRef<int64_t> iteratorTileShape);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_STRUCTUREDOPERATIONTILEFOOTPRINT_H
