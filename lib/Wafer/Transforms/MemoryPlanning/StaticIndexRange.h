//===- StaticIndexRange.h - Static index range evaluation -------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_MEMORYPLANNING_STATICINDEXRANGE_H
#define WAFER_TRANSFORMS_MEMORYPLANNING_STATICINDEXRANGE_H

#include "mlir/IR/Value.h"

#include <cstdint>

namespace wafer::memory_planning::detail {

struct StaticIndexRange {
  int64_t min = 0;
  int64_t max = 0;
  bool empty = false;
};

enum class StaticIndexRangeFailureKind {
  None,
  UnsupportedExpression,
  DynamicLoopBounds,
  InvalidLoopBounds,
  NonSingletonMultiplication,
  InvalidUnsignedDivision,
  ArithmeticOverflow,
  NegativeRange,
};

struct StaticIndexRangeResult {
  StaticIndexRange range;
  StaticIndexRangeFailureKind failure = StaticIndexRangeFailureKind::None;

  bool succeeded() const {
    return failure == StaticIndexRangeFailureKind::None;
  }
};

/// Conservatively evaluates a non-negative index value from constants,
/// constant-bounded scf.for induction variables, checked addition/subtraction,
/// multiplication with at least one singleton operand, and static unsigned
/// division. Unknown expressions and arithmetic overflow fail closed.
StaticIndexRangeResult evaluateNonNegativeStaticIndexRange(mlir::Value value);

} // namespace wafer::memory_planning::detail

#endif // WAFER_TRANSFORMS_MEMORYPLANNING_STATICINDEXRANGE_H
