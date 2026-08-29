//===- StaticIndexRange.h - Static index range evaluation -------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_INSTR_STATICINDEXRANGE_H
#define WAFER_ANALYSIS_INSTR_STATICINDEXRANGE_H

#include "mlir/IR/Operation.h"
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
/// multiplication with at least one singleton operand, static unsigned
/// division, signed min/max interval expressions, and identity-preserving
/// wafer.tile.region block/result edges.
/// When `use` is present, constant integer comparisons on enclosing scf.if
/// paths refine the same SSA values before arithmetic is evaluated. Unknown
/// expressions and arithmetic overflow fail closed.
StaticIndexRangeResult
evaluateNonNegativeStaticIndexRange(mlir::Value value,
                                    mlir::Operation *use = nullptr);

} // namespace wafer::memory_planning::detail

#endif // WAFER_ANALYSIS_INSTR_STATICINDEXRANGE_H
