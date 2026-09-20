//===- StaticIndexRange.h - Static index range evaluation -------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_INSTR_STATICINDEXRANGE_H
#define WAFER_ANALYSIS_INSTR_STATICINDEXRANGE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>

namespace mlir::scf {
class ForOp;
}

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
/// bounded scf.for induction variables with constant lower bound and step,
/// checked addition/subtraction, multiplication with at least one singleton
/// operand, unsigned division by a positive constant, signed min/max intervals,
/// and identity-preserving
/// wafer.tile.region block/result edges.
/// Fixed-width integer SSA and integer/index casts use InferIntRangeInterface
/// with full type ranges for unknown leaves. Only a proven non-negative final
/// range is accepted; finite-width truncation and wrapping are preserved.
/// When `use` is present, constant integer comparisons on enclosing scf.if
/// paths refine the same SSA values before arithmetic is evaluated. Unknown
/// expressions and arithmetic overflow fail closed.
StaticIndexRangeResult
evaluateNonNegativeStaticIndexRange(mlir::Value value,
                                    mlir::Operation *use = nullptr);

/// Prove that every reachable invocation of this current loop executes at
/// least once. Bounded, nonconstant upper limits are allowed; unknown is false.
bool proveNonEmptyLoop(mlir::scf::ForOp loop);

/// Exact remainder for a power-of-two modulus, when current index SSA proves
/// one. Unknown expressions return nullopt; interval endpoints alone never
/// prove divisibility of all dynamic values.
std::optional<uint64_t> getKnownIndexRemainder(mlir::Value value,
                                               uint64_t modulus);

/// Proves byte alignment of an actual packed memref view, including each
/// dynamic subview term and its source. Failure means unproven alignment.
mlir::LogicalResult proveByteAlignedPackedView(mlir::Value view);

} // namespace wafer::memory_planning::detail

#endif // WAFER_ANALYSIS_INSTR_STATICINDEXRANGE_H
