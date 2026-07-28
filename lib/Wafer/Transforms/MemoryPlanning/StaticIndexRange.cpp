//===- StaticIndexRange.cpp - Static index range evaluation -----*- C++ -*-===//

#include "MemoryPlanning/StaticIndexRange.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace wafer::memory_planning::detail {
namespace {

using Failure = StaticIndexRangeFailureKind;
using Result = StaticIndexRangeResult;

static Result failed(Failure failure) {
  return StaticIndexRangeResult{/*range=*/{}, failure};
}

class StaticIndexRangeEvaluator {
public:
  Result evaluate(mlir::Value value) {
    auto cached = cache.find(value);
    if (cached != cache.end())
      return cached->second;
    if (!active.insert(value).second)
      return failed(Failure::UnsupportedExpression);

    Result result = evaluateImpl(value);
    active.erase(value);
    cache.try_emplace(value, result);
    return result;
  }

private:
  Result evaluateImpl(mlir::Value value) {
    if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
      return Result{StaticIndexRange{*constant, *constant, /*empty=*/false}};

    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
    auto loop = blockArg && blockArg.getOwner()
                    ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                          blockArg.getOwner()->getParentOp())
                    : mlir::scf::ForOp{};
    if (loop && value == loop.getInductionVar())
      return evaluateLoopInductionVariable(loop);

    if (auto add = value.getDefiningOp<mlir::arith::AddIOp>())
      return evaluateAdd(add);
    if (auto subtract = value.getDefiningOp<mlir::arith::SubIOp>())
      return evaluateSubtract(subtract);
    if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>())
      return evaluateMultiply(multiply);
    if (auto divide = value.getDefiningOp<mlir::arith::DivUIOp>())
      return evaluateUnsignedDivide(divide);
    return failed(Failure::UnsupportedExpression);
  }

  Result evaluateLoopInductionVariable(mlir::scf::ForOp loop) {
    std::optional<int64_t> lower =
        mlir::getConstantIntValue(loop.getLowerBound());
    std::optional<int64_t> upper =
        mlir::getConstantIntValue(loop.getUpperBound());
    std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
    if (!lower || !upper || !step)
      return failed(Failure::DynamicLoopBounds);
    if (*lower < 0 || *upper < 0 || *step <= 0)
      return failed(Failure::InvalidLoopBounds);
    if (*upper <= *lower)
      return Result{
          StaticIndexRange{*lower, *lower, /*empty=*/true}};

    const int64_t distance = *upper - *lower - 1;
    int64_t delta = 0;
    int64_t maximum = 0;
    if (llvm::MulOverflow(distance / *step, *step, delta) ||
        llvm::AddOverflow(*lower, delta, maximum))
      return failed(Failure::ArithmeticOverflow);
    return Result{
        StaticIndexRange{*lower, maximum, /*empty=*/false}};
  }

  std::optional<std::pair<StaticIndexRange, StaticIndexRange>>
  evaluateOperands(mlir::Value lhsValue, mlir::Value rhsValue,
                   Failure &failure) {
    Result lhs = evaluate(lhsValue);
    if (!lhs.succeeded()) {
      failure = lhs.failure;
      return std::nullopt;
    }
    Result rhs = evaluate(rhsValue);
    if (!rhs.succeeded()) {
      failure = rhs.failure;
      return std::nullopt;
    }
    return std::make_pair(lhs.range, rhs.range);
  }

  Result evaluateAdd(mlir::arith::AddIOp add) {
    Failure failure = Failure::None;
    auto operands = evaluateOperands(add.getLhs(), add.getRhs(), failure);
    if (!operands)
      return failed(failure);
    if (operands->first.empty || operands->second.empty)
      return Result{StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true}};

    int64_t minimum = 0;
    int64_t maximum = 0;
    if (llvm::AddOverflow(operands->first.min, operands->second.min, minimum) ||
        llvm::AddOverflow(operands->first.max, operands->second.max, maximum))
      return failed(Failure::ArithmeticOverflow);
    return Result{StaticIndexRange{minimum, maximum, /*empty=*/false}};
  }

  Result evaluateSubtract(mlir::arith::SubIOp subtract) {
    Failure failure = Failure::None;
    auto operands =
        evaluateOperands(subtract.getLhs(), subtract.getRhs(), failure);
    if (!operands)
      return failed(failure);
    if (operands->first.empty || operands->second.empty)
      return Result{StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true}};

    int64_t minimum = 0;
    int64_t maximum = 0;
    if (llvm::SubOverflow(operands->first.min, operands->second.max, minimum) ||
        llvm::SubOverflow(operands->first.max, operands->second.min, maximum))
      return failed(Failure::ArithmeticOverflow);
    return Result{StaticIndexRange{minimum, maximum, /*empty=*/false}};
  }

  Result evaluateMultiply(mlir::arith::MulIOp multiply) {
    Failure failure = Failure::None;
    auto operands =
        evaluateOperands(multiply.getLhs(), multiply.getRhs(), failure);
    if (!operands)
      return failed(failure);
    if (operands->first.empty || operands->second.empty)
      return Result{StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true}};

    const bool lhsSingleton = operands->first.min == operands->first.max;
    const bool rhsSingleton = operands->second.min == operands->second.max;
    if (!lhsSingleton && !rhsSingleton)
      return failed(Failure::NonSingletonMultiplication);
    const int64_t factor =
        lhsSingleton ? operands->first.min : operands->second.min;
    const StaticIndexRange varying =
        lhsSingleton ? operands->second : operands->first;
    int64_t first = 0;
    int64_t second = 0;
    if (llvm::MulOverflow(varying.min, factor, first) ||
        llvm::MulOverflow(varying.max, factor, second))
      return failed(Failure::ArithmeticOverflow);
    return Result{StaticIndexRange{std::min(first, second),
                                   std::max(first, second),
                                   /*empty=*/false}};
  }

  Result evaluateUnsignedDivide(mlir::arith::DivUIOp divide) {
    Failure failure = Failure::None;
    auto operands =
        evaluateOperands(divide.getLhs(), divide.getRhs(), failure);
    if (!operands)
      return failed(failure);
    if (operands->first.empty || operands->second.empty)
      return Result{StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true}};
    if (operands->first.min != operands->first.max ||
        operands->second.min != operands->second.max ||
        operands->first.min < 0 || operands->second.min <= 0)
      return failed(Failure::InvalidUnsignedDivision);

    const int64_t quotient = static_cast<int64_t>(
        static_cast<uint64_t>(operands->first.min) /
        static_cast<uint64_t>(operands->second.min));
    return Result{
        StaticIndexRange{quotient, quotient, /*empty=*/false}};
  }

  llvm::DenseMap<mlir::Value, Result> cache;
  llvm::DenseSet<mlir::Value> active;
};

} // namespace

StaticIndexRangeResult
evaluateNonNegativeStaticIndexRange(mlir::Value value) {
  StaticIndexRangeResult result = StaticIndexRangeEvaluator().evaluate(value);
  if (result.succeeded() && !result.range.empty && result.range.min < 0)
    result.failure = StaticIndexRangeFailureKind::NegativeRange;
  return result;
}

} // namespace wafer::memory_planning::detail
