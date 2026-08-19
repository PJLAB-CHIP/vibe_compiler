//===- StaticIndexRange.cpp - Static index range evaluation -----*- C++ -*-===//

#include "MemoryPlanning/StaticIndexRange.h"

#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
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
  explicit StaticIndexRangeEvaluator(mlir::Operation *use) {
    collectEnclosingBranchConstraints(use);
  }

  Result evaluate(mlir::Value value) {
    auto cached = cache.find(value);
    if (cached != cache.end())
      return cached->second;
    if (!active.insert(value).second)
      return failed(Failure::UnsupportedExpression);

    Result result = applyConstraint(value, evaluateImpl(value));
    active.erase(value);
    cache.try_emplace(value, result);
    return result;
  }

private:
  struct Constraint {
    std::optional<int64_t> minimum;
    std::optional<int64_t> maximum;
  };

  static mlir::arith::CmpIPredicate
  invertPredicate(mlir::arith::CmpIPredicate predicate) {
    using Predicate = mlir::arith::CmpIPredicate;
    switch (predicate) {
    case Predicate::eq:
      return Predicate::ne;
    case Predicate::ne:
      return Predicate::eq;
    case Predicate::slt:
      return Predicate::sge;
    case Predicate::sle:
      return Predicate::sgt;
    case Predicate::sgt:
      return Predicate::sle;
    case Predicate::sge:
      return Predicate::slt;
    case Predicate::ult:
      return Predicate::uge;
    case Predicate::ule:
      return Predicate::ugt;
    case Predicate::ugt:
      return Predicate::ule;
    case Predicate::uge:
      return Predicate::ult;
    }
    llvm_unreachable("unhandled integer comparison predicate");
  }

  static mlir::arith::CmpIPredicate
  swapPredicate(mlir::arith::CmpIPredicate predicate) {
    using Predicate = mlir::arith::CmpIPredicate;
    switch (predicate) {
    case Predicate::eq:
    case Predicate::ne:
      return predicate;
    case Predicate::slt:
      return Predicate::sgt;
    case Predicate::sle:
      return Predicate::sge;
    case Predicate::sgt:
      return Predicate::slt;
    case Predicate::sge:
      return Predicate::sle;
    case Predicate::ult:
      return Predicate::ugt;
    case Predicate::ule:
      return Predicate::uge;
    case Predicate::ugt:
      return Predicate::ult;
    case Predicate::uge:
      return Predicate::ule;
    }
    llvm_unreachable("unhandled integer comparison predicate");
  }

  void constrainMinimum(mlir::Value value, int64_t minimum) {
    Constraint &constraint = constraints[value];
    if (!constraint.minimum || minimum > *constraint.minimum)
      constraint.minimum = minimum;
  }

  void constrainMaximum(mlir::Value value, int64_t maximum) {
    Constraint &constraint = constraints[value];
    if (!constraint.maximum || maximum < *constraint.maximum)
      constraint.maximum = maximum;
  }

  void addComparisonConstraint(mlir::Value value,
                               mlir::arith::CmpIPredicate predicate,
                               int64_t constant) {
    using Predicate = mlir::arith::CmpIPredicate;
    switch (predicate) {
    case Predicate::eq:
      constrainMinimum(value, constant);
      constrainMaximum(value, constant);
      return;
    case Predicate::ne:
      return;
    case Predicate::slt:
    case Predicate::ult:
      if (constant != std::numeric_limits<int64_t>::min())
        constrainMaximum(value, constant - 1);
      return;
    case Predicate::sle:
    case Predicate::ule:
      constrainMaximum(value, constant);
      return;
    case Predicate::sgt:
    case Predicate::ugt:
      if (constant != std::numeric_limits<int64_t>::max())
        constrainMinimum(value, constant + 1);
      return;
    case Predicate::sge:
    case Predicate::uge:
      constrainMinimum(value, constant);
      return;
    }
    llvm_unreachable("unhandled integer comparison predicate");
  }

  void collectComparisonConstraint(mlir::Value condition, bool selected) {
    auto compare = condition.getDefiningOp<mlir::arith::CmpIOp>();
    if (!compare)
      return;
    mlir::arith::CmpIPredicate predicate = compare.getPredicate();
    if (!selected)
      predicate = invertPredicate(predicate);

    if (std::optional<int64_t> rhs =
            mlir::getConstantIntValue(compare.getRhs())) {
      addComparisonConstraint(compare.getLhs(), predicate, *rhs);
      return;
    }
    if (std::optional<int64_t> lhs =
            mlir::getConstantIntValue(compare.getLhs()))
      addComparisonConstraint(compare.getRhs(), swapPredicate(predicate), *lhs);
  }

  void collectEnclosingBranchConstraints(mlir::Operation *use) {
    if (!use)
      return;
    mlir::Operation *nested = use;
    while (mlir::Operation *parent = nested->getParentOp()) {
      if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(parent)) {
        if (nested->getBlock() == ifOp.thenBlock())
          collectComparisonConstraint(ifOp.getCondition(), /*selected=*/true);
        else if (nested->getBlock() == ifOp.elseBlock())
          collectComparisonConstraint(ifOp.getCondition(), /*selected=*/false);
      }
      nested = parent;
    }
  }

  Result applyConstraint(mlir::Value value, Result result) const {
    if (!result.succeeded() || result.range.empty)
      return result;
    auto it = constraints.find(value);
    if (it == constraints.end())
      return result;
    if (it->second.minimum)
      result.range.min = std::max(result.range.min, *it->second.minimum);
    if (it->second.maximum)
      result.range.max = std::min(result.range.max, *it->second.maximum);
    if (result.range.min > result.range.max)
      result.range = StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true};
    return result;
  }

  Result evaluateImpl(mlir::Value value) {
    if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
      return Result{StaticIndexRange{*constant, *constant, /*empty=*/false}};

    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (blockArg && blockArg.getOwner()) {
      if (mlir::Value entry =
              analysis::getSingleExecutionRegionEntryOperand(blockArg))
        return evaluate(entry);
    }
    if (auto opResult = mlir::dyn_cast<mlir::OpResult>(value)) {
      if (mlir::Value exit =
              analysis::getSingleExecutionRegionExitOperand(opResult))
        return evaluate(exit);
    }
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
    if (auto maximum = value.getDefiningOp<mlir::arith::MaxSIOp>())
      return evaluateSignedExtremum(maximum.getLhs(), maximum.getRhs(),
                                    /*takeMaximum=*/true);
    if (auto minimum = value.getDefiningOp<mlir::arith::MinSIOp>())
      return evaluateSignedExtremum(minimum.getLhs(), minimum.getRhs(),
                                    /*takeMaximum=*/false);
    // Keep the hand-written arithmetic cases above for precise failure
    // classification, then accept any other current-IR index expression whose
    // registered ValueBounds model proves a finite closed interval. This
    // covers affine.apply without treating an unbounded block argument as a
    // zero or guessed range.
    return evaluateInterfaceBounds(value);
  }

  Result evaluateInterfaceBounds(mlir::Value value) {
    if (!value.getType().isIndex())
      return failed(Failure::UnsupportedExpression);

    using mlir::ValueBoundsConstraintSet;
    using mlir::presburger::BoundType;
    ValueBoundsConstraintSet::Variable variable(value);
    mlir::FailureOr<int64_t> lower =
        ValueBoundsConstraintSet::computeConstantBound(
            BoundType::LB, variable, nullptr, /*closedUB=*/true);
    mlir::FailureOr<int64_t> upper =
        ValueBoundsConstraintSet::computeConstantBound(
            BoundType::UB, variable, nullptr, /*closedUB=*/true);
    if (mlir::failed(lower) || mlir::failed(upper) || *lower > *upper)
      return failed(Failure::UnsupportedExpression);
    return Result{StaticIndexRange{*lower, *upper, /*empty=*/false}};
  }

  Result evaluateLoopInductionVariable(mlir::scf::ForOp loop) {
    Result lowerRange = evaluate(loop.getLowerBound());
    Result upperRange = evaluate(loop.getUpperBound());
    Result stepRange = evaluate(loop.getStep());
    if (!lowerRange.succeeded() || !upperRange.succeeded() ||
        !stepRange.succeeded() || lowerRange.range.empty ||
        upperRange.range.empty || stepRange.range.empty ||
        lowerRange.range.min != lowerRange.range.max ||
        upperRange.range.min != upperRange.range.max ||
        stepRange.range.min != stepRange.range.max)
      return failed(Failure::DynamicLoopBounds);
    int64_t lower = lowerRange.range.min;
    int64_t upper = upperRange.range.min;
    int64_t step = stepRange.range.min;
    if (lower < 0 || upper < 0 || step <= 0)
      return failed(Failure::InvalidLoopBounds);
    if (upper <= lower)
      return Result{StaticIndexRange{lower, lower, /*empty=*/true}};

    const int64_t distance = upper - lower - 1;
    int64_t delta = 0;
    int64_t maximum = 0;
    if (llvm::MulOverflow(distance / step, step, delta) ||
        llvm::AddOverflow(lower, delta, maximum))
      return failed(Failure::ArithmeticOverflow);
    return Result{StaticIndexRange{lower, maximum, /*empty=*/false}};
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
    auto operands = evaluateOperands(divide.getLhs(), divide.getRhs(), failure);
    if (!operands)
      return failed(failure);
    if (operands->first.empty || operands->second.empty)
      return Result{StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true}};
    if (operands->first.min != operands->first.max ||
        operands->second.min != operands->second.max ||
        operands->first.min < 0 || operands->second.min <= 0)
      return failed(Failure::InvalidUnsignedDivision);

    const int64_t quotient =
        static_cast<int64_t>(static_cast<uint64_t>(operands->first.min) /
                             static_cast<uint64_t>(operands->second.min));
    return Result{StaticIndexRange{quotient, quotient, /*empty=*/false}};
  }

  Result evaluateSignedExtremum(mlir::Value lhsValue, mlir::Value rhsValue,
                                bool takeMaximum) {
    Failure failure = Failure::None;
    auto operands = evaluateOperands(lhsValue, rhsValue, failure);
    if (!operands)
      return failed(failure);
    if (operands->first.empty || operands->second.empty)
      return Result{StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true}};
    auto combine = [takeMaximum](int64_t lhs, int64_t rhs) {
      return takeMaximum ? std::max(lhs, rhs) : std::min(lhs, rhs);
    };
    return Result{StaticIndexRange{
        combine(operands->first.min, operands->second.min),
        combine(operands->first.max, operands->second.max),
        /*empty=*/false}};
  }

  llvm::DenseMap<mlir::Value, Result> cache;
  llvm::DenseSet<mlir::Value> active;
  llvm::DenseMap<mlir::Value, Constraint> constraints;
};

} // namespace

StaticIndexRangeResult
evaluateNonNegativeStaticIndexRange(mlir::Value value, mlir::Operation *use) {
  StaticIndexRangeResult result =
      StaticIndexRangeEvaluator(use).evaluate(value);
  if (result.succeeded() && !result.range.empty && result.range.min < 0)
    result.failure = StaticIndexRangeFailureKind::NegativeRange;
  return result;
}

} // namespace wafer::memory_planning::detail
