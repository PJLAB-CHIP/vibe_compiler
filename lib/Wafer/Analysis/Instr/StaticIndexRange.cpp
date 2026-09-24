//===- StaticIndexRange.cpp - Static index range evaluation -----*- C++ -*-===//

#include "StaticIndexRange.h"

#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Analysis/ControlFlow/StaticLoopDomain.h"
#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
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
    if (unreachable)
      return Result{StaticIndexRange{0, 0, /*empty=*/true}};
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

  static StaticIndexExecution proveExecution(mlir::Operation *operation,
                                             mlir::Operation *scope) {
    using Execution = StaticIndexExecution;
    if (!operation || !scope || !scope->isProperAncestor(operation))
      return Execution::Unknown;
    llvm::SmallVector<analysis::StaticLoopDomain> loops;
    llvm::SmallVector<std::pair<mlir::Value, bool>> paths;
    for (auto *child = operation; child->getParentOp() != scope;) {
      auto *parent = child->getParentOp();
      if (auto loop = analysis::getStaticLoopDomain(parent)) {
        loops.push_back(*loop);
      } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(parent)) {
        paths.emplace_back(branch.getCondition(),
                           child->getParentRegion() == &branch.getThenRegion());
      } else {
        auto flow = analysis::getSingleExecutionRegionFlow(parent);
        if (!flow || child->getParentRegion() != flow->region)
          return Execution::Unknown;
      }
      child = parent;
    }
    // Bounds merely propose one tuple of actual induction values. In a
    // deterministic static nest, a tuple satisfying every path condition is
    // a constructive must-execute witness, not sampled coverage of the nest.
    StaticIndexRangeEvaluator bounds(operation), point(nullptr);
    auto unproved = [&]() {
      return bounds.resourceExhausted || point.resourceExhausted ||
                     bounds.rangeWork.isExhausted() ||
                     point.rangeWork.isExhausted()
                 ? Execution::ResourceExhausted
                 : Execution::Unknown;
    };
    for (auto domain : loops) {
      if (!bounds.rangeWork.charge())
        return Execution::ResourceExhausted;
      auto iv = domain.loop.getInductionVar();
      auto candidate = bounds.evaluate(iv);
      if (!candidate.succeeded() || candidate.range.empty)
        return unproved();
      int64_t value = candidate.range.min;
      if (value < domain.lower || value >= domain.upper ||
          (value - domain.lower) % domain.step != 0)
        return Execution::Unknown;
      point.cache.try_emplace(iv,
                              Result{StaticIndexRange{value, value, false}});
    }
    // This evaluator has no path constraints, nor cached Boolean assumptions.
    // A runtime-dependent/unknown branch cannot establish the witness.
    for (auto [condition, selected] : paths)
      if (point.knownBoolean(condition, point) != selected)
        return unproved();
    return Execution::Proven;
  }

private:
  bool chargePredicate(unsigned depth) {
    if (depth >= 64) {
      resourceExhausted = true;
      return false;
    }
    return rangeWork.charge();
  }

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

  static void constrainMinimum(Constraint &constraint, int64_t minimum) {
    if (!constraint.minimum || minimum > *constraint.minimum)
      constraint.minimum = minimum;
  }

  static void constrainMaximum(Constraint &constraint, int64_t maximum) {
    if (!constraint.maximum || maximum < *constraint.maximum)
      constraint.maximum = maximum;
  }

  void addComparisonConstraint(mlir::Value value,
                               mlir::arith::CmpIPredicate predicate,
                               StaticIndexRange other) {
    using Predicate = mlir::arith::CmpIPredicate;
    bool isUnsigned =
        predicate == Predicate::ult || predicate == Predicate::ule ||
        predicate == Predicate::ugt || predicate == Predicate::uge;
    // The stored interval is signed. An unsigned comparison only refines it
    // when both operand intervals are known non-negative.
    if (other.empty || (isUnsigned && other.min < 0))
      return;
    Constraint &constraint =
        isUnsigned ? unsignedConstraints[value] : constraints[value];
    if (constrainedSet.insert(value).second)
      constrainedValues.push_back(value);
    switch (predicate) {
    case Predicate::eq:
      constrainMinimum(constraint, other.min);
      constrainMaximum(constraint, other.max);
      return;
    case Predicate::ne:
      return;
    case Predicate::slt:
    case Predicate::ult:
      if (other.max != std::numeric_limits<int64_t>::min())
        constrainMaximum(constraint, other.max - 1);
      return;
    case Predicate::sle:
    case Predicate::ule:
      constrainMaximum(constraint, other.max);
      return;
    case Predicate::sgt:
    case Predicate::ugt:
      if (other.min != std::numeric_limits<int64_t>::max())
        constrainMinimum(constraint, other.min + 1);
      return;
    case Predicate::sge:
    case Predicate::uge:
      constrainMinimum(constraint, other.min);
      return;
    }
    llvm_unreachable("unhandled integer comparison predicate");
  }

  // Evaluate Boolean structure only from proven current SSA intervals. This
  // also recognizes unreachable branches without enumerating loop instances.
  std::optional<bool> knownBoolean(mlir::Value condition,
                                   StaticIndexRangeEvaluator &unconditioned,
                                   unsigned depth = 0) {
    if (!chargePredicate(depth))
      return std::nullopt;
    auto found = booleans.find(condition);
    if (found != booleans.end())
      return found->second;
    std::optional<bool> result;
    if (auto constant = mlir::getConstantIntValue(condition)) {
      result = *constant != 0;
    } else if (auto compare = condition.getDefiningOp<mlir::arith::CmpIOp>()) {
      auto lhs = unconditioned.evaluate(compare.getLhs());
      auto rhs = unconditioned.evaluate(compare.getRhs());
      if (lhs.succeeded() && rhs.succeeded() && !lhs.range.empty &&
          !rhs.range.empty) {
        using P = mlir::arith::CmpIPredicate;
        auto predicate = compare.getPredicate();
        bool unsignedCompare = predicate == P::ult || predicate == P::ule ||
                               predicate == P::ugt || predicate == P::uge;
        if (!unsignedCompare || (lhs.range.min >= 0 && rhs.range.min >= 0)) {
          auto a = lhs.range, b = rhs.range;
          switch (predicate) {
          case P::eq:
          case P::ne:
            if (a.max < b.min || b.max < a.min)
              result = predicate == P::ne;
            else if (a.min == a.max && b.min == b.max)
              result = predicate == P::eq;
            break;
          case P::slt:
          case P::ult:
            if (a.max < b.min)
              result = true;
            else if (a.min >= b.max)
              result = false;
            break;
          case P::sle:
          case P::ule:
            if (a.max <= b.min)
              result = true;
            else if (a.min > b.max)
              result = false;
            break;
          case P::sgt:
          case P::ugt:
            if (a.min > b.max)
              result = true;
            else if (a.max <= b.min)
              result = false;
            break;
          case P::sge:
          case P::uge:
            if (a.min >= b.max)
              result = true;
            else if (a.max < b.min)
              result = false;
            break;
          }
        }
      }
    } else if (auto *op = condition.getDefiningOp();
               op && mlir::isa<mlir::arith::AndIOp, mlir::arith::OrIOp,
                               mlir::arith::XOrIOp>(op)) {
      auto lhs = knownBoolean(op->getOperand(0), unconditioned, depth + 1);
      auto rhs = knownBoolean(op->getOperand(1), unconditioned, depth + 1);
      if (mlir::isa<mlir::arith::AndIOp>(op)) {
        if (lhs == false || rhs == false)
          result = false;
        else if (lhs && rhs)
          result = *lhs && *rhs;
      } else if (mlir::isa<mlir::arith::OrIOp>(op)) {
        if (lhs == true || rhs == true)
          result = true;
        else if (lhs && rhs)
          result = *lhs || *rhs;
      } else if (lhs && rhs) {
        result = *lhs != *rhs;
      }
    }
    booleans.try_emplace(condition, result);
    return result;
  }

  void constrainComparison(mlir::Value value,
                           mlir::arith::CmpIPredicate predicate,
                           StaticIndexRange other,
                           StaticIndexRangeEvaluator &unconditioned,
                           unsigned depth = 0) {
    if (!chargePredicate(depth))
      return;
    addComparisonConstraint(value, predicate, other);
    // Affine lowering spells signed translations as arith.addi/subi and
    // multiplication by -1. Preserve the same bounds through these exact
    // identities, after checking that their index arithmetic cannot overflow.
    // Unsigned comparisons are not invariant under signed translation.
    using P = mlir::arith::CmpIPredicate;
    if (!value.getType().isIndex() || predicate == P::ult ||
        predicate == P::ule || predicate == P::ugt || predicate == P::uge)
      return;
    auto *op = value.getDefiningOp();
    if (!op || !mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                          mlir::arith::MulIOp>(op))
      return;
    auto range = unconditioned.evaluate(value);
    if (!range.succeeded() || range.range.empty || other.empty)
      return;
    auto lhs = mlir::getConstantIntValue(op->getOperand(0));
    auto rhs = mlir::getConstantIntValue(op->getOperand(1));
    if (!lhs && !rhs)
      return;
    mlir::Value variable = op->getOperand(rhs ? 0 : 1);
    int64_t constant = rhs ? *rhs : *lhs;
    int64_t offset = 0;
    bool negate = false;
    if (mlir::isa<mlir::arith::AddIOp>(op)) {
      offset = constant;
    } else if (mlir::isa<mlir::arith::SubIOp>(op)) {
      negate = !rhs;
      if (negate)
        offset = constant;
      else if (llvm::SubOverflow(int64_t{0}, constant, offset))
        return;
    } else {
      if (constant != 1 && constant != -1)
        return;
      negate = constant == -1;
    }
    StaticIndexRange translated;
    if (negate) {
      if (llvm::SubOverflow(offset, other.max, translated.min) ||
          llvm::SubOverflow(offset, other.min, translated.max))
        return;
      predicate = swapPredicate(predicate);
    } else if (llvm::SubOverflow(other.min, offset, translated.min) ||
               llvm::SubOverflow(other.max, offset, translated.max)) {
      return;
    }
    constrainComparison(variable, predicate, translated, unconditioned,
                        depth + 1);
  }

  void collectComparisonConstraint(mlir::Value condition, bool selected,
                                   StaticIndexRangeEvaluator &unconditioned,
                                   unsigned depth = 0) {
    if (!chargePredicate(depth))
      return;
    if (auto known = knownBoolean(condition, unconditioned)) {
      unreachable |= *known != selected;
      return;
    }
    auto *definition = condition.getDefiningOp();
    if (definition &&
        mlir::isa<mlir::arith::AndIOp, mlir::arith::OrIOp, mlir::arith::XOrIOp>(
            definition)) {
      auto lhs = definition->getOperand(0), rhs = definition->getOperand(1);
      bool conjunction = mlir::isa<mlir::arith::AndIOp>(definition);
      bool disjunction = mlir::isa<mlir::arith::OrIOp>(definition);
      // True AND and false OR imply both operands. The opposite selection
      // only refines an operand when its sibling is proved neutral.
      if ((conjunction && selected) || (disjunction && !selected)) {
        collectComparisonConstraint(lhs, selected, unconditioned, depth + 1);
        collectComparisonConstraint(rhs, selected, unconditioned, depth + 1);
      } else {
        for (unsigned i = 0; i < 2; ++i)
          if (auto known =
                  knownBoolean(definition->getOperand(1 - i), unconditioned)) {
            bool operandSelection = selected;
            if (!conjunction && !disjunction)
              operandSelection ^= *known;
            else if (*known != conjunction)
              continue;
            collectComparisonConstraint(definition->getOperand(i),
                                        operandSelection, unconditioned,
                                        depth + 1);
          }
      }
      return;
    }
    auto compare = condition.getDefiningOp<mlir::arith::CmpIOp>();
    if (!compare)
      return;
    mlir::arith::CmpIPredicate predicate = compare.getPredicate();
    if (!selected)
      predicate = invertPredicate(predicate);

    Result rhs = unconditioned.evaluate(compare.getRhs());
    if (rhs.succeeded())
      constrainComparison(compare.getLhs(), predicate, rhs.range,
                          unconditioned);
    Result lhs = unconditioned.evaluate(compare.getLhs());
    if (lhs.succeeded())
      constrainComparison(compare.getRhs(), swapPredicate(predicate), lhs.range,
                          unconditioned);
  }

  void collectEnclosingBranchConstraints(mlir::Operation *use) {
    if (!use)
      return;
    // Process outer paths first. Each condition may use facts from its
    // already selected enclosing paths, never facts inferred from itself.
    // Reset value/Boolean caches at each constraint epoch.
    llvm::SmallVector<std::pair<mlir::Value, bool>> paths;
    mlir::Operation *nested = use;
    while (mlir::Operation *parent = nested->getParentOp()) {
      if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(parent)) {
        if (nested->getBlock() == ifOp.thenBlock())
          paths.push_back({ifOp.getCondition(), true});
        else if (nested->getBlock() == ifOp.elseBlock())
          paths.push_back({ifOp.getCondition(), false});
      }
      nested = parent;
    }
    StaticIndexRangeEvaluator prior(nullptr);
    for (auto [condition, selected] : llvm::reverse(paths)) {
      if (!rangeWork.charge(1 + constrainedValues.size()))
        break;
      prior.constraints = constraints;
      prior.unsignedConstraints = unsignedConstraints;
      prior.constrainedValues = constrainedValues;
      prior.cache.clear();
      booleans.clear();
      collectComparisonConstraint(condition, selected, prior);
      resourceExhausted |=
          prior.resourceExhausted || prior.rangeWork.isExhausted();
      if (unreachable)
        break;
    }
  }

  Result applyConstraint(mlir::Value value, Result result) {
    if (!result.succeeded() || result.range.empty)
      return result;
    auto apply = [&](const auto &bounds) {
      auto it = bounds.find(value);
      if (it == bounds.end())
        return;
      if (it->second.minimum)
        result.range.min = std::max(result.range.min, *it->second.minimum);
      if (it->second.maximum)
        result.range.max = std::min(result.range.max, *it->second.maximum);
    };
    apply(constraints);
    if (result.range.min >= 0)
      apply(unsignedConstraints);
    // Guard and address materializations may be distinct affine.apply SSA
    // values. Recover only a proven constant signed difference (or sum),
    // with actual operand bindings; no operation order or spelling is used.
    if (value.getType().isIndex() && result.range.min != result.range.max)
      for (mlir::Value guarded : constrainedValues) {
        if (guarded == value || !guarded.getType().isIndex())
          continue;
        auto addressSize = boundedAffineCone(value);
        auto guardSize = boundedAffineCone(guarded);
        if (!addressSize || !guardSize)
          continue;
        auto found = constraints.find(guarded);
        if (found == constraints.end())
          continue;
        auto identity =
            mlir::AffineMap::getMultiDimIdentityMap(1, value.getContext());
        mlir::affine::AffineValueMap address(identity, value);
        for (int64_t sign : {int64_t{1}, int64_t{-1}}) {
          if (affineWork == 0)
            break;
          --affineWork;
          uint64_t nodes = addressSize->nodes + guardSize->nodes + 4;
          if (!rangeWork.charge(nodes * nodes))
            break;
          auto map = mlir::AffineMap::get(
              1, 0, mlir::getAffineDimExpr(0, value.getContext()) * sign);
          mlir::affine::AffineValueMap other(map, guarded);
          address.composeSimplifyAndCanonicalize();
          other.composeSimplifyAndCanonicalize();
          llvm::SmallVector<mlir::Value> bindings;
          auto bind = [&](mlir::affine::AffineValueMap &valueMap) {
            llvm::SmallVector<mlir::AffineExpr> replacements;
            for (mlir::Value operand : valueMap.getOperands()) {
              auto found = llvm::find(bindings, operand);
              unsigned position = std::distance(bindings.begin(), found);
              if (found == bindings.end())
                bindings.push_back(operand);
              replacements.push_back(
                  mlir::getAffineDimExpr(position, value.getContext()));
            }
            return valueMap.getResult(0).replaceDimsAndSymbols(
                llvm::ArrayRef(replacements).take_front(valueMap.getNumDims()),
                llvm::ArrayRef(replacements).drop_front(valueMap.getNumDims()));
          };
          auto lhs = bind(address), rhs = bind(other);
          auto offset = mlir::dyn_cast<mlir::AffineConstantExpr>(
              mlir::simplifyAffineExpr(lhs - rhs, bindings.size(), 0));
          if (!offset)
            continue;
          const auto &bounds = found->second;
          auto translate =
              [&](std::optional<int64_t> bound) -> std::optional<int64_t> {
            int64_t scaled, translated;
            if (!bound || llvm::MulOverflow(*bound, sign, scaled) ||
                llvm::AddOverflow(scaled, offset.getValue(), translated))
              return std::nullopt;
            return translated;
          };
          if (auto lower =
                  translate(sign > 0 ? bounds.minimum : bounds.maximum))
            result.range.min = std::max(result.range.min, *lower);
          if (auto upper =
                  translate(sign > 0 ? bounds.maximum : bounds.minimum))
            result.range.max = std::min(result.range.max, *upper);
          break;
        }
      }
    if (result.range.min > result.range.max)
      result.range = StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true};
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    auto loop = argument ? mlir::dyn_cast<mlir::scf::ForOp>(
                               argument.getOwner()->getParentOp())
                         : mlir::scf::ForOp{};
    if (result.range.empty || !loop || value != loop.getInductionVar())
      return result;
    Result step = evaluate(loop.getStep());
    if (!step.succeeded() || step.range.empty || step.range.min <= 0 ||
        step.range.min != step.range.max)
      return result;
    const int64_t stride = step.range.min;
    Result lower = evaluate(loop.getLowerBound());
    std::optional<uint64_t> residue;
    if (lower.succeeded() && !lower.range.empty && lower.range.min >= 0 &&
        lower.range.min == lower.range.max)
      residue = lower.range.min % stride;
    else
      residue = getKnownIndexRemainder(loop.getLowerBound(), stride);
    if (!residue || result.range.min < 0)
      return result;
    const int64_t expected = static_cast<int64_t>(*residue);
    const int64_t remainder = result.range.max % stride;
    const int64_t adjustment = remainder >= expected
                                   ? remainder - expected
                                   : stride - (expected - remainder);
    result.range.max -= adjustment;
    if (result.range.max < result.range.min) {
      result.range = StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true};
      return result;
    }
    // Derive the first reachable point from the rounded maximum. This cannot
    // overflow and also works for non-power-of-two constant steps.
    result.range.min =
        result.range.max -
        ((result.range.max - result.range.min) / stride) * stride;
    return result;
  }

  struct AffineComplexity {
    uint64_t norm;
    uint64_t nodes;
  };

  std::optional<AffineComplexity> boundedAffineCone(mlir::Value value,
                                                    unsigned depth = 0) {
    auto cached = affineComplexity.find(value);
    if (cached != affineComplexity.end())
      return cached->second;
    if (depth >= 16 || affineWork == 0) {
      resourceExhausted = true;
      return std::nullopt;
    }
    --affineWork;
    constexpr uint64_t limit = uint64_t{1} << 50;
    constexpr uint64_t nodeLimit = 512;
    if (auto constant = mlir::getConstantIntValue(value)) {
      if (*constant < -int64_t(limit) || *constant > int64_t(limit))
        return std::nullopt;
      return AffineComplexity{uint64_t(*constant < 0 ? -*constant : *constant),
                              1};
    }
    auto apply = value.getDefiningOp<mlir::affine::AffineApplyOp>();
    if (!apply)
      return AffineComplexity{1, 1};
    llvm::SmallVector<AffineComplexity> operands;
    for (mlir::Value operand : apply.getMapOperands()) {
      auto complexity = boundedAffineCone(operand, depth + 1);
      if (!complexity)
        return std::nullopt;
      operands.push_back(*complexity);
    }
    std::function<std::optional<AffineComplexity>(mlir::AffineExpr, unsigned)>
        visit = [&](mlir::AffineExpr expression,
                    unsigned nesting) -> std::optional<AffineComplexity> {
      if (nesting >= 32 || affineWork == 0) {
        resourceExhausted = true;
        return std::nullopt;
      }
      --affineWork;
      if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
        if (c.getValue() < -int64_t(limit) || c.getValue() > int64_t(limit))
          return std::nullopt;
        return AffineComplexity{
            uint64_t(c.getValue() < 0 ? -c.getValue() : c.getValue()), 1};
      }
      if (auto d = mlir::dyn_cast<mlir::AffineDimExpr>(expression))
        return operands[d.getPosition()];
      if (auto s = mlir::dyn_cast<mlir::AffineSymbolExpr>(expression))
        return operands[apply.getAffineMap().getNumDims() + s.getPosition()];
      auto binary = mlir::cast<mlir::AffineBinaryOpExpr>(expression);
      auto lhs = visit(binary.getLHS(), nesting + 1),
           rhs = visit(binary.getRHS(), nesting + 1);
      if (!lhs || !rhs || lhs->nodes + rhs->nodes + 1 > nodeLimit)
        return std::nullopt;
      uint64_t norm;
      if (expression.getKind() == mlir::AffineExprKind::Mul) {
        if (rhs->norm && lhs->norm > limit / rhs->norm)
          return std::nullopt;
        norm = lhs->norm * rhs->norm;
      } else {
        if (lhs->norm > limit - rhs->norm)
          return std::nullopt;
        norm = lhs->norm + rhs->norm;
      }
      return AffineComplexity{norm, lhs->nodes + rhs->nodes + 1};
    };
    auto complexity = visit(apply.getAffineMap().getResult(0), 0);
    if (complexity)
      affineComplexity.try_emplace(value, *complexity);
    return complexity;
  }

  Result evaluateImpl(mlir::Value value) {
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

    // Integer values have finite-width, signed/unsigned semantics. In
    // particular, an index_cast must not lose a clamp on a runtime integer or
    // treat a preceding truncation/wrapping add as mathematical arithmetic.
    if (mlir::isa<mlir::IntegerType>(value.getType()) ||
        value.getDefiningOp<mlir::arith::IndexCastOp>() ||
        value.getDefiningOp<mlir::arith::IndexCastUIOp>())
      return evaluateIntegerRange(value);
    if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
      return Result{StaticIndexRange{*constant, *constant, /*empty=*/false}};

    if (auto add = value.getDefiningOp<mlir::arith::AddIOp>())
      return evaluateAdd(add);
    if (auto subtract = value.getDefiningOp<mlir::arith::SubIOp>())
      return evaluateSubtract(subtract);
    if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>())
      return evaluateMultiply(multiply);
    if (auto divide = value.getDefiningOp<mlir::arith::DivUIOp>())
      return evaluateUnsignedDivide(divide);
    if (auto divide = value.getDefiningOp<mlir::arith::CeilDivSIOp>())
      return evaluateSignedCeilDivide(divide);
    if (auto maximum = value.getDefiningOp<mlir::arith::MaxSIOp>())
      return evaluateSignedExtremum(maximum.getLhs(), maximum.getRhs(),
                                    /*takeMaximum=*/true);
    if (auto minimum = value.getDefiningOp<mlir::arith::MinSIOp>())
      return evaluateSignedExtremum(minimum.getLhs(), minimum.getRhs(),
                                    /*takeMaximum=*/false);
    if (auto apply = value.getDefiningOp<mlir::affine::AffineApplyOp>())
      return evaluateAffineApply(apply);
    // Keep the hand-written arithmetic cases above for precise failure
    // classification, then accept any other current-IR index expression whose
    // registered ValueBounds model proves a finite closed interval. This
    // covers affine.apply without treating an unbounded block argument as a
    // zero or guessed range.
    return evaluateInterfaceBounds(value);
  }

  Result evaluateIntegerRange(mlir::Value root) {
    using mlir::ConstantIntRanges;
    auto getModel = [](mlir::Value value) -> mlir::InferIntRangeInterface {
      mlir::Operation *op = value.getDefiningOp();
      if (!op || op->getNumRegions() != 0 ||
          (!mlir::isa<mlir::IntegerType>(value.getType()) &&
           !mlir::isa<mlir::arith::IndexCastOp, mlir::arith::IndexCastUIOp>(
               op)) ||
          !llvm::all_of(op->getOperandTypes(),
                        [](mlir::Type type) { return type.isIntOrIndex(); }))
        return {};
      return mlir::dyn_cast<mlir::InferIntRangeInterface>(op);
    };

    // Unknown leaves retain the full type range. The regionless SSA cone is
    // acyclic; an explicit postorder stack avoids recursion on long cast/index
    // chains. Shared operands are inferred once per query, not once per path.
    llvm::DenseMap<mlir::Value, ConstantIntRanges> ranges;
    llvm::SmallVector<std::pair<mlir::Value, bool>, 16> pending{{root, false}};
    while (!pending.empty()) {
      auto [value, ready] = pending.pop_back_val();
      if (ranges.contains(value))
        continue;
      auto model = getModel(value);
      if (model && !ready) {
        pending.emplace_back(value, true);
        for (mlir::Value operand : model->getOperands())
          if (!ranges.contains(operand))
            pending.emplace_back(operand, false);
        continue;
      }
      auto maximum = ConstantIntRanges::maxRange(
          ConstantIntRanges::getStorageBitwidth(value.getType()));
      if (model) {
        llvm::SmallVector<ConstantIntRanges, 4> operands;
        for (mlir::Value operand : model->getOperands())
          operands.push_back(ranges.find(operand)->second);
        model.inferResultRanges(
            operands, [&](mlir::Value result, const ConstantIntRanges &range) {
              if (result == value)
                maximum = range;
            });
      }
      ranges.try_emplace(value, std::move(maximum));
    }
    const ConstantIntRanges &range = ranges.find(root)->second;
    if (!range.smin().isSignedIntN(64) || !range.smax().isSignedIntN(64))
      return failed(Failure::UnsupportedExpression);
    return Result{StaticIndexRange{range.smin().getSExtValue(),
                                   range.smax().getSExtValue(),
                                   /*empty=*/false}};
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
        upperRange.range.empty || stepRange.range.empty)
      return failed(Failure::DynamicLoopBounds);
    int64_t lower = lowerRange.range.min;
    int64_t upper = upperRange.range.max;
    int64_t step = stepRange.range.min;
    if (lower < 0 || upper < 0 || step <= 0)
      return failed(Failure::InvalidLoopBounds);
    if (upper <= lower)
      return Result{StaticIndexRange{lower, lower, /*empty=*/true}};
    if (stepRange.range.min != stepRange.range.max)
      return Result{StaticIndexRange{lower, upper - 1, /*empty=*/false}};

    if (lowerRange.range.min != lowerRange.range.max) {
      int64_t maximum = upper - 1;
      // The lower bound may vary between invocations. An interval alone does
      // not prove a common induction grid; only an actual SSA congruence does.
      if (auto residue = getKnownIndexRemainder(loop.getLowerBound(), step)) {
        int64_t remainder = maximum % step;
        int64_t expected = static_cast<int64_t>(*residue);
        int64_t adjustment = remainder >= expected
                                 ? remainder - expected
                                 : step - (expected - remainder);
        maximum -= adjustment;
      }
      return Result{StaticIndexRange{lower, maximum, maximum < lower}};
    }

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

    int64_t minimum = std::numeric_limits<int64_t>::max();
    int64_t maximum = std::numeric_limits<int64_t>::min();
    for (int64_t lhs : {operands->first.min, operands->first.max})
      for (int64_t rhs : {operands->second.min, operands->second.max}) {
        int64_t product = 0;
        if (llvm::MulOverflow(lhs, rhs, product))
          return failed(Failure::ArithmeticOverflow);
        minimum = std::min(minimum, product);
        maximum = std::max(maximum, product);
      }
    return Result{StaticIndexRange{minimum, maximum, /*empty=*/false}};
  }

  Result evaluateUnsignedDivide(mlir::arith::DivUIOp divide) {
    Failure failure = Failure::None;
    auto operands = evaluateOperands(divide.getLhs(), divide.getRhs(), failure);
    if (!operands)
      return failed(failure);
    if (operands->first.empty || operands->second.empty)
      return Result{StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true}};
    if (operands->second.min != operands->second.max ||
        operands->first.min < 0 || operands->second.min <= 0)
      return failed(Failure::InvalidUnsignedDivision);

    const uint64_t divisor = operands->second.min;
    return Result{StaticIndexRange{
        static_cast<int64_t>(static_cast<uint64_t>(operands->first.min) /
                             divisor),
        static_cast<int64_t>(static_cast<uint64_t>(operands->first.max) /
                             divisor),
        /*empty=*/false}};
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
    return Result{
        StaticIndexRange{combine(operands->first.min, operands->second.min),
                         combine(operands->first.max, operands->second.max),
                         /*empty=*/false}};
  }

  Result evaluateSignedCeilDivide(mlir::arith::CeilDivSIOp divide) {
    Failure failure = Failure::None;
    auto operands = evaluateOperands(divide.getLhs(), divide.getRhs(), failure);
    if (!operands)
      return failed(failure);
    if (operands->first.empty || operands->second.empty)
      return Result{StaticIndexRange{0, 0, /*empty=*/true}};
    if (operands->second.min != operands->second.max ||
        operands->second.min <= 0)
      return failed(Failure::InvalidSignedDivision);
    int64_t divisor = operands->second.min;
    auto ceil = [divisor](int64_t numerator) {
      // Avoid the overflowing numerator + divisor - 1 formulation.
      return numerator / divisor + (numerator % divisor > 0);
    };
    return Result{StaticIndexRange{ceil(operands->first.min),
                                   ceil(operands->first.max), false}};
  }

  Result evaluateAffineApply(mlir::affine::AffineApplyOp apply) {
    mlir::AffineMap map = apply.getAffineMap();
    if (map.getNumResults() != 1)
      return evaluateInterfaceBounds(apply.getResult());
    mlir::OperandRange operands = apply.getMapOperands();
    std::function<Result(mlir::AffineExpr)> evaluateExpr =
        [&](mlir::AffineExpr expr) -> Result {
      if (auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
        int64_t value = constant.getValue();
        return Result{StaticIndexRange{value, value, /*empty=*/false}};
      }
      unsigned operand = 0;
      if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
        operand = dim.getPosition();
      } else if (auto symbol = mlir::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
        operand = map.getNumDims() + symbol.getPosition();
      } else {
        auto binary = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
        if (!binary)
          return failed(Failure::UnsupportedExpression);
        Result lhs = evaluateExpr(binary.getLHS());
        Result rhs = evaluateExpr(binary.getRHS());
        if (!lhs.succeeded())
          return lhs;
        if (!rhs.succeeded())
          return rhs;
        if (lhs.range.empty || rhs.range.empty)
          return Result{StaticIndexRange{/*min=*/0, /*max=*/0, /*empty=*/true}};
        if (expr.getKind() == mlir::AffineExprKind::Add) {
          int64_t minimum = 0;
          int64_t maximum = 0;
          if (llvm::AddOverflow(lhs.range.min, rhs.range.min, minimum) ||
              llvm::AddOverflow(lhs.range.max, rhs.range.max, maximum))
            return failed(Failure::ArithmeticOverflow);
          return Result{StaticIndexRange{minimum, maximum, /*empty=*/false}};
        }
        if (expr.getKind() == mlir::AffineExprKind::FloorDiv ||
            expr.getKind() == mlir::AffineExprKind::CeilDiv ||
            expr.getKind() == mlir::AffineExprKind::Mod) {
          if (rhs.range.min <= 0 || rhs.range.min != rhs.range.max)
            return failed(Failure::InvalidSignedDivision);
          int64_t divisor = rhs.range.min;
          if (expr.getKind() == mlir::AffineExprKind::FloorDiv)
            return Result{StaticIndexRange{
                llvm::divideFloorSigned(lhs.range.min, divisor),
                llvm::divideFloorSigned(lhs.range.max, divisor), false}};
          if (expr.getKind() == mlir::AffineExprKind::CeilDiv)
            return Result{StaticIndexRange{
                llvm::divideCeilSigned(lhs.range.min, divisor),
                llvm::divideCeilSigned(lhs.range.max, divisor), false}};
          return Result{StaticIndexRange{0, divisor - 1, false}};
        }
        if (expr.getKind() != mlir::AffineExprKind::Mul)
          return failed(Failure::UnsupportedExpression);
        const bool lhsSingleton = lhs.range.min == lhs.range.max;
        const bool rhsSingleton = rhs.range.min == rhs.range.max;
        if (!lhsSingleton && !rhsSingleton)
          return failed(Failure::NonSingletonMultiplication);
        const int64_t factor = lhsSingleton ? lhs.range.min : rhs.range.min;
        const StaticIndexRange varying = lhsSingleton ? rhs.range : lhs.range;
        int64_t first = 0;
        int64_t second = 0;
        if (llvm::MulOverflow(varying.min, factor, first) ||
            llvm::MulOverflow(varying.max, factor, second))
          return failed(Failure::ArithmeticOverflow);
        return Result{StaticIndexRange{std::min(first, second),
                                       std::max(first, second),
                                       /*empty=*/false}};
      }
      if (operand >= operands.size())
        return failed(Failure::UnsupportedExpression);
      return evaluate(operands[operand]);
    };
    Result result = evaluateExpr(map.getResult(0));
    if (!result.succeeded() || result.range.empty ||
        result.range.min == result.range.max)
      return result;
    bool division = false;
    map.getResult(0).walk([&](mlir::AffineExpr expression) {
      division |= expression.getKind() == mlir::AffineExprKind::FloorDiv ||
                  expression.getKind() == mlir::AffineExprKind::CeilDiv ||
                  expression.getKind() == mlir::AffineExprKind::Mod;
    });
    if (!division || !boundedAffineCone(apply.getResult()))
      return result;
    llvm::SmallVector<analysis::ClosedIndexInterval> ranges;
    llvm::SmallVector<mlir::Value> bindings;
    llvm::SmallVector<mlir::AffineExpr> replacements;
    for (mlir::Value operand : operands) {
      auto found = llvm::find(bindings, operand);
      unsigned position = std::distance(bindings.begin(), found);
      if (found == bindings.end()) {
        auto range = evaluate(operand);
        if (!range.succeeded() || range.range.empty)
          return result;
        bindings.push_back(operand);
        ranges.push_back({range.range.min, range.range.max});
      }
      replacements.push_back(
          mlir::getAffineDimExpr(position, apply.getContext()));
    }
    auto expression = map.getResult(0).replaceDimsAndSymbols(
        llvm::ArrayRef(replacements).take_front(map.getNumDims()),
        llvm::ArrayRef(replacements).drop_front(map.getNumDims()));
    auto interval = analysis::boundIndexExpression(
        mlir::AffineMap::get(bindings.size(), 0, expression), ranges,
        rangeWork);
    resourceExhausted |=
        interval.status == analysis::IndexRelationStatus::ResourceExhausted;
    if (interval.status == analysis::IndexRelationStatus::SoundBound) {
      result.range.min = std::max(result.range.min, interval.interval->minimum);
      result.range.max = std::min(result.range.max, interval.interval->maximum);
    }
    return result;
  }

  llvm::DenseMap<mlir::Value, Result> cache;
  llvm::DenseSet<mlir::Value> active;
  llvm::DenseMap<mlir::Value, Constraint> constraints;
  llvm::DenseMap<mlir::Value, Constraint> unsignedConstraints;
  llvm::SmallVector<mlir::Value> constrainedValues;
  llvm::DenseSet<mlir::Value> constrainedSet;
  llvm::DenseMap<mlir::Value, AffineComplexity> affineComplexity;
  bool unreachable = false;
  bool resourceExhausted = false;
  llvm::DenseMap<mlir::Value, std::optional<bool>> booleans;
  uint64_t affineWork = 4096;
  analysis::IndexRelationWork rangeWork{analysis::IndexRelationLimits{}};
};

class IndexRemainderEvaluator {
public:
  std::optional<uint64_t> evaluate(mlir::Value value, uint64_t modulus) {
    if (!value || !value.getType().isIndex() || !llvm::isPowerOf2_64(modulus))
      return std::nullopt;
    if (modulus == 1)
      return 0;
    auto key = std::make_pair(value, modulus);
    if (auto found = cache.find(key); found != cache.end())
      return found->second;
    if (!active.insert(key).second)
      return std::nullopt;
    auto result = evaluateImpl(value, modulus);
    active.erase(key);
    cache.try_emplace(key, result);
    return result;
  }

private:
  std::optional<uint64_t> evaluateImpl(mlir::Value value, uint64_t modulus) {
    const uint64_t mask = modulus - 1;
    if (auto constant = mlir::getConstantIntValue(value))
      return static_cast<uint64_t>(*constant) & mask;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto *owner = argument.getOwner()->getParentOp();
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(owner)) {
        if (value == loop.getInductionVar() &&
            evaluate(loop.getStep(), modulus) == std::optional<uint64_t>(0))
          return evaluate(loop.getLowerBound(), modulus);
      }
      if (auto region = mlir::dyn_cast<TileRegionOp>(owner))
        return evaluate(region.getInputs()[argument.getArgNumber()], modulus);
    }
    auto *op = value.getDefiningOp();
    if (!op || op->getNumOperands() != 2)
      return std::nullopt;
    auto lhs = evaluate(op->getOperand(0), modulus);
    auto rhs = evaluate(op->getOperand(1), modulus);
    if (mlir::isa<mlir::arith::MulIOp>(op)) {
      if (lhs && rhs)
        return (*lhs * *rhs) & mask;
      auto factor = lhs ? lhs : rhs;
      if (!factor)
        return std::nullopt;
      auto varying = lhs ? op->getOperand(1) : op->getOperand(0);
      auto reduced = evaluate(varying, modulus / std::gcd(*factor, modulus));
      return reduced ? std::optional<uint64_t>((*factor * *reduced) & mask)
                     : std::nullopt;
    }
    if (!lhs || !rhs)
      return std::nullopt;
    if (mlir::isa<mlir::arith::AddIOp>(op))
      return (*lhs + *rhs) & mask;
    if (mlir::isa<mlir::arith::SubIOp>(op))
      return (*lhs - *rhs) & mask;
    if (mlir::isa<mlir::arith::MinSIOp, mlir::arith::MaxSIOp,
                  mlir::arith::MinUIOp, mlir::arith::MaxUIOp>(op) &&
        lhs == rhs)
      return lhs;
    return std::nullopt;
  }
  llvm::DenseMap<std::pair<mlir::Value, uint64_t>, std::optional<uint64_t>>
      cache;
  llvm::DenseSet<std::pair<mlir::Value, uint64_t>> active;
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

StaticIndexExecution proveStaticIndexExecution(mlir::Operation *operation,
                                               mlir::Operation *scope) {
  return StaticIndexRangeEvaluator::proveExecution(operation, scope);
}

std::optional<uint64_t> getKnownIndexRemainder(mlir::Value value,
                                               uint64_t modulus) {
  return IndexRemainderEvaluator().evaluate(value, modulus);
}

bool proveNonEmptyLoop(mlir::scf::ForOp loop) {
  StaticIndexRangeEvaluator evaluator(loop);
  auto lower = evaluator.evaluate(loop.getLowerBound());
  auto upper = evaluator.evaluate(loop.getUpperBound());
  auto step = evaluator.evaluate(loop.getStep());
  return lower.succeeded() && upper.succeeded() && step.succeeded() &&
         !lower.range.empty && !upper.range.empty && !step.range.empty &&
         step.range.min > 0 && lower.range.max < upper.range.min;
}

mlir::LogicalResult proveByteAlignedPackedView(mlir::Value view) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(view.getType());
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (!type || !type.getElementType().isInteger(1) ||
      mlir::failed(mlir::getStridesAndOffset(type, strides, offset)))
    return mlir::failure();
  if (!mlir::ShapedType::isDynamic(offset))
    return mlir::success(offset >= 0 && offset % 8 == 0);
  if (auto cast = view.getDefiningOp<mlir::memref::CastOp>())
    return proveByteAlignedPackedView(cast.getSource());
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(view)) {
    if (auto region =
            mlir::dyn_cast<TileRegionOp>(argument.getOwner()->getParentOp()))
      return proveByteAlignedPackedView(
          region.getInputs()[argument.getArgNumber()]);
  }
  auto subview = view.getDefiningOp<mlir::memref::SubViewOp>();
  if (!subview ||
      mlir::failed(proveByteAlignedPackedView(subview.getSource())) ||
      mlir::failed(
          mlir::getStridesAndOffset(subview.getSourceType(), strides, offset)))
    return mlir::failure();
  IndexRemainderEvaluator evaluator;
  uint64_t remainder = 0;
  for (auto [index, stride] :
       llvm::zip_equal(subview.getMixedOffsets(), strides)) {
    if (mlir::ShapedType::isDynamic(stride) || stride < 0)
      return mlir::failure();
    if (auto constant = mlir::getConstantIntValue(index)) {
      remainder = (remainder + static_cast<uint64_t>(*constant) * stride) % 8;
      continue;
    }
    uint64_t modulus = 8 / std::gcd(static_cast<uint64_t>(stride), uint64_t(8));
    auto part = evaluator.evaluate(mlir::cast<mlir::Value>(index), modulus);
    if (!part)
      return mlir::failure();
    remainder = (remainder + *part * stride) % 8;
  }
  return mlir::success(remainder == 0);
}

} // namespace wafer::memory_planning::detail
