//===- TensorSubsetBlock.cpp - Guarded dense subset blocks ---------------===//

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"
#include "llvm/ADT/bit.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <numeric>

namespace wafer::analysis {
namespace {

struct Interval {
  int64_t lower, upper;
};

struct BlockExpression {
  mlir::AffineExpr base;
  llvm::SmallVector<int64_t, 4> coefficients;
};

struct BlockBounds {
  mlir::AffineExpr minimum, maximum;
};

/// Recover b + A*delta recursively. Arithmetic is checked before constructing
/// affine expressions; no sampled point or relaxed image authorizes a copy.
class BlockQuery {
public:
  BlockQuery(const TensorSubsetDemand &demand, llvm::ArrayRef<int64_t> shape,
             IndexRelationWork &work)
      : demand(demand), shape(shape), work(work),
        parameters(demand.parameters.size()),
        context(demand.domain.getContext()), blocking(shape.size()) {}

  // Every value is residue modulo stride; stride zero denotes a constant.
  // This bounded congruence proof avoids inventing impossible remainders for
  // row-major coordinates such as (4*p + u) mod 8 with fixed u=0.
  std::optional<std::pair<int64_t, int64_t>>
  congruence(mlir::AffineExpr expression, unsigned depth = 0) {
    if (!charge(depth))
      return std::nullopt;
    if (auto found = congruences.find(expression); found != congruences.end())
      return found->second;
    std::pair<int64_t, int64_t> result{1, 0};
    if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
      result = {0, c.getValue()};
    } else if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
      auto interval = range(expression);
      if (!interval)
        return std::nullopt;
      if (interval->lower == interval->upper)
        result = {0, interval->lower};
      else if (dim.getPosition() < parameters)
        result = {demand.parameters[dim.getPosition()].step, interval->lower};
    } else if (auto binary =
                   mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression)) {
      auto lhs = congruence(binary.getLHS(), depth + 1);
      auto rhs = congruence(binary.getRHS(), depth + 1);
      if (!lhs || !rhs)
        return std::nullopt;
      if (expression.getKind() == mlir::AffineExprKind::Add) {
        result.first = std::gcd(lhs->first, rhs->first);
        if (llvm::AddOverflow(lhs->second, rhs->second, result.second))
          return std::nullopt;
      } else if (rhs->first == 0) {
        int64_t factor = rhs->second;
        if (expression.getKind() == mlir::AffineExprKind::Mul) {
          if (factor == INT64_MIN ||
              llvm::MulOverflow(lhs->first, std::abs(factor), result.first) ||
              llvm::MulOverflow(lhs->second, factor, result.second))
            return std::nullopt;
        } else if (factor > 0) {
          if (expression.getKind() == mlir::AffineExprKind::Mod) {
            result = {std::gcd(lhs->first, factor), lhs->second};
          } else if (lhs->first % factor == 0) {
            result.first = lhs->first / factor;
            result.second =
                expression.getKind() == mlir::AffineExprKind::FloorDiv
                    ? llvm::divideFloorSigned(lhs->second, factor)
                    : llvm::divideCeilSigned(lhs->second, factor);
          }
        }
      }
    }
    if (result.first) {
      result.second %= result.first;
      if (result.second < 0)
        result.second += result.first;
    }
    congruences.try_emplace(expression, result);
    return result;
  }

  std::optional<Interval> range(mlir::AffineExpr expression,
                                unsigned depth = 0) {
    if (!charge(depth))
      return std::nullopt;
    if (auto found = ranges.find(expression); found != ranges.end())
      return found->second;
    std::optional<Interval> result;
    if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(expression))
      result = Interval{c.getValue(), c.getValue()};
    else if (auto d = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
      unsigned axis = d.getPosition();
      if (axis < parameters) {
        const auto &parameter = demand.parameters[axis];
        // Preserve the last actual IV. Using upper-1 invents inadmissible
        // tail origins and needlessly keeps source-bounds guards live.
        int64_t distance, last;
        if (parameter.step > 0 &&
            !llvm::SubOverflow(parameter.upper, parameter.lower, distance) &&
            distance > 0 &&
            !llvm::MulOverflow((distance - 1) / parameter.step, parameter.step,
                               last) &&
            !llvm::AddOverflow(parameter.lower, last, last))
          result = Interval{parameter.lower, last};
      } else if (axis - parameters < shape.size()) {
        result = Interval{0, demand.shape[axis - parameters] -
                                 shape[axis - parameters]};
      }
    } else if (auto binary =
                   mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression)) {
      auto lhs = range(binary.getLHS(), depth + 1);
      auto rhs = range(binary.getRHS(), depth + 1);
      if (!lhs || !rhs)
        return std::nullopt;
      int64_t lower, upper;
      switch (expression.getKind()) {
      case mlir::AffineExprKind::Add:
        if (!llvm::AddOverflow(lhs->lower, rhs->lower, lower) &&
            !llvm::AddOverflow(lhs->upper, rhs->upper, upper))
          result = Interval{lower, upper};
        break;
      case mlir::AffineExprKind::Mul:
        if (rhs->lower != rhs->upper)
          std::swap(lhs, rhs);
        if (rhs->lower == rhs->upper &&
            !llvm::MulOverflow(lhs->lower, rhs->lower, lower) &&
            !llvm::MulOverflow(lhs->upper, rhs->lower, upper))
          result = Interval{std::min(lower, upper), std::max(lower, upper)};
        break;
      case mlir::AffineExprKind::FloorDiv:
      case mlir::AffineExprKind::CeilDiv:
      case mlir::AffineExprKind::Mod:
        if (rhs->lower <= 0 || rhs->lower != rhs->upper)
          break;
        if (expression.getKind() == mlir::AffineExprKind::FloorDiv)
          result = Interval{llvm::divideFloorSigned(lhs->lower, rhs->lower),
                            llvm::divideFloorSigned(lhs->upper, rhs->lower)};
        else if (expression.getKind() == mlir::AffineExprKind::CeilDiv)
          result = Interval{llvm::divideCeilSigned(lhs->lower, rhs->lower),
                            llvm::divideCeilSigned(lhs->upper, rhs->lower)};
        else if (llvm::divideFloorSigned(lhs->lower, rhs->lower) ==
                 llvm::divideFloorSigned(lhs->upper, rhs->lower)) {
          auto remainder = [&](int64_t value) {
            int64_t r = value % rhs->lower;
            return r < 0 ? r + rhs->lower : r;
          };
          result = Interval{remainder(lhs->lower), remainder(lhs->upper)};
        } else {
          auto lattice = congruence(binary.getLHS());
          if (!lattice)
            return std::nullopt;
          int64_t step = std::gcd(lattice->first, rhs->lower);
          int64_t first = lattice->second % step;
          if (first < 0)
            first += step;
          result = Interval{first, rhs->lower - step + first};
        }
        break;
      default:
        break;
      }
    }
    if (result)
      ranges.try_emplace(expression, *result);
    return result;
  }

  bool charge(unsigned depth) {
    if (depth >= work.getLimits().maxVariables) {
      limited = true;
      return false;
    }
    return work.charge();
  }

  bool coefficient(int64_t value) {
    const llvm::DynamicAPInt maximum(
        int64_t(work.getLimits().maxAbsoluteCoefficient));
    if (llvm::abs(llvm::DynamicAPInt(value)) > maximum) {
      limited = true;
      return false;
    }
    return true;
  }

  std::optional<mlir::AffineExpr> add(mlir::AffineExpr a, mlir::AffineExpr b) {
    auto lhs = range(a), rhs = range(b);
    int64_t lower, upper;
    if (!lhs || !rhs || llvm::AddOverflow(lhs->lower, rhs->lower, lower) ||
        llvm::AddOverflow(lhs->upper, rhs->upper, upper))
      return std::nullopt;
    auto lhsNorm = norm(a), rhsNorm = norm(b);
    if (!lhsNorm || !rhsNorm || *lhsNorm > normLimit() - *rhsNorm) {
      limited = true;
      return std::nullopt;
    }
    auto expression = a + b;
    norms.try_emplace(expression, *lhsNorm + *rhsNorm);
    return expression;
  }

  std::optional<mlir::AffineExpr> scale(mlir::AffineExpr a, int64_t c) {
    auto bounds = range(a);
    int64_t lower, upper;
    if (!bounds || !coefficient(c) ||
        llvm::MulOverflow(bounds->lower, c, lower) ||
        llvm::MulOverflow(bounds->upper, c, upper))
      return std::nullopt;
    auto lhsNorm = norm(a);
    uint64_t factor = c < 0 ? uint64_t(-(c + 1)) + 1 : uint64_t(c);
    if (!lhsNorm || (factor && *lhsNorm > normLimit() / factor)) {
      limited = true;
      return std::nullopt;
    }
    auto expression = a * c;
    norms.try_emplace(expression, *lhsNorm * factor);
    return expression;
  }

  uint64_t normLimit() const {
    return std::min<uint64_t>(work.getLimits().maxAbsoluteCoefficient,
                              INT64_MAX / 4);
  }

  std::optional<uint64_t> norm(mlir::AffineExpr expression,
                               unsigned depth = 0) {
    if (!charge(depth))
      return std::nullopt;
    if (auto found = norms.find(expression); found != norms.end())
      return found->second;
    uint64_t value = 1;
    if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
      value = c.getValue() < 0 ? uint64_t(-(c.getValue() + 1)) + 1
                               : uint64_t(c.getValue());
    } else if (auto binary =
                   mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression)) {
      auto lhs = norm(binary.getLHS(), depth + 1);
      auto rhs = norm(binary.getRHS(), depth + 1);
      if (!lhs || !rhs)
        return std::nullopt;
      if (expression.getKind() == mlir::AffineExprKind::Mul) {
        if (*rhs && *lhs > normLimit() / *rhs) {
          limited = true;
          return std::nullopt;
        }
        value = *lhs * *rhs;
      } else {
        if (*lhs > normLimit() - *rhs) {
          limited = true;
          return std::nullopt;
        }
        value = *lhs + *rhs;
      }
    }
    if (value > normLimit()) {
      limited = true;
      return std::nullopt;
    }
    norms.try_emplace(expression, value);
    return value;
  }

  std::optional<mlir::AffineExpr> simplify(mlir::AffineExpr expression,
                                           bool fixedOrigins = false) {
    if (fixedOrigins) {
      llvm::SmallVector<mlir::AffineExpr> operands;
      for (unsigned i = 0; i < parameters + shape.size(); ++i) {
        auto dim = mlir::getAffineDimExpr(i, context);
        auto interval = range(dim);
        if (!interval)
          return std::nullopt;
        operands.push_back(interval->lower == interval->upper
                               ? constant(interval->lower)
                               : dim);
      }
      auto result = composeIndexMap(
          mlir::AffineMap::get(parameters + shape.size(), 0, expression),
          mlir::AffineMap::get(parameters + shape.size(), 0, operands, context),
          work);
      if (!result.isExact()) {
        limited |= result.status == IndexRelationStatus::ResourceExhausted;
        return std::nullopt;
      }
      return result.map.getResult(0);
    }
    auto result = composeIndexMap(
        mlir::AffineMap::getMultiDimIdentityMap(1, context),
        mlir::AffineMap::get(parameters + shape.size(), 0, expression), work);
    if (!result.isExact()) {
      limited |= result.status == IndexRelationStatus::ResourceExhausted;
      return std::nullopt;
    }
    return result.map.getResult(0);
  }

  mlir::AffineExpr constant(int64_t value) {
    return mlir::getAffineConstantExpr(value, context);
  }

  std::optional<BlockExpression> linearize(mlir::AffineExpr expression,
                                           unsigned depth = 0) {
    if (!charge(depth))
      return std::nullopt;
    if (auto found = expressions.find(expression); found != expressions.end())
      return found->second;
    BlockExpression result{expression,
                           llvm::SmallVector<int64_t, 4>(shape.size(), 0)};
    if (auto d = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
      if (d.getPosition() >= parameters)
        result.coefficients[d.getPosition() - parameters] = 1;
    } else if (!mlir::isa<mlir::AffineConstantExpr>(expression)) {
      auto binary = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression);
      if (!binary)
        return std::nullopt;
      auto lhs = linearize(binary.getLHS(), depth + 1);
      if (!lhs)
        return std::nullopt;
      if (expression.getKind() == mlir::AffineExprKind::Add) {
        auto rhs = linearize(binary.getRHS(), depth + 1);
        auto base = rhs ? add(lhs->base, rhs->base) : std::nullopt;
        if (!base)
          return std::nullopt;
        result.base = *base;
        for (unsigned i = 0; i < shape.size(); ++i)
          if (llvm::AddOverflow(lhs->coefficients[i], rhs->coefficients[i],
                                result.coefficients[i]))
            return std::nullopt;
      } else {
        auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(binary.getRHS());
        if (!c || !coefficient(c.getValue()))
          return std::nullopt;
        int64_t divisor = c.getValue();
        if (expression.getKind() == mlir::AffineExprKind::Mul) {
          auto base = scale(lhs->base, divisor);
          if (!base)
            return std::nullopt;
          result.base = *base;
          for (unsigned i = 0; i < shape.size(); ++i)
            if (llvm::MulOverflow(lhs->coefficients[i], divisor,
                                  result.coefficients[i]))
              return std::nullopt;
        } else {
          if (divisor <= 0)
            return std::nullopt;
          auto base = lhs->base;
          if (expression.getKind() == mlir::AffineExprKind::CeilDiv) {
            auto adjusted = add(base, constant(divisor - 1));
            if (!adjusted)
              return std::nullopt;
            base = *adjusted;
          }
          int64_t residual = 0;
          for (unsigned i = 0; i < shape.size(); ++i) {
            int64_t a = lhs->coefficients[i];
            int64_t q = llvm::divideFloorSigned(a, divisor);
            int64_t r = a % divisor;
            if (r < 0)
              r += divisor;
            int64_t contribution;
            if (llvm::MulOverflow(r, shape[i] - 1, contribution) ||
                llvm::AddOverflow(residual, contribution, residual))
              return std::nullopt;
            if (contribution)
              blocking.set(i);
            result.coefficients[i] =
                expression.getKind() == mlir::AffineExprKind::Mod ? r : q;
          }
          result.base = expression.getKind() == mlir::AffineExprKind::Mod
                            ? base % divisor
                            : base.floorDiv(divisor);
          if (residual) {
            // Euclidean remainder preserves the shared dividend. Bounding
            // two separate quotients loses that correlation, even when the
            // coordinate congruence proves that no carry is possible.
            int64_t allowance;
            if (llvm::SubOverflow(divisor - 1, residual, allowance))
              return std::nullopt;
            auto negative = scale(base % divisor, -1);
            auto room =
                negative ? add(constant(allowance), *negative) : std::nullopt;
            if (!room || !guard(*room, false))
              return std::nullopt;
          }
        }
      }
    }
    if (!range(result.base) ||
        !llvm::all_of(result.coefficients,
                      [&](int64_t value) { return coefficient(value); }))
      return std::nullopt;
    expressions.try_emplace(expression, result);
    return result;
  }

  std::optional<Interval> displacement(const BlockExpression &expression) {
    Interval result{0, 0};
    for (unsigned i = 0; i < shape.size(); ++i) {
      int64_t delta;
      if (!work.charge() ||
          llvm::MulOverflow(expression.coefficients[i], shape[i] - 1, delta) ||
          llvm::AddOverflow(result.lower, std::min<int64_t>(0, delta),
                            result.lower) ||
          llvm::AddOverflow(result.upper, std::max<int64_t>(0, delta),
                            result.upper))
        return std::nullopt;
    }
    return result;
  }

  bool guard(mlir::AffineExpr expression, bool equality) {
    auto bounds = range(expression);
    if (!bounds)
      return false;
    if (equality ? bounds->lower == 0 && bounds->upper == 0
                 : bounds->lower >= 0)
      return true;
    auto original = expression;
    auto simplified = simplify(expression, /*fixedOrigins=*/true);
    if (!simplified)
      return false;
    expression = *simplified;
    bounds = range(expression);
    if (!bounds)
      return false;
    if (equality ? bounds->lower == 0 && bounds->upper == 0
                 : bounds->lower >= 0)
      return true;
    for (unsigned axis = 0; axis < shape.size(); ++axis)
      if (shape[axis] > 1 && original.isFunctionOfDim(parameters + axis))
        blocking.set(axis);
    if (!work.charge(guards.size() + 1))
      return false;
    for (auto existing : guards)
      if (!existing.complement && existing.set.getNumConstraints() == 1 &&
          existing.set.getConstraint(0) == expression &&
          existing.set.isEq(0) == equality)
        return true;
    guards.push_back({mlir::IntegerSet::get(parameters + shape.size(), 0,
                                            {expression}, {equality}),
                      false});
    return true;
  }

  std::optional<BlockBounds> boundsOnBlock(mlir::AffineExpr expression,
                                           unsigned depth = 0) {
    if (!charge(depth))
      return std::nullopt;
    if (auto found = blockBounds.find(expression); found != blockBounds.end())
      return found->second;
    BlockBounds result{expression, expression};
    if (auto d = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
      if (d.getPosition() >= parameters) {
        auto upper =
            add(expression, constant(shape[d.getPosition() - parameters] - 1));
        if (!upper)
          return std::nullopt;
        result.maximum = *upper;
      }
    } else if (auto binary =
                   mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression)) {
      auto lhs = boundsOnBlock(binary.getLHS(), depth + 1);
      if (!lhs)
        return std::nullopt;
      if (expression.getKind() == mlir::AffineExprKind::Add) {
        auto rhs = boundsOnBlock(binary.getRHS(), depth + 1);
        auto minimum = rhs ? add(lhs->minimum, rhs->minimum) : std::nullopt;
        auto maximum = rhs ? add(lhs->maximum, rhs->maximum) : std::nullopt;
        if (!minimum || !maximum)
          return std::nullopt;
        result = {*minimum, *maximum};
      } else {
        auto constantRhs =
            mlir::dyn_cast<mlir::AffineConstantExpr>(binary.getRHS());
        if (!constantRhs)
          return std::nullopt;
        int64_t c = constantRhs.getValue();
        if (expression.getKind() == mlir::AffineExprKind::Mul) {
          auto minimum = scale(c < 0 ? lhs->maximum : lhs->minimum, c);
          auto maximum = scale(c < 0 ? lhs->minimum : lhs->maximum, c);
          if (!minimum || !maximum)
            return std::nullopt;
          result = {*minimum, *maximum};
        } else if (c <= 0) {
          return std::nullopt;
        } else if (expression.getKind() == mlir::AffineExprKind::FloorDiv) {
          result = {lhs->minimum.floorDiv(c), lhs->maximum.floorDiv(c)};
        } else if (expression.getKind() == mlir::AffineExprKind::CeilDiv) {
          result = {lhs->minimum.ceilDiv(c), lhs->maximum.ceilDiv(c)};
        } else if (expression.getKind() == mlir::AffineExprKind::Mod) {
          // A varying remainder is bounded without inventing a wrap pattern.
          // At unit shape the same expression retains exact point semantics.
          result = lhs->minimum == lhs->maximum
                       ? BlockBounds{lhs->minimum % c, lhs->maximum % c}
                       : BlockBounds{constant(0), constant(c - 1)};
        } else {
          return std::nullopt;
        }
      }
    } else if (!mlir::isa<mlir::AffineConstantExpr>(expression)) {
      return std::nullopt;
    }
    if (!range(result.minimum) || !range(result.maximum))
      return std::nullopt;
    blockBounds.try_emplace(expression, result);
    return result;
  }

  bool domain(IndexDomainCondition condition) {
    llvm::SmallBitVector conditionBlocking(shape.size());
    auto markBlocking = [&](mlir::AffineExpr expression) {
      for (unsigned axis = 0; axis < shape.size(); ++axis)
        if (shape[axis] > 1 && expression.isFunctionOfDim(parameters + axis))
          conditionBlocking.set(axis);
    };
    llvm::SmallVector<mlir::AffineExpr> constraints;
    llvm::SmallVector<bool> equalities;
    for (auto [expression, equality] : llvm::zip_equal(
             condition.set.getConstraints(), condition.set.getEqFlags())) {
      auto bounds = boundsOnBlock(expression);
      if (!bounds)
        return false;
      auto minimum = bounds->minimum, maximum = bounds->maximum;
      if (!condition.complement) {
        if (equality && minimum != maximum) {
          // Both extremal bounds equal zero is sufficient even when this
          // interval proof is conservative for a correlated expression.
          if (!guard(minimum, true) || !guard(maximum, true))
            return false;
        } else if (!guard(minimum, equality)) {
          return false;
        }
      } else {
        // NOT(all constraints): any constraint uniformly violated suffices.
        // Retain this as one complemented conjunction, never enumerate DNF.
        auto simplified = simplify(maximum, /*fixedOrigins=*/true);
        if (!simplified)
          return false;
        maximum = *simplified;
        auto maxRange = range(maximum);
        if (!maxRange)
          return false;
        if (maxRange->upper < 0)
          return true;
        if (maxRange->lower < 0) {
          markBlocking(bounds->maximum);
          constraints.push_back(maximum);
          equalities.push_back(false);
        }
        if (equality) {
          auto negative = scale(minimum, -1);
          if (!negative)
            return false;
          auto negativeRange = range(*negative);
          if (!negativeRange)
            return false;
          if (negativeRange->upper < 0)
            return true;
          if (negativeRange->lower < 0) {
            markBlocking(bounds->minimum);
            constraints.push_back(*negative);
            equalities.push_back(false);
          }
        }
      }
    }
    if (condition.complement && constraints.empty())
      return guard(constant(-1), false);
    if (condition.complement) {
      guards.push_back({mlir::IntegerSet::get(parameters + shape.size(), 0,
                                              constraints, equalities),
                        true});
      blocking |= conditionBlocking;
    }
    return true;
  }

  const TensorSubsetDemand &demand;
  llvm::ArrayRef<int64_t> shape;
  IndexRelationWork &work;
  unsigned parameters;
  mlir::MLIRContext *context;
  llvm::SmallBitVector blocking;
  llvm::SmallVector<IndexDomainCondition, 4> guards;
  bool limited = false;
  llvm::DenseMap<mlir::AffineExpr, Interval> ranges;
  llvm::DenseMap<mlir::AffineExpr, std::pair<int64_t, int64_t>> congruences;
  llvm::DenseMap<mlir::AffineExpr, uint64_t> norms;
  llvm::DenseMap<mlir::AffineExpr, BlockExpression> expressions;
  llvm::DenseMap<mlir::AffineExpr, BlockBounds> blockBounds;
};

} // namespace

TensorSubsetBlockResult queryTensorSubsetBlock(const TensorSubsetDemand &demand,
                                               const TensorSubsetSource &source,
                                               llvm::ArrayRef<int64_t> shape,
                                               IndexRelationWork &work) {
  using Status = TensorSubsetBlockStatus;
  auto reject = [](Status status, llvm::StringRef detail) {
    return TensorSubsetBlockResult{status, std::nullopt, {}, detail.str()};
  };
  auto type =
      source.source
          ? mlir::dyn_cast<mlir::RankedTensorType>(source.source.getType())
          : mlir::RankedTensorType{};
  if (!demand.domain.getAsOpaquePointer() || !source.coordinates || !type ||
      !type.hasStaticShape() || shape.size() != demand.shape.size() ||
      source.coordinates.getNumSymbols() ||
      source.coordinates.getNumDims() !=
          shape.size() + demand.parameters.size() ||
      source.coordinates.getNumResults() != unsigned(type.getRank()))
    return reject(Status::BrokenContract, "inconsistent tensor subset block");
  for (auto [size, extent] : llvm::zip_equal(shape, demand.shape))
    if (size <= 0 || size > extent)
      return reject(Status::BrokenContract,
                    "invalid tensor subset block extent");
  for (const auto &parameter : demand.parameters)
    if (parameter.upper <= parameter.lower || parameter.step <= 0)
      return reject(Status::BrokenContract,
                    "block query requires a live domain");

  // A contiguous row-major interval can cross quotient/remainder carries and
  // still be one rectangular reshape. Prove the source ordinal identity
  // before the per-coordinate affine-block proof, using the same derived F.
  llvm::SmallVector<unsigned> contiguousAxes;
  auto contiguous = [&]() -> std::optional<TensorSubsetBlock> {
    BlockQuery proof(demand, shape, work);
    // Preserve independent leading axes as a rectangular batch. Only the
    // remaining row-major suffix must be contiguous in its source volume.
    // Each leading coordinate must depend on precisely its matching delta;
    // the suffix ordinal must be independent of all peeled delta axes.
    unsigned prefix = 0;
    for (; prefix < std::min(shape.size(), size_t(type.getRank())); ++prefix) {
      BlockQuery axisProof(demand, shape, work);
      auto leading = axisProof.linearize(source.coordinates.getResult(prefix));
      if (!leading)
        break;
      bool identity = true;
      for (unsigned axis = 0; axis < shape.size(); ++axis)
        if (shape[axis] > 1 &&
            leading->coefficients[axis] != int64_t(axis == prefix))
          identity = false;
      if (!identity)
        break;
      proof.guards.append(axisProof.guards);
      proof.blocking |= axisProof.blocking;
    }
    int64_t volume = 1;
    for (int64_t extent : shape.drop_front(prefix))
      if (llvm::MulOverflow(volume, extent, volume))
        return std::nullopt;
    TensorSubsetBlock block;
    block.sourceSizes.resize(type.getRank(), 1);
    for (unsigned axis = 0; axis < prefix; ++axis)
      block.sourceSizes[axis] = shape[axis];
    int64_t remaining = volume;
    for (unsigned axis = type.getRank(); axis-- > prefix;) {
      int64_t extent = type.getDimSize(axis);
      if (extent <= 0)
        return std::nullopt;
      int64_t size = std::min(remaining, extent);
      if (remaining % size)
        return std::nullopt;
      block.sourceSizes[axis] = size;
      remaining /= size;
    }
    if (remaining != 1)
      return std::nullopt;
    auto ordinal = proof.constant(0);
    int64_t stride = 1;
    for (unsigned axis = type.getRank(); axis-- > prefix;) {
      auto term = proof.scale(source.coordinates.getResult(axis), stride);
      auto sum = term ? proof.add(ordinal, *term) : std::nullopt;
      if (!sum || llvm::MulOverflow(stride, type.getDimSize(axis), stride))
        return std::nullopt;
      ordinal = *sum;
    }
    auto simplified = proof.simplify(ordinal);
    if (!simplified)
      return std::nullopt;
    ordinal = *simplified;
    auto linear = proof.linearize(ordinal);
    if (!linear)
      return std::nullopt;
    stride = 1;
    for (unsigned axis = 0; axis < prefix; ++axis)
      if (shape[axis] > 1 && linear->coefficients[axis] != 0)
        return std::nullopt;
    for (unsigned axis = shape.size(); axis-- > prefix;) {
      if (shape[axis] > 1 && linear->coefficients[axis] != stride)
        return std::nullopt;
      if (llvm::MulOverflow(stride, shape[axis], stride))
        return std::nullopt;
    }
    bool outerVaries = false;
    for (unsigned axis = 0; axis < unsigned(type.getRank()); ++axis) {
      auto offset = source.coordinates.getResult(axis);
      auto negative = proof.scale(offset, -1);
      auto upper = negative ? proof.add(*negative,
                                        proof.constant(type.getDimSize(axis) -
                                                       block.sourceSizes[axis]))
                            : std::nullopt;
      if (!upper || !proof.guard(offset, false) || !proof.guard(*upper, false))
        return std::nullopt;
      if (axis == prefix)
        outerVaries = false;
      if (axis >= prefix && outerVaries &&
          (block.sourceSizes[axis] != type.getDimSize(axis) ||
           !proof.guard(offset, true)))
        return std::nullopt;
      outerVaries |= block.sourceSizes[axis] > 1;
    }
    for (auto condition : source.conditions)
      if (!proof.domain(condition))
        return std::nullopt;
    block.sourceOffsets = source.coordinates;
    block.guards = std::move(proof.guards);
    for (unsigned axis = 0; axis < shape.size(); ++axis)
      if (proof.blocking.test(axis))
        contiguousAxes.push_back(axis);
    return block;
  };
  if (auto block = contiguous())
    return {Status::Copy, std::move(block), std::move(contiguousAxes), {}};
  BlockQuery query(demand, shape, work);
  auto failure = [&] {
    return reject(work.isExhausted() || query.limited
                      ? Status::ResourceExhausted
                      : Status::Unsupported,
                  "tensor subset block arithmetic or work limit");
  };
  auto axes = [&] {
    llvm::SmallVector<unsigned, 4> result;
    for (unsigned i = 0; i < shape.size(); ++i)
      if (shape[i] > 1 && query.blocking.test(i))
        result.push_back(i);
    return result;
  };
  auto subdivide = [&] {
    return TensorSubsetBlockResult{
        Status::Subdivide, std::nullopt, axes(),
        "block does not have dense row-major copy geometry"};
  };
  for (auto condition : source.conditions)
    if (!query.domain(condition))
      return failure();
  llvm::SmallVector<BlockExpression> coordinates;
  llvm::SmallVector<mlir::AffineExpr> offsets;
  TensorSubsetBlock block;
  int64_t sourceVolume = 1, blockVolume = 1;
  for (auto expression : source.coordinates.getResults()) {
    auto coordinate = query.linearize(expression);
    auto delta = coordinate ? query.displacement(*coordinate) : std::nullopt;
    if (!delta)
      return failure();
    int64_t extent;
    if (llvm::SubOverflow(delta->upper, delta->lower, extent) ||
        llvm::AddOverflow(extent, int64_t{1}, extent) ||
        llvm::MulOverflow(sourceVolume, extent, sourceVolume))
      return failure();
    auto offset = query.add(coordinate->base, query.constant(delta->lower));
    if (!offset)
      return failure();
    unsigned axis = coordinates.size();
    auto negative = query.scale(*offset, -1);
    auto upper = negative
                     ? query.add(*negative,
                                 query.constant(type.getDimSize(axis) - extent))
                     : std::nullopt;
    if (!upper || !query.guard(*offset, false) || !query.guard(*upper, false))
      return failure();
    offsets.push_back(*offset);
    block.sourceSizes.push_back(extent);
    coordinates.push_back(std::move(*coordinate));
  }
  for (int64_t extent : shape)
    if (llvm::MulOverflow(blockVolume, extent, blockVolume))
      return failure();
  if (blockVolume != sourceVolume) {
    for (unsigned axis = 0; axis < shape.size(); ++axis)
      if (shape[axis] > 1)
        query.blocking.set(axis);
    return subdivide();
  }
  llvm::SmallVector<int64_t, 4> ordinal(shape.size(), 0);
  int64_t stride = 1, origin = 0;
  for (unsigned i = coordinates.size(); i-- > 0;) {
    auto delta = query.displacement(coordinates[i]);
    int64_t contribution;
    if (!delta || llvm::MulOverflow(delta->lower, stride, contribution) ||
        llvm::AddOverflow(origin, contribution, origin))
      return failure();
    for (unsigned j = 0; j < shape.size(); ++j)
      if (llvm::MulOverflow(coordinates[i].coefficients[j], stride,
                            contribution) ||
          llvm::AddOverflow(ordinal[j], contribution, ordinal[j]))
        return failure();
    if (llvm::MulOverflow(stride, block.sourceSizes[i], stride))
      return failure();
  }
  stride = 1;
  bool ordered = origin == 0;
  for (unsigned i = shape.size(); i-- > 0;) {
    if (shape[i] > 1 && ordinal[i] != stride) {
      ordered = false;
      query.blocking.set(i);
    }
    if (llvm::MulOverflow(stride, shape[i], stride))
      return failure();
  }
  if (!ordered)
    return subdivide();
  block.sourceOffsets = mlir::AffineMap::get(source.coordinates.getNumDims(), 0,
                                             offsets, type.getContext());
  block.guards = std::move(query.guards);
  return {Status::Copy, std::move(block), axes(), {}};
}

StaticTensorSubsetBlocksResult
queryStaticTensorSubsetBlocks(const TensorSubsetDemand &demand,
                              IndexRelationWork &work) {
  auto reject = [](IndexRelationStatus status, llvm::StringRef reason) {
    return StaticTensorSubsetBlocksResult{status, {}, reason.str()};
  };
  auto limited = [&] {
    return reject(IndexRelationStatus::ResourceExhausted,
                  "static subset blocks exceeded the request budget");
  };
  if (!demand.domain.getAsOpaquePointer() ||
      llvm::any_of(demand.shape, [](int64_t size) { return size <= 0; }))
    return reject(IndexRelationStatus::Invalid, "invalid static subset domain");
  for (const auto &parameter : demand.parameters) {
    int64_t distance;
    if (parameter.step <= 0 ||
        llvm::SubOverflow(parameter.upper, parameter.lower, distance))
      return reject(IndexRelationStatus::Invalid,
                    "invalid static subset parameter");
    if (distance <= 0)
      return reject(IndexRelationStatus::Unsupported,
                    "static subset blocks require a live parameter domain");
  }
  unsigned rank = demand.shape.size(), parameters = demand.parameters.size();
  llvm::SmallVector<int64_t> unit(rank, 1);
  BlockQuery ranges(demand, unit, work);
  auto *context = demand.domain.getContext();
  llvm::SmallVector<mlir::AffineExpr> bindings;
  for (auto [index, parameter] : llvm::enumerate(demand.parameters))
    bindings.push_back(
        parameter.upper - parameter.lower <= parameter.step
            ? mlir::getAffineConstantExpr(parameter.lower, context)
            : mlir::getAffineDimExpr(index, context));
  for (unsigned axis = 0; axis < rank; ++axis)
    bindings.push_back(
        demand.shape[axis] == 1
            ? mlir::getAffineConstantExpr(0, context)
            : mlir::getAffineDimExpr(parameters + axis, context));
  auto binding = mlir::AffineMap::get(parameters + rank, 0, bindings, context);
  StaticRectangularIndexSet whole{llvm::SmallVector<int64_t, 4>(rank, 0),
                                  demand.shape};
  struct BoxResult {
    IndexRelationStatus status;
    std::optional<StaticRectangularIndexSet> box;
  };
  auto rectangle = [&](mlir::IntegerSet condition) -> BoxResult {
    auto composed = composeIndexMap(
        mlir::AffineMap::get(condition.getNumDims(), 0,
                             condition.getConstraints(), context),
        binding, work);
    if (!composed.isExact())
      return {composed.status, std::nullopt};
    auto box = whole;
    llvm::SmallVector<std::pair<mlir::AffineExpr, bool>> constraints;
    for (auto [expression, equality] :
         llvm::zip_equal(composed.map.getResults(), condition.getEqFlags()))
      constraints.emplace_back(expression, equality);
    while (!constraints.empty()) {
      auto [expression, equality] = constraints.pop_back_val();
      auto interval = ranges.range(expression);
      if (!interval)
        return {work.isExhausted() ? IndexRelationStatus::ResourceExhausted
                                   : IndexRelationStatus::Unsupported,
                std::nullopt};
      if (equality ? interval->lower == 0 && interval->upper == 0
                   : interval->lower >= 0)
        continue;
      if (equality ? interval->upper < 0 || interval->lower > 0
                   : interval->upper < 0)
        return {IndexRelationStatus::Exact, std::nullopt};
      llvm::SmallVector<int64_t> coefficients(rank, 0);
      int64_t constant = 0, quotientCoefficient = 0;
      mlir::AffineBinaryOpExpr quotient(nullptr);
      // A single constant-divisor floor constraint has an exact integer
      // interval preimage. Other nonlinear/coupled forms retain the generic
      // block path; no relaxed rectangle authorizes a read.
      std::function<bool(mlir::AffineExpr, int64_t)> collect =
          [&](mlir::AffineExpr expr, int64_t scale) {
            if (!work.charge())
              return false;
            if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
              int64_t term;
              return !llvm::MulOverflow(c.getValue(), scale, term) &&
                     !llvm::AddOverflow(constant, term, constant);
            }
            if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
              if (dim.getPosition() < parameters)
                return false;
              unsigned axis = dim.getPosition() - parameters;
              return !llvm::AddOverflow(coefficients[axis], scale,
                                        coefficients[axis]);
            }
            auto binary = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
            if (!binary)
              return false;
            if (expr.getKind() == mlir::AffineExprKind::Add)
              return collect(binary.getLHS(), scale) &&
                     collect(binary.getRHS(), scale);
            if (expr.getKind() == mlir::AffineExprKind::FloorDiv) {
              if (quotient && quotient != binary)
                return false;
              quotient = binary;
              return !llvm::AddOverflow(quotientCoefficient, scale,
                                        quotientCoefficient);
            }
            auto factor =
                mlir::dyn_cast<mlir::AffineConstantExpr>(binary.getRHS());
            int64_t product;
            return expr.getKind() == mlir::AffineExprKind::Mul && factor &&
                   !llvm::MulOverflow(scale, factor.getValue(), product) &&
                   collect(binary.getLHS(), product);
          };
      if (!collect(expression, 1))
        return {work.isExhausted() ? IndexRelationStatus::ResourceExhausted
                                   : IndexRelationStatus::Unsupported,
                std::nullopt};
      if (quotient && quotientCoefficient) {
        auto divisor =
            mlir::dyn_cast<mlir::AffineConstantExpr>(quotient.getRHS());
        if (!divisor || divisor.getValue() <= 0 ||
            llvm::any_of(coefficients,
                         [](int64_t value) { return value != 0; }) ||
            constant == std::numeric_limits<int64_t>::min() ||
            quotientCoefficient == std::numeric_limits<int64_t>::min())
          return {IndexRelationStatus::Unsupported, std::nullopt};
        if (equality && constant % quotientCoefficient)
          return {IndexRelationStatus::Exact, std::nullopt};
        auto addBound = [&](bool lower) {
          int64_t bound =
              equality ? -constant / quotientCoefficient
              : lower  ? llvm::divideCeilSigned(-constant, quotientCoefficient)
                      : llvm::divideFloorSigned(constant, -quotientCoefficient);
          int64_t offset;
          if ((!lower && llvm::AddOverflow(bound, int64_t{1}, bound)) ||
              llvm::MulOverflow(bound, divisor.getValue(), offset) ||
              (lower ? llvm::SubOverflow(int64_t{0}, offset, offset)
                     : llvm::SubOverflow(offset, int64_t{1}, offset)))
            return false;
          // Compose through the same checked expression path before creating
          // the transformed inequality, including nested quotients.
          auto dim = mlir::getAffineDimExpr(0, context);
          auto form = mlir::AffineMap::get(1, 0, (lower ? dim : -dim) + offset,
                                           context);
          auto operand =
              mlir::AffineMap::get(parameters + rank, 0, quotient.getLHS());
          auto expanded = composeIndexMap(form, operand, work);
          if (!expanded.isExact())
            return false;
          constraints.emplace_back(expanded.map.getResult(0), false);
          return true;
        };
        if ((equality && (!addBound(true) || !addBound(false))) ||
            (!equality && !addBound(quotientCoefficient > 0)))
          return {work.isExhausted() ? IndexRelationStatus::ResourceExhausted
                                     : IndexRelationStatus::Unsupported,
                  std::nullopt};
        continue;
      }
      std::optional<unsigned> axis;
      for (unsigned i = 0; i < rank; ++i) {
        if (!coefficients[i])
          continue;
        if (axis)
          return {IndexRelationStatus::Unsupported, std::nullopt};
        axis = i;
      }
      if (!axis) {
        if (equality ? constant != 0 : constant < 0)
          return {IndexRelationStatus::Exact, std::nullopt};
        continue;
      }
      int64_t lower = box.offsets[*axis];
      int64_t upper = lower + box.sizes[*axis] - 1;
      int64_t coefficient = coefficients[*axis];
      if (equality) {
        if (constant % coefficient)
          return {IndexRelationStatus::Exact, std::nullopt};
        lower = std::max(lower, -constant / coefficient);
        upper = std::min(upper, -constant / coefficient);
      } else if (coefficient > 0) {
        lower = std::max(lower, llvm::divideCeilSigned(-constant, coefficient));
      } else {
        upper =
            std::min(upper, llvm::divideFloorSigned(constant, -coefficient));
      }
      if (upper < lower)
        return {IndexRelationStatus::Exact, std::nullopt};
      box.offsets[*axis] = lower;
      box.sizes[*axis] = upper - lower + 1;
    }
    return {IndexRelationStatus::Exact, std::move(box)};
  };
  auto intersect = [&](const StaticRectangularIndexSet &left,
                       const StaticRectangularIndexSet &right)
      -> std::optional<StaticRectangularIndexSet> {
    auto result = left;
    for (unsigned axis = 0; axis < rank; ++axis) {
      int64_t lower = std::max(left.offsets[axis], right.offsets[axis]);
      int64_t upper = std::min(left.offsets[axis] + left.sizes[axis],
                               right.offsets[axis] + right.sizes[axis]);
      if (upper <= lower)
        return std::nullopt;
      result.offsets[axis] = lower;
      result.sizes[axis] = upper - lower;
    }
    return result;
  };
  StaticTensorSubsetBlocksResult result{IndexRelationStatus::Exact, {}, {}};
  for (auto [sourceIndex, source] : llvm::enumerate(demand.sources)) {
    llvm::SmallVector<StaticRectangularIndexSet, 4> boxes{whole};
    for (auto condition : source.conditions) {
      auto window = rectangle(condition.set);
      if (window.status != IndexRelationStatus::Exact)
        return reject(window.status,
                      "static subset source is not a bounded rectangle");
      llvm::SmallVector<StaticRectangularIndexSet, 4> next;
      for (const auto &box : boxes) {
        if (!work.charge(rank + 1))
          return limited();
        auto overlap = window.box ? intersect(box, *window.box) : std::nullopt;
        if (!condition.complement) {
          if (overlap)
            next.push_back(std::move(*overlap));
          continue;
        }
        if (!overlap) {
          next.push_back(box);
          continue;
        }
        auto middle = box;
        for (unsigned axis = 0; axis < rank; ++axis) {
          if (!work.charge(rank + 1))
            return limited();
          if (middle.offsets[axis] < overlap->offsets[axis]) {
            auto lower = middle;
            lower.sizes[axis] = overlap->offsets[axis] - middle.offsets[axis];
            next.push_back(std::move(lower));
            middle.sizes[axis] -= overlap->offsets[axis] - middle.offsets[axis];
            middle.offsets[axis] = overlap->offsets[axis];
          }
          int64_t end = overlap->offsets[axis] + overlap->sizes[axis];
          if (middle.offsets[axis] + middle.sizes[axis] > end) {
            auto upper = middle;
            upper.sizes[axis] = middle.offsets[axis] + middle.sizes[axis] - end;
            upper.offsets[axis] = end;
            next.push_back(std::move(upper));
            middle.sizes[axis] = end - middle.offsets[axis];
          }
          if (next.size() > work.getLimits().maxRectangularPieces)
            return limited();
        }
      }
      boxes = std::move(next);
    }
    llvm::SmallVector<StaticRectangularIndexSet, 8> pending(boxes.rbegin(),
                                                            boxes.rend());
    while (!pending.empty()) {
      if (!work.charge(rank + 1) || result.blocks.size() + pending.size() >
                                        work.getLimits().maxRectangularPieces)
        return limited();
      auto box = pending.pop_back_val();
      auto proof = queryTensorSubsetBlock(demand, source, box.sizes, work);
      if (proof.status == TensorSubsetBlockStatus::ResourceExhausted)
        return limited();
      if (proof.status != TensorSubsetBlockStatus::Copy &&
          proof.status != TensorSubsetBlockStatus::Subdivide)
        return reject(IndexRelationStatus::Unsupported, proof.detail);
      llvm::SmallVector<mlir::AffineExpr> origin;
      origin.append(bindings.begin(), bindings.begin() + parameters);
      for (int64_t offset : box.offsets)
        origin.push_back(mlir::getAffineConstantExpr(offset, context));
      auto atOrigin = mlir::AffineMap::get(parameters, 0, origin, context);
      bool proved = proof.status == TensorSubsetBlockStatus::Copy;
      if (proved)
        for (auto guard : proof.block->guards) {
          auto checked = composeIndexMap(
              mlir::AffineMap::get(guard.set.getNumDims(), 0,
                                   guard.set.getConstraints(), context),
              atOrigin, work);
          if (!checked.isExact())
            return reject(checked.status, checked.reason);
          bool allTrue = true, anyFalse = false;
          for (auto [expression, equality] : llvm::zip_equal(
                   checked.map.getResults(), guard.set.getEqFlags())) {
            auto interval = ranges.range(expression);
            if (!interval)
              return reject(
                  work.isExhausted() ? IndexRelationStatus::ResourceExhausted
                                     : IndexRelationStatus::Unsupported,
                  "static block guard has no bounded parameter proof");
            allTrue &= equality ? interval->lower == 0 && interval->upper == 0
                                : interval->lower >= 0;
            anyFalse |= equality ? interval->upper < 0 || interval->lower > 0
                                 : interval->upper < 0;
          }
          if (!(guard.complement ? anyFalse : allTrue)) {
            proved = false;
            break;
          }
        }
      if (!proved) {
        std::optional<unsigned> axis;
        for (unsigned candidate : proof.blockingAxes)
          if (box.sizes[candidate] > 1) {
            axis = candidate;
            break;
          }
        if (!axis)
          for (unsigned candidate = 0; candidate < rank; ++candidate)
            if (box.sizes[candidate] > 1) {
              axis = candidate;
              break;
            }
        if (!axis)
          return reject(IndexRelationStatus::Unsupported,
                        "static unit subset has no ordered copy proof");
        // Static rectangles need no dynamic half-loop template. Take a
        // power-of-two prefix so an aligned large suffix keeps its alignment
        // when the remaining extent is odd.
        int64_t half = llvm::bit_floor(uint64_t(box.sizes[*axis] - 1));
        auto right = box;
        right.offsets[*axis] += half;
        right.sizes[*axis] -= half;
        box.sizes[*axis] = half;
        pending.push_back(std::move(right));
        pending.push_back(std::move(box));
        continue;
      }
      auto offsets =
          composeIndexMap(proof.block->sourceOffsets, atOrigin, work);
      if (!offsets.isExact())
        return reject(offsets.status, offsets.reason);
      proof.block->sourceOffsets = offsets.map;
      proof.block->guards.clear();
      result.blocks.push_back(
          {unsigned(sourceIndex), std::move(box), std::move(*proof.block)});
    }
  }
  int64_t volume = 1, covered = 0;
  for (int64_t size : demand.shape)
    if (llvm::MulOverflow(volume, size, volume))
      return limited();
  for (const auto &block : result.blocks) {
    int64_t size = 1;
    for (int64_t extent : block.destination.sizes)
      if (!work.charge() || llvm::MulOverflow(size, extent, size))
        return limited();
    if (llvm::AddOverflow(covered, size, covered))
      return limited();
  }
  // The queried last-writer paths are disjoint. Preserve their exact total
  // coverage after the static rectangular construction as well.
  if (covered != volume)
    return reject(IndexRelationStatus::Unsupported,
                  "static subset rectangles do not cover the whole demand");
  return result;
}

} // namespace wafer::analysis
