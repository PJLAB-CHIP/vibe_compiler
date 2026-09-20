//===- AttentionVisibility.cpp - Selected attention block visibility
//-------===//

#include "AttentionVisibility.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <numeric>

namespace wafer {
namespace {

// A residue and a modulus; modulus zero describes a constant. Unknown leaves
// have modulus one. This reads only the current scalar position expression.
struct Congruence {
  int64_t residue = 0;
  int64_t modulus = 1;
};

int64_t positiveRemainder(int64_t value, int64_t modulus) {
  int64_t result = value % modulus;
  return result < 0 ? result + modulus : result;
}

Congruence getCongruence(mlir::Value value) {
  if (auto constant = mlir::getConstantIntValue(value))
    return {*constant, 0};
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto loop =
        mlir::dyn_cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop || value != loop.getInductionVar())
      return {};
    auto step = mlir::getConstantIntValue(loop.getStep());
    if (!step || *step <= 0)
      return {};
    auto lower = getCongruence(loop.getLowerBound());
    int64_t modulus = std::gcd(lower.modulus, *step);
    return {positiveRemainder(lower.residue, modulus), modulus};
  }
  auto *definition = value.getDefiningOp();
  if (!mlir::isa_and_nonnull<mlir::arith::AddIOp, mlir::arith::SubIOp>(
          definition))
    return {};
  auto lhs = getCongruence(definition->getOperand(0));
  auto rhs = getCongruence(definition->getOperand(1));
  int64_t residue;
  bool overflow = mlir::isa<mlir::arith::AddIOp>(definition)
                      ? llvm::AddOverflow(lhs.residue, rhs.residue, residue)
                      : llvm::SubOverflow(lhs.residue, rhs.residue, residue);
  if (overflow)
    return {};
  int64_t modulus = std::gcd(lhs.modulus, rhs.modulus);
  return {modulus ? positiveRemainder(residue, modulus) : residue, modulus};
}

class PositionBounds : public mlir::ValueBoundsConstraintSet {
public:
  explicit PositionBounds(mlir::Operation *use)
      : ValueBoundsConstraintSet(use->getContext(),
                                 [](auto, auto, auto &) { return false; }) {
    for (auto *child = use; child->getParentOp(); child = child->getParentOp())
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(child->getParentOp()))
        addCondition(branch.getCondition(),
                     child->getParentRegion() == &branch.getThenRegion());
  }

  std::pair<std::optional<int64_t>, std::optional<int64_t>>
  getDifference(mlir::Value lhs, mlir::Value rhs) {
    // The pinned ValueBounds worklist stores column indices. Drain ancestor
    // symbols before inserting the new dimension, which shifts those columns.
    processWorklist();
    auto map =
        mlir::AffineMap::get(2, 0,
                             mlir::getAffineDimExpr(0, lhs.getContext()) -
                                 mlir::getAffineDimExpr(1, lhs.getContext()));
    int64_t position =
        populateConstraints(map, {{lhs, std::nullopt}, {rhs, std::nullopt}});
    return {cstr.getConstantBound64(mlir::presburger::BoundType::LB, position),
            cstr.getConstantBound64(mlir::presburger::BoundType::UB, position)};
  }

private:
  void addCondition(mlir::Value condition, bool truth) {
    if (auto both = condition.getDefiningOp<mlir::arith::AndIOp>()) {
      if (truth) {
        addCondition(both.getLhs(), true);
        addCondition(both.getRhs(), true);
      } else {
        for (auto [constant, other] : {std::pair{both.getLhs(), both.getRhs()},
                                       std::pair{both.getRhs(), both.getLhs()}})
          if (auto value = mlir::getConstantIntValue(constant); value && *value)
            addCondition(other, false);
      }
      return;
    }
    auto compare = condition.getDefiningOp<mlir::arith::CmpIOp>();
    if (!compare || !compare.getLhs().getType().isIndex())
      return;
    using P = mlir::arith::CmpIPredicate;
    auto predicate = truth
                         ? compare.getPredicate()
                         : mlir::arith::invertPredicate(compare.getPredicate());
    if (predicate == P::ult || predicate == P::ule || predicate == P::ugt ||
        predicate == P::uge) {
      for (auto value : compare.getOperands()) {
        auto lower = computeConstantBound(mlir::presburger::BoundType::LB,
                                          Variable(value));
        if (mlir::failed(lower) || *lower < 0)
          return;
      }
    }
    auto lhs = compare.getLhs();
    (void)getExpr(lhs);
    auto rhs = getExpr(compare.getRhs());
    switch (predicate) {
    case P::eq:
      bound(lhs) == rhs;
      break;
    case P::slt:
    case P::ult:
      bound(lhs) < rhs;
      break;
    case P::sle:
    case P::ule:
      bound(lhs) <= rhs;
      break;
    case P::sgt:
    case P::ugt:
      bound(lhs) > rhs;
      break;
    case P::sge:
    case P::uge:
      bound(lhs) >= rhs;
      break;
    case P::ne:
      break;
    }
  }
};

bool tightenVisibleLoop(mlir::RewriterBase &rewriter,
                        LinalgExtOnlineAttentionOp tile) {
  auto loop = mlir::dyn_cast<mlir::scf::ForOp>(tile->getParentOp());
  if (!loop)
    return false;
  auto step = mlir::getConstantIntValue(loop.getStep());
  if (!step || *step <= 0)
    return false;
  // Removing an invisible iteration must leave every carried value unchanged.
  // This checks the actual update and yield, including any unrelated states.
  auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  llvm::SmallVector<mlir::Value> previous{tile.getAccumulator(),
                                          tile.getMaximum(), tile.getSum()};
  for (auto [next, argument] :
       llvm::zip_equal(yield.getOperands(), loop.getRegionIterArgs())) {
    auto insertion = next.getDefiningOp<mlir::tensor::InsertSliceOp>();
    if (insertion) {
      auto result = mlir::dyn_cast<mlir::OpResult>(insertion.getSource());
      auto extraction = result && result.getOwner() == tile
                            ? previous[result.getResultNumber()]
                                  .getDefiningOp<mlir::tensor::ExtractSliceOp>()
                            : mlir::tensor::ExtractSliceOp{};
      if (!extraction || insertion.getDest() != argument ||
          extraction.getSource() != argument ||
          insertion.getMixedOffsets() != extraction.getMixedOffsets() ||
          insertion.getMixedSizes() != extraction.getMixedSizes() ||
          insertion.getMixedStrides() != extraction.getMixedStrides())
        return false;
      continue;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(next);
    if (result && result.getOwner() == tile)
      next = previous[result.getResultNumber()];
    if (next != argument)
      return false;
  }
  for (auto &op : loop.getBody()->without_terminator())
    if (&op != tile && !mlir::isMemoryEffectFree(&op))
      return false;
  auto base = mlir::ValueBoundsConstraintSet::computeConstantDelta(
      tile.getPositions()[1], loop.getInductionVar());
  if (mlir::failed(base))
    return false;
  llvm::SmallVector<mlir::Operation *> metadata;
  llvm::DenseSet<mlir::Operation *> visited;
  std::function<bool(mlir::Value)> invariant = [&](mlir::Value value) {
    if (loop.isDefinedOutsideOfLoop(value))
      return true;
    auto *definition = value.getDefiningOp();
    if (!definition || definition->getNumRegions() ||
        !mlir::isPure(definition) || !value.getType().isIndex())
      return false;
    if (!visited.insert(definition).second)
      return true;
    if (!llvm::all_of(definition->getOperands(), invariant))
      return false;
    metadata.push_back(definition);
    return true;
  };
  if (!invariant(tile.getPositions()[2]) ||
      (tile.getCausal() && !invariant(tile.getPositions()[0])))
    return false;
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  for (auto *op : metadata)
    rewriter.moveOpBefore(op, loop);
  rewriter.setInsertionPoint(loop);
  auto loc = tile.getLoc();
  mlir::Value end = tile.getPositions()[2];
  if (tile.getCausal()) {
    auto dimension = mlir::cast<mlir::AffineDimExpr>(
                         tile.getPositionMapAttr().getValue().getResult(0))
                         .getPosition();
    auto extent = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, tile.getStaticLoopRanges()[dimension]);
    auto queryEnd = rewriter.createOrFold<mlir::arith::AddIOp>(
        loc, tile.getPositions()[0], extent);
    end = rewriter.createOrFold<mlir::arith::MinSIOp>(loc, end, queryEnd);
  }
  if (*base) {
    auto offset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *base);
    end = rewriter.createOrFold<mlir::arith::SubIOp>(loc, end, offset);
  }
  auto upper = rewriter.createOrFold<mlir::arith::MinSIOp>(
      loc, loop.getUpperBound(), end);
  rewriter.modifyOpInPlace(loop, [&] { loop.setUpperBound(upper); });
  return true;
}

} // namespace

AttentionPositionRange getAttentionPositionRange(mlir::Value lhs,
                                                 mlir::Value rhs,
                                                 mlir::Operation *use,
                                                 int64_t lower, int64_t upper) {
  auto [minimum, maximum] = PositionBounds(use).getDifference(lhs, rhs);
  if (minimum && *minimum >= upper)
    return {upper, upper, 1};
  if (maximum && *maximum <= lower)
    return {lower, lower, 1};
  if (minimum && maximum && *minimum == *maximum)
    return {*minimum, *maximum, 1};
  auto left = getCongruence(lhs), right = getCongruence(rhs);
  int64_t residue;
  int64_t step = std::gcd(left.modulus, right.modulus);
  if (llvm::SubOverflow(left.residue, right.residue, residue)) {
    step = 1;
    residue = 0;
  }
  if (!step)
    return {std::clamp(residue, lower, upper),
            std::clamp(residue, lower, upper), 1};
  residue = positiveRemainder(residue, step);
  // Include one reachable grid point beyond each saturation endpoint. Clamping
  // the runtime offset to these points preserves the final 0/-inf pattern.
  int64_t first, last, span;
  int64_t lowerResidue = positiveRemainder(lower, step);
  int64_t upperResidue = positiveRemainder(upper, step);
  if (llvm::SubOverflow(lower, positiveRemainder(lowerResidue - residue, step),
                        first) ||
      llvm::AddOverflow(upper, positiveRemainder(residue - upperResidue, step),
                        last) ||
      llvm::SubOverflow(last, first, span))
    return {lower, upper, 1};
  if (minimum && *minimum > first) {
    int64_t adjustment =
        positiveRemainder(residue - positiveRemainder(*minimum, step), step);
    if (!llvm::AddOverflow(*minimum, adjustment, first))
      first = std::min(first, last);
  }
  if (maximum && *maximum < last) {
    int64_t adjustment =
        positiveRemainder(positiveRemainder(*maximum, step) - residue, step);
    if (!llvm::SubOverflow(*maximum, adjustment, last))
      last = std::max(first, last);
  }
  return {first, last, step};
}

bool proveAttentionPositionOrder(mlir::Value lhs, mlir::Value rhs,
                                 bool strict) {
  using Bounds = mlir::ValueBoundsConstraintSet;
  auto difference =
      mlir::AffineMap::get(2, 0,
                           mlir::getAffineDimExpr(1, lhs.getContext()) -
                               mlir::getAffineDimExpr(0, lhs.getContext()));
  auto stop = [](mlir::Value value, std::optional<int64_t> dim,
                 Bounds &bounds) {
    if (dim)
      return false;
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    auto loop = argument ? mlir::dyn_cast<mlir::scf::ForOp>(
                               argument.getOwner()->getParentOp())
                         : mlir::scf::ForOp{};
    if (!loop || value != loop.getInductionVar())
      return false;
    auto first = mlir::getConstantIntValue(loop.getLowerBound());
    auto end = mlir::getConstantIntValue(loop.getUpperBound());
    auto step = mlir::getConstantIntValue(loop.getStep());
    // Attention positions are nonnegative. This also makes every
    // subtraction/product below representable in signed int64.
    if (!first || !end || !step || *first < 0 || *end <= *first || *step <= 0)
      return false;
    int64_t last = *first + ((*end - *first - 1) / *step) * *step;
    bounds.bound(value) >= *first;
    bounds.bound(value) <= last;
    return true;
  };
  for (auto value : {lhs, rhs}) {
    auto bound = Bounds::computeConstantBound(mlir::presburger::BoundType::LB,
                                              Bounds::Variable(value), stop);
    if (mlir::failed(bound) || *bound < 0)
      return false;
  }
  auto lower = Bounds::computeConstantBound(
      mlir::presburger::BoundType::LB,
      Bounds::Variable(difference, llvm::ArrayRef<mlir::Value>{lhs, rhs}),
      stop);
  if (mlir::failed(lower))
    return false;
  auto left = getCongruence(lhs), right = getCongruence(rhs);
  int64_t modulus = std::gcd(left.modulus, right.modulus), residue;
  if (modulus && !llvm::SubOverflow(right.residue, left.residue, residue)) {
    int64_t adjustment =
        positiveRemainder(positiveRemainder(residue, modulus) -
                              positiveRemainder(*lower, modulus),
                          modulus);
    int64_t aligned;
    if (!llvm::AddOverflow(*lower, adjustment, aligned))
      *lower = aligned;
  }
  return *lower >= (strict ? 1 : 0);
}

mlir::LogicalResult materializeAttentionVisibility(mlir::RewriterBase &rewriter,
                                                   TileRegionOp region) {
  llvm::SmallVector<LinalgExtOnlineAttentionOp> tiles;
  region.walk([&](LinalgExtOnlineAttentionOp tile) {
    if (!tile.getPositions().empty())
      tiles.push_back(tile);
  });
  for (auto tile : tiles) {
    auto sizes = tile.getStaticLoopRanges();
    auto map = tile.getPositionMapAttr().getValue();
    for (auto expr : map.getResults())
      if (mlir::ShapedType::isDynamic(
              sizes[mlir::cast<mlir::AffineDimExpr>(expr).getPosition()]))
        return tile.emitOpError(
            "visibility requires statically sized selected tiles");
  }
  llvm::DenseSet<mlir::Operation *> visibleIterations;
  for (auto tile : tiles)
    if (tightenVisibleLoop(rewriter, tile))
      visibleIterations.insert(tile);
  for (auto tile : tiles) {
    mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPoint(tile);
    auto loc = tile.getLoc();
    auto sizes = tile.getStaticLoopRanges();
    auto map = tile.getPositionMapAttr().getValue();
    auto query = tile.getPositions()[0], key = tile.getPositions()[1];
    auto validEnd = tile.getPositions()[2];
    auto end = [&](mlir::Value start, unsigned position) {
      unsigned dimension =
          mlir::cast<mlir::AffineDimExpr>(map.getResult(position))
              .getPosition();
      auto extent =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizes[dimension]);
      return rewriter.createOrFold<mlir::arith::AddIOp>(loc, start, extent);
    };
    auto compare = [&](mlir::arith::CmpIPredicate predicate, mlir::Value lhs,
                       mlir::Value rhs) {
      if (proveAttentionPositionOrder(
              lhs, rhs, predicate == mlir::arith::CmpIPredicate::ult))
        return mlir::Value(
            rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1));
      if (proveAttentionPositionOrder(
              rhs, lhs, predicate == mlir::arith::CmpIPredicate::ule))
        return mlir::Value(
            rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1));
      return rewriter.createOrFold<mlir::arith::CmpIOp>(loc, predicate, lhs,
                                                        rhs);
    };
    auto visible = compare(mlir::arith::CmpIPredicate::ult, key, validEnd);
    auto full = compare(mlir::arith::CmpIPredicate::ule, end(key, 1), validEnd);
    if (tile.getCausal()) {
      auto causalVisible =
          compare(mlir::arith::CmpIPredicate::ult, key, end(query, 0));
      visible = rewriter.createOrFold<mlir::arith::AndIOp>(loc, visible,
                                                           causalVisible);
      auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto firstQueryEnd =
          rewriter.createOrFold<mlir::arith::AddIOp>(loc, query, one);
      auto causalFull =
          compare(mlir::arith::CmpIPredicate::ule, end(key, 1), firstQueryEnd);
      full = rewriter.createOrFold<mlir::arith::AndIOp>(loc, full, causalFull);
    }
    if (visibleIterations.contains(tile))
      visible = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    // Select constant predicates here: visibility is an actual materialization
    // boundary, so a downstream consumer must not need canonicalization to
    // discover that a published state has one storage owner.
    using Values = llvm::SmallVector<mlir::Value>;
    auto choose = [&](mlir::Value condition, auto thenBuilder,
                      auto elseBuilder) {
      llvm::APInt constant;
      if (mlir::matchPattern(condition, mlir::m_ConstantInt(&constant)))
        return constant.isZero() ? elseBuilder() : thenBuilder();
      auto branch = rewriter.create<mlir::scf::IfOp>(
          loc, tile->getResultTypes(), condition, true);
      rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
      rewriter.create<mlir::scf::YieldOp>(loc, thenBuilder());
      rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
      rewriter.create<mlir::scf::YieldOp>(loc, elseBuilder());
      rewriter.setInsertionPointAfter(branch);
      return Values(branch->getResults());
    };
    auto selected = choose(
        visible,
        [&]() -> Values {
          // Move pure, single-use input views only into an actual guarded
          // scope. Physical reads are subsequently materialized in this same
          // scope.
          for (mlir::Value input : {tile.getQuery(), tile.getKey(),
                                    tile.getValue(), tile.getMask()}) {
            if (!input)
              continue;
            auto slice = input.getDefiningOp<mlir::tensor::ExtractSliceOp>();
            if (slice && slice->hasOneUse() &&
                slice->getBlock() == tile->getBlock() &&
                rewriter.getInsertionBlock() != tile->getBlock())
              rewriter.moveOpBefore(slice, rewriter.getInsertionBlock(),
                                    rewriter.getInsertionPoint());
          }
          auto clone = [&](bool omitPositions) {
            mlir::IRMapping mapping;
            auto copy = mlir::cast<LinalgExtOnlineAttentionOp>(
                rewriter.clone(*tile, mapping));
            if (omitPositions) {
              copy.getPositionsMutable().clear();
              copy.setCausal(false);
              copy.removePositionMapAttr();
            }
            return Values(copy->getResults());
          };
          return choose(
              full, [&] { return clone(true); }, [&] { return clone(false); });
        },
        [&] {
          return Values{tile.getAccumulator(), tile.getMaximum(),
                        tile.getSum()};
        });
    rewriter.replaceOp(tile, selected);
  }
  return mlir::success();
}
} // namespace wafer
