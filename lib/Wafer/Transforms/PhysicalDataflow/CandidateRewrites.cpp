//===- CandidateRewrites.cpp - Actual physical-dataflow clones ----------===//

#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace wafer {
namespace {

static bool isPureTensorProducer(mlir::linalg::LinalgOp producer) {
  if (producer->getNumResults() != 1 ||
      !mlir::isa<mlir::RankedTensorType>(producer->getResult(0).getType()) ||
      !mlir::isMemoryEffectFree(producer) || !mlir::isSpeculatable(producer))
    return false;
  return llvm::all_of(producer->getOperands(), [](mlir::Value operand) {
    return !mlir::isa<mlir::MemRefType>(operand.getType());
  });
}

static bool isCompatibleConsumerUse(mlir::OpOperand &use,
                                    mlir::linalg::LinalgOp producer) {
  auto consumer = mlir::dyn_cast<mlir::linalg::LinalgOp>(use.getOwner());
  return consumer && consumer->getBlock() == producer->getBlock() &&
         producer->isBeforeInBlock(consumer) &&
         use.getOperandNumber() < consumer.getNumDpsInputs();
}

static bool hasNoOverflowPromise(mlir::Operation *operation) {
  if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(operation))
    return add.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none;
  if (auto sub = mlir::dyn_cast<mlir::arith::SubIOp>(operation))
    return sub.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none;
  if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(operation))
    return mul.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none;
  return true;
}

static bool isModularIntegerBinary(mlir::Operation *operation) {
  if (!operation || hasNoOverflowPromise(operation) ||
      operation->getNumOperands() != 2 || operation->getNumResults() != 1)
    return false;
  auto integer = mlir::dyn_cast<mlir::IntegerType>(
      operation->getResult(0).getType());
  return integer && integer.getWidth() > 1 &&
         operation->getOperand(0).getType() == integer &&
         operation->getOperand(1).getType() == integer;
}

static mlir::arith::AddIOp asModularAdd(mlir::Value value) {
  auto add = value.getDefiningOp<mlir::arith::AddIOp>();
  return add && isModularIntegerBinary(add) ? add : mlir::arith::AddIOp{};
}

static mlir::arith::MulIOp asModularMul(mlir::Value value) {
  auto mul = value.getDefiningOp<mlir::arith::MulIOp>();
  return mul && isModularIntegerBinary(mul) ? mul : mlir::arith::MulIOp{};
}

static mlir::arith::SubIOp asModularSub(mlir::Value value) {
  auto sub = value.getDefiningOp<mlir::arith::SubIOp>();
  return sub && isModularIntegerBinary(sub) ? sub : mlir::arith::SubIOp{};
}

struct CommonMultiplicand {
  mlir::Value factor;
  mlir::Value lhsOther;
  mlir::Value rhsOther;
};

static std::optional<CommonMultiplicand>
findCommonMultiplicand(mlir::arith::MulIOp lhs,
                       mlir::arith::MulIOp rhs) {
  if (lhs.getLhs() == rhs.getLhs())
    return CommonMultiplicand{lhs.getLhs(), lhs.getRhs(), rhs.getRhs()};
  if (lhs.getLhs() == rhs.getRhs())
    return CommonMultiplicand{lhs.getLhs(), lhs.getRhs(), rhs.getLhs()};
  if (lhs.getRhs() == rhs.getLhs())
    return CommonMultiplicand{lhs.getRhs(), lhs.getLhs(), rhs.getRhs()};
  if (lhs.getRhs() == rhs.getRhs())
    return CommonMultiplicand{lhs.getRhs(), lhs.getLhs(), rhs.getLhs()};
  return std::nullopt;
}

template <typename Rewrite>
static unsigned rewriteIntegerYields(mlir::func::FuncOp task,
                                     Rewrite &&rewrite) {
  unsigned changed = 0;
  task.walk([&](mlir::linalg::GenericOp generic) {
    if (!generic.getRegion().hasOneBlock())
      return;
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
        generic.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return;
    changed += rewrite(yield, yield.getValues().front()) ? 1u : 0u;
  });
  return changed;
}

static void eraseIfDead(mlir::Operation *operation) {
  if (operation && operation->use_empty())
    operation->erase();
}

} // namespace

unsigned
materializeConsumerLocalTensorRecomputation(mlir::func::FuncOp task) {
  llvm::SmallVector<mlir::linalg::LinalgOp, 8> producers;
  task.walk([&](mlir::linalg::LinalgOp producer) {
    if (isPureTensorProducer(producer))
      producers.push_back(producer);
  });

  unsigned changed = 0;
  for (mlir::linalg::LinalgOp producer : producers) {
    mlir::Value result = producer->getResult(0);
    llvm::SmallVector<mlir::OpOperand *, 4> compatibleUses;
    bool hasIncompatibleUse = false;
    for (mlir::OpOperand &use : result.getUses()) {
      if (isCompatibleConsumerUse(use, producer))
        compatibleUses.push_back(&use);
      else
        hasIncompatibleUse = true;
    }
    if (hasIncompatibleUse || compatibleUses.size() < 2)
      continue;

    llvm::stable_sort(compatibleUses,
                      [](mlir::OpOperand *lhs, mlir::OpOperand *rhs) {
                        return lhs->getOwner()->isBeforeInBlock(rhs->getOwner());
                      });
    // The original producer supplies the first consumer. Every later
    // consumer receives a real producer clone placed at that use site.
    for (mlir::OpOperand *use : llvm::drop_begin(compatibleUses)) {
      mlir::Operation *consumer = use->getOwner();
      mlir::OpBuilder builder(consumer);
      mlir::IRMapping mapping;
      mlir::Operation *clone = builder.clone(*producer.getOperation(), mapping);
      use->set(clone->getResult(0));
      ++changed;
    }
  }
  return changed;
}

unsigned hoistStaticLoopInvariantOperations(mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
  function->walk<mlir::WalkOrder::PostOrder>([&](mlir::Operation *operation) {
    if (auto loop = mlir::dyn_cast<mlir::LoopLikeOpInterface>(operation))
      loops.push_back(loop);
  });
  unsigned changed = 0;
  for (mlir::LoopLikeOpInterface loop : loops)
    changed += static_cast<unsigned>(mlir::moveLoopInvariantCode(loop));
  return changed;
}

unsigned reassociateIntegerElementwiseExpressions(mlir::func::FuncOp task) {
  return rewriteIntegerYields(
      task, [](mlir::linalg::YieldOp yield, mlir::Value yielded) {
        auto outer = asModularAdd(yielded);
        auto left = outer ? asModularAdd(outer.getLhs()) : mlir::arith::AddIOp{};
        if (!outer || !left || !outer->hasOneUse() || !left->hasOneUse())
          return false;
        mlir::OpBuilder builder(outer);
        auto right = builder.create<mlir::arith::AddIOp>(
            outer.getLoc(), left.getRhs(), outer.getRhs());
        auto reassociated = builder.create<mlir::arith::AddIOp>(
            outer.getLoc(), left.getLhs(), right);
        yield->setOperand(0, reassociated);
        eraseIfDead(outer);
        eraseIfDead(left);
        return true;
      });
}

unsigned balanceIntegerElementwiseReductionTrees(mlir::func::FuncOp task) {
  return rewriteIntegerYields(
      task, [](mlir::linalg::YieldOp yield, mlir::Value yielded) {
        auto outer = asModularAdd(yielded);
        auto middle =
            outer ? asModularAdd(outer.getLhs()) : mlir::arith::AddIOp{};
        auto inner =
            middle ? asModularAdd(middle.getLhs()) : mlir::arith::AddIOp{};
        if (!outer || !middle || !inner || !outer->hasOneUse() ||
            !middle->hasOneUse() || !inner->hasOneUse())
          return false;
        mlir::OpBuilder builder(outer);
        auto right = builder.create<mlir::arith::AddIOp>(
            outer.getLoc(), middle.getRhs(), outer.getRhs());
        auto balanced = builder.create<mlir::arith::AddIOp>(
            outer.getLoc(), inner.getResult(), right.getResult());
        yield->setOperand(0, balanced);
        eraseIfDead(outer);
        eraseIfDead(middle);
        return true;
      });
}

unsigned contractIntegerDistributiveExpressions(mlir::func::FuncOp task) {
  return rewriteIntegerYields(
      task, [](mlir::linalg::YieldOp yield, mlir::Value yielded) {
        auto difference = asModularSub(yielded);
        auto lhs = difference ? asModularMul(difference.getLhs())
                              : mlir::arith::MulIOp{};
        auto rhs = difference ? asModularMul(difference.getRhs())
                              : mlir::arith::MulIOp{};
        if (!difference || !lhs || !rhs || !difference->hasOneUse() ||
            !lhs->hasOneUse() || !rhs->hasOneUse())
          return false;

        std::optional<CommonMultiplicand> common =
            findCommonMultiplicand(lhs, rhs);
        if (!common)
          return false;

        mlir::OpBuilder builder(difference);
        auto terms = builder.create<mlir::arith::SubIOp>(
            difference.getLoc(), common->lhsOther, common->rhsOther);
        auto contracted = builder.create<mlir::arith::MulIOp>(
            difference.getLoc(), common->factor, terms);
        yield->setOperand(0, contracted);
        eraseIfDead(difference);
        eraseIfDead(lhs);
        eraseIfDead(rhs);
        return true;
      });
}

unsigned factorIntegerElementwiseExpressions(mlir::func::FuncOp task) {
  return rewriteIntegerYields(
      task, [](mlir::linalg::YieldOp yield, mlir::Value yielded) {
        auto sum = asModularAdd(yielded);
        auto lhs = sum ? asModularMul(sum.getLhs()) : mlir::arith::MulIOp{};
        auto rhs = sum ? asModularMul(sum.getRhs()) : mlir::arith::MulIOp{};
        if (!sum || !lhs || !rhs || !sum->hasOneUse() ||
            !lhs->hasOneUse() || !rhs->hasOneUse())
          return false;

        std::optional<CommonMultiplicand> common =
            findCommonMultiplicand(lhs, rhs);
        if (!common)
          return false;

        mlir::OpBuilder builder(sum);
        auto terms = builder.create<mlir::arith::AddIOp>(
            sum.getLoc(), common->lhsOther, common->rhsOther);
        auto factored = builder.create<mlir::arith::MulIOp>(
            sum.getLoc(), common->factor, terms);
        yield->setOperand(0, factored);
        eraseIfDead(sum);
        eraseIfDead(lhs);
        eraseIfDead(rhs);
        return true;
      });
}

} // namespace wafer
