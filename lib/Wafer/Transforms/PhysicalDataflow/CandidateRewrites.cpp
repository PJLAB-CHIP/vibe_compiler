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

enum class NumericDomain {
  ModularInteger,
  Floating,
};

enum class NumericBinaryKind {
  Add,
  Subtract,
  Multiply,
};

struct NumericBinary {
  mlir::Operation *operation;
  mlir::Value lhs;
  mlir::Value rhs;
  NumericDomain domain;

  mlir::Value getResult() const { return operation->getResult(0); }
  bool hasOneUse() const { return operation->hasOneUse(); }
};

static std::optional<NumericBinary>
matchNumericBinary(mlir::Value value, NumericBinaryKind kind) {
  auto matchInteger = [&](auto operation) -> std::optional<NumericBinary> {
    if (!operation || !isModularIntegerBinary(operation))
      return std::nullopt;
    return NumericBinary{operation, operation.getLhs(), operation.getRhs(),
                         NumericDomain::ModularInteger};
  };
  auto matchFloating = [&](auto operation) -> std::optional<NumericBinary> {
    if (!operation)
      return std::nullopt;
    auto resultType =
        mlir::dyn_cast<mlir::FloatType>(operation.getResult().getType());
    if (!resultType || operation.getLhs().getType() != resultType ||
        operation.getRhs().getType() != resultType)
      return std::nullopt;
    return NumericBinary{operation, operation.getLhs(), operation.getRhs(),
                         NumericDomain::Floating};
  };

  switch (kind) {
  case NumericBinaryKind::Add:
    if (auto integer = matchInteger(
            value.getDefiningOp<mlir::arith::AddIOp>()))
      return integer;
    return matchFloating(value.getDefiningOp<mlir::arith::AddFOp>());
  case NumericBinaryKind::Subtract:
    if (auto integer = matchInteger(
            value.getDefiningOp<mlir::arith::SubIOp>()))
      return integer;
    return matchFloating(value.getDefiningOp<mlir::arith::SubFOp>());
  case NumericBinaryKind::Multiply:
    if (auto integer = matchInteger(
            value.getDefiningOp<mlir::arith::MulIOp>()))
      return integer;
    return matchFloating(value.getDefiningOp<mlir::arith::MulFOp>());
  }
  llvm_unreachable("unknown numeric binary kind");
}

static bool hasUniformNumericDomain(
    llvm::ArrayRef<NumericBinary> operations) {
  if (operations.empty())
    return false;
  NumericDomain domain = operations.front().domain;
  return llvm::all_of(operations, [&](const NumericBinary &operation) {
    return operation.domain == domain;
  });
}

static mlir::Value
createNumericBinary(mlir::OpBuilder &builder, mlir::Location loc,
                    NumericBinaryKind kind, NumericDomain domain,
                    mlir::Value lhs, mlir::Value rhs) {
  if (domain == NumericDomain::ModularInteger) {
    switch (kind) {
    case NumericBinaryKind::Add:
      return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
    case NumericBinaryKind::Subtract:
      return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
    case NumericBinaryKind::Multiply:
      return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    }
  }

  switch (kind) {
  case NumericBinaryKind::Add:
    return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
  case NumericBinaryKind::Subtract:
    return builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
  case NumericBinaryKind::Multiply:
    return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
  }
  llvm_unreachable("unknown numeric binary kind");
}

struct CommonMultiplicand {
  mlir::Value factor;
  mlir::Value lhsOther;
  mlir::Value rhsOther;
};

static std::optional<CommonMultiplicand>
findCommonMultiplicand(const NumericBinary &lhs, const NumericBinary &rhs) {
  if (lhs.lhs == rhs.lhs)
    return CommonMultiplicand{lhs.lhs, lhs.rhs, rhs.rhs};
  if (lhs.lhs == rhs.rhs)
    return CommonMultiplicand{lhs.lhs, lhs.rhs, rhs.lhs};
  if (lhs.rhs == rhs.lhs)
    return CommonMultiplicand{lhs.rhs, lhs.lhs, rhs.rhs};
  if (lhs.rhs == rhs.rhs)
    return CommonMultiplicand{lhs.rhs, lhs.lhs, rhs.lhs};
  return std::nullopt;
}

template <typename Rewrite>
static unsigned rewriteNumericYields(mlir::func::FuncOp task,
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

unsigned reassociateElementwiseExpressions(mlir::func::FuncOp task) {
  return rewriteNumericYields(
      task, [](mlir::linalg::YieldOp yield, mlir::Value yielded) {
        std::optional<NumericBinary> outer =
            matchNumericBinary(yielded, NumericBinaryKind::Add);
        std::optional<NumericBinary> left =
            outer ? matchNumericBinary(outer->lhs, NumericBinaryKind::Add)
                  : std::nullopt;
        if (!outer || !left || !outer->hasOneUse() || !left->hasOneUse() ||
            !hasUniformNumericDomain({*outer, *left}))
          return false;
        mlir::OpBuilder builder(outer->operation);
        mlir::Value right = createNumericBinary(
            builder, outer->operation->getLoc(), NumericBinaryKind::Add,
            outer->domain, left->rhs, outer->rhs);
        mlir::Value reassociated = createNumericBinary(
            builder, outer->operation->getLoc(), NumericBinaryKind::Add,
            outer->domain, left->lhs, right);
        yield->setOperand(0, reassociated);
        eraseIfDead(outer->operation);
        eraseIfDead(left->operation);
        return true;
      });
}

unsigned balanceElementwiseReductionTrees(mlir::func::FuncOp task) {
  return rewriteNumericYields(
      task, [](mlir::linalg::YieldOp yield, mlir::Value yielded) {
        std::optional<NumericBinary> outer =
            matchNumericBinary(yielded, NumericBinaryKind::Add);
        std::optional<NumericBinary> middle =
            outer ? matchNumericBinary(outer->lhs, NumericBinaryKind::Add)
                  : std::nullopt;
        std::optional<NumericBinary> inner =
            middle ? matchNumericBinary(middle->lhs, NumericBinaryKind::Add)
                   : std::nullopt;
        if (!outer || !middle || !inner || !outer->hasOneUse() ||
            !middle->hasOneUse() || !inner->hasOneUse() ||
            !hasUniformNumericDomain({*outer, *middle, *inner}))
          return false;
        mlir::OpBuilder builder(outer->operation);
        mlir::Value right = createNumericBinary(
            builder, outer->operation->getLoc(), NumericBinaryKind::Add,
            outer->domain, middle->rhs, outer->rhs);
        mlir::Value balanced = createNumericBinary(
            builder, outer->operation->getLoc(), NumericBinaryKind::Add,
            outer->domain, inner->getResult(), right);
        yield->setOperand(0, balanced);
        eraseIfDead(outer->operation);
        eraseIfDead(middle->operation);
        return true;
      });
}

unsigned contractDistributiveExpressions(mlir::func::FuncOp task) {
  return rewriteNumericYields(
      task, [](mlir::linalg::YieldOp yield, mlir::Value yielded) {
        std::optional<NumericBinary> difference =
            matchNumericBinary(yielded, NumericBinaryKind::Subtract);
        std::optional<NumericBinary> lhs =
            difference
                ? matchNumericBinary(difference->lhs,
                                     NumericBinaryKind::Multiply)
                : std::nullopt;
        std::optional<NumericBinary> rhs =
            difference
                ? matchNumericBinary(difference->rhs,
                                     NumericBinaryKind::Multiply)
                : std::nullopt;
        if (!difference || !lhs || !rhs || !difference->hasOneUse() ||
            !lhs->hasOneUse() || !rhs->hasOneUse() ||
            !hasUniformNumericDomain({*difference, *lhs, *rhs}))
          return false;

        std::optional<CommonMultiplicand> common =
            findCommonMultiplicand(*lhs, *rhs);
        if (!common)
          return false;

        mlir::OpBuilder builder(difference->operation);
        mlir::Value terms = createNumericBinary(
            builder, difference->operation->getLoc(),
            NumericBinaryKind::Subtract, difference->domain, common->lhsOther,
            common->rhsOther);
        mlir::Value contracted = createNumericBinary(
            builder, difference->operation->getLoc(),
            NumericBinaryKind::Multiply, difference->domain, common->factor,
            terms);
        yield->setOperand(0, contracted);
        eraseIfDead(difference->operation);
        eraseIfDead(lhs->operation);
        eraseIfDead(rhs->operation);
        return true;
      });
}

unsigned factorElementwiseExpressions(mlir::func::FuncOp task) {
  return rewriteNumericYields(
      task, [](mlir::linalg::YieldOp yield, mlir::Value yielded) {
        std::optional<NumericBinary> sum =
            matchNumericBinary(yielded, NumericBinaryKind::Add);
        std::optional<NumericBinary> lhs =
            sum ? matchNumericBinary(sum->lhs, NumericBinaryKind::Multiply)
                : std::nullopt;
        std::optional<NumericBinary> rhs =
            sum ? matchNumericBinary(sum->rhs, NumericBinaryKind::Multiply)
                : std::nullopt;
        if (!sum || !lhs || !rhs || !sum->hasOneUse() ||
            !lhs->hasOneUse() || !rhs->hasOneUse() ||
            !hasUniformNumericDomain({*sum, *lhs, *rhs}))
          return false;

        std::optional<CommonMultiplicand> common =
            findCommonMultiplicand(*lhs, *rhs);
        if (!common)
          return false;

        mlir::OpBuilder builder(sum->operation);
        mlir::Value terms = createNumericBinary(
            builder, sum->operation->getLoc(), NumericBinaryKind::Add,
            sum->domain, common->lhsOther, common->rhsOther);
        mlir::Value factored = createNumericBinary(
            builder, sum->operation->getLoc(), NumericBinaryKind::Multiply,
            sum->domain, common->factor, terms);
        yield->setOperand(0, factored);
        eraseIfDead(sum->operation);
        eraseIfDead(lhs->operation);
        eraseIfDead(rhs->operation);
        return true;
      });
}

} // namespace wafer
