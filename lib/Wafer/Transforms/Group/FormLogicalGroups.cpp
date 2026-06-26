//===- FormLogicalGroups.cpp - Wafer logical group formation -------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer {
#define GEN_PASS_DEF_FORMLOGICALGROUPSPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

static bool isTensorType(mlir::Type type) {
  return mlir::isa<mlir::TensorType>(type);
}

static bool hasTensorResults(mlir::Operation *op) {
  return op->getNumResults() > 0 &&
         llvm::all_of(op->getResultTypes(), isTensorType);
}

static bool hasTensorInits(mlir::DestinationStyleOpInterface dpsOp) {
  return dpsOp.getNumDpsInits() > 0 &&
         llvm::all_of(dpsOp.getDpsInits().getTypes(), isTensorType);
}

static bool hasDuplicateBoundaryValues(mlir::ValueRange inputs,
                                       mlir::ValueRange outs) {
  llvm::DenseSet<mlir::Value> seen;
  for (mlir::Value input : inputs) {
    if (!seen.insert(input).second)
      return true;
  }
  for (mlir::Value out : outs) {
    if (!seen.insert(out).second)
      return true;
  }
  return false;
}

static bool isLinalgExtCollectiveRoot(mlir::Operation *op) {
  return mlir::isa<WaferLinalgExtCollectiveOpInterface>(op);
}

static bool isLinalgRoot(mlir::Operation *op) {
  if (!mlir::isa<mlir::linalg::LinalgOp>(op))
    return false;

  // A fill is usually the init producer for the real root. Grouping it alone
  // adds a boundary that does not help downstream tile feasibility.
  return !mlir::isa<mlir::linalg::FillOp>(op);
}

static bool isTensorLevelGroupableDpsOp(mlir::Operation *op) {
  if (op->getParentOfType<GroupOp>())
    return false;
  if (!hasTensorResults(op))
    return false;

  auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
  if (!dpsOp || !hasTensorInits(dpsOp))
    return false;
  if (!dpsOp.hasPureTensorSemantics())
    return false;

  return mlir::isa<mlir::linalg::LinalgOp>(op) || isLinalgExtCollectiveRoot(op);
}

static bool isLogicalGroupRoot(mlir::Operation *op) {
  if (!isTensorLevelGroupableDpsOp(op))
    return false;
  return isLinalgRoot(op) || isLinalgExtCollectiveRoot(op);
}

static bool isInternalSupportOp(mlir::Operation *op) {
  if (op->getParentOfType<GroupOp>())
    return false;
  if (op->getNumResults() == 0)
    return false;

  return mlir::isa<mlir::arith::ConstantOp, mlir::tensor::EmptyOp>(op);
}

static bool isSelectedDef(mlir::Value value,
                          const llvm::DenseSet<mlir::Operation *> &selected) {
  mlir::Operation *def = value.getDefiningOp();
  return def && selected.contains(def);
}

static bool allUsesInsideOrAllowedConsumer(
    mlir::Value value, const llvm::DenseSet<mlir::Operation *> &selected,
    mlir::Operation *allowedConsumer = nullptr) {
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (!selected.contains(owner) && owner != allowedConsumer)
      return false;
  }
  return true;
}

static bool
allResultsUsedBySelected(mlir::Operation *op,
                         const llvm::DenseSet<mlir::Operation *> &selected) {
  return llvm::all_of(op->getResults(), [&](mlir::Value result) {
    return allUsesInsideOrAllowedConsumer(result, selected);
  });
}

static bool hasExternalUse(mlir::Value value,
                           const llvm::DenseSet<mlir::Operation *> &selected) {
  for (mlir::OpOperand &use : value.getUses()) {
    if (!selected.contains(use.getOwner()))
      return true;
  }
  return false;
}

static bool
canAbsorbProducer(mlir::Operation *producer, mlir::Block *block,
                  const llvm::DenseSet<mlir::Operation *> &selected) {
  if (!producer || producer->getBlock() != block || selected.contains(producer))
    return false;
  if (!isTensorLevelGroupableDpsOp(producer))
    return false;

  return allResultsUsedBySelected(producer, selected);
}

static bool
canAbsorbConsumer(mlir::Operation *consumer, mlir::Block *block,
                  const llvm::DenseSet<mlir::Operation *> &selected) {
  if (!consumer || consumer->getBlock() != block || selected.contains(consumer))
    return false;
  if (!isTensorLevelGroupableDpsOp(consumer))
    return false;

  bool consumesSelectedValue = false;
  for (mlir::Value operand : consumer->getOperands()) {
    if (!isSelectedDef(operand, selected))
      continue;

    consumesSelectedValue = true;
    if (!allUsesInsideOrAllowedConsumer(operand, selected, consumer))
      return false;
  }

  return consumesSelectedValue;
}

static void
orderSelectedOps(mlir::Block *block,
                 const llvm::DenseSet<mlir::Operation *> &selected,
                 llvm::SmallVectorImpl<mlir::Operation *> &orderedOps) {
  orderedOps.clear();
  for (mlir::Operation &op : *block) {
    if (selected.contains(&op))
      orderedOps.push_back(&op);
  }
}

static void
expandLogicalGroupSelection(mlir::Operation *root,
                            llvm::DenseSet<mlir::Operation *> &selected) {
  mlir::Block *block = root->getBlock();
  selected.insert(root);

  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *> orderedOps;
    orderSelectedOps(block, selected, orderedOps);

    for (mlir::Operation *op : orderedOps) {
      for (mlir::Value operand : op->getOperands()) {
        if (mlir::Operation *producer = operand.getDefiningOp()) {
          if (canAbsorbProducer(producer, block, selected)) {
            selected.insert(producer);
            changed = true;
          }
        }
      }
    }

    orderSelectedOps(block, selected, orderedOps);
    for (mlir::Operation *op : orderedOps) {
      for (mlir::Value result : op->getResults()) {
        llvm::SmallVector<mlir::OpOperand *> uses;
        for (mlir::OpOperand &use : result.getUses())
          uses.push_back(&use);

        for (mlir::OpOperand *use : uses) {
          mlir::Operation *consumer = use->getOwner();
          if (canAbsorbConsumer(consumer, block, selected)) {
            selected.insert(consumer);
            changed = true;
          }
        }
      }
    }
  }
}

static mlir::Value
getExternalDpsInitForResult(mlir::Value value,
                            const llvm::DenseSet<mlir::Operation *> &selected) {
  mlir::Value current = value;
  while (auto result = mlir::dyn_cast<mlir::OpResult>(current)) {
    auto dpsOp =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(result.getOwner());
    if (!dpsOp)
      return {};

    if (result.getResultNumber() >=
        static_cast<unsigned>(dpsOp.getNumDpsInits()))
      return {};
    mlir::OpOperand *tiedOperand =
        dpsOp.getDpsInitOperand(result.getResultNumber());
    mlir::Value init = tiedOperand->get();
    mlir::Operation *initDef = init.getDefiningOp();
    if (!initDef || !selected.contains(initDef))
      return init;

    current = init;
  }

  return {};
}

static bool
collectYieldedValuesAndOuts(llvm::ArrayRef<mlir::Operation *> orderedOps,
                            const llvm::DenseSet<mlir::Operation *> &selected,
                            llvm::SmallVectorImpl<mlir::Value> &yieldedValues,
                            llvm::SmallVectorImpl<mlir::Value> &outs) {
  yieldedValues.clear();
  outs.clear();

  for (mlir::Operation *op : orderedOps) {
    for (mlir::Value result : op->getResults()) {
      if (hasExternalUse(result, selected))
        yieldedValues.push_back(result);
    }
  }

  if (yieldedValues.empty())
    return false;

  for (mlir::Value yielded : yieldedValues) {
    mlir::Value out = getExternalDpsInitForResult(yielded, selected);
    if (!out || !mlir::isa<mlir::TensorType>(out.getType()))
      return false;
    outs.push_back(out);
  }

  return true;
}

static void
absorbInternalSupportOps(mlir::Block *block,
                         llvm::DenseSet<mlir::Operation *> &selected,
                         mlir::ValueRange outs) {
  llvm::DenseSet<mlir::Value> outSet;
  for (mlir::Value out : outs)
    outSet.insert(out);

  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *> orderedOps;
    orderSelectedOps(block, selected, orderedOps);

    for (mlir::Operation *op : orderedOps) {
      for (mlir::Value operand : op->getOperands()) {
        if (outSet.contains(operand))
          continue;

        mlir::Operation *producer = operand.getDefiningOp();
        if (!producer || producer->getBlock() != block ||
            selected.contains(producer))
          continue;
        if (!isInternalSupportOp(producer))
          continue;
        if (!allResultsUsedBySelected(producer, selected))
          continue;

        selected.insert(producer);
        changed = true;
      }
    }
  }
}

static void
collectBoundaryInputs(llvm::ArrayRef<mlir::Operation *> orderedOps,
                      const llvm::DenseSet<mlir::Operation *> &selected,
                      mlir::ValueRange outs,
                      llvm::SmallVectorImpl<mlir::Value> &inputs) {
  llvm::DenseSet<mlir::Value> seen;
  for (mlir::Value out : outs)
    seen.insert(out);

  inputs.clear();
  for (mlir::Operation *op : orderedOps) {
    op->walk([&](mlir::Operation *nestedOp) {
      for (mlir::Value operand : nestedOp->getOperands()) {
        if (isSelectedDef(operand, selected))
          continue;

        bool definedInsideSelected = false;
        if (mlir::Operation *def = operand.getDefiningOp()) {
          for (mlir::Operation *scope = def; scope;
               scope = scope->getParentOp()) {
            if (selected.contains(scope)) {
              definedInsideSelected = true;
              break;
            }
          }
        } else if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(
                       operand)) {
          mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
          for (mlir::Operation *scope = parentOp; scope;
               scope = scope->getParentOp()) {
            if (selected.contains(scope)) {
              definedInsideSelected = true;
              break;
            }
          }
        }
        if (definedInsideSelected)
          continue;
        if (!seen.insert(operand).second)
          continue;
        inputs.push_back(operand);
      }
    });
  }
}

struct LogicalGroupSelection {
  llvm::DenseSet<mlir::Operation *> selected;
  llvm::SmallVector<mlir::Operation *> orderedOps;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outs;
  llvm::SmallVector<mlir::Value> yieldedValues;
};

static bool buildLogicalGroupSelection(mlir::Operation *root,
                                       LogicalGroupSelection &selection) {
  expandLogicalGroupSelection(root, selection.selected);
  orderSelectedOps(root->getBlock(), selection.selected, selection.orderedOps);

  if (!collectYieldedValuesAndOuts(selection.orderedOps, selection.selected,
                                   selection.yieldedValues, selection.outs))
    return false;

  absorbInternalSupportOps(root->getBlock(), selection.selected,
                           selection.outs);
  orderSelectedOps(root->getBlock(), selection.selected, selection.orderedOps);
  collectBoundaryInputs(selection.orderedOps, selection.selected,
                        selection.outs, selection.inputs);

  if (hasDuplicateBoundaryValues(selection.inputs, selection.outs))
    return false;

  return true;
}

static mlir::LogicalResult
formGroupForRoot(mlir::Operation *root,
                 llvm::DenseSet<mlir::Operation *> &consumed) {
  LogicalGroupSelection selection;
  if (!buildLogicalGroupSelection(root, selection))
    return mlir::success();

  for (mlir::Operation *op : selection.selected)
    consumed.insert(op);

  mlir::Operation *insertionPoint = selection.orderedOps.back();
  mlir::OpBuilder builder(insertionPoint);
  auto group =
      builder.create<GroupOp>(root->getLoc(), mlir::TypeRange(selection.outs),
                              selection.inputs, selection.outs);

  mlir::Region &body = group.getBody();
  body.push_back(new mlir::Block);
  mlir::Block &block = body.front();
  for (mlir::Value input : selection.inputs)
    block.addArgument(input.getType(), input.getLoc());
  for (mlir::Value out : selection.outs)
    block.addArgument(out.getType(), out.getLoc());

  mlir::IRMapping mapping;
  unsigned blockArgIndex = 0;
  for (mlir::Value input : selection.inputs)
    mapping.map(input, block.getArgument(blockArgIndex++));
  for (mlir::Value out : selection.outs)
    mapping.map(out, block.getArgument(blockArgIndex++));

  mlir::OpBuilder bodyBuilder(&block, block.end());
  for (mlir::Operation *op : selection.orderedOps)
    bodyBuilder.clone(*op, mapping);

  llvm::SmallVector<mlir::Value> yieldedValues;
  for (mlir::Value yielded : selection.yieldedValues)
    yieldedValues.push_back(mapping.lookup(yielded));
  bodyBuilder.create<GroupYieldOp>(root->getLoc(), yieldedValues);

  for (auto [result, yielded] :
       llvm::zip_equal(group.getResults(), selection.yieldedValues)) {
    llvm::SmallVector<mlir::OpOperand *> externalUses;
    for (mlir::OpOperand &use : yielded.getUses()) {
      if (!selection.selected.contains(use.getOwner()))
        externalUses.push_back(&use);
    }
    for (mlir::OpOperand *use : externalUses)
      use->set(result);
  }

  for (mlir::Operation *op : llvm::reverse(selection.orderedOps))
    op->erase();
  return mlir::success();
}

struct FormLogicalGroupsPass
    : public impl::FormLogicalGroupsPassBase<FormLogicalGroupsPass> {
  using impl::FormLogicalGroupsPassBase<
      FormLogicalGroupsPass>::FormLogicalGroupsPassBase;

  void runOnOperation() final {
    llvm::SmallVector<mlir::Operation *> roots;
    getOperation().walk([&](mlir::Operation *op) {
      if (isLogicalGroupRoot(op))
        roots.push_back(op);
    });

    llvm::DenseSet<mlir::Operation *> consumed;
    for (mlir::Operation *root : roots) {
      if (consumed.contains(root))
        continue;
      if (!root->getBlock())
        continue;
      if (root->getParentOfType<GroupOp>())
        continue;
      if (mlir::failed(formGroupForRoot(root, consumed))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace wafer
