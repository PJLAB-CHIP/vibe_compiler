//===- StructuredSchedulingScope.cpp - Tensor scheduling task scope ------===//

#include "Scheduling/StructuredSchedulingScope.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/Target/TopologyUtils.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace wafer::structured_scheduler {
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

static bool isLinalgExtCollectiveRoot(mlir::Operation *op) {
  return mlir::isa_and_nonnull<WaferLinalgExtCollectiveOpInterface>(op);
}

static bool isLinalgRoot(mlir::Operation *op) {
  if (!mlir::isa<mlir::linalg::LinalgOp>(op))
    return false;
  return !mlir::isa<mlir::linalg::FillOp>(op);
}

static bool isEligibleTensorLevelDpsOp(mlir::Operation *op) {
  if (!hasTensorResults(op))
    return false;
  auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
  if (!dpsOp || !hasTensorInits(dpsOp) || !dpsOp.hasPureTensorSemantics())
    return false;
  return mlir::isa<mlir::linalg::LinalgOp>(op) || isLinalgExtCollectiveRoot(op);
}

static bool isTensorLevelSchedulableDpsOp(mlir::Operation *op) {
  return isEligibleTensorLevelDpsOp(op);
}

static bool isInternalSupportOp(mlir::Operation *op) {
  if (op->getNumResults() == 0)
    return false;
  auto allStatic = [](llvm::ArrayRef<int64_t> values) {
    return llvm::all_of(values, [](int64_t value) {
      return !mlir::ShapedType::isDynamic(value);
    });
  };
  if (mlir::isa<mlir::arith::ConstantOp, mlir::tensor::EmptyOp,
                mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp>(op))
    return true;
  if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op))
    return allStatic(extract.getStaticOffsets()) &&
           allStatic(extract.getStaticSizes()) &&
           allStatic(extract.getStaticStrides());
  if (auto insert = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op))
    return allStatic(insert.getStaticOffsets()) &&
           allStatic(insert.getStaticSizes()) &&
           allStatic(insert.getStaticStrides());
  return false;
}

static bool isSelectedDef(mlir::Value value,
                          const llvm::DenseSet<mlir::Operation *> &selected) {
  mlir::Operation *def = value.getDefiningOp();
  return def && selected.contains(def);
}

static bool
isInsideSelectedOperation(mlir::Operation *operation,
                          const llvm::DenseSet<mlir::Operation *> &selected) {
  for (mlir::Operation *scope = operation; scope; scope = scope->getParentOp())
    if (selected.contains(scope))
      return true;
  return false;
}

static bool allUsesInsideOrAllowedConsumer(
    mlir::Value value, const llvm::DenseSet<mlir::Operation *> &selected,
    mlir::Operation *allowedConsumer = nullptr) {
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (!isInsideSelectedOperation(owner, selected) && owner != allowedConsumer)
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
  return llvm::any_of(value.getUses(), [&](mlir::OpOperand &use) {
    return !isInsideSelectedOperation(use.getOwner(), selected);
  });
}

static bool haveEqualRankedTensorShapes(mlir::Type lhs, mlir::Type rhs) {
  auto lhsTensor = mlir::dyn_cast<mlir::RankedTensorType>(lhs);
  auto rhsTensor = mlir::dyn_cast<mlir::RankedTensorType>(rhs);
  return lhsTensor && rhsTensor && lhsTensor.getShape() == rhsTensor.getShape();
}

static bool hasShapeCompatibleSelectedConsumers(
    mlir::Operation *producer,
    const llvm::DenseSet<mlir::Operation *> &selected) {
  for (mlir::Value result : producer->getResults()) {
    if (!mlir::isa<mlir::RankedTensorType>(result.getType()))
      continue;
    for (mlir::OpOperand &use : result.getUses()) {
      mlir::Operation *consumer = use.getOwner();
      if (!selected.contains(consumer))
        continue;
      for (mlir::Value consumerResult : consumer->getResults())
        if (mlir::isa<mlir::RankedTensorType>(consumerResult.getType()) &&
            !haveEqualRankedTensorShapes(result.getType(),
                                         consumerResult.getType()))
          return false;
    }
  }
  return true;
}

static bool hasCompatibleRootFamily(mlir::Operation *lhs,
                                    mlir::Operation *rhs) {
  return isLinalgExtCollectiveRoot(lhs) == isLinalgExtCollectiveRoot(rhs);
}

static mlir::RankedTensorType getSingleStaticResultType(mlir::Operation *op) {
  if (op->getNumResults() != 1)
    return {};
  auto type =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  return type && type.hasStaticShape() ? type : mlir::RankedTensorType{};
}

static bool
hasDirectSelectedDataflow(mlir::Operation *candidate,
                          const llvm::DenseSet<mlir::Operation *> &selected) {
  for (mlir::Value operand : candidate->getOperands())
    if (isSelectedDef(operand, selected))
      return true;
  for (mlir::Value result : candidate->getResults())
    for (mlir::OpOperand &use : result.getUses())
      if (isInsideSelectedOperation(use.getOwner(), selected))
        return true;
  return false;
}

static bool sharesExternalTensorDpsInput(
    mlir::Operation *candidate,
    const llvm::DenseSet<mlir::Operation *> &selected) {
  auto candidateDps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(candidate);
  if (!candidateDps)
    return false;
  llvm::DenseSet<mlir::Value> candidateInputs;
  for (mlir::Value input : candidateDps.getDpsInputs())
    if (mlir::isa<mlir::RankedTensorType>(input.getType()) &&
        !isSelectedDef(input, selected))
      candidateInputs.insert(input);
  if (candidateInputs.empty())
    return false;
  for (mlir::Operation *selectedOp : selected) {
    auto selectedDps =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(selectedOp);
    if (!selectedDps)
      continue;
    for (mlir::Value input : selectedDps.getDpsInputs())
      if (candidateInputs.contains(input) && !isSelectedDef(input, selected))
        return true;
  }
  return false;
}

static bool
canAbsorbSharedInputPeer(mlir::Operation *candidate, mlir::Operation *anchor,
                         mlir::Block *block,
                         const llvm::DenseSet<mlir::Operation *> &selected,
                         const llvm::DenseSet<mlir::Operation *> &forbidden) {
  if (!candidate || candidate->getBlock() != block ||
      selected.contains(candidate) || forbidden.contains(candidate) ||
      !isEligibleStructuredSchedulingRoot(candidate))
    return false;
  if (!mlir::isa_and_nonnull<mlir::func::FuncOp>(block->getParentOp()))
    return false;
  if (!mlir::isa<mlir::linalg::LinalgOp>(anchor) ||
      !mlir::isa<mlir::linalg::LinalgOp>(candidate))
    return false;
  if (!hasCompatibleRootFamily(anchor, candidate) ||
      anchor->getName() != candidate->getName())
    return false;
  mlir::RankedTensorType anchorType = getSingleStaticResultType(anchor);
  mlir::RankedTensorType candidateType = getSingleStaticResultType(candidate);
  if (!anchorType || !candidateType ||
      anchorType.getShape() != candidateType.getShape())
    return false;
  if (hasDirectSelectedDataflow(candidate, selected))
    return false;
  return sharesExternalTensorDpsInput(candidate, selected);
}

static bool hasFamilyCompatibleSelectedConsumers(
    mlir::Operation *producer,
    const llvm::DenseSet<mlir::Operation *> &selected) {
  for (mlir::Value result : producer->getResults())
    for (mlir::OpOperand &use : result.getUses())
      if (selected.contains(use.getOwner()) &&
          !hasCompatibleRootFamily(producer, use.getOwner()))
        return false;
  return true;
}

static bool hasShapeCompatibleSelectedOperands(
    mlir::Operation *consumer,
    const llvm::DenseSet<mlir::Operation *> &selected) {
  for (mlir::Value operand : consumer->getOperands()) {
    if (!isSelectedDef(operand, selected) ||
        !mlir::isa<mlir::RankedTensorType>(operand.getType()))
      continue;
    for (mlir::Value result : consumer->getResults())
      if (mlir::isa<mlir::RankedTensorType>(result.getType()) &&
          !haveEqualRankedTensorShapes(operand.getType(), result.getType()))
        return false;
  }
  return true;
}

static bool hasFamilyCompatibleSelectedOperands(
    mlir::Operation *consumer,
    const llvm::DenseSet<mlir::Operation *> &selected) {
  for (mlir::Value operand : consumer->getOperands()) {
    mlir::Operation *producer = operand.getDefiningOp();
    if (producer && selected.contains(producer) &&
        !hasCompatibleRootFamily(producer, consumer))
      return false;
  }
  return true;
}

static bool
canAbsorbProducer(mlir::Operation *producer, mlir::Block *block,
                  const llvm::DenseSet<mlir::Operation *> &selected,
                  const llvm::DenseSet<mlir::Operation *> &forbidden,
                  bool allowCrossShapeDataflow) {
  if (!producer || producer->getBlock() != block ||
      selected.contains(producer) || forbidden.contains(producer))
    return false;
  if (!isTensorLevelSchedulableDpsOp(producer) ||
      (!allowCrossShapeDataflow &&
       (!hasShapeCompatibleSelectedConsumers(producer, selected) ||
        !hasFamilyCompatibleSelectedConsumers(producer, selected))))
    return false;
  return allResultsUsedBySelected(producer, selected);
}

static bool
canAbsorbConsumer(mlir::Operation *consumer, mlir::Block *block,
                  const llvm::DenseSet<mlir::Operation *> &selected,
                  const llvm::DenseSet<mlir::Operation *> &forbidden,
                  bool allowCrossShapeDataflow) {
  if (!consumer || consumer->getBlock() != block ||
      selected.contains(consumer) || forbidden.contains(consumer))
    return false;
  if (!isTensorLevelSchedulableDpsOp(consumer) ||
      (!allowCrossShapeDataflow &&
       (!hasShapeCompatibleSelectedOperands(consumer, selected) ||
        !hasFamilyCompatibleSelectedOperands(consumer, selected))))
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

/// A single-use static reshape does not introduce a new tensor value or a
/// scheduling decision, but it can hide the tiled consumer of a logical
/// collective root. Admit only a closed, linear view chain: fanout or a
/// terminal view remains an explicit task boundary rather than silently
/// extending SPM residency.
static bool canAbsorbTransparentViewConsumer(
    mlir::Operation *candidate, mlir::Block *block,
    const llvm::DenseSet<mlir::Operation *> &selected,
    const llvm::DenseSet<mlir::Operation *> &forbidden) {
  if (!candidate || candidate->getBlock() != block ||
      selected.contains(candidate) || forbidden.contains(candidate) ||
      !mlir::isa<mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp>(
          candidate))
    return false;

  auto isStaticIfTensor = [](mlir::Value value) {
    if (!mlir::isa<mlir::TensorType>(value.getType()))
      return true;
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    return tensorType && tensorType.hasStaticShape();
  };
  if (!llvm::all_of(candidate->getOperands(), isStaticIfTensor) ||
      !llvm::all_of(candidate->getResults(), isStaticIfTensor))
    return false;

  bool consumesSelectedValue = false;
  for (mlir::Value operand : candidate->getOperands()) {
    if (!isSelectedDef(operand, selected))
      continue;
    consumesSelectedValue = true;
    if (!allUsesInsideOrAllowedConsumer(operand, selected, candidate))
      return false;
  }
  if (!consumesSelectedValue || candidate->getNumResults() == 0)
    return false;

  for (mlir::Value result : candidate->getResults()) {
    if (!result.hasOneUse())
      return false;
    mlir::Operation *consumer = *result.getUsers().begin();
    if (!consumer || consumer->getBlock() != block ||
        forbidden.contains(consumer) ||
        (!isEligibleStructuredSchedulingRoot(consumer) &&
         !mlir::isa<mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp>(
             consumer)))
      return false;
  }
  return true;
}

static void
orderSelectedOps(mlir::Block *block,
                 const llvm::DenseSet<mlir::Operation *> &selected,
                 llvm::SmallVectorImpl<mlir::Operation *> &orderedOps) {
  orderedOps.clear();
  for (mlir::Operation &op : *block)
    if (selected.contains(&op))
      orderedOps.push_back(&op);
}

static void
expandSelection(mlir::Operation *root,
                llvm::DenseSet<mlir::Operation *> &selected,
                const ScopeDiscoveryPolicy &policy,
                const llvm::DenseSet<mlir::Operation *> &forbidden) {
  mlir::Block *block = root->getBlock();
  selected.insert(root);
  bool rootIsLogicalCollective = isLinalgExtCollectiveRoot(root);
  bool rootIsTiledLogicalCollective =
      rootIsLogicalCollective &&
      classifyCandidateTraversalRoot(root) ==
          CandidateTraversalRootCapability::Tiled;
  bool changed = true;
  while (changed) {
    changed = false;
    if (policy.includeSharedInputPeers) {
      for (mlir::Operation &candidate : *block) {
        if (!canAbsorbSharedInputPeer(&candidate, root, block, selected,
                                      forbidden))
          continue;
        selected.insert(&candidate);
        changed = true;
      }
    }
    llvm::SmallVector<mlir::Operation *> orderedOps;
    orderSelectedOps(block, selected, orderedOps);
    for (mlir::Operation *op : orderedOps)
      for (mlir::Value operand : op->getOperands())
        if (canAbsorbProducer(operand.getDefiningOp(), block, selected,
                              forbidden, policy.allowCrossShapeDataflow)) {
          selected.insert(operand.getDefiningOp());
          changed = true;
        }

    orderSelectedOps(block, selected, orderedOps);
    for (mlir::Operation *op : orderedOps) {
      for (mlir::Value result : op->getResults()) {
        llvm::SmallVector<mlir::OpOperand *> uses;
        for (mlir::OpOperand &use : result.getUses())
          uses.push_back(&use);
        for (mlir::OpOperand *use : uses) {
          mlir::Operation *consumer = use->getOwner();
          // A tiled logical collective may be fused as the terminal consumer
          // of an upstream compute task.  When it is itself the discovery
          // root, keep it as a task boundary: absorbing a downstream consumer
          // would turn the collective into an internal producer, duplicate
          // its untiled source op during producer fusion, and lose the direct
          // output-boundary contract used by complete traversal.
          if (rootIsTiledLogicalCollective)
            continue;
          if (rootIsLogicalCollective &&
              canAbsorbTransparentViewConsumer(consumer, block, selected,
                                               forbidden)) {
            selected.insert(consumer);
            changed = true;
            continue;
          }
          if (!canAbsorbConsumer(consumer, block, selected, forbidden,
                                 policy.allowCrossShapeDataflow))
            continue;
          selected.insert(consumer);
          changed = true;
        }
      }
    }
  }
}

static bool
hasTiledSchedulingRoot(const llvm::DenseSet<mlir::Operation *> &selected) {
  return llvm::any_of(selected, [](mlir::Operation *operation) {
    return classifyCandidateTraversalRoot(operation) ==
           CandidateTraversalRootCapability::Tiled;
  });
}

/// Cuts only a full-traversal-only root whose tensor result leaves the current
/// scope. Repeating the cut handles a terminal chain, but a collective-only
/// scope is preserved so it remains one full-traversal candidate.
static void
cutTerminalFullTraversalOnlyRoots(llvm::DenseSet<mlir::Operation *> &selected,
                                  const ScopeDiscoveryPolicy &policy) {
  if (!policy.cutTerminalFullTraversalOnlyRoots)
    return;

  while (hasTiledSchedulingRoot(selected)) {
    llvm::SmallVector<mlir::Operation *, 2> terminalRoots;
    for (mlir::Operation *operation : selected) {
      if (classifyCandidateTraversalRoot(operation) !=
          CandidateTraversalRootCapability::FullTraversalOnly)
        continue;
      bool hasExternalTensorResult = false;
      bool hasSelectedConsumer = false;
      for (mlir::Value result : operation->getResults()) {
        for (mlir::OpOperand &use : result.getUses()) {
          if (isInsideSelectedOperation(use.getOwner(), selected)) {
            hasSelectedConsumer = true;
            break;
          }
          if (mlir::isa<mlir::TensorType>(result.getType()))
            hasExternalTensorResult = true;
        }
        if (hasSelectedConsumer)
          break;
      }
      if (hasExternalTensorResult && !hasSelectedConsumer)
        terminalRoots.push_back(operation);
    }
    if (terminalRoots.empty())
      return;
    for (mlir::Operation *operation : terminalRoots)
      selected.erase(operation);
  }
}

static mlir::Value
getExternalDpsInitForResult(mlir::Value value,
                            const llvm::DenseSet<mlir::Operation *> &selected) {
  mlir::Value current = value;
  while (auto result = mlir::dyn_cast<mlir::OpResult>(current)) {
    auto dpsOp =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(result.getOwner());
    if (!dpsOp || result.getResultNumber() >=
                      static_cast<unsigned>(dpsOp.getNumDpsInits()))
      return {};
    mlir::Value init = dpsOp.getDpsInitOperand(result.getResultNumber())->get();
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
  for (mlir::Operation *op : orderedOps)
    for (mlir::Value result : op->getResults())
      if (hasExternalUse(result, selected))
        yieldedValues.push_back(result);
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

static void absorbInternalSupportOps(
    mlir::Block *block, llvm::DenseSet<mlir::Operation *> &selected,
    mlir::ValueRange outs, const llvm::DenseSet<mlir::Operation *> &forbidden) {
  llvm::DenseSet<mlir::Value> outSet(outs.begin(), outs.end());
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *> orderedOps;
    orderSelectedOps(block, selected, orderedOps);
    for (mlir::Operation *op : orderedOps)
      op->walk([&](mlir::Operation *nestedOp) {
        for (mlir::Value operand : nestedOp->getOperands()) {
          if (outSet.contains(operand))
            continue;
          mlir::Operation *producer = operand.getDefiningOp();
          if (!producer || producer->getBlock() != block ||
              selected.contains(producer) || forbidden.contains(producer) ||
              !isInternalSupportOp(producer) ||
              !allResultsUsedBySelected(producer, selected))
            continue;
          selected.insert(producer);
          changed = true;
        }
      });
  }
}

static void
collectBoundaryInputs(llvm::ArrayRef<mlir::Operation *> orderedOps,
                      const llvm::DenseSet<mlir::Operation *> &selected,
                      mlir::ValueRange outs,
                      llvm::SmallVectorImpl<mlir::Value> &inputs) {
  llvm::DenseSet<mlir::Value> seen(outs.begin(), outs.end());
  inputs.clear();
  for (mlir::Operation *op : orderedOps) {
    op->walk([&](mlir::Operation *nestedOp) {
      for (mlir::Value operand : nestedOp->getOperands()) {
        if (isSelectedDef(operand, selected))
          continue;
        bool definedInsideSelected = false;
        if (mlir::Operation *def = operand.getDefiningOp()) {
          for (mlir::Operation *scope = def; scope;
               scope = scope->getParentOp())
            if (selected.contains(scope)) {
              definedInsideSelected = true;
              break;
            }
        } else if (auto blockArg =
                       mlir::dyn_cast<mlir::BlockArgument>(operand)) {
          for (mlir::Operation *scope = blockArg.getOwner()->getParentOp();
               scope; scope = scope->getParentOp())
            if (selected.contains(scope)) {
              definedInsideSelected = true;
              break;
            }
        }
        if (!definedInsideSelected && seen.insert(operand).second)
          inputs.push_back(operand);
      }
    });
  }
}

static bool hasDuplicateBoundaryValues(mlir::ValueRange inputs,
                                       mlir::ValueRange outs) {
  llvm::DenseSet<mlir::Value> seen;
  for (mlir::Value value : inputs)
    if (!seen.insert(value).second)
      return true;
  for (mlir::Value value : outs)
    if (!seen.insert(value).second)
      return true;
  return false;
}

static bool hasValidInsertionPoint(const StructuredSchedulingScope &scope) {
  if (scope.orderedOps.empty())
    return false;
  mlir::Operation *insertionPoint = scope.orderedOps.back();
  mlir::Block *block = insertionPoint->getBlock();
  for (mlir::Value yielded : scope.yieldedValues) {
    for (mlir::OpOperand &use : yielded.getUses()) {
      if (isInsideSelectedOperation(use.getOwner(), scope.selected))
        continue;
      mlir::Operation *anchor = use.getOwner();
      while (anchor && anchor->getBlock() != block)
        anchor = anchor->getParentOp();
      if (anchor &&
          (anchor == insertionPoint || anchor->isBeforeInBlock(insertionPoint)))
        return false;
    }
  }
  return true;
}

} // namespace

bool isEligibleStructuredSchedulingRoot(mlir::Operation *op) {
  if (!isEligibleTensorLevelDpsOp(op))
    return false;
  return isLinalgRoot(op) || isLinalgExtCollectiveRoot(op);
}

bool isInsideStructuredSchedulingScope(mlir::Operation *operation,
                                       const StructuredSchedulingScope &scope) {
  return isInsideSelectedOperation(operation, scope.selected);
}

bool buildStructuredSchedulingScope(
    mlir::Operation *root, StructuredSchedulingScope &scope,
    ScopeSelectionFailure &failure, const ScopeDiscoveryPolicy &policy,
    const llvm::DenseSet<mlir::Operation *> &forbiddenOperations) {
  scope = StructuredSchedulingScope{};
  failure = ScopeSelectionFailure::None;
  if (forbiddenOperations.contains(root))
    return false;
  expandSelection(root, scope.selected, policy, forbiddenOperations);
  cutTerminalFullTraversalOnlyRoots(scope.selected, policy);
  orderSelectedOps(root->getBlock(), scope.selected, scope.orderedOps);
  if (!collectYieldedValuesAndOuts(scope.orderedOps, scope.selected,
                                   scope.yieldedValues, scope.outs)) {
    failure = scope.yieldedValues.empty()
                  ? ScopeSelectionFailure::MissingExternalResult
                  : ScopeSelectionFailure::MissingTensorDestination;
    return false;
  }
  absorbInternalSupportOps(root->getBlock(), scope.selected, scope.outs,
                           forbiddenOperations);
  orderSelectedOps(root->getBlock(), scope.selected, scope.orderedOps);
  collectBoundaryInputs(scope.orderedOps, scope.selected, scope.outs,
                        scope.inputs);
  if (hasDuplicateBoundaryValues(scope.inputs, scope.outs)) {
    failure = ScopeSelectionFailure::DuplicateBoundary;
    return false;
  }
  if (!hasValidInsertionPoint(scope)) {
    failure = ScopeSelectionFailure::InterleavedExternalUse;
    return false;
  }
  scope.insertionPoint = scope.orderedOps.back();
  return true;
}

mlir::LogicalResult
refreshStructuredSchedulingScopeBoundary(StructuredSchedulingScope &scope) {
  if (!scope.insertionPoint || scope.orderedOps.empty())
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 4> yieldedValues;
  llvm::SmallVector<mlir::Value, 4> outs;
  llvm::SmallVector<mlir::Value, 8> inputs;
  if (!collectYieldedValuesAndOuts(scope.orderedOps, scope.selected,
                                   yieldedValues, outs))
    return mlir::failure();
  collectBoundaryInputs(scope.orderedOps, scope.selected, outs, inputs);
  if (hasDuplicateBoundaryValues(inputs, outs) ||
      inputs.size() != scope.inputs.size() ||
      outs.size() != scope.outs.size() ||
      yieldedValues.size() != scope.yieldedValues.size())
    return mlir::failure();

  // Do not inspect the cached Values here: a preceding task commit may have
  // erased their defining source op after rewiring every live use.  The
  // standalone candidate function is the immutable ABI/type contract, and
  // commitSelectedTaskCandidate checks the refreshed Values against it.
  scope.inputs = std::move(inputs);
  scope.outs = std::move(outs);
  scope.yieldedValues = std::move(yieldedValues);
  return hasValidInsertionPoint(scope) ? mlir::success() : mlir::failure();
}

llvm::StringRef getScopeSelectionFailureMessage(ScopeSelectionFailure failure) {
  switch (failure) {
  case ScopeSelectionFailure::MissingExternalResult:
    return "selected root has no result used outside the scheduling task";
  case ScopeSelectionFailure::MissingTensorDestination:
    return "an externally used result has no external tensor destination";
  case ScopeSelectionFailure::DuplicateBoundary:
    return "computed scheduling-task boundary contains duplicate SSA values";
  case ScopeSelectionFailure::InterleavedExternalUse:
    return "selected operations cross an earlier external result use";
  case ScopeSelectionFailure::None:
    return "structured scheduling scope selection failed";
  }
  llvm_unreachable("unknown structured scheduling scope failure");
}

mlir::LogicalResult discoverStructuredSchedulingScopes(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<StructuredSchedulingScope> &scopes,
    const ScopeDiscoveryPolicy &policy) {
  scopes.clear();
  llvm::SmallVector<mlir::Operation *> roots;
  module.walk([&](mlir::Operation *op) {
    if (isEligibleStructuredSchedulingRoot(op))
      roots.push_back(op);
  });
  llvm::DenseSet<mlir::Operation *> consumed;
  for (mlir::Operation *root : roots) {
    if (consumed.contains(root) || !root->getBlock())
      continue;
    StructuredSchedulingScope scope;
    ScopeSelectionFailure failure;
    if (!buildStructuredSchedulingScope(root, scope, failure, policy,
                                        consumed)) {
      StructuredSchedulingScope fallback;
      ScopeSelectionFailure fallbackFailure;
      ScopeDiscoveryPolicy fallbackPolicy = policy;
      fallbackPolicy.includeSharedInputPeers = false;
      if (!buildStructuredSchedulingScope(root, fallback, fallbackFailure,
                                          fallbackPolicy, consumed)) {
        root->emitOpError("cannot form required scheduling task: ")
            << getScopeSelectionFailureMessage(fallbackFailure);
        return mlir::failure();
      }
      scope = std::move(fallback);
    }
    for (mlir::Operation *op : scope.selected) {
      if (!consumed.insert(op).second) {
        root->emitOpError(
            "structured scheduling tasks must not overlap source operations");
        return mlir::failure();
      }
    }
    scopes.push_back(std::move(scope));
  }
  for (mlir::Operation *root : roots) {
    if (!consumed.contains(root)) {
      root->emitOpError("eligible tensor-level root remains outside a "
                        "structured scheduling task");
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::OwningOpRef<mlir::ModuleOp>
cloneScopeToStandaloneModule(const StructuredSchedulingScope &scope,
                             llvm::StringRef functionName) {
  if (!scope.insertionPoint || scope.orderedOps.empty() ||
      scope.outs.size() != scope.yieldedValues.size())
    return nullptr;
  mlir::Location loc = scope.insertionPoint->getLoc();
  auto module = mlir::ModuleOp::create(loc);
  mlir::ModuleOp sourceModule =
      scope.insertionPoint->getParentOfType<mlir::ModuleOp>();
  if (!sourceModule)
    return nullptr;
  cloneTargetExecutionFacts(sourceModule, module);
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  llvm::SmallVector<mlir::Type> argumentTypes;
  for (mlir::Value input : scope.inputs)
    argumentTypes.push_back(input.getType());
  for (mlir::Value out : scope.outs)
    argumentTypes.push_back(out.getType());
  llvm::SmallVector<mlir::Type> resultTypes;
  for (mlir::Value yielded : scope.yieldedValues)
    resultTypes.push_back(yielded.getType());
  auto func = moduleBuilder.create<mlir::func::FuncOp>(
      loc, functionName,
      moduleBuilder.getFunctionType(argumentTypes, resultTypes));
  mlir::Block *entry = func.addEntryBlock();
  mlir::OpBuilder builder(entry, entry->end());
  mlir::IRMapping mapping;
  unsigned argumentIndex = 0;
  for (mlir::Value input : scope.inputs) {
    mlir::Value replacement = entry->getArgument(argumentIndex++);
    // Keep the ABI slot stable for atomic commit, while preserving exact
    // constant semantics inside the private candidate.  Treating a constant
    // as an unconstrained function argument would lose the proof used by
    // numeric legality (for example pow(x, 2) -> x*x).
    if (auto constant = input.getDefiningOp<mlir::arith::ConstantOp>())
      replacement = builder.clone(*constant)->getResult(0);
    mapping.map(input, replacement);
  }
  for (mlir::Value out : scope.outs)
    mapping.map(out, entry->getArgument(argumentIndex++));
  for (mlir::Operation *op : scope.orderedOps)
    builder.clone(*op, mapping);
  llvm::SmallVector<mlir::Value> returned;
  for (mlir::Value yielded : scope.yieldedValues)
    returned.push_back(mapping.lookup(yielded));
  builder.create<mlir::func::ReturnOp>(loc, returned);
  return module;
}

mlir::func::FuncOp findSingleTaskFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp found;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (found)
      return {};
    found = func;
  }
  return found;
}

} // namespace wafer::structured_scheduler
