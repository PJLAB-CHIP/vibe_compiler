//===- CardBaselinePlacementClosure.cpp - Exact coordinate closure --------===//

#include "Wafer/Planning/Baseline/CardBaselinePlacement.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {

void setFailure(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
}

bool samePlacement(const StructuredDAGNodePlacement &lhs,
                   const StructuredDAGNodePlacement &rhs) {
  return lhs.iteratorPartitionFactors == rhs.iteratorPartitionFactors &&
         lhs.tiles == rhs.tiles;
}

mlir::FailureOr<unsigned>
mapAxisForwardThroughSupportOp(mlir::Operation *operation, unsigned operandAxis,
                               uint64_t shardCount) {
  if (mlir::isa<mlir::tensor::ExtractSliceOp, mlir::tensor::InsertSliceOp>(
          operation))
    return operandAxis;
  if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(operation)) {
    llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation =
        expand.getReassociationIndices();
    auto operandType =
        mlir::cast<mlir::RankedTensorType>(operation->getOperand(0).getType());
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation->getResult(0).getType());
    if (operandAxis >= reassociation.size() ||
        operandAxis >= static_cast<unsigned>(operandType.getRank()))
      return mlir::failure();
    llvm::ArrayRef<int64_t> group = reassociation[operandAxis];
    const int64_t extent = operandType.getShape()[operandAxis];
    if (extent <= 0 || shardCount == 0 ||
        extent % static_cast<int64_t>(shardCount) != 0)
      return mlir::failure();
    const int64_t shardSize = extent / static_cast<int64_t>(shardCount);
    llvm::SmallVector<int64_t, 4> memberExtents;
    for (int64_t member : group) {
      if (member < 0 || member >= resultType.getRank())
        return mlir::failure();
      const int64_t memberExtent = resultType.getShape()[member];
      if (memberExtent <= 0)
        return mlir::failure();
      memberExtents.push_back(memberExtent);
    }
    uint64_t leadingExtentProduct = 1;
    uint64_t trailingExtentProduct = 1;
    for (int64_t memberExtent : llvm::drop_begin(memberExtents))
      trailingExtentProduct *= memberExtent;
    for (auto [index, pair] :
         llvm::enumerate(llvm::zip(group, memberExtents))) {
      auto [member, memberExtent] = pair;
      if (leadingExtentProduct == 1 &&
          shardSize % static_cast<int64_t>(trailingExtentProduct) == 0)
        return static_cast<unsigned>(member);
      leadingExtentProduct *= memberExtent;
      if (index + 1 < memberExtents.size())
        trailingExtentProduct /= memberExtents[index + 1];
    }
    return mlir::failure();
  }
  if (auto collapse =
          mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(operation)) {
    auto operandType =
        mlir::cast<mlir::RankedTensorType>(operation->getOperand(0).getType());
    if (operandAxis >= static_cast<unsigned>(operandType.getRank()))
      return mlir::failure();
    const int64_t extent = operandType.getShape()[operandAxis];
    if (extent <= 0 || shardCount == 0 ||
        extent % static_cast<int64_t>(shardCount) != 0)
      return mlir::failure();
    llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation =
        collapse.getReassociationIndices();
    for (auto [resultAxis, group] : llvm::enumerate(reassociation)) {
      if (!llvm::is_contained(group, static_cast<int64_t>(operandAxis)))
        continue;
      for (int64_t member : group) {
        if (member == static_cast<int64_t>(operandAxis))
          break;
        if (operandType.getShape()[member] != 1)
          return mlir::failure();
      }
      return static_cast<unsigned>(resultAxis);
    }
    return mlir::failure();
  }
  auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getOperand(0).getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(operation->getResult(0).getType());
  if (operandType && resultType && operandType.hasStaticShape() &&
      resultType.hasStaticShape() &&
      operandType.getRank() == resultType.getRank() &&
      operandType.getShape() == resultType.getShape())
    return operandAxis;
  return mlir::failure();
}

mlir::LogicalResult
collectOutputSupportChain(mlir::Value value, mlir::Operation *rootOperation,
                          llvm::SmallVectorImpl<mlir::Operation *> &chain) {
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return mlir::failure();
  mlir::Operation *owner = result.getOwner();
  if (owner == rootOperation)
    return result.getResultNumber() == 0 ? mlir::success() : mlir::failure();
  if (!owner || owner->getNumResults() != 1 || !mlir::isMemoryEffectFree(owner))
    return mlir::failure();
  for (mlir::Value operand : owner->getOperands()) {
    if (mlir::succeeded(
            collectOutputSupportChain(operand, rootOperation, chain))) {
      chain.push_back(owner);
      return mlir::success();
    }
  }
  return mlir::failure();
}

mlir::FailureOr<unsigned>
mapRootShardAxisToOutput(const StructuredDAGAnalysis &dag, unsigned outputIndex,
                         const StructuredDAGNode &root, unsigned rootAxis,
                         uint64_t shardCount) {
  auto returnOperation = mlir::dyn_cast<mlir::func::ReturnOp>(
      dag.getFunction().getBody().front().getTerminator());
  if (!returnOperation || outputIndex >= returnOperation.getNumOperands())
    return mlir::failure();
  llvm::SmallVector<mlir::Operation *, 4> chain;
  if (mlir::failed(collectOutputSupportChain(
          returnOperation.getOperand(outputIndex), root.operation, chain)))
    return mlir::failure();
  unsigned axis = rootAxis;
  for (mlir::Operation *operation : chain) {
    mlir::FailureOr<unsigned> mapped =
        mapAxisForwardThroughSupportOp(operation, axis, shardCount);
    if (mlir::failed(mapped))
      return mlir::failure();
    axis = *mapped;
  }
  return axis;
}

mlir::FailureOr<unsigned>
getRootResultDimensionForIterator(const StructuredDAGNode &root,
                                  unsigned iteratorDimension) {
  if (!root.operation || root.operation->getNumResults() == 0)
    return mlir::failure();
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      root.operation->getResult(0).getType());
  if (!resultType)
    return mlir::failure();
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(root.operation)) {
    mlir::AffineMap resultMap =
        linalg.getIndexingMapMatchingResult(root.operation->getResult(0));
    std::optional<unsigned> resultDimension;
    for (auto [dimension, expression] :
         llvm::enumerate(resultMap.getResults())) {
      auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!iterator || iterator.getPosition() != iteratorDimension)
        continue;
      if (resultDimension)
        return mlir::failure();
      resultDimension = dimension;
    }
    if (resultDimension)
      return *resultDimension;
    return mlir::failure();
  }
  if (iteratorDimension < static_cast<unsigned>(resultType.getRank()))
    return iteratorDimension;
  return mlir::failure();
}

mlir::FailureOr<llvm::SmallVector<CardBaselineObservablePlacement, 4>>
deriveObservablePlacements(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> placements) {
  llvm::SmallVector<CardBaselineObservablePlacement, 4> outputs;
  outputs.reserve(dag.getFunction().getNumResults());
  auto returnOperation = mlir::dyn_cast<mlir::func::ReturnOp>(
      dag.getFunction().getBody().front().getTerminator());
  if (!returnOperation ||
      returnOperation.getNumOperands() != dag.getFunction().getNumResults())
    return mlir::failure();
  for (auto [outputIndex, roots] :
       llvm::enumerate(dag.getObservableOutputRootNodes())) {
    if (roots.empty() || roots.front() >= placements.size())
      return mlir::failure();
    const StructuredDAGNodePlacement &owner = placements[roots.front()];
    if (llvm::any_of(roots, [&](StructuredDAGNodeID root) {
          return root >= placements.size() ||
                 !samePlacement(owner, placements[root]);
        }))
      return mlir::failure();
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(
        returnOperation.getOperand(outputIndex).getType());
    if (!outputType || !outputType.hasStaticShape())
      return mlir::failure();
    std::optional<unsigned> partitionedIterator;
    for (auto [iterator, factor] :
         llvm::enumerate(owner.iteratorPartitionFactors)) {
      if (factor == 1)
        continue;
      if (partitionedIterator)
        return mlir::failure();
      partitionedIterator = iterator;
    }
    if (!partitionedIterator) {
      if (owner.tiles.size() != 1 ||
          llvm::any_of(owner.iteratorPartitionFactors,
                       [](uint32_t factor) { return factor != 1; }))
        return mlir::failure();
      outputs.push_back(CardBaselineObservablePlacement{
          static_cast<uint32_t>(outputIndex), std::nullopt, owner.tiles});
      continue;
    }
    std::optional<unsigned> outputAxis;
    for (StructuredDAGNodeID rootID : roots) {
      const StructuredDAGNode *root = dag.getNode(rootID);
      if (!root || !root->operation)
        return mlir::failure();
      mlir::FailureOr<unsigned> rootResultDimension =
          getRootResultDimensionForIterator(*root, *partitionedIterator);
      if (mlir::failed(rootResultDimension))
        return mlir::failure();
      mlir::FailureOr<unsigned> mapped = mapRootShardAxisToOutput(
          dag, static_cast<unsigned>(outputIndex), *root, *rootResultDimension,
          owner.tiles.size());
      if (mlir::failed(mapped) || (outputAxis && *outputAxis != *mapped))
        return mlir::failure();
      outputAxis = *mapped;
    }
    if (!outputAxis ||
        *outputAxis >= static_cast<unsigned>(outputType.getRank()) ||
        static_cast<uint64_t>(outputType.getShape()[*outputAxis]) <
            owner.tiles.size())
      return mlir::failure();
    outputs.push_back(CardBaselineObservablePlacement{
        static_cast<uint32_t>(outputIndex), *outputAxis, owner.tiles});
  }
  return outputs;
}

} // namespace

mlir::FailureOr<CardBaselinePlacementClosure> closeCardBaselinePlacement(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
    analysis::IREpoch epoch, std::string *failureReason,
    CardBaselinePlacementVerdict *verdict) {
  if (failureReason)
    failureReason->clear();
  auto failIndeterminate = [&](llvm::StringRef detail) {
    setFailure(failureReason, detail);
    if (verdict) {
      verdict->status = analysis::ExactDemandStatus::IndeterminateFailure;
      verdict->detail = detail.str();
    }
  };
  if (nodePlacements.size() != dag.getNodes().size()) {
    failIndeterminate("baseline placement closure requires every DAG node");
    return mlir::failure();
  }
  for (auto [node, placement] :
       llvm::zip_equal(dag.getNodes(), nodePlacements)) {
    if (placement.node != node.id || placement.tiles.empty()) {
      failIndeterminate("baseline placement closure has an invalid node");
      return mlir::failure();
    }
  }

  CardBaselinePlacementClosure closure;
  closure.nodePlacements.assign(nodePlacements.begin(), nodePlacements.end());
  auto outputs = deriveObservablePlacements(dag, closure.nodePlacements);
  if (mlir::failed(outputs)) {
    failIndeterminate(
        "baseline placement has inconsistent observable output ownership");
    return mlir::failure();
  }

  StructuredDAGExactDemandQuery demandQuery(dag, epoch);
  for (const StructuredDAGEdge &edge : dag.getEdges()) {
    const StructuredDAGNodePlacement &producer =
        closure.nodePlacements[edge.producer];
    const StructuredDAGNodePlacement &consumer =
        closure.nodePlacements[edge.consumer];
    std::string trialFailure;
    auto trial =
        buildEdgeShardTrial(dag, producer, consumer, epoch, &trialFailure);
    if (mlir::failed(trial)) {
      failIndeterminate(trialFailure);
      return mlir::failure();
    }
    analysis::ExactDemandResult demand = demandQuery.query(edge.id, *trial);
    if (demand.status == analysis::ExactDemandStatus::Satisfied)
      continue;
    setFailure(failureReason, demand.detail);
    if (verdict) {
      verdict->status = demand.status;
      verdict->detail = std::move(demand.detail);
    }
    return mlir::failure();
  }

  closure.outputPlacements = std::move(*outputs);
  if (verdict) {
    verdict->status = analysis::ExactDemandStatus::Satisfied;
    verdict->detail.clear();
  }
  return closure;
}

} // namespace wafer::compiler::detail
