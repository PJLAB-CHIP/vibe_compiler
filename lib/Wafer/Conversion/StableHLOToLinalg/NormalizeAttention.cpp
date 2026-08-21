//===- NormalizeAttention.cpp - Form structured attention semantics -----===//

#include "AttentionMatching.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer {

#define GEN_PASS_DEF_FORMATTENTIONOPSPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

mlir::Value
materializeScale(mlir::IRRewriter &rewriter,
                 const attention_normalization::AttentionMatch &match) {
  if (!match.scale) {
    auto elementType =
        mlir::cast<mlir::ShapedType>(match.query.getType()).getElementType();
    return rewriter.create<mlir::arith::ConstantOp>(
        match.root->getLoc(), rewriter.getFloatAttr(elementType, 1.0));
  }
  if (mlir::isa<mlir::FloatType>(match.scale.getType()))
    return match.scale;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(match.scale.getType());
  assert(type && "attention proof only returns scalar or ranked scale");
  llvm::SmallVector<mlir::Value, 4> indices;
  for (int64_t dimension = 0; dimension < type.getRank(); ++dimension)
    indices.push_back(
        rewriter.create<mlir::arith::ConstantIndexOp>(match.root->getLoc(), 0));
  return rewriter.create<mlir::tensor::ExtractOp>(match.root->getLoc(),
                                                  match.scale, indices);
}

void eraseDeadProducerClosure(mlir::IRRewriter &rewriter,
                              llvm::ArrayRef<mlir::Operation *> seeds) {
  llvm::SmallVector<mlir::Operation *, 32> worklist(seeds.begin(), seeds.end());
  llvm::DenseSet<mlir::Operation *> queued(seeds.begin(), seeds.end());
  while (!worklist.empty()) {
    mlir::Operation *operation = worklist.pop_back_val();
    if (!mlir::isOpTriviallyDead(operation))
      continue;
    llvm::SmallVector<mlir::Operation *, 8> producers;
    for (mlir::Value operand : operation->getOperands()) {
      if (mlir::Operation *producer = operand.getDefiningOp())
        producers.push_back(producer);
    }
    rewriter.eraseOp(operation);
    for (mlir::Operation *producer : producers) {
      if (queued.insert(producer).second)
        worklist.push_back(producer);
    }
  }
}

bool dependsOn(mlir::Value value, mlir::Operation *ancestor) {
  llvm::SmallVector<mlir::Value, 16> worklist{value};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    mlir::Operation *definition = current.getDefiningOp();
    if (!definition)
      continue;
    if (definition == ancestor)
      return true;
    llvm::append_range(worklist, definition->getOperands());
  }
  return false;
}

mlir::LogicalResult validateMatchOwnership(
    llvm::ArrayRef<attention_normalization::AttentionMatch> matches) {
  llvm::DenseSet<mlir::Operation *> roots;
  for (const attention_normalization::AttentionMatch &match : matches) {
    if (!match.root || !roots.insert(match.root).second)
      return mlir::failure();
  }
  for (const attention_normalization::AttentionMatch &producer : matches) {
    for (const attention_normalization::AttentionMatch &consumer : matches) {
      if (&producer == &consumer)
        continue;
      for (mlir::Value value : llvm::SmallVector<mlir::Value, 6>{
               consumer.query, consumer.key, consumer.value, consumer.scale,
               consumer.mask}) {
        if (value && dependsOn(value, producer.root))
          return consumer.root->emitError(
              "overlapping attention roots require one unambiguous owner");
      }
    }
  }
  return mlir::success();
}

mlir::Value reshapeAttentionResult(mlir::IRRewriter &rewriter,
                                   mlir::Location location, mlir::Value result,
                                   mlir::RankedTensorType targetType) {
  auto sourceType = mlir::cast<mlir::RankedTensorType>(result.getType());
  if (sourceType == targetType)
    return result;
  std::optional<llvm::SmallVector<mlir::ReassociationIndices>> reassociation =
      mlir::getReassociationIndicesForReshape(sourceType, targetType);
  assert(reassociation && "attention proof prevalidates result reshape");
  if (sourceType.getRank() > targetType.getRank())
    return rewriter.create<mlir::tensor::CollapseShapeOp>(
        location, targetType, result, *reassociation);
  return rewriter.create<mlir::tensor::ExpandShapeOp>(location, targetType,
                                                      result, *reassociation);
}

struct FormAttentionOpsPass final
    : impl::FormAttentionOpsPassBase<FormAttentionOpsPass> {
  using impl::FormAttentionOpsPassBase<
      FormAttentionOpsPass>::FormAttentionOpsPassBase;

  void runOnOperation() final {
    mlir::func::FuncOp function = getOperation();
    llvm::SmallVector<attention_normalization::AttentionMatch, 4> matches =
        attention_normalization::collectAttentionMatches(function);
    if (matches.empty())
      return;
    if (mlir::failed(validateMatchOwnership(matches)))
      return signalPassFailure();

    mlir::IRRewriter rewriter(&getContext());
    llvm::SmallVector<mlir::Operation *, 32> deadProducerSeeds;
    for (const attention_normalization::AttentionMatch &match : matches) {
      if (!match.root || match.root->getNumResults() != 1)
        return signalPassFailure();
      rewriter.setInsertionPoint(match.root);
      mlir::Value scale = materializeScale(rewriter, match);
      mlir::Value output = rewriter.create<mlir::tensor::EmptyOp>(
          match.root->getLoc(), match.outputType.getShape(),
          match.outputType.getElementType());
      auto attention = rewriter.create<LinalgExtAttentionOp>(
          match.root->getLoc(), mlir::TypeRange{match.outputType}, match.query,
          match.key, match.value, scale, match.mask, output, match.algorithm,
          rewriter.getAffineMapArrayAttr(match.indexingMaps));
      if (mlir::failed(mlir::verify(attention.getOperation())))
        return signalPassFailure();

      for (mlir::Value operand : match.root->getOperands()) {
        if (mlir::Operation *producer = operand.getDefiningOp())
          deadProducerSeeds.push_back(producer);
      }
      mlir::Value replacement = reshapeAttentionResult(
          rewriter, match.root->getLoc(), attention.getResult(0),
          mlir::cast<mlir::RankedTensorType>(
              match.root->getResult(0).getType()));
      rewriter.replaceOp(match.root, replacement);
    }

    eraseDeadProducerClosure(rewriter, deadProducerSeeds);
    if (mlir::failed(mlir::verify(function)))
      signalPassFailure();
  }
};

} // namespace

} // namespace wafer
