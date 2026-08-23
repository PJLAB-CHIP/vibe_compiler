//===- TemporalPartialReductionTraversal.cpp - Partial wave traversal ===//

#include "TemporalPartialReductionTraversal.h"

#include "Internal.h"
#include "ProducerTileFusion.h"
#include "TemporalWaveLoop.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::tensor_program_to_tile_region {
namespace {

mlir::LogicalResult fail(std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

} // namespace

mlir::LogicalResult materializeTemporalPartialReductionShard(
    mlir::func::FuncOp function, uint32_t structuredNodeId,
    llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
    const StructuredNodeTemporalTile &selectedTemporal,
    unsigned &functionalArgumentCount,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason) {
  if (selectedTemporal.structuredNodeId != structuredNodeId ||
      offsets.size() != sizes.size() ||
      sizes.size() != selectedTemporal.iteratorTileSizes.size() ||
      llvm::any_of(sizes, [](int64_t size) { return size <= 0; }))
    return fail(failureReason, "partial-reduction temporal shard is malformed");
  auto rootMapping = llvm::find_if(
      operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
        return mapping.structuredNodeId == structuredNodeId;
      });
  if (rootMapping == operationNodes.end() || !rootMapping->operation)
    return fail(failureReason,
                "partial-reduction temporal shard has no mapped root");
  mlir::Operation *root = rootMapping->operation;
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(root);
  auto partial = mlir::dyn_cast<mlir::PartialReductionOpInterface>(root);
  auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(root);
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(root);
  if (!tiling || !partial || !dps || !linalg ||
      dps.getNumDpsInits() != root->getNumResults())
    return fail(failureReason,
                "partial temporal traversal requires tiled DPS reduction");

  llvm::SmallVector<int, 2> reductionDimensions;
  for (auto [dimension, iteratorType] :
       llvm::enumerate(tiling.getLoopIteratorTypes()))
    if (iteratorType == mlir::utils::IteratorType::reduction)
      reductionDimensions.push_back(static_cast<int>(dimension));
  if (reductionDimensions.empty())
    return fail(failureReason,
                "partial temporal traversal has no reduction iterator");

  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> spatialSizes;
  for (auto [offset, size] : llvm::zip_equal(offsets, sizes)) {
    (void)offset;
    spatialSizes.push_back(builder.getIndexAttr(size));
  }
  llvm::SmallVector<mlir::OpFoldResult, 4> spatialOffsets;
  for (int64_t offset : offsets)
    spatialOffsets.push_back(builder.getIndexAttr(offset));
  llvm::SmallVector<mlir::Value, 2> neutral;
  llvm::SmallVector<llvm::SmallVector<mlir::AffineExpr, 4>, 2>
      partialResultMaps;
  partialResultMaps.reserve(root->getNumResults());
  for (unsigned result = 0; result < root->getNumResults(); ++result) {
    mlir::OpOperand *init = dps.getDpsInitOperand(result);
    llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
    llvm::SmallVector<mlir::OpFoldResult> resultSizes;
    if (!init || mlir::failed(tiling.getResultTilePosition(
                     builder, result, spatialOffsets, spatialSizes,
                     resultOffsets, resultSizes)))
      return fail(failureReason,
                  "partial temporal traversal has no result tile relation");
    mlir::AffineMap outputMap = linalg.getMatchingIndexingMap(init);
    llvm::SmallVector<mlir::AffineExpr, 4> partialMap(
        outputMap.getResults().begin(), outputMap.getResults().end());
    for (int reductionDimension : reductionDimensions)
      partialMap.push_back(builder.getAffineDimExpr(reductionDimension));
    llvm::SmallVector<int64_t, 4> outputTileShape;
    for (mlir::OpFoldResult size : resultSizes) {
      std::optional<int64_t> constant = mlir::getConstantIntValue(size);
      if (!constant || *constant <= 0)
        return fail(failureReason,
                    "partial temporal result tile is not static");
      outputTileShape.push_back(*constant);
    }
    llvm::SmallVector<int64_t, 4> partialShape;
    llvm::DenseSet<int> reductionSet(reductionDimensions.begin(),
                                     reductionDimensions.end());
    unsigned outputDimension = 0;
    for (unsigned dimension = 0;
         dimension < outputTileShape.size() + reductionDimensions.size();
         ++dimension) {
      if (reductionSet.contains(static_cast<int>(dimension))) {
        partialShape.push_back(sizes[dimension]);
      } else {
        if (outputDimension >= outputTileShape.size())
          return fail(failureReason,
                      "partial temporal output shape is inconsistent");
        partialShape.push_back(outputTileShape[outputDimension++]);
      }
    }
    llvm::SmallVector<mlir::Operation *, 4> combinerOps;
    if (!mlir::matchReduction(linalg.getRegionOutputArgs(), result,
                              combinerOps) ||
        combinerOps.size() != 1)
      return fail(failureReason,
                  "partial temporal traversal has no exact combiner");
    std::optional<mlir::TypedAttr> identity =
        mlir::arith::getNeutralElement(combinerOps.front());
    if (!identity)
      return fail(failureReason,
                  "partial temporal traversal has no neutral element");
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        root->getLoc(), partialShape,
        linalg.getRegionOutputArgs()[result].getType());
    auto constant =
        builder.create<mlir::arith::ConstantOp>(root->getLoc(), *identity);
    neutral.push_back(builder
                          .create<mlir::linalg::FillOp>(root->getLoc(),
                                                        constant.getResult(),
                                                        empty.getResult())
                          .getResult(0));
    if (partialMap.size() != partialShape.size())
      return fail(failureReason,
                  "partial temporal output map has inconsistent rank");
    partialResultMaps.push_back(std::move(partialMap));
  }

  TensorProgramScope scope(function, functionalArgumentCount);
  StructuredOpTemporalTile temporal{root, selectedTemporal.iteratorTileSizes,
                                    selectedTemporal.waveLoopOrder};
  llvm::SmallVector<StructuredOpTemporalTile, 1> temporalTiles{temporal};
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> traversed =
      materializeTemporalWaveLoopNest(
          builder, root->getLoc(), offsets, sizes,
          selectedTemporal.iteratorTileSizes, selectedTemporal.waveLoopOrder,
          neutral,
          [&](mlir::OpBuilder &leafBuilder,
              llvm::ArrayRef<mlir::OpFoldResult> leafOffsets,
              llvm::ArrayRef<int64_t> leafSizes, mlir::ValueRange outputs,
              llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops)
              -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> {
            llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
            for (int64_t size : leafSizes)
              mixedSizes.push_back(leafBuilder.getIndexAttr(size));
            llvm::SmallVector<llvm::SmallVector<mlir::OpFoldResult>, 2>
                relativeOffsets(root->getNumResults());
            llvm::SmallVector<llvm::SmallVector<mlir::OpFoldResult>, 2>
                partialSizes(root->getNumResults());
            for (unsigned result = 0; result < root->getNumResults();
                 ++result) {
              auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(
                  outputs[result].getType());
              if (!outputType || partialResultMaps[result].size() !=
                                     static_cast<size_t>(outputType.getRank()))
                return mlir::failure();
              for (auto [outputDimension, expression] :
                   llvm::enumerate(partialResultMaps[result])) {
                if (auto dimension =
                        mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
                  unsigned position = dimension.getPosition();
                  if (position >= leafOffsets.size())
                    return mlir::failure();
                  std::optional<int64_t> constant =
                      mlir::getConstantIntValue(leafOffsets[position]);
                  if (constant) {
                    relativeOffsets[result].push_back(leafBuilder.getIndexAttr(
                        *constant - offsets[position]));
                  } else {
                    mlir::Value value = mlir::getValueOrCreateConstantIndexOp(
                        leafBuilder, root->getLoc(), leafOffsets[position]);
                    mlir::Value base =
                        leafBuilder.create<mlir::arith::ConstantIndexOp>(
                            root->getLoc(), offsets[position]);
                    relativeOffsets[result].push_back(
                        leafBuilder
                            .create<mlir::arith::SubIOp>(root->getLoc(), value,
                                                         base)
                            .getResult());
                  }
                  partialSizes[result].push_back(
                      leafBuilder.getIndexAttr(leafSizes[position]));
                  continue;
                }
                auto constant =
                    mlir::dyn_cast<mlir::AffineConstantExpr>(expression);
                if (!constant || constant.getValue() != 0 ||
                    outputType.getDimSize(outputDimension) != 1)
                  return mlir::failure();
                relativeOffsets[result].push_back(leafBuilder.getIndexAttr(0));
                partialSizes[result].push_back(leafBuilder.getIndexAttr(1));
              }
            }

            mlir::FailureOr<PartialReductionTileMaterialization> tile =
                materializePartialReductionTile(root, leafBuilder, leafOffsets,
                                                mixedSizes, failureReason);
            if (mlir::failed(tile) ||
                tile->partialValues.size() != root->getNumResults())
              return mlir::failure();
            for (mlir::Operation *operation : tile->partialOperations) {
              recordStructuredOperationNodeMaterialization(root, operation,
                                                           &operationNodes);
              if (mlir::failed(fuseCandidateProducerSlices(
                      operation, root, scope, loops, temporalTiles,
                      /*nestedTemporalTiles=*/{}, leafBuilder.getListener(),
                      failureReason, &operationNodes)))
                return mlir::failure();
            }
            for (mlir::Operation *operation :
                 llvm::reverse(tile->mergeOperations))
              if (operation->use_empty())
                operation->erase();
            leafBuilder.setInsertionPointAfter(tile->partialOperations.back());
            llvm::SmallVector<mlir::Value, 2> updated;
            for (unsigned result = 0; result < root->getNumResults();
                 ++result) {
              auto outputType =
                  mlir::cast<mlir::RankedTensorType>(outputs[result].getType());
              llvm::SmallVector<mlir::OpFoldResult, 4> strides(
                  outputType.getRank(), leafBuilder.getIndexAttr(1));
              updated.push_back(
                  leafBuilder
                      .create<mlir::tensor::InsertSliceOp>(
                          root->getLoc(), tile->partialValues[result],
                          outputs[result], relativeOffsets[result],
                          partialSizes[result], strides)
                      .getResult());
            }
            return updated;
          },
          failureReason);
  if (mlir::failed(traversed) || traversed->size() != root->getNumResults())
    return mlir::failure();

  mlir::func::ReturnOp oldReturn = scope.getReturn();
  builder.setInsertionPoint(oldReturn);
  builder.create<mlir::func::ReturnOp>(oldReturn.getLoc(), *traversed);
  oldReturn.erase();
  llvm::SmallVector<mlir::Type, 2> resultTypes;
  for (mlir::Value value : *traversed)
    resultTypes.push_back(value.getType());
  function.setFunctionType(mlir::FunctionType::get(
      function.getContext(), function.getArgumentTypes(), resultTypes));

  eraseDeadCandidateSupportClosure(
      TensorProgramScope(function, functionalArgumentCount));
  retainLiveOperationNodes(function, operationNodes);
  if (mlir::failed(appendTileOutputDestinations(function, failureReason)))
    return mlir::failure();
  return bindFullResultsToOutputDestinations(function, functionalArgumentCount,
                                             failureReason);
}

} // namespace wafer::tensor_program_to_tile_region
