//===- TemporalRegionTraversal.cpp - Iterator wave loops --------------===//

#include "TemporalRegionTraversal.h"

#include "Internal.h"
#include "ProducerTileFusion.h"
#include "StructuredIterationTile.h"
#include "TemporalWaveLoop.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::tensor_program_to_tile_region {
namespace {

class TemporalTraversalBuilder {
public:
  TemporalTraversalBuilder(
      mlir::Operation *root, TensorProgramScope scope,
      llvm::ArrayRef<int64_t> spatialOffsets,
      llvm::ArrayRef<int64_t> spatialSizes,
      llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
      llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
      llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
      std::string *failureReason,
      llvm::SmallVectorImpl<MaterializedCoupledProducerTile>
          *sharedProducerTiles)
      : root(root), scope(scope),
        spatialOffsets(spatialOffsets.begin(), spatialOffsets.end()),
        spatialSizes(spatialSizes.begin(), spatialSizes.end()),
        operationTemporalTiles(operationTemporalTiles),
        nestedTemporalTiles(nestedTemporalTiles),
        operationNodes(operationNodes), failureReason(failureReason),
        sharedProducerTiles(sharedProducerTiles) {}

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
  materialize(mlir::ValueRange outputs) {
    auto selected = llvm::find_if(operationTemporalTiles,
                                  [&](const StructuredOpTemporalTile &tile) {
                                    return tile.operation == root;
                                  });
    if (!root || selected == operationTemporalTiles.end() ||
        spatialOffsets.size() != spatialSizes.size() ||
        spatialSizes.size() != selected->iteratorTileSizes.size() ||
        outputs.size() != root->getNumResults())
      return fail("temporal traversal request is structurally inconsistent");
    temporal = &*selected;
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(root);
    if (!tiling || tiling.getLoopIteratorTypes().size() != spatialSizes.size())
      return fail("temporal traversal root has no matching iterator domain");
    for (auto [extent, tile] :
         llvm::zip_equal(spatialSizes, temporal->iteratorTileSizes))
      if (extent <= 0 || tile <= 0)
        return fail("temporal traversal has a non-positive extent or tile");

    llvm::SmallVector<uint32_t, 4> active;
    for (auto [dimension, extent, tile] :
         llvm::enumerate(spatialSizes, temporal->iteratorTileSizes))
      if (std::min(extent, tile) < extent)
        active.push_back(static_cast<uint32_t>(dimension));
    if (temporal->waveLoopOrder.empty()) {
      order = active;
    } else {
      llvm::SmallVector<uint32_t, 4> seen;
      for (uint32_t dimension : temporal->waveLoopOrder) {
        if (dimension >= spatialSizes.size() ||
            llvm::is_contained(seen, dimension))
          return fail("temporal wave-loop order is malformed");
        seen.push_back(dimension);
        if (std::min(spatialSizes[dimension],
                     temporal->iteratorTileSizes[dimension]) <
            spatialSizes[dimension])
          order.push_back(dimension);
      }
    }
    llvm::SmallVector<uint32_t, 4> sortedOrder = order;
    llvm::sort(sortedOrder);
    if (sortedOrder != active)
      return fail("temporal wave-loop order does not cover active iterators");

    mlir::OpBuilder builder(root);
    return materializeTemporalWaveLoopNest(
        builder, root->getLoc(), spatialOffsets, spatialSizes,
        temporal->iteratorTileSizes, temporal->waveLoopOrder, outputs,
        [&](mlir::OpBuilder &leafBuilder,
            llvm::ArrayRef<mlir::OpFoldResult> leafOffsets,
            llvm::ArrayRef<int64_t> leafSizes, mlir::ValueRange leafOutputs,
            llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops) {
          return emitLeaf(leafBuilder, leafOffsets, leafSizes, leafOutputs,
                          loops);
        },
        failureReason);
  }

private:
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
  fail(llvm::StringRef detail) {
    setFailureReason(failureReason, detail);
    return mlir::failure();
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
  emitLeaf(mlir::OpBuilder &builder, llvm::ArrayRef<mlir::OpFoldResult> offsets,
           llvm::ArrayRef<int64_t> sizes, mlir::ValueRange outputs,
           llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops) {
    llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
    for (int64_t size : sizes)
      mixedSizes.push_back(builder.getIndexAttr(size));
    auto tile = materializeStructuredIterationTile(root, builder, offsets,
                                                   mixedSizes, failureReason);
    if (mlir::failed(tile) || tile->values.size() != outputs.size())
      return mlir::failure();

    for (unsigned result = 0; result < tile->values.size(); ++result) {
      auto tiledResult = mlir::dyn_cast<mlir::OpResult>(tile->values[result]);
      auto dps = tiledResult
                     ? mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
                           tiledResult.getOwner())
                     : mlir::DestinationStyleOpInterface{};
      if (!dps || tiledResult.getResultNumber() >= dps.getNumDpsInits())
        return fail("temporal leaf result has no matching DPS init");
      auto outputType =
          mlir::dyn_cast<mlir::RankedTensorType>(outputs[result].getType());
      if (!outputType)
        return fail("temporal output destination is not ranked");
      llvm::SmallVector<mlir::OpFoldResult, 4> strides(outputType.getRank(),
                                                       builder.getIndexAttr(1));
      builder.setInsertionPoint(tile->operations.front());
      auto accumulator = builder.create<mlir::tensor::ExtractSliceOp>(
          root->getLoc(), outputs[result], tile->resultOffsets[result],
          tile->resultSizes[result], strides);
      mlir::OpOperand *init =
          dps.getDpsInitOperand(tiledResult.getResultNumber());
      if (!init || init->get().getType() != accumulator.getType())
        return fail("temporal accumulator type does not match tiled result");
      mlir::Value oldInit = init->get();
      init->set(accumulator.getResult());
      if (mlir::Operation *definition = oldInit.getDefiningOp();
          definition && definition->use_empty() &&
          mlir::isOpTriviallyDead(definition))
        definition->erase();
    }

    builder.setInsertionPointAfter(tile->operations.back());
    for (mlir::Operation *operation : tile->operations) {
      recordStructuredOperationNodeMaterialization(root, operation,
                                                   &operationNodes);
      if (sharedProducerTiles)
        reuseMaterializedProducerTiles(tile->generatedSlices,
                                       *sharedProducerTiles);
      mlir::LogicalResult fused =
          sharedProducerTiles
              ? fuseCandidateProducerSlicesWithCache(
                    operation, root, scope, loops, operationTemporalTiles,
                    nestedTemporalTiles, builder.getListener(), failureReason,
                    &operationNodes, *sharedProducerTiles)
              : fuseCandidateProducerSlices(
                    operation, root, scope, loops, operationTemporalTiles,
                    nestedTemporalTiles, builder.getListener(), failureReason,
                    &operationNodes);
      if (mlir::failed(fused))
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Value, 2> updated;
    for (unsigned result = 0; result < tile->values.size(); ++result) {
      auto outputType =
          mlir::cast<mlir::RankedTensorType>(outputs[result].getType());
      llvm::SmallVector<mlir::OpFoldResult, 4> strides(outputType.getRank(),
                                                       builder.getIndexAttr(1));
      if (sharedProducerTiles) {
        auto remember = [&](mlir::OpResult producerResult) {
          sharedProducerTiles->push_back(MaterializedCoupledProducerTile{
              producerResult, tile->values[result].getParentBlock(),
              tile->values[result].getType(),
              llvm::to_vector<4>(tile->resultOffsets[result]),
              llvm::to_vector<4>(tile->resultSizes[result]), strides,
              tile->values[result]});
        };
        remember(mlir::cast<mlir::OpResult>(root->getResult(result)));
        if (auto materialized =
                mlir::dyn_cast<mlir::OpResult>(tile->values[result]);
            materialized && materialized != root->getResult(result))
          remember(materialized);
      }
      updated.push_back(builder
                            .create<mlir::tensor::InsertSliceOp>(
                                root->getLoc(), tile->values[result],
                                outputs[result], tile->resultOffsets[result],
                                tile->resultSizes[result], strides)
                            .getResult());
    }
    return updated;
  }

  mlir::Operation *root;
  TensorProgramScope scope;
  llvm::SmallVector<int64_t, 4> spatialOffsets;
  llvm::SmallVector<int64_t, 4> spatialSizes;
  llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles;
  llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles;
  const StructuredOpTemporalTile *temporal = nullptr;
  llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes;
  std::string *failureReason;
  llvm::SmallVectorImpl<MaterializedCoupledProducerTile> *sharedProducerTiles;
  llvm::SmallVector<uint32_t, 4> order;
};

} // namespace

mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
materializeTemporalRegionTraversal(
    mlir::Operation *root, TensorProgramScope scope,
    llvm::ArrayRef<int64_t> spatialOffsets,
    llvm::ArrayRef<int64_t> spatialSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
    mlir::ValueRange outputDestinations,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason,
    llvm::SmallVectorImpl<MaterializedCoupledProducerTile>
        *sharedProducerTiles) {
  return TemporalTraversalBuilder(root, scope, spatialOffsets, spatialSizes,
                                  operationTemporalTiles, nestedTemporalTiles,
                                  operationNodes, failureReason,
                                  sharedProducerTiles)
      .materialize(outputDestinations);
}

} // namespace wafer::tensor_program_to_tile_region
