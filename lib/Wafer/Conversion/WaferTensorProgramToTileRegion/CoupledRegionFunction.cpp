//===- CoupledRegionFunction.cpp - Multi-node region function ----------===//

#include "SingleRootTileRegionInternal.h"

#include "Internal.h"
#include "ProducerTileFusionInternal.h"
#include "StructuredIterationTile.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::tensor_program_to_tile_region {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

bool hasStructuredDependency(mlir::Operation *producer,
                             mlir::Operation *consumer) {
  if (!producer || !consumer || producer == consumer ||
      producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer))
    return false;
  for (unsigned result = 0; result < producer->getNumResults(); ++result)
    for (unsigned operand = 0; operand < consumer->getNumOperands();
         ++operand) {
      std::string ignored;
      if (mlir::succeeded(traceProducerToConsumerChain(
              producer, result, consumer, operand, &ignored)))
        return true;
    }
  return false;
}

bool reachesObservableBoundary(
    mlir::Value value,
    const llvm::DenseSet<mlir::Operation *> &selectedOperations,
    llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *user = use.getOwner();
    while (user && user->getBlock() != value.getParentBlock())
      user = user->getParentOp();
    if (!user)
      continue;
    if (mlir::isa<mlir::func::ReturnOp>(user))
      return true;
    if (selectedOperations.contains(user) || !mlir::isMemoryEffectFree(user))
      continue;
    for (mlir::Value result : user->getResults())
      if (reachesObservableBoundary(result, selectedOperations, visited))
        return true;
  }
  return false;
}

} // namespace

mlir::FailureOr<RootFragment> materializeCoupledRootFragment(
    TileModuleOp tileOwner,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    const StructuredNodeShardGroup &group, std::string *failureReason) {
  if (group.shards.size() < 2)
    return fail<RootFragment>(failureReason,
                              "coupled region requires several node shards");
  const TileId tile = group.shards.front().tile;
  llvm::DenseSet<uint32_t> selectedNodeIds;
  llvm::SmallVector<uint32_t, 8> orderedNodeIds;
  llvm::DenseMap<uint32_t, const StructuredNodeIterationShard *> shardsByNode;
  for (const StructuredNodeIterationShard &shard : group.shards) {
    if (shard.tile != tile ||
        shard.role != StructuredNodeIterationShardRole::Complete ||
        shard.reductionMergeTile ||
        !selectedNodeIds.insert(shard.structuredNodeId).second)
      return fail<RootFragment>(
          failureReason,
          "coupled region has mixed Tiles, partial reduction, or duplicates");
    orderedNodeIds.push_back(shard.structuredNodeId);
    shardsByNode.try_emplace(shard.structuredNodeId, &shard);
  }
  llvm::sort(orderedNodeIds);

  llvm::DenseMap<uint32_t, mlir::Operation *> sourceByNode;
  llvm::DenseSet<mlir::Operation *> seenOperations;
  for (const StructuredOperationNodeMapping &mapping : sourceOperationNodes) {
    if (!mapping.operation ||
        !seenOperations.insert(mapping.operation).second ||
        sourceByNode.count(mapping.structuredNodeId))
      return fail<RootFragment>(failureReason,
                                "coupled source mapping is malformed");
    sourceByNode.try_emplace(mapping.structuredNodeId, mapping.operation);
  }
  llvm::SmallVector<mlir::Operation *, 8> selectedOperations;
  for (uint32_t node : orderedNodeIds) {
    mlir::Operation *operation = sourceByNode.lookup(node);
    if (!operation)
      return fail<RootFragment>(failureReason,
                                "coupled group references an unknown node");
    selectedOperations.push_back(operation);
  }
  llvm::sort(selectedOperations,
             [](mlir::Operation *lhs, mlir::Operation *rhs) {
               return lhs->isBeforeInBlock(rhs);
             });
  llvm::DenseSet<mlir::Operation *> selectedOperationSet(
      selectedOperations.begin(), selectedOperations.end());

  llvm::SmallVector<mlir::Operation *, 4> sourceSinks;
  for (mlir::Operation *candidate : selectedOperations) {
    bool hasOutgoing =
        llvm::any_of(selectedOperations, [&](mlir::Operation *consumer) {
          return hasStructuredDependency(candidate, consumer);
        });
    bool observable =
        llvm::any_of(candidate->getResults(), [&](mlir::Value result) {
          llvm::DenseSet<mlir::Value> visited;
          return reachesObservableBoundary(result, selectedOperationSet,
                                           visited);
        });
    if (!hasOutgoing || observable)
      sourceSinks.push_back(candidate);
  }
  if (sourceSinks.empty())
    return fail<RootFragment>(failureReason,
                              "coupled group has no consumer sink");

  RootFragment result;
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
  unsigned functionalArgumentCount = 0;
  mlir::FailureOr<mlir::func::FuncOp> function = buildCoupledRootFunction(
      tileOwner.getBody().front(), sourceSinks, sourceOperationNodes,
      orderedNodeIds, failureReason, operationNodes, functionalArgumentCount);
  if (mlir::failed(function) ||
      mlir::failed(appendTileOutputDestinations(*function, failureReason)))
    return mlir::failure();

  TensorProgramScope scope(*function, functionalArgumentCount);
  mlir::func::ReturnOp returnOp = scope.getReturn();
  unsigned outputIndex = 0;
  llvm::SmallVector<MaterializedCoupledProducerTile, 8> sharedProducerTiles;
  for (mlir::Operation *sourceSink : sourceSinks) {
    auto sourceMapping =
        llvm::find_if(sourceOperationNodes,
                      [&](const StructuredOperationNodeMapping &mapping) {
                        return mapping.operation == sourceSink;
                      });
    if (sourceMapping == sourceOperationNodes.end())
      return fail<RootFragment>(failureReason,
                                "coupled sink has no structured identity");
    auto mapped = llvm::find_if(
        operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
          return mapping.structuredNodeId == sourceMapping->structuredNodeId;
        });
    const StructuredNodeIterationShard *shard =
        shardsByNode.lookup(sourceMapping->structuredNodeId);
    if (mapped == operationNodes.end() || !mapped->operation || !shard ||
        shard->offsets.size() != shard->sizes.size() ||
        llvm::any_of(shard->sizes, [](int64_t size) { return size <= 0; }))
      return fail<RootFragment>(failureReason,
                                "coupled sink shard is malformed");

    mlir::Operation *root = mapped->operation;
    mlir::OpBuilder builder(root);
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    for (auto [offset, size] : llvm::zip_equal(shard->offsets, shard->sizes)) {
      offsets.push_back(builder.getIndexAttr(offset));
      sizes.push_back(builder.getIndexAttr(size));
    }
    mlir::FailureOr<StructuredIterationTile> tile =
        materializeStructuredIterationTile(root, builder, offsets, sizes,
                                           failureReason);
    if (mlir::failed(tile))
      return mlir::failure();
    reuseMaterializedProducerTiles(tile->generatedSlices, sharedProducerTiles);
    llvm::SmallVector<mlir::LoopLikeOpInterface, 0> loops;
    for (mlir::Operation *tiledOperation : tile->operations) {
      recordStructuredOperationNodeMaterialization(root, tiledOperation,
                                                   &operationNodes);
      if (mlir::failed(fuseCandidateProducerSlicesWithCache(
              tiledOperation, root, scope, loops,
              /*operationTemporalTiles=*/{}, builder.getListener(),
              failureReason, &operationNodes, sharedProducerTiles)))
        return mlir::failure();
    }
    for (unsigned resultNumber = 0; resultNumber < tile->values.size();
         ++resultNumber) {
      llvm::SmallVector<mlir::OpFoldResult, 4> strides(
          tile->resultOffsets[resultNumber].size(), builder.getIndexAttr(1));
      auto appendCachedVersion = [&](mlir::OpResult producerResult) {
        sharedProducerTiles.push_back(MaterializedCoupledProducerTile{
            producerResult, tile->values[resultNumber].getParentBlock(),
            tile->values[resultNumber].getType(),
            llvm::to_vector<4>(tile->resultOffsets[resultNumber]),
            llvm::to_vector<4>(tile->resultSizes[resultNumber]), strides,
            tile->values[resultNumber]});
      };
      appendCachedVersion(
          mlir::cast<mlir::OpResult>(root->getResult(resultNumber)));
      if (auto materializedResult =
              mlir::dyn_cast<mlir::OpResult>(tile->values[resultNumber]);
          materializedResult &&
          materializedResult != root->getResult(resultNumber))
        appendCachedVersion(materializedResult);
      if (tile->values[resultNumber].getType() ==
          root->getResult(resultNumber).getType())
        for (mlir::OpOperand &use : llvm::make_early_inc_range(
                 root->getResult(resultNumber).getUses()))
          if (use.getOwner() != returnOp.getOperation())
            use.set(tile->values[resultNumber]);
    }

    builder.setInsertionPoint(returnOp);
    for (unsigned resultNumber = 0; resultNumber < tile->values.size();
         ++resultNumber, ++outputIndex) {
      mlir::FailureOr<mlir::Value> destination =
          getCandidateOutputBoundary(scope, outputIndex, failureReason);
      if (mlir::failed(destination))
        return mlir::failure();
      auto destinationType =
          mlir::dyn_cast<mlir::RankedTensorType>(destination->getType());
      if (!destinationType)
        return fail<RootFragment>(failureReason,
                                  "coupled sink output is not a ranked tensor");
      llvm::SmallVector<mlir::OpFoldResult, 4> strides(
          destinationType.getRank(), builder.getIndexAttr(1));
      auto inserted = builder.create<mlir::tensor::InsertSliceOp>(
          root->getLoc(), tile->values[resultNumber], *destination,
          tile->resultOffsets[resultNumber], tile->resultSizes[resultNumber],
          strides);
      returnOp->setOperand(outputIndex, inserted.getResult());
    }
  }
  if (outputIndex != returnOp.getNumOperands())
    return fail<RootFragment>(failureReason,
                              "coupled sinks did not cover every result");

  eraseDeadCandidateSupportClosure(scope);
  retainLiveOperationNodes(*function, operationNodes);
  TileRegionEmissionRelations emissionRelations;
  if (mlir::failed(convertTensorProgramToTileRegionFunctionInPlace(
          *function, functionalArgumentCount,
          /*currentLogicalPartition=*/0, failureReason,
          /*suppressDiagnostics=*/true, /*verifyResult=*/true,
          /*populateFallbackFailureReason=*/true,
          /*peerEndpoints=*/{}, /*selectedDDRStages=*/{}, &emissionRelations,
          operationNodes, /*requireOneStructuredRootPerRegion=*/false)))
    return mlir::failure();
  result.function = *function;

  unsigned regionCount = 0;
  result.function.walk([&](TileRegionOp region) {
    if (!region->getParentOfType<TileRegionOp>())
      ++regionCount;
  });
  llvm::DenseSet<uint32_t> emittedNodes;
  for (const StructuredOperationEmissionRelation &relation :
       emissionRelations.materializedBuffers.operationEmissions)
    if (relation.operation)
      emittedNodes.insert(relation.structuredNodeId);
  if (regionCount != 1 || emittedNodes.size() != selectedNodeIds.size() ||
      llvm::any_of(selectedNodeIds,
                   [&](uint32_t node) { return !emittedNodes.contains(node); }))
    return fail<RootFragment>(
        failureReason,
        "coupled function did not emit the exact selected node group");
  result.relations = std::move(emissionRelations.materializedBuffers);
  return result;
}

} // namespace wafer::tensor_program_to_tile_region
