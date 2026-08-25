//===- CoupledRegionFunction.cpp - Multi-node region function ----------===//

#include "SingleRootTileRegionInternal.h"

#include "Internal.h"
#include "ProducerTileFusionInternal.h"
#include "StructuredIterationTile.h"
#include "TemporalRegionTraversal.h"

#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"

namespace wafer::tensor_program_to_tile_region {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
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
  llvm::StringRef currentStage = "validate selected coupled group";
  bool completed = false;
  auto diagnoseEmptyFailure = llvm::make_scope_exit([&] {
    if (!completed && failureReason && failureReason->empty())
      *failureReason =
          (llvm::Twine(
               "selected coupled construction failed while trying to ") +
           currentStage)
              .str();
  });
  if (group.shards.size() < 2 && group.recomputedProducerNodes.empty())
    return fail<RootFragment>(failureReason,
                              "coupled region requires several node shards");
  const TileId tile = group.shards.front().tile;
  llvm::DenseSet<uint32_t> selectedNodeIds;
  llvm::SmallVector<uint32_t, 8> orderedNodeIds;
  llvm::DenseMap<uint32_t, const StructuredNodeIterationShard *> shardsByNode;
  for (const StructuredNodeIterationShard &shard : group.shards) {
    if (shard.tile != tile || !shard.reductionGroups.empty() ||
        !selectedNodeIds.insert(shard.structuredNodeId).second)
      return fail<RootFragment>(
          failureReason,
          "coupled region has mixed Tiles, partial reduction, or duplicates");
    orderedNodeIds.push_back(shard.structuredNodeId);
    shardsByNode.try_emplace(shard.structuredNodeId, &shard);
  }
  llvm::sort(orderedNodeIds);
  llvm::SmallVector<uint32_t, 4> recomputedNodeIds(
      group.recomputedProducerNodes.begin(),
      group.recomputedProducerNodes.end());
  llvm::sort(recomputedNodeIds);
  if (std::adjacent_find(recomputedNodeIds.begin(), recomputedNodeIds.end()) !=
          recomputedNodeIds.end() ||
      llvm::any_of(recomputedNodeIds, [&](uint32_t node) {
        return selectedNodeIds.contains(node);
      }))
    return fail<RootFragment>(
        failureReason,
        "recomputed producer identities are duplicated or already selected");
  llvm::SmallVector<uint32_t, 4> independentlyMaterializedNodeIds(
      group.independentlyMaterializedNodes.begin(),
      group.independentlyMaterializedNodes.end());
  llvm::sort(independentlyMaterializedNodeIds);
  if (std::adjacent_find(independentlyMaterializedNodeIds.begin(),
                         independentlyMaterializedNodeIds.end()) !=
          independentlyMaterializedNodeIds.end() ||
      llvm::any_of(independentlyMaterializedNodeIds, [&](uint32_t node) {
        return !selectedNodeIds.contains(node) &&
               !llvm::is_contained(recomputedNodeIds, node);
      }))
    return fail<RootFragment>(
        failureReason,
        "independent materialization identities are duplicated or not "
        "scheduled in the group");
  llvm::DenseMap<uint32_t, const StructuredNodeIterationShard *>
      recomputedShardsByNode;
  for (const StructuredNodeIterationShard &shard :
       group.recomputedProducerShards)
    if (!recomputedShardsByNode.try_emplace(shard.structuredNodeId, &shard)
             .second)
      return fail<RootFragment>(
          failureReason,
          "recomputed producer work has a duplicate node identity");

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
  for (uint32_t node : recomputedNodeIds)
    if (!sourceByNode.lookup(node))
      return fail<RootFragment>(failureReason,
                                "recompute references an unknown producer");
  llvm::sort(selectedOperations,
             [](mlir::Operation *lhs, mlir::Operation *rhs) {
               return lhs->isBeforeInBlock(rhs);
             });
  llvm::DenseSet<mlir::Operation *> selectedOperationSet(
      selectedOperations.begin(), selectedOperations.end());
  llvm::DenseSet<mlir::Operation *> independentSourceOperations;
  for (uint32_t node : independentlyMaterializedNodeIds)
    if (mlir::Operation *operation = sourceByNode.lookup(node))
      independentSourceOperations.insert(operation);
  auto sourceFunction =
      selectedOperations.front()->getParentOfType<mlir::func::FuncOp>();
  mlir::FailureOr<compiler::detail::StructuredDAGAnalysis> sourceDAG =
      compiler::detail::StructuredDAGAnalysis::create(sourceFunction,
                                                      failureReason);
  if (mlir::failed(sourceDAG))
    return mlir::failure();
  llvm::DenseSet<std::pair<mlir::Operation *, mlir::Operation *>>
      structuredDependencies;
  for (const compiler::detail::StructuredDAGEdge &edge :
       sourceDAG->getEdges()) {
    const compiler::detail::StructuredDAGNode *producer =
        sourceDAG->getNode(edge.producer);
    const compiler::detail::StructuredDAGNode *consumer =
        sourceDAG->getNode(edge.consumer);
    if (producer && consumer)
      structuredDependencies.insert({producer->operation, consumer->operation});
  }

  llvm::SmallVector<mlir::Operation *, 4> sourceSinks;
  for (mlir::Operation *candidate : selectedOperations) {
    bool hasOutgoing =
        llvm::any_of(selectedOperations, [&](mlir::Operation *consumer) {
          return structuredDependencies.contains({candidate, consumer});
        });
    bool observable =
        llvm::any_of(candidate->getResults(), [&](mlir::Value result) {
          llvm::DenseSet<mlir::Value> visited;
          return reachesObservableBoundary(result, selectedOperationSet,
                                           visited);
        });
    if (observable ||
        (!hasOutgoing && !independentSourceOperations.contains(candidate)))
      sourceSinks.push_back(candidate);
  }
  if (sourceSinks.empty())
    return fail<RootFragment>(failureReason,
                              "coupled group has no consumer sink");

  RootFragment result;
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
  unsigned functionalArgumentCount = 0;
  currentStage = "build the private coupled function";
  mlir::FailureOr<mlir::func::FuncOp> function = buildCoupledRootFunction(
      tileOwner.getBody().front(), sourceSinks, sourceOperationNodes,
      orderedNodeIds, recomputedNodeIds, independentlyMaterializedNodeIds,
      failureReason, operationNodes, functionalArgumentCount, result.boundaries,
      result.results);
  if (mlir::failed(function))
    return mlir::failure();
  for (const StructuredNodeLocalUse &use : group.localUses) {
    auto producer = llvm::find_if(
        operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
          return mapping.structuredNodeId == use.producerNodeId;
        });
    auto consumer = llvm::find_if(
        operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
          return mapping.structuredNodeId == use.consumerNodeId;
        });
    if (producer == operationNodes.end() || consumer == operationNodes.end() ||
        !producer->operation || !consumer->operation ||
        use.producerResult >= producer->operation->getNumResults() ||
        use.consumerOperand >= consumer->operation->getNumOperands() ||
        producer->operation->getResult(use.producerResult).getType() !=
            consumer->operation->getOperand(use.consumerOperand).getType())
      return fail<RootFragment>(
          failureReason,
          "selected local use has no current producer/consumer SSA edge");
    consumer->operation->setOperand(
        use.consumerOperand,
        producer->operation->getResult(use.producerResult));
  }
  llvm::BitVector eraseArguments((*function).getNumArguments());
  llvm::SmallVector<RootValueKey, 8> retainedBoundaries;
  for (auto [index, boundary] : llvm::enumerate(result.boundaries)) {
    mlir::BlockArgument argument = (*function).getArgument(index);
    if (argument.use_empty())
      eraseArguments.set(index);
    else
      retainedBoundaries.push_back(boundary);
  }
  const unsigned erasedArguments = eraseArguments.count();
  if (erasedArguments != 0) {
    (*function).eraseArguments(eraseArguments);
    functionalArgumentCount -= erasedArguments;
    result.boundaries = std::move(retainedBoundaries);
  }
  if (mlir::failed(appendTileOutputDestinations(*function, failureReason)))
    return mlir::failure();

  TensorProgramScope scope(*function, functionalArgumentCount);
  mlir::func::ReturnOp returnOp = scope.getReturn();
  llvm::SmallVector<StructuredOpTemporalTile, 8> mappedTemporalTiles;
  mappedTemporalTiles.reserve(group.temporalTiles.size());
  for (const StructuredNodeTemporalTile &temporal : group.temporalTiles) {
    auto mapped = llvm::find_if(
        operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
          return mapping.structuredNodeId == temporal.structuredNodeId;
        });
    if (mapped == operationNodes.end() || !mapped->operation)
      return fail<RootFragment>(
          failureReason,
          "coupled temporal assignment has no current operation mapping");
    const StructuredNodeIterationShard *shard =
        shardsByNode.lookup(temporal.structuredNodeId);
    if (!shard || shard->sizes.size() != temporal.iteratorTileSizes.size())
      return fail<RootFragment>(
          failureReason,
          "coupled temporal assignment has no matching iterator shard");
    if (llvm::none_of(llvm::zip_equal(shard->sizes, temporal.iteratorTileSizes),
                      [](auto values) {
                        auto [extent, tile] = values;
                        return tile < extent;
                      }))
      continue;
    mappedTemporalTiles.push_back(StructuredOpTemporalTile{
        mapped->operation, temporal.iteratorTileSizes, temporal.waveLoopOrder});
  }
  llvm::SmallVector<StructuredOpNestedTemporalTile, 8>
      mappedNestedTemporalTiles;
  mappedNestedTemporalTiles.reserve(group.nestedTemporalTiles.size());
  for (const StructuredNodeNestedTemporalTile &nested :
       group.nestedTemporalTiles) {
    auto producer = llvm::find_if(
        operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
          return mapping.structuredNodeId == nested.producerNodeId;
        });
    auto parent = llvm::find_if(
        operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
          return mapping.structuredNodeId == nested.parentNodeId;
        });
    if (producer == operationNodes.end() || parent == operationNodes.end() ||
        !producer->operation || !parent->operation ||
        nested.producerResult >= producer->operation->getNumResults() ||
        nested.parentOperand >= parent->operation->getNumOperands())
      return fail<RootFragment>(
          failureReason,
          "nested temporal assignment has no current producer/use mapping");
    mappedNestedTemporalTiles.push_back(
        {producer->operation, parent->operation, nested.producerResult,
         nested.parentOperand, nested.requestedResultExtents,
         nested.producerIterationExtents, nested.iteratorTileSizes,
         nested.waveLoopOrder});
  }
  unsigned outputIndex = 0;
  llvm::SmallVector<MaterializedCoupledProducerTile, 8> sharedProducerTiles;
  auto materializeIndependentRoot =
      [&](mlir::Operation *root,
          const StructuredNodeIterationShard &shard) -> mlir::LogicalResult {
    mlir::OpBuilder builder(root);
    auto selectedTemporal = llvm::find_if(
        mappedTemporalTiles, [&](const StructuredOpTemporalTile &temporal) {
          return temporal.operation == root;
        });
    if (selectedTemporal != mappedTemporalTiles.end()) {
      auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(root);
      if (!dps || dps.getNumDpsInits() != root->getNumResults()) {
        setFailureReason(
            failureReason,
            "independent temporal root has no matching DPS destinations");
        return mlir::failure();
      }
      llvm::SmallVector<mlir::Value, 2> destinations(dps.getDpsInits().begin(),
                                                     dps.getDpsInits().end());
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> traversed =
          materializeTemporalRegionTraversal(
              root, scope, shard.offsets, shard.sizes, mappedTemporalTiles,
              mappedNestedTemporalTiles, destinations, operationNodes,
              failureReason, &sharedProducerTiles);
      return mlir::succeeded(traversed) ? mlir::success() : mlir::failure();
    }

    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    for (auto [offset, size] : llvm::zip_equal(shard.offsets, shard.sizes)) {
      offsets.push_back(builder.getIndexAttr(offset));
      sizes.push_back(builder.getIndexAttr(size));
    }
    mlir::FailureOr<StructuredIterationTile> tile =
        materializeStructuredIterationTile(root, builder, offsets, sizes,
                                           failureReason);
    if (mlir::failed(tile)) {
      if (!failureReason || failureReason->empty())
        setFailureReason(failureReason,
                         "independent producer iterator tiling failed");
      return mlir::failure();
    }
    reuseMaterializedProducerTiles(tile->generatedSlices, sharedProducerTiles);
    llvm::SmallVector<mlir::LoopLikeOpInterface, 0> loops;
    for (mlir::Operation *tiledOperation : tile->operations) {
      recordStructuredOperationNodeMaterialization(root, tiledOperation,
                                                   &operationNodes);
      if (mlir::failed(fuseCandidateProducerSlicesWithCache(
              tiledOperation, root, scope, loops, mappedTemporalTiles,
              mappedNestedTemporalTiles, builder.getListener(), failureReason,
              &operationNodes, sharedProducerTiles))) {
        if (!failureReason || failureReason->empty())
          setFailureReason(failureReason,
                           "independent producer operand fusion failed");
        return mlir::failure();
      }
    }
    for (unsigned resultNumber = 0; resultNumber < tile->values.size();
         ++resultNumber) {
      llvm::SmallVector<mlir::OpFoldResult, 4> strides(
          tile->resultOffsets[resultNumber].size(), builder.getIndexAttr(1));
      auto remember = [&](mlir::OpResult producerResult) {
        sharedProducerTiles.push_back(MaterializedCoupledProducerTile{
            producerResult, tile->values[resultNumber].getParentBlock(),
            tile->values[resultNumber].getType(),
            llvm::to_vector<4>(tile->resultOffsets[resultNumber]),
            llvm::to_vector<4>(tile->resultSizes[resultNumber]), strides,
            tile->values[resultNumber]});
      };
      remember(mlir::cast<mlir::OpResult>(root->getResult(resultNumber)));
      if (auto materialized =
              mlir::dyn_cast<mlir::OpResult>(tile->values[resultNumber]);
          materialized && materialized != root->getResult(resultNumber))
        remember(materialized);
    }
    return mlir::success();
  };

  llvm::SmallVector<mlir::Operation *, 4> independentRoots;
  for (uint32_t node : independentlyMaterializedNodeIds) {
    mlir::Operation *source = sourceByNode.lookup(node);
    if (!source || llvm::is_contained(sourceSinks, source))
      continue;
    auto mapped = llvm::find_if(
        operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
          return mapping.structuredNodeId == node;
        });
    const StructuredNodeIterationShard *shard = shardsByNode.lookup(node);
    if (!shard)
      shard = recomputedShardsByNode.lookup(node);
    if (mapped == operationNodes.end() || !mapped->operation || !shard)
      return fail<RootFragment>(
          failureReason,
          "independent materialization has no cloned operation or shard");
    independentRoots.push_back(mapped->operation);
  }
  llvm::sort(independentRoots, [](mlir::Operation *lhs, mlir::Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  currentStage = "materialize independent selected producers";
  for (mlir::Operation *root : independentRoots) {
    auto mapping = llvm::find_if(
        operationNodes, [&](const StructuredOperationNodeMapping &candidate) {
          return candidate.operation == root;
        });
    const StructuredNodeIterationShard *shard =
        mapping == operationNodes.end()
            ? nullptr
            : shardsByNode.lookup(mapping->structuredNodeId);
    if (!shard && mapping != operationNodes.end())
      shard = recomputedShardsByNode.lookup(mapping->structuredNodeId);
    if (!shard) {
      setFailureReason(
          failureReason,
          "independent selected producer lost its exact iterator work");
      return mlir::failure();
    }
    if (mlir::failed(materializeIndependentRoot(root, *shard))) {
      if (!failureReason || failureReason->empty())
        setFailureReason(
            failureReason,
            "independent selected producer materialization failed");
      return mlir::failure();
    }
  }

  currentStage = "materialize selected consumer sinks";
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
    auto selectedTemporal = llvm::find_if(
        mappedTemporalTiles, [&](const StructuredOpTemporalTile &temporal) {
          return temporal.operation == root;
        });
    if (selectedTemporal != mappedTemporalTiles.end()) {
      llvm::SmallVector<mlir::Value, 2> destinations;
      destinations.reserve(root->getNumResults());
      for (unsigned resultNumber = 0; resultNumber < root->getNumResults();
           ++resultNumber) {
        mlir::FailureOr<mlir::Value> destination = getCandidateOutputBoundary(
            scope, outputIndex + resultNumber, failureReason);
        if (mlir::failed(destination))
          return mlir::failure();
        destinations.push_back(*destination);
      }
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> traversed =
          materializeTemporalRegionTraversal(
              root, scope, shard->offsets, shard->sizes, mappedTemporalTiles,
              mappedNestedTemporalTiles, destinations, operationNodes,
              failureReason, &sharedProducerTiles);
      if (mlir::failed(traversed) || traversed->size() != root->getNumResults())
        return mlir::failure();
      for (unsigned resultNumber = 0; resultNumber < traversed->size();
           ++resultNumber, ++outputIndex) {
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
            root->getResult(resultNumber).getType());
        if (!resultType)
          return fail<RootFragment>(
              failureReason, "coupled temporal result is not a ranked tensor");
        mlir::OpBuilder cacheBuilder(root);
        llvm::SmallVector<mlir::OpFoldResult, 4> iterationOffsets;
        llvm::SmallVector<mlir::OpFoldResult, 4> iterationSizes;
        for (auto [offset, size] :
             llvm::zip_equal(shard->offsets, shard->sizes)) {
          iterationOffsets.push_back(cacheBuilder.getIndexAttr(offset));
          iterationSizes.push_back(cacheBuilder.getIndexAttr(size));
        }
        llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
        llvm::SmallVector<mlir::OpFoldResult> resultSizes;
        auto tiling = mlir::cast<mlir::TilingInterface>(root);
        if (mlir::failed(tiling.getResultTilePosition(
                cacheBuilder, resultNumber, iterationOffsets, iterationSizes,
                resultOffsets, resultSizes)))
          return fail<RootFragment>(
              failureReason,
              "coupled temporal result has no exact iterator relation");
        bool coversFullResult =
            resultOffsets.size() == static_cast<size_t>(resultType.getRank()) &&
            resultSizes.size() == static_cast<size_t>(resultType.getRank());
        for (auto [dimension, offset, size] :
             llvm::enumerate(resultOffsets, resultSizes)) {
          std::optional<int64_t> constantOffset =
              mlir::getConstantIntValue(offset);
          std::optional<int64_t> constantSize = mlir::getConstantIntValue(size);
          coversFullResult &= constantOffset && *constantOffset == 0 &&
                              constantSize &&
                              *constantSize == resultType.getDimSize(dimension);
        }
        if (coversFullResult) {
          llvm::SmallVector<mlir::OpFoldResult, 4> strides(
              resultType.getRank(), cacheBuilder.getIndexAttr(1));
          sharedProducerTiles.push_back(MaterializedCoupledProducerTile{
              root->getResult(resultNumber),
              (*traversed)[resultNumber].getParentBlock(),
              (*traversed)[resultNumber].getType(), std::move(resultOffsets),
              std::move(resultSizes), std::move(strides),
              (*traversed)[resultNumber]});
        }
        for (mlir::OpOperand &use : llvm::make_early_inc_range(
                 root->getResult(resultNumber).getUses()))
          if (use.getOwner() != returnOp.getOperation())
            use.set((*traversed)[resultNumber]);
        returnOp->setOperand(outputIndex, (*traversed)[resultNumber]);
      }
      continue;
    }
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
              tiledOperation, root, scope, loops, mappedTemporalTiles,
              mappedNestedTemporalTiles, builder.getListener(), failureReason,
              &operationNodes, sharedProducerTiles)))
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
  currentStage = "lower the coupled function to TileRegion IR";
  TileRegionEmissionRelations emissionRelations;
  if (mlir::failed(convertTensorProgramToTileRegionFunctionInPlace(
          *function, functionalArgumentCount,
          /*currentLogicalPartition=*/0, failureReason,
          /*suppressDiagnostics=*/true, /*verifyResult=*/true,
          /*populateFallbackFailureReason=*/true,
          /*peerEndpoints=*/{}, /*selectedDDRStages=*/{}, &emissionRelations,
          operationNodes, /*requireOneStructuredRootPerRegion=*/false,
          group.representations, group.implementations)))
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
  llvm::DenseSet<uint32_t> expectedNodes = selectedNodeIds;
  if (regionCount != 1 || emittedNodes.size() != expectedNodes.size() ||
      llvm::any_of(selectedNodeIds, [&](uint32_t node) {
        return !emittedNodes.contains(node);
      })) {
    std::string detail;
    llvm::raw_string_ostream diagnostic(detail);
    diagnostic << "coupled function did not emit the exact selected node "
                  "group; expected=[";
    llvm::interleaveComma(expectedNodes, diagnostic);
    diagnostic << "], emitted=[";
    llvm::interleaveComma(emittedNodes, diagnostic);
    diagnostic << "], missing=[";
    bool firstMissing = true;
    for (uint32_t node : selectedNodeIds) {
      if (emittedNodes.contains(node))
        continue;
      if (!firstMissing)
        diagnostic << ',';
      firstMissing = false;
      diagnostic << node << ':';
      if (mlir::Operation *source = sourceByNode.lookup(node))
        diagnostic << source->getName();
      else
        diagnostic << "unknown";
    }
    diagnostic << "], regions=" << regionCount;
    return fail<RootFragment>(failureReason, diagnostic.str());
  }
  result.relations = std::move(emissionRelations.materializedBuffers);
  completed = true;
  return result;
}

} // namespace wafer::tensor_program_to_tile_region
