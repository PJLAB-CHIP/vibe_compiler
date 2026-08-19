//===- SingleRootRegionFunction.cpp - One structured-root function -----===//

#include "SingleRootTileRegionInternal.h"

#include "Internal.h"
#include "StructuredIterationTile.h"
#include "TemporalPartialReductionTraversal.h"
#include "TemporalRegionTraversal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <functional>

namespace wafer::tensor_program_to_tile_region {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

mlir::LogicalResult failResult(std::string *failureReason,
                               llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

} // namespace

struct RootClosure {
  llvm::SmallVector<mlir::Value, 8> boundaries;
  llvm::DenseSet<mlir::Operation *> operations;
};

mlir::FailureOr<RootClosure> collectRootClosure(
    llvm::ArrayRef<mlir::Operation *> roots, mlir::Block &sourceBody,
    const llvm::DenseSet<mlir::Operation *> &structuredOperations,
    const llvm::DenseSet<mlir::Operation *> &coupledOperations,
    std::string *failureReason) {
  if (roots.empty())
    return fail<RootClosure>(failureReason,
                             "region construction requires a sink root");
  for (mlir::Operation *root : roots)
    if (!root || root->getBlock() != &sourceBody ||
        !coupledOperations.contains(root) ||
        !mlir::isa<mlir::TilingInterface>(root) ||
        !mlir::isa<mlir::DestinationStyleOpInterface>(root) ||
        root->getNumResults() == 0 || !mlir::isMemoryEffectFree(root))
      return fail<RootClosure>(
          failureReason,
          "region construction requires pure top-level tiled DPS roots");

  RootClosure result;
  llvm::DenseSet<mlir::Value> seenBoundaries;
  llvm::DenseSet<mlir::Value> visitedValues;
  std::function<mlir::LogicalResult(mlir::Value)> collectValue =
      [&](mlir::Value value) -> mlir::LogicalResult {
    if (!value || !visitedValues.insert(value).second)
      return mlir::success();
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() != &sourceBody)
        return failResult(
            failureReason,
            "single-root dependency escaped the source function body");
      if (seenBoundaries.insert(value).second)
        result.boundaries.push_back(value);
      return mlir::success();
    }
    auto opResult = mlir::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *definition = opResult ? opResult.getOwner() : nullptr;
    if (!definition || definition->getBlock() != &sourceBody)
      return failResult(
          failureReason,
          "single-root dependency has no top-level source definition");
    if (structuredOperations.contains(definition) &&
        !coupledOperations.contains(definition)) {
      if (seenBoundaries.insert(value).second)
        result.boundaries.push_back(value);
      return mlir::success();
    }
    if (!mlir::isMemoryEffectFree(definition))
      return failResult(
          failureReason,
          "single-root dependency crosses an effectful operation");
    if (!result.operations.insert(definition).second)
      return mlir::success();
    for (mlir::Value operand : definition->getOperands())
      if (mlir::failed(collectValue(operand)))
        return mlir::failure();
    return mlir::success();
  };

  for (mlir::Operation *root : roots) {
    result.operations.insert(root);
    for (mlir::Value operand : root->getOperands())
      if (mlir::failed(collectValue(operand)))
        return mlir::failure();
  }
  return result;
}

mlir::FailureOr<mlir::func::FuncOp> buildRegionFunction(
    mlir::Block &destination, mlir::Operation *sourceRoot,
    llvm::ArrayRef<mlir::Operation *> sourceRoots,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    const llvm::DenseSet<mlir::Operation *> &structuredOperations,
    const llvm::DenseSet<mlir::Operation *> &coupledOperations,
    llvm::StringRef functionName, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    unsigned &functionalArgumentCount) {
  mlir::func::FuncOp sourceFunction =
      sourceRoot ? sourceRoot->getParentOfType<mlir::func::FuncOp>()
                 : mlir::func::FuncOp{};
  if (!sourceFunction || sourceFunction.isExternal() ||
      !sourceFunction.getBody().hasOneBlock())
    return fail<mlir::func::FuncOp>(
        failureReason,
        "single-root region requires one defined single-block source function");
  mlir::Block &sourceBody = sourceFunction.getBody().front();
  mlir::FailureOr<RootClosure> closure =
      collectRootClosure(sourceRoots, sourceBody, structuredOperations,
                         coupledOperations, failureReason);
  if (mlir::failed(closure))
    return mlir::failure();

  llvm::SmallVector<mlir::Type, 8> inputTypes;
  inputTypes.reserve(closure->boundaries.size());
  for (mlir::Value boundary : closure->boundaries)
    inputTypes.push_back(boundary.getType());
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (mlir::Operation *root : sourceRoots)
    resultTypes.append(root->getResultTypes().begin(),
                       root->getResultTypes().end());
  if (llvm::any_of(resultTypes, [](mlir::Type type) {
        auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
        return !tensor || !tensor.hasStaticShape();
      }))
    return fail<mlir::func::FuncOp>(
        failureReason,
        "region construction requires static ranked tensor results");

  mlir::OpBuilder builder(&destination, destination.end());
  auto function = mlir::func::FuncOp::create(
      sourceRoot->getLoc(), functionName,
      builder.getFunctionType(inputTypes, resultTypes));
  function.setPrivate();
  destination.push_back(function.getOperation());
  mlir::Block *body = function.addEntryBlock();

  mlir::IRMapping mapping;
  for (auto [boundary, argument] :
       llvm::zip_equal(closure->boundaries, body->getArguments()))
    mapping.map(boundary, argument);
  builder.setInsertionPointToStart(body);
  for (mlir::Operation &operation : sourceBody.without_terminator()) {
    if (!closure->operations.contains(&operation))
      continue;
    mlir::Operation *cloned = builder.clone(operation, mapping);
    auto node =
        llvm::find_if(sourceOperationNodes,
                      [&](const StructuredOperationNodeMapping &candidate) {
                        return candidate.operation == &operation;
                      });
    if (node != sourceOperationNodes.end())
      operationNodes.push_back({cloned, node->structuredNodeId});
  }
  llvm::SmallVector<mlir::Value, 4> returnedValues;
  for (mlir::Operation *root : sourceRoots) {
    mlir::Operation *mapped = mapping.lookupOrNull(root);
    if (!mapped || mapped->getNumResults() != root->getNumResults())
      return fail<mlir::func::FuncOp>(failureReason,
                                      "region construction omitted a root");
    returnedValues.append(mapped->getResults().begin(),
                          mapped->getResults().end());
  }
  builder.create<mlir::func::ReturnOp>(sourceRoot->getLoc(), returnedValues);
  functionalArgumentCount = body->getNumArguments();
  return function;
}

mlir::FailureOr<mlir::func::FuncOp> buildRootFunction(
    mlir::Block &destination, mlir::Operation *sourceRoot,
    uint32_t structuredNodeId,
    const llvm::DenseSet<mlir::Operation *> &structuredOperations,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    unsigned &functionalArgumentCount) {
  llvm::DenseSet<mlir::Operation *> coupledOperations{sourceRoot};
  llvm::SmallVector<StructuredOperationNodeMapping, 1> sourceNodes{
      {sourceRoot, structuredNodeId}};
  std::string name =
      (llvm::Twine("execute_node_") + llvm::Twine(structuredNodeId)).str();
  return buildRegionFunction(destination, sourceRoot, {sourceRoot}, sourceNodes,
                             structuredOperations, coupledOperations, name,
                             failureReason, operationNodes,
                             functionalArgumentCount);
}

mlir::FailureOr<mlir::func::FuncOp> buildCoupledRootFunction(
    mlir::Block &destination, llvm::ArrayRef<mlir::Operation *> sourceRoots,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    llvm::ArrayRef<uint32_t> coupledNodeIds, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    unsigned &functionalArgumentCount) {
  if (sourceRoots.empty() || coupledNodeIds.size() < 2)
    return fail<mlir::func::FuncOp>(
        failureReason, "coupled region requires several nodes and one sink");
  llvm::DenseSet<mlir::Operation *> structuredOperations;
  llvm::DenseSet<mlir::Operation *> coupledOperations;
  for (const StructuredOperationNodeMapping &mapping : sourceOperationNodes) {
    if (!mapping.operation ||
        !structuredOperations.insert(mapping.operation).second)
      return fail<mlir::func::FuncOp>(
          failureReason, "coupled region has a malformed node mapping");
    if (llvm::is_contained(coupledNodeIds, mapping.structuredNodeId))
      coupledOperations.insert(mapping.operation);
  }
  if (coupledOperations.size() != coupledNodeIds.size())
    return fail<mlir::func::FuncOp>(
        failureReason, "coupled region omitted one selected structured node");
  const uint32_t firstNode = *llvm::min_element(coupledNodeIds);
  std::string name =
      (llvm::Twine("execute_group_") + llvm::Twine(firstNode)).str();
  return buildRegionFunction(destination, sourceRoots.front(), sourceRoots,
                             sourceOperationNodes, structuredOperations,
                             coupledOperations, name, failureReason,
                             operationNodes, functionalArgumentCount);
}

mlir::LogicalResult materializeRootIteratorShard(
    mlir::func::FuncOp function, uint32_t structuredNodeId,
    llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
    unsigned functionalArgumentCount,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason) {
  if (offsets.size() != sizes.size() ||
      llvm::any_of(sizes, [](int64_t size) { return size <= 0; }))
    return failResult(failureReason,
                      "single-root iterator shard is empty or malformed");
  auto rootMapping = llvm::find_if(
      operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
        return mapping.structuredNodeId == structuredNodeId;
      });
  if (rootMapping == operationNodes.end() || !rootMapping->operation)
    return failResult(failureReason,
                      "single-root function has no structured root mapping");
  mlir::Operation *root = rootMapping->operation;
  if (mlir::failed(appendTileOutputDestinations(function, failureReason)))
    return mlir::failure();
  TensorProgramScope scope(function, functionalArgumentCount);
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  for (auto [offset, size] : llvm::zip_equal(offsets, sizes)) {
    mixedOffsets.push_back(builder.getIndexAttr(offset));
    mixedSizes.push_back(builder.getIndexAttr(size));
  }
  mlir::FailureOr<StructuredIterationTile> tile =
      materializeStructuredIterationTile(root, builder, mixedOffsets,
                                         mixedSizes, failureReason);
  if (mlir::failed(tile))
    return mlir::failure();

  llvm::SmallVector<mlir::LoopLikeOpInterface, 0> loops;
  for (mlir::Operation *tiledOperation : tile->operations) {
    recordStructuredOperationNodeMaterialization(root, tiledOperation,
                                                 &operationNodes);
    if (mlir::failed(fuseCandidateProducerSlices(
            tiledOperation, root, scope, loops,
            /*operationTemporalTiles=*/{}, builder.getListener(), failureReason,
            &operationNodes)))
      return mlir::failure();
  }

  auto returnOp = scope.getReturn();
  if (returnOp.getNumOperands() != tile->values.size())
    return failResult(
        failureReason,
        "single-root tile result count changed during materialization");
  builder.setInsertionPoint(returnOp);
  for (unsigned resultNumber = 0; resultNumber < tile->values.size();
       ++resultNumber) {
    mlir::FailureOr<mlir::Value> destination =
        getCandidateOutputBoundary(scope, resultNumber, failureReason);
    if (mlir::failed(destination))
      return mlir::failure();
    auto destinationType =
        mlir::dyn_cast<mlir::RankedTensorType>(destination->getType());
    if (!destinationType || tile->resultOffsets[resultNumber].size() !=
                                static_cast<size_t>(destinationType.getRank()))
      return failResult(failureReason,
                        "single-root result boundary rank is inconsistent");
    llvm::SmallVector<mlir::OpFoldResult, 4> strides(destinationType.getRank(),
                                                     builder.getIndexAttr(1));
    auto inserted = builder.create<mlir::tensor::InsertSliceOp>(
        root->getLoc(), tile->values[resultNumber], *destination,
        tile->resultOffsets[resultNumber], tile->resultSizes[resultNumber],
        strides);
    returnOp->setOperand(resultNumber, inserted.getResult());
  }

  eraseDeadCandidateSupportClosure(scope);
  llvm::DenseSet<mlir::Operation *> live;
  function.walk([&](mlir::Operation *operation) { live.insert(operation); });
  llvm::erase_if(operationNodes, [&](const auto &mapping) {
    return !mapping.operation || !live.contains(mapping.operation);
  });
  return mlir::success();
}

mlir::LogicalResult materializeTemporalRootIteratorShard(
    mlir::func::FuncOp function, uint32_t structuredNodeId,
    llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
    const StructuredNodeTemporalTile &selectedTemporal,
    unsigned functionalArgumentCount,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason) {
  if (selectedTemporal.structuredNodeId != structuredNodeId ||
      offsets.size() != sizes.size() ||
      sizes.size() != selectedTemporal.iteratorTileSizes.size() ||
      llvm::any_of(sizes, [](int64_t size) { return size <= 0; }))
    return failResult(failureReason, "single-root temporal shard is malformed");
  auto rootMapping = llvm::find_if(
      operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
        return mapping.structuredNodeId == structuredNodeId;
      });
  if (rootMapping == operationNodes.end() || !rootMapping->operation)
    return failResult(
        failureReason,
        "single-root temporal shard has no structured root mapping");
  if (mlir::failed(appendTileOutputDestinations(function, failureReason)))
    return mlir::failure();

  mlir::Operation *root = rootMapping->operation;
  StructuredOpTemporalTile temporal{root, selectedTemporal.iteratorTileSizes,
                                    selectedTemporal.waveLoopOrder};
  TensorProgramScope scope(function, functionalArgumentCount);
  mlir::func::ReturnOp returnOp = scope.getReturn();
  llvm::SmallVector<mlir::Value, 2> destinations;
  destinations.reserve(returnOp.getNumOperands());
  for (unsigned result = 0; result < returnOp.getNumOperands(); ++result) {
    mlir::FailureOr<mlir::Value> destination =
        getCandidateOutputBoundary(scope, result, failureReason);
    if (mlir::failed(destination))
      return mlir::failure();
    destinations.push_back(*destination);
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> traversed =
      materializeTemporalRegionTraversal(root, scope, offsets, sizes, temporal,
                                         destinations, operationNodes,
                                         failureReason);
  if (mlir::failed(traversed) || traversed->size() != returnOp.getNumOperands())
    return mlir::failure();
  for (auto [result, value] : llvm::enumerate(*traversed))
    returnOp->setOperand(result, value);

  eraseDeadCandidateSupportClosure(scope);
  retainLiveOperationNodes(function, operationNodes);
  return mlir::success();
}

void retainLiveOperationNodes(
    mlir::func::FuncOp function,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes) {
  llvm::DenseSet<mlir::Operation *> live;
  function.walk([&](mlir::Operation *operation) { live.insert(operation); });
  llvm::erase_if(operationNodes, [&](const auto &mapping) {
    return !mapping.operation || !live.contains(mapping.operation);
  });
}

mlir::LogicalResult materializePartialReductionShard(
    mlir::func::FuncOp function, uint32_t structuredNodeId,
    llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
    unsigned &functionalArgumentCount,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason) {
  auto rootMapping = llvm::find_if(
      operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
        return mapping.structuredNodeId == structuredNodeId;
      });
  if (rootMapping == operationNodes.end() || !rootMapping->operation ||
      offsets.size() != sizes.size() || offsets.empty())
    return failResult(
        failureReason,
        "partial-reduction shard has no mapped root or iterator rectangle");
  mlir::Operation *root = rootMapping->operation;
  TensorProgramScope scope(function, functionalArgumentCount);
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  for (auto [offset, size] : llvm::zip_equal(offsets, sizes)) {
    if (size <= 0)
      return failResult(failureReason,
                        "partial-reduction shard has an empty iterator");
    mixedOffsets.push_back(builder.getIndexAttr(offset));
    mixedSizes.push_back(builder.getIndexAttr(size));
  }
  mlir::FailureOr<PartialReductionTileMaterialization> partial =
      materializePartialReductionTile(root, builder, mixedOffsets, mixedSizes,
                                      failureReason);
  if (mlir::failed(partial) ||
      partial->partialValues.size() != root->getNumResults())
    return mlir::failure();

  llvm::SmallVector<mlir::LoopLikeOpInterface, 0> loops;
  for (mlir::Operation *partialOperation : partial->partialOperations) {
    recordStructuredOperationNodeMaterialization(root, partialOperation,
                                                 &operationNodes);
    if (mlir::failed(fuseCandidateProducerSlices(
            partialOperation, root, scope, loops,
            /*operationTemporalTiles=*/{}, builder.getListener(), failureReason,
            &operationNodes)))
      return mlir::failure();
  }

  // Contribution regions return neutral-initialized partial tensors. The
  // original DPS init is consumed exactly once by the merge region below; it
  // must not be folded independently into every spatial contribution.
  for (mlir::Operation *merge : llvm::reverse(partial->mergeOperations))
    if (merge->use_empty())
      merge->erase();
  mlir::func::ReturnOp oldReturn = scope.getReturn();
  builder.setInsertionPoint(oldReturn);
  builder.create<mlir::func::ReturnOp>(oldReturn.getLoc(),
                                       partial->partialValues);
  oldReturn.erase();
  llvm::SmallVector<mlir::Type, 2> resultTypes;
  for (mlir::Value value : partial->partialValues)
    resultTypes.push_back(value.getType());
  function.setFunctionType(mlir::FunctionType::get(
      function.getContext(), function.getArgumentTypes(), resultTypes));

  eraseDeadCandidateSupportClosure(
      TensorProgramScope(function, functionalArgumentCount));
  retainLiveOperationNodes(function, operationNodes);
  if (mlir::failed(appendTileOutputDestinations(function, failureReason)))
    return mlir::failure();
  return mlir::success();
}

mlir::FailureOr<RootFragment> materializeRootFragment(
    TileModuleOp tileOwner,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    const StructuredNodeIterationShard &shard,
    const StructuredNodeTemporalTile *temporal,
    const StructuredNodePhysicalRepresentation *representation,
    std::string *failureReason) {
  auto requested = llvm::find_if(
      sourceOperationNodes, [&](const StructuredOperationNodeMapping &mapping) {
        return mapping.structuredNodeId == shard.structuredNodeId;
      });
  if (requested == sourceOperationNodes.end() || !requested->operation)
    return fail<RootFragment>(failureReason,
                              "node shard references an unknown root");
  llvm::DenseSet<mlir::Operation *> structuredOperations;
  for (const StructuredOperationNodeMapping &mapping : sourceOperationNodes)
    if (!mapping.operation ||
        !structuredOperations.insert(mapping.operation).second)
      return fail<RootFragment>(
          failureReason,
          "structured operation mapping contains a null or duplicate root");

  RootFragment result;
  llvm::SmallVector<StructuredOperationNodeMapping, 4> operationNodes;
  unsigned functionalArgumentCount = 0;
  mlir::FailureOr<mlir::func::FuncOp> function =
      buildRootFunction(tileOwner.getBody().front(), requested->operation,
                        shard.structuredNodeId, structuredOperations,
                        failureReason, operationNodes, functionalArgumentCount);
  if (mlir::failed(function))
    return mlir::failure();
  const bool hasTemporalWaves =
      temporal &&
      llvm::any_of(llvm::zip_equal(shard.sizes, temporal->iteratorTileSizes),
                   [](auto values) {
                     auto [extent, tile] = values;
                     return tile < extent;
                   });
  mlir::LogicalResult materialized = mlir::failure();
  if (shard.role == StructuredNodeIterationShardRole::Complete) {
    materialized =
        hasTemporalWaves
            ? materializeTemporalRootIteratorShard(
                  *function, shard.structuredNodeId, shard.offsets, shard.sizes,
                  *temporal, functionalArgumentCount, operationNodes,
                  failureReason)
            : materializeRootIteratorShard(
                  *function, shard.structuredNodeId, shard.offsets, shard.sizes,
                  functionalArgumentCount, operationNodes, failureReason);
  } else if (hasTemporalWaves) {
    materialized = materializeTemporalPartialReductionShard(
        *function, shard.structuredNodeId, shard.offsets, shard.sizes,
        *temporal, functionalArgumentCount, operationNodes, failureReason);
  } else {
    materialized = materializePartialReductionShard(
        *function, shard.structuredNodeId, shard.offsets, shard.sizes,
        functionalArgumentCount, operationNodes, failureReason);
  }
  if (mlir::failed(materialized))
    return mlir::failure();

  TileRegionEmissionRelations emissionRelations;
  if (mlir::failed(convertTensorProgramToTileRegionFunctionInPlace(
          *function, functionalArgumentCount,
          /*currentLogicalPartition=*/0, failureReason,
          /*suppressDiagnostics=*/true, /*verifyResult=*/true,
          /*populateFallbackFailureReason=*/true,
          /*peerEndpoints=*/{}, /*selectedDDRStages=*/{}, &emissionRelations,
          operationNodes, /*requireOneStructuredRootPerRegion=*/true,
          representation
              ? llvm::ArrayRef<StructuredNodePhysicalRepresentation>(
                    representation, 1)
              : llvm::ArrayRef<StructuredNodePhysicalRepresentation>{})))
    return mlir::failure();
  result.function = *function;

  unsigned regionCount = 0;
  result.function.walk([&](TileRegionOp region) {
    if (!region->getParentOfType<TileRegionOp>())
      ++regionCount;
  });
  if (regionCount != 1)
    return fail<RootFragment>(
        failureReason,
        "single-root conversion did not produce exactly one outer region");
  llvm::DenseSet<uint32_t> emittedNodes;
  for (const StructuredOperationEmissionRelation &relation :
       emissionRelations.materializedBuffers.operationEmissions)
    if (relation.operation)
      emittedNodes.insert(relation.structuredNodeId);
  if (emittedNodes.size() != 1 ||
      !emittedNodes.contains(shard.structuredNodeId)) {
    std::string detail;
    llvm::raw_string_ostream diagnostic(detail);
    diagnostic << "single-root conversion requested node "
               << shard.structuredNodeId << " but emitted nodes [";
    llvm::interleaveComma(emittedNodes, diagnostic);
    diagnostic << ']';
    return fail<RootFragment>(failureReason, diagnostic.str());
  }
  result.relations = std::move(emissionRelations.materializedBuffers);
  return result;
}

} // namespace wafer::tensor_program_to_tile_region
