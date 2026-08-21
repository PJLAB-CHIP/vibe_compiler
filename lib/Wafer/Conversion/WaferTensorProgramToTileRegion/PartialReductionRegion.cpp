//===- PartialReductionRegion.cpp - Spatial reduction merge region ----===//

#include "SingleRootTileRegionInternal.h"

#include "Internal.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

namespace wafer::tensor_program_to_tile_region {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

} // namespace

mlir::FailureOr<RootFragment> materializeReductionMergeFragment(
    TileModuleOp tileOwner,
    llvm::ArrayRef<StructuredOperationNodeMapping> sourceOperationNodes,
    uint32_t structuredNodeId,
    const compiler::detail::ReductionGroupId &group,
    llvm::ArrayRef<const StructuredNodeIterationShard *> contributionShards,
    llvm::ArrayRef<mlir::func::FuncOp> contributionFunctions,
    const StructuredNodePhysicalRepresentation *representation,
    std::string *failureReason) {
  if (contributionShards.empty() ||
      contributionShards.size() != contributionFunctions.size())
    return fail<RootFragment>(
        failureReason,
        "partial-reduction merge requires every contribution function");
  auto requested = llvm::find_if(
      sourceOperationNodes, [&](const StructuredOperationNodeMapping &mapping) {
        return mapping.structuredNodeId == structuredNodeId;
      });
  if (requested == sourceOperationNodes.end() || !requested->operation)
    return fail<RootFragment>(failureReason,
                              "partial-reduction merge has no source root");
  llvm::DenseSet<mlir::Operation *> structuredOperations;
  for (const StructuredOperationNodeMapping &mapping : sourceOperationNodes)
    if (mapping.operation)
      structuredOperations.insert(mapping.operation);

  RootFragment result;
  llvm::SmallVector<StructuredOperationNodeMapping, 8> operationNodes;
  unsigned functionalArgumentCount = 0;
  mlir::FailureOr<mlir::func::FuncOp> function = buildRootFunction(
      tileOwner.getBody().front(), requested->operation, structuredNodeId,
      sourceOperationNodes, structuredOperations, failureReason,
      operationNodes, functionalArgumentCount, result.boundaries,
      result.results);
  if (mlir::failed(function))
    return mlir::failure();
  function->setName(
      (llvm::Twine("merge_node_") + llvm::Twine(structuredNodeId)).str());
  auto rootMapping = llvm::find_if(
      operationNodes, [&](const StructuredOperationNodeMapping &mapping) {
        return mapping.structuredNodeId == structuredNodeId;
      });
  mlir::Operation *root =
      rootMapping != operationNodes.end() ? rootMapping->operation : nullptr;
  auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(root);
  auto partial =
      mlir::dyn_cast_or_null<mlir::PartialReductionOpInterface>(root);
  if (!tiling || !partial)
    return fail<RootFragment>(
        failureReason, "partial-reduction merge requires TilingInterface and "
                       "PartialReductionOpInterface");

  mlir::Block &body = function->getBody().front();
  llvm::SmallVector<llvm::SmallVector<mlir::BlockArgument, 2>, 8>
      contributionArguments;
  for (mlir::func::FuncOp contribution : contributionFunctions) {
    if (!contribution || contribution.getNumResults() != root->getNumResults())
      return fail<RootFragment>(
          failureReason,
          "partial-reduction contribution result count is inconsistent");
    llvm::SmallVector<mlir::BlockArgument, 2> arguments;
    for (mlir::Type type : contribution.getResultTypes()) {
      const unsigned argumentIndex = function->getNumArguments();
      function->insertArgument(argumentIndex, type, mlir::DictionaryAttr{},
                               requested->operation->getLoc());
      arguments.push_back(body.getArgument(argumentIndex));
    }
    contributionArguments.push_back(std::move(arguments));
  }
  functionalArgumentCount = function->getNumArguments();

  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::Range, 4> domain = tiling.getIterationDomain(builder);
  llvm::SmallVector<mlir::OpFoldResult, 4> domainOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> domainSizes;
  llvm::SmallVector<int64_t, 4> staticOffsets;
  llvm::SmallVector<int64_t, 4> staticSizes;
  for (mlir::Range range : domain) {
    std::optional<int64_t> offset = mlir::getConstantIntValue(range.offset);
    std::optional<int64_t> size = mlir::getConstantIntValue(range.size);
    std::optional<int64_t> stride = mlir::getConstantIntValue(range.stride);
    if (!offset || !size || !stride || *size <= 0 || *stride != 1)
      return fail<RootFragment>(
          failureReason,
          "partial-reduction merge requires one static iteration domain");
    domainOffsets.push_back(builder.getIndexAttr(*offset));
    domainSizes.push_back(builder.getIndexAttr(*size));
    staticOffsets.push_back(*offset);
    staticSizes.push_back(*size);
  }
  llvm::SmallVector<int, 2> reductionDimensions;
  for (auto [dimension, iterator] :
       llvm::enumerate(tiling.getLoopIteratorTypes()))
    if (iterator == mlir::utils::IteratorType::reduction)
      reductionDimensions.push_back(static_cast<int>(dimension));
  if (reductionDimensions.empty())
    return fail<RootFragment>(failureReason,
                              "partial merge root has no reduction iterator");
  mlir::FailureOr<llvm::SmallVector<mlir::Value>> initial =
      partial.generateInitialTensorForPartialReduction(
          builder, root->getLoc(), domainSizes, reductionDimensions);
  if (mlir::failed(initial) || initial->size() != root->getNumResults())
    return fail<RootFragment>(
        failureReason,
        "partial merge cannot create complete neutral accumulator tensors");
  llvm::SmallVector<mlir::Value, 2> assembled(initial->begin(), initial->end());
  for (auto [shard, arguments] :
       llvm::zip_equal(contributionShards, contributionArguments)) {
    if (shard->offsets.size() != domain.size() ||
        shard->sizes.size() != domain.size())
      return fail<RootFragment>(
          failureReason,
          "partial contribution rank does not match the merge domain");
    for (unsigned resultNumber = 0; resultNumber < assembled.size();
         ++resultNumber) {
      auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
          arguments[resultNumber].getType());
      auto destinationType = mlir::dyn_cast<mlir::RankedTensorType>(
          assembled[resultNumber].getType());
      if (!sourceType || !destinationType ||
          sourceType.getRank() != static_cast<int64_t>(domain.size()) ||
          destinationType.getRank() != static_cast<int64_t>(domain.size()) ||
          !llvm::equal(sourceType.getShape(), shard->sizes))
        return fail<RootFragment>(
            failureReason,
            "partial contribution tensor does not preserve iterator shape");
      llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
      llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
      llvm::SmallVector<mlir::OpFoldResult, 4> strides;
      for (size_t dimension = 0; dimension < domain.size(); ++dimension) {
        const int64_t relative =
            shard->offsets[dimension] - staticOffsets[dimension];
        if (relative < 0 ||
            static_cast<__int128>(relative) + shard->sizes[dimension] >
                staticSizes[dimension])
          return fail<RootFragment>(
              failureReason,
              "partial contribution is outside the complete merge domain");
        offsets.push_back(builder.getIndexAttr(relative));
        sizes.push_back(builder.getIndexAttr(shard->sizes[dimension]));
        strides.push_back(builder.getIndexAttr(1));
      }
      assembled[resultNumber] =
          builder
              .create<mlir::tensor::InsertSliceOp>(
                  root->getLoc(), arguments[resultNumber],
                  assembled[resultNumber], offsets, sizes, strides)
              .getResult();
    }
  }

  mlir::FailureOr<mlir::MergeResult> merged = partial.mergeReductions(
      builder, root->getLoc(), assembled, reductionDimensions);
  if (mlir::failed(merged) || merged->mergeOps.empty() ||
      merged->replacements.size() != root->getNumResults())
    return fail<RootFragment>(failureReason,
                              "partial-reduction merge materialization failed");
  llvm::SmallVector<mlir::Value, 2> mergedValues;
  mlir::IRRewriter rewriter(builder);
  for (mlir::Operation *mergeOperation : merged->mergeOps) {
    auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(mergeOperation);
    if (!linalg)
      return fail<RootFragment>(failureReason,
                                "partial merge produced a non-Linalg op");
    mlir::Operation *materialized = mergeOperation;
    if (!mlir::isa<mlir::linalg::GenericOp>(mergeOperation)) {
      rewriter.setInsertionPoint(mergeOperation);
      mlir::FailureOr<mlir::linalg::GenericOp> generic =
          mlir::linalg::generalizeNamedOp(rewriter, linalg);
      if (mlir::failed(generic))
        return fail<RootFragment>(
            failureReason,
            "partial merge cannot be represented as linalg.generic");
      materialized = generic->getOperation();
    }
    recordStructuredOperationNodeMaterialization(root, materialized,
                                                 &operationNodes);
    mergedValues.append(materialized->getResults().begin(),
                        materialized->getResults().end());
  }
  if (mergedValues.size() != root->getNumResults())
    return fail<RootFragment>(failureReason,
                              "partial merge result count is inconsistent");

  mlir::func::ReturnOp oldReturn =
      mlir::cast<mlir::func::ReturnOp>(body.getTerminator());
  builder.setInsertionPoint(oldReturn);
  builder.create<mlir::func::ReturnOp>(oldReturn.getLoc(), mergedValues);
  oldReturn.erase();
  eraseDeadCandidateSupportClosure(
      TensorProgramScope(*function, functionalArgumentCount));
  retainLiveOperationNodes(*function, operationNodes);
  if (mlir::failed(appendTileOutputDestinations(*function, failureReason)))
    return mlir::failure();
  if (mlir::failed(bindFullResultsToOutputDestinations(
          *function, functionalArgumentCount, failureReason)))
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
  llvm::DenseSet<uint32_t> emittedNodes;
  for (const StructuredOperationEmissionRelation &relation :
       emissionRelations.materializedBuffers.operationEmissions)
    if (relation.operation)
      emittedNodes.insert(relation.structuredNodeId);
  if (regionCount != 1 || emittedNodes.size() != 1 ||
      !emittedNodes.contains(structuredNodeId))
    return fail<RootFragment>(
        failureReason, "partial merge did not produce one single-root region");
  result.relations = std::move(emissionRelations.materializedBuffers);
  TileRegionOp mergeRegion;
  result.function.walk([&](TileRegionOp region) {
    if (!region->getParentOfType<TileRegionOp>())
      mergeRegion = region;
  });
  if (!mergeRegion)
    return fail<RootFragment>(failureReason,
                              "partial merge has no materialized region");
  auto stripSubviews = [](mlir::Value value) {
    while (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>())
      value = subview.getSource();
    return value;
  };
  for (auto [shard, arguments] :
       llvm::zip_equal(contributionShards, contributionArguments)) {
    for (auto [resultIndex, argument] : llvm::enumerate(arguments)) {
      if (argument.getArgNumber() >=
          mergeRegion.getBody().front().getNumArguments())
        return fail<RootFragment>(
            failureReason,
            "partial merge contribution argument left region boundary");
      mlir::BlockArgument regionArgument =
          mergeRegion.getBody().front().getArgument(argument.getArgNumber());
      mlir::Value buffer;
      mergeRegion.walk([&](StorageLoadOp load) {
        if (!buffer && stripSubviews(load.getSource()) == regionArgument)
          buffer = load.getDest();
      });
      if (!buffer)
        return fail<RootFragment>(
            failureReason,
            "partial merge contribution has no exact input buffer");
      result.relations.partialReductionMergeInputs.push_back(
          {structuredNodeId, group, static_cast<unsigned>(resultIndex),
           shard->tile, buffer});
    }
  }
  return result;
}

} // namespace wafer::tensor_program_to_tile_region
