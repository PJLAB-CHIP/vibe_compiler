//===- TemporalTileMaterialization.cpp - Root temporal traversal ------===//

#include "Internal.h"
#include "TemporalWaveLoop.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

/// Give every pure ranked-tensor function result a structured output anchor.
/// Source programs routinely return a shape view or an insert/extract update
/// rather than the last compute op itself.  Those values are still ordinary
/// SSA dataflow and must not narrow structured-DAG search to workloads whose
/// return happens to be a DPS op.  A generic identity anchor exposes the result
/// domain through TilingInterface; producer fusion must then prove the exact
/// tile relation through the original view/update chain.  Failure to fuse
/// remains a candidate legality failure rather than a full-tensor fallback.
static mlir::LogicalResult materializeCandidateOutputAnchors(
    TensorProgramScope scope, llvm::ArrayRef<SpatialOutputShard> outputShards,
    std::string *failureReason) {
  auto returnOp =
      mlir::dyn_cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  if (!returnOp)
    return mlir::failure();

  mlir::OpBuilder builder(returnOp);
  for (auto [outputIndex, returned] : llvm::enumerate(returnOp.getOperands())) {
    mlir::Operation *definition = returned.getDefiningOp();
    if (classifyStructuredRoot(definition) == StructuredRootCapability::Tiled)
      continue;
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        scope, static_cast<unsigned>(outputIndex), failureReason);
    if (mlir::failed(outputBoundary))
      return mlir::failure();
    const bool selected =
        llvm::any_of(outputShards, [&](const SpatialOutputShard &shard) {
          return shard.outputIndex == outputIndex;
        });
    // A narrow final-region source represents every sibling result by its
    // private scheduling destination.  Give that no-store path the same
    // temporary typed root shape expected by the common traversal code; the
    // omitted-output branch below immediately restores the boundary and the
    // dead anchor never reaches TileRegion IR.
    const bool omittedBoundary = !selected && returned == *outputBoundary;
    auto toTensor =
        mlir::dyn_cast_or_null<mlir::bufferization::ToTensorOp>(definition);
    auto compilerOwnedAllocation =
        toTensor ? toTensor.getMemref().getDefiningOp<mlir::memref::AllocOp>()
                 : mlir::memref::AllocOp{};
    const bool compilerOwnedDDRBoundary =
        compilerOwnedAllocation &&
        isWaferDDRMemRefType(compilerOwnedAllocation.getType()) &&
        compilerOwnedAllocation.getDynamicSizes().empty() &&
        compilerOwnedAllocation.getSymbolOperands().empty();
    if ((!definition && !omittedBoundary) ||
        (definition && definition->getBlock() != &scope.getBody()) ||
        (definition && !mlir::isMemoryEffectFree(definition) &&
         !compilerOwnedDDRBoundary)) {
      setFailureReason(failureReason,
                       "candidate traversal root is unsupported");
      return mlir::failure();
    }
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(returned.getType());
    if (!resultType || !resultType.hasStaticShape()) {
      setFailureReason(
          failureReason,
          "candidate output anchor requires a static ranked tensor");
      return mlir::failure();
    }
    mlir::AffineMap identity = mlir::AffineMap::getMultiDimIdentityMap(
        resultType.getRank(), builder.getContext());
    llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
        resultType.getRank(), mlir::utils::IteratorType::parallel);
    auto anchor = builder.create<mlir::linalg::GenericOp>(
        returned.getLoc(), mlir::TypeRange{returned.getType()},
        mlir::ValueRange{returned}, mlir::ValueRange{*outputBoundary},
        llvm::ArrayRef<mlir::AffineMap>{identity, identity}, iteratorTypes,
        [&](mlir::OpBuilder &bodyBuilder, mlir::Location loc,
            mlir::ValueRange arguments) {
          bodyBuilder.create<mlir::linalg::YieldOp>(loc, arguments.front());
        });
    returnOp->setOperand(outputIndex, anchor.getResult(0));
  }
  return mlir::success();
}

mlir::Value
insertCandidateRootTile(mlir::Operation *root, mlir::Value tileValue,
                        mlir::Value outputDestination,
                        llvm::ArrayRef<int64_t> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes) {
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  offsets.reserve(candidateTileOffsets.size());
  for (int64_t offset : candidateTileOffsets)
    offsets.push_back(builder.getIndexAttr(offset));
  return insertCandidateRootTile(builder, root->getLoc(), tileValue,
                                 outputDestination, offsets,
                                 candidateTileSizes);
}

mlir::Value
insertCandidateRootTile(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value tileValue, mlir::Value outputDestination,
                        llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes) {
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  sizes.reserve(candidateTileSizes.size());
  strides.reserve(candidateTileSizes.size());
  for (int64_t size : candidateTileSizes) {
    sizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }

  auto inserted = builder.create<mlir::tensor::InsertSliceOp>(
      loc, tileValue, outputDestination, candidateTileOffsets, sizes, strides);
  return inserted.getResult();
}

mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>>
collectCandidateRoots(TensorProgramScope scope, bool rejectProducerChains,
                      std::string *failureReason) {
  auto returnOp =
      mlir::dyn_cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  if (!returnOp || returnOp.getOperands().empty()) {
    setFailureReason(
        failureReason,
        "candidate tile materialization requires function results");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Operation *, 4> roots;
  for (auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
    mlir::Operation *rootOperation = value.getDefiningOp();
    StructuredRootCapability capability = classifyStructuredRoot(rootOperation);
    if (capability == StructuredRootCapability::FullTraversalOnly) {
      setFailureReason(failureReason,
                       "candidate traversal root supports full traversal only");
      return mlir::failure();
    }
    if (capability == StructuredRootCapability::Unsupported) {
      setFailureReason(failureReason,
                       "candidate traversal root is unsupported");
      return mlir::failure();
    }
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(rootOperation);
    if (rootOperation->getBlock() != &scope.getBody() ||
        rootOperation->getNumResults() != 1 || !dps ||
        dps.getNumDpsInits() != 1 || rootOperation->getResult(0) != value) {
      setFailureReason(failureReason,
                       "candidate multi-output coverage requires "
                       "single-result yielded roots");
      return mlir::failure();
    }

    if (rejectProducerChains) {
      for (mlir::Value input : dps.getDpsInputs()) {
        if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
          continue;
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(input);
        if (blockArg && blockArg.getOwner() == &scope.getBody())
          continue;
        // Static tensor reshapes are shape-only views.  Keeping them in the
        // The scheduling scope does not introduce an independently tiled
        // producer: every
        // root tile still slices the same canonical linear element sequence,
        // and the normal tile-region reshape lowering decides whether the
        // physical layout can alias or must be materialized.  Other producer
        // chains remain rejected because complete traversal does not yet
        // carry their intermediate tensors across output tiles.
        auto isStaticShapeOnlyViewChain = [&](mlir::Value value) {
          mlir::Value current = value;
          while (mlir::Operation *def = current.getDefiningOp()) {
            if (def->getBlock() != &scope.getBody())
              return false;
            if (auto expand =
                    mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(def)) {
              current = expand.getSrc();
              continue;
            }
            if (auto collapse =
                    mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(def)) {
              current = collapse.getSrc();
              continue;
            }
            return false;
          }
          auto sourceArg = mlir::dyn_cast<mlir::BlockArgument>(current);
          return sourceArg && sourceArg.getOwner() == &scope.getBody();
        };
        if (isStaticShapeOnlyViewChain(input))
          continue;
        setFailureReason(
            failureReason,
            "complete candidate traversal does not support tensor producer "
            "chains");
        return mlir::failure();
      }

      mlir::Value init = dps.getDpsInits().front();
      bool hasDirectOutputInit = isTensorProgramOutputBoundary(
          scope, init, static_cast<unsigned>(index));
      auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(rootOperation);
      bool hasReduction = linalg && !getReductionLoopDims(linalg).empty();
      if (!hasDirectOutputInit &&
          (!hasReduction || !init.getDefiningOp<mlir::linalg::FillOp>())) {
        setFailureReason(
            failureReason,
            "complete candidate traversal does not support output producer "
            "chains");
        return mlir::failure();
      }
    }
    roots.push_back(rootOperation);
  }

  return roots;
}

/// Materializes one output's selected spatial shard as a compact temporal
/// traversal.  The first full tile is a finite prologue, remaining full-size
/// steady tiles live in structured scf.for loops, and every non-divisible
/// dimension contributes one statically-shaped tail class.  The traversal
/// carries the private full-card scheduling destination through ordinary SSA,
/// so each leaf
/// writes exactly its own subview while the producer closure is fused into
/// that leaf from current TilingInterface relations.
static mlir::FailureOr<mlir::Value> materializeTemporalRootTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned outputIndex, llvm::ArrayRef<int64_t> shardOffsets,
    llvm::ArrayRef<int64_t> shardSizes,
    llvm::ArrayRef<int64_t> temporalTileSizes, unsigned dimension,
    mlir::Value output, llvm::SmallVectorImpl<mlir::OpFoldResult> &tileOffsets,
    llvm::SmallVectorImpl<int64_t> &tileSizes,
    llvm::SmallVectorImpl<mlir::LoopLikeOpInterface> &loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes) {
  if (dimension != 0 || !tileOffsets.empty() || !tileSizes.empty() ||
      !loops.empty()) {
    setFailureReason(failureReason,
                     "temporal output traversal requires one root entry");
    return mlir::failure();
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> traversed =
      materializeTemporalWaveLoopNest(
          builder, root->getLoc(), shardOffsets, shardSizes, temporalTileSizes,
          /*waveLoopOrder=*/{}, mlir::ValueRange(output),
          [&](mlir::OpBuilder &leafBuilder,
              llvm::ArrayRef<mlir::OpFoldResult> leafOffsets,
              llvm::ArrayRef<int64_t> leafSizes, mlir::ValueRange outputs,
              llvm::MutableArrayRef<mlir::LoopLikeOpInterface> leafLoops)
              -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> {
            mlir::FailureOr<mlir::Value> tile =
                materializeCandidateRootTileValue(
                    leafBuilder, scope, root, outputIndex, leafOffsets,
                    leafSizes, leafLoops, operationTemporalTiles, failureReason,
                    operationNodes);
            if (mlir::failed(tile) || outputs.size() != 1)
              return mlir::failure();
            return llvm::SmallVector<mlir::Value, 2>{insertCandidateRootTile(
                leafBuilder, root->getLoc(), *tile, outputs.front(),
                leafOffsets, leafSizes)};
          },
          failureReason);
  if (mlir::failed(traversed) || traversed->size() != 1)
    return mlir::failure();
  return traversed->front();
}

mlir::LogicalResult materializeCandidateOutputTileSlices(
    TensorProgramScope scope, llvm::ArrayRef<SpatialOutputShard> outputShards,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes,
    llvm::ArrayRef<mlir::Operation *> preservedOperations) {
  if (mlir::failed(materializeCandidateOutputAnchors(scope, outputShards,
                                                     failureReason)))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> roots =
      collectCandidateRoots(scope, /*rejectProducerChains=*/false,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();

  if (outputShards.empty()) {
    setFailureReason(failureReason,
                     "spatial output materialization requires a shard");
    return mlir::failure();
  }

  llvm::SmallVector<const SpatialOutputShard *, 4> selectedByOutput(
      roots->size(), nullptr);
  for (const SpatialOutputShard &shard : outputShards) {
    if (shard.outputIndex >= roots->size()) {
      setFailureReason(failureReason,
                       "spatial output shard index is outside function "
                       "results");
      return mlir::failure();
    }
    if (selectedByOutput[shard.outputIndex]) {
      setFailureReason(failureReason,
                       "spatial output shard index is duplicated");
      return mlir::failure();
    }
    if (shard.offsets.size() != shard.sizes.size() ||
        shard.sizes.size() != shard.temporalTileSizes.size() ||
        llvm::any_of(llvm::zip_equal(shard.offsets, shard.sizes,
                                     shard.temporalTileSizes),
                     [](auto values) {
                       auto [offset, size, temporalSize] = values;
                       return offset < 0 || size <= 0 || temporalSize <= 0 ||
                              temporalSize > size;
                     })) {
      setFailureReason(
          failureReason,
          "spatial output shard has an invalid temporal tile domain");
      return mlir::failure();
    }
    selectedByOutput[shard.outputIndex] = &shard;
  }

  mlir::Operation *partialOutputEffect = nullptr;
  llvm::DenseSet<mlir::Operation *> preservedEffectOwners(
      preservedOperations.begin(), preservedOperations.end());
  if (outputShards.size() != roots->size())
    for (mlir::Operation &operation : scope.getBody().without_terminator())
      if (!preservedEffectOwners.contains(&operation) &&
          !mlir::isMemoryEffectFree(&operation)) {
        partialOutputEffect = &operation;
        break;
      }
  if (partialOutputEffect) {
    std::string reason =
        "partial output spatial materialization requires a pure tensor "
        "program; first non-pure operation=";
    reason += partialOutputEffect->getName().getStringRef();
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  for (auto [index, root] : llvm::enumerate(*roots)) {
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        scope, static_cast<unsigned>(index), failureReason);
    if (mlir::failed(outputBoundary))
      return mlir::failure();
    const SpatialOutputShard *shard = selectedByOutput[index];
    if (!shard) {
      // This Tile does not own any domain of this observable result.  Keep the
      // full-card ABI result but make its actual body a typed no-store path.
      returnOp->setOperand(index, *outputBoundary);
      continue;
    }
    mlir::OpBuilder builder(root);
    llvm::SmallVector<mlir::OpFoldResult, 4> tileOffsets;
    llvm::SmallVector<int64_t, 4> tileSizes;
    llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
    mlir::FailureOr<mlir::Value> traversed = materializeTemporalRootTraversal(
        builder, scope, root, static_cast<unsigned>(index), shard->offsets,
        shard->sizes, shard->temporalTileSizes,
        /*dimension=*/0, *outputBoundary, tileOffsets, tileSizes, loops,
        operationTemporalTiles, failureReason, operationNodes);
    if (mlir::failed(traversed))
      return mlir::failure();
    returnOp->setOperand(index, *traversed);
  }

  // Replacing every function result leaves the original untiled roots and
  // their producer/consumer closure dead.  Let the ordinary effect-aware
  // reverse walk erase that closure in use-before-def order.  Erasing roots
  // directly is incorrect when one observable result is also the producer of
  // another observable result (or the same value is returned twice).
  eraseDeadCandidateSupportClosure(scope, preservedOperations);
  if (operationNodes) {
    llvm::DenseSet<mlir::Operation *> liveOperations;
    scope.getBody().getParentOp()->walk(
        [&](mlir::Operation *operation) { liveOperations.insert(operation); });
    llvm::erase_if(*operationNodes, [&](const auto &mapping) {
      return !liveOperations.contains(mapping.operation);
    });
  }
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
