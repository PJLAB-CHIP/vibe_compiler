//===- TileMaterialization.cpp - Candidate root tile materialization -===//

#include "Internal.h"

#include "mlir/IR/PatternMatch.h"

#include <deque>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

static std::optional<ComputeReduceKind>
inferCandidateReduceKind(mlir::linalg::GenericOp generic,
                         std::string *failureReason) {
  if (generic.getRegionInputArgs().size() != 1 ||
      generic.getRegionOutputArgs().size() != 1) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires one input and one accumulator");
    return std::nullopt;
  }
  return matchExactReductionKind(generic.getRegionOutputArgs(), /*redPos=*/0,
                                 generic.getRegionInputArgs().front(),
                                 "candidate reduction split", failureReason);
}

static mlir::FailureOr<ComputeReduceKind>
getCandidateCombineKind(mlir::linalg::LinalgOp root,
                        std::string *failureReason) {
  if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
                mlir::linalg::MatmulTransposeBOp, mlir::linalg::BatchMatmulOp,
                mlir::linalg::BatchMatmulTransposeAOp,
                mlir::linalg::BatchMatmulTransposeBOp>(root.getOperation()))
    return ComputeReduceKind::Sum;
  if (auto generic =
          mlir::dyn_cast<mlir::linalg::GenericOp>(root.getOperation())) {
    std::optional<ComputeReduceKind> kind =
        inferCandidateReduceKind(generic, failureReason);
    if (!kind)
      return mlir::failure();
    return *kind;
  }

  setFailureReason(
      failureReason,
      "candidate reduction split requires matmul, batch_matmul, or generic "
      "reduction root");
  return mlir::failure();
}

static mlir::LogicalResult fuseCandidateProducerSlices(
    mlir::Operation *tiledConsumer, TensorProgramScope scope,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  // With a structured loop nest, a fused tile is created in a nested block and
  // therefore cannot be mistaken for another untiled scope producer.  The
  // direct single-tile API has no enclosing loop, so remember the finite set
  // of source producers that existed before fusion.  Otherwise slices
  // generated from a newly tiled clone can recursively fuse that clone again.
  llvm::DenseSet<mlir::Operation *> sourceProducers;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    if (&operation != tiledConsumer &&
        mlir::isa<mlir::TilingInterface>(&operation))
      sourceProducers.insert(&operation);

  std::deque<mlir::tensor::ExtractSliceOp> worklist;
  llvm::DenseSet<mlir::Operation *> seen;
  auto enqueueSlices = [&](llvm::ArrayRef<mlir::Operation *> operations,
                           mlir::Operation *downstreamProducer = nullptr) {
    for (mlir::Operation *operation : operations) {
      auto slice =
          mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(operation);
      mlir::Operation *sourceProducer =
          slice ? slice.getSource().getDefiningOp() : nullptr;
      // Generated operand slices must move strictly upstream through the
      // original source graph.  Some TilingInterface implementations also
      // report a destination/result slice whose source is the producer just
      // fused. Re-enqueuing that slice would clone the same producer forever
      // when there is no enclosing loop block to separate tiled clones.
      if (slice && downstreamProducer && sourceProducer &&
          sourceProducer->getBlock() == downstreamProducer->getBlock() &&
          !sourceProducer->isBeforeInBlock(downstreamProducer))
        continue;
      if (slice && seen.insert(operation).second)
        worklist.push_back(slice);
    }
  };

  llvm::SmallVector<mlir::Operation *, 4> initialSlices;
  for (mlir::Value operand : tiledConsumer->getOperands())
    if (auto slice = operand.getDefiningOp<mlir::tensor::ExtractSliceOp>())
      initialSlices.push_back(slice.getOperation());
  enqueueSlices(initialSlices);

  mlir::IRRewriter rewriter(scope.getContext());
  while (!worklist.empty()) {
    mlir::tensor::ExtractSliceOp slice = worklist.front();
    worklist.pop_front();
    auto producerResult = mlir::dyn_cast<mlir::OpResult>(slice.getSource());
    if (!producerResult ||
        producerResult.getOwner()->getBlock() != &scope.getBody() ||
        !sourceProducers.contains(producerResult.getOwner()))
      continue;
    // A destination slice generated while fusing a DPS producer may point at
    // tensor.empty.  Tiling that placeholder creates another equivalent
    // empty destination slice and can grow the worklist without making the
    // consumer tile more precise.  Empty destinations carry no dataflow to
    // fuse; leave their slice for ordinary dead-support cleanup.
    if (mlir::isa<mlir::tensor::EmptyOp>(producerResult.getOwner()))
      continue;
    if (!mlir::isa<mlir::TilingInterface>(producerResult.getOwner()))
      continue;

    std::optional<mlir::scf::SCFFuseProducerOfSliceResult> fused =
        mlir::scf::tileAndFuseProducerOfSlice(rewriter, slice, loops);
    if (!fused) {
      setFailureReason(failureReason, "candidate producer tile fusion failed");
      return mlir::failure();
    }
    enqueueSlices(fused->generatedSlices, producerResult.getOwner());
    if (slice->use_empty())
      rewriter.eraseOp(slice);
  }
  return mlir::success();
}

void eraseDeadCandidateSupportClosure(TensorProgramScope scope) {
  // Tiled producer fusion leaves the original untiled producer and its pure
  // destination/view support dead.  This cleanup is required by both the
  // complete traversal API and the direct single-tile API: otherwise the
  // latter lowers a dead full-shape producer and creates SPM pressure that is
  // unrelated to the requested tile.
  llvm::SmallVector<mlir::Operation *, 8> operations;
  for (mlir::Operation &op : scope.getBody().without_terminator())
    operations.push_back(&op);
  for (mlir::Operation *op : llvm::reverse(operations)) {
    if (!op->use_empty())
      continue;
    if (mlir::isa<mlir::linalg::LinalgOp, mlir::tensor::EmptyOp,
                  mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp,
                  mlir::tensor::ExtractSliceOp, mlir::arith::ConstantOp>(op))
      op->erase();
  }
}

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    TensorProgramScope scope, mlir::linalg::LinalgOp root, unsigned outputIndex,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  if (root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires one result");
    return mlir::failure();
  }
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType) {
    setFailureReason(failureReason,
                     "candidate tile materialization result is not ranked");
    return mlir::failure();
  }
  if (mlir::failed(validateCandidateTile(resultType, candidateTileOffsets,
                                         candidateTileSizes, failureReason)))
    return mlir::failure();

  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets;
  mixedOffsets.reserve(candidateTileOffsets.size());
  for (int64_t offset : candidateTileOffsets)
    mixedOffsets.push_back(builder.getIndexAttr(offset));
  llvm::SmallVector<mlir::LoopLikeOpInterface, 0> loops;
  return materializeCandidateRootTileValue(
      builder, scope, root, outputIndex, mixedOffsets, candidateTileSizes,
      candidateReductionTileSizes, loops, failureReason);
}

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    mlir::linalg::LinalgOp root, unsigned outputIndex,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason) {
  if (root.getNumDpsInits() != 1 || root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires one DPS output");
    return mlir::failure();
  }
  if (!root.hasOnlyProjectedPermutations()) {
    setFailureReason(
        failureReason,
        "candidate tile materialization requires permutation-only maps");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(root);
  bool hasDirectOutputInit = isTensorProgramOutputBoundary(
      scope, root.getDpsInits().front(), outputIndex);
  if (!hasDirectOutputInit && reductionLoopDims.empty()) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires direct output "
                     "boundary init for non-reduction roots");
    return mlir::failure();
  }

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType) {
    setFailureReason(failureReason,
                     "candidate tile materialization result is not ranked");
    return mlir::failure();
  }
  if (candidateTileOffsets.size() != candidateTileSizes.size() ||
      candidateTileSizes.size() != static_cast<size_t>(resultType.getRank()) ||
      llvm::any_of(candidateTileSizes,
                   [](int64_t size) { return size <= 0; })) {
    setFailureReason(failureReason,
                     "candidate tile materialization rank or size mismatch");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      root.getIndexingMapsArray();
  unsigned outputMapIndex = static_cast<unsigned>(root.getNumDpsInputs());
  if (outputMapIndex >= indexingMaps.size()) {
    setFailureReason(failureReason,
                     "candidate tile materialization missing output map");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 2> rootReductionTileSizes;
  if (!reductionLoopDims.empty())
    rootReductionTileSizes.assign(candidateReductionTileSizes.begin(),
                                  candidateReductionTileSizes.end());

  if (!rootReductionTileSizes.empty()) {
    if (reductionLoopDims.size() != 1) {
      setFailureReason(
          failureReason,
          "candidate reduction split requires exactly one reduction axis");
      return mlir::failure();
    }
  }

  mlir::FailureOr<llvm::SmallVector<ReductionChunk, 8>> reductionChunks =
      buildReductionChunks(root, rootReductionTileSizes, failureReason);
  if (mlir::failed(reductionChunks))
    return mlir::failure();
  if (reductionChunks->size() > 1) {
    if (mlir::failed(
            verifyCandidateReductionSplitNumericLegality(root, failureReason)))
      return mlir::failure();
    mlir::FailureOr<ComputeReduceKind> kind =
        getCandidateCombineKind(root, failureReason);
    if (mlir::failed(kind))
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 4> valuesToTile(root->operand_begin(),
                                                 root->operand_end());
  unsigned initOperandIndex = static_cast<unsigned>(root.getNumDpsInputs());
  mlir::Value accumulator;

  for (const ReductionChunk &chunk : *reductionChunks) {
    CandidateLoopTile loopTile;
    if (mlir::failed(buildCandidateLoopTile(
            builder, root.getLoc(), root, indexingMaps[outputMapIndex],
            candidateTileOffsets, candidateTileSizes, chunk.offsets,
            chunk.sizes, loopTile, failureReason)))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> tiledOperands =
        mlir::linalg::makeTiledShapes(builder, root.getLoc(), root,
                                      valuesToTile, loopTile.ivs,
                                      loopTile.tileSizes, loopTile.sizeBounds,
                                      /*omitPartialTileCheck=*/true);
    if (accumulator && initOperandIndex < tiledOperands.size()) {
      mlir::Operation *unusedInitSlice =
          tiledOperands[initOperandIndex].getDefiningOp();
      if (accumulator.getType() != tiledOperands[initOperandIndex].getType()) {
        setFailureReason(failureReason,
                         "candidate reduction split accumulator type mismatch");
        return mlir::failure();
      }
      tiledOperands[initOperandIndex] = accumulator;
      if (unusedInitSlice && unusedInitSlice->use_empty())
        unusedInitSlice->erase();
    }

    llvm::SmallVector<mlir::Type, 2> resultTypes =
        mlir::linalg::getTensorOutputTypes(root, tiledOperands);
    if (resultTypes.size() != 1) {
      setFailureReason(
          failureReason,
          "candidate tile materialization expected one tiled result type");
      return mlir::failure();
    }

    mlir::Operation *tiled =
        mlir::clone(builder, root.getOperation(), resultTypes, tiledOperands);
    auto tiledLinalg = mlir::cast<mlir::linalg::LinalgOp>(tiled);
    mlir::linalg::offsetIndices(builder, tiledLinalg, loopTile.loopOffsets);

    if (mlir::failed(
            fuseCandidateProducerSlices(tiled, scope, loops, failureReason)))
      return mlir::failure();

    builder.setInsertionPointAfter(tiled);
    accumulator = tiled->getResult(0);
  }

  if (!accumulator) {
    setFailureReason(failureReason,
                     "candidate tile materialization produced no tiled result");
    return mlir::failure();
  }
  return accumulator;
}

mlir::FailureOr<mlir::Value>
getCandidateOutputBoundary(TensorProgramScope scope, unsigned outputIndex,
                           std::string *failureReason) {
  unsigned inputCount = scope.getInputCount();
  mlir::Block &body = scope.getBody();
  if (inputCount + outputIndex >= body.getNumArguments()) {
    setFailureReason(failureReason,
                     "candidate tile materialization missing output boundary");
    return mlir::failure();
  }
  return body.getArgument(inputCount + outputIndex);
}

mlir::Value
insertCandidateRootTile(mlir::linalg::LinalgOp root, mlir::Value tileValue,
                        mlir::Value outputDestination,
                        llvm::ArrayRef<int64_t> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes) {
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  offsets.reserve(candidateTileOffsets.size());
  for (int64_t offset : candidateTileOffsets)
    offsets.push_back(builder.getIndexAttr(offset));
  return insertCandidateRootTile(builder, root.getLoc(), tileValue,
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

mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
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

  llvm::SmallVector<mlir::linalg::LinalgOp, 4> roots;
  llvm::DenseSet<mlir::Operation *> seenRoots;
  for (auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
    mlir::Operation *rootOperation = value.getDefiningOp();
    CandidateTraversalRootCapability capability =
        classifyCandidateTraversalRoot(rootOperation);
    if (capability == CandidateTraversalRootCapability::FullTraversalOnly) {
      setFailureReason(failureReason,
                       "candidate traversal root supports full traversal only");
      return mlir::failure();
    }
    if (capability == CandidateTraversalRootCapability::Unsupported) {
      setFailureReason(failureReason,
                       "candidate traversal root is unsupported");
      return mlir::failure();
    }
    auto root = mlir::cast<mlir::linalg::LinalgOp>(rootOperation);
    if (root->getBlock() != &scope.getBody() ||
        !seenRoots.insert(root.getOperation()).second ||
        root->getNumResults() != 1 || root.getNumDpsInits() != 1 ||
        root->getResult(0) != value) {
      setFailureReason(failureReason,
                       "candidate multi-output coverage requires distinct "
                       "single-result yielded roots");
      return mlir::failure();
    }

    unsigned matchingYieldUses = 0;
    for (mlir::OpOperand &use : value.getUses()) {
      if (use.getOwner() == returnOp.getOperation() &&
          use.getOperandNumber() == index) {
        ++matchingYieldUses;
        continue;
      }
      setFailureReason(failureReason, "candidate multi-output coverage "
                                      "requires independent yielded roots");
      return mlir::failure();
    }
    if (matchingYieldUses != 1) {
      setFailureReason(failureReason, "candidate multi-output coverage "
                                      "requires independent yielded roots");
      return mlir::failure();
    }

    if (rejectProducerChains) {
      for (mlir::Value input : root.getDpsInputs()) {
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

      mlir::Value init = root.getDpsInits().front();
      bool hasDirectOutputInit = isTensorProgramOutputBoundary(
          scope, init, static_cast<unsigned>(index));
      bool hasReduction = !getReductionLoopDims(root).empty();
      if (!hasDirectOutputInit &&
          (!hasReduction || !init.getDefiningOp<mlir::linalg::FillOp>())) {
        setFailureReason(
            failureReason,
            "complete candidate traversal does not support output producer "
            "chains");
        return mlir::failure();
      }
    }
    roots.push_back(root);
  }

  return roots;
}

mlir::LogicalResult materializeCandidateTileSlices(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      collectCandidateRoots(scope, /*rejectProducerChains=*/false,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();

  if (!candidateReductionTileSizes.empty() &&
      llvm::none_of(*roots, [](mlir::linalg::LinalgOp root) {
        return !getReductionLoopDims(root).empty();
      })) {
    setFailureReason(failureReason,
                     "candidate reduction split requires a reduction root");
    return mlir::failure();
  }

  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());

  llvm::SmallVector<mlir::Value, 4> insertedValues;
  for (auto [index, root] : llvm::enumerate(*roots)) {
    mlir::FailureOr<mlir::Value> tileValue = materializeCandidateRootTileValue(
        scope, root, static_cast<unsigned>(index), candidateTileOffsets,
        candidateTileSizes, candidateReductionTileSizes, failureReason);
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        scope, static_cast<unsigned>(index), failureReason);
    if (mlir::failed(tileValue) || mlir::failed(outputBoundary))
      return mlir::failure();
    insertedValues.push_back(
        insertCandidateRootTile(root, *tileValue, *outputBoundary,
                                candidateTileOffsets, candidateTileSizes));
  }

  for (auto [index, inserted] : llvm::enumerate(insertedValues))
    returnOp->setOperand(index, inserted);
  for (mlir::linalg::LinalgOp root : *roots)
    root->erase();
  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
