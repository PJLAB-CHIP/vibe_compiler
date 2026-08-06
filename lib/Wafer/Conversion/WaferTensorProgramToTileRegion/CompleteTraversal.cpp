//===- CompleteTraversal.cpp - Complete candidate traversal -----------===//

#include "Internal.h"

#include <algorithm>
#include <iterator>
#include <limits>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

static mlir::FailureOr<uint64_t>
getCandidateOutputTileCount(mlir::RankedTensorType resultType,
                            llvm::ArrayRef<int64_t> candidateTileSizes,
                            std::string *failureReason) {
  if (candidateTileSizes.size() != static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "candidate tile rank does not match program result rank");
    return mlir::failure();
  }

  for (auto [bound, tileSize] :
       llvm::zip(resultType.getShape(), candidateTileSizes)) {
    if (mlir::ShapedType::isDynamic(bound)) {
      setFailureReason(failureReason,
                       "complete candidate traversal requires static result "
                       "shape");
      return mlir::failure();
    }
    if (bound <= 0 || tileSize <= 0 || tileSize > bound) {
      setFailureReason(
          failureReason,
          "complete candidate traversal tile size is outside result bounds");
      return mlir::failure();
    }
  }

  uint64_t tileCount = 0;
  switch (wafer::detail::checkedStaticTileProduct(
      resultType.getShape(), candidateTileSizes, tileCount)) {
  case wafer::detail::CheckedStaticTileProductStatus::Success:
    return tileCount;
  case wafer::detail::CheckedStaticTileProductStatus::Overflow:
    setFailureReason(
        failureReason,
        "complete candidate traversal output tile count is not representable");
    return mlir::failure();
  case wafer::detail::CheckedStaticTileProductStatus::InvalidInput:
    setFailureReason(
        failureReason,
        "complete candidate traversal has invalid static output ranges");
    return mlir::failure();
  }
  llvm_unreachable("unknown checked tile product status");
}

static bool haveEquivalentExternalSlices(mlir::tensor::ExtractSliceOp lhs,
                                         mlir::tensor::ExtractSliceOp rhs,
                                         TensorProgramScope scope) {
  auto lhsSource = mlir::dyn_cast<mlir::BlockArgument>(lhs.getSource());
  auto rhsSource = mlir::dyn_cast<mlir::BlockArgument>(rhs.getSource());
  if (!lhsSource || !rhsSource || lhsSource != rhsSource ||
      lhsSource.getOwner() != &scope.getBody())
    return false;
  return lhs.getType() == rhs.getType() &&
         llvm::equal(lhs.getMixedOffsets(), rhs.getMixedOffsets()) &&
         llvm::equal(lhs.getMixedSizes(), rhs.getMixedSizes()) &&
         llvm::equal(lhs.getMixedStrides(), rhs.getMixedStrides());
}

static void deduplicateExternalSlicesInBlock(mlir::Block &block,
                                             TensorProgramScope scope) {
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> canonicalSlices;
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> duplicateSlices;
  for (mlir::Operation &op : block.without_terminator()) {
    auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op);
    if (!slice)
      continue;
    auto existing = llvm::find_if(canonicalSlices, [&](auto candidate) {
      return haveEquivalentExternalSlices(candidate, slice, scope);
    });
    if (existing == canonicalSlices.end()) {
      canonicalSlices.push_back(slice);
      continue;
    }
    slice.getResult().replaceAllUsesWith(existing->getResult());
    duplicateSlices.push_back(slice);
  }
  for (mlir::tensor::ExtractSliceOp duplicate : duplicateSlices)
    duplicate->erase();
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
materializeLoopedRootsTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    llvm::ArrayRef<mlir::Operation *> roots,
    llvm::ArrayRef<unsigned> outputIndices, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> tileSizes,
    llvm::ArrayRef<int64_t> reductionTileSizes,
    CandidateTileTraversalKind traversalKind, unsigned dim,
    mlir::ValueRange outputs,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets,
    llvm::SmallVectorImpl<int64_t> &sizes,
    llvm::SmallVectorImpl<mlir::LoopLikeOpInterface> &loops,
    std::string *failureReason) {
  if (roots.size() != outputs.size() || roots.size() != outputIndices.size()) {
    setFailureReason(failureReason,
                     "shared traversal root/output count mismatch");
    return mlir::failure();
  }
  if (dim == shape.size()) {
    llvm::SmallVector<mlir::Value, 4> nextOutputs;
    nextOutputs.reserve(roots.size());
    for (auto [index, root, output] : llvm::enumerate(roots, outputs)) {
      mlir::FailureOr<mlir::Value> tile =
          traversalKind == CandidateTileTraversalKind::PartialReduction
              ? materializeCandidatePartialReductionRootTileValue(
                    builder, scope, root, offsets, sizes, reductionTileSizes,
                    loops, failureReason)
              : materializeCandidateRootTileValue(
                    builder, scope, root, outputIndices[index], offsets, sizes,
                    reductionTileSizes, loops, failureReason);
      if (mlir::failed(tile))
        return mlir::failure();
      nextOutputs.push_back(insertCandidateRootTile(
          builder, root->getLoc(), *tile, output, offsets, sizes));
    }
    deduplicateExternalSlicesInBlock(*builder.getInsertionBlock(), scope);
    return nextOutputs;
  }

  mlir::Location loc = roots.front()->getLoc();
  int64_t tailSize = shape[dim] % tileSizes[dim];
  int64_t mainEnd = shape[dim] - tailSize;

  llvm::SmallVector<mlir::Value, 4> currentOutputs(outputs.begin(),
                                                   outputs.end());

  if (mainEnd > 0) {
    auto lower = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto upper = builder.create<mlir::arith::ConstantIndexOp>(loc, mainEnd);
    auto step =
        builder.create<mlir::arith::ConstantIndexOp>(loc, tileSizes[dim]);
    auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step,
                                                 currentOutputs);

    offsets.push_back(loop.getInductionVar());
    sizes.push_back(tileSizes[dim]);
    loops.push_back(mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
    mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> next =
        materializeLoopedRootsTraversal(
            bodyBuilder, scope, roots, outputIndices, shape, tileSizes,
            reductionTileSizes, traversalKind, dim + 1,
            loop.getRegionIterArgs(), offsets, sizes, loops, failureReason);
    loops.pop_back();
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(next))
      return mlir::failure();

    bodyBuilder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange(*next));
    builder.setInsertionPointAfter(loop);
    currentOutputs.assign(loop.getResults().begin(), loop.getResults().end());
  }

  if (tailSize > 0) {
    offsets.push_back(builder.getIndexAttr(mainEnd));
    sizes.push_back(tailSize);
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> tail =
        materializeLoopedRootsTraversal(builder, scope, roots, outputIndices,
                                        shape, tileSizes, reductionTileSizes,
                                        traversalKind, dim + 1, currentOutputs,
                                        offsets, sizes, loops, failureReason);
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(tail))
      return mlir::failure();
    currentOutputs = std::move(*tail);
  }
  return currentOutputs;
}

struct OperandDrivenTraversalSeed {
  mlir::Operation *root = nullptr;
  unsigned operandNumber = 0;
  mlir::RankedTensorType type;
};

static bool isScalarInitializedTargetCompute(mlir::Operation *operation) {
  if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
                mlir::linalg::MatmulTransposeBOp, mlir::linalg::BatchMatmulOp,
                mlir::linalg::BatchMatmulTransposeAOp,
                mlir::linalg::BatchMatmulTransposeBOp>(operation))
    return true;
  auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(operation);
  return generic && !getReductionLoopDims(generic).empty();
}

static mlir::Value getConservativeTraversalDestination(mlir::Operation *root,
                                                       mlir::Value dpsInit) {
  if (!isScalarInitializedTargetCompute(root))
    return dpsInit;
  auto fill = dpsInit.getDefiningOp<mlir::linalg::FillOp>();
  if (!fill || fill.getDpsInits().size() != 1)
    return dpsInit;

  // Tile materialization still consumes the fill result to recover the exact
  // scalar initialization required by target GEMM/reduce.  The complete
  // traversal, however, writes every output tile and must carry the fill's
  // destination object rather than a full-shape value produced by the fill.
  return fill.getDpsInits().front();
}

static mlir::FailureOr<OperandDrivenTraversalSeed>
findOperandDrivenTraversalSeed(TensorProgramScope scope,
                               llvm::ArrayRef<mlir::Operation *> roots,
                               std::string *failureReason) {
  if (roots.size() != 1) {
    setFailureReason(
        failureReason,
        "operand-driven complete traversal requires one yielded consumer");
    return mlir::failure();
  }
  mlir::Operation *root = roots.front();
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(root);
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!tiling || !resultType || !resultType.hasStaticShape()) {
    setFailureReason(
        failureReason,
        "operand-driven complete traversal requires a static tiled consumer");
    return mlir::failure();
  }

  mlir::OpBuilder builder(root);
  for (auto [operandNumber, operand] : llvm::enumerate(root->getOperands())) {
    auto boundary = mlir::dyn_cast<mlir::BlockArgument>(operand);
    auto operandType =
        mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (!boundary || boundary.getOwner() != &scope.getBody() ||
        boundary.getArgNumber() >= scope.getInputCount() || !operandType ||
        !operandType.hasStaticShape() || operandType != resultType)
      continue;

    llvm::SmallVector<mlir::OpFoldResult, 4> offsets(operandType.getRank(),
                                                     builder.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    for (int64_t size : operandType.getShape())
      sizes.push_back(builder.getIndexAttr(size));

    std::string relationFailure;
    mlir::FailureOr<OperandTileIterationDomain> iteration =
        mapOperandTileToIterationDomain(root, builder,
                                        static_cast<unsigned>(operandNumber),
                                        offsets, sizes, &relationFailure);
    if (mlir::failed(iteration))
      continue;
    llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
    llvm::SmallVector<mlir::OpFoldResult> resultSizes;
    if (mlir::failed(tiling.getResultTilePosition(
            builder, /*resultNumber=*/0, iteration->offsets, iteration->sizes,
            resultOffsets, resultSizes)))
      continue;
    if (!llvm::equal(resultOffsets, offsets) ||
        !llvm::equal(resultSizes, sizes))
      continue;
    return OperandDrivenTraversalSeed{
        root, static_cast<unsigned>(operandNumber), operandType};
  }

  setFailureReason(
      failureReason,
      "operand-driven complete traversal found no exact boundary tile seed");
  return mlir::failure();
}

static mlir::FailureOr<mlir::Value> materializeLoopedOperandTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    const OperandDrivenTraversalSeed &seed, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> tileSizes, unsigned dim, mlir::Value output,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets,
    llvm::SmallVectorImpl<int64_t> &sizes,
    llvm::SmallVectorImpl<mlir::LoopLikeOpInterface> &loops,
    std::string *failureReason) {
  if (dim == shape.size()) {
    llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
    for (int64_t size : sizes)
      mixedSizes.push_back(builder.getIndexAttr(size));
    mlir::FailureOr<mlir::Value> tile =
        materializeCandidateOperandConsumerTileValue(
            builder, scope, seed.root, seed.operandNumber, offsets, mixedSizes,
            loops, failureReason);
    if (mlir::failed(tile))
      return mlir::failure();
    mlir::Value next = insertCandidateRootTile(builder, seed.root->getLoc(),
                                               *tile, output, offsets, sizes);
    deduplicateExternalSlicesInBlock(*builder.getInsertionBlock(), scope);
    return next;
  }

  mlir::Location loc = seed.root->getLoc();
  int64_t tailSize = shape[dim] % tileSizes[dim];
  int64_t mainEnd = shape[dim] - tailSize;
  mlir::Value currentOutput = output;

  if (mainEnd > 0) {
    auto lower = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto upper = builder.create<mlir::arith::ConstantIndexOp>(loc, mainEnd);
    auto step =
        builder.create<mlir::arith::ConstantIndexOp>(loc, tileSizes[dim]);
    auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step,
                                                 currentOutput);
    offsets.push_back(loop.getInductionVar());
    sizes.push_back(tileSizes[dim]);
    loops.push_back(mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
    mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
    mlir::FailureOr<mlir::Value> next = materializeLoopedOperandTraversal(
        bodyBuilder, scope, seed, shape, tileSizes, dim + 1,
        loop.getRegionIterArgs().front(), offsets, sizes, loops, failureReason);
    loops.pop_back();
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(next))
      return mlir::failure();
    bodyBuilder.create<mlir::scf::YieldOp>(loc, *next);
    builder.setInsertionPointAfter(loop);
    currentOutput = loop.getResult(0);
  }

  if (tailSize > 0) {
    offsets.push_back(builder.getIndexAttr(mainEnd));
    sizes.push_back(tailSize);
    mlir::FailureOr<mlir::Value> tail = materializeLoopedOperandTraversal(
        builder, scope, seed, shape, tileSizes, dim + 1, currentOutput, offsets,
        sizes, loops, failureReason);
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(tail))
      return mlir::failure();
    currentOutput = *tail;
  }
  return currentOutput;
}

static mlir::LogicalResult materializeCompleteOperandDrivenTraversal(
    TensorProgramScope scope, llvm::ArrayRef<mlir::Operation *> roots,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  if (!candidateReductionTileSizes.empty()) {
    setFailureReason(failureReason,
                     "operand-driven traversal does not invent a reduction "
                     "partition");
    return mlir::failure();
  }
  mlir::FailureOr<OperandDrivenTraversalSeed> seed =
      findOperandDrivenTraversalSeed(scope, roots, failureReason);
  if (mlir::failed(seed))
    return mlir::failure();
  if (mlir::failed(getCandidateOutputTileCount(seed->type, candidateTileSizes,
                                               failureReason)))
    return mlir::failure();

  mlir::FailureOr<mlir::Value> outputBoundary =
      getCandidateOutputBoundary(scope, /*outputIndex=*/0, failureReason);
  if (mlir::failed(outputBoundary))
    return mlir::failure();
  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  mlir::OpBuilder builder(seed->root);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
  mlir::FailureOr<mlir::Value> completeOutput =
      materializeLoopedOperandTraversal(
          builder, scope, *seed, seed->type.getShape(), candidateTileSizes,
          /*dim=*/0, *outputBoundary, offsets, sizes, loops, failureReason);
  if (mlir::failed(completeOutput))
    return mlir::failure();
  returnOp->setOperand(0, *completeOutput);
  seed->root->erase();
  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

mlir::LogicalResult materializeCompleteCandidateTraversal(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    CandidateTileTraversalKind traversalKind, std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> roots =
      collectCandidateRoots(scope, /*rejectProducerChains=*/false,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();
  if (traversalKind == CandidateTileTraversalKind::OperandDriven)
    return materializeCompleteOperandDrivenTraversal(
        scope, *roots, candidateTileSizes, candidateReductionTileSizes,
        failureReason);

  struct RootTraversalGroup {
    llvm::SmallVector<int64_t, 4> shape;
    llvm::SmallVector<mlir::Operation *, 4> roots;
    llvm::SmallVector<unsigned, 4> outputIndices;
    llvm::SmallVector<mlir::Value, 4> outputBoundaries;
  };
  llvm::SmallVector<RootTraversalGroup, 4> groups;

  bool hasReductionRoot = false;
  for (auto [outputIndex, root] : llvm::enumerate(*roots)) {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape()) {
      setFailureReason(failureReason,
                       "complete candidate traversal result is not a static "
                       "ranked tensor");
      return mlir::failure();
    }
    // Compact loops avoid eager instance expansion, but every traversal
    // domain must still be representable before subview construction.
    if (mlir::failed(getCandidateOutputTileCount(resultType, candidateTileSizes,
                                                 failureReason)))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        scope, static_cast<unsigned>(outputIndex), failureReason);
    if (mlir::failed(outputBoundary))
      return mlir::failure();
    auto group = llvm::find_if(groups, [&](const RootTraversalGroup &current) {
      return llvm::equal(current.shape, resultType.getShape());
    });
    if (group == groups.end()) {
      RootTraversalGroup next;
      next.shape.assign(resultType.getShape().begin(),
                        resultType.getShape().end());
      groups.push_back(std::move(next));
      group = std::prev(groups.end());
    }
    group->roots.push_back(root);
    group->outputIndices.push_back(static_cast<unsigned>(outputIndex));
    group->outputBoundaries.push_back(*outputBoundary);

    if (traversalKind == CandidateTileTraversalKind::PartialReduction) {
      mlir::FailureOr<mlir::linalg::LinalgOp> computeRoot =
          getCandidatePartialReductionComputeRoot(root, failureReason);
      if (mlir::failed(computeRoot))
        return mlir::failure();
      hasReductionRoot |= !getReductionLoopDims(*computeRoot).empty();
      if (!mlir::isa<mlir::PartialReductionOpInterface>(
              computeRoot->getOperation()) ||
          mlir::failed(verifyCandidateReductionSplitNumericLegality(
              *computeRoot, failureReason)))
        return mlir::failure();
      continue;
    }
    if (auto linalgRoot = mlir::dyn_cast<mlir::linalg::LinalgOp>(root))
      hasReductionRoot |= !getReductionLoopDims(linalgRoot).empty();
  }
  if (!candidateReductionTileSizes.empty() && !hasReductionRoot) {
    setFailureReason(failureReason,
                     "candidate reduction split requires a reduction root");
    return mlir::failure();
  }
  if (traversalKind == CandidateTileTraversalKind::PartialReduction &&
      !hasReductionRoot) {
    setFailureReason(failureReason,
                     "partial-reduction traversal requires a reduction root");
    return mlir::failure();
  }
  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  // Equal-shape roots share one traversal. Different result domains receive
  // independent traversals in the same actual residency program; recursively
  // tiled shared producers may therefore form bounded consumer-compatible
  // versions instead of forcing a full-shape spill or rejecting multi-root
  // programs outright.
  for (RootTraversalGroup &group : groups) {
    mlir::OpBuilder builder(group.roots.front());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<int64_t, 4> sizes;
    llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> completeOutputs =
        materializeLoopedRootsTraversal(
            builder, scope, group.roots, group.outputIndices, group.shape,
            candidateTileSizes, candidateReductionTileSizes, traversalKind,
            /*dim=*/0, group.outputBoundaries, offsets, sizes, loops,
            failureReason);
    if (mlir::failed(completeOutputs))
      return mlir::failure();
    for (auto [index, output] :
         llvm::zip(group.outputIndices, *completeOutputs))
      returnOp->setOperand(index, output);
  }
  for (mlir::Operation *root : *roots)
    root->erase();

  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

static mlir::LogicalResult materializeSeparatedCompleteRankTraversalsImpl(
    TensorProgramScope scope,
    std::optional<llvm::ArrayRef<int64_t>> candidateTileSizes,
    std::string *failureReason) {
  llvm::SmallVector<mlir::Operation *, 16> roots;
  for (mlir::Operation &operation : scope.getBody().without_terminator()) {
    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(operation)) {
      bool isReturned =
          llvm::any_of(fill->getUsers(), [](mlir::Operation *user) {
            return mlir::isa<mlir::func::ReturnOp>(user);
          });
      // Keep a non-terminal fill attached to its consumer so reduction and
      // overwrite lowering can prove the exact initialization scalar while
      // materializing that consumer's tile.
      if (!isReturned)
        continue;
    }
    CandidateTraversalRootCapability capability =
        classifyCandidateTraversalRoot(&operation);
    if (capability == CandidateTraversalRootCapability::Unsupported)
      continue;
    if (operation.getNumResults() != 1) {
      setFailureReason(
          failureReason,
          "conservative complete-rank baseline requires single-result "
          "structured roots");
      return mlir::failure();
    }
    roots.push_back(&operation);
  }
  if (roots.empty()) {
    setFailureReason(failureReason,
                     "conservative complete-rank baseline found no "
                     "structured roots");
    return mlir::failure();
  }

  // C1's mandatory baseline is deliberately spill-conservative: every
  // original tensor.empty that is not the public function output becomes an
  // explicit compiler-owned DDR tensor. Each structured root then traverses
  // into that destination before the next root is tiled. This keeps one
  // complete-rank decision clone and one residency region, but prevents an
  // unsupported producer-view relation from silently materializing a
  // full-shape SPM producer.
  llvm::SmallVector<mlir::tensor::EmptyOp, 16> emptyDestinations;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(operation))
      emptyDestinations.push_back(empty);
  for (mlir::tensor::EmptyOp empty : emptyDestinations) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(empty.getType());
    if (!tensorType || !tensorType.hasStaticShape()) {
      setFailureReason(
          failureReason,
          "conservative complete-rank spill requires static tensor.empty "
          "destinations");
      return mlir::failure();
    }
    auto ddrType = mlir::MemRefType::get(
        tensorType.getShape(), tensorType.getElementType(),
        mlir::MemRefLayoutAttrInterface{},
        wafer::MemoryAttr::get(scope.getContext(), wafer::MemorySpace::DDR,
                               wafer::MemLayout::Tensor));
    mlir::OpBuilder builder(empty);
    auto allocation =
        builder.create<mlir::memref::AllocOp>(empty.getLoc(), ddrType);
    auto tensor = builder.create<mlir::bufferization::ToTensorOp>(
        empty.getLoc(), allocation.getResult(), /*restrict=*/true,
        /*writable=*/true);
    empty.getResult().replaceAllUsesWith(tensor.getResult());
    empty.erase();
  }

  for (mlir::Operation *root : roots) {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        llvm::any_of(resultType.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      setFailureReason(failureReason,
                       "conservative complete-rank traversal requires "
                       "positive static result shapes");
      return mlir::failure();
    }
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(root);
    if (!dps || dps.getNumDpsInits() != 1) {
      setFailureReason(
          failureReason,
          "conservative complete-rank baseline requires one DPS destination");
      return mlir::failure();
    }

    // A typed collective that is explicitly classified as full-traversal-only
    // remains one complete tensor operation in this conservative clone. Its
    // DPS destination is already an explicit DDR boundary/allocation, and the
    // Tile body emitter materializes the matching typed communication op. Do
    // not invent a tile relation for an operation whose interface does not
    // provide one; coordinated collective tiling is a separate candidate
    // action rather than a baseline legality requirement.
    if (classifyCandidateTraversalRoot(root) ==
        CandidateTraversalRootCapability::FullTraversalOnly)
      continue;

    llvm::SmallVector<int64_t, 4> shape(resultType.getShape().begin(),
                                        resultType.getShape().end());
    llvm::SmallVector<int64_t, 4> tileSizes;
    if (candidateTileSizes)
      tileSizes.assign(candidateTileSizes->begin(), candidateTileSizes->end());
    else
      tileSizes.assign(shape.size(), 1);
    if (mlir::failed(
            getCandidateOutputTileCount(resultType, tileSizes, failureReason)))
      return mlir::failure();

    unsigned outputIndex = 0;
    if (auto argument =
            mlir::dyn_cast<mlir::BlockArgument>(dps.getDpsInits().front());
        argument && argument.getOwner() == &scope.getBody() &&
        argument.getArgNumber() >= scope.getInputCount())
      outputIndex = argument.getArgNumber() - scope.getInputCount();

    mlir::Value originalResult = root->getResult(0);
    mlir::Value outputDestination =
        getConservativeTraversalDestination(root, dps.getDpsInits().front());
    mlir::OpBuilder builder(root);
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<int64_t, 4> sizes;
    llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> completeOutputs =
        materializeLoopedRootsTraversal(
            builder, scope, llvm::ArrayRef<mlir::Operation *>{root},
            llvm::ArrayRef<unsigned>{outputIndex}, shape, tileSizes,
            /*reductionTileSizes=*/{}, CandidateTileTraversalKind::ResultDriven,
            /*dim=*/0, mlir::ValueRange{outputDestination}, offsets, sizes,
            loops, failureReason);
    if (mlir::failed(completeOutputs))
      return mlir::failure();
    if (completeOutputs->size() != 1) {
      setFailureReason(
          failureReason,
          "conservative complete-rank traversal produced invalid arity");
      return mlir::failure();
    }
    originalResult.replaceAllUsesWith(completeOutputs->front());
    root->erase();
    eraseDeadCandidateSupportClosure(scope);
  }

  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

mlir::LogicalResult
materializeConservativeCompleteRankTraversals(TensorProgramScope scope,
                                              std::string *failureReason) {
  return materializeSeparatedCompleteRankTraversalsImpl(scope, std::nullopt,
                                                        failureReason);
}

mlir::LogicalResult materializeSeparatedCompleteRankTraversals(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileSizes,
    std::string *failureReason) {
  return materializeSeparatedCompleteRankTraversalsImpl(
      scope, candidateTileSizes, failureReason);
}

} // namespace wafer::tensor_program_to_tile_region
