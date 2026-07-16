//===- CompleteTraversal.cpp - Complete candidate traversal -----------===//

#include "Internal.h"

#include <algorithm>
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
    llvm::ArrayRef<mlir::linalg::LinalgOp> roots, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> tileSizes,
    llvm::ArrayRef<int64_t> reductionTileSizes, unsigned dim,
    mlir::ValueRange outputs,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets,
    llvm::SmallVectorImpl<int64_t> &sizes,
    llvm::SmallVectorImpl<mlir::LoopLikeOpInterface> &loops,
    std::string *failureReason) {
  if (roots.size() != outputs.size()) {
    setFailureReason(failureReason,
                     "shared traversal root/output count mismatch");
    return mlir::failure();
  }
  if (dim == shape.size()) {
    llvm::SmallVector<mlir::Value, 4> nextOutputs;
    nextOutputs.reserve(roots.size());
    for (auto [index, root, output] : llvm::enumerate(roots, outputs)) {
      mlir::FailureOr<mlir::Value> tile = materializeCandidateRootTileValue(
          builder, scope, root, static_cast<unsigned>(index), offsets, sizes,
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
        materializeLoopedRootsTraversal(bodyBuilder, scope, roots, shape,
                                        tileSizes, reductionTileSizes, dim + 1,
                                        loop.getRegionIterArgs(), offsets,
                                        sizes, loops, failureReason);
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
        materializeLoopedRootsTraversal(
            builder, scope, roots, shape, tileSizes, reductionTileSizes,
            dim + 1, currentOutputs, offsets, sizes, loops, failureReason);
    sizes.pop_back();
    offsets.pop_back();
    if (mlir::failed(tail))
      return mlir::failure();
    currentOutputs = std::move(*tail);
  }
  return currentOutputs;
}

mlir::LogicalResult materializeCompleteCandidateTraversal(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      collectCandidateRoots(scope, /*rejectProducerChains=*/false,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();

  auto firstResultType = mlir::dyn_cast<mlir::RankedTensorType>(
      (*roots).front()->getResult(0).getType());
  if (!firstResultType) {
    setFailureReason(failureReason,
                     "complete candidate traversal result is not ranked");
    return mlir::failure();
  }

  // Compact loops avoid eager instance expansion, but the traversal domain
  // must still be representable.  Prove rank, bounds and checked tile-count
  // arithmetic before any subview/layout construction can observe it.
  if (mlir::failed(getCandidateOutputTileCount(
          firstResultType, candidateTileSizes, failureReason)))
    return mlir::failure();

  bool hasReductionRoot = false;
  for (mlir::linalg::LinalgOp root : *roots) {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!resultType || resultType.getShape() != firstResultType.getShape()) {
      setFailureReason(
          failureReason,
          "complete candidate traversal requires equal static result shapes");
      return mlir::failure();
    }
    hasReductionRoot |= !getReductionLoopDims(root).empty();
  }
  if (!candidateReductionTileSizes.empty() && !hasReductionRoot) {
    setFailureReason(failureReason,
                     "candidate reduction split requires a reduction root");
    return mlir::failure();
  }
  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  llvm::SmallVector<mlir::Value, 4> outputBoundaries;
  for (auto [index, root] : llvm::enumerate(*roots)) {
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        scope, static_cast<unsigned>(index), failureReason);
    if (mlir::failed(outputBoundary))
      return mlir::failure();
    outputBoundaries.push_back(*outputBoundary);
  }

  mlir::OpBuilder builder((*roots).front());
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  llvm::SmallVector<mlir::LoopLikeOpInterface, 4> loops;
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> completeOutputs =
      materializeLoopedRootsTraversal(
          builder, scope, *roots, firstResultType.getShape(),
          candidateTileSizes, candidateReductionTileSizes, /*dim=*/0,
          outputBoundaries, offsets, sizes, loops, failureReason);
  if (mlir::failed(completeOutputs))
    return mlir::failure();

  for (auto [index, output] : llvm::enumerate(*completeOutputs))
    returnOp->setOperand(index, output);
  for (mlir::linalg::LinalgOp root : *roots)
    root->erase();

  eraseDeadCandidateSupportClosure(scope);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
