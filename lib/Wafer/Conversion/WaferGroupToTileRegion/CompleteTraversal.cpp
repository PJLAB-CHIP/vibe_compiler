//===- CompleteTraversal.cpp - Complete candidate traversal -----------===//

#include "Internal.h"

#include <algorithm>
#include <limits>

using namespace wafer;

namespace wafer::group_to_tile_region {

struct CandidateOutputTile {
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
};

static mlir::FailureOr<uint64_t>
getCandidateOutputTileCount(mlir::RankedTensorType resultType,
                            llvm::ArrayRef<int64_t> candidateTileSizes,
                            std::string *failureReason) {
  if (candidateTileSizes.size() != static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "candidate tile rank does not match group result rank");
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

static mlir::LogicalResult checkCompleteCandidateExpansionBudget(
    llvm::ArrayRef<mlir::linalg::LinalgOp> roots,
    mlir::RankedTensorType resultType,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  mlir::FailureOr<uint64_t> outputTileCount = getCandidateOutputTileCount(
      resultType, candidateTileSizes, failureReason);
  if (mlir::failed(outputTileCount))
    return mlir::failure();

  llvm::SmallVector<uint64_t, 4> reductionChunkCounts;
  reductionChunkCounts.reserve(roots.size());
  for (mlir::linalg::LinalgOp root : roots) {
    if (candidateReductionTileSizes.empty() ||
        getReductionLoopDims(root).empty()) {
      reductionChunkCounts.push_back(1);
      continue;
    }
    mlir::FailureOr<uint64_t> chunkCount = getCandidateReductionChunkCount(
        root, candidateReductionTileSizes, failureReason);
    if (mlir::failed(chunkCount))
      return mlir::failure();
    reductionChunkCounts.push_back(*chunkCount);
  }

  uint64_t materializationCount = 0;
  switch (wafer::detail::checkCompleteCandidateExpansionBudget(
      *outputTileCount, reductionChunkCounts, materializationCount)) {
  case wafer::detail::CompleteCandidateExpansionStatus::WithinBudget:
    return mlir::success();
  case wafer::detail::CompleteCandidateExpansionStatus::BudgetExceeded:
    setFailureReason(
        failureReason,
        "complete candidate traversal exceeds the eager materialization "
        "budget; this is an implementation resource limit, not an IR or "
        "target legality restriction");
    return mlir::failure();
  case wafer::detail::CompleteCandidateExpansionStatus::CountOverflow:
    setFailureReason(
        failureReason,
        "complete candidate traversal expansion count is not representable");
    return mlir::failure();
  case wafer::detail::CompleteCandidateExpansionStatus::InvalidInput:
    setFailureReason(
        failureReason,
        "complete candidate traversal has invalid static expansion counts");
    return mlir::failure();
  }
  llvm_unreachable("unknown complete candidate expansion status");
}

static void buildCandidateOutputTileProducts(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> tileSizes,
    unsigned dim, llvm::SmallVectorImpl<int64_t> &currentOffsets,
    llvm::SmallVectorImpl<int64_t> &currentSizes,
    llvm::SmallVectorImpl<CandidateOutputTile> &tiles) {
  if (dim == shape.size()) {
    tiles.push_back(
        CandidateOutputTile{llvm::SmallVector<int64_t, 4>(
                                currentOffsets.begin(), currentOffsets.end()),
                            llvm::SmallVector<int64_t, 4>(currentSizes.begin(),
                                                          currentSizes.end())});
    return;
  }

  for (int64_t offset = 0; offset < shape[dim];) {
    int64_t size = std::min(tileSizes[dim], shape[dim] - offset);
    currentOffsets.push_back(offset);
    currentSizes.push_back(size);
    buildCandidateOutputTileProducts(shape, tileSizes, dim + 1, currentOffsets,
                                     currentSizes, tiles);
    currentOffsets.pop_back();
    currentSizes.pop_back();
    offset += size;
  }
}

static mlir::FailureOr<llvm::SmallVector<CandidateOutputTile, 8>>
buildCandidateOutputTiles(mlir::RankedTensorType resultType,
                          llvm::ArrayRef<int64_t> candidateTileSizes,
                          std::string *failureReason) {
  mlir::FailureOr<uint64_t> tileCount = getCandidateOutputTileCount(
      resultType, candidateTileSizes, failureReason);
  if (mlir::failed(tileCount))
    return mlir::failure();

  llvm::SmallVector<CandidateOutputTile, 8> tiles;
  tiles.reserve(static_cast<size_t>(*tileCount));
  llvm::SmallVector<int64_t, 4> currentOffsets;
  llvm::SmallVector<int64_t, 4> currentSizes;
  buildCandidateOutputTileProducts(resultType.getShape(), candidateTileSizes,
                                   /*dim=*/0, currentOffsets, currentSizes,
                                   tiles);
  return tiles;
}

mlir::LogicalResult materializeCompleteCandidateTraversal(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      collectCandidateRoots(group, /*rejectProducerChains=*/true,
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
  if (mlir::failed(checkCompleteCandidateExpansionBudget(
          *roots, firstResultType, candidateTileSizes,
          candidateReductionTileSizes, failureReason)))
    return mlir::failure();

  mlir::FailureOr<llvm::SmallVector<CandidateOutputTile, 8>> tiles =
      buildCandidateOutputTiles(firstResultType, candidateTileSizes,
                                failureReason);
  if (mlir::failed(tiles))
    return mlir::failure();

  auto yield =
      mlir::cast<GroupYieldOp>(group.getBody().front().getTerminator());
  llvm::SmallVector<mlir::Value, 4> completeOutputs;
  for (auto [index, root] : llvm::enumerate(*roots)) {
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        group, static_cast<unsigned>(index), failureReason);
    if (mlir::failed(outputBoundary))
      return mlir::failure();
    mlir::Value output = *outputBoundary;
    for (const CandidateOutputTile &tile : *tiles) {
      mlir::FailureOr<mlir::Value> tileValue =
          materializeCandidateRootTileValue(
              group, root, static_cast<unsigned>(index), tile.offsets,
              tile.sizes, candidateReductionTileSizes, failureReason);
      if (mlir::failed(tileValue))
        return mlir::failure();
      output = insertCandidateRootTile(root, *tileValue, output, tile.offsets,
                                       tile.sizes);
    }
    completeOutputs.push_back(output);
  }

  for (auto [index, output] : llvm::enumerate(completeOutputs))
    yield->setOperand(index, output);
  for (mlir::linalg::LinalgOp root : *roots)
    root->erase();
  return mlir::success();
}

} // namespace wafer::group_to_tile_region
