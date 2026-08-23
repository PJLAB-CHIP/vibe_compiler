//===- EdgeFragmentPlanning.cpp - Peer fragment geometry -----------===//

#include "EdgeFragmentPlanning.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

} // namespace

bool isContained(llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
                 llvm::ArrayRef<int64_t> containerOffsets,
                 llvm::ArrayRef<int64_t> containerSizes) {
  if (offsets.size() != sizes.size() ||
      offsets.size() != containerOffsets.size() ||
      offsets.size() != containerSizes.size())
    return false;
  return llvm::all_of(
      llvm::zip_equal(offsets, sizes, containerOffsets, containerSizes),
      [](auto values) {
        auto [offset, size, containerOffset, containerSize] = values;
        return offset >= 0 && containerOffset >= 0 && size > 0 &&
               containerSize > 0 && size <= containerSize &&
               offset >= containerOffset &&
               offset - containerOffset <= containerSize - size;
      });
}

bool overlaps(llvm::ArrayRef<int64_t> lhsOffsets,
              llvm::ArrayRef<int64_t> lhsSizes,
              llvm::ArrayRef<int64_t> rhsOffsets,
              llvm::ArrayRef<int64_t> rhsSizes) {
  if (lhsOffsets.size() != lhsSizes.size() ||
      lhsOffsets.size() != rhsOffsets.size() ||
      lhsOffsets.size() != rhsSizes.size())
    return true;
  return llvm::all_of(
      llvm::zip_equal(lhsOffsets, lhsSizes, rhsOffsets, rhsSizes),
      [](auto values) {
        auto [lhsOffset, lhsSize, rhsOffset, rhsSize] = values;
        if (lhsOffset < 0 || rhsOffset < 0 || lhsSize <= 0 || rhsSize <= 0)
          return false;
        if (lhsOffset <= rhsOffset)
          return lhsSize > rhsOffset - lhsOffset;
        return rhsSize > lhsOffset - rhsOffset;
      });
}

bool isFullStaticResultDomain(mlir::Operation *operation, unsigned resultNumber,
                              llvm::ArrayRef<int64_t> offsets,
                              llvm::ArrayRef<int64_t> sizes) {
  if (!operation || resultNumber >= operation->getNumResults())
    return false;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getResult(resultNumber).getType());
  return type && type.hasStaticShape() &&
         offsets.size() == static_cast<size_t>(type.getRank()) &&
         sizes.size() == offsets.size() &&
         llvm::all_of(llvm::zip_equal(offsets, sizes, type.getShape()),
                      [](auto values) {
                        auto [offset, size, extent] = values;
                        return offset == 0 && size == extent;
                      });
}

bool isOneFullTemporalWave(
    mlir::Operation *operation,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles) {
  auto selected = llvm::find_if(operationTemporalTiles,
                                [&](const StructuredOpTemporalTile &tile) {
                                  return tile.operation == operation;
                                });
  if (selected == operationTemporalTiles.end())
    return true;
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!linalg)
    return false;
  llvm::SmallVector<int64_t, 4> ranges = linalg.getStaticLoopRanges();
  return selected->iteratorTileSizes.size() == ranges.size() &&
         llvm::all_of(llvm::zip_equal(selected->iteratorTileSizes, ranges),
                      [](auto values) {
                        auto [tile, extent] = values;
                        return tile > 0 && extent > 0 && tile >= extent;
                      });
}

bool isOneFullTemporalWaveClosure(
    mlir::Operation *operation,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles) {
  if (!operation)
    return false;
  llvm::DenseSet<mlir::Operation *> visited;
  llvm::SmallVector<mlir::Operation *, 16> worklist{operation};
  while (!worklist.empty()) {
    mlir::Operation *current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;
    if (mlir::isa<mlir::linalg::LinalgOp>(current) &&
        !isOneFullTemporalWave(current, operationTemporalTiles))
      return false;
    for (mlir::Value operand : current->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      if (definition && definition->getBlock() == operation->getBlock())
        worklist.push_back(definition);
    }
  }
  return true;
}

bool isOneFullTemporalWaveForResultDemand(
    mlir::Operation *operation, unsigned resultNumber,
    llvm::ArrayRef<int64_t> resultSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles) {
  auto selected = llvm::find_if(operationTemporalTiles,
                                [&](const StructuredOpTemporalTile &tile) {
                                  return tile.operation == operation;
                                });
  if (selected == operationTemporalTiles.end())
    return true;
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!linalg || resultNumber >= operation->getNumResults())
    return false;
  llvm::SmallVector<int64_t, 4> ranges = linalg.getStaticLoopRanges();
  mlir::AffineMap resultMap =
      linalg.getIndexingMapMatchingResult(operation->getResult(resultNumber));
  if (!resultMap || resultMap.getNumResults() != resultSizes.size() ||
      selected->iteratorTileSizes.size() != ranges.size())
    return false;

  llvm::SmallVector<int64_t, 4> required(ranges.begin(), ranges.end());
  for (auto [resultDimension, expression] :
       llvm::enumerate(resultMap.getResults())) {
    auto loopDimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!loopDimension || loopDimension.getPosition() >= required.size() ||
        resultSizes[resultDimension] <= 0)
      return false;
    required[loopDimension.getPosition()] = resultSizes[resultDimension];
  }
  return llvm::all_of(llvm::zip_equal(selected->iteratorTileSizes, required),
                      [](auto values) {
                        auto [tile, extent] = values;
                        return tile > 0 && extent > 0 && tile >= extent;
                      });
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>> getResultTemporalTileSizes(
    mlir::Operation *operation, unsigned resultNumber,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason) {
  auto resultType = operation && resultNumber < operation->getNumResults()
                        ? mlir::dyn_cast<mlir::RankedTensorType>(
                              operation->getResult(resultNumber).getType())
                        : mlir::RankedTensorType{};
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(operation);
  auto selected = llvm::find_if(operationTemporalTiles,
                                [&](const StructuredOpTemporalTile &tile) {
                                  return tile.operation == operation;
                                });
  if (!resultType || !resultType.hasStaticShape() || !linalg ||
      selected == operationTemporalTiles.end())
    return fail<llvm::SmallVector<int64_t, 4>>(
        failureReason,
        "peer fragment temporal projection requires one selected static "
        "structured producer");
  mlir::AffineMap resultMap =
      linalg.getIndexingMapMatchingResult(operation->getResult(resultNumber));
  if (!resultMap ||
      resultMap.getNumResults() != static_cast<unsigned>(resultType.getRank()))
    return fail<llvm::SmallVector<int64_t, 4>>(
        failureReason,
        "peer fragment temporal projection lacks its exact result map");
  llvm::SmallVector<int64_t, 4> resultTiles;
  resultTiles.reserve(resultType.getRank());
  for (auto [resultDimension, expression] :
       llvm::enumerate(resultMap.getResults())) {
    auto loopDimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!loopDimension ||
        loopDimension.getPosition() >= selected->iteratorTileSizes.size())
      return fail<llvm::SmallVector<int64_t, 4>>(
          failureReason,
          "peer fragment temporal projection is not a projected iterator "
          "domain");
    const int64_t tile =
        selected->iteratorTileSizes[loopDimension.getPosition()];
    const int64_t extent = resultType.getDimSize(resultDimension);
    if (tile <= 0 || extent <= 0)
      return fail<llvm::SmallVector<int64_t, 4>>(
          failureReason,
          "peer fragment temporal projection has a nonpositive extent");
    resultTiles.push_back(std::min(tile, extent));
  }
  return resultTiles;
}

} // namespace wafer::tensor_program_to_tile_region
