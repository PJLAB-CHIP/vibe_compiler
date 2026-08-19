//===- EdgeFragmentPlanning.cpp - Peer fragment geometry -----------===//

#include "EdgeFragmentPlanning.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>
#include <optional>

using namespace wafer;

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

mlir::LogicalResult splitIndependentPeerFragmentsAtTemporalWaves(
    llvm::MutableArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason) {
  llvm::DenseMap<int64_t, int64_t> nextPayloadSlice;
  for (SpatialEdgeStrategy &strategy : edgeStrategies) {
    if (strategy.action != SpatialEdgeAction::PeerFragments)
      continue;
    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> temporalTiles =
        getResultTemporalTileSizes(strategy.producer, strategy.producerResult,
                                   operationTemporalTiles, failureReason);
    if (mlir::failed(temporalTiles))
      return mlir::failure();
    auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
        strategy.producer->getResult(strategy.producerResult).getType());
    const unsigned elementBits =
        producerType.getElementType().isIntOrFloat()
            ? producerType.getElementType().getIntOrFloatBitWidth()
            : 0;
    if (elementBits == 0 || elementBits % 8 != 0)
      return failResult(
          failureReason,
          "peer fragment temporal projection requires a byte-addressable "
          "element type");

    llvm::SmallVector<SpatialEdgeFragment, 16> splitFragments;
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      if (fragment.offsets.size() != temporalTiles->size() ||
          fragment.sizes.size() != temporalTiles->size())
        return failResult(
            failureReason,
            "peer fragment temporal projection has an inconsistent rank");
      if (!isContained(fragment.offsets, fragment.sizes,
                       strategy.producerOffsets, strategy.producerSizes))
        return failResult(
            failureReason,
            "dependent fragment extends outside its consumer demand");
      std::optional<size_t> splitDimension;
      for (size_t dimension = 0; dimension < temporalTiles->size(); ++dimension)
        if ((*temporalTiles)[dimension] < fragment.sizes[dimension]) {
          splitDimension = dimension;
          break;
        }
      llvm::SmallVector<llvm::SmallVector<std::pair<int64_t, int64_t>, 4>, 4>
          dimensionSegments(temporalTiles->size());
      for (size_t dimension = 0; dimension < temporalTiles->size();
           ++dimension) {
        const int64_t begin = fragment.offsets[dimension];
        const int64_t size = fragment.sizes[dimension];
        const int64_t tile =
            splitDimension == dimension ? (*temporalTiles)[dimension] : size;
        if (begin < 0 || size <= 0 || tile <= 0)
          return failResult(
              failureReason,
              "peer fragment temporal projection has an invalid domain");
        if (begin > std::numeric_limits<int64_t>::max() - size)
          return failResult(
              failureReason,
              "dependent fragment extends outside its consumer demand");
        const int64_t end = begin + size;
        for (int64_t offset = begin; offset < end;) {
          const int64_t nextGrid = ((offset / tile) + 1) * tile;
          const int64_t next = std::min(end, nextGrid);
          dimensionSegments[dimension].push_back({offset, next - offset});
          offset = next;
        }
      }

      llvm::SmallVector<int64_t, 4> offsets(temporalTiles->size());
      llvm::SmallVector<int64_t, 4> sizes(temporalTiles->size());
      std::function<mlir::LogicalResult(size_t)> appendDimension =
          [&](size_t dimension) -> mlir::LogicalResult {
        if (dimension != dimensionSegments.size()) {
          for (auto [offset, size] : dimensionSegments[dimension]) {
            offsets[dimension] = offset;
            sizes[dimension] = size;
            if (mlir::failed(appendDimension(dimension + 1)))
              return mlir::failure();
          }
          return mlir::success();
        }
        uint64_t elements = 1;
        for (int64_t size : sizes) {
          if (elements > std::numeric_limits<uint64_t>::max() /
                             static_cast<uint64_t>(size))
            return failResult(failureReason,
                              "peer temporal fragment byte count overflows");
          elements *= static_cast<uint64_t>(size);
        }
        const uint64_t bytes = elements * (elementBits / 8);
        if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max())
          return failResult(
              failureReason,
              "peer temporal fragment exceeds the target payload range");
        SpatialEdgeFragment split = fragment;
        split.offsets = offsets;
        split.sizes = sizes;
        split.bytes = split.kind == SpatialEdgeFragmentKind::Peer ? bytes : 0;
        if (split.kind == SpatialEdgeFragmentKind::Peer)
          split.payloadSlice = nextPayloadSlice[split.communicationId]++;
        splitFragments.push_back(std::move(split));
        return mlir::success();
      };
      if (mlir::failed(appendDimension(0)))
        return mlir::failure();
    }
    strategy.fragments = std::move(splitFragments);
  }
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
