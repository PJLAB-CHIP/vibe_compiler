//===- CardBaselinePlacement.cpp -------------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselinePlacement.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <tuple>

namespace wafer::compiler::detail {

std::optional<llvm::SmallVector<CardBaselineSpatialAxis, 4>>
getCardBaselineSpatialAxes(const StructuredDAGNode &node) {
  if (!node.operation || node.operation->getNumResults() == 0)
    return std::nullopt;
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      node.operation->getResult(0).getType());
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(node.operation);
  if (!resultType || !resultType.hasStaticShape() || resultType.getRank() <= 0 ||
      !tiling)
    return std::nullopt;
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      tiling.getLoopIteratorTypes();
  if (iteratorTypes.empty())
    return std::nullopt;

  llvm::SmallVector<CardBaselineSpatialAxis, 4> axes;
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(node.operation)) {
    mlir::AffineMap resultMap =
        linalg.getIndexingMapMatchingResult(node.operation->getResult(0));
    if (!resultMap || resultMap.getNumDims() != iteratorTypes.size() ||
        resultMap.getNumResults() !=
            static_cast<unsigned>(resultType.getRank()))
      return std::nullopt;
    for (auto [resultDimension, expression] :
         llvm::enumerate(resultMap.getResults())) {
      auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!iterator || iterator.getPosition() >= iteratorTypes.size() ||
          iteratorTypes[iterator.getPosition()] !=
              mlir::utils::IteratorType::parallel)
        continue;
      const int64_t extent = resultType.getShape()[resultDimension];
      if (extent <= 0)
        continue;
      CardBaselineSpatialAxis axis;
      axis.iteratorDimension = iterator.getPosition();
      axis.resultDimension = static_cast<unsigned>(resultDimension);
      axis.extent = static_cast<uint64_t>(extent);
      axis.unitPartitionFactors.assign(iteratorTypes.size(), 1);
      axes.push_back(std::move(axis));
    }
  } else if (static_cast<size_t>(resultType.getRank()) ==
             iteratorTypes.size()) {
    for (auto [dimension, extent] :
         llvm::enumerate(resultType.getShape())) {
      if (extent <= 0 ||
          iteratorTypes[dimension] != mlir::utils::IteratorType::parallel)
        continue;
      CardBaselineSpatialAxis axis;
      axis.iteratorDimension = static_cast<unsigned>(dimension);
      axis.resultDimension = static_cast<unsigned>(dimension);
      axis.extent = static_cast<uint64_t>(extent);
      axis.unitPartitionFactors.assign(iteratorTypes.size(), 1);
      axes.push_back(std::move(axis));
    }
  }
  llvm::sort(axes, [](const CardBaselineSpatialAxis &lhs,
                      const CardBaselineSpatialAxis &rhs) {
    return std::tuple(std::numeric_limits<uint64_t>::max() - lhs.extent,
                      lhs.iteratorDimension, lhs.resultDimension) <
           std::tuple(std::numeric_limits<uint64_t>::max() - rhs.extent,
                      rhs.iteratorDimension, rhs.resultDimension);
  });
  if (axes.empty())
    return std::nullopt;
  return axes;
}

mlir::FailureOr<DeterministicSpatialAdvance>
advanceDeterministicSpatialCoordinate(
    llvm::SmallVectorImpl<StructuredDAGNodePlacement> &placements,
    std::string *failureReason) {
  if (placements.empty()) {
    if (failureReason)
      *failureReason = "deterministic spatial coordinate has no placement";
    return mlir::failure();
  }
  size_t maximumParticipants = 1;
  for (const StructuredDAGNodePlacement &placement : placements) {
    if (placement.tiles.empty() ||
        placement.iteratorPartitionFactors.empty()) {
      if (failureReason)
        *failureReason =
            "deterministic spatial coordinate has no Tile or iterator";
      return mlir::failure();
    }
    llvm::SmallVector<unsigned, 2> partitionedIterators;
    for (auto [iterator, factor] :
         llvm::enumerate(placement.iteratorPartitionFactors))
      if (factor != 1)
        partitionedIterators.push_back(iterator);
    if (partitionedIterators.empty()) {
      if (placement.tiles.size() != 1) {
        if (failureReason)
          *failureReason =
              "unpartitioned spatial coordinate is not singleton/unit";
        return mlir::failure();
      }
      continue;
    }
    if (partitionedIterators.size() != 1) {
      if (failureReason)
        *failureReason =
            "deterministic baseline coordinate partitions several iterators";
      return mlir::failure();
    }
    const unsigned iterator = partitionedIterators.front();
    if (placement.iteratorPartitionFactors[iterator] !=
        placement.tiles.size()) {
      if (failureReason)
        *failureReason =
            "partitioned spatial coordinate has inconsistent iterator "
            "factors";
      return mlir::failure();
    }
    maximumParticipants = std::max(maximumParticipants, placement.tiles.size());
  }
  if (maximumParticipants == 1)
    return DeterministicSpatialAdvance::Exhausted;

  const size_t nextMaximumParticipants = maximumParticipants - 1;
  for (StructuredDAGNodePlacement &placement : placements) {
    if (placement.tiles.size() <= nextMaximumParticipants)
      continue;
    std::optional<unsigned> partitionedIterator;
    for (auto [iterator, factor] :
         llvm::enumerate(placement.iteratorPartitionFactors))
      if (factor != 1) {
        partitionedIterator = iterator;
        break;
      }
    if (!partitionedIterator)
      continue;
    placement.tiles.erase(placement.tiles.begin() + nextMaximumParticipants,
                          placement.tiles.end());
    placement.iteratorPartitionFactors[*partitionedIterator] =
        static_cast<uint32_t>(nextMaximumParticipants);
  }
  return DeterministicSpatialAdvance::Advanced;
}

} // namespace wafer::compiler::detail
