//===- TemporalTileShape.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"

#include "Wafer/Target/Core/Tx81InstructionLimits.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

namespace wafer::compiler::detail {
namespace {

uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  return product > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(product);
}

uint64_t ceilDivide(uint64_t numerator, uint64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

uint64_t getElementCount(llvm::ArrayRef<int64_t> shape) {
  uint64_t elements = 1;
  for (int64_t extent : shape)
    elements = saturatingMultiply(elements, static_cast<uint64_t>(extent));
  return elements;
}

uint64_t saturatingAlignTo(uint64_t value, uint64_t alignment) {
  if (alignment <= 1)
    return value;
  const uint64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  const uint64_t increment = alignment - remainder;
  return value > std::numeric_limits<uint64_t>::max() - increment
             ? std::numeric_limits<uint64_t>::max()
             : value + increment;
}

int64_t getWaveBreakpointAtOrBelow(int64_t extent, int64_t requested) {
  requested = std::clamp<int64_t>(requested, 1, extent);
  const uint64_t waves = ceilDivide(static_cast<uint64_t>(extent),
                                    static_cast<uint64_t>(requested));
  return static_cast<int64_t>(ceilDivide(static_cast<uint64_t>(extent), waves));
}

} // namespace

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
deriveLocalIteratorExtents(mlir::Operation *operation,
                           llvm::ArrayRef<uint32_t> partitionFactors,
                           std::string *failureReason) {
  auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(operation);
  if (!tiling) {
    if (failureReason)
      *failureReason = "structured operation has no iterator domain";
    return mlir::failure();
  }
  llvm::SmallVector<int64_t, 4> extents;
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation)) {
    extents = linalg.getStaticLoopRanges();
  } else if (operation->getNumResults() == 1) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(
        operation->getResult(0).getType());
    if (type && type.hasStaticShape())
      extents.assign(type.getShape().begin(), type.getShape().end());
  }
  if (extents.size() != partitionFactors.size() ||
      extents.size() != tiling.getLoopIteratorTypes().size()) {
    if (failureReason)
      *failureReason = "structured iterator/factor rank is inconsistent";
    return mlir::failure();
  }
  for (auto [dimension, factor] : llvm::enumerate(partitionFactors)) {
    if (factor == 0 || extents[dimension] <= 0) {
      if (failureReason)
        *failureReason = "structured iterator extent or factor is invalid";
      return mlir::failure();
    }
    extents[dimension] =
        extents[dimension] / factor + (extents[dimension] % factor != 0);
  }
  return extents;
}

uint64_t estimateAlignedTileResidencyBytes(llvm::ArrayRef<int64_t> tileShape,
                                           uint64_t elementBytes,
                                           uint64_t tensorMultiplicity,
                                           const TargetMemoryPolicy &memory) {
  const uint64_t allocationBytes =
      saturatingMultiply(getElementCount(tileShape), elementBytes);
  const uint64_t alignedAllocation = saturatingAlignTo(
      allocationBytes, static_cast<uint64_t>(memory.spmAlignment));
  return saturatingMultiply(alignedAllocation, tensorMultiplicity);
}

int64_t getNextLowerTemporalWaveTileSize(int64_t fullExtent,
                                         int64_t currentTileSize) {
  assert(fullExtent > 0 && currentTileSize > 1 &&
         currentTileSize <= fullExtent);
  const uint64_t currentWaves =
      ceilDivide(static_cast<uint64_t>(fullExtent),
                 static_cast<uint64_t>(currentTileSize));
  const uint64_t nextClassUpperBound =
      (static_cast<uint64_t>(fullExtent) - 1) / currentWaves;
  const int64_t next = getWaveBreakpointAtOrBelow(
      fullExtent, static_cast<int64_t>(nextClassUpperBound));
  assert(next >= 1 && next < currentTileSize);
  return next;
}

int64_t getNextLowerDivisibleTemporalTileSize(int64_t fullExtent,
                                              int64_t currentTileSize) {
  assert(fullExtent > 0 && currentTileSize > 1 &&
         currentTileSize <= fullExtent);
  int64_t best = 0;
  for (int64_t divisor = 1; divisor <= fullExtent / divisor; ++divisor) {
    if (fullExtent % divisor != 0)
      continue;
    if (divisor < currentTileSize)
      best = std::max(best, divisor);
    const int64_t quotient = fullExtent / divisor;
    if (quotient < currentTileSize)
      best = std::max(best, quotient);
  }
  assert(best > 0 && best < currentTileSize);
  return best;
}

std::optional<unsigned>
selectTemporalTileRefinementAxis(llvm::ArrayRef<int64_t> fullShape,
                                 llvm::ArrayRef<int64_t> currentShape,
                                 uint64_t knownBytesPerIterationPoint) {
  assert(fullShape.size() == currentShape.size());
  struct Selection {
    unsigned dimension = 0;
    uint64_t residencyReduction = 0;
    uint64_t addedWaves = 1;
  };
  std::optional<Selection> selected;
  for (auto [dimension, current] : llvm::enumerate(currentShape)) {
    if (current <= 1)
      continue;
    const int64_t next =
        getNextLowerTemporalWaveTileSize(fullShape[dimension], current);
    uint64_t otherElements = 1;
    for (auto [otherDimension, extent] : llvm::enumerate(currentShape))
      if (otherDimension != dimension)
        otherElements =
            saturatingMultiply(otherElements, static_cast<uint64_t>(extent));
    const uint64_t removedElements = saturatingMultiply(
        otherElements, static_cast<uint64_t>(current - next));
    const uint64_t reduction = saturatingMultiply(
        removedElements, std::max<uint64_t>(1, knownBytesPerIterationPoint));
    const uint64_t currentAxisWaves =
        ceilDivide(static_cast<uint64_t>(fullShape[dimension]),
                   static_cast<uint64_t>(current));
    const uint64_t nextAxisWaves =
        ceilDivide(static_cast<uint64_t>(fullShape[dimension]),
                   static_cast<uint64_t>(next));
    Selection candidate{static_cast<unsigned>(dimension), reduction,
                        nextAxisWaves - currentAxisWaves};
    if (!selected) {
      selected = candidate;
      continue;
    }
    const unsigned __int128 candidateBenefit =
        static_cast<unsigned __int128>(candidate.residencyReduction) *
        selected->addedWaves;
    const unsigned __int128 selectedBenefit =
        static_cast<unsigned __int128>(selected->residencyReduction) *
        candidate.addedWaves;
    if (candidateBenefit > selectedBenefit ||
        (candidateBenefit == selectedBenefit &&
         (candidate.residencyReduction > selected->residencyReduction ||
          (candidate.residencyReduction == selected->residencyReduction &&
           (candidate.addedWaves < selected->addedWaves ||
            (candidate.addedWaves == selected->addedWaves &&
             candidate.dimension < selected->dimension))))))
      selected = candidate;
  }
  return selected ? std::optional<unsigned>(selected->dimension) : std::nullopt;
}

llvm::SmallVector<int64_t, 4> deriveCapacityTemporalTileShape(
    llvm::ArrayRef<int64_t> maximumShardShape, uint64_t elementBytes,
    uint64_t tensorMultiplicity, const TargetMemoryPolicy &memory,
    unsigned additionalWaveRefinements) {
  llvm::SmallVector<int64_t, 4> shape(maximumShardShape.begin(),
                                      maximumShardShape.end());
  if (shape.size() <= 4) {
    constexpr std::array<uint32_t, 4> nativeDimensionLimits = {
        Tx81InstructionLimits::dataShapeOuterMax,
        Tx81InstructionLimits::dataShapeOuterMax,
        Tx81InstructionLimits::dataShapeOuterMax,
        Tx81InstructionLimits::dataShapeChannelMax};
    const size_t leadingDimensions = 4 - shape.size();
    for (auto [dimension, extent] : llvm::enumerate(shape))
      shape[dimension] = std::min<int64_t>(
          extent, nativeDimensionLimits[leadingDimensions + dimension]);
  }

  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  const uint64_t knownBytesPerIterationPoint =
      saturatingMultiply(elementBytes, tensorMultiplicity);
  auto refineOnce = [&]() {
    std::optional<unsigned> dimension = selectTemporalTileRefinementAxis(
        maximumShardShape, shape, knownBytesPerIterationPoint);
    if (!dimension)
      return false;
    shape[*dimension] = getNextLowerTemporalWaveTileSize(
        maximumShardShape[*dimension], shape[*dimension]);
    return true;
  };
  while (estimateAlignedTileResidencyBytes(
             shape, elementBytes, tensorMultiplicity, memory) > capacity &&
         refineOnce()) {
  }
  for (unsigned refinement = 0;
       refinement < additionalWaveRefinements && refineOnce(); ++refinement) {
  }
  return shape;
}

} // namespace wafer::compiler::detail
