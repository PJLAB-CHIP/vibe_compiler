//===- StructuredOperationTileFootprint.cpp --------------------------===//

#include "Wafer/Compiler/Planning/StructuredOperationTileFootprint.h"

#include "Wafer/Compiler/Planning/TemporalTileShape.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <limits>

namespace wafer::compiler::detail {
namespace {

using SignedRange = std::pair<int64_t, int64_t>;

uint64_t saturatingAdd(uint64_t lhs, uint64_t rhs) {
  return lhs > std::numeric_limits<uint64_t>::max() - rhs
             ? std::numeric_limits<uint64_t>::max()
             : lhs + rhs;
}

uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  return product > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(product);
}

uint64_t saturatingAlignTo(uint64_t value, uint64_t alignment) {
  if (alignment <= 1)
    return value;
  const uint64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  return saturatingAdd(value, alignment - remainder);
}

uint64_t ceilDivide(uint64_t numerator, uint64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

std::optional<uint64_t> getElementByteWidth(mlir::Type type) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped)
    return std::nullopt;
  mlir::Type elementType = shaped.getElementType();
  unsigned bits = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(elementType))
    bits = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(elementType))
    bits = floating.getWidth();
  else
    return std::nullopt;
  return std::max<uint64_t>(1, ceilDivide(bits, 8));
}

std::optional<int64_t> narrowSigned(__int128 value) {
  if (value > std::numeric_limits<int64_t>::max() ||
      value < std::numeric_limits<int64_t>::min())
    return std::nullopt;
  return static_cast<int64_t>(value);
}

std::optional<SignedRange>
getAffineTileRange(mlir::AffineExpr expression,
                   llvm::ArrayRef<int64_t> iteratorTileShape) {
  if (auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression))
    return SignedRange{constant.getValue(), constant.getValue()};
  if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= iteratorTileShape.size() ||
        iteratorTileShape[dimension.getPosition()] <= 0)
      return std::nullopt;
    return SignedRange{0, iteratorTileShape[dimension.getPosition()] - 1};
  }
  if (mlir::isa<mlir::AffineSymbolExpr>(expression))
    return std::nullopt;
  auto binary = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression);
  if (!binary)
    return std::nullopt;
  std::optional<SignedRange> lhs =
      getAffineTileRange(binary.getLHS(), iteratorTileShape);
  std::optional<SignedRange> rhs =
      getAffineTileRange(binary.getRHS(), iteratorTileShape);
  if (!lhs || !rhs)
    return std::nullopt;
  switch (expression.getKind()) {
  case mlir::AffineExprKind::Add: {
    std::optional<int64_t> lower =
        narrowSigned(static_cast<__int128>(lhs->first) + rhs->first);
    std::optional<int64_t> upper =
        narrowSigned(static_cast<__int128>(lhs->second) + rhs->second);
    return lower && upper ? std::optional<SignedRange>{{*lower, *upper}}
                          : std::nullopt;
  }
  case mlir::AffineExprKind::Mul: {
    const std::array<__int128, 4> products{
        static_cast<__int128>(lhs->first) * rhs->first,
        static_cast<__int128>(lhs->first) * rhs->second,
        static_cast<__int128>(lhs->second) * rhs->first,
        static_cast<__int128>(lhs->second) * rhs->second};
    auto [minimum, maximum] =
        std::minmax_element(products.begin(), products.end());
    std::optional<int64_t> lower = narrowSigned(*minimum);
    std::optional<int64_t> upper = narrowSigned(*maximum);
    return lower && upper ? std::optional<SignedRange>{{*lower, *upper}}
                          : std::nullopt;
  }
  default:
    return std::nullopt;
  }
}

std::optional<llvm::SmallVector<int64_t, 4>> getMappedTileShape(
    mlir::AffineMap map, int64_t resultRank,
    llvm::ArrayRef<int64_t> iteratorTileShape) {
  if (!map || map.getNumDims() != iteratorTileShape.size() ||
      map.getNumResults() != static_cast<unsigned>(resultRank))
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(resultRank);
  for (mlir::AffineExpr expression : map.getResults()) {
    std::optional<SignedRange> range =
        getAffineTileRange(expression, iteratorTileShape);
    if (!range || range->second < range->first)
      return std::nullopt;
    shape.push_back(range->second - range->first + 1);
  }
  return shape;
}

std::optional<uint64_t>
getAlignedPhysicalBytes(mlir::MLIRContext *context,
                        llvm::ArrayRef<int64_t> shape,
                        mlir::Type elementType, MemLayout layout,
                        const TargetMemoryPolicy &memory) {
  auto type = mlir::MemRefType::get(
      shape, elementType, mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(context, MemorySpace::SPM, layout));
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info || info->physicalBytes < 0)
    return std::nullopt;
  return saturatingAlignTo(static_cast<uint64_t>(info->physicalBytes),
                           static_cast<uint64_t>(memory.spmAlignment));
}

template <typename Estimate, typename NextTileSize>
std::optional<unsigned> selectRefinementAxis(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> fullIteratorShape,
    llvm::ArrayRef<int64_t> currentIteratorTileShape, Estimate estimate,
    NextTileSize nextTileSize, bool allowEqualEstimate = false) {
  std::optional<uint64_t> current = estimate(operation, currentIteratorTileShape);
  if (!current)
    return selectTemporalTileRefinementAxis(
        fullIteratorShape, currentIteratorTileShape, 1);

  struct Selection {
    unsigned dimension = 0;
    uint64_t residencyReduction = 0;
    uint64_t addedWaves = 1;
  };
  std::optional<Selection> selected;
  for (auto [dimension, tileSize] :
       llvm::enumerate(currentIteratorTileShape)) {
    if (tileSize <= 1)
      continue;
    llvm::SmallVector<int64_t, 4> next(currentIteratorTileShape.begin(),
                                       currentIteratorTileShape.end());
    next[dimension] =
        nextTileSize(fullIteratorShape[dimension], tileSize);
    std::optional<uint64_t> nextResidency = estimate(operation, next);
    if (!nextResidency || *nextResidency > *current ||
        (!allowEqualEstimate && *nextResidency == *current))
      continue;
    const uint64_t currentWaves = ceilDivide(
        static_cast<uint64_t>(fullIteratorShape[dimension]),
        static_cast<uint64_t>(tileSize));
    const uint64_t nextWaves = ceilDivide(
        static_cast<uint64_t>(fullIteratorShape[dimension]),
        static_cast<uint64_t>(next[dimension]));
    Selection candidate{static_cast<unsigned>(dimension),
                        *current - *nextResidency,
                        nextWaves - currentWaves};
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
  return selected ? std::optional<unsigned>(selected->dimension)
                  : std::nullopt;
}

} // namespace

std::optional<uint64_t> estimateStructuredOperationTileResidencyBytes(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> iteratorTileShape,
    const TargetMemoryPolicy &memory) {
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(operation);
  if (!linalg)
    return std::nullopt;
  llvm::SmallVector<mlir::AffineMap, 4> maps = linalg.getIndexingMapsArray();
  if (maps.size() != operation->getNumOperands())
    return std::nullopt;

  uint64_t residency = 0;
  bool hasKnownBuffer = false;
  for (auto [operandNumber, operand] :
       llvm::enumerate(operation->getOperands())) {
    mlir::AffineMap map = maps[operandNumber];
    auto shaped = mlir::dyn_cast<mlir::ShapedType>(operand.getType());
    std::optional<uint64_t> elementBytes =
        getElementByteWidth(operand.getType());
    if (!shaped || !shaped.hasRank() || !elementBytes)
      continue;
    std::optional<llvm::SmallVector<int64_t, 4>> shape =
        getMappedTileShape(map, shaped.getRank(), iteratorTileShape);
    if (!shape)
      continue;
    uint64_t elements = 1;
    for (int64_t extent : *shape)
      elements = saturatingMultiply(elements, static_cast<uint64_t>(extent));
    residency = saturatingAdd(
        residency,
        saturatingAlignTo(saturatingMultiply(elements, *elementBytes),
                          static_cast<uint64_t>(memory.spmAlignment)));
    hasKnownBuffer = true;
  }
  return hasKnownBuffer ? std::optional<uint64_t>(residency) : std::nullopt;
}

std::optional<uint64_t> getStructuredOperationLoweringSPMUpperBoundBytes(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> localIteratorShape,
    llvm::ArrayRef<int64_t> iteratorTileShape,
    const TargetMemoryPolicy &memory) {
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(operation);
  if (!linalg || localIteratorShape.size() != iteratorTileShape.size())
    return std::nullopt;
  llvm::SmallVector<mlir::AffineMap, 4> maps = linalg.getIndexingMapsArray();
  if (maps.size() != operation->getNumOperands())
    return std::nullopt;

  uint64_t upperBound = 0;
  std::optional<uint64_t> resultTemporaryBytes;
  for (auto [operandNumber, operand] :
       llvm::enumerate(operation->getOperands())) {
    mlir::AffineMap map = maps[operandNumber];
    auto shaped = mlir::dyn_cast<mlir::ShapedType>(operand.getType());
    if (!shaped || !shaped.hasRank() || !shaped.hasStaticShape())
      continue;
    std::optional<llvm::SmallVector<int64_t, 4>> shape =
        getMappedTileShape(map, shaped.getRank(), iteratorTileShape);
    if (!shape)
      return std::nullopt;
    std::optional<uint64_t> tensorBytes = getAlignedPhysicalBytes(
        operation->getContext(), *shape, shaped.getElementType(),
        MemLayout::Tensor, memory);
    if (!tensorBytes)
      return std::nullopt;
    upperBound = saturatingAdd(upperBound, *tensorBytes);

    // The current compute lowering may require a blocked compute-layout copy
    // in addition to the Tensor staging allocation. Account for both here;
    // cost ranking may use the cheaper logical estimate, but baseline
    // legality cannot assume the staging allocation disappears early.
    if (!shape->empty()) {
      const MemLayout computeLayout =
          shape->size() > 2 ? MemLayout::NCx : MemLayout::Cx;
      std::optional<uint64_t> computeBytes = getAlignedPhysicalBytes(
          operation->getContext(), *shape, shaped.getElementType(),
          computeLayout, memory);
      if (!computeBytes)
        return std::nullopt;
      upperBound = saturatingAdd(upperBound, *computeBytes);
    }

    if (linalg.isDpsInit(&operation->getOpOperand(operandNumber)))
      resultTemporaryBytes = tensorBytes;
  }

  // A generic scalar body can materialize one result-shaped buffer per
  // scalar value while constructing the wave. Named operations use dedicated
  // lowering and do not materialize their scalar payload region this way.
  if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(operation)) {
    if (!resultTemporaryBytes && generic.getNumDpsInits() != 0) {
      auto resultType = mlir::dyn_cast<mlir::ShapedType>(
          generic.getDpsInits().front().getType());
      std::optional<llvm::SmallVector<int64_t, 4>> resultShape =
          getMappedTileShape(generic.getMatchingIndexingMap(
                                 &generic->getOpOperand(
                                     generic.getNumDpsInputs())),
                             resultType ? resultType.getRank() : 0,
                             iteratorTileShape);
      if (!resultType || !resultShape)
        return std::nullopt;
      resultTemporaryBytes = getAlignedPhysicalBytes(
          operation->getContext(), *resultShape, resultType.getElementType(),
          MemLayout::Tensor, memory);
    }
    if (!resultTemporaryBytes)
      return std::nullopt;
    for (mlir::Operation &bodyOp : generic.getBody()->without_terminator()) {
      if (bodyOp.getNumResults() == 0 ||
          llvm::all_of(bodyOp.getResultTypes(),
                       [](mlir::Type type) {
                         return mlir::isa<mlir::IndexType>(type);
                       }))
        continue;
      upperBound = saturatingAdd(upperBound, *resultTemporaryBytes);
    }
  }

  // Temporal traversal emits an explicit prologue, one reusable steady-loop
  // body, and (only for a non-divisible extent with at least two full waves)
  // one explicit tail. These are distinct static allocations in the selected
  // TileRegion and the current asynchronous lowering may keep their resource
  // lifetimes overlapping. Bound the complete static wave-class set instead
  // of pretending one isolated wave is the allocation unit.
  uint64_t waveClasses = 1;
  for (auto [fullExtent, tileExtent] :
       llvm::zip_equal(localIteratorShape, iteratorTileShape)) {
    if (fullExtent <= 0 || tileExtent <= 0 || tileExtent > fullExtent)
      return std::nullopt;
    if (tileExtent == fullExtent)
      continue;
    const uint64_t waves = ceilDivide(static_cast<uint64_t>(fullExtent),
                                      static_cast<uint64_t>(tileExtent));
    const uint64_t classes =
        fullExtent % tileExtent == 0 || waves == 2 ? 2 : 3;
    waveClasses = saturatingMultiply(waveClasses, classes);
  }
  return saturatingMultiply(upperBound, waveClasses);
}

std::optional<unsigned> selectStructuredOperationTileRefinementAxis(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> fullIteratorShape,
    llvm::ArrayRef<int64_t> currentIteratorTileShape,
    const TargetMemoryPolicy &memory) {
  return selectRefinementAxis(
      operation, fullIteratorShape, currentIteratorTileShape,
      [&](mlir::Operation *candidate, llvm::ArrayRef<int64_t> shape) {
        return estimateStructuredOperationTileResidencyBytes(candidate, shape,
                                                              memory);
      },
      getNextLowerTemporalWaveTileSize);
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
deriveStructuredOperationTemporalTileShape(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> localIteratorShape,
    const TargetMemoryPolicy &memory) {
  llvm::SmallVector<int64_t, 4> shape(localIteratorShape.begin(),
                                      localIteratorShape.end());
  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  while (true) {
    std::optional<uint64_t> residency =
        getStructuredOperationLoweringSPMUpperBoundBytes(
            operation, localIteratorShape, shape, memory);
    if (!residency)
      return mlir::failure();
    if (*residency <= capacity)
      break;

    struct BaselineRefinement {
      unsigned dimension = 0;
      int64_t tileSize = 0;
      uint64_t residencyReduction = 0;
      uint64_t addedWaves = 1;
    };
    std::optional<BaselineRefinement> selected;
    for (auto [dimension, currentTileSize] : llvm::enumerate(shape)) {
      int64_t nextTileSize = currentTileSize;
      while (nextTileSize > 1) {
        nextTileSize = getNextLowerDivisibleTemporalTileSize(
            localIteratorShape[dimension], nextTileSize);
        llvm::SmallVector<int64_t, 4> next(shape.begin(), shape.end());
        next[dimension] = nextTileSize;
        std::optional<uint64_t> nextResidency =
            getStructuredOperationLoweringSPMUpperBoundBytes(
                operation, localIteratorShape, next, memory);
        if (!nextResidency)
          return mlir::failure();
        // The first split introduces a second static wave class and can be
        // byte-neutral (or slightly larger after alignment). Continue along
        // this one axis until the complete static class set actually shrinks;
        // this is one deterministic greedy transition, not a candidate set.
        if (*nextResidency >= *residency)
          continue;
        const uint64_t currentWaves = ceilDivide(
            static_cast<uint64_t>(localIteratorShape[dimension]),
            static_cast<uint64_t>(currentTileSize));
        const uint64_t nextWaves = ceilDivide(
            static_cast<uint64_t>(localIteratorShape[dimension]),
            static_cast<uint64_t>(nextTileSize));
        BaselineRefinement candidate{
            static_cast<unsigned>(dimension), nextTileSize,
            *residency - *nextResidency, nextWaves - currentWaves};
        if (!selected) {
          selected = candidate;
          break;
        }
        const unsigned __int128 candidateBenefit =
            static_cast<unsigned __int128>(candidate.residencyReduction) *
            selected->addedWaves;
        const unsigned __int128 selectedBenefit =
            static_cast<unsigned __int128>(selected->residencyReduction) *
            candidate.addedWaves;
        if (candidateBenefit > selectedBenefit ||
            (candidateBenefit == selectedBenefit &&
             (candidate.residencyReduction >
                  selected->residencyReduction ||
              (candidate.residencyReduction ==
                   selected->residencyReduction &&
               (candidate.addedWaves < selected->addedWaves ||
                (candidate.addedWaves == selected->addedWaves &&
                 candidate.dimension < selected->dimension))))))
          selected = candidate;
        break;
      }
    }
    if (!selected)
      return mlir::failure();
    shape[selected->dimension] = selected->tileSize;
  }
  return shape;
}

std::optional<llvm::SmallVector<int64_t, 4>>
getStructuredResultTileShape(mlir::Operation *operation,
                             unsigned resultNumber,
                             llvm::ArrayRef<int64_t> iteratorTileShape) {
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(operation);
  if (!linalg || resultNumber >= operation->getNumResults())
    return std::nullopt;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getResult(resultNumber).getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  return getMappedTileShape(
      linalg.getIndexingMapMatchingResult(operation->getResult(resultNumber)),
      type.getRank(), iteratorTileShape);
}

std::optional<llvm::SmallVector<int64_t, 4>>
getStructuredOperandTileShape(mlir::Operation *operation,
                              unsigned operandNumber,
                              llvm::ArrayRef<int64_t> iteratorTileShape) {
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(operation);
  if (!linalg || operandNumber >= operation->getNumOperands())
    return std::nullopt;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getOperand(operandNumber).getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  return getMappedTileShape(
      linalg.getMatchingIndexingMap(&linalg->getOpOperand(operandNumber)),
      type.getRank(), iteratorTileShape);
}

} // namespace wafer::compiler::detail
