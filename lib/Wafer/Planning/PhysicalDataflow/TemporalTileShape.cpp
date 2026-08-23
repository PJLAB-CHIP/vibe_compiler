//===- TemporalTileShape.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

using SignedRange = std::pair<int64_t, int64_t>;

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
