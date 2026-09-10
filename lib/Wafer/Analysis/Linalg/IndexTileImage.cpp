//===- IndexTileImage.cpp - Exact parametric rectangular demands --------===//

#include "Wafer/Analysis/Linalg/IndexRelation.h"

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <cstdlib>
#include <limits>

using namespace mlir;
using namespace mlir::presburger;

namespace wafer::analysis {
namespace {

RectangularTileImageResult reject(IndexRelationStatus status,
                                  llvm::StringRef reason) {
  return {status, std::nullopt, reason.str()};
}

struct LinearCoordinate {
  llvm::SmallVector<int64_t, 6> coefficients;
  int64_t constant;
};

// A bounded affine function maps each source coordinate to a translated copy
// of its image at the origin. Separability and interval coverage are cheap
// construction proofs; coupled coordinates use the generic exact-image query.
bool provesDenseRectangle(llvm::ArrayRef<LinearCoordinate> coordinates,
                          llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<bool, 6> assigned(sizes.size(), false);
  for (const auto &coordinate : coordinates) {
    llvm::SmallVector<std::pair<int64_t, int64_t>, 6> terms;
    for (auto [axis, coefficient] : llvm::enumerate(coordinate.coefficients)) {
      if (!coefficient || sizes[axis] == 1)
        continue;
      if (assigned[axis])
        return false;
      assigned[axis] = true;
      terms.emplace_back(std::abs(coefficient), sizes[axis]);
    }
    llvm::sort(terms);
    __int128 reach = 0;
    for (auto [coefficient, size] : terms) {
      if (coefficient > reach + 1)
        return false;
      reach += static_cast<__int128>(coefficient) * (size - 1);
    }
  }
  return true;
}

// A recovered affine expression is only a subset of the relation's contract.
// Substitute it into every current constraint and check extrema over the box.
// This proves totality including all intermediate bounds in bounded work,
// without general Presburger subtraction or dropping a clipped domain.
bool provesCompleteAffineDomain(const IntegerRelation &relation,
                                llvm::ArrayRef<LinearCoordinate> coordinates,
                                llvm::ArrayRef<int64_t> shape) {
  const unsigned rank = shape.size();
  auto holds = [&](llvm::ArrayRef<llvm::DynamicAPInt> row, bool equality) {
    llvm::SmallVector<llvm::DynamicAPInt, 8> expression(row.take_front(rank));
    llvm::DynamicAPInt constant = row.back();
    for (auto [source, coordinate] : llvm::enumerate(coordinates)) {
      constant += row[rank + source] * llvm::DynamicAPInt(coordinate.constant);
      for (unsigned axis = 0; axis < rank; ++axis)
        expression[axis] += row[rank + source] *
                            llvm::DynamicAPInt(coordinate.coefficients[axis]);
    }
    auto minimum = constant, maximum = constant;
    for (auto [coefficient, extent] : llvm::zip_equal(expression, shape)) {
      auto distance = coefficient * llvm::DynamicAPInt(extent - 1);
      if (coefficient < 0)
        minimum += distance;
      else
        maximum += distance;
    }
    return equality ? minimum == 0 && maximum == 0 : minimum >= 0;
  };
  for (unsigned row = 0; row < relation.getNumEqualities(); ++row)
    if (!holds(relation.getEquality(row), true))
      return false;
  for (unsigned row = 0; row < relation.getNumInequalities(); ++row)
    if (!holds(relation.getInequality(row), false))
      return false;
  return true;
}

} // namespace

RectangularTileImageResult IndexRelation::getRectangularTileImage(
    MLIRContext *context, llvm::ArrayRef<int64_t> destinationShape,
    llvm::ArrayRef<int64_t> sourceShape, llvm::ArrayRef<int64_t> tileSizes,
    const IndexRelationLimits &limits) const {
  const unsigned rank = destinationShape.size();
  if (!context || rank != getDestinationRank() ||
      sourceShape.size() != getSourceRank() || tileSizes.size() != rank ||
      llvm::any_of(sourceShape, [](int64_t size) { return size <= 0; }) ||
      llvm::any_of(llvm::zip_equal(destinationShape, tileSizes), [](auto pair) {
        auto [extent, size] = pair;
        return extent <= 0 || size <= 0 || size > extent;
      }))
    return reject(IndexRelationStatus::Invalid,
                  "tile image requires positive static shapes and valid sizes");
  if (getStatus() != IndexRelationStatus::Exact)
    return reject(getStatus(), "tile image requires an exact relation");

  auto map = getProjectedAffineMap(context);
  if (!map)
    return reject(IndexRelationStatus::Unsupported,
                  "relation has no translated rectangular tile representation");
  const auto &constraints = getPresburgerRelation().getDisjunct(0);
  if (constraints.getNumVars() > limits.maxVariables ||
      constraints.getNumConstraints() > limits.maxConstraintsPerDisjunct)
    return reject(IndexRelationStatus::ResourceExhausted,
                  "tile relation constraint count exceeds budget");
  llvm::SmallVector<LinearCoordinate, 6> coordinates;
  llvm::SmallVector<AffineExpr, 6> offsetExpressions, sizeExpressions;
  llvm::SmallVector<bool, 6> dependent(rank, false);
  for (auto [expression, sourceExtent] :
       llvm::zip_equal(map->getResults(), sourceShape)) {
    llvm::SmallVector<int64_t, 8> coefficients;
    if (failed(getFlattenedAffineExpr(expression, rank, 0, &coefficients)) ||
        coefficients.size() != rank + 1)
      return reject(IndexRelationStatus::Unsupported,
                    "tile image requires integral affine translations");
    LinearCoordinate coordinate{
        llvm::SmallVector<int64_t, 6>(coefficients.begin(),
                                      coefficients.begin() + rank),
        coefficients.back()};
    __int128 minimum = coordinate.constant, maximum = coordinate.constant;
    AffineExpr offset = getAffineConstantExpr(coordinate.constant, context);
    AffineExpr size = getAffineConstantExpr(1, context);
    for (auto [axis, coefficient] : llvm::enumerate(coordinate.coefficients)) {
      if (coefficient == std::numeric_limits<int64_t>::min() ||
          static_cast<uint64_t>(std::abs(coefficient)) >
              limits.maxAbsoluteCoefficient)
        return reject(IndexRelationStatus::ResourceExhausted,
                      "tile image coefficient exceeds budget");
      dependent[axis] = dependent[axis] || coefficient != 0;
      __int128 distance =
          static_cast<__int128>(coefficient) * (destinationShape[axis] - 1);
      minimum += std::min<__int128>(0, distance);
      maximum += std::max<__int128>(0, distance);
      offset = offset + getAffineDimExpr(axis, context) * coefficient;
      if (coefficient < 0)
        offset =
            offset + (getAffineDimExpr(rank + axis, context) - 1) * coefficient;
      size =
          size + (getAffineDimExpr(axis, context) - 1) * std::abs(coefficient);
    }
    if (minimum < 0 || maximum >= sourceExtent)
      return reject(IndexRelationStatus::Unsupported,
                    "source bounds clip the complete destination domain");
    coordinates.push_back(std::move(coordinate));
    offsetExpressions.push_back(offset);
    sizeExpressions.push_back(size);
  }
  // Recovering an expression alone does not prove totality: intermediate
  // bounds or prior domain intersections may have removed points.
  auto complete = fromAffineMap(*map, destinationShape, sourceShape, limits);
  if (!complete.isExact())
    return reject(complete.status, complete.reason);
  if (!provesCompleteAffineDomain(constraints, coordinates, destinationShape))
    return reject(IndexRelationStatus::Unsupported,
                  "relation does not cover its complete affine tile domain");

  RectangularTileImage result;
  result.offsetMap = AffineMap::get(2 * rank, 0, offsetExpressions, context);
  result.sizeMap = AffineMap::get(rank, 0, sizeExpressions, context);
  llvm::SmallVector<unsigned, 6> active, ragged;
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (!dependent[axis])
      result.invariantDimensions.push_back(axis);
    else if (tileSizes[axis] < destinationShape[axis])
      active.push_back(axis);
    if (destinationShape[axis] % tileSizes[axis])
      ragged.push_back(axis);
  }
  uint64_t classes = 1;
  for (unsigned axis : ragged) {
    (void)axis;
    if (classes > limits.maxRectangularPieces / 2)
      return reject(IndexRelationStatus::ResourceExhausted,
                    "tile image main/tail classes exceed budget");
    classes *= 2;
  }
  if (classes > limits.maxRectangularPieces)
    return reject(IndexRelationStatus::ResourceExhausted,
                  "tile image main/tail classes exceed budget");
  for (uint64_t mask = 0; mask < classes; ++mask) {
    llvm::SmallVector<int64_t, 6> offsets(rank, 0), sizes(tileSizes);
    for (auto [bit, axis] : llvm::enumerate(ragged))
      if (mask & (uint64_t{1} << bit)) {
        sizes[axis] = destinationShape[axis] % tileSizes[axis];
        offsets[axis] = destinationShape[axis] - sizes[axis];
      }
    if (!provesDenseRectangle(coordinates, sizes)) {
      auto image = getExactStaticRectangularImage(offsets, sizes, limits);
      if (!image.isExact())
        return reject(image.status, image.reason);
    }
    ++result.checkedTileClasses;
  }
  // All tiles in each class are translations of its representative. Because
  // the complete affine domain was proved above, source bounds never clip
  // those translations. Thus the bounds maps are exact for the entire grid.

  // Fast injectivity proof for rectangular translations: every varying tile
  // coordinate has a separating source coordinate. This also covers windows
  // with invariant axes and negative strides, without operation-specific rules.
  bool separated = true;
  for (unsigned axis : active) {
    bool witness = false;
    for (const auto &coordinate : coordinates) {
      if (!coordinate.coefficients[axis])
        continue;
      if (llvm::any_of(active, [&](unsigned other) {
            return other != axis && coordinate.coefficients[other] != 0;
          }))
        continue;
      __int128 span = 1;
      for (auto [coefficient, size] :
           llvm::zip_equal(coordinate.coefficients, tileSizes))
        span += static_cast<__int128>(std::abs(coefficient)) * (size - 1);
      __int128 shift =
          static_cast<__int128>(std::abs(coordinate.coefficients[axis])) *
          tileSizes[axis];
      witness |= shift >= span;
    }
    separated &= witness;
  }
  if (separated) {
    result.distinctTilesDisjoint = true;
    return {IndexRelationStatus::Exact, std::move(result), {}};
  }

  // q -> x, with x in the bounded tile q*B .. q*B+B-1. Invariant q
  // coordinates have already been proved irrelevant and are existentially
  // eliminated. Compose with the actual access relation, then test whether
  // any source point belongs to two distinct remaining tile coordinates.
  if (active.size() + rank + sourceShape.size() > limits.maxVariables)
    return reject(IndexRelationStatus::ResourceExhausted,
                  "tile disjointness relation exceeds variable budget");
  IntegerRelation grid(PresburgerSpace::getRelationSpace(active.size(), rank));
  for (unsigned axis = 0; axis < rank; ++axis) {
    unsigned x = active.size() + axis;
    grid.addBound(BoundType::LB, x, 0);
    grid.addBound(BoundType::UB, x, destinationShape[axis] - 1);
  }
  for (auto [q, axis] : llvm::enumerate(active)) {
    grid.addBound(BoundType::LB, q, 0);
    grid.addBound(BoundType::UB, q,
                  llvm::divideCeil(destinationShape[axis], tileSizes[axis]) -
                      1);
    llvm::SmallVector<int64_t, 12> lower(grid.getNumVars() + 1, 0);
    lower[q] = -tileSizes[axis];
    lower[active.size() + axis] = 1;
    grid.addInequality(lower);
    for (int64_t &coefficient : lower)
      coefficient = -coefficient;
    lower.back() = tileSizes[axis] - 1;
    grid.addInequality(lower);
  }
  IndexRelation tiledDomain(PresburgerRelation(grid),
                            IndexRelationStatus::Exact);
  auto tiled = tiledDomain.compose(*this, limits);
  if (!tiled.isExact())
    return reject(tiled.status, tiled.reason);
  auto disjoint = tiled.get()->isInjective(limits);
  if (disjoint.status != IndexRelationStatus::Exact)
    return reject(disjoint.status, disjoint.reason);
  result.distinctTilesDisjoint = disjoint.isProvenTrue();
  return {IndexRelationStatus::Exact, std::move(result), {}};
}

} // namespace wafer::analysis
