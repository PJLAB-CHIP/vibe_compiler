//===- IndexRelation.cpp - MLIR-backed logical index relations -----------===//

#include "Wafer/Analysis/Linalg/IndexRelation.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::presburger;

namespace wafer::analysis {
namespace {

static IndexRelationResult fail(IndexRelationStatus status,
                                llvm::StringRef reason) {
  return IndexRelationResult{status, std::nullopt, reason.str()};
}

static IndexSetResult failSet(IndexRelationStatus status,
                              llvm::StringRef reason) {
  return IndexSetResult{status, std::nullopt, reason.str()};
}

static StaticRectangularIndexSetResult failRectangle(IndexRelationStatus status,
                                                     llvm::StringRef reason) {
  return StaticRectangularIndexSetResult{status, std::nullopt, reason.str()};
}

static StaticRectangularIndexSetPiecesResult
failRectanglePieces(IndexRelationStatus status, llvm::StringRef reason) {
  return StaticRectangularIndexSetPiecesResult{status, {}, reason.str()};
}

static IndexRelationQueryResult failQuery(IndexRelationStatus status,
                                          llvm::StringRef reason) {
  return IndexRelationQueryResult{status, std::nullopt, reason.str()};
}

static bool isShapeValid(llvm::ArrayRef<int64_t> shape) {
  return llvm::all_of(shape, [](int64_t dim) {
    return dim >= 0 || ShapedType::isDynamic(dim);
  });
}

static bool hasDynamicDim(llvm::ArrayRef<int64_t> shape) {
  return llvm::any_of(shape, ShapedType::isDynamic);
}

static bool exceedsVariableLimit(unsigned destinationRank, unsigned sourceRank,
                                 const IndexRelationLimits &limits) {
  return destinationRank > limits.maxVariables ||
         sourceRank > limits.maxVariables - destinationRank;
}

static bool exceedsDisjunctWorkLimits(const IntegerRelation &disjunct,
                                      const IndexRelationLimits &limits) {
  if (disjunct.getNumVars() > limits.maxVariables ||
      disjunct.getNumConstraints() > limits.maxConstraintsPerDisjunct ||
      disjunct.getNumLocalVars() > limits.maxLocalVariablesPerDisjunct)
    return true;
  const llvm::DynamicAPInt maximumCoefficient(
      static_cast<int64_t>(limits.maxAbsoluteCoefficient));
  auto rowExceeds = [&](llvm::ArrayRef<llvm::DynamicAPInt> row) {
    return llvm::any_of(row, [&](const llvm::DynamicAPInt &coefficient) {
      return llvm::abs(coefficient) > maximumCoefficient;
    });
  };
  for (unsigned row = 0; row < disjunct.getNumEqualities(); ++row)
    if (rowExceeds(disjunct.getEquality(row)))
      return true;
  for (unsigned row = 0; row < disjunct.getNumInequalities(); ++row)
    if (rowExceeds(disjunct.getInequality(row)))
      return true;
  return false;
}

static bool exceedsRelationLimits(const PresburgerRelation &relation,
                                  const IndexRelationLimits &limits) {
  return relation.getNumVars() > limits.maxVariables ||
         relation.getNumDisjuncts() > limits.maxDisjuncts ||
         llvm::any_of(relation.getAllDisjuncts(),
                      [&](const IntegerRelation &disjunct) {
                        return exceedsDisjunctWorkLimits(disjunct, limits);
                      });
}

static bool exceedsSetLimits(const PresburgerSet &set,
                             const IndexRelationLimits &limits) {
  return set.getNumVars() > limits.maxVariables ||
         set.getNumDisjuncts() > limits.maxDisjuncts ||
         llvm::any_of(set.getAllDisjuncts(),
                      [&](const IntegerRelation &disjunct) {
                        return exceedsDisjunctWorkLimits(disjunct, limits);
                      });
}

static bool isCompatibleSet(const PresburgerSet &set, unsigned rank) {
  const PresburgerSpace &space = set.getSpace();
  return space.getNumDomainVars() == 0 && space.getNumSetDimVars() == rank &&
         space.getNumSymbolVars() == 0;
}

/// Recognize the exact representation produced by staticRectangularDomain:
/// a conjunction of independent unit-coefficient inclusive bounds. This is a
/// structural proof and deliberately performs no Presburger optimization.
static StaticRectangularIndexSetResult
recoverDirectStaticRectangle(const IntegerRelation &box, unsigned rank) {
  if (box.getNumSymbolVars() != 0 || box.getNumLocalVars() != 0 ||
      box.getNumEqualities() != 0)
    return failRectangle(IndexRelationStatus::Unsupported,
                         "index-set disjunct is not a direct static rectangle");

  llvm::SmallVector<std::optional<int64_t>, 4> lowerBounds(rank);
  llvm::SmallVector<std::optional<int64_t>, 4> upperBounds(rank);
  for (unsigned row = 0; row < box.getNumInequalities(); ++row) {
    llvm::SmallVector<int64_t, 8> inequality = box.getInequality64(row);
    std::optional<unsigned> dimension;
    int64_t coefficient = 0;
    for (unsigned index = 0; index < rank; ++index) {
      if (inequality[index] == 0)
        continue;
      if (dimension || (inequality[index] != 1 && inequality[index] != -1))
        return failRectangle(
            IndexRelationStatus::Unsupported,
            "index-set disjunct is not a direct static rectangle");
      dimension = index;
      coefficient = inequality[index];
    }
    if (!dimension)
      return failRectangle(
          IndexRelationStatus::Unsupported,
          "index-set disjunct is not a direct static rectangle");
    const int64_t constant = inequality.back();
    if (coefficient == 1) {
      if (constant == std::numeric_limits<int64_t>::min())
        return failRectangle(IndexRelationStatus::Invalid,
                             "index-set rectangle lower bound overflows");
      const int64_t lower = -constant;
      lowerBounds[*dimension] = lowerBounds[*dimension]
                                    ? std::max(*lowerBounds[*dimension], lower)
                                    : lower;
    } else {
      upperBounds[*dimension] =
          upperBounds[*dimension] ? std::min(*upperBounds[*dimension], constant)
                                  : constant;
    }
  }
  if (!llvm::all_of(lowerBounds,
                    [](const auto &bound) { return bound.has_value(); }) ||
      !llvm::all_of(upperBounds,
                    [](const auto &bound) { return bound.has_value(); }))
    return failRectangle(IndexRelationStatus::Unsupported,
                         "index-set rectangle has an unbounded dimension");

  StaticRectangularIndexSet rectangle;
  rectangle.offsets.reserve(rank);
  rectangle.sizes.reserve(rank);
  for (unsigned dimension = 0; dimension < rank; ++dimension) {
    int64_t size = 0;
    if (*upperBounds[dimension] < *lowerBounds[dimension] ||
        llvm::SubOverflow(*upperBounds[dimension], *lowerBounds[dimension],
                          size) ||
        llvm::AddOverflow(size, int64_t{1}, size))
      return failRectangle(IndexRelationStatus::Invalid,
                           "index-set rectangle is empty or overflows");
    rectangle.offsets.push_back(*lowerBounds[dimension]);
    rectangle.sizes.push_back(size);
  }
  return StaticRectangularIndexSetResult{
      IndexRelationStatus::Exact, std::move(rectangle), {}};
}

static PresburgerRelation getUnboundedIdentityRelation(unsigned rank) {
  IntegerRelation identity(PresburgerSpace::getRelationSpace(rank, rank));
  for (unsigned index = 0; index < rank; ++index) {
    llvm::SmallVector<int64_t, 8> equality(identity.getNumVars() + 1, 0);
    equality[index] = 1;
    equality[rank + index] = -1;
    identity.addEquality(equality);
  }
  return PresburgerRelation(identity);
}

static void addStaticShapeBounds(IntegerRelation &relation,
                                 llvm::ArrayRef<int64_t> destinationShape,
                                 llvm::ArrayRef<int64_t> sourceShape) {
  for (auto [index, dim] : llvm::enumerate(destinationShape)) {
    if (ShapedType::isDynamic(dim))
      continue;
    relation.addBound(BoundType::LB, index, 0);
    relation.addBound(BoundType::UB, index, dim - 1);
  }
  for (auto [index, dim] : llvm::enumerate(sourceShape)) {
    if (ShapedType::isDynamic(dim))
      continue;
    unsigned position = destinationShape.size() + index;
    relation.addBound(BoundType::LB, position, 0);
    relation.addBound(BoundType::UB, position, dim - 1);
  }
}

static IndexRelationResult
finishExactOrBound(IntegerRelation relation, bool isBound,
                   const IndexRelationLimits &limits) {
  PresburgerRelation presburger(relation);
  if (exceedsRelationLimits(presburger, limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "index relation exceeds variable or disjunct budget");
  return IndexRelationResult{
      isBound ? IndexRelationStatus::SoundBound : IndexRelationStatus::Exact,
      IndexRelation(std::move(presburger), isBound
                                               ? IndexRelationStatus::SoundBound
                                               : IndexRelationStatus::Exact),
      {}};
}

static std::optional<int64_t> resolveConstantIndex(OpFoldResult value) {
  if (auto attr =
          llvm::dyn_cast_if_present<IntegerAttr>(value.dyn_cast<Attribute>()))
    return attr.getInt();
  Value dynamic = value.dyn_cast<Value>();
  if (!dynamic || !dynamic.getType().isIndex())
    return std::nullopt;
  FailureOr<int64_t> constant = ValueBoundsConstraintSet::computeConstantBound(
      BoundType::EQ, ValueBoundsConstraintSet::Variable(dynamic));
  return succeeded(constant) ? std::optional<int64_t>(*constant) : std::nullopt;
}

static std::optional<int64_t>
getStaticElementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0)
      return std::nullopt;
    if (llvm::MulOverflow(count, dim, count))
      return std::nullopt;
  }
  return count;
}

static std::optional<llvm::SmallVector<int64_t, 4>>
getRowMajorStrides(llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t, 4> strides(shape.size(), 1);
  int64_t running = 1;
  for (size_t reverseIndex = 0; reverseIndex < shape.size(); ++reverseIndex) {
    size_t index = shape.size() - reverseIndex - 1;
    strides[index] = running;
    if (llvm::MulOverflow(running, shape[index], running))
      return std::nullopt;
  }
  return strides;
}

} // namespace

unsigned IndexRelation::getDestinationRank() const {
  return relation.getNumDomainVars();
}

unsigned IndexRelation::getSourceRank() const {
  return relation.getNumRangeVars();
}

bool IndexRelation::contains(llvm::ArrayRef<int64_t> destination,
                             llvm::ArrayRef<int64_t> source) const {
  if (destination.size() != getDestinationRank() ||
      source.size() != getSourceRank())
    return false;
  llvm::SmallVector<int64_t, 8> point(destination.begin(), destination.end());
  point.append(source.begin(), source.end());
  return relation.containsPoint(point);
}

bool IndexRelation::hasCanonicalRowMajorReshapeConstruction() const {
  return status == IndexRelationStatus::Exact &&
         canonicalRowMajorOrderByConstruction &&
         totalBoundedAffineMapByConstruction && rectangleDestinationShape &&
         rectangleSourceShape;
}

std::optional<AffineMap>
IndexRelation::getProjectedAffineMap(MLIRContext *context) const {
  if (!context || status != IndexRelationStatus::Exact)
    return std::nullopt;
  if (relation.getNumDisjuncts() != 1)
    return std::nullopt;

  const IntegerRelation &disjunct = relation.getDisjunct(0);
  const unsigned destinationRank = getDestinationRank();
  const unsigned sourceRank = getSourceRank();
  if (disjunct.getNumSymbolVars() != 0 || disjunct.getNumLocalVars() != 0)
    return std::nullopt;

  llvm::SmallVector<AffineExpr, 4> results;
  results.reserve(sourceRank);
  for (unsigned source = 0; source < sourceRank; ++source) {
    std::optional<AffineExpr> projected;
    const unsigned sourcePosition = destinationRank + source;
    for (unsigned equality = 0; equality < disjunct.getNumEqualities();
         ++equality) {
      llvm::SmallVector<int64_t, 8> coefficients =
          disjunct.getEquality64(equality);
      int64_t sourceCoefficient = coefficients[sourcePosition];
      if (sourceCoefficient == 0)
        continue;

      bool hasOtherSource = false;
      for (unsigned other = 0; other < sourceRank; ++other) {
        if (other != source && coefficients[destinationRank + other] != 0) {
          hasOtherSource = true;
          break;
        }
      }
      if (hasOtherSource)
        continue;

      // sum(dst_i * a_i) + src * b + constant == 0.
      // Recover src = -(sum(dst_i * a_i) + constant) / b only when
      // every coefficient is exactly integral.
      bool integral = true;
      for (unsigned destination = 0; destination < destinationRank;
           ++destination)
        integral &= coefficients[destination] % sourceCoefficient == 0;
      const int64_t constant = coefficients.back();
      integral &= constant % sourceCoefficient == 0;
      if (!integral)
        continue;

      AffineExpr expression =
          getAffineConstantExpr(-constant / sourceCoefficient, context);
      for (unsigned destination = 0; destination < destinationRank;
           ++destination) {
        int64_t multiplier = -coefficients[destination] / sourceCoefficient;
        if (multiplier != 0)
          expression =
              expression + getAffineDimExpr(destination, context) * multiplier;
      }
      projected = expression;
      break;
    }
    if (!projected)
      return std::nullopt;
    results.push_back(*projected);
  }
  return AffineMap::get(destinationRank, 0, results, context);
}

IndexRelationResult IndexRelation::identity(llvm::ArrayRef<int64_t> shape,
                                            const IndexRelationLimits &limits) {
  if (!isShapeValid(shape) ||
      exceedsVariableLimit(shape.size(), shape.size(), limits))
    return fail(IndexRelationStatus::Invalid,
                "identity relation rank or shape is invalid");
  IntegerRelation identity(
      PresburgerSpace::getRelationSpace(shape.size(), shape.size()));
  for (unsigned index = 0; index < shape.size(); ++index) {
    llvm::SmallVector<int64_t, 8> equality(identity.getNumVars() + 1, 0);
    equality[index] = 1;
    equality[shape.size() + index] = -1;
    identity.addEquality(equality);
  }
  addStaticShapeBounds(identity, shape, shape);
  IndexRelationResult result =
      finishExactOrBound(std::move(identity), /*isBound=*/false, limits);
  if (!result.isExact())
    return result;
  result.relation->functionalByConstruction = true;
  result.relation->canonicalRowMajorOrderByConstruction = true;
  result.relation->injectiveByConstruction = true;
  result.relation->totalBoundedAffineMapByConstruction = true;
  llvm::SmallVector<int64_t, 4> pattern;
  pattern.reserve(shape.size());
  for (unsigned dimension = 0; dimension < shape.size(); ++dimension)
    pattern.push_back(dimension);
  result.relation->projectedRectanglePattern = std::move(pattern);
  result.relation->rectangleDestinationShape =
      llvm::SmallVector<int64_t, 4>(shape);
  result.relation->rectangleSourceShape = llvm::SmallVector<int64_t, 4>(shape);
  return result;
}

IndexRelationResult IndexRelation::fromAffineMap(
    AffineMap map, llvm::ArrayRef<int64_t> destinationShape,
    llvm::ArrayRef<int64_t> sourceShape, const IndexRelationLimits &limits) {
  if (!map || map.getNumDims() != destinationShape.size() ||
      map.getNumResults() != sourceShape.size() ||
      !isShapeValid(destinationShape) || !isShapeValid(sourceShape))
    return fail(IndexRelationStatus::Invalid,
                "affine relation rank or shape is invalid");
  if (map.getNumSymbols() != 0)
    return fail(IndexRelationStatus::Unsupported,
                "symbolic affine relations require explicit symbol bounds");
  if (exceedsVariableLimit(destinationShape.size(), sourceShape.size(), limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "index relation exceeds variable budget");

  IntegerRelation relation(PresburgerSpace::getRelationSpace());
  if (failed(mlir::affine::getRelationFromMap(map, relation)))
    return fail(IndexRelationStatus::Unsupported,
                "affine map cannot be flattened to Presburger constraints");
  relation.setSpace(PresburgerSpace::getRelationSpace(
      destinationShape.size(), sourceShape.size(), map.getNumSymbols(),
      relation.getNumLocalVars()));
  addStaticShapeBounds(relation, destinationShape, sourceShape);
  bool isBound = hasDynamicDim(destinationShape) || hasDynamicDim(sourceShape);
  IndexRelationResult result =
      finishExactOrBound(std::move(relation), isBound, limits);
  if (!result.isExact())
    return result;
  // An affine map is single-valued by construction; the flattened relation
  // preserves that property.
  result.relation->functionalByConstruction = true;
  // A fully constant map is also total and in-bounds by construction when
  // every constant lies inside its corresponding source dimension. This is
  // especially important for an ordered reduction slice: its rank-zero
  // iteration domain maps to one source tuple, and re-proving that singleton
  // fact with generic Presburger subtraction would make compile work scale
  // with the logical tensor extent.
  if (llvm::all_of(llvm::zip_equal(map.getResults(), sourceShape),
                   [](auto values) {
                     auto [expression, extent] = values;
                     auto constant =
                         mlir::dyn_cast<mlir::AffineConstantExpr>(expression);
                     return constant && constant.getValue() >= 0 &&
                            constant.getValue() < extent;
                   }))
    result.relation->totalBoundedAffineMapByConstruction = true;
  // A projected permutation map is a rectangle pattern by construction: the
  // relation is exactly the map, so no equivalence proof is required. The
  // pattern enables the arithmetic rectangular image/preimage fast paths.
  if (map.isProjectedPermutation(/*allowZeroInResults=*/true)) {
    llvm::SmallVector<int64_t, 4> pattern;
    llvm::SmallVector<IndexRelation::RowMajorRectangleMapping, 4>
        rowMajorMappings;
    bool valid = true;
    bool hasExactRowMajorMapping = true;
    for (auto [sourceDimension, expression] :
         llvm::enumerate(map.getResults())) {
      IndexRelation::RowMajorRectangleMapping mapping;
      mapping.sourceDimensions.push_back(sourceDimension);
      if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
        pattern.push_back(dimension.getPosition());
        mapping.destinationDimensions.push_back(dimension.getPosition());
        rowMajorMappings.push_back(std::move(mapping));
        continue;
      }
      auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression);
      if (!constant || constant.getValue() != 0) {
        valid = false;
        break;
      }
      pattern.push_back(-1);
      // A constant projection into a non-singleton source dimension is still
      // handled exactly by projectedRectanglePattern, but it is not a
      // row-major reshape group: only coordinate zero is in the image.
      if (sourceShape[sourceDimension] == 1)
        rowMajorMappings.push_back(std::move(mapping));
      else
        hasExactRowMajorMapping = false;
    }
    if (valid && pattern.size() == map.getNumResults()) {
      result.relation->projectedRectanglePattern = std::move(pattern);
      if (hasExactRowMajorMapping)
        result.relation->rowMajorRectangleMappings =
            std::move(rowMajorMappings);
      result.relation->rectangleDestinationShape =
          llvm::SmallVector<int64_t, 4>(destinationShape);
      result.relation->rectangleSourceShape =
          llvm::SmallVector<int64_t, 4>(sourceShape);

      // Functionality alone does not make the bounded map total: a source
      // extent smaller than the mapped destination extent clips the relation
      // domain. Prove the stronger property dimension by dimension before a
      // caller may skip the generic domain/range checks.
      bool totalAndBounded = true;
      for (auto [sourceDimension, expression] :
           llvm::enumerate(map.getResults())) {
        if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
          if (destinationShape[dimension.getPosition()] >
              sourceShape[sourceDimension]) {
            totalAndBounded = false;
            break;
          }
          continue;
        }
        auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression);
        if (!constant || constant.getValue() < 0 ||
            constant.getValue() >= sourceShape[sourceDimension]) {
          totalAndBounded = false;
          break;
        }
      }
      result.relation->totalBoundedAffineMapByConstruction = totalAndBounded;
    }
  }
  // A projected slice may pin one or more source dimensions to non-zero
  // constants. MLIR's projected-permutation predicate intentionally accepts
  // only constant zero, but arbitrary in-bounds constants have the same
  // closed-form totality proof. Record only total-and-bounded here: the
  // rectangle image helpers encode constant-zero offsets and must not receive
  // a stronger pattern for a non-zero slice.
  if (!result.relation->totalBoundedAffineMapByConstruction) {
    bool totalAndBounded = true;
    for (auto [expression, sourceExtent] :
         llvm::zip_equal(map.getResults(), sourceShape)) {
      if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
        if (dimension.getPosition() >= destinationShape.size() ||
            destinationShape[dimension.getPosition()] > sourceExtent) {
          totalAndBounded = false;
          break;
        }
        continue;
      }
      auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression);
      if (!constant || constant.getValue() < 0 ||
          constant.getValue() >= sourceExtent) {
        totalAndBounded = false;
        break;
      }
    }
    result.relation->totalBoundedAffineMapByConstruction = totalAndBounded;
  }
  if (result.relation->totalBoundedAffineMapByConstruction &&
      map.isProjectedPermutation(/*allowZeroInResults=*/true) &&
      !hasDynamicDim(destinationShape) && !hasDynamicDim(sourceShape)) {
    std::optional<int64_t> destinationElements =
        getStaticElementCount(destinationShape);
    std::optional<int64_t> sourceElements = getStaticElementCount(sourceShape);
    std::optional<llvm::SmallVector<int64_t, 4>> destinationStrides =
        getRowMajorStrides(destinationShape);
    std::optional<llvm::SmallVector<int64_t, 4>> sourceStrides =
        getRowMajorStrides(sourceShape);
    bool sameOrder = destinationElements && sourceElements &&
                     *destinationElements == *sourceElements &&
                     destinationStrides && sourceStrides;
    llvm::SmallVector<int64_t, 4> coefficients(destinationShape.size(), 0);
    if (sameOrder) {
      for (auto [sourceDimension, expression] :
           llvm::enumerate(map.getResults())) {
        if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
          int64_t coefficient = 0;
          if (llvm::AddOverflow(coefficients[dimension.getPosition()],
                                (*sourceStrides)[sourceDimension],
                                coefficient)) {
            sameOrder = false;
            break;
          }
          coefficients[dimension.getPosition()] = coefficient;
          continue;
        }
        auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression);
        if (!constant || constant.getValue() != 0) {
          sameOrder = false;
          break;
        }
      }
    }
    if (sameOrder)
      for (unsigned dimension = 0; dimension < destinationShape.size();
           ++dimension)
        if (destinationShape[dimension] != 1 &&
            coefficients[dimension] != (*destinationStrides)[dimension]) {
          sameOrder = false;
          break;
        }
    result.relation->canonicalRowMajorOrderByConstruction = sameOrder;
  }
  return result;
}

IndexRelationResult IndexRelation::fromCommonIterationDomain(
    AffineMap iterationToDestination, llvm::ArrayRef<int64_t> destinationShape,
    AffineMap iterationToSource, llvm::ArrayRef<int64_t> sourceShape,
    llvm::ArrayRef<int64_t> iterationShape, const IndexRelationLimits &limits) {
  IndexRelationResult toDestination = fromAffineMap(
      iterationToDestination, iterationShape, destinationShape, limits);
  if (!toDestination.relation)
    return toDestination;
  IndexRelationResult toSource =
      fromAffineMap(iterationToSource, iterationShape, sourceShape, limits);
  if (!toSource.relation)
    return toSource;
  IndexRelationResult destinationToIteration =
      toDestination.relation->inverse(limits);
  if (!destinationToIteration.relation)
    return destinationToIteration;
  IndexRelationResult result =
      destinationToIteration.relation->compose(*toSource.relation, limits);
  if (!result.isExact())
    return result;
  mlir::AffineMap destinationToIterationMap =
      mlir::inversePermutation(iterationToDestination);
  if (destinationToIterationMap) {
    mlir::AffineMap projected =
        iterationToSource.compose(destinationToIterationMap);
    if (projected &&
        projected.isProjectedPermutation(/*allowZeroInResults=*/true)) {
      llvm::SmallVector<int64_t, 4> pattern;
      for (AffineExpr expression : projected.getResults()) {
        if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
          pattern.push_back(dimension.getPosition());
          continue;
        }
        auto constant = dyn_cast<AffineConstantExpr>(expression);
        if (!constant || constant.getValue() != 0) {
          pattern.clear();
          break;
        }
        pattern.push_back(-1);
      }
      if (pattern.size() == projected.getNumResults()) {
        IndexRelationResult projectedRelation =
            fromAffineMap(projected, destinationShape, sourceShape, limits);
        if (projectedRelation.isExact() &&
            projectedRelation.relation->isEquivalentTo(*result.relation, limits)
                .isProvenTrue()) {
          result.relation->projectedRectanglePattern = std::move(pattern);
          result.relation->rowMajorRectangleMappings =
              projectedRelation.relation->rowMajorRectangleMappings;
          result.relation->rectangleDestinationShape =
              llvm::SmallVector<int64_t, 4>(destinationShape);
          result.relation->rectangleSourceShape =
              llvm::SmallVector<int64_t, 4>(sourceShape);
          result.relation->canonicalRowMajorOrderByConstruction =
              projectedRelation.relation->canonicalRowMajorOrderByConstruction;
          result.relation->totalBoundedAffineMapByConstruction =
              projectedRelation.relation->totalBoundedAffineMapByConstruction;
        }
      }
    }
  } else {
    // A complete-reduction destination maps every iterator to a constant
    // position, so no inverse permutation exists. The composed relation is
    // still exactly "every destination point covers the complete source
    // domain"; attach the closed-form full-image pattern so the arithmetic
    // rectangle fast paths never enter generic Presburger equality
    // recovery. An unconstrained source dimension is encoded as -2.
    const bool allConstantZeroDestination = llvm::all_of(
        iterationToDestination.getResults(), [](mlir::AffineExpr expression) {
          auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression);
          return constant && constant.getValue() == 0;
        });
    const bool singletonDestination = llvm::all_of(
        destinationShape, [](int64_t extent) { return extent == 1; });
    const bool sourceCoversCompleteDomain =
        iterationToSource.isProjectedPermutation(
            /*allowZeroInResults=*/true) &&
        llvm::all_of(llvm::enumerate(iterationToSource.getResults()),
                     [&](auto indexedExpression) {
                       const unsigned sourceDimension =
                           indexedExpression.index();
                       mlir::AffineExpr expression = indexedExpression.value();
                       if (auto dimension =
                               mlir::dyn_cast<mlir::AffineDimExpr>(expression))
                         return iterationShape[dimension.getPosition()] ==
                                sourceShape[sourceDimension];
                       auto constant =
                           mlir::dyn_cast<mlir::AffineConstantExpr>(expression);
                       return constant && constant.getValue() == 0 &&
                              sourceShape[sourceDimension] == 1;
                     });
    if (allConstantZeroDestination &&
        iterationToDestination.getNumResults() == destinationShape.size() &&
        singletonDestination && sourceCoversCompleteDomain) {
      llvm::SmallVector<int64_t, 4> pattern(sourceShape.size(), -2);
      result.relation->projectedRectanglePattern = std::move(pattern);
      result.relation->rectangleDestinationShape =
          llvm::SmallVector<int64_t, 4>(destinationShape);
      result.relation->rectangleSourceShape =
          llvm::SmallVector<int64_t, 4>(sourceShape);
    }
  }
  return result;
}

IndexRelationResult IndexRelation::staticSlice(
    llvm::ArrayRef<int64_t> destinationShape,
    llvm::ArrayRef<int64_t> sourceShape, llvm::ArrayRef<int64_t> offsets,
    llvm::ArrayRef<int64_t> strides, const IndexRelationLimits &limits) {
  if (destinationShape.size() != sourceShape.size() ||
      offsets.size() != sourceShape.size() ||
      strides.size() != sourceShape.size() || !isShapeValid(destinationShape) ||
      !isShapeValid(sourceShape) || hasDynamicDim(destinationShape) ||
      hasDynamicDim(sourceShape) ||
      llvm::any_of(offsets, [](int64_t offset) { return offset < 0; }) ||
      llvm::any_of(strides, [](int64_t stride) { return stride <= 0; }))
    return fail(IndexRelationStatus::Invalid,
                "static slice requires equal-rank static shapes and valid "
                "offsets/strides");

  MLIRContext context;
  llvm::SmallVector<AffineExpr, 4> results;
  results.reserve(sourceShape.size());
  for (auto [index, pair] : llvm::enumerate(llvm::zip(offsets, strides))) {
    auto [offset, stride] = pair;
    results.push_back(getAffineConstantExpr(offset, &context) +
                      getAffineDimExpr(index, &context) * stride);
  }
  AffineMap map = AffineMap::get(destinationShape.size(), 0, results, &context);
  IndexRelationResult result =
      fromAffineMap(map, destinationShape, sourceShape, limits);
  if (!result.isExact())
    return result;

  // A slice whose destination domain reaches outside the source shape is
  // malformed rather than an exact empty/truncated mapping.
  for (size_t index = 0; index < sourceShape.size(); ++index) {
    int64_t span = 0;
    int64_t last = 0;
    if (destinationShape[index] > 0 &&
        (llvm::MulOverflow(destinationShape[index] - 1, strides[index], span) ||
         llvm::AddOverflow(offsets[index], span, last)))
      return fail(IndexRelationStatus::Invalid,
                  "static slice index arithmetic overflows");
    if (destinationShape[index] > 0 && last >= sourceShape[index])
      return fail(IndexRelationStatus::Invalid,
                  "static slice exceeds source domain");
  }
  result.relation->injectiveByConstruction = true;
  return result;
}

IndexRelationResult IndexRelation::slice(
    llvm::ArrayRef<int64_t> destinationShape,
    llvm::ArrayRef<int64_t> sourceShape, llvm::ArrayRef<OpFoldResult> offsets,
    llvm::ArrayRef<OpFoldResult> strides, const IndexRelationLimits &limits) {
  llvm::SmallVector<int64_t, 4> constantOffsets;
  llvm::SmallVector<int64_t, 4> constantStrides;
  for (OpFoldResult offset : offsets) {
    std::optional<int64_t> constant = resolveConstantIndex(offset);
    if (!constant)
      return fail(IndexRelationStatus::Unsupported,
                  "slice offset is not constant under ValueBounds");
    constantOffsets.push_back(*constant);
  }
  for (OpFoldResult stride : strides) {
    std::optional<int64_t> constant = resolveConstantIndex(stride);
    if (!constant)
      return fail(IndexRelationStatus::Unsupported,
                  "slice stride is not constant under ValueBounds");
    constantStrides.push_back(*constant);
  }
  return staticSlice(destinationShape, sourceShape, constantOffsets,
                     constantStrides, limits);
}

IndexRelationResult
IndexRelation::staticReshape(llvm::ArrayRef<int64_t> destinationShape,
                             llvm::ArrayRef<int64_t> sourceShape,
                             const IndexRelationLimits &limits) {
  if (!isShapeValid(destinationShape) || !isShapeValid(sourceShape) ||
      hasDynamicDim(destinationShape) || hasDynamicDim(sourceShape))
    return fail(IndexRelationStatus::Unsupported,
                "reshape relation requires static shapes");
  if (exceedsVariableLimit(destinationShape.size(), sourceShape.size(), limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "index relation exceeds variable budget");

  std::optional<int64_t> destinationElements =
      getStaticElementCount(destinationShape);
  std::optional<int64_t> sourceElements = getStaticElementCount(sourceShape);
  std::optional<llvm::SmallVector<int64_t, 4>> destinationStrides =
      getRowMajorStrides(destinationShape);
  std::optional<llvm::SmallVector<int64_t, 4>> sourceStrides =
      getRowMajorStrides(sourceShape);
  if (!destinationElements || !sourceElements || !destinationStrides ||
      !sourceStrides)
    return fail(IndexRelationStatus::Invalid,
                "reshape element count or stride overflows");
  if (*destinationElements != *sourceElements)
    return fail(IndexRelationStatus::Invalid,
                "reshape source and destination element counts differ");

  // Recover the reshape's exact reassociation as equal-product row-major
  // groups. This is a construction proof for both collapse and expansion;
  // it does not depend on a solver rediscovering quotient/remainder facts.
  llvm::SmallVector<RowMajorRectangleMapping, 4> rowMajorMappings;
  size_t destinationDimension = 0;
  size_t sourceDimension = 0;
  bool hasRowMajorMapping =
      llvm::all_of(destinationShape,
                   [](int64_t extent) { return extent > 0; }) &&
      llvm::all_of(sourceShape, [](int64_t extent) { return extent > 0; });
  while (hasRowMajorMapping && destinationDimension < destinationShape.size() &&
         sourceDimension < sourceShape.size()) {
    RowMajorRectangleMapping mapping;
    int64_t destinationProduct = destinationShape[destinationDimension];
    int64_t sourceProduct = sourceShape[sourceDimension];
    mapping.destinationDimensions.push_back(destinationDimension++);
    mapping.sourceDimensions.push_back(sourceDimension++);
    while (destinationProduct != sourceProduct) {
      if (destinationProduct < sourceProduct &&
          destinationDimension < destinationShape.size()) {
        if (llvm::MulOverflow(destinationProduct,
                              destinationShape[destinationDimension],
                              destinationProduct)) {
          hasRowMajorMapping = false;
          break;
        }
        mapping.destinationDimensions.push_back(destinationDimension++);
        continue;
      }
      if (sourceProduct < destinationProduct &&
          sourceDimension < sourceShape.size()) {
        if (llvm::MulOverflow(sourceProduct, sourceShape[sourceDimension],
                              sourceProduct)) {
          hasRowMajorMapping = false;
          break;
        }
        mapping.sourceDimensions.push_back(sourceDimension++);
        continue;
      }
      hasRowMajorMapping = false;
      break;
    }
    if (hasRowMajorMapping)
      rowMajorMappings.push_back(std::move(mapping));
  }
  if (hasRowMajorMapping && destinationDimension < destinationShape.size()) {
    RowMajorRectangleMapping mapping;
    while (destinationDimension < destinationShape.size()) {
      if (destinationShape[destinationDimension] != 1) {
        hasRowMajorMapping = false;
        break;
      }
      mapping.destinationDimensions.push_back(destinationDimension++);
    }
    if (hasRowMajorMapping)
      rowMajorMappings.push_back(std::move(mapping));
  }
  if (hasRowMajorMapping && sourceDimension < sourceShape.size()) {
    RowMajorRectangleMapping mapping;
    while (sourceDimension < sourceShape.size()) {
      if (sourceShape[sourceDimension] != 1) {
        hasRowMajorMapping = false;
        break;
      }
      mapping.sourceDimensions.push_back(sourceDimension++);
    }
    if (hasRowMajorMapping)
      rowMajorMappings.push_back(std::move(mapping));
  }

  const bool isAffineCollapse =
      hasRowMajorMapping &&
      llvm::all_of(rowMajorMappings,
                   [](const RowMajorRectangleMapping &mapping) {
                     return mapping.sourceDimensions.size() <= 1;
                   });
  if (isAffineCollapse) {
    MLIRContext context;
    llvm::SmallVector<AffineExpr, 4> results(
        sourceShape.size(), getAffineConstantExpr(0, &context));
    for (const RowMajorRectangleMapping &mapping : rowMajorMappings) {
      if (mapping.sourceDimensions.empty())
        continue;
      AffineExpr expression = getAffineConstantExpr(0, &context);
      int64_t stride = 1;
      for (unsigned destination :
           llvm::reverse(mapping.destinationDimensions)) {
        expression =
            expression + getAffineDimExpr(destination, &context) * stride;
        if (llvm::MulOverflow(stride, destinationShape[destination], stride)) {
          hasRowMajorMapping = false;
          break;
        }
      }
      if (!hasRowMajorMapping)
        break;
      results[mapping.sourceDimensions.front()] = expression;
    }
    IndexRelationResult result = fromAffineMap(
        AffineMap::get(destinationShape.size(), 0, results, &context),
        destinationShape, sourceShape, limits);
    if (result.isExact()) {
      result.relation->rowMajorRectangleMappings = std::move(rowMajorMappings);
      result.relation->rectangleDestinationShape =
          llvm::SmallVector<int64_t, 4>(destinationShape);
      result.relation->rectangleSourceShape =
          llvm::SmallVector<int64_t, 4>(sourceShape);
      result.relation->canonicalRowMajorOrderByConstruction = true;
      result.relation->totalBoundedAffineMapByConstruction = true;
      return result;
    }
  }

  IntegerRelation relation(PresburgerSpace::getRelationSpace(
      destinationShape.size(), sourceShape.size()));
  llvm::SmallVector<int64_t, 8> equality(relation.getNumVars() + 1, 0);
  for (auto [index, stride] : llvm::enumerate(*destinationStrides))
    equality[index] = stride;
  for (auto [index, stride] : llvm::enumerate(*sourceStrides))
    equality[destinationShape.size() + index] = -stride;
  relation.addEquality(equality);
  addStaticShapeBounds(relation, destinationShape, sourceShape);
  IndexRelationResult result =
      finishExactOrBound(std::move(relation), /*isBound=*/false, limits);
  if (result.isExact() && hasRowMajorMapping) {
    result.relation->rowMajorRectangleMappings = std::move(rowMajorMappings);
    result.relation->rectangleDestinationShape =
        llvm::SmallVector<int64_t, 4>(destinationShape);
    result.relation->rectangleSourceShape =
        llvm::SmallVector<int64_t, 4>(sourceShape);
    result.relation->functionalByConstruction = true;
    result.relation->canonicalRowMajorOrderByConstruction = true;
    result.relation->totalBoundedAffineMapByConstruction = true;
  }
  return result;
}

IndexRelationResult
IndexRelation::staticConcatPiece(llvm::ArrayRef<int64_t> destinationShape,
                                 llvm::ArrayRef<int64_t> sourceShape,
                                 unsigned axis, int64_t destinationOffset,
                                 const IndexRelationLimits &limits) {
  if (destinationShape.size() != sourceShape.size() ||
      axis >= destinationShape.size() || destinationOffset < 0 ||
      !isShapeValid(destinationShape) || !isShapeValid(sourceShape) ||
      hasDynamicDim(destinationShape) || hasDynamicDim(sourceShape))
    return fail(IndexRelationStatus::Invalid,
                "concat piece requires equal-rank static shapes, a valid "
                "axis, and a non-negative destination offset");
  for (unsigned index = 0; index < destinationShape.size(); ++index) {
    if (index != axis && destinationShape[index] != sourceShape[index])
      return fail(IndexRelationStatus::Invalid,
                  "concat piece non-concat dimensions must match");
  }
  int64_t pieceEnd = 0;
  if (llvm::AddOverflow(destinationOffset, sourceShape[axis], pieceEnd) ||
      pieceEnd > destinationShape[axis])
    return fail(IndexRelationStatus::Invalid,
                "concat piece exceeds destination domain");

  MLIRContext context;
  llvm::SmallVector<AffineExpr, 4> results;
  results.reserve(sourceShape.size());
  for (unsigned index = 0; index < sourceShape.size(); ++index) {
    AffineExpr expression = getAffineDimExpr(index, &context);
    if (index == axis)
      expression = expression - destinationOffset;
    results.push_back(expression);
  }
  return fromAffineMap(
      AffineMap::get(destinationShape.size(), 0, results, &context),
      destinationShape, sourceShape, limits);
}

IndexRelationResult
IndexRelation::staticInsertSlice(llvm::ArrayRef<int64_t> destinationShape,
                                 llvm::ArrayRef<int64_t> sourceShape,
                                 llvm::ArrayRef<int64_t> offsets,
                                 const IndexRelationLimits &limits) {
  if (destinationShape.size() != sourceShape.size() ||
      offsets.size() != destinationShape.size() ||
      !isShapeValid(destinationShape) || !isShapeValid(sourceShape) ||
      hasDynamicDim(destinationShape) || hasDynamicDim(sourceShape))
    return fail(IndexRelationStatus::Invalid,
                "insert slice requires equal-rank static shapes with one "
                "offset per dimension");
  if (llvm::any_of(offsets, [](int64_t offset) { return offset < 0; }))
    return fail(IndexRelationStatus::Invalid,
                "insert slice offsets must be non-negative");
  for (auto [index, size] : llvm::enumerate(sourceShape)) {
    int64_t pieceEnd = 0;
    if (llvm::AddOverflow(offsets[index], size, pieceEnd) ||
        pieceEnd > destinationShape[index])
      return fail(IndexRelationStatus::Invalid,
                  "insert slice piece exceeds destination domain");
  }

  MLIRContext context;
  llvm::SmallVector<AffineExpr, 4> results;
  results.reserve(sourceShape.size());
  for (unsigned index = 0; index < sourceShape.size(); ++index) {
    AffineExpr expression = getAffineDimExpr(index, &context);
    if (offsets[index] != 0)
      expression = expression - offsets[index];
    results.push_back(expression);
  }
  return fromAffineMap(
      AffineMap::get(destinationShape.size(), 0, results, &context),
      destinationShape, sourceShape, limits);
}

IndexSetResult IndexRelation::staticDomain(llvm::ArrayRef<int64_t> shape,
                                           const IndexRelationLimits &limits) {
  llvm::SmallVector<int64_t, 4> offsets(shape.size(), 0);
  return staticRectangularDomain(offsets, shape, limits);
}

IndexSetResult
IndexRelation::staticRectangularDomain(llvm::ArrayRef<int64_t> offsets,
                                       llvm::ArrayRef<int64_t> sizes,
                                       const IndexRelationLimits &limits) {
  if (offsets.size() != sizes.size() || !isShapeValid(sizes) ||
      hasDynamicDim(sizes) ||
      llvm::any_of(offsets, [](int64_t offset) { return offset < 0; }))
    return failSet(IndexRelationStatus::Unsupported,
                   "static index domain requires static non-negative bounds");
  if (sizes.size() > limits.maxVariables)
    return failSet(IndexRelationStatus::ResourceExhausted,
                   "index domain exceeds variable budget");
  IntegerPolyhedron domain(PresburgerSpace::getSetSpace(sizes.size()));
  for (auto [index, bounds] : llvm::enumerate(llvm::zip(offsets, sizes))) {
    auto [offset, size] = bounds;
    int64_t upper = 0;
    if (llvm::AddOverflow(offset, size, upper))
      return failSet(IndexRelationStatus::Invalid,
                     "static index domain bound overflows");
    domain.addBound(BoundType::LB, index, offset);
    domain.addBound(BoundType::UB, index, upper - 1);
  }
  PresburgerSet set(domain);
  if (exceedsSetLimits(set, limits))
    return failSet(IndexRelationStatus::ResourceExhausted,
                   "index domain exceeds variable or disjunct budget");
  return IndexSetResult{IndexRelationStatus::Exact, std::move(set), {}};
}

StaticRectangularIndexSetResult IndexSetResult::getExactStaticRectangularDomain(
    const IndexRelationLimits &limits) const {
  if (status != IndexRelationStatus::Exact || !set)
    return failRectangle(
        status,
        reason.empty() ? "rectangular recovery requires an exact set" : reason);
  // Reject over-budget structure before any emptiness, extremum or equality
  // query. Variable/disjunct counts alone do not bound
  // Presburger work when a disjunct has many constraints, locals or very
  // large coefficients.
  if (exceedsSetLimits(*set, limits))
    return failRectangle(IndexRelationStatus::ResourceExhausted,
                         "rectangular recovery exceeds index-set budget");

  // A single conjunction of unit, one-dimensional lower/upper bounds is
  // exactly a static rectangle by construction. Recover it arithmetically
  // before invoking any Presburger emptiness/extremum/equality query. This is
  // the common representation of balanced execution and ownership shards;
  // rebuilding a many-piece union and asking a generic solver to rediscover
  // the same box proof is not bounded by the structural limits above.
  if (set->getNumDisjuncts() == 1) {
    const unsigned rank = set->getSpace().getNumSetDimVars();
    if (rank == 0 && set->getDisjunct(0).getNumConstraints() == 0)
      return StaticRectangularIndexSetResult{
          IndexRelationStatus::Exact, StaticRectangularIndexSet{{}, {}}, {}};
    StaticRectangularIndexSetResult direct =
        recoverDirectStaticRectangle(set->getDisjunct(0), rank);
    if (direct.isExact())
      return direct;
  }
  if (set->isIntegerEmpty())
    return failRectangle(IndexRelationStatus::Unsupported,
                         "empty index demand has no transfer rectangle");

  const unsigned rank = set->getSpace().getNumSetDimVars();
  // Every nonempty zero-dimensional Presburger set is the singleton {()}.
  // Avoid generic set equality here: pinned MLIR's symbolic optimizer requires
  // at least one set dimension and asserts for this scalar case.
  if (rank == 0)
    return StaticRectangularIndexSetResult{
        IndexRelationStatus::Exact, StaticRectangularIndexSet{{}, {}}, {}};
  llvm::SmallVector<int64_t, 4> offsets(rank,
                                        std::numeric_limits<int64_t>::max());
  llvm::SmallVector<int64_t, 4> inclusiveUpper(
      rank, std::numeric_limits<int64_t>::min());
  bool sawNonEmptyDisjunct = false;
  for (const IntegerRelation &stored : set->getAllDisjuncts()) {
    if (stored.isIntegerEmpty())
      continue;
    sawNonEmptyDisjunct = true;
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      std::optional<int64_t> lower =
          stored.getConstantBound64(BoundType::LB, dimension);
      std::optional<int64_t> upper =
          stored.getConstantBound64(BoundType::UB, dimension);
      if (!lower || !upper)
        return failRectangle(
            IndexRelationStatus::Unsupported,
            "exact index demand has no finite static rectangular bounds");
      offsets[dimension] = std::min(offsets[dimension], *lower);
      inclusiveUpper[dimension] = std::max(inclusiveUpper[dimension], *upper);
    }
  }
  if (!sawNonEmptyDisjunct)
    return failRectangle(IndexRelationStatus::Unsupported,
                         "empty index demand has no transfer rectangle");

  llvm::SmallVector<int64_t, 4> sizes;
  sizes.reserve(rank);
  for (unsigned dimension = 0; dimension < rank; ++dimension) {
    int64_t size = 0;
    if (inclusiveUpper[dimension] < offsets[dimension] ||
        llvm::SubOverflow(inclusiveUpper[dimension], offsets[dimension],
                          size) ||
        llvm::AddOverflow(size, int64_t{1}, size))
      return failRectangle(IndexRelationStatus::Invalid,
                           "exact index demand rectangle overflows");
    sizes.push_back(size);
  }
  IndexSetResult rectangle =
      IndexRelation::staticRectangularDomain(offsets, sizes, limits);
  if (!rectangle.isExact())
    return failRectangle(rectangle.status, rectangle.reason);
  if (!set->isEqual(*rectangle.set))
    return failRectangle(
        IndexRelationStatus::Unsupported,
        "exact index demand is not one dense static rectangle");
  return StaticRectangularIndexSetResult{
      IndexRelationStatus::Exact,
      StaticRectangularIndexSet{std::move(offsets), std::move(sizes)},
      {}};
}

StaticRectangularIndexSetPiecesResult
IndexSetResult::getExactStaticRectangularDisjuncts(
    const IndexRelationLimits &limits) const {
  if (status != IndexRelationStatus::Exact || !set)
    return failRectanglePieces(
        status, reason.empty()
                    ? "rectangular disjunct recovery requires an exact set"
                    : reason);
  if (exceedsSetLimits(*set, limits))
    return failRectanglePieces(
        IndexRelationStatus::ResourceExhausted,
        "rectangular disjunct recovery exceeds index-set budget");

  const unsigned rank = set->getSpace().getNumSetDimVars();
  llvm::SmallVector<StaticRectangularIndexSet, 8> rectangles;
  rectangles.reserve(set->getNumDisjuncts());
  for (const IntegerRelation &disjunct : set->getAllDisjuncts()) {
    if (rank == 0 && disjunct.getNumConstraints() == 0) {
      rectangles.push_back(StaticRectangularIndexSet{{}, {}});
      continue;
    }
    StaticRectangularIndexSetResult rectangle =
        recoverDirectStaticRectangle(disjunct, rank);
    if (!rectangle.isExact())
      return failRectanglePieces(rectangle.status, rectangle.reason);
    rectangles.push_back(std::move(*rectangle.domain));
  }
  return StaticRectangularIndexSetPiecesResult{
      IndexRelationStatus::Exact, std::move(rectangles), {}};
}

IndexRelationResult
IndexRelation::compose(const IndexRelation &next,
                       const IndexRelationLimits &limits) const {
  if (getSourceRank() != next.getDestinationRank())
    return fail(IndexRelationStatus::Invalid,
                "index relation composition rank mismatch");
  if (exceedsVariableLimit(getDestinationRank(), next.getSourceRank(), limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "composed index relation exceeds variable budget");

  // If every intermediate coordinate is produced by one current row-major
  // group, substitute those groups into the next relation's reassociation.
  // This covers projected indexing followed by either collapse or expansion.
  // The total/bounded premises are essential: without them an intermediate
  // bound may clip the apparent map.
  std::optional<llvm::SmallVector<RowMajorRectangleMapping, 4>>
      composedMappings;
  const bool composedCanonicalRowMajorOrder =
      status == IndexRelationStatus::Exact &&
      next.status == IndexRelationStatus::Exact &&
      canonicalRowMajorOrderByConstruction &&
      next.canonicalRowMajorOrderByConstruction && rectangleDestinationShape &&
      rectangleSourceShape && next.rectangleDestinationShape &&
      next.rectangleSourceShape &&
      *rectangleSourceShape == *next.rectangleDestinationShape;
  if (status == IndexRelationStatus::Exact &&
      next.status == IndexRelationStatus::Exact &&
      totalBoundedAffineMapByConstruction &&
      next.totalBoundedAffineMapByConstruction && rowMajorRectangleMappings &&
      next.rowMajorRectangleMappings && rectangleDestinationShape &&
      rectangleSourceShape && next.rectangleDestinationShape &&
      next.rectangleSourceShape &&
      *rectangleSourceShape == *next.rectangleDestinationShape) {
    llvm::SmallVector<std::optional<unsigned>, 4> intermediateProducer(
        rectangleSourceShape->size());
    bool validMappings = true;
    for (auto [mappingIndex, mapping] :
         llvm::enumerate(*rowMajorRectangleMappings)) {
      if (mapping.sourceDimensions.empty())
        continue;
      if (mapping.sourceDimensions.size() != 1 ||
          mapping.sourceDimensions.front() >= intermediateProducer.size() ||
          intermediateProducer[mapping.sourceDimensions.front()]) {
        validMappings = false;
        break;
      }
      intermediateProducer[mapping.sourceDimensions.front()] = mappingIndex;
    }
    llvm::SmallVector<RowMajorRectangleMapping, 4> mappings;
    for (const RowMajorRectangleMapping &nextMapping :
         *next.rowMajorRectangleMappings) {
      RowMajorRectangleMapping mapping;
      mapping.sourceDimensions = nextMapping.sourceDimensions;
      for (unsigned intermediateDimension : nextMapping.destinationDimensions) {
        if (!validMappings ||
            intermediateDimension >= intermediateProducer.size() ||
            !intermediateProducer[intermediateDimension]) {
          validMappings = false;
          break;
        }
        llvm::append_range(mapping.destinationDimensions,
                           (*rowMajorRectangleMappings)
                               [*intermediateProducer[intermediateDimension]]
                                   .destinationDimensions);
      }
      if (!validMappings)
        break;
      mappings.push_back(std::move(mapping));
    }
    if (validMappings)
      composedMappings = std::move(mappings);
  }
  // mergeAndCompose substitutes integral local equalities while retaining
  // every intermediate bound. Plain PresburgerRelation::compose leaves these
  // locals existential, obscuring an otherwise affine view/access chain.
  // Its direction is next.mergeAndCompose(this): A->B followed by B->C.
  PresburgerRelation composed =
      PresburgerRelation::getEmpty(PresburgerSpace::getRelationSpace(
          getDestinationRank(), next.getSourceRank(),
          relation.getNumSymbolVars()));
  if (static_cast<uint64_t>(relation.getNumDisjuncts()) *
          next.relation.getNumDisjuncts() >
      limits.maxDisjuncts)
    return fail(IndexRelationStatus::ResourceExhausted,
                "composed index relation exceeds disjunct budget");
  for (const IntegerRelation &before : relation.getAllDisjuncts())
    for (const IntegerRelation &after : next.relation.getAllDisjuncts()) {
      IntegerRelation joined = after;
      if (before.getNumSymbolVars() == 0 && after.getNumSymbolVars() == 0) {
        // The pinned API requires identifier storage even with no symbols to
        // align. Empty identifiers carry no operation/value identity.
        IntegerRelation input = before;
        input.resetIds();
        joined.resetIds();
        joined.mergeAndCompose(input);
        auto space = joined.getSpace();
        space.disableIds();
        joined.setSpace(space);
      } else {
        // Symbolic relations retain their existing positional symbol contract.
        joined = before;
        joined.compose(after);
      }
      if (exceedsDisjunctWorkLimits(joined, limits))
        return fail(IndexRelationStatus::ResourceExhausted,
                    "composed index relation exceeds constraint budget");
      composed.unionInPlace(joined);
    }
  if (exceedsRelationLimits(composed, limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "composed index relation exceeds variable or disjunct budget");
  IndexRelationStatus composedStatus =
      status == IndexRelationStatus::Exact &&
              next.status == IndexRelationStatus::Exact
          ? IndexRelationStatus::Exact
          : IndexRelationStatus::SoundBound;
  IndexRelationResult result{
      composedStatus, IndexRelation(std::move(composed), composedStatus), {}};
  if (composedStatus == IndexRelationStatus::Exact) {
    result.relation->functionalByConstruction =
        functionalByConstruction && next.functionalByConstruction;
    result.relation->injectiveByConstruction =
        (injectiveByConstruction ||
         hasCanonicalRowMajorReshapeConstruction()) &&
        (next.injectiveByConstruction ||
         next.hasCanonicalRowMajorReshapeConstruction());
  }
  if (composedStatus == IndexRelationStatus::Exact && composedMappings) {
    result.relation->rowMajorRectangleMappings = std::move(composedMappings);
    result.relation->rectangleDestinationShape = rectangleDestinationShape;
    result.relation->rectangleSourceShape = next.rectangleSourceShape;
    result.relation->functionalByConstruction = true;
    result.relation->totalBoundedAffineMapByConstruction = true;
  }
  if (composedStatus == IndexRelationStatus::Exact &&
      composedCanonicalRowMajorOrder) {
    result.relation->rectangleDestinationShape = rectangleDestinationShape;
    result.relation->rectangleSourceShape = next.rectangleSourceShape;
    result.relation->functionalByConstruction = true;
    result.relation->canonicalRowMajorOrderByConstruction = true;
    result.relation->totalBoundedAffineMapByConstruction = true;
  }
  return result;
}

IndexRelationResult
IndexRelation::inverse(const IndexRelationLimits &limits) const {
  PresburgerRelation inverted = relation;
  inverted.inverse();
  if (exceedsRelationLimits(inverted, limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "inverse index relation exceeds variable or disjunct budget");
  IndexRelationResult result{
      status, IndexRelation(std::move(inverted), status), {}};
  result.relation->functionalByConstruction = injectiveByConstruction;
  result.relation->injectiveByConstruction = functionalByConstruction;
  if (status == IndexRelationStatus::Exact &&
      canonicalRowMajorOrderByConstruction && rectangleDestinationShape &&
      rectangleSourceShape) {
    result.relation->rectangleDestinationShape = rectangleSourceShape;
    result.relation->rectangleSourceShape = rectangleDestinationShape;
    if (rowMajorRectangleMappings) {
      llvm::SmallVector<RowMajorRectangleMapping, 4> invertedMappings;
      for (const RowMajorRectangleMapping &mapping : *rowMajorRectangleMappings)
        invertedMappings.push_back(RowMajorRectangleMapping{
            mapping.sourceDimensions, mapping.destinationDimensions});
      result.relation->rowMajorRectangleMappings = std::move(invertedMappings);
    }
    result.relation->functionalByConstruction = true;
    result.relation->canonicalRowMajorOrderByConstruction = true;
    result.relation->totalBoundedAffineMapByConstruction = true;
  }
  return result;
}

IndexRelationResult IndexRelation::intersectDestinationDomain(
    const PresburgerSet &domain, const IndexRelationLimits &limits) const {
  if (!isCompatibleSet(domain, getDestinationRank()))
    return fail(IndexRelationStatus::Invalid,
                "destination domain rank or symbols are incompatible");
  PresburgerRelation restricted = relation.intersectDomain(domain);
  if (exceedsRelationLimits(restricted, limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "destination-domain intersection exceeds budget");
  // An arbitrary restriction keeps single-valuedness but invalidates proofs
  // about the complete rectangular domain. Do not let a stale construction
  // pattern bypass the intersection.
  IndexRelation restrictedRelation(std::move(restricted), status);
  restrictedRelation.functionalByConstruction = functionalByConstruction;
  restrictedRelation.injectiveByConstruction =
      injectiveByConstruction || hasCanonicalRowMajorReshapeConstruction();
  if ((projectedRectanglePattern || rowMajorRectangleMappings) &&
      rectangleDestinationShape && rectangleSourceShape) {
    StaticRectangularIndexSetResult rectangle = IndexSetResult{
        IndexRelationStatus::Exact,
        domain,
        {}}.getExactStaticRectangularDomain(limits);
    if (rectangle.isExact() &&
        llvm::all_of(rectangle.domain->offsets,
                     [](int64_t offset) { return offset == 0; }) &&
        rectangle.domain->sizes == *rectangleDestinationShape) {
      restrictedRelation.projectedRectanglePattern = projectedRectanglePattern;
      restrictedRelation.rowMajorRectangleMappings = rowMajorRectangleMappings;
      restrictedRelation.rectangleDestinationShape = rectangleDestinationShape;
      restrictedRelation.rectangleSourceShape = rectangleSourceShape;
      restrictedRelation.canonicalRowMajorOrderByConstruction =
          canonicalRowMajorOrderByConstruction;
      restrictedRelation.totalBoundedAffineMapByConstruction =
          totalBoundedAffineMapByConstruction;
    }
  }
  return IndexRelationResult{status, std::move(restrictedRelation), {}};
}

IndexRelationResult
IndexRelation::intersectSourceDomain(const PresburgerSet &domain,
                                     const IndexRelationLimits &limits) const {
  if (!isCompatibleSet(domain, getSourceRank()))
    return fail(IndexRelationStatus::Invalid,
                "source domain rank or symbols are incompatible");
  PresburgerRelation restricted = relation.intersectRange(domain);
  if (exceedsRelationLimits(restricted, limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "source-domain intersection exceeds budget");
  // Range restriction likewise invalidates totality and the original
  // projected-rectangle proof while preserving single-valuedness.
  IndexRelation restrictedRelation(std::move(restricted), status);
  restrictedRelation.functionalByConstruction = functionalByConstruction;
  restrictedRelation.injectiveByConstruction =
      injectiveByConstruction || hasCanonicalRowMajorReshapeConstruction();
  if ((projectedRectanglePattern || rowMajorRectangleMappings) &&
      rectangleDestinationShape && rectangleSourceShape) {
    StaticRectangularIndexSetResult rectangle = IndexSetResult{
        IndexRelationStatus::Exact,
        domain,
        {}}.getExactStaticRectangularDomain(limits);
    if (rectangle.isExact() &&
        llvm::all_of(rectangle.domain->offsets,
                     [](int64_t offset) { return offset == 0; }) &&
        rectangle.domain->sizes == *rectangleSourceShape) {
      restrictedRelation.projectedRectanglePattern = projectedRectanglePattern;
      restrictedRelation.rowMajorRectangleMappings = rowMajorRectangleMappings;
      restrictedRelation.rectangleDestinationShape = rectangleDestinationShape;
      restrictedRelation.rectangleSourceShape = rectangleSourceShape;
      restrictedRelation.canonicalRowMajorOrderByConstruction =
          canonicalRowMajorOrderByConstruction;
      restrictedRelation.totalBoundedAffineMapByConstruction =
          totalBoundedAffineMapByConstruction;
    }
  }
  return IndexRelationResult{status, std::move(restrictedRelation), {}};
}

IndexSetResult IndexRelation::image(const PresburgerSet &destinationDomain,
                                    const IndexRelationLimits &limits) const {
  IndexRelationResult restricted =
      intersectDestinationDomain(destinationDomain, limits);
  if (!restricted.relation)
    return failSet(restricted.status, restricted.reason);
  PresburgerSet image = restricted.relation->relation.getRangeSet();
  if (exceedsSetLimits(image, limits))
    return failSet(IndexRelationStatus::ResourceExhausted,
                   "index relation image exceeds budget");
  return IndexSetResult{restricted.status, std::move(image), {}};
}

StaticRectangularIndexSetResult IndexRelation::getExactStaticRectangularImage(
    llvm::ArrayRef<int64_t> destinationOffsets,
    llvm::ArrayRef<int64_t> destinationSizes,
    const IndexRelationLimits &limits) const {
  if (status != IndexRelationStatus::Exact)
    return failRectangle(IndexRelationStatus::SoundBound,
                         "rectangular image requires an exact relation");
  if (destinationOffsets.size() != getDestinationRank() ||
      destinationSizes.size() != getDestinationRank())
    return failRectangle(IndexRelationStatus::Invalid,
                         "rectangular image destination rank is invalid");

  if (rowMajorRectangleMappings && rectangleDestinationShape &&
      rectangleSourceShape) {
    for (auto [offset, size, extent] :
         llvm::zip_equal(destinationOffsets, destinationSizes,
                         *rectangleDestinationShape)) {
      int64_t upper = 0;
      if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, upper) ||
          upper > extent)
        return failRectangle(IndexRelationStatus::Invalid,
                             "rectangular image destination is out of bounds");
    }

    StaticRectangularIndexSet result;
    result.offsets.assign(rectangleSourceShape->size(), 0);
    result.sizes.assign(rectangleSourceShape->size(), 0);
    llvm::SmallVector<bool, 4> coveredSource(rectangleSourceShape->size(),
                                             false);
    bool exactRectangle = true;
    for (const RowMajorRectangleMapping &mapping : *rowMajorRectangleMappings) {
      int64_t linearOffset = 0;
      int64_t destinationProduct = 1;
      for (unsigned destination :
           llvm::reverse(mapping.destinationDimensions)) {
        if (destination >= rectangleDestinationShape->size()) {
          exactRectangle = false;
          break;
        }
        int64_t contribution = 0;
        if (llvm::MulOverflow(destinationOffsets[destination],
                              destinationProduct, contribution) ||
            llvm::AddOverflow(linearOffset, contribution, linearOffset) ||
            llvm::MulOverflow(destinationProduct,
                              (*rectangleDestinationShape)[destination],
                              destinationProduct)) {
          exactRectangle = false;
          break;
        }
      }
      if (!exactRectangle)
        break;

      int64_t sourceProduct = 1;
      for (unsigned source : mapping.sourceDimensions) {
        if (source >= rectangleSourceShape->size() || coveredSource[source] ||
            (*rectangleSourceShape)[source] <= 0 ||
            llvm::MulOverflow(sourceProduct, (*rectangleSourceShape)[source],
                              sourceProduct)) {
          exactRectangle = false;
          break;
        }
      }
      if (!exactRectangle || destinationProduct != sourceProduct) {
        exactRectangle = false;
        break;
      }

      std::optional<size_t> firstVarying;
      for (auto [ordinal, destination] :
           llvm::enumerate(mapping.destinationDimensions))
        if (destinationSizes[destination] > 1) {
          firstVarying = ordinal;
          break;
        }
      int64_t linearSize = 1;
      if (firstVarying) {
        const unsigned varyingDestination =
            mapping.destinationDimensions[*firstVarying];
        int64_t innerExtent = 1;
        for (unsigned destination : llvm::drop_begin(
                 mapping.destinationDimensions, *firstVarying + 1)) {
          if (destinationOffsets[destination] != 0 ||
              destinationSizes[destination] !=
                  (*rectangleDestinationShape)[destination] ||
              llvm::MulOverflow(innerExtent,
                                (*rectangleDestinationShape)[destination],
                                innerExtent)) {
            exactRectangle = false;
            break;
          }
        }
        if (!exactRectangle ||
            llvm::MulOverflow(destinationSizes[varyingDestination], innerExtent,
                              linearSize)) {
          exactRectangle = false;
          break;
        }
      }
      int64_t linearLimit = 0;
      if (llvm::AddOverflow(linearOffset, linearSize, linearLimit) ||
          linearLimit > sourceProduct) {
        exactRectangle = false;
        break;
      }

      if (mapping.sourceDimensions.empty()) {
        if (linearOffset != 0 || linearSize != 1)
          exactRectangle = false;
        continue;
      }

      std::optional<size_t> varyingSource;
      int64_t varyingOffset = 0;
      int64_t varyingSize = 0;
      for (size_t candidate = 0; candidate < mapping.sourceDimensions.size();
           ++candidate) {
        int64_t innerExtent = 1;
        bool validInnerExtent = true;
        for (unsigned source :
             llvm::drop_begin(mapping.sourceDimensions, candidate + 1))
          if (llvm::MulOverflow(innerExtent, (*rectangleSourceShape)[source],
                                innerExtent)) {
            validInnerExtent = false;
            break;
          }
        if (!validInnerExtent || linearOffset % innerExtent != 0 ||
            linearSize % innerExtent != 0)
          continue;
        const unsigned source = mapping.sourceDimensions[candidate];
        const int64_t offset =
            (linearOffset / innerExtent) % (*rectangleSourceShape)[source];
        const int64_t size = linearSize / innerExtent;
        int64_t limit = 0;
        if (size > 0 && !llvm::AddOverflow(offset, size, limit) &&
            limit <= (*rectangleSourceShape)[source]) {
          varyingSource = candidate;
          varyingOffset = offset;
          varyingSize = size;
          break;
        }
      }
      if (!varyingSource) {
        exactRectangle = false;
        break;
      }

      int64_t sourceStride = sourceProduct;
      for (auto [ordinal, source] : llvm::enumerate(mapping.sourceDimensions)) {
        sourceStride /= (*rectangleSourceShape)[source];
        coveredSource[source] = true;
        if (ordinal < *varyingSource) {
          result.offsets[source] =
              (linearOffset / sourceStride) % (*rectangleSourceShape)[source];
          result.sizes[source] = 1;
        } else if (ordinal == *varyingSource) {
          result.offsets[source] = varyingOffset;
          result.sizes[source] = varyingSize;
        } else {
          result.offsets[source] = 0;
          result.sizes[source] = (*rectangleSourceShape)[source];
        }
      }
    }
    exactRectangle &=
        llvm::all_of(coveredSource, [](bool covered) { return covered; });
    if (exactRectangle)
      return StaticRectangularIndexSetResult{
          IndexRelationStatus::Exact, std::move(result), {}};
  }

  if (projectedRectanglePattern && rectangleDestinationShape &&
      rectangleSourceShape) {
    for (auto [offset, size, extent] :
         llvm::zip_equal(destinationOffsets, destinationSizes,
                         *rectangleDestinationShape)) {
      int64_t upper = 0;
      if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, upper) ||
          upper > extent)
        return failRectangle(IndexRelationStatus::Invalid,
                             "rectangular image destination is out of bounds");
    }
    StaticRectangularIndexSet result;
    result.offsets.reserve(projectedRectanglePattern->size());
    result.sizes.reserve(projectedRectanglePattern->size());
    bool sourceBoundsPreserveProjection = true;
    for (auto [sourceDimension, mappedDimension] :
         llvm::enumerate(*projectedRectanglePattern)) {
      const int64_t sourceExtent = (*rectangleSourceShape)[sourceDimension];
      if (mappedDimension == -2) {
        // Unconstrained source dimension of a complete-reduction relation:
        // the image covers the complete source extent for every destination
        // rectangle.
        if (sourceExtent <= 0) {
          sourceBoundsPreserveProjection = false;
          break;
        }
        result.offsets.push_back(0);
        result.sizes.push_back(sourceExtent);
        continue;
      }
      if (mappedDimension >= 0) {
        const unsigned position = static_cast<unsigned>(mappedDimension);
        int64_t upper = 0;
        if (llvm::AddOverflow(destinationOffsets[position],
                              destinationSizes[position], upper) ||
            destinationOffsets[position] < 0 || upper > sourceExtent) {
          sourceBoundsPreserveProjection = false;
          break;
        }
        result.offsets.push_back(destinationOffsets[position]);
        result.sizes.push_back(destinationSizes[position]);
        continue;
      }
      if (sourceExtent <= 0) {
        sourceBoundsPreserveProjection = false;
        break;
      }
      result.offsets.push_back(0);
      result.sizes.push_back(1);
    }
    if (sourceBoundsPreserveProjection)
      return StaticRectangularIndexSetResult{
          IndexRelationStatus::Exact, std::move(result), {}};
  }

  IndexSetResult destination =
      staticRectangularDomain(destinationOffsets, destinationSizes, limits);
  if (!destination.isExact())
    return failRectangle(destination.status, destination.reason);
  IndexSetResult exactImage = image(*destination.set, limits);
  return exactImage.getExactStaticRectangularDomain(limits);
}

StaticRectangularIndexSetPiecesResult
IndexRelation::getExactStaticRectangularImagePieces(
    llvm::ArrayRef<int64_t> destinationOffsets,
    llvm::ArrayRef<int64_t> destinationSizes,
    const IndexRelationLimits &limits) const {
  auto failPieces = [](IndexRelationStatus failureStatus,
                       llvm::StringRef reason) {
    return StaticRectangularIndexSetPiecesResult{
        failureStatus, {}, reason.str()};
  };
  if (status != IndexRelationStatus::Exact)
    return failPieces(IndexRelationStatus::SoundBound,
                      "rectangular image decomposition requires an exact "
                      "relation");
  if (destinationOffsets.size() != getDestinationRank() ||
      destinationSizes.size() != getDestinationRank())
    return failPieces(IndexRelationStatus::Invalid,
                      "rectangular image destination rank is invalid");
  if (!rowMajorRectangleMappings || !rectangleDestinationShape ||
      !rectangleSourceShape) {
    StaticRectangularIndexSetResult rectangle = getExactStaticRectangularImage(
        destinationOffsets, destinationSizes, limits);
    if (!rectangle.isExact())
      return failPieces(rectangle.status, rectangle.reason);
    return StaticRectangularIndexSetPiecesResult{
        IndexRelationStatus::Exact, {std::move(*rectangle.domain)}, {}};
  }

  for (auto [offset, size, extent] : llvm::zip_equal(
           destinationOffsets, destinationSizes, *rectangleDestinationShape)) {
    int64_t upper = 0;
    if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, upper) ||
        upper > extent)
      return failPieces(IndexRelationStatus::Invalid,
                        "rectangular image destination is out of bounds");
  }

  llvm::SmallVector<StaticRectangularIndexSet, 8> result(1);
  result.front().offsets.assign(rectangleSourceShape->size(), 0);
  result.front().sizes.assign(rectangleSourceShape->size(), 0);
  llvm::SmallVector<bool, 4> coveredSource(rectangleSourceShape->size(), false);

  for (const RowMajorRectangleMapping &mapping : *rowMajorRectangleMappings) {
    int64_t destinationProduct = 1;
    for (unsigned destination : mapping.destinationDimensions) {
      if (destination >= rectangleDestinationShape->size() ||
          (*rectangleDestinationShape)[destination] <= 0 ||
          llvm::MulOverflow(destinationProduct,
                            (*rectangleDestinationShape)[destination],
                            destinationProduct))
        return failPieces(IndexRelationStatus::Invalid,
                          "row-major destination group is invalid");
    }
    int64_t sourceProduct = 1;
    for (unsigned source : mapping.sourceDimensions) {
      if (source >= rectangleSourceShape->size() || coveredSource[source] ||
          (*rectangleSourceShape)[source] <= 0 ||
          llvm::MulOverflow(sourceProduct, (*rectangleSourceShape)[source],
                            sourceProduct))
        return failPieces(IndexRelationStatus::Invalid,
                          "row-major source group is invalid");
      coveredSource[source] = true;
    }
    if (destinationProduct != sourceProduct)
      return failPieces(IndexRelationStatus::Invalid,
                        "row-major group element counts differ");

    struct LinearInterval {
      int64_t offset = 0;
      int64_t size = 0;
    };
    llvm::SmallVector<LinearInterval, 8> linearIntervals;
    if (mapping.destinationDimensions.empty()) {
      linearIntervals.push_back({0, 1});
    } else {
      size_t contiguousDimension = mapping.destinationDimensions.size() - 1;
      for (size_t candidate = 0;
           candidate < mapping.destinationDimensions.size(); ++candidate) {
        const bool innerDimensionsAreComplete = llvm::all_of(
            llvm::drop_begin(mapping.destinationDimensions, candidate + 1),
            [&](unsigned destination) {
              return destinationOffsets[destination] == 0 &&
                     destinationSizes[destination] ==
                         (*rectangleDestinationShape)[destination];
            });
        if (innerDimensionsAreComplete) {
          contiguousDimension = candidate;
          break;
        }
      }

      uint64_t prefixCount = 1;
      for (unsigned destination :
           llvm::ArrayRef<unsigned>(mapping.destinationDimensions)
               .take_front(contiguousDimension)) {
        if (static_cast<uint64_t>(destinationSizes[destination]) >
            limits.maxRectangularPieces / prefixCount)
          return failPieces(IndexRelationStatus::ResourceExhausted,
                            "row-major image decomposition exceeds piece "
                            "budget");
        prefixCount *= static_cast<uint64_t>(destinationSizes[destination]);
      }
      for (uint64_t ordinal = 0; ordinal < prefixCount; ++ordinal) {
        llvm::SmallVector<int64_t, 4> coordinates(destinationOffsets.begin(),
                                                  destinationOffsets.end());
        uint64_t remainingOrdinal = ordinal;
        for (unsigned destination : llvm::reverse(
                 llvm::ArrayRef<unsigned>(mapping.destinationDimensions)
                     .take_front(contiguousDimension))) {
          const uint64_t extent =
              static_cast<uint64_t>(destinationSizes[destination]);
          coordinates[destination] +=
              static_cast<int64_t>(remainingOrdinal % extent);
          remainingOrdinal /= extent;
        }

        int64_t linearOffset = 0;
        int64_t stride = 1;
        for (unsigned destination :
             llvm::reverse(mapping.destinationDimensions)) {
          int64_t contribution = 0;
          if (llvm::MulOverflow(coordinates[destination], stride,
                                contribution) ||
              llvm::AddOverflow(linearOffset, contribution, linearOffset) ||
              llvm::MulOverflow(
                  stride, (*rectangleDestinationShape)[destination], stride))
            return failPieces(IndexRelationStatus::Invalid,
                              "row-major destination interval overflows");
        }
        const unsigned contiguousDestination =
            mapping.destinationDimensions[contiguousDimension];
        int64_t innerExtent = 1;
        for (unsigned destination : llvm::drop_begin(
                 mapping.destinationDimensions, contiguousDimension + 1))
          if (llvm::MulOverflow(innerExtent,
                                (*rectangleDestinationShape)[destination],
                                innerExtent))
            return failPieces(IndexRelationStatus::Invalid,
                              "row-major destination interval overflows");
        int64_t linearSize = 0;
        if (llvm::MulOverflow(destinationSizes[contiguousDestination],
                              innerExtent, linearSize))
          return failPieces(IndexRelationStatus::Invalid,
                            "row-major destination interval overflows");
        linearIntervals.push_back({linearOffset, linearSize});
      }
    }

    llvm::SmallVector<StaticRectangularIndexSet, 8> mappingPieces;
    for (LinearInterval interval : linearIntervals) {
      if (mapping.sourceDimensions.empty()) {
        if (interval.offset != 0 || interval.size != 1)
          return failPieces(IndexRelationStatus::Invalid,
                            "row-major dropped dimensions are not singleton");
        StaticRectangularIndexSet empty;
        empty.offsets.assign(rectangleSourceShape->size(), 0);
        empty.sizes.assign(rectangleSourceShape->size(), 0);
        mappingPieces.push_back(std::move(empty));
        continue;
      }

      int64_t position = interval.offset;
      int64_t remaining = interval.size;
      while (remaining > 0) {
        std::optional<size_t> selectedDimension;
        int64_t selectedBlock = 0;
        int64_t selectedOffset = 0;
        int64_t selectedSize = 0;
        for (size_t candidate = 0; candidate < mapping.sourceDimensions.size();
             ++candidate) {
          int64_t innerExtent = 1;
          for (unsigned source :
               llvm::drop_begin(mapping.sourceDimensions, candidate + 1))
            if (llvm::MulOverflow(innerExtent, (*rectangleSourceShape)[source],
                                  innerExtent))
              return failPieces(IndexRelationStatus::Invalid,
                                "row-major source interval overflows");
          if (position % innerExtent != 0 || remaining < innerExtent)
            continue;
          const unsigned source = mapping.sourceDimensions[candidate];
          const int64_t digit =
              (position / innerExtent) % (*rectangleSourceShape)[source];
          const int64_t count = std::min(
              (*rectangleSourceShape)[source] - digit, remaining / innerExtent);
          int64_t block = 0;
          if (count <= 0 || llvm::MulOverflow(count, innerExtent, block))
            continue;
          if (block > selectedBlock) {
            selectedDimension = candidate;
            selectedBlock = block;
            selectedOffset = digit;
            selectedSize = count;
          }
        }
        if (!selectedDimension || selectedBlock <= 0)
          return failPieces(IndexRelationStatus::Invalid,
                            "row-major source interval cannot be decomposed");
        if (mappingPieces.size() >= limits.maxRectangularPieces)
          return failPieces(IndexRelationStatus::ResourceExhausted,
                            "row-major image decomposition exceeds piece "
                            "budget");

        StaticRectangularIndexSet piece;
        piece.offsets.assign(rectangleSourceShape->size(), 0);
        piece.sizes.assign(rectangleSourceShape->size(), 0);
        int64_t sourceStride = sourceProduct;
        for (auto [ordinal, source] :
             llvm::enumerate(mapping.sourceDimensions)) {
          sourceStride /= (*rectangleSourceShape)[source];
          if (ordinal < *selectedDimension) {
            piece.offsets[source] =
                (position / sourceStride) % (*rectangleSourceShape)[source];
            piece.sizes[source] = 1;
          } else if (ordinal == *selectedDimension) {
            piece.offsets[source] = selectedOffset;
            piece.sizes[source] = selectedSize;
          } else {
            piece.offsets[source] = 0;
            piece.sizes[source] = (*rectangleSourceShape)[source];
          }
        }
        mappingPieces.push_back(std::move(piece));
        position += selectedBlock;
        remaining -= selectedBlock;
      }
    }

    if (!mapping.sourceDimensions.empty()) {
      if (mappingPieces.empty() ||
          result.size() > limits.maxRectangularPieces / mappingPieces.size())
        return failPieces(IndexRelationStatus::ResourceExhausted,
                          "row-major image decomposition exceeds piece "
                          "budget");
      llvm::SmallVector<StaticRectangularIndexSet, 8> combined;
      combined.reserve(result.size() * mappingPieces.size());
      for (const StaticRectangularIndexSet &base : result)
        for (const StaticRectangularIndexSet &piece : mappingPieces) {
          StaticRectangularIndexSet value = base;
          for (unsigned source : mapping.sourceDimensions) {
            value.offsets[source] = piece.offsets[source];
            value.sizes[source] = piece.sizes[source];
          }
          combined.push_back(std::move(value));
        }
      result = std::move(combined);
    }
  }

  if (!llvm::all_of(coveredSource, [](bool covered) { return covered; }))
    return failPieces(IndexRelationStatus::Invalid,
                      "row-major image leaves a source dimension undefined");
  return StaticRectangularIndexSetPiecesResult{
      IndexRelationStatus::Exact, std::move(result), {}};
}

IndexSetResult
IndexRelation::preimage(const PresburgerSet &sourceDomain,
                        const IndexRelationLimits &limits) const {
  if (!isCompatibleSet(sourceDomain, getSourceRank()))
    return failSet(IndexRelationStatus::Invalid,
                   "source domain rank or symbols are incompatible");
  if (projectedRectanglePattern && rectangleDestinationShape &&
      rectangleSourceShape) {
    StaticRectangularIndexSetResult rectangle = IndexSetResult{
        IndexRelationStatus::Exact,
        sourceDomain,
        {}}.getExactStaticRectangularDomain(limits);
    if (rectangle.isExact()) {
      const StaticRectangularIndexSet &source = *rectangle.domain;
      bool sourceRectangleIsInBounds = true;
      for (auto [offset, size, extent] : llvm::zip_equal(
               source.offsets, source.sizes, *rectangleSourceShape)) {
        int64_t upper = 0;
        if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, upper) ||
            upper > extent) {
          sourceRectangleIsInBounds = false;
          break;
        }
      }
      bool nonEmpty = true;
      bool destinationBoundsPreserveProjection = sourceRectangleIsInBounds;
      llvm::SmallVector<int64_t, 4> offsets(getDestinationRank(), 0);
      llvm::SmallVector<int64_t, 4> sizes(getDestinationRank(), 0);
      for (unsigned destinationDim = 0; destinationDim < getDestinationRank();
           ++destinationDim)
        sizes[destinationDim] = (*rectangleDestinationShape)[destinationDim];
      for (unsigned sourceDim = 0; sourceDim < getSourceRank(); ++sourceDim) {
        const int64_t mapped = (*projectedRectanglePattern)[sourceDim];
        if (mapped == -2) {
          // Unconstrained source dimension: the destination domain is fully
          // covered and the destination rectangle is the complete shape.
          continue;
        }
        if (mapped < 0) {
          if (!(source.offsets[sourceDim] <= 0 &&
                0 < source.offsets[sourceDim] + source.sizes[sourceDim]))
            nonEmpty = false;
          continue;
        }
        int64_t upper = 0;
        if (llvm::AddOverflow(source.offsets[sourceDim],
                              source.sizes[sourceDim], upper) ||
            source.offsets[sourceDim] < 0 ||
            upper > (*rectangleDestinationShape)[mapped]) {
          destinationBoundsPreserveProjection = false;
          break;
        }
        offsets[mapped] = source.offsets[sourceDim];
        sizes[mapped] = source.sizes[sourceDim];
      }
      if (destinationBoundsPreserveProjection && nonEmpty) {
        IndexSetResult preimage =
            staticRectangularDomain(offsets, sizes, limits);
        if (preimage.isExact())
          return preimage;
      } else if (destinationBoundsPreserveProjection) {
        PresburgerSet empty = PresburgerSet::getEmpty(
            PresburgerSpace::getSetSpace(getDestinationRank()));
        return IndexSetResult{IndexRelationStatus::Exact, std::move(empty), {}};
      }
    }
  }
  IndexRelationResult restricted = intersectSourceDomain(sourceDomain, limits);
  if (!restricted.relation)
    return failSet(restricted.status, restricted.reason);
  PresburgerSet preimage = restricted.relation->relation.getDomainSet();
  if (exceedsSetLimits(preimage, limits))
    return failSet(IndexRelationStatus::ResourceExhausted,
                   "index relation preimage exceeds budget");
  return IndexSetResult{restricted.status, std::move(preimage), {}};
}

IndexRelationQueryResult
IndexRelation::isFunctional(const IndexRelationLimits &limits) const {
  if (status != IndexRelationStatus::Exact)
    return failQuery(IndexRelationStatus::SoundBound,
                     "functionality requires an exact relation");
  if (functionalByConstruction)
    return IndexRelationQueryResult{IndexRelationStatus::Exact, true, {}};
  PresburgerRelation sharedDestination = relation;
  sharedDestination.inverse();
  sharedDestination.compose(relation);
  if (exceedsRelationLimits(sharedDestination, limits))
    return failQuery(IndexRelationStatus::ResourceExhausted,
                     "functionality proof exceeds budget");
  return IndexRelationQueryResult{
      IndexRelationStatus::Exact,
      sharedDestination.isSubsetOf(
          getUnboundedIdentityRelation(getSourceRank())),
      {}};
}

IndexRelationQueryResult
IndexRelation::isInjective(const IndexRelationLimits &limits) const {
  if (status != IndexRelationStatus::Exact)
    return failQuery(IndexRelationStatus::SoundBound,
                     "injectivity requires an exact relation");
  if (injectiveByConstruction || hasCanonicalRowMajorReshapeConstruction())
    return {IndexRelationStatus::Exact, true, {}};
  if (projectedRectanglePattern && rectangleDestinationShape) {
    const llvm::SmallVector<int64_t, 4> &pattern = *projectedRectanglePattern;
    const llvm::SmallVector<int64_t, 4> &destinationShape =
        *rectangleDestinationShape;
    // Closed-form injectivity for the rectangle pattern. An unconstrained
    // (-2) source dimension means every destination point covers the
    // complete source: injective only for a single-point destination. A
    // constant (-1) source dimension pins its value; the destination dims
    // it does not cover must then be extent one.
    if (llvm::is_contained(pattern, int64_t{-2}))
      return IndexRelationQueryResult{
          IndexRelationStatus::Exact,
          llvm::all_of(destinationShape,
                       [](int64_t extent) { return extent == 1; }),
          {}};
    llvm::SmallVector<int64_t, 4> coverage(destinationShape.size(), 0);
    for (int64_t mapped : pattern) {
      if (mapped < 0)
        continue;
      if (static_cast<size_t>(mapped) >= coverage.size())
        return failQuery(IndexRelationStatus::SoundBound,
                         "rectangle pattern is out of destination rank");
      ++coverage[mapped];
    }
    for (auto [dimension, count] : llvm::enumerate(coverage)) {
      if (count > 1)
        return IndexRelationQueryResult{IndexRelationStatus::Exact, false, {}};
      if (count == 0 && destinationShape[dimension] != 1)
        return IndexRelationQueryResult{IndexRelationStatus::Exact, false, {}};
    }
    return IndexRelationQueryResult{IndexRelationStatus::Exact, true, {}};
  }
  PresburgerRelation inverseRelation = relation;
  inverseRelation.inverse();
  PresburgerRelation sharedSource = relation;
  sharedSource.compose(inverseRelation);
  if (exceedsRelationLimits(sharedSource, limits))
    return failQuery(IndexRelationStatus::ResourceExhausted,
                     "injectivity proof exceeds budget");
  return IndexRelationQueryResult{
      IndexRelationStatus::Exact,
      sharedSource.isSubsetOf(
          getUnboundedIdentityRelation(getDestinationRank())),
      {}};
}

IndexRelationQueryResult
IndexRelation::isBijective(const IndexRelationLimits &limits) const {
  IndexRelationQueryResult functional = isFunctional(limits);
  if (functional.status != IndexRelationStatus::Exact || !functional.value)
    return functional;
  IndexRelationQueryResult injective = isInjective(limits);
  if (injective.status != IndexRelationStatus::Exact || !injective.value)
    return injective;
  return IndexRelationQueryResult{IndexRelationStatus::Exact, true, {}};
}

IndexRelationQueryResult
IndexRelation::isEquivalentTo(const IndexRelation &other,
                              const IndexRelationLimits &limits) const {
  if (getDestinationRank() != other.getDestinationRank() ||
      getSourceRank() != other.getSourceRank())
    return failQuery(IndexRelationStatus::Invalid,
                     "equivalence requires matching relation ranks");
  if (exceedsRelationLimits(relation, limits) ||
      exceedsRelationLimits(other.relation, limits))
    return failQuery(IndexRelationStatus::ResourceExhausted,
                     "equivalence proof exceeds budget");
  if (status != IndexRelationStatus::Exact ||
      other.status != IndexRelationStatus::Exact)
    return failQuery(IndexRelationStatus::SoundBound,
                     "equivalence requires exact relations");
  if (relation.isObviouslyEqual(other.relation))
    return {IndexRelationStatus::Exact, true, {}};
  if (projectedRectanglePattern && other.projectedRectanglePattern &&
      rectangleDestinationShape && other.rectangleDestinationShape &&
      rectangleSourceShape && other.rectangleSourceShape &&
      *projectedRectanglePattern == *other.projectedRectanglePattern &&
      *rectangleDestinationShape == *other.rectangleDestinationShape &&
      *rectangleSourceShape == *other.rectangleSourceShape)
    return IndexRelationQueryResult{IndexRelationStatus::Exact, true, {}};
  return IndexRelationQueryResult{
      IndexRelationStatus::Exact, relation.isEqual(other.relation), {}};
}

IndexRelationQueryResult
IndexRelation::implies(const IndexRelation &other,
                       const IndexRelationLimits &limits) const {
  if (getDestinationRank() != other.getDestinationRank() ||
      getSourceRank() != other.getSourceRank())
    return failQuery(IndexRelationStatus::Invalid,
                     "implication requires matching relation ranks");
  if (exceedsRelationLimits(relation, limits) ||
      exceedsRelationLimits(other.relation, limits))
    return failQuery(IndexRelationStatus::ResourceExhausted,
                     "implication proof exceeds budget");
  if (status != IndexRelationStatus::Exact ||
      other.status != IndexRelationStatus::Exact)
    return failQuery(IndexRelationStatus::SoundBound,
                     "implication requires exact relations");
  return IndexRelationQueryResult{
      IndexRelationStatus::Exact, relation.isSubsetOf(other.relation), {}};
}

std::optional<CanonicalReshapeRelations>
getCanonicalReshapeRelations(mlir::MLIRContext *context,
                             llvm::ArrayRef<int64_t> sourceShape,
                             llvm::ArrayRef<int64_t> destinationShape) {
  if (!context)
    return std::nullopt;

  auto getPrefixProducts = [](llvm::ArrayRef<int64_t> shape)
      -> std::optional<llvm::SmallVector<int64_t, 4>> {
    llvm::SmallVector<int64_t, 4> prefixes{1};
    int64_t product = 1;
    for (int64_t extent : shape) {
      if (extent <= 0 || llvm::MulOverflow(product, extent, product))
        return std::nullopt;
      prefixes.push_back(product);
    }
    return prefixes;
  };
  std::optional<llvm::SmallVector<int64_t, 4>> sourcePrefixes =
      getPrefixProducts(sourceShape);
  std::optional<llvm::SmallVector<int64_t, 4>> destinationPrefixes =
      getPrefixProducts(destinationShape);
  if (!sourcePrefixes || !destinationPrefixes ||
      sourcePrefixes->back() != destinationPrefixes->back())
    return std::nullopt;

  llvm::SmallVector<int64_t, 8> boundaries(sourcePrefixes->begin(),
                                           sourcePrefixes->end());
  boundaries.append(destinationPrefixes->begin(), destinationPrefixes->end());
  llvm::sort(boundaries);
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                   boundaries.end());
  llvm::SmallVector<int64_t, 4> iterationShape;
  iterationShape.reserve(boundaries.size() - 1);
  for (size_t index = 1; index < boundaries.size(); ++index) {
    if (boundaries[index - 1] <= 0 ||
        boundaries[index] % boundaries[index - 1] != 0)
      return std::nullopt;
    iterationShape.push_back(boundaries[index] / boundaries[index - 1]);
  }

  auto buildMap =
      [&](llvm::ArrayRef<int64_t> shape,
          llvm::ArrayRef<int64_t> prefixes) -> std::optional<mlir::AffineMap> {
    llvm::SmallVector<mlir::AffineExpr, 4> results;
    results.reserve(shape.size());
    for (size_t logicalDim = 0; logicalDim < shape.size(); ++logicalDim) {
      if (shape[logicalDim] == 1) {
        results.push_back(mlir::getAffineConstantExpr(0, context));
        continue;
      }
      auto beginIt = llvm::find(boundaries, prefixes[logicalDim]);
      auto endIt = llvm::find(boundaries, prefixes[logicalDim + 1]);
      if (beginIt == boundaries.end() || endIt == boundaries.end() ||
          beginIt >= endIt)
        return std::nullopt;
      size_t begin = std::distance(boundaries.begin(), beginIt);
      size_t end = std::distance(boundaries.begin(), endIt);
      mlir::AffineExpr expression = mlir::getAffineConstantExpr(0, context);
      for (size_t axis = begin; axis < end; ++axis)
        expression = expression * iterationShape[axis] +
                     mlir::getAffineDimExpr(axis, context);
      results.push_back(expression);
    }
    return mlir::AffineMap::get(iterationShape.size(), 0, results, context);
  };

  std::optional<mlir::AffineMap> sourceMap =
      buildMap(sourceShape, *sourcePrefixes);
  std::optional<mlir::AffineMap> destinationMap =
      buildMap(destinationShape, *destinationPrefixes);
  if (!sourceMap || !destinationMap)
    return std::nullopt;
  IndexRelationResult sourceRelation =
      IndexRelation::fromAffineMap(*sourceMap, iterationShape, sourceShape);
  IndexRelationResult destinationRelation = IndexRelation::fromAffineMap(
      *destinationMap, iterationShape, destinationShape);
  if (!sourceRelation.isExact() || !destinationRelation.isExact())
    return std::nullopt;
  return CanonicalReshapeRelations{std::move(iterationShape),
                                   std::move(*sourceRelation.relation),
                                   std::move(*destinationRelation.relation)};
}

} // namespace wafer::analysis
