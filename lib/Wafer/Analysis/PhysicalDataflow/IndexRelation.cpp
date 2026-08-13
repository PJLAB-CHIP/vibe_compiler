//===- IndexRelation.cpp - MLIR-backed logical index relations -----------===//

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

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

static bool exceedsRelationLimits(const PresburgerRelation &relation,
                                  const IndexRelationLimits &limits) {
  return relation.getNumVars() > limits.maxVariables ||
         relation.getNumDisjuncts() > limits.maxDisjuncts;
}

static bool exceedsSetLimits(const PresburgerSet &set,
                             const IndexRelationLimits &limits) {
  return set.getNumVars() > limits.maxVariables ||
         set.getNumDisjuncts() > limits.maxDisjuncts;
}

static bool isCompatibleSet(const PresburgerSet &set, unsigned rank) {
  const PresburgerSpace &space = set.getSpace();
  return space.getNumDomainVars() == 0 && space.getNumSetDimVars() == rank &&
         space.getNumSymbolVars() == 0;
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
  MLIRContext context;
  AffineMap map = AffineMap::getMultiDimIdentityMap(shape.size(), &context);
  return fromAffineMap(map, shape, shape, limits);
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
  return finishExactOrBound(std::move(relation), isBound, limits);
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
        IndexRelationResult projectedRelation = fromAffineMap(
            projected, destinationShape, sourceShape, limits);
        if (projectedRelation.isExact() &&
            projectedRelation.relation
                ->isEquivalentTo(*result.relation, limits)
                .isProvenTrue()) {
          result.relation->projectedRectanglePattern = std::move(pattern);
          result.relation->projectedRectangleDestinationShape =
              llvm::SmallVector<int64_t, 4>(destinationShape);
        }
      }
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

  IntegerRelation relation(PresburgerSpace::getRelationSpace(
      destinationShape.size(), sourceShape.size()));
  llvm::SmallVector<int64_t, 8> equality(relation.getNumVars() + 1, 0);
  for (auto [index, stride] : llvm::enumerate(*destinationStrides))
    equality[index] = stride;
  for (auto [index, stride] : llvm::enumerate(*sourceStrides))
    equality[destinationShape.size() + index] = -stride;
  relation.addEquality(equality);
  addStaticShapeBounds(relation, destinationShape, sourceShape);
  return finishExactOrBound(std::move(relation), /*isBound=*/false, limits);
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
  if (set->isIntegerEmpty())
    return failRectangle(IndexRelationStatus::Unsupported,
                         "empty index demand has no transfer rectangle");
  if (set->getNumVars() > limits.maxVariables ||
      set->getNumDisjuncts() > limits.maxDisjuncts)
    return failRectangle(IndexRelationStatus::ResourceExhausted,
                         "rectangular recovery exceeds index-set budget");

  const unsigned rank = set->getSpace().getNumSetDimVars();
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

IndexRelationResult
IndexRelation::compose(const IndexRelation &next,
                       const IndexRelationLimits &limits) const {
  if (getSourceRank() != next.getDestinationRank())
    return fail(IndexRelationStatus::Invalid,
                "index relation composition rank mismatch");
  if (exceedsVariableLimit(getDestinationRank(), next.getSourceRank(), limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "composed index relation exceeds variable budget");
  PresburgerRelation composed = relation;
  composed.compose(next.relation);
  if (exceedsRelationLimits(composed, limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "composed index relation exceeds variable or disjunct budget");
  IndexRelationStatus composedStatus =
      status == IndexRelationStatus::Exact &&
              next.status == IndexRelationStatus::Exact
          ? IndexRelationStatus::Exact
          : IndexRelationStatus::SoundBound;
  return IndexRelationResult{
      composedStatus, IndexRelation(std::move(composed), composedStatus), {}};
}

IndexRelationResult
IndexRelation::inverse(const IndexRelationLimits &limits) const {
  PresburgerRelation inverted = relation;
  inverted.inverse();
  if (exceedsRelationLimits(inverted, limits))
    return fail(IndexRelationStatus::ResourceExhausted,
                "inverse index relation exceeds variable or disjunct budget");
  return IndexRelationResult{
      status, IndexRelation(std::move(inverted), status), {}};
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
  return IndexRelationResult{
      status, IndexRelation(std::move(restricted), status), {}};
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
  return IndexRelationResult{
      status, IndexRelation(std::move(restricted), status), {}};
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

  if (projectedRectanglePattern && projectedRectangleDestinationShape) {
    for (auto [offset, size, extent] :
         llvm::zip_equal(destinationOffsets, destinationSizes,
                         *projectedRectangleDestinationShape)) {
      int64_t upper = 0;
      if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, upper) ||
          upper > extent)
        return failRectangle(IndexRelationStatus::Invalid,
                             "rectangular image destination is out of bounds");
    }
    StaticRectangularIndexSet result;
    result.offsets.reserve(projectedRectanglePattern->size());
    result.sizes.reserve(projectedRectanglePattern->size());
    for (int64_t mappedDimension : *projectedRectanglePattern) {
      if (mappedDimension >= 0) {
        const unsigned position = static_cast<unsigned>(mappedDimension);
        result.offsets.push_back(destinationOffsets[position]);
        result.sizes.push_back(destinationSizes[position]);
        continue;
      }
      result.offsets.push_back(0);
      result.sizes.push_back(1);
    }
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

IndexSetResult
IndexRelation::preimage(const PresburgerSet &sourceDomain,
                        const IndexRelationLimits &limits) const {
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
