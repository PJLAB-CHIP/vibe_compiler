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

} // namespace wafer::analysis
