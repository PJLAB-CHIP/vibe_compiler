//===- PhysicalAccessRelation.cpp - Logical-to-physical access ----------===//

#include "Wafer/Analysis/PhysicalDataflow/PhysicalAccessRelation.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::presburger;

namespace wafer::analysis {
namespace {

static PresburgerSet getPointSet(llvm::ArrayRef<int64_t> point) {
  IntegerPolyhedron set(PresburgerSpace::getSetSpace(point.size()));
  for (auto [index, value] : llvm::enumerate(point))
    set.addBound(BoundType::EQ, index, value);
  return PresburgerSet(set);
}

static bool isPointInShape(llvm::ArrayRef<int64_t> point,
                           llvm::ArrayRef<int64_t> shape) {
  if (point.size() != shape.size())
    return false;
  return llvm::all_of(llvm::zip_equal(point, shape), [](auto pair) {
    auto [index, dim] = pair;
    return dim >= 0 && index >= 0 && index < dim;
  });
}

static std::optional<int64_t> linearize(llvm::ArrayRef<int64_t> point,
                                        llvm::ArrayRef<int64_t> shape) {
  if (!isPointInShape(point, shape))
    return std::nullopt;
  int64_t linear = 0;
  for (auto [index, dim] : llvm::zip_equal(point, shape)) {
    if (llvm::MulOverflow(linear, dim, linear) ||
        llvm::AddOverflow(linear, index, linear))
      return std::nullopt;
  }
  return linear;
}

static std::optional<llvm::SmallVector<int64_t, 4>>
delinearize(int64_t linear, llvm::ArrayRef<int64_t> shape) {
  if (linear < 0)
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> point(shape.size(), 0);
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    if (shape[dim] <= 0)
      return std::nullopt;
    point[dim] = linear % shape[dim];
    linear /= shape[dim];
  }
  if (linear != 0)
    return std::nullopt;
  return point;
}

static mlir::FailureOr<WaferPhysicalElementSpan> getEndpointSpan(
    mlir::MemRefType type, llvm::ArrayRef<int64_t> logicalPoint,
    const PhysicalLayoutRelation &physicalLayout,
    const std::optional<WaferStaticPhysicalOffsetCalculator> &calculator) {
  if (!calculator)
    return physicalLayout.getPhysicalElementSpan(logicalPoint);
  int64_t bitOffset = 0;
  int64_t bitLength = 0;
  if (llvm::MulOverflow(calculator->getByteOffsetForValidIndices(logicalPoint),
                        int64_t{8}, bitOffset) ||
      llvm::MulOverflow(calculator->getInfo().elementBytes, int64_t{8},
                        bitLength) ||
      bitLength <= 0)
    return mlir::failure();
  return WaferPhysicalElementSpan{bitOffset, bitLength};
}

} // namespace

mlir::FailureOr<PhysicalAccessRelation> PhysicalAccessRelation::create(
    mlir::MemRefType endpointType, llvm::ArrayRef<int64_t> iterationShape,
    const IndexRelation &iterationToLogical, bool requireInjective) {
  if (!endpointType || !endpointType.hasStaticShape() ||
      iterationToLogical.getStatus() != IndexRelationStatus::Exact ||
      iterationToLogical.getDestinationRank() != iterationShape.size() ||
      iterationToLogical.getSourceRank() !=
          static_cast<unsigned>(endpointType.getRank()) ||
      !iterationToLogical.isFunctional().isProvenTrue() ||
      (requireInjective && !iterationToLogical.isInjective().isProvenTrue()))
    return mlir::failure();

  IndexSetResult iterationDomain = IndexRelation::staticDomain(iterationShape);
  IndexSetResult endpointDomain =
      IndexRelation::staticDomain(endpointType.getShape());
  if (!iterationDomain.isExact() || !endpointDomain.isExact())
    return mlir::failure();
  PresburgerSet relationDomain =
      iterationToLogical.getPresburgerRelation().getDomainSet();
  PresburgerSet relationRange =
      iterationToLogical.getPresburgerRelation().getRangeSet();
  if (!relationDomain.isEqual(*iterationDomain.set) ||
      !relationRange.isSubsetOf(*endpointDomain.set))
    return mlir::failure();

  mlir::FailureOr<PhysicalLayoutRelation> physicalLayout =
      PhysicalLayoutRelation::create(endpointType);
  if (mlir::failed(physicalLayout))
    return mlir::failure();
  IndexRelationResult physicalBitOffsets = iterationToLogical.compose(
      physicalLayout->getLogicalToPhysicalBitOffset());
  IndexRelationResult physicalElementOrdinals = iterationToLogical.compose(
      physicalLayout->getLogicalToPhysicalElementOrdinal());
  // Both component relations are exact functions. The encoding interface owns
  // non-overlap of valid physical element spans, so writer injectivity is
  // already established by the logical relation check above. Re-proving the
  // composed blocked relation with a generic solver here is both redundant and
  // a candidate-hot-path scalability hazard.
  if (!physicalBitOffsets.isExact() || !physicalElementOrdinals.isExact())
    return mlir::failure();

  mlir::AffineMap projectedAffineMap =
      iterationToLogical.getProjectedAffineMap(endpointType.getContext())
          .value_or(mlir::AffineMap{});
  bool canonicalLinearOrder = false;
  if (!projectedAffineMap) {
    IndexRelationResult canonical =
        IndexRelation::staticReshape(iterationShape, endpointType.getShape());
    if (canonical.isExact())
      canonicalLinearOrder =
          iterationToLogical.isEquivalentTo(*canonical.get()).isProvenTrue();
  }

  const int64_t footprint = physicalLayout->getPhysicalFootprintBytes();
  const int64_t alignment = physicalLayout->getMinimumAlignmentBytes();
  const int64_t valid = physicalLayout->getValidElementCount();
  const int64_t padding = physicalLayout->getPaddingElementCount();

  return PhysicalAccessRelation(
      endpointType, llvm::SmallVector<int64_t, 4>(iterationShape),
      iterationToLogical, std::move(*physicalLayout),
      std::move(*physicalBitOffsets.relation),
      std::move(*physicalElementOrdinals.relation),
      WaferStaticPhysicalOffsetCalculator::create(endpointType),
      projectedAffineMap, canonicalLinearOrder, footprint, alignment, valid,
      padding);
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
PhysicalAccessRelation::getLogicalPoint(
    llvm::ArrayRef<int64_t> iterationPoint) const {
  if (!isPointInShape(iterationPoint, iterationShape))
    return mlir::failure();

  // A rank-zero endpoint has exactly one logical point.  Preserve the
  // relation membership check, but do not route the empty coordinate through
  // affine-map recovery or Presburger sample projection: both representations
  // legitimately contain zero result variables, which is otherwise
  // indistinguishable from a missing projection in their convenience APIs.
  if (endpointType.getRank() == 0) {
    if (!iterationToLogical.contains(iterationPoint, {}))
      return mlir::failure();
    return llvm::SmallVector<int64_t, 4>{};
  }

  if (projectedAffineMap) {
    mlir::IndexType indexType = mlir::IndexType::get(endpointType.getContext());
    llvm::SmallVector<mlir::Attribute, 4> operands;
    operands.reserve(iterationPoint.size());
    for (int64_t index : iterationPoint)
      operands.push_back(mlir::IntegerAttr::get(indexType, index));
    llvm::SmallVector<mlir::Attribute, 4> folded;
    if (mlir::failed(projectedAffineMap.constantFold(operands, folded)) ||
        folded.size() !=
            static_cast<size_t>(projectedAffineMap.getNumResults()))
      return mlir::failure();
    llvm::SmallVector<int64_t, 4> logical;
    logical.reserve(folded.size());
    for (mlir::Attribute result : folded) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(result);
      if (!integer)
        return mlir::failure();
      logical.push_back(integer.getInt());
    }
    return logical;
  }

  if (canonicalLinearOrder) {
    std::optional<int64_t> linear = linearize(iterationPoint, iterationShape);
    if (!linear)
      return mlir::failure();
    std::optional<llvm::SmallVector<int64_t, 4>> logical =
        delinearize(*linear, endpointType.getShape());
    if (!logical)
      return mlir::failure();
    return std::move(*logical);
  }

  PresburgerRelation restricted =
      iterationToLogical.getPresburgerRelation().intersectDomain(
          getPointSet(iterationPoint));
  llvm::SmallVector<llvm::DynamicAPInt, 8> sample;
  if (!restricted.findIntegerSample(sample))
    return mlir::failure();
  const unsigned destinationRank = iterationToLogical.getDestinationRank();
  const unsigned sourceRank = iterationToLogical.getSourceRank();
  if (sample.size() < destinationRank + sourceRank)
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> logical;
  logical.reserve(sourceRank);
  for (unsigned index = 0; index < sourceRank; ++index)
    logical.push_back(static_cast<int64_t>(sample[destinationRank + index]));
  return logical;
}

mlir::FailureOr<WaferPhysicalElementSpan>
PhysicalAccessRelation::getPhysicalElementSpan(
    llvm::ArrayRef<int64_t> iterationPoint) const {
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> logical =
      getLogicalPoint(iterationPoint);
  if (mlir::failed(logical))
    return mlir::failure();
  return getEndpointSpan(endpointType, *logical, physicalLayout,
                         offsetCalculator);
}

IndexRelationQueryResult PhysicalAccessRelation::hasSamePhysicalElementMapping(
    const PhysicalAccessRelation &other) const {
  if (iterationShape != other.iterationShape)
    return IndexRelationQueryResult{IndexRelationStatus::Invalid, std::nullopt,
                                    "physical mappings require the same "
                                    "iteration domain"};
  if (physicalLayout.getElementBitWidth() !=
      other.physicalLayout.getElementBitWidth())
    return IndexRelationQueryResult{IndexRelationStatus::Exact, false, {}};
  return iterationToPhysicalBitOffset.isEquivalentTo(
      other.iterationToPhysicalBitOffset);
}

IndexRelationQueryResult PhysicalAccessRelation::hasSamePhysicalTraversal(
    const PhysicalAccessRelation &other) const {
  if (iterationShape != other.iterationShape)
    return IndexRelationQueryResult{IndexRelationStatus::Invalid, std::nullopt,
                                    "physical traversals require the same "
                                    "iteration domain"};
  // CT traverses the destination footprint. The source may have a longer
  // physical tail (notably a packed i1 mask), but it must contain every
  // destination ordinal that the instruction can read. Exact ordinal
  // equivalence over the valid iteration domain separately proves that the
  // corresponding logical elements line up.
  if (physicalLayout.getPhysicalElementCount() <
      other.physicalLayout.getPhysicalElementCount())
    return IndexRelationQueryResult{IndexRelationStatus::Exact, false, {}};
  return iterationToPhysicalElementOrdinal.isEquivalentTo(
      other.iterationToPhysicalElementOrdinal);
}

} // namespace wafer::analysis
