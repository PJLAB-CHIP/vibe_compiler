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
    WaferPhysicalEncodingAttrInterface encoding,
    const std::optional<WaferStaticPhysicalOffsetCalculator> &calculator) {
  if (!calculator)
    return encoding.getPhysicalElementSpan(type, logicalPoint);
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

  auto encoding = mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
      endpointType.getMemorySpace());
  if (!encoding)
    return mlir::failure();
  mlir::FailureOr<int64_t> footprint =
      encoding.getPhysicalFootprintBytes(endpointType);
  mlir::FailureOr<int64_t> alignment =
      encoding.getMinimumAlignmentBytes(endpointType);
  mlir::FailureOr<int64_t> valid = encoding.getValidElementCount(endpointType);
  mlir::FailureOr<int64_t> padding =
      encoding.getPaddingElementCount(endpointType);
  if (mlir::failed(footprint) || mlir::failed(alignment) ||
      mlir::failed(valid) || mlir::failed(padding) || *footprint < 0 ||
      *alignment <= 0 || *valid < 0 || *padding < 0)
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

  return PhysicalAccessRelation(
      endpointType, llvm::SmallVector<int64_t, 4>(iterationShape),
      iterationToLogical, encoding,
      WaferStaticPhysicalOffsetCalculator::create(endpointType),
      projectedAffineMap, canonicalLinearOrder, *footprint, *alignment, *valid,
      *padding);
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
PhysicalAccessRelation::getLogicalPoint(
    llvm::ArrayRef<int64_t> iterationPoint) const {
  if (!isPointInShape(iterationPoint, iterationShape))
    return mlir::failure();

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
  return getEndpointSpan(endpointType, *logical, encoding, offsetCalculator);
}

} // namespace wafer::analysis
