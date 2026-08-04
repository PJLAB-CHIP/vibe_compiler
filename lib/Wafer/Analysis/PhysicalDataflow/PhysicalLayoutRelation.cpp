//===- PhysicalLayoutRelation.cpp - Encoding-owned physical map ----------===//

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::presburger;

namespace wafer::analysis {
namespace {

static std::optional<PresburgerSet>
getPieceDomain(llvm::ArrayRef<int64_t> shape,
               const WaferPhysicalLayoutPiece &piece) {
  if (piece.logicalLowerBounds.size() != shape.size() ||
      piece.logicalUpperBounds.size() != shape.size() ||
      piece.logicalTilePeriods.size() != shape.size())
    return std::nullopt;
  IntegerPolyhedron domain(PresburgerSpace::getSetSpace(shape.size()));
  for (auto [dim, bounds] : llvm::enumerate(llvm::zip_equal(
           piece.logicalLowerBounds, piece.logicalUpperBounds))) {
    auto [lower, upper] = bounds;
    if (shape[dim] < 0 || lower < 0 || upper < lower || upper > shape[dim] ||
        piece.logicalTilePeriods[dim] < 0)
      return std::nullopt;
    domain.addBound(BoundType::LB, dim, lower);
    domain.addBound(BoundType::UB, dim, upper - 1);
  }
  return PresburgerSet(domain);
}

static bool exceedsLimits(const PresburgerRelation &relation,
                          const IndexRelationLimits &limits) {
  return relation.getNumVars() > limits.maxVariables ||
         relation.getNumDisjuncts() > limits.maxDisjuncts;
}

} // namespace

mlir::FailureOr<PhysicalLayoutRelation>
PhysicalLayoutRelation::create(mlir::MemRefType type,
                               const IndexRelationLimits &limits) {
  if (!type || !type.hasStaticShape())
    return mlir::failure();
  auto encoding = mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
      type.getMemorySpace());
  if (!encoding)
    return mlir::failure();

  mlir::FailureOr<int64_t> footprint = encoding.getPhysicalFootprintBytes(type);
  mlir::FailureOr<int64_t> alignment = encoding.getMinimumAlignmentBytes(type);
  mlir::FailureOr<int64_t> valid = encoding.getValidElementCount(type);
  mlir::FailureOr<int64_t> padding = encoding.getPaddingElementCount(type);
  mlir::FailureOr<int64_t> elementBits =
      encoding.getPhysicalElementBitWidth(type);
  mlir::FailureOr<llvm::SmallVector<WaferPhysicalLayoutPiece, 2>> pieces =
      encoding.getPhysicalLayoutPieces(type);
  if (mlir::failed(footprint) || mlir::failed(alignment) ||
      mlir::failed(valid) || mlir::failed(padding) ||
      mlir::failed(elementBits) || mlir::failed(pieces) || *footprint < 0 ||
      *alignment <= 0 || *valid < 0 || *padding < 0 || *elementBits <= 0)
    return mlir::failure();

  int64_t footprintBits = 0;
  if (llvm::MulOverflow(*footprint, int64_t{8}, footprintBits))
    return mlir::failure();
  int64_t offsetDomainSize = 0;
  if (*valid > 0) {
    if (footprintBits < *elementBits ||
        llvm::AddOverflow(footprintBits - *elementBits, int64_t{1},
                          offsetDomainSize) ||
        offsetDomainSize <= 0)
      return mlir::failure();
  }

  std::optional<PresburgerRelation> combined;
  llvm::SmallVector<PresburgerSet, 2> pieceDomains;
  bool byteAddressable = *elementBits % 8 == 0;
  for (const WaferPhysicalLayoutPiece &piece : *pieces) {
    if (!piece.logicalToPhysicalBitOffset ||
        piece.logicalToPhysicalBitOffset.getNumDims() !=
            static_cast<unsigned>(type.getRank()) ||
        piece.logicalToPhysicalBitOffset.getNumSymbols() != 0 ||
        piece.logicalToPhysicalBitOffset.getNumResults() != 1)
      return mlir::failure();
    byteAddressable &=
        piece.logicalToPhysicalBitOffset.getResult(0).isMultipleOf(8);
    std::optional<PresburgerSet> pieceDomain =
        getPieceDomain(type.getShape(), piece);
    if (!pieceDomain)
      return mlir::failure();
    for (const PresburgerSet &existing : pieceDomains)
      if (!existing.intersect(*pieceDomain).isIntegerEmpty())
        return mlir::failure();
    pieceDomains.push_back(*pieceDomain);
    IndexRelationResult pieceRelation = IndexRelation::fromAffineMap(
        piece.logicalToPhysicalBitOffset, type.getShape(), {offsetDomainSize},
        limits);
    if (!pieceRelation.isExact())
      return mlir::failure();
    IndexRelationResult restricted =
        pieceRelation.get()->intersectDestinationDomain(*pieceDomain, limits);
    if (!restricted.isExact())
      return mlir::failure();
    if (!combined)
      combined = restricted.get()->getPresburgerRelation();
    else
      combined->unionInPlace(restricted.get()->getPresburgerRelation());
  }

  if (!combined) {
    if (*valid != 0)
      return mlir::failure();
    IntegerRelation empty(PresburgerSpace::getRelationSpace(type.getRank(), 1));
    empty.addBound(BoundType::LB, type.getRank(), 0);
    empty.addBound(BoundType::UB, type.getRank(), -1);
    combined = PresburgerRelation(empty);
  }
  if (exceedsLimits(*combined, limits))
    return mlir::failure();

  IndexRelation relation(std::move(*combined), IndexRelationStatus::Exact);
  IndexSetResult logicalDomain =
      IndexRelation::staticDomain(type.getShape(), limits);
  if (!logicalDomain.isExact() ||
      !relation.getPresburgerRelation().getDomainSet().isEqual(
          *logicalDomain.set))
    return mlir::failure();

  return PhysicalLayoutRelation(type, std::move(relation), std::move(*pieces),
                                *elementBits, *footprint, *alignment, *valid,
                                *padding, byteAddressable);
}

mlir::FailureOr<WaferPhysicalElementSpan>
PhysicalLayoutRelation::getPhysicalElementSpan(
    llvm::ArrayRef<int64_t> logicalPoint) const {
  auto encoding = mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
      type.getMemorySpace());
  if (!encoding)
    return mlir::failure();
  return encoding.getPhysicalElementSpan(type, logicalPoint);
}

} // namespace wafer::analysis
