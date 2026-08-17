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
  int64_t elementDomainSize = 0;
  if (footprintBits % *elementBits != 0)
    return mlir::failure();
  elementDomainSize = footprintBits / *elementBits;
  if (*valid > 0) {
    if (footprintBits < *elementBits ||
        llvm::AddOverflow(footprintBits - *elementBits, int64_t{1},
                          offsetDomainSize) ||
        offsetDomainSize <= 0)
      return mlir::failure();
  }

  // Layout pieces are per-dimension bounded boxes (getPieceDomain builds one
  // lower/upper bound per dimension), so pairwise disjointness is a plain
  // interval-overlap test. The generic integer-emptiness proof on the
  // same boxes can enter unbounded Presburger simplex pivoting and is never
  // needed for the rectangle case.
  auto rectanglesOverlap = [&](const WaferPhysicalLayoutPiece &lhs,
                               const WaferPhysicalLayoutPiece &rhs) {
    for (auto [dim, bounds] : llvm::enumerate(
             llvm::zip_equal(lhs.logicalLowerBounds, lhs.logicalUpperBounds,
                             rhs.logicalLowerBounds, rhs.logicalUpperBounds))) {
      auto [lhsLower, lhsUpper, rhsLower, rhsUpper] = bounds;
      if (lhsLower >= rhsUpper || rhsLower >= lhsUpper)
        return false;
    }
    return true;
  };
  llvm::SmallVector<WaferPhysicalLayoutPiece, 2> previousPieces;

  std::optional<PresburgerRelation> combined;
  std::optional<PresburgerRelation> combinedOrdinals;
  llvm::SmallVector<PresburgerSet, 2> pieceDomains;
  bool byteAddressable = *elementBits % 8 == 0;
  for (const WaferPhysicalLayoutPiece &piece : *pieces) {
    if (!piece.logicalToPhysicalBitOffset ||
        piece.logicalToPhysicalBitOffset.getNumDims() !=
            static_cast<unsigned>(type.getRank()) ||
        piece.logicalToPhysicalBitOffset.getNumSymbols() != 0 ||
        piece.logicalToPhysicalBitOffset.getNumResults() != 1)
      return mlir::failure();
    mlir::AffineExpr bitOffset = piece.logicalToPhysicalBitOffset.getResult(0);
    if (!bitOffset.isMultipleOf(*elementBits))
      return mlir::failure();
    byteAddressable &=
        piece.logicalToPhysicalBitOffset.getResult(0).isMultipleOf(8);
    std::optional<PresburgerSet> pieceDomain =
        getPieceDomain(type.getShape(), piece);
    if (!pieceDomain)
      return mlir::failure();
    for (const WaferPhysicalLayoutPiece &previous : previousPieces)
      if (rectanglesOverlap(previous, piece))
        return mlir::failure();
    previousPieces.push_back(piece);
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
    mlir::AffineMap ordinalMap = mlir::AffineMap::get(
        type.getRank(), 0, bitOffset.floorDiv(*elementBits));
    IndexRelationResult pieceOrdinal = IndexRelation::fromAffineMap(
        ordinalMap, type.getShape(), {elementDomainSize}, limits);
    if (!pieceOrdinal.isExact())
      return mlir::failure();
    IndexRelationResult restrictedOrdinal =
        pieceOrdinal.get()->intersectDestinationDomain(*pieceDomain, limits);
    if (!restrictedOrdinal.isExact())
      return mlir::failure();
    if (!combined)
      combined = restricted.get()->getPresburgerRelation();
    else
      combined->unionInPlace(restricted.get()->getPresburgerRelation());
    if (!combinedOrdinals)
      combinedOrdinals = restrictedOrdinal.get()->getPresburgerRelation();
    else
      combinedOrdinals->unionInPlace(
          restrictedOrdinal.get()->getPresburgerRelation());
  }

  if (!combined) {
    if (*valid != 0)
      return mlir::failure();
    IntegerRelation empty(PresburgerSpace::getRelationSpace(type.getRank(), 1));
    empty.addBound(BoundType::LB, type.getRank(), 0);
    empty.addBound(BoundType::UB, type.getRank(), -1);
    combined = PresburgerRelation(empty);
    combinedOrdinals = *combined;
  }
  if (!combinedOrdinals || exceedsLimits(*combined, limits) ||
      exceedsLimits(*combinedOrdinals, limits))
    return mlir::failure();

  IndexRelation relation(std::move(*combined), IndexRelationStatus::Exact);
  IndexRelation ordinalRelation(std::move(*combinedOrdinals),
                                IndexRelationStatus::Exact);
  // The piece boxes are pairwise disjoint axis-aligned rectangles. Disjoint
  // rectangles cover the complete logical domain exactly when their volumes
  // sum to the domain volume; the generic set-equality proof on the same
  // boxes can enter unbounded Presburger elimination and is never needed
  // for the rectangle case.
  int64_t totalVolume = 1;
  for (int64_t extent : type.getShape())
    if (llvm::MulOverflow(totalVolume, extent, totalVolume))
      return mlir::failure();
  int64_t coveredVolume = 0;
  for (const WaferPhysicalLayoutPiece &piece : *pieces) {
    int64_t pieceVolume = 1;
    for (auto [lower, upper] : llvm::zip_equal(
             piece.logicalLowerBounds, piece.logicalUpperBounds))
      if (llvm::MulOverflow(pieceVolume, upper - lower, pieceVolume))
        return mlir::failure();
    if (llvm::AddOverflow(coveredVolume, pieceVolume, coveredVolume))
      return mlir::failure();
  }
  if (coveredVolume != totalVolume)
    return mlir::failure();

  return PhysicalLayoutRelation(type, std::move(relation),
                                std::move(ordinalRelation), std::move(*pieces),
                                *elementBits, elementDomainSize, *footprint,
                                *alignment, *valid, *padding, byteAddressable);
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
