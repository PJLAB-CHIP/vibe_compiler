//===- PhysicalAccessRelation.h - Logical-to-physical access -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_TILE_PHYSICALACCESSRELATION_H
#define WAFER_ANALYSIS_TILE_PHYSICALACCESSRELATION_H

#include "Wafer/Analysis/Tile/PhysicalLayoutRelation.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer::analysis {

/// Invocation-local composition of an iteration-to-logical IndexRelation and
/// one endpoint's physical encoding.  IndexRelation deliberately remains a
/// layout-independent logical object; this analysis is the boundary that owns
/// logical-index -> physical-bit-span composition.  Point/span queries use the
/// verified logical projection and encoding directly.  The heavier composed
/// Presburger relations are created lazily only for relation-equivalence
/// queries.
///
/// The object owns no Operation or Value handles and is invalidated with the
/// IR epoch that supplied its MemRefType.  It is suitable for legality, cost,
/// descriptor and diagnostic queries, but is never serialized into IR.
class PhysicalAccessRelation {
public:
  /// Construct an exact total mapping from `iterationShape` to logical
  /// elements of `endpointType`.  The relation must be functional, cover the
  /// complete iteration domain, and stay inside the endpoint logical domain.
  /// Writers additionally request injectivity so no destination element is
  /// produced more than once.
  static mlir::FailureOr<PhysicalAccessRelation>
  create(mlir::MemRefType endpointType, llvm::ArrayRef<int64_t> iterationShape,
         const IndexRelation &iterationToLogical, bool requireInjective);

  mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
  getLogicalPoint(llvm::ArrayRef<int64_t> iterationPoint) const;

  mlir::FailureOr<WaferPhysicalElementSpan>
  getPhysicalElementSpan(llvm::ArrayRef<int64_t> iterationPoint) const;

  int64_t getPhysicalFootprintBytes() const { return physicalFootprintBytes; }
  int64_t getMinimumAlignmentBytes() const { return minimumAlignmentBytes; }
  int64_t getValidElementCount() const { return validElementCount; }
  int64_t getPaddingElementCount() const { return paddingElementCount; }

  mlir::MemRefType getEndpointType() const { return endpointType; }
  llvm::ArrayRef<int64_t> getIterationShape() const { return iterationShape; }
  const IndexRelation &getLogicalRelation() const { return iterationToLogical; }
  const PhysicalLayoutRelation &getPhysicalLayoutRelation() const {
    return physicalLayout;
  }
  /// Prove that two endpoints map the same iteration domain to the same
  /// physical element spans relative to their respective view bases.
  IndexRelationQueryResult
  hasSamePhysicalElementMapping(const PhysicalAccessRelation &other) const;

  /// Prove that two endpoints visit logical elements in the same physical
  /// element order and that this source covers the destination traversal.
  /// Unlike exact span equality, this deliberately normalizes each endpoint
  /// by its own element bit width, so a longer packed-mask source may feed a
  /// compatible dtype-changing CT route without pretending byte offsets or
  /// physical tail sizes are equal.
  IndexRelationQueryResult
  hasSamePhysicalTraversal(const PhysicalAccessRelation &other) const;

private:
  PhysicalAccessRelation(
      mlir::MemRefType endpointType,
      llvm::SmallVector<int64_t, 4> iterationShape,
      IndexRelation iterationToLogical, PhysicalLayoutRelation physicalLayout,
      std::optional<WaferStaticPhysicalOffsetCalculator> offsetCalculator,
      mlir::AffineMap projectedAffineMap, bool canonicalLinearOrder,
      int64_t physicalFootprintBytes, int64_t minimumAlignmentBytes,
      int64_t validElementCount, int64_t paddingElementCount)
      : endpointType(endpointType), iterationShape(std::move(iterationShape)),
        iterationToLogical(std::move(iterationToLogical)),
        physicalLayout(std::move(physicalLayout)),
        offsetCalculator(std::move(offsetCalculator)),
        projectedAffineMap(projectedAffineMap),
        canonicalLinearOrder(canonicalLinearOrder),
        physicalFootprintBytes(physicalFootprintBytes),
        minimumAlignmentBytes(minimumAlignmentBytes),
        validElementCount(validElementCount),
        paddingElementCount(paddingElementCount) {}

  IndexRelationQueryResult materializePhysicalRelations() const;

  mlir::MemRefType endpointType;
  llvm::SmallVector<int64_t, 4> iterationShape;
  IndexRelation iterationToLogical;
  PhysicalLayoutRelation physicalLayout;
  mutable bool physicalRelationsAttempted = false;
  mutable IndexRelationStatus physicalRelationsStatus =
      IndexRelationStatus::Invalid;
  mutable std::string physicalRelationsFailureReason;
  mutable std::optional<IndexRelation> iterationToPhysicalBitOffset;
  mutable std::optional<IndexRelation> iterationToPhysicalElementOrdinal;
  std::optional<WaferStaticPhysicalOffsetCalculator> offsetCalculator;
  mlir::AffineMap projectedAffineMap;
  bool canonicalLinearOrder = false;
  int64_t physicalFootprintBytes = 0;
  int64_t minimumAlignmentBytes = 0;
  int64_t validElementCount = 0;
  int64_t paddingElementCount = 0;
};

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_TILE_PHYSICALACCESSRELATION_H
