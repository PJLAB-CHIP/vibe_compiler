//===- IndexRelation.h - MLIR-backed logical index relations ---*- C++ -*-===//

#ifndef WAFER_ANALYSIS_LINALG_INDEXRELATION_H
#define WAFER_ANALYSIS_LINALG_INDEXRELATION_H

#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer::analysis {

/// Precision/failure classes are explicit so only Exact results can authorize
/// an IR rewrite. SoundBound is an over-approximation suitable for pruning or
/// diagnostics, never for equivalence or movement elimination.
enum class IndexRelationStatus {
  Exact,
  SoundBound,
  Unsupported,
  Invalid,
  ResourceExhausted,
};

struct IndexRelationLimits {
  unsigned maxVariables = 32;
  unsigned maxDisjuncts = 8;
  unsigned maxConstraintsPerDisjunct = 1024;
  unsigned maxLocalVariablesPerDisjunct = 32;
  uint64_t maxAbsoluteCoefficient = uint64_t{1} << 50;
  uint64_t maxRectangularPieces = 4096;
};

struct IndexRelationResult;
struct IndexSetResult;
struct IndexRelationQueryResult;
struct StaticRectangularIndexSetResult;
struct StaticRectangularIndexSetPiecesResult;

/// A transformation-local adapter over MLIR Presburger relations. Domain
/// variables are destination logical indexes and range variables are source
/// logical indexes. The object owns no Operation/Value handles and must be
/// rebuilt by its transformation owner after relevant IR mutation.
class IndexRelation {
public:
  IndexRelation(mlir::presburger::PresburgerRelation relation,
                IndexRelationStatus status)
      : relation(std::move(relation)), status(status) {}

  unsigned getDestinationRank() const;
  unsigned getSourceRank() const;
  IndexRelationStatus getStatus() const { return status; }

  bool contains(llvm::ArrayRef<int64_t> destination,
                llvm::ArrayRef<int64_t> source) const;

  const mlir::presburger::PresburgerRelation &getPresburgerRelation() const {
    return relation;
  }

  /// Recover a standard affine map from the exact Presburger relation when
  /// every source coordinate is an integral affine expression of the
  /// destination coordinates.  This is a derived query, not retained side
  /// state: transformations may use it to lower the same relation without
  /// rebuilding an operation-specific index language.  Piecewise/quasi-affine
  /// relations such as a general reshape return std::nullopt.
  std::optional<mlir::AffineMap>
  getProjectedAffineMap(mlir::MLIRContext *context) const;

  static IndexRelationResult
  identity(llvm::ArrayRef<int64_t> shape,
           const IndexRelationLimits &limits = IndexRelationLimits());

  static IndexRelationResult
  fromAffineMap(mlir::AffineMap map, llvm::ArrayRef<int64_t> destinationShape,
                llvm::ArrayRef<int64_t> sourceShape,
                const IndexRelationLimits &limits = IndexRelationLimits());

  /// Compose two exact mappings out of one common iteration domain and return
  /// the destination-to-source relation. The result is allowed to be
  /// one-to-many: iterators projected out of the destination remain quantified
  /// in the exact source demand instead of making the relation unsupported.
  static IndexRelationResult fromCommonIterationDomain(
      mlir::AffineMap iterationToDestination,
      llvm::ArrayRef<int64_t> destinationShape,
      mlir::AffineMap iterationToSource, llvm::ArrayRef<int64_t> sourceShape,
      llvm::ArrayRef<int64_t> iterationShape,
      const IndexRelationLimits &limits = IndexRelationLimits());

  static IndexRelationResult
  staticSlice(llvm::ArrayRef<int64_t> destinationShape,
              llvm::ArrayRef<int64_t> sourceShape,
              llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> strides,
              const IndexRelationLimits &limits = IndexRelationLimits());

  /// Resolve mixed index operands with MLIR ValueBounds, then build the exact
  /// static slice relation. Unknown dynamic values return Unsupported.
  static IndexRelationResult
  slice(llvm::ArrayRef<int64_t> destinationShape,
        llvm::ArrayRef<int64_t> sourceShape,
        llvm::ArrayRef<mlir::OpFoldResult> offsets,
        llvm::ArrayRef<mlir::OpFoldResult> strides,
        const IndexRelationLimits &limits = IndexRelationLimits());

  /// Build the exact row-major logical relation between equal-element-count
  /// static shapes. This covers collapse/expand reassociation without adding a
  /// private relation expression language.
  static IndexRelationResult
  staticReshape(llvm::ArrayRef<int64_t> destinationShape,
                llvm::ArrayRef<int64_t> sourceShape,
                const IndexRelationLimits &limits = IndexRelationLimits());

  /// Build one exact destination-to-source piece of a logical concat. The
  /// source coordinate on `axis` starts at zero while the corresponding
  /// destination interval starts at `destinationOffset`.
  static IndexRelationResult
  staticConcatPiece(llvm::ArrayRef<int64_t> destinationShape,
                    llvm::ArrayRef<int64_t> sourceShape, unsigned axis,
                    int64_t destinationOffset,
                    const IndexRelationLimits &limits = IndexRelationLimits());

  /// Build the exact destination-to-source relation of one static unit-stride
  /// insert_slice: the source coordinate on every dimension is the
  /// destination coordinate shifted by the per-dimension offset. Strided or
  /// negative inserts are not expressible as one piece and stay outside this
  /// builder.
  static IndexRelationResult
  staticInsertSlice(llvm::ArrayRef<int64_t> destinationShape,
                    llvm::ArrayRef<int64_t> sourceShape,
                    llvm::ArrayRef<int64_t> offsets,
                    const IndexRelationLimits &limits = IndexRelationLimits());

  /// Build a static rectangular index domain.
  static IndexSetResult
  staticDomain(llvm::ArrayRef<int64_t> shape,
               const IndexRelationLimits &limits = IndexRelationLimits());

  /// Build a static half-open rectangular index domain.
  static IndexSetResult staticRectangularDomain(
      llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
      const IndexRelationLimits &limits = IndexRelationLimits());

  /// Compose this A->B relation with next B->C and return A->C.
  IndexRelationResult
  compose(const IndexRelation &next,
          const IndexRelationLimits &limits = IndexRelationLimits()) const;

  IndexRelationResult
  inverse(const IndexRelationLimits &limits = IndexRelationLimits()) const;

  IndexRelationResult intersectDestinationDomain(
      const mlir::presburger::PresburgerSet &domain,
      const IndexRelationLimits &limits = IndexRelationLimits()) const;

  IndexRelationResult intersectSourceDomain(
      const mlir::presburger::PresburgerSet &domain,
      const IndexRelationLimits &limits = IndexRelationLimits()) const;

  IndexSetResult
  image(const mlir::presburger::PresburgerSet &destinationDomain,
        const IndexRelationLimits &limits = IndexRelationLimits()) const;

  /// Query the exact image of one static destination rectangle and recover it
  /// as one dense source rectangle. Exact projected mappings use a symbolic
  /// fast path; all other relations use the generic Presburger image and
  /// equality proof. Neither path accepts a bounding box approximation.
  StaticRectangularIndexSetResult getExactStaticRectangularImage(
      llvm::ArrayRef<int64_t> destinationOffsets,
      llvm::ArrayRef<int64_t> destinationSizes,
      const IndexRelationLimits &limits = IndexRelationLimits()) const;

  /// Decompose one exact rectangular destination image into a finite,
  /// disjoint row-major rectangle union when the construction proof supports
  /// it. This is the exact logical set in another representation, never a
  /// bounding box; unusual piece counts fail with ResourceExhausted.
  StaticRectangularIndexSetPiecesResult getExactStaticRectangularImagePieces(
      llvm::ArrayRef<int64_t> destinationOffsets,
      llvm::ArrayRef<int64_t> destinationSizes,
      const IndexRelationLimits &limits = IndexRelationLimits()) const;

  IndexSetResult
  preimage(const mlir::presburger::PresburgerSet &sourceDomain,
           const IndexRelationLimits &limits = IndexRelationLimits()) const;

  IndexRelationQueryResult
  isFunctional(const IndexRelationLimits &limits = IndexRelationLimits()) const;
  IndexRelationQueryResult
  isInjective(const IndexRelationLimits &limits = IndexRelationLimits()) const;
  IndexRelationQueryResult
  isBijective(const IndexRelationLimits &limits = IndexRelationLimits()) const;
  IndexRelationQueryResult isEquivalentTo(
      const IndexRelation &other,
      const IndexRelationLimits &limits = IndexRelationLimits()) const;
  IndexRelationQueryResult
  implies(const IndexRelation &other,
          const IndexRelationLimits &limits = IndexRelationLimits()) const;
  /// True when construction proves that the relation is total over its full
  /// static destination box and every mapped point lies in the static source
  /// box. This is stronger than affine-map functionality: source bounds can
  /// clip an otherwise single-valued affine map.
  bool hasTotalBoundedAffineMapConstruction() const {
    return totalBoundedAffineMapByConstruction;
  }

  /// True when construction proves that destination and source coordinates
  /// have the same row-major linear ordinal over their complete static boxes.
  /// This is the exact canonical static-reshape relation and does not invoke a
  /// generic Presburger equivalence query.
  bool hasCanonicalRowMajorReshapeConstruction() const;

private:
  struct RowMajorRectangleMapping {
    llvm::SmallVector<unsigned, 4> destinationDimensions;
    llvm::SmallVector<unsigned, 4> sourceDimensions;
  };

  mlir::presburger::PresburgerRelation relation;
  IndexRelationStatus status;
  /// Derived rectangular projected-map pattern. Each source coordinate holds
  /// its destination dimension, -1 for constant zero, or -2 for a complete
  /// source dimension. It supports projected image/preimage queries.
  std::optional<llvm::SmallVector<int64_t, 4>> projectedRectanglePattern;
  /// Exact row-major reshape/projection image pattern. Every entry equates
  /// the linear ordinal of one ordered destination-dimension group with one
  /// ordered source-dimension group. Groups are disjoint on each side, so a
  /// qualifying destination rectangle has a Cartesian-product source image.
  std::optional<llvm::SmallVector<RowMajorRectangleMapping, 4>>
      rowMajorRectangleMappings;
  std::optional<llvm::SmallVector<int64_t, 4>> rectangleDestinationShape;
  std::optional<llvm::SmallVector<int64_t, 4>> rectangleSourceShape;
  /// True when construction already proves the relation single-valued (for
  /// example an affine map flattened to Presburger constraints). The
  /// functionality query returns proven-true without running the generic
  /// self-composition proof for such relations.
  bool functionalByConstruction = false;
  bool canonicalRowMajorOrderByConstruction = false;
  /// True only for an affine construction whose source bounds cannot clip
  /// any point in the complete destination box.
  bool totalBoundedAffineMapByConstruction = false;

  friend struct IndexRelationResult;
};

struct IndexRelationResult {
  IndexRelationStatus status = IndexRelationStatus::Invalid;
  std::optional<IndexRelation> relation;
  std::string reason;

  bool isExact() const {
    return status == IndexRelationStatus::Exact && relation.has_value();
  }

  const IndexRelation *get() const { return relation ? &*relation : nullptr; }
  IndexRelation *get() { return relation ? &*relation : nullptr; }
};

struct IndexSetResult {
  IndexRelationStatus status = IndexRelationStatus::Invalid;
  std::optional<mlir::presburger::PresburgerSet> set;
  std::string reason;

  bool isExact() const {
    return status == IndexRelationStatus::Exact && set.has_value();
  }

  bool contains(llvm::ArrayRef<int64_t> point) const {
    return set && set->getSpace().getNumSetDimVars() == point.size() &&
           set->containsPoint(point);
  }

  /// Recover one dense static rectangle iff it is exactly equal to this set.
  /// The returned bounds are derived from Presburger extrema and then proved
  /// equal to the complete set; a bounding box is never accepted as demand.
  StaticRectangularIndexSetResult getExactStaticRectangularDomain(
      const IndexRelationLimits &limits = IndexRelationLimits()) const;

  /// Recover every stored disjunct as one dense static rectangle without
  /// invoking Presburger subtraction, subset, lexicographic optimization, or
  /// coalescing. This is the bounded path for sets that are deliberately
  /// assembled as unions of rectangular execution/ownership pieces. If any
  /// disjunct is not already a direct rectangular conjunction, the query
  /// fails closed instead of asking the generic solver to rediscover it.
  StaticRectangularIndexSetPiecesResult getExactStaticRectangularDisjuncts(
      const IndexRelationLimits &limits = IndexRelationLimits()) const;
};

struct StaticRectangularIndexSet {
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
};

struct StaticRectangularIndexSetResult {
  IndexRelationStatus status = IndexRelationStatus::Invalid;
  std::optional<StaticRectangularIndexSet> domain;
  std::string reason;

  bool isExact() const {
    return status == IndexRelationStatus::Exact && domain.has_value();
  }
};

struct StaticRectangularIndexSetPiecesResult {
  IndexRelationStatus status = IndexRelationStatus::Invalid;
  llvm::SmallVector<StaticRectangularIndexSet, 8> domains;
  std::string reason;

  bool isExact() const { return status == IndexRelationStatus::Exact; }
};

struct IndexRelationQueryResult {
  IndexRelationStatus status = IndexRelationStatus::Invalid;
  std::optional<bool> value;
  std::string reason;

  bool isProvenTrue() const {
    return status == IndexRelationStatus::Exact && value.value_or(false);
  }
};

/// A common rectangular iteration domain and two projected-affine mappings
/// that express the canonical row-major correspondence between equal-element
/// static shapes. Singleton dimensions are represented by constant-zero map
/// results. This is shared by physical-equivalence proof and descriptor
/// synthesis so neither rebuilds reshape semantics independently.
struct CanonicalReshapeRelations {
  llvm::SmallVector<int64_t, 4> iterationShape;
  IndexRelation iterationToSource;
  IndexRelation iterationToDest;
};

std::optional<CanonicalReshapeRelations>
getCanonicalReshapeRelations(mlir::MLIRContext *context,
                             llvm::ArrayRef<int64_t> sourceShape,
                             llvm::ArrayRef<int64_t> destinationShape);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_LINALG_INDEXRELATION_H
