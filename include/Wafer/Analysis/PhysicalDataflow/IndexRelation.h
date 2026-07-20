//===- IndexRelation.h - MLIR-backed logical index relations ---*- C++ -*-===//

#ifndef WAFER_ANALYSIS_PHYSICALDATAFLOW_INDEXRELATION_H
#define WAFER_ANALYSIS_PHYSICALDATAFLOW_INDEXRELATION_H

#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/ArrayRef.h"

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
};

struct IndexRelationResult;

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

  static IndexRelationResult
  identity(llvm::ArrayRef<int64_t> shape,
           const IndexRelationLimits &limits = IndexRelationLimits());

  static IndexRelationResult
  fromAffineMap(mlir::AffineMap map, llvm::ArrayRef<int64_t> destinationShape,
                llvm::ArrayRef<int64_t> sourceShape,
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

  /// Compose this A->B relation with next B->C and return A->C.
  IndexRelationResult
  compose(const IndexRelation &next,
          const IndexRelationLimits &limits = IndexRelationLimits()) const;

private:
  mlir::presburger::PresburgerRelation relation;
  IndexRelationStatus status;

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

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_PHYSICALDATAFLOW_INDEXRELATION_H
