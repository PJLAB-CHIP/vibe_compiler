//===- StructuredImplementationAlternative.h -----------------*- C++ -*-===//

#ifndef WAFER_COMPILER_STRUCTUREDIMPLEMENTATIONALTERNATIVE_H
#define WAFER_COMPILER_STRUCTUREDIMPLEMENTATIONALTERNATIVE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace wafer::compiler::detail {

using StructuredAlternativeTileShape = llvm::SmallVector<int64_t, 4>;

/// Current-SSA domains proven by one implementation-alternative provider.
/// These facts are query-local and must not survive mutation of the source IR.
struct StructuredAlternativeDomain {
  llvm::SmallVector<int64_t, 4> outputShape;
  llvm::SmallVector<int64_t, 2> reductionShape;
};

/// Generic structural parameters supplied to a concrete implementation point.
/// The provider owns their interpretation and the common search must not infer
/// algorithm semantics from the vectors or the stable key.
struct StructuredAlternativeParameters {
  llvm::SmallVector<int64_t, 4> outputTileSizes;
  llvm::SmallVector<int64_t, 2> reductionTileSizes;
  int64_t parallelPartitionCount = 1;
};

/// Cheap, target-independent structural facts for common bounded enumeration.
/// The optional estimates are derived by the concrete provider from the
/// current typed SSA and one exact parameter point.  They are ordering priors
/// only: common search may use them to spend its bounded beam on plausible
/// points, but legality, lifetime, placement, and final cost remain exact
/// post-materialization gates.  Compute work includes any re-execution induced
/// by the parameter point.  An absent estimate is ordinary and must not make a
/// provider worse.
struct StructuredAlternativeStructuralEstimates {
  uint64_t logicalOutputElements = 0;
  uint64_t logicalReductionElements = 0;
  uint64_t outputTileCount = 0;
  uint64_t reductionTileCount = 0;
  uint64_t parallelPartitionCount = 0;
  uint64_t structuredWorkUnitUpperBound = 0;
  std::optional<uint64_t> estimatedPeakLiveBytes;
  std::optional<uint64_t> estimatedComputeScalarOps;
};

/// An opaque deterministic key for one point in a canonical provider
/// query. stableKey is for equality/debugging only and must never be parsed to
/// recover semantics. stableOrdinal is provider-local and query-local; the
/// coordinated search assigns global semantic ordinals.
struct StructuredAlternativeKey {
  std::string stableKey;
  uint64_t stableOrdinal = 0;
};

/// Transient coverage returned by one successful graph materialization.  The
/// operation references are valid only in the supplied isolated module and
/// only until common lowering mutates it.  They tell the ordinary complete-
/// rank fallback which top-level operations already belong to the provider's
/// complete implementation graph; they are never persisted, interpreted as
/// algorithm semantics, or used as a lowering side channel.
struct StructuredImplementationAlternativeMaterialization {
  llvm::SmallVector<mlir::Operation *, 32> coveredTopLevelOperations;
};

/// Finite parameter axes offered by the common enumerator. Empty output or
/// reduction axes request the exact full current-SSA domain. Providers filter
/// invalid combinations and never select a winner from the legal points.
struct StructuredImplementationAlternativeQuery {
  /// Optional exact provider-domain shapes used by narrow tests and callers
  /// that already own a typed domain. Common coordinated search leaves these
  /// empty: it must not infer a provider's internal domain from public
  /// function result shapes.
  llvm::ArrayRef<StructuredAlternativeTileShape> outputTileShapes;
  llvm::ArrayRef<StructuredAlternativeTileShape> reductionTileShapes;
  llvm::ArrayRef<int64_t> parallelPartitionCounts;
  /// Target-generic scalar seeds. A provider first proves its exact current-
  /// SSA domain, then clamps each positive seed independently into every
  /// dimension of that domain. These are enumeration hints, not semantics.
  llvm::ArrayRef<int64_t> outputTileSizeSeeds;
  llvm::ArrayRef<int64_t> reductionTileSizeSeeds;
};

/// One immutable, query-local graph implementation point. A point carries no
/// source Operation pointers: materialization re-proves capability against the
/// supplied isolated clone and mutates only that discardable clone.
class StructuredImplementationAlternativePoint {
public:
  virtual ~StructuredImplementationAlternativePoint() = default;

  const StructuredAlternativeKey &getKey() const { return key; }
  const StructuredAlternativeDomain &getDomain() const { return domain; }
  const StructuredAlternativeParameters &getParameters() const {
    return parameters;
  }
  const StructuredAlternativeStructuralEstimates &
  getStructuralEstimates() const {
    return estimates;
  }

  virtual mlir::LogicalResult
  materialize(mlir::ModuleOp isolatedStructuredModule,
              std::string *failureReason = nullptr,
              StructuredImplementationAlternativeMaterialization *result =
                  nullptr) const = 0;

protected:
  StructuredImplementationAlternativePoint(
      StructuredAlternativeKey key, StructuredAlternativeDomain domain,
      StructuredAlternativeParameters parameters,
      StructuredAlternativeStructuralEstimates estimates);

private:
  StructuredAlternativeKey key;
  StructuredAlternativeDomain domain;
  StructuredAlternativeParameters parameters;
  StructuredAlternativeStructuralEstimates estimates;
};

using StructuredImplementationAlternativePoints =
    llvm::SmallVector<std::unique_ptr<StructuredImplementationAlternativePoint>,
                      8>;

/// A compiler-private capability provider. A semantic non-match is success
/// with no appended points; failure is reserved for an invalid query contract
/// or a provider invariant. Providers only enumerate and materialize points.
class StructuredImplementationAlternativeProvider {
public:
  virtual ~StructuredImplementationAlternativeProvider() = default;

  virtual llvm::StringRef getStableKey() const = 0;

  virtual mlir::LogicalResult
  query(mlir::ModuleOp currentStructuredModule,
        const StructuredImplementationAlternativeQuery &query,
        StructuredImplementationAlternativePoints &points,
        std::string *failureReason = nullptr) const = 0;
};

/// Validates one exact domain/parameter combination and derives overflow-safe
/// structural counts. This helper deliberately has no target/resource model.
mlir::FailureOr<StructuredAlternativeStructuralEstimates>
buildStructuredAlternativeStructuralEstimates(
    const StructuredAlternativeDomain &domain,
    const StructuredAlternativeParameters &parameters,
    std::string *failureReason = nullptr);

/// Materializes one opaque point only after reserving its actual work. The
/// short-lived pipeline creates an isolated complete-rank structured module,
/// lets the point re-prove and rewrite its current SSA, then consumes that
/// module through ordinary conservative TensorProgram-to-Tile
/// lowering. No provider-specific semantics enter the common coordinator.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeStructuredImplementationAlternativeToTileRegion(
    mlir::ModuleOp sourceModule,
    const StructuredImplementationAlternativePoint &point, int64_t logicalRank,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_STRUCTUREDIMPLEMENTATIONALTERNATIVE_H
