//===- GlobalTileRelation.h - Rank/global static tile relation -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_PHYSICALDATAFLOW_GLOBALTILERELATION_H
#define WAFER_ANALYSIS_PHYSICALDATAFLOW_GLOBALTILERELATION_H

#include "Wafer/Frontend/Program/Program.h"

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace wafer::compiler {

/// One rectangular strided logical tile. This is a recomputable analysis
/// value used while constructing an all-rank candidate; it is never persisted
/// in IR, packages, or a side table.
struct StaticTileRegion {
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  friend bool operator==(const StaticTileRegion &lhs,
                         const StaticTileRegion &rhs) {
    return lhs.offsets == rhs.offsets && lhs.sizes == rhs.sizes &&
           lhs.strides == rhs.strides;
  }
  friend bool operator!=(const StaticTileRegion &lhs,
                         const StaticTileRegion &rhs) {
    return !(lhs == rhs);
  }
};

enum class StaticTileRelation {
  Equivalent,
  Disjoint,
  OverlappingOrUnknown,
};

/// One typed memref.subview step from a program-boundary memref toward an
/// actual instruction operand. Dynamic offsets retain their SSA expression;
/// sizes and strides are required to be positive constants by the resolver.
struct SymbolicTileViewStep {
  mlir::Operation *viewOperation = nullptr;
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
};

/// Recomputable typed view of an actual boundary operand. `viewSteps` are in
/// root-to-leaf order. `staticLocalTile` is present only when every composed
/// offset is constant; dynamic loop offsets remain explicit SSA in the steps.
/// This value is analysis-only and is never persisted as an IR side channel.
struct ResolvedBoundaryTileView {
  unsigned argumentIndex = 0;
  mlir::Type leafType;
  llvm::SmallVector<SymbolicTileViewStep, 4> viewSteps;
  std::optional<StaticTileRegion> staticLocalTile;
};

/// Compose a rank-local work tile through the verified program-boundary slice
/// into global logical coordinates. Every extent, stride, bound, and
/// arithmetic operation is checked; malformed or non-representable relations
/// fail closed.
llvm::Expected<StaticTileRegion>
mapRankLocalTileToGlobal(const frontend::ProgramRankSlice &rankSlice,
                         llvm::ArrayRef<int64_t> globalShape,
                         llvm::ArrayRef<int64_t> localShape,
                         const StaticTileRegion &localTile);

/// Invert a global logical tile through one verified rank slice. This is used
/// to prove that a proposed owner can source exactly the consumer's tile.
/// Non-integral, out-of-slice, or differently strided projections fail.
llvm::Expected<StaticTileRegion>
mapGlobalTileToRankLocal(const frontend::ProgramRankSlice &rankSlice,
                         llvm::ArrayRef<int64_t> globalShape,
                         llvm::ArrayRef<int64_t> localShape,
                         const StaticTileRegion &globalTile);

/// Exact equality is distinguished from proven bounding-box disjointness.
/// Anything else is deliberately fail-closed as overlap/unknown.
StaticTileRelation compareStaticTiles(const StaticTileRegion &lhs,
                                      const StaticTileRegion &rhs);

/// Resolve casts, tile-region boundary aliases, and a chain of rank-preserving
/// memref.subview operations back to one function argument. Every dynamic
/// offset must have statically proven non-negative/in-bounds loop bounds;
/// dynamic sizes/strides and unknown index expressions fail closed.
std::optional<ResolvedBoundaryTileView>
resolveBoundaryTileView(mlir::Value value, llvm::ArrayRef<int64_t> localShape);

/// Compare two actual boundary views in global program coordinates. Fully
/// static views use exact rank/global composition. Dynamic views are
/// equivalent only when their verified rank slices are identical and every
/// typed loop/view expression is structurally equivalent; otherwise the
/// result is fail-closed.
StaticTileRelation
compareRankBoundaryTileViews(const ResolvedBoundaryTileView &lhs,
                             const frontend::ProgramRankSlice &lhsSlice,
                             const ResolvedBoundaryTileView &rhs,
                             const frontend::ProgramRankSlice &rhsSlice,
                             llvm::ArrayRef<int64_t> globalShape,
                             llvm::ArrayRef<int64_t> localShape);

/// Invocation-local semantic hash for equality prefiltering only. LLVM may
/// seed hash values per process, so the numeric result must never determine
/// canonical group order, communication IDs, or physical owner selection.
/// Hash equality is not a legality proof; callers must still use
/// `compareRankBoundaryTileViews`.
std::optional<size_t>
hashRankBoundaryTileView(const ResolvedBoundaryTileView &view,
                         const frontend::ProgramRankSlice &rankSlice,
                         llvm::ArrayRef<int64_t> globalShape,
                         llvm::ArrayRef<int64_t> localShape);

/// Compare/hash a typed operation occurrence path to its function root.
/// Function symbol names are ignored; operation, block, region, and nesting
/// positions are retained so sibling structured occurrences cannot alias.
/// The hash is invocation-local and must not drive canonical ordering.
bool haveEquivalentStructuredOperationPaths(mlir::Operation *lhs,
                                            mlir::Operation *rhs);
std::optional<size_t> hashStructuredOperationPath(mlir::Operation *operation);

} // namespace wafer::compiler

#endif // WAFER_ANALYSIS_PHYSICALDATAFLOW_GLOBALTILERELATION_H
