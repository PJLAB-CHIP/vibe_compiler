//===- WholeVariantAttemptPlan.h - Invocation-local tuple plan -*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLEVARIANTATTEMPTPLAN_H
#define WAFER_COMPILER_WHOLEVARIANTATTEMPTPLAN_H

#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wafer::compiler::detail {

/// The only metadata needed to reproduce the bounded whole-variant tuple walk.
/// It is copied from an actual rank candidate and lives only for one compiler
/// invocation. It is neither candidate semantics nor a serialized artifact.
struct RankVariantSlotMetadata {
  int64_t stableOrdinal = 0;
  wafer::RankArtifactKind artifactKind = wafer::RankArtifactKind::Spill;
  bool reservedBaseline = false;
  wafer::RankBufferingKind bufferingKind = wafer::RankBufferingKind::Single;
  uint32_t bufferingPlanOrdinal = 0;
  wafer::RankWorkerPlacementKind workerPlacementKind =
      wafer::RankWorkerPlacementKind::Unplaced;
  uint32_t workerPlacementPlanOrdinal = 0;
};

using RankVariantMetadataFrontier = std::vector<RankVariantSlotMetadata>;

enum class WholeVariantAttemptPlanFailure : uint8_t {
  None,
  CandidateDomain,
  ReservedBaseline,
};

/// Exact replay of the bounded Cartesian/coordinated attempt sequence over
/// rank-frontier metadata. Candidate indices always refer to the original
/// frontier slots; no frontier is compacted or reordered.
struct WholeVariantAttemptPlan {
  WholeVariantAttemptPlanFailure failure = WholeVariantAttemptPlanFailure::None;
  std::vector<size_t> reservedBaselineIndices;
  std::vector<std::vector<size_t>> optimizedCandidateIndices;
  /// Bounded complete typed physical-realization tuples appended to the
  /// ordinary attempt sequence and retained for post-import composition.
  std::vector<std::vector<size_t>> workerPlacedCandidateIndices;
  std::vector<std::vector<size_t>> fixedSlotCandidateIndices;
  /// Bounded complete Single+Unplaced correspondence tuples. NoC seed
  /// selection consumes this explicit band so saturated physical bands cannot
  /// starve ordinary dataflow opportunities.
  std::vector<std::vector<size_t>> genericCandidateIndices;
  std::vector<std::vector<size_t>> requiredModuleIndices;
  size_t boundedProductPositionCount = 0;
  size_t boundedProductUniqueAttemptCount = 0;
  size_t coordinatedAttemptCount = 0;

  bool isValid() const {
    return failure == WholeVariantAttemptPlanFailure::None;
  }

  size_t getRequiredModuleCount() const;
};

/// Reproduces the production coordinator's reserved-baseline allowance,
/// 64-position Cartesian walk, and bounded 64 complete-correspondence walk
/// without inspecting or materializing MLIR modules. When more than 64
/// complete tuples exist, the correspondence walk samples deterministic
/// evenly spaced quantiles including both endpoints, so no full upstream
/// prefix can starve the rest of the domain. On malformed metadata,
/// requiredModuleIndices conservatively names every slot so the coordinator
/// retains its existing structural diagnostics. Complete disjoint-worker and
/// otherwise fixed-slot tuples have separate frontier-bounded attempt/import
/// allowances so later physical-dataflow stages can compose them without an
/// unbounded import. A bounded ordinary Single+Unplaced correspondence band is
/// retained independently so typed physical bands cannot starve generic NoC
/// seeds under the fixed total seed budget.
WholeVariantAttemptPlan buildWholeVariantAttemptPlan(
    llvm::ArrayRef<RankVariantMetadataFrontier> frontiers,
    int64_t expectedRankCount);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLEVARIANTATTEMPTPLAN_H
