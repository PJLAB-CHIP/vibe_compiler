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
/// 64-position Cartesian walk, and 64 complete-correspondence walk without
/// inspecting or materializing MLIR modules. On malformed metadata,
/// requiredModuleIndices conservatively names every slot so the coordinator
/// retains its existing structural diagnostics.
WholeVariantAttemptPlan buildWholeVariantAttemptPlan(
    llvm::ArrayRef<RankVariantMetadataFrontier> frontiers,
    int64_t expectedRankCount);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLEVARIANTATTEMPTPLAN_H
