//===- ScopedFeasibility.h - Pure partial-state bounds -------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Search/CoupledRegion.h"

#include "Wafer/Target/Core/TargetMemory.h"

#include <optional>

namespace wafer::compiler::detail {

enum class FeasibilityCoordinate : uint8_t {
  PhysicalRepresentation,
  DataMovement,
  Buffering,
  EventSchedule,
};

enum class ScopedFeasibilityKind : uint8_t {
  LowerBound,
  Deferred,
  ExactRejection,
  Indeterminate,
};

enum class ScopedFeasibilityReason : uint8_t {
  None,
  MissingCoordinates,
  UnsupportedFootprint,
  MinimumFootprintExceedsSPM,
};

struct ScopedFeasibilityAssignmentKey {
  StructuredDAGNodeID node = 0;
  TileId tile{0};
  llvm::SmallVector<StructuredDAGNodeID, 4> groupNodes;
  llvm::SmallVector<int64_t, 4> spatialOffsets;
  llvm::SmallVector<int64_t, 4> spatialSizes;
  llvm::SmallVector<int64_t, 4> temporalTileSizes;
  llvm::SmallVector<uint32_t, 4> waveLoopOrder;

  friend bool operator==(const ScopedFeasibilityAssignmentKey &lhs,
                         const ScopedFeasibilityAssignmentKey &rhs) {
    return lhs.node == rhs.node && lhs.tile == rhs.tile &&
           lhs.groupNodes == rhs.groupNodes &&
           lhs.spatialOffsets == rhs.spatialOffsets &&
           lhs.spatialSizes == rhs.spatialSizes &&
           lhs.temporalTileSizes == rhs.temporalTileSizes &&
           lhs.waveLoopOrder == rhs.waveLoopOrder;
  }
};

struct ScopedSPMLowerBound {
  StructuredDAGNodeID node = 0;
  TileId tile{0};
  uint64_t minimumRequiredBytes = 0;
  std::optional<uint64_t> nonBindingResidencyEstimateBytes;
};

struct ScopedSPMRejection {
  ScopedFeasibilityAssignmentKey key;
  uint64_t minimumRequiredBytes = 0;
  uint64_t capacityBytes = 0;
};

struct ScopedFeasibilityResult {
  ScopedFeasibilityKind kind = ScopedFeasibilityKind::Indeterminate;
  ScopedFeasibilityReason reason =
      ScopedFeasibilityReason::UnsupportedFootprint;
  llvm::SmallVector<ScopedSPMLowerBound, 16> lowerBounds;
  llvm::SmallVector<FeasibilityCoordinate, 4> requiredCoordinates;
  std::optional<ScopedSPMRejection> rejection;
};

/// Computes only facts proven from immutable TensorProgram semantics, one
/// closed spatial/group/temporal assignment and explicit target memory facts.
/// It does not create IR, run lowering/packing, select a coordinate, or cache
/// a materialized candidate. `unresolvedCoordinates` is supplied by the
/// partial-state owner; an empty list asks only for the proven lower bounds.
mlir::FailureOr<ScopedFeasibilityResult> analyzeScopedFeasibility(
    const CardProgramAnalysis &program,
    const SpatialAssignment &spatial,
    const analysis::ExactDemandProof &demand,
    const CoupledRegionDomain &coupledDomain,
    const CoupledRegionAssignment &coupledAssignment,
    const CardTemporalDomain &temporalDomain,
    const CardTemporalAssignment &temporalAssignment,
    const TargetMemoryPolicy &memory,
    llvm::ArrayRef<FeasibilityCoordinate> unresolvedCoordinates);

} // namespace wafer::compiler::detail
