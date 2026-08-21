//===- FeasibilityProof.h - Canonical resource proof schema -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_FEASIBILITYPROOF_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_FEASIBILITYPROOF_H

#include "Wafer/Planning/PhysicalDataflow/AttentionWorkDescription.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

using FeasibilityResourceOrigin =
    std::variant<StorageObjectId, AttentionValueId>;

struct FeasibilityResourceId {
  FeasibilityResourceOrigin origin;

  friend bool operator==(const FeasibilityResourceId &lhs,
                         const FeasibilityResourceId &rhs) {
    return lhs.origin == rhs.origin;
  }
  friend bool operator<(const FeasibilityResourceId &lhs,
                        const FeasibilityResourceId &rhs) {
    return lhs.origin < rhs.origin;
  }
};

struct FeasibilityResourceDemand {
  FeasibilityResourceId id;
  TileId tile{0};
  int64_t sizeBytes = 0;
  int64_t alignmentBytes = 1;
  int64_t begin = 0;
  int64_t end = 0;
};

struct FeasibilityResourceConflict {
  FeasibilityResourceId lhs;
  FeasibilityResourceId rhs;

  friend bool operator==(const FeasibilityResourceConflict &left,
                         const FeasibilityResourceConflict &right) {
    return left.lhs == right.lhs && left.rhs == right.rhs;
  }
  friend bool operator<(const FeasibilityResourceConflict &left,
                        const FeasibilityResourceConflict &right) {
    if (!(left.lhs == right.lhs))
      return left.lhs < right.lhs;
    return left.rhs < right.rhs;
  }
};

struct TileSPMResourceProblem {
  TileId tile{0};
  int64_t arenaBegin = 0;
  int64_t arenaEnd = 0;
  std::vector<FeasibilityResourceDemand> demands;
  std::vector<FeasibilityResourceConflict> conflicts;
};

struct DDRPayloadProblem {
  MovementActionId action;
  int64_t bytes = 0;
  int64_t largestContiguousBytes = 0;
};

struct CanonicalResourceProblem {
  std::vector<TileSPMResourceProblem> tileSPM;
  std::vector<DDRPayloadProblem> ddrPayloads;
  std::vector<ScheduleNodeId> scheduleNodes;
  std::vector<AttentionActionId> attentionActions;
};

enum class ResourceProofKind : uint8_t {
  TileSPM,
  DDRPayload,
  ScheduleCoverage,
  AttentionCoverage,
};

enum class ResourceProofMethod : uint8_t {
  ValidatedPlacement,
  ExactSolverProof,
  DirectCapacityProof,
};

struct ResourceProblemProof {
  ResourceProofKind kind = ResourceProofKind::TileSPM;
  ResourceProofMethod method = ResourceProofMethod::DirectCapacityProof;
  std::optional<TileId> tile;
  std::vector<FeasibilityResourceId> resources;
  std::vector<MovementActionId> movements;
  std::vector<ScheduleNodeId> scheduleNodes;
  std::vector<AttentionActionId> attentionActions;
};

struct FeasibilityDependencyKey {
  int64_t spmBegin = 0;
  int64_t spmEnd = 0;
  int64_t spmAlignment = 0;
  int64_t ddrCapacityBytes = 0;
  int64_t ddrLargestContiguousBytes = 0;
  int64_t ddrAlignmentBytes = 0;
  std::vector<FeasibilityResourceId> resources;
  std::vector<MovementActionId> movements;
  std::vector<ScheduleNodeId> scheduleNodes;
  std::vector<AttentionActionId> attentionActions;
};

enum class FullFeasibilityCoverage : uint8_t {
  EveryPlannedResourceClosed,
};

struct FullFeasibilityProof {
  FeasibilityDependencyKey dependencyKey;
  std::vector<ResourceProblemProof> resourceProblems;
  FullFeasibilityCoverage coverage =
      FullFeasibilityCoverage::EveryPlannedResourceClosed;
};

struct CanonicalFeasibilityCoordinate {
  CanonicalResourceProblem problem;
  FullFeasibilityProof proof;
};

enum class CanonicalResourceRejectionReason : uint8_t {
  SPMCapacity,
  DDRPayloadLimit,
};

struct CanonicalResourceRejection {
  CanonicalResourceRejectionReason reason =
      CanonicalResourceRejectionReason::SPMCapacity;
  std::optional<TileId> tile;
  std::vector<FeasibilityResourceId> resources;
  std::optional<MovementActionId> movement;
  std::optional<int64_t> requiredBytes;
  int64_t capacityBytes = 0;
};

enum class CanonicalFeasibilityIndeterminateReason : uint8_t {
  ResourceWorkExhausted,
  HeuristicNoFit,
};

struct CanonicalFeasibilityIndeterminate {
  CanonicalFeasibilityIndeterminateReason reason =
      CanonicalFeasibilityIndeterminateReason::ResourceWorkExhausted;
  std::optional<TileId> tile;
  std::vector<FeasibilityResourceId> resources;
};

enum class UnsupportedCanonicalFeasibilityReason : uint8_t {
  NonFiniteDomain,
  UnsupportedElementOrEncoding,
};

struct UnsupportedCanonicalFeasibility {
  UnsupportedCanonicalFeasibilityReason reason =
      UnsupportedCanonicalFeasibilityReason::NonFiniteDomain;
  std::optional<FeasibilityResourceId> resource;
  std::string detail;
};

enum class BrokenCanonicalFeasibilityReason : uint8_t {
  PlanCoverageMismatch,
  DuplicateIdentity,
  ArithmeticOverflow,
  InvalidResourceProblem,
  InvalidSolverResult,
};

struct BrokenCanonicalFeasibility {
  BrokenCanonicalFeasibilityReason reason =
      BrokenCanonicalFeasibilityReason::PlanCoverageMismatch;
  std::string detail;
};

using CanonicalFeasibilityOutcome =
    std::variant<CanonicalFeasibilityCoordinate, CanonicalResourceRejection,
                 CanonicalFeasibilityIndeterminate,
                 UnsupportedCanonicalFeasibility, BrokenCanonicalFeasibility>;

const CanonicalFeasibilityCoordinate *
getCanonicalFeasibilityCoordinate(const CanonicalFeasibilityOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_FEASIBILITYPROOF_H
