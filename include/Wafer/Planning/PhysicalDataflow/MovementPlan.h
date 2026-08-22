//===- MovementPlan.h - Explicit canonical movement schema ----*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_MOVEMENTPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_MOVEMENTPLAN_H

#include "Wafer/Planning/PhysicalDataflow/RepresentationPlan.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

struct ExternalLoadId {
  BoundaryRegionValueId destination;

  friend bool operator==(const ExternalLoadId &lhs, const ExternalLoadId &rhs) {
    return lhs.destination == rhs.destination;
  }
  friend bool operator<(const ExternalLoadId &lhs, const ExternalLoadId &rhs) {
    return lhs.destination < rhs.destination;
  }
};

struct DDRBoundaryTransferId {
  BoundaryRegionValueId destination;

  friend bool operator==(const DDRBoundaryTransferId &lhs,
                         const DDRBoundaryTransferId &rhs) {
    return lhs.destination == rhs.destination;
  }
  friend bool operator<(const DDRBoundaryTransferId &lhs,
                        const DDRBoundaryTransferId &rhs) {
    return lhs.destination < rhs.destination;
  }
};

using ReductionGatherValue =
    std::variant<ReductionPartialValueId, CoupledComponentValueId>;

struct ReductionGatherId {
  ReductionGroupId group;
  LogicalShardId contribution;
  ReductionGatherValue value;

  friend bool operator==(const ReductionGatherId &lhs,
                         const ReductionGatherId &rhs) {
    return lhs.group == rhs.group && lhs.contribution == rhs.contribution &&
           lhs.value == rhs.value;
  }
  friend bool operator<(const ReductionGatherId &lhs,
                        const ReductionGatherId &rhs) {
    if (lhs.group != rhs.group)
      return lhs.group < rhs.group;
    if (lhs.contribution != rhs.contribution)
      return lhs.contribution < rhs.contribution;
    if (lhs.value.index() != rhs.value.index())
      return lhs.value.index() < rhs.value.index();
    if (const auto *partial = std::get_if<ReductionPartialValueId>(&lhs.value))
      return *partial < std::get<ReductionPartialValueId>(rhs.value);
    return std::get<CoupledComponentValueId>(lhs.value) <
           std::get<CoupledComponentValueId>(rhs.value);
  }
};

struct ResultPublicationId {
  ExecutionResultValueId source;

  friend bool operator==(const ResultPublicationId &lhs,
                         const ResultPublicationId &rhs) {
    return lhs.source == rhs.source;
  }
  friend bool operator<(const ResultPublicationId &lhs,
                        const ResultPublicationId &rhs) {
    return lhs.source < rhs.source;
  }
};

struct ResultDiscardId {
  ExecutionResultValueId source;

  friend bool operator==(const ResultDiscardId &lhs,
                         const ResultDiscardId &rhs) {
    return lhs.source == rhs.source;
  }
  friend bool operator<(const ResultDiscardId &lhs,
                        const ResultDiscardId &rhs) {
    return lhs.source < rhs.source;
  }
};

struct ExternalLoadPlan {
  ExternalLoadId id;
  PhysicalVersionId destination;
};

struct DDRBoundaryTransferPlan {
  DDRBoundaryTransferId id;
  PhysicalVersionId source;
  PhysicalVersionId destination;
};

struct ReductionGatherPlan {
  ReductionGatherId id;
  PhysicalVersionId source;
  ExecutionInstanceId mergeExecution;
};

struct ResultPublicationPlan {
  ResultPublicationId id;
  PhysicalVersionId source;
};

/// Explicit disposition for an execution result whose exact piece is produced
/// but has no downstream carrier because a later pure tensor reconstruction
/// overwrites or otherwise removes that piece. It is not a movement action.
struct ResultDiscardPlan {
  ResultDiscardId id;
  PhysicalVersionId source;
};

struct MovementPlan {
  std::vector<ExternalLoadPlan> externalLoads;
  std::vector<DDRBoundaryTransferPlan> ddrTransfers;
  std::vector<ReductionGatherPlan> reductionGathers;
  std::vector<ResultPublicationPlan> publications;
  std::vector<ResultDiscardPlan> discards;
};

using MovementActionId = std::variant<ExternalLoadId, DDRBoundaryTransferId,
                                      ReductionGatherId, ResultPublicationId>;

struct MovementResourceDescription {
  MovementActionId action;
  analysis::ExactIndexSet exactDomain;
  mlir::Type elementType;
  std::optional<TileId> sourceTile;
  std::optional<TileId> destinationTile;
};

struct CanonicalMovementCoordinate {
  MovementPlan plan;
  std::vector<MovementResourceDescription> resources;
};

enum class BrokenMovementPlanReason : uint8_t {
  DuplicateRootWork,
  PlanWorkMismatch,
  MissingPhysicalVersion,
  DuplicateAction,
  CoverageMismatch,
};

struct BrokenMovementPlan {
  BrokenMovementPlanReason reason = BrokenMovementPlanReason::PlanWorkMismatch;
  std::optional<analysis::RootRegionWorkId> work;
  std::string detail;
};

enum class UnsupportedMovementFeature : uint8_t {
  EffectfulOutputPath,
};

struct UnsupportedMovementPlan {
  UnsupportedMovementFeature feature =
      UnsupportedMovementFeature::EffectfulOutputPath;
  std::optional<analysis::RootRegionWorkId> work;
  std::string detail;
};

using CanonicalMovementPlanOutcome =
    std::variant<CanonicalMovementCoordinate, UnsupportedMovementPlan,
                 BrokenMovementPlan>;

const CanonicalMovementCoordinate *
getCanonicalMovementCoordinate(const CanonicalMovementPlanOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_MOVEMENTPLAN_H
