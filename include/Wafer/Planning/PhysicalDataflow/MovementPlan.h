//===- MovementPlan.h - Explicit canonical movement schema ----*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_MOVEMENTPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_MOVEMENTPLAN_H

#include "Wafer/Planning/PhysicalDataflow/RepresentationPlan.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
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

  friend bool operator==(const ExternalLoadPlan &lhs,
                         const ExternalLoadPlan &rhs) {
    return lhs.id == rhs.id && lhs.destination == rhs.destination;
  }
  friend bool operator<(const ExternalLoadPlan &lhs,
                        const ExternalLoadPlan &rhs) {
    return std::tie(lhs.id, lhs.destination) <
           std::tie(rhs.id, rhs.destination);
  }
};

enum class MovementRealizationKind : uint8_t {
  DDRStage,
  TargetRoutedPeer,
  SoftwareRelay,
};

struct MovementHop {
  TileId source{0};
  TileId destination{0};

  friend bool operator==(const MovementHop &lhs, const MovementHop &rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
  friend bool operator<(const MovementHop &lhs, const MovementHop &rhs) {
    return std::tuple(lhs.source.getValue(), lhs.destination.getValue()) <
           std::tuple(rhs.source.getValue(), rhs.destination.getValue());
  }
};

struct MovementRealization {
  MovementRealizationKind kind = MovementRealizationKind::DDRStage;
  std::vector<MovementHop> hops;

  friend bool operator==(const MovementRealization &lhs,
                         const MovementRealization &rhs) {
    return lhs.kind == rhs.kind && lhs.hops == rhs.hops;
  }
  friend bool operator<(const MovementRealization &lhs,
                        const MovementRealization &rhs) {
    return std::tie(lhs.kind, lhs.hops) < std::tie(rhs.kind, rhs.hops);
  }
};

struct DDRBoundaryTransferPlan {
  DDRBoundaryTransferId id;
  PhysicalVersionId source;
  PhysicalVersionId destination;
  MovementRealization realization;

  friend bool operator==(const DDRBoundaryTransferPlan &lhs,
                         const DDRBoundaryTransferPlan &rhs) {
    return lhs.id == rhs.id && lhs.source == rhs.source &&
           lhs.destination == rhs.destination &&
           lhs.realization == rhs.realization;
  }
  friend bool operator<(const DDRBoundaryTransferPlan &lhs,
                        const DDRBoundaryTransferPlan &rhs) {
    return std::tie(lhs.id, lhs.source, lhs.destination, lhs.realization) <
           std::tie(rhs.id, rhs.source, rhs.destination, rhs.realization);
  }
};

struct ReductionGatherPlan {
  ReductionGatherId id;
  PhysicalVersionId source;
  ExecutionInstanceId mergeExecution;
  MovementRealization realization;

  friend bool operator==(const ReductionGatherPlan &lhs,
                         const ReductionGatherPlan &rhs) {
    return lhs.id == rhs.id && lhs.source == rhs.source &&
           lhs.mergeExecution == rhs.mergeExecution &&
           lhs.realization == rhs.realization;
  }
  friend bool operator<(const ReductionGatherPlan &lhs,
                        const ReductionGatherPlan &rhs) {
    return std::tie(lhs.id, lhs.source, lhs.mergeExecution, lhs.realization) <
           std::tie(rhs.id, rhs.source, rhs.mergeExecution, rhs.realization);
  }
};

struct ResultPublicationPlan {
  ResultPublicationId id;
  PhysicalVersionId source;

  friend bool operator==(const ResultPublicationPlan &lhs,
                         const ResultPublicationPlan &rhs) {
    return lhs.id == rhs.id && lhs.source == rhs.source;
  }
  friend bool operator<(const ResultPublicationPlan &lhs,
                        const ResultPublicationPlan &rhs) {
    return std::tie(lhs.id, lhs.source) < std::tie(rhs.id, rhs.source);
  }
};

/// Explicit disposition for an execution result whose exact piece is produced
/// but has no downstream carrier because a later pure tensor reconstruction
/// overwrites or otherwise removes that piece. It is not a movement action.
struct ResultDiscardPlan {
  ResultDiscardId id;
  PhysicalVersionId source;

  friend bool operator==(const ResultDiscardPlan &lhs,
                         const ResultDiscardPlan &rhs) {
    return lhs.id == rhs.id && lhs.source == rhs.source;
  }
  friend bool operator<(const ResultDiscardPlan &lhs,
                        const ResultDiscardPlan &rhs) {
    return std::tie(lhs.id, lhs.source) < std::tie(rhs.id, rhs.source);
  }
};

struct MovementPlan {
  std::vector<ExternalLoadPlan> externalLoads;
  std::vector<DDRBoundaryTransferPlan> ddrTransfers;
  std::vector<ReductionGatherPlan> reductionGathers;
  std::vector<ResultPublicationPlan> publications;
  std::vector<ResultDiscardPlan> discards;

  friend bool operator==(const MovementPlan &lhs, const MovementPlan &rhs) {
    return lhs.externalLoads == rhs.externalLoads &&
           lhs.ddrTransfers == rhs.ddrTransfers &&
           lhs.reductionGathers == rhs.reductionGathers &&
           lhs.publications == rhs.publications && lhs.discards == rhs.discards;
  }
  friend bool operator<(const MovementPlan &lhs, const MovementPlan &rhs) {
    return std::tie(lhs.externalLoads, lhs.ddrTransfers, lhs.reductionGathers,
                    lhs.publications, lhs.discards) <
           std::tie(rhs.externalLoads, rhs.ddrTransfers, rhs.reductionGathers,
                    rhs.publications, rhs.discards);
  }
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
