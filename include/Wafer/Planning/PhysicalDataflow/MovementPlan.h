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

using MovementActionId = std::variant<ExternalLoadId, DDRBoundaryTransferId,
                                      ReductionGatherId, ResultPublicationId>;

enum class PeerTransferGraphKind : uint8_t {
  TargetRoutedPeer,
  SoftwareRelay,
  SoftwareFanout,
  ExternalLoadFanout,
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

struct PeerTransferGraphPlan {
  PeerTransferGraphKind kind = PeerTransferGraphKind::TargetRoutedPeer;
  std::vector<MovementHop> hops;
  /// All logical actions served by this one payload graph. Every graph carries
  /// its sorted, all-and-only action set and is stored once in MovementPlan.
  /// An action absent from every graph uses its explicit DDR carrier.
  std::vector<MovementActionId> actions;
  /// Present only for ExternalLoadFanout. This member performs the one DDR
  /// load that seeds the peer graph; every other member is satisfied by the
  /// explicit endpoint transfers.
  std::optional<ExternalLoadId> ddrRoot;

  friend bool operator==(const PeerTransferGraphPlan &lhs,
                         const PeerTransferGraphPlan &rhs) {
    return lhs.kind == rhs.kind && lhs.hops == rhs.hops &&
           lhs.actions == rhs.actions && lhs.ddrRoot == rhs.ddrRoot;
  }
  friend bool operator<(const PeerTransferGraphPlan &lhs,
                        const PeerTransferGraphPlan &rhs) {
    return std::tie(lhs.kind, lhs.hops, lhs.actions, lhs.ddrRoot) <
           std::tie(rhs.kind, rhs.hops, rhs.actions, rhs.ddrRoot);
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

struct DDRBoundaryTransferPlan {
  DDRBoundaryTransferId id;
  PhysicalVersionId source;
  PhysicalVersionId destination;

  friend bool operator==(const DDRBoundaryTransferPlan &lhs,
                         const DDRBoundaryTransferPlan &rhs) {
    return lhs.id == rhs.id && lhs.source == rhs.source &&
           lhs.destination == rhs.destination;
  }
  friend bool operator<(const DDRBoundaryTransferPlan &lhs,
                        const DDRBoundaryTransferPlan &rhs) {
    return std::tie(lhs.id, lhs.source, lhs.destination) <
           std::tie(rhs.id, rhs.source, rhs.destination);
  }
};

struct ReductionGatherPlan {
  ReductionGatherId id;
  PhysicalVersionId source;
  ExecutionInstanceId mergeExecution;

  friend bool operator==(const ReductionGatherPlan &lhs,
                         const ReductionGatherPlan &rhs) {
    return lhs.id == rhs.id && lhs.source == rhs.source &&
           lhs.mergeExecution == rhs.mergeExecution;
  }
  friend bool operator<(const ReductionGatherPlan &lhs,
                        const ReductionGatherPlan &rhs) {
    return std::tie(lhs.id, lhs.source, lhs.mergeExecution) <
           std::tie(rhs.id, rhs.source, rhs.mergeExecution);
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
  std::vector<PeerTransferGraphPlan> peerGraphs;
  std::vector<ResultPublicationPlan> publications;
  std::vector<ResultDiscardPlan> discards;

  friend bool operator==(const MovementPlan &lhs, const MovementPlan &rhs) {
    return lhs.externalLoads == rhs.externalLoads &&
           lhs.ddrTransfers == rhs.ddrTransfers &&
           lhs.reductionGathers == rhs.reductionGathers &&
           lhs.peerGraphs == rhs.peerGraphs &&
           lhs.publications == rhs.publications && lhs.discards == rhs.discards;
  }
  friend bool operator<(const MovementPlan &lhs, const MovementPlan &rhs) {
    return std::tie(lhs.externalLoads, lhs.ddrTransfers, lhs.reductionGathers,
                    lhs.peerGraphs, lhs.publications, lhs.discards) <
           std::tie(rhs.externalLoads, rhs.ddrTransfers, rhs.reductionGathers,
                    rhs.peerGraphs, rhs.publications, rhs.discards);
  }
};

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
