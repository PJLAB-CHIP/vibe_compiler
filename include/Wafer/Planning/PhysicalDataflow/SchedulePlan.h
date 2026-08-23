//===- SchedulePlan.h - Typed schedule prefix and closure ----*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEPLAN_H

#include "Wafer/IR/NCCCompletion.h"
#include "Wafer/Planning/PhysicalDataflow/BufferPlan.h"
#include "Wafer/Planning/PhysicalDataflow/ExecutionStructurePlan.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

using ScheduleNodeId = StorageAccessSite;

struct CanonicalScheduleWorkerBinding {
  /// Canonical default for worker-capable issues emitted by this high-level
  /// node. The node itself is not asserted to be an NCC operation.
  ScheduleNodeId node;
  NCCWorker worker = NCCWorker::Worker0;

  friend bool operator==(const CanonicalScheduleWorkerBinding &lhs,
                         const CanonicalScheduleWorkerBinding &rhs) {
    return lhs.node == rhs.node && lhs.worker == rhs.worker;
  }
};

struct CanonicalSchedulePrefix {
  std::vector<ScheduleNodeId> order;
  std::vector<CanonicalScheduleWorkerBinding> workerBindings;

  friend bool operator==(const CanonicalSchedulePrefix &lhs,
                         const CanonicalSchedulePrefix &rhs) {
    return lhs.order == rhs.order && lhs.workerBindings == rhs.workerBindings;
  }
};

struct ScheduleDependency {
  ScheduleNodeId predecessor;
  ScheduleNodeId successor;
  StorageObjectId object;

  friend bool operator==(const ScheduleDependency &lhs,
                         const ScheduleDependency &rhs) {
    return lhs.predecessor == rhs.predecessor &&
           lhs.successor == rhs.successor && lhs.object == rhs.object;
  }
  friend bool operator<(const ScheduleDependency &lhs,
                        const ScheduleDependency &rhs) {
    if (!(lhs.predecessor == rhs.predecessor))
      return lhs.predecessor < rhs.predecessor;
    if (!(lhs.successor == rhs.successor))
      return lhs.successor < rhs.successor;
    return lhs.object < rhs.object;
  }
};

struct CanonicalScheduleCoordinate {
  CanonicalSchedulePrefix plan;
  std::vector<ScheduleDependency> dependencies;
};

struct EventWorkerBinding {
  EventId event;
  NCCWorker worker = NCCWorker::Worker0;

  friend bool operator==(const EventWorkerBinding &lhs,
                         const EventWorkerBinding &rhs) {
    return lhs.event == rhs.event && lhs.worker == rhs.worker;
  }
  friend bool operator<(const EventWorkerBinding &lhs,
                        const EventWorkerBinding &rhs) {
    return std::tie(lhs.event, lhs.worker) < std::tie(rhs.event, rhs.worker);
  }
};

struct ResourceInstanceId {
  ResourceKey resource;
  uint32_t lane = 0;

  friend bool operator==(const ResourceInstanceId &lhs,
                         const ResourceInstanceId &rhs) {
    return lhs.resource == rhs.resource && lhs.lane == rhs.lane;
  }
  friend bool operator<(const ResourceInstanceId &lhs,
                        const ResourceInstanceId &rhs) {
    return std::tie(lhs.resource, lhs.lane) < std::tie(rhs.resource, rhs.lane);
  }
};

struct EventResourceBinding {
  EventId event;
  ResourceInstanceId instance;

  friend bool operator==(const EventResourceBinding &lhs,
                         const EventResourceBinding &rhs) {
    return lhs.event == rhs.event && lhs.instance == rhs.instance;
  }
  friend bool operator<(const EventResourceBinding &lhs,
                        const EventResourceBinding &rhs) {
    return std::tie(lhs.event, lhs.instance) <
           std::tie(rhs.event, rhs.instance);
  }
};

struct ResourceSequence {
  ResourceInstanceId instance;
  std::vector<EventId> events;

  friend bool operator==(const ResourceSequence &lhs,
                         const ResourceSequence &rhs) {
    return lhs.instance == rhs.instance && lhs.events == rhs.events;
  }
  friend bool operator<(const ResourceSequence &lhs,
                        const ResourceSequence &rhs) {
    return std::tie(lhs.instance, lhs.events) <
           std::tie(rhs.instance, rhs.events);
  }
};

struct CardControlScope {
  CardId card{0};
  friend bool operator==(CardControlScope lhs, CardControlScope rhs) {
    return lhs.card == rhs.card;
  }
  friend bool operator<(CardControlScope lhs, CardControlScope rhs) {
    return lhs.card.getValue() < rhs.card.getValue();
  }
};

struct TileControlScope {
  TileId tile{0};
  friend bool operator==(TileControlScope lhs, TileControlScope rhs) {
    return lhs.tile == rhs.tile;
  }
  friend bool operator<(TileControlScope lhs, TileControlScope rhs) {
    return lhs.tile.getValue() < rhs.tile.getValue();
  }
};

using ControlScopeId = std::variant<CardControlScope, TileControlScope>;

struct ControlOrder {
  ControlScopeId scope;
  std::vector<EventId> events;

  friend bool operator==(const ControlOrder &lhs, const ControlOrder &rhs) {
    return lhs.scope == rhs.scope && lhs.events == rhs.events;
  }
  friend bool operator<(const ControlOrder &lhs, const ControlOrder &rhs) {
    return std::tie(lhs.scope, lhs.events) < std::tie(rhs.scope, rhs.events);
  }
};

struct EventBoundaryId {
  EventId after;

  friend bool operator==(const EventBoundaryId &lhs,
                         const EventBoundaryId &rhs) {
    return lhs.after == rhs.after;
  }
  friend bool operator<(const EventBoundaryId &lhs,
                        const EventBoundaryId &rhs) {
    return lhs.after < rhs.after;
  }
};

struct CompletionPlacement {
  EventId issue;
  EventId completion;
  EventBoundaryId boundary;
  CompletionProtocol protocol = CompletionProtocol::Synchronous;
  uint32_t participantMask = 0;

  friend bool operator==(const CompletionPlacement &lhs,
                         const CompletionPlacement &rhs) {
    return lhs.issue == rhs.issue && lhs.completion == rhs.completion &&
           lhs.boundary == rhs.boundary && lhs.protocol == rhs.protocol &&
           lhs.participantMask == rhs.participantMask;
  }
  friend bool operator<(const CompletionPlacement &lhs,
                        const CompletionPlacement &rhs) {
    return std::tie(lhs.issue, lhs.completion, lhs.boundary, lhs.protocol,
                    lhs.participantMask) < std::tie(rhs.issue, rhs.completion,
                                                    rhs.boundary, rhs.protocol,
                                                    rhs.participantMask);
  }
};

/// Complete plan-level schedule for one fixed structure/storage generation.
/// It intentionally carries no timestamp, pending-set, calendar, or cost.
struct ClosedSchedulePlan {
  ExecutionStructurePlan structure;
  BufferPlan buffers;
  std::vector<EventWorkerBinding> workerBindings;
  std::vector<EventResourceBinding> resourceBindings;
  std::vector<ResourceSequence> resourceSequences;
  std::vector<ControlOrder> controlOrders;
  std::vector<CompletionPlacement> completionPlacements;

  friend bool operator==(const ClosedSchedulePlan &lhs,
                         const ClosedSchedulePlan &rhs) {
    return lhs.structure == rhs.structure && lhs.buffers == rhs.buffers &&
           lhs.workerBindings == rhs.workerBindings &&
           lhs.resourceBindings == rhs.resourceBindings &&
           lhs.resourceSequences == rhs.resourceSequences &&
           lhs.controlOrders == rhs.controlOrders &&
           lhs.completionPlacements == rhs.completionPlacements;
  }
  friend bool operator<(const ClosedSchedulePlan &lhs,
                        const ClosedSchedulePlan &rhs) {
    return std::tie(lhs.structure, lhs.buffers, lhs.workerBindings,
                    lhs.resourceBindings, lhs.resourceSequences,
                    lhs.controlOrders, lhs.completionPlacements) <
           std::tie(rhs.structure, rhs.buffers, rhs.workerBindings,
                    rhs.resourceBindings, rhs.resourceSequences,
                    rhs.controlOrders, rhs.completionPlacements);
  }
};

enum class BrokenSchedulePlanReason : uint8_t {
  EmptyScheduleInput,
  DuplicateSerializedExecution,
  DuplicateStorageObject,
  DuplicateStorageResource,
  DuplicateLifetime,
  MissingStorageResource,
  MissingLifetime,
  UnknownStorageObject,
  UnknownExecutionSite,
  EmptyUseSet,
  CyclicDependency,
};

struct BrokenSchedulePlan {
  BrokenSchedulePlanReason reason =
      BrokenSchedulePlanReason::EmptyScheduleInput;
  std::optional<ScheduleNodeId> node;
  std::string detail;
};

using CanonicalSchedulePlanOutcome =
    std::variant<CanonicalScheduleCoordinate, BrokenSchedulePlan>;

const CanonicalScheduleCoordinate *
getCanonicalScheduleCoordinate(const CanonicalSchedulePlanOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEPLAN_H
