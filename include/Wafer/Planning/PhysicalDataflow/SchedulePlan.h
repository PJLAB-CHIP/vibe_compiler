//===- SchedulePlan.h - Typed schedule prefix and closure ----*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEPLAN_H

#include "Wafer/IR/NCCCompletion.h"
#include "Wafer/Planning/PhysicalDataflow/BufferPlan.h"
#include "Wafer/Planning/PhysicalDataflow/ExecutionStructurePlan.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
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

class EventOrder {
public:
  using const_iterator = std::vector<EventId>::const_iterator;

  EventOrder() : events(std::make_shared<const std::vector<EventId>>()) {}
  EventOrder(std::initializer_list<EventId> events)
      : events(
            std::make_shared<const std::vector<EventId>>(std::move(events))) {}
  EventOrder(std::vector<EventId> events)
      : events(
            std::make_shared<const std::vector<EventId>>(std::move(events))) {}

  EventOrder &operator=(std::vector<EventId> value) {
    events = std::make_shared<const std::vector<EventId>>(std::move(value));
    return *this;
  }

  const std::vector<EventId> &get() const { return *events; }
  const EventId *data() const { return events->data(); }
  size_t size() const { return events->size(); }
  bool empty() const { return events->empty(); }
  const EventId &front() const { return events->front(); }
  const EventId &back() const { return events->back(); }
  const EventId &operator[](size_t index) const { return (*events)[index]; }
  const_iterator begin() const { return events->begin(); }
  const_iterator end() const { return events->end(); }
  operator const std::vector<EventId> &() const { return *events; }

  friend bool operator==(const EventOrder &lhs, const EventOrder &rhs) {
    return lhs.events == rhs.events || lhs.get() == rhs.get();
  }
  friend bool operator<(const EventOrder &lhs, const EventOrder &rhs) {
    return lhs.get() < rhs.get();
  }

private:
  std::shared_ptr<const std::vector<EventId>> events;
};

struct ResourceSequence {
  ResourceInstanceId instance;
  EventOrder events;

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
  std::shared_ptr<const PipelineScopeId> pipeline =
      std::make_shared<const PipelineScopeId>();

  CardControlScope() = default;
  CardControlScope(CardId card, PipelineScopeId pipeline)
      : card(card),
        pipeline(std::make_shared<const PipelineScopeId>(std::move(pipeline))) {
  }
  CardControlScope(CardId card, std::shared_ptr<const PipelineScopeId> pipeline)
      : card(card), pipeline(std::move(pipeline)) {}

  friend bool operator==(const CardControlScope &lhs,
                         const CardControlScope &rhs) {
    return lhs.card == rhs.card &&
           (lhs.pipeline == rhs.pipeline || *lhs.pipeline == *rhs.pipeline);
  }
  friend bool operator<(const CardControlScope &lhs,
                        const CardControlScope &rhs) {
    if (lhs.card != rhs.card)
      return lhs.card.getValue() < rhs.card.getValue();
    if (lhs.pipeline == rhs.pipeline)
      return false;
    return *lhs.pipeline < *rhs.pipeline;
  }
};

struct TileControlScope {
  TileId tile{0};
  std::shared_ptr<const PipelineScopeId> pipeline =
      std::make_shared<const PipelineScopeId>();

  TileControlScope() = default;
  TileControlScope(TileId tile, PipelineScopeId pipeline)
      : tile(tile),
        pipeline(std::make_shared<const PipelineScopeId>(std::move(pipeline))) {
  }
  TileControlScope(TileId tile, std::shared_ptr<const PipelineScopeId> pipeline)
      : tile(tile), pipeline(std::move(pipeline)) {}

  friend bool operator==(const TileControlScope &lhs,
                         const TileControlScope &rhs) {
    return lhs.tile == rhs.tile &&
           (lhs.pipeline == rhs.pipeline || *lhs.pipeline == *rhs.pipeline);
  }
  friend bool operator<(const TileControlScope &lhs,
                        const TileControlScope &rhs) {
    if (lhs.tile != rhs.tile)
      return lhs.tile.getValue() < rhs.tile.getValue();
    if (lhs.pipeline == rhs.pipeline)
      return false;
    return *lhs.pipeline < *rhs.pipeline;
  }
};

using ControlScopeId = std::variant<CardControlScope, TileControlScope>;

struct ControlOrder {
  ControlScopeId scope;
  EventOrder events;

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
  CompletionProtocol protocol = CompletionProtocol::Unknown;
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
  std::shared_ptr<const ExecutionStructurePlan> structure =
      std::make_shared<const ExecutionStructurePlan>();
  std::shared_ptr<const BufferPlan> buffers =
      std::make_shared<const BufferPlan>();
  std::vector<EventWorkerBinding> workerBindings;
  std::vector<EventResourceBinding> resourceBindings;
  std::vector<ResourceSequence> resourceSequences;
  std::vector<ControlOrder> controlOrders;
  std::vector<CompletionPlacement> completionPlacements;

  const ExecutionStructurePlan &getStructure() const { return *structure; }
  const BufferPlan &getBuffers() const { return *buffers; }
  void setStructure(ExecutionStructurePlan value) {
    structure =
        std::make_shared<const ExecutionStructurePlan>(std::move(value));
  }
  void setBuffers(BufferPlan value) {
    buffers = std::make_shared<const BufferPlan>(std::move(value));
  }
  void
  setGeneration(std::shared_ptr<const ExecutionStructurePlan> structureValue,
                std::shared_ptr<const BufferPlan> bufferValue) {
    structure = std::move(structureValue);
    buffers = std::move(bufferValue);
  }

  friend bool operator==(const ClosedSchedulePlan &lhs,
                         const ClosedSchedulePlan &rhs) {
    return (lhs.structure == rhs.structure ||
            lhs.getStructure() == rhs.getStructure()) &&
           (lhs.buffers == rhs.buffers ||
            lhs.getBuffers() == rhs.getBuffers()) &&
           lhs.workerBindings == rhs.workerBindings &&
           lhs.resourceBindings == rhs.resourceBindings &&
           lhs.resourceSequences == rhs.resourceSequences &&
           lhs.controlOrders == rhs.controlOrders &&
           lhs.completionPlacements == rhs.completionPlacements;
  }
  friend bool operator<(const ClosedSchedulePlan &lhs,
                        const ClosedSchedulePlan &rhs) {
    return std::tie(lhs.getStructure(), lhs.getBuffers(), lhs.workerBindings,
                    lhs.resourceBindings, lhs.resourceSequences,
                    lhs.controlOrders, lhs.completionPlacements) <
           std::tie(rhs.getStructure(), rhs.getBuffers(), rhs.workerBindings,
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
