//===- EventGraph.h - Typed event and resource problem -------*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_EVENTGRAPH_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_EVENTGRAPH_H

#include "Wafer/IR/NCCCompletion.h"
#include "Wafer/Planning/PhysicalDataflow/BufferPlan.h"
#include "Wafer/Planning/PhysicalDataflow/SerializedExecutionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

namespace event_graph_detail {
class EventGraphBuilder;
}

enum class PlannedEventKind : uint8_t {
  ComputeIssue,
  MovementIssue,
  Completion,
  Wait,
  LocalCombine,
  BufferReady,
  BufferRelease,
  ObservableWrite,
};

struct ExecutionEventAction {
  ExecutionInstanceId execution;

  friend bool operator==(const ExecutionEventAction &lhs,
                         const ExecutionEventAction &rhs) {
    return lhs.execution == rhs.execution;
  }
  friend bool operator<(const ExecutionEventAction &lhs,
                        const ExecutionEventAction &rhs) {
    return lhs.execution < rhs.execution;
  }
};

/// One semantic transfer endpoint. `hop` comes from the selected movement
/// realization; it is not a router path inferred by the scheduler.
enum class MovementEventPhase : uint8_t {
  Logical,
  DDRLoad,
  DDRStore,
  PeerSend,
  PeerReceive,
  LocalCombine,
};

struct MovementEventAction {
  MovementActionId action;
  MovementEventPhase phase = MovementEventPhase::Logical;
  uint32_t payloadSlice = 0;
  std::optional<MovementHop> hop;

  friend bool operator==(const MovementEventAction &lhs,
                         const MovementEventAction &rhs) {
    return lhs.action == rhs.action && lhs.phase == rhs.phase &&
           lhs.payloadSlice == rhs.payloadSlice && lhs.hop == rhs.hop;
  }
  friend bool operator<(const MovementEventAction &lhs,
                        const MovementEventAction &rhs) {
    return std::tie(lhs.action, lhs.phase, lhs.payloadSlice, lhs.hop) <
           std::tie(rhs.action, rhs.phase, rhs.payloadSlice, rhs.hop);
  }
};

/// The semantic object identifies the definition/use lifetime. The storage
/// object identifies the selected physical allocation after alias/reuse.
struct BufferEventAction {
  StorageObjectId semanticObject;
  StorageObjectId storageObject;

  friend bool operator==(const BufferEventAction &lhs,
                         const BufferEventAction &rhs) {
    return lhs.semanticObject == rhs.semanticObject &&
           lhs.storageObject == rhs.storageObject;
  }
  friend bool operator<(const BufferEventAction &lhs,
                        const BufferEventAction &rhs) {
    return std::tie(lhs.semanticObject, lhs.storageObject) <
           std::tie(rhs.semanticObject, rhs.storageObject);
  }
};

using EventAction =
    std::variant<ExecutionEventAction, MovementEventAction, BufferEventAction>;

struct EventId {
  EventAction action;
  PlannedEventKind kind = PlannedEventKind::ComputeIssue;

  friend bool operator==(const EventId &lhs, const EventId &rhs) {
    return lhs.action == rhs.action && lhs.kind == rhs.kind;
  }
  friend bool operator!=(const EventId &lhs, const EventId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const EventId &lhs, const EventId &rhs) {
    return std::tie(lhs.action, lhs.kind) < std::tie(rhs.action, rhs.kind);
  }
};

struct PlannedEvent {
  EventId id;
  CardId card{0};
  std::optional<TileId> tile;
  std::vector<NCCWorker> workerDomain;

  friend bool operator==(const PlannedEvent &lhs, const PlannedEvent &rhs) {
    return lhs.id == rhs.id && lhs.card == rhs.card && lhs.tile == rhs.tile &&
           lhs.workerDomain == rhs.workerDomain;
  }
};

enum class EventDependencyReason : uint8_t {
  SSAValue,
  NestedExecution,
  LoopRecurrence,
  EffectOrder,
  TransferReady,
  BufferLifetime,
  Completion,
  Publication,
};

struct EventDependency {
  EventId predecessor;
  EventId successor;
  EventDependencyReason reason = EventDependencyReason::SSAValue;

  friend bool operator==(const EventDependency &lhs,
                         const EventDependency &rhs) {
    return lhs.predecessor == rhs.predecessor &&
           lhs.successor == rhs.successor && lhs.reason == rhs.reason;
  }
  friend bool operator<(const EventDependency &lhs,
                        const EventDependency &rhs) {
    return std::tie(lhs.predecessor, lhs.successor, lhs.reason) <
           std::tie(rhs.predecessor, rhs.successor, rhs.reason);
  }
};

struct PhysicalRangeBox {
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;

  friend bool operator==(const PhysicalRangeBox &lhs,
                         const PhysicalRangeBox &rhs) {
    return lhs.offsets == rhs.offsets && lhs.sizes == rhs.sizes;
  }
  friend bool operator<(const PhysicalRangeBox &lhs,
                        const PhysicalRangeBox &rhs) {
    return std::tie(lhs.offsets, lhs.sizes) < std::tie(rhs.offsets, rhs.sizes);
  }
};

struct TileEngineResource {
  TileId tile{0};
  friend bool operator==(TileEngineResource lhs, TileEngineResource rhs) {
    return lhs.tile == rhs.tile;
  }
  friend bool operator<(TileEngineResource lhs, TileEngineResource rhs) {
    return lhs.tile.getValue() < rhs.tile.getValue();
  }
};

struct NCCWorkerResource {
  TileId tile{0};
  NCCWorker worker = NCCWorker::Worker0;
  friend bool operator==(NCCWorkerResource lhs, NCCWorkerResource rhs) {
    return lhs.tile == rhs.tile && lhs.worker == rhs.worker;
  }
  friend bool operator<(NCCWorkerResource lhs, NCCWorkerResource rhs) {
    return std::tuple(lhs.tile.getValue(), lhs.worker) <
           std::tuple(rhs.tile.getValue(), rhs.worker);
  }
};

struct DirectDTESenderResource {
  TileId tile{0};
  friend bool operator==(DirectDTESenderResource lhs,
                         DirectDTESenderResource rhs) {
    return lhs.tile == rhs.tile;
  }
  friend bool operator<(DirectDTESenderResource lhs,
                        DirectDTESenderResource rhs) {
    return lhs.tile.getValue() < rhs.tile.getValue();
  }
};

/// Current receiver FSM pool has four interchangeable lanes per destination
/// Tile. The pool key is exact; ScheduleDomain assigns canonical lane numbers.
struct DTEReceiverFSMResource {
  TileId tile{0};
  friend bool operator==(DTEReceiverFSMResource lhs,
                         DTEReceiverFSMResource rhs) {
    return lhs.tile == rhs.tile;
  }
  friend bool operator<(DTEReceiverFSMResource lhs,
                        DTEReceiverFSMResource rhs) {
    return lhs.tile.getValue() < rhs.tile.getValue();
  }
};

struct SPMRangeResource {
  StorageObjectId object;
  std::vector<PhysicalRangeBox> boxes;
  friend bool operator==(const SPMRangeResource &lhs,
                         const SPMRangeResource &rhs) {
    return lhs.object == rhs.object && lhs.boxes == rhs.boxes;
  }
  friend bool operator<(const SPMRangeResource &lhs,
                        const SPMRangeResource &rhs) {
    return std::tie(lhs.object, lhs.boxes) < std::tie(rhs.object, rhs.boxes);
  }
};

struct CardDDRResource {
  CardId card{0};
  friend bool operator==(CardDDRResource lhs, CardDDRResource rhs) {
    return lhs.card == rhs.card;
  }
  friend bool operator<(CardDDRResource lhs, CardDDRResource rhs) {
    return lhs.card.getValue() < rhs.card.getValue();
  }
};

struct DirectedNoCLinkResource {
  MovementHop link;
  friend bool operator==(const DirectedNoCLinkResource &lhs,
                         const DirectedNoCLinkResource &rhs) {
    return lhs.link == rhs.link;
  }
  friend bool operator<(const DirectedNoCLinkResource &lhs,
                        const DirectedNoCLinkResource &rhs) {
    return lhs.link < rhs.link;
  }
};

struct OpaqueNoCTransferResource {
  MovementHop endpoints;
  friend bool operator==(const OpaqueNoCTransferResource &lhs,
                         const OpaqueNoCTransferResource &rhs) {
    return lhs.endpoints == rhs.endpoints;
  }
  friend bool operator<(const OpaqueNoCTransferResource &lhs,
                        const OpaqueNoCTransferResource &rhs) {
    return lhs.endpoints < rhs.endpoints;
  }
};

using ResourceKey =
    std::variant<TileEngineResource, NCCWorkerResource, DirectDTESenderResource,
                 DTEReceiverFSMResource, SPMRangeResource, CardDDRResource,
                 DirectedNoCLinkResource, OpaqueNoCTransferResource>;

enum class ResourceUseMode : uint8_t { Read, Write, Exclusive, CapacityUnits };
enum class ResourceIntervalKind : uint8_t {
  IssueToCompletion,
  Instantaneous,
  UntilEvent,
};
enum class ResourceKnowledge : uint8_t { Exact, LowerBound, Estimate };

struct PlannedResourceUse {
  EventId event;
  ResourceKey resource;
  ResourceUseMode mode = ResourceUseMode::Exclusive;
  ResourceIntervalKind interval = ResourceIntervalKind::IssueToCompletion;
  std::optional<EventId> until;
  ResourceKnowledge knowledge = ResourceKnowledge::Exact;

  friend bool operator==(const PlannedResourceUse &lhs,
                         const PlannedResourceUse &rhs) {
    return lhs.event == rhs.event && lhs.resource == rhs.resource &&
           lhs.mode == rhs.mode && lhs.interval == rhs.interval &&
           lhs.until == rhs.until && lhs.knowledge == rhs.knowledge;
  }
  friend bool operator<(const PlannedResourceUse &lhs,
                        const PlannedResourceUse &rhs) {
    return std::tie(lhs.resource, lhs.event, lhs.mode, lhs.interval, lhs.until,
                    lhs.knowledge) < std::tie(rhs.resource, rhs.event, rhs.mode,
                                              rhs.interval, rhs.until,
                                              rhs.knowledge);
  }
};

struct DisjunctiveResourceOrder {
  ResourceKey resource;
  std::vector<EventId> events;

  friend bool operator==(const DisjunctiveResourceOrder &lhs,
                         const DisjunctiveResourceOrder &rhs) {
    return lhs.resource == rhs.resource && lhs.events == rhs.events;
  }
};

enum class CompletionProtocol : uint8_t {
  Unknown,
  NoAsynchronousCompletion,
  DirectDTE,
  NCCParticipant,
  NCCSynchronousWriteback,
};

struct CompletionObligation {
  EventId issue;
  EventId completion;
  CompletionProtocol protocol = CompletionProtocol::Unknown;
  uint32_t participantMask = 0;

  friend bool operator==(const CompletionObligation &lhs,
                         const CompletionObligation &rhs) {
    return lhs.issue == rhs.issue && lhs.completion == rhs.completion &&
           lhs.protocol == rhs.protocol &&
           lhs.participantMask == rhs.participantMask;
  }
  friend bool operator<(const CompletionObligation &lhs,
                        const CompletionObligation &rhs) {
    return std::tie(lhs.issue, lhs.completion, lhs.protocol,
                    lhs.participantMask) < std::tie(rhs.issue, rhs.completion,
                                                    rhs.protocol,
                                                    rhs.participantMask);
  }
};

struct EventComponent {
  std::vector<EventId> events;

  friend bool operator==(const EventComponent &lhs, const EventComponent &rhs) {
    return lhs.events == rhs.events;
  }
};

struct ExecutionEventContract {
  ExecutionInstanceId execution;
  std::vector<NCCWorker> workerDomain;
  NCCCompletionKind completion = NCCCompletionKind::None;
  uint32_t participantMask = 0;
  /// A source execution may have no standalone target issue when its exact
  /// SSA value is consumed inside one or more selected local executions. The
  /// dependencies retain that semantic occurrence without inventing work.
  std::vector<ExecutionInstanceId> foldedInto;
};

/// Exact target routing is an explicit input fact. In its absence a selected
/// peer hop remains an opaque endpoint transfer and cannot claim link use.
struct ExactMovementRoute {
  std::vector<MovementActionId> graphActions;
  MovementHop transfer;
  std::vector<MovementHop> links;
};

struct EventGraphLimits {
  uint64_t maxEvents = 100000;
  uint64_t maxDependencies = 400000;
};

class EventGraph {
public:
  llvm::ArrayRef<PlannedEvent> getEvents() const { return events; }
  llvm::ArrayRef<EventDependency> getHardDependencies() const {
    return hardDependencies;
  }
  llvm::ArrayRef<DisjunctiveResourceOrder> getOrderChoices() const {
    return orderChoices;
  }
  llvm::ArrayRef<CompletionObligation> getCompletionObligations() const {
    return completionObligations;
  }
  llvm::ArrayRef<PlannedResourceUse> getResourceUses() const {
    return resourceUses;
  }
  llvm::ArrayRef<EventComponent> getComponents() const { return components; }

  bool contains(const EventId &event) const;

  /// Returns the stable exact ready set after applying completed events and
  /// caller-selected resource precedences. It never chooses one ready event.
  std::optional<std::vector<EventId>>
  getReadyEvents(llvm::ArrayRef<EventId> completed,
                 llvm::ArrayRef<EventDependency> selectedOrders = {}) const;

  friend bool operator==(const EventGraph &lhs, const EventGraph &rhs) {
    return lhs.events == rhs.events &&
           lhs.hardDependencies == rhs.hardDependencies &&
           lhs.orderChoices == rhs.orderChoices &&
           lhs.completionObligations == rhs.completionObligations &&
           lhs.resourceUses == rhs.resourceUses &&
           lhs.components == rhs.components;
  }

private:
  EventGraph(std::vector<PlannedEvent> events,
             std::vector<EventDependency> hardDependencies,
             std::vector<DisjunctiveResourceOrder> orderChoices,
             std::vector<CompletionObligation> completionObligations,
             std::vector<PlannedResourceUse> resourceUses,
             std::vector<EventComponent> components)
      : events(std::move(events)),
        hardDependencies(std::move(hardDependencies)),
        orderChoices(std::move(orderChoices)),
        completionObligations(std::move(completionObligations)),
        resourceUses(std::move(resourceUses)),
        components(std::move(components)) {}

  std::vector<PlannedEvent> events;
  std::vector<EventDependency> hardDependencies;
  std::vector<DisjunctiveResourceOrder> orderChoices;
  std::vector<CompletionObligation> completionObligations;
  std::vector<PlannedResourceUse> resourceUses;
  std::vector<EventComponent> components;

  friend class event_graph_detail::EventGraphBuilder;
};

enum class EventGraphFailureKind : uint8_t {
  ExactRejection,
  Deferred,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

enum class EventGraphFailureReason : uint8_t {
  HardDependencyCycle,
  EmptyWorkerDomain,
  MissingPlanFact,
  UnsupportedExecutionContract,
  UnsupportedResourceRange,
  WorkLimit,
  MalformedPlan,
};

struct EventGraphFailure {
  EventGraphFailureKind kind = EventGraphFailureKind::CompilerBug;
  EventGraphFailureReason reason = EventGraphFailureReason::MalformedPlan;
  std::vector<EventId> witness;
  std::string detail;
};

struct ExecutionEventContractResult {
  std::vector<ExecutionEventContract> contracts;
  std::optional<EventGraphFailure> failure;

  bool succeeded() const { return !failure.has_value(); }
};

struct EventGraphBuildResult {
  std::optional<EventGraph> graph;
  std::optional<EventGraphFailure> failure;

  bool succeeded() const { return graph.has_value(); }
};

EventGraphBuildResult buildEventGraph(
    CardId card, const RegionPlan &regions, const TemporalPlan &temporal,
    const SerializedExecutionPlan &serialized,
    const CanonicalMovementCoordinate &movement,
    const CanonicalStorageCoordinate &storage, const BufferPlan &buffers,
    llvm::ArrayRef<ExecutionEventContract> executionContracts = {},
    llvm::ArrayRef<ExactMovementRoute> exactRoutes = {},
    const EventGraphLimits &limits = EventGraphLimits());

/// Derives the current target-abstract completion contract for every selected
/// execution from its typed structured root. A fixed-mode attention root is a
/// composite of selected Linalg NCC issues and therefore exposes the same
/// worker domain with one execution-level pending obligation; it does not
/// create per-action or per-K2 completion. Unknown implementations remain
/// typed Unsupported and never become implicit synchronous work.
ExecutionEventContractResult deriveExecutionEventContracts(
    const SerializedExecutionPlan &serialized,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks);

ExecutionEventContractResult deriveExecutionEventContracts(
    const SerializedExecutionPlan &serialized, const RegionPlan &regions,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_EVENTGRAPH_H
