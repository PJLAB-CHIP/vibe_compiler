//===- EventGraph.cpp - Typed event and resource problem --------------===//

#include "Wafer/Planning/PhysicalDataflow/EventGraph.h"

#include "Wafer/Analysis/Structured/ScalarInitialization.h"

#include "Wafer/Support/CompileTiming.h"

#include "Wafer/Target/Core/DirectDTE.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <type_traits>
#include <utility>

namespace wafer::compiler::detail {

static_assert(TargetDirectDTEResourceLimits::senderSlotsPerTile == 1,
              "EventGraph Direct-DTE sender model requires one target slot");

namespace event_graph_detail {

TileId tileOf(const ExecutionInstanceId &execution) {
  return std::visit([](const auto &source) { return source.work.tile; },
                    execution.source);
}

EventId executionEvent(const ExecutionInstanceId &execution,
                       PlannedEventKind kind) {
  return {ExecutionEventAction{execution}, kind};
}

EventId movementEvent(const MovementActionId &action, PlannedEventKind kind,
                      MovementEventPhase phase = MovementEventPhase::Logical,
                      uint32_t payloadSlice = 0,
                      std::optional<MovementHop> hop = std::nullopt) {
  return {MovementEventAction{action, phase, payloadSlice, std::move(hop)},
          kind};
}

EventGraphBuildResult failed(EventGraphFailureKind kind,
                             EventGraphFailureReason reason,
                             llvm::StringRef detail,
                             std::vector<EventId> witness = {}) {
  return {{},
          EventGraphFailure{kind, reason, std::move(witness), detail.str()}};
}

struct MovementActionDescription {
  MovementActionId id;
  std::optional<PeerTransferGraphPlan> peerGraph;
  const MovementResourceDescription *resource = nullptr;
  bool publication = false;
  bool gather = false;
};

struct EventRecord {
  PlannedEvent planned;
  uint32_t ordinal = 0;
};

class EventGraphBuilder {
public:
  EventGraphBuilder(CardId card, const RegionPlan &regions,
                    const TemporalPlan &temporal,
                    const SerializedExecutionPlan &serialized,
                    const CanonicalMovementCoordinate &movement,
                    const CanonicalStorageCoordinate &storage,
                    const BufferPlan &buffers,
                    llvm::ArrayRef<ExecutionEventContract> contracts,
                    llvm::ArrayRef<ExactMovementRoute> exactRoutes,
                    const EventGraphLimits &limits)
      : card(card), regions(regions), temporal(temporal),
        serialized(serialized), movement(movement), storage(storage),
        buffers(buffers), contracts(contracts), exactRoutes(exactRoutes),
        limits(limits) {}

  EventGraphBuildResult build() {
    {
      wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                     "collect-event-inputs");
      if (!collectInputs())
        return std::move(*failure);
    }
    {
      wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                     "build-execution-events");
      if (!addExecutionEvents())
        return std::move(*failure);
    }
    {
      wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                     "build-movement-events");
      if (!addMovementEvents())
        return std::move(*failure);
    }
    {
      wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                     "build-storage-events");
      if (!addStorageEvents())
        return std::move(*failure);
    }
    {
      wafer::support::ScopedCompileTimingSpan timing(
          "query", "physical-search", "build-event-dependencies");
      if (!addNestedDependencies() || !addBufferOrderDependencies())
        return std::move(*failure);
    }
    if (events.size() > limits.maxEvents ||
        dependencies.size() > limits.maxDependencies)
      return failed(EventGraphFailureKind::Indeterminate,
                    EventGraphFailureReason::WorkLimit,
                    "event graph construction exceeded its work limit");

    std::vector<EventId> cycle = [&]() {
      wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                     "find-event-cycle");
      return findCycle();
    }();
    if (!cycle.empty())
      return failed(EventGraphFailureKind::ExactRejection,
                    EventGraphFailureReason::HardDependencyCycle,
                    "event graph has a hard dependency cycle",
                    std::move(cycle));

    {
      wafer::support::ScopedCompileTimingSpan timing(
          "query", "physical-search", "build-event-order-choices");
      buildOrderChoices();
    }
    {
      wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                     "build-event-components");
      buildComponents();
    }

    std::vector<PlannedEvent> plannedEvents;
    for (auto &[id, event] : events) {
      (void)id;
      plannedEvents.push_back(std::move(event.planned));
    }
    std::vector<EventDependency> hardDependencies(dependencies.begin(),
                                                  dependencies.end());
    std::vector<CompletionObligation> completionObligations(completions.begin(),
                                                            completions.end());
    std::vector<PlannedResourceUse> plannedResourceUses(resourceUses.begin(),
                                                        resourceUses.end());
    EventGraph graph(std::move(plannedEvents), std::move(hardDependencies),
                     std::move(orderChoices), std::move(completionObligations),
                     std::move(plannedResourceUses), std::move(components));
    return {std::move(graph), {}};
  }

private:
  bool fail(EventGraphFailureKind kind, EventGraphFailureReason reason,
            llvm::StringRef detail) {
    failure = failed(kind, reason, detail);
    return false;
  }

  bool collectInputs() {
    if (serialized.executions.empty() || buffers.storageObjects.empty())
      return fail(EventGraphFailureKind::Deferred,
                  EventGraphFailureReason::MissingPlanFact,
                  "event graph requires executions and selected storage");

    for (const ExecutionInstanceId &execution : serialized.executions)
      if (!executionSet.insert(execution).second)
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "event graph input has duplicate executions");

    for (const RegionGroupPlan &group : regions.groups) {
      for (const ExecutionInstancePlan &execution : group.executions) {
        auto [position, inserted] = executionPlans.try_emplace(
            execution.id, std::make_pair(group.tile, &execution));
        (void)position;
        if (!inserted)
          return fail(EventGraphFailureKind::CompilerBug,
                      EventGraphFailureReason::MalformedPlan,
                      "region plan has duplicate required executions");
      }
    }
    for (const ExecutionInstanceId &execution : executionSet) {
      auto plan = executionPlans.find(execution);
      if (plan == executionPlans.end() ||
          plan->second.first != tileOf(execution))
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "serialized execution is missing from its region plan");
    }

    std::set<ExecutionInstanceId> scoped;
    for (const TemporalScopePlan &scope : temporal.scopes) {
      const ExecutionInstanceId *execution = getRequiredExecution(scope.id);
      if (!execution)
        continue;
      if (!executionSet.count(*execution))
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "temporal scope references an unknown execution");
      scoped.insert(*execution);
    }
    std::set<ExecutionInstanceId> temporallyScheduled;
    for (const ExecutionInstanceId &execution : executionSet)
      if (!std::holds_alternative<RequiredMergeExecution>(execution.source))
        temporallyScheduled.insert(execution);
    if (scoped != temporallyScheduled)
      return fail(EventGraphFailureKind::Deferred,
                  EventGraphFailureReason::MissingPlanFact,
                  "event graph requires temporal coverage for every iterated "
                  "execution");

    for (const MovementResourceDescription &resource : movement.resources)
      if (!movementResources.try_emplace(resource.action, &resource).second)
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "movement coordinate has duplicate action resources");

    std::map<MovementActionId, const PeerTransferGraphPlan *> selectedGraphs;
    for (const PeerTransferGraphPlan &graph : movement.plan.peerGraphs) {
      if (graph.actions.empty())
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "movement plan has a malformed peer graph");
      for (const MovementActionId &action : graph.actions)
        if (!selectedGraphs.try_emplace(action, &graph).second)
          return fail(EventGraphFailureKind::CompilerBug,
                      EventGraphFailureReason::MalformedPlan,
                      "movement peer graphs overlap one semantic action");
    }
    auto getPeerGraph = [&](const MovementActionId &id) {
      auto selected = selectedGraphs.find(id);
      return selected == selectedGraphs.end()
                 ? std::optional<PeerTransferGraphPlan>{}
                 : std::optional<PeerTransferGraphPlan>(*selected->second);
    };

    auto addMovement = [&](MovementActionId id,
                           std::optional<PeerTransferGraphPlan> peerGraph,
                           bool publication, bool gather) -> bool {
      auto resource = movementResources.find(id);
      if (resource == movementResources.end())
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "selected movement action has no resource description");
      MovementActionDescription description{
          id, std::move(peerGraph), resource->second, publication, gather};
      if (!movementActions.try_emplace(id, std::move(description)).second)
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "movement plan has duplicate semantic actions");
      return true;
    };
    for (const ExternalLoadPlan &load : movement.plan.externalLoads)
      if (!addMovement(load.id, getPeerGraph(MovementActionId(load.id)), false,
                       false))
        return false;
    for (const DDRBoundaryTransferPlan &transfer : movement.plan.ddrTransfers)
      if (!addMovement(transfer.id, getPeerGraph(MovementActionId(transfer.id)),
                       false, false))
        return false;
    for (const ReductionGatherPlan &gather : movement.plan.reductionGathers)
      if (!addMovement(gather.id, getPeerGraph(MovementActionId(gather.id)),
                       false, true))
        return false;
    for (const ResultPublicationPlan &publication : movement.plan.publications)
      if (!addMovement(publication.id, std::nullopt, true, false))
        return false;
    if (movementActions.size() != movement.resources.size())
      return fail(EventGraphFailureKind::CompilerBug,
                  EventGraphFailureReason::MalformedPlan,
                  "movement action and resource inventories differ");
    if (llvm::any_of(selectedGraphs, [&](const auto &entry) {
          return !movementActions.count(entry.first);
        }))
      return fail(EventGraphFailureKind::CompilerBug,
                  EventGraphFailureReason::MalformedPlan,
                  "movement peer graph references an unknown action");

    using AssembledBoundaryKey =
        std::tuple<analysis::RootRegionWorkId, analysis::RootBoundaryId,
                   analysis::RootUseId>;
    std::map<AssembledBoundaryKey, std::vector<DDRBoundaryTransferId>>
        assembledGroups;
    for (const DDRBoundaryTransferPlan &transfer :
         movement.plan.ddrTransfers) {
      if (selectedGraphs.count(MovementActionId(transfer.id)))
        continue;
      const BoundaryRegionValueId &destination = transfer.id.destination;
      assembledGroups[{destination.work, destination.fragment.source,
                       destination.fragment.use}]
          .push_back(transfer.id);
    }
    for (auto &[key, actions] : assembledGroups) {
      (void)key;
      llvm::sort(actions);
      actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
      if (actions.size() < 2)
        continue;
      const DDRBoundaryTransferId root = actions.front();
      assembledBoundaryGroups.emplace(root, actions);
      for (const DDRBoundaryTransferId &action : actions)
        assembledBoundaryRoot.emplace(action, root);
    }
    std::map<PhysicalVersionId, std::vector<DDRBoundaryTransferId>>
        storesBySource;
    for (const DDRBoundaryTransferPlan &transfer :
         movement.plan.ddrTransfers)
      if (assembledBoundaryRoot.count(transfer.id))
        storesBySource[transfer.source].push_back(transfer.id);
    for (auto &[source, actions] : storesBySource) {
      (void)source;
      llvm::sort(actions);
      actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
      const DDRBoundaryTransferId root = actions.front();
      for (const DDRBoundaryTransferId &action : actions)
        assembledBoundaryStoreRoot.emplace(action, root);
    }

    for (const StorageObjectPlan &object : storage.plan.storageObjects)
      if (!semanticObjects.insert(object.id).second)
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "canonical storage has duplicate semantic objects");
    for (const StorageResourceDescription &resource : storage.resources) {
      if (resource.exactDomain.getBoxes().empty())
        return fail(EventGraphFailureKind::Unsupported,
                    EventGraphFailureReason::UnsupportedResourceRange,
                    "event graph requires a finite exact storage range");
      if (!storageResources.try_emplace(resource.object, &resource).second)
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "canonical storage has duplicate resources");
    }
    for (const StorageLifetimeDescription &lifetime : storage.lifetimes)
      if (!storageLifetimes.try_emplace(lifetime.object, &lifetime).second)
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "canonical storage has duplicate lifetimes");
    if (semanticObjects.size() != storageResources.size() ||
        semanticObjects.size() != storageLifetimes.size())
      return fail(EventGraphFailureKind::Deferred,
                  EventGraphFailureReason::MissingPlanFact,
                  "canonical storage resource/lifetime coverage differs");

    for (const StorageObjectPlan &object : buffers.storageObjects)
      if (!selectedObjects.insert(object.id).second ||
          !selectedObjectTiles.try_emplace(object.id, object.tile).second ||
          !storageResources.count(object.id))
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "selected storage references an unknown physical object");
    for (const PhysicalVersionStorageBinding &binding : buffers.versionBindings)
      if (!selectedVersionObjects.try_emplace(binding.version, binding.object)
               .second ||
          !selectedObjects.count(binding.object))
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "selected version binding is incomplete");
    for (const ReductionGatherStorageBinding &binding :
         buffers.gatherStagingBindings)
      if (!selectedGatherObjects
               .try_emplace(binding.gather, binding.stagingObject)
               .second ||
          !selectedObjects.count(binding.stagingObject))
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "selected gather storage binding is incomplete");

    for (const ExecutionEventContract &contract : contracts) {
      if (!executionSet.count(contract.execution) ||
          !executionContracts.try_emplace(contract.execution, &contract).second)
        return fail(
            EventGraphFailureKind::CompilerBug,
            EventGraphFailureReason::MalformedPlan,
            "execution completion contract has an unknown or duplicate action");
    }
    for (const ExactMovementRoute &route : exactRoutes) {
      auto graph =
          llvm::find_if(movement.plan.peerGraphs,
                        [&](const PeerTransferGraphPlan &candidate) {
                          return candidate.actions == route.graphActions;
                        });
      if (graph == movement.plan.peerGraphs.end() || route.links.empty() ||
          !llvm::is_contained(graph->hops, route.transfer) ||
          !routes
               .try_emplace(std::make_pair(route.graphActions, route.transfer),
                            &route)
               .second)
        return fail(
            EventGraphFailureKind::CompilerBug,
            EventGraphFailureReason::MalformedPlan,
            "exact movement route has an unknown, duplicate, or empty action");
      if (route.links.front().source != route.transfer.source ||
          route.links.back().destination != route.transfer.destination)
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "exact movement route disagrees with its endpoints");
      for (auto indexedLink : llvm::enumerate(route.links)) {
        const MovementHop &link = indexedLink.value();
        if (link.source == link.destination ||
            (indexedLink.index() > 0 &&
             route.links[indexedLink.index() - 1].destination != link.source))
          return fail(EventGraphFailureKind::CompilerBug,
                      EventGraphFailureReason::MalformedPlan,
                      "exact movement route is discontinuous");
      }
    }
    return true;
  }

  bool addEvent(EventId id, std::optional<TileId> tile,
                llvm::ArrayRef<NCCWorker> workerDomain = {}) {
    std::vector<NCCWorker> workers(workerDomain.begin(), workerDomain.end());
    llvm::sort(workers);
    workers.erase(std::unique(workers.begin(), workers.end()), workers.end());
    PlannedEvent planned{std::move(id), card, tile, std::move(workers)};
    auto existing = events.find(planned.id);
    if (existing != events.end()) {
      if (!(existing->second.planned == planned))
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "one EventId has inconsistent descriptors");
      return true;
    }
    if (events.size() > std::numeric_limits<uint32_t>::max())
      return fail(EventGraphFailureKind::Indeterminate,
                  EventGraphFailureReason::WorkLimit,
                  "event graph ordinal domain is not representable");
    EventId key = planned.id;
    events.emplace(
        std::move(key),
        EventRecord{std::move(planned), static_cast<uint32_t>(events.size())});
    if (events.size() > limits.maxEvents)
      return fail(EventGraphFailureKind::Indeterminate,
                  EventGraphFailureReason::WorkLimit,
                  "event graph exceeded its event limit");
    return true;
  }

  bool addDependency(EventId predecessor, EventId successor,
                     EventDependencyReason reason) {
    auto predecessorEvent = events.find(predecessor);
    auto successorEvent = events.find(successor);
    if (predecessorEvent == events.end() || successorEvent == events.end()) {
      const bool missingPredecessor = predecessorEvent == events.end();
      const EventId &missing = missingPredecessor ? predecessor : successor;
      std::string detail;
      llvm::raw_string_ostream diagnostic(detail);
      diagnostic << "event dependency references an unknown "
                 << (missingPredecessor ? "predecessor" : "successor")
                 << " reason=" << static_cast<unsigned>(reason)
                 << " kind=" << static_cast<unsigned>(missing.kind);
      if (const auto *movement =
              std::get_if<MovementEventAction>(&missing.action))
        diagnostic << " movement_phase="
                   << static_cast<unsigned>(movement->phase)
                   << " payload=" << movement->payloadSlice << " action_kind="
                   << stringifyMovementActionKind(movement->action);
      if (const auto *movement =
              std::get_if<MovementEventAction>(&missing.action))
        if (movement->hop)
          diagnostic << " hop=" << movement->hop->source.getValue() << "->"
                     << movement->hop->destination.getValue();
      return fail(EventGraphFailureKind::CompilerBug,
                  EventGraphFailureReason::MalformedPlan, diagnostic.str());
    }
    if (predecessor == successor)
      return fail(EventGraphFailureKind::ExactRejection,
                  EventGraphFailureReason::HardDependencyCycle,
                  "event dependency forms a self cycle");
    dependencies.insert({std::move(predecessor), std::move(successor), reason});
    if (dependencies.size() > limits.maxDependencies)
      return fail(EventGraphFailureKind::Indeterminate,
                  EventGraphFailureReason::WorkLimit,
                  "event graph exceeded its dependency limit");
    return true;
  }

  void addResourceUse(EventId event, ResourceKey resource, ResourceUseMode mode,
                      EventId until,
                      ResourceKnowledge knowledge = ResourceKnowledge::Exact) {
    resourceUses.insert({std::move(event), std::move(resource), mode,
                         ResourceIntervalKind::IssueToCompletion,
                         std::move(until), knowledge});
  }

  bool addExecutionEvents() {
    for (const ExecutionInstanceId &execution : executionSet) {
      TileId tile = tileOf(execution);
      EventId issue = executionEvent(execution, PlannedEventKind::ComputeIssue);
      EventId completion =
          executionEvent(execution, PlannedEventKind::Completion);
      llvm::ArrayRef<NCCWorker> workers;
      uint32_t participants = 0;
      CompletionProtocol protocol = CompletionProtocol::Unknown;
      auto contract = executionContracts.find(execution);
      if (contract == executionContracts.end())
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "execution has no typed completion contract");
      workers = contract->second->workerDomain;
      const NCCCompletionKind kind = contract->second->completion;
      participants = contract->second->participantMask;
      if ((kind == NCCCompletionKind::OrderedAsynchronousIssue ||
           kind == NCCCompletionKind::SynchronousWriteback) &&
          workers.empty())
        return fail(EventGraphFailureKind::ExactRejection,
                    EventGraphFailureReason::EmptyWorkerDomain,
                    "worker-capable execution has an empty worker domain");
      if (kind == NCCCompletionKind::ParticipantJoin &&
          (participants == 0 || (participants & ~kAllNCCWorkersMask) != 0))
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "NCC participant completion has an invalid mask");
      switch (kind) {
      case NCCCompletionKind::None:
        protocol = CompletionProtocol::NoAsynchronousCompletion;
        break;
      case NCCCompletionKind::OrderedAsynchronousIssue:
      case NCCCompletionKind::ParticipantJoin:
        protocol = CompletionProtocol::NCCParticipant;
        break;
      case NCCCompletionKind::SynchronousWriteback:
        protocol = CompletionProtocol::NCCSynchronousWriteback;
        break;
      }
      if (!addEvent(issue, tile, workers) || !addEvent(completion, tile) ||
          !addDependency(issue, completion, EventDependencyReason::Completion))
        return false;
      completions.insert({issue, completion, protocol, participants});
      if (contract->second->foldedInto.empty())
        addResourceUse(issue, TileEngineResource{tile},
                       ResourceUseMode::CapacityUnits, completion,
                       ResourceKnowledge::Estimate);
    }
    for (const auto &[execution, contract] : executionContracts)
      for (const ExecutionInstanceId &consumer : contract->foldedInto) {
        if (!executionSet.count(consumer) || consumer == execution)
          return fail(EventGraphFailureKind::CompilerBug,
                      EventGraphFailureReason::MalformedPlan,
                      "folded execution references an unknown consumer");
        if (!addDependency(
                executionEvent(execution, PlannedEventKind::Completion),
                executionEvent(consumer, PlannedEventKind::ComputeIssue),
                EventDependencyReason::SSAValue))
          return false;
      }
    return true;
  }

  bool addMovementEvents() {
    llvm::SmallVector<NCCWorker, 3> allWorkers;
    for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker)
      allWorkers.push_back(static_cast<NCCWorker>(worker));

    auto payloadCount =
        [&](const MovementActionId &action) -> std::optional<uint32_t> {
      auto resource = movementResources.find(action);
      if (resource == movementResources.end())
        return std::nullopt;
      llvm::ArrayRef<analysis::StaticRectangularIndexSet> boxes =
          resource->second->exactDomain.getBoxes();
      if (!boxes.empty()) {
        if (boxes.size() > std::numeric_limits<uint32_t>::max())
          return std::nullopt;
        return static_cast<uint32_t>(boxes.size());
      }
      auto normalized =
          analysis::normalizeFiniteExactIndexSet(resource->second->exactDomain);
      if (mlir::failed(normalized) || normalized->getBoxes().empty() ||
          normalized->getBoxes().size() > std::numeric_limits<uint32_t>::max())
        return std::nullopt;
      return static_cast<uint32_t>(normalized->getBoxes().size());
    };

    auto addLogicalAction = [&](const MovementActionId &action,
                                std::optional<TileId> tile) {
      return addEvent(movementEvent(action, PlannedEventKind::MovementIssue),
                      tile) &&
             addEvent(movementEvent(action, PlannedEventKind::Completion),
                      tile);
    };

    auto addDDRPhase =
        [&](const MovementActionId &action, MovementEventPhase phase,
            uint32_t payloadSlice, bool completesLogicalAction,
            TileId tile) -> std::optional<std::pair<EventId, EventId>> {
      EventId logicalIssue =
          movementEvent(action, PlannedEventKind::MovementIssue);
      EventId logicalCompletion =
          movementEvent(action, PlannedEventKind::Completion);
      EventId issue = movementEvent(action, PlannedEventKind::MovementIssue,
                                    phase, payloadSlice);
      EventId completion = movementEvent(action, PlannedEventKind::Completion,
                                         phase, payloadSlice);
      if (!addEvent(issue, tile, allWorkers) || !addEvent(completion, tile) ||
          !addDependency(logicalIssue, issue,
                         EventDependencyReason::TransferReady) ||
          !addDependency(issue, completion,
                         EventDependencyReason::Completion))
        return std::nullopt;
      if (completesLogicalAction &&
          !addDependency(completion, logicalCompletion,
                         EventDependencyReason::Completion))
        return std::nullopt;
      completions.insert(
          {issue, completion, CompletionProtocol::NCCParticipant, 0});
      addResourceUse(issue, CardDDRResource{card},
                     ResourceUseMode::CapacityUnits, completion,
                     ResourceKnowledge::Estimate);
      return std::pair<EventId, EventId>{issue, completion};
    };

    for (const auto &[action, description] : movementActions) {
      const std::optional<TileId> source = description.resource->sourceTile;
      const std::optional<TileId> destination =
          description.resource->destinationTile;
      const std::optional<TileId> scope = destination ? destination : source;
      if (!addLogicalAction(action, scope))
        return false;
      if (description.peerGraph)
        continue;

      std::optional<uint32_t> pieces = payloadCount(action);
      if (!pieces)
        return fail(EventGraphFailureKind::Unsupported,
                    EventGraphFailureReason::UnsupportedResourceRange,
                    "DDR movement has no finite exact payload cover");
      const bool cardTransfer =
          std::holds_alternative<DDRBoundaryTransferId>(action) ||
          std::holds_alternative<ReductionGatherId>(action);
      const uint32_t phaseCount = cardTransfer ? 1 : *pieces;
      for (uint32_t payloadSlice = 0; payloadSlice < phaseCount;
           ++payloadSlice) {
        if (std::holds_alternative<ExternalLoadId>(action)) {
          if (!destination || !addDDRPhase(action, MovementEventPhase::DDRLoad,
                                           payloadSlice,
                                           /*completesLogicalAction=*/true,
                                           *destination))
            return false;
        } else if (std::holds_alternative<ResultPublicationId>(action)) {
          if (!source || !addDDRPhase(action, MovementEventPhase::DDRStore,
                                      payloadSlice,
                                      /*completesLogicalAction=*/true, *source))
            return false;
        } else {
          if (!source || !destination)
            return fail(EventGraphFailureKind::CompilerBug,
                        EventGraphFailureReason::MalformedPlan,
                        "DDR movement has incomplete endpoints");
          const auto *boundary = std::get_if<DDRBoundaryTransferId>(&action);
          auto assembled = boundary ? assembledBoundaryRoot.find(*boundary)
                                    : assembledBoundaryRoot.end();
          const bool assembledMember =
              assembled != assembledBoundaryRoot.end();
          const bool assembledRoot =
              assembledMember && assembled->first == assembled->second;
          auto selectedStore =
              boundary ? assembledBoundaryStoreRoot.find(*boundary)
                       : assembledBoundaryStoreRoot.end();
          const bool assembledStoreRoot =
              assembledMember &&
              selectedStore != assembledBoundaryStoreRoot.end() &&
              selectedStore->first == selectedStore->second;
          std::optional<std::pair<EventId, EventId>> store;
          if (!assembledMember || assembledStoreRoot)
            store = addDDRPhase(
                action, MovementEventPhase::DDRStore, payloadSlice,
                /*completesLogicalAction=*/!assembledMember, *source);
          std::optional<std::pair<EventId, EventId>> load;
          if (!assembledMember || assembledRoot)
            load = addDDRPhase(
                action, MovementEventPhase::DDRLoad, payloadSlice,
                /*completesLogicalAction=*/!assembledMember, *destination);
          if (assembledMember) {
            if ((assembledStoreRoot && !store) || (assembledRoot && !load))
              return false;
            continue;
          }
          if (!store || !load ||
              !addDependency(store->second, load->first,
                             EventDependencyReason::TransferReady))
            return false;
        }
      }

      if (description.publication) {
        EventId observable =
            movementEvent(action, PlannedEventKind::ObservableWrite);
        EventId completion =
            movementEvent(action, PlannedEventKind::Completion);
        if (!addEvent(observable, std::nullopt) ||
            !addDependency(completion, observable,
                           EventDependencyReason::Publication))
          return false;
      }
    }

    for (const auto &[root, actions] : assembledBoundaryGroups) {
      MovementActionId rootAction(root);
      EventId loadIssue = movementEvent(
          rootAction, PlannedEventKind::MovementIssue,
          MovementEventPhase::DDRLoad, /*payloadSlice=*/0);
      EventId loadCompletion = movementEvent(
          rootAction, PlannedEventKind::Completion,
          MovementEventPhase::DDRLoad, /*payloadSlice=*/0);
      for (const DDRBoundaryTransferId &member : actions) {
        MovementActionId action(member);
        auto selectedStore = assembledBoundaryStoreRoot.find(member);
        if (selectedStore == assembledBoundaryStoreRoot.end())
          return fail(EventGraphFailureKind::CompilerBug,
                      EventGraphFailureReason::MalformedPlan,
                      "assembled boundary member has no store root");
        MovementActionId storeAction(selectedStore->second);
        EventId logicalIssue =
            movementEvent(action, PlannedEventKind::MovementIssue);
        EventId storeIssue = movementEvent(
            storeAction, PlannedEventKind::MovementIssue,
            MovementEventPhase::DDRStore, /*payloadSlice=*/0);
        EventId storeCompletion = movementEvent(
            storeAction, PlannedEventKind::Completion,
            MovementEventPhase::DDRStore, /*payloadSlice=*/0);
        EventId logicalCompletion =
            movementEvent(action, PlannedEventKind::Completion);
        if (!addDependency(logicalIssue, storeIssue,
                           EventDependencyReason::TransferReady) ||
            !addDependency(storeCompletion, loadIssue,
                           EventDependencyReason::TransferReady) ||
            !addDependency(loadCompletion, logicalCompletion,
                           EventDependencyReason::Completion))
          return false;
      }
    }

    std::set<std::vector<MovementActionId>> emittedGraphs;
    for (const auto &[action, description] : movementActions) {
      if (!description.peerGraph ||
          !emittedGraphs.insert(description.peerGraph->actions).second)
        continue;
      const PeerTransferGraphPlan &graph = *description.peerGraph;
      const MovementActionId anchor = graph.actions.front();
      std::map<int64_t, MovementHop> incoming;
      std::set<int64_t> roots;
      for (const MovementHop &hop : graph.hops) {
        roots.insert(hop.source.getValue());
        if (!incoming.try_emplace(hop.destination.getValue(), hop).second)
          return fail(EventGraphFailureKind::CompilerBug,
                      EventGraphFailureReason::MalformedPlan,
                      "peer graph gives one Tile multiple parents");
      }
      for (const auto &[tile, hop] : incoming) {
        (void)hop;
        roots.erase(tile);
      }
      if (roots.size() != 1)
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "peer graph does not have one root");
      const int64_t root = *roots.begin();
      std::optional<uint32_t> pieces = payloadCount(anchor);
      if (!pieces)
        return fail(EventGraphFailureKind::Unsupported,
                    EventGraphFailureReason::UnsupportedResourceRange,
                    "peer movement has no finite exact payload cover");

      for (uint32_t payloadSlice = 0; payloadSlice < *pieces; ++payloadSlice)
        for (const MovementHop &hop : graph.hops) {
          EventId receiveIssue =
              movementEvent(anchor, PlannedEventKind::MovementIssue,
                            MovementEventPhase::PeerReceive, payloadSlice, hop);
          EventId receiveCompletion =
              movementEvent(anchor, PlannedEventKind::Completion,
                            MovementEventPhase::PeerReceive, payloadSlice, hop);
          EventId sendIssue =
              movementEvent(anchor, PlannedEventKind::MovementIssue,
                            MovementEventPhase::PeerSend, payloadSlice, hop);
          EventId sendCompletion =
              movementEvent(anchor, PlannedEventKind::Completion,
                            MovementEventPhase::PeerSend, payloadSlice, hop);
          if (!addEvent(receiveIssue, hop.destination) ||
              !addEvent(receiveCompletion, hop.destination) ||
              !addEvent(sendIssue, hop.source) ||
              !addEvent(sendCompletion, hop.source) ||
              !addDependency(receiveIssue, receiveCompletion,
                             EventDependencyReason::Completion) ||
              !addDependency(receiveIssue, sendCompletion,
                             EventDependencyReason::Completion) ||
              !addDependency(sendIssue, receiveCompletion,
                             EventDependencyReason::Completion) ||
              !addDependency(sendIssue, sendCompletion,
                             EventDependencyReason::Completion))
            return false;
          completions.insert(
              {sendIssue, sendCompletion, CompletionProtocol::DirectDTE, 0});
          completions.insert({receiveIssue, receiveCompletion,
                              CompletionProtocol::DirectDTE, 0});
          addResourceUse(sendIssue, DirectDTESenderResource{hop.source},
                         ResourceUseMode::Exclusive, sendCompletion);
          addResourceUse(receiveIssue, DTEReceiverFSMResource{hop.destination},
                         ResourceUseMode::CapacityUnits, receiveCompletion);
          addResourceUse(sendIssue, OpaqueNoCTransferResource{hop},
                         ResourceUseMode::CapacityUnits, receiveCompletion,
                         ResourceKnowledge::Estimate);
          auto exact = routes.find(std::make_pair(graph.actions, hop));
          if (exact != routes.end())
            for (const MovementHop &link : exact->second->links)
              addResourceUse(sendIssue, DirectedNoCLinkResource{link},
                             ResourceUseMode::CapacityUnits, receiveCompletion);
        }
      for (uint32_t payloadSlice = 0; payloadSlice < *pieces; ++payloadSlice)
        for (const MovementHop &hop : graph.hops) {
          auto parent = incoming.find(hop.source.getValue());
          if (parent == incoming.end())
            continue;
          EventId parentCompletion = movementEvent(
              anchor, PlannedEventKind::Completion,
              MovementEventPhase::PeerReceive, payloadSlice, parent->second);
          EventId sendIssue =
              movementEvent(anchor, PlannedEventKind::MovementIssue,
                            MovementEventPhase::PeerSend, payloadSlice, hop);
          if (!addDependency(parentCompletion, sendIssue,
                             EventDependencyReason::TransferReady))
            return false;
        }

      for (const MovementActionId &member : graph.actions) {
        auto selected = movementActions.find(member);
        if (selected == movementActions.end() ||
            !selected->second.resource->destinationTile)
          return fail(EventGraphFailureKind::CompilerBug,
                      EventGraphFailureReason::MalformedPlan,
                      "peer graph member has no terminal resource");
        EventId issue = movementEvent(member, PlannedEventKind::MovementIssue);
        EventId completion =
            movementEvent(member, PlannedEventKind::Completion);
        for (uint32_t payloadSlice = 0; payloadSlice < *pieces; ++payloadSlice)
          for (const MovementHop &hop : graph.hops)
            if (hop.source.getValue() == root) {
              EventId rootSend = movementEvent(
                  anchor, PlannedEventKind::MovementIssue,
                  MovementEventPhase::PeerSend, payloadSlice, hop);
              if (!addDependency(issue, rootSend,
                                 EventDependencyReason::TransferReady))
                return false;
            }
        const int64_t terminal =
            selected->second.resource->destinationTile->getValue();
        if (graph.kind == PeerTransferGraphKind::ExternalLoadFanout &&
            graph.ddrRoot && member == MovementActionId(*graph.ddrRoot)) {
          if (terminal != root)
            return fail(EventGraphFailureKind::CompilerBug,
                        EventGraphFailureReason::MalformedPlan,
                        "external peer root and DDR endpoint disagree");
          for (uint32_t payloadSlice = 0; payloadSlice < *pieces;
               ++payloadSlice) {
            auto load = addDDRPhase(member, MovementEventPhase::DDRLoad,
                                    payloadSlice,
                                    /*completesLogicalAction=*/true,
                                    TileId(root));
            if (!load)
              return false;
            for (const MovementHop &hop : graph.hops)
              if (hop.source.getValue() == root) {
                EventId sendIssue = movementEvent(
                    anchor, PlannedEventKind::MovementIssue,
                    MovementEventPhase::PeerSend, payloadSlice, hop);
                if (!addDependency(load->second, sendIssue,
                                   EventDependencyReason::TransferReady))
                  return false;
              }
          }
          continue;
        }
        auto terminalHop = incoming.find(terminal);
        if (terminalHop == incoming.end())
          return fail(EventGraphFailureKind::CompilerBug,
                      EventGraphFailureReason::MalformedPlan,
                      "peer graph does not reach one member terminal");
        for (uint32_t payloadSlice = 0; payloadSlice < *pieces;
             ++payloadSlice) {
          EventId terminalReceiveIssue =
              movementEvent(anchor, PlannedEventKind::MovementIssue,
                            MovementEventPhase::PeerReceive, payloadSlice,
                            terminalHop->second);
          EventId terminalReceiveCompletion =
              movementEvent(anchor, PlannedEventKind::Completion,
                            MovementEventPhase::PeerReceive, payloadSlice,
                            terminalHop->second);
          if (graph.kind == PeerTransferGraphKind::ExternalLoadFanout) {
            if (!addDependency(issue, terminalReceiveIssue,
                               EventDependencyReason::TransferReady) ||
                !addDependency(terminalReceiveCompletion, completion,
                               EventDependencyReason::Completion))
              return false;
            continue;
          }
          EventId combineIssue =
              movementEvent(member, PlannedEventKind::LocalCombine,
                            MovementEventPhase::LocalCombine, payloadSlice);
          EventId combineCompletion =
              movementEvent(member, PlannedEventKind::Completion,
                            MovementEventPhase::LocalCombine, payloadSlice);
          if (!addDependency(issue, terminalReceiveIssue,
                             EventDependencyReason::TransferReady) ||
              !addEvent(combineIssue, TileId(terminal), allWorkers) ||
              !addEvent(combineCompletion, TileId(terminal)) ||
              !addDependency(terminalReceiveCompletion, combineIssue,
                             EventDependencyReason::TransferReady) ||
              !addDependency(combineIssue, combineCompletion,
                             EventDependencyReason::Completion) ||
              !addDependency(combineCompletion, completion,
                             EventDependencyReason::Completion))
            return false;
          completions.insert({combineIssue, combineCompletion,
                              CompletionProtocol::NCCParticipant, 0});
          addResourceUse(combineIssue, TileEngineResource{TileId(terminal)},
                         ResourceUseMode::CapacityUnits, combineCompletion,
                         ResourceKnowledge::Estimate);
        }
      }
    }
    return true;
  }

  std::optional<EventId> siteEvent(const StorageAccessSite &site,
                                   bool completion) const {
    if (const auto *execution = std::get_if<ExecutionInstanceId>(&site)) {
      EventId event = executionEvent(
          *execution, completion ? PlannedEventKind::Completion
                                 : PlannedEventKind::ComputeIssue);
      return events.count(event) ? std::optional<EventId>(event) : std::nullopt;
    }
    if (const auto *transfer = std::get_if<PeerTransferSiteId>(&site)) {
      if (transfer->graphActions.empty())
        return std::nullopt;
      EventId event =
          movementEvent(transfer->graphActions.front(),
                        completion ? PlannedEventKind::Completion
                                   : PlannedEventKind::MovementIssue,
                        transfer->endpoint == PeerTransferSiteId::Endpoint::Send
                            ? MovementEventPhase::PeerSend
                            : MovementEventPhase::PeerReceive,
                        transfer->payloadSlice, transfer->hop);
      return events.count(event) ? std::optional<EventId>(event) : std::nullopt;
    }
    const MovementActionId &action = std::get<MovementActionId>(site);
    EventId event =
        movementEvent(action, completion ? PlannedEventKind::Completion
                                         : PlannedEventKind::MovementIssue);
    return events.count(event) ? std::optional<EventId>(event) : std::nullopt;
  }

  std::optional<StorageObjectId>
  selectedObjectFor(const StorageObjectId &semantic) const {
    if (const auto *version =
            std::get_if<PhysicalVersionId>(&semantic.origin)) {
      auto selected = selectedVersionObjects.find(*version);
      return selected == selectedVersionObjects.end()
                 ? std::nullopt
                 : std::optional<StorageObjectId>(selected->second);
    }
    if (const auto *gather =
            std::get_if<ReductionGatherStagingId>(&semantic.origin)) {
      auto selected = selectedGatherObjects.find(gather->gather);
      return selected == selectedGatherObjects.end()
                 ? std::nullopt
                 : std::optional<StorageObjectId>(selected->second);
    }
    return selectedObjects.count(semantic)
               ? std::optional<StorageObjectId>(semantic)
               : std::nullopt;
  }

  std::optional<TileId> selectedTile(const StorageObjectId &object) const {
    auto found = selectedObjectTiles.find(object);
    return found == selectedObjectTiles.end()
               ? std::nullopt
               : std::optional<TileId>(found->second);
  }

  const SPMRangeResource *
  spmResource(const StorageObjectId &object,
              std::optional<uint32_t> payloadSlice = std::nullopt) {
    auto full = fullSPMResources.find(object);
    if (full == fullSPMResources.end()) {
      const StorageResourceDescription &resource = *storageResources.at(object);
      std::vector<PhysicalRangeBox> boxes;
      for (const analysis::StaticRectangularIndexSet &box :
           resource.residentDomain.getBoxes())
        boxes.push_back({box.offsets, box.sizes});
      llvm::sort(boxes);
      full =
          fullSPMResources
              .try_emplace(object, SPMRangeResource{object, std::move(boxes)})
              .first;
    }
    if (!payloadSlice)
      return &full->second;
    if (*payloadSlice >= full->second.boxes.size())
      return nullptr;
    auto key = std::make_pair(object, *payloadSlice);
    auto slice = slicedSPMResources.find(key);
    if (slice == slicedSPMResources.end())
      slice = slicedSPMResources
                  .try_emplace(key,
                               SPMRangeResource{
                                   object, {full->second.boxes[*payloadSlice]}})
                  .first;
    return &slice->second;
  }

  bool addStorageResourceUses(const StorageAccessSite &site,
                              const StorageObjectId &object,
                              ResourceUseMode mode) {
    auto addUse = [&](EventId issue, EventId completion,
                      std::optional<uint32_t> payloadSlice) {
      const SPMRangeResource *resource = spmResource(object, payloadSlice);
      if (!resource || events.find(issue) == events.end() ||
          events.find(completion) == events.end())
        return false;
      addResourceUse(issue, ResourceKey(*resource), mode, completion);
      return true;
    };

    if (const auto *transfer = std::get_if<PeerTransferSiteId>(&site)) {
      std::optional<EventId> issue = siteEvent(site, false);
      std::optional<EventId> completion = siteEvent(site, true);
      return issue && completion &&
             addUse(*issue, *completion, transfer->payloadSlice);
    }
    const auto *action = std::get_if<MovementActionId>(&site);
    if (action) {
      auto description = movementActions.find(*action);
      auto resource = movementResources.find(*action);
      std::optional<TileId> objectTile = selectedTile(object);
      if (description == movementActions.end() ||
          resource == movementResources.end() || !objectTile)
        return false;
      if (!description->second.peerGraph) {
        llvm::ArrayRef<analysis::StaticRectangularIndexSet> boxes =
            resource->second->exactDomain.getBoxes();
        std::optional<analysis::ExactIndexSet> normalized;
        if (boxes.empty()) {
          auto result = analysis::normalizeFiniteExactIndexSet(
              resource->second->exactDomain);
          if (mlir::failed(result))
            return false;
          normalized.emplace(std::move(*result));
          boxes = normalized->getBoxes();
        }
        if (boxes.empty() ||
            boxes.size() > std::numeric_limits<uint32_t>::max())
          return false;
        llvm::SmallVector<MovementEventPhase, 2> phases;
        if (std::holds_alternative<ExternalLoadId>(*action)) {
          if (resource->second->destinationTile == objectTile)
            phases.push_back(MovementEventPhase::DDRLoad);
        } else if (std::holds_alternative<ResultPublicationId>(*action)) {
          if (resource->second->sourceTile == objectTile)
            phases.push_back(MovementEventPhase::DDRStore);
        } else {
          if (resource->second->sourceTile == objectTile)
            phases.push_back(MovementEventPhase::DDRStore);
          if (resource->second->destinationTile == objectTile)
            phases.push_back(MovementEventPhase::DDRLoad);
        }
        if (!phases.empty()) {
          const bool cardTransfer =
              std::holds_alternative<DDRBoundaryTransferId>(*action) ||
              std::holds_alternative<ReductionGatherId>(*action);
          const uint32_t phaseCount = cardTransfer ? 1 : boxes.size();
          for (uint32_t payloadSlice = 0; payloadSlice < phaseCount;
               ++payloadSlice)
            for (MovementEventPhase phase : phases) {
              MovementActionId eventAction = *action;
              if (phase == MovementEventPhase::DDRLoad)
                if (const auto *transfer =
                        std::get_if<DDRBoundaryTransferId>(&*action)) {
                  auto root = assembledBoundaryRoot.find(*transfer);
                  if (root != assembledBoundaryRoot.end())
                    eventAction = MovementActionId(root->second);
                }
              if (phase == MovementEventPhase::DDRStore)
                if (const auto *transfer =
                        std::get_if<DDRBoundaryTransferId>(&*action)) {
                  auto root = assembledBoundaryStoreRoot.find(*transfer);
                  if (root != assembledBoundaryStoreRoot.end())
                    eventAction = MovementActionId(root->second);
                }
              if (!addUse(movementEvent(eventAction,
                                        PlannedEventKind::MovementIssue, phase,
                                        payloadSlice),
                          movementEvent(eventAction,
                                        PlannedEventKind::Completion, phase,
                                        payloadSlice),
                          cardTransfer ? std::nullopt
                                       : std::optional<uint32_t>(payloadSlice)))
                return false;
            }
          return true;
        }
      }
    }

    std::optional<EventId> issue = siteEvent(site, false);
    std::optional<EventId> completion = siteEvent(site, true);
    return issue && completion && addUse(*issue, *completion, std::nullopt);
  }

  bool addStorageEvents() {
    for (const auto &[semantic, lifetime] : storageLifetimes) {
      std::optional<StorageObjectId> selected = selectedObjectFor(semantic);
      if (!selected || !selectedObjects.count(*selected))
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "semantic storage lifetime has no selected object");
      std::optional<TileId> tile = selectedTile(*selected);
      if (!tile)
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "selected storage object has no Tile");
      BufferEventAction buffer{semantic, *selected};
      EventId ready{buffer, PlannedEventKind::BufferReady};
      EventId release{buffer, PlannedEventKind::BufferRelease};
      if (!addEvent(ready, tile) || !addEvent(release, tile))
        return false;
      bufferReady.try_emplace(semantic, ready);
      bufferRelease.try_emplace(semantic, release);

      std::optional<EventId> definition =
          siteEvent(lifetime->definition, /*completion=*/true);
      std::optional<EventId> definitionIssue =
          siteEvent(lifetime->definition, /*completion=*/false);
      if (!definition || !definitionIssue)
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "storage definition has no planned event");
      if (!addDependency(*definition, ready,
                         EventDependencyReason::BufferLifetime))
        return false;
      if (!addStorageResourceUses(lifetime->definition, *selected,
                                  ResourceUseMode::Write))
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "storage definition has no exact physical event");

      if (lifetime->uses.empty())
        return fail(EventGraphFailureKind::CompilerBug,
                    EventGraphFailureReason::MalformedPlan,
                    "storage lifetime has no semantic use");
      for (const StorageAccessSite &use : lifetime->uses) {
        std::optional<EventId> useIssue = siteEvent(use, false);
        std::optional<EventId> useCompletion = siteEvent(use, true);
        if (!useIssue || !useCompletion)
          return fail(EventGraphFailureKind::Deferred,
                      EventGraphFailureReason::MissingPlanFact,
                      "storage use has no planned event");
        if (*useIssue == *definitionIssue) {
          if (!(*useCompletion == *definition))
            return fail(EventGraphFailureKind::CompilerBug,
                        EventGraphFailureReason::MalformedPlan,
                        "one execution has inconsistent internal storage "
                        "completion");
          if (!addDependency(*useCompletion, release,
                             EventDependencyReason::BufferLifetime))
            return false;
          continue;
        }
        if (!addDependency(ready, *useIssue, EventDependencyReason::SSAValue) ||
            !addDependency(*useCompletion, release,
                           EventDependencyReason::BufferLifetime))
          return false;
        if (!addStorageResourceUses(use, *selected, ResourceUseMode::Read))
          return fail(EventGraphFailureKind::Deferred,
                      EventGraphFailureReason::MissingPlanFact,
                      "storage use has no exact physical event");
      }
    }
    return true;
  }

  bool addNestedDependencies() {
    for (const auto &[child, placement] : executionPlans) {
      const auto *nested = std::get_if<ExecutionInstancePlan::NestedUnder>(
          &placement.second->placement);
      if (!nested)
        continue;
      ExecutionInstanceId parent{nested->consumer};
      if (!executionSet.count(parent))
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "nested execution parent is absent from the event graph");
      if (!addDependency(executionEvent(parent, PlannedEventKind::ComputeIssue),
                         executionEvent(child, PlannedEventKind::ComputeIssue),
                         EventDependencyReason::NestedExecution) ||
          !addDependency(executionEvent(child, PlannedEventKind::Completion),
                         executionEvent(parent, PlannedEventKind::Completion),
                         EventDependencyReason::NestedExecution))
        return false;
    }
    return true;
  }

  bool addBufferOrderDependencies() {
    for (const BufferOrderRequirement &order : buffers.orderRequirements) {
      StorageObjectId earlier{StorageObjectOrigin{order.earlier}};
      StorageObjectId later{StorageObjectOrigin{order.later}};
      auto release = bufferRelease.find(earlier);
      auto ready = bufferReady.find(later);
      if (release == bufferRelease.end() || ready == bufferReady.end())
        return fail(EventGraphFailureKind::Deferred,
                    EventGraphFailureReason::MissingPlanFact,
                    "buffer reuse order has no semantic lifetime endpoints");
      if (!addDependency(release->second, ready->second,
                         EventDependencyReason::BufferLifetime))
        return false;
    }
    return true;
  }

  std::map<EventId, std::set<EventId>> adjacency() const {
    std::map<EventId, std::set<EventId>> result;
    for (const auto &[id, event] : events) {
      (void)event;
      result.try_emplace(id);
    }
    for (const EventDependency &edge : dependencies)
      result[edge.predecessor].insert(edge.successor);
    return result;
  }

  std::optional<uint32_t> eventOrdinal(const EventId &event) const {
    auto found = events.find(event);
    return found == events.end()
               ? std::nullopt
               : std::optional<uint32_t>(found->second.ordinal);
  }

  std::vector<EventId> findCycle() const {
    std::vector<std::vector<uint32_t>> successors(events.size());
    for (const EventDependency &edge : dependencies) {
      std::optional<uint32_t> predecessor = eventOrdinal(edge.predecessor);
      std::optional<uint32_t> successor = eventOrdinal(edge.successor);
      if (predecessor && successor)
        successors[*predecessor].push_back(*successor);
    }
    std::vector<const EventId *> ids(events.size(), nullptr);
    for (const auto &[id, event] : events)
      ids[event.ordinal] = &id;
    std::vector<uint8_t> colors(events.size(), 0);
    std::vector<uint32_t> stack;
    std::vector<size_t> stackPositions(events.size(), 0);
    std::vector<EventId> cycle;
    std::function<bool(uint32_t)> visit = [&](uint32_t event) {
      colors[event] = 1;
      stackPositions[event] = stack.size();
      stack.push_back(event);
      for (uint32_t successor : successors[event]) {
        if (colors[successor] == 0) {
          if (visit(successor))
            return true;
        } else if (colors[successor] == 1) {
          for (uint32_t member : llvm::ArrayRef<uint32_t>(stack).drop_front(
                   stackPositions[successor]))
            cycle.push_back(*ids[member]);
          cycle.push_back(*ids[successor]);
          return true;
        }
      }
      stack.pop_back();
      colors[event] = 2;
      return false;
    };
    for (const auto &[id, event] : events) {
      (void)id;
      if (colors[event.ordinal] == 0 && visit(event.ordinal))
        break;
    }
    return cycle;
  }

  bool reaches(const EventId &source, const EventId &destination,
               const std::map<EventId, std::set<EventId>> &successors) const {
    std::set<EventId> visited;
    std::vector<EventId> stack{source};
    while (!stack.empty()) {
      EventId current = stack.back();
      stack.pop_back();
      if (!visited.insert(current).second)
        continue;
      auto next = successors.find(current);
      if (next == successors.end())
        continue;
      for (const EventId &successor : next->second) {
        if (successor == destination)
          return true;
        stack.push_back(successor);
      }
    }
    return false;
  }

  static bool conflicts(const PlannedResourceUse &lhs,
                        const PlannedResourceUse &rhs) {
    if (lhs.knowledge != ResourceKnowledge::Exact ||
        rhs.knowledge != ResourceKnowledge::Exact)
      return false;
    if (lhs.mode == ResourceUseMode::Exclusive ||
        rhs.mode == ResourceUseMode::Exclusive)
      return true;
    return lhs.mode == ResourceUseMode::Write ||
           rhs.mode == ResourceUseMode::Write;
  }

  void buildOrderChoices() {
    if (llvm::none_of(resourceUses, [](const PlannedResourceUse &use) {
          return std::holds_alternative<DirectDTESenderResource>(use.resource);
        }))
      return;
    std::map<ResourceKey, std::vector<const PlannedResourceUse *>> byResource;
    for (const PlannedResourceUse &use : resourceUses)
      byResource[use.resource].push_back(&use);
    const auto successors = adjacency();
    for (auto &[resource, uses] : byResource) {
      if (!std::holds_alternative<DirectDTESenderResource>(resource))
        continue;
      std::set<EventId> unordered;
      for (size_t left = 0; left < uses.size(); ++left)
        for (size_t right = left + 1; right < uses.size(); ++right) {
          if (uses[left]->event == uses[right]->event ||
              !conflicts(*uses[left], *uses[right]))
            continue;
          if (reaches(uses[left]->event, uses[right]->event, successors) ||
              reaches(uses[right]->event, uses[left]->event, successors))
            continue;
          unordered.insert(uses[left]->event);
          unordered.insert(uses[right]->event);
        }
      if (unordered.size() > 1)
        orderChoices.push_back(
            {resource,
             std::vector<EventId>(unordered.begin(), unordered.end())});
    }
  }

  void buildComponents() {
    std::vector<uint32_t> parents(events.size());
    for (uint32_t ordinal = 0; ordinal < parents.size(); ++ordinal)
      parents[ordinal] = ordinal;
    std::function<uint32_t(uint32_t)> findRoot = [&](uint32_t event) {
      if (parents[event] == event)
        return event;
      parents[event] = findRoot(parents[event]);
      return parents[event];
    };
    auto unite = [&](const EventId &lhs, const EventId &rhs) {
      std::optional<uint32_t> left = eventOrdinal(lhs);
      std::optional<uint32_t> right = eventOrdinal(rhs);
      if (!left || !right)
        return;
      left = findRoot(*left);
      right = findRoot(*right);
      if (*left == *right)
        return;
      if (*right < *left)
        std::swap(left, right);
      parents[*right] = *left;
    };
    for (const EventDependency &edge : dependencies)
      unite(edge.predecessor, edge.successor);
    for (const DisjunctiveResourceOrder &choice : orderChoices)
      for (size_t index = 1; index < choice.events.size(); ++index)
        unite(choice.events.front(), choice.events[index]);

    std::vector<std::vector<EventId>> members(events.size());
    for (const auto &[id, event] : events)
      members[findRoot(event.ordinal)].push_back(id);
    std::set<uint32_t> emitted;
    for (const auto &[id, event] : events) {
      (void)id;
      uint32_t root = findRoot(event.ordinal);
      if (emitted.insert(root).second)
        components.push_back({std::move(members[root])});
    }
  }

  CardId card;
  const RegionPlan &regions;
  const TemporalPlan &temporal;
  const SerializedExecutionPlan &serialized;
  const CanonicalMovementCoordinate &movement;
  const CanonicalStorageCoordinate &storage;
  const BufferPlan &buffers;
  llvm::ArrayRef<ExecutionEventContract> contracts;
  llvm::ArrayRef<ExactMovementRoute> exactRoutes;
  EventGraphLimits limits;

  std::set<ExecutionInstanceId> executionSet;
  std::map<ExecutionInstanceId,
           std::pair<TileId, const ExecutionInstancePlan *>>
      executionPlans;
  std::map<ExecutionInstanceId, const ExecutionEventContract *>
      executionContracts;
  std::map<MovementActionId, const MovementResourceDescription *>
      movementResources;
  std::map<MovementActionId, MovementActionDescription> movementActions;
  std::map<DDRBoundaryTransferId, DDRBoundaryTransferId>
      assembledBoundaryRoot;
  std::map<DDRBoundaryTransferId, DDRBoundaryTransferId>
      assembledBoundaryStoreRoot;
  std::map<DDRBoundaryTransferId, std::vector<DDRBoundaryTransferId>>
      assembledBoundaryGroups;
  std::map<std::pair<std::vector<MovementActionId>, MovementHop>,
           const ExactMovementRoute *>
      routes;
  std::set<StorageObjectId> semanticObjects;
  std::set<StorageObjectId> selectedObjects;
  std::map<StorageObjectId, TileId> selectedObjectTiles;
  std::map<StorageObjectId, const StorageResourceDescription *>
      storageResources;
  std::map<StorageObjectId, const StorageLifetimeDescription *>
      storageLifetimes;
  std::map<PhysicalVersionId, StorageObjectId> selectedVersionObjects;
  std::map<ReductionGatherId, StorageObjectId> selectedGatherObjects;
  std::map<StorageObjectId, EventId> bufferReady;
  std::map<StorageObjectId, EventId> bufferRelease;
  std::map<StorageObjectId, SPMRangeResource> fullSPMResources;
  std::map<std::pair<StorageObjectId, uint32_t>, SPMRangeResource>
      slicedSPMResources;

  std::map<EventId, EventRecord> events;
  std::set<EventDependency> dependencies;
  std::set<CompletionObligation> completions;
  std::set<PlannedResourceUse> resourceUses;
  std::vector<DisjunctiveResourceOrder> orderChoices;
  std::vector<EventComponent> components;
  std::optional<EventGraphBuildResult> failure;
};

} // namespace event_graph_detail

static std::optional<unsigned>
getPassthroughInput(mlir::linalg::GenericOp generic) {
  if (generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
    return std::nullopt;
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
      generic.getBody()->getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return std::nullopt;
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(yield.getValues()[0]);
  if (!argument || argument.getOwner() != generic.getBody() ||
      argument.getArgNumber() >= generic.getNumDpsInputs())
    return std::nullopt;
  return argument.getArgNumber();
}

static bool isMetadataPassthroughExecution(
    mlir::Operation *operation, const ExecutionInstanceId &execution,
    const analysis::RootRegionWork &work) {
  auto generic = mlir::dyn_cast_or_null<mlir::linalg::GenericOp>(operation);
  const auto *required =
      std::get_if<RequiredRootExecution>(&execution.source);
  std::optional<unsigned> input =
      generic ? getPassthroughInput(generic) : std::nullopt;
  if (!generic || !required || !input)
    return false;
  auto piece = llvm::find_if(work.execution, [&](const auto &candidate) {
    return candidate.shard == required->shard;
  });
  if (piece == work.execution.end())
    return false;
  llvm::SmallVector<mlir::AffineMap, 4> maps =
      generic.getIndexingMapsArray();
  if (*input >= maps.size() || maps.size() != generic.getNumDpsInputs() + 1)
    return false;
  llvm::SmallVector<unsigned, 4> resultDims;
  for (mlir::AffineExpr expression : maps.back().getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() >= piece->iterationDomain.size())
      return false;
    resultDims.push_back(dimension.getPosition());
  }
  size_t nextResult = 0;
  std::set<unsigned> retained;
  for (mlir::AffineExpr expression : maps[*input].getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension)
      return false;
    while (nextResult < resultDims.size() &&
           resultDims[nextResult] != dimension.getPosition())
      ++nextResult;
    if (nextResult == resultDims.size())
      return false;
    retained.insert(dimension.getPosition());
    ++nextResult;
  }
  return llvm::all_of(resultDims, [&](unsigned dimension) {
    return retained.count(dimension) ||
           piece->iterationDomain[dimension].size == 1;
  });
}

static ExecutionEventContractResult deriveExecutionEventContractsImpl(
    const SerializedExecutionPlan &serialized, const RegionPlan *regions,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> roots;
  for (const analysis::RootRegionWork &work : rootWorks)
    if (!work.rootOperation ||
        !roots.try_emplace(work.id, &work).second)
      return {{},
              EventGraphFailure{EventGraphFailureKind::CompilerBug,
                                EventGraphFailureReason::MalformedPlan,
                                {},
                                "execution contracts have malformed root "
                                "work"}};
  std::set<ExecutionInstanceId> serializedExecutions(
      serialized.executions.begin(), serialized.executions.end());
  ExecutionEventContractResult result;
  for (const ExecutionInstanceId &execution : serialized.executions) {
    analysis::RootRegionWorkId work = std::visit(
        [](const auto &source) { return source.work; }, execution.source);
    auto operation = roots.find(work);
    if (operation == roots.end())
      return {{},
              EventGraphFailure{EventGraphFailureKind::CompilerBug,
                                EventGraphFailureReason::MalformedPlan,
                                {},
                                "execution contract has no typed root"}};
    ExecutionEventContract contract;
    contract.execution = execution;
    mlir::Operation *rootOperation = operation->second->rootOperation;
    if (!mlir::isMemoryEffectFree(rootOperation))
      return {{},
              EventGraphFailure{
                  EventGraphFailureKind::Unsupported,
                  EventGraphFailureReason::UnsupportedExecutionContract,
                  {},
                  "effectful structured execution needs an explicit event "
                  "contract"}};
    std::set<ExecutionInstanceId> foldedInto;
    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(rootOperation);
        regions && fill && analysis::onlyFeedsScalarInitializedReduction(
                               fill.getResult(0))) {
      for (const RegionGroupPlan &group : regions->groups)
        for (const LocalUseBinding &binding : group.localBindings) {
          const auto *producer =
              std::get_if<ExecutionInstanceId>(&binding.producer);
          if (!producer || !(*producer == execution) ||
              binding.fragment.source.kind !=
                  analysis::RootBoundaryKind::StructuredResult ||
              binding.fragment.source.semantic != work.root ||
              binding.fragment.source.index != 0)
            continue;
          analysis::RootRegionWorkId consumerWork{
              binding.fragment.use.destinationShard.root, group.tile};
          ExecutionInstanceId consumer{RequiredRootExecution{
              consumerWork, binding.fragment.use.destinationShard}};
          if (consumer == execution || !serializedExecutions.count(consumer) ||
              llvm::none_of(group.executions, [&](const auto &candidate) {
                return candidate.id == consumer;
              }))
            continue;
          foldedInto.insert(std::move(consumer));
        }
    }
    if (!foldedInto.empty()) {
      contract.completion = NCCCompletionKind::None;
      contract.foldedInto.assign(foldedInto.begin(), foldedInto.end());
    } else if (isMetadataPassthroughExecution(
                   rootOperation, execution, *operation->second)) {
      contract.completion = NCCCompletionKind::None;
    } else if (mlir::isa<mlir::linalg::LinalgOp, LinalgExtAttentionOp>(
                   rootOperation)) {
      for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker)
        contract.workerDomain.push_back(static_cast<NCCWorker>(worker));
      contract.completion = NCCCompletionKind::OrderedAsynchronousIssue;
    } else if (mlir::isa<mlir::ViewLikeOpInterface>(rootOperation)) {
      contract.completion = NCCCompletionKind::None;
    } else {
      return {{},
              EventGraphFailure{
                  EventGraphFailureKind::Unsupported,
                  EventGraphFailureReason::UnsupportedExecutionContract,
                  {},
                  "selected execution has no target completion contract"}};
    }
    result.contracts.push_back(std::move(contract));
  }
  llvm::sort(result.contracts, [](const ExecutionEventContract &lhs,
                                  const ExecutionEventContract &rhs) {
    return lhs.execution < rhs.execution;
  });
  if (std::adjacent_find(result.contracts.begin(), result.contracts.end(),
                         [](const ExecutionEventContract &lhs,
                            const ExecutionEventContract &rhs) {
                           return lhs.execution == rhs.execution;
                         }) != result.contracts.end())
    return {{},
            EventGraphFailure{EventGraphFailureKind::CompilerBug,
                              EventGraphFailureReason::MalformedPlan,
                              {},
                              "execution contracts contain duplicates"}};
  return result;
}

ExecutionEventContractResult deriveExecutionEventContracts(
    const SerializedExecutionPlan &serialized,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  return deriveExecutionEventContractsImpl(serialized, nullptr, rootWorks);
}

ExecutionEventContractResult deriveExecutionEventContracts(
    const SerializedExecutionPlan &serialized, const RegionPlan &regions,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  return deriveExecutionEventContractsImpl(serialized, &regions, rootWorks);
}

bool EventGraph::contains(const EventId &event) const {
  auto found = std::lower_bound(
      events.begin(), events.end(), event,
      [](const PlannedEvent &planned, const EventId &candidate) {
        return planned.id < candidate;
      });
  return found != events.end() && found->id == event;
}

std::optional<std::vector<EventId>> EventGraph::getReadyEvents(
    llvm::ArrayRef<EventId> completed,
    llvm::ArrayRef<EventDependency> selectedOrders) const {
  std::set<EventId> done;
  for (const EventId &event : completed)
    if (!contains(event) || !done.insert(event).second)
      return std::nullopt;
  std::map<EventId, std::set<EventId>> predecessors;
  for (const PlannedEvent &event : events)
    predecessors.try_emplace(event.id);
  auto add = [&](const EventDependency &edge) {
    if (!contains(edge.predecessor) || !contains(edge.successor) ||
        edge.predecessor == edge.successor)
      return false;
    predecessors[edge.successor].insert(edge.predecessor);
    return true;
  };
  for (const EventDependency &edge : hardDependencies)
    if (!add(edge))
      return std::nullopt;
  std::set<EventDependency> selected;
  for (const EventDependency &edge : selectedOrders) {
    const bool allowed =
        edge.reason == EventDependencyReason::EffectOrder &&
        llvm::any_of(orderChoices, [&](const auto &choice) {
          return llvm::is_contained(choice.events, edge.predecessor) &&
                 llvm::is_contained(choice.events, edge.successor);
        });
    if (!allowed || !selected.insert(edge).second || !add(edge))
      return std::nullopt;
  }

  // A completion prefix is an order ideal of the selected dependency graph.
  for (const EventId &event : done)
    if (llvm::any_of(predecessors[event], [&](const EventId &predecessor) {
          return !done.count(predecessor);
        }))
      return std::nullopt;

  // Reject a selected resource precedence that closes a cycle.
  std::map<EventId, size_t> indegree;
  std::map<EventId, std::set<EventId>> successors;
  for (const auto &[event, required] : predecessors) {
    indegree[event] = required.size();
    for (const EventId &predecessor : required)
      successors[predecessor].insert(event);
  }
  std::set<EventId> cycleReady;
  for (const auto &[event, degree] : indegree)
    if (degree == 0)
      cycleReady.insert(event);
  size_t visited = 0;
  while (!cycleReady.empty()) {
    EventId event = *cycleReady.begin();
    cycleReady.erase(cycleReady.begin());
    ++visited;
    for (const EventId &successor : successors[event])
      if (--indegree[successor] == 0)
        cycleReady.insert(successor);
  }
  if (visited != events.size())
    return std::nullopt;

  std::vector<EventId> ready;
  for (const auto &[event, required] : predecessors)
    if (!done.count(event) &&
        llvm::all_of(required, [&](const EventId &dependency) {
          return done.count(dependency);
        }))
      ready.push_back(event);
  return ready;
}

EventGraphBuildResult buildEventGraph(
    CardId card, const RegionPlan &regions, const TemporalPlan &temporal,
    const SerializedExecutionPlan &serialized,
    const CanonicalMovementCoordinate &movement,
    const CanonicalStorageCoordinate &storage, const BufferPlan &buffers,
    llvm::ArrayRef<ExecutionEventContract> executionContracts,
    llvm::ArrayRef<ExactMovementRoute> exactRoutes,
    const EventGraphLimits &limits) {
  return event_graph_detail::EventGraphBuilder(
             card, regions, temporal, serialized, movement, storage, buffers,
             executionContracts, exactRoutes, limits)
      .build();
}

} // namespace wafer::compiler::detail
