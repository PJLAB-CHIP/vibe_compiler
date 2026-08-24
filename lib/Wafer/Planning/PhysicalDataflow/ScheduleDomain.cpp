//===- ScheduleDomain.cpp - Fixed-generation event schedule -----------===//

#include "Wafer/Planning/PhysicalDataflow/ScheduleDomain.h"

#include "Wafer/Target/Core/DirectDTE.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

ScheduleDomainResult failed(ScheduleDomainFailureKind kind,
                            llvm::StringRef detail) {
  return {{}, ScheduleDomainFailure{kind, detail.str()}};
}

using Edge = std::pair<EventId, EventId>;

bool isPermutation(llvm::ArrayRef<EventId> expected,
                   llvm::ArrayRef<EventId> actual) {
  if (expected.size() != actual.size())
    return false;
  std::set<EventId> expectedSet(expected.begin(), expected.end());
  std::set<EventId> actualSet(actual.begin(), actual.end());
  return expectedSet.size() == expected.size() && expectedSet == actualSet;
}

bool respects(llvm::ArrayRef<EventId> order, const std::set<Edge> &edges) {
  std::map<EventId, size_t> positions;
  for (auto [index, event] : llvm::enumerate(order))
    if (!positions.try_emplace(event, index).second)
      return false;
  for (const auto &[predecessor, successor] : edges) {
    auto before = positions.find(predecessor);
    auto after = positions.find(successor);
    if (before != positions.end() && after != positions.end() &&
        before->second >= after->second)
      return false;
  }
  return true;
}

std::optional<std::vector<EventId>>
getFirstTopologicalOrder(llvm::ArrayRef<EventId> events,
                         const std::set<Edge> &edges) {
  std::map<EventId, size_t> indegree;
  std::map<EventId, std::set<EventId>> successors;
  std::set<EventId> members(events.begin(), events.end());
  if (members.size() != events.size())
    return std::nullopt;
  for (const EventId &event : members)
    indegree.try_emplace(event, 0);
  for (const auto &[predecessor, successor] : edges)
    if (members.count(predecessor) && members.count(successor) &&
        successors[predecessor].insert(successor).second)
      ++indegree[successor];
  std::set<EventId> ready;
  for (const auto &[event, degree] : indegree)
    if (degree == 0)
      ready.insert(event);
  std::vector<EventId> order;
  while (!ready.empty()) {
    EventId event = *ready.begin();
    ready.erase(ready.begin());
    order.push_back(event);
    for (const EventId &successor : successors[event])
      if (--indegree[successor] == 0)
        ready.insert(successor);
  }
  return order.size() == events.size()
             ? std::optional<std::vector<EventId>>(std::move(order))
             : std::nullopt;
}

enum class NextOrderKind : uint8_t { Order, End, Indeterminate };
struct NextOrder {
  NextOrderKind kind = NextOrderKind::End;
  std::vector<EventId> order;
};

NextOrder getNextTopologicalOrder(llvm::ArrayRef<EventId> events,
                                  llvm::ArrayRef<EventId> current,
                                  const std::set<Edge> &edges, uint64_t limit) {
  if (!isPermutation(events, current) || !respects(current, edges))
    return {NextOrderKind::Indeterminate, {}};
  std::vector<EventId> next(current.begin(), current.end());
  uint64_t steps = 0;
  while (std::next_permutation(next.begin(), next.end())) {
    if (++steps > limit)
      return {NextOrderKind::Indeterminate, {}};
    if (respects(next, edges))
      return {NextOrderKind::Order, std::move(next)};
  }
  return {NextOrderKind::End, {}};
}

bool isAcyclic(llvm::ArrayRef<PlannedEvent> events,
               const std::set<Edge> &edges) {
  std::vector<EventId> ids;
  for (const PlannedEvent &event : events)
    ids.push_back(event.id);
  return getFirstTopologicalOrder(ids, edges).has_value();
}

bool reaches(const EventId &source, const EventId &destination,
             const std::set<Edge> &edges) {
  std::map<EventId, std::vector<EventId>> successors;
  for (const auto &[predecessor, successor] : edges)
    successors[predecessor].push_back(successor);
  std::set<EventId> visited;
  std::vector<EventId> stack{source};
  while (!stack.empty()) {
    EventId current = stack.back();
    stack.pop_back();
    if (!visited.insert(current).second)
      continue;
    for (const EventId &successor : successors[current]) {
      if (successor == destination)
        return true;
      stack.push_back(successor);
    }
  }
  return false;
}

std::set<Edge> getHardEdges(llvm::ArrayRef<EventDependency> dependencies) {
  std::set<Edge> edges;
  for (const EventDependency &dependency : dependencies)
    edges.insert({dependency.predecessor, dependency.successor});
  return edges;
}

std::optional<EventId> getResourceReleaseEvent(const ScheduleDomainInput &input,
                                               const ResourceKey &resource,
                                               const EventId &event) {
  std::optional<EventId> release;
  for (const PlannedResourceUse &use : input.resourceUses) {
    if (!(use.event == event) || !(use.resource == resource) ||
        use.knowledge != ResourceKnowledge::Exact)
      continue;
    EventId candidate =
        use.interval == ResourceIntervalKind::Instantaneous || !use.until
            ? event
            : *use.until;
    if (release && !(*release == candidate))
      return std::nullopt;
    release = candidate;
  }
  return release;
}

bool rangesOverlap(const SPMRangeResource &lhs, const SPMRangeResource &rhs) {
  if (!(lhs.object == rhs.object))
    return false;
  for (const PhysicalRangeBox &left : lhs.boxes)
    for (const PhysicalRangeBox &right : rhs.boxes) {
      if (left.offsets.size() != right.offsets.size() ||
          left.sizes.size() != right.sizes.size() ||
          left.offsets.size() != left.sizes.size())
        continue;
      bool overlap = true;
      for (auto [leftOffset, leftSize, rightOffset, rightSize] :
           llvm::zip_equal(left.offsets, left.sizes, right.offsets,
                           right.sizes))
        overlap &= leftSize > 0 && rightSize > 0 &&
                   static_cast<__int128>(leftOffset) <
                       static_cast<__int128>(rightOffset) + rightSize &&
                   static_cast<__int128>(rightOffset) <
                       static_cast<__int128>(leftOffset) + leftSize;
      if (overlap)
        return true;
    }
  return false;
}

bool modesConflict(ResourceUseMode lhs, ResourceUseMode rhs) {
  return lhs == ResourceUseMode::Exclusive ||
         rhs == ResourceUseMode::Exclusive || lhs == ResourceUseMode::Write ||
         rhs == ResourceUseMode::Write;
}

bool hasExactSPMConflict(const ScheduleDomainInput &input, const EventId &lhs,
                         const EventId &rhs) {
  for (const PlannedResourceUse &left : input.resourceUses) {
    if (!(left.event == lhs) || left.knowledge != ResourceKnowledge::Exact)
      continue;
    const auto *leftRange = std::get_if<SPMRangeResource>(&left.resource);
    if (!leftRange)
      continue;
    for (const PlannedResourceUse &right : input.resourceUses) {
      if (!(right.event == rhs) || right.knowledge != ResourceKnowledge::Exact)
        continue;
      const auto *rightRange = std::get_if<SPMRangeResource>(&right.resource);
      if (rightRange && rangesOverlap(*leftRange, *rightRange) &&
          modesConflict(left.mode, right.mode))
        return true;
    }
  }
  return false;
}

bool hasExactResourceConflict(const ScheduleDomainInput &input,
                              const EventId &lhs, const EventId &rhs,
                              llvm::ArrayRef<EventResourceBinding> bindings) {
  if (hasExactSPMConflict(input, lhs, rhs))
    return true;
  for (const PlannedResourceUse &left : input.resourceUses) {
    if (!(left.event == lhs) || left.knowledge != ResourceKnowledge::Exact)
      continue;
    for (const PlannedResourceUse &right : input.resourceUses) {
      if (!(right.event == rhs) ||
          right.knowledge != ResourceKnowledge::Exact ||
          !(left.resource == right.resource))
        continue;
      if (std::holds_alternative<DTEReceiverFSMResource>(left.resource)) {
        auto leftBinding = llvm::find_if(bindings, [&](const auto &binding) {
          return binding.event == lhs &&
                 binding.instance.resource == left.resource;
        });
        auto rightBinding = llvm::find_if(bindings, [&](const auto &binding) {
          return binding.event == rhs &&
                 binding.instance.resource == right.resource;
        });
        if (leftBinding != bindings.end() && rightBinding != bindings.end()) {
          if (leftBinding->instance.lane != rightBinding->instance.lane)
            continue;
          return true;
        }
      }
      if (modesConflict(left.mode, right.mode))
        return true;
    }
  }
  return false;
}

std::vector<CompletionPlacement> deriveCompletionPlacements(
    const ScheduleDomainInput &input,
    const std::map<EventId, NCCWorker> &selectedWorkers,
    llvm::ArrayRef<ControlOrder> controlOrders,
    llvm::ArrayRef<EventResourceBinding> resourceBindings,
    const std::set<Edge> &selectedEdges,
    const std::map<EventId, std::optional<uint32_t>> &eventStages) {
  std::map<EventId, std::pair<const std::vector<EventId> *, size_t>> positions;
  for (const ControlOrder &control : controlOrders)
    for (auto [index, event] : llvm::enumerate(control.events))
      positions.emplace(event, std::make_pair(&control.events, index));

  std::vector<CompletionPlacement> placements;
  for (const CompletionObligation &obligation : input.completionObligations) {
    uint32_t participants = obligation.participantMask;
    if (obligation.protocol == CompletionProtocol::NCCParticipant &&
        participants == 0) {
      auto worker = selectedWorkers.find(obligation.issue);
      auto completion = positions.find(obligation.completion);
      if (worker == selectedWorkers.end() || completion == positions.end())
        return {};
      bool sameWorkerHandoff = false;
      const std::vector<EventId> &order = *completion->second.first;
      for (size_t index = completion->second.second + 1; index < order.size();
           ++index) {
        if ((order[index].kind == PlannedEventKind::BufferRelease ||
             order[index].kind == PlannedEventKind::ObservableWrite) &&
            reaches(obligation.completion, order[index], selectedEdges))
          break;
        if (!hasExactSPMConflict(input, obligation.issue, order[index]))
          continue;
        auto consumerWorker = selectedWorkers.find(order[index]);
        sameWorkerHandoff = consumerWorker != selectedWorkers.end() &&
                            consumerWorker->second == worker->second;
        break;
      }
      if (!sameWorkerHandoff)
        participants = uint32_t{1} << static_cast<uint32_t>(worker->second);
    }
    if (obligation.protocol == CompletionProtocol::NCCSynchronousWriteback) {
      auto worker = selectedWorkers.find(obligation.issue);
      if (worker == selectedWorkers.end())
        return {};
      participants = uint32_t{1} << static_cast<uint32_t>(worker->second);
    }
    EventId boundary = obligation.completion;
    auto completion = positions.find(obligation.completion);
    const bool emitsCompletion =
        (obligation.protocol == CompletionProtocol::NCCParticipant &&
         participants != 0) ||
        obligation.protocol == CompletionProtocol::DirectDTE;
    if (completion != positions.end() && emitsCompletion) {
      const std::vector<EventId> &order = *completion->second.first;
      for (size_t index = completion->second.second + 1; index < order.size();
           ++index) {
        if (eventStages.at(order[index]) !=
            eventStages.at(obligation.completion))
          break;
        const bool terminalObserver =
            (order[index].kind == PlannedEventKind::LocalCombine ||
             order[index].kind == PlannedEventKind::BufferRelease ||
             order[index].kind == PlannedEventKind::ObservableWrite) &&
            reaches(obligation.completion, order[index], selectedEdges);
        if (terminalObserver ||
            hasExactResourceConflict(input, obligation.issue, order[index],
                                     resourceBindings))
          break;
        boundary = order[index];
      }
    }
    placements.push_back({obligation.issue, obligation.completion,
                          EventBoundaryId{boundary}, obligation.protocol,
                          participants});
  }
  llvm::sort(placements);
  return placements;
}

std::map<EventId, std::optional<uint32_t>>
getEventStages(const ExecutionStructurePlan &structure) {
  std::map<EventId, std::optional<uint32_t>> stages;
  for (const ExecutionStructureChoice &choice : structure.scopes) {
    std::map<EventId, uint32_t> selected;
    if (const auto *pipeline =
            std::get_if<PipelinedExecutionStructure>(&choice))
      for (const EventStageAssignment &assignment : pipeline->eventStages)
        selected.emplace(assignment.event, assignment.stage.getValue());
    for (const EventId &event : getPipelineScope(choice).events) {
      auto stage = selected.find(event);
      stages.emplace(event, stage == selected.end()
                                ? std::optional<uint32_t>{}
                                : std::optional<uint32_t>{stage->second});
    }
  }
  return stages;
}

std::optional<std::vector<EventResourceBinding>>
deriveReceiverFSMBindings(const ScheduleDomainInput &input,
                          llvm::ArrayRef<ControlOrder> controls) {
  std::map<EventId, std::pair<const ControlOrder *, size_t>> positions;
  for (const ControlOrder &control : controls)
    for (auto [index, event] : llvm::enumerate(control.events))
      positions.emplace(event, std::make_pair(&control, index));
  struct Interval {
    EventId issue;
    EventId completion;
    DTEReceiverFSMResource resource;
    size_t begin = 0;
    size_t end = 0;
  };
  std::map<DTEReceiverFSMResource, std::vector<Interval>> byReceiver;
  for (const PlannedResourceUse &use : input.resourceUses) {
    const auto *resource = std::get_if<DTEReceiverFSMResource>(&use.resource);
    if (!resource)
      continue;
    auto issue = positions.find(use.event);
    auto completion = use.until ? positions.find(*use.until) : positions.end();
    if (use.knowledge != ResourceKnowledge::Exact || !use.until ||
        issue == positions.end() || completion == positions.end() ||
        issue->second.first != completion->second.first ||
        issue->second.second >= completion->second.second)
      return std::nullopt;
    byReceiver[*resource].push_back({use.event, *use.until, *resource,
                                     issue->second.second,
                                     completion->second.second});
  }

  std::vector<EventResourceBinding> bindings;
  for (auto &[resource, intervals] : byReceiver) {
    llvm::sort(intervals, [](const Interval &lhs, const Interval &rhs) {
      return lhs.issue < rhs.issue;
    });
    std::vector<uint32_t> colors(intervals.size(), 0);
    std::function<bool(size_t)> color = [&](size_t index) {
      if (index == intervals.size())
        return true;
      for (uint32_t lane = 0;
           lane < TargetDirectDTEResourceLimits::receiverFSMsPerTile; ++lane) {
        bool available = true;
        for (size_t previous = 0; previous < index; ++previous) {
          const bool overlap =
              intervals[previous].begin < intervals[index].end &&
              intervals[index].begin < intervals[previous].end;
          if (overlap && colors[previous] == lane) {
            available = false;
            break;
          }
        }
        if (!available)
          continue;
        colors[index] = lane;
        if (color(index + 1))
          return true;
      }
      return false;
    };
    if (!color(0))
      return std::nullopt;
    for (auto [interval, lane] : llvm::zip_equal(intervals, colors))
      bindings.push_back({interval.issue, ResourceInstanceId{resource, lane}});
  }
  llvm::sort(bindings);
  return bindings;
}

bool mandatoryReceiverFSMConflictsAreColorable(const ScheduleDomainInput &input,
                                               const std::set<Edge> &hard) {
  struct Interval {
    EventId issue;
    EventId completion;
  };
  std::map<DTEReceiverFSMResource, std::vector<Interval>> byReceiver;
  for (const PlannedResourceUse &use : input.resourceUses)
    if (const auto *resource =
            std::get_if<DTEReceiverFSMResource>(&use.resource)) {
      if (use.knowledge != ResourceKnowledge::Exact || !use.until)
        return false;
      byReceiver[*resource].push_back({use.event, *use.until});
    }
  for (auto &[resource, intervals] : byReceiver) {
    (void)resource;
    llvm::sort(intervals, [](const Interval &lhs, const Interval &rhs) {
      return lhs.issue < rhs.issue;
    });
    std::vector<std::vector<bool>> conflicts(
        intervals.size(), std::vector<bool>(intervals.size(), false));
    for (size_t left = 0; left < intervals.size(); ++left)
      for (size_t right = left + 1; right < intervals.size(); ++right) {
        const bool cannotPlaceLeftFirst =
            reaches(intervals[right].issue, intervals[left].completion, hard);
        const bool cannotPlaceRightFirst =
            reaches(intervals[left].issue, intervals[right].completion, hard);
        conflicts[left][right] = conflicts[right][left] =
            cannotPlaceLeftFirst && cannotPlaceRightFirst;
      }
    std::vector<uint32_t> colors(intervals.size(), 0);
    std::function<bool(size_t)> color = [&](size_t index) {
      if (index == intervals.size())
        return true;
      for (uint32_t lane = 0;
           lane < TargetDirectDTEResourceLimits::receiverFSMsPerTile; ++lane) {
        bool available = true;
        for (size_t previous = 0; previous < index; ++previous)
          if (conflicts[index][previous] && colors[previous] == lane) {
            available = false;
            break;
          }
        if (!available)
          continue;
        colors[index] = lane;
        if (color(index + 1))
          return true;
      }
      return false;
    };
    if (!color(0))
      return false;
  }
  return true;
}

} // namespace

bool ScheduleDomain::addSelectedResourceEdges(const ScheduleCursor &cursor,
                                              std::set<Edge> &edges) const {
  if (cursor.resourceOrders.size() != resources.size())
    return false;
  for (auto [domain, order] : llvm::zip_equal(resources, cursor.resourceOrders))
    for (size_t index = 1; index < order.size(); ++index) {
      std::optional<EventId> release =
          getResourceReleaseEvent(input, domain.resource, order[index - 1]);
      if (!release)
        return false;
      edges.insert({*release, order[index]});
    }
  return true;
}

std::optional<ScheduleCursor> ScheduleDomain::getInitialCursor() const {
  ScheduleCursor cursor;
  cursor.workerIndices.assign(workers.size(), 0);
  const std::set<Edge> hard = getHardEdges(input.hardDependencies);
  for (const ResourceDomain &resource : resources) {
    auto order = getFirstTopologicalOrder(resource.events, hard);
    if (!order)
      return std::nullopt;
    cursor.resourceOrders.push_back(std::move(*order));
  }
  std::set<Edge> controlEdges = hard;
  if (!addSelectedResourceEdges(cursor, controlEdges))
    return std::nullopt;
  for (const ControlDomain &control : controls) {
    auto order = getFirstTopologicalOrder(control.events, controlEdges);
    if (!order)
      return std::nullopt;
    cursor.controlOrders.push_back(std::move(*order));
  }
  return cursor;
}

ClosedSchedulePlan
ScheduleDomain::buildPlan(const ScheduleCursor &cursor) const {
  ClosedSchedulePlan plan;
  plan.structure = input.structure;
  plan.buffers = input.buffers;
  for (auto [domain, selected] : llvm::zip_equal(workers, cursor.workerIndices))
    if (selected < domain.workers.size())
      plan.workerBindings.push_back({domain.event, domain.workers[selected]});
  plan.resourceBindings = fixedResourceBindings;
  for (const EventWorkerBinding &binding : plan.workerBindings) {
    auto event =
        llvm::find_if(input.events, [&](const PlannedEvent &candidate) {
          return candidate.id == binding.event;
        });
    if (event != input.events.end() && event->tile)
      plan.resourceBindings.push_back(
          {binding.event,
           ResourceInstanceId{NCCWorkerResource{*event->tile, binding.worker},
                              0}});
  }
  llvm::sort(plan.workerBindings);
  llvm::sort(plan.resourceBindings);
  for (auto [domain, order] : llvm::zip_equal(resources, cursor.resourceOrders))
    plan.resourceSequences.push_back(
        {ResourceInstanceId{domain.resource, 0}, order});
  for (auto [domain, order] : llvm::zip_equal(controls, cursor.controlOrders))
    plan.controlOrders.push_back({domain.scope, order});
  if (auto receiverFSMs = deriveReceiverFSMBindings(input, plan.controlOrders))
    plan.resourceBindings.insert(plan.resourceBindings.end(),
                                 receiverFSMs->begin(), receiverFSMs->end());
  llvm::sort(plan.resourceBindings);
  std::map<EventId, NCCWorker> selectedWorkers;
  for (const EventWorkerBinding &binding : plan.workerBindings)
    selectedWorkers.emplace(binding.event, binding.worker);
  std::set<Edge> completionEdges = getHardEdges(input.hardDependencies);
  if (!addSelectedResourceEdges(cursor, completionEdges))
    return plan;
  plan.completionPlacements = deriveCompletionPlacements(
      input, selectedWorkers, plan.controlOrders, plan.resourceBindings,
      completionEdges, getEventStages(input.structure));
  llvm::sort(plan.resourceSequences);
  llvm::sort(plan.controlOrders);
  llvm::sort(plan.completionPlacements);
  return plan;
}

bool ScheduleDomain::contains(const ClosedSchedulePlan &plan) const {
  if (!isForGeneration(plan.structure, plan.buffers) ||
      plan.workerBindings.size() != workers.size() ||
      plan.resourceSequences.size() != resources.size() ||
      plan.controlOrders.size() != controls.size() ||
      plan.completionPlacements.size() != input.completionObligations.size())
    return false;
  std::map<EventId, NCCWorker> selectedWorkers;
  for (auto [domain, binding] : llvm::zip_equal(workers, plan.workerBindings)) {
    if (!(binding.event == domain.event) ||
        !llvm::is_contained(domain.workers, binding.worker) ||
        !selectedWorkers.try_emplace(binding.event, binding.worker).second)
      return false;
  }
  std::set<Edge> edges = getHardEdges(input.hardDependencies);
  for (auto [domain, sequence] :
       llvm::zip_equal(resources, plan.resourceSequences)) {
    if (!(sequence.instance == ResourceInstanceId{domain.resource, 0}) ||
        !isPermutation(domain.events, sequence.events) ||
        !respects(sequence.events, edges))
      return false;
    for (size_t index = 1; index < sequence.events.size(); ++index) {
      std::optional<EventId> release = getResourceReleaseEvent(
          input, domain.resource, sequence.events[index - 1]);
      if (!release)
        return false;
      edges.insert({*release, sequence.events[index]});
    }
  }
  const std::set<Edge> completionEdges = edges;
  for (auto [domain, control] : llvm::zip_equal(controls, plan.controlOrders)) {
    if (!(control.scope == domain.scope) ||
        !isPermutation(domain.events, control.events) ||
        !respects(control.events, edges))
      return false;
    for (size_t index = 1; index < control.events.size(); ++index)
      edges.insert({control.events[index - 1], control.events[index]});
  }
  if (!isAcyclic(input.events, edges))
    return false;

  std::vector<EventResourceBinding> expectedResources = fixedResourceBindings;
  for (const EventWorkerBinding &binding : plan.workerBindings) {
    auto event =
        llvm::find_if(input.events, [&](const PlannedEvent &candidate) {
          return candidate.id == binding.event;
        });
    if (event == input.events.end() || !event->tile)
      return false;
    expectedResources.push_back(
        {binding.event,
         ResourceInstanceId{NCCWorkerResource{*event->tile, binding.worker},
                            0}});
  }
  auto receiverFSMs = deriveReceiverFSMBindings(input, plan.controlOrders);
  if (!receiverFSMs)
    return false;
  expectedResources.insert(expectedResources.end(), receiverFSMs->begin(),
                           receiverFSMs->end());
  llvm::sort(expectedResources);
  if (plan.resourceBindings != expectedResources)
    return false;

  std::vector<CompletionPlacement> expectedCompletions =
      deriveCompletionPlacements(input, selectedWorkers, plan.controlOrders,
                                 plan.resourceBindings, completionEdges,
                                 getEventStages(input.structure));
  return plan.completionPlacements == expectedCompletions;
}

std::optional<ScheduleCursor>
ScheduleDomain::getCursor(const ClosedSchedulePlan &plan) const {
  if (!contains(plan))
    return std::nullopt;
  ScheduleCursor cursor;
  for (auto [domain, binding] : llvm::zip_equal(workers, plan.workerBindings)) {
    auto worker = llvm::find(domain.workers, binding.worker);
    cursor.workerIndices.push_back(
        static_cast<uint32_t>(std::distance(domain.workers.begin(), worker)));
  }
  for (const ResourceSequence &sequence : plan.resourceSequences)
    cursor.resourceOrders.push_back(sequence.events);
  for (const ControlOrder &control : plan.controlOrders)
    cursor.controlOrders.push_back(control.events);
  return cursor;
}

ScheduleDomain::AdvanceResult
ScheduleDomain::advance(ScheduleCursor &cursor) const {
  for (size_t reverse = 0; reverse < workers.size(); ++reverse) {
    const size_t index = workers.size() - reverse - 1;
    if (++cursor.workerIndices[index] < workers[index].workers.size()) {
      for (size_t reset = index + 1; reset < workers.size(); ++reset)
        cursor.workerIndices[reset] = 0;
      return {AdvanceKind::Advanced, {}};
    }
    cursor.workerIndices[index] = 0;
  }

  std::set<Edge> hard = getHardEdges(input.hardDependencies);
  if (!addSelectedResourceEdges(cursor, hard))
    return {AdvanceKind::End, {}};
  for (size_t reverse = 0; reverse < controls.size(); ++reverse) {
    const size_t index = controls.size() - reverse - 1;
    NextOrder next = getNextTopologicalOrder(controls[index].events,
                                             cursor.controlOrders[index], hard,
                                             limits.maxSuccessorSteps);
    if (next.kind == NextOrderKind::Indeterminate)
      return {AdvanceKind::Indeterminate,
              "control-order successor exceeded its work limit"};
    if (next.kind == NextOrderKind::Order) {
      cursor.controlOrders[index] = std::move(next.order);
      for (size_t reset = index + 1; reset < controls.size(); ++reset) {
        auto first = getFirstTopologicalOrder(controls[reset].events, hard);
        if (!first)
          return {AdvanceKind::End, {}};
        cursor.controlOrders[reset] = std::move(*first);
      }
      return {AdvanceKind::Advanced, {}};
    }
    auto first = getFirstTopologicalOrder(controls[index].events, hard);
    if (!first)
      return {AdvanceKind::End, {}};
    cursor.controlOrders[index] = std::move(*first);
  }

  const std::set<Edge> hardOnly = getHardEdges(input.hardDependencies);
  for (size_t reverse = 0; reverse < resources.size(); ++reverse) {
    const size_t index = resources.size() - reverse - 1;
    NextOrder next = getNextTopologicalOrder(
        resources[index].events, cursor.resourceOrders[index], hardOnly,
        limits.maxSuccessorSteps);
    if (next.kind == NextOrderKind::Indeterminate)
      return {AdvanceKind::Indeterminate,
              "resource-order successor exceeded its work limit"};
    if (next.kind == NextOrderKind::Order) {
      cursor.resourceOrders[index] = std::move(next.order);
      for (size_t reset = index + 1; reset < resources.size(); ++reset) {
        auto first =
            getFirstTopologicalOrder(resources[reset].events, hardOnly);
        if (!first)
          return {AdvanceKind::End, {}};
        cursor.resourceOrders[reset] = std::move(*first);
      }
      std::set<Edge> controlEdges = hardOnly;
      if (!addSelectedResourceEdges(cursor, controlEdges))
        return {AdvanceKind::End, {}};
      for (size_t control = 0; control < controls.size(); ++control) {
        auto first =
            getFirstTopologicalOrder(controls[control].events, controlEdges);
        if (!first)
          return {AdvanceKind::End, {}};
        cursor.controlOrders[control] = std::move(*first);
      }
      return {AdvanceKind::Advanced, {}};
    }
    auto first = getFirstTopologicalOrder(resources[index].events, hardOnly);
    if (!first)
      return {AdvanceKind::End, {}};
    cursor.resourceOrders[index] = std::move(*first);
  }
  return {AdvanceKind::End, {}};
}

ScheduleSuccessor ScheduleDomain::getFirstPlan() const {
  std::optional<ScheduleCursor> initial = getInitialCursor();
  if (!initial)
    return {ScheduleSuccessorKind::End};
  ScheduleCursor cursor = std::move(*initial);
  uint64_t steps = 0;
  while (true) {
    ClosedSchedulePlan plan = buildPlan(cursor);
    if (contains(plan))
      return {ScheduleSuccessorKind::Plan, std::move(plan), std::move(cursor)};
    if (++steps > limits.maxSuccessorSteps)
      return {ScheduleSuccessorKind::Indeterminate,
              {},
              {},
              "schedule first-leaf search exceeded its work limit"};
    AdvanceResult advanced = advance(cursor);
    if (advanced.kind == AdvanceKind::End)
      return {ScheduleSuccessorKind::End};
    if (advanced.kind == AdvanceKind::Indeterminate)
      return {ScheduleSuccessorKind::Indeterminate,
              {},
              {},
              std::move(advanced.detail)};
  }
}

ScheduleSuccessor
ScheduleDomain::getNextPlan(const ScheduleCursor &cursor) const {
  if (!getCursor(buildPlan(cursor)))
    return {ScheduleSuccessorKind::CompilerBug,
            {},
            {},
            "schedule cursor is outside the current domain"};
  ScheduleCursor next = cursor;
  uint64_t steps = 0;
  while (true) {
    AdvanceResult advanced = advance(next);
    if (advanced.kind == AdvanceKind::End)
      return {ScheduleSuccessorKind::End};
    if (advanced.kind == AdvanceKind::Indeterminate)
      return {ScheduleSuccessorKind::Indeterminate,
              {},
              {},
              std::move(advanced.detail)};
    ClosedSchedulePlan plan = buildPlan(next);
    if (contains(plan))
      return {ScheduleSuccessorKind::Plan, std::move(plan), std::move(next)};
    if (++steps > limits.maxSuccessorSteps)
      return {ScheduleSuccessorKind::Indeterminate,
              {},
              {},
              "schedule successor search exceeded its work limit"};
  }
}

ScheduleDomainResult buildScheduleDomain(ScheduleDomainInput input,
                                         const ScheduleDomainLimits &limits) {
  if (input.structure.scopes.empty() || input.buffers.storageObjects.empty() ||
      input.events.empty() || limits.maxSuccessorSteps == 0)
    return failed(ScheduleDomainFailureKind::BrokenContract,
                  "schedule domain requires fixed K/I events and a work limit");
  llvm::sort(input.events,
             [](const PlannedEvent &lhs, const PlannedEvent &rhs) {
               return lhs.id < rhs.id;
             });
  for (size_t index = 1; index < input.events.size(); ++index)
    if (input.events[index - 1].id == input.events[index].id)
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule domain has duplicate events");
  std::set<EventId> eventIds;
  for (const PlannedEvent &event : input.events)
    eventIds.insert(event.id);
  std::set<EventId> structureEvents;
  std::map<EventId, PipelineScopeId> pipelineScopes;
  for (const ExecutionStructureChoice &choice : input.structure.scopes) {
    const PipelineScopeId &pipeline = getPipelineScope(choice);
    for (const EventId &event : pipeline.events)
      if (!eventIds.count(event) || !structureEvents.insert(event).second ||
          !pipelineScopes.try_emplace(event, pipeline).second)
        return failed(ScheduleDomainFailureKind::BrokenContract,
                      "schedule structure has stale or overlapping events");
  }
  if (structureEvents != eventIds)
    return failed(ScheduleDomainFailureKind::BrokenContract,
                  "schedule structure does not cover every event");
  std::set<StorageObjectId> storageObjects;
  for (const StorageObjectPlan &object : input.buffers.storageObjects)
    if (!storageObjects.insert(object.id).second)
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule buffers have duplicate storage objects");
  llvm::sort(input.hardDependencies);
  llvm::sort(input.orderChoices, [](const auto &lhs, const auto &rhs) {
    return lhs.resource < rhs.resource;
  });
  llvm::sort(input.completionObligations);
  llvm::sort(input.resourceUses);
  llvm::sort(input.slotLifetimes);
  if (std::adjacent_find(input.completionObligations.begin(),
                         input.completionObligations.end()) !=
          input.completionObligations.end() ||
      std::adjacent_find(input.slotLifetimes.begin(),
                         input.slotLifetimes.end()) !=
          input.slotLifetimes.end())
    return failed(ScheduleDomainFailureKind::BrokenContract,
                  "schedule completion or slot lifetime facts are duplicated");
  std::set<EventId> completionIssues;
  for (const CompletionObligation &obligation : input.completionObligations)
    if (obligation.protocol == CompletionProtocol::Unknown ||
        !eventIds.count(obligation.issue) ||
        !eventIds.count(obligation.completion) ||
        !completionIssues.insert(obligation.issue).second ||
        (obligation.participantMask & ~kAllNCCWorkersMask) != 0 ||
        ((obligation.protocol == CompletionProtocol::DirectDTE ||
          obligation.protocol ==
              CompletionProtocol::NoAsynchronousCompletion) &&
         obligation.participantMask != 0))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule completion protocol is unknown or unbound");
  for (const EventDependency &dependency : input.hardDependencies)
    if (!eventIds.count(dependency.predecessor) ||
        !eventIds.count(dependency.successor) ||
        dependency.predecessor == dependency.successor)
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule hard dependency references an invalid event");
  for (const PlannedResourceUse &use : input.resourceUses)
    if (!eventIds.count(use.event) ||
        (use.until && !eventIds.count(*use.until)) ||
        (use.knowledge == ResourceKnowledge::Exact &&
         use.interval != ResourceIntervalKind::Instantaneous && !use.until) ||
        (use.interval == ResourceIntervalKind::Instantaneous && use.until))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule resource use has an invalid exact interval");
  const std::set<Edge> hard = getHardEdges(input.hardDependencies);
  if (!isAcyclic(input.events, hard))
    return failed(ScheduleDomainFailureKind::ExactRejection,
                  "schedule hard dependency graph is cyclic");
  if (!mandatoryReceiverFSMConflictsAreColorable(input, hard))
    return failed(ScheduleDomainFailureKind::ExactRejection,
                  "mandatory receiver FSM live ranges exceed the target "
                  "per-Tile limit");

  std::vector<ScheduleDomain::WorkerDomain> workers;
  for (PlannedEvent &event : input.events) {
    llvm::sort(event.workerDomain);
    event.workerDomain.erase(
        std::unique(event.workerDomain.begin(), event.workerDomain.end()),
        event.workerDomain.end());
    if (llvm::any_of(event.workerDomain, [](NCCWorker worker) {
          return static_cast<uint32_t>(worker) >= kNCCWorkerCount;
        }))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule event has an invalid worker domain");
    if (!event.workerDomain.empty())
      workers.push_back({event.id, event.workerDomain});
  }
  std::vector<ScheduleDomain::ResourceDomain> resources;
  std::set<ResourceKey> resourceKeys;
  for (DisjunctiveResourceOrder &choice : input.orderChoices) {
    llvm::sort(choice.events);
    if (choice.events.size() < 2 ||
        std::adjacent_find(choice.events.begin(), choice.events.end()) !=
            choice.events.end() ||
        !resourceKeys.insert(choice.resource).second ||
        std::holds_alternative<DTEReceiverFSMResource>(choice.resource))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule resource order choice is malformed");
    for (const EventId &event : choice.events) {
      if (!eventIds.count(event))
        return failed(ScheduleDomainFailureKind::BrokenContract,
                      "schedule resource choice references an unknown event");
      if (!getResourceReleaseEvent(input, choice.resource, event))
        return failed(
            ScheduleDomainFailureKind::BrokenContract,
            "schedule resource choice has no exact matching live interval");
    }
    if (!getFirstTopologicalOrder(choice.events, hard))
      return failed(ScheduleDomainFailureKind::ExactRejection,
                    "schedule resource domain is cyclic");
    resources.push_back({choice.resource, choice.events});
  }
  llvm::sort(resources, [](const auto &lhs, const auto &rhs) {
    return lhs.resource < rhs.resource;
  });

  std::map<ControlScopeId, std::vector<EventId>> byControl;
  for (const PlannedEvent &event : input.events) {
    auto pipeline = pipelineScopes.find(event.id);
    if (pipeline == pipelineScopes.end())
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule event has no execution-structure scope");
    ControlScopeId scope =
        event.tile
            ? ControlScopeId{TileControlScope{*event.tile, pipeline->second}}
            : ControlScopeId{CardControlScope{event.card, pipeline->second}};
    byControl[scope].push_back(event.id);
  }
  std::vector<ScheduleDomain::ControlDomain> controls;
  for (auto &[scope, events] : byControl) {
    llvm::sort(events);
    controls.push_back({scope, std::move(events)});
  }

  if (input.components.empty())
    return failed(ScheduleDomainFailureKind::BrokenContract,
                  "schedule input has no event components");
  std::set<EventId> componentEvents;
  for (EventComponent &component : input.components) {
    llvm::sort(component.events);
    if (component.events.empty() ||
        std::adjacent_find(component.events.begin(), component.events.end()) !=
            component.events.end())
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule input has an empty or duplicate component");
    for (const EventId &event : component.events)
      if (!eventIds.count(event) || !componentEvents.insert(event).second)
        return failed(ScheduleDomainFailureKind::BrokenContract,
                      "schedule components overlap or reference stale events");
  }
  if (componentEvents != eventIds)
    return failed(ScheduleDomainFailureKind::BrokenContract,
                  "schedule components do not cover every event");

  std::vector<EventResourceBinding> fixedBindings;
  for (const PlannedResourceUse &use : input.resourceUses) {
    if (!eventIds.count(use.event))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule resource use references an unknown event");
    if (use.knowledge == ResourceKnowledge::Exact &&
        !std::holds_alternative<DTEReceiverFSMResource>(use.resource))
      fixedBindings.push_back({use.event, ResourceInstanceId{use.resource, 0}});
  }
  llvm::sort(fixedBindings);
  fixedBindings.erase(std::unique(fixedBindings.begin(), fixedBindings.end()),
                      fixedBindings.end());

  for (const CompletionObligation &obligation : input.completionObligations)
    if (!eventIds.count(obligation.issue) ||
        !eventIds.count(obligation.completion) ||
        !reaches(obligation.issue, obligation.completion, hard))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule completion obligation is unmatched");
  for (const SlotLifetimeRequirement &lifetime : input.slotLifetimes) {
    auto structure = llvm::find_if(
        input.structure.scopes, [&](const ExecutionStructureChoice &choice) {
          const auto *pipeline =
              std::get_if<PipelinedExecutionStructure>(&choice);
          return pipeline && pipeline->recurrence == lifetime.occurrence &&
                 pipeline->iteration == lifetime.iteration;
        });
    auto family = llvm::find_if(
        input.buffers.slotFamilies, [&](const SlotFamilyPlan &candidate) {
          return candidate.id == lifetime.family &&
                 candidate.occurrence == lifetime.occurrence;
        });
    if (structure == input.structure.scopes.end() ||
        family == input.buffers.slotFamilies.end() ||
        family->multiplicity < lifetime.minimumMultiplicity ||
        family->multiplicity > lifetime.maximumMultiplicity ||
        (family->multiplicity > 1 &&
         family->rotationIterators !=
             llvm::SmallVector<uint32_t, 4>{lifetime.iteration.recurrenceAxis}))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule input has a stale slot lifetime generation");
    for (const EventId &event : lifetime.readyEvents)
      if (!eventIds.count(event))
        return failed(ScheduleDomainFailureKind::BrokenContract,
                      "slot lifetime has an unknown ready event");
    for (const EventId &event : lifetime.releaseEvents)
      if (!eventIds.count(event))
        return failed(ScheduleDomainFailureKind::BrokenContract,
                      "slot lifetime has an unknown release event");
  }

  ScheduleDomain domain(std::move(input), std::move(workers),
                        std::move(resources), std::move(controls),
                        std::move(fixedBindings), limits);
  ScheduleSuccessor first = domain.getFirstPlan();
  if (first.getKind() == ScheduleSuccessorKind::End)
    return failed(ScheduleDomainFailureKind::ExactRejection,
                  "fixed K/I schedule domain has no feasible leaf");
  if (first.getKind() == ScheduleSuccessorKind::Indeterminate)
    return failed(ScheduleDomainFailureKind::Indeterminate, first.getDetail());
  if (first.getKind() != ScheduleSuccessorKind::Plan)
    return failed(ScheduleDomainFailureKind::BrokenContract,
                  "schedule domain failed to construct its first leaf");
  return {std::move(domain), {}};
}

} // namespace wafer::compiler::detail
