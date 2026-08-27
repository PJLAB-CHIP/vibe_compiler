//===- ScheduleDomain.cpp - Fixed-generation event schedule -----------===//

#include "Wafer/Planning/PhysicalDataflow/ScheduleDomain.h"

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/Core/DirectDTE.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
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
  std::vector<const EventId *> expectedOrder;
  std::vector<const EventId *> actualOrder;
  expectedOrder.reserve(expected.size());
  actualOrder.reserve(actual.size());
  for (const EventId &event : expected)
    expectedOrder.push_back(&event);
  for (const EventId &event : actual)
    actualOrder.push_back(&event);
  auto less = [](const EventId *lhs, const EventId *rhs) {
    return *lhs < *rhs;
  };
  llvm::sort(expectedOrder, less);
  llvm::sort(actualOrder, less);
  for (auto [left, right] : llvm::zip_equal(expectedOrder, actualOrder))
    if (!(*left == *right))
      return false;
  return std::adjacent_find(expectedOrder.begin(), expectedOrder.end(),
                            [](const EventId *lhs, const EventId *rhs) {
                              return *lhs == *rhs;
                            }) == expectedOrder.end();
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
  std::vector<EventId> members(events.begin(), events.end());
  llvm::sort(members);
  if (std::adjacent_find(members.begin(), members.end()) != members.end())
    return std::nullopt;
  auto indexOf = [&](const EventId &event) -> std::optional<size_t> {
    auto found = llvm::lower_bound(members, event);
    return found == members.end() || !(*found == event)
               ? std::nullopt
               : std::optional<size_t>(std::distance(members.begin(), found));
  };
  std::vector<size_t> indegree(members.size(), 0);
  std::vector<std::vector<size_t>> successors(members.size());
  for (const auto &[predecessor, successor] : edges) {
    std::optional<size_t> before = indexOf(predecessor);
    std::optional<size_t> after = indexOf(successor);
    if (!before || !after)
      continue;
    successors[*before].push_back(*after);
    ++indegree[*after];
  }
  std::set<size_t> ready;
  for (auto [event, degree] : llvm::enumerate(indegree))
    if (degree == 0)
      ready.insert(event);
  std::vector<EventId> order;
  while (!ready.empty()) {
    size_t event = *ready.begin();
    ready.erase(ready.begin());
    order.push_back(members[event]);
    for (size_t successor : successors[event])
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

struct ResourceUseIndex {
  ResourceUseIndex(const ScheduleDomainInput &input,
                   const BufferPlan &buffers) {
    for (const PlannedResourceUse &use : input.resourceUses) {
      byEvent[use.event].push_back(&use);
      if (use.knowledge != ResourceKnowledge::Exact)
        continue;
      if (const auto *range = std::get_if<SPMRangeResource>(&use.resource)) {
        spmUses[range->object].push_back(&use);
        if (const auto *movement =
                std::get_if<MovementEventAction>(&use.event.action))
          movementRanges[movement->action].push_back(&use);
      } else {
        exactResourceUses[use.resource].push_back(&use);
      }
    }
    for (const PhysicalVersionStorageBinding &binding :
         buffers.versionBindings)
      if (binding.kind == StorageBindingKind::Reuse)
        selectedReuseObjects.insert(binding.object);
    for (const SlotFamilyPlan &family : buffers.slotFamilies)
      selectedReuseObjects.insert(family.id.objects.begin(),
                                  family.id.objects.end());
    for (const BufferOrderRequirement &requirement :
         buffers.orderRequirements)
      for (const PhysicalVersionStorageBinding &binding :
           buffers.versionBindings)
        if (binding.version == requirement.earlier)
          selectedReuseObjects.insert(binding.object);
  }

  std::map<EventId, std::vector<const PlannedResourceUse *>> byEvent;
  std::map<StorageObjectId, std::vector<const PlannedResourceUse *>> spmUses;
  std::map<ResourceKey, std::vector<const PlannedResourceUse *>>
      exactResourceUses;
  std::map<MovementActionId, std::vector<const PlannedResourceUse *>>
      movementRanges;
  std::set<StorageObjectId> selectedReuseObjects;
};

std::vector<CompletionPlacement> deriveCompletionPlacements(
    const ScheduleDomainInput &input, const BufferPlan &buffers,
    const std::map<EventId, NCCWorker> &selectedWorkers,
    llvm::ArrayRef<ControlOrder> controlOrders,
    llvm::ArrayRef<EventResourceBinding> resourceBindings,
    const std::set<Edge> &selectedEdges,
    const std::map<EventId, std::optional<uint32_t>> &eventStages) {
  std::optional<wafer::support::ScopedCompileTimingSpan> phase;
  phase.emplace("planning-algorithm", "schedule-domain",
                "index-completion-events");
  struct EventPosition {
    const EventOrder *order = nullptr;
    size_t index = 0;
    uint32_t ordinal = 0;
    uint32_t control = 0;
  };
  std::map<EventId, EventPosition> positions;
  uint32_t nextOrdinal = 0;
  for (auto [controlOrdinal, control] : llvm::enumerate(controlOrders))
    for (auto [index, event] : llvm::enumerate(control.events))
      positions.emplace(
          event, EventPosition{&control.events, index, nextOrdinal++,
                               static_cast<uint32_t>(controlOrdinal)});
  if (positions.size() != input.events.size())
    return {};
  ResourceUseIndex resourceUses(input, buffers);
  std::map<const PlannedResourceUse *, std::pair<uint32_t, size_t>,
           std::less<const PlannedResourceUse *>>
      usePositions;
  for (const PlannedResourceUse &use : input.resourceUses) {
    auto position = positions.find(use.event);
    if (position == positions.end())
      return {};
    usePositions.emplace(
        &use, std::make_pair(position->second.control, position->second.index));
  }
  auto sortUseBucket = [&](auto &bucket) {
    llvm::sort(bucket, [&](const PlannedResourceUse *lhs,
                           const PlannedResourceUse *rhs) {
      const auto &left = usePositions.at(lhs);
      const auto &right = usePositions.at(rhs);
      return left != right ? left < right : *lhs < *rhs;
    });
  };
  for (auto &[object, uses] : resourceUses.spmUses) {
    (void)object;
    sortUseBucket(uses);
  }
  for (auto &[resource, uses] : resourceUses.exactResourceUses) {
    (void)resource;
    sortUseBucket(uses);
  }
  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "index-completion-dag");
  std::vector<std::vector<uint32_t>> selectedSuccessors(positions.size());
  std::vector<std::vector<uint32_t>> selectedPredecessors(positions.size());
  std::vector<uint32_t> indegree(positions.size(), 0);
  for (const auto &[predecessor, successor] : selectedEdges) {
    auto source = positions.find(predecessor);
    auto destination = positions.find(successor);
    if (source != positions.end() && destination != positions.end()) {
      selectedSuccessors[source->second.ordinal].push_back(
          destination->second.ordinal);
      selectedPredecessors[destination->second.ordinal].push_back(
          source->second.ordinal);
      ++indegree[destination->second.ordinal];
    }
  }

  std::set<uint32_t> ready;
  for (auto [ordinal, degree] : llvm::enumerate(indegree))
    if (degree == 0)
      ready.insert(static_cast<uint32_t>(ordinal));
  std::vector<uint32_t> topological;
  topological.reserve(positions.size());
  while (!ready.empty()) {
    const uint32_t current = *ready.begin();
    ready.erase(ready.begin());
    topological.push_back(current);
    for (uint32_t successor : selectedSuccessors[current])
      if (--indegree[successor] == 0)
        ready.insert(successor);
  }
  if (topological.size() != positions.size())
    return {};
  std::vector<uint32_t> topologicalRank(positions.size());
  for (auto [rank, ordinal] : llvm::enumerate(topological))
    topologicalRank[ordinal] = static_cast<uint32_t>(rank);

  const uint32_t noComponent = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> componentOf(positions.size(), noComponent);
  std::vector<std::vector<uint32_t>> componentNodes(input.components.size());
  for (auto [componentOrdinal, component] :
       llvm::enumerate(input.components))
    for (const EventId &event : component.events) {
      auto position = positions.find(event);
      if (position == positions.end() ||
          componentOf[position->second.ordinal] != noComponent)
        return {};
      componentOf[position->second.ordinal] =
          static_cast<uint32_t>(componentOrdinal);
      componentNodes[componentOrdinal].push_back(position->second.ordinal);
    }
  if (llvm::any_of(componentOf,
                   [&](uint32_t component) { return component == noComponent; }))
    return {};
  std::vector<std::vector<uint32_t>> componentTopological(
      input.components.size());
  for (uint32_t ordinal : topological)
    componentTopological[componentOf[ordinal]].push_back(ordinal);
  std::vector<std::vector<uint32_t>> componentControls(
      input.components.size());
  for (auto [controlOrdinal, control] : llvm::enumerate(controlOrders)) {
    if (control.events.empty())
      return {};
    auto first = positions.find(control.events.front());
    if (first == positions.end())
      return {};
    const uint32_t component = componentOf[first->second.ordinal];
    if (llvm::any_of(control.events, [&](const EventId &event) {
          auto position = positions.find(event);
          return position == positions.end() ||
                 componentOf[position->second.ordinal] != component;
        }))
      return {};
    componentControls[component].push_back(
        static_cast<uint32_t>(controlOrdinal));
  }
  for (auto [source, successors] : llvm::enumerate(selectedSuccessors))
    if (llvm::any_of(successors, [&](uint32_t destination) {
          return componentOf[source] != componentOf[destination];
        }))
      return {};

  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "index-reachable-completion-boundaries");
  const size_t noPosition = std::numeric_limits<size_t>::max();
  std::vector<size_t> earliestReachableIssue(positions.size(), noPosition);
  std::vector<size_t> propagatedPosition(positions.size(), noPosition);
  for (auto [component, controls] : llvm::enumerate(componentControls))
    for (uint32_t controlOrdinal : controls) {
      for (uint32_t ordinal : componentNodes[component])
        propagatedPosition[ordinal] = noPosition;
      const EventOrder &order = controlOrders[controlOrdinal].events;
      for (auto [index, event] : llvm::enumerate(order))
        if (event.kind == PlannedEventKind::ComputeIssue ||
            event.kind == PlannedEventKind::MovementIssue ||
            event.kind == PlannedEventKind::LocalCombine)
          propagatedPosition[positions.find(event)->second.ordinal] = index;
      for (uint32_t ordinal :
           llvm::reverse(componentTopological[component]))
        for (uint32_t successor : selectedSuccessors[ordinal])
          propagatedPosition[ordinal] =
              std::min(propagatedPosition[ordinal],
                       propagatedPosition[successor]);
      for (const EventId &event : order) {
        const uint32_t ordinal = positions.find(event)->second.ordinal;
        earliestReachableIssue[ordinal] = propagatedPosition[ordinal];
      }
    }

  struct IndexedTerminalObserver {
    uint32_t control = 0;
    size_t position = 0;
    uint32_t ordinal = 0;
    const PlannedResourceUse *rangeUse = nullptr;
  };
  std::map<StorageObjectId, std::vector<IndexedTerminalObserver>>
      releasesByObject;
  std::map<StorageObjectId, std::vector<IndexedTerminalObserver>>
      observablesByObject;
  std::vector<std::vector<size_t>> stageRunEnd(controlOrders.size());
  for (auto [controlOrdinal, control] : llvm::enumerate(controlOrders)) {
    for (auto [index, event] : llvm::enumerate(control.events)) {
      const uint32_t ordinal = positions.find(event)->second.ordinal;
      if (event.kind == PlannedEventKind::BufferRelease)
        if (const auto *buffer =
                std::get_if<BufferEventAction>(&event.action))
          if (resourceUses.selectedReuseObjects.count(buffer->storageObject))
            releasesByObject[buffer->storageObject].push_back(
                {static_cast<uint32_t>(controlOrdinal), index, ordinal,
                 nullptr});
      if (event.kind == PlannedEventKind::ObservableWrite)
        if (const auto *movement =
                std::get_if<MovementEventAction>(&event.action)) {
          auto ranges = resourceUses.movementRanges.find(movement->action);
          if (ranges != resourceUses.movementRanges.end())
            for (const PlannedResourceUse *use : ranges->second)
              if (const auto *range =
                      std::get_if<SPMRangeResource>(&use->resource))
                observablesByObject[range->object].push_back(
                    {static_cast<uint32_t>(controlOrdinal), index, ordinal,
                     use});
        }
    }
    stageRunEnd[controlOrdinal].resize(control.events.size());
    if (control.events.empty())
      continue;
    stageRunEnd[controlOrdinal].back() = control.events.size();
    for (size_t reverse = 1; reverse < control.events.size(); ++reverse) {
      const size_t index = control.events.size() - reverse - 1;
      stageRunEnd[controlOrdinal][index] =
          eventStages.at(control.events[index]) ==
                  eventStages.at(control.events[index + 1])
              ? stageRunEnd[controlOrdinal][index + 1]
              : index + 1;
    }
  }
  auto sortObservers = [](auto &observers) {
    llvm::sort(observers, [](const IndexedTerminalObserver &lhs,
                             const IndexedTerminalObserver &rhs) {
      if (std::tie(lhs.control, lhs.position, lhs.ordinal) !=
          std::tie(rhs.control, rhs.position, rhs.ordinal))
        return std::tie(lhs.control, lhs.position, lhs.ordinal) <
               std::tie(rhs.control, rhs.position, rhs.ordinal);
      if (!lhs.rangeUse || !rhs.rangeUse)
        return static_cast<bool>(rhs.rangeUse);
      return *lhs.rangeUse < *rhs.rangeUse;
    });
  };
  for (auto &[object, observers] : releasesByObject) {
    (void)object;
    sortObservers(observers);
  }
  for (auto &[object, observers] : observablesByObject) {
    (void)object;
    sortObservers(observers);
  }

  auto selectedReceiverLane = [&](const EventId &event,
                                  const ResourceKey &resource)
      -> std::optional<uint32_t> {
    auto binding = llvm::lower_bound(
        resourceBindings, event,
        [](const EventResourceBinding &candidate, const EventId &event) {
          return candidate.event < event;
        });
    for (; binding != resourceBindings.end() && binding->event == event;
         ++binding)
      if (binding->instance.resource == resource)
        return binding->instance.lane;
    return std::nullopt;
  };
  auto usesConflict = [&](const PlannedResourceUse &left,
                          const PlannedResourceUse &right) {
    if (left.knowledge != ResourceKnowledge::Exact ||
        right.knowledge != ResourceKnowledge::Exact)
      return false;
    const auto *leftRange = std::get_if<SPMRangeResource>(&left.resource);
    const auto *rightRange = std::get_if<SPMRangeResource>(&right.resource);
    if (leftRange || rightRange)
      return leftRange && rightRange && rangesOverlap(*leftRange, *rightRange) &&
             modesConflict(left.mode, right.mode);
    if (!(left.resource == right.resource))
      return false;
    if (std::holds_alternative<DTEReceiverFSMResource>(left.resource)) {
      std::optional<uint32_t> leftLane =
          selectedReceiverLane(left.event, left.resource);
      std::optional<uint32_t> rightLane =
          selectedReceiverLane(right.event, right.resource);
      if (leftLane && rightLane)
        return *leftLane == *rightLane;
    }
    return modesConflict(left.mode, right.mode);
  };
  struct ConflictPositions {
    std::optional<size_t> spm;
    std::optional<size_t> any;
  };
  auto firstConflictPositions = [&](const EventId &issue,
                                    const EventPosition &completion) {
    ConflictPositions result;
    auto issueUses = resourceUses.byEvent.find(issue);
    if (issueUses == resourceUses.byEvent.end())
      return result;
    size_t firstSPM = noPosition;
    size_t firstAny = noPosition;
    for (const PlannedResourceUse *left : issueUses->second) {
      if (left->knowledge != ResourceKnowledge::Exact)
        continue;
      llvm::ArrayRef<const PlannedResourceUse *> candidates;
      const auto *range = std::get_if<SPMRangeResource>(&left->resource);
      if (range) {
        auto bucket = resourceUses.spmUses.find(range->object);
        if (bucket == resourceUses.spmUses.end())
          continue;
        candidates = bucket->second;
      } else {
        auto bucket = resourceUses.exactResourceUses.find(left->resource);
        if (bucket == resourceUses.exactResourceUses.end())
          continue;
        candidates = bucket->second;
      }
      const auto start = std::make_pair(completion.control,
                                        completion.index + 1);
      auto candidate = llvm::lower_bound(
          candidates, start,
          [&](const PlannedResourceUse *use,
              const std::pair<uint32_t, size_t> &position) {
            return usePositions.at(use) < position;
          });
      for (; candidate != candidates.end(); ++candidate) {
        const auto &[control, index] = usePositions.at(*candidate);
        const size_t currentFirst = range ? firstSPM : firstAny;
        if (control != completion.control || index >= currentFirst)
          break;
        if (!usesConflict(*left, **candidate))
          continue;
        if (range)
          firstSPM = index;
        else
          firstAny = index;
        break;
      }
    }
    if (firstSPM != noPosition)
      result.spm = firstSPM;
    const size_t first = std::min(firstSPM, firstAny);
    if (first != noPosition)
      result.any = first;
    return result;
  };

  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "index-completion-conflicts");
  std::vector<ConflictPositions> conflictsByObligation(
      input.completionObligations.size());
  for (auto [obligationIndex, obligation] :
       llvm::enumerate(input.completionObligations)) {
    auto completion = positions.find(obligation.completion);
    if (completion == positions.end())
      return {};
    conflictsByObligation[obligationIndex] =
        firstConflictPositions(obligation.issue, completion->second);
  }

  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "index-completion-terminal-queries");
  struct TerminalQuery {
    uint32_t source = 0;
    uint32_t destination = 0;
    size_t obligation = 0;
    size_t position = 0;
  };
  std::vector<std::vector<TerminalQuery>> terminalQueries(
      input.components.size());
  for (auto [obligationIndex, obligation] :
       llvm::enumerate(input.completionObligations)) {
    auto completion = positions.find(obligation.completion);
    if (completion == positions.end())
      return {};
    const EventPosition &completionPosition = completion->second;
    const EventOrder &order = *completionPosition.order;
    const ConflictPositions &conflicts =
        conflictsByObligation[obligationIndex];
    std::optional<size_t> terminalLimit;
    if (obligation.protocol == CompletionProtocol::NCCParticipant &&
        obligation.participantMask == 0) {
      terminalLimit = conflicts.spm.value_or(order.size() - 1);
    } else if (obligation.protocol == CompletionProtocol::DirectDTE ||
               (obligation.protocol == CompletionProtocol::NCCParticipant &&
                obligation.participantMask != 0)) {
      size_t firstBarrier =
          stageRunEnd[completionPosition.control][completionPosition.index];
      const size_t actualIssue =
          earliestReachableIssue[completionPosition.ordinal];
      if (actualIssue > completionPosition.index)
        firstBarrier = std::min(firstBarrier, actualIssue);
      if (conflicts.any)
        firstBarrier = std::min(firstBarrier, *conflicts.any);
      terminalLimit = firstBarrier == order.size() ? order.size() - 1
                                                   : firstBarrier;
    }
    if (!terminalLimit || *terminalLimit <= completionPosition.index)
      continue;
    std::vector<std::pair<size_t, uint32_t>> relevantObservers;
    auto issueUses = resourceUses.byEvent.find(obligation.issue);
    if (issueUses == resourceUses.byEvent.end())
      continue;
    const auto firstPosition =
        std::make_pair(completionPosition.control,
                       completionPosition.index + 1);
    auto appendObservers = [&](const auto &observers, auto isRelevant) {
      auto observer = llvm::lower_bound(
          observers, firstPosition,
          [](const IndexedTerminalObserver &candidate,
             const std::pair<uint32_t, size_t> &position) {
            return std::make_pair(candidate.control, candidate.position) <
                   position;
          });
      for (; observer != observers.end() &&
             observer->control == completionPosition.control &&
             observer->position <= *terminalLimit;
           ++observer)
        if (isRelevant(*observer))
          relevantObservers.emplace_back(observer->position,
                                         observer->ordinal);
    };
    for (const PlannedResourceUse *issueUse : issueUses->second) {
      if (issueUse->knowledge != ResourceKnowledge::Exact)
        continue;
      const auto *issueRange =
          std::get_if<SPMRangeResource>(&issueUse->resource);
      if (!issueRange)
        continue;
      auto releases = releasesByObject.find(issueRange->object);
      if (releases != releasesByObject.end())
        appendObservers(releases->second,
                        [](const IndexedTerminalObserver &) { return true; });
      auto observables = observablesByObject.find(issueRange->object);
      if (observables != observablesByObject.end())
        appendObservers(
            observables->second,
            [&](const IndexedTerminalObserver &observer) {
              const auto *observerRange =
                  std::get_if<SPMRangeResource>(&observer.rangeUse->resource);
              return observerRange &&
                     rangesOverlap(*issueRange, *observerRange) &&
                     modesConflict(ResourceUseMode::Write,
                                   observer.rangeUse->mode);
            });
    }
    llvm::sort(relevantObservers);
    relevantObservers.erase(
        std::unique(relevantObservers.begin(), relevantObservers.end()),
        relevantObservers.end());
    for (const auto &[position, destination] : relevantObservers) {
      if (topologicalRank[completionPosition.ordinal] >=
          topologicalRank[destination])
        continue;
      terminalQueries[componentOf[completionPosition.ordinal]].push_back(
          {completionPosition.ordinal, destination, obligationIndex,
           position});
    }
  }

  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "solve-completion-terminal-reachability");
  std::vector<std::optional<size_t>> terminalBarriers(
      input.completionObligations.size());
  std::vector<uint32_t> visitedGeneration(positions.size(), 0);
  uint32_t currentGeneration = 0;
  auto startTraversal = [&]() {
    if (++currentGeneration == 0) {
      std::fill(visitedGeneration.begin(), visitedGeneration.end(), 0);
      ++currentGeneration;
    }
  };
  auto recordReachable = [&](const TerminalQuery &query) {
    std::optional<size_t> &barrier = terminalBarriers[query.obligation];
    if (!barrier || query.position < *barrier)
      barrier = query.position;
  };
  for (std::vector<TerminalQuery> &queries : terminalQueries) {
    if (queries.empty())
      continue;
    std::map<uint32_t, std::vector<const TerminalQuery *>> bySource;
    std::map<uint32_t, std::vector<const TerminalQuery *>> byDestination;
    for (const TerminalQuery &query : queries) {
      bySource[query.source].push_back(&query);
      byDestination[query.destination].push_back(&query);
    }
    if (bySource.size() <= byDestination.size()) {
      for (const auto &[source, sourceQueries] : bySource) {
        startTraversal();
        uint32_t maximumRank = 0;
        for (const TerminalQuery *query : sourceQueries)
          maximumRank =
              std::max(maximumRank, topologicalRank[query->destination]);
        std::vector<uint32_t> stack{source};
        while (!stack.empty()) {
          const uint32_t current = stack.back();
          stack.pop_back();
          if (visitedGeneration[current] == currentGeneration)
            continue;
          visitedGeneration[current] = currentGeneration;
          for (uint32_t successor : selectedSuccessors[current])
            if (topologicalRank[successor] <= maximumRank)
              stack.push_back(successor);
        }
        for (const TerminalQuery *query : sourceQueries)
          if (visitedGeneration[query->destination] == currentGeneration)
            recordReachable(*query);
      }
    } else {
      for (const auto &[destination, destinationQueries] : byDestination) {
        startTraversal();
        uint32_t minimumRank = std::numeric_limits<uint32_t>::max();
        for (const TerminalQuery *query : destinationQueries)
          minimumRank = std::min(minimumRank,
                                 topologicalRank[query->source]);
        std::vector<uint32_t> stack{destination};
        while (!stack.empty()) {
          const uint32_t current = stack.back();
          stack.pop_back();
          if (visitedGeneration[current] == currentGeneration)
            continue;
          visitedGeneration[current] = currentGeneration;
          for (uint32_t predecessor : selectedPredecessors[current])
            if (topologicalRank[predecessor] >= minimumRank)
              stack.push_back(predecessor);
        }
        for (const TerminalQuery *query : destinationQueries)
          if (visitedGeneration[query->source] == currentGeneration)
            recordReachable(*query);
      }
    }
  }

  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "derive-completion-placements");
  std::vector<CompletionPlacement> placements;
  for (auto [obligationIndex, obligation] :
       llvm::enumerate(input.completionObligations)) {
    auto completion = positions.find(obligation.completion);
    if (completion == positions.end())
      return {};
    const EventPosition &completionPosition = completion->second;
    const EventOrder &order = *completionPosition.order;
    const ConflictPositions conflicts =
        conflictsByObligation[obligationIndex];
    const std::optional<size_t> terminalBarrier =
        terminalBarriers[obligationIndex];
    uint32_t participants = obligation.participantMask;
    if (obligation.protocol == CompletionProtocol::NCCParticipant &&
        participants == 0) {
      auto worker = selectedWorkers.find(obligation.issue);
      if (worker == selectedWorkers.end())
        return {};
      bool sameWorkerHandoff = false;
      if (conflicts.spm &&
          (!terminalBarrier || *conflicts.spm < *terminalBarrier)) {
        auto consumerWorker = selectedWorkers.find(order[*conflicts.spm]);
        sameWorkerHandoff = consumerWorker != selectedWorkers.end() &&
                            consumerWorker->second == worker->second;
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
    const bool emitsCompletion =
        (obligation.protocol == CompletionProtocol::NCCParticipant &&
         participants != 0) ||
        obligation.protocol == CompletionProtocol::DirectDTE;
    if (emitsCompletion) {
      size_t firstBarrier =
          stageRunEnd[completionPosition.control][completionPosition.index];
      const size_t actualIssue =
          earliestReachableIssue[completionPosition.ordinal];
      if (actualIssue > completionPosition.index)
        firstBarrier = std::min(firstBarrier, actualIssue);
      if (terminalBarrier)
        firstBarrier = std::min(firstBarrier, *terminalBarrier);
      if (conflicts.any)
        firstBarrier = std::min(firstBarrier, *conflicts.any);
      if (firstBarrier > completionPosition.index + 1)
        boundary = order[firstBarrier - 1];
    }
    placements.push_back({obligation.issue, obligation.completion,
                          EventBoundaryId{boundary}, obligation.protocol,
                          participants});
  }
  llvm::sort(placements);
  phase.reset();
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
      return std::tie(lhs.begin, lhs.end, lhs.issue) <
             std::tie(rhs.begin, rhs.end, rhs.issue);
    });
    std::vector<uint32_t> colors;
    std::array<size_t, TargetDirectDTEResourceLimits::receiverFSMsPerTile>
        laneEnds{};
    colors.reserve(intervals.size());
    for (const Interval &interval : intervals) {
      std::optional<uint32_t> selected;
      for (uint32_t lane = 0;
           lane < TargetDirectDTEResourceLimits::receiverFSMsPerTile; ++lane)
        if (laneEnds[lane] <= interval.begin) {
          selected = lane;
          break;
        }
      if (!selected)
        return std::nullopt;
      colors.push_back(*selected);
      laneEnds[*selected] = interval.end;
    }
    for (auto [interval, lane] : llvm::zip_equal(intervals, colors))
      bindings.push_back({interval.issue, ResourceInstanceId{resource, lane}});
  }
  llvm::sort(bindings);
  return bindings;
}

std::optional<std::vector<EventResourceBinding>> deriveResourceBindings(
    const ScheduleDomainInput &input,
    llvm::ArrayRef<EventResourceBinding> fixedBindings,
    llvm::ArrayRef<EventWorkerBinding> workerBindings,
    llvm::ArrayRef<ControlOrder> controls) {
  std::vector<EventResourceBinding> bindings(fixedBindings.begin(),
                                             fixedBindings.end());
  for (const EventWorkerBinding &binding : workerBindings) {
    auto event = llvm::lower_bound(
        input.events, binding.event,
        [](const PlannedEvent &candidate, const EventId &event) {
          return candidate.id < event;
        });
    if (event == input.events.end() || !(event->id == binding.event) ||
        !event->tile)
      return std::nullopt;
    bindings.push_back(
        {binding.event,
         ResourceInstanceId{NCCWorkerResource{*event->tile, binding.worker},
                            0}});
  }
  auto receiverBindings = deriveReceiverFSMBindings(input, controls);
  if (!receiverBindings)
    return std::nullopt;
  bindings.insert(bindings.end(), receiverBindings->begin(),
                  receiverBindings->end());
  llvm::sort(bindings);
  return bindings;
}

bool mandatoryReceiverFSMConflictsAreColorable(const ScheduleDomainInput &input,
                                               const std::set<Edge> &hard) {
  struct Interval {
    EventId issue;
    EventId completion;
  };
  std::map<Edge, bool> reachability;
  auto reachesHard = [&](const EventId &source, const EventId &destination) {
    auto [entry, inserted] =
        reachability.try_emplace({source, destination}, false);
    if (inserted)
      entry->second = reaches(source, destination, hard);
    return entry->second;
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
    if (intervals.size() <= TargetDirectDTEResourceLimits::receiverFSMsPerTile)
      continue;
    llvm::sort(intervals, [](const Interval &lhs, const Interval &rhs) {
      return lhs.issue < rhs.issue;
    });
    std::vector<std::vector<bool>> conflicts(
        intervals.size(), std::vector<bool>(intervals.size(), false));
    for (size_t left = 0; left < intervals.size(); ++left)
      for (size_t right = left + 1; right < intervals.size(); ++right) {
        const bool cannotPlaceLeftFirst =
            reachesHard(intervals[right].issue, intervals[left].completion);
        const bool cannotPlaceRightFirst =
            reachesHard(intervals[left].issue, intervals[right].completion);
        conflicts[left][right] = conflicts[right][left] =
            cannotPlaceLeftFirst && cannotPlaceRightFirst;
      }
    constexpr size_t forbiddenCliqueSize =
        TargetDirectDTEResourceLimits::receiverFSMsPerTile + 1;
    std::function<bool(std::vector<size_t>, size_t)> hasForbiddenClique =
        [&](std::vector<size_t> candidates, size_t depth) {
          if (depth == forbiddenCliqueSize)
            return true;
          if (depth + candidates.size() < forbiddenCliqueSize)
            return false;
          while (!candidates.empty()) {
            const size_t vertex = candidates.front();
            candidates.erase(candidates.begin());
            std::vector<size_t> next;
            for (size_t candidate : candidates)
              if (conflicts[vertex][candidate])
                next.push_back(candidate);
            if (hasForbiddenClique(std::move(next), depth + 1))
              return true;
          }
          return false;
        };
    std::vector<size_t> candidates(intervals.size());
    std::iota(candidates.begin(), candidates.end(), size_t{0});
    if (hasForbiddenClique(std::move(candidates), 0))
      return false;
  }
  return true;
}

std::optional<std::vector<EventId>>
getFirstReceiverFeasibleControlOrder(const ScheduleDomainInput &input,
                                     llvm::ArrayRef<EventId> events,
                                     const std::set<Edge> &edges) {
  std::vector<EventId> members(events.begin(), events.end());
  llvm::sort(members);
  if (std::adjacent_find(members.begin(), members.end()) != members.end())
    return std::nullopt;
  auto indexOf = [&](const EventId &event) -> std::optional<size_t> {
    auto found = llvm::lower_bound(members, event);
    return found == members.end() || !(*found == event)
               ? std::nullopt
               : std::optional<size_t>(std::distance(members.begin(), found));
  };
  std::vector<std::vector<DTEReceiverFSMResource>> starts(members.size());
  std::vector<std::vector<DTEReceiverFSMResource>> ends(members.size());
  bool hasReceiverIntervals = false;
  for (const PlannedResourceUse &use : input.resourceUses) {
    const auto *resource = std::get_if<DTEReceiverFSMResource>(&use.resource);
    if (!resource || use.knowledge != ResourceKnowledge::Exact || !use.until)
      continue;
    std::optional<size_t> start = indexOf(use.event);
    std::optional<size_t> end = indexOf(*use.until);
    if (start.has_value() != end.has_value())
      return std::nullopt;
    if (!start)
      continue;
    starts[*start].push_back(*resource);
    ends[*end].push_back(*resource);
    hasReceiverIntervals = true;
  }
  if (!hasReceiverIntervals)
    return std::nullopt;

  std::vector<size_t> indegree(members.size(), 0);
  std::vector<std::vector<size_t>> successors(members.size());
  for (const Edge &edge : edges) {
    std::optional<size_t> source = indexOf(edge.first);
    std::optional<size_t> destination = indexOf(edge.second);
    if (!source || !destination)
      continue;
    successors[*source].push_back(*destination);
    ++indegree[*destination];
  }

  auto priority = [&](size_t event) {
    if (!ends[event].empty())
      return 0u;
    if (starts[event].empty())
      return 1u;
    return 2u;
  };
  std::array<std::set<size_t>, 3> ready;
  for (auto [event, degree] : llvm::enumerate(indegree))
    if (degree == 0)
      ready[priority(event)].insert(event);
  std::vector<EventId> order;
  std::map<DTEReceiverFSMResource, uint32_t> active;
  while (order.size() != members.size()) {
    std::optional<size_t> selected;
    std::map<DTEReceiverFSMResource, uint32_t> selectedActive;
    for (unsigned bucket = 0; bucket < ready.size() && !selected; ++bucket) {
      for (size_t event : ready[bucket]) {
        std::map<DTEReceiverFSMResource, uint32_t> nextActive = active;
        bool valid = true;
        for (const DTEReceiverFSMResource &resource : ends[event]) {
          auto current = nextActive.find(resource);
          if (current == nextActive.end() || current->second == 0) {
            valid = false;
            break;
          }
          --current->second;
        }
        if (!valid)
          continue;
        for (const DTEReceiverFSMResource &resource : starts[event])
          if (++nextActive[resource] >
              TargetDirectDTEResourceLimits::receiverFSMsPerTile) {
            valid = false;
            break;
          }
        if (!valid)
          continue;
        selected = event;
        selectedActive = std::move(nextActive);
        break;
      }
    }
    if (!selected)
      return std::nullopt;
    ready[priority(*selected)].erase(*selected);
    active = std::move(selectedActive);
    order.push_back(members[*selected]);
    for (size_t successor : successors[*selected])
      if (--indegree[successor] == 0)
        ready[priority(successor)].insert(successor);
  }
  return llvm::all_of(active,
                      [](const auto &entry) { return entry.second == 0; })
             ? std::optional<std::vector<EventId>>(std::move(order))
             : std::nullopt;
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

std::optional<ScheduleCursor>
ScheduleDomain::getReceiverFeasibleProposalCursor() const {
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
  std::vector<EventId> allEvents;
  for (const PlannedEvent &event : input.events)
    allEvents.push_back(event.id);
  auto global =
      getFirstReceiverFeasibleControlOrder(input, allEvents, controlEdges);
  if (!global)
    global = getFirstTopologicalOrder(allEvents, controlEdges);
  if (!global)
    return std::nullopt;
  std::vector<size_t> globalPositions(input.events.size(), 0);
  for (auto [position, event] : llvm::enumerate(*global)) {
    auto source = llvm::lower_bound(
        input.events, event,
        [](const PlannedEvent &candidate, const EventId &event) {
          return candidate.id < event;
        });
    if (source == input.events.end() || !(source->id == event))
      return std::nullopt;
    globalPositions[std::distance(input.events.begin(), source)] = position;
  }
  for (const ControlDomain &control : controls) {
    std::vector<std::pair<size_t, EventId>> positioned;
    positioned.reserve(control.events.size());
    for (const EventId &event : control.events) {
      auto found = llvm::lower_bound(
          input.events, event,
          [](const PlannedEvent &candidate, const EventId &event) {
            return candidate.id < event;
          });
      if (found == input.events.end() || !(found->id == event))
        return std::nullopt;
      positioned.emplace_back(
          globalPositions[std::distance(input.events.begin(), found)], event);
    }
    llvm::sort(positioned, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    std::vector<EventId> projected;
    projected.reserve(positioned.size());
    for (auto &entry : positioned)
      projected.push_back(std::move(entry.second));
    cursor.controlOrders.push_back(std::move(projected));
  }
  cursor.proposal = true;
  return cursor;
}

ClosedSchedulePlan
ScheduleDomain::buildPlan(const ScheduleCursor &cursor) const {
  std::optional<wafer::support::ScopedCompileTimingSpan> phase;
  phase.emplace("planning-algorithm", "schedule-domain",
                "copy-schedule-generation");
  ClosedSchedulePlan plan;
  plan.setGeneration(generationStructure, generationBuffers);
  for (auto [domain, selected] : llvm::zip_equal(workers, cursor.workerIndices))
    if (selected < domain.workers.size())
      plan.workerBindings.push_back({domain.event, domain.workers[selected]});
  llvm::sort(plan.workerBindings);
  for (auto [domain, order] : llvm::zip_equal(resources, cursor.resourceOrders))
    plan.resourceSequences.push_back(
        {ResourceInstanceId{domain.resource, 0}, order});
  for (auto [domain, order] : llvm::zip_equal(controls, cursor.controlOrders))
    plan.controlOrders.push_back({domain.scope, order});
  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain", "bind-receiver-fsms");
  auto resourceBindings =
      deriveResourceBindings(input, fixedResourceBindings, plan.workerBindings,
                             plan.controlOrders);
  if (!resourceBindings)
    return plan;
  plan.resourceBindings = std::move(*resourceBindings);
  phase.reset();
  std::map<EventId, NCCWorker> selectedWorkers;
  for (const EventWorkerBinding &binding : plan.workerBindings)
    selectedWorkers.emplace(binding.event, binding.worker);
  std::set<Edge> completionEdges = getHardEdges(input.hardDependencies);
  if (!addSelectedResourceEdges(cursor, completionEdges))
    return plan;
  phase.emplace("planning-algorithm", "schedule-domain", "place-completions");
  plan.completionPlacements = deriveCompletionPlacements(
      input, *generationBuffers, selectedWorkers, plan.controlOrders,
      plan.resourceBindings, completionEdges,
      getEventStages(*generationStructure));
  phase.reset();
  return plan;
}

bool ScheduleDomain::contains(const ClosedSchedulePlan &plan) const {
  if (firstPlan && plan == firstPlan->first)
    return true;
  std::optional<wafer::support::ScopedCompileTimingSpan> phase;
  phase.emplace("planning-algorithm", "schedule-domain",
                "validate-schedule-generation");
  if (((plan.structure != generationStructure ||
        plan.buffers != generationBuffers) &&
       !isForGeneration(plan.getStructure(), plan.getBuffers())) ||
      plan.workerBindings.size() != workers.size() ||
      plan.resourceSequences.size() != resources.size() ||
      plan.controlOrders.size() != controls.size() ||
      plan.completionPlacements.size() != input.completionObligations.size())
    return false;
  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "validate-schedule-workers");
  std::map<EventId, NCCWorker> selectedWorkers;
  for (auto [domain, binding] : llvm::zip_equal(workers, plan.workerBindings)) {
    if (!(binding.event == domain.event) ||
        !llvm::is_contained(domain.workers, binding.worker) ||
        !selectedWorkers.try_emplace(binding.event, binding.worker).second)
      return false;
  }
  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "validate-schedule-resource-orders");
  std::set<Edge> edges = getHardEdges(input.hardDependencies);
  for (auto [domain, sequence] :
       llvm::zip_equal(resources, plan.resourceSequences)) {
    if (!(sequence.instance == ResourceInstanceId{domain.resource, 0}) ||
        !isPermutation(domain.events, sequence.events.get()) ||
        !respects(sequence.events.get(), edges))
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
  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "validate-schedule-control-orders");
  auto eventIndex = [&](const EventId &event) -> std::optional<size_t> {
    auto found = llvm::lower_bound(
        input.events, event,
        [](const PlannedEvent &candidate, const EventId &event) {
          return candidate.id < event;
        });
    return found == input.events.end() || !(found->id == event)
               ? std::nullopt
               : std::optional<size_t>(
                     std::distance(input.events.begin(), found));
  };
  const uint32_t noControl = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> expectedControl(input.events.size(), noControl);
  for (auto [controlOrdinal, domain] : llvm::enumerate(controls))
    for (const EventId &event : domain.events) {
      std::optional<size_t> index = eventIndex(event);
      if (!index || expectedControl[*index] != noControl)
        return false;
      expectedControl[*index] = static_cast<uint32_t>(controlOrdinal);
    }
  std::vector<uint32_t> actualControl(input.events.size(), noControl);
  std::vector<size_t> actualPosition(input.events.size(), 0);
  for (auto [controlOrdinal, pair] :
       llvm::enumerate(llvm::zip_equal(controls, plan.controlOrders))) {
    const auto &[domain, control] = pair;
    if (!(control.scope == domain.scope) ||
        control.events.size() != domain.events.size())
      return false;
    for (auto [position, event] : llvm::enumerate(control.events)) {
      std::optional<size_t> index = eventIndex(event);
      if (!index || expectedControl[*index] != controlOrdinal ||
          actualControl[*index] != noControl)
        return false;
      actualControl[*index] = static_cast<uint32_t>(controlOrdinal);
      actualPosition[*index] = position;
    }
    for (size_t index = 1; index < control.events.size(); ++index)
      edges.insert({control.events[index - 1], control.events[index]});
  }
  for (const auto &[predecessor, successor] : completionEdges) {
    std::optional<size_t> before = eventIndex(predecessor);
    std::optional<size_t> after = eventIndex(successor);
    if (!before || !after)
      return false;
    if (actualControl[*before] == actualControl[*after] &&
        actualPosition[*before] >= actualPosition[*after])
      return false;
  }
  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "validate-schedule-dag");
  if (!isAcyclic(input.events, edges))
    return false;

  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "validate-schedule-resources");
  auto expectedResources =
      deriveResourceBindings(input, fixedResourceBindings, plan.workerBindings,
                             plan.controlOrders);
  if (!expectedResources)
    return false;
  if (plan.resourceBindings != *expectedResources)
    return false;

  phase.reset();
  phase.emplace("planning-algorithm", "schedule-domain",
                "validate-schedule-completions");
  std::vector<CompletionPlacement> expectedCompletions =
      deriveCompletionPlacements(input, *generationBuffers, selectedWorkers,
                                 plan.controlOrders, plan.resourceBindings,
                                 completionEdges,
                                 getEventStages(*generationStructure));
  const bool valid = plan.completionPlacements == expectedCompletions;
  phase.reset();
  return valid;
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
  std::set<Edge> hard = getHardEdges(input.hardDependencies);
  if (!addSelectedResourceEdges(cursor, hard))
    return {AdvanceKind::End, {}};
  for (size_t reverse = 0; reverse < controls.size(); ++reverse) {
    const size_t index = controls.size() - reverse - 1;
    NextOrder next = getNextTopologicalOrder(controls[index].events,
                                             cursor.controlOrders[index].get(),
                                             hard, limits.maxSuccessorSteps);
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
        resources[index].events, cursor.resourceOrders[index].get(), hardOnly,
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

  for (size_t reverse = 0; reverse < workers.size(); ++reverse) {
    const size_t index = workers.size() - reverse - 1;
    if (++cursor.workerIndices[index] < workers[index].workers.size()) {
      for (size_t reset = index + 1; reset < workers.size(); ++reset)
        cursor.workerIndices[reset] = 0;
      return {AdvanceKind::Advanced, {}};
    }
    cursor.workerIndices[index] = 0;
  }
  return {AdvanceKind::End, {}};
}

ScheduleSuccessor ScheduleDomain::getFirstPlan() const {
  if (firstPlan)
    return {ScheduleSuccessorKind::Plan, firstPlan->first, firstPlan->second};
  std::optional<ScheduleCursor> receiverProposal;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "planning-algorithm", "schedule-domain", "receiver-feasible-proposal");
    receiverProposal = getReceiverFeasibleProposalCursor();
  }
  if (receiverProposal) {
    ScheduleCursor &proposal = *receiverProposal;
    ClosedSchedulePlan plan = buildPlan(proposal);
    bool accepted = false;
    {
      wafer::support::ScopedCompileTimingSpan timing(
          "planning-algorithm", "schedule-domain", "validate-proposal");
      accepted = contains(plan);
    }
    if (accepted)
      return {ScheduleSuccessorKind::Plan, std::move(plan),
              std::move(proposal)};
  }
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
  ClosedSchedulePlan currentPlan = buildPlan(cursor);
  if (!contains(currentPlan))
    return {ScheduleSuccessorKind::CompilerBug,
            {},
            {},
            "schedule cursor is outside the current domain"};
  ScheduleCursor next;
  bool advanceBeforeCheck = true;
  if (cursor.proposal) {
    std::optional<ScheduleCursor> canonical = getInitialCursor();
    if (!canonical)
      return {ScheduleSuccessorKind::End};
    next = std::move(*canonical);
    advanceBeforeCheck = false;
  } else {
    next = cursor;
  }
  uint64_t steps = 0;
  while (true) {
    if (advanceBeforeCheck) {
      AdvanceResult advanced = advance(next);
      if (advanced.kind == AdvanceKind::End)
        return {ScheduleSuccessorKind::End};
      if (advanced.kind == AdvanceKind::Indeterminate)
        return {ScheduleSuccessorKind::Indeterminate,
                {},
                {},
                std::move(advanced.detail)};
    }
    advanceBeforeCheck = true;
    ClosedSchedulePlan plan = buildPlan(next);
    if (contains(plan) && !(plan == currentPlan))
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
  std::optional<wafer::support::ScopedCompileTimingSpan> phase;
  phase.emplace("query", "physical-search", "normalize-schedule-input");
  llvm::sort(input.events,
             [](const PlannedEvent &lhs, const PlannedEvent &rhs) {
               return lhs.id < rhs.id;
             });
  for (size_t index = 1; index < input.events.size(); ++index)
    if (input.events[index - 1].id == input.events[index].id)
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule domain has duplicate events");
  auto eventIndex = [&](const EventId &event) -> std::optional<size_t> {
    auto found = llvm::lower_bound(
        input.events, event,
        [](const PlannedEvent &candidate, const EventId &event) {
          return candidate.id < event;
        });
    return found == input.events.end() || !(found->id == event)
               ? std::nullopt
               : std::optional<size_t>(
                     std::distance(input.events.begin(), found));
  };
  auto hasEvent = [&](const EventId &event) {
    return eventIndex(event).has_value();
  };
  std::vector<const PipelineScopeId *> pipelineScopes(input.events.size(),
                                                      nullptr);
  for (const ExecutionStructureChoice &choice : input.structure.scopes) {
    const PipelineScopeId &pipeline = getPipelineScope(choice);
    for (const EventId &event : pipeline.events) {
      std::optional<size_t> index = eventIndex(event);
      if (!index || pipelineScopes[*index])
        return failed(ScheduleDomainFailureKind::BrokenContract,
                      "schedule structure has stale or overlapping events");
      pipelineScopes[*index] = &pipeline;
    }
  }
  if (llvm::any_of(pipelineScopes,
                   [](const PipelineScopeId *scope) { return !scope; }))
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
        !hasEvent(obligation.issue) || !hasEvent(obligation.completion) ||
        !completionIssues.insert(obligation.issue).second ||
        (obligation.participantMask & ~kAllNCCWorkersMask) != 0 ||
        ((obligation.protocol == CompletionProtocol::DirectDTE ||
          obligation.protocol ==
              CompletionProtocol::NoAsynchronousCompletion) &&
         obligation.participantMask != 0))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule completion protocol is unknown or unbound");
  for (const EventDependency &dependency : input.hardDependencies)
    if (!hasEvent(dependency.predecessor) || !hasEvent(dependency.successor) ||
        dependency.predecessor == dependency.successor)
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule hard dependency references an invalid event");
  for (const PlannedResourceUse &use : input.resourceUses)
    if (!hasEvent(use.event) || (use.until && !hasEvent(*use.until)) ||
        (use.knowledge == ResourceKnowledge::Exact &&
         use.interval != ResourceIntervalKind::Instantaneous && !use.until) ||
        (use.interval == ResourceIntervalKind::Instantaneous && use.until))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule resource use has an invalid exact interval");
  const std::set<Edge> hard = getHardEdges(input.hardDependencies);
  phase.reset();
  const bool acyclic = [&]() {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "physical-search", "validate-schedule-hard-dag");
    return isAcyclic(input.events, hard);
  }();
  if (!acyclic)
    return failed(ScheduleDomainFailureKind::ExactRejection,
                  "schedule hard dependency graph is cyclic");
  const bool receiversColorable = [&]() {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "physical-search", "validate-schedule-receivers");
    return mandatoryReceiverFSMConflictsAreColorable(input, hard);
  }();
  if (!receiversColorable)
    return failed(ScheduleDomainFailureKind::ExactRejection,
                  "mandatory receiver FSM live ranges exceed the target "
                  "per-Tile limit");
  phase.emplace("query", "physical-search", "build-schedule-choice-domains");
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
      if (!hasEvent(event))
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

  struct ControlGroupKey {
    CardId card{0};
    std::optional<TileId> tile;
    const PipelineScopeId *pipeline = nullptr;
  };
  struct ControlGroupKeyLess {
    bool operator()(const ControlGroupKey &lhs,
                    const ControlGroupKey &rhs) const {
      if (lhs.card != rhs.card)
        return lhs.card.getValue() < rhs.card.getValue();
      if (lhs.tile.has_value() != rhs.tile.has_value())
        return lhs.tile.has_value() < rhs.tile.has_value();
      if (lhs.tile && lhs.tile != rhs.tile)
        return lhs.tile->getValue() < rhs.tile->getValue();
      return std::less<const PipelineScopeId *>{}(lhs.pipeline, rhs.pipeline);
    }
  };
  std::map<ControlGroupKey, std::vector<EventId>, ControlGroupKeyLess>
      byControl;
  for (auto [index, event] : llvm::enumerate(input.events))
    byControl[{event.card, event.tile, pipelineScopes[index]}].push_back(
        event.id);
  std::vector<ScheduleDomain::ControlDomain> controls;
  std::map<const PipelineScopeId *, std::shared_ptr<const PipelineScopeId>,
           std::less<const PipelineScopeId *>>
      sharedPipelines;
  for (auto &[key, events] : byControl) {
    llvm::sort(events);
    auto [shared, inserted] = sharedPipelines.try_emplace(key.pipeline);
    if (inserted)
      shared->second = std::make_shared<const PipelineScopeId>(*key.pipeline);
    ControlScopeId scope =
        key.tile ? ControlScopeId{TileControlScope{*key.tile, shared->second}}
                 : ControlScopeId{CardControlScope{key.card, shared->second}};
    controls.push_back({std::move(scope), std::move(events)});
  }
  llvm::sort(controls, [](const auto &lhs, const auto &rhs) {
    return lhs.scope < rhs.scope;
  });

  if (input.components.empty())
    return failed(ScheduleDomainFailureKind::BrokenContract,
                  "schedule input has no event components");
  std::vector<uint8_t> componentCoverage(input.events.size(), 0);
  for (EventComponent &component : input.components) {
    llvm::sort(component.events);
    if (component.events.empty() ||
        std::adjacent_find(component.events.begin(), component.events.end()) !=
            component.events.end())
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule input has an empty or duplicate component");
    for (const EventId &event : component.events) {
      std::optional<size_t> index = eventIndex(event);
      if (!index || componentCoverage[*index]++)
        return failed(ScheduleDomainFailureKind::BrokenContract,
                      "schedule components overlap or reference stale events");
    }
  }
  if (llvm::any_of(componentCoverage, [](uint8_t count) { return count != 1; }))
    return failed(ScheduleDomainFailureKind::BrokenContract,
                  "schedule components do not cover every event");

  std::vector<EventResourceBinding> fixedBindings;
  for (const PlannedResourceUse &use : input.resourceUses) {
    if (!hasEvent(use.event))
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
    if (!hasEvent(obligation.issue) || !hasEvent(obligation.completion) ||
        !hard.count(Edge{obligation.issue, obligation.completion}))
      return failed(ScheduleDomainFailureKind::BrokenContract,
                    "schedule completion obligation has no direct hard edge");
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
      if (!hasEvent(event))
        return failed(ScheduleDomainFailureKind::BrokenContract,
                      "slot lifetime has an unknown ready event");
    for (const EventId &event : lifetime.releaseEvents)
      if (!hasEvent(event))
        return failed(ScheduleDomainFailureKind::BrokenContract,
                      "slot lifetime has an unknown release event");
  }

  phase.reset();
  auto generationStructure = std::make_shared<const ExecutionStructurePlan>(
      std::move(input.structure));
  auto generationBuffers =
      std::make_shared<const BufferPlan>(std::move(input.buffers));
  ScheduleDomain domain(
      std::move(input), std::move(workers), std::move(resources),
      std::move(controls), std::move(fixedBindings),
      std::move(generationStructure), std::move(generationBuffers), limits);
  ScheduleSuccessor first = [&]() {
    wafer::support::ScopedCompileTimingSpan timing("query", "physical-search",
                                                   "build-first-schedule");
    return domain.getFirstPlan();
  }();
  if (first.getKind() == ScheduleSuccessorKind::End)
    return failed(ScheduleDomainFailureKind::ExactRejection,
                  "fixed K/I schedule domain has no feasible leaf");
  if (first.getKind() == ScheduleSuccessorKind::Indeterminate)
    return failed(ScheduleDomainFailureKind::Indeterminate, first.getDetail());
  if (first.getKind() != ScheduleSuccessorKind::Plan)
    return failed(ScheduleDomainFailureKind::BrokenContract,
                  "schedule domain failed to construct its first leaf");
  domain.firstPlan.emplace(*first.getPlan(), *first.getCursor());
  return {std::move(domain), {}};
}

} // namespace wafer::compiler::detail
