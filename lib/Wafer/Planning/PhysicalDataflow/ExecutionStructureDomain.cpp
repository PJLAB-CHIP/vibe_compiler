//===- ExecutionStructureDomain.cpp - Finite structure domain --------===//

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructureDomain.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

ExecutionStructureDomainResult failed(ExecutionStructureDomainFailureKind kind,
                                      llvm::StringRef detail) {
  return {{}, ExecutionStructureDomainFailure{kind, detail.str()}};
}

std::optional<uint64_t> getTripCount(const OccurrenceRelationId &recurrence) {
  uint64_t tripCount = 1;
  for (uint64_t count : recurrence.axisOccurrences)
    if (count == 0 || tripCount > std::numeric_limits<uint64_t>::max() / count)
      return std::nullopt;
    else
      tripCount *= count;
  return tripCount;
}

bool hasSteadyState(uint64_t tripCount, uint32_t stageCount,
                    uint64_t launchDistance) {
  return stageCount >= 2 && launchDistance > 0 && tripCount > 1 &&
         uint64_t(stageCount - 1) <= (tripCount - 1) / launchDistance;
}

bool incrementStages(std::vector<uint32_t> &stages, uint32_t stageCount) {
  for (size_t reverse = 0; reverse < stages.size(); ++reverse) {
    const size_t index = stages.size() - reverse - 1;
    if (++stages[index] < stageCount) {
      std::fill(stages.begin() + index + 1, stages.end(), 0);
      return true;
    }
    stages[index] = 0;
  }
  return false;
}

bool isStageAssignmentLegal(llvm::ArrayRef<EventId> events,
                            llvm::ArrayRef<EventDependency> dependencies,
                            llvm::ArrayRef<uint32_t> assignments,
                            uint32_t stageCount) {
  if (events.size() != assignments.size() || stageCount < 2)
    return false;
  std::vector<bool> used(stageCount, false);
  std::map<EventId, uint32_t> stages;
  for (auto [event, stage] : llvm::zip_equal(events, assignments)) {
    if (stage >= stageCount || !stages.try_emplace(event, stage).second)
      return false;
    used[stage] = true;
  }
  if (llvm::is_contained(used, false))
    return false;
  for (const EventDependency &dependency : dependencies) {
    auto predecessor = stages.find(dependency.predecessor);
    auto successor = stages.find(dependency.successor);
    if (predecessor == stages.end() || successor == stages.end() ||
        predecessor->second > successor->second)
      return false;
  }
  return true;
}

} // namespace

ExecutionStructurePlan ExecutionStructureDomain::buildPlan(
    const ExecutionStructureCursor &cursor) const {
  ExecutionStructurePlan plan;
  if (cursor.scopes.size() != scopes.size())
    return plan;
  for (auto [scope, choice] : llvm::zip_equal(scopes, cursor.scopes)) {
    if (choice.serialized) {
      plan.scopes.push_back(SerializedExecutionStructure{scope.id});
      continue;
    }
    PipelinedExecutionStructure pipelined;
    pipelined.scope = scope.id;
    pipelined.recurrence = scope.id.recurrences.front();
    pipelined.launchDistance = choice.launchDistance;
    for (auto [event, stage] :
         llvm::zip_equal(scope.stageableEvents, choice.eventStages))
      pipelined.eventStages.push_back({event, StageId(stage)});
    plan.scopes.push_back(std::move(pipelined));
  }
  return plan;
}

ExecutionStructureSuccessor ExecutionStructureDomain::getFirstPlan() const {
  if (scopes.empty())
    return {ExecutionStructureSuccessorKind::CompilerBug,
            {},
            {},
            "execution-structure domain has no scopes"};
  ExecutionStructureCursor cursor;
  cursor.scopes.resize(scopes.size());
  ExecutionStructurePlan plan = buildPlan(cursor);
  if (!contains(plan))
    return {ExecutionStructureSuccessorKind::CompilerBug,
            {},
            {},
            "execution-structure domain has no serialized identity"};
  return {ExecutionStructureSuccessorKind::Plan, std::move(plan),
          std::move(cursor)};
}

ExecutionStructureDomain::ScopeAdvance ExecutionStructureDomain::advanceScope(
    size_t index, ExecutionStructureCursor::ScopeCursor &cursor) const {
  const ScopeDomain &scope = scopes[index];
  if (!scope.pipelinedEligible)
    return {ScopeAdvanceKind::End, {}};
  const uint32_t maximumStages = static_cast<uint32_t>(
      std::min<uint64_t>(scope.stageableEvents.size(), scope.tripCount));
  const uint32_t boundedMaximum =
      limits.maxStages == 0 ? maximumStages
                            : std::min(maximumStages, limits.maxStages);
  uint64_t steps = 0;
  auto overLimit = [&]() { return ++steps > limits.maxSuccessorSteps; };
  auto searchAssignment = [&](uint32_t stageCount,
                              std::vector<uint32_t> &assignments,
                              bool includeCurrent) -> std::optional<bool> {
    bool current = includeCurrent;
    while (current || incrementStages(assignments, stageCount)) {
      current = false;
      if (overLimit())
        return std::nullopt;
      if (isStageAssignmentLegal(scope.stageableEvents, scope.dependencies,
                                 assignments, stageCount))
        return true;
    }
    return false;
  };

  if (!cursor.serialized && hasSteadyState(scope.tripCount, cursor.stageCount,
                                           cursor.launchDistance + 1)) {
    ++cursor.launchDistance;
    return {ScopeAdvanceKind::Choice, {}};
  }

  uint32_t stageCount = cursor.serialized ? 2 : cursor.stageCount;
  std::vector<uint32_t> assignments =
      cursor.serialized ? std::vector<uint32_t>(scope.stageableEvents.size(), 0)
                        : cursor.eventStages;
  bool includeCurrent = cursor.serialized;
  while (stageCount <= boundedMaximum) {
    std::optional<bool> found =
        searchAssignment(stageCount, assignments, includeCurrent);
    if (!found)
      return {ScopeAdvanceKind::Indeterminate,
              "execution-structure successor exceeded its work limit"};
    if (*found && hasSteadyState(scope.tripCount, stageCount, 1)) {
      cursor.serialized = false;
      cursor.stageCount = stageCount;
      cursor.eventStages = std::move(assignments);
      cursor.launchDistance = 1;
      return {ScopeAdvanceKind::Choice, {}};
    }
    ++stageCount;
    assignments.assign(scope.stageableEvents.size(), 0);
    includeCurrent = true;
  }
  return {ScopeAdvanceKind::End, {}};
}

ExecutionStructureSuccessor ExecutionStructureDomain::getNextPlan(
    const ExecutionStructureCursor &cursor) const {
  if (cursor.scopes.size() != scopes.size() || !contains(buildPlan(cursor)))
    return {ExecutionStructureSuccessorKind::CompilerBug,
            {},
            {},
            "execution-structure cursor is outside the current domain"};
  ExecutionStructureCursor next = cursor;
  for (size_t reverse = 0; reverse < scopes.size(); ++reverse) {
    const size_t index = scopes.size() - reverse - 1;
    ScopeAdvance advanced = advanceScope(index, next.scopes[index]);
    if (advanced.kind == ScopeAdvanceKind::Indeterminate)
      return {ExecutionStructureSuccessorKind::Indeterminate,
              {},
              {},
              std::move(advanced.detail)};
    if (advanced.kind == ScopeAdvanceKind::Choice) {
      for (size_t reset = index + 1; reset < next.scopes.size(); ++reset)
        next.scopes[reset] = {};
      ExecutionStructurePlan plan = buildPlan(next);
      if (!contains(plan))
        return {ExecutionStructureSuccessorKind::CompilerBug,
                {},
                {},
                "execution-structure successor produced an invalid plan"};
      return {ExecutionStructureSuccessorKind::Plan, std::move(plan),
              std::move(next)};
    }
    next.scopes[index] = {};
  }
  return {ExecutionStructureSuccessorKind::End};
}

bool ExecutionStructureDomain::containsScope(
    size_t index, const ExecutionStructureChoice &choice) const {
  const ScopeDomain &scope = scopes[index];
  if (!(getPipelineScope(choice) == scope.id))
    return false;
  if (std::holds_alternative<SerializedExecutionStructure>(choice))
    return true;
  if (!scope.pipelinedEligible || scope.id.recurrences.size() != 1)
    return false;
  const auto &pipelined = std::get<PipelinedExecutionStructure>(choice);
  if (!(pipelined.recurrence == scope.id.recurrences.front()) ||
      pipelined.launchDistance == 0 ||
      pipelined.eventStages.size() != scope.stageableEvents.size())
    return false;
  std::vector<uint32_t> assignments;
  assignments.reserve(pipelined.eventStages.size());
  uint32_t stageCount = 0;
  for (auto [expected, actual] :
       llvm::zip_equal(scope.stageableEvents, pipelined.eventStages)) {
    if (!(expected == actual.event))
      return false;
    assignments.push_back(actual.stage.getValue());
    stageCount = std::max(stageCount, actual.stage.getValue() + 1);
  }
  const uint32_t maximumStages = static_cast<uint32_t>(
      std::min<uint64_t>(scope.stageableEvents.size(), scope.tripCount));
  if (stageCount > maximumStages ||
      (limits.maxStages != 0 && stageCount > limits.maxStages) ||
      !isStageAssignmentLegal(scope.stageableEvents, scope.dependencies,
                              assignments, stageCount) ||
      !hasSteadyState(scope.tripCount, stageCount, pipelined.launchDistance))
    return false;
  return true;
}

bool ExecutionStructureDomain::contains(
    const ExecutionStructurePlan &plan) const {
  if (plan.scopes.size() != scopes.size())
    return false;
  for (auto [index, choice] : llvm::enumerate(plan.scopes))
    if (!containsScope(index, choice))
      return false;
  return true;
}

ExecutionStructureDomainResult buildExecutionStructureDomain(
    llvm::ArrayRef<ExecutionStructureScopeDescription> descriptions,
    const ExecutionStructureLimits &limits) {
  if (descriptions.empty())
    return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                  "execution-structure domain requires a nonempty scope set");
  if (limits.maxSuccessorSteps == 0)
    return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                  "execution-structure successor work limit must be positive");
  std::vector<ExecutionStructureScopeDescription> scopes(descriptions.begin(),
                                                         descriptions.end());
  std::set<EventId> allEvents;
  for (ExecutionStructureScopeDescription &scope : scopes) {
    llvm::sort(scope.id.events);
    llvm::sort(scope.id.recurrences);
    llvm::sort(scope.stageableEvents);
    llvm::sort(scope.dependencies);
    if (scope.id.events.empty() ||
        std::adjacent_find(scope.id.events.begin(), scope.id.events.end()) !=
            scope.id.events.end() ||
        std::adjacent_find(scope.id.recurrences.begin(),
                           scope.id.recurrences.end()) !=
            scope.id.recurrences.end() ||
        std::adjacent_find(scope.stageableEvents.begin(),
                           scope.stageableEvents.end()) !=
            scope.stageableEvents.end() ||
        std::adjacent_find(scope.dependencies.begin(),
                           scope.dependencies.end()) !=
            scope.dependencies.end())
      return failed(
          ExecutionStructureDomainFailureKind::BrokenContract,
          "execution-structure scope has an empty or duplicate inventory");
    std::set<EventId> scopeEvents(scope.id.events.begin(),
                                  scope.id.events.end());
    std::set<EventId> stageable(scope.stageableEvents.begin(),
                                scope.stageableEvents.end());
    for (const EventId &event : scope.id.events)
      if (!allEvents.insert(event).second)
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "execution-structure scopes overlap");
    for (const EventId &event : scope.stageableEvents)
      if (!scopeEvents.count(event))
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "stageable event is outside its structure scope");
    for (const EventDependency &dependency : scope.dependencies)
      if (dependency.predecessor == dependency.successor ||
          !stageable.count(dependency.predecessor) ||
          !stageable.count(dependency.successor))
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "structure dependency is outside its stageable events");
    std::map<EventId, size_t> indegree;
    std::map<EventId, std::set<EventId>> successors;
    for (const EventId &event : scope.stageableEvents)
      indegree.try_emplace(event, 0);
    for (const EventDependency &dependency : scope.dependencies)
      if (successors[dependency.predecessor]
              .insert(dependency.successor)
              .second)
        ++indegree[dependency.successor];
    std::set<EventId> ready;
    for (const auto &[event, degree] : indegree)
      if (degree == 0)
        ready.insert(event);
    size_t visited = 0;
    while (!ready.empty()) {
      EventId event = *ready.begin();
      ready.erase(ready.begin());
      ++visited;
      for (const EventId &successor : successors[event])
        if (--indegree[successor] == 0)
          ready.insert(successor);
    }
    if (visited != scope.stageableEvents.size())
      return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                    "execution-structure scope has a cyclic hard dependency");
    if (scope.pipelinedEligible) {
      if (scope.id.recurrences.size() != 1 || scope.tripCount < 2 ||
          scope.stageableEvents.size() < 2)
        return failed(
            ExecutionStructureDomainFailureKind::BrokenContract,
            "eligible pipeline scope lacks recurrence, trips, or events");
      std::optional<uint64_t> recurrenceTripCount =
          getTripCount(scope.id.recurrences.front());
      if (!recurrenceTripCount || *recurrenceTripCount != scope.tripCount)
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "pipeline trip count differs from its recurrence");
    }
  }
  llvm::sort(scopes,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  for (size_t index = 1; index < scopes.size(); ++index)
    if (scopes[index - 1].id == scopes[index].id)
      return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                    "execution-structure domain has duplicate scopes");
  return {ExecutionStructureDomain(std::move(scopes), limits), {}};
}

ExecutionStructureDomainResult buildExecutionStructureDomain(
    const EventGraph &graph,
    llvm::ArrayRef<TemporalScopeDescriptor> temporalScopes,
    const TemporalPlan &temporal, const ExecutionStructureLimits &limits) {
  if (graph.getEvents().empty() || graph.getComponents().empty())
    return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                  "execution-structure domain requires a nonempty EventGraph");
  if (limits.maxSuccessorSteps == 0)
    return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                  "execution-structure successor work limit must be positive");

  std::map<TraversalScopeId, const TemporalScopeDescriptor *> descriptors;
  for (const TemporalScopeDescriptor &descriptor : temporalScopes)
    if (!descriptors.try_emplace(descriptor.id, &descriptor).second)
      return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                    "temporal domain has duplicate scope descriptors");
  std::map<ExecutionInstanceId, std::vector<OccurrenceRelationId>>
      occurrencesByExecution;
  std::set<TraversalScopeId> selectedScopes;
  for (const TemporalScopePlan &selected : temporal.scopes) {
    auto descriptor = descriptors.find(selected.id);
    if (descriptor == descriptors.end() ||
        !selectedScopes.insert(selected.id).second ||
        selected.iteratorTileSizes.size() !=
            descriptor->second->iterationExtents.size())
      return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                    "selected temporal plan and descriptors disagree");
    OccurrenceRelationId recurrence;
    recurrence.scope = selected.id;
    for (auto [extent, tile] :
         llvm::zip_equal(descriptor->second->iterationExtents,
                         selected.iteratorTileSizes)) {
      if (extent <= 0 || tile <= 0)
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "temporal occurrence has a non-positive extent or tile");
      recurrence.axisOccurrences.push_back(
          static_cast<uint64_t>(1 + (extent - 1) / tile));
    }
    if (!getTripCount(recurrence))
      return failed(ExecutionStructureDomainFailureKind::Indeterminate,
                    "temporal occurrence count overflows");
    const ExecutionInstanceId *execution = getRequiredExecution(selected.id);
    if (execution)
      occurrencesByExecution[*execution].push_back(std::move(recurrence));
  }

  std::set<EventId> allEvents;
  std::vector<ExecutionStructureScopeDescription> scopeDomains;
  for (const EventComponent &component : graph.getComponents()) {
    if (component.events.empty())
      return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                    "EventGraph has an empty component");
    ExecutionStructureScopeDescription scope;
    scope.id.events = component.events;
    llvm::sort(scope.id.events);
    if (std::adjacent_find(scope.id.events.begin(), scope.id.events.end()) !=
        scope.id.events.end())
      return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                    "EventGraph component has duplicate events");
    for (const EventId &event : scope.id.events) {
      if (!graph.contains(event) || !allEvents.insert(event).second)
        return failed(
            ExecutionStructureDomainFailureKind::BrokenContract,
            "EventGraph components overlap or reference unknown events");
      if (event.kind != PlannedEventKind::ObservableWrite)
        scope.stageableEvents.push_back(event);
      const auto *execution = std::get_if<ExecutionEventAction>(&event.action);
      if (!execution)
        continue;
      auto occurrences = occurrencesByExecution.find(execution->execution);
      if (occurrences != occurrencesByExecution.end())
        scope.id.recurrences.insert(scope.id.recurrences.end(),
                                    occurrences->second.begin(),
                                    occurrences->second.end());
    }
    llvm::sort(scope.id.recurrences);
    scope.id.recurrences.erase(
        std::unique(scope.id.recurrences.begin(), scope.id.recurrences.end()),
        scope.id.recurrences.end());
    llvm::sort(scope.stageableEvents);
    std::set<EventId> stageable(scope.stageableEvents.begin(),
                                scope.stageableEvents.end());
    for (const EventDependency &dependency : graph.getHardDependencies())
      if (stageable.count(dependency.predecessor) &&
          stageable.count(dependency.successor))
        scope.dependencies.push_back(dependency);
    if (scope.id.recurrences.size() == 1) {
      std::optional<uint64_t> tripCount =
          getTripCount(scope.id.recurrences.front());
      if (!tripCount)
        return failed(ExecutionStructureDomainFailureKind::Indeterminate,
                      "pipeline occurrence count overflows");
      scope.tripCount = *tripCount;
      scope.pipelinedEligible =
          scope.tripCount >= 2 && scope.stageableEvents.size() >= 2;
    }
    scopeDomains.push_back(std::move(scope));
  }
  if (allEvents.size() != graph.getEvents().size())
    return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                  "EventGraph components do not cover every event");
  return buildExecutionStructureDomain(scopeDomains, limits);
}

} // namespace wafer::compiler::detail
