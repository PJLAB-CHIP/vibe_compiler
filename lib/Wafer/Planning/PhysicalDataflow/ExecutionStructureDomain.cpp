//===- ExecutionStructureDomain.cpp - Finite structure domain --------===//

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructureDomain.h"

#include "mlir/Interfaces/TilingInterface.h"

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

bool hasFiniteOccurrenceCount(const OccurrenceRelationId &recurrence) {
  uint64_t product = 1;
  for (uint64_t count : recurrence.axisOccurrences) {
    if (count == 0 || product > std::numeric_limits<uint64_t>::max() / count)
      return false;
    product *= count;
  }
  return true;
}

bool hasSteadyState(const PipelineIterationClass &iteration,
                    uint32_t stageCount) {
  return stageCount >= 2 && iteration.prefixCount == 1 &&
         iteration.tailCount <= 1 && iteration.steadyTripCount >= stageCount;
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
                            llvm::ArrayRef<PipelineDependence> dependences,
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
  for (const PipelineDependence &dependence : dependences) {
    auto source = stages.find(dependence.source);
    auto destination = stages.find(dependence.destination);
    if (source == stages.end() || destination == stages.end() ||
        dependence.iterationDistance > 1 ||
        (dependence.iterationDistance == 0 &&
         source->second > destination->second))
      return false;
  }
  return true;
}

std::optional<ExecutionStructureLowering>
getLowering(const ExecutionStructureScopeDescription &scope,
            llvm::ArrayRef<uint32_t> assignments,
            const ExecutionStructureLimits &limits) {
  if (scope.capability != CyclicExecutionCapability::SCFDistanceOne ||
      !scope.iteration || assignments.size() != scope.stageableEvents.size())
    return std::nullopt;
  std::map<EventId, uint32_t> stages;
  for (auto [event, stage] :
       llvm::zip_equal(scope.stageableEvents, assignments))
    stages.emplace(event, stage);

  bool requiresFiniteUnroll = false;
  for (const CompletionObligation &obligation : scope.completionObligations) {
    auto issue = stages.find(obligation.issue);
    auto completion = stages.find(obligation.completion);
    if ((issue == stages.end()) != (completion == stages.end()))
      return std::nullopt;
    if (obligation.protocol == CompletionProtocol::Unknown ||
        obligation.protocol == CompletionProtocol::NCCSynchronousWriteback)
      return std::nullopt;
    if (obligation.protocol == CompletionProtocol::DirectDTE &&
        issue != stages.end() && issue->second != completion->second)
      requiresFiniteUnroll = true;
  }
  if (!requiresFiniteUnroll)
    return ExecutionStructureLowering::SCFDistanceOne;

  if (limits.maxFiniteUnrolledOperations == 0 ||
      scope.stageableEvents.empty() ||
      scope.iteration->steadyTripCount >
          limits.maxFiniteUnrolledOperations / scope.stageableEvents.size())
    return std::nullopt;
  return ExecutionStructureLowering::FiniteUnrolled;
}

PipelineDependenceKind getDependenceKind(EventDependencyReason reason) {
  switch (reason) {
  case EventDependencyReason::Completion:
    return PipelineDependenceKind::AsyncCompletion;
  case EventDependencyReason::BufferLifetime:
    return PipelineDependenceKind::BufferLifetime;
  case EventDependencyReason::EffectOrder:
    return PipelineDependenceKind::Effect;
  case EventDependencyReason::SSAValue:
  case EventDependencyReason::NestedExecution:
  case EventDependencyReason::LoopRecurrence:
  case EventDependencyReason::TransferReady:
  case EventDependencyReason::Publication:
    return PipelineDependenceKind::DataReady;
  }
  return PipelineDependenceKind::DataReady;
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
    std::optional<ExecutionStructureLowering> lowering =
        getLowering(scope, choice.eventStages, limits);
    if (!scope.iteration || !lowering)
      return {};
    PipelinedExecutionStructure pipelined;
    pipelined.scope = scope.id;
    pipelined.recurrence = scope.id.recurrences.front();
    pipelined.iteration = *scope.iteration;
    pipelined.launchDistance = 1;
    pipelined.dependences = scope.dependences;
    pipelined.completionObligations = scope.completionObligations;
    pipelined.lowering = *lowering;
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
  if (scope.capability != CyclicExecutionCapability::SCFDistanceOne ||
      !scope.iteration)
    return {ScopeAdvanceKind::End, {}};
  const uint32_t maximumStages = static_cast<uint32_t>(std::min<uint64_t>(
      scope.stageableEvents.size(), scope.iteration->steadyTripCount));
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
      if (isStageAssignmentLegal(scope.stageableEvents, scope.dependences,
                                 assignments, stageCount) &&
          getLowering(scope, assignments, limits))
        return true;
    }
    return false;
  };

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
    if (*found && hasSteadyState(*scope.iteration, stageCount)) {
      cursor.serialized = false;
      cursor.stageCount = stageCount;
      cursor.eventStages = std::move(assignments);
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
  if (scope.capability != CyclicExecutionCapability::SCFDistanceOne ||
      !scope.iteration || scope.id.recurrences.size() != 1)
    return false;
  const auto &pipelined = std::get<PipelinedExecutionStructure>(choice);
  if (!(pipelined.recurrence == scope.id.recurrences.front()) ||
      !(pipelined.iteration == *scope.iteration) ||
      pipelined.launchDistance != 1 ||
      pipelined.dependences != scope.dependences ||
      pipelined.completionObligations != scope.completionObligations ||
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
  const uint32_t maximumStages = static_cast<uint32_t>(std::min<uint64_t>(
      scope.stageableEvents.size(), scope.iteration->steadyTripCount));
  std::optional<ExecutionStructureLowering> lowering =
      getLowering(scope, assignments, limits);
  return stageCount <= maximumStages &&
         (limits.maxStages == 0 || stageCount <= limits.maxStages) &&
         isStageAssignmentLegal(scope.stageableEvents, scope.dependences,
                                assignments, stageCount) &&
         hasSteadyState(*scope.iteration, stageCount) && lowering &&
         *lowering == pipelined.lowering;
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
  if (limits.maxSuccessorSteps == 0 || limits.maxFiniteUnrolledOperations == 0)
    return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                  "execution-structure work limits must be positive");
  std::vector<ExecutionStructureScopeDescription> scopes(descriptions.begin(),
                                                         descriptions.end());
  std::set<EventId> allEvents;
  for (ExecutionStructureScopeDescription &scope : scopes) {
    llvm::sort(scope.id.events);
    llvm::sort(scope.id.recurrences);
    llvm::sort(scope.stageableEvents);
    llvm::sort(scope.dependences);
    llvm::sort(scope.completionObligations);
    if (scope.id.events.empty() ||
        std::adjacent_find(scope.id.events.begin(), scope.id.events.end()) !=
            scope.id.events.end() ||
        std::adjacent_find(scope.id.recurrences.begin(),
                           scope.id.recurrences.end()) !=
            scope.id.recurrences.end() ||
        std::adjacent_find(scope.stageableEvents.begin(),
                           scope.stageableEvents.end()) !=
            scope.stageableEvents.end() ||
        std::adjacent_find(scope.dependences.begin(),
                           scope.dependences.end()) !=
            scope.dependences.end() ||
        std::adjacent_find(scope.completionObligations.begin(),
                           scope.completionObligations.end()) !=
            scope.completionObligations.end())
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
    for (const PipelineDependence &dependence : scope.dependences)
      if ((dependence.source == dependence.destination &&
           dependence.iterationDistance == 0) ||
          dependence.iterationDistance > 1 ||
          !stageable.count(dependence.source) ||
          !stageable.count(dependence.destination))
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "pipeline dependence is outside its supported event "
                      "and distance domain");
    for (const CompletionObligation &obligation : scope.completionObligations)
      if (!scopeEvents.count(obligation.issue) ||
          !scopeEvents.count(obligation.completion))
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "completion obligation is outside its structure scope");

    std::map<EventId, size_t> indegree;
    std::map<EventId, std::set<EventId>> successors;
    for (const EventId &event : scope.stageableEvents)
      indegree.try_emplace(event, 0);
    for (const PipelineDependence &dependence : scope.dependences)
      if (dependence.iterationDistance == 0 &&
          successors[dependence.source].insert(dependence.destination).second)
        ++indegree[dependence.destination];
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
                    "distance-zero pipeline dependence graph is cyclic");

    if (scope.iteration) {
      const PipelineIterationClass &iteration = *scope.iteration;
      if (scope.id.recurrences.size() != 1 ||
          !hasFiniteOccurrenceCount(scope.id.recurrences.front()) ||
          iteration.recurrenceAxis >=
              scope.id.recurrences.front().axisOccurrences.size() ||
          iteration.prefixCount != 1 || iteration.tailCount > 1 ||
          scope.id.recurrences.front()
                  .axisOccurrences[iteration.recurrenceAxis] !=
              iteration.prefixCount + iteration.steadyTripCount +
                  iteration.tailCount)
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "pipeline iteration class disagrees with its exact "
                      "temporal recurrence");
    }
    if (scope.capability == CyclicExecutionCapability::SCFDistanceOne &&
        (!scope.iteration || scope.iteration->steadyTripCount < 2 ||
         scope.stageableEvents.size() < 2))
      return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                    "cyclic-capable scope lacks a steady loop or events");
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
    const TemporalPlan &temporal,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const ExecutionStructureLimits &limits) {
  if (graph.getEvents().empty() || graph.getComponents().empty())
    return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                  "execution-structure domain requires a nonempty EventGraph");
  if (limits.maxSuccessorSteps == 0 || limits.maxFiniteUnrolledOperations == 0)
    return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                  "execution-structure work limits must be positive");

  std::map<TraversalScopeId, const TemporalScopeDescriptor *> descriptors;
  for (const TemporalScopeDescriptor &descriptor : temporalScopes)
    if (!descriptors.try_emplace(descriptor.id, &descriptor).second)
      return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                    "temporal domain has duplicate scope descriptors");
  std::map<analysis::RootRegionWorkId, mlir::Operation *> roots;
  for (const analysis::RootRegionWork &work : rootWorks)
    if (!work.rootOperation ||
        !roots.try_emplace(work.id, work.rootOperation).second)
      return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                    "execution-structure input has malformed root work");

  struct OccurrenceFacts {
    OccurrenceRelationId recurrence;
    std::optional<PipelineIterationClass> iteration;
  };
  std::map<ExecutionInstanceId, std::vector<OccurrenceFacts>>
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
    OccurrenceFacts facts;
    facts.recurrence.scope = selected.id;
    std::vector<uint32_t> active;
    for (auto [axis, extent, tile] :
         llvm::enumerate(descriptor->second->iterationExtents,
                         selected.iteratorTileSizes)) {
      if (extent <= 0 || tile <= 0 || tile > extent)
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "temporal occurrence has an invalid extent or tile");
      const uint64_t occurrences =
          static_cast<uint64_t>(1 + (extent - 1) / tile);
      facts.recurrence.axisOccurrences.push_back(occurrences);
      if (occurrences > 1)
        active.push_back(static_cast<uint32_t>(axis));
    }
    if (!hasFiniteOccurrenceCount(facts.recurrence))
      return failed(ExecutionStructureDomainFailureKind::Indeterminate,
                    "temporal occurrence count overflows");

    std::vector<uint32_t> orderedActive;
    if (selected.waveLoopOrder.empty()) {
      orderedActive = active;
    } else {
      for (uint32_t axis : selected.waveLoopOrder)
        if (llvm::is_contained(active, axis))
          orderedActive.push_back(axis);
      std::vector<uint32_t> sorted = orderedActive;
      llvm::sort(sorted);
      if (sorted != active)
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "temporal loop order does not cover active axes");
    }
    if (!orderedActive.empty()) {
      const uint32_t axis = orderedActive.back();
      const int64_t extent = descriptor->second->iterationExtents[axis];
      const int64_t tile = selected.iteratorTileSizes[axis];
      const uint64_t full = static_cast<uint64_t>(extent / tile);
      const uint64_t tail = extent % tile == 0 ? 0 : 1;
      facts.iteration =
          PipelineIterationClass{axis, 1, full == 0 ? 0 : full - 1, tail};
      if (facts.recurrence.axisOccurrences[axis] != full + tail)
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "temporal prefix/steady/tail partition is inconsistent");
    }
    const ExecutionInstanceId *execution = getRequiredExecution(selected.id);
    if (execution)
      occurrencesByExecution[*execution].push_back(std::move(facts));
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

    std::map<OccurrenceRelationId, PipelineIterationClass> iterations;
    std::set<ExecutionInstanceId> componentExecutions;
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
      componentExecutions.insert(execution->execution);
      auto occurrences = occurrencesByExecution.find(execution->execution);
      if (occurrences == occurrencesByExecution.end())
        continue;
      for (const OccurrenceFacts &facts : occurrences->second) {
        scope.id.recurrences.push_back(facts.recurrence);
        if (facts.iteration) {
          auto [position, inserted] =
              iterations.try_emplace(facts.recurrence, *facts.iteration);
          if (!inserted && !(position->second == *facts.iteration))
            return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                          "one recurrence has inconsistent iteration classes");
        }
      }
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
        scope.dependences.push_back({dependency.predecessor,
                                     dependency.successor, 0,
                                     getDependenceKind(dependency.reason)});
    for (const CompletionObligation &obligation :
         graph.getCompletionObligations()) {
      const bool hasIssue =
          llvm::is_contained(scope.id.events, obligation.issue);
      const bool hasCompletion =
          llvm::is_contained(scope.id.events, obligation.completion);
      if (hasIssue != hasCompletion)
        return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                      "completion obligation crosses EventGraph components");
      if (hasIssue)
        scope.completionObligations.push_back(obligation);
    }

    bool supported = scope.id.recurrences.size() == 1 &&
                     componentExecutions.size() == 1 &&
                     scope.stageableEvents.size() >= 2;
    if (supported) {
      auto iteration = iterations.find(scope.id.recurrences.front());
      supported = iteration != iterations.end() &&
                  iteration->second.steadyTripCount >= 2;
      if (supported)
        scope.iteration = iteration->second;
    }
    if (supported) {
      for (const CompletionObligation &obligation : scope.completionObligations)
        if (obligation.protocol == CompletionProtocol::Unknown ||
            obligation.protocol ==
                CompletionProtocol::NCCSynchronousWriteback) {
          supported = false;
          break;
        }
    }
    if (supported) {
      const ExecutionInstanceId &execution = *componentExecutions.begin();
      analysis::RootRegionWorkId work = std::visit(
          [](const auto &source) { return source.work; }, execution.source);
      auto root = roots.find(work);
      auto tiling = root == roots.end()
                        ? mlir::TilingInterface{}
                        : mlir::dyn_cast<mlir::TilingInterface>(root->second);
      if (!tiling || !scope.iteration ||
          scope.iteration->recurrenceAxis >=
              tiling.getLoopIteratorTypes().size()) {
        supported = false;
      } else if (tiling
                     .getLoopIteratorTypes()[scope.iteration->recurrenceAxis] ==
                 mlir::utils::IteratorType::reduction) {
        EventId completion{ExecutionEventAction{execution},
                           PlannedEventKind::Completion};
        EventId issue{ExecutionEventAction{execution},
                      PlannedEventKind::ComputeIssue};
        if (!stageable.count(completion) || !stageable.count(issue))
          supported = false;
        else
          scope.dependences.push_back(
              {completion, issue, 1, PipelineDependenceKind::DataReady});
      } else if (tiling
                     .getLoopIteratorTypes()[scope.iteration->recurrenceAxis] !=
                 mlir::utils::IteratorType::parallel) {
        supported = false;
      }
    }
    if (supported)
      scope.capability = CyclicExecutionCapability::SCFDistanceOne;
    llvm::sort(scope.dependences);
    scope.dependences.erase(
        std::unique(scope.dependences.begin(), scope.dependences.end()),
        scope.dependences.end());
    llvm::sort(scope.completionObligations);
    scopeDomains.push_back(std::move(scope));
  }
  if (allEvents.size() != graph.getEvents().size())
    return failed(ExecutionStructureDomainFailureKind::BrokenContract,
                  "EventGraph components do not cover every event");
  return buildExecutionStructureDomain(scopeDomains, limits);
}

} // namespace wafer::compiler::detail
