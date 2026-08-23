//===- ExecutionStructureDomainTest.cpp -------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructureDomain.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

ExecutionInstanceId makeExecution(uint32_t anchor) {
  SemanticRootKey root;
  root.anchorIndex = anchor;
  analysis::RootRegionWorkId work{root, TileId(anchor % 2)};
  LogicalShardId shard{root, {anchor}};
  return ExecutionInstanceId{RequiredRootExecution{work, shard}};
}

EventId makeEvent(uint32_t anchor) {
  return {ExecutionEventAction{makeExecution(anchor)},
          PlannedEventKind::ComputeIssue};
}

ExecutionStructureScopeDescription
makeScope(uint32_t firstAnchor, unsigned eventCount, uint64_t tripCount,
          llvm::ArrayRef<std::pair<unsigned, unsigned>> edges = {}) {
  ExecutionStructureScopeDescription scope;
  for (unsigned index = 0; index < eventCount; ++index)
    scope.id.events.push_back(makeEvent(firstAnchor + index));
  scope.stageableEvents = scope.id.events;
  OccurrenceRelationId recurrence;
  recurrence.scope = TraversalScopeId{
      RegionExecutionId{makeExecution(firstAnchor)}, TopLevelWorkPieceId{0}};
  recurrence.axisOccurrences = {1, tripCount, 1};
  scope.id.recurrences.push_back(recurrence);
  scope.tripCount = tripCount;
  scope.pipelinedEligible = eventCount >= 2 && tripCount >= 2;
  for (auto [before, after] : edges)
    scope.dependencies.push_back({scope.stageableEvents[before],
                                  scope.stageableEvents[after],
                                  EventDependencyReason::SSAValue});
  return scope;
}

std::optional<std::set<ExecutionStructurePlan>>
enumerate(const ExecutionStructureDomain &domain, size_t limit = 100000) {
  std::set<ExecutionStructurePlan> plans;
  ExecutionStructureSuccessor next = domain.getFirstPlan();
  while (next.getKind() == ExecutionStructureSuccessorKind::Plan) {
    if (!next.getPlan() || !next.getCursor() ||
        !domain.contains(*next.getPlan()) || plans.size() >= limit ||
        !plans.insert(*next.getPlan()).second)
      return std::nullopt;
    ExecutionStructureCursor cursor = *next.getCursor();
    next = domain.getNextPlan(cursor);
  }
  return next.getKind() == ExecutionStructureSuccessorKind::End
             ? std::optional<std::set<ExecutionStructurePlan>>(std::move(plans))
             : std::nullopt;
}

bool dependencyLegal(const ExecutionStructureScopeDescription &scope,
                     llvm::ArrayRef<uint32_t> stages) {
  std::map<EventId, uint32_t> byEvent;
  for (auto [event, stage] : llvm::zip_equal(scope.stageableEvents, stages))
    byEvent[event] = stage;
  return llvm::all_of(scope.dependencies, [&](const auto &dependency) {
    return byEvent[dependency.predecessor] <= byEvent[dependency.successor];
  });
}

std::set<ExecutionStructurePlan>
reference(const ExecutionStructureScopeDescription &scope,
          uint32_t maximumStages = 0) {
  std::set<ExecutionStructurePlan> plans;
  plans.insert(
      ExecutionStructurePlan{{SerializedExecutionStructure{scope.id}}});
  if (!scope.pipelinedEligible)
    return plans;
  const uint32_t upper = static_cast<uint32_t>(
      std::min<uint64_t>(scope.stageableEvents.size(), scope.tripCount));
  const uint32_t bounded =
      maximumStages == 0 ? upper : std::min(upper, maximumStages);
  for (uint32_t stageCount = 2; stageCount <= bounded; ++stageCount) {
    std::vector<uint32_t> assignment(scope.stageableEvents.size(), 0);
    std::function<void(size_t)> visit = [&](size_t index) {
      if (index != assignment.size()) {
        for (uint32_t stage = 0; stage < stageCount; ++stage) {
          assignment[index] = stage;
          visit(index + 1);
        }
        return;
      }
      std::vector<bool> used(stageCount, false);
      for (uint32_t stage : assignment)
        used[stage] = true;
      if (llvm::is_contained(used, false) ||
          !dependencyLegal(scope, assignment))
        return;
      for (uint64_t distance = 1;
           uint64_t(stageCount - 1) <= (scope.tripCount - 1) / distance;
           ++distance) {
        PipelinedExecutionStructure pipelined;
        pipelined.scope = scope.id;
        pipelined.recurrence = scope.id.recurrences.front();
        pipelined.launchDistance = distance;
        for (auto [event, stage] :
             llvm::zip_equal(scope.stageableEvents, assignment))
          pipelined.eventStages.push_back({event, StageId(stage)});
        plans.insert(ExecutionStructurePlan{{std::move(pipelined)}});
      }
    };
    visit(0);
  }
  return plans;
}

TEST(ExecutionStructureDomainTest,
     TinyExactSuccessorMatchesIndependentOrderedPartitionOracle) {
  for (unsigned eventCount = 2; eventCount <= 6; ++eventCount) {
    SCOPED_TRACE(eventCount);
    ExecutionStructureScopeDescription scope =
        makeScope(0, eventCount, 7, {{0, eventCount - 1}});
    ExecutionStructureDomainResult result =
        buildExecutionStructureDomain({scope});
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    auto actual = enumerate(*result.domain);
    ASSERT_TRUE(actual);
    EXPECT_EQ(*actual, reference(scope));
  }
}

TEST(ExecutionStructureDomainTest,
     RealScaleAlignedAndRaggedRecurrencesRetainExactAxisCounts) {
  for (uint64_t sequenceExtent :
       {uint64_t{1024}, uint64_t{1025}, uint64_t{1031}}) {
    SCOPED_TRACE(sequenceExtent);
    const uint64_t sequenceOccurrences = (sequenceExtent + 127) / 128;
    ExecutionStructureScopeDescription scope =
        makeScope(0, 3, 2 * sequenceOccurrences, {{0, 1}, {1, 2}});
    scope.id.recurrences.front().axisOccurrences = {2, sequenceOccurrences, 1};
    ExecutionStructureLimits limits;
    limits.maxStages = 3;
    ExecutionStructureDomainResult result =
        buildExecutionStructureDomain({scope}, limits);
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    ExecutionStructureSuccessor serialized = result.domain->getFirstPlan();
    ASSERT_EQ(serialized.getKind(), ExecutionStructureSuccessorKind::Plan);
    ASSERT_TRUE(serialized.getCursor());
    ExecutionStructureSuccessor pipeline =
        result.domain->getNextPlan(*serialized.getCursor());
    ASSERT_EQ(pipeline.getKind(), ExecutionStructureSuccessorKind::Plan);
    ASSERT_TRUE(pipeline.getPlan());
    const auto &choice =
        std::get<PipelinedExecutionStructure>(pipeline.getPlan()->scopes[0]);
    EXPECT_EQ(choice.recurrence.axisOccurrences,
              (llvm::SmallVector<uint64_t, 4>{2, sequenceOccurrences, 1}));
    EXPECT_TRUE(result.domain->contains(*pipeline.getPlan()));
  }
}

TEST(ExecutionStructureDomainTest,
     IndependentScopesFormCartesianProductWithoutCardGlobalMerge) {
  ExecutionStructureScopeDescription first = makeScope(0, 2, 3);
  ExecutionStructureScopeDescription second = makeScope(10, 2, 3);
  ExecutionStructureDomainResult combined =
      buildExecutionStructureDomain({second, first});
  ASSERT_TRUE(combined.succeeded())
      << (combined.failure ? combined.failure->detail : "");
  auto plans = enumerate(*combined.domain);
  ASSERT_TRUE(plans);
  EXPECT_EQ(plans->size(), reference(first).size() * reference(second).size());

  std::reverse(first.id.events.begin(), first.id.events.end());
  std::reverse(first.stageableEvents.begin(), first.stageableEvents.end());
  ExecutionStructureDomainResult reordered =
      buildExecutionStructureDomain({first, second});
  ASSERT_TRUE(reordered.succeeded())
      << (reordered.failure ? reordered.failure->detail : "");
  auto reorderedPlans = enumerate(*reordered.domain);
  ASSERT_TRUE(reorderedPlans);
  EXPECT_EQ(*plans, *reorderedPlans);
}

TEST(ExecutionStructureDomainTest,
     IneligibleAndFailureClassesDoNotRemoveSerializedIdentity) {
  ExecutionStructureScopeDescription oneTrip = makeScope(0, 3, 1);
  oneTrip.pipelinedEligible = false;
  ExecutionStructureDomainResult serializedOnly =
      buildExecutionStructureDomain({oneTrip});
  ASSERT_TRUE(serializedOnly.succeeded());
  auto plans = enumerate(*serializedOnly.domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 1u);
  EXPECT_TRUE(std::holds_alternative<SerializedExecutionStructure>(
      plans->begin()->scopes.front()));

  ExecutionStructureScopeDescription multipleRecurrences = makeScope(10, 3, 7);
  OccurrenceRelationId secondRecurrence =
      multipleRecurrences.id.recurrences.front();
  secondRecurrence.scope = TraversalScopeId{
      RegionExecutionId{makeExecution(20)}, TopLevelWorkPieceId{0}};
  multipleRecurrences.id.recurrences.push_back(secondRecurrence);
  multipleRecurrences.pipelinedEligible = false;
  ExecutionStructureDomainResult multi =
      buildExecutionStructureDomain({multipleRecurrences});
  ASSERT_TRUE(multi.succeeded());
  auto multiPlans = enumerate(*multi.domain);
  ASSERT_TRUE(multiPlans);
  EXPECT_EQ(multiPlans->size(), 1u);

  ExecutionStructureScopeDescription malformed = makeScope(0, 3, 7);
  malformed.stageableEvents.push_back(malformed.stageableEvents.front());
  ExecutionStructureDomainResult broken =
      buildExecutionStructureDomain({malformed});
  ASSERT_FALSE(broken.succeeded());
  ASSERT_TRUE(broken.failure);
  EXPECT_EQ(broken.failure->kind,
            ExecutionStructureDomainFailureKind::BrokenContract);

  ExecutionStructureLimits limits;
  limits.maxSuccessorSteps = 1;
  ExecutionStructureDomainResult limited =
      buildExecutionStructureDomain({makeScope(0, 4, 7)}, limits);
  ASSERT_TRUE(limited.succeeded());
  ExecutionStructureSuccessor first = limited.domain->getFirstPlan();
  ASSERT_TRUE(first.getCursor());
  ExecutionStructureSuccessor next =
      limited.domain->getNextPlan(*first.getCursor());
  EXPECT_EQ(next.getKind(), ExecutionStructureSuccessorKind::Indeterminate);
}

} // namespace
