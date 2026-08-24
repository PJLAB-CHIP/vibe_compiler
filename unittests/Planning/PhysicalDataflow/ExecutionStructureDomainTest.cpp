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

struct Edge {
  unsigned source = 0;
  unsigned destination = 0;
  uint32_t distance = 0;
};

ExecutionStructureScopeDescription
makeScope(uint32_t firstAnchor, unsigned eventCount, uint64_t steadyTripCount,
          uint64_t tailCount = 0, llvm::ArrayRef<Edge> edges = {}) {
  ExecutionStructureScopeDescription scope;
  for (unsigned index = 0; index < eventCount; ++index)
    scope.id.events.push_back(makeEvent(firstAnchor + index));
  scope.stageableEvents = scope.id.events;
  OccurrenceRelationId recurrence;
  recurrence.scope = TraversalScopeId{
      RegionExecutionId{makeExecution(firstAnchor)}, TopLevelWorkPieceId{0}};
  recurrence.axisOccurrences = {1, 1 + steadyTripCount + tailCount, 1};
  scope.id.recurrences.push_back(recurrence);
  scope.iteration =
      PipelineIterationClass{/*recurrenceAxis=*/1,
                             /*prefixCount=*/1, steadyTripCount, tailCount};
  scope.capability = eventCount >= 2 && steadyTripCount >= 2
                         ? CyclicExecutionCapability::SCFDistanceOne
                         : CyclicExecutionCapability::Unsupported;
  for (const Edge &edge : edges)
    scope.dependences.push_back({scope.stageableEvents[edge.source],
                                 scope.stageableEvents[edge.destination],
                                 edge.distance,
                                 PipelineDependenceKind::DataReady});
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
  return llvm::all_of(scope.dependences, [&](const auto &dependence) {
    return dependence.iterationDistance == 1 ||
           byEvent[dependence.source] <= byEvent[dependence.destination];
  });
}

std::set<ExecutionStructurePlan>
reference(const ExecutionStructureScopeDescription &scope,
          uint32_t maximumStages = 0) {
  std::set<ExecutionStructurePlan> plans;
  plans.insert(
      ExecutionStructurePlan{{SerializedExecutionStructure{scope.id}}});
  if (scope.capability != CyclicExecutionCapability::SCFDistanceOne ||
      !scope.iteration)
    return plans;
  const uint32_t upper = static_cast<uint32_t>(std::min<uint64_t>(
      scope.stageableEvents.size(), scope.iteration->steadyTripCount));
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
      PipelinedExecutionStructure pipelined;
      pipelined.scope = scope.id;
      pipelined.recurrence = scope.id.recurrences.front();
      pipelined.iteration = *scope.iteration;
      pipelined.launchDistance = 1;
      pipelined.dependences = scope.dependences;
      pipelined.completionObligations = scope.completionObligations;
      for (auto [event, stage] :
           llvm::zip_equal(scope.stageableEvents, assignment))
        pipelined.eventStages.push_back({event, StageId(stage)});
      plans.insert(ExecutionStructurePlan{{std::move(pipelined)}});
    };
    visit(0);
  }
  return plans;
}

TEST(ExecutionStructureDomainTest,
     TinyExactSuccessorMatchesIndependentOrderedPartitionOracle) {
  // Tiny event counts are intentional: this is a bounded complete oracle. The
  // same algorithm has representative 1024/1025/1031 coverage below.
  for (unsigned eventCount = 2; eventCount <= 6; ++eventCount) {
    SCOPED_TRACE(eventCount);
    ExecutionStructureScopeDescription scope =
        makeScope(0, eventCount, /*steadyTripCount=*/7,
                  /*tailCount=*/0, {{0, eventCount - 1, 0}});
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
     RealScaleAlignedAndRaggedRecurrencesRetainPrefixSteadyAndTail) {
  for (uint64_t sequenceExtent :
       {uint64_t{1024}, uint64_t{1025}, uint64_t{1031}}) {
    SCOPED_TRACE(sequenceExtent);
    const uint64_t totalOccurrences = (sequenceExtent + 127) / 128;
    const uint64_t tailCount = sequenceExtent % 128 == 0 ? 0 : 1;
    const uint64_t steadyTripCount = sequenceExtent / 128 - 1;
    ExecutionStructureScopeDescription scope =
        makeScope(0, 3, steadyTripCount, tailCount, {{0, 1, 0}, {1, 2, 0}});
    scope.id.recurrences.front().axisOccurrences = {2, totalOccurrences, 1};
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
              (llvm::SmallVector<uint64_t, 4>{2, totalOccurrences, 1}));
    EXPECT_EQ(choice.iteration.prefixCount, 1u);
    EXPECT_EQ(choice.iteration.steadyTripCount, 7u);
    EXPECT_EQ(choice.iteration.tailCount, tailCount);
    EXPECT_EQ(choice.launchDistance, 1u);
    EXPECT_TRUE(result.domain->contains(*pipeline.getPlan()));
  }
}

TEST(ExecutionStructureDomainTest,
     DistanceOneAndDirectDTESelectOnlySupportedLowering) {
  ExecutionStructureScopeDescription scope = makeScope(
      0, 3, /*steadyTripCount=*/7, /*tailCount=*/1, {{2, 0, 1}, {0, 1, 0}});
  scope.completionObligations.push_back({scope.stageableEvents[0],
                                         scope.stageableEvents[1],
                                         CompletionProtocol::DirectDTE, 0});
  ExecutionStructureDomainResult result =
      buildExecutionStructureDomain({scope});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  bool sawSCF = false;
  bool sawFinite = false;
  for (const ExecutionStructurePlan &plan : *plans) {
    const auto *pipeline =
        std::get_if<PipelinedExecutionStructure>(&plan.scopes.front());
    if (!pipeline)
      continue;
    sawSCF |= pipeline->lowering == ExecutionStructureLowering::SCFDistanceOne;
    sawFinite |=
        pipeline->lowering == ExecutionStructureLowering::FiniteUnrolled;
  }
  EXPECT_TRUE(sawSCF);
  EXPECT_TRUE(sawFinite);

  ExecutionStructureLimits limited;
  limited.maxFiniteUnrolledOperations = 1;
  ExecutionStructureDomainResult bounded =
      buildExecutionStructureDomain({scope}, limited);
  ASSERT_TRUE(bounded.succeeded());
  auto boundedPlans = enumerate(*bounded.domain);
  ASSERT_TRUE(boundedPlans);
  EXPECT_TRUE(llvm::none_of(*boundedPlans, [](const auto &plan) {
    const auto *pipeline =
        std::get_if<PipelinedExecutionStructure>(&plan.scopes.front());
    return pipeline &&
           pipeline->lowering == ExecutionStructureLowering::FiniteUnrolled;
  }));
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
     IneligibleAndFailureClassesDoNotInventPipelinedFallback) {
  ExecutionStructureScopeDescription noSteady = makeScope(0, 3, 1);
  ExecutionStructureDomainResult serializedOnly =
      buildExecutionStructureDomain({noSteady});
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
  multipleRecurrences.capability = CyclicExecutionCapability::Unsupported;
  multipleRecurrences.iteration.reset();
  ExecutionStructureDomainResult multi =
      buildExecutionStructureDomain({multipleRecurrences});
  ASSERT_TRUE(multi.succeeded());
  auto multiPlans = enumerate(*multi.domain);
  ASSERT_TRUE(multiPlans);
  EXPECT_EQ(multiPlans->size(), 1u);

  ExecutionStructureScopeDescription synchronous = makeScope(30, 2, 7);
  synchronous.completionObligations.push_back(
      {synchronous.stageableEvents[0], synchronous.stageableEvents[1],
       CompletionProtocol::NCCSynchronousWriteback, 1});
  ExecutionStructureDomainResult synchronousResult =
      buildExecutionStructureDomain({synchronous});
  ASSERT_TRUE(synchronousResult.succeeded());
  auto synchronousPlans = enumerate(*synchronousResult.domain);
  ASSERT_TRUE(synchronousPlans);
  EXPECT_EQ(synchronousPlans->size(), 1u);

  ExecutionStructureScopeDescription malformed = makeScope(0, 3, 7);
  malformed.stageableEvents.push_back(malformed.stageableEvents.front());
  ExecutionStructureDomainResult broken =
      buildExecutionStructureDomain({malformed});
  ASSERT_FALSE(broken.succeeded());
  ASSERT_TRUE(broken.failure);
  EXPECT_EQ(broken.failure->kind,
            ExecutionStructureDomainFailureKind::BrokenContract);

  ExecutionStructureScopeDescription unsupportedDistance = makeScope(0, 3, 7);
  unsupportedDistance.dependences.push_back(
      {unsupportedDistance.stageableEvents[0],
       unsupportedDistance.stageableEvents[1], 2,
       PipelineDependenceKind::DataReady});
  ExecutionStructureDomainResult distance =
      buildExecutionStructureDomain({unsupportedDistance});
  ASSERT_FALSE(distance.succeeded());
  ASSERT_TRUE(distance.failure);
  EXPECT_EQ(distance.failure->kind,
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
