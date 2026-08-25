//===- ScheduleDomainTest.cpp ----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/ScheduleDomain.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

ExecutionInstanceId makeExecution(uint32_t anchor, TileId tile = TileId(0)) {
  SemanticRootKey root;
  root.anchorIndex = anchor;
  analysis::RootRegionWorkId work{root, tile};
  LogicalShardId shard{root, {anchor}};
  return ExecutionInstanceId{RequiredRootExecution{work, shard}};
}

EventId makeEvent(uint32_t anchor,
                  PlannedEventKind kind = PlannedEventKind::ComputeIssue,
                  TileId tile = TileId(0)) {
  return {ExecutionEventAction{makeExecution(anchor, tile)}, kind};
}

ScheduleDomainInput
makeInput(unsigned eventCount, bool workerCapable,
          llvm::ArrayRef<std::pair<unsigned, unsigned>> edges = {}) {
  ScheduleDomainInput input;
  PhysicalVersionId version{ExecutionResultValueId{makeExecution(100), 0}};
  StorageObjectId object{StorageObjectOrigin{version}};
  input.buffers.storageObjects.push_back({object, TileId(0)});
  input.buffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  PipelineScopeId scope;
  for (unsigned index = 0; index < eventCount; ++index) {
    EventId id = makeEvent(index);
    scope.events.push_back(id);
    PlannedEvent event{id, CardId(0), TileId(0), {}};
    if (workerCapable)
      event.workerDomain = {NCCWorker::Worker0, NCCWorker::Worker1,
                            NCCWorker::Worker2};
    input.events.push_back(std::move(event));
  }
  input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  for (auto [before, after] : edges)
    input.hardDependencies.push_back({scope.events[before], scope.events[after],
                                      EventDependencyReason::SSAValue});
  input.components.push_back({scope.events});
  return input;
}

std::optional<std::set<ClosedSchedulePlan>>
enumerate(const ScheduleDomain &domain, size_t limit = 100000) {
  std::set<ClosedSchedulePlan> plans;
  ScheduleSuccessor next = domain.getFirstPlan();
  while (next.getKind() == ScheduleSuccessorKind::Plan) {
    if (!next.getPlan() || !next.getCursor() ||
        !domain.contains(*next.getPlan()) || plans.size() >= limit ||
        !plans.insert(*next.getPlan()).second)
      return std::nullopt;
    ScheduleCursor cursor = *next.getCursor();
    next = domain.getNextPlan(cursor);
  }
  return next.getKind() == ScheduleSuccessorKind::End
             ? std::optional<std::set<ClosedSchedulePlan>>(std::move(plans))
             : std::nullopt;
}

std::optional<std::set<std::vector<EventId>>>
enumerateControlOrders(const ScheduleDomain &domain, size_t limit = 100000) {
  std::set<std::vector<EventId>> orders;
  ScheduleSuccessor next = domain.getFirstPlan();
  while (next.getKind() == ScheduleSuccessorKind::Plan) {
    if (!next.getPlan() || !next.getCursor() ||
        !domain.contains(*next.getPlan()) ||
        next.getPlan()->controlOrders.size() != 1 || orders.size() >= limit ||
        !orders.insert(next.getPlan()->controlOrders.front().events).second)
      return std::nullopt;
    ScheduleCursor cursor = *next.getCursor();
    next = domain.getNextPlan(cursor);
  }
  return next.getKind() == ScheduleSuccessorKind::End
             ? std::optional<std::set<std::vector<EventId>>>(std::move(orders))
             : std::nullopt;
}

TEST(ScheduleDomainTest, WorkerAndControlCountsMatchEighteenAndFiftyFour) {
  ScheduleDomainResult independent =
      buildScheduleDomain(makeInput(2, /*workerCapable=*/true));
  ASSERT_TRUE(independent.succeeded())
      << (independent.failure ? independent.failure->detail : "");
  auto independentPlans = enumerate(*independent.domain);
  ASSERT_TRUE(independentPlans);
  EXPECT_EQ(independentPlans->size(), 18u);

  ScheduleDomainResult fanout = buildScheduleDomain(
      makeInput(3, /*workerCapable=*/true, {{0, 1}, {0, 2}}));
  ASSERT_TRUE(fanout.succeeded())
      << (fanout.failure ? fanout.failure->detail : "");
  auto fanoutPlans = enumerate(*fanout.domain);
  ASSERT_TRUE(fanoutPlans);
  EXPECT_EQ(fanoutPlans->size(), 54u);
}

TEST(ScheduleDomainTest,
     TwoToSevenEventControlSuccessorsMatchPermutationReference) {
  for (unsigned count = 2; count <= 7; ++count) {
    SCOPED_TRACE(count);
    ScheduleDomainResult result =
        buildScheduleDomain(makeInput(count, /*workerCapable=*/false));
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    auto orders = enumerateControlOrders(*result.domain);
    ASSERT_TRUE(orders);
    size_t factorial = 1;
    for (unsigned value = 2; value <= count; ++value)
      factorial *= value;
    EXPECT_EQ(orders->size(), factorial);
  }
}

TEST(ScheduleDomainTest,
     SharedResourceSequenceConstrainsControlWithoutDuplicateLeaves) {
  ScheduleDomainInput input = makeInput(2, /*workerCapable=*/false);
  ResourceKey sender = DirectDTESenderResource{TileId(0)};
  input.orderChoices.push_back(
      {sender, {input.events[0].id, input.events[1].id}});
  for (const PlannedEvent &event : input.events)
    input.resourceUses.push_back({event.id, sender, ResourceUseMode::Exclusive,
                                  ResourceIntervalKind::Instantaneous,
                                  std::nullopt, ResourceKnowledge::Exact});
  input.resourceUses.push_back(
      {input.events.front().id,
       OpaqueNoCTransferResource{MovementHop{TileId(0), TileId(1)}},
       ResourceUseMode::CapacityUnits, ResourceIntervalKind::IssueToCompletion,
       std::nullopt, ResourceKnowledge::Estimate});
  ScheduleDomainResult result = buildScheduleDomain(input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 2u);
  for (const ClosedSchedulePlan &plan : *plans) {
    ASSERT_EQ(plan.resourceSequences.size(), 1u);
    ASSERT_EQ(plan.controlOrders.size(), 1u);
    EXPECT_EQ(plan.resourceSequences.front().events,
              plan.controlOrders.front().events);
    EXPECT_EQ(plan.resourceBindings.size(), 2u);
  }

  ScheduleDomainInput reversed = input;
  std::reverse(reversed.events.begin(), reversed.events.end());
  std::reverse(reversed.orderChoices.front().events.begin(),
               reversed.orderChoices.front().events.end());
  std::reverse(reversed.resourceUses.begin(), reversed.resourceUses.end());
  ScheduleDomainResult reordered = buildScheduleDomain(std::move(reversed));
  ASSERT_TRUE(reordered.succeeded())
      << (reordered.failure ? reordered.failure->detail : "");
  auto reorderedPlans = enumerate(*reordered.domain);
  ASSERT_TRUE(reorderedPlans);
  EXPECT_EQ(*plans, *reorderedPlans);
}

TEST(ScheduleDomainTest,
     MultiTileControlScopesStayIndependentWithoutCardTotalOrder) {
  ScheduleDomainInput input = makeInput(0, /*workerCapable=*/false);
  input.events.clear();
  input.structure.scopes.clear();
  input.components.clear();
  EventId tile0First = makeEvent(10, PlannedEventKind::ComputeIssue, TileId(0));
  EventId tile0Second =
      makeEvent(11, PlannedEventKind::ComputeIssue, TileId(0));
  EventId tile1First = makeEvent(12, PlannedEventKind::ComputeIssue, TileId(1));
  EventId tile1Second =
      makeEvent(13, PlannedEventKind::ComputeIssue, TileId(1));
  input.events = {{tile0First, CardId(0), TileId(0), {}},
                  {tile0Second, CardId(0), TileId(0), {}},
                  {tile1First, CardId(0), TileId(1), {}},
                  {tile1Second, CardId(0), TileId(1), {}}};
  PipelineScopeId scope{{tile0First, tile0Second, tile1First, tile1Second}, {}};
  input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  input.components.push_back({scope.events});

  ScheduleDomainResult result = buildScheduleDomain(input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  EXPECT_EQ(plans->size(), 4u);
  for (const ClosedSchedulePlan &plan : *plans) {
    ASSERT_EQ(plan.controlOrders.size(), 2u);
    for (const ControlOrder &control : plan.controlOrders) {
      const auto *tile = std::get_if<TileControlScope>(&control.scope);
      ASSERT_NE(tile, nullptr);
      ASSERT_EQ(control.events.size(), 2u);
      EXPECT_TRUE(llvm::all_of(control.events, [&](const EventId &event) {
        auto planned = llvm::find_if(input.events, [&](const auto &candidate) {
          return candidate.id == event;
        });
        return planned != input.events.end() && planned->tile == tile->tile;
      }));
    }
  }
}

TEST(ScheduleDomainTest,
     AsyncExclusiveResourceSequenceOrdersReleaseBeforeNextIssue) {
  ScheduleDomainInput input = makeInput(0, /*workerCapable=*/false);
  input.events.clear();
  input.structure.scopes.clear();
  EventId firstIssue = makeEvent(20, PlannedEventKind::MovementIssue);
  EventId firstCompletion = makeEvent(20, PlannedEventKind::Completion);
  EventId secondIssue = makeEvent(21, PlannedEventKind::MovementIssue);
  EventId secondCompletion = makeEvent(21, PlannedEventKind::Completion);
  PipelineScopeId scope{
      {firstIssue, firstCompletion, secondIssue, secondCompletion}, {}};
  input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  input.events = {{firstIssue, CardId(0), TileId(0), {}},
                  {firstCompletion, CardId(0), TileId(0), {}},
                  {secondIssue, CardId(0), TileId(0), {}},
                  {secondCompletion, CardId(0), TileId(0), {}}};
  input.hardDependencies = {
      {firstIssue, firstCompletion, EventDependencyReason::Completion},
      {secondIssue, secondCompletion, EventDependencyReason::Completion}};
  DirectDTESenderResource sender{TileId(0)};
  input.resourceUses = {{firstIssue, sender, ResourceUseMode::Exclusive,
                         ResourceIntervalKind::IssueToCompletion,
                         firstCompletion, ResourceKnowledge::Exact},
                        {secondIssue, sender, ResourceUseMode::Exclusive,
                         ResourceIntervalKind::IssueToCompletion,
                         secondCompletion, ResourceKnowledge::Exact}};
  input.orderChoices = {{sender, {firstIssue, secondIssue}}};
  input.components = {{scope.events}};
  ScheduleDomainResult result = buildScheduleDomain(input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  ASSERT_FALSE(plans->empty());
  for (const ClosedSchedulePlan &plan : *plans) {
    ASSERT_EQ(plan.resourceSequences.size(), 1u);
    ASSERT_EQ(plan.controlOrders.size(), 1u);
    const std::vector<EventId> &resource =
        plan.resourceSequences.front().events;
    const std::vector<EventId> &control = plan.controlOrders.front().events;
    auto position = [&](const EventId &event) {
      return static_cast<size_t>(
          std::distance(control.begin(), llvm::find(control, event)));
    };
    EventId release =
        resource.front() == firstIssue ? firstCompletion : secondCompletion;
    EXPECT_LT(position(release), position(resource.back()));
  }
}

TEST(ScheduleDomainTest,
     CompletionPlacementUsesSelectedWorkerAndKeepsDTESeparate) {
  ScheduleDomainInput input = makeInput(0, /*workerCapable=*/false);
  input.events.clear();
  input.structure.scopes.clear();
  EventId issue = makeEvent(0, PlannedEventKind::ComputeIssue);
  EventId completion = makeEvent(0, PlannedEventKind::Completion);
  EventId dteIssue = makeEvent(1, PlannedEventKind::MovementIssue);
  EventId dteCompletion = makeEvent(1, PlannedEventKind::Completion);
  PipelineScopeId scope{{issue, completion, dteIssue, dteCompletion}, {}};
  input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  input.events = {
      {issue,
       CardId(0),
       TileId(0),
       {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
      {completion, CardId(0), TileId(0), {}},
      {dteIssue, CardId(0), TileId(0), {}},
      {dteCompletion, CardId(0), TileId(0), {}}};
  input.hardDependencies = {
      {issue, completion, EventDependencyReason::Completion},
      {dteIssue, dteCompletion, EventDependencyReason::Completion}};
  input.completionObligations = {
      {issue, completion, CompletionProtocol::NCCParticipant, 0},
      {dteIssue, dteCompletion, CompletionProtocol::DirectDTE, 0}};
  input.components = {{scope.events}};
  ScheduleDomainResult result = buildScheduleDomain(input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  EXPECT_EQ(plans->size(), 18u); // 3 workers x 6 legal interleavings.
  for (const ClosedSchedulePlan &plan : *plans) {
    ASSERT_EQ(plan.workerBindings.size(), 1u);
    ASSERT_EQ(plan.completionPlacements.size(), 2u);
    auto ncc = llvm::find_if(plan.completionPlacements, [&](const auto &entry) {
      return entry.issue == issue;
    });
    auto dte = llvm::find_if(plan.completionPlacements, [&](const auto &entry) {
      return entry.issue == dteIssue;
    });
    ASSERT_NE(ncc, plan.completionPlacements.end());
    ASSERT_NE(dte, plan.completionPlacements.end());
    EXPECT_EQ(ncc->participantMask, uint32_t{1} << static_cast<uint32_t>(
                                        plan.workerBindings.front().worker));
    EXPECT_EQ(dte->participantMask, 0u);
    ASSERT_EQ(plan.controlOrders.size(), 1u);
    ASSERT_FALSE(plan.controlOrders.front().events.empty());
    const EventId &latest = plan.controlOrders.front().events.back();
    EXPECT_EQ(ncc->boundary.after, latest);
    EXPECT_EQ(dte->boundary.after, latest);
  }
}

TEST(ScheduleDomainTest, CompletionPlacementDoesNotCrossSelectedKStage) {
  ScheduleDomainInput input = makeInput(0, /*workerCapable=*/false);
  input.events.clear();
  input.structure.scopes.clear();
  EventId issue = makeEvent(0, PlannedEventKind::ComputeIssue);
  EventId completion = makeEvent(0, PlannedEventKind::Completion);
  EventId laterStage = makeEvent(1, PlannedEventKind::BufferReady);
  input.events = {
      {issue,
       CardId(0),
       TileId(0),
       {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
      {completion, CardId(0), TileId(0), {}},
      {laterStage, CardId(0), TileId(0), {}}};
  input.hardDependencies = {
      {issue, completion, EventDependencyReason::Completion}};
  input.completionObligations = {
      {issue, completion, CompletionProtocol::NCCParticipant, 0}};
  OccurrenceRelationId recurrence{
      TraversalScopeId{RegionExecutionId{makeExecution(0)},
                       TopLevelWorkPieceId{0}},
      {8}};
  PipelineScopeId scope{{issue, completion, laterStage}, {recurrence}};
  PipelinedExecutionStructure pipeline;
  pipeline.scope = scope;
  pipeline.recurrence = recurrence;
  pipeline.iteration = {0, 1, 7, 0};
  pipeline.eventStages = {
      {issue, StageId(0)}, {completion, StageId(0)}, {laterStage, StageId(1)}};
  input.structure.scopes.push_back(std::move(pipeline));
  input.components = {{scope.events}};

  ScheduleDomainResult result = buildScheduleDomain(input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  ScheduleSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(first.getPlan());
  ASSERT_EQ(first.getPlan()->completionPlacements.size(), 1u);
  EXPECT_EQ(first.getPlan()->completionPlacements.front().boundary.after,
            completion);
}

TEST(ScheduleDomainTest, BufferReleasePreventsSameWorkerCompletionElision) {
  ScheduleDomainInput input = makeInput(0, /*workerCapable=*/false);
  input.events.clear();
  input.structure.scopes.clear();
  StorageObjectId object = input.buffers.storageObjects.front().id;
  EventId firstIssue = makeEvent(70, PlannedEventKind::ComputeIssue);
  EventId firstCompletion = makeEvent(70, PlannedEventKind::Completion);
  EventId release{BufferEventAction{object, object},
                  PlannedEventKind::BufferRelease};
  EventId secondIssue = makeEvent(71, PlannedEventKind::ComputeIssue);
  EventId secondCompletion = makeEvent(71, PlannedEventKind::Completion);
  input.events = {
      {firstIssue,
       CardId(0),
       TileId(0),
       {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
      {firstCompletion, CardId(0), TileId(0), {}},
      {release, CardId(0), TileId(0), {}},
      {secondIssue,
       CardId(0),
       TileId(0),
       {NCCWorker::Worker0, NCCWorker::Worker1, NCCWorker::Worker2}},
      {secondCompletion, CardId(0), TileId(0), {}}};
  input.hardDependencies = {
      {firstIssue, firstCompletion, EventDependencyReason::Completion},
      {firstCompletion, release, EventDependencyReason::BufferLifetime},
      {release, secondIssue, EventDependencyReason::BufferLifetime},
      {secondIssue, secondCompletion, EventDependencyReason::Completion}};
  input.completionObligations = {
      {firstIssue, firstCompletion, CompletionProtocol::NCCParticipant, 0},
      {secondIssue, secondCompletion, CompletionProtocol::NCCParticipant, 0}};
  const PhysicalVersionId earlier = std::get<PhysicalVersionId>(object.origin);
  const PhysicalVersionId later{ExecutionResultValueId{makeExecution(71), 0}};
  input.buffers.versionBindings.push_back(
      {later, object, StorageBindingKind::Reuse});
  input.buffers.orderRequirements.push_back(
      {earlier, later, BufferOrderKind::ReuseAfterCompletion});
  SPMRangeResource range{object, {{{0, 0, 0}, {2, 1025, 128}}}};
  input.resourceUses = {{firstIssue, range, ResourceUseMode::Write,
                         ResourceIntervalKind::IssueToCompletion,
                         firstCompletion, ResourceKnowledge::Exact},
                        {secondIssue, range, ResourceUseMode::Read,
                         ResourceIntervalKind::IssueToCompletion,
                         secondCompletion, ResourceKnowledge::Exact}};
  PipelineScopeId scope{
      {firstIssue, firstCompletion, release, secondIssue, secondCompletion},
      {}};
  input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  input.components = {{scope.events}};

  ScheduleDomainResult result = buildScheduleDomain(input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  ScheduleSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(first.getPlan());
  auto placement = llvm::find_if(first.getPlan()->completionPlacements,
                                 [&](const CompletionPlacement &candidate) {
                                   return candidate.issue == firstIssue;
                                 });
  ASSERT_NE(placement, first.getPlan()->completionPlacements.end());
  EXPECT_EQ(placement->participantMask, 1u);
}

TEST(ScheduleDomainTest, FixedGenerationSlotLifetimeAndFailuresStayTyped) {
  ScheduleDomainInput input = makeInput(2, /*workerCapable=*/false, {{0, 1}});
  StorageObjectId object = input.buffers.storageObjects.front().id;
  OccurrenceRelationId occurrence{
      TraversalScopeId{RegionExecutionId{makeExecution(0)},
                       TopLevelWorkPieceId{0}},
      {2, 8, 1}};
  PipelineIterationClass iteration{1, 1, 7, 0};
  PipelineScopeId scope = getPipelineScope(input.structure.scopes.front());
  scope.recurrences = {occurrence};
  PipelinedExecutionStructure pipeline;
  pipeline.scope = scope;
  pipeline.recurrence = occurrence;
  pipeline.iteration = iteration;
  pipeline.eventStages = {{input.events[0].id, StageId(0)},
                          {input.events[1].id, StageId(1)}};
  pipeline.dependences = {{input.events[0].id, input.events[1].id, 0,
                           PipelineDependenceKind::DataReady}};
  input.structure.scopes = {pipeline};
  SlotFamilyId family{{object}};
  input.buffers.slotFamilies.push_back({family, occurrence, 3, {1}});
  input.slotLifetimes.push_back({family,
                                 occurrence,
                                 iteration,
                                 {input.events[0].id},
                                 {input.events[1].id},
                                 1,
                                 2,
                                 8});
  ScheduleDomainResult result = buildScheduleDomain(input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  ScheduleSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_NE(first.getPlan(), nullptr);
  EXPECT_TRUE(result.domain->isForGeneration(input.structure, input.buffers));
  BufferPlan stale = input.buffers;
  stale.slotFamilies.front().multiplicity = 1;
  EXPECT_FALSE(result.domain->isForGeneration(input.structure, stale));
  ClosedSchedulePlan stalePlan = *first.getPlan();
  stalePlan.buffers = stale;
  EXPECT_FALSE(result.domain->contains(stalePlan));

  ScheduleDomainInput staleInput = input;
  staleInput.buffers = stale;
  ScheduleDomainResult staleResult = buildScheduleDomain(staleInput);
  ASSERT_FALSE(staleResult.succeeded());
  ASSERT_TRUE(staleResult.failure);
  EXPECT_EQ(staleResult.failure->kind,
            ScheduleDomainFailureKind::BrokenContract);

  ScheduleDomainInput staleIteration = input;
  staleIteration.slotLifetimes.front().iteration.tailCount = 1;
  ScheduleDomainResult staleIterationResult =
      buildScheduleDomain(staleIteration);
  ASSERT_FALSE(staleIterationResult.succeeded());
  ASSERT_TRUE(staleIterationResult.failure);
  EXPECT_EQ(staleIterationResult.failure->kind,
            ScheduleDomainFailureKind::BrokenContract);

  ScheduleDomainInput cyclic = makeInput(2, false, {{0, 1}, {1, 0}});
  ScheduleDomainResult cycleResult = buildScheduleDomain(cyclic);
  ASSERT_FALSE(cycleResult.succeeded());
  ASSERT_TRUE(cycleResult.failure);
  EXPECT_EQ(cycleResult.failure->kind,
            ScheduleDomainFailureKind::ExactRejection);

  ScheduleDomainInput unknown = makeInput(2, false, {{0, 1}});
  unknown.completionObligations.push_back({unknown.events[0].id,
                                           unknown.events[1].id,
                                           CompletionProtocol::Unknown, 0});
  ScheduleDomainResult unknownResult = buildScheduleDomain(unknown);
  ASSERT_FALSE(unknownResult.succeeded());
  ASSERT_TRUE(unknownResult.failure);
  EXPECT_EQ(unknownResult.failure->kind,
            ScheduleDomainFailureKind::BrokenContract);

  ScheduleDomainInput missingResource = makeInput(2, false);
  missingResource.orderChoices.push_back(
      {DirectDTESenderResource{TileId(0)},
       {missingResource.events[0].id, missingResource.events[1].id}});
  ScheduleDomainResult missingResourceResult =
      buildScheduleDomain(missingResource);
  ASSERT_FALSE(missingResourceResult.succeeded());
  ASSERT_TRUE(missingResourceResult.failure);
  EXPECT_EQ(missingResourceResult.failure->kind,
            ScheduleDomainFailureKind::BrokenContract);

  ScheduleDomainLimits limits;
  limits.maxSuccessorSteps = 1;
  ScheduleDomainInput chain =
      makeInput(7, false, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}});
  ScheduleDomainResult limited = buildScheduleDomain(chain, limits);
  ASSERT_TRUE(limited.succeeded());
  ScheduleSuccessor only = limited.domain->getFirstPlan();
  ASSERT_NE(only.getCursor(), nullptr);
  EXPECT_EQ(limited.domain->getNextPlan(*only.getCursor()).getKind(),
            ScheduleSuccessorKind::Indeterminate);
}

ScheduleDomainInput makeReceiverFSMInput(unsigned count, bool forceOverlap) {
  ScheduleDomainInput input;
  PhysicalVersionId version{ExecutionResultValueId{makeExecution(500), 0}};
  StorageObjectId object{StorageObjectOrigin{version}};
  input.buffers.storageObjects.push_back({object, TileId(0)});
  input.buffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  PipelineScopeId scope;
  std::vector<EventId> issues;
  std::vector<EventId> completions;
  std::set<EventDependency> dependencies;
  for (unsigned index = 0; index < count; ++index) {
    EventId issue = makeEvent(600 + index, PlannedEventKind::MovementIssue);
    EventId completion = makeEvent(600 + index, PlannedEventKind::Completion);
    issues.push_back(issue);
    completions.push_back(completion);
    scope.events.push_back(issue);
    scope.events.push_back(completion);
    input.events.push_back({issue, CardId(0), TileId(0), {}});
    input.events.push_back({completion, CardId(0), TileId(0), {}});
    dependencies.insert({issue, completion, EventDependencyReason::Completion});
    input.completionObligations.push_back(
        {issue, completion, CompletionProtocol::DirectDTE, 0});
    input.resourceUses.push_back({issue, DTEReceiverFSMResource{TileId(0)},
                                  ResourceUseMode::CapacityUnits,
                                  ResourceIntervalKind::IssueToCompletion,
                                  completion, ResourceKnowledge::Exact});
  }
  if (forceOverlap)
    for (const EventId &issue : issues)
      for (const EventId &completion : completions)
        dependencies.insert(
            {issue, completion, EventDependencyReason::Completion});
  input.hardDependencies.assign(dependencies.begin(), dependencies.end());
  llvm::sort(scope.events);
  input.structure.scopes.push_back(SerializedExecutionStructure{scope});
  input.components.push_back({scope.events});
  return input;
}

TEST(ScheduleDomainTest,
     ReceiverFSMLiveIntervalsUseFourCanonicalLanesAndRejectFiveOverlap) {
  ScheduleDomainResult four =
      buildScheduleDomain(makeReceiverFSMInput(4, /*forceOverlap=*/true));
  ASSERT_TRUE(four.succeeded()) << (four.failure ? four.failure->detail : "");
  ScheduleSuccessor fourPlan = four.domain->getFirstPlan();
  ASSERT_EQ(fourPlan.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(fourPlan.getPlan());
  std::set<uint32_t> lanes;
  for (const EventResourceBinding &binding :
       fourPlan.getPlan()->resourceBindings)
    if (std::holds_alternative<DTEReceiverFSMResource>(
            binding.instance.resource))
      lanes.insert(binding.instance.lane);
  EXPECT_EQ(lanes, (std::set<uint32_t>{0, 1, 2, 3}));

  ScheduleDomainResult five =
      buildScheduleDomain(makeReceiverFSMInput(5, /*forceOverlap=*/true));
  ASSERT_FALSE(five.succeeded());
  ASSERT_TRUE(five.failure);
  EXPECT_EQ(five.failure->kind, ScheduleDomainFailureKind::ExactRejection);

  ScheduleDomainResult reusable =
      buildScheduleDomain(makeReceiverFSMInput(5, /*forceOverlap=*/false));
  ASSERT_TRUE(reusable.succeeded())
      << (reusable.failure ? reusable.failure->detail : "");
  ScheduleSuccessor reusablePlan = reusable.domain->getFirstPlan();
  ASSERT_EQ(reusablePlan.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_TRUE(reusablePlan.getPlan());
  std::map<EventId, uint32_t> reusableLanes;
  for (const EventResourceBinding &binding :
       reusablePlan.getPlan()->resourceBindings)
    if (std::holds_alternative<DTEReceiverFSMResource>(
            binding.instance.resource)) {
      EXPECT_LT(binding.instance.lane, 4u);
      reusableLanes.emplace(binding.event, binding.instance.lane);
    }
  ASSERT_EQ(reusablePlan.getPlan()->controlOrders.size(), 1u);
  const std::vector<EventId> &order =
      reusablePlan.getPlan()->controlOrders.front().events;
  auto position = [&](const EventId &event) {
    return static_cast<size_t>(
        std::distance(order.begin(), llvm::find(order, event)));
  };
  for (const CompletionPlacement &placement :
       reusablePlan.getPlan()->completionPlacements) {
    auto lane = reusableLanes.find(placement.issue);
    ASSERT_NE(lane, reusableLanes.end());
    for (size_t index = position(placement.completion) + 1;
         index < order.size(); ++index) {
      auto nextLane = reusableLanes.find(order[index]);
      if (nextLane == reusableLanes.end() || nextLane->second != lane->second)
        continue;
      EXPECT_LT(position(placement.boundary.after), index);
      break;
    }
  }
}

} // namespace
