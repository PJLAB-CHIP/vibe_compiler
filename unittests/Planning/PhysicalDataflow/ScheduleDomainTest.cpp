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
  ResourceKey ddr = CardDDRResource{CardId(0)};
  input.orderChoices.push_back({ddr, {input.events[0].id, input.events[1].id}});
  for (const PlannedEvent &event : input.events)
    input.resourceUses.push_back({event.id, ddr, ResourceUseMode::Exclusive,
                                  ResourceIntervalKind::IssueToCompletion,
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
    EXPECT_EQ(ncc->boundary.after, completion);
    EXPECT_EQ(dte->boundary.after, dteCompletion);
  }
}

TEST(ScheduleDomainTest, FixedGenerationSlotLifetimeAndFailuresStayTyped) {
  ScheduleDomainInput input = makeInput(2, /*workerCapable=*/false, {{0, 1}});
  StorageObjectId object = input.buffers.storageObjects.front().id;
  OccurrenceRelationId occurrence{
      TraversalScopeId{RegionExecutionId{makeExecution(0)},
                       TopLevelWorkPieceId{0}},
      {2, 8, 1}};
  SlotFamilyId family{{object}};
  input.buffers.slotFamilies.push_back({family, occurrence, 3, {0, 1}});
  input.slotLifetimes.push_back({family,
                                 occurrence,
                                 {input.events[0].id},
                                 {input.events[1].id},
                                 2,
                                 3,
                                 16});
  ScheduleDomainResult result = buildScheduleDomain(input);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  ScheduleSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), ScheduleSuccessorKind::Plan);
  ASSERT_NE(first.getPlan(), nullptr);
  EXPECT_TRUE(result.domain->isForGeneration(input.structure, input.buffers));
  BufferPlan stale = input.buffers;
  stale.slotFamilies.front().multiplicity = 2;
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

} // namespace
