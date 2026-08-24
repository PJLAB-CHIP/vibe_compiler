//===- StructureSpecificStorageDomainTest.cpp -------------------------===//

#include "Wafer/Planning/PhysicalDataflow/StructureSpecificStorageDomain.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
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

PhysicalVersionId makeVersion(uint32_t anchor) {
  return PhysicalVersionId{ExecutionResultValueId{makeExecution(anchor), 0}};
}

struct StorageProblem {
  BufferPlan initial;
  ExecutionStructurePlan structure;
  std::vector<PlannedEvent> events;
};

StorageProblem makeProblem(unsigned objectCount, uint64_t sequenceOccurrences,
                           uint32_t releaseStage = 2,
                           bool activeBatchAxis = true,
                           uint64_t tailCount = 0) {
  StorageProblem problem;
  OccurrenceRelationId occurrence;
  occurrence.scope = TraversalScopeId{RegionExecutionId{makeExecution(100)},
                                      TopLevelWorkPieceId{0}};
  occurrence.axisOccurrences =
      activeBatchAxis
          ? llvm::SmallVector<uint64_t, 4>{2, sequenceOccurrences, 1}
          : llvm::SmallVector<uint64_t, 4>{sequenceOccurrences};
  PipelineScopeId scope;
  scope.recurrences.push_back(occurrence);
  PipelinedExecutionStructure pipeline;
  pipeline.recurrence = occurrence;
  const uint32_t recurrenceAxis = activeBatchAxis ? 1 : 0;
  pipeline.iteration = {recurrenceAxis, 1, sequenceOccurrences - 1 - tailCount,
                        tailCount};
  pipeline.launchDistance = 1;
  for (unsigned index = 0; index < objectCount; ++index) {
    PhysicalVersionId version = makeVersion(index);
    StorageObjectId object{StorageObjectOrigin{version}};
    problem.initial.storageObjects.push_back(
        {object, TileId(static_cast<int64_t>(index % 2))});
    problem.initial.versionBindings.push_back(
        {version, object, StorageBindingKind::Fresh});
    BufferEventAction action{object, object};
    EventId ready{action, PlannedEventKind::BufferReady};
    EventId release{action, PlannedEventKind::BufferRelease};
    scope.events.push_back(ready);
    scope.events.push_back(release);
    pipeline.eventStages.push_back({ready, StageId(0)});
    pipeline.eventStages.push_back({release, StageId(releaseStage)});
    problem.events.push_back(
        {ready, CardId(0), TileId(static_cast<int64_t>(index % 2)), {}});
    problem.events.push_back(
        {release, CardId(0), TileId(static_cast<int64_t>(index % 2)), {}});
  }
  llvm::sort(scope.events);
  llvm::sort(pipeline.eventStages);
  pipeline.scope = scope;
  problem.structure.scopes.push_back(std::move(pipeline));
  return problem;
}

std::optional<std::vector<BufferPlan>>
enumerate(const StructureSpecificStorageDomain &domain, size_t limit = 100000) {
  std::vector<BufferPlan> plans;
  StructureSpecificStorageSuccessor next = domain.getFirstPlan();
  while (next.getKind() == StructureSpecificStorageSuccessorKind::Plan) {
    if (!next.getPlan() || !next.getCursor() ||
        !domain.contains(*next.getPlan()) || plans.size() >= limit)
      return std::nullopt;
    plans.push_back(*next.getPlan());
    StructureSpecificStorageCursor cursor = *next.getCursor();
    next = domain.getNextPlan(cursor);
  }
  return next.getKind() == StructureSpecificStorageSuccessorKind::End
             ? std::optional<std::vector<BufferPlan>>(std::move(plans))
             : std::nullopt;
}

TEST(StructureSpecificStorageDomainTest,
     AlignedAndRaggedLiveDistanceEnumeratesExactMultiplicityAndRotation) {
  for (uint64_t extent : {uint64_t{1024}, uint64_t{1025}, uint64_t{1031}}) {
    SCOPED_TRACE(extent);
    const uint64_t sequenceOccurrences = (extent + 127) / 128;
    StorageProblem problem = makeProblem(1, sequenceOccurrences);
    auto &pipeline =
        std::get<PipelinedExecutionStructure>(problem.structure.scopes.front());
    pipeline.iteration.tailCount = extent % 128 == 0 ? 0 : 1;
    pipeline.iteration.steadyTripCount =
        sequenceOccurrences - 1 - pipeline.iteration.tailCount;
    StructureSpecificStorageDomainResult result =
        buildStructureSpecificStorageDomain(problem.structure, problem.initial,
                                            problem.events);
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    EXPECT_TRUE(result.domain->isForStructure(problem.structure));
    ExecutionStructurePlan sibling = problem.structure;
    ++std::get<PipelinedExecutionStructure>(sibling.scopes.front())
          .launchDistance;
    EXPECT_FALSE(result.domain->isForStructure(sibling));
    ASSERT_EQ(result.domain->getLifetimeRequirements().size(), 1u);
    const SlotLifetimeRequirement &lifetime =
        result.domain->getLifetimeRequirements().front();
    EXPECT_EQ(lifetime.iteration, pipeline.iteration);
    EXPECT_EQ(lifetime.liveStageDistance, 2u);
    EXPECT_EQ(lifetime.minimumMultiplicity, 3u);
    EXPECT_EQ(lifetime.maximumMultiplicity, sequenceOccurrences);
    auto plans = enumerate(*result.domain);
    ASSERT_TRUE(plans);
    EXPECT_EQ(plans->size(), size_t(lifetime.maximumMultiplicity - 3 + 1));
    for (const BufferPlan &plan : *plans) {
      ASSERT_EQ(plan.slotFamilies.size(), 1u);
      const SlotFamilyPlan &family = plan.slotFamilies.front();
      EXPECT_FALSE(family.occurrence.axisOccurrences.empty());
      EXPECT_GE(family.multiplicity, 3u);
      EXPECT_EQ(family.rotationIterators, (llvm::SmallVector<uint32_t, 4>{1}));
    }
    EXPECT_FALSE(result.domain->contains(problem.initial));
  }
}

TEST(StructureSpecificStorageDomainTest,
     SerializedClearsStalePreStructureFamiliesAndKeepsBindings) {
  StorageProblem problem = makeProblem(1, 8);
  const PipelineScopeId scope =
      getPipelineScope(problem.structure.scopes.front());
  problem.structure.scopes = {SerializedExecutionStructure{scope}};
  problem.initial.slotFamilies.push_back(
      {SlotFamilyId{{problem.initial.storageObjects.front().id}},
       scope.recurrences.front(),
       5,
       {0, 1}});
  BufferPlan expected = problem.initial;
  expected.slotFamilies.clear();
  StructureSpecificStorageDomainResult result =
      buildStructureSpecificStorageDomain(problem.structure, problem.initial,
                                          problem.events);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  StructureSpecificStorageSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), StructureSpecificStorageSuccessorKind::Plan);
  ASSERT_NE(first.getPlan(), nullptr);
  EXPECT_EQ(*first.getPlan(), expected);
  EXPECT_TRUE(result.domain->getLifetimeRequirements().empty());
  ASSERT_NE(first.getCursor(), nullptr);
  EXPECT_EQ(result.domain->getNextPlan(*first.getCursor()).getKind(),
            StructureSpecificStorageSuccessorKind::End);
}

TEST(StructureSpecificStorageDomainTest,
     SharedAliasObjectAndGatherStagingEachHaveOnePhysicalFamily) {
  StorageProblem problem = makeProblem(1, 8);
  PhysicalVersionId alias = makeVersion(50);
  StorageObjectId shared = problem.initial.storageObjects.front().id;
  problem.initial.versionBindings.push_back(
      {alias, shared, StorageBindingKind::IdentityAlias});
  BufferEventAction aliasAction{StorageObjectId{StorageObjectOrigin{alias}},
                                shared};
  EventId aliasReady{aliasAction, PlannedEventKind::BufferReady};
  EventId aliasRelease{aliasAction, PlannedEventKind::BufferRelease};

  ReductionGatherId gather;
  StorageObjectId staging{
      StorageObjectOrigin{ReductionGatherStagingId{gather}}};
  problem.initial.storageObjects.push_back({staging, TileId(0)});
  problem.initial.gatherStagingBindings.push_back({gather, staging});
  BufferEventAction stagingAction{staging, staging};
  EventId stagingReady{stagingAction, PlannedEventKind::BufferReady};
  EventId stagingRelease{stagingAction, PlannedEventKind::BufferRelease};

  auto &selected =
      std::get<PipelinedExecutionStructure>(problem.structure.scopes.front());
  for (const auto &[event, stage] :
       {std::pair<EventId, uint32_t>{aliasReady, 0},
        {aliasRelease, 2},
        {stagingReady, 0},
        {stagingRelease, 2}}) {
    selected.scope.events.push_back(event);
    selected.eventStages.push_back({event, StageId(stage)});
    problem.events.push_back({event, CardId(0), TileId(0), {}});
  }
  llvm::sort(selected.scope.events);
  llvm::sort(selected.eventStages);
  StructureSpecificStorageDomainResult result =
      buildStructureSpecificStorageDomain(problem.structure, problem.initial,
                                          problem.events);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  ASSERT_EQ(result.domain->getLifetimeRequirements().size(), 2u);
  std::set<StorageObjectId> familyObjects;
  for (const SlotLifetimeRequirement &lifetime :
       result.domain->getLifetimeRequirements()) {
    ASSERT_EQ(lifetime.family.objects.size(), 1u);
    familyObjects.insert(lifetime.family.objects.front());
  }
  EXPECT_EQ(familyObjects, (std::set<StorageObjectId>{shared, staging}));
}

TEST(StructureSpecificStorageDomainTest,
     TwoToSixObjectProductsMatchIndependentMultiplicityReference) {
  for (unsigned objects = 2; objects <= 6; ++objects) {
    SCOPED_TRACE(objects);
    StorageProblem problem =
        makeProblem(objects, /*sequenceOccurrences=*/3,
                    /*releaseStage=*/1, /*activeBatchAxis=*/false);
    StructureSpecificStorageDomainResult result =
        buildStructureSpecificStorageDomain(problem.structure, problem.initial,
                                            problem.events);
    ASSERT_TRUE(result.succeeded())
        << (result.failure ? result.failure->detail : "");
    auto plans = enumerate(*result.domain);
    ASSERT_TRUE(plans);
    size_t expected = 1;
    for (unsigned index = 0; index < objects; ++index)
      expected *= 2; // Tiny oracle: multiplicities 2..3, one rotation.
    EXPECT_EQ(plans->size(), expected);
  }
}

TEST(StructureSpecificStorageDomainTest,
     MissingLifetimeEarlyReleaseAndInvalidIterationRemainTyped) {
  StorageProblem missing = makeProblem(1, 8);
  auto &missingPipeline =
      std::get<PipelinedExecutionStructure>(missing.structure.scopes.front());
  EventId release = missingPipeline.scope.events.back();
  llvm::erase(missingPipeline.scope.events, release);
  llvm::erase_if(missingPipeline.eventStages, [&](const auto &assignment) {
    return assignment.event == release;
  });
  llvm::erase_if(missing.events,
                 [&](const auto &event) { return event.id == release; });
  StructureSpecificStorageDomainResult missingResult =
      buildStructureSpecificStorageDomain(missing.structure, missing.initial,
                                          missing.events);
  ASSERT_FALSE(missingResult.succeeded());
  ASSERT_TRUE(missingResult.failure);
  EXPECT_EQ(missingResult.failure->kind,
            StructureSpecificStorageFailureKind::BrokenContract);

  StorageProblem early = makeProblem(1, 8, /*releaseStage=*/0);
  auto &earlyPipeline =
      std::get<PipelinedExecutionStructure>(early.structure.scopes.front());
  earlyPipeline.eventStages.front().stage = StageId(1);
  llvm::sort(earlyPipeline.eventStages);
  StructureSpecificStorageDomainResult earlyResult =
      buildStructureSpecificStorageDomain(early.structure, early.initial,
                                          early.events);
  ASSERT_FALSE(earlyResult.succeeded());
  ASSERT_TRUE(earlyResult.failure);
  EXPECT_EQ(earlyResult.failure->kind,
            StructureSpecificStorageFailureKind::ExactRejection);

  StorageProblem wide = makeProblem(1, 3, 1, false);
  auto &widePipeline =
      std::get<PipelinedExecutionStructure>(wide.structure.scopes.front());
  widePipeline.iteration.recurrenceAxis = 4;
  StructureSpecificStorageDomainResult limited =
      buildStructureSpecificStorageDomain(wide.structure, wide.initial,
                                          wide.events);
  ASSERT_FALSE(limited.succeeded());
  ASSERT_TRUE(limited.failure);
  EXPECT_EQ(limited.failure->kind,
            StructureSpecificStorageFailureKind::BrokenContract);
}

} // namespace
