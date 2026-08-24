//===- ActualResultControllerTest.cpp --------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/ActualResultController.h"

#include "Wafer/Target/Core/RuntimeLaunchContract.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
#include <set>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

ExecutionInstanceId makeExecution(uint32_t anchor) {
  SemanticRootKey root;
  root.anchorIndex = anchor;
  RootRegionWorkId work{root, TileId(0)};
  LogicalShardId shard{root, {anchor}};
  return ExecutionInstanceId{RequiredRootExecution{work, shard}};
}

struct KeyParts {
  SpatialPlan spatial;
  RegionPlan regions;
  TemporalPlan temporal;
  RepresentationPlan representations;
  MovementPlan movement;
  BufferPlan initialBuffers;
  ExecutionStructurePlan structure;
  BufferPlan buffers;
  ClosedSchedulePlan schedule;
};

KeyParts makeKeyParts(uint32_t anchor, int64_t extent = 1024) {
  KeyParts parts;
  ExecutionInstanceId execution = makeExecution(anchor);
  const auto &rootExecution = std::get<RequiredRootExecution>(execution.source);

  NodeSpatialPlan spatial;
  spatial.root = rootExecution.work.root;
  spatial.axes = {{0, IteratorPartitionScheme::BalancedParts, 1},
                  {1, IteratorPartitionScheme::UniformExtent, 128},
                  {2, IteratorPartitionScheme::UniformExtent, 128}};
  spatial.embedding.push_back(TileId(0));
  parts.spatial.nodes.push_back(std::move(spatial));

  RegionGroupPlan region;
  region.tile = TileId(0);
  region.mandatoryRoots.push_back(rootExecution.work);
  region.executions.push_back({execution, ExecutionInstancePlan::TopLevel{}});
  parts.regions.groups.push_back(std::move(region));

  parts.temporal.scopes.push_back(
      {TraversalScopeId{RegionExecutionId(execution), TopLevelWorkPieceId{0}},
       {2, extent, 128},
       {0, 1, 2}});

  PhysicalVersionId version{ExecutionResultValueId{execution, 0}};
  parts.representations.logicalValues.push_back(
      {version.logicalValue, version});
  parts.representations.physicalVersions.push_back(
      {version, MemLayout::Tensor});
  ResultPublicationId publication{ExecutionResultValueId{execution, 0}};
  parts.movement.publications.push_back({publication, version});

  StorageObjectId object{StorageObjectOrigin(version)};
  parts.initialBuffers.storageObjects.push_back({object, TileId(0)});
  parts.initialBuffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  parts.buffers = parts.initialBuffers;

  EventId event{ExecutionEventAction{execution},
                PlannedEventKind::ComputeIssue};
  PipelineScopeId scope{{event}, {}};
  parts.structure.scopes.push_back(SerializedExecutionStructure{scope});
  parts.schedule.structure = parts.structure;
  parts.schedule.buffers = parts.buffers;
  parts.schedule.controlOrders.push_back(
      {TileControlScope{TileId(0), scope}, {event}});
  return parts;
}

CompleteCandidateKey closeKey(KeyParts parts) {
  std::string failureReason;
  auto key = CompleteCandidateKey::createFromValidatedPlans(
      std::move(parts.spatial), std::move(parts.regions),
      std::move(parts.temporal), std::move(parts.representations),
      std::move(parts.movement), std::move(parts.initialBuffers),
      std::move(parts.structure), std::move(parts.buffers),
      std::move(parts.schedule), &failureReason);
  EXPECT_TRUE(mlir::succeeded(key)) << failureReason;
  return std::move(*key);
}

CompleteCandidateKey makeKey(uint32_t anchor, int64_t extent = 1024) {
  return closeKey(makeKeyParts(anchor, extent));
}

ActualResultController makeController(
    uint64_t credits, std::optional<SearchCostCohort> cohort = {},
    ExactRejectionCachePolicy cache = ExactRejectionCachePolicy::Enabled) {
  return ActualResultController(
      ActualResultControllerOptions{credits, std::move(cohort), cache});
}

RuntimeLaunchContract makeLaunch() {
  return llvm::cantFail(RuntimeLaunchContract::createKernel(
      KernelLaunchForm::Grid, KernelEntryABI::TileMajorPointerTable,
      {RuntimeLaunchPhaseRole::Main}));
}

FullFeasibilityResult accepted(uint64_t instructions, bool known = true) {
  CardInstructionProgramCost cost;
  cost.aggregateInstructionCount.value = instructions;
  cost.aggregateDDRReadBytes.value = instructions * 2;
  cost.aggregateDDRWriteBytes.value = instructions * 3;
  cost.minimumHopLinkByteDemand.value = instructions * 4;
  if (!known)
    cost.aggregateInstructionCount.knowledge =
        ScheduleCostKnowledge::Unavailable;
  CardExecutableCompilationResult compilation;
  compilation.status = CardExecutableCompilationStatus::Accepted;
  compilation.executable.emplace(std::vector<compiler::TileExecutable>{},
                                 makeLaunch(), std::move(cost));
  FullFeasibilityResult result;
  result.status = FullFeasibilityStatus::Accepted;
  result.compilation.emplace(std::move(compilation));
  return result;
}

FullFeasibilityResult exactRejected(uint32_t rootAnchor,
                                    bool spmCapacity = false) {
  FullFeasibilityResult result;
  result.status = FullFeasibilityStatus::ExactRejection;
  CardExecutableCompilationResult compilation;
  compilation.status = CardExecutableCompilationStatus::ProvenExactRejection;
  if (spmCapacity) {
    SemanticRootKey root;
    root.anchorIndex = rootAnchor;
    result.causalRoots.push_back(root);
    CardExecutableTileFailure tile;
    tile.tileId = TileId(0);
    tile.memoryPlanning.kind = TileMemoryPlanningFailureKind::SPMAllocation;
    tile.memoryPlanning.spmPlanningFailureKind =
        SPMMemoryPlanningFailureKind::CapacityOverflow;
    compilation.tileFailures.push_back(std::move(tile));
  }
  result.compilation.emplace(std::move(compilation));
  return result;
}

TEST(ActualResultControllerTest,
     CompleteKeyUsesEveryAxisAndRejectsStaleGenerations) {
  // This is a value-schema test: each axis' domain tests own plan legality.
  // CompleteCandidateKey joins those already-validated values and must retain
  // every field while independently rejecting stale K/I/J generation joins.
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    KeyParts baseParts = makeKeyParts(0, extent);
    CompleteCandidateKey base = closeKey(baseParts);
    std::set<CompleteCandidateKey> variants;
    variants.insert(base);

    KeyParts spatial = baseParts;
    spatial.spatial.nodes.front().axes.front().parameter = 2;
    variants.insert(closeKey(std::move(spatial)));
    KeyParts regions = baseParts;
    regions.regions.groups.front().tile = TileId(1);
    variants.insert(closeKey(std::move(regions)));
    KeyParts temporal = baseParts;
    --temporal.temporal.scopes.front().iteratorTileSizes[1];
    variants.insert(closeKey(std::move(temporal)));
    KeyParts representations = baseParts;
    representations.representations.physicalVersions.front().encoding =
        MemLayout::NCx;
    variants.insert(closeKey(std::move(representations)));
    KeyParts movement = baseParts;
    movement.movement.publications.clear();
    variants.insert(closeKey(std::move(movement)));
    KeyParts initial = baseParts;
    initial.initialBuffers.slotFamilies.push_back(
        {SlotFamilyId{{initial.initialBuffers.storageObjects.front().id}},
         OccurrenceRelationId{},
         1,
         {}});
    variants.insert(closeKey(std::move(initial)));
    KeyParts structure = baseParts;
    structure.structure.scopes.push_back(structure.structure.scopes.front());
    structure.schedule.structure = structure.structure;
    variants.insert(closeKey(std::move(structure)));
    KeyParts buffers = baseParts;
    buffers.buffers.slotFamilies.push_back(
        {SlotFamilyId{{buffers.buffers.storageObjects.front().id}},
         OccurrenceRelationId{},
         2,
         {0}});
    buffers.schedule.buffers = buffers.buffers;
    variants.insert(closeKey(std::move(buffers)));
    KeyParts schedule = baseParts;
    schedule.schedule.workerBindings.push_back(
        {schedule.schedule.controlOrders.front().events.front(),
         NCCWorker::Worker1});
    variants.insert(closeKey(std::move(schedule)));
    EXPECT_EQ(variants.size(), 10u);

    KeyParts stale = baseParts;
    stale.schedule.buffers.slotFamilies.push_back(
        {SlotFamilyId{{stale.buffers.storageObjects.front().id}},
         OccurrenceRelationId{},
         2,
         {0}});
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(CompleteCandidateKey::createFromValidatedPlans(
        std::move(stale.spatial), std::move(stale.regions),
        std::move(stale.temporal), std::move(stale.representations),
        std::move(stale.movement), std::move(stale.initialBuffers),
        std::move(stale.structure), std::move(stale.buffers),
        std::move(stale.schedule), &failureReason)));
    EXPECT_EQ(failureReason,
              "complete candidate key has a stale schedule generation");
  }
}

TEST(ActualResultControllerTest,
     ExplicitCohortDerivesKnownUnknownOverflowAndStrictBound) {
  std::string failureReason;
  auto cohort = SearchCostCohort::create(1, 2, 3, 4, &failureReason);
  ASSERT_TRUE(mlir::succeeded(cohort)) << failureReason;
  EXPECT_TRUE(
      mlir::failed(SearchCostCohort::create(0, 1, 1, 1, &failureReason)));

  CardInstructionProgramCost cost;
  cost.aggregateInstructionCount.value = 1;
  cost.aggregateDDRReadBytes.value = 2;
  cost.aggregateDDRWriteBytes.value = 3;
  cost.minimumHopLinkByteDemand.value = 4;
  SearchObjective known = deriveSearchObjective(cost, *cohort);
  const auto *knownValue = std::get_if<KnownSearchObjective>(&known);
  ASSERT_NE(knownValue, nullptr);
  EXPECT_EQ(knownValue->ticks, 30u);
  EXPECT_TRUE(std::holds_alternative<UnknownSearchObjective>(
      deriveSearchObjective(cost, std::nullopt)));
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Unavailable;
  EXPECT_EQ(
      std::get<UnknownSearchObjective>(deriveSearchObjective(cost, *cohort))
          .reason,
      SearchObjectiveUnknownReason::MetricUnavailable);
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Known;
  cost.aggregateInstructionCount.value = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(
      std::get<UnknownSearchObjective>(deriveSearchObjective(cost, *cohort))
          .reason,
      SearchObjectiveUnknownReason::ArithmeticOverflow);

  ActualResultController controller = makeController(1, *cohort);
  CompleteCandidateKey key = makeKey(0);
  ASSERT_EQ(controller.reserve(key), CandidateReservation::Granted);
  ASSERT_EQ(controller.record(key, accepted(1)),
            CandidateRecordOutcome::Accepted);
  SearchLowerBound equal{key, KnownSearchObjective{30, *cohort}};
  SearchLowerBound worse{key, KnownSearchObjective{31, *cohort}};
  SearchLowerBound unknown{
      key, UnknownSearchObjective{SearchObjectiveUnknownReason::NoCohort}};
  EXPECT_FALSE(controller.canPrune(equal));
  EXPECT_TRUE(controller.canPrune(worse));
  EXPECT_FALSE(controller.canPrune(unknown));
}

TEST(ActualResultControllerTest,
     TwoToSevenAcceptedResultsMatchIndependentWinnerOracleInAnyOrder) {
  // Two-to-seven is intentionally a bounded independent controller oracle;
  // the same mechanism has real-scale 1024/1025 keys in the other tests.
  auto cohort = *SearchCostCohort::create(1, 1, 1, 1);
  for (unsigned count = 2; count <= 7; ++count) {
    SCOPED_TRACE(count);
    ActualResultController controller = makeController(count, cohort);
    struct Record {
      CompleteCandidateKey key;
      uint64_t instructions = 0;
    };
    std::vector<Record> records;
    for (uint32_t index = 0; index < count; ++index)
      records.push_back(
          {makeKey(index, index % 2 ? 1025 : 1024),
           1 + (static_cast<uint64_t>(index) * 5 + 3) % (count + 1)});
    const Record *expected = &records.front();
    for (const Record &record : records)
      if (record.instructions < expected->instructions ||
          (record.instructions == expected->instructions &&
           record.key < expected->key))
        expected = &record;
    CompleteCandidateKey expectedKey = expected->key;
    std::reverse(records.begin(), records.end());
    for (const Record &record : records) {
      ASSERT_EQ(controller.reserve(record.key), CandidateReservation::Granted);
      ASSERT_EQ(controller.record(record.key, accepted(record.instructions)),
                CandidateRecordOutcome::Accepted);
    }
    SearchControllerResult result =
        controller.finish(SearchFrontierStatus::Exhausted);
    ASSERT_TRUE(result.winner);
    EXPECT_EQ(result.winner->key, expectedKey);
    EXPECT_EQ(result.coverage, SearchControllerCoverage::ComparableBest);
    EXPECT_EQ(result.statistics.accepted, count);
  }
}

TEST(ActualResultControllerTest,
     UnknownObjectivesUseCompleteKeyButRemainFeasibleUnranked) {
  ActualResultController controller = makeController(3);
  for (uint32_t index : {2u, 0u, 1u}) {
    CompleteCandidateKey key = makeKey(index);
    ASSERT_EQ(controller.reserve(key), CandidateReservation::Granted);
    ASSERT_EQ(controller.record(key, accepted(10, /*known=*/false)),
              CandidateRecordOutcome::Accepted);
  }
  SearchControllerResult result =
      controller.finish(SearchFrontierStatus::Exhausted);
  ASSERT_TRUE(result.winner);
  EXPECT_EQ(result.winner->key, makeKey(0));
  EXPECT_EQ(result.coverage, SearchControllerCoverage::FeasibleUnranked);
}

TEST(ActualResultControllerTest,
     ExactFeedbackDoesNotMatchACompleteKeyWithTheSameSchedule) {
  KeyParts rejectedParts = makeKeyParts(0, 1025);
  KeyParts siblingParts = rejectedParts;
  siblingParts.representations.physicalVersions.front().encoding =
      MemLayout::NCx;
  CompleteCandidateKey rejected = closeKey(std::move(rejectedParts));
  CompleteCandidateKey sibling = closeKey(std::move(siblingParts));
  ActualResultController controller = makeController(4);
  CompleteCandidateKey unsupported = makeKey(2, 1025);
  CompleteCandidateKey indeterminate = makeKey(3, 1025);
  for (const CompleteCandidateKey *key :
       {&rejected, &sibling, &unsupported, &indeterminate})
    ASSERT_EQ(controller.reserve(*key), CandidateReservation::Granted);
  EXPECT_EQ(controller.record(rejected, exactRejected(0, true)),
            CandidateRecordOutcome::ExactRejection);
  EXPECT_TRUE(controller.isForbidden(rejected));
  EXPECT_FALSE(controller.isForbidden(sibling));
  const ExactCompleteRejection *proof =
      controller.findExactCompleteRejection(rejected);
  ASSERT_NE(proof, nullptr);
  ASSERT_EQ(proof->causalRoots.size(), 1u);
  EXPECT_EQ(proof->causalRoots.front().anchorIndex, 0u);
  EXPECT_EQ(controller.record(sibling, accepted(1)),
            CandidateRecordOutcome::Accepted);
  FullFeasibilityResult unsupportedResult;
  unsupportedResult.status = FullFeasibilityStatus::Unsupported;
  EXPECT_EQ(controller.record(unsupported, std::move(unsupportedResult)),
            CandidateRecordOutcome::Unsupported);
  FullFeasibilityResult indeterminateResult;
  indeterminateResult.status = FullFeasibilityStatus::Indeterminate;
  EXPECT_EQ(controller.record(indeterminate, std::move(indeterminateResult)),
            CandidateRecordOutcome::Indeterminate);
  SearchControllerResult result =
      controller.finish(SearchFrontierStatus::Exhausted);
  ASSERT_TRUE(result.winner);
  EXPECT_EQ(result.coverage, SearchControllerCoverage::FeasiblePartial);
  EXPECT_EQ(result.statistics.exactRejected, 1u);
  EXPECT_EQ(result.statistics.unsupported, 1u);
  EXPECT_EQ(result.statistics.indeterminate, 1u);
}

TEST(ActualResultControllerTest,
     ExactCachePolicyDoesNotChangeAcceptedSetOrWinner) {
  std::vector<CompleteCandidateKey> winners;
  std::vector<SearchControllerCoverage> coverage;
  std::vector<size_t> cacheSizes;
  for (ExactRejectionCachePolicy policy :
       {ExactRejectionCachePolicy::Enabled,
        ExactRejectionCachePolicy::Disabled}) {
    ActualResultController controller = makeController(2, std::nullopt, policy);
    CompleteCandidateKey rejected = makeKey(0, 1024);
    CompleteCandidateKey acceptedKey = makeKey(1, 1025);
    ASSERT_EQ(controller.reserve(rejected), CandidateReservation::Granted);
    ASSERT_EQ(controller.record(rejected, exactRejected(0)),
              CandidateRecordOutcome::ExactRejection);
    ASSERT_EQ(controller.reserve(acceptedKey), CandidateReservation::Granted);
    ASSERT_EQ(controller.record(acceptedKey, accepted(1, false)),
              CandidateRecordOutcome::Accepted);
    SearchControllerResult result =
        controller.finish(SearchFrontierStatus::Exhausted);
    ASSERT_TRUE(result.winner);
    winners.push_back(result.winner->key);
    coverage.push_back(result.coverage);
    cacheSizes.push_back(result.exactCompleteRejections);
  }
  EXPECT_EQ(winners[0], winners[1]);
  EXPECT_EQ(coverage[0], coverage[1]);
  EXPECT_EQ(cacheSizes[0], 1u);
  EXPECT_EQ(cacheSizes[1], 0u);
}

TEST(ActualResultControllerTest,
     ReservationAndMalformedActualResultsFailClosedWithExactCounts) {
  ActualResultController controller = makeController(1);
  CompleteCandidateKey first = makeKey(0);
  CompleteCandidateKey second = makeKey(1);
  ASSERT_EQ(controller.reserve(first), CandidateReservation::Granted);
  EXPECT_EQ(controller.reserve(first), CandidateReservation::Duplicate);
  EXPECT_EQ(controller.reserve(second), CandidateReservation::Exhausted);
  FullFeasibilityResult malformedAccepted;
  malformedAccepted.status = FullFeasibilityStatus::Accepted;
  CardExecutableCompilationResult compilation;
  compilation.status = CardExecutableCompilationStatus::Accepted;
  malformedAccepted.compilation.emplace(std::move(compilation));
  EXPECT_EQ(controller.record(first, std::move(malformedAccepted)),
            CandidateRecordOutcome::CompilerBug);
  EXPECT_EQ(controller.reserve(second), CandidateReservation::Closed);
  SearchControllerResult failed =
      controller.finish(SearchFrontierStatus::Incomplete);
  EXPECT_EQ(failed.coverage, SearchControllerCoverage::Failed);
  EXPECT_EQ(failed.statistics.reserved, 1u);
  EXPECT_EQ(failed.statistics.duplicateReservations, 1u);
  EXPECT_EQ(failed.statistics.exhaustedReservations, 1u);
  EXPECT_EQ(failed.statistics.closedReservations, 1u);
  EXPECT_EQ(failed.statistics.compilerBugs, 1u);

  ActualResultController noFeasible = makeController(2);
  for (uint32_t index = 0; index < 2; ++index) {
    CompleteCandidateKey key = makeKey(index);
    ASSERT_EQ(noFeasible.reserve(key), CandidateReservation::Granted);
    ASSERT_EQ(noFeasible.record(key, exactRejected(index)),
              CandidateRecordOutcome::ExactRejection);
  }
  SearchControllerResult exhausted =
      noFeasible.finish(SearchFrontierStatus::Exhausted);
  EXPECT_FALSE(exhausted.winner);
  EXPECT_EQ(exhausted.coverage, SearchControllerCoverage::NoFeasible);

  ActualResultController typedBug = makeController(1);
  CompleteCandidateKey bugKey = makeKey(4, 1025);
  ASSERT_EQ(typedBug.reserve(bugKey), CandidateReservation::Granted);
  FullFeasibilityResult compilerBug;
  compilerBug.status = FullFeasibilityStatus::CompilerBug;
  EXPECT_EQ(typedBug.record(bugKey, std::move(compilerBug)),
            CandidateRecordOutcome::CompilerBug);
  EXPECT_EQ(typedBug.finish(SearchFrontierStatus::Incomplete).coverage,
            SearchControllerCoverage::Failed);

  ActualResultController unreserved = makeController(1);
  CompleteCandidateKey unreservedKey = makeKey(5, 1025);
  EXPECT_EQ(unreserved.record(unreservedKey, accepted(1)),
            CandidateRecordOutcome::CompilerBug);
  EXPECT_EQ(unreserved.getStatistics().compilerBugs, 1u);

  ActualResultController inFlight = makeController(1);
  CompleteCandidateKey inFlightKey = makeKey(6, 1025);
  ASSERT_EQ(inFlight.reserve(inFlightKey), CandidateReservation::Granted);
  EXPECT_EQ(inFlight.finish(SearchFrontierStatus::Exhausted).coverage,
            SearchControllerCoverage::Failed);
}

} // namespace
