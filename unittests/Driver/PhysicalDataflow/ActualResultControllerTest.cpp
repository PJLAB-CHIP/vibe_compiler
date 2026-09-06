//===- ActualResultControllerTest.cpp --------------------------------===//

#include "Wafer/Driver/PhysicalDataflow/ActualResultController.h"

#include "Wafer/Target/RuntimeLaunchContract.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
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
};

KeyParts makeKeyParts(uint32_t anchor) {
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
  region.executions.push_back({execution});
  parts.regions.groups.push_back(std::move(region));

  return parts;
}

StructuralCandidateKey makeKey(uint32_t anchor, uint64_t regionCount = 1) {
  KeyParts parts = makeKeyParts(anchor);
  for (uint64_t index = 1; index < regionCount; ++index) {
    ExecutionInstanceId execution =
        makeExecution(anchor + static_cast<uint32_t>(index));
    const auto &root = std::get<RequiredRootExecution>(execution.source);
    RegionGroupPlan region;
    region.tile = TileId(0);
    region.mandatoryRoots.push_back(root.work);
    region.executions.push_back({execution});
    parts.regions.groups.push_back(std::move(region));
  }
  llvm::sort(parts.regions.groups);
  return StructuralCandidateKey::create(std::move(parts.spatial),
                                        std::move(parts.regions));
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

SearchCostPolicy unitCostPolicy() {
  SearchCostPolicy policy;
  policy.ddrNominalBytesPerSecond = UINT64_C(1000000000000);
  policy.directionalNoCBytesPerSecond = UINT64_C(1000000000000);
  policy.dteEndpointBytesPerSecondEstimate = UINT64_C(1000000000000);
  policy.dteMessageStartupPicosecondsEstimate = 1;
  policy.noCHopPicosecondsEstimate = 1;
  policy.instructionFixedPicosecondsEstimate = 1;
  policy.dteWaitedEventPicosecondsEstimate = 1;
  policy.nccParticipantWaitPicosecondsEstimate = 1;
  policy.f16Bf16NPULogicalOpsPerSecondPerTile = UINT64_C(1000000000000);
  policy.f16Bf16VectorLogicalOpsPerSecondPerTile = UINT64_C(1000000000000);
  policy.f32VectorLogicalOpsPerSecondPerTile = UINT64_C(1000000000000);
  policy.spmExplicitMovementBytesPerSecondPerTileEstimate =
      UINT64_C(1000000000000);
  return policy;
}

ActualCandidateResult accepted(uint64_t instructions, bool known = true) {
  InstructionProgramAggregateCost cost;
  cost.aggregateInstructionCount.value = instructions;
  cost.aggregateDDRReadBytes.value = instructions * 2;
  cost.aggregateDDRWriteBytes.value = instructions * 3;
  cost.aggregateNoC.staticIssueSiteCount.value = instructions == 0 ? 0 : 1;
  cost.modeledNoCRoute.peakDirectedLinkByteDemand.value = instructions * 4;
  if (!known)
    cost.aggregateInstructionCount.knowledge =
        ScheduleCostKnowledge::Unavailable;
  ExecutableCompilationResult compilation;
  compilation.status = ExecutableCompilationStatus::Accepted;
  compilation.executable.emplace(std::vector<compiler::TileExecutable>{},
                                 makeLaunch(), std::move(cost));
  ActualCandidateResult result;
  result.status = ActualCandidateStatus::Accepted;
  result.compilation.emplace(std::move(compilation));
  return result;
}

ActualCandidateResult acceptedWithCost(InstructionProgramAggregateCost cost) {
  ExecutableCompilationResult compilation;
  compilation.status = ExecutableCompilationStatus::Accepted;
  compilation.executable.emplace(std::vector<compiler::TileExecutable>{},
                                 makeLaunch(), std::move(cost));
  ActualCandidateResult result;
  result.status = ActualCandidateStatus::Accepted;
  result.compilation.emplace(std::move(compilation));
  return result;
}

ActualCandidateResult exactRejected(uint32_t rootAnchor,
                                    bool spmCapacity = false) {
  ActualCandidateResult result;
  result.status = ActualCandidateStatus::ExactRejection;
  ExecutableCompilationResult compilation;
  compilation.status = ExecutableCompilationStatus::ProvenExactRejection;
  if (spmCapacity) {
    SemanticRootKey root;
    root.anchorIndex = rootAnchor;
    result.causalRoots.push_back(root);
    ExecutableTileFailure tile;
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
     ExplicitCohortDerivesResourceTermsUnknownOverflowAndStrictBound) {
  std::string failureReason;
  SearchCostPolicy policy = unitCostPolicy();
  auto cohort = SearchCostCohort::create(policy, &failureReason);
  ASSERT_TRUE(mlir::succeeded(cohort)) << failureReason;
  policy.ddrNominalBytesPerSecond = 0;
  EXPECT_TRUE(mlir::failed(SearchCostCohort::create(policy, &failureReason)));
  policy = unitCostPolicy();
  policy.profileIdentity = 0;
  EXPECT_TRUE(mlir::failed(SearchCostCohort::create(policy, &failureReason)));

  InstructionProgramAggregateCost cost;
  cost.aggregateInstructionCount.value = 1;
  cost.aggregateDDRReadBytes.value = 2;
  cost.aggregateDDRWriteBytes.value = 3;
  cost.aggregateNoC.staticIssueSiteCount.value = 1;
  cost.modeledNoCRoute.peakDirectedLinkByteDemand.value = 4;
  cost.maximumTileNoCTransmitBytes.value = 10;
  cost.maximumTileNoCTransmitMessageCount.value = 2;
  cost.minimumHopMessageDemand.value = 3;
  SearchObjective known = deriveSearchObjective(cost, *cohort);
  const auto *knownValue = std::get_if<KnownSearchObjective>(&known);
  ASSERT_NE(knownValue, nullptr);
  EXPECT_EQ(knownValue->durations.instructionControlPicoseconds, 1u);
  EXPECT_EQ(knownValue->durations.ddrPicoseconds, 5u);
  EXPECT_EQ(knownValue->durations.nocPicoseconds, 4u);
  EXPECT_EQ(knownValue->durations.dteEndpointPicoseconds, 10u);
  EXPECT_EQ(knownValue->durations.dteStartupPicoseconds, 2u);
  EXPECT_EQ(knownValue->durations.nocHopPicoseconds, 3u);
  EXPECT_TRUE(std::holds_alternative<UnknownSearchObjective>(
      deriveSearchObjective(cost, std::nullopt)));
  cost.minimumHopMessageDemand.knowledge = ScheduleCostKnowledge::Unavailable;
  EXPECT_EQ(std::get<UnknownSearchObjective>(
                deriveSearchObjective(cost, *cohort))
                .reason,
            SearchObjectiveUnknownReason::MetricUnavailable);
  cost.minimumHopMessageDemand.knowledge = ScheduleCostKnowledge::Known;
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Unavailable;
  EXPECT_EQ(
      std::get<UnknownSearchObjective>(deriveSearchObjective(cost, *cohort))
          .reason,
      SearchObjectiveUnknownReason::MetricUnavailable);
  cost.aggregateInstructionCount.knowledge = ScheduleCostKnowledge::Known;
  cost.aggregateInstructionCount.value = std::numeric_limits<uint64_t>::max();
  SearchCostPolicy overflowPolicy = unitCostPolicy();
  overflowPolicy.instructionFixedPicosecondsEstimate = 2;
  auto overflowCohort = *SearchCostCohort::create(overflowPolicy);
  EXPECT_EQ(std::get<UnknownSearchObjective>(
                deriveSearchObjective(cost, overflowCohort))
                .reason,
            SearchObjectiveUnknownReason::ArithmeticOverflow);

  ActualResultController controller = makeController(1, *cohort);
  StructuralCandidateKey key = makeKey(0);
  ASSERT_EQ(controller.reserve(key), CandidateReservation::Granted);
  ASSERT_EQ(controller.record(key, accepted(1)),
            CandidateRecordOutcome::Accepted);
  SearchResourceDurations equalDurations;
  equalDurations.instructionControlPicoseconds = 1;
  equalDurations.ddrPicoseconds = 5;
  equalDurations.nocPicoseconds = 4;
  SearchResourceDurations worseDurations = equalDurations;
  worseDurations.instructionControlPicoseconds = 2;
  SearchLowerBound equal{key, KnownSearchObjective{equalDurations, *cohort}};
  SearchLowerBound worse{key, KnownSearchObjective{worseDurations, *cohort}};
  SearchLowerBound unknown{
      key, UnknownSearchObjective{SearchObjectiveUnknownReason::NoCohort}};
  EXPECT_FALSE(controller.canPrune(equal));
  EXPECT_TRUE(controller.canPrune(worse));
  EXPECT_FALSE(controller.canPrune(unknown));
}

TEST(ActualResultControllerTest,
     TwoToSevenAcceptedResultsMatchIndependentWinnerOracleInAnyOrder) {
  // Two-to-seven is intentionally a bounded independent controller oracle;
  // real-scale IR coverage belongs to the caller-owned actualizer tests.
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  for (unsigned count = 2; count <= 7; ++count) {
    SCOPED_TRACE(count);
    ActualResultController controller = makeController(count, cohort);
    struct Record {
      StructuralCandidateKey key;
      uint64_t instructions = 0;
    };
    std::vector<Record> records;
    for (uint32_t index = 0; index < count; ++index)
      records.push_back(
          {makeKey(index),
           1 + (static_cast<uint64_t>(index) * 5 + 3) % (count + 1)});
    const Record *expected = &records.front();
    for (const Record &record : records)
      if (record.instructions < expected->instructions ||
          (record.instructions == expected->instructions &&
           record.key < expected->key))
        expected = &record;
    StructuralCandidateKey expectedKey = expected->key;
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
     SameInstructionCountKeepsNEAndVectorTradeoffIncomparable) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  auto makeCost = [](uint64_t ne, uint64_t vector) {
    InstructionProgramAggregateCost cost;
    cost.aggregateInstructionCount.value = 10;
    cost.aggregateCompute.npuF16Bf16LogicalOps.value = ne;
    cost.aggregateCompute.vectorF16Bf16LogicalOps.value = vector;
    return cost;
  };

  SearchObjective neHeavy = deriveSearchObjective(makeCost(1024, 0), cohort);
  SearchObjective vectorHeavy =
      deriveSearchObjective(makeCost(0, 1024), cohort);
  EXPECT_EQ(compareSearchObjectives(neHeavy, vectorHeavy),
            SearchObjectiveComparison::Incomparable);

  SearchObjective lessOfBoth =
      deriveSearchObjective(makeCost(1024, 1024), cohort);
  SearchObjective moreOfBoth =
      deriveSearchObjective(makeCost(1025, 1031), cohort);
  EXPECT_EQ(compareSearchObjectives(lessOfBoth, moreOfBoth),
            SearchObjectiveComparison::Better);
  const auto *known = std::get_if<KnownSearchObjective>(&lessOfBoth);
  ASSERT_NE(known, nullptr);
  EXPECT_EQ(known->durations.neF16Bf16Picoseconds, 1024u);
  EXPECT_EQ(known->durations.vectorF16Bf16Picoseconds, 1024u);
  EXPECT_EQ(known->durations.instructionControlPicoseconds, 10u);
}

TEST(ActualResultControllerTest,
     IncomparableSafeRefinementsPreferFewerSelectedRegionGroups) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  auto makeCost = [](uint64_t ne, uint64_t vector) {
    InstructionProgramAggregateCost cost;
    cost.aggregateInstructionCount.value = 10;
    cost.aggregateCompute.npuF16Bf16LogicalOps.value = ne;
    cost.aggregateCompute.vectorF16Bf16LogicalOps.value = vector;
    cost.aggregateDDRReadBytes.value = 100;
    cost.aggregateDDRWriteBytes.value = 100;
    cost.aggregateSPMMovementBytes.value = 100;
    return cost;
  };
  ActualResultController controller = makeController(3, cohort);
  for (auto [key, result] : llvm::zip_equal(
           std::array{makeKey(0, 12), makeKey(1, 10), makeKey(2, 5)},
           std::array{acceptedWithCost(makeCost(100, 100)),
                      acceptedWithCost(makeCost(50, 100)),
                      acceptedWithCost(makeCost(100, 50))})) {
    ASSERT_EQ(controller.reserve(key), CandidateReservation::Granted);
    ASSERT_EQ(controller.record(key, std::move(result)),
              CandidateRecordOutcome::Accepted);
  }
  SearchControllerResult result =
      controller.finish(SearchFrontierStatus::Exhausted);
  ASSERT_TRUE(result.winner);
  EXPECT_EQ(result.winner->key, makeKey(2, 5));
  EXPECT_EQ(result.coverage, SearchControllerCoverage::FeasibleUnranked);
}

TEST(ActualResultControllerTest,
     FewerRegionGroupsCannotOverrideAReferenceRegression) {
  auto cohort = *SearchCostCohort::create(unitCostPolicy());
  auto makeCost = [](uint64_t ne, uint64_t vector) {
    InstructionProgramAggregateCost cost;
    cost.aggregateInstructionCount.value = 10;
    cost.aggregateCompute.npuF16Bf16LogicalOps.value = ne;
    cost.aggregateCompute.vectorF16Bf16LogicalOps.value = vector;
    cost.aggregateDDRReadBytes.value = 100;
    cost.aggregateDDRWriteBytes.value = 100;
    cost.aggregateSPMMovementBytes.value = 100;
    return cost;
  };
  ActualResultController controller = makeController(3, cohort);
  for (auto [key, result] : llvm::zip_equal(
           std::array{makeKey(0, 12), makeKey(1, 10), makeKey(2, 1)},
           std::array{acceptedWithCost(makeCost(100, 100)),
                      acceptedWithCost(makeCost(50, 100)),
                      acceptedWithCost(makeCost(150, 50))})) {
    ASSERT_EQ(controller.reserve(key), CandidateReservation::Granted);
    ASSERT_EQ(controller.record(key, std::move(result)),
              CandidateRecordOutcome::Accepted);
  }
  SearchControllerResult result =
      controller.finish(SearchFrontierStatus::Exhausted);
  ASSERT_TRUE(result.winner);
  EXPECT_EQ(result.winner->key, makeKey(1, 10));
  EXPECT_EQ(result.coverage, SearchControllerCoverage::FeasibleUnranked);
}

TEST(ActualResultControllerTest,
     UnknownObjectivesUseCompleteKeyButRemainFeasibleUnranked) {
  ActualResultController controller = makeController(3);
  for (uint32_t index : {2u, 0u, 1u}) {
    StructuralCandidateKey key = makeKey(index);
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
     ExactFeedbackDoesNotMatchASiblingStructuralChoice) {
  StructuralCandidateKey rejected = makeKey(0);
  StructuralCandidateKey sibling = makeKey(1);
  ActualResultController controller = makeController(4);
  StructuralCandidateKey unsupported = makeKey(2);
  StructuralCandidateKey indeterminate = makeKey(3);
  for (const StructuralCandidateKey *key :
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
  ActualCandidateResult unsupportedResult;
  unsupportedResult.status = ActualCandidateStatus::Unsupported;
  EXPECT_EQ(controller.record(unsupported, std::move(unsupportedResult)),
            CandidateRecordOutcome::Unsupported);
  ActualCandidateResult indeterminateResult;
  indeterminateResult.status = ActualCandidateStatus::Indeterminate;
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
     DisabledExactCacheKeepsActualCapacityTypedWithoutGuessingCausalRoots) {
  ActualResultController controller =
      makeController(1, std::nullopt, ExactRejectionCachePolicy::Disabled);
  StructuralCandidateKey key = makeKey(0);
  ASSERT_EQ(controller.reserve(key), CandidateReservation::Granted);
  ActualCandidateResult rejection = exactRejected(0, /*spmCapacity=*/true);
  rejection.causalRoots.clear();
  EXPECT_EQ(controller.record(key, std::move(rejection)),
            CandidateRecordOutcome::ExactRejection);
  EXPECT_FALSE(controller.isForbidden(key));
  SearchControllerResult result =
      controller.finish(SearchFrontierStatus::Exhausted);
  EXPECT_EQ(result.coverage, SearchControllerCoverage::NoFeasible);
  EXPECT_EQ(result.statistics.exactRejected, 1u);
  EXPECT_EQ(result.exactCompleteRejections, 0u);
}

TEST(ActualResultControllerTest,
     ExactCachePolicyDoesNotChangeAcceptedSetOrWinner) {
  std::vector<StructuralCandidateKey> winners;
  std::vector<SearchControllerCoverage> coverage;
  std::vector<size_t> cacheSizes;
  for (ExactRejectionCachePolicy policy :
       {ExactRejectionCachePolicy::Enabled,
        ExactRejectionCachePolicy::Disabled}) {
    ActualResultController controller = makeController(2, std::nullopt, policy);
    StructuralCandidateKey rejected = makeKey(0);
    StructuralCandidateKey acceptedKey = makeKey(1);
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
  StructuralCandidateKey first = makeKey(0);
  StructuralCandidateKey second = makeKey(1);
  ASSERT_EQ(controller.reserve(first), CandidateReservation::Granted);
  EXPECT_EQ(controller.reserve(first), CandidateReservation::Duplicate);
  EXPECT_EQ(controller.reserve(second), CandidateReservation::Exhausted);
  ActualCandidateResult malformedAccepted;
  malformedAccepted.status = ActualCandidateStatus::Accepted;
  ExecutableCompilationResult compilation;
  compilation.status = ExecutableCompilationStatus::Accepted;
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
    StructuralCandidateKey key = makeKey(index);
    ASSERT_EQ(noFeasible.reserve(key), CandidateReservation::Granted);
    ASSERT_EQ(noFeasible.record(key, exactRejected(index)),
              CandidateRecordOutcome::ExactRejection);
  }
  SearchControllerResult exhausted =
      noFeasible.finish(SearchFrontierStatus::Exhausted);
  EXPECT_FALSE(exhausted.winner);
  EXPECT_EQ(exhausted.coverage, SearchControllerCoverage::NoFeasible);

  ActualResultController typedBug = makeController(1);
  StructuralCandidateKey bugKey = makeKey(4);
  ASSERT_EQ(typedBug.reserve(bugKey), CandidateReservation::Granted);
  ActualCandidateResult compilerBug;
  compilerBug.status = ActualCandidateStatus::CompilerBug;
  EXPECT_EQ(typedBug.record(bugKey, std::move(compilerBug)),
            CandidateRecordOutcome::CompilerBug);
  EXPECT_EQ(typedBug.finish(SearchFrontierStatus::Incomplete).coverage,
            SearchControllerCoverage::Failed);

  ActualResultController unreserved = makeController(1);
  StructuralCandidateKey unreservedKey = makeKey(5);
  EXPECT_EQ(unreserved.record(unreservedKey, accepted(1)),
            CandidateRecordOutcome::CompilerBug);
  EXPECT_EQ(unreserved.getStatistics().compilerBugs, 1u);

  ActualResultController inFlight = makeController(1);
  StructuralCandidateKey inFlightKey = makeKey(6);
  ASSERT_EQ(inFlight.reserve(inFlightKey), CandidateReservation::Granted);
  EXPECT_EQ(inFlight.finish(SearchFrontierStatus::Exhausted).coverage,
            SearchControllerCoverage::Failed);
}

} // namespace
