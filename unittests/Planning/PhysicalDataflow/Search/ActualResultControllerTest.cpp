//===- ActualResultControllerTest.cpp --------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/Search/ActualResultController.h"

#include "Wafer/Target/Core/RuntimeLaunchContract.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
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

ClosedSchedulePlan makePlan(uint32_t anchor) {
  ClosedSchedulePlan plan;
  EventId event{ExecutionEventAction{makeExecution(anchor)},
                PlannedEventKind::ComputeIssue};
  PipelineScopeId scope{{event}, {}};
  plan.structure.scopes.push_back(SerializedExecutionStructure{scope});
  PhysicalVersionId version{ExecutionResultValueId{makeExecution(anchor), 0}};
  StorageObjectId object{StorageObjectOrigin{version}};
  plan.buffers.storageObjects.push_back({object, TileId(0)});
  plan.buffers.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  plan.controlOrders.push_back({TileControlScope{TileId(0), scope}, {event}});
  return plan;
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
  compilation.executable.emplace(std::vector<wafer::compiler::TileExecutable>{},
                                 makeLaunch(), std::move(cost));
  FullFeasibilityResult result;
  result.status = FullFeasibilityStatus::Accepted;
  result.compilation.emplace(std::move(compilation));
  return result;
}

FullFeasibilityResult exactRejected(uint32_t rootAnchor) {
  FullFeasibilityResult result;
  result.status = FullFeasibilityStatus::ExactRejection;
  SemanticRootKey root;
  root.anchorIndex = rootAnchor;
  result.causalRoots.push_back(root);
  CardExecutableCompilationResult compilation;
  compilation.status = CardExecutableCompilationStatus::ProvenExactRejection;
  result.compilation.emplace(std::move(compilation));
  return result;
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

  ActualResultController controller(1, *cohort);
  ClosedSchedulePlan plan = makePlan(0);
  ASSERT_EQ(controller.reserve(plan), CandidateReservation::Granted);
  ASSERT_EQ(controller.record(plan, accepted(1)),
            CandidateRecordOutcome::Accepted);
  SearchObjective equal = KnownSearchObjective{30, *cohort};
  SearchObjective worse = KnownSearchObjective{31, *cohort};
  SearchObjective unknown =
      UnknownSearchObjective{SearchObjectiveUnknownReason::NoCohort};
  EXPECT_FALSE(controller.canPrune(equal));
  EXPECT_TRUE(controller.canPrune(worse));
  EXPECT_FALSE(controller.canPrune(unknown));
}

TEST(ActualResultControllerTest,
     TwoToSevenAcceptedResultsSelectDeterministicallyInAnyInputOrder) {
  auto cohort = *SearchCostCohort::create(1, 1, 1, 1);
  for (unsigned count = 2; count <= 7; ++count) {
    SCOPED_TRACE(count);
    ActualResultController controller(count, cohort);
    std::vector<uint32_t> order;
    for (uint32_t index = 0; index < count; ++index)
      order.push_back(index);
    std::reverse(order.begin(), order.end());
    for (uint32_t index : order) {
      ClosedSchedulePlan plan = makePlan(index);
      ASSERT_EQ(controller.reserve(plan), CandidateReservation::Granted);
      ASSERT_EQ(controller.record(plan, accepted(index + 1)),
                CandidateRecordOutcome::Accepted);
    }
    SearchControllerResult result = controller.finish(true);
    ASSERT_TRUE(result.winner);
    EXPECT_EQ(result.winner->plan, makePlan(0));
    EXPECT_EQ(result.coverage, SearchControllerCoverage::ComparableBest);
    EXPECT_EQ(result.statistics.accepted, count);
  }
}

TEST(ActualResultControllerTest,
     UnknownObjectivesUseSemanticCommitKeyButRemainFeasibleUnranked) {
  ActualResultController controller(3);
  for (uint32_t index : {2u, 0u, 1u}) {
    ClosedSchedulePlan plan = makePlan(index);
    ASSERT_EQ(controller.reserve(plan), CandidateReservation::Granted);
    ASSERT_EQ(controller.record(plan, accepted(10, /*known=*/false)),
              CandidateRecordOutcome::Accepted);
  }
  SearchControllerResult result = controller.finish(true);
  ASSERT_TRUE(result.winner);
  EXPECT_EQ(result.winner->plan, makePlan(0));
  EXPECT_EQ(result.coverage, SearchControllerCoverage::FeasibleUnranked);
}

TEST(ActualResultControllerTest,
     ExactUnsupportedAndIndeterminateHaveDistinctCacheAndCoverageEffects) {
  ActualResultController controller(4);
  ClosedSchedulePlan rejected = makePlan(0);
  ClosedSchedulePlan sibling = makePlan(1);
  ClosedSchedulePlan unsupported = makePlan(2);
  ClosedSchedulePlan indeterminate = makePlan(3);
  for (const ClosedSchedulePlan *plan :
       {&rejected, &sibling, &unsupported, &indeterminate})
    ASSERT_EQ(controller.reserve(*plan), CandidateReservation::Granted);
  EXPECT_EQ(controller.record(rejected, exactRejected(0)),
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
  EXPECT_EQ(controller.getExactCompleteRejectionCount(), 1u);
  SearchControllerResult result = controller.finish(true);
  ASSERT_TRUE(result.winner);
  EXPECT_EQ(result.coverage, SearchControllerCoverage::FeasiblePartial);
  EXPECT_EQ(result.statistics.exactRejected, 1u);
  EXPECT_EQ(result.statistics.unsupported, 1u);
  EXPECT_EQ(result.statistics.indeterminate, 1u);
}

TEST(ActualResultControllerTest,
     ReservationAllowanceMalformedKeysAndCompilerBugFailClosed) {
  ActualResultController controller(1);
  ClosedSchedulePlan malformed;
  EXPECT_EQ(controller.reserve(malformed), CandidateReservation::Invalid);
  ClosedSchedulePlan first = makePlan(0);
  ClosedSchedulePlan second = makePlan(1);
  ASSERT_EQ(controller.reserve(first), CandidateReservation::Granted);
  EXPECT_EQ(controller.reserve(first), CandidateReservation::Duplicate);
  EXPECT_EQ(controller.reserve(second), CandidateReservation::Exhausted);
  FullFeasibilityResult bug;
  bug.status = FullFeasibilityStatus::CompilerBug;
  EXPECT_EQ(controller.record(first, std::move(bug)),
            CandidateRecordOutcome::CompilerBug);
  SearchControllerResult result = controller.finish(false);
  EXPECT_EQ(result.coverage, SearchControllerCoverage::Failed);

  ActualResultController noFeasible(2);
  for (uint32_t index = 0; index < 2; ++index) {
    ClosedSchedulePlan plan = makePlan(index);
    ASSERT_EQ(noFeasible.reserve(plan), CandidateReservation::Granted);
    ASSERT_EQ(noFeasible.record(plan, exactRejected(index)),
              CandidateRecordOutcome::ExactRejection);
  }
  SearchControllerResult exhausted = noFeasible.finish(true);
  EXPECT_FALSE(exhausted.winner);
  EXPECT_EQ(exhausted.coverage, SearchControllerCoverage::NoFeasible);
}

} // namespace
