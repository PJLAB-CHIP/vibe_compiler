//===- WholeCardCandidateSelectionTest.cpp ------------------------------===//

#include "../../lib/Wafer/Compiler/WholeCardCandidateSelection.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/PhysicalTileExecutablesInternal.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace {

using wafer::analysis::InstructionProgramCost;
using wafer::analysis::ScheduleCostKnowledge;
using wafer::analysis::ScheduleCostReason;
using wafer::analysis::WholeCardInstructionProgramCost;
using wafer::compiler::detail::WholeCardCandidateCostView;
using wafer::compiler::detail::WholeCardCandidateSelection;
using wafer::compiler::detail::WholeCardQualificationSelection;
using wafer::compiler::detail::WholeCardSelectionBasis;
using wafer::compiler::detail::WholeVariantSelectionMode;

static WholeCardInstructionProgramCost makeCost(uint64_t ddrReadBytes) {
  WholeCardInstructionProgramCost cost;
  cost.aggregateDDRReadBytes.value = ddrReadBytes;
  return cost;
}

static void addSingleRankCost(WholeCardInstructionProgramCost &cost) {
  InstructionProgramCost rank;
  rank.compute = cost.aggregateCompute;
  rank.ddrReadBytes = cost.aggregateDDRReadBytes;
  rank.ddrWriteBytes = cost.aggregateDDRWriteBytes;
  rank.spmMovementBytes = cost.aggregateSPMMovementBytes;
  rank.gatherScatterBytes = cost.aggregateGatherScatterBytes;
  rank.noc = cost.aggregateNoC;
  rank.instructionCount = cost.aggregateInstructionCount;
  rank.nccParticipantWaitCount = cost.aggregateNCCParticipantWaitCount;
  rank.intrinsicNCCDrainCount = cost.aggregateIntrinsicNCCDrainCount;
  rank.qualifiedOverlapWindowCount = cost.aggregateQualifiedOverlapWindowCount;
  rank.directDTEComputeOverlapWindowCount =
      cost.aggregateDirectDTEComputeOverlapWindowCount;
  cost.rankCosts.push_back(rank);
}

static mlir::FailureOr<WholeCardCandidateSelection>
plan(std::vector<WholeCardCandidateCostView> views,
     WholeVariantSelectionMode mode = WholeVariantSelectionMode::Production) {
  return wafer::compiler::detail::selectWholeCardCandidateByCost(views, mode);
}

TEST(WholeCardCandidateSelectionTest,
     SelectsOnlyACompleteTargetModelMarginWinner) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {7, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::EstimatedBenefit);
  ASSERT_TRUE(result->selectedDuration.makespan.nominalPicoseconds.isKnown());
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     ReservedModeReturnsTheUniqueBaselineWithoutPromotion) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(1);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views, WholeVariantSelectionMode::ReservedBaseline);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::ReservedBaseline);
}

TEST(WholeCardCandidateSelectionTest,
     ModeledSPMAndGatherScatterTradeOffAgainstDDRBenefit) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  candidate.aggregateSPMMovementBytes.value = 1;
  candidate.maximumRankSPMMovementBytes.value = 1;
  candidate.aggregateGatherScatterBytes.value = 1;
  candidate.maximumRankGatherScatterBytes.value = 1;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::EstimatedBenefit);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     LargeSPMRegressionCannotClearTheProductionMargin) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  candidate.aggregateSPMMovementBytes.value = 1'500'000'000;
  candidate.maximumRankSPMMovementBytes.value = 1'500'000'000;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     KnownPrimitiveImprovementDoesNotNeedWholeProgramMargin) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(995'000'000);
  addSingleRankCost(baseline);
  addSingleRankCost(candidate);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::EstimatedBenefit);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     ControlDeltaMustClearTheMarginAgainstRegressedWaitWork) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(995'000'000);
  baseline.aggregateInstructionCount.value = 1'000'000;
  candidate.aggregateInstructionCount.value = 900'000;
  candidate.aggregateNCCParticipantWaitCount.value = 1;
  addSingleRankCost(baseline);
  addSingleRankCost(candidate);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::EstimatedBenefit);
}

TEST(WholeCardCandidateSelectionTest,
     ControlDeltaThatDoesNotClearTheMarginKeepsBaseline) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(995'000'000);
  baseline.aggregateInstructionCount.value = 2;
  candidate.aggregateInstructionCount.value = 1;
  candidate.aggregateNCCParticipantWaitCount.value = 1;
  addSingleRankCost(baseline);
  addSingleRankCost(candidate);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
}

TEST(WholeCardCandidateSelectionTest,
     DTEWaitDeltaUsesTheSameChangedControlWorkMargin) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(995'000'000);
  baseline.aggregateInstructionCount.value = 1'000'000;
  candidate.aggregateInstructionCount.value = 900'000;
  candidate.aggregateNoC.waitOperationCount.value = 1;
  candidate.aggregateNoC.waitedEventCount.value = 1;
  addSingleRankCost(baseline);
  addSingleRankCost(candidate);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::EstimatedBenefit);
}

TEST(WholeCardCandidateSelectionTest,
     PrimitiveImprovementCannotReverseAWeakerScheduleContext) {
  WholeCardInstructionProgramCost baseline = makeCost(150'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(149'000'000);
  baseline.aggregateCompute.vectorF16Bf16LogicalOps.value = 64'000'000;
  candidate.aggregateCompute.vectorF16Bf16LogicalOps.value = 63'000'000;
  baseline.aggregateQualifiedOverlapWindowCount.value = 1;
  addSingleRankCost(baseline);
  addSingleRankCost(candidate);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
}

TEST(WholeCardCandidateSelectionTest,
     UnknownOverlapWitnessIsNotASequentialScheduleProof) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(995'000'000);
  candidate.aggregateQualifiedOverlapWindowCount = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  addSingleRankCost(baseline);
  addSingleRankCost(candidate);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
}

TEST(WholeCardCandidateSelectionTest,
     CompositeComputePriorCannotHideAnEngineTradeoff) {
  WholeCardInstructionProgramCost baseline = makeCost(0);
  WholeCardInstructionProgramCost candidate = makeCost(0);
  baseline.aggregateCompute.vectorF16Bf16LogicalOps.value = 64'000'000;
  candidate.aggregateCompute.npuF16Bf16LogicalOps.value = 7'960'000'000;
  addSingleRankCost(baseline);
  addSingleRankCost(candidate);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
}

TEST(WholeCardCandidateSelectionTest,
     AuditFactsDoNotAbsolutelyVetoAModeledBenefit) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  baseline.aggregateEventCount.value = 1;
  baseline.aggregateInstructionCount.value = 1;
  baseline.aggregateReadyOrderPriorityInversions.value = 1;
  baseline.maximumRankDDRHighWaterBytes.value = 1024;
  baseline.summedRankDDRHighWaterBytes.value = 1024;
  candidate.aggregateEventCount.value = 1'000;
  candidate.aggregateInstructionCount.value = 1'000;
  candidate.aggregateReadyOrderPriorityInversions.value = 1'000;
  candidate.maximumRankDDRHighWaterBytes.value = 2048;
  candidate.summedRankDDRHighWaterBytes.value = 2048;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::EstimatedBenefit);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     UnknownSPMMovementKeepsTheConservativeBaseline) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  candidate.aggregateSPMMovementBytes = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.maximumRankSPMMovementBytes = candidate.aggregateSPMMovementBytes;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     SameUnknownModeledServiceCannotProveDominance) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(995'000'000);
  baseline.aggregateSPMMovementBytes = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  baseline.maximumRankSPMMovementBytes = baseline.aggregateSPMMovementBytes;
  candidate.aggregateSPMMovementBytes = baseline.aggregateSPMMovementBytes;
  candidate.maximumRankSPMMovementBytes = baseline.maximumRankSPMMovementBytes;
  addSingleRankCost(baseline);
  addSingleRankCost(candidate);
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::ReservedBaseline);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     StructuralDepthAndAllocationSitesDoNotVetoModeledBenefit) {
  WholeCardInstructionProgramCost baseline = makeCost(595'722'240);
  WholeCardInstructionProgramCost candidate = makeCost(8'519'680);
  baseline.aggregateSPMMovementBytes.value = 1'185'562'624;
  baseline.maximumRankSPMMovementBytes.value = 1'185'562'624;
  candidate.aggregateSPMMovementBytes.value = 50'978'816;
  candidate.maximumRankSPMMovementBytes.value = 50'978'816;
  baseline.maximumRankDataDependencyDepth.value = 6;
  candidate.maximumRankDataDependencyDepth.value = 549;
  baseline.aggregateCompilerOwnedSPMBufferCount.value = 46;
  baseline.maximumRankCompilerOwnedSPMBufferCount.value = 46;
  baseline.aggregateCompilerOwnedDDRBufferCount.value = 2;
  baseline.maximumRankCompilerOwnedDDRBufferCount.value = 2;
  candidate.aggregateCompilerOwnedSPMBufferCount.value = 83;
  candidate.maximumRankCompilerOwnedSPMBufferCount.value = 83;
  candidate.aggregateCompilerOwnedDDRBufferCount.value = 4;
  candidate.maximumRankCompilerOwnedDDRBufferCount.value = 4;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::EstimatedBenefit);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     UnmodeledConvertWorkStillKeepsTheConservativeBaseline) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  candidate.aggregateCompute.vectorOtherLogicalOps.value = 1;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     ExactAuditDominanceCannotInventAnUncalibratedBenefit) {
  WholeCardInstructionProgramCost baseline = makeCost(0);
  WholeCardInstructionProgramCost candidate = baseline;
  baseline.aggregateGatherScatterBytes.value = 1024;
  baseline.maximumRankGatherScatterBytes.value = 1024;
  candidate.aggregateGatherScatterBytes.value = 512;
  candidate.maximumRankGatherScatterBytes.value = 512;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
  EXPECT_EQ(result->basis, WholeCardSelectionBasis::ReservedBaseline);
  EXPECT_EQ(result->selectedDuration.makespan.nominalPicoseconds.value, 0u);
}

TEST(WholeCardCandidateSelectionTest,
     AcceptedSPMHighWaterIsCapacityOnlyNotExecutionCost) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  baseline.maximumRankSPMHighWaterBytes.value = 1024;
  baseline.summedRankSPMHighWaterBytes.value = 1024;
  candidate.maximumRankSPMHighWaterBytes.value = 2048;
  candidate.summedRankSPMHighWaterBytes.value = 2048;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
}

TEST(WholeCardCandidateSelectionTest,
     DifferentStructuralDepthUnknownsDoNotVetoModeledBenefit) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  baseline.maximumRankDataDependencyDepth = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.maximumRankDataDependencyDepth = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::DynamicLoopTripCount};
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
  EXPECT_EQ(result->paretoIndices.size(), 2u);
}

TEST(WholeCardCandidateSelectionTest,
     SameUnknownDispositionDoesNotBecomeZeroOrBlockKnownImprovement) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost candidate = makeCost(100'000'000);
  baseline.maximumRankDataDependencyDepth = {
      0, ScheduleCostKnowledge::Unknown,
      ScheduleCostReason::UnsupportedControlFlow};
  candidate.maximumRankDataDependencyDepth =
      baseline.maximumRankDataDependencyDepth;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &candidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 1u);
}

TEST(WholeCardCandidateSelectionTest,
     StableOrdinalBreaksOnlyACompleteHardwareTupleTie) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost first = makeCost(100'000'000);
  WholeCardInstructionProgramCost second = first;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {9, false, &first},
      {3, false, &second},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(views[result->selectedIndex].stableSemanticOrdinal, 3);
}

TEST(WholeCardCandidateSelectionTest,
     EqualNominalButCostIncomparableCandidatesFallBackToBaseline) {
  WholeCardInstructionProgramCost baseline = makeCost(2'000'000'000);
  WholeCardInstructionProgramCost ddrCandidate = makeCost(100'000'000);
  WholeCardInstructionProgramCost computeCandidate = makeCost(0);
  // ceil(5,333,333,336 * 1e12 / 8e12) == the 100 MB DDR nominal.
  computeCandidate.aggregateCompute.npuF16Bf16LogicalOps.value = 5'333'333'336;
  std::vector<WholeCardCandidateCostView> views = {
      {0, true, &baseline},
      {1, false, &ddrCandidate},
      {2, false, &computeCandidate},
  };

  auto result = plan(views);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->selectedIndex, 0u);
}

TEST(WholeCardCandidateSelectionTest, DecisionIsIndependentOfCandidateOrder) {
  WholeCardInstructionProgramCost baseline = makeCost(1'000'000'000);
  WholeCardInstructionProgramCost worse = makeCost(400'000'000);
  WholeCardInstructionProgramCost winner = makeCost(100'000'000);
  std::vector<WholeCardCandidateCostView> first = {
      {0, true, &baseline}, {8, false, &worse}, {4, false, &winner}};
  std::vector<WholeCardCandidateCostView> second = {
      {4, false, &winner}, {0, true, &baseline}, {8, false, &worse}};

  auto firstResult = plan(first);
  auto secondResult = plan(second);
  ASSERT_TRUE(mlir::succeeded(firstResult));
  ASSERT_TRUE(mlir::succeeded(secondResult));
  EXPECT_EQ(first[firstResult->selectedIndex].stableSemanticOrdinal, 4);
  EXPECT_EQ(second[secondResult->selectedIndex].stableSemanticOrdinal, 4);
}

TEST(WholeCardCandidateSelectionTest, RejectsMissingOrDuplicateBaseline) {
  WholeCardInstructionProgramCost cost = makeCost(1);
  EXPECT_TRUE(mlir::failed(plan({{0, false, &cost}})));
  EXPECT_TRUE(mlir::failed(plan({{0, true, &cost}, {1, true, &cost}})));
}

TEST(WholeCardCandidateSelectionTest,
     UsesScheduleActionOrdinalWithinOneStructuredVariant) {
  WholeCardInstructionProgramCost cost = makeCost(1);
  auto distinctActions = plan({{4, true, &cost, 0}, {4, false, &cost, 1}});
  ASSERT_TRUE(mlir::succeeded(distinctActions));
  EXPECT_EQ(distinctActions->selectedIndex, 0u);

  EXPECT_TRUE(mlir::failed(plan({{4, true, &cost, 0}, {4, false, &cost, 0}})));
}

class CoordinatedQualificationSelectionTest : public ::testing::Test {
protected:
  CoordinatedQualificationSelectionTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeCollectiveModule(bool send, llvm::StringRef phase,
                       llvm::StringRef secondPhase = {}) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  func.func @main() {
    %buffer = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %token0 = wafer.instr.dte_)mlir"
       << (send ? "send" : "recv")
       << R"mlir( %buffer {peer = 1 : i64, bytes = 8 : i64,
        message = #wafer.dte_message<communication = 7, phase = )mlir"
       << phase << R"mlir(, round = 0, slice = 0>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token0 : !async.token
)mlir";
    if (!secondPhase.empty())
      os << "    %token1 = wafer.instr.dte_" << (send ? "send" : "recv")
         << R"mlir( %buffer {peer = 1 : i64, bytes = 8 : i64,
        message = #wafer.dte_message<communication = 8, phase = )mlir"
         << secondPhase << R"mlir(, round = 0, slice = 0>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token1 : !async.token
)mlir";
    os << R"mlir(    return
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  wafer::compiler::detail::EvaluatedWholeCardCandidate
  makeExecutable(int64_t semanticOrdinal, uint32_t actionOrdinal,
                 bool reservedBaseline, llvm::StringRef phase,
                 llvm::StringRef secondPhase = {},
                 bool implementationAlternative = false) {
    std::vector<wafer::compiler::RankExecutable> ranks;
    for (int64_t logicalRank = 0; logicalRank < 2; ++logicalRank) {
      mlir::OwningOpRef<mlir::ModuleOp> module = makeCollectiveModule(
          /*send=*/logicalRank == 0, phase, secondPhase);
      EXPECT_TRUE(module);
      ranks.push_back(wafer::compiler::PhysicalTileExecutablesBuilder::makeRank(
          logicalRank, std::move(module), "main", /*programBindings=*/{},
          wafer::compiler::TransportContract::DirectDTE));
    }
    const wafer::RuntimeLaunchPhaseRole mainPhase =
        wafer::RuntimeLaunchPhaseRole::Main;
    wafer::RuntimeLaunchContract launch =
        llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
            wafer::KernelLaunchForm::Grid,
            wafer::KernelEntryABI::RankMajorPointerTable,
            llvm::ArrayRef<wafer::RuntimeLaunchPhaseRole>(&mainPhase, 1)));
    wafer::compiler::detail::WholeCardExecutableResult accepted(
        std::move(ranks), std::move(launch),
        wafer::analysis::WholeCardInstructionProgramCost{});
    return wafer::compiler::detail::EvaluatedWholeCardCandidate(
        semanticOrdinal, reservedBaseline, std::move(accepted), actionOrdinal,
        wafer::compiler::detail::WholeCardScheduleActionKey{},
        implementationAlternative);
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CoordinatedQualificationSelectionTest,
       CollectiveCharacterizationFiltersFinalTypedPhasesThenUsesStableOrder) {
  std::vector<wafer::compiler::detail::EvaluatedWholeCardCandidate> variants;
  variants.push_back(makeExecutable(
      /*semanticOrdinal=*/0, /*actionOrdinal=*/0,
      /*reservedBaseline=*/true, "all_gather_ring"));
  variants.push_back(makeExecutable(
      /*semanticOrdinal=*/9, /*actionOrdinal=*/2,
      /*reservedBaseline=*/false, "all_gather_direct"));
  variants.push_back(makeExecutable(
      /*semanticOrdinal=*/3, /*actionOrdinal=*/7,
      /*reservedBaseline=*/false, "all_gather_direct"));
  variants.push_back(makeExecutable(
      /*semanticOrdinal=*/1, /*actionOrdinal=*/1,
      /*reservedBaseline=*/false, "all_gather_direct", "all_gather_ring"));

  mlir::FailureOr<WholeCardQualificationSelection> selected =
      wafer::compiler::detail::selectWholeCardQualificationCandidate(
          variants, WholeVariantSelectionMode::CharacterizeAllGatherDirect);
  ASSERT_TRUE(mlir::succeeded(selected));
  ASSERT_EQ(selected->matchingIndices.size(), 2u);
  EXPECT_EQ(variants[selected->selectedIndex].stableSemanticOrdinal, 3);
  EXPECT_EQ(variants[selected->selectedIndex].scheduleActionOrdinal, 7u);
}

TEST_F(CoordinatedQualificationSelectionTest,
       NoCResidentRingRequiresCurrentTypedNoCAndBoundaryOnlyDDR) {
  std::vector<wafer::compiler::detail::EvaluatedWholeCardCandidate> variants;
  variants.push_back(makeExecutable(
      /*semanticOrdinal=*/0, /*actionOrdinal=*/0,
      /*reservedBaseline=*/true, "all_reduce_ring"));
  variants.push_back(makeExecutable(
      /*semanticOrdinal=*/4, /*actionOrdinal=*/3,
      /*reservedBaseline=*/false, "all_reduce_ring",
      /*secondPhase=*/{}, /*implementationAlternative=*/true));

  mlir::FailureOr<WholeCardQualificationSelection> selected =
      wafer::compiler::detail::selectWholeCardQualificationCandidate(
          variants, WholeVariantSelectionMode::QualifyNoCResidentAllReduceRing);
  ASSERT_TRUE(mlir::succeeded(selected));
  EXPECT_EQ(selected->matchingIndices.size(), 1u);
  EXPECT_EQ(selected->selectedIndex, 1u);
  EXPECT_TRUE(
      variants[selected->selectedIndex].implementationAlternativeOrigin);

  std::string diagnostics;
  llvm::raw_string_ostream diagnosticsStream(diagnostics);
  mlir::FailureOr<wafer::compiler::detail::EvaluatedWholeCardCandidate> winner =
      wafer::compiler::detail::selectEvaluatedWholeCardCandidate(
          std::move(variants),
          WholeVariantSelectionMode::QualifyNoCResidentAllReduceRing,
          diagnosticsStream);
  ASSERT_TRUE(mlir::succeeded(winner));
  diagnosticsStream.flush();
  EXPECT_NE(diagnostics.find("implementation_alternative=true"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("reserved_baseline=false"), std::string::npos);
}

TEST_F(CoordinatedQualificationSelectionTest,
       QualificationCandidateSetReductionDoesNotApplyProductionCostPromotion) {
  std::vector<wafer::compiler::detail::EvaluatedWholeCardCandidate> variants;
  variants.push_back(makeExecutable(
      /*semanticOrdinal=*/0, /*actionOrdinal=*/0,
      /*reservedBaseline=*/true, "all_gather_ring"));
  variants.push_back(makeExecutable(
      /*semanticOrdinal=*/8, /*actionOrdinal=*/4,
      /*reservedBaseline=*/false, "all_gather_direct"));
  variants.push_back(makeExecutable(
      /*semanticOrdinal=*/2, /*actionOrdinal=*/6,
      /*reservedBaseline=*/false, "all_gather_direct"));

  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::reduceEvaluatedWholeCardCandidates(
          variants, WholeVariantSelectionMode::CharacterizeAllGatherDirect)));
  EXPECT_EQ(variants.size(), 3u);
}

} // namespace
