//===- CandidateSelectionTest.cpp - Ranking cost policy -----------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

#include "gtest/gtest.h"

namespace {

using wafer::tensor_program_scheduling::CandidateArtifactSource;
using wafer::tensor_program_scheduling::CandidateCheckResult;
using wafer::tensor_program_scheduling::CandidateSpec;
using wafer::tensor_program_scheduling::CandidateStats;
using wafer::tensor_program_scheduling::estimateCandidateTimePs;
using wafer::tensor_program_scheduling::estimateTargetSPMRequiredLiveBytes;
using wafer::tensor_program_scheduling::estimateTargetSPMWorkingSetBytes;
using wafer::tensor_program_scheduling::evaluateCandidateOnStandaloneTaskText;
using wafer::tensor_program_scheduling::getRankingCostFailure;
using wafer::tensor_program_scheduling::hasStrictExecutionCostDominance;
using wafer::tensor_program_scheduling::SelectionConfig;

TEST(CandidateSelectionTest, SerializesSPMMovementIssueAndEventCosts) {
  CandidateStats zero;
  EXPECT_EQ(estimateCandidateTimePs(zero), 0);

  CandidateStats spmOnly;
  spmOnly.program.spmMovementBytes.value = 1024;
  int64_t spmTime = estimateCandidateTimePs(spmOnly);
  EXPECT_GT(spmTime, 0);

  CandidateStats issueOnly;
  issueOnly.program.instructionCount.value = 1;
  int64_t issueTime = estimateCandidateTimePs(issueOnly);
  EXPECT_GT(issueTime, 0);

  CandidateStats eventOnly;
  eventOnly.program.eventCount.value = 1;
  int64_t eventTime = estimateCandidateTimePs(eventOnly);
  EXPECT_GT(eventTime, 0);

  CandidateStats combined;
  combined.program.spmMovementBytes.value = 1024;
  combined.program.instructionCount.value = 1;
  combined.program.eventCount.value = 1;
  EXPECT_EQ(estimateCandidateTimePs(combined), spmTime + issueTime + eventTime);
}

TEST(CandidateSelectionTest, RejectsUnknownOrUnsupportedRankingDimensions) {
  CandidateStats unknown;
  unknown.program.spmMovementBytes.knowledge =
      wafer::analysis::ScheduleCostKnowledge::Unknown;
  unknown.program.spmMovementBytes.reason =
      wafer::analysis::ScheduleCostReason::UnknownResourceBytes;
  std::optional<std::string> unknownFailure = getRankingCostFailure(unknown);
  ASSERT_TRUE(unknownFailure);
  EXPECT_NE(unknownFailure->find("ranking-cost: SPM movement bytes is unknown "
                                 "(unknown-resource-bytes)"),
            std::string::npos)
      << *unknownFailure;

  CandidateStats unsupported;
  unsupported.program.eventCount.knowledge =
      wafer::analysis::ScheduleCostKnowledge::Unsupported;
  unsupported.program.eventCount.reason =
      wafer::analysis::ScheduleCostReason::UnsupportedInstructionSemantics;
  std::optional<std::string> unsupportedFailure =
      getRankingCostFailure(unsupported);
  ASSERT_TRUE(unsupportedFailure);
  EXPECT_NE(unsupportedFailure->find("ranking-cost: event count is unsupported "
                                     "(unsupported-instruction-semantics)"),
            std::string::npos)
      << *unsupportedFailure;
}

TEST(CandidateSelectionTest,
     UsesExactDominanceWhenCoarseTimeSaturatesForUncalibratedCompute) {
  CandidateStats spill;
  spill.program.compute.vectorOtherLogicalOps.value = 128;
  spill.program.ddrReadBytes.value = 4096;
  spill.program.ddrWriteBytes.value = 4096;
  spill.program.spmMovementBytes.value = 8192;
  spill.program.instructionCount.value = 8;

  CandidateStats resident = spill;
  resident.program.ddrReadBytes.value = 0;
  resident.program.ddrWriteBytes.value = 0;
  resident.program.spmMovementBytes.value = 4096;
  resident.program.instructionCount.value = 6;

  EXPECT_EQ(estimateCandidateTimePs(spill),
            std::numeric_limits<int64_t>::max());
  EXPECT_EQ(estimateCandidateTimePs(resident),
            std::numeric_limits<int64_t>::max());
  EXPECT_TRUE(hasStrictExecutionCostDominance(resident, spill));
  EXPECT_FALSE(hasStrictExecutionCostDominance(spill, resident));

  CandidateStats tradeoff = resident;
  tradeoff.program.spmMovementBytes.value = 8193;
  EXPECT_FALSE(hasStrictExecutionCostDominance(tradeoff, spill));

  CandidateStats unknown = resident;
  unknown.program.ddrReadBytes.knowledge =
      wafer::analysis::ScheduleCostKnowledge::Unknown;
  unknown.program.ddrReadBytes.reason =
      wafer::analysis::ScheduleCostReason::UnknownResourceBytes;
  EXPECT_FALSE(hasStrictExecutionCostDominance(unknown, spill));
}

TEST(CandidateSelectionTest,
     UsesCompleteTraversalInsteadOfPartialRepresentativeForResourceGate) {
  constexpr llvm::StringLiteral task = R"mlir(
module {
  func.func @transpose(%input: tensor<8x16xf16>,
                       %out: tensor<16x8xf16>)
      -> tensor<16x8xf16> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d1, d0)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<8x16xf16>)
        outs(%out : tensor<16x8xf16>) {
      ^bb0(%value: f16, %init: f16):
        linalg.yield %value : f16
      } -> tensor<16x8xf16>
    return %result : tensor<16x8xf16>
  }
}
)mlir";

  wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
  SelectionConfig config(policy);
  config.logicalRank = 0;
  CandidateSpec candidate{/*tileSizes=*/{8, 4},
                          /*reductionSplitSizes=*/{}};

  CandidateCheckResult result = evaluateCandidateOnStandaloneTaskText(
      task, /*traversalShape=*/{16, 8}, candidate, config);
  EXPECT_TRUE(result.failureReason.empty()) << result.failureReason;
  EXPECT_EQ(result.artifactSource,
            CandidateArtifactSource::CompleteTraversalAPI);
}

TEST(CandidateSelectionTest,
     EvaluatesTargetReciprocalAsDistinctCompleteImplementationCandidate) {
  constexpr llvm::StringLiteral task = R"mlir(
module {
  func.func @reciprocal(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%value: f32, %init: f32):
      %one = arith.constant 1.0 : f32
      %reciprocal = arith.divf %one, %value : f32
      linalg.yield %reciprocal : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir";

  wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
  SelectionConfig config(policy);
  config.logicalRank = 0;

  CandidateSpec baseline{/*tileSizes=*/{2},
                         /*reductionSplitSizes=*/{}};
  CandidateSpec alternative = baseline;
  alternative.selectedImplementationAlternative =
      wafer::TargetImplementationKind::GenericReciprocal;

  CandidateCheckResult baselineResult = evaluateCandidateOnStandaloneTaskText(
      task, /*traversalShape=*/{4}, baseline, config);
  CandidateCheckResult alternativeResult =
      evaluateCandidateOnStandaloneTaskText(task, /*traversalShape=*/{4},
                                            alternative, config);
  EXPECT_TRUE(baselineResult.failureReason.empty())
      << baselineResult.failureReason;
  EXPECT_TRUE(alternativeResult.failureReason.empty())
      << alternativeResult.failureReason;
  EXPECT_EQ(baselineResult.artifactSource,
            CandidateArtifactSource::CompleteTraversalAPI);
  EXPECT_EQ(alternativeResult.artifactSource,
            CandidateArtifactSource::CompleteTraversalAPI);
  EXPECT_LT(alternativeResult.stats.program.instructionCount.value,
            baselineResult.stats.program.instructionCount.value);
}

TEST(CandidateSelectionTest, AccountsForCXPaddingInMatmulWorkingSet) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @matmul(%lhs: tensor<16x4096xf16>,
                    %rhs: tensor<4096x688xf16>,
                    %out: tensor<16x688xf16>) -> tensor<16x688xf16> {
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<16x688xf16>) -> tensor<16x688xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<16x4096xf16>, tensor<4096x688xf16>)
        outs(%init : tensor<16x688xf16>) -> tensor<16x688xf16>
    return %result : tensor<16x688xf16>
  }
  func.func @fused_transpose_matmul(
      %lhs: tensor<16x4096xf16>, %weight: tensor<688x4096xf16>,
      %out: tensor<16x688xf16>) -> tensor<16x688xf16> {
    %transposed_out = tensor.empty() : tensor<4096x688xf16>
    %rhs = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%weight : tensor<688x4096xf16>)
        outs(%transposed_out : tensor<4096x688xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<4096x688xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<16x688xf16>) -> tensor<16x688xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<16x4096xf16>, tensor<4096x688xf16>)
        outs(%init : tensor<16x688xf16>) -> tensor<16x688xf16>
    return %result : tensor<16x688xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function =
      source->lookupSymbol<mlir::func::FuncOp>("matmul");
  ASSERT_TRUE(function);

  std::optional<int64_t> padded = estimateTargetSPMWorkingSetBytes(
      function,
      CandidateSpec{/*tileSizes=*/{16, 172},
                    /*reductionSplitSizes=*/{}},
      /*spmAlignment=*/256);
  ASSERT_TRUE(padded);
  EXPECT_EQ(*padded, 3255808);

  std::optional<int64_t> capacityDirected = estimateTargetSPMWorkingSetBytes(
      function,
      CandidateSpec{/*tileSizes=*/{16, 86},
                    /*reductionSplitSizes=*/{}},
      /*spmAlignment=*/256);
  ASSERT_TRUE(capacityDirected);
  EXPECT_EQ(*capacityDirected, 1758976);
  EXPECT_LE(*capacityDirected, 3080192 - 65536);

  mlir::func::FuncOp fused =
      source->lookupSymbol<mlir::func::FuncOp>("fused_transpose_matmul");
  ASSERT_TRUE(fused);
  std::optional<int64_t> fusedPadded = estimateTargetSPMWorkingSetBytes(
      fused,
      CandidateSpec{/*tileSizes=*/{16, 172},
                    /*reductionSplitSizes=*/{}},
      /*spmAlignment=*/256);
  ASSERT_TRUE(fusedPadded);
  EXPECT_EQ(*fusedPadded, 4664832);
  std::optional<int64_t> fusedRequired = estimateTargetSPMRequiredLiveBytes(
      fused,
      CandidateSpec{/*tileSizes=*/{16, 172},
                    /*reductionSplitSizes=*/{}},
      /*spmAlignment=*/256);
  ASSERT_TRUE(fusedRequired);
  EXPECT_EQ(*fusedRequired, 3255808);
  EXPECT_GT(*fusedRequired, 3080192 - 65536);

  std::optional<int64_t> fusedCapacityDirected =
      estimateTargetSPMWorkingSetBytes(
          fused,
          CandidateSpec{/*tileSizes=*/{16, 86},
                        /*reductionSplitSizes=*/{}},
          /*spmAlignment=*/256);
  ASSERT_TRUE(fusedCapacityDirected);
  EXPECT_EQ(*fusedCapacityDirected, 2463488);
  EXPECT_LE(*fusedCapacityDirected, 3080192 - 65536);
  std::optional<int64_t> fusedCapacityRequired =
      estimateTargetSPMRequiredLiveBytes(
          fused,
          CandidateSpec{/*tileSizes=*/{16, 86},
                        /*reductionSplitSizes=*/{}},
          /*spmAlignment=*/256);
  ASSERT_TRUE(fusedCapacityRequired);
  EXPECT_EQ(*fusedCapacityRequired, 1758976);
  EXPECT_LE(*fusedCapacityRequired, 3080192 - 65536);
}

} // namespace
