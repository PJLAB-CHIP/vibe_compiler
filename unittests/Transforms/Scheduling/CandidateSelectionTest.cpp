//===- CandidateSelectionTest.cpp - Candidate exact gates ---------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

#include "gtest/gtest.h"

#include <cstdint>

namespace {

using wafer::tensor_program_scheduling::CandidateArtifactSource;
using wafer::tensor_program_scheduling::CandidateCheckResult;
using wafer::tensor_program_scheduling::CandidateSpec;
using wafer::tensor_program_scheduling::CandidateStats;
using wafer::tensor_program_scheduling::estimateTargetSPMRequiredLiveBytes;
using wafer::tensor_program_scheduling::estimateTargetSPMWorkingSetBytes;
using wafer::tensor_program_scheduling::evaluateCandidateOnStandaloneTaskText;
using wafer::tensor_program_scheduling::failsCheapSPMBound;
using wafer::tensor_program_scheduling::getCheapTargetGeometryFailure;
using wafer::tensor_program_scheduling::getRankingCostFailure;
using wafer::tensor_program_scheduling::getStaticRootReductionRanges;
using wafer::tensor_program_scheduling::getTraversalComputeRootLinalgOps;
using wafer::tensor_program_scheduling::getYieldedRootLinalgOps;
using wafer::tensor_program_scheduling::SelectionConfig;

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

TEST(CandidateSelectionTest,
     LooksThroughShapePreservingAllReduceOnlyForTraversalPressure) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @direct_matmul(
      %lhs: tensor<4096x256xf16>, %rhs: tensor<256x4096xf16>,
      %out: tensor<4096x4096xf16>) -> tensor<4096x4096xf16> {
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<4096x256xf16>, tensor<256x4096xf16>)
        outs(%out : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
    return %result : tensor<4096x4096xf16>
  }

  func.func @all_reduce_matmul(
      %lhs: tensor<4096x256xf16>, %rhs: tensor<256x4096xf16>,
      %partial_out: tensor<4096x4096xf16>,
      %collective_out: tensor<4096x4096xf16>)
      -> tensor<4096x4096xf16> {
    %partial = linalg.matmul
        ins(%lhs, %rhs : tensor<4096x256xf16>, tensor<256x4096xf16>)
        outs(%partial_out : tensor<4096x4096xf16>)
        -> tensor<4096x4096xf16>
    %reduced = wafer.linalg_ext.collective.all_reduce
        ins(%partial : tensor<4096x4096xf16>)
        outs(%collective_out : tensor<4096x4096xf16>) {
      ^bb0(%lhs_value: f16, %rhs_value: f16):
        %sum = arith.addf %lhs_value, %rhs_value : f16
        wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 35 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4096x4096xf16>
    return %reduced : tensor<4096x4096xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::func::FuncOp direct =
      source->lookupSymbol<mlir::func::FuncOp>("direct_matmul");
  mlir::func::FuncOp wrapped =
      source->lookupSymbol<mlir::func::FuncOp>("all_reduce_matmul");
  ASSERT_TRUE(direct);
  ASSERT_TRUE(wrapped);

  auto traversalRoots = getTraversalComputeRootLinalgOps(wrapped);
  ASSERT_TRUE(traversalRoots);
  ASSERT_EQ(traversalRoots->size(), 1u);
  EXPECT_TRUE(mlir::isa<mlir::linalg::MatmulOp>(
      traversalRoots->front().getOperation()));
  EXPECT_FALSE(getYieldedRootLinalgOps(wrapped));

  // The all-reduce wrapper directs M/N search through the local GEMM, but it
  // must not expose local K=256 as a compiler-created reduction split axis.
  auto reductionRanges = getStaticRootReductionRanges(wrapped);
  ASSERT_TRUE(mlir::succeeded(reductionRanges));
  EXPECT_TRUE(reductionRanges->empty());

  CandidateSpec tile{/*tileSizes=*/{256, 256},
                     /*reductionSplitSizes=*/{}};
  std::optional<int64_t> directWorking =
      estimateTargetSPMWorkingSetBytes(direct, tile, /*spmAlignment=*/256);
  std::optional<int64_t> wrappedWorking =
      estimateTargetSPMWorkingSetBytes(wrapped, tile, /*spmAlignment=*/256);
  std::optional<int64_t> directRequired =
      estimateTargetSPMRequiredLiveBytes(direct, tile,
                                         /*spmAlignment=*/256);
  std::optional<int64_t> wrappedRequired =
      estimateTargetSPMRequiredLiveBytes(wrapped, tile,
                                         /*spmAlignment=*/256);
  ASSERT_TRUE(directWorking);
  ASSERT_TRUE(wrappedWorking);
  ASSERT_TRUE(directRequired);
  ASSERT_TRUE(wrappedRequired);
  EXPECT_EQ(*wrappedWorking - *directWorking,
            3 * 256 * 256 * static_cast<int64_t>(sizeof(uint16_t)));
  EXPECT_EQ(*wrappedRequired, *directRequired);
  EXPECT_LE(*wrappedWorking, 3080192 - 65536);

  EXPECT_FALSE(failsCheapSPMBound(wrapped, tile, /*spmBase=*/65536,
                                  /*spmLimit=*/3080192));
  EXPECT_TRUE(failsCheapSPMBound(wrapped,
                                 CandidateSpec{/*tileSizes=*/{4096, 4096},
                                               /*reductionSplitSizes=*/{}},
                                 /*spmBase=*/65536, /*spmLimit=*/3080192));
  EXPECT_TRUE(
      getCheapTargetGeometryFailure(wrapped,
                                    CandidateSpec{/*tileSizes=*/{65536, 1},
                                                  /*reductionSplitSizes=*/{}},
                                    /*reductionRanges=*/{}));
}

TEST(CandidateSelectionTest,
     UsesWorstCollectiveRoleForDirectAllReduceRequiredLiveBound) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect, wafer::WaferDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @all_reduce(%input: tensor<4096x4096xf16>,
                        %out: tensor<4096x4096xf16>)
      -> tensor<4096x4096xf16> {
    %reduced = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<4096x4096xf16>)
        outs(%out : tensor<4096x4096xf16>) {
      ^bb0(%lhs: f16, %rhs: f16):
        %sum = arith.addf %lhs, %rhs : f16
        wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 61 : i64,
       rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                              8, 9, 10, 11, 12, 13, 14, 15>}
        -> tensor<4096x4096xf16>
    return %reduced : tensor<4096x4096xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function =
      source->lookupSymbol<mlir::func::FuncOp>("all_reduce");
  ASSERT_TRUE(function);

  std::optional<int64_t> oversized = estimateTargetSPMRequiredLiveBytes(
      function,
      CandidateSpec{/*tileSizes=*/{512, 1024},
                    /*reductionSplitSizes=*/{}},
      /*spmAlignment=*/256);
  ASSERT_TRUE(oversized);
  EXPECT_EQ(*oversized, 3 * 512 * 1024 *
                            static_cast<int64_t>(sizeof(uint16_t)));
  EXPECT_GT(*oversized, 3080192 - 65536);

  std::optional<int64_t> fitting = estimateTargetSPMRequiredLiveBytes(
      function,
      CandidateSpec{/*tileSizes=*/{512, 512},
                    /*reductionSplitSizes=*/{}},
      /*spmAlignment=*/256);
  ASSERT_TRUE(fitting);
  EXPECT_EQ(*fitting,
            3 * 512 * 512 * static_cast<int64_t>(sizeof(uint16_t)));
  EXPECT_LE(*fitting, 3080192 - 65536);
}

} // namespace
