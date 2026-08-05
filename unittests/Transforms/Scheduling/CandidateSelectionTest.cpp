//===- CandidateSelectionTest.cpp - Candidate exact gates ---------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>

namespace {

using wafer::tensor_program_scheduling::CandidateArtifactSource;
using wafer::tensor_program_scheduling::CandidateCheckResult;
using wafer::tensor_program_scheduling::CandidateEvaluationExecutor;
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
using wafer::tensor_program_scheduling::SelectedCandidate;
using wafer::tensor_program_scheduling::SelectionConfig;

class CandidateSearchExecutionTest : public ::testing::Test {
protected:
  CandidateSearchExecutionTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::math::MathDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    wafer::WaferDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);
    wafer::registerTargetImplementationExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  struct SelectionRun {
    mlir::OwningOpRef<mlir::ModuleOp> source;
    mlir::OwningOpRef<mlir::ModuleOp> taskModule;
    std::optional<SelectedCandidate> selected;
  };

  SelectionRun select(unsigned taskAlternativeOrdinal,
                      int64_t candidateParallelism,
                      CandidateEvaluationExecutor *evaluationExecutor = nullptr,
                      unsigned spmWorkingSetMultiplicity = 1,
                      int64_t spmCapacityBytes = 2048) {
    SelectionRun run;
    run.source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<16x16xf16>,
                  %rhs: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<16x16xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<16x16xf16>) -> tensor<16x16xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>)
        outs(%init : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %result : tensor<16x16xf16>
  }
}
)mlir",
                                                         context.get());
    if (!run.source) {
      ADD_FAILURE() << "failed to parse candidate-search source";
      return run;
    }

    llvm::SmallVector<wafer::structured_scheduler::StructuredSchedulingScope, 1>
        scopes;
    if (mlir::failed(
            wafer::structured_scheduler::discoverStructuredSchedulingScopes(
                *run.source, scopes)) ||
        scopes.size() != 1) {
      ADD_FAILURE() << "failed to discover one scheduling scope";
      return run;
    }
    run.taskModule =
        wafer::structured_scheduler::cloneScopeToStandaloneModule(scopes[0]);
    if (!run.taskModule) {
      ADD_FAILURE() << "failed to clone standalone scheduling task";
      return run;
    }
    mlir::func::FuncOp task =
        wafer::structured_scheduler::findSingleTaskFunction(*run.taskModule);
    if (!task) {
      ADD_FAILURE() << "standalone module has no task";
      return run;
    }

    wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
    SelectionConfig config(policy);
    config.logicalRank = 0;
    config.preferredTileSizes = {16, 8, 4, 2, 1};
    config.maxCandidatesPerDim = 5;
    config.maxSearchCandidates = 16;
    config.searchBeamWidth = 8;
    config.candidateParallelism = candidateParallelism;
    config.evaluationExecutor = evaluationExecutor;
    config.taskAlternativeOrdinal = taskAlternativeOrdinal;
    config.spmWorkingSetMultiplicity = spmWorkingSetMultiplicity;
    // The capacity-directed 8x16 seed passes. The queued full tile and the
    // first deterministic refinement then fail exact placement before the
    // next refinement passes. This fixes both passing ordinals and the
    // preceding-failure path without changing queue order.
    config.spmLimit = config.spmBase + spmCapacityBytes;

    mlir::FailureOr<SelectedCandidate> selected =
        wafer::tensor_program_scheduling::selectCandidateForScope(
            scopes[0], task, "main#0", config);
    if (mlir::failed(selected)) {
      ADD_FAILURE() << "candidate selection failed";
      return run;
    }
    run.selected.emplace(std::move(*selected));
    return run;
  }

  static std::string printModule(mlir::ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream os(text);
    module.print(os);
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

class InterfaceTraversalCandidateSelectionTest : public ::testing::Test {
protected:
  InterfaceTraversalCandidateSelectionTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::math::MathDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    wafer::WaferDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);
    wafer::registerTargetImplementationExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  struct Run {
    mlir::OwningOpRef<mlir::ModuleOp> source;
    mlir::OwningOpRef<mlir::ModuleOp> taskModule;
    std::optional<SelectedCandidate> selected;
  };

  Run select(llvm::StringRef sourceText,
             wafer::CandidateTileTraversalKind traversalKind,
             unsigned taskAlternativeOrdinal) {
    Run run;
    run.source =
        mlir::parseSourceString<mlir::ModuleOp>(sourceText, context.get());
    if (!run.source) {
      ADD_FAILURE() << "failed to parse interface traversal source";
      return run;
    }
    llvm::SmallVector<wafer::structured_scheduler::StructuredSchedulingScope, 1>
        scopes;
    if (mlir::failed(
            wafer::structured_scheduler::discoverStructuredSchedulingScopes(
                *run.source, scopes)) ||
        scopes.size() != 1) {
      ADD_FAILURE() << "failed to discover one interface traversal scope";
      return run;
    }
    run.taskModule =
        wafer::structured_scheduler::cloneScopeToStandaloneModule(scopes[0]);
    mlir::func::FuncOp task =
        run.taskModule ? wafer::structured_scheduler::findSingleTaskFunction(
                             *run.taskModule)
                       : mlir::func::FuncOp{};
    if (!task) {
      ADD_FAILURE() << "failed to clone interface traversal task";
      return run;
    }

    wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
    SelectionConfig config(policy);
    config.logicalRank = 0;
    config.preferredTileSizes = {8, 4, 2, 1};
    config.maxCandidatesPerDim = 4;
    config.maxSearchCandidates = 8;
    config.searchBeamWidth = 4;
    config.traversalKind = traversalKind;
    config.taskAlternativeOrdinal = taskAlternativeOrdinal;
    mlir::FailureOr<SelectedCandidate> selected =
        wafer::tensor_program_scheduling::selectCandidateForScope(
            scopes.front(), task, "interface-traversal", config);
    if (mlir::failed(selected)) {
      ADD_FAILURE() << "interface traversal candidate selection failed";
      return run;
    }
    run.selected.emplace(std::move(*selected));
    return run;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

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

  CandidateStats unknownDrain;
  unknownDrain.program.nccParticipantWaitCount.knowledge =
      wafer::analysis::ScheduleCostKnowledge::Unknown;
  unknownDrain.program.nccParticipantWaitCount.reason =
      wafer::analysis::ScheduleCostReason::ConditionalControlFlow;
  std::optional<std::string> unknownDrainFailure =
      getRankingCostFailure(unknownDrain);
  ASSERT_TRUE(unknownDrainFailure);
  EXPECT_NE(unknownDrainFailure->find(
                "ranking-cost: NCC participant wait count is unknown "
                "(conditional-control-flow)"),
            std::string::npos)
      << *unknownDrainFailure;

  CandidateStats unknownSteadyDrain;
  unknownSteadyDrain.program.steadyStateNCCParticipantWaitCount.knowledge =
      wafer::analysis::ScheduleCostKnowledge::Unknown;
  unknownSteadyDrain.program.steadyStateNCCParticipantWaitCount.reason =
      wafer::analysis::ScheduleCostReason::DynamicLoopTripCount;
  std::optional<std::string> unknownSteadyDrainFailure =
      getRankingCostFailure(unknownSteadyDrain);
  ASSERT_TRUE(unknownSteadyDrainFailure);
  EXPECT_NE(unknownSteadyDrainFailure->find(
                "ranking-cost: steady-state NCC participant wait count is "
                "unknown (dynamic-loop-trip-count)"),
            std::string::npos)
      << *unknownSteadyDrainFailure;
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

  // Ordinary traversal uses the wrapper only for M/N pressure. It must not
  // expose local K=256 as a reduction split axis unless the independent
  // PartialReduction interface recipe was selected explicitly.
  auto reductionRanges = getStaticRootReductionRanges(wrapped);
  ASSERT_TRUE(mlir::succeeded(reductionRanges));
  EXPECT_TRUE(reductionRanges->empty());
  auto partialReductionRanges = getStaticRootReductionRanges(
      wrapped, wafer::CandidateTileTraversalKind::PartialReduction);
  ASSERT_TRUE(mlir::succeeded(partialReductionRanges));
  EXPECT_EQ(*partialReductionRanges, (llvm::SmallVector<int64_t, 2>{256}));

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
  CandidateSpec oversizedPartial{/*tileSizes=*/{256, 256},
                                 /*reductionSplitSizes=*/{256}};
  oversizedPartial.traversalKind =
      wafer::CandidateTileTraversalKind::PartialReduction;
  EXPECT_TRUE(failsCheapSPMBound(wrapped, oversizedPartial,
                                 /*spmBase=*/65536,
                                 /*spmLimit=*/3080192));
  CandidateSpec fittingPartial{/*tileSizes=*/{16, 16},
                               /*reductionSplitSizes=*/{64}};
  fittingPartial.traversalKind =
      wafer::CandidateTileTraversalKind::PartialReduction;
  EXPECT_FALSE(failsCheapSPMBound(wrapped, fittingPartial,
                                  /*spmBase=*/65536,
                                  /*spmLimit=*/3080192));
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
  EXPECT_EQ(*oversized,
            3 * 512 * 1024 * static_cast<int64_t>(sizeof(uint16_t)));
  EXPECT_GT(*oversized, 3080192 - 65536);

  std::optional<int64_t> fitting = estimateTargetSPMRequiredLiveBytes(
      function,
      CandidateSpec{/*tileSizes=*/{512, 512},
                    /*reductionSplitSizes=*/{}},
      /*spmAlignment=*/256);
  ASSERT_TRUE(fitting);
  EXPECT_EQ(*fitting, 3 * 512 * 512 * static_cast<int64_t>(sizeof(uint16_t)));
  EXPECT_LE(*fitting, 3080192 - 65536);
}

TEST(CandidateSelectionTest,
     UnitLocalReductionDoesNotBorrowCTReduceTraversalNarrowing) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @unit_local_reduce(%input: tensor<1x458752xf16>)
      -> tensor<458752xf16> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<458752xf16>
    %out = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<458752xf16>) -> tensor<458752xf16>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>
        ],
        iterator_types = ["reduction", "parallel"]
      } ins(%input : tensor<1x458752xf16>)
        outs(%out : tensor<458752xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %sum = arith.addf %value, %acc : f16
      linalg.yield %sum : f16
    } -> tensor<458752xf16>
    return %result : tensor<458752xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp task =
      source->lookupSymbol<mlir::func::FuncOp>("unit_local_reduce");
  ASSERT_TRUE(task);

  CandidateSpec fullTraversal{/*tileSizes=*/{458752},
                              /*reductionSplitSizes=*/{}};
  EXPECT_FALSE(getCheapTargetGeometryFailure(task, fullTraversal,
                                             /*reductionRanges=*/{1}));
  EXPECT_TRUE(getCheapTargetGeometryFailure(task, fullTraversal,
                                            /*reductionRanges=*/{2}));

  CandidateSpec narrowTraversalWideReduction{/*tileSizes=*/{57344},
                                             /*reductionSplitSizes=*/{65536}};
  EXPECT_TRUE(getCheapTargetGeometryFailure(task, narrowTraversalWideReduction,
                                            /*reductionRanges=*/{65536}));
}

TEST_F(InterfaceTraversalCandidateSelectionTest,
       UnitLocalReductionCanSelectTraversalBeyondCTReduceLimit) {
  Run run = select(R"mlir(
module {
  func.func @unit_local_reduce(%input: tensor<1x458752xf16>)
      -> tensor<458752xf16> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<458752xf16>
    %out = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<458752xf16>) -> tensor<458752xf16>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d1)>
        ],
        iterator_types = ["reduction", "parallel"]
      } ins(%input : tensor<1x458752xf16>)
        outs(%out : tensor<458752xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %sum = arith.addf %value, %acc : f16
      linalg.yield %sum : f16
    } -> tensor<458752xf16>
    return %result : tensor<458752xf16>
  }
}
)mlir",
                   wafer::CandidateTileTraversalKind::ResultDriven,
                   /*taskAlternativeOrdinal=*/0);
  ASSERT_TRUE(run.selected);
  ASSERT_TRUE(run.selected->module);
  EXPECT_EQ(run.selected->spec.tileSizes,
            (llvm::SmallVector<int64_t, 2>{114688}));
  EXPECT_TRUE(run.selected->spec.reductionSplitSizes.empty());
  EXPECT_EQ(run.selected->candidateCount, 3);
  EXPECT_EQ(run.selected->completeEvaluationCount, 3);
  EXPECT_EQ(run.selected->rejectedCount, 2);
}

TEST_F(InterfaceTraversalCandidateSelectionTest,
       IndependentRecipeAcceptsOperandDrivenActualClone) {
  Run run = select(R"mlir(
module {
  func.func @pointwise(%lhs: tensor<4x6xf32>, %rhs: tensor<4x6xf32>,
                       %out: tensor<4x6xf32>) -> tensor<4x6xf32> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%lhs, %rhs : tensor<4x6xf32>, tensor<4x6xf32>)
        outs(%out : tensor<4x6xf32>) {
    ^bb0(%left: f32, %right: f32, %old: f32):
      %sum = arith.addf %left, %right : f32
      linalg.yield %sum : f32
    } -> tensor<4x6xf32>
    return %result : tensor<4x6xf32>
  }
}
)mlir",
                   wafer::CandidateTileTraversalKind::OperandDriven,
                   /*taskAlternativeOrdinal=*/0);
  ASSERT_TRUE(run.selected);
  ASSERT_TRUE(run.selected->module);
  EXPECT_EQ(run.selected->spec.traversalKind,
            wafer::CandidateTileTraversalKind::OperandDriven);
  EXPECT_EQ(run.selected->candidateCount, 1);
  EXPECT_EQ(run.selected->completeEvaluationCount, 1);
  EXPECT_EQ(run.selected->rejectedCount, 0);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*run.selected->module)));
}

TEST_F(InterfaceTraversalCandidateSelectionTest,
       IndependentRecipeAcceptsPartialReductionActualClone) {
  Run run = select(R"mlir(
module {
  func.func @sum(%input: tensor<4x8xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32)
        outs(%out : tensor<4xf32>) -> tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir",
                   wafer::CandidateTileTraversalKind::PartialReduction,
                   /*taskAlternativeOrdinal=*/0);
  ASSERT_TRUE(run.selected);
  ASSERT_TRUE(run.selected->module);
  EXPECT_EQ(run.selected->spec.traversalKind,
            wafer::CandidateTileTraversalKind::PartialReduction);
  EXPECT_EQ(run.selected->spec.reductionSplitSizes,
            (llvm::SmallVector<int64_t, 2>{8}));
  EXPECT_EQ(run.selected->candidateCount, 1);
  EXPECT_EQ(run.selected->completeEvaluationCount, 1);
  EXPECT_EQ(run.selected->rejectedCount, 0);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*run.selected->module)));
}

TEST_F(InterfaceTraversalCandidateSelectionTest,
       PartialReductionRecipeFeedsExistingTypedAllReduceProtocol) {
  Run run = select(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  func.func @local_sum_all_reduce(%input: tensor<4x8xf32>,
                                  %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %local = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf32>)
        outs(%init : tensor<4xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    %reduced = wafer.linalg_ext.collective.all_reduce
        ins(%local : tensor<4xf32>)
        outs(%out : tensor<4xf32>) {
    ^bb0(%left: f32, %right: f32):
      %sum = arith.addf %left, %right : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 73 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xf32>
    return %reduced : tensor<4xf32>
  }
}
)mlir",
                   wafer::CandidateTileTraversalKind::PartialReduction,
                   /*taskAlternativeOrdinal=*/0);
  ASSERT_TRUE(run.selected);
  ASSERT_TRUE(run.selected->module);
  EXPECT_EQ(run.selected->spec.traversalKind,
            wafer::CandidateTileTraversalKind::PartialReduction);
  EXPECT_EQ(run.selected->spec.reductionSplitSizes,
            (llvm::SmallVector<int64_t, 2>{8}));
  unsigned dteIssues = 0;
  unsigned localReductions = 0;
  run.selected->module->walk([&](mlir::Operation *operation) {
    dteIssues +=
        mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp>(operation);
    localReductions += mlir::isa<wafer::InstrElementwiseOp>(operation);
  });
  EXPECT_GT(dteIssues, 0u);
  EXPECT_GT(localReductions, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*run.selected->module)));
}

TEST(CandidateSelectionTest,
     OperandDrivenProductionCloneRejectsUnknownBoundaryCover) {
  constexpr llvm::StringLiteral task = R"mlir(
module {
  func.func @broadcast(%input: tensor<6xf32>, %out: tensor<4x6xf32>)
      -> tensor<4x6xf32> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<6xf32>) outs(%out : tensor<4x6xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    } -> tensor<4x6xf32>
    return %result : tensor<4x6xf32>
  }
}
)mlir";
  wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
  SelectionConfig config(policy);
  config.logicalRank = 0;
  CandidateSpec candidate{/*tileSizes=*/{2, 3},
                          /*reductionSplitSizes=*/{}};
  candidate.traversalKind = wafer::CandidateTileTraversalKind::OperandDriven;
  CandidateCheckResult result = evaluateCandidateOnStandaloneTaskText(
      task, /*traversalShape=*/{4, 6}, candidate, config);
  EXPECT_NE(result.failureReason.find(
                "operand-driven complete traversal found no exact boundary "
                "tile seed"),
            std::string::npos)
      << result.failureReason;
  EXPECT_TRUE(result.acceptedModuleText.empty());
}

TEST(CandidateSelectionTest,
     PartialReductionProductionCloneRejectsUnsupportedNumericRegrouping) {
  constexpr llvm::StringLiteral task = R"mlir(
module {
  func.func @unsigned_max(%input: tensor<4x8xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %zero = arith.constant 0 : i32
    %init = linalg.fill ins(%zero : i32)
        outs(%out : tensor<4xi32>) -> tensor<4xi32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xi32>)
        outs(%init : tensor<4xi32>) {
    ^bb0(%value: i32, %acc: i32):
      %maximum = arith.maxui %value, %acc : i32
      linalg.yield %maximum : i32
    } -> tensor<4xi32>
    return %result : tensor<4xi32>
  }
}
)mlir";
  wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
  SelectionConfig config(policy);
  config.logicalRank = 0;
  CandidateSpec candidate{/*tileSizes=*/{2},
                          /*reductionSplitSizes=*/{4}};
  candidate.traversalKind = wafer::CandidateTileTraversalKind::PartialReduction;
  CandidateCheckResult result = evaluateCandidateOnStandaloneTaskText(
      task, /*traversalShape=*/{4}, candidate, config);
  EXPECT_NE(result.failureReason.find(
                "candidate reduction split cannot preserve unsigned min/max "
                "semantics"),
            std::string::npos)
      << result.failureReason;
  EXPECT_TRUE(result.acceptedModuleText.empty());
}

TEST(CandidateSelectionTest,
     AdmitsQualifiedF16NativePartialReductionBeforeMaterialization) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @matmul(%lhs: tensor<2x1024xf16>,
                    %rhs: tensor<1024x2xf16>,
                    %out: tensor<2x2xf16>) -> tensor<2x2xf16> {
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<2x1024xf16>, tensor<1024x2xf16>)
        outs(%out : tensor<2x2xf16>) -> tensor<2x2xf16>
    return %result : tensor<2x2xf16>
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function =
      source->lookupSymbol<mlir::func::FuncOp>("matmul");
  ASSERT_TRUE(function);

  CandidateSpec candidate{/*tileSizes=*/{2, 2},
                          /*reductionSplitSizes=*/{1024}};
  candidate.traversalKind = wafer::CandidateTileTraversalKind::PartialReduction;
  std::optional<std::string> oversized = getCheapTargetGeometryFailure(
      function, candidate, {1024});
  EXPECT_FALSE(oversized);

  candidate.reductionSplitSizes = {512};
  EXPECT_FALSE(getCheapTargetGeometryFailure(function, candidate, {1024}));
}

TEST(CandidateSelectionTest,
     TypedAllReducePartialRecipeRejectsUnsupportedLocalRegrouping) {
  constexpr llvm::StringLiteral task = R"mlir(
module {
  func.func @local_unsigned_max_all_reduce(%input: tensor<4x8xi32>,
                                           %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<4xi32>
    %init = linalg.fill ins(%zero : i32)
        outs(%empty : tensor<4xi32>) -> tensor<4xi32>
    %local = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xi32>)
        outs(%init : tensor<4xi32>) {
    ^bb0(%value: i32, %acc: i32):
      %maximum = arith.maxui %value, %acc : i32
      linalg.yield %maximum : i32
    } -> tensor<4xi32>
    %reduced = wafer.linalg_ext.collective.all_reduce
        ins(%local : tensor<4xi32>)
        outs(%out : tensor<4xi32>) {
    ^bb0(%left: i32, %right: i32):
      %maximum = arith.maxui %left, %right : i32
      wafer.linalg_ext.collective.yield %maximum : i32
    } {channel_id = 74 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xi32>
    return %reduced : tensor<4xi32>
  }
}
)mlir";
  wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
  SelectionConfig config(policy);
  config.logicalRank = 0;
  CandidateSpec candidate{/*tileSizes=*/{2},
                          /*reductionSplitSizes=*/{4}};
  candidate.traversalKind = wafer::CandidateTileTraversalKind::PartialReduction;
  CandidateCheckResult result = evaluateCandidateOnStandaloneTaskText(
      task, /*traversalShape=*/{4}, candidate, config);
  EXPECT_NE(result.failureReason.find(
                "candidate reduction split cannot preserve unsigned min/max "
                "semantics"),
            std::string::npos)
      << result.failureReason;
  EXPECT_TRUE(result.acceptedModuleText.empty());
}

TEST_F(CandidateSearchExecutionTest,
       StopsOrdinalZeroAfterOneCompletePassingEvaluation) {
  SelectionRun serial =
      select(/*taskAlternativeOrdinal=*/0, /*candidateParallelism=*/1);
  ASSERT_TRUE(serial.selected);
  ASSERT_TRUE(serial.selected->module);
  EXPECT_EQ(serial.selected->candidateCount, 2);
  EXPECT_EQ(serial.selected->completeEvaluationCount, 1);
  EXPECT_EQ(serial.selected->rejectedCount, 0);
  EXPECT_EQ(serial.selected->spec.tileSizes,
            (llvm::SmallVector<int64_t, 4>{8, 16}));

  SelectionRun parallel =
      select(/*taskAlternativeOrdinal=*/0, /*candidateParallelism=*/4);
  ASSERT_TRUE(parallel.selected);
  ASSERT_TRUE(parallel.selected->module);
  // The fixed semantic batch visits the cheap-rejected full tile in both
  // execution modes. The accepted worker module is imported rather than
  // lowered a second time on the owner.
  EXPECT_EQ(parallel.selected->candidateCount, 2);
  EXPECT_EQ(parallel.selected->completeEvaluationCount, 1);
  EXPECT_EQ(parallel.selected->rejectedCount, 0);
  EXPECT_EQ(parallel.selected->candidateCount,
            serial.selected->candidateCount);
  EXPECT_EQ(parallel.selected->completeEvaluationCount,
            serial.selected->completeEvaluationCount);
  EXPECT_EQ(printModule(*parallel.selected->module),
            printModule(*serial.selected->module));
}

TEST_F(CandidateSearchExecutionTest,
       ConcurrentWorkingSetRequirementDirectsTileSearch) {
  SelectionRun baseline =
      select(/*taskAlternativeOrdinal=*/0, /*candidateParallelism=*/1,
             /*evaluationExecutor=*/nullptr,
             /*spmWorkingSetMultiplicity=*/1,
             /*spmCapacityBytes=*/4096);
  SelectionRun concurrent =
      select(/*taskAlternativeOrdinal=*/0, /*candidateParallelism=*/1,
             /*evaluationExecutor=*/nullptr,
             /*spmWorkingSetMultiplicity=*/2,
             /*spmCapacityBytes=*/4096);
  ASSERT_TRUE(baseline.selected);
  ASSERT_TRUE(concurrent.selected);

  int64_t baselineTileElements = 1;
  for (int64_t size : baseline.selected->spec.tileSizes)
    baselineTileElements *= size;
  int64_t concurrentTileElements = 1;
  for (int64_t size : concurrent.selected->spec.tileSizes)
    concurrentTileElements *= size;
  EXPECT_LT(concurrentTileElements, baselineTileElements);
  EXPECT_NE(concurrent.selected->spec.tileSizes,
            baseline.selected->spec.tileSizes);
}

TEST_F(CandidateSearchExecutionTest,
       FindsOrdinalOneAfterEarlierFailureAndMatchesParallelImport) {
  SelectionRun serial =
      select(/*taskAlternativeOrdinal=*/1, /*candidateParallelism=*/1);
  ASSERT_TRUE(serial.selected);
  ASSERT_TRUE(serial.selected->module);
  EXPECT_EQ(serial.selected->candidateCount, 6);
  EXPECT_EQ(serial.selected->completeEvaluationCount, 5);
  EXPECT_EQ(serial.selected->rejectedCount, 2);

  SelectionRun parallel =
      select(/*taskAlternativeOrdinal=*/1, /*candidateParallelism=*/4);
  ASSERT_TRUE(parallel.selected);
  ASSERT_TRUE(parallel.selected->module);
  EXPECT_EQ(parallel.selected->rejectedCount, 2);
  EXPECT_EQ(parallel.selected->candidateCount,
            serial.selected->candidateCount);
  EXPECT_EQ(parallel.selected->completeEvaluationCount,
            serial.selected->completeEvaluationCount);
  EXPECT_EQ(parallel.selected->spec.tileSizes, serial.selected->spec.tileSizes);
  EXPECT_EQ(parallel.selected->spec.reductionSplitSizes,
            serial.selected->spec.reductionSplitSizes);
  EXPECT_EQ(printModule(*parallel.selected->module),
            printModule(*serial.selected->module));
}

TEST_F(CandidateSearchExecutionTest,
       ReusesBoundedWorkerContextsAcrossTaskSelections) {
  CandidateEvaluationExecutor executor(/*workerCount=*/2);
  EXPECT_EQ(executor.getWorkerConstructionCount(), 0u);
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    SelectionRun run = select(/*taskAlternativeOrdinal=*/1,
                              /*candidateParallelism=*/2, &executor);
    ASSERT_TRUE(run.selected);
    ASSERT_TRUE(run.selected->module);
  }

  EXPECT_EQ(executor.getWorkerCount(), 2u);
  EXPECT_EQ(executor.getWorkerConstructionCount(), 2u);
  EXPECT_GE(executor.getContextConstructionCount(), 1u);
  EXPECT_LE(executor.getContextConstructionCount(), 2u);
  // Every request above has byte-identical standalone task IR. Each persistent
  // worker parses it at most once even though three selections and several
  // fixed batches were evaluated.
  EXPECT_GE(executor.getTaskParseCount(), 1u);
  EXPECT_LE(executor.getTaskParseCount(), 2u);
}

TEST(CandidateSearchExecutorTest, StartsLazilyAndBoundsWorkerResources) {
  CandidateEvaluationExecutor executor(/*workerCount=*/1000);
  EXPECT_EQ(executor.getWorkerCount(), 64u);
  EXPECT_EQ(executor.getWorkerConstructionCount(), 0u);
  EXPECT_EQ(executor.getContextConstructionCount(), 0u);
  EXPECT_EQ(executor.getTaskParseCount(), 0u);
}

} // namespace
