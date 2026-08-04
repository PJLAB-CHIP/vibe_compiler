//===- RankCandidateFrontierTest.cpp - Actual-clone frontier tests ------===//

#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"
#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Testing.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace {

constexpr wafer::TargetProfileId kTargetProfile =
    wafer::TargetProfileId::waferTx81SingleCardKernelV1();

using RankCandidateSignature =
    std::tuple<int64_t, wafer::RankArtifactKind, bool, wafer::RankBufferingKind,
               uint32_t, wafer::RankWorkerPlacementKind, uint32_t, std::string>;

static RankCandidateSignature
getRankCandidateSignature(wafer::ScheduledRankCandidate &candidate) {
  std::string moduleText;
  llvm::raw_string_ostream os(moduleText);
  candidate.module->print(os);
  os.flush();
  return {candidate.stableOrdinal,
          candidate.artifactKind,
          candidate.reservedBaseline,
          candidate.bufferingKind,
          candidate.bufferingPlanOrdinal,
          candidate.workerPlacementKind,
          candidate.workerPlacementPlanOrdinal,
          std::move(moduleText)};
}

static std::vector<RankCandidateSignature> getRankCandidateSignatures(
    std::vector<wafer::ScheduledRankCandidate> &frontier) {
  std::vector<RankCandidateSignature> signatures;
  signatures.reserve(frontier.size());
  for (wafer::ScheduledRankCandidate &candidate : frontier)
    signatures.push_back(getRankCandidateSignature(candidate));
  return signatures;
}

TEST(RankCandidateFrontierTest,
     DetectsExactlyTheTypedCollectiveRankDependency) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect, wafer::WaferDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> rankInvariant =
      mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
  ASSERT_TRUE(rankInvariant);
  EXPECT_TRUE(wafer::isTensorProgramSchedulingRankInvariant(*rankInvariant));

  mlir::OwningOpRef<mlir::ModuleOp> rankDependent =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<1xf16>) -> tensor<1xf16> {
    %out = tensor.empty() : tensor<1xf16>
    %result = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<1xf16>)
        outs(%out : tensor<1xf16>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %sum = arith.addf %lhs, %rhs : f16
      wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 1 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<1xf16>
    return %result : tensor<1xf16>
  }
}
)mlir",
                                              &context);
  ASSERT_TRUE(rankDependent);
  EXPECT_FALSE(wafer::isTensorProgramSchedulingRankInvariant(*rankDependent));
}

TEST(RankCandidateFrontierTest,
     CanonicalRequestShardMergeMatchesUnshardedFrontier) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<4x6xf16>, %rhs: tensor<4x6xf16>,
                  %out: tensor<4x6xf16>) -> tensor<4x6xf16> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%lhs, %rhs : tensor<4x6xf16>, tensor<4x6xf16>)
        outs(%out : tensor<4x6xf16>) {
    ^bb0(%left: f16, %right: f16, %old: f16):
      %sum = arith.addf %left, %right : f16
      linalg.yield %sum : f16
    } -> tensor<4x6xf16>
    return %result : tensor<4x6xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig unshardedConfig;
  unshardedConfig.logicalRank = 0;
  unshardedConfig.candidateParallelism = 1;
  unshardedConfig.targetProfile = kTargetProfile;
  auto unsharded =
      wafer::buildScheduledRankCandidateFrontier(*source, unshardedConfig);
  ASSERT_TRUE(mlir::succeeded(unsharded));
  std::vector<RankCandidateSignature> expected =
      getRankCandidateSignatures(*unsharded);

  constexpr uint32_t requestShardCount = 16;
  std::vector<wafer::ScheduledRankCandidate> shardCandidates;
  for (uint32_t shardIndex = 0; shardIndex < requestShardCount; ++shardIndex) {
    wafer::TensorProgramSchedulingConfig shardConfig = unshardedConfig;
    shardConfig.requestShardIndex = shardIndex;
    shardConfig.requestShardCount = requestShardCount;
    auto shard =
        wafer::buildScheduledRankCandidateFrontier(*source, shardConfig);
    ASSERT_TRUE(mlir::succeeded(shard));
    for (wafer::ScheduledRankCandidate &candidate : *shard)
      shardCandidates.push_back(std::move(candidate));
  }
  std::stable_sort(shardCandidates.begin(), shardCandidates.end(),
                   [](const wafer::ScheduledRankCandidate &lhs,
                      const wafer::ScheduledRankCandidate &rhs) {
                     return lhs.frontierOrderOrdinal < rhs.frontierOrderOrdinal;
                   });

  wafer::RankFrontierAdmissionState admission;
  std::vector<wafer::ScheduledRankCandidate> merged;
  for (wafer::ScheduledRankCandidate &candidate : shardCandidates) {
    if (!candidate.reservedBaseline &&
        !admission.tryAdmit(candidate.bufferingKind,
                            candidate.workerPlacementKind))
      continue;
    merged.push_back(std::move(candidate));
  }

  EXPECT_EQ(getRankCandidateSignatures(merged), expected);
}

TEST(RankCandidateFrontierTest,
     NoOptionalOptimizationsRetainsOnlyFullyGatedReservedBaseline) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<4x6xf16>, %rhs: tensor<4x6xf16>,
                  %out: tensor<4x6xf16>) -> tensor<4x6xf16> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%lhs, %rhs : tensor<4x6xf16>, tensor<4x6xf16>)
        outs(%out : tensor<4x6xf16>) {
    ^bb0(%left: f16, %right: f16, %old: f16):
      %sum = arith.addf %left, %right : f16
      linalg.yield %sum : f16
    } -> tensor<4x6xf16>
    return %result : tensor<4x6xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  config.optimizations = wafer::OptimizationConfig::none();
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_EQ(frontier->size(), 1u);
  EXPECT_TRUE(frontier->front().reservedBaseline);
  EXPECT_EQ(frontier->front().artifactKind, wafer::RankArtifactKind::Spill);
  EXPECT_EQ(frontier->front().bufferingKind, wafer::RankBufferingKind::Single);
  EXPECT_EQ(frontier->front().workerPlacementKind,
            wafer::RankWorkerPlacementKind::Unplaced);
}

TEST(RankCandidateFrontierTest,
     RejectsMissingTargetProfileBeforeCandidateAnalysis) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::parseSourceString<mlir::ModuleOp>("module {}", &context);
  ASSERT_TRUE(source);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  EXPECT_TRUE(mlir::failed(
      wafer::buildScheduledRankCandidateFrontier(*source, config)));
  EXPECT_NE(diagnostics.find("target-profile must be explicitly provided"),
            std::string::npos)
      << diagnostics;
}

TEST(RankCandidateFrontierTest,
     AcceptsOperandInterfaceRecipeWithoutReplacingResultSearchPhase) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<4x6xf32>, %rhs: tensor<4x6xf32>,
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
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));

  // Result recipes occupy 12 slots for each of at most 16 source variants,
  // with four scope policies per slot. Interface traversal starts in the next
  // private ordinal band so it cannot replace a result request or masquerade
  // as one of its physical derivations.
  constexpr int64_t interfaceOrdinalBase = 16 * 12 * 4;
  bool sawResultRecipe = false;
  bool sawInterfaceRecipe = false;
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    ASSERT_TRUE(candidate.module);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*candidate.module)));
    if (candidate.reservedBaseline)
      continue;
    sawResultRecipe |= candidate.stableOrdinal < interfaceOrdinalBase;
    sawInterfaceRecipe |= candidate.stableOrdinal >= interfaceOrdinalBase;
  }
  EXPECT_TRUE(sawResultRecipe);
  EXPECT_TRUE(sawInterfaceRecipe);

  bool sourceContainsTileRegion = false;
  source->walk([&](wafer::TileRegionOp) { sourceContainsTileRegion = true; });
  EXPECT_FALSE(sourceContainsTileRegion);
}

TEST(RankCandidateFrontierTest,
     IndependentAdmissionBandsSurviveGeneralFrontierSaturation) {
  wafer::RankFrontierAdmissionState admission;
  for (unsigned index = 0; index < wafer::kGeneralRankFrontierAdmissionLimit;
       ++index)
    EXPECT_TRUE(admission.tryAdmit(wafer::RankBufferingKind::Single,
                                   wafer::RankWorkerPlacementKind::Unplaced));
  EXPECT_FALSE(admission.tryAdmit(wafer::RankBufferingKind::Single,
                                  wafer::RankWorkerPlacementKind::Unplaced));

  // Both orthogonal realizations remain admissible after the general band is
  // saturated; neither relies on generation order or unused general capacity.
  EXPECT_TRUE(admission.tryAdmit(wafer::RankBufferingKind::StaticFixedSlot,
                                 wafer::RankWorkerPlacementKind::Unplaced));
  EXPECT_TRUE(
      admission.tryAdmit(wafer::RankBufferingKind::Single,
                         wafer::RankWorkerPlacementKind::DisjointComponents));
  EXPECT_EQ(admission.getGeneralCount(),
            wafer::kGeneralRankFrontierAdmissionLimit);
  EXPECT_EQ(admission.getFixedSlotCount(), 1u);
  EXPECT_EQ(admission.getWorkerPlacementCount(), 1u);
  EXPECT_FALSE(admission.allBandsFull());

  for (unsigned index = 1; index < wafer::kFixedSlotRankFrontierAdmissionLimit;
       ++index)
    EXPECT_TRUE(admission.tryAdmit(wafer::RankBufferingKind::StaticFixedSlot,
                                   wafer::RankWorkerPlacementKind::Unplaced));
  for (unsigned index = 1;
       index < wafer::kWorkerPlacementRankFrontierAdmissionLimit; ++index)
    EXPECT_TRUE(
        admission.tryAdmit(wafer::RankBufferingKind::StaticFixedSlot,
                           wafer::RankWorkerPlacementKind::DisjointComponents));
  EXPECT_TRUE(admission.allBandsFull());
  EXPECT_FALSE(
      admission.tryAdmit(wafer::RankBufferingKind::Single,
                         wafer::RankWorkerPlacementKind::DisjointComponents));
}

TEST(RankCandidateFrontierTest,
     BuildsWorkerPlacementThroughProductionRankFrontier) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%a: tensor<1024xf32>, %b: tensor<1024xf32>,
                  %c: tensor<1024xf32>) -> tensor<1024xf32> {
    %init = tensor.empty() : tensor<1024xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c
          : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)
      outs(%init : tensor<1024xf32>) {
    ^bb0(%a_value: f32, %b_value: f32, %c_value: f32, %unused: f32):
      %ab = arith.addf %a_value, %b_value : f32
      %abc = arith.addf %ab, %c_value : f32
      linalg.yield %abc : f32
    } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = wafer::TargetProfileId::waferTx81SingleCardKernelV3();
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_LE(frontier->size(), wafer::kMaximumScheduledRankFrontierSize);

  unsigned generalCount = 0;
  unsigned workerCount = 0;
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    if (candidate.reservedBaseline)
      continue;
    if (candidate.workerPlacementKind ==
        wafer::RankWorkerPlacementKind::DisjointComponents) {
      ++workerCount;
      EXPECT_GT(candidate.workerPlacementPlanOrdinal, 0u);
      continue;
    }
    if (candidate.bufferingKind == wafer::RankBufferingKind::Single)
      ++generalCount;
  }
  EXPECT_GT(generalCount, 0u);
  EXPECT_GT(workerCount, 0u);
  EXPECT_LE(workerCount, wafer::kWorkerPlacementRankFrontierAdmissionLimit);
}

TEST(RankCandidateFrontierTest,
     PreservesReservedSpillAndResidentAsIndependentPlacedClones) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"],
    endpoints = array<i64: 0, 0, 0, 0, 0, 0, 0, 1>,
    policy = "explicit", shape = array<i64: 2>, topology = @default
  }
  func.func @main(%lhs: tensor<256x128xf16>,
                  %rhs: tensor<128x1024xf16>) -> tensor<256x1024xf16> {
    %zero = arith.constant 0.0 : f16
    %matmul_empty = tensor.empty() : tensor<256x1024xf16>
    %matmul_init = linalg.fill ins(%zero : f16)
        outs(%matmul_empty : tensor<256x1024xf16>) -> tensor<256x1024xf16>
    %matmul = linalg.matmul
        ins(%lhs, %rhs : tensor<256x128xf16>, tensor<128x1024xf16>)
        outs(%matmul_init : tensor<256x1024xf16>) -> tensor<256x1024xf16>
    %collective_out = tensor.empty() : tensor<256x1024xf16>
    %collective = wafer.linalg_ext.collective.all_reduce
        ins(%matmul : tensor<256x1024xf16>)
        outs(%collective_out : tensor<256x1024xf16>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16):
      %sum = arith.addf %lhs_value, %rhs_value : f16
      wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 43 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<256x1024xf16>
    return %collective : tensor<256x1024xf16>
  }
}
)mlir",
      &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GE(frontier->size(), 2u);
  EXPECT_LE(frontier->size(), wafer::kMaximumScheduledRankFrontierSize);

  unsigned baselineCount = 0;
  std::optional<unsigned> baselineGatherScatterCount;
  std::optional<unsigned> optimizedMinimumGatherScatterCount;
  bool sawResidentAlternative = false;
  bool sawRingAllReduce = false;
  bool sawTreeReduce = false;
  bool sawTreeBroadcast = false;
  auto recordCommunicationPhase = [&](wafer::DTEProtocolPhase phase) {
    sawRingAllReduce |= phase == wafer::DTEProtocolPhase::AllReduceRing;
    sawTreeReduce |= phase == wafer::DTEProtocolPhase::AllReduceTreeReduce;
    sawTreeBroadcast |=
        phase == wafer::DTEProtocolPhase::AllReduceTreeBroadcast;
  };
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    baselineCount += candidate.reservedBaseline;
    unsigned gatherScatterCount = 0;
    candidate.module->walk(
        [&](wafer::InstrGatherScatterOp) { ++gatherScatterCount; });
    if (candidate.reservedBaseline) {
      EXPECT_EQ(candidate.artifactKind, wafer::RankArtifactKind::Spill);
      EXPECT_EQ(candidate.bufferingKind, wafer::RankBufferingKind::Single);
      EXPECT_EQ(candidate.bufferingPlanOrdinal, 0u);
      baselineGatherScatterCount = gatherScatterCount;
    } else if (!optimizedMinimumGatherScatterCount ||
               gatherScatterCount < *optimizedMinimumGatherScatterCount) {
      optimizedMinimumGatherScatterCount = gatherScatterCount;
    }
    bool candidateResident = false;
    candidate.module->walk([&](wafer::TileRegionOp region) {
      candidateResident |=
          llvm::any_of(region.getResultTypes(), [](mlir::Type type) {
            return wafer::isWaferSPMMemRefType(type);
          });
    });
    sawResidentAlternative |= !candidate.reservedBaseline && candidateResident;

    candidate.module->walk([&](mlir::memref::AllocOp allocation) {
      if (wafer::isWaferSPMMemRefType(allocation.getType()))
        EXPECT_TRUE(allocation->hasAttr(wafer::kWaferSPMOffsetAttrName));
      if (wafer::isWaferDDRMemRefType(allocation.getType()))
        EXPECT_FALSE(allocation->hasAttr(wafer::kWaferDDROffsetAttrName));
    });
    candidate.module->walk([&](wafer::InstrDTESendOp send) {
      EXPECT_FALSE(send.getBinding().has_value());
      recordCommunicationPhase(send.getMessage().getPhase());
    });
    candidate.module->walk([&](wafer::InstrDTERecvOp recv) {
      EXPECT_FALSE(recv.getBinding().has_value());
      recordCommunicationPhase(recv.getMessage().getPhase());
    });
  }
  EXPECT_EQ(baselineCount, 1u);
  ASSERT_TRUE(baselineGatherScatterCount.has_value());
  ASSERT_TRUE(optimizedMinimumGatherScatterCount.has_value());
  EXPECT_GT(*baselineGatherScatterCount, *optimizedMinimumGatherScatterCount);
  EXPECT_TRUE(sawResidentAlternative);
  EXPECT_TRUE(sawRingAllReduce);
  EXPECT_TRUE(sawTreeReduce);
  EXPECT_TRUE(sawTreeBroadcast);

  config.candidateParallelism = 4;
  auto parallelFrontier =
      wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(parallelFrontier));
  ASSERT_EQ(parallelFrontier->size(), frontier->size());
  for (auto &&[serial, parallel] :
       llvm::zip_equal(*frontier, *parallelFrontier)) {
    EXPECT_EQ(serial.stableOrdinal, parallel.stableOrdinal);
    EXPECT_EQ(serial.artifactKind, parallel.artifactKind);
    EXPECT_EQ(serial.bufferingKind, parallel.bufferingKind);
    EXPECT_EQ(serial.bufferingPlanOrdinal, parallel.bufferingPlanOrdinal);
    EXPECT_EQ(serial.reservedBaseline, parallel.reservedBaseline);
    std::string serialText;
    llvm::raw_string_ostream serialStream(serialText);
    serial.module->print(serialStream);
    std::string parallelText;
    llvm::raw_string_ostream parallelStream(parallelText);
    parallel.module->print(parallelStream);
    EXPECT_EQ(serialStream.str(), parallelStream.str());
  }

  // Candidate generation and evaluation never mutate the shared source.
  bool sourceContainsTileRegion = false;
  source->walk([&](wafer::TileRegionOp) { sourceContainsTileRegion = true; });
  EXPECT_FALSE(sourceContainsTileRegion);
}

TEST(RankCandidateFrontierTest,
     SendsLoopInvariantActualSourceCloneThroughRankExactGates) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<4xf32>, %initial: tensor<4xf32>)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result = scf.for %iv = %c0 to %c2 step %c1
        iter_args(%iter = %initial) -> tensor<4xf32> {
      %empty = tensor.empty() : tensor<4xf32>
      %invariant = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        linalg.yield %value : f32
      } -> tensor<4xf32>
      scf.yield %invariant : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_EQ(llvm::count_if(*frontier,
                           [](const auto &candidate) {
                             return candidate.reservedBaseline;
                           }),
            1u);
  EXPECT_TRUE(llvm::any_of(*frontier, [](const auto &candidate) {
    return candidate.stableOrdinal >= 48;
  }));
  // Both generation paths are isolated from the shared source.
  bool sourceContainsTileRegion = false;
  source->walk([&](wafer::TileRegionOp) { sourceContainsTileRegion = true; });
  EXPECT_FALSE(sourceContainsTileRegion);
}

TEST(RankCandidateFrontierTest,
     BuildsLongSteadyPlacedStaticFixedSlotsFromProductionTensorProgram) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<8388608xf16>, %rhs: tensor<8388608xf16>)
      -> tensor<8388608xf16> {
    %out = tensor.empty() : tensor<8388608xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%lhs, %rhs : tensor<8388608xf16>, tensor<8388608xf16>)
      outs(%out : tensor<8388608xf16>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16, %unused: f16):
      %value = arith.addf %lhs_value, %rhs_value : f16
      linalg.yield %value : f16
    } -> tensor<8388608xf16>
    return %sum : tensor<8388608xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);
  std::string sourceBefore;
  llvm::raw_string_ostream sourceBeforeStream(sourceBefore);
  source->print(sourceBeforeStream);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto serialFrontier =
      wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(serialFrontier));

  wafer::ScheduledRankCandidate *fixed = nullptr;
  for (wafer::ScheduledRankCandidate &candidate : *serialFrontier) {
    if (!candidate.reservedBaseline &&
        candidate.bufferingKind == wafer::RankBufferingKind::StaticFixedSlot &&
        candidate.bufferingPlanOrdinal > 0) {
      fixed = &candidate;
      break;
    }
  }
  ASSERT_NE(fixed, nullptr);

  llvm::SmallVector<mlir::scf::ForOp, 2> loops;
  fixed->module->walk([&](mlir::scf::ForOp loop) { loops.push_back(loop); });
  ASSERT_EQ(loops.size(), 1u);
  mlir::scf::ForOp steady = loops.front();
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(steady.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(steady.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(steady.getStep());
  ASSERT_TRUE(lower);
  ASSERT_TRUE(upper);
  ASSERT_TRUE(step);
  ASSERT_GT(*step, 0);
  ASSERT_GT(*upper, *lower);
  EXPECT_GE((*upper - *lower + *step - 1) / *step, 32);

  unsigned rdmaCount = 0;
  unsigned computeCount = 0;
  unsigned wdmaCount = 0;
  unsigned joinCount = 0;
  steady.walk([&](wafer::InstrRDMAOp) { ++rdmaCount; });
  steady.walk([&](wafer::InstrElementwiseOp) { ++computeCount; });
  steady.walk([&](wafer::InstrWDMAOp) { ++wdmaCount; });
  fixed->module->walk([&](wafer::SyncNCCJoinOp) { ++joinCount; });
  EXPECT_EQ(rdmaCount, 2u);
  EXPECT_EQ(computeCount, 1u);
  EXPECT_EQ(wdmaCount, 1u);
  EXPECT_EQ(joinCount, 1u);

  unsigned nestedAllocations = 0;
  unsigned externalSPMAllocations = 0;
  std::set<int64_t> slotOffsets;
  fixed->module->walk([&](mlir::memref::AllocOp allocation) {
    if (!wafer::isWaferSPMMemRefType(allocation.getType()))
      return;
    if (allocation->getParentOfType<mlir::scf::ForOp>()) {
      ++nestedAllocations;
      return;
    }
    ++externalSPMAllocations;
    auto offset = allocation->getAttrOfType<wafer::SPMOffsetAttr>(
        wafer::kWaferSPMOffsetAttrName);
    ASSERT_TRUE(offset);
    EXPECT_GE(offset.getOffset(), 65536);
    EXPECT_LT(offset.getOffset(), 3080192);
    slotOffsets.insert(offset.getOffset());
  });
  EXPECT_EQ(nestedAllocations, 0u);
  EXPECT_EQ(externalSPMAllocations, 6u);
  EXPECT_EQ(slotOffsets.size(), 6u);

  config.candidateParallelism = 4;
  auto parallelFrontier =
      wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(parallelFrontier));
  ASSERT_EQ(parallelFrontier->size(), serialFrontier->size());
  for (auto &&[serial, parallel] :
       llvm::zip_equal(*serialFrontier, *parallelFrontier)) {
    EXPECT_EQ(serial.stableOrdinal, parallel.stableOrdinal);
    EXPECT_EQ(serial.artifactKind, parallel.artifactKind);
    EXPECT_EQ(serial.bufferingKind, parallel.bufferingKind);
    EXPECT_EQ(serial.bufferingPlanOrdinal, parallel.bufferingPlanOrdinal);
    EXPECT_EQ(serial.reservedBaseline, parallel.reservedBaseline);
    std::string serialText;
    llvm::raw_string_ostream serialStream(serialText);
    serial.module->print(serialStream);
    std::string parallelText;
    llvm::raw_string_ostream parallelStream(parallelText);
    parallel.module->print(parallelStream);
    EXPECT_EQ(serialStream.str(), parallelStream.str());
  }

  std::string sourceAfter;
  llvm::raw_string_ostream sourceAfterStream(sourceAfter);
  source->print(sourceAfterStream);
  EXPECT_EQ(sourceAfterStream.str(), sourceBeforeStream.str());
  bool sourceContainsTileRegion = false;
  source->walk([&](wafer::TileRegionOp) { sourceContainsTileRegion = true; });
  EXPECT_FALSE(sourceContainsTileRegion);
}

TEST(RankCandidateFrontierTest,
     SendsDirectAndRingAllGatherClonesThroughTheSameRankFrontier) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"],
    endpoints = array<i64: 0, 0, 0, 0, 0, 0, 0, 1>,
    policy = "explicit", shape = array<i64: 2>, topology = @default
  }
  func.func @main(%input: tensor<4xf32>) -> tensor<8xf32> {
    %out = tensor.empty() : tensor<8xf32>
    %gathered = wafer.linalg_ext.collective.all_gather
        ins(%input : tensor<4xf32>) outs(%out : tensor<8xf32>)
        {axis = 0 : i64, channel_id = 47 : i64,
         rank_group = array<i64: 0, 1>} -> tensor<8xf32>
    return %gathered : tensor<8xf32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_LE(frontier->size(), wafer::kMaximumScheduledRankFrontierSize);
  EXPECT_EQ(llvm::count_if(*frontier,
                           [](const auto &candidate) {
                             return candidate.reservedBaseline;
                           }),
            1u);

  bool sawRing = false;
  bool sawDirect = false;
  auto recordPhase = [&](wafer::DTEProtocolPhase phase) {
    sawRing |= phase == wafer::DTEProtocolPhase::AllGatherRing;
    sawDirect |= phase == wafer::DTEProtocolPhase::AllGatherDirect;
  };
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    candidate.module->walk([&](wafer::InstrDTESendOp send) {
      recordPhase(send.getMessage().getPhase());
    });
    candidate.module->walk([&](wafer::InstrDTERecvOp recv) {
      recordPhase(recv.getMessage().getPhase());
    });
  }
  EXPECT_TRUE(sawRing);
  EXPECT_TRUE(sawDirect);
}

TEST(RankCandidateFrontierTest,
     SendsEveryModularAlgebraSourceCloneThroughRankExactGates) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @reassociate(%a: tensor<4xi32>, %b: tensor<4xi32>,
                         %c: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %unused: i32):
      %ab = arith.addi %av, %bv : i32
      %abc = arith.addi %ab, %cv : i32
      linalg.yield %abc : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  func.func @tree(%a: tensor<4xi32>, %b: tensor<4xi32>,
                  %c: tensor<4xi32>, %d: tensor<4xi32>,
                  %out: tensor<4xi32>) -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c, %d : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>,
                            tensor<4xi32>) outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %dv: i32, %unused: i32):
      %ab = arith.addi %av, %bv : i32
      %abc = arith.addi %ab, %cv : i32
      %abcd = arith.addi %abc, %dv : i32
      linalg.yield %abcd : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  func.func @distribute(%a: tensor<4xi32>, %b: tensor<4xi32>,
                        %c: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %unused: i32):
      %ab = arith.muli %av, %bv : i32
      %ac = arith.muli %av, %cv : i32
      %r0 = arith.subi %ab, %ac : i32
      linalg.yield %r0 : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  func.func @factor(%a: tensor<4xi32>, %b: tensor<4xi32>,
                    %c: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %unused: i32):
      %ab = arith.muli %av, %bv : i32
      %ac = arith.muli %av, %cv : i32
      %r0 = arith.addi %ab, %ac : i32
      linalg.yield %r0 : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_LE(frontier->size(), wafer::kMaximumScheduledRankFrontierSize);

  std::set<int64_t> sourceBands;
  std::set<std::pair<unsigned, unsigned>> addMulCounts;
  bool sawSourceRecipeJointState = false;
  bool sawSourcePolicyJointState = false;
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    // Source recipes use a stable private ordinal band. Reaching four
    // non-baseline bands proves reassociation, tree balancing, distribution,
    // and factorization clones survived the complete rank gates; the winner
    // never reads these ordinals as mechanism facts.
    sourceBands.insert(candidate.stableOrdinal / 144);
    int64_t sourceIndex = candidate.stableOrdinal / 144;
    int64_t recipeIndex = (candidate.stableOrdinal / 4) % 36;
    int64_t policyIndex = candidate.stableOrdinal % 4;
    sawSourceRecipeJointState |= sourceIndex > 0 && recipeIndex > 0;
    sawSourcePolicyJointState |= sourceIndex > 0 && policyIndex > 0;
    unsigned adds = 0;
    unsigned multiplies = 0;
    candidate.module->walk([&](wafer::InstrElementwiseOp elementwise) {
      adds += elementwise.getKind() == wafer::InstrElementwiseKind::Add;
      multiplies += elementwise.getKind() == wafer::InstrElementwiseKind::Mul;
    });
    addMulCounts.insert({adds, multiplies});
  }
  EXPECT_GE(sourceBands.size(), 5u);
  EXPECT_TRUE(addMulCounts.count({6, 4}));
  EXPECT_TRUE(addMulCounts.count({6, 3}));
  EXPECT_TRUE(addMulCounts.count({6, 2}));
  EXPECT_TRUE(sawSourceRecipeJointState);
  EXPECT_TRUE(sawSourcePolicyJointState);
}

TEST(RankCandidateFrontierTest,
     SendsUnannotatedF16FactorCloneThroughRankExactGates) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%a: tensor<16xf16>, %b: tensor<16xf16>,
                  %c: tensor<16xf16>) -> tensor<16xf16> {
    %out = tensor.empty() : tensor<16xf16>
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<16xf16>, tensor<16xf16>, tensor<16xf16>)
      outs(%out : tensor<16xf16>) {
    ^bb0(%av: f16, %bv: f16, %cv: f16, %unused: f16):
      %ab = arith.mulf %av, %bv : f16
      %ac = arith.mulf %av, %cv : f16
      %result = arith.addf %ab, %ac : f16
      linalg.yield %result : f16
    } -> tensor<16xf16>
    return %r : tensor<16xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_LE(frontier->size(), wafer::kMaximumScheduledRankFrontierSize);

  bool sawBaseline = false;
  bool sawFactored = false;
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    unsigned adds = 0;
    unsigned multiplies = 0;
    candidate.module->walk([&](wafer::InstrElementwiseOp elementwise) {
      adds += elementwise.getKind() == wafer::InstrElementwiseKind::Add;
      multiplies += elementwise.getKind() == wafer::InstrElementwiseKind::Mul;
    });
    sawBaseline |= adds == 1 && multiplies == 2;
    sawFactored |= adds == 1 && multiplies == 1;
  }
  EXPECT_TRUE(sawBaseline);
  EXPECT_TRUE(sawFactored);
}

TEST(RankCandidateFrontierTest,
     SendsShareAndRecomputeActualClonesThroughRankExactGates) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<4xi32>, %producer_out: tensor<4xi32>,
                  %left_out: tensor<4xi32>, %right_out: tensor<4xi32>)
      -> (tensor<4xi32>, tensor<4xi32>) {
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<4xi32>) outs(%producer_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32):
      %incremented = arith.addi %value, %value : i32
      linalg.yield %incremented : i32
    } -> tensor<4xi32>
    %left = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%left_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32):
      %result = arith.addi %value, %value : i32
      linalg.yield %result : i32
    } -> tensor<4xi32>
    %right = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%right_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32):
      %result = arith.addi %value, %value : i32
      linalg.yield %result : i32
    } -> tensor<4xi32>
    return %left, %right : tensor<4xi32>, tensor<4xi32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_LE(frontier->size(), wafer::kMaximumScheduledRankFrontierSize);

  std::set<unsigned> elementwiseCounts;
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    unsigned count = 0;
    candidate.module->walk([&](wafer::InstrElementwiseOp) { ++count; });
    elementwiseCounts.insert(count);
  }
  EXPECT_TRUE(elementwiseCounts.count(3));
  EXPECT_TRUE(elementwiseCounts.count(4));
}

TEST(RankCandidateFrontierTest,
     SendsImplementationAndTaskBeamClonesThroughRankExactGates) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<16xf32>, %out: tensor<16xf32>)
      -> tensor<16xf32> {
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<16xf32>) outs(%out : tensor<16xf32>) {
    ^bb0(%value: f32, %init: f32):
      %one = arith.constant 1.0 : f32
      %reciprocal = arith.divf %one, %value : f32
      linalg.yield %reciprocal : f32
    } -> tensor<16xf32>
    return %result : tensor<16xf32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));

  bool sawReciprocal = false;
  bool sawDivision = false;
  bool sawSecondTaskCandidate = false;
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    candidate.module->walk([&](wafer::InstrElementwiseOp elementwise) {
      sawReciprocal |=
          elementwise.getKind() == wafer::InstrElementwiseKind::Recip;
      sawDivision |= elementwise.getKind() == wafer::InstrElementwiseKind::Div;
    });
    // The reciprocal implementation occupies recipe 1. Four bounded physical
    // layout recipes occupy recipes 2..5 and four implementation/layout joint
    // recipes occupy recipes 6..9, so recipe 10 requests the second complete
    // task candidate. Each recipe has four scope policies in the
    // invocation-local ordinal layout.
    sawSecondTaskCandidate |= candidate.stableOrdinal == 40;
  }
  EXPECT_TRUE(sawReciprocal);
  EXPECT_TRUE(sawDivision);
  EXPECT_TRUE(sawSecondTaskCandidate);
}

TEST(RankCandidateFrontierTest,
     AdmitsMovementReducedPhysicalLayoutActualClone) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %lhs: tensor<4x64xf16>, %rhs0: tensor<64x64xf16>,
      %rhs1: tensor<64x64xf16>, %out: tensor<4x64xf16>)
      -> tensor<4x64xf16> {
    %zero = arith.constant 0.0 : f16
    %first_empty = tensor.empty() : tensor<4x64xf16>
    %first_init = linalg.fill ins(%zero : f16)
        outs(%first_empty : tensor<4x64xf16>) -> tensor<4x64xf16>
    %first = linalg.matmul
        ins(%lhs, %rhs0 : tensor<4x64xf16>, tensor<64x64xf16>)
        outs(%first_init : tensor<4x64xf16>) -> tensor<4x64xf16>
    %point_empty = tensor.empty() : tensor<4x64xf16>
    %point = linalg.generic {
        indexing_maps = [affine_map<(d0, d1)->(d0, d1)>,
                         affine_map<(d0, d1)->(d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%first : tensor<4x64xf16>)
      outs(%point_empty : tensor<4x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %one = arith.constant 1.0 : f16
      %reciprocal = arith.divf %one, %value : f16
      linalg.yield %reciprocal : f16
    } -> tensor<4x64xf16>
    %second_init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<4x64xf16>) -> tensor<4x64xf16>
    %second = linalg.matmul
        ins(%point, %rhs1 : tensor<4x64xf16>, tensor<64x64xf16>)
        outs(%second_init : tensor<4x64xf16>) -> tensor<4x64xf16>
    return %second : tensor<4x64xf16>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  config.targetProfile = kTargetProfile;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));

  std::optional<uint64_t> baselineMovement;
  std::optional<uint64_t> bestOptimizedMovement;
  std::optional<uint64_t> bestReciprocalMovement;
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    wafer::analysis::InstructionProgramCost cost =
        wafer::analysis::analyzeInstructionProgramCost(
            *candidate.module,
            wafer::analysis::getTargetScheduleCostPolicy(kTargetProfile));
    ASSERT_TRUE(cost.spmMovementBytes.isKnown());
    if (candidate.reservedBaseline) {
      baselineMovement = cost.spmMovementBytes.value;
      continue;
    }
    if (!bestOptimizedMovement ||
        cost.spmMovementBytes.value < *bestOptimizedMovement)
      bestOptimizedMovement = cost.spmMovementBytes.value;
    bool hasReciprocal = false;
    candidate.module->walk([&](wafer::InstrElementwiseOp elementwise) {
      hasReciprocal |=
          elementwise.getKind() == wafer::InstrElementwiseKind::Recip;
    });
    if (hasReciprocal &&
        (!bestReciprocalMovement ||
         cost.spmMovementBytes.value < *bestReciprocalMovement))
      bestReciprocalMovement = cost.spmMovementBytes.value;
  }
  ASSERT_TRUE(baselineMovement);
  ASSERT_TRUE(bestOptimizedMovement);
  ASSERT_TRUE(bestReciprocalMovement);
  EXPECT_LT(*bestOptimizedMovement, *baselineMovement);
  EXPECT_LT(*bestReciprocalMovement, *baselineMovement);
}

} // namespace
