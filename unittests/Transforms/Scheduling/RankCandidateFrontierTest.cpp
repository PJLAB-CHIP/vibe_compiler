//===- RankCandidateFrontierTest.cpp - Actual-clone frontier tests ------===//

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/PhysicalDataflow.h"
#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

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
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <optional>
#include <set>

namespace {

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
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GE(frontier->size(), 2u);
  EXPECT_LE(frontier->size(), 257u);

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
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_LE(frontier->size(), 257u);
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
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_LE(frontier->size(), 257u);

  std::set<int64_t> sourceBands;
  std::set<std::pair<unsigned, unsigned>> addMulCounts;
  bool sawSourceRecipeJointState = false;
  bool sawSourcePolicyJointState = false;
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    // Source recipes use a stable private ordinal band. Reaching four
    // non-baseline bands proves reassociation, tree balancing, distribution,
    // and factorization clones survived the complete rank gates; the winner
    // never reads these ordinals as mechanism facts.
    sourceBands.insert(candidate.stableOrdinal / 48);
    int64_t sourceIndex = candidate.stableOrdinal / 48;
    int64_t recipeIndex = (candidate.stableOrdinal / 4) % 12;
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
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_LE(frontier->size(), 257u);

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
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_LE(frontier->size(), 257u);

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
    // The reciprocal implementation occupies recipe 1. Recipe 2 requests
    // the second complete task candidate, and each recipe has four scope
    // policies in the invocation-local ordinal layout.
    sawSecondTaskCandidate |= candidate.stableOrdinal == 8;
  }
  EXPECT_TRUE(sawReciprocal);
  EXPECT_TRUE(sawDivision);
  EXPECT_TRUE(sawSecondTaskCandidate);
}

} // namespace
