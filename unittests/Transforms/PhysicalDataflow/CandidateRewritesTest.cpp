//===- CandidateRewritesTest.cpp - Actual-clone producer tests ---------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

class CandidateRewritesTest : public ::testing::Test {
protected:
  CandidateRewritesTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect, wafer::WaferDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);
    wafer::registerTargetImplementationExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  }

  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
};

TEST_F(CandidateRewritesTest, MaterializesConsumerLocalPureTensorClone) {
  auto module = parse(R"mlir(
module {
  func.func @fanout(%input: tensor<4xi32>, %producer_out: tensor<4xi32>,
                    %left_out: tensor<4xi32>, %right_out: tensor<4xi32>)
      -> (tensor<4xi32>, tensor<4xi32>) {
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<4xi32>) outs(%producer_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32):
      linalg.yield %value : i32
    } -> tensor<4xi32>
    %left = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%left_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32):
      linalg.yield %value : i32
    } -> tensor<4xi32>
    %right = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%right_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32):
      linalg.yield %value : i32
    } -> tensor<4xi32>
    return %left, %right : tensor<4xi32>, tensor<4xi32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("fanout");
  ASSERT_TRUE(function);

  EXPECT_EQ(wafer::materializeConsumerLocalTensorRecomputation(function), 1u);
  unsigned genericCount = 0;
  function.walk([&](mlir::linalg::GenericOp) { ++genericCount; });
  EXPECT_EQ(genericCount, 4u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest, KeepsProducerWithExternallyVisibleUseShared) {
  auto module = parse(R"mlir(
module {
  func.func @visible(%input: tensor<4xi32>, %producer_out: tensor<4xi32>,
                     %left_out: tensor<4xi32>, %right_out: tensor<4xi32>)
      -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<4xi32>) outs(%producer_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32): linalg.yield %value : i32
    } -> tensor<4xi32>
    %left = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%left_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32): linalg.yield %value : i32
    } -> tensor<4xi32>
    %right = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%right_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32): linalg.yield %value : i32
    } -> tensor<4xi32>
    return %producer, %left, %right : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("visible");
  EXPECT_EQ(wafer::materializeConsumerLocalTensorRecomputation(function), 0u);
}

TEST_F(CandidateRewritesTest,
       AppliesTypedPBQPLayoutProposalAndErasesActualMovements) {
  auto module = parse(R"mlir(
module {
  func.func @chain(%input: memref<3x65xf16, #wafer.memory<spm, cx>>)
      -> memref<3x65xf16, #wafer.memory<spm, cx>> {
    %tensor = wafer.tile.materialize_layout %input
        : memref<3x65xf16, #wafer.memory<spm, cx>>
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %first = wafer.tile.elementwise #wafer.elementwise_kind<neg> %tensor
        : (memref<3x65xf16, #wafer.memory<spm, tensor>>)
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %cx = wafer.tile.materialize_layout %first
        : memref<3x65xf16, #wafer.memory<spm, tensor>>
       -> memref<3x65xf16, #wafer.memory<spm, cx>>
    %second = wafer.tile.elementwise #wafer.elementwise_kind<neg> %cx
        : (memref<3x65xf16, #wafer.memory<spm, cx>>)
       -> memref<3x65xf16, #wafer.memory<spm, cx>>
    return %second : memref<3x65xf16, #wafer.memory<spm, cx>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::PhysicalLayoutProposalResult result;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/0, &result, &failureReason)))
      << failureReason;
  EXPECT_GE(result.proposalCount, 1u);
  EXPECT_EQ(result.movementCommandsBefore, 2u);
  EXPECT_EQ(result.movementCommandsAfter, 0u);
  EXPECT_GT(result.movementBytesBefore, result.movementBytesAfter);

  unsigned materializations = 0;
  unsigned elementwise = 0;
  module->walk([&](wafer::LayoutMaterializeOp) { ++materializations; });
  module->walk([&](wafer::ComputeElementwiseOp op) {
    ++elementwise;
    auto type = mlir::cast<mlir::MemRefType>(op.getResult().getType());
    EXPECT_EQ(wafer::getWaferMemoryAttr(type).getLayout(),
              wafer::MemLayout::Cx);
  });
  EXPECT_EQ(materializations, 0u);
  EXPECT_EQ(elementwise, 2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest,
       SharesOneFanoutSecondaryPhysicalVersionAcrossConsumers) {
  auto module = parse(R"mlir(
module {
  func.func @fanout(%input: memref<3x65xf16, #wafer.memory<spm, cx>>)
      -> (memref<3x65xf16, #wafer.memory<spm, cx>>,
          memref<3x65xf16, #wafer.memory<spm, tensor>>,
          memref<3x65xf16, #wafer.memory<spm, tensor>>) {
    %tensor = wafer.tile.materialize_layout %input
        : memref<3x65xf16, #wafer.memory<spm, cx>>
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %producer = wafer.tile.elementwise #wafer.elementwise_kind<neg> %tensor
        : (memref<3x65xf16, #wafer.memory<spm, tensor>>)
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %cx = wafer.tile.materialize_layout %producer
        : memref<3x65xf16, #wafer.memory<spm, tensor>>
       -> memref<3x65xf16, #wafer.memory<spm, cx>>
    %child = wafer.tile.elementwise #wafer.elementwise_kind<neg> %cx
        : (memref<3x65xf16, #wafer.memory<spm, cx>>)
       -> memref<3x65xf16, #wafer.memory<spm, cx>>
    %left = wafer.tile.copy %producer
        : memref<3x65xf16, #wafer.memory<spm, tensor>>
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %right = wafer.tile.copy %producer
        : memref<3x65xf16, #wafer.memory<spm, tensor>>
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    return %child, %left, %right
        : memref<3x65xf16, #wafer.memory<spm, cx>>,
          memref<3x65xf16, #wafer.memory<spm, tensor>>,
          memref<3x65xf16, #wafer.memory<spm, tensor>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::PhysicalLayoutProposalResult result;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/0, &result, &failureReason)))
      << failureReason;
  EXPECT_EQ(result.movementCommandsBefore, 4u);
  EXPECT_EQ(result.movementCommandsAfter, 3u);

  llvm::SmallVector<wafer::LayoutMaterializeOp, 2> materializations;
  llvm::SmallVector<wafer::MoveCopyOp, 2> copies;
  module->walk(
      [&](wafer::LayoutMaterializeOp op) { materializations.push_back(op); });
  module->walk([&](wafer::MoveCopyOp op) { copies.push_back(op); });
  ASSERT_EQ(materializations.size(), 1u);
  ASSERT_EQ(copies.size(), 2u);
  EXPECT_EQ(copies[0].getSource(), materializations.front().getResult());
  EXPECT_EQ(copies[1].getSource(), materializations.front().getResult());
  auto secondarySourceType = mlir::cast<mlir::MemRefType>(
      materializations.front().getSource().getType());
  EXPECT_EQ(wafer::getWaferMemoryAttr(secondarySourceType).getLayout(),
            wafer::MemLayout::Cx);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest,
       PhysicalLayoutProposalRunsThroughCompleteActualCloneGates) {
  auto module = parse(R"mlir(
module {
  func.func @chain(
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
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<4x64xf16>
    %second_init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<4x64xf16>) -> tensor<4x64xf16>
    %second = linalg.matmul
        ins(%point, %rhs1 : tensor<4x64xf16>, tensor<64x64xf16>)
        outs(%second_init : tensor<4x64xf16>) -> tensor<4x64xf16>
    return %second : tensor<4x64xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto task = module->lookupSymbol<mlir::func::FuncOp>("chain");
  ASSERT_TRUE(task);

  wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
  wafer::tensor_program_scheduling::SelectionConfig baselineConfig(
      policy, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  baselineConfig.logicalRank = 0;
  wafer::tensor_program_scheduling::CandidateSpec candidate;
  candidate.tileSizes = {4, 64};
  auto baseline = wafer::tensor_program_scheduling::evaluateCompleteCandidate(
      task, {4, 64}, candidate, baselineConfig);
  ASSERT_TRUE(baseline.failureReason.empty()) << baseline.failureReason;
  ASSERT_TRUE(baseline.stats.program.spmMovementBytes.isKnown());

  auto optimizedConfig = baselineConfig;
  optimizedConfig.physicalLayoutProposalOrdinal = 0;
  auto optimized = wafer::tensor_program_scheduling::evaluateCompleteCandidate(
      task, {4, 64}, candidate, optimizedConfig);
  ASSERT_TRUE(optimized.failureReason.empty()) << optimized.failureReason;
  ASSERT_TRUE(optimized.stats.program.spmMovementBytes.isKnown());
  EXPECT_LT(optimized.stats.program.spmMovementBytes.value,
            baseline.stats.program.spmMovementBytes.value);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*optimized.module)));
}

TEST_F(CandidateRewritesTest,
       BitpackedRelationAndSelectUseOneSharedBlockedTraversal) {
  auto module = parse(R"mlir(
module {
  func.func @relation_select(
      %input: memref<1x2048xf16, #wafer.memory<spm, cx>>)
      -> memref<1x2048xf16, #wafer.memory<spm, cx>> {
    %tensor = wafer.tile.materialize_layout %input
        : memref<1x2048xf16, #wafer.memory<spm, cx>>
       -> memref<1x2048xf16, #wafer.memory<spm, tensor>>
    %predicate = wafer.tile.elementwise #wafer.elementwise_kind<eq>
        %tensor, %tensor
        : (memref<1x2048xf16, #wafer.memory<spm, tensor>>,
           memref<1x2048xf16, #wafer.memory<spm, tensor>>)
       -> memref<1x2048xi1, #wafer.memory<spm, tensor>>
    %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
        %predicate, %tensor, %tensor
        : (memref<1x2048xi1, #wafer.memory<spm, tensor>>,
           memref<1x2048xf16, #wafer.memory<spm, tensor>>,
           memref<1x2048xf16, #wafer.memory<spm, tensor>>)
       -> memref<1x2048xf16, #wafer.memory<spm, tensor>>
    %cx = wafer.tile.materialize_layout %selected
        : memref<1x2048xf16, #wafer.memory<spm, tensor>>
       -> memref<1x2048xf16, #wafer.memory<spm, cx>>
    return %cx : memref<1x2048xf16, #wafer.memory<spm, cx>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::PhysicalLayoutProposalResult result;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/0, &result, &failureReason)))
      << failureReason;
  EXPECT_EQ(result.movementCommandsBefore, 2u);
  EXPECT_EQ(result.movementCommandsAfter, 0u);
  module->walk([&](wafer::ComputeElementwiseOp op) {
    auto type = mlir::cast<mlir::MemRefType>(op.getResult().getType());
    EXPECT_EQ(wafer::getWaferMemoryAttr(type).getLayout(),
              wafer::MemLayout::Cx);
  });

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*module, &failureReason)))
      << failureReason;
  unsigned relations = 0;
  module->walk([&](wafer::InstrElementwiseOp op) {
    relations += op.getKind() == wafer::InstrElementwiseKind::Eq;
  });
  EXPECT_EQ(relations, 1u);
  unsigned bit2fp = 0;
  unsigned maskMoves = 0;
  module->walk([&](wafer::InstrBit2FpOp) { ++bit2fp; });
  module->walk([&](wafer::InstrMaskMoveOp) { ++maskMoves; });
  EXPECT_EQ(bit2fp, 1u);
  EXPECT_EQ(maskMoves, 1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest,
       CarriesBlockedPointwiseLayoutIntoCompositeReduceLowering) {
  auto module = parse(R"mlir(
module {
  func.func @pointwise_reduce(
      %input: memref<3x65xf16, #wafer.memory<spm, cx>>)
      -> memref<65xf16, #wafer.memory<spm, cx>> {
    %tensor = wafer.tile.materialize_layout %input
        : memref<3x65xf16, #wafer.memory<spm, cx>>
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %negated = wafer.tile.elementwise #wafer.elementwise_kind<neg> %tensor
        : (memref<3x65xf16, #wafer.memory<spm, tensor>>)
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %cx = wafer.tile.materialize_layout %negated
        : memref<3x65xf16, #wafer.memory<spm, tensor>>
       -> memref<3x65xf16, #wafer.memory<spm, cx>>
    %reduced = wafer.tile.reduce #wafer.reduce_kind<sum> %cx
        {dimensions = array<i64: 0>, init_value = 0.000000e+00 : f16}
        : (memref<3x65xf16, #wafer.memory<spm, cx>>)
       -> memref<65xf16, #wafer.memory<spm, cx>>
    return %reduced : memref<65xf16, #wafer.memory<spm, cx>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::PhysicalLayoutProposalResult result;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/0, &result, &failureReason)))
      << failureReason;
  EXPECT_EQ(result.movementCommandsBefore, 2u);
  EXPECT_EQ(result.movementCommandsAfter, 0u);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("pointwise_reduce");
  auto pointwise = *function.getOps<wafer::ComputeElementwiseOp>().begin();
  auto pointwiseType =
      mlir::cast<mlir::MemRefType>(pointwise.getResult().getType());
  EXPECT_EQ(wafer::getWaferMemoryAttr(pointwiseType).getLayout(),
            wafer::MemLayout::Cx);

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*module, &failureReason)))
      << failureReason;
  unsigned reductions = 0;
  unsigned gathers = 0;
  module->walk([&](wafer::InstrElementwiseOp op) {
    reductions += op.getKind() == wafer::InstrElementwiseKind::Add;
  });
  module->walk([&](wafer::InstrGatherScatterOp) { ++gathers; });
  EXPECT_EQ(reductions, 3u);
  EXPECT_GT(gathers, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest,
       JointlyAssignsBlockedPointwiseAndTreeAllReducePhysicalLayout) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 4>,
       policy = "all_available", endpoints = array<i64>}
  func.func @pointwise_all_reduce(
      %input: memref<3x65xf16, #wafer.memory<spm, cx>>)
      -> memref<3x65xf16, #wafer.memory<spm, cx>> {
    %tensor = wafer.tile.materialize_layout %input
        : memref<3x65xf16, #wafer.memory<spm, cx>>
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %first = wafer.tile.elementwise #wafer.elementwise_kind<neg> %tensor
        : (memref<3x65xf16, #wafer.memory<spm, tensor>>)
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %recv = memref.alloc()
        : memref<3x65xf16, #wafer.memory<spm, tensor>>
    %collective = wafer.tile.all_reduce #wafer.reduce_kind<sum> %first using %recv
        {local_rank = 2 : i64, group_size = 4 : i64,
         rank_group = array<i64: 0, 1, 2, 3>, bytes = 390 : i64,
         communication_id = 32 : i64}
        : (memref<3x65xf16, #wafer.memory<spm, tensor>>,
           memref<3x65xf16, #wafer.memory<spm, tensor>>)
       -> memref<3x65xf16, #wafer.memory<spm, tensor>>
    %cx = wafer.tile.materialize_layout %collective
        : memref<3x65xf16, #wafer.memory<spm, tensor>>
       -> memref<3x65xf16, #wafer.memory<spm, cx>>
    %second = wafer.tile.elementwise #wafer.elementwise_kind<neg> %cx
        : (memref<3x65xf16, #wafer.memory<spm, cx>>)
       -> memref<3x65xf16, #wafer.memory<spm, cx>>
    return %second : memref<3x65xf16, #wafer.memory<spm, cx>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::PhysicalLayoutProposalResult result;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/0, &result, &failureReason)))
      << failureReason;
  EXPECT_EQ(result.movementCommandsBefore, 2u);
  EXPECT_EQ(result.movementCommandsAfter, 0u);
  auto function =
      module->lookupSymbol<mlir::func::FuncOp>("pointwise_all_reduce");
  auto collective = *function.getOps<wafer::CommAllReduceOp>().begin();
  auto collectiveType =
      mlir::cast<mlir::MemRefType>(collective.getResult().getType());
  EXPECT_EQ(wafer::getWaferMemoryAttr(collectiveType).getLayout(),
            wafer::MemLayout::Cx);
  EXPECT_EQ(collective.getBytes(), 512);
  EXPECT_EQ(llvm::range_size(function.getOps<mlir::memref::AllocOp>()), 1u);

  wafer::tile_region_to_instr::TileRegionToInstrOptions options;
  options.allReduceSchedule =
      wafer::tile_region_to_instr::AllReduceSchedule::Tree;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *module, options, &failureReason)))
      << failureReason;
  unsigned treeMessages = 0;
  module->walk([&](wafer::InstrDTESendOp send) {
    treeMessages += send.getMessage().getPhase() ==
                        wafer::DTEProtocolPhase::AllReduceTreeReduce ||
                    send.getMessage().getPhase() ==
                        wafer::DTEProtocolPhase::AllReduceTreeBroadcast;
    if (send.getMessage().getPhase() ==
            wafer::DTEProtocolPhase::AllReduceTreeReduce ||
        send.getMessage().getPhase() ==
            wafer::DTEProtocolPhase::AllReduceTreeBroadcast)
      EXPECT_EQ(send.getBytes(), 512);
  });
  EXPECT_GT(treeMessages, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest,
       ComposesInverseNonMetadataReshapesAcrossPointwiseCompute) {
  auto module = parse(R"mlir(
module {
  func.func @inverse_reshape(
      %input: memref<2x65xf16, #wafer.memory<spm, cx>>)
      -> memref<2x65xf16, #wafer.memory<spm, cx>> {
    %reshaped = wafer.tile.reshape %input
        : memref<2x65xf16, #wafer.memory<spm, cx>>
       -> memref<5x26xf16, #wafer.memory<spm, cx>>
    %negated = wafer.tile.elementwise #wafer.elementwise_kind<neg> %reshaped
        : (memref<5x26xf16, #wafer.memory<spm, cx>>)
       -> memref<5x26xf16, #wafer.memory<spm, cx>>
    %restored = wafer.tile.reshape %negated
        : memref<5x26xf16, #wafer.memory<spm, cx>>
       -> memref<2x65xf16, #wafer.memory<spm, cx>>
    return %restored : memref<2x65xf16, #wafer.memory<spm, cx>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::PhysicalLayoutProposalResult result;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/0, &result, &failureReason)))
      << failureReason;
  EXPECT_EQ(result.movementCommandsBefore, 2u);
  EXPECT_EQ(result.movementCommandsAfter, 0u);
  unsigned reshapes = 0;
  module->walk([&](wafer::ViewReshapeOp) { ++reshapes; });
  EXPECT_EQ(reshapes, 0u);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("inverse_reshape");
  ASSERT_TRUE(function);
  auto elementwise = *function.getOps<wafer::ComputeElementwiseOp>().begin();
  auto resultType =
      mlir::cast<mlir::MemRefType>(elementwise.getResult().getType());
  EXPECT_EQ(resultType.getShape(), llvm::ArrayRef<int64_t>({2, 65}));

  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(*module, &failureReason)))
      << failureReason;
  unsigned gathers = 0;
  module->walk([&](wafer::InstrGatherScatterOp) { ++gathers; });
  EXPECT_EQ(gathers, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest,
       KeepsPreViewWhenAnInterveningEffectMayMutateTheSource) {
  auto module = parse(R"mlir(
module {
  func.func @effect_barrier(
      %input: memref<2x3xf16, #wafer.memory<spm, tensor>>)
      -> memref<2x3xf16, #wafer.memory<spm, tensor>> {
    %snapshot = wafer.tile.copy %input
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       -> memref<2x3xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    wafer.tile.fill %input, %zero
        : memref<2x3xf16, #wafer.memory<spm, tensor>>, f16
    %negated = wafer.tile.elementwise #wafer.elementwise_kind<neg> %snapshot
        : (memref<2x3xf16, #wafer.memory<spm, tensor>>)
       -> memref<2x3xf16, #wafer.memory<spm, tensor>>
    %result = wafer.tile.copy %negated
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       -> memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %result : memref<2x3xf16, #wafer.memory<spm, tensor>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  std::string failureReason;
  EXPECT_TRUE(mlir::succeeded(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/0, /*result=*/nullptr, &failureReason)))
      << failureReason;
  EXPECT_EQ(llvm::range_size(module->getOps<mlir::func::FuncOp>()), 1u);
  llvm::SmallVector<wafer::MoveCopyOp, 2> copies;
  module->walk([&](wafer::MoveCopyOp copy) { copies.push_back(copy); });
  ASSERT_EQ(copies.size(), 1u);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("effect_barrier");
  auto compute = *function.getOps<wafer::ComputeElementwiseOp>().begin();
  EXPECT_EQ(compute.getInputs().front(), copies.front().getResult());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest,
       ComposesInverseTransposeDiamondWithExactRelations) {
  auto module = parse(R"mlir(
module {
  func.func @transpose_diamond(
      %lhs: memref<2x3xf16, #wafer.memory<spm, tensor>>,
      %rhs: memref<2x3xf16, #wafer.memory<spm, tensor>>)
      -> memref<2x3xf16, #wafer.memory<spm, tensor>> {
    %lhs_t = wafer.tile.transpose %lhs {permutation = array<i64: 1, 0>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       -> memref<3x2xf16, #wafer.memory<spm, tensor>>
    %rhs_t = wafer.tile.transpose %rhs {permutation = array<i64: 1, 0>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       -> memref<3x2xf16, #wafer.memory<spm, tensor>>
    %sum_t = wafer.tile.elementwise #wafer.elementwise_kind<add> %lhs_t, %rhs_t
        : (memref<3x2xf16, #wafer.memory<spm, tensor>>,
           memref<3x2xf16, #wafer.memory<spm, tensor>>)
       -> memref<3x2xf16, #wafer.memory<spm, tensor>>
    %sum = wafer.tile.transpose %sum_t {permutation = array<i64: 1, 0>}
        : memref<3x2xf16, #wafer.memory<spm, tensor>>
       -> memref<2x3xf16, #wafer.memory<spm, tensor>>
    return %sum : memref<2x3xf16, #wafer.memory<spm, tensor>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::PhysicalLayoutProposalResult result;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/0, &result, &failureReason)))
      << failureReason;
  EXPECT_EQ(result.movementCommandsBefore, 3u);
  EXPECT_EQ(result.movementCommandsAfter, 0u);
  unsigned transposes = 0;
  module->walk([&](wafer::MoveTransposeOp) { ++transposes; });
  EXPECT_EQ(transposes, 0u);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("transpose_diamond");
  auto add = *function.getOps<wafer::ComputeElementwiseOp>().begin();
  auto type = mlir::cast<mlir::MemRefType>(add.getResult().getType());
  EXPECT_EQ(type.getShape(), llvm::ArrayRef<int64_t>({2, 3}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest,
       ComposesBroadcastAndRankReducedSliceWithExactRelations) {
  auto module = parse(R"mlir(
module {
  func.func @broadcast_slice(
      %lhs: memref<3xf16, #wafer.memory<spm, tensor>>,
      %rhs: memref<3xf16, #wafer.memory<spm, tensor>>)
      -> memref<3xf16, #wafer.memory<spm, tensor>> {
    %lhs_b = wafer.tile.broadcast %lhs {dimensions = array<i64: 1>}
        : memref<3xf16, #wafer.memory<spm, tensor>>
       -> memref<4x3xf16, #wafer.memory<spm, tensor>>
    %rhs_b = wafer.tile.broadcast %rhs {dimensions = array<i64: 1>}
        : memref<3xf16, #wafer.memory<spm, tensor>>
       -> memref<4x3xf16, #wafer.memory<spm, tensor>>
    %sum_b = wafer.tile.elementwise #wafer.elementwise_kind<add> %lhs_b, %rhs_b
        : (memref<4x3xf16, #wafer.memory<spm, tensor>>,
           memref<4x3xf16, #wafer.memory<spm, tensor>>)
       -> memref<4x3xf16, #wafer.memory<spm, tensor>>
    %sum = wafer.tile.extract_slice %sum_b {
        offsets = array<i64: 2, 0>, sizes = array<i64: 1, 3>,
        strides = array<i64: 1, 1>}
        : memref<4x3xf16, #wafer.memory<spm, tensor>>
       -> memref<3xf16, #wafer.memory<spm, tensor>>
    return %sum : memref<3xf16, #wafer.memory<spm, tensor>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::PhysicalLayoutProposalResult result;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/0, &result, &failureReason)))
      << failureReason;
  EXPECT_EQ(result.movementCommandsBefore, 3u);
  EXPECT_EQ(result.movementCommandsAfter, 0u);
  unsigned broadcasts = 0;
  unsigned slices = 0;
  module->walk([&](wafer::MoveBroadcastOp) { ++broadcasts; });
  module->walk([&](wafer::MoveExtractSliceOp) { ++slices; });
  EXPECT_EQ(broadcasts, 0u);
  EXPECT_EQ(slices, 0u);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("broadcast_slice");
  auto add = *function.getOps<wafer::ComputeElementwiseOp>().begin();
  auto type = mlir::cast<mlir::MemRefType>(add.getResult().getType());
  EXPECT_EQ(type.getShape(), llvm::ArrayRef<int64_t>({3}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest,
       RejectedPhysicalProposalLeavesActualCloneUnchanged) {
  auto module = parse(R"mlir(
module {
  func.func @inverse_reshape(
      %input: memref<2x65xf16, #wafer.memory<spm, cx>>)
      -> memref<2x65xf16, #wafer.memory<spm, cx>> {
    %reshaped = wafer.tile.reshape %input
        : memref<2x65xf16, #wafer.memory<spm, cx>>
       -> memref<5x26xf16, #wafer.memory<spm, cx>>
    %negated = wafer.tile.elementwise #wafer.elementwise_kind<neg> %reshaped
        : (memref<5x26xf16, #wafer.memory<spm, cx>>)
       -> memref<5x26xf16, #wafer.memory<spm, cx>>
    %restored = wafer.tile.reshape %negated
        : memref<5x26xf16, #wafer.memory<spm, cx>>
       -> memref<2x65xf16, #wafer.memory<spm, cx>>
    return %restored : memref<2x65xf16, #wafer.memory<spm, cx>>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream(before) << *module;

  std::string failureReason;
  EXPECT_TRUE(mlir::failed(wafer::applyPhysicalLayoutProposal(
      *module, /*proposalOrdinal=*/99, /*result=*/nullptr, &failureReason)));
  EXPECT_FALSE(failureReason.empty());
  std::string after;
  llvm::raw_string_ostream(after) << *module;
  EXPECT_EQ(after, before);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest, HoistsOnlySpeculatableLoopInvariantWork) {
  auto module = parse(R"mlir(
module {
  func.func @licm(%a: i32, %b: i32, %buffer: memref<1xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      %sum = arith.addi %a, %b : i32
      memref.store %sum, %buffer[%c0] : memref<1xi32>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("licm");
  EXPECT_EQ(wafer::hoistStaticLoopInvariantOperations(function), 1u);
  auto loop = *function.getOps<mlir::scf::ForOp>().begin();
  EXPECT_TRUE(loop.getBody()->getOps<mlir::arith::AddIOp>().empty());
  EXPECT_FALSE(loop.getBody()->getOps<mlir::memref::StoreOp>().empty());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest, RewritesEachModularIntegerAlgebraVariant) {
  auto module = parse(R"mlir(
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
                  %c: tensor<4xi32>, %d: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%a, %b, %c, %d : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
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
)mlir");
  ASSERT_TRUE(module);
  EXPECT_EQ(wafer::reassociateElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("reassociate")),
            1u);
  EXPECT_EQ(wafer::balanceElementwiseReductionTrees(
                module->lookupSymbol<mlir::func::FuncOp>("tree")),
            1u);
  EXPECT_EQ(wafer::contractDistributiveExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("distribute")),
            1u);
  EXPECT_EQ(wafer::factorElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("factor")),
            1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
  wafer::tensor_program_scheduling::SelectionConfig config(
      policy, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  config.logicalRank = 0;
  wafer::tensor_program_scheduling::CandidateSpec candidate;
  candidate.tileSizes = {4};
  for (llvm::StringRef functionName :
       {"reassociate", "tree", "distribute", "factor"}) {
    auto acceptance =
        wafer::tensor_program_scheduling::evaluateCandidateOnOriginalTask(
            module->lookupSymbol<mlir::func::FuncOp>(functionName), {4},
            candidate, config);
    EXPECT_TRUE(acceptance.failureReason.empty())
        << functionName.str() << ": " << acceptance.failureReason;
    EXPECT_EQ(acceptance.artifactSource,
              wafer::tensor_program_scheduling::CandidateArtifactSource::
                  CompleteTraversalAPI)
        << functionName.str();
  }
}

TEST_F(CandidateRewritesTest,
       RewritesF16AndBF16AlgebraWithoutNumericAnnotations) {
  auto module = parse(R"mlir(
module {
  func.func @reassociate_f16(
      %a: tensor<4xf16>, %b: tensor<4xf16>, %c: tensor<4xf16>,
      %out: tensor<4xf16>) -> tensor<4xf16> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xf16>, tensor<4xf16>, tensor<4xf16>)
      outs(%out : tensor<4xf16>) {
    ^bb0(%av: f16, %bv: f16, %cv: f16, %unused: f16):
      %ab = arith.addf %av, %bv : f16
      %abc = arith.addf %ab, %cv : f16
      linalg.yield %abc : f16
    } -> tensor<4xf16>
    return %r : tensor<4xf16>
  }
  func.func @tree_bf16(
      %a: tensor<4xbf16>, %b: tensor<4xbf16>, %c: tensor<4xbf16>,
      %d: tensor<4xbf16>, %out: tensor<4xbf16>) -> tensor<4xbf16> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c, %d
          : tensor<4xbf16>, tensor<4xbf16>, tensor<4xbf16>,
            tensor<4xbf16>) outs(%out : tensor<4xbf16>) {
    ^bb0(%av: bf16, %bv: bf16, %cv: bf16, %dv: bf16, %unused: bf16):
      %ab = arith.addf %av, %bv : bf16
      %abc = arith.addf %ab, %cv : bf16
      %abcd = arith.addf %abc, %dv : bf16
      linalg.yield %abcd : bf16
    } -> tensor<4xbf16>
    return %r : tensor<4xbf16>
  }
  func.func @distribute_f16(
      %a: tensor<4xf16>, %b: tensor<4xf16>, %c: tensor<4xf16>,
      %out: tensor<4xf16>) -> tensor<4xf16> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xf16>, tensor<4xf16>, tensor<4xf16>)
      outs(%out : tensor<4xf16>) {
    ^bb0(%av: f16, %bv: f16, %cv: f16, %unused: f16):
      %ab = arith.mulf %av, %bv : f16
      %ac = arith.mulf %av, %cv : f16
      %r0 = arith.subf %ab, %ac : f16
      linalg.yield %r0 : f16
    } -> tensor<4xf16>
    return %r : tensor<4xf16>
  }
  func.func @factor_bf16(
      %a: tensor<4xbf16>, %b: tensor<4xbf16>, %c: tensor<4xbf16>,
      %out: tensor<4xbf16>) -> tensor<4xbf16> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xbf16>, tensor<4xbf16>, tensor<4xbf16>)
      outs(%out : tensor<4xbf16>) {
    ^bb0(%av: bf16, %bv: bf16, %cv: bf16, %unused: bf16):
      %ab = arith.mulf %av, %bv : bf16
      %ac = arith.mulf %av, %cv : bf16
      %r0 = arith.addf %ab, %ac : bf16
      linalg.yield %r0 : bf16
    } -> tensor<4xbf16>
    return %r : tensor<4xbf16>
  }
}
)mlir");
  ASSERT_TRUE(module);

  EXPECT_EQ(wafer::reassociateElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("reassociate_f16")),
            1u);
  EXPECT_EQ(wafer::balanceElementwiseReductionTrees(
                module->lookupSymbol<mlir::func::FuncOp>("tree_bf16")),
            1u);
  EXPECT_EQ(wafer::contractDistributiveExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("distribute_f16")),
            1u);
  EXPECT_EQ(wafer::factorElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("factor_bf16")),
            1u);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  auto reassociate =
      module->lookupSymbol<mlir::func::FuncOp>("reassociate_f16");
  auto reassociateGeneric =
      *reassociate.getOps<mlir::linalg::GenericOp>().begin();
  auto reassociateYield = mlir::cast<mlir::linalg::YieldOp>(
      reassociateGeneric.getBody()->getTerminator());
  auto reassociated =
      reassociateYield.getValues().front().getDefiningOp<mlir::arith::AddFOp>();
  ASSERT_TRUE(reassociated);
  auto right = reassociated.getRhs().getDefiningOp<mlir::arith::AddFOp>();
  ASSERT_TRUE(right);

  auto factor = module->lookupSymbol<mlir::func::FuncOp>("factor_bf16");
  auto factorGeneric = *factor.getOps<mlir::linalg::GenericOp>().begin();
  auto factorYield = mlir::cast<mlir::linalg::YieldOp>(
      factorGeneric.getBody()->getTerminator());
  auto factored =
      factorYield.getValues().front().getDefiningOp<mlir::arith::MulFOp>();
  ASSERT_TRUE(factored);
  auto terms = factored.getRhs().getDefiningOp<mlir::arith::AddFOp>();
  ASSERT_TRUE(terms);
}

TEST_F(CandidateRewritesTest, RejectsPoisonChangingIntegerAlgebra) {
  auto module = parse(R"mlir(
module {
  func.func @no_wrap(%a: tensor<4xi32>, %b: tensor<4xi32>,
                     %c: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %unused: i32):
      %ab = arith.addi %av, %bv overflow<nsw> : i32
      %abc = arith.addi %ab, %cv : i32
      linalg.yield %abc : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  func.func @no_wrap_distribution(
      %a: tensor<4xi32>, %b: tensor<4xi32>, %c: tensor<4xi32>,
      %out: tensor<4xi32>) -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %unused: i32):
      %ab = arith.muli %av, %bv : i32
      %ac = arith.muli %av, %cv : i32
      %r0 = arith.subi %ab, %ac overflow<nsw> : i32
      linalg.yield %r0 : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  EXPECT_EQ(wafer::reassociateElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("no_wrap")),
            0u);
  EXPECT_EQ(
      wafer::contractDistributiveExpressions(
          module->lookupSymbol<mlir::func::FuncOp>("no_wrap_distribution")),
      0u);
}

} // namespace
