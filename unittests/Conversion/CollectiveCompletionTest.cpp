#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <string>
#include <tuple>

namespace {

void registerConversionDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
}

mlir::func::FuncOp findSingleFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp found;
  module.walk([&](mlir::func::FuncOp function) {
    EXPECT_FALSE(found);
    found = function;
  });
  return found;
}

template <typename OpT>
llvm::SmallVector<OpT, 4> collectOps(mlir::ModuleOp module) {
  llvm::SmallVector<OpT, 4> operations;
  module.walk([&](OpT operation) { operations.push_back(operation); });
  return operations;
}

void expectBefore(mlir::Operation *before, mlir::Operation *after) {
  ASSERT_NE(before, nullptr);
  ASSERT_NE(after, nullptr);
  ASSERT_EQ(before->getBlock(), after->getBlock());
  EXPECT_TRUE(before->isBeforeInBlock(after));
}

mlir::OwningOpRef<mlir::ModuleOp>
lowerTensorProgram(mlir::MLIRContext &context, llvm::StringRef sourceText,
                   int64_t currentLogicalRank) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(&context));
  EXPECT_TRUE(source);
  if (!source)
    return {};

  mlir::func::FuncOp function = findSingleFunction(*source);
  EXPECT_TRUE(function);
  if (!function)
    return {};

  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  std::string failureReason;
  EXPECT_TRUE(mlir::succeeded(wafer::lowerTensorProgramToTileRegionModule(
      function, lowered, &failureReason, currentLogicalRank)))
      << failureReason;
  if (!lowered)
    return {};
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*lowered)));
  return lowered;
}

void expectValidInstructionLowering(mlir::ModuleOp module) {
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstrModule(module, &failureReason)))
      << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
}

void expectSingletonCollectiveIdentity(mlir::MLIRContext &context,
                                       llvm::StringRef sourceText) {
  auto lowered =
      lowerTensorProgram(context, sourceText, /*currentLogicalRank=*/0);
  ASSERT_TRUE(lowered);

  EXPECT_TRUE(collectOps<wafer::CommAllGatherOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::CommReduceScatterOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::CommAllReduceOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTESendOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTERecvOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTEWaitOp>(*lowered).empty());

  auto loads = collectOps<wafer::StorageLoadOp>(*lowered);
  auto stores = collectOps<wafer::StorageStoreOp>(*lowered);
  ASSERT_EQ(loads.size(), 1u);
  ASSERT_EQ(stores.size(), 1u);
  EXPECT_TRUE(stores.front().getSource() == loads.front().getDest());

  expectValidInstructionLowering(*lowered);
  EXPECT_TRUE(collectOps<wafer::InstrDTESendOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTERecvOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTEWaitOp>(*lowered).empty());
}

std::string makeCollectivePermuteSource(llvm::StringRef pairs,
                                        bool includeChannel = true) {
  std::string source = R"mlir(
module {
  func.func @permute(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = wafer.linalg_ext.collective.collective_permute
        ins(%input : tensor<4xf32>)
        outs(%out : tensor<4xf32>)
        {source_target_pairs = array<i64: )mlir";
  source.append(pairs.begin(), pairs.end());
  source += ">";
  if (includeChannel)
    source += ", channel_id = 9 : i64";
  source += R"mlir(}
        -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir";
  return source;
}

class CollectiveCompletionTest : public ::testing::Test {
protected:
  CollectiveCompletionTest() {
    registerConversionDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
};

TEST_F(CollectiveCompletionTest,
       SingletonAllGatherWithoutChannelIsResidentIdentity) {
  expectSingletonCollectiveIdentity(context, R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 1>,
       policy = "all_available", endpoints = array<i64>}
  func.func @all_gather(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = wafer.linalg_ext.collective.all_gather
        ins(%input : tensor<4xf32>)
        outs(%out : tensor<4xf32>)
        {axis = 0 : i64, rank_group = array<i64: 0>} -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir");
}

TEST_F(CollectiveCompletionTest,
       SingletonReduceScatterWithoutChannelIsResidentIdentity) {
  expectSingletonCollectiveIdentity(context, R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 1>,
       policy = "all_available", endpoints = array<i64>}
  func.func @reduce_scatter(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = wafer.linalg_ext.collective.reduce_scatter
        ins(%input : tensor<4xf32>)
        outs(%out : tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {axis = 0 : i64, rank_group = array<i64: 0>} -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir");
}

TEST_F(CollectiveCompletionTest,
       SingletonAllReduceWithoutChannelIsResidentIdentity) {
  expectSingletonCollectiveIdentity(context, R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 1>,
       policy = "all_available", endpoints = array<i64>}
  func.func @all_reduce(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<4xf32>)
        outs(%out : tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {rank_group = array<i64: 0>} -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
)mlir");
}

TEST_F(CollectiveCompletionTest,
       AllToAllCompletesRemoteResultAssemblyBeforeConsumer) {
  auto lowered = lowerTensorProgram(context,
                                    R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  func.func @all_to_all(%input: tensor<4x2xf32>,
                        %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = wafer.linalg_ext.collective.all_to_all
        ins(%input : tensor<4x2xf32>)
        outs(%out : tensor<2x4xf32>)
        {split_axis = 0 : i64, concat_axis = 1 : i64,
         split_count = 2 : i64, rank_group = array<i64: 0, 1>,
         channel_id = 7 : i64}
        -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}
)mlir",
                                    /*currentLogicalRank=*/0);
  ASSERT_TRUE(lowered);

  auto inserts = collectOps<wafer::MoveInsertSliceOp>(*lowered);
  auto fences = collectOps<wafer::SyncLocalFenceOp>(*lowered);
  auto sends = collectOps<wafer::InstrDTESendOp>(*lowered);
  auto recvs = collectOps<wafer::InstrDTERecvOp>(*lowered);
  auto waits = collectOps<wafer::InstrDTEWaitOp>(*lowered);
  auto stores = collectOps<wafer::StorageStoreOp>(*lowered);
  ASSERT_EQ(inserts.size(), 2u);
  ASSERT_EQ(fences.size(), 2u);
  ASSERT_EQ(sends.size(), 1u);
  ASSERT_EQ(recvs.size(), 1u);
  ASSERT_EQ(waits.size(), 1u);
  ASSERT_EQ(stores.size(), 1u);

  expectBefore(inserts.front(), fences.front());
  expectBefore(fences.front(), sends.front());
  expectBefore(fences.front(), recvs.front());
  expectBefore(waits.front(), inserts.back());
  expectBefore(inserts.back(), fences.back());
  expectBefore(fences.back(), stores.front());
  expectValidInstructionLowering(*lowered);
}

TEST_F(CollectiveCompletionTest,
       SingletonAllToAllWithoutChannelCompletesLocalResultAssembly) {
  auto lowered = lowerTensorProgram(context,
                                    R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 1>,
       policy = "all_available", endpoints = array<i64>}
  func.func @all_to_all(%input: tensor<4x2xf32>,
                        %out: tensor<4x2xf32>) -> tensor<4x2xf32> {
    %result = wafer.linalg_ext.collective.all_to_all
        ins(%input : tensor<4x2xf32>)
        outs(%out : tensor<4x2xf32>)
        {split_axis = 0 : i64, concat_axis = 1 : i64,
         split_count = 1 : i64, rank_group = array<i64: 0>}
        -> tensor<4x2xf32>
    return %result : tensor<4x2xf32>
  }
}
)mlir",
                                    /*currentLogicalRank=*/0);
  ASSERT_TRUE(lowered);

  auto inserts = collectOps<wafer::MoveInsertSliceOp>(*lowered);
  auto fences = collectOps<wafer::SyncLocalFenceOp>(*lowered);
  auto stores = collectOps<wafer::StorageStoreOp>(*lowered);
  EXPECT_TRUE(collectOps<wafer::InstrDTESendOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTERecvOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTEWaitOp>(*lowered).empty());
  ASSERT_EQ(inserts.size(), 1u);
  ASSERT_EQ(fences.size(), 1u);
  ASSERT_EQ(stores.size(), 1u);
  expectBefore(inserts.front(), fences.front());
  expectBefore(fences.front(), stores.front());
  expectValidInstructionLowering(*lowered);
}

TEST_F(CollectiveCompletionTest,
       AllToAllDirectUsesGlobalGroupIndexCyclicRounds) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 4>,
       policy = "explicit",
       endpoints = array<i64: 0, 0, 0, 0,
                              0, 0, 1, 1,
                              0, 0, 0, 1,
                              0, 0, 1, 0>}
  func.func @all_to_all(%input: tensor<8x2xf32>,
                        %out: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %result = wafer.linalg_ext.collective.all_to_all
        ins(%input : tensor<8x2xf32>)
        outs(%out : tensor<2x8xf32>)
        {split_axis = 0 : i64, concat_axis = 1 : i64,
         split_count = 4 : i64, rank_group = array<i64: 0, 1, 2, 3>,
         channel_id = 7 : i64}
        -> tensor<2x8xf32>
    return %result : tensor<2x8xf32>
  }
}
)mlir";
  using Transfer = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
  llvm::SmallVector<Transfer, 16> sendTransfers;
  llvm::SmallVector<Transfer, 16> recvTransfers;
  const int64_t expectedSendPeers[] = {1, 2, 3};
  const int64_t expectedRecvPeers[] = {3, 2, 1};
  const int64_t expectedSendPayloadSlices[] = {1, 2, 3};
  for (int64_t logicalRank = 0; logicalRank < 4; ++logicalRank) {
    auto lowered = lowerTensorProgram(context, source, logicalRank);
    ASSERT_TRUE(lowered);

    auto sends = collectOps<wafer::InstrDTESendOp>(*lowered);
    auto recvs = collectOps<wafer::InstrDTERecvOp>(*lowered);
    ASSERT_EQ(sends.size(), 3u);
    ASSERT_EQ(recvs.size(), 3u);
    for (size_t round = 0; round < sends.size(); ++round) {
      EXPECT_EQ(sends[round].getMessage().getRound(),
                static_cast<int64_t>(round + 1));
      EXPECT_EQ(recvs[round].getMessage().getRound(),
                static_cast<int64_t>(round + 1));
      sendTransfers.emplace_back(logicalRank, sends[round].getPeer(),
                                 sends[round].getMessage().getCommunicationId(),
                                 sends[round].getMessage().getRound(),
                                 sends[round].getMessage().getPayloadSlice());
      recvTransfers.emplace_back(recvs[round].getPeer(), logicalRank,
                                 recvs[round].getMessage().getCommunicationId(),
                                 recvs[round].getMessage().getRound(),
                                 recvs[round].getMessage().getPayloadSlice());
    }
    if (logicalRank == 0) {
      for (size_t round = 0; round < sends.size(); ++round) {
        EXPECT_EQ(sends[round].getPeer(), expectedSendPeers[round]);
        EXPECT_EQ(recvs[round].getPeer(), expectedRecvPeers[round]);
        EXPECT_EQ(sends[round].getMessage().getPayloadSlice(),
                  expectedSendPayloadSlices[round]);
        EXPECT_EQ(recvs[round].getMessage().getPayloadSlice(), 0);
      }
    }
    expectValidInstructionLowering(*lowered);
  }
  llvm::sort(sendTransfers);
  llvm::sort(recvTransfers);
  EXPECT_EQ(sendTransfers, recvTransfers);
}

TEST_F(CollectiveCompletionTest,
       AllToAllDirectDoesNotRequireBoundedTopologyRing) {
  auto lowered = lowerTensorProgram(context,
                                    R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 17>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @mesh
      {topology = @topology, axes = ["rank"], shape = array<i64: 17>,
       policy = "all_available", endpoints = array<i64>}
  func.func @all_to_all(%input: tensor<34x1xf32>,
                        %out: tensor<2x17xf32>) -> tensor<2x17xf32> {
    %result = wafer.linalg_ext.collective.all_to_all
        ins(%input : tensor<34x1xf32>)
        outs(%out : tensor<2x17xf32>)
        {split_axis = 0 : i64, concat_axis = 1 : i64,
         split_count = 17 : i64,
         rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                                 8, 9, 10, 11, 12, 13, 14, 15, 16>,
         channel_id = 7 : i64}
        -> tensor<2x17xf32>
    return %result : tensor<2x17xf32>
  }
}
)mlir",
                                    /*currentLogicalRank=*/0);
  ASSERT_TRUE(lowered);

  auto sends = collectOps<wafer::InstrDTESendOp>(*lowered);
  auto recvs = collectOps<wafer::InstrDTERecvOp>(*lowered);
  ASSERT_EQ(sends.size(), 16u);
  ASSERT_EQ(recvs.size(), 16u);
  EXPECT_EQ(sends.front().getPeer(), 1);
  EXPECT_EQ(recvs.front().getPeer(), 16);
  EXPECT_EQ(sends.back().getPeer(), 16);
  EXPECT_EQ(recvs.back().getPeer(), 1);
  expectValidInstructionLowering(*lowered);
}

TEST_F(CollectiveCompletionTest,
       PermuteSendAndReceiveAvoidsFillRaceAndCompletesBothDirections) {
  auto lowered =
      lowerTensorProgram(context, makeCollectivePermuteSource("0, 1, 1, 2"),
                         /*currentLogicalRank=*/1);
  ASSERT_TRUE(lowered);

  auto fences = collectOps<wafer::SyncLocalFenceOp>(*lowered);
  auto sends = collectOps<wafer::InstrDTESendOp>(*lowered);
  auto recvs = collectOps<wafer::InstrDTERecvOp>(*lowered);
  auto waits = collectOps<wafer::InstrDTEWaitOp>(*lowered);
  auto stores = collectOps<wafer::StorageStoreOp>(*lowered);
  EXPECT_TRUE(collectOps<wafer::ComputeFillOp>(*lowered).empty());
  ASSERT_EQ(fences.size(), 1u);
  ASSERT_EQ(sends.size(), 1u);
  ASSERT_EQ(recvs.size(), 1u);
  ASSERT_EQ(waits.size(), 1u);
  ASSERT_EQ(stores.size(), 1u);
  EXPECT_TRUE(recvs.front().getBuffer().getDefiningOp<mlir::memref::AllocOp>());
  expectBefore(fences.front(), sends.front());
  expectBefore(fences.front(), recvs.front());
  expectBefore(sends.front(), waits.front());
  expectBefore(recvs.front(), waits.front());
  expectBefore(waits.front(), stores.front());
  expectValidInstructionLowering(*lowered);
}

TEST_F(CollectiveCompletionTest,
       PermuteSendOnlyCompletesZeroResultAndSendSourceBeforeIssue) {
  auto lowered =
      lowerTensorProgram(context, makeCollectivePermuteSource("0, 1"),
                         /*currentLogicalRank=*/0);
  ASSERT_TRUE(lowered);

  auto fills = collectOps<wafer::ComputeFillOp>(*lowered);
  auto fences = collectOps<wafer::SyncLocalFenceOp>(*lowered);
  auto sends = collectOps<wafer::InstrDTESendOp>(*lowered);
  auto waits = collectOps<wafer::InstrDTEWaitOp>(*lowered);
  auto stores = collectOps<wafer::StorageStoreOp>(*lowered);
  EXPECT_TRUE(collectOps<wafer::InstrDTERecvOp>(*lowered).empty());
  ASSERT_EQ(fills.size(), 1u);
  ASSERT_EQ(fences.size(), 1u);
  ASSERT_EQ(sends.size(), 1u);
  ASSERT_EQ(waits.size(), 1u);
  ASSERT_EQ(stores.size(), 1u);
  expectBefore(fills.front(), fences.front());
  expectBefore(fences.front(), sends.front());
  expectBefore(sends.front(), waits.front());
  expectBefore(waits.front(), stores.front());
  expectValidInstructionLowering(*lowered);
}

TEST_F(CollectiveCompletionTest,
       PermuteReceiveOnlyUsesDTECompletionWithoutZeroFill) {
  auto lowered =
      lowerTensorProgram(context, makeCollectivePermuteSource("0, 1"),
                         /*currentLogicalRank=*/1);
  ASSERT_TRUE(lowered);

  auto recvs = collectOps<wafer::InstrDTERecvOp>(*lowered);
  auto waits = collectOps<wafer::InstrDTEWaitOp>(*lowered);
  auto stores = collectOps<wafer::StorageStoreOp>(*lowered);
  EXPECT_TRUE(collectOps<wafer::ComputeFillOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTESendOp>(*lowered).empty());
  ASSERT_EQ(recvs.size(), 1u);
  ASSERT_EQ(waits.size(), 1u);
  ASSERT_EQ(stores.size(), 1u);
  expectBefore(recvs.front(), waits.front());
  expectBefore(waits.front(), stores.front());
  expectValidInstructionLowering(*lowered);
}

TEST_F(CollectiveCompletionTest,
       PermuteUnmappedRankCompletesDefinedZeroBeforeConsumer) {
  auto lowered =
      lowerTensorProgram(context, makeCollectivePermuteSource("0, 1"),
                         /*currentLogicalRank=*/2);
  ASSERT_TRUE(lowered);

  auto fills = collectOps<wafer::ComputeFillOp>(*lowered);
  auto fences = collectOps<wafer::SyncLocalFenceOp>(*lowered);
  auto stores = collectOps<wafer::StorageStoreOp>(*lowered);
  EXPECT_TRUE(collectOps<wafer::InstrDTESendOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTERecvOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTEWaitOp>(*lowered).empty());
  ASSERT_EQ(fills.size(), 1u);
  ASSERT_EQ(fences.size(), 1u);
  ASSERT_EQ(stores.size(), 1u);
  expectBefore(fills.front(), fences.front());
  expectBefore(fences.front(), stores.front());
  expectValidInstructionLowering(*lowered);
}

TEST_F(CollectiveCompletionTest,
       PermuteSelfCopyWithoutChannelCompletesLocalMovementBeforeConsumer) {
  auto lowered = lowerTensorProgram(
      context, makeCollectivePermuteSource("0, 0", /*includeChannel=*/false),
      /*currentLogicalRank=*/0);
  ASSERT_TRUE(lowered);

  auto copies = collectOps<wafer::MoveCopyOp>(*lowered);
  auto fences = collectOps<wafer::SyncLocalFenceOp>(*lowered);
  auto stores = collectOps<wafer::StorageStoreOp>(*lowered);
  EXPECT_TRUE(collectOps<wafer::ComputeFillOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTESendOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTERecvOp>(*lowered).empty());
  EXPECT_TRUE(collectOps<wafer::InstrDTEWaitOp>(*lowered).empty());
  ASSERT_EQ(copies.size(), 1u);
  ASSERT_EQ(fences.size(), 1u);
  ASSERT_EQ(stores.size(), 1u);
  expectBefore(copies.front(), fences.front());
  expectBefore(fences.front(), stores.front());
  expectValidInstructionLowering(*lowered);
}

} // namespace
