#include "Wafer/Conversion/WaferTileRegionToInstr/Internal.h"
#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

class CommunicationAlternativesTest : public ::testing::Test {
protected:
  CommunicationAlternativesTest() {
    wafer::registerAllDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef body) {
    return mlir::parseSourceString<mlir::ModuleOp>(body, &context);
  }

  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
};

TEST_F(CommunicationAlternativesTest,
       MaterializesDirectAllGatherBeyondExactRingBound) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 17>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 17>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%boundary: memref<68xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<68xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<68xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<68xf32, #wafer.memory<ddr, tensor>>):
      %local = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %gather = memref.alloc()
          : memref<68xf32, #wafer.memory<spm, tensor>>
      wafer.tile.all_gather %local into %gather
          {local_rank = 1 : i64, group_size = 17 : i64,
           rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15, 16>,
           bytes = 16 : i64,
           communication_id = 11 : i64}
          : memref<4xf32, #wafer.memory<spm, tensor>>
         -> memref<68xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<68xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  wafer::tile_region_to_instr::TileRegionToInstrOptions options;
  options.allGatherSchedule =
      wafer::tile_region_to_instr::AllGatherSchedule::Direct;
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *module, options, &failure)))
      << failure;
  unsigned directMessages = 0;
  unsigned directReceivesIntoGatherSlots = 0;
  module->walk([&](wafer::InstrDTESendOp send) {
    directMessages += send.getMessage().getPhase() ==
                      wafer::DTEProtocolPhase::AllGatherDirect;
  });
  module->walk([&](wafer::InstrDTERecvOp recv) {
    if (recv.getMessage().getPhase() !=
        wafer::DTEProtocolPhase::AllGatherDirect)
      return;
    directReceivesIntoGatherSlots +=
        recv.getBuffer().getDefiningOp<mlir::memref::SubViewOp>() != nullptr;
  });
  EXPECT_EQ(directMessages, 16u);
  EXPECT_EQ(directReceivesIntoGatherSlots, 16u);
  unsigned localCopies = 0;
  module->walk([&](wafer::InstrGatherScatterOp) { ++localCopies; });
  EXPECT_EQ(localCopies, 2u);
  bool retainedTileCollective = false;
  module->walk([&](wafer::CommAllGatherOp) { retainedTileCollective = true; });
  EXPECT_FALSE(retainedTileCollective);
}

TEST_F(CommunicationAlternativesTest,
       RingAllGatherConsumesExplicitTopologyPlacement) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 4>,
       policy = "explicit",
       endpoints = array<i64: 0, 0, 0, 0,
                              0, 0, 1, 1,
                              0, 0, 0, 1,
                              0, 0, 1, 0>}
  func.func @main(%boundary: memref<16xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<16xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<16xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<16xf32, #wafer.memory<ddr, tensor>>):
      %local = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %gather = memref.alloc()
          : memref<16xf32, #wafer.memory<spm, tensor>>
      wafer.tile.all_gather %local into %gather
          {local_rank = 0 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 16 : i64,
           communication_id = 12 : i64}
          : memref<4xf32, #wafer.memory<spm, tensor>>
         -> memref<16xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<16xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  wafer::tile_region_to_instr::TileRegionToInstrOptions options;
  options.allGatherSchedule =
      wafer::tile_region_to_instr::AllGatherSchedule::Ring;
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *module, options, &failure)))
      << failure;

  unsigned sends = 0;
  unsigned recvs = 0;
  module->walk([&](wafer::InstrDTESendOp send) {
    if (send.getMessage().getPhase() != wafer::DTEProtocolPhase::AllGatherRing)
      return;
    ++sends;
    EXPECT_EQ(send.getPeer(), 2);
  });
  module->walk([&](wafer::InstrDTERecvOp recv) {
    if (recv.getMessage().getPhase() != wafer::DTEProtocolPhase::AllGatherRing)
      return;
    ++recvs;
    EXPECT_EQ(recv.getPeer(), 3);
  });
  EXPECT_EQ(sends, 3u);
  EXPECT_EQ(recvs, 3u);
}

TEST_F(CommunicationAlternativesTest,
       MaterializesTopologyRingReduceScatterClone) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 4>,
       policy = "explicit",
       endpoints = array<i64: 0, 0, 0, 0,
                              0, 0, 1, 1,
                              0, 0, 0, 1,
                              0, 0, 1, 0>}
  func.func @main(%boundary: memref<4xi32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xi32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xi32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xi32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<16xi32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xi32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.reduce_scatter #wafer.reduce_kind<sum> %input using %recv
          {axis = 0 : i64, local_rank = 2 : i64, group_size = 4 : i64,
           rank_group = array<i64: 2, 3, 0, 1>, bytes = 16 : i64,
           communication_id = 13 : i64}
          : (memref<16xi32, #wafer.memory<spm, tensor>>,
             memref<4xi32, #wafer.memory<spm, tensor>>)
         -> memref<4xi32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xi32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  wafer::tile_region_to_instr::TileRegionToInstrOptions options;
  options.reduceScatterSchedule =
      wafer::tile_region_to_instr::ReduceScatterSchedule::Ring;
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *module, options, &failure)))
      << failure;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  llvm::SmallVector<wafer::InstrDTESendOp> sends;
  llvm::SmallVector<wafer::InstrDTERecvOp> recvs;
  llvm::SmallVector<wafer::InstrDTEWaitOp> waits;
  llvm::SmallVector<wafer::InstrElementwiseOp> reductions;
  module->walk([&](wafer::InstrDTESendOp op) { sends.push_back(op); });
  module->walk([&](wafer::InstrDTERecvOp op) { recvs.push_back(op); });
  module->walk([&](wafer::InstrDTEWaitOp op) { waits.push_back(op); });
  module->walk([&](wafer::InstrElementwiseOp op) { reductions.push_back(op); });

  ASSERT_EQ(sends.size(), 3u);
  ASSERT_EQ(recvs.size(), 3u);
  ASSERT_EQ(waits.size(), 3u);
  ASSERT_EQ(reductions.size(), 3u);

  // The explicit placement yields the minimum-hop cycle
  // rank 2 -> rank 0 -> rank 3 -> rank 1 -> rank 2, represented by
  // rank_group indices [0, 2, 1, 3].  Local group index 2 therefore sends to
  // logical rank 3, receives from logical rank 2, and finishes with slice 2.
  constexpr int64_t expectedSendSlices[] = {0, 3, 1};
  constexpr int64_t expectedRecvSlices[] = {3, 1, 2};
  for (size_t index = 0; index < sends.size(); ++index) {
    EXPECT_EQ(sends[index].getPeer(), 3);
    EXPECT_EQ(recvs[index].getPeer(), 2);
    EXPECT_EQ(sends[index].getBytes(), 16);
    EXPECT_EQ(recvs[index].getBytes(), 16);
    EXPECT_EQ(sends[index].getMessage().getPhase(),
              wafer::DTEProtocolPhase::ReduceScatterRing);
    EXPECT_EQ(recvs[index].getMessage().getPhase(),
              wafer::DTEProtocolPhase::ReduceScatterRing);
    EXPECT_EQ(sends[index].getMessage().getRound(),
              static_cast<int64_t>(index));
    EXPECT_EQ(recvs[index].getMessage().getRound(),
              static_cast<int64_t>(index));
    EXPECT_EQ(sends[index].getMessage().getPayloadSlice(),
              expectedSendSlices[index]);
    EXPECT_EQ(recvs[index].getMessage().getPayloadSlice(),
              expectedRecvSlices[index]);
    EXPECT_EQ(waits[index].getTokens().size(), 2u);
  }

  EXPECT_TRUE(
      sends.front().getBuffer().getDefiningOp<mlir::memref::SubViewOp>());
  for (wafer::InstrElementwiseOp reduction : reductions) {
    ASSERT_EQ(reduction.getInputs().size(), 2u);
    EXPECT_TRUE(
        reduction.getInputs().front().getDefiningOp<mlir::memref::SubViewOp>());
    EXPECT_EQ(mlir::cast<mlir::MemRefType>(reduction.getDest().getType())
                  .getNumElements(),
              4);
  }
}

TEST_F(CommunicationAlternativesTest,
       KeepsDirectReduceScatterBeyondExactRingBound) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 17>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 17>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%boundary: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<68xf32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.reduce_scatter #wafer.reduce_kind<sum> %input using %recv
          {axis = 0 : i64, local_rank = 1 : i64, group_size = 17 : i64,
           rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15, 16>,
           bytes = 16 : i64,
           communication_id = 18 : i64}
          : (memref<68xf32, #wafer.memory<spm, tensor>>,
             memref<4xf32, #wafer.memory<spm, tensor>>)
         -> memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  wafer::tile_region_to_instr::TileRegionToInstrOptions options;
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *module, options, &failure)))
      << failure;

  unsigned directMessages = 0;
  unsigned ringMessages = 0;
  module->walk([&](wafer::InstrDTESendOp send) {
    directMessages += send.getMessage().getPhase() ==
                      wafer::DTEProtocolPhase::ReduceScatterDirect;
    ringMessages += send.getMessage().getPhase() ==
                    wafer::DTEProtocolPhase::ReduceScatterRing;
  });
  EXPECT_EQ(directMessages, 16u);
  EXPECT_EQ(ringMessages, 0u);
}

TEST_F(CommunicationAlternativesTest, MaterializesChunkedRingAllReduceClone) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 4>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%boundary: memref<4xi32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xi32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xi32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xi32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<4xi32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xi32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.all_reduce #wafer.reduce_kind<sum> %input using %recv
          {local_rank = 2 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 16 : i64,
           communication_id = 14 : i64}
          : (memref<4xi32, #wafer.memory<spm, tensor>>,
             memref<4xi32, #wafer.memory<spm, tensor>>)
         -> memref<4xi32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xi32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  wafer::tile_region_to_instr::TileRegionToInstrOptions options;
  options.allReduceSchedule =
      wafer::tile_region_to_instr::AllReduceSchedule::Ring;
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *module, options, &failure)))
      << failure;
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  llvm::SmallVector<wafer::InstrDTESendOp> sends;
  llvm::SmallVector<wafer::InstrDTERecvOp> recvs;
  llvm::SmallVector<wafer::InstrDTEWaitOp> waits;
  llvm::SmallVector<wafer::InstrElementwiseOp> reductions;
  llvm::SmallVector<wafer::InstrGatherScatterOp> localCopies;
  module->walk([&](wafer::InstrDTESendOp op) { sends.push_back(op); });
  module->walk([&](wafer::InstrDTERecvOp op) { recvs.push_back(op); });
  module->walk([&](wafer::InstrDTEWaitOp op) { waits.push_back(op); });
  module->walk([&](wafer::InstrElementwiseOp op) { reductions.push_back(op); });
  module->walk(
      [&](wafer::InstrGatherScatterOp op) { localCopies.push_back(op); });

  ASSERT_EQ(sends.size(), 6u);
  ASSERT_EQ(recvs.size(), 6u);
  ASSERT_EQ(waits.size(), 6u);
  ASSERT_EQ(reductions.size(), 3u);
  // One accumulator initialization plus one receive-buffer-to-accumulator
  // copy for each all-gather round.  Receiving directly into the accumulator
  // would alias the simultaneous send root and violate Direct DTE isolation.
  ASSERT_EQ(localCopies.size(), 4u);

  // Rank 2 sends slices 2,1,0 during reduce-scatter, owns slice 3,
  // then sends 3,2,1 during all-gather.  Rounds continue across both
  // phases so no two messages from the collective share an identity.
  constexpr int64_t expectedSendSlices[] = {2, 1, 0, 3, 2, 1};
  constexpr int64_t expectedRecvSlices[] = {1, 0, 3, 2, 1, 0};
  for (size_t index = 0; index < sends.size(); ++index) {
    EXPECT_EQ(sends[index].getBytes(), 4);
    EXPECT_EQ(recvs[index].getBytes(), 4);
    EXPECT_EQ(sends[index].getMessage().getPhase(),
              wafer::DTEProtocolPhase::AllReduceRing);
    EXPECT_EQ(recvs[index].getMessage().getPhase(),
              wafer::DTEProtocolPhase::AllReduceRing);
    EXPECT_EQ(sends[index].getMessage().getRound(),
              static_cast<int64_t>(index));
    EXPECT_EQ(recvs[index].getMessage().getRound(),
              static_cast<int64_t>(index));
    EXPECT_EQ(sends[index].getMessage().getPayloadSlice(),
              expectedSendSlices[index]);
    EXPECT_EQ(recvs[index].getMessage().getPayloadSlice(),
              expectedRecvSlices[index]);
    auto sendView =
        sends[index].getBuffer().getDefiningOp<mlir::memref::SubViewOp>();
    auto recvView =
        recvs[index].getBuffer().getDefiningOp<mlir::memref::SubViewOp>();
    ASSERT_TRUE(sendView);
    ASSERT_TRUE(recvView);
    EXPECT_EQ(waits[index].getTokens().size(), 2u);
    if (index >= 3)
      EXPECT_NE(sendView.getSource(), recvView.getSource());
  }

  for (wafer::InstrElementwiseOp reduction : reductions) {
    auto destType = mlir::cast<mlir::MemRefType>(reduction.getDest().getType());
    EXPECT_EQ(destType.getNumElements(), 1);
    ASSERT_EQ(reduction.getInputs().size(), 2u);
    for (mlir::Value input : reduction.getInputs())
      EXPECT_EQ(mlir::cast<mlir::MemRefType>(input.getType()).getNumElements(),
                1);
  }

  // Per rank this is 2 * (P - 1) messages of B / P bytes, hence the full
  // four-rank program injects 2 * (P - 1) * B = 96 bytes.
  int64_t perRankSendBytes = 0;
  for (wafer::InstrDTESendOp send : sends)
    perRankSendBytes += send.getBytes();
  EXPECT_EQ(perRankSendBytes, 24);
  EXPECT_EQ(perRankSendBytes * 4, 96);
}

TEST_F(CommunicationAlternativesTest, MaterializesTreeAllReduceClone) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 4>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%boundary: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.all_reduce #wafer.reduce_kind<sum> %input using %recv
          {local_rank = 2 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 16 : i64,
           communication_id = 15 : i64}
          : (memref<4xf32, #wafer.memory<spm, tensor>>,
             memref<4xf32, #wafer.memory<spm, tensor>>)
         -> memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  wafer::tile_region_to_instr::TileRegionToInstrOptions options;
  options.allReduceSchedule =
      wafer::tile_region_to_instr::AllReduceSchedule::Tree;
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *module, options, &failure)))
      << failure;
  bool sawReduce = false;
  bool sawBroadcast = false;
  module->walk([&](wafer::InstrDTESendOp send) {
    if (send.getMessage().getPhase() ==
        wafer::DTEProtocolPhase::AllReduceTreeReduce) {
      sawReduce = true;
      EXPECT_EQ(send.getPeer(), 1);
    }
    if (send.getMessage().getPhase() ==
        wafer::DTEProtocolPhase::AllReduceTreeBroadcast) {
      sawBroadcast = true;
      EXPECT_EQ(send.getPeer(), 3);
    }
  });
  EXPECT_TRUE(sawReduce);
  EXPECT_TRUE(sawBroadcast);
}

TEST_F(CommunicationAlternativesTest,
       FloatingAllReduceUsesOrderedTreeAndRejectsRing) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 4>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%boundary: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.all_reduce #wafer.reduce_kind<sum> %input using %recv
          {local_rank = 1 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 16 : i64,
           communication_id = 19 : i64}
          : (memref<4xf32, #wafer.memory<spm, tensor>>,
             memref<4xf32, #wafer.memory<spm, tensor>>)
         -> memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir";

  auto autoModule = parse(source);
  ASSERT_TRUE(autoModule);
  wafer::tile_region_to_instr::TileRegionToInstrOptions autoOptions;
  std::string autoFailure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *autoModule, autoOptions, &autoFailure)))
      << autoFailure;
  bool sawTree = false;
  bool sawRing = false;
  autoModule->walk([&](wafer::InstrDTESendOp send) {
    sawTree |= send.getMessage().getPhase() ==
                   wafer::DTEProtocolPhase::AllReduceTreeReduce ||
               send.getMessage().getPhase() ==
                   wafer::DTEProtocolPhase::AllReduceTreeBroadcast;
    sawRing |=
        send.getMessage().getPhase() == wafer::DTEProtocolPhase::AllReduceRing;
  });
  EXPECT_TRUE(sawTree);
  EXPECT_FALSE(sawRing);

  auto ringModule = parse(source);
  ASSERT_TRUE(ringModule);
  wafer::tile_region_to_instr::TileRegionToInstrOptions ringOptions;
  ringOptions.allReduceSchedule =
      wafer::tile_region_to_instr::AllReduceSchedule::Ring;
  std::string ringFailure;
  EXPECT_TRUE(
      mlir::failed(wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *ringModule, ringOptions, &ringFailure)));
  EXPECT_NE(ringFailure.find("reorders reduction leaves"), std::string::npos);
}

TEST_F(CommunicationAlternativesTest,
       DefaultsToTreeWhenRingCannotFormNonzeroChunks) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%boundary: memref<4xi32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xi32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xi32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xi32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<4xi32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xi32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.all_reduce #wafer.reduce_kind<sum> %input using %recv
          {local_rank = 2 : i64, group_size = 16 : i64,
           rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                                  8, 9, 10, 11, 12, 13, 14, 15>,
           bytes = 16 : i64, communication_id = 17 : i64}
          : (memref<4xi32, #wafer.memory<spm, tensor>>,
             memref<4xi32, #wafer.memory<spm, tensor>>)
         -> memref<4xi32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xi32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir";

  auto autoModule = parse(source);
  ASSERT_TRUE(autoModule);
  wafer::tile_region_to_instr::TileRegionToInstrOptions autoOptions;
  std::string autoFailure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *autoModule, autoOptions, &autoFailure)))
      << autoFailure;
  bool sawTree = false;
  bool sawRing = false;
  autoModule->walk([&](wafer::InstrDTESendOp send) {
    sawTree |= send.getMessage().getPhase() ==
                   wafer::DTEProtocolPhase::AllReduceTreeReduce ||
               send.getMessage().getPhase() ==
                   wafer::DTEProtocolPhase::AllReduceTreeBroadcast;
    sawRing |=
        send.getMessage().getPhase() == wafer::DTEProtocolPhase::AllReduceRing;
  });
  EXPECT_TRUE(sawTree);
  EXPECT_FALSE(sawRing);

  auto ringModule = parse(source);
  ASSERT_TRUE(ringModule);
  wafer::tile_region_to_instr::TileRegionToInstrOptions ringOptions;
  ringOptions.allReduceSchedule =
      wafer::tile_region_to_instr::AllReduceSchedule::Ring;
  std::string ringFailure;
  EXPECT_TRUE(
      mlir::failed(wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *ringModule, ringOptions, &ringFailure)));
  EXPECT_NE(ringFailure.find("evenly divisible contiguous axis"),
            std::string::npos);
}

TEST_F(CommunicationAlternativesTest,
       RejectsNonDivisibleRingButKeepsTreeAllReduceClone) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 4>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%boundary: memref<6xi32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<6xi32, #wafer.memory<ddr, tensor>>)
        -> (memref<6xi32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<6xi32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<6xi32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<6xi32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.all_reduce #wafer.reduce_kind<sum> %input using %recv
          {local_rank = 2 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 24 : i64,
           communication_id = 16 : i64}
          : (memref<6xi32, #wafer.memory<spm, tensor>>,
             memref<6xi32, #wafer.memory<spm, tensor>>)
         -> memref<6xi32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<6xi32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir";

  auto ringModule = parse(source);
  ASSERT_TRUE(ringModule);
  wafer::tile_region_to_instr::TileRegionToInstrOptions ringOptions;
  ringOptions.allReduceSchedule =
      wafer::tile_region_to_instr::AllReduceSchedule::Ring;
  std::string ringFailure;
  EXPECT_TRUE(
      mlir::failed(wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *ringModule, ringOptions, &ringFailure)));
  EXPECT_NE(ringFailure.find("evenly divisible contiguous axis"),
            std::string::npos);

  auto treeModule = parse(source);
  ASSERT_TRUE(treeModule);
  wafer::tile_region_to_instr::TileRegionToInstrOptions treeOptions;
  treeOptions.allReduceSchedule =
      wafer::tile_region_to_instr::AllReduceSchedule::Tree;
  std::string treeFailure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::tile_region_to_instr::convertTileRegionToInstrModule(
          *treeModule, treeOptions, &treeFailure)))
      << treeFailure;
  bool sawTreeMessage = false;
  treeModule->walk([&](wafer::InstrDTESendOp send) {
    sawTreeMessage |= send.getMessage().getPhase() ==
                          wafer::DTEProtocolPhase::AllReduceTreeReduce ||
                      send.getMessage().getPhase() ==
                          wafer::DTEProtocolPhase::AllReduceTreeBroadcast;
  });
  EXPECT_TRUE(sawTreeMessage);
}

} // namespace
