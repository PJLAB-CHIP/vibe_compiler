#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Frontend/InitImporterDialects.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

TEST_F(CommunicationAlternativesTest, MaterializesDirectAllGatherClone) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 4>,
       policy = "all_available", endpoints = array<i64>}
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
          {local_rank = 1 : i64, group_size = 4 : i64,
           rank_group = array<i64: 0, 1, 2, 3>, bytes = 16 : i64,
           communication_id = 11 : i64}
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
  wafer::TileRegionToInstrOptions options;
  options.allGatherSchedule = wafer::AllGatherSchedule::Direct;
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(wafer::convertTileRegionToInstrModule(
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
  EXPECT_GT(directMessages, 0u);
  EXPECT_EQ(directReceivesIntoGatherSlots, 3u);
  unsigned localCopies = 0;
  module->walk([&](wafer::InstrGatherScatterOp) { ++localCopies; });
  EXPECT_EQ(localCopies, 2u);
  bool retainedTileCollective = false;
  module->walk(
      [&](wafer::CommAllGatherOp) { retainedTileCollective = true; });
  EXPECT_FALSE(retainedTileCollective);
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
  wafer::TileRegionToInstrOptions options;
  options.allReduceSchedule = wafer::AllReduceSchedule::Tree;
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(wafer::convertTileRegionToInstrModule(
      *module, options, &failure)))
      << failure;
  bool sawReduce = false;
  bool sawBroadcast = false;
  module->walk([&](wafer::InstrDTESendOp send) {
    sawReduce |= send.getMessage().getPhase() ==
                 wafer::DTEProtocolPhase::AllReduceTreeReduce;
    sawBroadcast |= send.getMessage().getPhase() ==
                    wafer::DTEProtocolPhase::AllReduceTreeBroadcast;
  });
  EXPECT_TRUE(sawReduce);
  EXPECT_TRUE(sawBroadcast);
}

} // namespace
