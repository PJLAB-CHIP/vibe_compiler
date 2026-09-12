//===- ReadOnlyInputSharingTest.cpp -----------------------------------===//

#include "Wafer/Transforms/Tile/ReadOnlyInputSharing.h"
#include "TestSupport/Driver/CompilerTesting.h"

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

namespace {
using namespace wafer;
using namespace wafer::compiler::detail;

enum class SharingMismatch { None, Window, Input, Layout };

std::string makeSharingInput(int64_t count, int64_t extent,
                             SharingMismatch mismatch = SharingMismatch::None) {
  std::string text;
  llvm::raw_string_ostream out(text);
  const std::string source = "memref<2x" + std::to_string(extent + 16) +
                             "x64xf16, #wafer.memory<ddr, tensor>>";
  const std::string shape = "memref<2x" + std::to_string(extent) + "x64xf16";
  const std::string ddr = shape + ", #wafer.memory<ddr, tensor>>";
  out << "module { wafer.target.topology @topology {card_grid = array<i64: 1, "
         "1>, "
         "card_interconnect = \"mesh\", tile_grid = array<i64: "
      << (count == 4 ? "2, 2" : "4, 4")
      << ">, unavailable_tiles = array<i64>}\n";
  for (int64_t tile = 0; tile < count; ++tile) {
    int64_t row = mismatch == SharingMismatch::Window ? tile : 1;
    const char *layouts[] = {"tensor", "ntensor", "cx", "ncx"};
    const std::string spm =
        shape + ", #wafer.memory<spm, " +
        layouts[mismatch == SharingMismatch::Layout ? tile % 4 : 0] + ">>";
    const std::string converted =
        shape + ", #wafer.memory<spm, " +
        (mismatch == SharingMismatch::Layout && tile % 4 == 3 ? "tensor"
                                                              : "ncx") +
        ">>";
    const std::string view = shape + ", strided<[" +
                             std::to_string((extent + 16) * 64) +
                             ", 64, 1], offset: " + std::to_string(row * 64) +
                             ">, #wafer.memory<ddr, tensor>>";
    out << "wafer.tile.module card_id = 0 tile_id = " << tile << " {\n"
        << "func.func @entry(";
    for (int64_t input = 0;
         input < (mismatch == SharingMismatch::Input ? count : 1); ++input)
      out << (input ? ", " : "") << "%src" << input << ": " << source
          << " {wafer.program_argument = #wafer.program_argument<" << input
          << ">}";
    out << ") -> " << ddr << " {\n%dst = memref.alloc() : " << ddr << "\n"
        << "%result = wafer.tile.region(%src"
        << (mismatch == SharingMismatch::Input ? tile : 0)
        << ", %dst : " << source << ", " << ddr << ") -> (" << ddr
        << ") {\n^bb0(%input: " << source << ", %output: " << ddr << "):\n"
        << "%view = memref.subview %input[0, " << row << ", 0] [2, " << extent
        << ", 64] [1, 1, 1] : " << source << " to " << view << "\n"
        << "%local = memref.alloc() : " << spm << "\n"
        << "wafer.tile.load %view into %local : " << view << " into " << spm
        << "\n"
        << "%converted = wafer.tile.materialize_layout %local : " << spm
        << " -> " << converted << "\n"
        << "wafer.tile.store %converted, %output : " << converted << " -> "
        << ddr << "\n"
        << "wafer.tile.yield %output : " << ddr << "\n}\n"
        << "return %result : " << ddr << "\n}\n}\n";
  }
  out << "}\n";
  return text;
}

TEST(ReadOnlyInputSharingTest, EqualStaticWindowsReachActualCompletionAndSPM) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (int64_t count : {4, 16})
    for (int64_t extent : {1024, 1025, 1031})
      for (bool share : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << count << "/" << extent << "/" << share);
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            makeSharingInput(count, extent), &context);
        ASSERT_TRUE(module);
        ASSERT_TRUE(mlir::succeeded(verifyTileModuleCollection(*module)));
        ASSERT_TRUE(hasReadOnlyInputSharing(*module));
        StructuredMaterializationRelations relations;
        rebuildCurrentBufferOwnerRelations(*module, relations);
        if (share) {
          auto result = materializeReadOnlyInputSharing(*module, relations);
          ASSERT_TRUE(result.succeeded()) << result.detail;
          EXPECT_EQ(result.statistics.peerSends, count - 1);
          EXPECT_EQ(result.statistics.peerReceives, count - 1);
        }
        unsigned loads = 0, sends = 0, receives = 0;
        module->walk([&](StorageLoadOp load) {
          ++loads;
          auto view = load.getSource().getDefiningOp<mlir::memref::SubViewOp>();
          ASSERT_TRUE(view);
          EXPECT_EQ(view.getStaticOffsets()[1], 1);
          EXPECT_EQ(view.getStaticSizes()[1], extent);
        });
        module->walk([&](CommPeerSendOp send) {
          ++sends;
          EXPECT_EQ(send.getBytes(), 256 * extent);
          EXPECT_EQ(send->getParentOfType<TileModuleOp>().getTileId(), 0);
          EXPECT_GT(send.getPeer(), 0);
        });
        module->walk([&](CommPeerRecvOp receive) {
          ++receives;
          EXPECT_EQ(receive.getBytes(), 256 * extent);
          EXPECT_EQ(receive.getPeer(), 0);
        });
        EXPECT_EQ(loads, share ? 1 : count);
        EXPECT_EQ(sends, share ? count - 1 : 0);
        EXPECT_EQ(receives, sends);
        std::string detail;
        auto standalone =
            createStandaloneTileModules(std::move(module), &detail, &relations);
        ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
        llvm::SmallVector<mlir::ModuleOp> instructions;
        for (auto &tile : *standalone) {
          TileRegionOp region;
          tile.module->walk([&](TileRegionOp current) { region = current; });
          ASSERT_TRUE(region);
          TileRegionToInstrLoweringSession session(context);
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(
              convertBufferizationCopiesToInstr(*tile.module, session)));
          instructions.push_back(*tile.module);
        }
        auto waits = rebuildRequiredDirectDTEWaits(instructions);
        ASSERT_TRUE(waits.succeeded()) << waits.detail;
        instructions.clear();
        for (auto &tile : *standalone) {
          ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
          TileMemoryPlanningFailure failure;
          auto planned = planTileMemory(std::move(tile.module), &failure);
          ASSERT_TRUE(mlir::succeeded(planned));
          tile.module = std::move(*planned);
          tile.module->walk([&](mlir::memref::AllocOp alloc) {
            if (isWaferSPMMemRefType(alloc.getType())) {
              EXPECT_TRUE(alloc->hasAttr(kWaferSPMOffsetAttrName));
            }
          });
          instructions.push_back(*tile.module);
        }
        EXPECT_TRUE(
            mlir::succeeded(verifyDirectDTETransportSchedule(instructions)));
        EXPECT_TRUE(mlir::succeeded(bindDirectDTETransport(instructions)));
        if (count == 16) {
          llvm::SmallVector<TileId> tileIds;
          for (auto &tile : *standalone)
            tileIds.push_back(tile.tileId);
          auto config =
              wafer::compiler::ExecutionConfig::createForSingleCard(1);
          ASSERT_TRUE(static_cast<bool>(config));
          auto cost = wafer::compiler::testing::verifyProgramResources(
              instructions, tileIds, *config);
          ASSERT_TRUE(mlir::succeeded(cost));
          ASSERT_TRUE(cost->maximumTileNoCTransmitMessageCount.isKnown());
          EXPECT_EQ(cost->maximumTileNoCTransmitMessageCount.value,
                    share ? 15 : 0);
        }
      }
}

TEST(ReadOnlyInputSharingTest, RequiresIdentityExactWindowAndReadOnlyEffects) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (unsigned variant = 0; variant != 7; ++variant) {
    SCOPED_TRACE(variant);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        makeSharingInput(4, 1031,
                         variant == 0   ? SharingMismatch::Window
                         : variant == 5 ? SharingMismatch::Input
                         : variant == 6 ? SharingMismatch::Layout
                                        : SharingMismatch::None),
        &context);
    ASSERT_TRUE(module);
    if (variant == 1)
      module->walk([&](mlir::func::FuncOp function) {
        if (function->getParentOfType<TileModuleOp>().getTileId() == 0)
          function.removeArgAttr(0, kWaferProgramArgumentAttrName);
      });
    llvm::SmallVector<StorageLoadOp> loads;
    module->walk([&](StorageLoadOp load) { loads.push_back(load); });
    for (auto load : loads) {
      mlir::OpBuilder builder(load);
      builder.setInsertionPointAfter(load);
      if (variant == 2 &&
          load->getParentOfType<TileModuleOp>().getTileId() == 0)
        builder.create<StorageStoreOp>(load.getLoc(), load.getDest(),
                                       load.getSource());
      else if (variant == 3)
        builder.create<StorageLoadOp>(load.getLoc(), load.getSource(),
                                      load.getDest());
      else if (variant == 4) {
        auto zero =
            builder.create<mlir::arith::ConstantIndexOp>(load.getLoc(), 0);
        auto one =
            builder.create<mlir::arith::ConstantIndexOp>(load.getLoc(), 1);
        auto loop =
            builder.create<mlir::scf::ForOp>(load.getLoc(), zero, one, one);
        load->moveBefore(loop.getBody()->getTerminator());
      }
    }
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    EXPECT_FALSE(hasReadOnlyInputSharing(*module));
    StructuredMaterializationRelations relations;
    rebuildCurrentBufferOwnerRelations(*module, relations);
    auto result = materializeReadOnlyInputSharing(*module, relations);
    EXPECT_EQ(result.failure, BoundaryMovementFailureKind::Unsupported);
    EXPECT_EQ(result.statistics.peerSends, 0u);
  }
}

} // namespace
