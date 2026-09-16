//===- AccessReuseTest.cpp -----------------------------------===//

#include "Wafer/Transforms/Tile/AccessReuse.h"
#include "TestSupport/Driver/CompilerTesting.h"
#include "TestSupport/Transforms/AccessReuseInputs.h"

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Instr/CommunicationConstruction.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

namespace {
using namespace wafer;
using namespace wafer::compiler::detail;

enum class SharingMismatch { None, Window, Input, Layout };

void checkMemoryAndCompletion(mlir::OwningOpRef<mlir::ModuleOp> module,
                              StructuredMaterializationRelations &relations,
                              bool expectCapacity = false) {
  std::string detail;
  auto standalone =
      createStandaloneTileModules(std::move(module), &detail, &relations);
  ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
  llvm::SmallVector<mlir::ModuleOp> modules;
  llvm::SmallVector<TileId> tiles;
  for (auto &tile : *standalone) {
    TileRegionToInstrLoweringSession conversion(*tile.module->getContext());
    llvm::SmallVector<TileRegionOp> regions;
    tile.module->walk([&](TileRegionOp region) { regions.push_back(region); });
    for (auto region : regions)
      ASSERT_TRUE(
          mlir::succeeded(convertTileRegionToInstr(region, conversion)));
    ASSERT_TRUE(mlir::succeeded(
        convertBufferizationCopiesToInstr(*tile.module, conversion)));
    modules.push_back(*tile.module);
    tiles.push_back(tile.tileId);
  }
  auto constructed = constructCommunication(modules, tiles);
  ASSERT_TRUE(constructed.outcome.succeeded()) << constructed.outcome.detail;
  for (auto &tile : *standalone) {
    ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
    TileMemoryPlanningFailure failure;
    auto planned =
        planTileMemory(std::move(tile.module), &failure, nullptr, false);
    if (expectCapacity) {
      EXPECT_TRUE(mlir::failed(planned));
      EXPECT_TRUE(failure.spmCapacityOverflow);
      EXPECT_EQ(failure.kind, TileMemoryPlanningFailureKind::SPMAllocation);
      ASSERT_FALSE(failure.spmIndividuallyOversizedDemands.empty());
      EXPECT_GE(failure.spmLargestDemandBytes, 8u * 1024 * 1024);
    } else {
      ASSERT_TRUE(mlir::succeeded(planned));
      tile.module = std::move(*planned);
    }
  }
}

TEST(AccessReuseTest, ScopedWindowsCoverMainAndTailAndReachActualMemory) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (llvm::StringRef dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031})
      for (auto kind : {AccessReuseKind::Resident, AccessReuseKind::TwoLevel}) {
        SCOPED_TRACE(::testing::Message()
                     << dtype.str() << "/" << extent << "/" << int(kind));
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            wafer::testing::makeAccessReuseInput(4, extent, dtype), &context);
        ASSERT_TRUE(module);
        auto facts = analysis::analyzeAccessReuse(*module);
        ASSERT_FALSE(facts.scopes.empty()) << facts.detail;
        AccessReuseChoice choice;
        for (const auto &window : facts.scopes) {
          if (!window.outerLoops.empty())
            continue;
          EXPECT_EQ(window.sizes, (llvm::SmallVector<int64_t>{1, extent, 64}));
          EXPECT_EQ(window.windowBytes, uint64_t(extent * 128));
          EXPECT_EQ(window.readBytes, uint64_t(16 * extent * 128));
          AccessReuseAction action{kind, window.reads, window.scope, {}};
          if (kind == AccessReuseKind::TwoLevel)
            for (const auto &inner : facts.scopes)
              if (inner.source == window.source &&
                  inner.windowBytes == 256 * 128) {
                action.innerScope = inner.scope;
                break;
              }
          choice.actions.push_back(std::move(action));
        }
        ASSERT_EQ(choice.actions.size(), 4u);
        StructuredMaterializationRelations relations;
        rebuildCurrentBufferOwnerRelations(*module, relations);
        auto result = materializeAccessReuse(*module, relations, choice);
        ASSERT_TRUE(result.succeeded()) << result.detail;
        EXPECT_EQ(result.residentWindows, 4u);
        EXPECT_EQ(result.twoLevelWindows,
                  kind == AccessReuseKind::TwoLevel ? 4u : 0u);
        unsigned loads = 0;
        module->walk([&](StorageLoadOp) { ++loads; });
        EXPECT_EQ(loads, 4u);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        checkMemoryAndCompletion(std::move(module), relations);
      }
}

TEST(AccessReuseTest, SlidingWindowsLoadOnlyNewRowsAndReachActualMemory) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (llvm::StringRef dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031}) {
      auto module = mlir::parseSourceString<mlir::ModuleOp>(
          wafer::testing::makeAccessReuseInput(4, extent, dtype, true),
          &context);
      ASSERT_TRUE(module);
      auto facts = analysis::analyzeAccessReuse(*module);
      ASSERT_EQ(facts.sliding.size(), 4u);
      AccessReuseChoice choice;
      for (const auto &window : facts.sliding) {
        EXPECT_EQ(window.axis, 1u);
        EXPECT_EQ(window.shift, 1);
        choice.actions.push_back(
            {AccessReuseKind::Sliding, {window.access.load}, window.scope, {}});
      }
      StructuredMaterializationRelations relations;
      rebuildCurrentBufferOwnerRelations(*module, relations);
      auto result = materializeAccessReuse(*module, relations, choice);
      ASSERT_TRUE(result.succeeded()) << result.detail;
      EXPECT_EQ(result.slidingWindows, 4u);
      unsigned full = 0, incremental = 0;
      module->walk([&](StorageLoadOp load) {
        auto type = mlir::cast<mlir::MemRefType>(load.getSource().getType());
        if (type.getDimSize(1) == 8)
          ++full;
        if (type.getDimSize(1) == 1)
          ++incremental;
      });
      EXPECT_EQ(full, 4u);
      EXPECT_EQ(incremental, 4u);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      checkMemoryAndCompletion(std::move(module), relations);
    }
}

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

std::string makeLoopSharingInput(int64_t count, int64_t extent,
                                 bool differentWindows = false,
                                 bool differentTrips = false) {
  std::string text;
  llvm::raw_string_ostream out(text);
  std::string full = "memref<2x" + std::to_string(extent + 256) +
                     "x64xf16, #wafer.memory<ddr, tensor>>";
  int64_t outputExtent =
      differentWindows || differentTrips ? extent + 256 : extent;
  std::string output = "memref<2x" + std::to_string(outputExtent) +
                       "x64xf16, #wafer.memory<ddr, tensor>>";
  out << "module { wafer.target.topology @topology {card_grid = array<i64: 1, "
         "1>, "
         "card_interconnect = \"mesh\", tile_grid = array<i64: "
      << (count == 4 ? "2, 2" : "4, 4")
      << ">, unavailable_tiles = array<i64>}\n";
  for (int64_t tile = 0; tile < count; ++tile) {
    out << "wafer.tile.module card_id = 0 tile_id = " << tile
        << " { func.func @entry(%src: " << full
        << " {wafer.program_argument = #wafer.program_argument<0>}) -> "
        << output << " { %dst = memref.alloc() : " << output
        << " wafer.tile.region(%src, %dst : " << full << ", " << output
        << ") -> () { ^bb0(%input: " << full << ", %output: " << output << "): "
        << "%c0 = arith.constant 0 : index %c1 = arith.constant 1 : index "
        << "%c2 = arith.constant 2 : index %step = arith.constant 256 : index "
        << "%end = arith.constant "
        << (differentTrips ? 256 * (tile + 1) : 1024)
        << " : index %shift = arith.constant " << (differentWindows ? tile : 0)
        << " : index scf.for %b = %c0 to %c2 step %c1 { "
        << "scf.for %m = %c0 to %end step %step { "
        << "%row = arith.addi %m, %shift : index ";
    auto panel = [&](int64_t rows, llvm::StringRef offset,
                     llvm::StringRef tag) {
      std::string shape = "memref<1x" + std::to_string(rows) + "x64xf16";
      std::string view = shape + ", strided<[" +
                         std::to_string((extent + 256) * 64) +
                         ", 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>";
      std::string writeView =
          shape + ", strided<[" + std::to_string(outputExtent * 64) +
          ", 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>";
      std::string local = shape + ", #wafer.memory<spm, tensor>>";
      out << "%read" << tag << " = memref.subview %input[%b, " << offset
          << ", 0] [1, " << rows << ", 64] [1, 1, 1] : " << full << " to "
          << view << " %local" << tag << " = memref.alloc() : " << local
          << " wafer.tile.load %read" << tag << " into %local" << tag << " : "
          << view << " into " << local << " %write" << tag
          << " = memref.subview %output[%b, " << offset << ", 0] [1, " << rows
          << ", 64] [1, 1, 1] : " << output << " to " << writeView
          << " wafer.tile.store %local" << tag << ", %write" << tag << " : "
          << local << " -> " << writeView << " ";
    };
    panel(256, "%row", "main");
    out << "} ";
    if (extent > 1024 && !differentWindows && !differentTrips) {
      out << "%tail = arith.constant 1024 : index ";
      panel(extent - 1024, "%tail", "tail");
    }
    out << "} wafer.tile.yield } return %dst : " << output << " } }\n";
  }
  out << "}\n";
  return text;
}

TEST(AccessReuseTest, LoopWindowsReachConstructionCompletionAndBinding) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (int64_t count : {4, 16})
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(::testing::Message() << count << "/" << extent);
      auto module = mlir::parseSourceString<mlir::ModuleOp>(
          makeLoopSharingInput(count, extent), &context);
      ASSERT_TRUE(module);
      ASSERT_TRUE(!wafer::analysis::analyzeAccessReuse(*module).peers.empty());
      StructuredMaterializationRelations relations;
      rebuildCurrentBufferOwnerRelations(*module, relations);
      auto shared = materializeAccessReuse(
          *module, relations,
          selectPeerAccessReuse(wafer::analysis::analyzeAccessReuse(*module)));
      ASSERT_TRUE(shared.succeeded()) << shared.detail;
      const unsigned families = extent > 1024 ? 2 : 1;
      EXPECT_EQ(shared.movement.peerSends, (count - 1) * families);
      EXPECT_EQ(shared.movement.peerReceives, (count - 1) * families);
      unsigned loads = 0;
      module->walk([&](StorageLoadOp) { ++loads; });
      EXPECT_EQ(loads, families);
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone));
      llvm::SmallVector<mlir::ModuleOp> instructions;
      llvm::SmallVector<TileId> tileIds;
      for (auto &tile : *standalone) {
        TileRegionToInstrLoweringSession conversion(context);
        llvm::SmallVector<TileRegionOp> regions;
        tile.module->walk(
            [&](TileRegionOp region) { regions.push_back(region); });
        for (auto region : regions)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, conversion)));
        ASSERT_TRUE(mlir::succeeded(
            convertBufferizationCopiesToInstr(*tile.module, conversion)));
        instructions.push_back(*tile.module);
        tileIds.push_back(tile.tileId);
      }
      auto construction = constructCommunication(instructions, tileIds);
      ASSERT_TRUE(construction.outcome.succeeded())
          << construction.outcome.detail;
      EXPECT_EQ(analyzeCurrentCommunicationOrder(instructions, tileIds).status,
                CommunicationOrderStatus::Acyclic);
      instructions.clear();
      for (auto &tile : *standalone) {
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
        auto planned = planTileMemory(std::move(tile.module));
        ASSERT_TRUE(mlir::succeeded(planned));
        tile.module = std::move(*planned);
        instructions.push_back(*tile.module);
      }
      EXPECT_TRUE(
          mlir::succeeded(verifyDirectDTETransportSchedule(instructions)));
      EXPECT_TRUE(mlir::succeeded(bindDirectDTETransport(instructions)));
    }
}

TEST(AccessReuseTest, DifferentLoopWindowsAndCountsAreNotShared) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (bool trips : {false, true}) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        makeLoopSharingInput(4, 1024, !trips, trips), &context);
    ASSERT_TRUE(module);
    EXPECT_FALSE(!wafer::analysis::analyzeAccessReuse(*module).peers.empty());
  }
}

TEST(AccessReuseTest, EqualStaticWindowsReachActualCompletionAndSPM) {
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
        ASSERT_TRUE(
            !wafer::analysis::analyzeAccessReuse(*module).peers.empty());
        StructuredMaterializationRelations relations;
        rebuildCurrentBufferOwnerRelations(*module, relations);
        if (share) {
          auto result = materializeAccessReuse(
              *module, relations,
              selectPeerAccessReuse(
                  wafer::analysis::analyzeAccessReuse(*module)));
          ASSERT_TRUE(result.succeeded()) << result.detail;
          EXPECT_EQ(result.movement.peerSends, count - 1);
          EXPECT_EQ(result.movement.peerReceives, count - 1);
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

TEST(AccessReuseTest, RequiresIdentityExactWindowAndReadOnlyEffects) {
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
    EXPECT_EQ(!wafer::analysis::analyzeAccessReuse(*module).peers.empty(),
              variant == 4);
    StructuredMaterializationRelations relations;
    rebuildCurrentBufferOwnerRelations(*module, relations);
    auto result = materializeAccessReuse(
        *module, relations,
        selectPeerAccessReuse(wafer::analysis::analyzeAccessReuse(*module)));
    EXPECT_EQ(result.failure, variant == 4
                                  ? AccessReuseFailureKind::None
                                  : AccessReuseFailureKind::Unsupported);
    EXPECT_EQ(result.movement.peerSends, variant == 4 ? 3u : 0u);
  }
}

// Arithmetic oracle: four 256 KiB reads, three sends on a 2x2 mesh.
// This policy isolates the strict threshold from hardware calibration.
TEST(AccessReuseTest, NetBenefitThresholdIsStrictAndQueriesDoNotMutate) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      makeSharingInput(4, 1024), &context);
  ASSERT_TRUE(module);
  std::string before, after;
  llvm::raw_string_ostream(before) << *module;
  auto facts = analysis::analyzeAccessReuse(*module);
  ASSERT_EQ(facts.peers.size(), 1u);
  analysis::SearchCostPolicy policy;
  policy.ddrNominalBytesPerSecond = 1000000000000ULL;
  policy.dteEndpointBytesPerSecondEstimate = 2000000000000ULL;
  policy.directionalNoCBytesPerSecond = 2000000000000ULL;
  policy.dteFirstMessagePicosecondsEstimate = 1;
  policy.dteMessageStartupPicosecondsEstimate = 1;
  policy.noCHopPicosecondsEstimate = 1;
  for (uint64_t threshold : {131064, 131065, 131066}) {
    policy.instructionFixedPicosecondsEstimate = threshold;
    auto choices = proposeAccessReuse(facts, policy);
    EXPECT_EQ(choices.opportunities, 1u);
    EXPECT_EQ(choices.choices.size(), threshold == 131064 ? 1u : 0u);
    EXPECT_EQ(choices.lowBenefit, threshold == 131064 ? 0u : 1u);
    EXPECT_EQ(choices.unknownBenefit, 0u);
  }
  policy.ddrNominalBytesPerSecond = 0;
  auto unknown = proposeAccessReuse(facts, policy);
  EXPECT_TRUE(unknown.choices.empty());
  EXPECT_EQ(unknown.unknownBenefit, 1u);
  llvm::raw_string_ostream(after) << *module;
  EXPECT_EQ(before, after);
}

TEST(AccessReuseTest, ScopeQueriesAreBoundedAndClonesRequireMappedAnchors) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      wafer::testing::makeAccessReuseInput(4, 1031, "f16"), &context);
  ASSERT_TRUE(module);
  auto bounded = analysis::analyzeAccessReuse(*module, {}, 1);
  EXPECT_EQ(bounded.scopeQueries, 1u);
  EXPECT_GT(bounded.indeterminateScopes, 0u);
  auto facts = analysis::analyzeAccessReuse(*module);
  auto choices = proposeAccessReuse(facts, analysis::SearchCostPolicy{});
  ASSERT_FALSE(choices.choices.empty());
  mlir::IRMapping absent;
  EXPECT_TRUE(
      mlir::failed(mapAccessReuseChoice(choices.choices.front(), absent)));
  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      mlir::cast<mlir::ModuleOp>(module->getOperation()->clone(mapping)));
  auto mapped = mapAccessReuseChoice(choices.choices.front(), mapping);
  ASSERT_TRUE(mlir::succeeded(mapped));
  StructuredMaterializationRelations relations;
  rebuildCurrentBufferOwnerRelations(*clone, relations);
  auto reused = materializeAccessReuse(*clone, relations, *mapped);
  EXPECT_TRUE(reused.succeeded()) << reused.detail;
  EXPECT_GT(reused.residentWindows, 0u);
  checkMemoryAndCompletion(std::move(clone), relations);
}

TEST(AccessReuseTest, PromotionCapacityComesOnlyFromActualMemoryPlanning) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (bool promote : {false, true}) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        wafer::testing::makeAccessReuseInput(4, 65536, "f16"), &context);
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations;
    rebuildCurrentBufferOwnerRelations(*module, relations);
    if (promote) {
      auto facts = analysis::analyzeAccessReuse(*module);
      AccessReuseChoice choice;
      for (const auto &window : facts.scopes)
        if (window.outerLoops.empty())
          choice.actions.push_back(
              {AccessReuseKind::Resident, window.reads, window.scope, {}});
      ASSERT_EQ(choice.actions.size(), 4u);
      auto materialized = materializeAccessReuse(*module, relations, choice);
      ASSERT_TRUE(materialized.succeeded()) << materialized.detail;
      EXPECT_EQ(materialized.residentWindows, 4u);
    }
    checkMemoryAndCompletion(std::move(module), relations, promote);
  }
}

TEST(AccessReuseTest, ExactScopesDoNotFillHolesOrAssumeUnknownEffects) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (unsigned variant : {0u, 1u, 2u}) {
    const bool holes = variant == 0;
    auto text = wafer::testing::makeAccessReuseInput(4, 1024, "f16");
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    if (holes) {
      module->walk([&](mlir::scf::ForOp loop) {
        if (mlir::getConstantIntValue(loop.getStep()) == 256) {
          mlir::OpBuilder builder(loop);
          loop.setStep(
              builder.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 512));
        }
      });
    } else if (variant == 1) {
      // A declaration with no effect contract may write through escaped state.
      module->walk([&](mlir::func::FuncOp function) {
        if (function.isExternal())
          return;
        mlir::OpBuilder builder(function);
        auto opaque = builder.create<mlir::func::FuncOp>(
            function.getLoc(), "opaque", builder.getFunctionType({}, {}));
        opaque.setPrivate();
        builder.setInsertionPointToStart(&function.front());
        builder.create<mlir::func::CallOp>(function.getLoc(), opaque,
                                           mlir::ValueRange{});
      });
    }
    if (variant == 2) {
      module->walk([&](mlir::func::FuncOp function) {
        auto region = *function.getOps<TileRegionOp>().begin();
        auto loop =
            *region.getBody().front().getOps<mlir::scf::ForOp>().begin();
        unsigned index = function.getNumArguments();
        function.insertArgument(index, mlir::IndexType::get(&context),
                                mlir::DictionaryAttr{}, function.getLoc());
        region.getInputsMutable().append(function.getArgument(index));
        auto bound = region.getBody().front().addArgument(
            mlir::IndexType::get(&context), function.getLoc());
        loop.setUpperBound(bound);
      });
    }
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    auto facts = analysis::analyzeAccessReuse(*module);
    if (holes) {
      EXPECT_FALSE(facts.scopes.empty()); // Inner exact reuse is still allowed.
      EXPECT_TRUE(llvm::none_of(facts.scopes, [](const auto &window) {
        return window.outerLoops.empty();
      }));
    } else {
      EXPECT_TRUE(facts.reads.empty());
      EXPECT_TRUE(facts.scopes.empty());
      EXPECT_TRUE(facts.peers.empty());
    }
  }
}

TEST(AccessReuseTest, JointChoiceIncludesThreeIndependentProfitableInputs) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      wafer::testing::makeAccessReuseInput(4, 1031, "f16"), &context);
  ASSERT_TRUE(module);
  module->walk([&](mlir::func::FuncOp function) {
    auto region = *function.getOps<TileRegionOp>().begin();
    auto first = function.getArgument(0);
    for (unsigned index : {1u, 2u}) {
      function.insertArgument(index, first.getType(), mlir::DictionaryAttr{},
                              function.getLoc());
      function.setArgAttr(index, kWaferProgramArgumentAttrName,
                          ProgramArgumentAttr::get(&context, index));
      mlir::IRMapping mapping;
      mapping.map(first, function.getArgument(index));
      mlir::OpBuilder builder(function.front().getTerminator());
      builder.clone(*region, mapping);
    }
  });
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  auto facts = analysis::analyzeAccessReuse(*module);
  auto choices = proposeAccessReuse(facts, analysis::SearchCostPolicy{});
  ASSERT_FALSE(choices.choices.empty());
  llvm::SmallDenseSet<int64_t, 4> resources;
  for (const auto &action : choices.choices.front().actions)
    for (auto load : action.reads)
      for (const auto &read : facts.reads)
        if (read.load == load)
          resources.insert(read.argument);
  EXPECT_EQ(resources.size(), 3u);
  StructuredMaterializationRelations relations;
  rebuildCurrentBufferOwnerRelations(*module, relations);
  auto applied =
      materializeAccessReuse(*module, relations, choices.choices.front());
  ASSERT_TRUE(applied.succeeded()) << applied.detail;
  EXPECT_EQ(applied.residentWindows, 12u);
  checkMemoryAndCompletion(std::move(module), relations);
}

TEST(AccessReuseTest, PermutedWindowAxisRetainsExactTailCoverage) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  for (int64_t extent : {1024, 1031}) {
    auto text = wafer::testing::makeAccessReuseInput(4, extent, "f16");
    auto replace = [&](const std::string &from, const std::string &to) {
      size_t cursor = 0;
      while ((cursor = text.find(from, cursor)) != std::string::npos) {
        text.replace(cursor, from.size(), to);
        cursor += to.size();
      }
    };
    for (int64_t rows : {extent, int64_t(256), extent % 256}) {
      replace("x" + std::to_string(rows) + "x64xf16",
              "x64x" + std::to_string(rows) + "xf16");
      replace("[1, " + std::to_string(rows) + ", 64]",
              "[1, 64, " + std::to_string(rows) + "]");
    }
    replace(", 64, 1]", ", " + std::to_string(extent) + ", 1]");
    for (const std::string row : {"%row", "%end"}) {
      replace("%input[%c0, " + row + ", 0]", "%input[%c0, 0, " + row + "]");
      replace("%output[%batch, " + row + ", 0]",
              "%output[%batch, 0, " + row + "]");
    }
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module) << text;
    auto facts = analysis::analyzeAccessReuse(*module);
    AccessReuseChoice choice;
    for (const auto &window : facts.scopes)
      if (window.outerLoops.empty()) {
        EXPECT_EQ(window.sizes, (llvm::SmallVector<int64_t>{1, 64, extent}));
        choice.actions.push_back(
            {AccessReuseKind::Resident, window.reads, window.scope, {}});
      }
    ASSERT_EQ(choice.actions.size(), 4u);
    StructuredMaterializationRelations relations;
    rebuildCurrentBufferOwnerRelations(*module, relations);
    auto applied = materializeAccessReuse(*module, relations, choice);
    ASSERT_TRUE(applied.succeeded()) << applied.detail;
    checkMemoryAndCompletion(std::move(module), relations);
  }
}

} // namespace
