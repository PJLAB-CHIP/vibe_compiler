//===- CoordinatedTerminalEvaluationTest.cpp ----------------------------===//

#include "../../lib/Wafer/Compiler/CoordinatedTerminalEvaluation.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>

namespace {

class CoordinatedTerminalEvaluationTest : public ::testing::Test {
protected:
  CoordinatedTerminalEvaluationTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeProgram(int64_t rankCount) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: )mlir"
       << rankCount << R"mlir(>, policy = "all_available",
       endpoints = array<i64>}
  func.func @main(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out = tensor.empty() : tensor<8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>) outs(%out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    } -> tensor<8xf16>
    return %result : tensor<8xf16>
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeOversizedTileProgram(int64_t elements) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  func.func @main(%boundary: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    %unused = wafer.tile.region(%boundary
        : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%ddr: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>):
      %zero = arith.constant 0.000000e+00 : f16
      %spm = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %ddr into %spm
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.fill %spm, %zero
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.yield %ddr
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeReadyOrderInstrProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %compute_dest = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %compute_source = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %dma_dest = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %ddr = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.elementwise <neg> %compute_source into %compute_dest
        : memref<4xf16, #wafer.memory<spm, tensor>>
      into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [1]
    wafer.instr.rdma %ddr to %dma_dest
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeRepairableSPMPressureTileProgram(int64_t elements) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 1>,
       policy = "explicit", endpoints = array<i64: 0, 0, 0, 0>}
  func.func @main(%input: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>)
      -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>> {
    %output = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    %result = wafer.tile.region(%input, %output
        : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>,
          memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%in: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>,
         %out: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>):
      %long_lived = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %long_lived
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      %intervening = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %intervening
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %intervening, %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
         -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.store %long_lived, %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
         -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    }
    return %result : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeDDRBoundaryRepairableSPMPressureTileProgram(int64_t elements) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 1>,
       policy = "explicit", endpoints = array<i64: 0, 0, 0, 0>}
  func.func @main(%input: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>)
      -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>> {
    %output = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    %result = wafer.tile.region(%input, %output
        : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>,
          memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%in: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>,
         %out: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>):
      %first = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %first
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %first, %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
         -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
      %second = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %second
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %second, %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
         -> memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %out
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    }
    return %result : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeFixedSlotInstrProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %slot
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeWorkerInstrProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %c = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %a, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %b, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %c, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0, 1, 2]
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeReadyOrderTileProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 1>,
       policy = "explicit", endpoints = array<i64: 0, 0, 0, 0>}
  func.func @main(
      %input: memref<8xf16, #wafer.memory<ddr, tensor>>)
      -> memref<8xf16, #wafer.memory<ddr, tensor>> {
    %output = memref.alloc()
        : memref<8xf16, #wafer.memory<ddr, tensor>>
    %result = wafer.tile.region(%input, %output
        : memref<8xf16, #wafer.memory<ddr, tensor>>,
          memref<8xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<8xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%in: memref<8xf16, #wafer.memory<ddr, tensor>>,
         %out: memref<8xf16, #wafer.memory<ddr, tensor>>):
      %compute_source = memref.alloc()
          : memref<8xf16, #wafer.memory<spm, tensor>>
      %computed = wafer.tile.elementwise #wafer.elementwise_kind<neg>
          %compute_source
          : (memref<8xf16, #wafer.memory<spm, tensor>>)
         -> memref<8xf16, #wafer.memory<spm, tensor>>
      %loaded = memref.alloc()
          : memref<8xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %in into %loaded
          : memref<8xf16, #wafer.memory<ddr, tensor>>
        into memref<8xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %computed, %out
          : memref<8xf16, #wafer.memory<spm, tensor>>
         -> memref<8xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %out
          : memref<8xf16, #wafer.memory<ddr, tensor>>
    }
    return %result : memref<8xf16, #wafer.memory<ddr, tensor>>
  }
}
)mlir",
                                                   context.get());
  }

  static llvm::SmallVector<mlir::ModuleOp, 2> views(
      llvm::ArrayRef<mlir::OwningOpRef<mlir::ModuleOp>> owners) {
    llvm::SmallVector<mlir::ModuleOp, 2> result;
    for (const auto &owner : owners)
      result.push_back(*owner);
    return result;
  }

  wafer::frontend::FrontendProgramVerificationResult
  replicatedProgram(int64_t rankCount, int64_t elements = 8) {
    auto makeBoundary = [&](int64_t index) {
      wafer::frontend::ProgramBoundaryBinding binding;
      binding.index = index;
      binding.programIndex = index;
      binding.distribution =
          wafer::frontend::ProgramDistributionKind::Replicated;
      binding.globalShape = {elements};
      binding.localShape = {elements};
      binding.dtype = "f16";
      for (int64_t rank = 0; rank < rankCount; ++rank) {
        wafer::frontend::ProgramRankSlice slice;
        slice.logicalRank = rank;
        slice.replicaId = rank;
        slice.offsets = {0};
        slice.sizes = {elements};
        slice.strides = {1};
        binding.rankSlices.push_back(std::move(slice));
      }
      return binding;
    };
    wafer::frontend::FrontendProgramVerificationResult program;
    program.logicalRankCount = rankCount;
    program.programUserInputCount = 1;
    program.distributedInputs.push_back(makeBoundary(0));
    program.distributedOutputs.push_back(makeBoundary(0));
    return program;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CoordinatedTerminalEvaluationTest,
       PublishesReadyOrderOnlyAsACompleteRankAction) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> parents;
  parents.push_back(makeReadyOrderInstrProgram());
  parents.push_back(makeReadyOrderInstrProgram());
  ASSERT_TRUE(parents[0] && parents[1]);
  wafer::OptimizationConfig config = wafer::OptimizationConfig::none();
  config.enable(wafer::OptimizationKind::ReadyOrderScheduling);
  uint64_t work = 0;
  auto actions = wafer::compiler::detail::
      deriveCoordinatedTerminalScheduleActions(views(parents), config, &work);
  ASSERT_TRUE(mlir::succeeded(actions));
  ASSERT_EQ(actions->size(), 2u);
  EXPECT_EQ(actions->front().stableOrdinal, 0u);
  EXPECT_EQ(actions->front().artifactKind, wafer::RankArtifactKind::Spill);
  EXPECT_EQ(actions->back().stableOrdinal, 1u);
  EXPECT_EQ(actions->back().artifactKind,
            wafer::RankArtifactKind::SpillReady);
  EXPECT_EQ(actions->back().rankModules.size(), 2u);
  EXPECT_EQ(work, 4u);
  for (const auto &action : *actions) {
    EXPECT_EQ(action.rankModules.size(), 2u);
    for (const auto &module : action.rankModules)
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }

  // A rank without a local move still receives an identity sibling under the
  // same all-rank action; it never disappears into a partial rank frontier.
  auto immovable = makeWorkerInstrProgram();
  ASSERT_TRUE(immovable);
  llvm::SmallVector<mlir::ModuleOp, 2> mismatched = {*parents.front(),
                                                     *immovable};
  actions = wafer::compiler::detail::deriveCoordinatedTerminalScheduleActions(
      mismatched, config, &work);
  ASSERT_TRUE(mlir::succeeded(actions));
  ASSERT_EQ(actions->size(), 2u);
  EXPECT_EQ(actions->back().rankModules.size(), 2u);
}

TEST_F(CoordinatedTerminalEvaluationTest,
       MaterializesFixedSlotAndWorkerActionsWithoutARankProduct) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> fixedParents;
  fixedParents.push_back(makeFixedSlotInstrProgram());
  fixedParents.push_back(makeFixedSlotInstrProgram());
  ASSERT_TRUE(fixedParents[0] && fixedParents[1]);
  wafer::OptimizationConfig fixedConfig = wafer::OptimizationConfig::none();
  fixedConfig.enable(wafer::OptimizationKind::StaticFixedSlotBuffering);
  uint64_t fixedWork = 0;
  auto fixedActions = wafer::compiler::detail::
      deriveCoordinatedTerminalScheduleActions(views(fixedParents),
                                               fixedConfig, &fixedWork);
  ASSERT_TRUE(mlir::succeeded(fixedActions));
  auto fixed = llvm::find_if(*fixedActions, [](const auto &action) {
    return action.bufferingKind == wafer::RankBufferingKind::StaticFixedSlot;
  });
  ASSERT_NE(fixed, fixedActions->end());
  EXPECT_EQ(fixed->bufferingPlanOrdinal, 1u);
  EXPECT_EQ(fixed->rankModules.size(), 2u);
  EXPECT_LE(fixedActions->size(),
            wafer::compiler::detail::kMaximumCoordinatedTerminalActions);

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> workerParents;
  workerParents.push_back(makeWorkerInstrProgram());
  workerParents.push_back(makeWorkerInstrProgram());
  ASSERT_TRUE(workerParents[0] && workerParents[1]);
  wafer::OptimizationConfig workerConfig = wafer::OptimizationConfig::none();
  workerConfig.enable(wafer::OptimizationKind::DisjointWorkerPlacement);
  uint64_t workerWork = 0;
  auto workerActions = wafer::compiler::detail::
      deriveCoordinatedTerminalScheduleActions(views(workerParents),
                                               workerConfig, &workerWork);
  ASSERT_TRUE(mlir::succeeded(workerActions));
  auto placed = llvm::find_if(*workerActions, [](const auto &action) {
    return action.workerPlacementKind ==
           wafer::RankWorkerPlacementKind::DisjointComponents;
  });
  ASSERT_NE(placed, workerActions->end());
  EXPECT_EQ(placed->workerPlacementPlanOrdinal, 1u);
  EXPECT_EQ(placed->rankModules.size(), 2u);
  EXPECT_EQ(workerActions->size(), 2u);
  EXPECT_EQ(workerWork, 4u);
}

TEST_F(CoordinatedTerminalEvaluationTest,
       LowersEachTileParentOnceBeforeMultipleTerminalActions) {
  auto tile = makeReadyOrderTileProgram();
  ASSERT_TRUE(tile);
  constexpr int64_t rankCount = 1;
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 1);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));

  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.reservedBaseline = true;
  variant.terminalReservation = ledger->getMandatoryBaselineReservation();
  variant.ranks.emplace_back(
      0, std::move(tile),
      std::make_shared<const std::string>("ready-order Tile parent"));
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CoordinatedTerminalFailure failure;
  auto gated = wafer::compiler::detail::evaluateCoordinatedTileVariant(
      variant, replicatedProgram(rankCount), *executionConfig,
      wafer::OptimizationConfig::production(), *ledger, diagnostics, failure);
  ASSERT_TRUE(mlir::succeeded(gated)) << diagnosticsText;
  ASSERT_GT(gated->size(), 1u);
  EXPECT_TRUE(llvm::any_of(*gated, [](const auto &survivor) {
    return survivor.variant.selectedArtifactKinds.front() ==
           wafer::RankArtifactKind::SpillReady;
  }));
  llvm::SmallSet<uint32_t, 8> actionOrdinals;
  for (const auto &survivor : *gated)
    EXPECT_TRUE(actionOrdinals.insert(survivor.terminalActionOrdinal).second);

  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    TileToInstrLowering)],
            1u);
  EXPECT_GT(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    TerminalScheduleAction)],
            1u);
}

TEST_F(CoordinatedTerminalEvaluationTest,
       FullyGatesAllRanksWithoutMutatingTileParents) {
  constexpr int64_t rankCount = 16;
  auto source = makeProgram(rankCount);
  ASSERT_TRUE(source);
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig searchConfig;
  searchConfig.rankCount = rankCount;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, searchConfig, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);

  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  for (const auto &tileParent : *frontier) {
    ASSERT_TRUE(wafer::containsTileDataflowOperations(
        tileParent.ranks.front().module.get().getOperation()));
    wafer::compiler::detail::CoordinatedTerminalFailure failure;
    auto gated = wafer::compiler::detail::evaluateCoordinatedTileVariant(
        tileParent, replicatedProgram(rankCount), *executionConfig,
        searchConfig.optimizations, *ledger, diagnostics, failure);
    ASSERT_TRUE(mlir::succeeded(gated))
        << "semantic ordinal " << tileParent.stableSemanticOrdinal << "\n"
        << diagnosticsText << "\n"
        << *tileParent.ranks.front().selectedTileIR;
    EXPECT_EQ(failure.kind,
              wafer::compiler::detail::CoordinatedTerminalFailureKind::None);
    ASSERT_FALSE(gated->empty());
    for (const auto &survivor : *gated) {
      ASSERT_EQ(survivor.variant.ranks.size(), rankCount);
      for (const wafer::compiler::RankExecutable &rank :
           survivor.variant.ranks) {
        EXPECT_FALSE(wafer::containsTileDataflowOperations(
            rank.getModule().getOperation()));
        bool hasInstruction = false;
        rank.getModule().walk([&](wafer::WaferInstructionOpInterface) {
          hasInstruction = true;
        });
        EXPECT_TRUE(hasInstruction);
      }
    }
    EXPECT_TRUE(wafer::containsTileDataflowOperations(
        tileParent.ranks.front().module.get().getOperation()));
  }
  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.terminalReserved, 0u);
  EXPECT_EQ(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    TileToInstrLowering)],
            rankCount * frontier->size());
  EXPECT_GE(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::ABIValidation)],
            rankCount * frontier->size());
}

TEST_F(CoordinatedTerminalEvaluationTest,
       TypedSPMFailurePreservesUnplacedTileParentAndClosesAttempt) {
  constexpr int64_t rankCount = 1;
  auto first = makeOversizedTileProgram(/*elements=*/2'000'000);
  ASSERT_TRUE(first);
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 rankCount);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));

  auto selectedTileIR = std::make_shared<const std::string>("actual Tile");
  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.reservedBaseline = true;
  variant.terminalReservation = ledger->getMandatoryBaselineReservation();
  variant.ranks.emplace_back(0, std::move(first), selectedTileIR);
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  auto program = replicatedProgram(rankCount);
  wafer::compiler::detail::CoordinatedTerminalFailure failure;
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  EXPECT_TRUE(
      mlir::failed(wafer::compiler::detail::evaluateCoordinatedTileVariant(
          variant, program, *executionConfig,
          wafer::OptimizationConfig::production(), *ledger, diagnostics,
          failure)));
  EXPECT_EQ(failure.kind, wafer::compiler::detail::
                              CoordinatedTerminalFailureKind::SPMAllocation);
  EXPECT_EQ(failure.logicalRank, 0);
  EXPECT_TRUE(wafer::containsTileDataflowOperations(
      variant.ranks.front().module.get().getOperation()));
  bool hasPlacement = false;
  variant.ranks.front().module.get().walk(
      [&](mlir::memref::AllocOp allocation) {
        hasPlacement |= allocation->hasAttr(wafer::kWaferSPMOffsetAttrName) ||
                        allocation->hasAttr(wafer::kWaferDDROffsetAttrName);
      });
  EXPECT_FALSE(hasPlacement);
  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.terminalReserved, 0u);
  EXPECT_EQ(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    TileToInstrLowering)],
            1u);
  EXPECT_GT(snapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    SPMAllocationProblem)],
            0u);
}

TEST_F(CoordinatedTerminalEvaluationTest,
       SelectiveSpillRepairReentersTheSameExactTerminalGate) {
  constexpr int64_t rankCount = 1;
  constexpr int64_t elements = 900'000;
  auto tileParent = makeRepairableSPMPressureTileProgram(elements);
  ASSERT_TRUE(tileParent);
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 1);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));
  auto terminal = wafer::compiler::detail::CoordinatedWorkLedger::
      getTerminalActionUpperBound(rankCount);
  ASSERT_TRUE(mlir::succeeded(terminal));
  auto parentReservation = ledger->tryReserveTerminalAction(*terminal);
  ASSERT_TRUE(parentReservation);

  wafer::compiler::detail::CoordinatedTileVariant parent;
  parent.stableSemanticOrdinal = 5;
  parent.terminalReservation = *parentReservation;
  parent.ranks.emplace_back(
      0, std::move(tileParent),
      std::make_shared<const std::string>("repairable SPM Tile parent"));
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  auto program = replicatedProgram(rankCount, elements);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CoordinatedTerminalFailure failure;
  auto rejected = wafer::compiler::detail::evaluateCoordinatedTileVariant(
      parent, program, *executionConfig, wafer::OptimizationConfig::none(),
      *ledger, diagnostics, failure);
  ASSERT_TRUE(mlir::failed(rejected));
  ASSERT_EQ(failure.kind, wafer::compiler::detail::
                              CoordinatedTerminalFailureKind::SPMAllocation)
      << diagnosticsText;
  EXPECT_TRUE(wafer::containsTileDataflowOperations(
      parent.ranks.front().module.get().getOperation()));

  std::string repairFailure;
  auto repaired = wafer::compiler::detail::materializeCoordinatedTileRepair(
      parent,
      wafer::compiler::detail::CoordinatedTileRepairAction::SelectiveSpill,
      /*stableSemanticOrdinal=*/6, *ledger, &repairFailure);
  ASSERT_TRUE(mlir::succeeded(repaired)) << repairFailure;
  const auto repairSnapshot = ledger->getSnapshot();
  EXPECT_EQ(repairSnapshot.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    RepairExpansion)],
            1u);

  auto accepted = wafer::compiler::detail::evaluateCoordinatedTileVariant(
      *repaired, program, *executionConfig, wafer::OptimizationConfig::none(),
      *ledger, diagnostics, failure);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticsText;
  ASSERT_EQ(accepted->size(), 1u);
  EXPECT_EQ(failure.kind,
            wafer::compiler::detail::CoordinatedTerminalFailureKind::None);
  mlir::ModuleOp finalModule = accepted->front().variant.ranks.front().getModule();
  unsigned loads = 0;
  unsigned stores = 0;
  finalModule.walk([&](wafer::InstrRDMAOp) { ++loads; });
  finalModule.walk([&](wafer::InstrWDMAOp) { ++stores; });
  EXPECT_GE(loads, 3u);
  EXPECT_GE(stores, 3u);
  EXPECT_FALSE(wafer::containsTileDataflowOperations(finalModule.getOperation()));

  bool parentHasOffset = false;
  parent.ranks.front().module.get().walk(
      [&](mlir::memref::AllocOp allocation) {
        parentHasOffset |= allocation->hasAttr(wafer::kWaferSPMOffsetAttrName) ||
                           allocation->hasAttr(wafer::kWaferDDROffsetAttrName);
      });
  EXPECT_FALSE(parentHasOffset);
}

TEST_F(CoordinatedTerminalEvaluationTest,
       DDRBoundaryRepairReentersAndCannotBypassTheExactSPMGate) {
  constexpr int64_t rankCount = 1;
  constexpr int64_t elements = 900'000;
  auto tileParent =
      makeDDRBoundaryRepairableSPMPressureTileProgram(elements);
  ASSERT_TRUE(tileParent);
  auto ledger =
      wafer::compiler::detail::CoordinatedWorkLedger::create(rankCount);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone,
                 1);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));
  auto terminal = wafer::compiler::detail::CoordinatedWorkLedger::
      getTerminalActionUpperBound(rankCount);
  ASSERT_TRUE(mlir::succeeded(terminal));
  auto parentReservation = ledger->tryReserveTerminalAction(*terminal);
  ASSERT_TRUE(parentReservation);

  wafer::compiler::detail::CoordinatedTileVariant parent;
  parent.stableSemanticOrdinal = 7;
  parent.terminalReservation = *parentReservation;
  parent.ranks.emplace_back(
      0, std::move(tileParent),
      std::make_shared<const std::string>("DDR-cut repairable Tile parent"));
  auto executionConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      rankCount, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig))
      << llvm::toString(executionConfig.takeError());
  auto program = replicatedProgram(rankCount, elements);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CoordinatedTerminalFailure failure;
  auto rejected = wafer::compiler::detail::evaluateCoordinatedTileVariant(
      parent, program, *executionConfig, wafer::OptimizationConfig::none(),
      *ledger, diagnostics, failure);
  ASSERT_TRUE(mlir::failed(rejected));
  ASSERT_EQ(failure.kind, wafer::compiler::detail::
                              CoordinatedTerminalFailureKind::SPMAllocation)
      << diagnosticsText;
  const uint64_t terminalReservedAfterParentFailure =
      ledger->getSnapshot().terminalReserved;

  std::string repairFailure;
  auto repaired = wafer::compiler::detail::materializeCoordinatedTileRepair(
      parent,
      wafer::compiler::detail::CoordinatedTileRepairAction::
          SplitAtExplicitDDRBoundary,
      /*stableSemanticOrdinal=*/8, *ledger, &repairFailure);
  ASSERT_TRUE(mlir::succeeded(repaired)) << repairFailure;
  unsigned repairedRegions = 0;
  repaired->ranks.front().module.get().walk(
      [&](wafer::TileRegionOp) { ++repairedRegions; });
  EXPECT_EQ(repairedRegions, 2u);

  auto rejectedRepair =
      wafer::compiler::detail::evaluateCoordinatedTileVariant(
      *repaired, program, *executionConfig, wafer::OptimizationConfig::none(),
      *ledger, diagnostics, failure);
  EXPECT_TRUE(mlir::failed(rejectedRepair));
  EXPECT_EQ(failure.kind, wafer::compiler::detail::
                              CoordinatedTerminalFailureKind::SPMAllocation)
      << diagnosticsText;
  EXPECT_EQ(ledger->getSnapshot().terminalReserved,
            terminalReservedAfterParentFailure);

  bool parentHasOffset = false;
  parent.ranks.front().module.get().walk(
      [&](mlir::memref::AllocOp allocation) {
        parentHasOffset |= allocation->hasAttr(wafer::kWaferSPMOffsetAttrName) ||
                           allocation->hasAttr(wafer::kWaferDDROffsetAttrName);
      });
  EXPECT_FALSE(parentHasOffset);
  bool repairedHasOffset = false;
  repaired->ranks.front().module.get().walk(
      [&](mlir::memref::AllocOp allocation) {
        repairedHasOffset |=
            allocation->hasAttr(wafer::kWaferSPMOffsetAttrName) ||
            allocation->hasAttr(wafer::kWaferDDROffsetAttrName);
      });
  EXPECT_FALSE(repairedHasOffset);
}

} // namespace
