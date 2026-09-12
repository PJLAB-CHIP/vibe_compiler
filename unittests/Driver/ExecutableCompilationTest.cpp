//===- ExecutableCompilationTest.cpp -------------------------------===//

#include "Wafer/Driver/ExecutableCompilation.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Driver/PhysicalDataflow/BaselineCurrentIR.h"
#include "Wafer/Driver/PhysicalDataflow/SearchCurrentIR.h"
#include "Wafer/Driver/ProgramData/ProgramData.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/TargetMemory.h"
#include "Wafer/Transforms/Instr/MemoryPlanning.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include "TestSupport/CodeGen/ExecutableTestSupport.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {

TEST(SearchCurrentIROptionsTest,
     RetentionWidthDoesNotChangeProposalGeneration) {
  wafer::compiler::detail::SearchCurrentIROptions options;
  for (uint64_t width : {1, 2, 3, 4, 8, 16}) {
    options.limits.width = width;
    EXPECT_EQ(options.getInitialProposalLimit(), 6u);
    EXPECT_EQ(options.getRefinementLimit(), 2u);
  }
}

class ExecutableCompilationTest : public ::testing::Test {
protected:
  ExecutableCompilationTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  llvm::SmallVector<wafer::TileId, 16> tileIds() const {
    llvm::SmallVector<wafer::TileId, 16> result;
    for (int64_t tile = 0; tile < 16; ++tile)
      result.push_back(wafer::TileId(tile));
    return result;
  }

  wafer::compiler::ExecutionConfig executionConfig() const {
    auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
    EXPECT_TRUE(static_cast<bool>(config));
    return *config;
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  oversizedTileModuleCollection(bool addInvalidPeer = false) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
)mlir";
    for (int64_t tile = 0; tile < 16; ++tile) {
      os << "  wafer.tile.module card_id = 0 tile_id = " << tile;
      if (tile != 0) {
        os << R"mlir( {
      func.func @main() {
        return
      }
    }
)mlir";
        continue;
      }
      os << R"mlir( {
      func.func @main(
          %boundary: memref<2000000xf16, #wafer.memory<ddr, tensor>>) {
        %unused = wafer.tile.region(
            %boundary : memref<2000000xf16,
                #wafer.memory<ddr, tensor>>) ->
            (memref<2000000xf16, #wafer.memory<ddr, tensor>>) {
        ^bb0(%ddr: memref<2000000xf16,
            #wafer.memory<ddr, tensor>>):
          %zero = arith.constant 0.000000e+00 : f16
          %spm = memref.alloc()
              : memref<2000000xf16, #wafer.memory<spm, tensor>>
          wafer.tile.load %ddr into %spm
              : memref<2000000xf16, #wafer.memory<ddr, tensor>>
                into memref<2000000xf16, #wafer.memory<spm, tensor>>
          wafer.tile.fill %spm, %zero
              : memref<2000000xf16, #wafer.memory<spm, tensor>>, f16
)mlir";
      if (addInvalidPeer)
        os << R"mlir(          %invalid_peer = wafer.instr.dte_send %spm
              {peer = 16 : i64, bytes = 16 : i64,
               message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
              : memref<2000000xf16, #wafer.memory<spm, tensor>> -> !async.token
)mlir";
      os << R"mlir(
          wafer.tile.yield %ddr
              : memref<2000000xf16, #wafer.memory<ddr, tensor>>
        }
        return
      }
    }
)mlir";
    }
    os << "}\n";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  canonicalInstructionModule(bool includeCompletion = true,
                             bool tensorBoundary = false) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
)mlir";
    if (tensorBoundary) {
      os << R"mlir(  func.func @main(%input: tensor<2x1025x64xf16>) {
    return
  }
)mlir";
    } else {
      os << R"mlir(  func.func @main() {
    %c0 = arith.constant 0 : index
    %unused = wafer.tile.region(%c0 : index) -> (index) {
    ^bb0(%index: index):
      %zero = arith.constant 0.000000e+00 : f16
      %spm = memref.alloc()
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %spm, %zero
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>, f16
)mlir";
      if (includeCompletion)
        os << "      wafer.instr.ncc_join [0]\n";
      os << R"mlir(      wafer.tile.yield %index : index
    }
    return
  }
)mlir";
    }
    os << "}\n";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> physicalTileModuleCollection() {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
)mlir";
    for (int64_t tile = 0; tile < 16; ++tile) {
      os << "  wafer.tile.module card_id = 0 tile_id = " << tile << " {\n";
      if (tile == 0) {
        os << R"mlir(    func.func @main() {
      %c0 = arith.constant 0 : index
      %unused = wafer.tile.region(%c0 : index) -> (index) {
      ^bb0(%index: index):
        %zero = arith.constant 0.000000e+00 : f16
        %buffer = memref.alloc()
            : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
        wafer.tile.fill %buffer, %zero
            : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>, f16
        %value = memref.load %buffer[%index, %index, %index]
            : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
        wafer.tile.yield %index : index
      }
      return
    }
)mlir";
      } else {
        os << R"mlir(    func.func @main() {
      return
    }
)mlir";
      }
      os << "  }\n";
    }
    os << "}\n";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  std::vector<wafer::compiler::detail::CanonicalInstructionTile>
  makeCanonicalInstructionTiles(mlir::ModuleOp source) {
    std::vector<wafer::compiler::detail::CanonicalInstructionTile> result;
    result.reserve(16);
    for (int64_t tile = 0; tile < 16; ++tile) {
      mlir::OwningOpRef<mlir::ModuleOp> module(
          mlir::cast<mlir::ModuleOp>(source->clone()));
      wafer::StructuredMaterializationRelations relations;
      module->walk([&](mlir::memref::AllocOp allocation) {
        relations.buffers.push_back({allocation, allocation.getResult(),
                                     wafer::MaterializedBufferRole::Scratch});
      });
      result.push_back({wafer::CardId(0), wafer::TileId(tile),
                        std::move(module), std::move(relations)});
    }
    return result;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST(ExecutableCompilationFailureTest,
     PreservesTypedMemoryPlanningFailureClasses) {
  using wafer::SPMMemoryPlanningFailureKind;
  using wafer::compiler::detail::ExecutableCompilationStatus;
  using wafer::compiler::detail::TileMemoryPlanningFailure;
  using wafer::compiler::detail::TileMemoryPlanningFailureKind;

  TileMemoryPlanningFailure failure;
  failure.kind = TileMemoryPlanningFailureKind::SPMAllocation;
  failure.spmPlanningFailureKind =
      SPMMemoryPlanningFailureKind::CapacityOverflow;
  EXPECT_EQ(wafer::compiler::detail::classifyTileMemoryPlanningFailure(failure),
            ExecutableCompilationStatus::ProvenExactRejection);

  failure.spmPlanningFailureKind =
      SPMMemoryPlanningFailureKind::ResourceExhausted;
  EXPECT_EQ(wafer::compiler::detail::classifyTileMemoryPlanningFailure(failure),
            ExecutableCompilationStatus::IndeterminateFailure);

  failure.spmPlanningFailureKind =
      SPMMemoryPlanningFailureKind::UnsupportedLifetime;
  EXPECT_EQ(wafer::compiler::detail::classifyTileMemoryPlanningFailure(failure),
            ExecutableCompilationStatus::UnsupportedFailure);

  failure.spmPlanningFailureKind =
      SPMMemoryPlanningFailureKind::MissingCompletion;
  EXPECT_EQ(wafer::compiler::detail::classifyTileMemoryPlanningFailure(failure),
            ExecutableCompilationStatus::CompilerFailure);

  failure.kind = TileMemoryPlanningFailureKind::Contract;
  failure.spmPlanningFailureKind = SPMMemoryPlanningFailureKind::None;
  EXPECT_EQ(wafer::compiler::detail::classifyTileMemoryPlanningFailure(failure),
            ExecutableCompilationStatus::CompilerFailure);
}

TEST_F(ExecutableCompilationTest,
       ProductionTransferCleanupRetargetsCurrentRelationsAndFeedsMiniMalloc) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %token = arith.constant false
    %unused = wafer.tile.region(%token : i1) -> (i1) {
    ^bb0(%arg0: i1):
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.000000e+00 : f16
      %source = memref.alloc()
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      %dest = memref.alloc()
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %source, %zero
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.ncc_join [0]
      wafer.instr.gather_scatter %source to %dest
          {byte_count = 262144 : i64, inner_bytes = 262144 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
         to memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
      %value = memref.load %dest[%c0, %c0, %c0]
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0 : i1
    }
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  mlir::memref::AllocOp source;
  mlir::memref::AllocOp dest;
  wafer::InstrGatherScatterOp gather;
  module->walk([&](mlir::memref::AllocOp allocation) {
    if (!source)
      source = allocation;
    else
      dest = allocation;
  });
  module->walk(
      [&](wafer::InstrGatherScatterOp operation) { gather = operation; });
  ASSERT_TRUE(source && dest && gather);
  wafer::StructuredMaterializationRelations relations;
  relations.buffers.push_back(
      {gather, source, wafer::MaterializedBufferRole::Operand});
  relations.buffers.push_back(
      {gather, dest, wafer::MaterializedBufferRole::Result});

  mlir::FailureOr<unsigned> eliminated =
      wafer::compiler::detail::cleanupCanonicalInstructionTransfers(*module,
                                                                    relations);
  ASSERT_TRUE(mlir::succeeded(eliminated));
  EXPECT_EQ(*eliminated, 1u);
  unsigned gathers = 0;
  unsigned allocations = 0;
  module->walk([&](wafer::InstrGatherScatterOp) { ++gathers; });
  module->walk([&](mlir::memref::AllocOp) { ++allocations; });
  EXPECT_EQ(gathers, 0u);
  EXPECT_EQ(allocations, 1u);
  EXPECT_FALSE(relations.buffers.empty());
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
  ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*module)));
  EXPECT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
      *module, /*spmBase=*/0, /*spmLimit=*/3 * 1024 * 1024,
      /*spmAlignment=*/16)));
}

TEST_F(ExecutableCompilationTest,
       TransferCleanupRebuildsOwnersAfterCreatingReplacementViews) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::string body = llvm::formatv(R"mlir(
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f16
      %source = memref.alloc()
          : memref<2x{0}x4xf16, #wafer.memory<spm, tensor>>
      %dest = memref.alloc()
          : memref<{0}x2x4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %source, %zero
          : memref<2x{0}x4xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.ncc_join [0]
      wafer.instr.gather_scatter %source to %dest
          {{byte_count = {1} : i64, inner_bytes = {1} : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x{0}x4xf16, #wafer.memory<spm, tensor>>
         to memref<{0}x2x4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
      %value = memref.load %dest[%c0, %c0, %c0]
          : memref<{0}x2x4xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0 : i1
    )mlir",
                                     extent, extent * 16)
                           .str();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        "module { func.func @main() { %token = arith.constant false "
        "%unused = wafer.tile.region(%token : i1) -> (i1) { ^bb0(%arg0: i1): " +
            body + "} return } }",
        context.get());
    ASSERT_TRUE(module);
    wafer::StructuredMaterializationRelations relations;
    wafer::compiler::detail::rebuildCurrentBufferOwnerRelations(
        module->getOperation(), relations);
    auto eliminated =
        wafer::compiler::detail::cleanupCanonicalInstructionTransfers(
            *module, relations);
    ASSERT_TRUE(mlir::succeeded(eliminated));
    EXPECT_EQ(*eliminated, 1u);
    unsigned views = 0;
    module->walk([&](mlir::memref::ReinterpretCastOp view) {
      ++views;
      EXPECT_TRUE(llvm::any_of(relations.buffers, [&](const auto &relation) {
        return relation.owner == view.getOperation() &&
               relation.buffer == view.getResult();
      }));
    });
    EXPECT_EQ(views, 1u);
    EXPECT_TRUE(mlir::succeeded(
        wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
            module->getOperation(), relations)));
    ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*module)));
    EXPECT_TRUE(mlir::succeeded(
        wafer::planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
  }
}

TEST_F(ExecutableCompilationTest,
       CanonicalInstrLeafOwnsActualMemoryTargetAndHighWater) {
  auto module = canonicalInstructionModule();
  ASSERT_TRUE(module);
  auto expectedTileIds = tileIds();
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::ExecutableLoweringStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;

  auto result =
      wafer::compiler::detail::compileCanonicalInstructionTilesToExecutable(
          makeCanonicalInstructionTiles(*module), wafer::CardId(0),
          expectedTileIds, program, executionConfig(), diagnostics, programData,
          &statistics);
  diagnostics.flush();
  ASSERT_TRUE(result.isAccepted()) << diagnosticText;
  ASSERT_TRUE(result.executable);
  ASSERT_EQ(result.executable->tiles.size(), 16u);
  ASSERT_EQ(result.executable->resourceCost.tileCosts.size(), 16u);
  const wafer::TargetMemoryPolicy memory = wafer::getTargetMemoryPolicy();
  for (auto [tileIndex, tile] : llvm::enumerate(result.executable->tiles)) {
    mlir::memref::AllocOp allocation;
    tile.getModule().walk([&](mlir::memref::AllocOp candidate) {
      if (wafer::isWaferSPMMemRefType(candidate.getType()))
        allocation = candidate;
    });
    ASSERT_TRUE(allocation);
    auto offset = allocation->getAttrOfType<wafer::SPMOffsetAttr>(
        wafer::kWaferSPMOffsetAttrName);
    ASSERT_TRUE(offset);
    auto physical = wafer::computeWaferPhysicalTensorInfo(allocation.getType());
    ASSERT_TRUE(physical.has_value());
    ASSERT_GE(offset.getOffset(), memory.spmBase);
    const uint64_t expectedHighWater =
        static_cast<uint64_t>(offset.getOffset() - memory.spmBase) +
        physical->physicalBytes;
    const auto &cost = result.executable->resourceCost.tileCosts[tileIndex];
    ASSERT_TRUE(cost.spmHighWaterBytes.isKnown());
    EXPECT_EQ(cost.spmHighWaterBytes.value, expectedHighWater);
    ASSERT_TRUE(cost.compilerOwnedSPMBufferCount.isKnown());
    EXPECT_EQ(cost.compilerOwnedSPMBufferCount.value, 1u);
  }
  EXPECT_EQ(statistics.executableCompilationInvocations, 0u);
  EXPECT_EQ(statistics.actualMemoryTargetGateInvocations, 1u);
  EXPECT_EQ(statistics.tileModuleLoweringAttempts, 1u);
  EXPECT_EQ(statistics.tileModuleLoweringSuccesses, 1u);
  EXPECT_EQ(statistics.deviceExecutablesProduced, 1u);
}

TEST_F(ExecutableCompilationTest,
       CurrentIRDownstreamReachesInstrMemoryAndExecutableExactlyOnce) {
  auto module = physicalTileModuleCollection();
  ASSERT_TRUE(module);
  wafer::StructuredMaterializationRelations relations;
  wafer::compiler::detail::rebuildCurrentBufferOwnerRelations(
      module->getOperation(), relations);
  auto expectedTileIds = tileIds();
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::ProgramDataHandoff programData;
  wafer::compiler::detail::CurrentIRDownstreamOptions options;
  options.captureTileDataflowIR = true;
  options.tilePipelineParallelism = 1;
  wafer::compiler::detail::CurrentIRDownstreamStatistics downstream;
  wafer::compiler::detail::ExecutableLoweringStatistics executable;

  auto result = wafer::compiler::detail::compileCurrentIRCandidateToExecutable(
      std::move(module), std::move(relations), wafer::CardId(0),
      expectedTileIds, program, executionConfig(), diagnostics, programData,
      options, &downstream, &executable);
  diagnostics.flush();
  ASSERT_TRUE(result.isAccepted())
      << result.gate << ": " << result.detail << "\n"
      << diagnosticText;
  ASSERT_TRUE(result.executable);
  ASSERT_TRUE(result.physicalIRInventory);
  EXPECT_EQ(result.physicalIRInventory->tileModules, 16u);
  EXPECT_GT(result.physicalIRInventory->tileRegions, 0u);
  ASSERT_EQ(result.physicalIRInventory->tiles.size(), 16u);
  EXPECT_EQ(result.physicalIRInventory->tiles.front().tile, wafer::TileId(0));
  EXPECT_EQ(result.physicalIRInventory->tiles.front().regions, 1u);
  for (const auto &tile : llvm::drop_begin(result.physicalIRInventory->tiles))
    EXPECT_EQ(tile.regions, 0u);
  EXPECT_EQ(result.executable->tiles.size(), 16u);
  EXPECT_EQ(result.tileDataflowIRTrace.size(), 16u);
  EXPECT_EQ(downstream.tileRegionsLowered, 1u);
  EXPECT_GE(downstream.instructionOperations, 1u);
  EXPECT_GE(downstream.nccJoinOperations, 1u);
  EXPECT_EQ(executable.actualMemoryTargetGateInvocations, 1u);
  EXPECT_EQ(executable.deviceExecutablesProduced, 1u);
}

TEST(ExecutableCompilationPolicyTest,
     DeterministicBaselineTraversesEveryCurrentIRStageWithoutSearchState) {
  wafer::compiler::testing::ParsedProgram parsed =
      wafer::compiler::testing::parseProgram();
  ASSERT_TRUE(parsed.module);
  auto program = wafer::compiler::testing::programMetadata();
  wafer::compiler::ProgramDataHandoff programData;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::BaselineCurrentIROptions options;
  options.downstream.captureTileDataflowIR = true;
  options.downstream.tilePipelineParallelism = 1;
  wafer::compiler::detail::BaselineCurrentIRStatistics baseline;
  wafer::compiler::detail::ExecutableLoweringStatistics executable;

  auto result = wafer::compiler::detail::compileBaselineCurrentIR(
      *parsed.module, program, wafer::compiler::testing::executionConfig(),
      diagnostics, programData, options, &baseline, &executable);
  diagnostics.flush();
  ASSERT_TRUE(result.isAccepted())
      << result.gate << ": " << result.detail << "\n"
      << diagnosticText;
  ASSERT_TRUE(result.executable);
  EXPECT_EQ(result.executable->tiles.size(), 16u);
  EXPECT_EQ(result.tileDataflowIRTrace.size(), 16u);
  EXPECT_EQ(baseline.attempts, 1u);
  EXPECT_EQ(baseline.spatialMaterializations, 1u);
  EXPECT_GT(baseline.temporalApplications, 0u);
  EXPECT_EQ(baseline.layoutInvocations, 1u);
  EXPECT_EQ(executable.actualMemoryTargetGateInvocations, 1u);
}

TEST(ExecutableCompilationPolicyTest,
     ParallelCurrentIRStagesPreserveExactInstructionAndStorage) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto parsed =
        wafer::compiler::testing::parseRealScaleDependentProgram(extent);
    ASSERT_TRUE(parsed.module);
    auto program =
        wafer::compiler::testing::realScaleDependentProgramMetadata(extent);
    std::vector<std::string> serial;
    for (unsigned workers : {1u, 4u, 16u}) {
      SCOPED_TRACE(workers);
      wafer::compiler::ProgramDataHandoff data;
      std::string diagnosticText;
      llvm::raw_string_ostream diagnostics(diagnosticText);
      wafer::compiler::detail::BaselineCurrentIROptions options;
      options.downstream.tilePipelineParallelism = workers;
      wafer::compiler::detail::ExecutableLoweringStatistics statistics;
      auto result = wafer::compiler::detail::compileBaselineCurrentIR(
          *parsed.module, program, wafer::compiler::testing::executionConfig(),
          diagnostics, data, options, nullptr, &statistics);
      ASSERT_TRUE(result.isAccepted()) << result.gate << ": " << result.detail;
      ASSERT_TRUE(result.executable);
      ASSERT_EQ(result.executable->tiles.size(), 16u);
      EXPECT_EQ(statistics.maximumTilePipelineWorkers, workers);
      std::vector<std::string> actual;
      for (const auto &tile : result.executable->tiles) {
        std::string text;
        llvm::raw_string_ostream stream(text);
        tile.getModule().print(stream);
        actual.push_back(std::move(text));
      }
      if (workers == 1)
        serial = std::move(actual);
      else
        EXPECT_EQ(actual, serial);
    }
  }
}

TEST(ExecutableCompilationPolicyTest,
     CommunicationClosurePreservesBaselineAndSearchDDRAlternatives) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto parsed = wafer::compiler::testing::parseProgram();
    std::string body = llvm::formatv(R"mlir(
      %empty = tensor.empty() : tensor<16x1x{0}xf16>
      %produced = linalg.add
          ins(%input, %input : tensor<16x1x{0}xf16>, tensor<16x1x{0}xf16>)
          outs(%empty : tensor<16x1x{0}xf16>) -> tensor<16x1x{0}xf16>
      %output = tensor.empty() : tensor<16x16x1x{0}xf16>
      %expanded = linalg.broadcast ins(%produced : tensor<16x1x{0}xf16>)
          outs(%output : tensor<16x16x1x{0}xf16>) dimensions = [0]
      %result = linalg.add
          ins(%lhs, %expanded : tensor<16x16x1x{0}xf16>, tensor<16x16x1x{0}xf16>)
          outs(%output : tensor<16x16x1x{0}xf16>) -> tensor<16x16x1x{0}xf16>
      // Both broadcast directions are consumed. Coordinating one edge cannot
      // remove the actual exchange required by the orthogonal edge.
      %other = linalg.broadcast ins(%produced : tensor<16x1x{0}xf16>)
          outs(%output : tensor<16x16x1x{0}xf16>) dimensions = [1]
      %combined = linalg.add
          ins(%result, %other : tensor<16x16x1x{0}xf16>, tensor<16x16x1x{0}xf16>)
          outs(%output : tensor<16x16x1x{0}xf16>) -> tensor<16x16x1x{0}xf16>
      return %combined : tensor<16x16x1x{0}xf16>
    )mlir",
                                     extent)
                           .str();
    std::string source =
        "module { wafer.target.topology @default "
        "{card_grid = array<i64: 1, 1>, card_interconnect = \"mesh\", "
        "tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>} "
        "wafer.execution.mesh @default_mesh "
        "{axes = [\"card\"], shape = array<i64: 1>} " +
        llvm::formatv("func.func @main(%lhs: tensor<16x16x1x{0}xf16>, "
                      "%input: tensor<16x1x{0}xf16>) -> "
                      "tensor<16x16x1x{0}xf16> {{",
                      extent)
            .str() +
        body + "} }";
    parsed.module =
        mlir::parseSourceString<mlir::ModuleOp>(source, parsed.context.get());
    ASSERT_TRUE(parsed.module);
    wafer::frontend::FrontendProgramVerificationResult program;
    program.numPartitions = 1;
    program.programUserInputCount = 2;
    program.distributedInputs = {
        wafer::compiler::testing::boundary(0, {16, 16, 1, extent}),
        wafer::compiler::testing::boundary(1, {16, 1, extent})};
    program.distributedOutputs = {
        wafer::compiler::testing::boundary(0, {16, 16, 1, extent})};
    std::string diagnosticText;
    llvm::raw_string_ostream diagnostics(diagnosticText);
    wafer::compiler::ProgramDataHandoff baselineData;
    wafer::compiler::detail::BaselineCurrentIROptions baselineOptions;
    baselineOptions.downstream.tilePipelineParallelism = 1;
    wafer::compiler::detail::BaselineCurrentIRStatistics baseline;
    auto baselineResult = wafer::compiler::detail::compileBaselineCurrentIR(
        *parsed.module, program, wafer::compiler::testing::executionConfig(),
        diagnostics, baselineData, baselineOptions, &baseline);
    ASSERT_TRUE(baselineResult.isAccepted()) << baselineResult.detail;
    EXPECT_EQ(baseline.communicationRegionClosures, 0u);
    ASSERT_TRUE(baselineResult.physicalIRInventory);
    EXPECT_GE(baselineResult.physicalIRInventory->tileRegions, 32u);

    llvm::SmallVector<mlir::ModuleOp> modules;
    llvm::SmallVector<wafer::TileId> tileIds;
    unsigned publications = 0, acquisitions = 0;
    for (auto &tile : baselineResult.executable->tiles) {
      auto module = tile.getModule();
      modules.push_back(module);
      tileIds.push_back(tile.getTileId());
      module.walk([&](wafer::SyncDDRPublishOp op) {
        ++publications;
        EXPECT_TRUE(
            mlir::isa_and_nonnull<wafer::SyncNCCJoinOp>(op->getPrevNode()));
      });
      module.walk([&](wafer::SyncDDRAcquireOp) { ++acquisitions; });
    }
    EXPECT_EQ(publications, 16u);
    EXPECT_EQ(acquisitions, 240u);
    auto completed = wafer::verifySharedDDRCompletion(modules, tileIds);
    EXPECT_TRUE(completed.succeeded()) << completed.detail;
    // Mutate only a private Tile Module clone; all other current owners stay
    // read-only. Missing/misplaced acquire must fail the same product gate.
    mlir::OwningOpRef<mlir::ModuleOp> missingAcquire(modules.front().clone());
    wafer::SyncDDRAcquireOp acquire;
    missingAcquire->walk([&](wafer::SyncDDRAcquireOp op) { acquire = op; });
    ASSERT_TRUE(acquire);
    auto original = modules.front();
    modules.front() = *missingAcquire;
    auto *next = acquire->getNextNode();
    auto *block = acquire->getBlock();
    acquire->moveBefore(block->getTerminator());
    EXPECT_EQ(wafer::verifySharedDDRCompletion(modules, tileIds).failure,
              wafer::SharedDDRCompletionFailure::Contract);
    acquire->moveBefore(next);
    EXPECT_TRUE(wafer::verifySharedDDRCompletion(modules, tileIds).succeeded());
    mlir::OpBuilder acquireBuilder(acquire);
    auto *extraAcquire = acquireBuilder.clone(*acquire);
    EXPECT_EQ(wafer::verifySharedDDRCompletion(modules, tileIds).failure,
              wafer::SharedDDRCompletionFailure::Contract);
    extraAcquire->erase();
    auto readyArgument = mlir::cast<mlir::BlockArgument>(acquire.getReady());
    while (!mlir::isa<mlir::func::FuncOp>(
        readyArgument.getOwner()->getParentOp())) {
      auto operand =
          wafer::analysis::getSingleExecutionRegionEntryOperand(readyArgument);
      ASSERT_TRUE(mlir::isa_and_nonnull<mlir::BlockArgument>(operand));
      readyArgument = mlir::cast<mlir::BlockArgument>(operand);
    }
    auto readyFunction =
        mlir::cast<mlir::func::FuncOp>(readyArgument.getOwner()->getParentOp());
    auto readyBinding = readyFunction.getArgAttrOfType<wafer::DDRBindingAttr>(
        readyArgument.getArgNumber(), wafer::kWaferDDRBindingAttrName);
    auto readyGlobal = missingAcquire->lookupSymbol<mlir::memref::GlobalOp>(
        readyBinding.getResource().getValue());
    ASSERT_TRUE(readyGlobal);
    auto initializer = readyGlobal.getInitialValueAttr();
    readyGlobal.setInitialValueAttr(acquireBuilder.getUnitAttr());
    EXPECT_EQ(wafer::verifySharedDDRCompletion(modules, tileIds).failure,
              wafer::SharedDDRCompletionFailure::Contract);
    readyGlobal.setInitialValueAttr(initializer);
    auto zeroIndex = acquireBuilder.create<mlir::arith::ConstantIndexOp>(
        acquire.getLoc(), 0);
    auto zeroByte = acquireBuilder.create<mlir::arith::ConstantIntOp>(
        acquire.getLoc(), 0, 8);
    auto unrelatedWrite = acquireBuilder.create<mlir::memref::StoreOp>(
        acquire.getLoc(), zeroByte, acquire.getReady(),
        mlir::ValueRange{zeroIndex});
    EXPECT_EQ(wafer::verifySharedDDRCompletion(modules, tileIds).failure,
              wafer::SharedDDRCompletionFailure::Unsupported);
    unrelatedWrite.erase();
    zeroByte.erase();
    zeroIndex.erase();
    acquire.erase();
    EXPECT_EQ(wafer::verifySharedDDRCompletion(modules, tileIds).failure,
              wafer::SharedDDRCompletionFailure::Contract);
    modules.front() = original;

    mlir::OwningOpRef<mlir::ModuleOp> duplicatePublish(original.clone());
    wafer::SyncDDRPublishOp publish;
    duplicatePublish->walk([&](wafer::SyncDDRPublishOp op) { publish = op; });
    ASSERT_TRUE(publish);
    mlir::OpBuilder builder(publish);
    builder.clone(*publish);
    modules.front() = *duplicatePublish;
    EXPECT_EQ(wafer::verifySharedDDRCompletion(modules, tileIds).failure,
              wafer::SharedDDRCompletionFailure::Contract);
    modules.front() = original;

    wafer::compiler::ProgramDataHandoff searchData;
    wafer::compiler::detail::SearchCurrentIROptions options;
    options.limits = wafer::SearchLimits{8, 42};
    options.downstream.tilePipelineParallelism = 1;
    wafer::compiler::detail::SearchCurrentIRStatistics search;
    auto result = wafer::compiler::detail::compileSearchCurrentIR(
        *parsed.module, program, wafer::compiler::testing::executionConfig(),
        diagnostics, searchData, options, &search);
    ASSERT_TRUE(result.isAccepted()) << result.detail << diagnosticText;
    EXPECT_GT(search.regionPreservingAccepted, 0u);
    EXPECT_GT(search.mergedRegionAccepted, 0u);
    EXPECT_GT(search.sharedDDRAccepted, 0u);
    EXPECT_LE(search.temporalCandidateActualizations, 42u);
    EXPECT_EQ(result.executable->tiles.size(), 16u);
  }
}

TEST(ExecutableCompilationPolicyTest,
     SearchActuallyCompilesReadOnlyInputSharingThroughLayoutConsumers) {
  using namespace wafer::compiler::detail;
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto parsed = wafer::compiler::testing::parseProgram();
    ASSERT_TRUE(parsed.module);
    std::string text = llvm::formatv(R"mlir(
module {{
  wafer.target.topology @default {{card_grid = array<i64: 1, 1>,
      card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {{axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<1x{0}x256xf16>, %rhs: tensor<1x256x512xf16>)
      -> tensor<1x{0}x512xf16> {{
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<1x{0}x512xf16>
    %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<1x{0}x512xf16>) -> tensor<1x{0}x512xf16>
    %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<1x{0}x256xf16>, tensor<1x256x512xf16>)
        outs(%init : tensor<1x{0}x512xf16>) -> tensor<1x{0}x512xf16>
    return %result : tensor<1x{0}x512xf16>
  }
}
)mlir",
                                     extent)
                           .str();
    parsed.module =
        mlir::parseSourceString<mlir::ModuleOp>(text, parsed.context.get());
    ASSERT_TRUE(parsed.module) << text;
    wafer::frontend::FrontendProgramVerificationResult program;
    program.numPartitions = 1;
    program.programUserInputCount = 2;
    program.distributedInputs = {
        wafer::compiler::testing::boundary(0, {1, extent, 256}),
        wafer::compiler::testing::boundary(1, {1, 256, 512})};
    program.distributedOutputs = {
        wafer::compiler::testing::boundary(0, {1, extent, 512})};
    wafer::compiler::ProgramDataHandoff data;
    std::string diagnosticText;
    llvm::raw_string_ostream diagnostics(diagnosticText);
    SearchCurrentIROptions options;
    options.limits = wafer::SearchLimits{8, 42};
    options.downstream.tilePipelineParallelism = 1;
    SearchCurrentIRStatistics statistics;
    auto result = compileSearchCurrentIR(
        *parsed.module, program, wafer::compiler::testing::executionConfig(),
        diagnostics, data, options, &statistics);
    ASSERT_TRUE(result.isAccepted()) << result.detail << diagnosticText;
    EXPECT_GT(statistics.inputSharingCandidates, 0u);
    EXPECT_GT(statistics.inputSharingAccepted, 0u);
    EXPECT_GT(statistics.acceptedCandidates, statistics.inputSharingAccepted);
    EXPECT_EQ(statistics.traversal.candidateActualizations, 42u);
    ASSERT_TRUE(result.executable);
    EXPECT_EQ(result.executable->tiles.size(), 16u);
  }
}

TEST(ExecutableCompilationPolicyTest,
     SearchRetainsLeafCapacityFeedbackWhenAlternativesAreIncomplete) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto parsed = wafer::compiler::testing::parseProgram();
    std::string body = llvm::formatv(R"mlir(
      %empty = tensor.empty() : tensor<16x128x{0}xf16>
      %produced = linalg.add
          ins(%input, %input : tensor<16x128x{0}xf16>, tensor<16x128x{0}xf16>)
          outs(%empty : tensor<16x128x{0}xf16>) -> tensor<16x128x{0}xf16>
      %output = tensor.empty() : tensor<16x16x128x{0}xf16>
      %expanded = linalg.broadcast ins(%produced : tensor<16x128x{0}xf16>)
          outs(%output : tensor<16x16x128x{0}xf16>) dimensions = [0]
      %result = linalg.add
          ins(%lhs, %expanded : tensor<16x16x128x{0}xf16>, tensor<16x16x128x{0}xf16>)
          outs(%output : tensor<16x16x128x{0}xf16>) -> tensor<16x16x128x{0}xf16>
      // Both broadcast directions are consumed. Coordinating one edge cannot
      // remove the actual exchange required by the orthogonal edge.
      %other = linalg.broadcast ins(%produced : tensor<16x128x{0}xf16>)
          outs(%output : tensor<16x16x128x{0}xf16>) dimensions = [1]
      %combined = linalg.add
          ins(%result, %other : tensor<16x16x128x{0}xf16>, tensor<16x16x128x{0}xf16>)
          outs(%output : tensor<16x16x128x{0}xf16>) -> tensor<16x16x128x{0}xf16>
      return %combined : tensor<16x16x128x{0}xf16>
    )mlir",
                                     extent)
                           .str();
    std::string source =
        "module { wafer.target.topology @default "
        "{card_grid = array<i64: 1, 1>, card_interconnect = \"mesh\", "
        "tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>} "
        "wafer.execution.mesh @default_mesh "
        "{axes = [\"card\"], shape = array<i64: 1>} " +
        llvm::formatv("func.func @main(%lhs: tensor<16x16x128x{0}xf16>, "
                      "%input: tensor<16x128x{0}xf16>) -> "
                      "tensor<16x16x128x{0}xf16> {{",
                      extent)
            .str() +
        body + "} }";
    parsed.module =
        mlir::parseSourceString<mlir::ModuleOp>(source, parsed.context.get());
    ASSERT_TRUE(parsed.module);
    wafer::frontend::FrontendProgramVerificationResult program;
    program.numPartitions = 1;
    program.programUserInputCount = 2;
    program.distributedInputs = {
        wafer::compiler::testing::boundary(0, {16, 16, 128, extent}),
        wafer::compiler::testing::boundary(1, {16, 128, extent})};
    program.distributedOutputs = {
        wafer::compiler::testing::boundary(0, {16, 16, 128, extent})};
    wafer::compiler::ProgramDataHandoff programData;
    std::string diagnosticText;
    llvm::raw_string_ostream diagnostics(diagnosticText);
    wafer::compiler::detail::SearchCurrentIROptions options;
    options.limits = wafer::SearchLimits{1, 8};

    options.downstream.tilePipelineParallelism = 1;
    wafer::compiler::detail::SearchCurrentIRStatistics search;
    auto result = wafer::compiler::detail::compileSearchCurrentIR(
        *parsed.module, program, wafer::compiler::testing::executionConfig(),
        diagnostics, programData, options, &search);
    // A short prefix may contain only actual capacity rejections. It must
    // retain that feedback without promoting the incomplete domain to an
    // exact rejection. Acceptance belongs to the extended prefix below.
    if (result.isAccepted()) {
      EXPECT_EQ(
          search.coverage,
          wafer::compiler::detail::SearchControllerCoverage::FeasiblePartial);
    } else {
      EXPECT_EQ(result.status,
                wafer::compiler::detail::ExecutableCompilationStatus::
                    IndeterminateFailure)
          << result.detail << diagnosticText;
      EXPECT_EQ(search.coverage,
                wafer::compiler::detail::SearchControllerCoverage::
                    IncompleteNoCandidate);
    }
    EXPECT_FALSE(result.isProvenExactRejection());
    EXPECT_EQ(search.traversal.peakRetainedBranches, 1u);
    EXPECT_EQ(search.traversal.candidateActualizations, 8u);
    EXPECT_GT(search.actualCapacityRefinements, 0u) << diagnosticText;
    EXPECT_GT(search.traversal.resumedCandidates, 0u);
    EXPECT_GT(search.movementCandidateActualizations, 2u);
    EXPECT_GT(search.temporalCandidateActualizations, 2u);
    EXPECT_LE(search.movementCandidateActualizations, 8u);
    const auto prefixRefinements = search.actualCapacityRefinements;
    const auto prefixAccepted = search.acceptedCandidates;
    options.limits.trials = 42;
    search = {};
    result = wafer::compiler::detail::compileSearchCurrentIR(
        *parsed.module, program, wafer::compiler::testing::executionConfig(),
        diagnostics, programData, options, &search);
    ASSERT_TRUE(result.isAccepted()) << result.detail << diagnosticText;
    EXPECT_EQ(
        search.coverage,
        wafer::compiler::detail::SearchControllerCoverage::FeasiblePartial);
    EXPECT_EQ(search.traversal.candidateActualizations, 42u);
    EXPECT_GE(search.actualCapacityRefinements, prefixRefinements);
    EXPECT_GE(search.acceptedCandidates, prefixAccepted);
  }
}

TEST(ExecutableCompilationPolicyTest,
     SearchActualizesCurrentIRAndRetainsTheAcceptedOwner) {
  wafer::compiler::testing::ParsedProgram parsed =
      wafer::compiler::testing::parseProgram();
  ASSERT_TRUE(parsed.module);
  auto program = wafer::compiler::testing::programMetadata();
  wafer::compiler::ProgramDataHandoff programData;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::SearchCurrentIROptions options;

  options.termination =
      wafer::compiler::detail::SearchTerminationPolicy::FirstAccepted;
  options.downstream.captureTileDataflowIR = true;
  options.downstream.tilePipelineParallelism = 1;
  wafer::compiler::detail::SearchCurrentIRStatistics search;
  wafer::compiler::detail::ExecutableLoweringStatistics executable;

  auto result = wafer::compiler::detail::compileSearchCurrentIR(
      *parsed.module, program, wafer::compiler::testing::executionConfig(),
      diagnostics, programData, options, &search, &executable);
  diagnostics.flush();
  ASSERT_TRUE(result.isAccepted())
      << result.gate << ": " << result.detail << "\n"
      << diagnosticText;
  ASSERT_TRUE(result.executable);
  EXPECT_EQ(result.executable->tiles.size(), 16u);
  EXPECT_EQ(result.tileDataflowIRTrace.size(), 16u);
  EXPECT_EQ(search.structuralMaterializations, 1u);
  EXPECT_GE(search.temporalCandidateActualizations, 1u);
  EXPECT_LE(search.temporalCandidateActualizations, 2u);
  EXPECT_EQ(search.layoutInvocations, search.temporalCandidateActualizations);
  EXPECT_GE(search.acceptedCandidates, 1u);
  EXPECT_EQ(search.controller.accepted, 1u);
  EXPECT_EQ(search.coverage,
            wafer::compiler::detail::SearchControllerCoverage::FeasiblePartial);
  EXPECT_EQ(executable.actualMemoryTargetGateInvocations,
            search.temporalCandidateActualizations);
  EXPECT_EQ(search.movementCandidateActualizations,
            search.temporalCandidateActualizations);
  EXPECT_EQ(search.recursiveDoublingCandidates, 0u);
}

TEST(ExecutableCompilationPolicyTest,
     SpatialOperandReuseReducesActualReadsThroughTheCommonLeaf) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto parsed = wafer::compiler::testing::parseProgram();
    ASSERT_TRUE(parsed.module);
    const std::string rhs = "tensor<1x4096x" + std::to_string(extent) + "xf16>";
    const std::string output =
        "tensor<1x16x" + std::to_string(extent) + "xf16>";
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(module {
      wafer.target.topology @default
        {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
         tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
      wafer.execution.mesh @default_mesh
        {axes = ["card"], shape = array<i64: 1>}
      func.func @main(%lhs: tensor<1x16x4096xf16>, %rhs: )mlir"
           << rhs << ") -> " << output << " {\n"
           << "%zero = arith.constant 0.0 : f16\n"
           << "%empty = tensor.empty() : " << output << "\n"
           << "%init = linalg.fill ins(%zero : f16) outs(%empty : " << output
           << ") -> " << output << "\n"
           << "%result = linalg.batch_matmul ins(%lhs, %rhs : "
              "tensor<1x16x4096xf16>, "
           << rhs << ") outs(%init : " << output << ") -> " << output << "\n"
           << "%point = linalg.add ins(%result, %result : " << output << ", "
           << output << ") outs(%empty : " << output << ") -> " << output
           << "\n"
           << "return %point : " << output << "\n}}";
    parsed.module = mlir::parseSourceString<mlir::ModuleOp>(
        stream.str(), parsed.context.get());
    ASSERT_TRUE(parsed.module);
    wafer::frontend::FrontendProgramVerificationResult program;
    program.numPartitions = 1;
    program.programUserInputCount = 2;
    program.distributedInputs = {
        wafer::compiler::testing::boundary(0, {1, 16, 4096}),
        wafer::compiler::testing::boundary(1, {1, 4096, extent})};
    program.distributedOutputs = {
        wafer::compiler::testing::boundary(0, {1, 16, extent})};
    uint64_t baselineReads = 0;
    for (bool search : {false, true}) {
      SCOPED_TRACE(search);
      wafer::compiler::ProgramDataHandoff data;
      std::string diagnosticText;
      llvm::raw_string_ostream diagnostics(diagnosticText);
      auto result = [&]() {
        if (search) {
          wafer::compiler::detail::SearchCurrentIROptions options;
          options.termination =
              wafer::compiler::detail::SearchTerminationPolicy::FirstAccepted;
          options.downstream.tilePipelineParallelism = 1;
          return wafer::compiler::detail::compileSearchCurrentIR(
              *parsed.module, program,
              wafer::compiler::testing::executionConfig(), diagnostics, data,
              options);
        }
        wafer::compiler::detail::BaselineCurrentIROptions options;
        options.downstream.tilePipelineParallelism = 1;
        return wafer::compiler::detail::compileBaselineCurrentIR(
            *parsed.module, program,
            wafer::compiler::testing::executionConfig(), diagnostics, data,
            options);
      }();
      ASSERT_TRUE(result.isAccepted())
          << result.gate << ": " << result.detail << "\n"
          << diagnosticText;
      ASSERT_TRUE(result.executable);
      ASSERT_EQ(result.executable->tiles.size(), 16u);
      const auto &reads = result.executable->resourceCost.aggregateDDRReadBytes;
      ASSERT_TRUE(reads.isKnown());
      if (search)
        EXPECT_LT(reads.value, baselineReads / 2);
      else
        baselineReads = reads.value;
      for (const auto &tile : result.executable->tiles) {
        unsigned contractions = 0;
        unsigned publications = 0;
        tile.getModule().walk([&](wafer::SyncDDRPublishOp) { ++publications; });
        tile.getModule().walk([&](wafer::InstrGemmOp gemm) {
          ++contractions;
          if (search)
            EXPECT_GT(gemm.getM(), 1);
          else
            EXPECT_EQ(gemm.getM(), 1);
        });
        EXPECT_GT(contractions, 0u);
        if (search) {
          EXPECT_EQ(publications, 0u)
              << "coordinated GEMM, pointwise and initializer need no peer DDR";
        }
        EXPECT_TRUE(mlir::succeeded(mlir::verify(tile.getModule())));
      }
    }
  }
}

TEST(ExecutableCompilationPolicyTest,
     SearchRefinesTemporalChoiceOnlyAfterActualSPMCapacityRejection) {
  wafer::compiler::testing::ParsedProgram parsed =
      wafer::compiler::testing::parseLargeTemporalProgram();
  ASSERT_TRUE(parsed.module);
  auto program = wafer::compiler::testing::largeTemporalProgramMetadata();
  wafer::compiler::ProgramDataHandoff programData;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::SearchCurrentIROptions options;
  options.termination =
      wafer::compiler::detail::SearchTerminationPolicy::FirstAccepted;
  options.downstream.tilePipelineParallelism = 1;
  wafer::compiler::detail::SearchCurrentIRStatistics search;
  wafer::compiler::detail::ExecutableLoweringStatistics executable;

  auto result = wafer::compiler::detail::compileSearchCurrentIR(
      *parsed.module, program, wafer::compiler::testing::executionConfig(),
      diagnostics, programData, options, &search, &executable);
  diagnostics.flush();
  ASSERT_TRUE(result.isAccepted())
      << result.gate << ": " << result.detail << "\n"
      << diagnosticText;
  EXPECT_GT(search.structuralMaterializations, 0u);
  EXPECT_GT(search.traversal.resumedCandidates, 0u);
  EXPECT_GT(search.exactRejectedCandidates, 0u);
  EXPECT_EQ(search.acceptedCandidates, 1u);
  EXPECT_EQ(search.temporalCandidateActualizations,
            search.exactRejectedCandidates + 1);
  EXPECT_EQ(search.layoutInvocations, search.temporalCandidateActualizations);
  EXPECT_EQ(executable.actualMemoryTargetGateInvocations,
            search.temporalCandidateActualizations);
  EXPECT_EQ(search.movementCandidateActualizations,
            search.temporalCandidateActualizations);
  EXPECT_EQ(search.recursiveDoublingCandidates, 0u);
}

TEST(ExecutableCompilationPolicyTest,
     SearchSelectsTheCoherentEndpointWithinConfiguredLimits) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    wafer::compiler::testing::ParsedProgram parsed =
        wafer::compiler::testing::parseRealScaleDependentProgram(extent);
    ASSERT_TRUE(parsed.module);
    auto program =
        wafer::compiler::testing::realScaleDependentProgramMetadata(extent);
    wafer::compiler::ProgramDataHandoff programData;
    std::string diagnosticText;
    llvm::raw_string_ostream diagnostics(diagnosticText);
    wafer::compiler::detail::SearchCurrentIROptions options;
    options.limits = wafer::SearchLimits{2, 32};

    options.termination =
        wafer::compiler::detail::SearchTerminationPolicy::Exhaustive;
    options.downstream.tilePipelineParallelism = 1;
    wafer::compiler::detail::SearchCurrentIRStatistics search;
    wafer::compiler::detail::ExecutableLoweringStatistics executable;

    auto result = wafer::compiler::detail::compileSearchCurrentIR(
        *parsed.module, program, wafer::compiler::testing::executionConfig(),
        diagnostics, programData, options, &search, &executable);
    diagnostics.flush();
    ASSERT_TRUE(result.isAccepted())
        << result.gate << ": " << result.detail << "\n"
        << diagnosticText;
    ASSERT_TRUE(result.physicalIRInventory);
    EXPECT_GE(search.structuralMaterializations, 2u);
    EXPECT_LE(search.traversal.peakRetainedBranches, 2u);
    EXPECT_EQ(search.traversal.candidateActualizations, 32u);
    EXPECT_GE(search.controller.accepted, 2u);
    EXPECT_EQ(result.physicalIRInventory->tileModules, 16u);
    // The singleton has 32 Regions. The coherent endpoint remains reachable
    // with only two simultaneously retained branches and removes all sixteen
    // producer/consumer boundaries, independent of total visited structures.
    EXPECT_EQ(result.physicalIRInventory->tileRegions, 16u);
    ASSERT_EQ(result.physicalIRInventory->tiles.size(), 16u);
    for (const auto &tile : result.physicalIRInventory->tiles)
      EXPECT_EQ(tile.regions, 1u);
  }
}

TEST(ExecutableCompilationPolicyTest,
     BaselineRefinesOnlyAfterActualSPMCapacityRejection) {
  wafer::compiler::testing::ParsedProgram parsed =
      wafer::compiler::testing::parseLargeTemporalProgram();
  ASSERT_TRUE(parsed.module);
  auto program = wafer::compiler::testing::largeTemporalProgramMetadata();
  wafer::compiler::ProgramDataHandoff programData;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::BaselineCurrentIROptions options;
  options.maximumCapacityAttempts = 16;
  options.downstream.tilePipelineParallelism = 1;
  wafer::compiler::detail::BaselineCurrentIRStatistics baseline;
  wafer::compiler::detail::ExecutableLoweringStatistics executable;

  auto result = wafer::compiler::detail::compileBaselineCurrentIR(
      *parsed.module, program, wafer::compiler::testing::executionConfig(),
      diagnostics, programData, options, &baseline, &executable);
  diagnostics.flush();
  ASSERT_TRUE(result.isAccepted())
      << result.gate << ": " << result.detail << "\n"
      << diagnosticText << " attempts=" << baseline.attempts
      << " refinements=" << baseline.capacityRefinements;
  EXPECT_GT(baseline.attempts, 1u);
  EXPECT_EQ(baseline.capacityRefinements, baseline.attempts - 1);
  EXPECT_EQ(baseline.spatialMaterializations, baseline.attempts);
  EXPECT_EQ(baseline.layoutInvocations, baseline.attempts);
  EXPECT_EQ(executable.actualMemoryTargetGateInvocations, baseline.attempts);
}

TEST_F(ExecutableCompilationTest,
       CanonicalInstrLeafRejectsTensorBoundaryWithoutBufferizing) {
  auto module = canonicalInstructionModule(/*includeCompletion=*/true,
                                           /*tensorBoundary=*/true);
  ASSERT_TRUE(module);
  auto expectedTileIds = tileIds();
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  std::string mlirDiagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(mlirDiagnosticText);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  wafer::compiler::detail::ExecutableLoweringStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;

  auto result =
      wafer::compiler::detail::compileCanonicalInstructionTilesToExecutable(
          makeCanonicalInstructionTiles(*module), wafer::CardId(0),
          expectedTileIds, program, executionConfig(), diagnostics, programData,
          &statistics, /*tilePipelineParallelism=*/1);
  EXPECT_EQ(
      result.status,
      wafer::compiler::detail::ExecutableCompilationStatus::CompilerFailure);
  EXPECT_EQ(result.gate, "tile-memory-planning");
  ASSERT_EQ(result.tileFailures.size(), 16u);
  for (const auto &failure : result.tileFailures)
    EXPECT_EQ(failure.memoryPlanning.kind,
              wafer::compiler::detail::TileMemoryPlanningFailureKind::Contract);
  EXPECT_EQ(statistics.actualMemoryTargetGateInvocations, 1u);
  EXPECT_EQ(statistics.tileModuleLoweringAttempts, 0u);
}

TEST_F(ExecutableCompilationTest,
       CanonicalInstrLeafRejectsMissingCompletionWithoutRepair) {
  auto module = canonicalInstructionModule(/*includeCompletion=*/false);
  ASSERT_TRUE(module);
  auto expectedTileIds = tileIds();
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  std::string mlirDiagnosticText;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(mlirDiagnosticText);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  wafer::compiler::detail::ExecutableLoweringStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;

  auto result =
      wafer::compiler::detail::compileCanonicalInstructionTilesToExecutable(
          makeCanonicalInstructionTiles(*module), wafer::CardId(0),
          expectedTileIds, program, executionConfig(), diagnostics, programData,
          &statistics, /*tilePipelineParallelism=*/1);
  diagnostics.flush();
  EXPECT_EQ(
      result.status,
      wafer::compiler::detail::ExecutableCompilationStatus::CompilerFailure);
  EXPECT_EQ(result.gate, "spm-allocation");
  ASSERT_EQ(result.tileFailures.size(), 16u);
  for (const auto &failure : result.tileFailures)
    EXPECT_EQ(failure.memoryPlanning.spmPlanningFailureKind,
              wafer::SPMMemoryPlanningFailureKind::MissingCompletion);
  EXPECT_NE(mlirDiagnosticText.find("missing_local_completion"),
            std::string::npos)
      << mlirDiagnosticText;
  EXPECT_EQ(statistics.actualMemoryTargetGateInvocations, 1u);
  EXPECT_EQ(statistics.tileModuleLoweringAttempts, 0u);
}

TEST(ExecutableCompilationPolicyTest,
     SharedDDRChecksActualDTESendAndWaitOrder) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (unsigned mode : {0, 1, 2, 3, 4}) {
      bool cycle = mode == 1;
      SCOPED_TRACE(extent);
      SCOPED_TRACE(mode);
      auto parsed = wafer::compiler::testing::parseProgram();
      llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 2> owners;
      llvm::SmallVector<mlir::ModuleOp, 2> modules;
      llvm::SmallVector<wafer::TileId, 2> tiles;
      for (int64_t tile = 0; tile < 2; ++tile) {
        std::string shape = "1x" + std::to_string(extent) + "x1xf16";
        std::string ddr = "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
        std::string spm = "memref<" + shape + ", #wafer.memory<spm, tensor>>";
        std::string text;
        llvm::raw_string_ostream ir(text);
        ir << "module { memref.global \"private\" @data : " << ddr
           << " {wafer.ddr_resource = #wafer.ddr_resource<0>}\n"
           << "func.func @entry(%data: " << ddr << " {wafer.ddr_binding = "
           << "#wafer.ddr_binding<@data, id = 0, " << (tile ? "read" : "write")
           << ">}, %condition: i1) {\n";
        unsigned regionIndex = 0;
        auto beginRegion = [&]() {
          ir << "%region" << regionIndex++
             << " = wafer.tile.region(%data, %condition : " << ddr
             << ", i1) -> (i1) { ^bb0(%arg: " << ddr
             << ", %flag: i1): %buffer = memref.alloc() : " << spm << "\n";
        };
        auto transport = [&]() {
          ir << "%event = wafer.instr.dte_" << (tile ? "send" : "recv")
             << " %buffer {peer = " << 1 - tile
             << " : i64, bytes = " << 2 * extent
             << " : i64, message = #wafer.dte_message<communication = 0, round "
                "= 0, slice = 0>} : "
             << spm << " -> !async.token\n"
             << "wafer.instr.dte_wait %event : !async.token\n";
        };
        beginRegion();
        if (!tile && cycle)
          transport();
        if (mode == 4)
          ir << "%lb = arith.constant 0 : index\n"
                "%ub = arith.constant 3 : index\n"
                "%step = arith.constant 1 : index\n"
                "scf.for %i = %lb to %ub step %step {\n";
        ir << "wafer.instr."
           << (tile ? "rdma %arg to %buffer" : "wdma %buffer to %arg")
           << " {byte_count = " << 2 * extent
           << " : i64, inner_bytes = " << 2 * extent << " : i64, "
           << (tile ? "src" : "dst") << "_iterations = array<i64: 1, 1, 1>, "
           << (tile ? "src" : "dst")
           << "_strides = array<i64: 0, 0, 0>} : " << (tile ? ddr : spm)
           << " to " << (tile ? spm : ddr) << "\n";
        if (mode == 4)
          ir << "}\n";
        ir << "wafer.instr.ncc_join [0]\n";
        if (tile || mode == 3)
          transport();
        ir << "wafer.tile.yield %flag : i1 }\n";
        if (!tile && !cycle && mode != 3) {
          beginRegion();
          if (mode == 2)
            ir << "scf.if %flag {\n";
          transport();
          if (mode == 2)
            ir << "}\n";
          ir << "wafer.tile.yield %flag : i1 }\n";
        }
        ir << "return } }\n";
        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(text, parsed.context.get());
        ASSERT_TRUE(module) << text;
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        modules.push_back(*module);
        owners.push_back(std::move(module));
        tiles.push_back(wafer::TileId(tile));
      }
      auto completion = wafer::materializeSharedDDRCompletion(modules, tiles);
      if (mode == 1 || mode == 2) {
        EXPECT_EQ(completion.failure,
                  wafer::SharedDDRCompletionFailure::Unsupported);
        EXPECT_NE(completion.detail.find(cycle ? "cycle" : "conditional"),
                  std::string::npos);
        if (cycle) {
          for (llvm::StringRef evidence :
               {"cycle-edge from=", "kind=tile-order", "kind=token-completion",
                "kind=ddr-publication", "tile=0", "tile=1"})
            EXPECT_NE(completion.detail.find(evidence.str()), std::string::npos)
                << completion.detail;
        }
        continue;
      }
      ASSERT_TRUE(completion.succeeded()) << completion.detail;
      EXPECT_TRUE(wafer::verifySharedDDRCompletion(modules, tiles).succeeded());
      unsigned publishes = 0, acquires = 0, waits = 0;
      for (auto module : modules) {
        ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
        module.walk([&](wafer::SyncDDRPublishOp op) {
          ++publishes;
          EXPECT_TRUE(mlir::isa<wafer::TileRegionOp>(op->getParentOp()));
          EXPECT_FALSE(op->getParentOfType<mlir::scf::ForOp>());
          if (mode == 4) {
            EXPECT_TRUE(mlir::isa<mlir::scf::ForOp>(op->getPrevNode()));
          }
          if (mode == 3) {
            wafer::InstrDTERecvOp receive;
            op->getParentOp()->walk(
                [&](wafer::InstrDTERecvOp recv) { receive = recv; });
            ASSERT_TRUE(receive);
            EXPECT_TRUE(op->isBeforeInBlock(receive));
          }
        });
        module.walk([&](wafer::SyncDDRAcquireOp op) {
          ++acquires;
          EXPECT_TRUE(mlir::isa<wafer::TileRegionOp>(op->getParentOp()));
          EXPECT_FALSE(op->getParentOfType<mlir::scf::ForOp>());
          if (mode == 4) {
            EXPECT_TRUE(mlir::isa<mlir::scf::ForOp>(op->getNextNode()));
          }
        });
        module.walk([&](wafer::InstrDTEWaitOp) { ++waits; });
      }
      EXPECT_EQ(publishes, 1u);
      EXPECT_EQ(acquires, 1u);
      EXPECT_EQ(waits, 2u);
    }
  }
}

TEST(ExecutableCompilationPolicyTest,
     SharedDDRManyResourcesPreserveExactReadersAndFreshValidation) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    // A wide actual communication graph exercises publication lookup scaling;
    // every payload retains a real rank-3 DMA extent.
    constexpr unsigned resources = 1024;
    auto parsed = wafer::compiler::testing::parseProgram();
    llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> owners;
    llvm::SmallVector<mlir::ModuleOp> modules;
    llvm::SmallVector<wafer::TileId> tiles;
    std::string shape = "1x" + std::to_string(extent) + "x1xf16";
    std::string ddr = "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
    std::string spm = "memref<" + shape + ", #wafer.memory<spm, tensor>>";
    for (unsigned tile = 0; tile < 4; ++tile) {
      std::string text;
      llvm::raw_string_ostream ir(text);
      ir << "module {\n";
      for (unsigned id = 0; id < resources; ++id)
        ir << "memref.global \"private\" @data" << id << " : " << ddr
           << " {wafer.ddr_resource = #wafer.ddr_resource<" << id << ">}\n";
      ir << "func.func @entry(";
      for (unsigned id = 0; id < resources; ++id) {
        if (id)
          ir << ", ";
        unsigned relative = (tile + 4 - id % 4) % 4;
        ir << "%data" << id << ": " << ddr << " {wafer.ddr_binding = "
           << "#wafer.ddr_binding<@data" << id << ", id = " << id << ", "
           << (relative == 0   ? "write"
               : relative == 3 ? "none"
                               : "read")
           << ">}";
      }
      ir << ") {\n%condition = arith.constant true\n";
      for (unsigned id = 0; id < resources; ++id) {
        unsigned relative = (tile + 4 - id % 4) % 4;
        if (relative == 3)
          continue;
        bool write = relative == 0;
        ir << "%result" << id << " = wafer.tile.region(%data" << id
           << ", %condition : " << ddr << ", i1) -> (i1) { ^bb0(%arg: " << ddr
           << ", %flag: i1): "
           << "%buffer = memref.alloc() : " << spm << "\n"
           << "wafer.instr."
           << (write ? "wdma %buffer to %arg" : "rdma %arg to %buffer")
           << " {byte_count = " << 2 * extent
           << " : i64, inner_bytes = " << 2 * extent << " : i64, "
           << (write ? "dst" : "src") << "_iterations = array<i64: 1, 1, 1>, "
           << (write ? "dst" : "src")
           << "_strides = array<i64: 0, 0, 0>} : " << (write ? spm : ddr)
           << " to " << (write ? ddr : spm)
           << "\nwafer.instr.ncc_join [0]\nwafer.tile.yield %flag : i1 }\n";
      }
      ir << "return } }";
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(text, parsed.context.get());
      ASSERT_TRUE(module);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      modules.push_back(*module);
      owners.push_back(std::move(module));
      tiles.push_back(wafer::TileId(tile));
    }
    auto result = wafer::materializeSharedDDRCompletion(modules, tiles);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    unsigned publications = 0, acquisitions = 0;
    wafer::SyncDDRPublishOp first;
    for (auto module : modules) {
      module.walk([&](wafer::SyncDDRPublishOp op) {
        ++publications;
        if (!first)
          first = op;
      });
      module.walk([&](wafer::SyncDDRAcquireOp) { ++acquisitions; });
    }
    EXPECT_EQ(publications, resources);
    EXPECT_EQ(acquisitions, 2 * resources);
    ASSERT_TRUE(first);
    mlir::OpBuilder builder(first);
    auto *duplicate = builder.clone(*first);
    EXPECT_EQ(wafer::verifySharedDDRCompletion(modules, tiles).failure,
              wafer::SharedDDRCompletionFailure::Contract);
    duplicate->erase();
    EXPECT_TRUE(wafer::verifySharedDDRCompletion(modules, tiles).succeeded());
    first.erase();
    EXPECT_EQ(wafer::verifySharedDDRCompletion(modules, tiles).failure,
              wafer::SharedDDRCompletionFailure::Contract);
  }
}

TEST(ExecutableCompilationPolicyTest,
     UnknownSpatialRootIsUnsupportedAtBothEntries) {
  using namespace wafer::compiler;
  using namespace wafer::compiler::detail;
  for (int64_t extent : {1024, 1025})
    for (bool search : {false, true}) {
      SCOPED_TRACE(std::to_string(extent) + ":" + std::to_string(search));
      auto parsed = wafer::compiler::testing::parseProgram();
      const std::string input =
          "tensor<2x" + std::to_string(extent) + "x128xf16>";
      const std::string output =
          "tensor<2x" + std::to_string(extent) + "x1x128xf16>";
      std::string source;
      llvm::raw_string_ostream os(source);
      os << R"mlir(module {
        wafer.target.topology @target {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
        wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
      )mlir"
         << "func.func @main(%input: " << input << ") -> " << output << " {\n"
         << "%empty = tensor.empty() : " << output << "\n"
         << "%result = tensor.pack %input inner_dims_pos = [2] inner_tiles = "
            "[128] into %empty : "
         << input << " -> " << output << "\n"
         << "return %result : " << output << "\n}}";
      parsed.module = mlir::parseSourceString<mlir::ModuleOp>(
          os.str(), parsed.context.get());
      ASSERT_TRUE(parsed.module);
      wafer::frontend::FrontendProgramVerificationResult program;
      program.numPartitions = 1;
      program.programUserInputCount = 1;
      program.distributedInputs = {
          wafer::compiler::testing::boundary(0, {2, extent, 128})};
      program.distributedOutputs = {
          wafer::compiler::testing::boundary(0, {2, extent, 1, 128})};
      ProgramDataHandoff data;
      std::string diagnosticsText;
      llvm::raw_string_ostream diagnostics(diagnosticsText);
      auto result = search ? compileSearchCurrentIR(
                                 *parsed.module, program,
                                 wafer::compiler::testing::executionConfig(),
                                 diagnostics, data, SearchCurrentIROptions{})
                           : compileBaselineCurrentIR(
                                 *parsed.module, program,
                                 wafer::compiler::testing::executionConfig(),
                                 diagnostics, data, BaselineCurrentIROptions{});
      EXPECT_EQ(result.status, ExecutableCompilationStatus::UnsupportedFailure)
          << result.gate << ": " << result.detail;
      EXPECT_EQ(result.gate,
                search ? "search-planning-problem" : "baseline-spatial");
      EXPECT_FALSE(result.executable);
    }
}

TEST(ExecutableCompilationPolicyTest,
     RelationWorkLimitRemainsIndeterminateAtBothEntries) {
  using namespace wafer::compiler;
  using namespace wafer::compiler::detail;
  // Rank 17 deliberately exceeds the default 32-variable relation budget
  // (17 iteration + 17 operand coordinates); extents remain static and real
  // scale.
  for (bool search : {false, true}) {
    auto parsed = wafer::compiler::testing::parseProgram();
    std::vector<int64_t> shape(17, 1);
    shape[15] = 1025;
    shape[16] = 128;
    std::string type = "tensor<";
    for (int64_t extent : shape)
      type += std::to_string(extent) + "x";
    type += "f16>";
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
      wafer.target.topology @target {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
      wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
    )mlir"
       << "func.func @main(%input: " << type << ") -> " << type << " {\n"
       << "%e = tensor.empty() : " << type
       << "\n%r = linalg.map ins(%input : " << type << ") outs(%e : " << type
       << ") (%x: f16) { %v = arith.addf %x,%x : f16\nlinalg.yield %v : f16 "
          "}\nreturn %r : "
       << type << "\n}}";
    parsed.module =
        mlir::parseSourceString<mlir::ModuleOp>(os.str(), parsed.context.get());
    ASSERT_TRUE(parsed.module);
    wafer::frontend::FrontendProgramVerificationResult program;
    program.numPartitions = 1;
    program.programUserInputCount = 1;
    program.distributedInputs = {wafer::compiler::testing::boundary(0, shape)};
    program.distributedOutputs = {wafer::compiler::testing::boundary(0, shape)};
    ProgramDataHandoff data;
    std::string text;
    llvm::raw_string_ostream diagnostics(text);
    SearchCurrentIROptions options;
    options.planningCredits = 8;
    auto result = search ? compileSearchCurrentIR(
                               *parsed.module, program,
                               wafer::compiler::testing::executionConfig(),
                               diagnostics, data, options)
                         : compileBaselineCurrentIR(
                               *parsed.module, program,
                               wafer::compiler::testing::executionConfig(),
                               diagnostics, data, BaselineCurrentIROptions{});
    EXPECT_EQ(result.status, ExecutableCompilationStatus::IndeterminateFailure)
        << result.gate << ": " << result.detail;
    EXPECT_FALSE(result.executable);
  }
}

} // namespace
