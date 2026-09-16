//===- ExecutableCompilationTest.cpp -------------------------------===//

#include "Wafer/Driver/ExecutableCompilation.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Driver/PhysicalDataflow/BaselineCurrentIR.h"
#include "Wafer/Driver/PhysicalDataflow/SearchCurrentIR.h"
#include "Wafer/Driver/ProgramData/ProgramData.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/TargetMemory.h"
#include "Wafer/Transforms/Instr/CommunicationConstruction.h"
#include "Wafer/Transforms/Instr/CommunicationScheduling.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"
#include "Wafer/Transforms/Instr/MemoryPlanning.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

  auto timing =
      std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
  wafer::support::ScopedCompileTimingActivation activation(timing);
  auto result = wafer::compiler::detail::compileBaselineCurrentIR(
      *parsed.module, program, wafer::compiler::testing::executionConfig(),
      diagnostics, programData, options, &baseline, &executable);
  timing->finishAndPrintSummary();
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
  EXPECT_NE(diagnosticText.find("| stage | current-ir-downstream | "
                                "final-direct-dte-completion | 1 |"),
            std::string::npos);
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
     SearchAccountsForFilteredAndEvaluatedReuse) {
  using namespace wafer::compiler::detail;
  for (int64_t extent : {1024, 1025, 1031, 4096}) {
    SCOPED_TRACE(extent);
    int64_t k = extent == 4096 ? 4096 : 256;
    int64_t n = extent == 4096 ? 4096 : 512;
    auto parsed = wafer::compiler::testing::parseProgram();
    ASSERT_TRUE(parsed.module);
    std::string text = llvm::formatv(R"mlir(
module {{
  wafer.target.topology @default {{card_grid = array<i64: 1, 1>,
      card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {{axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<1x{0}x{1}xf16>, %rhs: tensor<1x{1}x{2}xf16>)
      -> tensor<1x{0}x{2}xf16> {{
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<1x{0}x{2}xf16>
    %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<1x{0}x{2}xf16>) -> tensor<1x{0}x{2}xf16>
    %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<1x{0}x{1}xf16>, tensor<1x{1}x{2}xf16>)
        outs(%init : tensor<1x{0}x{2}xf16>) -> tensor<1x{0}x{2}xf16>
    return %result : tensor<1x{0}x{2}xf16>
  }
}
)mlir",
                                     extent, k, n)
                           .str();
    parsed.module =
        mlir::parseSourceString<mlir::ModuleOp>(text, parsed.context.get());
    ASSERT_TRUE(parsed.module) << text;
    wafer::frontend::FrontendProgramVerificationResult program;
    program.numPartitions = 1;
    program.programUserInputCount = 2;
    program.distributedInputs = {
        wafer::compiler::testing::boundary(0, {1, extent, k}),
        wafer::compiler::testing::boundary(1, {1, k, n})};
    program.distributedOutputs = {
        wafer::compiler::testing::boundary(0, {1, extent, n})};
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
    if (extent == 4096) {
      EXPECT_GT(statistics.accessReuseCandidates, 0u);
      EXPECT_GT(statistics.accessReuseAccepted, 0u);
      EXPECT_GT(statistics.accessReuseCapacityRejected, 0u);
      EXPECT_GT(statistics.residentReuseAccepted, 0u);
      ASSERT_TRUE(statistics.minimumResidentDDRReadBytes);
      EXPECT_EQ(*statistics.minimumResidentDDRReadBytes, 64u * 1024 * 1024);
    }
    EXPECT_GT(statistics.accessReuseQueries, 0u);
    EXPECT_GT(statistics.accessReuseEligible, 0u);
    EXPECT_GT(statistics.accessReuseLowBenefit, 0u);
    EXPECT_GE(statistics.accessReuseQueued, statistics.accessReuseCandidates);
    EXPECT_GT(statistics.acceptedCandidates, statistics.accessReuseAccepted);
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
    options.limits = wafer::SearchLimits{1, 2};

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
    EXPECT_EQ(search.traversal.candidateActualizations, 2u);
    EXPECT_GT(search.actualCapacityRefinements, 0u) << diagnosticText;
    EXPECT_GT(search.traversal.resumedCandidates, 0u);
    EXPECT_EQ(search.movementCandidateActualizations, 2u);
    // The certified repair is served before other realizations of the same
    // oversized point. Two actual leaves therefore visit two Temporal points.
    EXPECT_GE(search.temporalCandidateActualizations, 2u);
    EXPECT_EQ(search.peakSessionTemporalPrefixes, 1u);
    EXPECT_EQ(search.temporalBackpressureTurns, 0u);
    EXPECT_LE(search.movementCandidateActualizations, 2u);
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
     LargeContractionKeepsWideStateInsideItsSelectedRegion) {
  for (auto [extent, dtype] : {std::pair<int64_t, const char *>{4096, "f16"},
                               {4097, "f16"},
                               {4096, "bf16"}}) {
    SCOPED_TRACE(extent);
    SCOPED_TRACE(dtype);
    auto parsed = wafer::compiler::testing::parseProgram();
    ASSERT_TRUE(parsed.module);
    std::string tensor = "tensor<1x" + std::to_string(extent) + "x" +
                         std::to_string(extent) + "x" + dtype + ">";
    std::string source =
        "module { wafer.target.topology @default "
        "{card_grid = array<i64: 1, 1>, card_interconnect = \"mesh\", "
        "tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>} "
        "wafer.execution.mesh @default_mesh "
        "{axes = [\"card\"], shape = array<i64: 1>} "
        "func.func @main(%a: " +
        tensor + ", %b: " + tensor + ") -> " + tensor +
        " { %zero = arith.constant 0.0 : " + dtype +
        " %empty = tensor.empty() : " + tensor +
        " %init = linalg.fill ins(%zero : " + dtype +
        ") outs(%empty : " + tensor + ") -> " + tensor +
        " %r = linalg.batch_matmul ins(%a, %b : " + tensor + ", " + tensor +
        ") outs(%init : " + tensor + ") -> " + tensor +
        " return %r : " + tensor + " } }";
    parsed.module =
        mlir::parseSourceString<mlir::ModuleOp>(source, parsed.context.get());
    ASSERT_TRUE(parsed.module);
    wafer::frontend::FrontendProgramVerificationResult program;
    program.numPartitions = 1;
    program.programUserInputCount = 2;
    program.distributedInputs = {
        wafer::compiler::testing::boundary(0, {1, extent, extent}),
        wafer::compiler::testing::boundary(1, {1, extent, extent})};
    program.distributedOutputs = {
        wafer::compiler::testing::boundary(0, {1, extent, extent})};
    for (auto &input : program.distributedInputs)
      input.dtype = std::string(dtype) == "bf16"
                        ? wafer::ProgramElementType::BF16
                        : wafer::ProgramElementType::F16;
    program.distributedOutputs[0].dtype = program.distributedInputs[0].dtype;
    wafer::compiler::ProgramDataHandoff data;
    std::string diagnosticText;
    llvm::raw_string_ostream diagnostics(diagnosticText);
    wafer::compiler::detail::SearchCurrentIROptions options;
    options.termination =
        wafer::compiler::detail::SearchTerminationPolicy::FirstAccepted;
    wafer::compiler::detail::SearchCurrentIRStatistics search;
    auto result = wafer::compiler::detail::compileSearchCurrentIR(
        *parsed.module, program, wafer::compiler::testing::executionConfig(),
        diagnostics, data, options, &search);
    ASSERT_TRUE(result.isAccepted()) << result.detail << diagnosticText;
    ASSERT_EQ(result.executable->tiles.size(), 16u);
    EXPECT_GT(search.actualCapacityRefinements, 0u);
    // The original logical contraction still returns the original dtype.
    unsigned sourceContractions = 0, sourceConversions = 0;
    parsed.module->walk([&](mlir::linalg::LinalgOp op) {
      if (mlir::linalg::isaContractionOpInterface(op)) {
        ++sourceContractions;
        EXPECT_FALSE(
            mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
                .getElementType()
                .isF32());
      }
    });
    parsed.module->walk([&](mlir::arith::TruncFOp) { ++sourceConversions; });
    EXPECT_EQ(sourceContractions, 1u);
    EXPECT_EQ(sourceConversions, 0u);
    uint64_t multiplyAdds = 0;
    unsigned widePartials = 0, narrowOutputs = 0, conversions = 0;
    for (const auto &tile : result.executable->tiles) {
      tile.getModule().walk([&](wafer::InstrConvertOp) { ++conversions; });
      tile.getModule().walk([&](wafer::InstrGemmOp gemm) {
        uint64_t work = gemm.getBatchCount().value_or(1) * gemm.getM() *
                        gemm.getN() * gemm.getK();
        for (auto *parent = gemm->getParentOp(); parent;
             parent = parent->getParentOp()) {
          auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent);
          if (!loop)
            continue;
          auto lower = mlir::getConstantIntValue(loop.getLowerBound());
          auto upper = mlir::getConstantIntValue(loop.getUpperBound());
          auto step = mlir::getConstantIntValue(loop.getStep());
          ASSERT_TRUE(lower && upper && step && *step > 0);
          work *= (*upper - *lower + *step - 1) / *step;
        }
        multiplyAdds += work;
        if (gemm.getPsum()) {
          ++widePartials;
          EXPECT_TRUE(mlir::cast<mlir::MemRefType>(gemm.getPsum().getType())
                          .getElementType()
                          .isF32());
        }
        auto type = mlir::cast<mlir::MemRefType>(gemm.getDest().getType());
        narrowOutputs += !type.getElementType().isF32();
      });
      EXPECT_TRUE(mlir::succeeded(mlir::verify(tile.getModule())));
    }
    EXPECT_EQ(multiplyAdds, uint64_t(extent) * extent * extent);
    EXPECT_GT(widePartials, 0u);
    EXPECT_GT(narrowOutputs, 0u);
    EXPECT_EQ(conversions, 0u);
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
  uint64_t capacityCallbacks = 0;
  auto observeCapacity =
      [&](wafer::CardId, wafer::TileId,
          const wafer::SPMMemoryPlanningFailure &failure,
          const wafer::StructuredMaterializationRelations &) {
        EXPECT_EQ(failure.kind,
                  wafer::SPMMemoryPlanningFailureKind::CapacityOverflow);
        EXPECT_GT(failure.demandCount, 0u);
        ++capacityCallbacks;
      };
  options.downstream.capacityObserver = observeCapacity;
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
  EXPECT_GT(capacityCallbacks, 0u);
  EXPECT_GT(search.actualCapacityRefinements, 0u);
  EXPECT_LE(search.actualCapacityRefinements, capacityCallbacks);
  EXPECT_GT(search.temporalCandidateActualizations, 1u);
  // A retained realization may evaluate another layout placement at the same
  // Temporal point; every actual candidate still reaches the common gate.
  EXPECT_GE(search.layoutInvocations, search.temporalCandidateActualizations);
  EXPECT_EQ(executable.actualMemoryTargetGateInvocations,
            search.exactRejectedCandidates + search.acceptedCandidates);
  EXPECT_EQ(search.movementCandidateActualizations,
            executable.actualMemoryTargetGateInvocations);
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
    // All active Tiles fuse the producer/consumer boundary. The search may
    // choose fewer than sixteen partitions; unused target Tiles stay empty.
    ASSERT_EQ(result.physicalIRInventory->tiles.size(), 16u);
    uint64_t activeTiles = 0;
    for (const auto &tile : result.physicalIRInventory->tiles) {
      EXPECT_LE(tile.regions, 1u);
      activeTiles += tile.regions != 0;
    }
    EXPECT_GT(activeTiles, 1u);
    EXPECT_EQ(result.physicalIRInventory->tileRegions, activeTiles);
  }
}

TEST(ExecutableCompilationPolicyTest,
     TemporalBackpressureResumesActualPrefixesWithoutChargingYields) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto parsed =
        wafer::compiler::testing::parseRealScaleDependentProgram(extent);
    ASSERT_TRUE(parsed.module);
    auto program =
        wafer::compiler::testing::realScaleDependentProgramMetadata(extent);
    wafer::compiler::ProgramDataHandoff data;
    std::string text;
    llvm::raw_string_ostream diagnostics(text);
    wafer::compiler::detail::SearchCurrentIROptions options;
    options.limits = wafer::SearchLimits{2, 8};
    options.downstream.tilePipelineParallelism = 1;
    wafer::compiler::detail::SearchCurrentIRStatistics search;
    wafer::compiler::detail::ExecutableLoweringStatistics executable;
    auto timing =
        std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
    wafer::support::ScopedCompileTimingActivation activation(timing);
    auto result = wafer::compiler::detail::compileSearchCurrentIR(
        *parsed.module, program, wafer::compiler::testing::executionConfig(),
        diagnostics, data, options, &search, &executable);
    timing->finishAndPrintSummary();
    ASSERT_TRUE(result.isAccepted()) << result.detail << text;
    EXPECT_EQ(text.find("| stage | current-ir-downstream | "
                        "final-direct-dte-completion |"),
              std::string::npos);
    EXPECT_NE(text.find("| completion | direct-dte | rebuild-waits | 16 |"),
              std::string::npos);
    ASSERT_TRUE(result.executable && result.physicalIRInventory);
    EXPECT_EQ(search.peakSessionTemporalPrefixes, 1u);
    // Accepted prefixes are retained in the realization slot while another
    // Temporal point improves. This small budget need not reach the third
    // Spatial direction that introduces a SharedDDR exchange.
    EXPECT_GT(search.temporalCandidateActualizations, 1u);
    EXPECT_LE(search.peakSessionIRModules, 10u);
    EXPECT_GT(search.traversal.stageYields, 0u);
    EXPECT_EQ(search.traversal.candidateActualizations, 8u);
    EXPECT_EQ(search.controller.accepted, 8u) << text;
    EXPECT_EQ(search.controller.unsupported, 0u) << text;
    EXPECT_EQ(search.controller.indeterminate, 0u) << text;
    EXPECT_EQ(executable.actualMemoryTargetGateInvocations, 8u) << text;
    EXPECT_EQ(result.executable->tiles.size(), 16u);
    EXPECT_EQ(result.physicalIRInventory->tileModules, 16u);
    EXPECT_EQ(
        search.coverage,
        wafer::compiler::detail::SearchControllerCoverage::FeasiblePartial);
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
     CommunicationConstructionDoesNotEraseForeignCompletionTokens) {
  auto parsed = wafer::compiler::testing::parseProgram();
  // Zero-rank, no payload: bounded verifier-valid token-contract negative.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @entry() {
    wafer.tile.region() -> () {
      %token = async.execute { async.yield }
      wafer.instr.dte_wait %token : !async.token
      wafer.tile.yield
    }
    return
  }
}
)mlir",
                                                        parsed.context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  auto result = wafer::compiler::detail::constructCommunication(
      {*module}, {wafer::TileId(0)});
  EXPECT_EQ(result.outcome.failure,
            wafer::SharedDDRCompletionFailure::Contract);
  unsigned waits = 0;
  module->walk([&](wafer::InstrDTEWaitOp) { ++waits; });
  EXPECT_EQ(waits, 1u);
}

TEST(ExecutableCompilationPolicyTest,
     ConstructsReadyRingsAndPreservesRealDataCycles) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  for (unsigned count : {2u, 4u, 16u})
    for (int64_t extent : {1024, 1025, 1031})
      for (unsigned mode : {0u, 1u, 2u, 3u, 4u})
        for (unsigned repeated : {0u, 1u, 2u}) {
          SCOPED_TRACE(::testing::Message() << count << '/' << extent << '/'
                                            << mode << '/' << repeated);
          auto parsed = wafer::compiler::testing::parseProgram();
          llvm::SmallVector<StandaloneTileModule, 16> owners;
          llvm::SmallVector<mlir::ModuleOp> modules;
          llvm::SmallVector<TileId> ids;
          std::string type =
              std::string(mode >= 3 ? "memref<2x" : "memref<1x") +
              std::to_string(extent) + "x64xf16, #wafer.memory<spm, tensor>>";
          const int64_t bytes = extent * 128;
          for (unsigned tile = 0; tile < count; ++tile) {
            unsigned previous = (tile + count - 1) % count;
            unsigned next = (tile + 1) % count;
            std::string text;
            llvm::raw_string_ostream ir(text);
            ir << "module { func.func @entry() { wafer.tile.region() -> () {\n"
               << "%source = memref.alloc() : " << type << '\n'
               << "%dest = memref.alloc() : " << type << '\n'
               << "%other = memref.alloc() : " << type << '\n'
               << "%one = arith.constant 1.0 : f16\n"
               << "wafer.instr.fill %source, %one : " << type << ", f16\n";
            auto send = [&](unsigned round) {
              ir << "%s" << round << " = wafer.instr.dte_send %"
                 << (mode == 2 ? "dest" : "source") << " {peer = " << next
                 << " : i64, bytes = " << bytes
                 << " : i64, message = #wafer.dte_message<communication = "
                 << tile << ", round = " << round << ", slice = 0>} : " << type
                 << " -> !async.token\nwafer.instr.dte_wait %s" << round
                 << " : !async.token\n";
            };
            auto recv = [&](unsigned round) {
              ir << "%r" << round << " = wafer.instr.dte_recv %"
                 << (mode >= 3 ? "source"
                     : round   ? "other"
                               : "dest")
                 << " {peer = " << previous
                 << " : i64, buffer_offset = " << (mode == 3 ? bytes : 0)
                 << " : i64, bytes = " << bytes
                 << " : i64, message = #wafer.dte_message<communication = "
                 << previous << ", round = " << round
                 << ", slice = 0>} : " << type
                 << " -> !async.token\nwafer.instr.dte_wait %r" << round
                 << " : !async.token\n";
            };
            if (mode >= 2) {
              // Mode 3 uses the other half of an initialized allocation and
              // must remain legal. Modes 2/4 forward the not-yet-received range
              // and must not be admitted merely because a root was initialized.
              recv(0);
              send(0);
            } else if (mode == 1 && tile % 2) {
              send(1);
              recv(1);
              send(0);
              recv(0);
            } else {
              send(0);
              recv(0);
              if (mode == 1) {
                send(1);
                recv(1);
              }
            }
            ir << "wafer.tile.yield\n}\nreturn\n}\n}\n";
            auto module = mlir::parseSourceString<mlir::ModuleOp>(
                text, parsed.context.get());
            ASSERT_TRUE(module) << text;
            if (repeated) {
              module->walk([&](TileRegionOp region) {
                auto &body = region.getBody().front();
                llvm::SmallVector<mlir::Operation *> original;
                for (auto &operation : body.without_terminator())
                  original.push_back(&operation);
                mlir::OpBuilder builder(&body, body.begin());
                auto lower = builder.create<mlir::arith::ConstantIndexOp>(
                    region.getLoc(), 0);
                auto upper = builder.create<mlir::arith::ConstantIndexOp>(
                    region.getLoc(),
                    repeated == 2 && tile == count - 1 ? 1024 : 1025);
                auto step = builder.create<mlir::arith::ConstantIndexOp>(
                    region.getLoc(), 1);
                auto loop = builder.create<mlir::scf::ForOp>(
                    region.getLoc(), lower, upper, step);
                for (auto *operation : original)
                  operation->moveBefore(loop.getBody()->getTerminator());
              });
            }
            ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
            modules.push_back(*module);
            ids.push_back(TileId(tile));
            owners.push_back({CardId(0), TileId(tile), std::move(module), {}});
          }
          auto printModules = [&]() {
            std::string text;
            llvm::raw_string_ostream out(text);
            for (auto module : modules)
              module.print(out);
            return text;
          };
          std::string before = printModules();
          auto limited = scheduleCurrentCommunication(modules, ids, 0);
          EXPECT_EQ(limited.status,
                    repeated == 2 ? CommunicationSchedulingStatus::Unsupported
                                  : CommunicationSchedulingStatus::WorkLimit);
          EXPECT_EQ(printModules(), before);
          auto result = constructCommunication(modules, ids);
          if (repeated == 2 || mode == 2 || mode == 4) {
            EXPECT_EQ(result.outcome.failure,
                      SharedDDRCompletionFailure::Unsupported);
            EXPECT_EQ(printModules(), before);
            continue;
          }
          ASSERT_TRUE(result.outcome.succeeded()) << result.outcome.detail;
          EXPECT_EQ(result.statistics.replacedMessages, 0u);
          EXPECT_EQ(analyzeCurrentCommunicationOrder(modules, ids).status,
                    CommunicationOrderStatus::Acyclic);
          uint64_t sends = 0, receives = 0, waits = 0;
          for (auto module : modules) {
            module.walk([&](InstrDTESendOp send) {
              ++sends;
              EXPECT_EQ(send.getBytes(), static_cast<uint64_t>(bytes));
            });
            module.walk([&](InstrDTERecvOp recv) {
              ++receives;
              EXPECT_EQ(recv.getBytes(), static_cast<uint64_t>(bytes));
            });
            module.walk(
                [&](InstrDTEWaitOp wait) { waits += wait.getTokens().size(); });
          }
          EXPECT_EQ(sends, count * (mode == 1 ? 2u : 1u));
          EXPECT_EQ(receives, sends);
          EXPECT_EQ(waits, 2 * sends);
          for (auto &owner : owners) {
            ASSERT_TRUE(
                mlir::succeeded(rebuildRequiredNCCJoins(*owner.module)));
            TileMemoryPlanningFailure failure;
            auto planned = planTileMemory(std::move(owner.module), &failure);
            ASSERT_TRUE(mlir::succeeded(planned));
            owner.module = std::move(*planned);
          }
          ASSERT_TRUE(mlir::succeeded(bindDirectDTETransport(modules)));
        }
}

TEST(ExecutableCompilationPolicyTest,
     ConstructsFaninWithoutOverbookingReceiverFSMs) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  for (int64_t extent : {1024, 1025, 1031}) {
    auto parsed = wafer::compiler::testing::parseProgram();
    llvm::SmallVector<StandaloneTileModule, 16> owners;
    llvm::SmallVector<mlir::ModuleOp> modules;
    llvm::SmallVector<TileId> ids;
    std::string type = "memref<1x" + std::to_string(extent) +
                       "x64xf16, #wafer.memory<spm, tensor>>";
    for (unsigned tile = 0; tile < 16; ++tile) {
      std::string text;
      llvm::raw_string_ostream ir(text);
      ir << "module { func.func @entry() {\n";
      if (tile < 6) {
        ir << "wafer.tile.region() -> () {\n";
        const unsigned first = tile ? tile : 1, last = tile ? tile : 5;
        for (unsigned source = first; source <= last; ++source) {
          ir << "%b" << source << " = memref.alloc() : " << type << '\n';
          if (tile)
            ir << "%one = arith.constant 1.0 : f16\nwafer.instr.fill %b"
               << source << ", %one : " << type << ", f16\n";
          ir << "%t" << source << " = wafer.instr.dte_"
             << (tile ? "send" : "recv") << " %b" << source
             << " {peer = " << (tile ? 0 : source)
             << " : i64, bytes = " << extent * 128
             << " : i64, message = #wafer.dte_message<communication = "
             << source << ", round = 0, slice = 0>} : " << type
             << " -> !async.token\nwafer.instr.dte_wait %t" << source
             << " : !async.token\n";
        }
        ir << "wafer.tile.yield\n}\n";
      }
      ir << "return\n}\n}\n";
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(text, parsed.context.get());
      ASSERT_TRUE(module) << text;
      modules.push_back(*module);
      ids.push_back(TileId(tile));
      owners.push_back({CardId(0), TileId(tile), std::move(module), {}});
    }
    auto result = constructCommunication(modules, ids);
    ASSERT_TRUE(result.outcome.succeeded()) << result.outcome.detail;
    EXPECT_EQ(result.statistics.replacedMessages, 0u);
    unsigned live = 0, peak = 0, receives = 0;
    modules.front().walk([&](mlir::Operation *op) {
      if (mlir::isa<InstrDTERecvOp>(op)) {
        ++receives;
        peak = std::max(peak, ++live);
      } else if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(op)) {
        for (auto token : wait.getTokens())
          if (token.getDefiningOp<InstrDTERecvOp>()) {
            ASSERT_GT(live, 0u);
            --live;
          }
      }
    });
    EXPECT_EQ(receives, 5u);
    EXPECT_EQ(peak, 4u);
    EXPECT_EQ(live, 0u);
    for (auto &owner : owners) {
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*owner.module)));
      TileMemoryPlanningFailure failure;
      auto planned = planTileMemory(std::move(owner.module), &failure);
      ASSERT_TRUE(mlir::succeeded(planned));
      owner.module = std::move(*planned);
    }
    ASSERT_TRUE(mlir::succeeded(bindDirectDTETransport(modules)));
    EXPECT_EQ(analyzeCurrentCommunicationOrder(modules, ids).status,
              CommunicationOrderStatus::Acyclic);
  }
}

TEST(ExecutableCompilationPolicyTest,
     CommunicationProposalUsesCurrentOrderAndPreservesValidSharing) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  for (int64_t count : {2, 4, 16})
    for (int64_t extent : {1024, 1025, 1031})
      for (unsigned mode : {0u, 1u, 2u, 3u, 4u, 5u}) {
        if (mode >= 4 && count == 2)
          continue; // Native fanout needs two receivers plus its sender.
        SCOPED_TRACE(::testing::Message()
                     << count << "/" << extent << "/" << mode);
        auto parsed = wafer::compiler::testing::parseProgram();
        llvm::SmallVector<StandaloneTileModule, 16> tiles;
        llvm::SmallVector<mlir::ModuleOp> modules;
        llvm::SmallVector<TileId> ids;
        const std::string shape = "1x" + std::to_string(extent) + "x64xf16";
        const std::string ddr =
            "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
        const std::string spm =
            "memref<" + shape + ", #wafer.memory<spm, tensor>>";
        const int64_t bytes = 128 * extent;
        for (int64_t tile = 0; tile < count; ++tile) {
          std::string text;
          llvm::raw_string_ostream ir(text);
          ir << "module {\n";
          if (tile != 0 && tile != count - 1 && !(mode >= 4 && tile == 1)) {
            ir << "func.func @entry() { return }\n}\n";
          } else {
            const bool sender = tile == count - 1;
            ir << "memref.global \"private\" @data : " << ddr
               << " {wafer.ddr_resource = #wafer.ddr_resource<0>}\n"
               << "func.func @entry(%input: " << ddr
               << " {wafer.program_argument = #wafer.program_argument<0>}, "
                  "%data: "
               << ddr
               << " {wafer.ddr_binding = #wafer.ddr_binding<@data, id = 0, "
               << (sender ? "write" : "read") << ">}, %out: " << ddr << ") {\n"
               << "wafer.tile.region(%input, %data, %out : " << ddr << ", "
               << ddr << ", " << ddr << ") -> () { ^bb0(%i: " << ddr
               << ", %d: " << ddr << ", %o: " << ddr << "):\n"
               << "%b = memref.alloc() : " << spm << "\n";
            auto dma = [&](bool read, llvm::StringRef from,
                           llvm::StringRef to) {
              ir << "wafer.instr." << (read ? "rdma" : "wdma") << " " << from
                 << " to " << to << " {byte_count = " << bytes
                 << " : i64, inner_bytes = " << bytes << " : i64, "
                 << (read ? "src" : "dst") << "_strides = array<i64: 0, 0, 0>, "
                 << (read ? "src" : "dst")
                 << "_iterations = array<i64: 1, 1, 1>} : "
                 << (read ? ddr : spm) << " to " << (read ? spm : ddr) << "\n";
            };
            auto transport = [&]() {
              if (sender && mode >= 4) {
                ir << "%token = wafer.instr.dte_"
                   << (mode == 4 ? "broadcast" : "scatter")
                   << " %b {source_offset = 128 : i64, peers = array<i64: 0, "
                      "1>, bytes = 256 : i64, "
                   << "messages = [#wafer.dte_message<communication = 7, round "
                      "= 0, slice = 0>, "
                   << "#wafer.dte_message<communication = 7, round = 0, slice "
                      "= 1>]} : "
                   << spm
                   << " -> !async.token\nwafer.instr.dte_wait %token : "
                      "!async.token\n";
                return;
              }
              ir << "%token = wafer.instr.dte_" << (sender ? "send" : "recv")
                 << " %b {peer = " << (sender ? 0 : count - 1)
                 << " : i64, bytes = " << (mode >= 4 ? 256 : bytes)
                 << " : i64, message = #wafer.dte_message<communication = 7, "
                    "round = 0, slice = "
                 << (mode >= 4 ? tile : 0) << ">} : " << spm
                 << " -> !async.token\nwafer.instr.dte_wait %token : "
                    "!async.token\n";
            };
            if (sender) {
              if (mode == 2)
                dma(true, "%i", "%b");
              else
                ir << "%one = arith.constant 1.0 : f16\nwafer.instr.fill %b, "
                      "%one : "
                   << spm << ", f16\n";
              if (mode != 0)
                transport();
              if (mode >= 2) {
                ir << "wafer.tile.yield\n}\nwafer.tile.region(%data : " << ddr
                   << ") -> () { ^bb0(%d: " << ddr << "):\n"
                   << "%b = memref.alloc() : " << spm
                   << "\n%one = arith.constant 1.0 : f16\nwafer.instr.fill %b, "
                      "%one : "
                   << spm << ", f16\n";
              }
              dma(false, "%b", "%d");
              if (mode == 0)
                transport();
            } else {
              dma(true, "%d", "%b");
              transport();
              dma(false, "%b", "%o");
            }
            ir << "wafer.tile.yield\n}\n";
            if (tile == 0 || sender) {
              // An independent repeated exchange must not disable the static
              // DDR representation repair required by this same transaction.
              ir << "wafer.tile.region() -> () { "
                 << "%lo = arith.constant 0 : index "
                 << "%hi = arith.constant " << extent << " : index "
                 << "%step = arith.constant 1 : index "
                 << "%value = arith.constant 1.0 : f16 "
                 << "scf.for %i = %lo to %hi step %step { "
                 << "%panel = memref.alloc() : " << spm << " ";
              if (sender)
                ir << "wafer.instr.fill %panel, %value : " << spm << ", f16 ";
              ir << "%event = wafer.instr.dte_" << (sender ? "send" : "recv")
                 << " %panel {peer = " << (sender ? 0 : count - 1)
                 << " : i64, bytes = 256 : i64, message = "
                 << "#wafer.dte_message<communication = 99, round = 0, slice = "
                    "0>} : "
                 << spm
                 << " -> !async.token wafer.instr.dte_wait %event : "
                    "!async.token "
                 << "} wafer.tile.yield }\n";
            }
            ir << "return\n}\n}\n";
          }
          auto module = mlir::parseSourceString<mlir::ModuleOp>(
              text, parsed.context.get());
          ASSERT_TRUE(module) << text;
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          modules.push_back(*module);
          ids.push_back(TileId(tile));
          tiles.push_back({CardId(0), TileId(tile), std::move(module), {}});
        }
        ASSERT_TRUE(rebuildRequiredDirectDTEWaits(modules).succeeded());
        ASSERT_TRUE(materializeSharedDDRNotifications(modules).succeeded());
        auto before = analyzeCurrentCommunicationOrder(modules, ids);
        EXPECT_EQ(before.status, mode ? CommunicationOrderStatus::Cycle
                                      : CommunicationOrderStatus::Acyclic);
        std::string original;
        llvm::raw_string_ostream printed(original);
        for (auto module : modules)
          module.print(printed);
        auto proposal = constructCommunication(modules, ids);
        ASSERT_TRUE(proposal.outcome.succeeded()) << proposal.outcome.detail;
        EXPECT_EQ(proposal.statistics.replacedMessages, mode >= 4   ? 2u
                                                        : mode >= 2 ? 1u
                                                                    : 0u);
        EXPECT_EQ(proposal.statistics.localInputReads, mode == 2 ? 1u : 0u);
        EXPECT_EQ(proposal.statistics.ddrPackets, mode >= 3 ? 1u : 0u);
        if (mode == 1) {
          EXPECT_GT(proposal.statistics.issuePlacements, 0u);
        }
        EXPECT_EQ(analyzeCurrentCommunicationOrder(modules, ids).status,
                  CommunicationOrderStatus::Acyclic);
        unsigned packetWrites = 0, packetReads = 0;
        for (auto [tile, module] : llvm::enumerate(modules))
          for (auto entry : module.getOps<mlir::func::FuncOp>())
            for (auto argument : entry.getArguments()) {
              auto binding = entry.getArgAttrOfType<DDRBindingAttr>(
                  argument.getArgNumber(), kWaferDDRBindingAttrName);
              auto type = mlir::cast<mlir::MemRefType>(argument.getType());
              if (!binding || binding.getResourceId() == 0 ||
                  !type.getElementType().isF16())
                continue;
              EXPECT_GE(mode, 3u);
              EXPECT_EQ(computeWaferPhysicalTensorInfo(type)->physicalBytes,
                        mode >= 4 ? (mode == 5 ? 512 : 256) : bytes);
              entry.walk([&](TileRegionOp region) {
                for (auto [index, input] : llvm::enumerate(region.getInputs())) {
                  if (input != argument)
                    continue;
                  auto local = region.getBody().front().getArgument(index);
                  for (auto *user : local.getUsers()) {
                    if (auto write = mlir::dyn_cast<InstrWDMAOp>(user)) {
                      ++packetWrites;
                      EXPECT_EQ(tile, static_cast<size_t>(count - 1));
                      EXPECT_EQ(write.getByteCount(),
                                mode >= 4 ? (mode == 5 ? 512u : 256u)
                                          : static_cast<uint64_t>(bytes));
                      EXPECT_EQ(write.getSrcOffset().value_or(0),
                                mode >= 4 ? 128 : 0);
                      EXPECT_EQ(write.getDstOffset().value_or(0), 0);
                    } else if (auto read = mlir::dyn_cast<InstrRDMAOp>(user)) {
                      ++packetReads;
                      EXPECT_EQ(read.getByteCount(),
                                mode >= 4 ? 256u
                                          : static_cast<uint64_t>(bytes));
                      EXPECT_EQ(read.getSrcOffset().value_or(0),
                                mode == 5 ? static_cast<int64_t>(tile) * 256
                                          : 0);
                      EXPECT_EQ(read.getDstOffset().value_or(0), 0);
                    }
                  }
                }
              });
            }
        EXPECT_EQ(packetWrites, mode >= 3 ? 1u : 0u);
        EXPECT_EQ(packetReads, mode >= 4 ? 2u : mode == 3 ? 1u : 0u);
        if (!mode) {
          std::string after;
          llvm::raw_string_ostream output(after);
          for (auto module : modules)
            module.print(output);
          EXPECT_EQ(after, original);
        }
        for (auto &tile : tiles) {
          ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
          TileMemoryPlanningFailure failure;
          auto planned = planTileMemory(std::move(tile.module), &failure);
          ASSERT_TRUE(mlir::succeeded(planned));
          tile.module = std::move(*planned);
        }
      }
}

TEST(ExecutableCompilationPolicyTest,
     CommunicationIssuePlacementSummarizesEachCurrentOperationOnce) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  constexpr unsigned messages = 32;
  constexpr unsigned nestedWrites = 128;
  auto parsed = wafer::compiler::testing::parseProgram();
  llvm::SmallVector<StandaloneTileModule, 2> tiles;
  llvm::SmallVector<mlir::ModuleOp, 2> modules;
  llvm::SmallVector<TileId, 2> ids{TileId(0), TileId(1)};
  for (unsigned tile = 0; tile < 2; ++tile) {
    std::string text;
    llvm::raw_string_ostream ir(text);
    llvm::StringRef ddr = "memref<1x1025x64xf16, #wafer.memory<ddr, tensor>>";
    llvm::StringRef spm = "memref<1x1025x64xf16, #wafer.memory<spm, tensor>>";
    ir << "module { memref.global \"private\" @data : " << ddr
       << " {wafer.ddr_resource = #wafer.ddr_resource<0>}\n"
       << "func.func @entry(%data: " << ddr
       << " {wafer.ddr_binding = #wafer.ddr_binding<@data, id = 0, "
       << (tile ? "write" : "read") << ">}) {\n"
       << "wafer.tile.region(%data : " << ddr << ") -> () { ^bb0(%d: " << ddr
       << "):\n"
       << "%b = memref.alloc() : " << spm << "\n"
       << "%one = arith.constant 1.0 : f16\n";
    auto dma = [&]() {
      ir << "wafer.instr." << (tile ? "wdma %b to %d" : "rdma %d to %b")
         << " {byte_count = 131200 : i64, inner_bytes = 131200 : i64, "
         << (tile ? "dst" : "src") << "_strides = array<i64: 0,0,0>, "
         << (tile ? "dst" : "src")
         << "_iterations = array<i64: 1,1,1>} : " << (tile ? spm : ddr)
         << " to " << (tile ? ddr : spm) << '\n';
    };
    if (tile)
      ir << "wafer.instr.fill %b, %one : " << spm << ", f16\n";
    else
      dma();
    for (unsigned message = 0; message < messages; ++message)
      ir << "%t" << message << " = wafer.instr.dte_" << (tile ? "send" : "recv")
         << " %b {peer = " << (1 - tile)
         << " : i64, bytes = 128 : i64, message = "
            "#wafer.dte_message<communication = "
         << message << ", round = 0, slice = 0>} : " << spm
         << " -> !async.token\nwafer.instr.dte_wait %t" << message
         << " : !async.token\n";
    if (tile) {
      // The unrelated loop is a bounded oracle for repeated recursive effect
      // queries. Its actual writes must not move any source-buffer epoch.
      ir << "%other = memref.alloc() : " << spm << '\n'
         << "%lo = arith.constant 0 : index\n"
         << "%hi = arith.constant 1025 : index\n"
         << "%step = arith.constant 1 : index\n"
         << "scf.for %i = %lo to %hi step %step {\n";
      for (unsigned index = 0; index < nestedWrites; ++index)
        ir << "wafer.instr.fill %other, %one : " << spm << ", f16\n";
      ir << "}\n";
      dma();
    }
    ir << "wafer.tile.yield\n}\nreturn\n}\n}\n";
    auto module =
        mlir::parseSourceString<mlir::ModuleOp>(text, parsed.context.get());
    ASSERT_TRUE(module) << text;
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    modules.push_back(*module);
    tiles.push_back({CardId(0), TileId(tile), std::move(module), {}});
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  auto timing = std::make_shared<support::CompileTimingSession>(stream);
  support::ScopedCompileTimingActivation activation(timing);
  auto result = constructCommunication(modules, ids);
  ASSERT_TRUE(result.outcome.succeeded()) << result.outcome.detail;
  EXPECT_GE(result.statistics.issuePlacements, messages);
  EXPECT_EQ(result.statistics.replacedMessages, 0u);
  EXPECT_EQ(analyzeCurrentCommunicationOrder(modules, ids).status,
            CommunicationOrderStatus::Acyclic);
  timing->finishAndPrintSummary();
  llvm::StringRef marker = "name=effect-summaries value=";
  size_t offset = text.find(marker.str());
  ASSERT_NE(offset, std::string::npos);
  llvm::StringRef number = llvm::StringRef(text)
                               .drop_front(offset + marker.size())
                               .take_until([](char c) { return c == ' '; });
  uint64_t summaries = 0;
  ASSERT_FALSE(number.getAsInteger(10, summaries));
  // Each top-level op contributes at most one summary, independent of the
  // number of messages and the nested body length.
  EXPECT_LE(summaries, 2 * messages + 24);
  EXPECT_GT(summaries, 0u);
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
                "kind=ddr-publication", "tile=0", "tile=1", "shared-resource=",
                "writer-last-access-cut=", "reader-first-access-cut="})
            EXPECT_NE(completion.detail.find(evidence.str()), std::string::npos)
                << completion.detail;
          llvm::SmallVector<wafer::StandaloneTileModule, 2> candidate;
          for (auto [index, owner] : llvm::enumerate(owners))
            candidate.push_back(
                {wafer::CardId(0), tiles[index], std::move(owner), {}});
          auto proposed =
              wafer::compiler::detail::constructCommunication(modules, tiles);
          EXPECT_EQ(proposed.outcome.failure,
                    wafer::SharedDDRCompletionFailure::Unsupported);
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
      bool firstArgument = true;
      for (unsigned id = 0; id < resources; ++id) {
        unsigned relative = (tile + 4 - id % 4) % 4;
        if (relative == 3)
          continue;
        if (!firstArgument)
          ir << ", ";
        firstArgument = false;
        ir << "%data" << id << ": " << ddr << " {wafer.ddr_binding = "
           << "#wafer.ddr_binding<@data" << id << ", id = " << id << ", "
           << (relative == 0 ? "write" : "read") << ">}";
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
      auto entry = *module.getOps<mlir::func::FuncOp>().begin();
      constexpr unsigned localResources = resources * 3 / 4;
      ASSERT_EQ(entry.getNumArguments(), 2 * localResources);
      for (unsigned index = 0; index < localResources; ++index) {
        auto data = entry.getArgAttrOfType<wafer::DDRBindingAttr>(
            index, wafer::kWaferDDRBindingAttrName);
        auto ready = entry.getArgAttrOfType<wafer::DDRBindingAttr>(
            localResources + index, wafer::kWaferDDRBindingAttrName);
        ASSERT_TRUE(data && ready);
        EXPECT_EQ(ready.getResourceId(), resources + data.getResourceId());
        EXPECT_EQ(ready.getAccess(), data.getAccess());
        EXPECT_NE(ready.getAccess(), wafer::DDRAccess::None);
        EXPECT_FALSE(entry.getArgument(localResources + index).use_empty());
        EXPECT_EQ(entry.getArgument(localResources + index).getType(),
                  mlir::MemRefType::get(
                      {64}, mlir::IntegerType::get(parsed.context.get(), 8),
                      mlir::MemRefLayoutAttrInterface{},
                      wafer::MemoryAttr::get(parsed.context.get(),
                                             wafer::MemorySpace::DDR,
                                             wafer::MemLayout::Tensor)));
      }
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
