//===- ExecutableCompilationTest.cpp -------------------------------===//

#include "Wafer/CodeGen/Executable/ExecutableCompilation.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Program/ProgramData.h"
#include "Wafer/Target/Core/TargetMemory.h"
#include "Wafer/Transforms/Instr/MemoryPlanning.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

namespace {

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

  std::vector<wafer::compiler::detail::CanonicalInstructionTile>
  makeCanonicalInstructionTiles(mlir::ModuleOp source) {
    std::vector<wafer::compiler::detail::CanonicalInstructionTile> result;
    result.reserve(16);
    for (int64_t tile = 0; tile < 16; ++tile)
      result.push_back({wafer::CardId(0),
                        wafer::TileId(tile),
                        mlir::OwningOpRef<mlir::ModuleOp>(
                            mlir::cast<mlir::ModuleOp>(source->clone())),
                        {}});
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
  relations.scratchBuffers.push_back({0, source});
  relations.scratchBuffers.push_back({0, dest});
  relations.operationEmissions.push_back({0, gather.getOperation()});

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
  ASSERT_EQ(relations.scratchBuffers.size(), 2u);
  EXPECT_EQ(relations.scratchBuffers[0].buffer,
            relations.scratchBuffers[1].buffer);
  EXPECT_TRUE(relations.operationEmissions.empty());
  ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(*module)));
  EXPECT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
      *module, /*spmBase=*/0, /*spmLimit=*/3 * 1024 * 1024,
      /*spmAlignment=*/16)));
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

} // namespace
