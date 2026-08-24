//===- CardExecutableCompilationTest.cpp -------------------------------===//

#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Program/ProgramData.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

class CardExecutableCompilationTest : public ::testing::Test {
protected:
  CardExecutableCompilationTest() {
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

  mlir::OwningOpRef<mlir::ModuleOp> oversizedCardModule() {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  wafer.card.module card_id = 0 {
)mlir";
    for (int64_t tile = 0; tile < 16; ++tile) {
      os << "    wafer.tile.module tile_id = " << tile;
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
          wafer.tile.yield %ddr
              : memref<2000000xf16, #wafer.memory<ddr, tensor>>
        }
        return
      }
    }
)mlir";
    }
    os << R"mlir(  }
})mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CardExecutableCompilationTest,
       ClassifiesExactCapacityRejectionWithoutRepairingCardModule) {
  auto cardModule = oversizedCardModule();
  ASSERT_TRUE(cardModule);
  auto repeatedCardModule = oversizedCardModule();
  ASSERT_TRUE(repeatedCardModule);
  auto expectedTileIds = tileIds();
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::CardExecutableLoweringStatistics statistics;
  wafer::compiler::detail::CardExecutablePreparation preparation;
  wafer::compiler::ProgramDataHandoff programData;

  auto result = wafer::compiler::detail::compileCardModuleToExecutable(
      std::move(cardModule), wafer::CardId(0), expectedTileIds,
      /*materializationRelations=*/{}, preparation, program, executionConfig(),
      diagnostics, programData, &statistics);
  auto repeated = wafer::compiler::detail::compileCardModuleToExecutable(
      std::move(repeatedCardModule), wafer::CardId(0), expectedTileIds,
      /*materializationRelations=*/{}, preparation, program, executionConfig(),
      diagnostics, programData, &statistics);
  diagnostics.flush();

  EXPECT_FALSE(result.isAccepted());
  EXPECT_TRUE(result.isProvenExactRejection());
  EXPECT_EQ(result.gate, "spm-allocation");
  EXPECT_EQ(repeated.status, result.status);
  EXPECT_EQ(repeated.gate, result.gate);
  ASSERT_FALSE(repeated.tileFailures.empty());
  EXPECT_EQ(repeated.tileFailures.front().memoryPlanning.spmPlanningFailureKind,
            result.tileFailures.front().memoryPlanning.spmPlanningFailureKind);
  EXPECT_EQ(statistics.cardModuleCompilationInvocations, 2u);
  EXPECT_EQ(statistics.tileModuleLoweringAttempts, 0u);
  ASSERT_FALSE(result.tileFailures.empty());
  EXPECT_TRUE(result.tileFailures.front().memoryPlanning.spmCapacityOverflow);
  EXPECT_NE(diagnosticText.find("outcome=exact-rejection"), std::string::npos)
      << diagnosticText;
}

TEST_F(CardExecutableCompilationTest,
       KeepsIncompleteInvocationIndeterminateInsteadOfRejectingCandidate) {
  auto expectedTileIds = tileIds();
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::CardExecutableLoweringStatistics statistics;
  wafer::compiler::detail::CardExecutablePreparation preparation;
  wafer::compiler::ProgramDataHandoff programData;

  auto result = wafer::compiler::detail::compileCardModuleToExecutable(
      {}, wafer::CardId(0), expectedTileIds,
      /*materializationRelations=*/{}, preparation, program, executionConfig(),
      diagnostics, programData, &statistics);
  diagnostics.flush();

  EXPECT_FALSE(result.isAccepted());
  EXPECT_FALSE(result.isProvenExactRejection());
  EXPECT_EQ(result.status,
            wafer::compiler::detail::CardExecutableCompilationStatus::
                IndeterminateFailure);
  EXPECT_EQ(result.gate, "compilation-contract");
  EXPECT_EQ(statistics.cardModuleCompilationInvocations, 1u);
  EXPECT_NE(diagnosticText.find("outcome=indeterminate"), std::string::npos)
      << diagnosticText;
}

TEST_F(CardExecutableCompilationTest,
       TypedSelectedPreparationFailureStopsBeforeTileLowering) {
  class RejectingPreparation final
      : public wafer::compiler::detail::CardExecutablePreparation {
  public:
    mlir::LogicalResult prepareTileDataflow(
        llvm::MutableArrayRef<wafer::compiler::detail::CandidateTileDataflowIR>
            tiles,
        wafer::compiler::detail::CardExecutablePreparationFailure &failure)
        final {
      observedTiles = tiles.size();
      failure.kind = wafer::compiler::detail::
          CardExecutablePreparationFailureKind::Unsupported;
      failure.detail = "selected test preparation is unsupported";
      return mlir::failure();
    }

    size_t observedTiles = 0;
  } preparation;

  auto cardModule = oversizedCardModule();
  ASSERT_TRUE(cardModule);
  auto expectedTileIds = tileIds();
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::CardExecutableLoweringStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto result = wafer::compiler::detail::compileCardModuleToExecutable(
      std::move(cardModule), wafer::CardId(0), expectedTileIds,
      /*materializationRelations=*/{}, preparation, program, executionConfig(),
      diagnostics, programData, &statistics);
  EXPECT_EQ(result.status,
            wafer::compiler::detail::CardExecutableCompilationStatus::
                UnsupportedFailure);
  EXPECT_EQ(result.gate, "selected-tile-dataflow-preparation");
  EXPECT_EQ(preparation.observedTiles, expectedTileIds.size());
  EXPECT_EQ(statistics.tileModuleLoweringAttempts, 0u);
}

} // namespace
