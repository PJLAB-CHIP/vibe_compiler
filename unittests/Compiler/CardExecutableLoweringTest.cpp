//===- CardExecutableLoweringTest.cpp ----------------------------===//

#include "../../lib/Wafer/Compiler/CardExecutableLowering.h"
#include "../../lib/Wafer/Compiler/BoundedTileExecutor.h"
#include "../../lib/Wafer/Compiler/CardExecutableInternal.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/TargetCodeGenInternal.h"
#include "Wafer/Compiler/ProgramData.h"

#include "Wafer/IR/WaferDialect.h"

#include "Wafer/Support/CompileWorkStatistics.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

namespace {

TEST(CardExecutableLoweringFailureTest,
     KeepsClassificationOutOfDiagnosticText) {
  using wafer::compiler::detail::CardExecutableLoweringFailure;
  using wafer::compiler::detail::CardExecutableLoweringFailureKind;

  CardExecutableLoweringFailure stageFailure{
      CardExecutableLoweringFailureKind::TileDomain, "arbitrary detail"};
  EXPECT_FALSE(stageFailure.isProvenExactRejection());
  EXPECT_EQ(stageFailure.getDiagnosticLabel(), "tile-domain");

  CardExecutableLoweringFailure indeterminate{
      CardExecutableLoweringFailureKind::RuntimeLaunchContract,
      "tile-domain text must not affect classification"};
  EXPECT_FALSE(indeterminate.isProvenExactRejection());
  EXPECT_EQ(indeterminate.getDiagnosticLabel(), "runtime-launch-contract");
}

class CardExecutableLoweringTest : public ::testing::Test {
protected:
  CardExecutableLoweringTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_shared<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseTileModule(llvm::StringRef entry) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
)mlir";
    os << entry << "\n}\n";
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  wafer::frontend::FrontendProgramVerificationResult emptyProgram() const {
    wafer::frontend::FrontendProgramVerificationResult program;
    program.numPartitions = 1;
    return program;
  }

  llvm::Expected<wafer::compiler::ExecutionConfig> executionConfig() const {
    return wafer::compiler::ExecutionConfig::createForSingleCard(1);
  }

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>>
  makeTileModules(mlir::ModuleOp source) const {
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
    modules.reserve(16);
    for (int64_t tileId = 0; tileId < 16; ++tileId)
      modules.emplace_back(mlir::cast<mlir::ModuleOp>(source->clone()));
    return modules;
  }

  mlir::DialectRegistry registry;
  std::shared_ptr<mlir::MLIRContext> context;
};

TEST_F(CardExecutableLoweringTest, ConsumesCompleteTileDomain) {
  auto module = parseTileModule(R"mlir(
  func.func @main() {
    return
  })mlir");
  ASSERT_TRUE(module);
  auto config = executionConfig();
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> tiles =
      makeTileModules(*module);
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::CardExecutableLoweringStatistics statistics;
  wafer::compiler::detail::CardExecutableLoweringFailure failure;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::lowerTileModulesToCardExecutable(
      std::move(tiles), emptyProgram(), *config, diagnostics, failure,
      programData, &statistics);

  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticText;
  ASSERT_EQ(executable->tiles.size(), 16u);
  for (auto [expectedTileId, tile] : llvm::enumerate(executable->tiles)) {
    EXPECT_EQ(tile.getCardId(), wafer::CardId(0));
    EXPECT_EQ(tile.getTileId(),
              wafer::TileId(static_cast<int64_t>(expectedTileId)));
    EXPECT_EQ(tile.getEntryLocalCompletionKind(),
              wafer::compiler::EntryLocalCompletionKind::ReturnAfterLocalDrain);
  }
  EXPECT_FALSE(failure);
  EXPECT_EQ(statistics.tileModuleLoweringAttempts, 1u);
  EXPECT_EQ(statistics.tileModuleLoweringSuccesses, 1u);
  EXPECT_EQ(statistics.cardExecutablesProduced, 1u);
}

TEST_F(CardExecutableLoweringTest,
       DefersTargetABIFailureToRetainedTargetOutput) {
  auto module = parseTileModule(R"mlir(
  func.func @main(
      %input: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    return
  })mlir");
  ASSERT_TRUE(module);
  auto config = executionConfig();
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> tiles =
      makeTileModules(*module);
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::CardExecutableLoweringStatistics statistics;
  wafer::compiler::detail::CardExecutableLoweringFailure failure;
  wafer::compiler::ProgramDataHandoff programData;
  auto lowered = wafer::compiler::detail::lowerTileModulesToCardExecutable(
      std::move(tiles), emptyProgram(), *config, diagnostics, failure,
      programData, &statistics);

  ASSERT_TRUE(mlir::succeeded(lowered)) << diagnosticText;
  EXPECT_FALSE(failure);
  EXPECT_EQ(statistics.tileModuleLoweringAttempts, 1u);
  EXPECT_EQ(statistics.tileModuleLoweringSuccesses, 1u);
  EXPECT_EQ(statistics.cardExecutablesProduced, 1u);

  wafer::compiler::CardExecutable executable =
      wafer::compiler::CardExecutableBuilder::makeCardExecutable(
          *config, std::move(lowered->runtimeLaunchContract), context,
          std::move(lowered->tiles),
          std::make_unique<wafer::compiler::ProgramDataHandoff>());
  wafer::compiler::detail::TargetLLVMCompilationStatistics targetStats;
  targetStats.targetABIPreparationAttempts = 99;
  targetStats.targetLoweringAttempts = 99;
  targetStats.targetTranslationAttempts = 99;
  targetStats.maximumTilePipelineWorkers = 99;
  llvm::Expected<wafer::compiler::TargetLLVMModules> targetModules =
      wafer::compiler::detail::compileCardExecutableToTargetLLVMModulesImpl(
          executable, diagnostics, std::nullopt,
          wafer::compiler::detail::ProfileCaptureKind::None, &targetStats);
  ASSERT_FALSE(static_cast<bool>(targetModules));
  llvm::consumeError(targetModules.takeError());
  EXPECT_NE(diagnosticText.find("target ABI preparation failed"),
            std::string::npos)
      << diagnosticText;
  EXPECT_EQ(targetStats.targetABIPreparationAttempts, 16u);
  EXPECT_EQ(targetStats.targetLoweringAttempts, 0u);
  EXPECT_EQ(targetStats.targetTranslationAttempts, 0u);
}

TEST_F(CardExecutableLoweringTest,
       FinalTargetOutputRunsEachTileOnceWithBoundedParallelism) {
  auto module = parseTileModule(R"mlir(
  func.func @main() {
    return
  })mlir");
  ASSERT_TRUE(module);
  auto config = executionConfig();
  ASSERT_TRUE(static_cast<bool>(config));

  auto workSession =
      std::make_shared<wafer::support::CompileWorkStatisticsSession>();
  wafer::support::ScopedCompileWorkStatisticsActivation workActivation(
      workSession);
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::CardExecutableLoweringStatistics loweringStats;
  wafer::compiler::detail::CardExecutableLoweringFailure failure;
  wafer::compiler::ProgramDataHandoff programData;
  auto lowered = wafer::compiler::detail::lowerTileModulesToCardExecutable(
      makeTileModules(*module), emptyProgram(), *config, diagnostics, failure,
      programData, &loweringStats);
  ASSERT_TRUE(mlir::succeeded(lowered)) << diagnosticText;

  wafer::support::CompileWorkStatistics cardExecutableWork =
      workSession->snapshot();
  EXPECT_EQ(cardExecutableWork.targetABIModuleClones, 0u);
  EXPECT_EQ(cardExecutableWork.targetLoweringInvocations, 0u);
  EXPECT_EQ(cardExecutableWork.targetTranslationInvocations, 0u);

  wafer::compiler::CardExecutable executable =
      wafer::compiler::CardExecutableBuilder::makeCardExecutable(
          *config, std::move(lowered->runtimeLaunchContract), context,
          std::move(lowered->tiles),
          std::make_unique<wafer::compiler::ProgramDataHandoff>());
  wafer::compiler::detail::TargetLLVMCompilationStatistics targetStats;
  llvm::Expected<wafer::compiler::TargetLLVMModules> targetModules =
      wafer::compiler::detail::compileCardExecutableToTargetLLVMModulesImpl(
          executable, diagnostics, std::nullopt,
          wafer::compiler::detail::ProfileCaptureKind::None, &targetStats);
  ASSERT_TRUE(static_cast<bool>(targetModules))
      << diagnosticText << llvm::toString(targetModules.takeError());

  const wafer::support::CompileWorkStatistics finalWork =
      workSession->snapshot();
  EXPECT_EQ(targetStats.targetABIPreparationAttempts, 16u);
  EXPECT_EQ(targetStats.targetLoweringAttempts, 16u);
  EXPECT_EQ(targetStats.targetTranslationAttempts, 16u);
  EXPECT_EQ(finalWork.targetABIModuleClones, 16u);
  EXPECT_EQ(finalWork.targetLoweringInvocations, 16u);
  EXPECT_EQ(finalWork.targetTranslationInvocations, 16u);
  const unsigned expectedWorkers =
      wafer::compiler::detail::getBoundedTilePipelineWorkerCount(
          context.get(), /*tileCount=*/16);
  EXPECT_EQ(targetStats.maximumTilePipelineWorkers, expectedWorkers);
  if (expectedWorkers > 1)
    EXPECT_GT(targetStats.maximumTilePipelineWorkers, 1u);

  ASSERT_EQ(targetModules->getModules().size(), 16u);
  for (auto [launchSlot, targetModule] :
       llvm::enumerate(targetModules->getModules()))
    EXPECT_EQ(targetModule.getLaunchSlotId(),
              wafer::LaunchSlotId(static_cast<int64_t>(launchSlot)));
}

TEST_F(CardExecutableLoweringTest, RejectsIncompleteTileDomainBeforeMutation) {
  auto module = parseTileModule(R"mlir(
  func.func @main() {
    return
  })mlir");
  ASSERT_TRUE(module);
  auto config = executionConfig();
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> tiles =
      makeTileModules(*module);
  tiles.pop_back();
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::CardExecutableLoweringStatistics statistics;
  wafer::compiler::detail::CardExecutableLoweringFailure failure;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::lowerTileModulesToCardExecutable(
      std::move(tiles), emptyProgram(), *config, diagnostics, failure,
      programData, &statistics);

  EXPECT_TRUE(mlir::failed(executable));
  EXPECT_EQ(
      failure.kind,
      wafer::compiler::detail::CardExecutableLoweringFailureKind::TileDomain);
  EXPECT_FALSE(failure.isProvenExactRejection());
  EXPECT_EQ(statistics.tileModuleLoweringAttempts, 1u);
  EXPECT_EQ(statistics.tileModuleLoweringSuccesses, 0u);
}

TEST_F(CardExecutableLoweringTest,
       BoundaryDDRProofAcceptsIdentityButRejectsAlternatingLoopRoots) {
  auto verify = [&](bool alternateRoots) {
    std::string source = R"mlir(
module {
  func.func @main()
      -> memref<4xf32, #wafer.memory<ddr, tensor>> {
    %source = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %output0 = memref.alloc()
        : memref<4xf32, #wafer.memory<ddr, tensor>>
    %output1 = memref.alloc()
        : memref<4xf32, #wafer.memory<ddr, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %results:2 = scf.for %index = %c0 to %c2 step %c1
        iter_args(%current0 = %output0, %current1 = %output1)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>,
            memref<4xf32, #wafer.memory<ddr, tensor>>) {
      wafer.instr.wdma %source to %current0
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
         to memref<4xf32, #wafer.memory<ddr, tensor>>
)mlir";
    source += alternateRoots ? R"mlir(
      scf.yield %current1, %current0
)mlir"
                             : R"mlir(
      scf.yield %current0, %current1
)mlir";
    source += R"mlir(
          : memref<4xf32, #wafer.memory<ddr, tensor>>,
            memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    return %output0 : memref<4xf32, #wafer.memory<ddr, tensor>>
  }
}
)mlir";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
    if (!module) {
      ADD_FAILURE() << "failed to parse loop-carried DDR fixture";
      return false;
    }
    wafer::compiler::TileExecutable tile =
        wafer::compiler::CardExecutableBuilder::makeTileExecutable(
            wafer::CardId(0), wafer::TileId(0), wafer::LaunchSlotId(0),
            std::move(module), "main",
            /*programBindings=*/{}, wafer::compiler::TransportContract::None);
    return wafer::compiler::detail::hasBoundaryOnlyDDRMovementEvidence(tile);
  };

  EXPECT_TRUE(verify(/*alternateRoots=*/false));
  EXPECT_FALSE(verify(/*alternateRoots=*/true));
}

} // namespace
