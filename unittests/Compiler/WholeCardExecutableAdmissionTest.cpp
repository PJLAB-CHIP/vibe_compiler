//===- WholeCardExecutableAdmissionTest.cpp ----------------------------===//

#include "../../lib/Wafer/Compiler/WholeCardExecutableAdmission.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

#include "Wafer/IR/WaferDialect.h"

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

class WholeCardExecutableAdmissionTest : public ::testing::Test {
protected:
  WholeCardExecutableAdmissionTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
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
    return wafer::compiler::ExecutionConfig::createForSingleCard(
        1, wafer::RuntimeLaunchKind::Kernel);
  }

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>>
  makePhysicalTileModules(mlir::ModuleOp source) const {
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
    modules.reserve(16);
    for (int64_t tileId = 0; tileId < 16; ++tileId)
      modules.emplace_back(mlir::cast<mlir::ModuleOp>(source->clone()));
    return modules;
  }

  std::vector<std::shared_ptr<const std::string>>
  makeSelectedTileEvidence() const {
    return std::vector<std::shared_ptr<const std::string>>(
        16, std::make_shared<const std::string>("selected tile evidence"));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(WholeCardExecutableAdmissionTest,
       ConsumesCompletePhysicalTileDomainAndPreservesSelectedTileEvidence) {
  auto module = parseTileModule(R"mlir(
  func.func @main() {
    return
  })mlir");
  ASSERT_TRUE(module);
  auto config = executionConfig();
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> tiles =
      makePhysicalTileModules(*module);
  std::vector<std::shared_ptr<const std::string>> selectedTileIR =
      makeSelectedTileEvidence();
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::WholeCardSynthesisStatistics statistics;
  std::string failureGate;
  auto accepted = wafer::compiler::detail::admitWholeCardExecutable(
      std::move(tiles), selectedTileIR, emptyProgram(), *config, diagnostics,
      &failureGate, &statistics);

  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->tiles.size(), 16u);
  for (auto [expectedTileId, tile] : llvm::enumerate(accepted->tiles)) {
    EXPECT_EQ(tile.getPhysicalCardId(), wafer::PhysicalCardId(0));
    EXPECT_EQ(tile.getPhysicalTileId(),
              wafer::PhysicalTileId(static_cast<int64_t>(expectedTileId)));
    EXPECT_EQ(tile.getSelectedTileIR(), "selected tile evidence");
    EXPECT_EQ(tile.getEntryLocalCompletionKind(),
              wafer::compiler::EntryLocalCompletionKind::ReturnAfterLocalDrain);
  }
  EXPECT_TRUE(failureGate.empty());
  EXPECT_EQ(statistics.preTargetAttempts, 1u);
  EXPECT_EQ(statistics.preTargetAccepted, 1u);
  EXPECT_EQ(statistics.targetGateInvocations, 1u);
  EXPECT_EQ(statistics.targetTileGateInvocations, 16u);
  EXPECT_EQ(statistics.admittedExecutableCount, 1u);
}

TEST_F(WholeCardExecutableAdmissionTest,
       ReportsLateTargetABIFailureAfterPreTargetAdmission) {
  auto module = parseTileModule(R"mlir(
  func.func @main(
      %input: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    return
  })mlir");
  ASSERT_TRUE(module);
  auto config = executionConfig();
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> tiles =
      makePhysicalTileModules(*module);
  std::vector<std::shared_ptr<const std::string>> selectedTileIR =
      makeSelectedTileEvidence();
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::WholeCardSynthesisStatistics statistics;
  std::string failureGate;
  auto accepted = wafer::compiler::detail::admitWholeCardExecutable(
      std::move(tiles), selectedTileIR, emptyProgram(), *config, diagnostics,
      &failureGate, &statistics);

  EXPECT_TRUE(mlir::failed(accepted));
  EXPECT_EQ(failureGate, "target-abi-preparation");
  EXPECT_NE(diagnosticText.find("target-abi-preparation"), std::string::npos)
      << diagnosticText;
  EXPECT_EQ(statistics.preTargetAttempts, 1u);
  EXPECT_EQ(statistics.preTargetAccepted, 1u);
  EXPECT_EQ(statistics.targetGateInvocations, 1u);
  EXPECT_EQ(statistics.targetTileGateInvocations, 16u);
  EXPECT_EQ(statistics.admittedExecutableCount, 0u);
}

TEST_F(WholeCardExecutableAdmissionTest,
       RejectsIncompletePhysicalTileDomainBeforeMutation) {
  auto module = parseTileModule(R"mlir(
  func.func @main() {
    return
  })mlir");
  ASSERT_TRUE(module);
  auto config = executionConfig();
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> tiles =
      makePhysicalTileModules(*module);
  tiles.pop_back();
  std::vector<std::shared_ptr<const std::string>> selectedTileIR =
      makeSelectedTileEvidence();
  selectedTileIR.pop_back();
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::WholeCardSynthesisStatistics statistics;
  std::string failureGate;
  auto accepted = wafer::compiler::detail::admitWholeCardExecutable(
      std::move(tiles), selectedTileIR, emptyProgram(), *config, diagnostics,
      &failureGate, &statistics);

  EXPECT_TRUE(mlir::failed(accepted));
  EXPECT_EQ(failureGate, "physical-tile-domain");
  EXPECT_EQ(statistics.preTargetAttempts, 1u);
  EXPECT_EQ(statistics.preTargetAccepted, 0u);
  EXPECT_EQ(statistics.targetGateInvocations, 0u);
  EXPECT_EQ(statistics.targetTileGateInvocations, 0u);
}

TEST_F(WholeCardExecutableAdmissionTest,
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
    wafer::compiler::PhysicalTileExecutable tile =
        wafer::compiler::ExecutableBundleBuilder::makePhysicalTile(
            wafer::PhysicalCardId(0), wafer::PhysicalTileId(0),
            wafer::LaunchSlotId(0), std::move(module), "main",
            /*programBindings=*/{}, wafer::compiler::TransportContract::None);
    return wafer::compiler::detail::hasBoundaryOnlyDDRMovementEvidence(tile);
  };

  EXPECT_TRUE(verify(/*alternateRoots=*/false));
  EXPECT_FALSE(verify(/*alternateRoots=*/true));
}

} // namespace
