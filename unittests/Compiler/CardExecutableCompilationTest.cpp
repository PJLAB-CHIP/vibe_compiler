//===- CardExecutableCompilationTest.cpp -------------------------------===//

#include "../../lib/Wafer/Compiler/CardExecutableCompilation.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

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

  llvm::SmallVector<wafer::PhysicalTileId, 16> tileIds() const {
    llvm::SmallVector<wafer::PhysicalTileId, 16> result;
    for (int64_t tile = 0; tile < 16; ++tile)
      result.push_back(wafer::PhysicalTileId(tile));
    return result;
  }

  wafer::compiler::ExecutionConfig executionConfig() const {
    auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
        1, wafer::RuntimeLaunchKind::Kernel);
    EXPECT_TRUE(static_cast<bool>(config));
    return *config;
  }

  mlir::OwningOpRef<mlir::ModuleOp> oversizedCardProgram() {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  wafer.card.program card_id = 0 {
)mlir";
    for (int64_t tile = 0; tile < 16; ++tile) {
      os << "    wafer.tile.program tile_id = " << tile;
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
       ClassifiesExactCapacityRejectionWithoutRepairingCardProgram) {
  auto cardProgram = oversizedCardProgram();
  ASSERT_TRUE(cardProgram);
  auto repeatedCardProgram = oversizedCardProgram();
  ASSERT_TRUE(repeatedCardProgram);
  auto expectedTileIds = tileIds();
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::compiler::detail::WholeCardSynthesisStatistics statistics;

  auto result = wafer::compiler::detail::compileCardProgramToExecutable(
      std::move(cardProgram), wafer::PhysicalCardId(0), expectedTileIds,
      /*selectedBufferRequests=*/{}, program, executionConfig(), diagnostics,
      &statistics);
  auto repeated = wafer::compiler::detail::compileCardProgramToExecutable(
      std::move(repeatedCardProgram), wafer::PhysicalCardId(0), expectedTileIds,
      /*selectedBufferRequests=*/{}, program, executionConfig(), diagnostics,
      &statistics);
  diagnostics.flush();

  EXPECT_FALSE(result.isAccepted());
  EXPECT_TRUE(result.isProvenExactRejection());
  EXPECT_EQ(result.gate, "spm-allocation");
  EXPECT_EQ(repeated.status, result.status);
  EXPECT_EQ(repeated.gate, result.gate);
  ASSERT_FALSE(repeated.tileFailures.empty());
  EXPECT_EQ(repeated.tileFailures.front().finalization.spmPlanningFailureKind,
            result.tileFailures.front().finalization.spmPlanningFailureKind);
  EXPECT_EQ(statistics.cardProgramCompilationInvocations, 2u);
  EXPECT_EQ(statistics.preTargetAttempts, 0u);
  ASSERT_FALSE(result.tileFailures.empty());
  EXPECT_TRUE(result.tileFailures.front().finalization.spmCapacityOverflow);
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
  wafer::compiler::detail::WholeCardSynthesisStatistics statistics;

  auto result = wafer::compiler::detail::compileCardProgramToExecutable(
      {}, wafer::PhysicalCardId(0), expectedTileIds,
      /*selectedBufferRequests=*/{}, program, executionConfig(), diagnostics,
      &statistics);
  diagnostics.flush();

  EXPECT_FALSE(result.isAccepted());
  EXPECT_FALSE(result.isProvenExactRejection());
  EXPECT_EQ(result.status,
            wafer::compiler::detail::CardExecutableCompilationStatus::
                IndeterminateFailure);
  EXPECT_EQ(result.gate, "compilation-contract");
  EXPECT_EQ(statistics.cardProgramCompilationInvocations, 1u);
  EXPECT_NE(diagnosticText.find("outcome=indeterminate"), std::string::npos)
      << diagnosticText;
}

} // namespace
