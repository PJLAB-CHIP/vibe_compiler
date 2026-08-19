//===- CardExecutableSearchTest.cpp -----------------------------------===//

#include "Wafer/Planning/Search/CardExecutableSearch.h"
#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "gtest/gtest.h"

namespace {
using namespace wafer::compiler::testing;

TEST(CardExecutableSearchTest,
     PartialMechanismsReturnAcceptedBaselineWithoutRematerialization) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto baseline = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, programMetadata(), executionConfig(), diagnostics,
      programData, &statistics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << diagnosticsText;
  ASSERT_EQ(baseline->executable.tiles.size(), 16u);
  mlir::Operation *firstTileOwner =
      baseline->executable.tiles.front().getModule().getOperation();

  auto selected = wafer::compiler::detail::runCardExecutableSearch(
      *parsed.module, *baseline->programAnalysis,
      std::move(baseline->executable));

  ASSERT_TRUE(mlir::succeeded(selected));
  ASSERT_EQ(selected->tiles.size(), 16u);
  EXPECT_EQ(selected->tiles.front().getModule().getOperation(), firstTileOwner);
  EXPECT_EQ(statistics.baselineCardModuleMaterializations, 1u);
  EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(diagnosticsText.find("card-executable-selection"),
            std::string::npos);
}

} // namespace
