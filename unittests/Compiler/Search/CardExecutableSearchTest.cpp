//===- CardExecutableSearchTest.cpp -----------------------------------===//

#include "Wafer/Compiler/Search/CardExecutableSearch.h"
#include "../TestSupport/CardExecutableTestSupport.h"

#include "gtest/gtest.h"

namespace {
using namespace wafer::compiler::testing;

TEST(CardExecutableSearchTest,
     EmptyMechanismSetReturnsAcceptedBaselineWithoutRematerialization) {
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

  wafer::compiler::detail::CardExecutableLoweringResult selected =
      wafer::compiler::detail::runCardExecutableSearch(
          std::move(baseline->executable));

  ASSERT_EQ(selected.tiles.size(), 16u);
  EXPECT_EQ(selected.tiles.front().getModule().getOperation(), firstTileOwner);
  EXPECT_EQ(statistics.baselineCardModuleMaterializations, 1u);
  EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(diagnosticsText.find("card-executable-selection"),
            std::string::npos);
}

} // namespace
