//===- SearchRoutingTest.cpp ------------------------------------------===//

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

using namespace wafer::compiler::testing;

TEST(SearchRoutingTest,
     ExplicitSearchReturnsRetainedActualWinnerWithoutBaseline) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string sourceBefore;
  llvm::raw_string_ostream beforeStream(sourceBefore);
  parsed.module->print(beforeStream);
  beforeStream.flush();
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto timing =
      std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
  wafer::support::ScopedCompileTimingActivation timingActivation(timing);
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::buildCardExecutable(
      parsed.context, *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, std::nullopt,
      programData);
  diagnostics.flush();
  ASSERT_TRUE(static_cast<bool>(executable)) << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("physical-search result "
                                 "coverage=feasible-partial"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("candidate_actualizations=1"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("root_works="), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("region_states=1"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("temporal_states=1"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("temporal_unsupported=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("temporal_indeterminate=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("structural_readiness_queries=0"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("representation_states=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("representation_unsupported=0"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("movement_states=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("movement_unsupported=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("storage_states=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("storage_unsupported=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("event_graph_queries=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("event_graphs_built=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("execution_structure_queries=0"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("execution_structure_states=0"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("structure_storage_queries=0"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("structure_storage_states=0"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("schedule_queries=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("schedule_states=0"), std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("deterministic-card-executable-baseline"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(
      diagnosticsText.find("card-executable-compilation outcome=accepted"),
      std::string::npos)
      << diagnosticsText;
  std::string sourceAfter;
  llvm::raw_string_ostream afterStream(sourceAfter);
  parsed.module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(sourceAfter, sourceBefore);
}

TEST(SearchRoutingTest, NoneStillOwnsTheIndependentAcceptedPath) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::buildCardExecutable(
      parsed.context, *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::none(), diagnostics, std::nullopt,
      programData);
  diagnostics.flush();
  ASSERT_TRUE(static_cast<bool>(executable)) << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("physical-search"), std::string::npos)
      << diagnosticsText;
}

} // namespace
