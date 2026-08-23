//===- SearchRoutingTest.cpp ------------------------------------------===//

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

namespace {

using namespace wafer::compiler::testing;

TEST(SearchRoutingTest,
     ExplicitSearchStopsAtRepresentationWithoutBaselineOrActualCandidate) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string sourceBefore;
  llvm::raw_string_ostream beforeStream(sourceBefore);
  parsed.module->print(beforeStream);
  beforeStream.flush();
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::buildCardExecutable(
      parsed.context, *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, std::nullopt,
      programData);
  EXPECT_FALSE(static_cast<bool>(executable));
  llvm::consumeError(executable.takeError());
  diagnostics.flush();
  EXPECT_NE(diagnosticsText.find("physical-search incomplete "
                                 "required_coordinate=representation"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("candidate_actualizations=0"),
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
  EXPECT_NE(diagnosticsText.find("structural_readiness_queries=1"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("deterministic-card-executable-baseline"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("card-executable-compilation outcome="),
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
