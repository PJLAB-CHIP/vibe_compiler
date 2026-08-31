//===- SearchRoutingTest.cpp ------------------------------------------===//

#include "TestSupport/CodeGen/ExecutableTestSupport.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <optional>

namespace {

using namespace wafer::compiler::testing;

void expectPolicyUnavailable(wafer::OptimizationConfig policy,
                             llvm::StringRef policyName, bool enableTiming) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string sourceBefore;
  llvm::raw_string_ostream beforeStream(sourceBefore);
  parsed.module->print(beforeStream);
  beforeStream.flush();
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  std::shared_ptr<wafer::support::CompileTimingSession> timing;
  std::optional<wafer::support::ScopedCompileTimingActivation> timingActivation;
  if (enableTiming) {
    timing =
        std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
    timingActivation.emplace(timing);
  }
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::buildDeviceExecutable(
      parsed.context, *parsed.module, programMetadata(), executionConfig(),
      policy, diagnostics, std::nullopt, programData);
  diagnostics.flush();
  ASSERT_FALSE(static_cast<bool>(executable));
  const std::string error = llvm::toString(executable.takeError());
  EXPECT_NE(error.find("operation_not_supported"), std::string::npos) << error;
  EXPECT_NE(diagnosticsText.find("operation_not_supported"), std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("optimization-policy=" + policyName.str()),
            std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("analyze-card-program"), std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("deterministic-device-executable-baseline"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("physical-search"), std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("actual-memory-target"), std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("wrote verified package"), std::string::npos)
      << diagnosticsText;
  std::string sourceAfter;
  llvm::raw_string_ostream afterStream(sourceAfter);
  parsed.module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(sourceAfter, sourceBefore);
}

TEST(SearchRoutingTest, SearchIsUnavailableBeforeCandidateAnalysis) {
  expectPolicyUnavailable(wafer::OptimizationConfig::search(), "search",
                          /*enableTiming=*/true);
}

TEST(SearchRoutingTest, NoneBuildsOneCurrentIRDeviceExecutable) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::buildDeviceExecutable(
      parsed.context, *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::none(), diagnostics, std::nullopt,
      programData);
  diagnostics.flush();
  if (!executable) {
    const std::string error = llvm::toString(executable.takeError());
    FAIL() << diagnosticsText << error;
  }
  EXPECT_EQ(executable->getTileExecutables().size(), 16u);
  EXPECT_EQ(diagnosticsText.find("operation_not_supported"), std::string::npos)
      << diagnosticsText;
}

} // namespace
