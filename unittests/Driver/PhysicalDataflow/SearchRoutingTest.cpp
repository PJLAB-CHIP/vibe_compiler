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

TEST(SearchRoutingTest, SearchBuildsOneCurrentIRDeviceExecutable) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto timing =
      std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
  std::optional<wafer::support::ScopedCompileTimingActivation> timingActivation;
  timingActivation.emplace(timing);
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::buildDeviceExecutable(
      parsed.context, *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, std::nullopt,
      programData);
  diagnostics.flush();
  if (!executable) {
    const std::string error = llvm::toString(executable.takeError());
    FAIL() << diagnosticsText << error;
  }
  EXPECT_EQ(executable->getTileExecutables().size(), 16u);
  EXPECT_NE(diagnosticsText.find("search-current-ir coverage="),
            std::string::npos);
  EXPECT_EQ(diagnosticsText.find("operation_not_supported"), std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("deterministic-device-executable-baseline"),
            std::string::npos)
      << diagnosticsText;
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
