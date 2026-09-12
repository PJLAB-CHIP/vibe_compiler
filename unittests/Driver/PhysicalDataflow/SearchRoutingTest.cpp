//===- SearchRoutingTest.cpp ------------------------------------------===//

#include "TestSupport/CodeGen/ExecutableTestSupport.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <limits>
#include <memory>
#include <optional>

namespace {

using namespace wafer::compiler::testing;

TEST(SearchRoutingTest, CompileCountersRetainOverflowAsUnknownEvidence) {
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::support::CompileTimingSession timing(diagnostics);
  timing.addCounter("test", "work", std::numeric_limits<uint64_t>::max());
  timing.addCounter("test", "work", 1);
  timing.finishAndPrintSummary();
  diagnostics.flush();
  EXPECT_NE(diagnosticsText.find("compile-counter category=test name=work "
                                 "value=18446744073709551615 overflow=true"),
            std::string::npos)
      << diagnosticsText;
}

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
  timing->finishAndPrintSummary();
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
  for (llvm::StringRef counter : {
           "compile-counter category=accepted-physical-ir name=tile-regions",
           "compile-counter category=accepted-physical-ir "
           "name=tile-0-regions value=1",
           "compile-counter category=accepted-instr "
           "name=instructions-executions",
           "compile-counter category=accepted-instr name=tiles value=16",
           "compile-counter category=layout name=solver-work",
           "compile-counter category=movement name=ddr-loads",
           "compile-counter category=search "
           "name=region-proposal-count value=1",
           "compile-counter category=search "
           "name=region-proposal-0-merges value=0",
           "compile-counter category=search "
           "name=region-proposal-0-regions value=16",
           "compile-counter category=search "
           "name=region-proposal-0-maximum-roots value=1",
           "compile-counter category=search "
           "name=region-candidate-0-objective-known value=1",
           "compile-counter category=search "
           "name=candidate-actualizations value=42",
           "compile-counter category=search name=width value=8",
           "compile-counter category=search name=trials value=42",
           "compile-counter category=search "
           "name=accepted-candidates value=42",
           "compile-counter category=search "
           "name=unsupported-candidates value=0",
       })
    EXPECT_NE(diagnosticsText.find(counter.str()), std::string::npos)
        << counter.str() << "\n"
        << diagnosticsText;
}

TEST(SearchRoutingTest, PublicLimitsBoundTheActualSearchWork) {
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
      wafer::OptimizationConfig::search(wafer::SearchLimits{1, 1}), diagnostics,
      std::nullopt, programData);
  timing->finishAndPrintSummary();
  diagnostics.flush();
  if (!executable) {
    const std::string error = llvm::toString(executable.takeError());
    FAIL() << diagnosticsText << error;
  }
  for (llvm::StringRef counter : {
           "compile-counter category=search name=width value=1",
           "compile-counter category=search name=trials value=1",
           "compile-counter category=search name=trials-used value=1",
           "compile-counter category=search name=trials-remaining value=0",
           "compile-counter category=search name=structural-states value=1",
           "compile-counter category=search "
           "name=candidate-actualizations value=1",
           "compile-counter category=search "
           "name=accepted-candidates value=1",
           "compile-counter category=search "
           "name=unsupported-candidates value=0",
       })
    EXPECT_NE(diagnosticsText.find(counter.str()), std::string::npos)
        << counter.str() << "\n"
        << diagnosticsText;
}

TEST(SearchRoutingTest, ZeroPublicSearchLimitFailsBeforeActualization) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::buildDeviceExecutable(
      parsed.context, *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(wafer::SearchLimits{0, 1}), diagnostics,
      std::nullopt, programData);
  ASSERT_FALSE(executable);
  EXPECT_NE(llvm::toString(executable.takeError()).find("positive work limits"),
            std::string::npos);
  diagnostics.flush();
  EXPECT_EQ(diagnosticsText.find("search-current-ir coverage="),
            std::string::npos);
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
  EXPECT_EQ(diagnosticsText.find("search-current-ir"), std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("name=region-refinement"), std::string::npos)
      << diagnosticsText;
}

} // namespace
