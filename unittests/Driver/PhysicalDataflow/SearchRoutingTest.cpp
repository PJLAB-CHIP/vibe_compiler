//===- SearchRoutingTest.cpp ------------------------------------------===//

#include "TestSupport/CodeGen/ExecutableTestSupport.h"
#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/PhysicalDataflow/SearchCurrentIR.h"
#include "Wafer/Support/CompileTiming.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

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
           "name=accepted-candidates value=",
           "compile-counter category=search "
           "name=unsupported-candidates value=",
       })
    EXPECT_NE(diagnosticsText.find(counter.str()), std::string::npos)
        << counter.str() << "\n"
        << diagnosticsText;
  auto count = [&](llvm::StringRef name) -> uint64_t {
    auto prefix =
        "compile-counter category=search name=" + name.str() + " value=";
    auto text = llvm::StringRef(diagnosticsText);
    auto position = text.find(prefix);
    if (position == llvm::StringRef::npos) {
      ADD_FAILURE() << "missing counter " << name.str();
      return 0;
    }
    uint64_t value = 0;
    EXPECT_FALSE(text.drop_front(position + prefix.size())
                     .split(' ')
                     .first.getAsInteger(10, value));
    return value;
  };
  // A retained pipeline can become inapplicable after retile. That actual
  // attempt still consumes a standard trial and must not fail the query.
  EXPECT_GT(count("accepted-candidates"), 0u);
  EXPECT_EQ(count("accepted-candidates") + count("unsupported-candidates") +
                count("exact-rejected-candidates") +
                count("indeterminate-candidates"),
            count("candidate-actualizations"));
  EXPECT_EQ(count("trials-used"), count("candidate-actualizations"));
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

TEST(SearchRoutingTest, InspectionTransfersTheSameAcceptedIROwner) {
  using namespace wafer::compiler::detail;
  for (int64_t extent : {1024, 1025}) {
    auto parsed = parseRealScaleDependentProgram(extent);
    ASSERT_TRUE(parsed.module);
    std::string text;
    llvm::raw_string_ostream diagnostics(text);
    wafer::compiler::ProgramDataHandoff data;
    SearchCurrentIROptions options;
    options.limits = wafer::SearchLimits{2, 8};
    std::vector<mlir::Operation *> observedOwners;
    auto observe = [&](uint64_t, const ActualCandidateResult &candidate) {
      if (!candidate.isAccepted())
        return CandidateObservationAction::Continue;
      for (const auto &tile : candidate.compilation->executable->tiles)
        observedOwners.push_back(tile.getModule());
      return CandidateObservationAction::Inspect;
    };
    options.candidateObserver = observe;
    auto result = compileSearchCurrentIR(
        *parsed.module, realScaleDependentProgramMetadata(extent),
        executionConfig(), diagnostics, data, options);
    ASSERT_TRUE(result.isAccepted()) << text << result.detail;
    ASSERT_FALSE(observedOwners.empty());
    ASSERT_EQ(result.executable->tiles.size(), observedOwners.size());
    for (size_t i = 0; i < observedOwners.size(); ++i) {
      EXPECT_EQ(result.executable->tiles[i].getModule(), observedOwners[i]);
      EXPECT_TRUE(mlir::succeeded(
          mlir::verify(result.executable->tiles[i].getModule())));
    }
  }
}

TEST(SearchRoutingTest, MissingInspectionDoesNotReturnAnEarlierWinner) {
  using namespace wafer::compiler::detail;
  auto parsed = parseProgram();
  ASSERT_TRUE(parsed.module);
  std::string text;
  llvm::raw_string_ostream diagnostics(text);
  wafer::compiler::ProgramDataHandoff data;
  SearchCurrentIROptions options;
  options.limits = wafer::SearchLimits{1, 1};
  bool accepted = false;
  auto observe = [&](uint64_t index, const ActualCandidateResult &candidate) {
    EXPECT_EQ(index, 0u);
    accepted = candidate.isAccepted();
    return CandidateObservationAction::Continue;
  };
  options.candidateObserver = observe;
  auto result =
      compileSearchCurrentIR(*parsed.module, programMetadata(),
                             executionConfig(), diagnostics, data, options);
  ASSERT_TRUE(accepted) << text;
  EXPECT_EQ(result.status, ExecutableCompilationStatus::IndeterminateFailure);
  EXPECT_FALSE(result.executable);
}

TEST(SearchRoutingTest, ScalarExecutionChoicesConsumeTheSameActualLeafBudget) {
  using namespace wafer::compiler::detail;
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto parsed = parseRealScaleDependentProgram(extent);
    ASSERT_TRUE(parsed.module);
    auto function = *parsed.module->getOps<mlir::func::FuncOp>().begin();
    auto generic = *function.getOps<mlir::linalg::GenericOp>().begin();
    auto *original = &generic.getBody()->front();
    mlir::OpBuilder builder(original);
    auto loc = original->getLoc();
    auto lhs = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getF32FloatAttr(3.25));
    auto rhs = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getF32FloatAttr(1.75));
    auto quotient = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
    auto wide = builder.create<mlir::arith::ExtFOp>(loc, builder.getF32Type(),
                                                    original->getOperand(0));
    auto added = builder.create<mlir::arith::AddFOp>(loc, wide, quotient);
    auto narrowed =
        builder.create<mlir::arith::TruncFOp>(loc, builder.getF16Type(), added);
    original->getResult(0).replaceAllUsesWith(narrowed);
    original->erase();
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*parsed.module)));
    std::string text;
    llvm::raw_string_ostream diagnostics(text);
    auto timing =
        std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
    wafer::support::ScopedCompileTimingActivation activation(timing);
    wafer::compiler::ProgramDataHandoff data;
    SearchCurrentIROptions options;
    options.limits = wafer::SearchLimits{2, 8};
    SearchCurrentIRStatistics statistics;
    auto result = compileSearchCurrentIR(
        *parsed.module, realScaleDependentProgramMetadata(extent),
        executionConfig(), diagnostics, data, options, &statistics);
    timing->finishAndPrintSummary();
    diagnostics.flush();
    ASSERT_TRUE(result.isAccepted()) << text << result.detail;
    EXPECT_NE(text.find("-cpu-scalars value=0"), std::string::npos) << text;
    EXPECT_NE(text.find("-cpu-scalars value=1"), std::string::npos) << text;
    EXPECT_NE(text.find("name=cpu-scalar-materializations value="),
              std::string::npos)
        << text;
    EXPECT_EQ(statistics.traversal.candidateActualizations,
              statistics.traversal.trialsUsed);
    EXPECT_LE(statistics.traversal.trialsUsed, 8u);
    EXPECT_GT(statistics.acceptedCandidates, 0u);
    for (const auto &tile : result.executable->tiles)
      EXPECT_TRUE(mlir::succeeded(mlir::verify(tile.getModule())));
    // Transfer the selected owner to the public target LLVM consumer, including
    // its ordinary program bindings and managed entry ABI preparation.
    auto lowered = result.takeExecutable();
    auto executable =
        wafer::compiler::DeviceExecutableBuilder::makeDeviceExecutable(
            executionConfig(), std::move(lowered.runtimeLaunchContract),
            parsed.context, std::move(lowered.tiles),
            std::make_unique<wafer::compiler::ProgramDataHandoff>(
                std::move(data)));
    auto targets = wafer::compiler::compileDeviceExecutableToTargetLLVMModules(
        executable, diagnostics);
    ASSERT_TRUE(static_cast<bool>(targets))
        << text << llvm::toString(targets.takeError());
    EXPECT_EQ(targets->getModules().size(), 16u);
  }
}

} // namespace
