//===- CompilationTest.cpp - Typed compiler request tests ----------------===//

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/Package.h"
#include "Wafer/Compiler/TargetCodeGen.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

TEST(CompilationTest, ExecutionConfigSeparatesPartitionsFromTiles) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config));
  EXPECT_EQ(config->getNumPartitions(), 1);
  EXPECT_EQ(config->getTileCount(), 16);

  for (int64_t rejected :
       {std::numeric_limits<int64_t>::min(), int64_t{-1}, int64_t{0},
        int64_t{2}, int64_t{8}, int64_t{15}, int64_t{16}, int64_t{17},
        std::numeric_limits<int64_t>::max()}) {
    auto rejectedConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
        rejected, wafer::RuntimeLaunchKind::Kernel);
    ASSERT_FALSE(static_cast<bool>(rejectedConfig));
    EXPECT_FALSE(llvm::toString(rejectedConfig.takeError()).empty());
  }
}

TEST(CompilationTest, ExecutionConfigEqualityCoversTypedDomainsAndLaunchKind) {
  auto first = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  auto same = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  auto differentLaunchKind =
      wafer::compiler::ExecutionConfig::createForSingleCard(
          1, wafer::RuntimeLaunchKind::Model);
  ASSERT_TRUE(static_cast<bool>(first));
  ASSERT_TRUE(static_cast<bool>(same));
  ASSERT_TRUE(static_cast<bool>(differentLaunchKind));
  EXPECT_EQ(*first, *same);
  EXPECT_NE(*first, *differentLaunchKind);
  EXPECT_EQ(differentLaunchKind->getNumPartitions(), 1);
  EXPECT_EQ(differentLaunchKind->getTileCount(), 16);
}

TEST(CompilationTest, CompilationRequestOwnsSourceAndHasNoImplicitDefaults) {
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::ExecutionConfig>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::CompilationRequest>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::CompilationRequest>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::CompilationRequest>);
  static_assert(!std::is_default_constructible_v<
                wafer::compiler::TileExecutable>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::TileExecutable>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::TileExecutable>);
  static_assert(!std::is_default_constructible_v<
                wafer::compiler::CardExecutable>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::CardExecutable>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::CardExecutable>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::TargetLLVMModule>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::TargetLLVMModule>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::TargetLLVMModule>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::TargetLLVMModules>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::TargetLLVMModules>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::TargetLLVMModules>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::LinkedTargetModules>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::LinkedTargetModules>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::LinkedTargetModules>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::VerifiedPackage>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::VerifiedPackage>);
  static_assert(std::is_move_constructible_v<wafer::compiler::VerifiedPackage>);

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config));
  std::string source = "/tmp/source.program";
  auto request =
      wafer::compiler::CompilationRequest::create(source, std::move(*config));
  ASSERT_TRUE(static_cast<bool>(request));
  source.assign("/tmp/changed-after-request-construction.program");
  EXPECT_EQ(request->getSourceProgramDirectory(), "/tmp/source.program");
  EXPECT_EQ(request->getExecutionConfig().getNumPartitions(), 1);
  EXPECT_EQ(request->getExecutionConfig().getTileCount(), 16);
  EXPECT_EQ(request->getExecutionConfig().getTargetIdentityId(),
            wafer::TargetIdentityId::waferTx81SingleCard());
  EXPECT_EQ(request->getExecutionConfig().getRuntimeLaunchKind(),
            wafer::RuntimeLaunchKind::Kernel);
}

TEST(CompilationTest, CompilationRequestRejectsEmptySourceLocator) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config));
  auto request = wafer::compiler::CompilationRequest::create("", *config);
  ASSERT_FALSE(static_cast<bool>(request));
  EXPECT_FALSE(llvm::toString(request.takeError()).empty());
}

TEST(CompilationTest, ProfileOptionsRequireCompleteTileKernelDomain) {
  auto fullCard = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(fullCard));
  auto accepted = wafer::compiler::CompilationOptions::profile(*fullCard);
  ASSERT_TRUE(static_cast<bool>(accepted));
  EXPECT_TRUE(accepted->shouldProduceProfileInstrumentation());
  EXPECT_FALSE(accepted->shouldReportDetailedTiming());
  wafer::OptimizationConfig none = wafer::OptimizationConfig::none();
  auto acceptedNone =
      wafer::compiler::CompilationOptions::profile(*fullCard, none);
  ASSERT_TRUE(static_cast<bool>(acceptedNone));
  EXPECT_EQ(acceptedNone->getOptimizationConfig(), none);
  auto acceptedTimed = wafer::compiler::CompilationOptions::profile(
      *fullCard, none, wafer::compiler::CompilationTimingMode::Detailed);
  ASSERT_TRUE(static_cast<bool>(acceptedTimed));
  EXPECT_TRUE(acceptedTimed->shouldReportDetailedTiming());

  auto model = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Model);
  ASSERT_TRUE(static_cast<bool>(model));
  auto rejectedModel = wafer::compiler::CompilationOptions::profile(*model);
  ASSERT_FALSE(static_cast<bool>(rejectedModel));
  EXPECT_NE(llvm::toString(rejectedModel.takeError())
                .find("complete-card Tile kernel launch"),
            std::string::npos);
  EXPECT_FALSE(wafer::compiler::CompilationOptions::standard()
                   .shouldProduceProfileInstrumentation());
  EXPECT_FALSE(wafer::compiler::CompilationOptions::standard()
                   .shouldReportDetailedTiming());
  EXPECT_TRUE(wafer::compiler::CompilationOptions::standard(
                  wafer::OptimizationConfig::search(),
                  wafer::compiler::CompilationTimingMode::Detailed)
                  .shouldReportDetailedTiming());
}

TEST(CompilationTest, OptimizationConfigHasExactlySearchAndNonePolicies) {
  wafer::OptimizationConfig search = wafer::OptimizationConfig::search();
  wafer::OptimizationConfig none = wafer::OptimizationConfig::none();

  EXPECT_TRUE(search.isSearch());
  EXPECT_FALSE(search.isNone());
  EXPECT_FALSE(none.isSearch());
  EXPECT_TRUE(none.isNone());
  EXPECT_NE(search, none);

  wafer::compiler::CompilationOptions options =
      wafer::compiler::CompilationOptions::standard(none);
  EXPECT_EQ(options.getOptimizationConfig(), none);
}

TEST(CompilationTest, DetailedTimingAggregatesInvocationLocalSpans) {
  std::string output;
  llvm::raw_string_ostream diagnostics(output);
  auto session =
      std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
  {
    wafer::support::ScopedCompileTimingActivation activation(session);
    wafer::support::ScopedCompileTimingSpan span("stage", "test-pipeline",
                                                 "test-item", "request=7");
    span.markFailed();
  }
  std::thread worker([session] {
    wafer::support::ScopedCompileTimingActivation activation(session);
    wafer::support::ScopedCompileTimingSpan span("analysis", "test-worker",
                                                 "parallel-item");
  });
  worker.join();
  std::vector<std::thread> workers;
  workers.reserve(16);
  for (size_t workerIndex = 0; workerIndex < 16; ++workerIndex) {
    workers.emplace_back([session] {
      wafer::support::ScopedCompileTimingActivation activation(session);
      for (size_t iteration = 0; iteration < 64; ++iteration) {
        wafer::support::ScopedCompileTimingSpan span("analysis", "test-worker",
                                                     "sharded-parallel-item");
      }
    });
  }
  for (std::thread &parallelWorker : workers)
    parallelWorker.join();
  session->finishAndPrintSummary();
  EXPECT_NE(output.find("compile-timing-summary-begin"), std::string::npos);
  size_t failedRow = output.find("| stage | test-pipeline | test-item | 1 |");
  ASSERT_NE(failedRow, std::string::npos);
  size_t failedRowEnd = output.find('\n', failedRow);
  ASSERT_NE(failedRowEnd, std::string::npos);
  llvm::StringRef failedRowText(output.data() + failedRow,
                                failedRowEnd - failedRow);
  EXPECT_TRUE(failedRowText.ends_with("| 1 |"));
  EXPECT_NE(output.find("| analysis | test-worker | parallel-item | 1 |"),
            std::string::npos);
  EXPECT_NE(
      output.find("| analysis | test-worker | sharded-parallel-item | 1024 |"),
      std::string::npos);
  EXPECT_NE(output.find("compile-timing-summary-end transaction_wall_ms="),
            std::string::npos);
}

} // namespace
