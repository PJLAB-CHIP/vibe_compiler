//===- CompilationTest.cpp - Typed compiler request tests ----------------===//

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/Package.h"
#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

namespace {

TEST(CompilationTest, ExecutionConfigAcceptsOnlyCurrentSingleCardDomains) {
  for (int64_t accepted : {int64_t{1}, int64_t{16}}) {
    auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
        accepted, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
        wafer::RuntimeLaunchKind::Kernel);
    ASSERT_TRUE(static_cast<bool>(config));
    EXPECT_EQ(config->getRankCount(), accepted);
  }

  for (int64_t rejected : {std::numeric_limits<int64_t>::min(), int64_t{-1},
                           int64_t{0}, int64_t{2}, int64_t{8}, int64_t{15},
                           int64_t{17}, std::numeric_limits<int64_t>::max()}) {
    auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
        rejected, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
        wafer::RuntimeLaunchKind::Kernel);
    ASSERT_FALSE(static_cast<bool>(config));
    EXPECT_FALSE(llvm::toString(config.takeError()).empty());
  }
}

TEST(CompilationTest, ExecutionConfigEqualityCoversRankProfileAndLaunchKind) {
  wafer::TargetProfileId profile =
      wafer::TargetProfileId::waferTx81SingleCardKernelV1();
  auto first = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, profile, wafer::RuntimeLaunchKind::Kernel);
  auto same = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, profile, wafer::RuntimeLaunchKind::Kernel);
  auto differentRank = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, profile, wafer::RuntimeLaunchKind::Kernel);
  auto differentLaunchKind =
      wafer::compiler::ExecutionConfig::createForSingleCard(
          16, profile, wafer::RuntimeLaunchKind::Model);
  ASSERT_TRUE(static_cast<bool>(first));
  ASSERT_TRUE(static_cast<bool>(same));
  ASSERT_TRUE(static_cast<bool>(differentRank));
  ASSERT_TRUE(static_cast<bool>(differentLaunchKind));
  EXPECT_EQ(*first, *same);
  EXPECT_NE(*first, *differentRank);
  EXPECT_NE(*differentRank, *differentLaunchKind);

  auto invalidModel = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, profile, wafer::RuntimeLaunchKind::Model);
  ASSERT_FALSE(static_cast<bool>(invalidModel));
  EXPECT_NE(llvm::toString(invalidModel.takeError())
                .find("model runtime launch requires execution-ranks=16"),
            std::string::npos);

  auto unqualifiedModel = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV2(),
      wafer::RuntimeLaunchKind::Model);
  ASSERT_FALSE(static_cast<bool>(unqualifiedModel));
  EXPECT_NE(llvm::toString(unqualifiedModel.takeError()).find("not qualified"),
            std::string::npos);
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
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::RankExecutable>);
  static_assert(!std::is_copy_constructible_v<wafer::compiler::RankExecutable>);
  static_assert(std::is_move_constructible_v<wafer::compiler::RankExecutable>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::ExecutableBundle>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::ExecutableBundle>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::ExecutableBundle>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::TargetLLVMModule>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::TargetLLVMModule>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::TargetLLVMModule>);
  static_assert(!std::is_default_constructible_v<
                wafer::compiler::TargetLLVMModuleBundle>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::TargetLLVMModuleBundle>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::TargetLLVMModuleBundle>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::TargetArtifactBundle>);
  static_assert(
      !std::is_copy_constructible_v<wafer::compiler::TargetArtifactBundle>);
  static_assert(
      std::is_move_constructible_v<wafer::compiler::TargetArtifactBundle>);
  static_assert(
      !std::is_default_constructible_v<wafer::compiler::PackageBundle>);
  static_assert(!std::is_copy_constructible_v<wafer::compiler::PackageBundle>);
  static_assert(std::is_move_constructible_v<wafer::compiler::PackageBundle>);

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config));
  std::string source = "/tmp/source.program";
  auto request =
      wafer::compiler::CompilationRequest::create(source, std::move(*config));
  ASSERT_TRUE(static_cast<bool>(request));
  source.assign("/tmp/changed-after-request-construction.program");
  EXPECT_EQ(request->getSourceProgramDirectory(), "/tmp/source.program");
  EXPECT_EQ(request->getExecutionConfig().getRankCount(), 1);
  EXPECT_EQ(request->getExecutionConfig().getTargetProfileId(),
            wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  EXPECT_EQ(request->getExecutionConfig().getRuntimeLaunchKind(),
            wafer::RuntimeLaunchKind::Kernel);
}

TEST(CompilationTest, CompilationRequestRejectsEmptySourceLocator) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(config));
  auto request = wafer::compiler::CompilationRequest::create("", *config);
  ASSERT_FALSE(static_cast<bool>(request));
  EXPECT_FALSE(llvm::toString(request.takeError()).empty());
}

TEST(CompilationTest, ProfileOptionsRequireCompleteSingleCardRankDomain) {
  auto rankOne = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(rankOne));

  auto rejected = wafer::compiler::CompilationOptions::profile(*rankOne);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("16-rank kernel launch"),
            std::string::npos);

  auto fullCard = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(fullCard));
  auto accepted = wafer::compiler::CompilationOptions::profile(*fullCard);
  ASSERT_TRUE(static_cast<bool>(accepted));
  EXPECT_TRUE(accepted->shouldProduceProfileCompanion());
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
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::RuntimeLaunchKind::Model);
  ASSERT_TRUE(static_cast<bool>(model));
  auto rejectedModel = wafer::compiler::CompilationOptions::profile(*model);
  ASSERT_FALSE(static_cast<bool>(rejectedModel));
  EXPECT_NE(
      llvm::toString(rejectedModel.takeError()).find("16-rank kernel launch"),
      std::string::npos);
  EXPECT_FALSE(wafer::compiler::CompilationOptions::standard()
                   .shouldProduceProfileCompanion());
  EXPECT_FALSE(wafer::compiler::CompilationOptions::standard()
                   .shouldReportDetailedTiming());
  EXPECT_TRUE(wafer::compiler::CompilationOptions::standard(
                  wafer::OptimizationConfig::production(),
                  wafer::compiler::CompilationTimingMode::Detailed)
                  .shouldReportDetailedTiming());
}

TEST(CompilationTest, OptimizationKindsHaveStableUniqueRoundTripNames) {
  std::set<std::string> names;
  EXPECT_EQ(wafer::getSupportedOptimizationKinds().size(),
            static_cast<size_t>(wafer::OptimizationKind::Count));
  for (wafer::OptimizationKind kind : wafer::getSupportedOptimizationKinds()) {
    llvm::StringRef name = wafer::stringifyOptimizationKind(kind);
    EXPECT_FALSE(name.empty());
    EXPECT_TRUE(names.insert(name.str()).second);
    EXPECT_EQ(wafer::parseOptimizationKind(name), kind);
  }
  EXPECT_FALSE(wafer::parseOptimizationKind("not-an-optimization"));
}

TEST(CompilationTest, OptimizationConfigSupportsPresetsAndComposition) {
  wafer::OptimizationConfig production =
      wafer::OptimizationConfig::production();
  wafer::OptimizationConfig none = wafer::OptimizationConfig::none();
  for (wafer::OptimizationKind kind : wafer::getSupportedOptimizationKinds()) {
    EXPECT_TRUE(production.isEnabled(kind));
    EXPECT_FALSE(none.isEnabled(kind));
  }

  none.enable(wafer::OptimizationKind::FullBufferResidency);
  none.enable(wafer::OptimizationKind::ReadyOrderScheduling);
  EXPECT_TRUE(none.isEnabled(wafer::OptimizationKind::FullBufferResidency));
  EXPECT_TRUE(none.isEnabled(wafer::OptimizationKind::ReadyOrderScheduling));
  none.disable(wafer::OptimizationKind::FullBufferResidency);
  EXPECT_FALSE(none.isEnabled(wafer::OptimizationKind::FullBufferResidency));

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
  }
  session->finishAndPrintSummary();
  EXPECT_NE(output.find("compile-timing-summary-begin"), std::string::npos);
  EXPECT_NE(output.find("| stage | test-pipeline | test-item | 1 |"),
            std::string::npos);
  EXPECT_NE(output.find("compile-timing-summary-end transaction_wall_ms="),
            std::string::npos);
}

} // namespace
