//===- CompilationTest.cpp - Typed compiler request tests ----------------===//

#include "Wafer/Compiler/Compilation.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace {

TEST(CompilationTest, ExecutionConfigAcceptsOnlyCurrentSingleCardDomains) {
  for (int64_t accepted : {int64_t{1}, int64_t{16}}) {
    auto config =
        wafer::compiler::ExecutionConfig::createForSingleCard(accepted);
    ASSERT_TRUE(static_cast<bool>(config));
    EXPECT_EQ(config->getRankCount(), accepted);
  }

  for (int64_t rejected : {std::numeric_limits<int64_t>::min(), int64_t{-1},
                           int64_t{0}, int64_t{2}, int64_t{8}, int64_t{15},
                           int64_t{17}, std::numeric_limits<int64_t>::max()}) {
    auto config =
        wafer::compiler::ExecutionConfig::createForSingleCard(rejected);
    ASSERT_FALSE(static_cast<bool>(config));
    EXPECT_FALSE(llvm::toString(config.takeError()).empty());
  }
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

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  std::string source = "/tmp/source.program";
  auto request =
      wafer::compiler::CompilationRequest::create(source, std::move(*config));
  ASSERT_TRUE(static_cast<bool>(request));
  source.assign("/tmp/changed-after-request-construction.program");
  EXPECT_EQ(request->getSourceProgramDirectory(), "/tmp/source.program");
  EXPECT_EQ(request->getExecutionConfig().getRankCount(), 1);
}

TEST(CompilationTest, CompilationRequestRejectsEmptySourceLocator) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  auto request = wafer::compiler::CompilationRequest::create("", *config);
  ASSERT_FALSE(static_cast<bool>(request));
  EXPECT_FALSE(llvm::toString(request.takeError()).empty());
}

} // namespace
