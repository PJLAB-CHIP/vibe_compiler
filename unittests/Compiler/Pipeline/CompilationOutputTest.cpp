//===- CompilationOutputTest.cpp - Output directory rename tests --------===//

#include "Wafer/Compiler/Pipeline/CompilationInternal.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <system_error>

namespace {

TEST(CompilationOutputTest,
     SinglePublicationRenamePublishesAllOrNothing) {
  llvm::SmallString<256> prefix;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, prefix);
  llvm::sys::path::append(prefix, "wafer-single-publication-rename");
  llvm::SmallString<256> root;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(prefix, root));
  auto cleanup = llvm::make_scope_exit(
      [&] { (void)llvm::sys::fs::remove_directories(root); });

  // The common delivery root carries both products; one no-replace rename
  // publishes or publishes nothing.
  llvm::SmallString<256> delivery(root);
  llvm::sys::path::append(delivery, "delivery");
  llvm::SmallString<256> deliveryPackage(delivery);
  llvm::sys::path::append(deliveryPackage, "package");
  llvm::SmallString<256> deliveryInstrumentation(delivery);
  llvm::sys::path::append(deliveryInstrumentation, "package.profile");
  ASSERT_FALSE(llvm::sys::fs::create_directories(deliveryPackage));
  ASSERT_FALSE(llvm::sys::fs::create_directories(deliveryInstrumentation));

  llvm::SmallString<256> output(root);
  llvm::sys::path::append(output, "output");
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);

  // A competing writer occupies the final name: the rename must fail and
  // leave both products staged and the competitor untouched.
  ASSERT_FALSE(llvm::sys::fs::create_directories(output));
  {
    llvm::SmallString<256> competitorMember(output);
    llvm::sys::path::append(competitorMember, "sentinel");
    std::error_code error;
    llvm::raw_fd_ostream competitor(competitorMember, error,
                                    llvm::sys::fs::OF_Text);
    ASSERT_FALSE(error);
    competitor << "competitor\n";
    competitor.close();
    ASSERT_FALSE(competitor.has_error());
  }
  EXPECT_TRUE(wafer::compiler::detail::renameDirectoryNoReplace(
      delivery, output, diagnostics));
  diagnostics.flush();
  EXPECT_NE(diagnosticText.find("appeared before rename"), std::string::npos);
  EXPECT_TRUE(wafer::compiler::detail::isDirectory(deliveryPackage));
  EXPECT_TRUE(wafer::compiler::detail::isDirectory(deliveryInstrumentation));
  ASSERT_FALSE(llvm::sys::fs::remove_directories(output));

  // Without a competing writer the single rename publishes the whole
  // delivery root and the staged name disappears.
  EXPECT_FALSE(wafer::compiler::detail::renameDirectoryNoReplace(
      delivery, output, diagnostics));
  EXPECT_TRUE(wafer::compiler::detail::isDirectory(output));
  EXPECT_TRUE(wafer::compiler::detail::isDirectory(
      [&] {
        llvm::SmallString<256> publishedPackage(output);
        llvm::sys::path::append(publishedPackage, "package");
        return publishedPackage;
      }()));
  EXPECT_FALSE(wafer::compiler::detail::pathEntryExists(delivery));
}

} // namespace
