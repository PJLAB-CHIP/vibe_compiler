//===- CompilationPublicationTest.cpp - Publication transaction tests ---===//

#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <system_error>

namespace {

int publicationCallCount = 0;

bool failSecondPublication(llvm::StringRef source,
                           llvm::StringRef destination,
                           llvm::raw_ostream &diagnostics) {
  ++publicationCallCount;
  if (publicationCallCount == 2)
    return true;
  return wafer::compiler::detail::publishDirectoryNoReplace(
      source, destination, diagnostics);
}

TEST(CompilationPublicationTest,
     RollsPackageBackWhenCompanionPublicationFails) {
  llvm::SmallString<256> prefix;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, prefix);
  llvm::sys::path::append(prefix, "wafer-companion-publication");
  llvm::SmallString<256> root;
  ASSERT_FALSE(
      llvm::sys::fs::createUniqueDirectory(prefix, root));
  auto cleanup = llvm::make_scope_exit(
      [&] { (void)llvm::sys::fs::remove_directories(root); });

  llvm::SmallString<256> stagedPackage(root);
  llvm::sys::path::append(stagedPackage, "staged-package");
  llvm::SmallString<256> stagedCompanion(root);
  llvm::sys::path::append(stagedCompanion, "staged-companion");
  ASSERT_FALSE(llvm::sys::fs::create_directory(stagedPackage));
  ASSERT_FALSE(llvm::sys::fs::create_directory(stagedCompanion));

  llvm::SmallString<256> packageMember(stagedPackage);
  llvm::sys::path::append(packageMember, "manifest.json");
  std::error_code error;
  llvm::raw_fd_ostream packageOutput(packageMember, error,
                                    llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  packageOutput << "{}\n";
  packageOutput.close();
  ASSERT_FALSE(packageOutput.has_error());

  llvm::SmallString<256> companionMember(stagedCompanion);
  llvm::sys::path::append(companionMember, "activation.json");
  llvm::raw_fd_ostream companionOutput(companionMember, error,
                                      llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  companionOutput << "{}\n";
  companionOutput.close();
  ASSERT_FALSE(companionOutput.has_error());

  llvm::SmallString<256> outputPackage(root);
  llvm::sys::path::append(outputPackage, "published");
  llvm::SmallString<256> outputCompanion(root);
  llvm::sys::path::append(outputCompanion, "published.qualification");

  publicationCallCount = 0;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::publishPackageAndCompanionNoReplace(
          stagedPackage, outputPackage, stagedCompanion, outputCompanion,
          diagnostics, failSecondPublication)));
  diagnostics.flush();

  EXPECT_EQ(publicationCallCount, 2);
  EXPECT_TRUE(wafer::compiler::detail::isDirectory(stagedPackage));
  EXPECT_TRUE(wafer::compiler::detail::isDirectory(stagedCompanion));
  EXPECT_FALSE(wafer::compiler::detail::pathEntryExists(outputPackage));
  EXPECT_FALSE(wafer::compiler::detail::pathEntryExists(outputCompanion));
  EXPECT_NE(diagnosticText.find("package publication was rolled back"),
            std::string::npos);
}

} // namespace
