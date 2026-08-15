//===- CompilationOutputTest.cpp - Output directory rename tests --------===//

#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <system_error>

namespace {

int renameCallCount = 0;

bool failSecondRename(llvm::StringRef source, llvm::StringRef destination,
                      llvm::raw_ostream &diagnostics) {
  ++renameCallCount;
  if (renameCallCount == 2)
    return true;
  return wafer::compiler::detail::renameDirectoryNoReplace(source, destination,
                                                           diagnostics);
}

TEST(CompilationOutputTest, RestoresPackageWhenProfileDirectoryRenameFails) {
  llvm::SmallString<256> prefix;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, prefix);
  llvm::sys::path::append(prefix, "wafer-profile-directory-rename");
  llvm::SmallString<256> root;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(prefix, root));
  auto cleanup = llvm::make_scope_exit(
      [&] { (void)llvm::sys::fs::remove_directories(root); });

  llvm::SmallString<256> stagedPackage(root);
  llvm::sys::path::append(stagedPackage, "staged-package");
  llvm::SmallString<256> stagedInstrumentation(root);
  llvm::sys::path::append(stagedInstrumentation, "staged-instrumentation");
  ASSERT_FALSE(llvm::sys::fs::create_directory(stagedPackage));
  ASSERT_FALSE(llvm::sys::fs::create_directory(stagedInstrumentation));

  llvm::SmallString<256> packageMember(stagedPackage);
  llvm::sys::path::append(packageMember, "manifest.json");
  std::error_code error;
  llvm::raw_fd_ostream packageOutput(packageMember, error,
                                     llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  packageOutput << "{}\n";
  packageOutput.close();
  ASSERT_FALSE(packageOutput.has_error());

  llvm::SmallString<256> instrumentationMember(stagedInstrumentation);
  llvm::sys::path::append(instrumentationMember, "activation.json");
  llvm::raw_fd_ostream instrumentationOutput(instrumentationMember, error,
                                             llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  instrumentationOutput << "{}\n";
  instrumentationOutput.close();
  ASSERT_FALSE(instrumentationOutput.has_error());

  llvm::SmallString<256> outputPackage(root);
  llvm::sys::path::append(outputPackage, "output");
  llvm::SmallString<256> outputInstrumentation(root);
  llvm::sys::path::append(outputInstrumentation, "output.profile");

  renameCallCount = 0;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  EXPECT_TRUE(
      mlir::failed(wafer::compiler::detail::renamePackageAndProfileNoReplace(
          stagedPackage, outputPackage, stagedInstrumentation,
          outputInstrumentation, diagnostics, failSecondRename)));
  diagnostics.flush();

  EXPECT_EQ(renameCallCount, 2);
  EXPECT_TRUE(wafer::compiler::detail::isDirectory(stagedPackage));
  EXPECT_TRUE(wafer::compiler::detail::isDirectory(stagedInstrumentation));
  EXPECT_FALSE(wafer::compiler::detail::pathEntryExists(outputPackage));
  EXPECT_FALSE(wafer::compiler::detail::pathEntryExists(outputInstrumentation));
  EXPECT_NE(diagnosticText.find("package was restored to staging"),
            std::string::npos);
}

} // namespace
