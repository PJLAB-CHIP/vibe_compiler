//===- TargetPackagePublication.cpp - Staged target/package build -------===//

#include "CompilationInternal.h"
#include "PackageInternal.h"
#include "TargetArtifactInternal.h"

#include "Wafer/Compiler/Package.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

#include <utility>

namespace wafer::compiler::detail {

mlir::LogicalResult stageTargetPackage(
    llvm::StringRef groupedProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    std::optional<int64_t> failAfterTargetLogicalRank,
    std::optional<int64_t> failAfterPackageLogicalRank,
    std::optional<ExecutableBundle> &executableBundle,
    std::optional<TargetLLVMModuleBundle> &targetLLVMModuleBundle) {
  llvm::Expected<ExecutableBundle> compiledExecutableBundle =
      compileGroupedProgramToExecutableBundleImpl(groupedProgramDirectory,
                                                  executionConfig, diagnostics,
                                                  failAfterLogicalRank);
  if (!compiledExecutableBundle) {
    llvm::consumeError(compiledExecutableBundle.takeError());
    return mlir::failure();
  }

  llvm::SmallString<256> stagedTargetArtifacts(transactionRoot);
  llvm::sys::path::append(stagedTargetArtifacts, "target-artifacts");
  llvm::Expected<TargetLLVMModuleBundle> targetLLVMModules =
      compileExecutableBundleToTargetLLVMModulesImpl(
          *compiledExecutableBundle, diagnostics, failAfterTargetLogicalRank);
  if (!targetLLVMModules) {
    llvm::consumeError(targetLLVMModules.takeError());
    return mlir::failure();
  }
  llvm::Expected<TargetArtifactBundle> targetArtifacts =
      compileTargetLLVMModuleBundleToTargetArtifacts(
          *targetLLVMModules, stagedTargetArtifacts, targetToolchain,
          diagnostics);
  if (!targetArtifacts) {
    llvm::consumeError(targetArtifacts.takeError());
    return mlir::failure();
  }

  llvm::SmallString<256> stagedPackage(transactionRoot);
  llvm::sys::path::append(stagedPackage, "package");
  llvm::Expected<PackageBundle> package = assemblePackageBundleImpl(
      groupedProgramDirectory, *compiledExecutableBundle, *targetArtifacts,
      stagedPackage, diagnostics, failAfterPackageLogicalRank);
  if (!package) {
    llvm::consumeError(package.takeError());
    return mlir::failure();
  }

  executableBundle.emplace(std::move(*compiledExecutableBundle));
  targetLLVMModuleBundle.emplace(std::move(*targetLLVMModules));
  return mlir::success();
}

} // namespace wafer::compiler::detail
