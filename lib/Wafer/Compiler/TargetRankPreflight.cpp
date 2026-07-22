//===- TargetRankPreflight.cpp - All-rank target LLVM preflight ----------===//

#include "TargetArtifactInternal.h"

#include "Wafer/Support/TargetPolicy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

llvm::Expected<TargetLLVMModuleBundle>
compileExecutableBundleToTargetLLVMModulesImpl(
    const ExecutableBundle &executableBundle, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank) {
  const std::vector<RankExecutable> &ranks =
      executableBundle.getRankExecutables();
  const ExecutionConfig &executionConfig =
      executableBundle.getExecutionConfig();
  if (ranks.size() != static_cast<size_t>(executionConfig.getRankCount()))
    return fail(diagnostics, "target LLVM rank domain is incomplete");

  const TargetProfileRecord &targetProfile =
      getTargetProfileRecord(executionConfig.getTargetProfileId());
  std::vector<PreparedTargetRank> preparedRanks;
  preparedRanks.reserve(ranks.size());
  for (auto [expectedRank, rank] : llvm::enumerate(ranks)) {
    if (rank.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return fail(diagnostics, "target LLVM rank domain is not canonical");
    mlir::FailureOr<PreparedTargetRank> prepared =
        prepareTargetABI(rank, executionConfig);
    if (mlir::failed(prepared))
      return fail(diagnostics,
                  "target ABI preparation failed for logical rank " +
                      std::to_string(expectedRank));
    if (mlir::failed(lowerToTargetLLVM(*prepared)))
      return fail(diagnostics, "target lowering failed for logical rank " +
                                   std::to_string(expectedRank));
    if (mlir::failed(verifyLoweredKernelABI(*prepared, rank.getEntrySymbol())))
      return fail(diagnostics,
                  "target ABI verification failed for logical rank " +
                      std::to_string(expectedRank));
    if (prepared->logicalRank != static_cast<int64_t>(expectedRank) ||
        prepared->targetProfile != targetProfile.id ||
        prepared->launchABI != executionConfig.getTargetLaunchABIId() ||
        prepared->targetIdentity != targetProfile.targetIdentity ||
        prepared->kernelRuntimeABI != targetProfile.kernelRuntimeABI ||
        prepared->moduleFormat != targetProfile.moduleFormat)
      return fail(diagnostics,
                  "prepared target profile readback failed for logical rank " +
                      std::to_string(expectedRank));
    preparedRanks.push_back(std::move(*prepared));
  }

  std::vector<TargetLLVMModule> modules;
  modules.reserve(ranks.size());
  for (auto [expectedRank, prepared] : llvm::enumerate(preparedRanks)) {
    llvm::Expected<TargetLLVMModule> translated = translatePreparedTargetRank(
        std::move(prepared), ranks[expectedRank].getEntrySymbol());
    if (!translated)
      return fail(diagnostics,
                  "target LLVM translation/readback failed for logical rank " +
                      std::to_string(expectedRank) + ": " +
                      llvm::toString(translated.takeError()));
    modules.push_back(std::move(*translated));
    if (failAfterLogicalRank &&
        static_cast<int64_t>(expectedRank) == *failAfterLogicalRank)
      return fail(diagnostics,
                  "test-only injected target failure after logical rank " +
                      std::to_string(expectedRank));
  }

  return TargetLLVMModuleBundleBuilder::makeBundle(executionConfig,
                                                   std::move(modules));
}

mlir::LogicalResult
lowerTargetABIForTesting(const RankExecutable &rankExecutable,
                         const ExecutionConfig &executionConfig) {
  mlir::FailureOr<PreparedTargetRank> prepared =
      prepareTargetABI(rankExecutable, executionConfig);
  if (mlir::failed(prepared) || mlir::failed(lowerToTargetLLVM(*prepared)))
    return mlir::failure();
  return verifyLoweredKernelABI(*prepared, rankExecutable.getEntrySymbol());
}

llvm::Error
verifyTargetLLVMModuleForTesting(const TargetLLVMModule &targetModule) {
  return verifyTargetLLVMModule(
      targetModule.getModule(), targetModule.getLogicalRank(),
      targetModule.getEntrySymbol(), targetModule.getTargetProfileId(),
      targetModule.getTargetIdentityId(), targetModule.getKernelRuntimeABIId(),
      targetModule.getModuleFormat(), targetModule.getKernelABISlots());
}

} // namespace wafer::compiler::detail
