//===- Compilation.cpp - Typed Wafer compiler facade --------------------===//

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Compiler/Testing.h"
#include "Wafer/Support/OptimizationQualificationArchiveStore.h"

#include "CompilationInternal.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <optional>
#include <utility>

#ifndef WAFER_OPTIMIZATION_QUALIFICATION_ARCHIVE_ROOT
#define WAFER_OPTIMIZATION_QUALIFICATION_ARCHIVE_ROOT ""
#endif

namespace wafer::compiler {
namespace {

std::optional<detail::CompilationOptimizationPolicyV1>
getActiveProductionPolicy(llvm::raw_ostream &diagnostics) {
  static std::once_flag once;
  static std::optional<detail::CompilationOptimizationPolicyV1> policy;
  static std::string failure;
  std::call_once(once, [] {
    OptimizationQualificationArchiveStoreV1 store(
        WAFER_OPTIMIZATION_QUALIFICATION_ARCHIVE_ROOT);
    ActiveQualifiedOptimizationPolicySelectionV1 selection;
    if (!store.loadActiveQualifiedOptimizationPolicy(selection, &failure))
      return;
    OptimizationQualificationProposal proposal =
        getCurrentOptimizationQualificationProposal();
    if (!validateOptimizationQualificationProposal(proposal, &failure))
      return;
    if (selection.qualifiedSet.proposalDigest !=
        digestOptimizationQualificationProposalV1(proposal)) {
      failure =
          "active qualified optimization set does not bind current proposal";
      return;
    }
    policy = detail::CompilationOptimizationPolicyV1{
        std::move(proposal), getAllOnOptimizationConfiguration(),
        EquivalentInputVariantV1::Original,
        /*requiredTensorNormalizationRepetitions=*/1};
  });
  if (!policy) {
    detail::reject(diagnostics,
                   "active optimization qualification is unavailable: " +
                       failure);
    return std::nullopt;
  }
  return policy;
}

} // namespace

llvm::Expected<ExecutionConfig>
ExecutionConfig::createForSingleCard(int64_t executionRankCount,
                                     TargetProfileId targetProfile) {
  if (executionRankCount != 1 && executionRankCount != 16)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "execution-ranks must be exactly 1 or 16 for the single-card compiler");
  return ExecutionConfig(executionRankCount, targetProfile);
}

llvm::Expected<CompilationRequest>
CompilationRequest::create(llvm::StringRef sourceProgramDirectory,
                           ExecutionConfig executionConfig) {
  if (sourceProgramDirectory.empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "source program directory must not be empty");
  return CompilationRequest(sourceProgramDirectory, executionConfig);
}

llvm::Expected<ExecutableBundle>
compileTensorProgramToExecutableBundle(llvm::StringRef tensorProgramDirectory,
                                       ExecutionConfig executionConfig,
                                       llvm::raw_ostream &diagnostics) {
  auto policy = getActiveProductionPolicy(diagnostics);
  if (!policy)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "active optimization qualification is unavailable");
  return detail::compileTensorProgramToExecutableBundleImpl(
      tensorProgramDirectory, executionConfig, diagnostics, std::nullopt,
      *policy);
}

mlir::FailureOr<ExecutableBundle> compileProgram(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  auto policy = getActiveProductionPolicy(diagnostics);
  if (!policy)
    return mlir::failure();
  std::optional<ExecutableBundle> retainedExecutableBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics, std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle, nullptr, *policy)))
    return mlir::failure();
  if (!retainedExecutableBundle) {
    detail::reject(
        diagnostics,
        "successful compilation did not retain its executable bundle");
    return mlir::failure();
  }
  return std::move(*retainedExecutableBundle);
}

mlir::FailureOr<TargetCompilationProduct> compileProgramWithTargetLLVMBundle(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  auto policy = getActiveProductionPolicy(diagnostics);
  if (!policy)
    return mlir::failure();
  std::optional<ExecutableBundle> retainedExecutableBundle;
  std::optional<TargetLLVMModuleBundle> retainedTargetLLVMModuleBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics, std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle,
          &retainedTargetLLVMModuleBundle, *policy)))
    return mlir::failure();
  if (!retainedExecutableBundle || !retainedTargetLLVMModuleBundle) {
    detail::reject(diagnostics,
                   "successful compilation did not retain its complete target "
                   "compilation product");
    return mlir::failure();
  }
  return TargetCompilationProduct(std::move(*retainedExecutableBundle),
                                  std::move(*retainedTargetLLVMModuleBundle));
}

mlir::LogicalResult testing::compileProgramWithRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLogicalRank < 0 ||
      failAfterLogicalRank >= request.getExecutionConfig().getRankCount()) {
    detail::reject(diagnostics,
                   "test-only failure rank is outside ExecutionConfig");
    return mlir::failure();
  }
  auto policy = getActiveProductionPolicy(diagnostics);
  if (!policy)
    return mlir::failure();
  return detail::runCompilationTransaction(
      std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, diagnostics, failAfterLogicalRank, std::nullopt,
      std::nullopt, nullptr, nullptr, *policy);
}

mlir::LogicalResult testing::compileProgramWithTargetRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLogicalRank < 0 ||
      failAfterLogicalRank >= request.getExecutionConfig().getRankCount()) {
    detail::reject(diagnostics,
                   "test-only target failure rank is outside ExecutionConfig");
    return mlir::failure();
  }
  auto policy = getActiveProductionPolicy(diagnostics);
  if (!policy)
    return mlir::failure();
  return detail::runCompilationTransaction(
      std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, diagnostics, std::nullopt, failAfterLogicalRank,
      std::nullopt, nullptr, nullptr, *policy);
}

mlir::LogicalResult testing::compileProgramWithPackageRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLogicalRank < 0 ||
      failAfterLogicalRank >= request.getExecutionConfig().getRankCount()) {
    detail::reject(diagnostics,
                   "test-only package failure rank is outside ExecutionConfig");
    return mlir::failure();
  }
  auto policy = getActiveProductionPolicy(diagnostics);
  if (!policy)
    return mlir::failure();
  return detail::runCompilationTransaction(
      std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, diagnostics, std::nullopt, std::nullopt,
      failAfterLogicalRank, nullptr, nullptr, *policy);
}

} // namespace wafer::compiler
