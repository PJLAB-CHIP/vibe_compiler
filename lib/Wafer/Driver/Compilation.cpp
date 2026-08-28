//===- Compilation.cpp - Typed Wafer compiler facade --------------------===//

#include "Wafer/Driver/Compilation.h"
#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/CompilationResult.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/TestSupport/CompilerTesting.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Package/Writer/PackageInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler {

char CompilationFailure::ID = 0;

llvm::StringRef stringifyCompilationStage(CompilationStage stage) {
  switch (stage) {
  case CompilationStage::SourceVerification:
    return "source-verification";
  case CompilationStage::SpmdPartitioning:
    return "spmd-partitioning";
  case CompilationStage::TensorProgramPreparation:
    return "tensor-program-preparation";
  case CompilationStage::ExecutableCompilation:
    return "executable-compilation";
  case CompilationStage::TargetCodeGeneration:
    return "target-code-generation";
  case CompilationStage::PackageAssembly:
    return "package-assembly";
  case CompilationStage::PackageCommit:
    return "package-commit";
  }
  llvm_unreachable("unknown compilation stage");
}

llvm::Expected<ExecutionConfig>
ExecutionConfig::createForSingleCard(int64_t numPartitions) {
  if (numPartitions != 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "num-partitions must be exactly 1 for the single-card compiler");
  return ExecutionConfig(numPartitions, kSingleCardTileCount);
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

llvm::Expected<CompilationOptions>
CompilationOptions::profile(const ExecutionConfig &executionConfig,
                            OptimizationConfig optimizations,
                            CompilationTimingMode timing) {
  if (executionConfig.getTileCount() != ExecutionConfig::kSingleCardTileCount)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile compilation requires a complete-card Tile kernel "
        "launch");
  return CompilationOptions(/*profileInstrumentation=*/true, optimizations,
                            timing);
}

static llvm::Expected<CompilationResult>
compileProgramImpl(CompilationRequest request, llvm::StringRef outputDirectory,
                   llvm::StringRef xlaSpmdPartitionerHelper,
                   const TargetToolchain &targetToolchain,
                   CompilationOptions options, llvm::raw_ostream &diagnostics,
                   std::optional<int64_t> failAfterLaunchSlot,
                   std::optional<int64_t> failAfterTargetLaunchSlot,
                   std::optional<int64_t> failAfterPackageLaunchSlot,
                   detail::CommitFailureInjection commitFailureInjection) {
  const ExecutionConfig executionConfig = request.getExecutionConfig();
  const bool profileRequested = options.shouldProduceProfileInstrumentation();
  std::optional<ExecutablePackage> retainedPackage;
  std::optional<ProfileInstrumentationProduct> retainedProfileProduct;
  CompilationStage failureStage = CompilationStage::SourceVerification;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics, options, failAfterLaunchSlot,
          failAfterTargetLaunchSlot, failAfterPackageLaunchSlot, nullptr,
          nullptr, nullptr, &retainedPackage, &retainedProfileProduct,
          &failureStage, commitFailureInjection)))
    return llvm::make_error<CompilationFailure>(failureStage);
  if (!retainedPackage)
    llvm_unreachable("successful transaction did not retain its package");
  if (profileRequested && !retainedProfileProduct)
    llvm_unreachable(
        "successful transaction did not retain profile instrumentation");
  return CompilationResultBuilder::make(executionConfig,
                                        std::move(*retainedPackage),
                                        std::move(retainedProfileProduct));
}

llvm::Expected<CompilationResult>
compileProgram(CompilationRequest request, llvm::StringRef outputDirectory,
               llvm::StringRef xlaSpmdPartitionerHelper,
               const TargetToolchain &targetToolchain,
               CompilationOptions options, llvm::raw_ostream &diagnostics) {
  return compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, std::move(options), diagnostics, std::nullopt,
      std::nullopt, std::nullopt, detail::CommitFailureInjection::None);
}

llvm::Expected<CompiledProgram> compileProgramWithTargetLLVMModules(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  if (options.shouldProduceProfileInstrumentation())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "the internal qualification entry does not support profile "
        "instrumentation; use compileProgram for the production entry");
  std::optional<DeviceExecutable> retainedDeviceExecutable;
  std::optional<TargetLLVMModules> retainedTargetLLVMModules;
  std::optional<CompilationIRTrace> retainedIRTrace;
  CompilationStage failureStage = CompilationStage::SourceVerification;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics, options, std::nullopt, std::nullopt,
          std::nullopt, &retainedDeviceExecutable, &retainedTargetLLVMModules,
          &retainedIRTrace, nullptr, nullptr, &failureStage)))
    return llvm::make_error<CompilationFailure>(failureStage);
  if (!retainedDeviceExecutable || !retainedTargetLLVMModules ||
      !retainedIRTrace)
    llvm_unreachable("successful transaction did not retain compiled program");
  return CompiledProgram(std::move(*retainedDeviceExecutable),
                         std::move(*retainedTargetLLVMModules),
                         std::move(*retainedIRTrace));
}

llvm::Expected<CompilationResult>
testing::compileProgramWithExecutableLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLaunchSlot < 0 ||
      failAfterLaunchSlot >= request.getExecutionConfig().getTileCount()) {
    detail::reject(diagnostics, "test-only executable launch slot is outside "
                                "ExecutionConfig");
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "test-only executable launch slot is outside ExecutionConfig");
  }
  return compileProgramImpl(std::move(request), outputDirectory,
                            xlaSpmdPartitionerHelper, targetToolchain,
                            CompilationOptions::standard(), diagnostics,
                            failAfterLaunchSlot, std::nullopt, std::nullopt,
                            detail::CommitFailureInjection::None);
}

llvm::Expected<CompilationResult>
testing::compileProgramWithTargetLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLaunchSlot < 0 ||
      failAfterLaunchSlot >= request.getExecutionConfig().getTileCount()) {
    detail::reject(diagnostics,
                   "test-only target launch slot is outside ExecutionConfig");
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "test-only target launch slot is outside ExecutionConfig");
  }
  return compileProgramImpl(std::move(request), outputDirectory,
                            xlaSpmdPartitionerHelper, targetToolchain,
                            CompilationOptions::standard(), diagnostics,
                            std::nullopt, failAfterLaunchSlot, std::nullopt,
                            detail::CommitFailureInjection::None);
}

llvm::Expected<CompilationResult>
testing::compileProgramWithPackageLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLaunchSlot < 0 ||
      failAfterLaunchSlot >= request.getExecutionConfig().getTileCount()) {
    detail::reject(diagnostics,
                   "test-only package launch slot is outside ExecutionConfig");
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "test-only package launch slot is outside ExecutionConfig");
  }
  return compileProgramImpl(std::move(request), outputDirectory,
                            xlaSpmdPartitionerHelper, targetToolchain,
                            CompilationOptions::standard(), diagnostics,
                            std::nullopt, std::nullopt, failAfterLaunchSlot,
                            detail::CommitFailureInjection::None);
}

llvm::Expected<CompilationResult>
testing::compileProgramWithCommitVerificationFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  return compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, std::move(options), diagnostics, std::nullopt,
      std::nullopt, std::nullopt,
      detail::CommitFailureInjection::FailAfterVerification);
}

llvm::Expected<CompilationResult>
testing::compileProgramWithPackageBindingFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  return compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, std::move(options), diagnostics, std::nullopt,
      std::nullopt, std::nullopt,
      detail::CommitFailureInjection::CorruptPackageProgramData);
}

llvm::Expected<CompilationResult>
testing::compileProgramWithProfileBindingFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  if (!options.shouldProduceProfileInstrumentation())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "test-only profile binding failure requires profile instrumentation");
  return compileProgramImpl(std::move(request), outputDirectory,
                            xlaSpmdPartitionerHelper, targetToolchain,
                            std::move(options), diagnostics, std::nullopt,
                            std::nullopt, std::nullopt,
                            detail::CommitFailureInjection::CorruptProfilePlan);
}

} // namespace wafer::compiler
