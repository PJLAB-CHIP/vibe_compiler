//===- CompilerTesting.cpp - Compiler test-only adapters ----------------===//

#include "Wafer/TestSupport/CompilerTesting.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/ProgramResourceVerification.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>

namespace wafer::compiler::testing {

mlir::FailureOr<TransportContract>
bindDirectDTETransport(llvm::ArrayRef<mlir::ModuleOp> tileModules) {
  return detail::bindDirectDTETransport(tileModules);
}

mlir::FailureOr<analysis::InstructionProgramAggregateCost>
verifyProgramResources(llvm::ArrayRef<mlir::ModuleOp> tileModules,
                       llvm::ArrayRef<TileId> tileIds,
                       const ExecutionConfig &executionConfig) {
  return detail::verifyProgramResources(tileModules, tileIds, executionConfig);
}

llvm::Expected<CompilationResult> compileProgramWithExecutableLaunchSlotFailure(
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
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, CompilationOptions::standard(), diagnostics,
      failAfterLaunchSlot, std::nullopt, std::nullopt,
      detail::CommitFailureInjection::None);
}

llvm::Expected<CompilationResult> compileProgramWithTargetLaunchSlotFailure(
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
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, CompilationOptions::standard(), diagnostics,
      std::nullopt, failAfterLaunchSlot, std::nullopt,
      detail::CommitFailureInjection::None);
}

llvm::Expected<CompilationResult> compileProgramWithPackageLaunchSlotFailure(
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
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, CompilationOptions::standard(), diagnostics,
      std::nullopt, std::nullopt, failAfterLaunchSlot,
      detail::CommitFailureInjection::None);
}

llvm::Expected<CompilationResult> compileProgramWithCommitVerificationFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, std::move(options), diagnostics, std::nullopt,
      std::nullopt, std::nullopt,
      detail::CommitFailureInjection::FailAfterVerification);
}

llvm::Expected<CompilationResult> compileProgramWithPackageBindingFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, std::move(options), diagnostics, std::nullopt,
      std::nullopt, std::nullopt,
      detail::CommitFailureInjection::CorruptPackageProgramData);
}

llvm::Expected<CompilationResult> compileProgramWithProfileBindingFailure(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  if (!options.shouldProduceProfileInstrumentation())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "test-only profile binding failure requires profile instrumentation");
  return detail::compileProgramImpl(
      std::move(request), outputDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, std::move(options), diagnostics, std::nullopt,
      std::nullopt, std::nullopt,
      detail::CommitFailureInjection::CorruptProfilePlan);
}

} // namespace wafer::compiler::testing
