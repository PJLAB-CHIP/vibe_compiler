//===- Compilation.cpp - Typed Wafer compiler facade --------------------===//

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetCodeGen.h"
#include "Wafer/Compiler/Testing.h"
#include "Wafer/IR/WaferDialect.h"

#include "CompilationInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
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

llvm::Expected<ExecutionConfig>
ExecutionConfig::createForSingleCard(int64_t numPartitions,
                                     RuntimeLaunchKind runtimeLaunchKind) {
  if (numPartitions != 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "num-partitions must be exactly 1 for the single-card compiler");
  return ExecutionConfig(numPartitions, kSingleCardPhysicalTileCount,
                         runtimeLaunchKind);
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
  if (executionConfig.getPhysicalTileCount() !=
          ExecutionConfig::kSingleCardPhysicalTileCount ||
      executionConfig.getRuntimeLaunchKind() != RuntimeLaunchKind::Kernel)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile compilation requires a complete-card physical Tile kernel "
        "launch");
  return CompilationOptions(/*profileInstrumentation=*/true, optimizations,
                            timing);
}

mlir::FailureOr<PhysicalTileExecutables>
compileProgram(CompilationRequest request,
               llvm::StringRef outputProgramDirectory,
               llvm::StringRef xlaSpmdPartitionerHelper,
               const TargetToolchain &targetToolchain,
               CompilationOptions options, llvm::raw_ostream &diagnostics) {
  std::optional<PhysicalTileExecutables> retainedPhysicalTileExecutables;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics, options, std::nullopt, std::nullopt,
          std::nullopt, &retainedPhysicalTileExecutables, nullptr)))
    return mlir::failure();
  if (!retainedPhysicalTileExecutables) {
    detail::reject(
        diagnostics,
        "successful compilation did not retain physical Tile executables");
    return mlir::failure();
  }
  return std::move(*retainedPhysicalTileExecutables);
}

mlir::FailureOr<CompiledProgram> compileProgramWithTargetLLVMModules(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics) {
  std::optional<PhysicalTileExecutables> retainedPhysicalTileExecutables;
  std::optional<TargetLLVMModules> retainedTargetLLVMModules;
  std::optional<CompilationIRTrace> retainedIRTrace;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics, options, std::nullopt, std::nullopt,
          std::nullopt, &retainedPhysicalTileExecutables,
          &retainedTargetLLVMModules, &retainedIRTrace)))
    return mlir::failure();
  if (!retainedPhysicalTileExecutables || !retainedTargetLLVMModules ||
      !retainedIRTrace) {
    detail::reject(diagnostics,
                   "successful compilation did not retain the complete "
                   "compiled program");
    return mlir::failure();
  }
  return CompiledProgram(std::move(*retainedPhysicalTileExecutables),
                         std::move(*retainedTargetLLVMModules),
                         std::move(*retainedIRTrace));
}

mlir::LogicalResult testing::compileProgramWithExecutableLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLaunchSlot < 0 ||
      failAfterLaunchSlot >=
          request.getExecutionConfig().getPhysicalTileCount()) {
    detail::reject(diagnostics, "test-only executable launch slot is outside "
                                "ExecutionConfig");
    return mlir::failure();
  }
  return detail::runCompilationTransaction(
      std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, diagnostics, CompilationOptions::standard(),
      failAfterLaunchSlot, std::nullopt, std::nullopt, nullptr, nullptr);
}

mlir::LogicalResult testing::compileProgramWithTargetLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLaunchSlot < 0 ||
      failAfterLaunchSlot >=
          request.getExecutionConfig().getPhysicalTileCount()) {
    detail::reject(diagnostics,
                   "test-only target launch slot is outside ExecutionConfig");
    return mlir::failure();
  }
  return detail::runCompilationTransaction(
      std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, diagnostics, CompilationOptions::standard(),
      std::nullopt, failAfterLaunchSlot, std::nullopt, nullptr, nullptr);
}

mlir::LogicalResult testing::compileProgramWithPackageLaunchSlotFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLaunchSlot,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLaunchSlot < 0 ||
      failAfterLaunchSlot >=
          request.getExecutionConfig().getPhysicalTileCount()) {
    detail::reject(diagnostics,
                   "test-only package launch slot is outside ExecutionConfig");
    return mlir::failure();
  }
  return detail::runCompilationTransaction(
      std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, diagnostics, CompilationOptions::standard(),
      std::nullopt, std::nullopt, failAfterLaunchSlot, nullptr, nullptr);
}

} // namespace wafer::compiler
