//===- DeviceExecutable.cpp - Device executable ownership ----------------===//

#include "Wafer/CodeGen/DeviceExecutableInternal.h"

#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer::compiler {

llvm::Expected<ExecutionConfig>
ExecutionConfig::createForSingleCard(int64_t numPartitions) {
  if (numPartitions != 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "num-partitions must be exactly 1 for the single-card compiler");
  return ExecutionConfig(numPartitions, kSingleCardTileCount);
}

DeviceExecutable::DeviceExecutable(DeviceExecutable &&) = default;
DeviceExecutable &DeviceExecutable::operator=(DeviceExecutable &&) = default;
DeviceExecutable::~DeviceExecutable() = default;

DeviceExecutable::DeviceExecutable(
    ExecutionConfig executionConfig,
    RuntimeLaunchContract runtimeLaunchContract,
    std::shared_ptr<mlir::MLIRContext> context,
    std::vector<TileExecutable> tiles,
    std::unique_ptr<ProgramDataHandoff> programData)
    : executionConfig(executionConfig),
      runtimeLaunchContract(std::move(runtimeLaunchContract)),
      context(std::move(context)), tiles(std::move(tiles)),
      programData(std::move(programData)) {}

const ProgramDataHandoff &DeviceExecutable::getProgramDataHandoff() const {
  return *programData;
}

static llvm::Expected<DeviceExecutable>
compileTensorProgramModuleToDeviceExecutable(
    std::shared_ptr<mlir::MLIRContext> &, mlir::ModuleOp,
    const frontend::FrontendProgramVerificationResult &,
    const ExecutionConfig &, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t>,
    ProgramDataHandoff &, CompilationIRTrace &, bool) {
  if (!optimizations.isNone() && !optimizations.isSearch())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "device executable compilation requires an optimization policy");

  const llvm::StringRef policy = optimizations.isNone() ? "none" : "search";
  diagnostics << "wafer-compile: operation_not_supported: optimization-policy="
              << policy
              << " is unavailable while the current-IR executable pipeline is "
                 "being rebuilt\n";
  return llvm::createStringError(
      std::errc::operation_not_supported,
      "operation_not_supported: optimization-policy=%s is unavailable while "
      "the current-IR executable pipeline is being rebuilt",
      policy.str().c_str());
}

static llvm::Expected<DeviceExecutable> buildDeviceExecutableImpl(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace,
    bool requestTileIRTrace) {
  wafer::support::ScopedCompileTimingSpan deviceExecutableTiming(
      "stage", "tensor-program-to-executable", "device-executable");
  return compileTensorProgramModuleToDeviceExecutable(
      context, tensorModule, program, executionConfig, optimizations,
      diagnostics, failAfterLaunchSlot, programData, irTrace,
      requestTileIRTrace);
}

llvm::Expected<DeviceExecutable> detail::buildDeviceExecutable(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData) {
  CompilationIRTrace discardedTrace;
  return buildDeviceExecutableImpl(
      context, tensorModule, std::move(program), executionConfig, optimizations,
      diagnostics, failAfterLaunchSlot, programData, discardedTrace,
      /*requestTileIRTrace=*/false);
}

llvm::Expected<DeviceExecutable> detail::buildDeviceExecutableWithIRTrace(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace) {
  return buildDeviceExecutableImpl(context, tensorModule, std::move(program),
                                   executionConfig, optimizations, diagnostics,
                                   failAfterLaunchSlot, programData, irTrace,
                                   /*requestTileIRTrace=*/true);
}

} // namespace wafer::compiler
