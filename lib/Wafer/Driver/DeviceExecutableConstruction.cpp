//===- DeviceExecutableConstruction.cpp - Product policy routing -------===//

#include "Wafer/CodeGen/DeviceExecutableInternal.h"

#include "PhysicalDataflow/BaselineCurrentIR.h"
#include "PhysicalDataflow/PhysicalDataflowInstrumentation.h"
#include "PhysicalDataflow/SearchCurrentIR.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/ExternalProcess.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace wafer::compiler::detail {
namespace {

static llvm::Expected<DeviceExecutable> compileCurrentPolicy(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, ProgramDataHandoff &programData,
    CompilationIRTrace *irTrace) {
  if (!optimizations.isNone() && !optimizations.isSearch())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "device executable compilation requires an optimization policy");
  ExecutableLoweringStatistics executableStatistics;
  ExecutableCompilationResult compiled;
  if (optimizations.isSearch()) {
    SearchCurrentIROptions options;
    std::optional<SearchLimits> limits = optimizations.getSearchLimits();
    if (!limits)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "search optimization policy omitted its work limits");
    options.limits = *limits;
    options.deadline = std::chrono::steady_clock::now() +
                       std::chrono::seconds(
                           wafer::support::kExternalProcessTimeoutSeconds);
    options.downstream.captureTileDataflowIR = irTrace != nullptr;
    SearchCurrentIRStatistics searchStatistics;
    compiled = compileSearchCurrentIR(tensorModule, program, executionConfig,
                                      diagnostics, programData, options,
                                      &searchStatistics, &executableStatistics);
  } else {
    BaselineCurrentIROptions options;
    options.downstream.captureTileDataflowIR = irTrace != nullptr;
    BaselineCurrentIRStatistics baselineStatistics;
    compiled = compileBaselineCurrentIR(
        tensorModule, program, executionConfig, diagnostics, programData,
        options, &baselineStatistics, &executableStatistics);
  }
  if (!compiled.isAccepted()) {
    diagnostics << "wafer-compile: " << compiled.gate << ": " << compiled.detail
                << "\n";
    return llvm::createStringError(
        compiled.status == ExecutableCompilationStatus::IndeterminateFailure
            ? llvm::errc::resource_unavailable_try_again
            : llvm::errc::invalid_argument,
        "%s: %s", compiled.gate.c_str(), compiled.detail.c_str());
  }

  if (compiled.physicalIRInventory && compiled.executable)
    recordAcceptedPhysicalDataflowInstrumentation(
        *compiled.physicalIRInventory, compiled.executable->resourceCost);

  ExecutableLoweringResult lowered = compiled.takeExecutable();
  if (irTrace) {
    if (compiled.tileDataflowIRTrace.size() != lowered.tiles.size())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "current-IR trace does not match accepted Tile domain");
    irTrace->tiles.clear();
    irTrace->tiles.reserve(lowered.tiles.size());
    for (auto &&[tile, text] :
         llvm::zip_equal(lowered.tiles, compiled.tileDataflowIRTrace))
      irTrace->tiles.push_back(TileIRTrace{tile.getCardId(), tile.getTileId(),
                                           tile.getLaunchSlotId(), text});
  }
  programData.discardUnadoptedCandidates();
  auto ownedProgramData =
      std::make_unique<ProgramDataHandoff>(std::move(programData));
  return DeviceExecutableBuilder::makeDeviceExecutable(
      executionConfig, std::move(lowered.runtimeLaunchContract), context,
      std::move(lowered.tiles), std::move(ownedProgramData));
}

static llvm::Expected<DeviceExecutable> buildDeviceExecutableImpl(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData, CompilationIRTrace *irTrace) {
  (void)failAfterLaunchSlot;
  wafer::support::ScopedCompileTimingSpan timing(
      "stage", "tensor-program-to-executable", "device-executable");
  llvm::Expected<DeviceExecutable> result =
      compileCurrentPolicy(context, tensorModule, program, executionConfig,
                           optimizations, diagnostics, programData, irTrace);
  if (!result)
    timing.markFailed();
  return result;
}

} // namespace

llvm::Expected<DeviceExecutable> buildDeviceExecutable(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData) {
  return buildDeviceExecutableImpl(
      context, tensorModule, std::move(program), executionConfig, optimizations,
      diagnostics, failAfterLaunchSlot, programData, /*irTrace=*/nullptr);
}

llvm::Expected<DeviceExecutable> buildDeviceExecutableWithIRTrace(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace) {
  return buildDeviceExecutableImpl(context, tensorModule, std::move(program),
                                   executionConfig, optimizations, diagnostics,
                                   failAfterLaunchSlot, programData, &irTrace);
}

} // namespace wafer::compiler::detail
