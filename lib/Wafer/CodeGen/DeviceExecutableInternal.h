//===- DeviceExecutableInternal.h - Internal construction -*- C++
//-*-===//

#ifndef WAFER_CODEGEN_DEVICEEXECUTABLEINTERNAL_H
#define WAFER_CODEGEN_DEVICEEXECUTABLEINTERNAL_H

#include "Wafer/CodeGen/DeviceExecutable.h"
#include "Wafer/Driver/ProgramData/ProgramData.h"
#include "Wafer/Support/OptimizationConfig.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler {

/// Internal constructors for verified device executables.
struct DeviceExecutableBuilder {
  static TileExecutable
  makeTileExecutable(CardId cardId, TileId tileId, LaunchSlotId launchSlotId,
                     mlir::OwningOpRef<mlir::ModuleOp> module,
                     llvm::StringRef entrySymbol,
                     std::vector<ProgramResourceBinding> programBindings,
                     TransportContract transportContract) {
    return TileExecutable(cardId, tileId, launchSlotId, std::move(module),
                          entrySymbol, std::move(programBindings),
                          transportContract);
  }

  static DeviceExecutable
  makeDeviceExecutable(ExecutionConfig executionConfig,
                       RuntimeLaunchContract runtimeLaunchContract,
                       std::shared_ptr<mlir::MLIRContext> context,
                       std::vector<TileExecutable> tiles,
                       std::unique_ptr<ProgramDataHandoff> programData) {
    return DeviceExecutable(executionConfig, std::move(runtimeLaunchContract),
                            std::move(context), std::move(tiles),
                            std::move(programData));
  }
};

namespace detail {

/// Verifies the exact program/DDR binding coverage consumed by target ABI
/// preparation. Candidate admission and final target lowering call this same
/// implementation so a DeviceExecutable cannot be accepted and fail later at
/// the unchanged ABI boundary.
mlir::LogicalResult verifyProgramResourceBoundary(
    mlir::func::FuncOp function,
    llvm::ArrayRef<ProgramResourceBinding> programBindings);

mlir::LogicalResult
verifyExactExecutionConfig(mlir::ModuleOp module,
                           const ExecutionConfig &executionConfig);

llvm::Expected<DeviceExecutable> buildDeviceExecutable(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData);

llvm::Expected<DeviceExecutable> buildDeviceExecutableWithIRTrace(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace);

} // namespace detail
} // namespace wafer::compiler

#endif // WAFER_CODEGEN_DEVICEEXECUTABLEINTERNAL_H
