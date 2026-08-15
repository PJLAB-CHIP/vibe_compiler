//===- PhysicalTileExecutablesInternal.h - Internal construction -*- C++
//-*-===//

#ifndef WAFER_COMPILER_PHYSICALTILEEXECUTABLESINTERNAL_H
#define WAFER_COMPILER_PHYSICALTILEEXECUTABLESINTERNAL_H

#include "Wafer/Compiler/Compilation.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler {

/// Internal constructors for physical Tile executables.
struct PhysicalTileExecutablesBuilder {
  static PhysicalTileExecutable
  makePhysicalTile(PhysicalCardId physicalCardId, PhysicalTileId physicalTileId,
                   LaunchSlotId launchSlotId,
                   mlir::OwningOpRef<mlir::ModuleOp> module,
                   llvm::StringRef entrySymbol,
                   std::vector<ProgramResourceBinding> programBindings,
                   TransportContract transportContract) {
    return PhysicalTileExecutable(
        physicalCardId, physicalTileId, launchSlotId, std::move(module),
        entrySymbol, std::move(programBindings), transportContract);
  }

  static PhysicalTileExecutables
  makeExecutables(ExecutionConfig executionConfig,
                  RuntimeLaunchContract runtimeLaunchContract,
                  std::shared_ptr<mlir::MLIRContext> context,
                  std::vector<PhysicalTileExecutable> physicalTiles) {
    return PhysicalTileExecutables(
        executionConfig, std::move(runtimeLaunchContract), std::move(context),
        std::move(physicalTiles));
  }
};

namespace detail {

mlir::LogicalResult
verifyExactExecutionConfig(mlir::ModuleOp module,
                           const ExecutionConfig &executionConfig);

llvm::Expected<PhysicalTileExecutables> buildPhysicalTileExecutables(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot);

llvm::Expected<PhysicalTileExecutables> buildPhysicalTileExecutablesWithIRTrace(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    CompilationIRTrace &irTrace);

} // namespace detail
} // namespace wafer::compiler

#endif // WAFER_COMPILER_PHYSICALTILEEXECUTABLESINTERNAL_H
