//===- ExecutableBundleInternal.h - Internal bundle construction -*- C++
//-*-===//

#ifndef WAFER_COMPILER_EXECUTABLEBUNDLEINTERNAL_H
#define WAFER_COMPILER_EXECUTABLEBUNDLEINTERNAL_H

#include "Wafer/Compiler/Compilation.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler {

/// Internal construction seam shared by the current coordinator and
/// final bundle owner.  Construction remains unavailable to public callers.
struct ExecutableBundleBuilder {
  static PhysicalTileExecutable makePhysicalTile(
      PhysicalCardId physicalCardId, PhysicalTileId physicalTileId,
      LaunchSlotId launchSlotId,
      mlir::OwningOpRef<mlir::ModuleOp> module, llvm::StringRef entrySymbol,
      std::vector<ProgramResourceBinding> programBindings,
      TransportContract transportContract,
      llvm::StringRef selectedTileIR = {}) {
    return PhysicalTileExecutable(
        physicalCardId, physicalTileId, launchSlotId, std::move(module),
        entrySymbol,
        std::move(programBindings), transportContract, selectedTileIR);
  }

  static ExecutableBundle
  makeBundle(ExecutionConfig executionConfig,
             RuntimeLaunchContract runtimeLaunchContract,
             std::shared_ptr<mlir::MLIRContext> context,
             std::vector<PhysicalTileExecutable> physicalTiles) {
    return ExecutableBundle(executionConfig, std::move(runtimeLaunchContract),
                            std::move(context), std::move(physicalTiles));
  }
};

namespace detail {

mlir::LogicalResult
verifyExactExecutionConfig(mlir::ModuleOp module,
                           const ExecutionConfig &executionConfig);

llvm::Expected<ExecutableBundle> buildExecutableBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot);

} // namespace detail
} // namespace wafer::compiler

#endif // WAFER_COMPILER_EXECUTABLEBUNDLEINTERNAL_H
