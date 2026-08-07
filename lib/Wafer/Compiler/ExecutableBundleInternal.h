//===- ExecutableBundleInternal.h - Internal bundle construction -*- C++
//-*-===//

#ifndef WAFER_COMPILER_EXECUTABLEBUNDLEINTERNAL_H
#define WAFER_COMPILER_EXECUTABLEBUNDLEINTERNAL_H

#include "Wafer/Compiler/Compilation.h"
#include "WholeVariantSelection.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler {

/// Internal construction seam shared by the rank-frontier coordinator and
/// final bundle owner.  Construction remains unavailable to public callers.
struct ExecutableBundleBuilder {
  static RankExecutable
  makeRank(int64_t logicalRank, mlir::OwningOpRef<mlir::ModuleOp> module,
           llvm::StringRef entrySymbol,
           std::vector<RankProgramBinding> programBindings,
           TransportContract transportContract,
           llvm::StringRef selectedTileIR = {}) {
    return RankExecutable(logicalRank, std::move(module), entrySymbol,
                          std::move(programBindings), transportContract,
                          selectedTileIR);
  }

  static ExecutableBundle
  makeBundle(ExecutionConfig executionConfig,
             RuntimeLaunchContract runtimeLaunchContract,
             std::shared_ptr<mlir::MLIRContext> context,
             std::vector<RankExecutable> ranks) {
    return ExecutableBundle(executionConfig, std::move(runtimeLaunchContract),
                            std::move(context), std::move(ranks));
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
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLogicalRank,
    WholeVariantSelectionMode selectionMode =
        WholeVariantSelectionMode::Production);

inline llvm::Expected<ExecutableBundle> buildExecutableBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    WholeVariantSelectionMode selectionMode =
        WholeVariantSelectionMode::Production) {
  return buildExecutableBundle(context, tensorModule, std::move(program),
                               executionConfig,
                               OptimizationConfig::production(), diagnostics,
                               failAfterLogicalRank, selectionMode);
}

} // namespace detail
} // namespace wafer::compiler

#endif // WAFER_COMPILER_EXECUTABLEBUNDLEINTERNAL_H
